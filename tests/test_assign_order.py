"""Assignment order is part of a logical node's execution identity."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.planner.serialization import stable_json
from agent_memory.policy.logical import QueryExpr


def _legacy_snapshot(name: str) -> dict[str, Any]:
    """Load a schema-v2 LOTUS snapshot captured after add({"x": 1}) at 3c577e32."""
    path = Path(__file__).with_name("fixtures") / "assign_snapshots_3c577e32.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))[name]
    for key in ("state", "node_state"):
        snapshot[key] = {node: pd.DataFrame(**frame) for node, frame in snapshot[key].items()}
    return snapshot


def test_legacy_single_assign_restores_and_continues() -> None:
    class Single(am.Memory):
        log = am.Log({"x": "integer"})
        view = log.assign(y=log.col("x"))

    resumed = Single(adapter=LotusAdapter())
    resumed._runtime.restore_state(_legacy_snapshot("Single"))
    resumed.add({"x": 3})
    fresh = Single(adapter=LotusAdapter())
    for x in (1, 3):
        fresh.add({"x": x})
    pd.testing.assert_frame_equal(resumed._runtime._state["view"], fresh._runtime._state["view"])
    assert resumed._runtime._state["view"]["y"].tolist() == [1, 3]


def test_legacy_reversed_assign_snapshot_is_rejected() -> None:
    class Ordered(am.Memory):
        log = am.Log({"x": "integer"})
        view = log.assign(x=2, y=log.col("x")).concat(log.assign(y=log.col("x"), x=2))

    memory = Ordered(adapter=LotusAdapter())
    old = _legacy_snapshot("Ordered")
    assert old["state"]["view"]["y"].tolist() == [2, 2]
    with pytest.raises(ValueError, match="plan fingerprint does not match"):
        memory._runtime.restore_state(old)
    memory.add({"x": 1})
    assert memory._runtime._state["view"]["y"].tolist() == [2, 1]


class _Branches(am.Memory):
    log = am.Log({"x": "integer"})
    both = log.assign(x=2, y=log.col("x")).concat(
        log.assign(y=log.col("x"), x=2)
    )
    repeated = log.assign(x=2, y=log.col("x")).concat(
        log.assign(x=2, y=log.col("x"))
    )


def test_assign_identity_distinguishes_order_but_still_shares_identical_nodes() -> None:
    left, right = _Branches.both.expr.inputs
    assert len({left, right}) == 2
    assert stable_json(left) != stable_json(right)
    plan = _Branches.differentiate_policy()
    assert len(set(plan.nodes[plan.view_outputs["both"]].input_node_ids)) == 2
    assert len(set(plan.nodes[plan.view_outputs["repeated"]].input_node_ids)) == 1


def test_orderless_params_and_single_assign_keep_their_identity() -> None:
    for assignments in ({}, {"x": 1}):
        query = QueryExpr("assign", params={"assignments": assignments})
        expected = '{}' if not assignments else '{"x":1}'
        assert stable_json(query) == (
            '{"inputs":[],"op":"assign","params":{"assignments":' + expected + '}}'
        )
    a = QueryExpr("example", params={"a": 1, "b": 2})
    b = QueryExpr("example", params={"b": 2, "a": 1})
    assert a == b and hash(a) == hash(b)
    assert stable_json(a) == stable_json(b)


@pytest.mark.parametrize("backend", [LotusAdapter, DocetlAdapter])
def test_runtime_respects_assign_order_across_restore(backend: type) -> None:
    memory = _Branches(adapter=backend())
    memory.add({"x": 1})
    view = memory._runtime._state["both"]
    assert list(view.columns) == ["x", "y"]
    assert view["y"].tolist() == [2, 1]
    assert memory._runtime._state["repeated"]["y"].tolist() == [2, 2]

    restored = _Branches(adapter=backend())
    restored._runtime.restore_state(memory._runtime.snapshot_state())
    for instance in (memory, restored):
        instance.add({"x": 3})
    pd.testing.assert_frame_equal(restored._runtime._state["both"], memory._runtime._state["both"])
    assert sorted(restored._runtime._state["both"]["y"]) == [1, 2, 2, 3]
