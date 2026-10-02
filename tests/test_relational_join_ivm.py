"""Incremental maintenance tests for deterministic relational inner joins."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.runtime import executor as executor_module
from agent_memory.adapters import LotusAdapter
from agent_memory.planner.differential_policy import DifferentialNode
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.relation import LOG_ADD_SEQ_COLUMN


class _BipartiteJoinMemory(am.Memory):
    log = am.Log(
        {
            "side": "Join side.",
            "key": "Join key.",
            "value": "Row value.",
        }
    )
    _left = (
        log.filter(log.col("side") == "left")
        .select(["key", "value"])
        .alias("left")
    )
    _right = (
        log.filter(log.col("side") == "right")
        .select(["key", "value"])
        .alias("right")
    )
    pairs = _left.join(_right, on="key", how="inner")


class _ExplodedSelfJoinMemory(am.Memory):
    log = am.Log({"items": "JSON array of items."})
    _items = (
        log.explode(column="items", output_col="item")
        .select(["item"])
        .assign(key="all")
    )
    _left = _items.alias("left")
    _right = _items.alias("right")
    pairs = _left.join(_right, on="key", how="inner")


class _TemporalSelfJoinMemory(am.Memory):
    log = am.Log({"value": "Event value."}, system_columns=True)
    _events = log.select([LOG_ADD_SEQ_COLUMN, "value"])
    _earlier = _events.alias("earlier")
    _later = _events.alias("later")
    pairs = _earlier.join(
        _later,
        on=_earlier.col(LOG_ADD_SEQ_COLUMN) < _later.col(LOG_ADD_SEQ_COLUMN),
        how="inner",
    )


class _TemporalFilteredMemory(am.Memory):
    log = am.Log({"value": "Event value."}, system_columns=True)
    _events = log.select([LOG_ADD_SEQ_COLUMN, "value"])
    _earlier = _events.alias("earlier")
    _later = _events.alias("later")
    _pairs = _earlier.join(
        _later,
        on=_earlier.col(LOG_ADD_SEQ_COLUMN) < _later.col(LOG_ADD_SEQ_COLUMN),
        how="inner",
    )
    links = _pairs.sem_filter(
        instruction="The earlier and later events should be linked."
    )


class _RetractingParentMemory(am.Memory):
    log = am.Log({"key": "Group key.", "value": "Numeric value."})
    _minimum = log.group_by("key").min(column="value", output_col="value")
    _left = _minimum.alias("left")
    _right = _minimum.alias("right")
    pairs = _left.join(_right, on="key", how="inner")


class _TrackingAdapter(LotusAdapter):
    """Deterministic adapter that records root execution and accepts sem_filter rows."""

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self.root_queries: list[QueryExpr] = []
        self.sem_filter_input_sizes: list[int] = []

    def execute(
        self,
        query: QueryExpr,
        inputs: Mapping[str, Any],
    ) -> pd.DataFrame:
        is_root = self._depth == 0
        if is_root:
            self.root_queries.append(query)
        self._depth += 1
        try:
            if query.op == "sem_filter":
                source = self.execute(query.inputs[0], inputs)
                self.sem_filter_input_sizes.append(len(source))
                return source
            return super().execute(query, inputs)
        finally:
            self._depth -= 1


def _only_join_node(memory_type: type[am.Memory]) -> DifferentialNode:
    nodes = [
        node
        for node in memory_type.differentiate_policy().nodes.values()
        if node.query.op == "join"
    ]
    assert len(nodes) == 1
    return nodes[0]


def _query_nodes(query: QueryExpr) -> list[QueryExpr]:
    return [query, *(node for item in query.inputs for node in _query_nodes(item))]


def _bag(frame: pd.DataFrame) -> Counter[tuple[object, ...]]:
    return Counter(tuple(row) for row in frame.itertuples(index=False, name=None))


def _full_view(memory: am.Memory, view_name: str) -> pd.DataFrame:
    query = memory.spec().views[view_name].query
    return LotusAdapter().execute(query, {"log": memory._runtime._state["log"]})


def test_compiler_builds_bag_preserving_inner_join_maintenance_query() -> None:
    node = _only_join_node(_BipartiteJoinMemory)

    assert node.execution_kind == "relational_state"
    assert node.maintenance_query is not None
    nodes = _query_nodes(node.maintenance_query)
    joins = [query for query in nodes if query.op == "join"]
    concats = [query for query in nodes if query.op == "concat"]
    materialized_names = {
        str(query.params["name"])
        for query in nodes
        if query.op == "materialized_view"
    }

    assert len(joins) == 3
    assert len(concats) == 3
    assert all(query.params == node.query.params for query in joins)
    assert all(query.inputs[0].params["name"] == "left" for query in joins)
    assert all(query.inputs[1].params["name"] == "right" for query in joins)
    assert node.node_id in materialized_names
    for parent_id in node.input_node_ids:
        assert parent_id in materialized_names
        assert f"{parent_id}__inserted" in materialized_names


def test_inner_join_maintains_left_only_and_right_only_insertions() -> None:
    memory = _BipartiteJoinMemory()

    memory.add({"side": "right", "key": "k", "value": "r1"})
    assert memory._runtime._state["pairs"].empty

    memory.add({"side": "left", "key": "k", "value": "l1"})
    memory.add({"side": "left", "key": "k", "value": "l2"})
    memory.add({"side": "right", "key": "k", "value": "r2"})

    assert _bag(memory._runtime._state["pairs"]) == Counter(
        {
            ("k", "l1", "r1"): 1,
            ("k", "l2", "r1"): 1,
            ("k", "l1", "r2"): 1,
            ("k", "l2", "r2"): 1,
        }
    )


def test_self_join_multirow_delta_preserves_bag_multiplicity() -> None:
    memory = _ExplodedSelfJoinMemory()

    memory.add({"items": '["a", "a"]'})
    assert _bag(memory._runtime._state["pairs"]) == Counter(
        {("a", "all", "a"): 4}
    )
    assert _bag(memory._runtime._state["pairs"]) == _bag(_full_view(memory, "pairs"))

    memory.add({"items": '["a"]'})
    assert _bag(memory._runtime._state["pairs"]) == Counter(
        {("a", "all", "a"): 9}
    )
    assert _bag(memory._runtime._state["pairs"]) == _bag(_full_view(memory, "pairs"))


def test_predicate_self_join_matches_full_recompute_after_each_append() -> None:
    memory = _TemporalSelfJoinMemory()

    for value in ("a", "b", "c"):
        memory.add({"value": value})
        assert _bag(memory._runtime._state["pairs"]) == _bag(
            _full_view(memory, "pairs")
        )

    pairs = memory._runtime._state["pairs"]
    assert list(
        zip(
            pairs[f"{LOG_ADD_SEQ_COLUMN}:earlier"],
            pairs[f"{LOG_ADD_SEQ_COLUMN}:later"],
            strict=True,
        )
    ) == [(0, 1), (0, 2), (1, 2)]


def test_downstream_sem_filter_receives_only_new_join_pairs() -> None:
    adapter = _TrackingAdapter()
    memory = _TemporalFilteredMemory(adapter=adapter)

    for value in ("a", "b", "c"):
        memory.add({"value": value})

    assert adapter.sem_filter_input_sizes == [1, 2]
    assert len(memory._runtime._state["links"]) == 3


def test_parent_retraction_uses_affected_join_parts() -> None:
    adapter = _TrackingAdapter()
    memory = _RetractingParentMemory(adapter=adapter)
    node = _only_join_node(_RetractingParentMemory)

    memory.add({"key": "k", "value": 5})
    adapter.root_queries.clear()
    memory.add({"key": "k", "value": 3})

    assert node.query not in adapter.root_queries
    assert any(query.op == "join" for query in adapter.root_queries)
    assert node.maintenance_query not in adapter.root_queries
    assert memory._runtime._state["pairs"].to_dict(orient="records") == [
        {"key": "k", "value:left": 3, "value:right": 3}
    ]


def test_non_inner_join_keeps_deterministic_full_recompute_path() -> None:
    class OuterJoinMemory(am.Memory):
        log = am.Log({"key": "Join key."})
        pairs = log.join(log, on="key", how="outer")

    node = _only_join_node(OuterJoinMemory)

    assert node.execution_kind == "deterministic"
    assert node.maintenance_query is None


def test_inner_join_snapshot_round_trip_and_fingerprint_validation() -> None:
    original = _TemporalSelfJoinMemory()
    original.add({"value": "a"})
    original.add({"value": "b"})
    snapshot = original._runtime.snapshot_state()

    restored = _TemporalSelfJoinMemory()
    restored._runtime.restore_state(snapshot)
    original.add({"value": "c"})
    restored.add({"value": "c"})

    assert _bag(restored._runtime._state["pairs"]) == _bag(
        original._runtime._state["pairs"]
    )

    incompatible = dict(snapshot)
    incompatible["plan_fingerprint"] = "pre-relational-state-plan"
    with pytest.raises(ValueError, match="plan fingerprint"):
        _TemporalSelfJoinMemory()._runtime.restore_state(incompatible)


@pytest.mark.parametrize("value", ["next", "unchanged"])
def test_join_does_not_rekey_old_pairs_on_append_or_no_change(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    class FilteredMemory(am.Memory):
        log = am.Log({"value": "Value."}, system_columns=True)
        _kept = log.filter(log.col("value") != "unchanged")
        _left = _kept.alias("left")
        _right = _kept.alias("right")
        pairs = _left.join(
            _right, on=_left.col(LOG_ADD_SEQ_COLUMN) < _right.col(LOG_ADD_SEQ_COLUMN),
        )

    memory = FilteredMemory()
    for v in ("a", "b", "c"):
        memory.add({"value": v})
    node = _only_join_node(FilteredMemory)
    old = memory._runtime.snapshot_state()["node_state"][node.node_id].copy()
    original = executor_module._row_key
    old_keys = {original(row) for row in old.itertuples(index=False, name=None)}
    inspected = 0

    def count_pair_keys(row: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal inspected
        if len(row) == len(old.columns) and original(row) in old_keys:
            inspected += 1
        return original(row)

    monkeypatch.setattr(executor_module, "_row_key", count_pair_keys)
    memory.add({"value": value})
    current = memory._runtime.snapshot_state()["node_state"][node.node_id]
    pd.testing.assert_frame_equal(current.loc[old.index], old)
    assert inspected == 0
    assert len(current) == (6 if value == "next" else 3)


@pytest.mark.parametrize("indexed", [False, True])
def test_join_delta_cancels_replacements_and_preserves_nested_null_rows(indexed: bool) -> None:
    old = pd.DataFrame({"value": [None, None, {"a": [1, None]}]}, index=pd.Index(["a", "b", "c"]))
    removed = old.iloc[[0, 2]]
    added = pd.DataFrame({"value": [{"a": [1, None]}, "new"]})
    index = executor_module._index_join_rows(old) if indexed else None
    update = executor_module._apply_join_delta(old, removed, added, allocate=lambda: "d", row_index=index)
    assert update.retracted_rows.index.tolist() == ["b"]
    assert update.inserted_rows.index.tolist() == ["d"]
    pd.testing.assert_frame_equal(update.output_rows.loc[["a", "c"]], old.loc[["a", "c"]])
    assert update.output_rows.loc["d", "value"] == "new"
    if indexed:
        assert index == executor_module._index_join_rows(update.output_rows)
    with pytest.raises(RuntimeError, match="retractions exceed"):
        executor_module._apply_join_delta(old, pd.concat([old, old]), old.iloc[:0], allocate=lambda: "unused")


def test_replacing_unprojected_join_input_does_not_repeat_semantic_work() -> None:
    class ProjectedMemory(am.Memory):
        log = am.Log({"key": "key", "value": "value"})
        _state = log.group_by("key").min(column="value", output_col="value")
        _left, _right = _state.alias("l"), _state.alias("r")
        pairs = _left.join(_right, on="key").select(["key"])
        result = pairs.sem_filter(instruction="Keep all rows.")

    adapter = _TrackingAdapter()
    memory = ProjectedMemory(adapter=adapter)
    memory.add({"key": "a", "value": 9})
    memory.add({"key": "b", "value": 5})
    before = memory._runtime.snapshot_state()
    calls = list(adapter.sem_filter_input_sizes)
    memory.add({"key": "a", "value": 3})
    assert adapter.sem_filter_input_sizes == calls
    assert _bag(memory._runtime._state["pairs"]) == Counter({("a",): 1, ("b",): 1})
    node = _only_join_node(ProjectedMemory)
    old = before["node_state"][node.node_id]
    unchanged = old.loc[old["key"] == "b"]
    current = memory._runtime.snapshot_state()["node_state"][node.node_id]
    pd.testing.assert_frame_equal(current.loc[unchanged.index], unchanged)


def test_join_append_retains_compiled_task_order() -> None:
    memory = _ExplodedSelfJoinMemory()
    memory.add({"items": '["a", "b"]'})
    before = memory._runtime.snapshot_state()["node_state"]
    memory.add({"items": '["c", "d"]'})
    after = memory._runtime.snapshot_state()["node_state"]
    node = _only_join_node(_ExplodedSelfJoinMemory)
    inputs = {node.node_id: before[node.node_id]}
    for parent in node.input_node_ids:
        inputs[parent] = before[parent]
        inputs[f"{parent}__inserted"] = executor_module.NodeOutputUpdate.between(
            before[parent], after[parent],
        ).inserted_rows
    assert node.maintenance_query is not None
    expected = LotusAdapter().execute(node.maintenance_query, inputs)
    pd.testing.assert_frame_equal(
        after[node.node_id].reset_index(drop=True), expected.reset_index(drop=True),
    )


@pytest.mark.parametrize("indexed", [False, True])
def test_join_delta_random_bags_match_net_changes(indexed: bool) -> None:
    import random
    from itertools import count

    rng = random.Random(41)
    for _ in range(50):
        values = [rng.randrange(4) for _ in range(rng.randrange(20))]
        old = pd.DataFrame({"value": values}, index=pd.Index([f"old:{i}" for i in range(len(values))]))
        positions = rng.sample(range(len(values)), rng.randrange(len(values) + 1))
        removed = old.iloc[positions]
        new_values = [rng.randrange(4) for _ in range(rng.randrange(10))]
        added = pd.DataFrame({"value": new_values})
        ids = count()
        index = executor_module._index_join_rows(old) if indexed else None
        update = executor_module._apply_join_delta(
            old, removed, added, allocate=lambda: f"new:{next(ids)}", row_index=index,
        )
        expected = Counter(v for i, v in enumerate(values) if i not in positions) + Counter(new_values)
        actual = Counter(update.output_rows["value"])
        assert actual == expected
        assert Counter(update.inserted_rows["value"]) == expected - Counter(values)
        assert Counter(update.retracted_rows["value"]) == Counter(values) - expected
        assert update.output_rows.index.is_unique
        if indexed:
            assert index == executor_module._index_join_rows(update.output_rows)
