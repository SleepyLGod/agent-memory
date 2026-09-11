"""Exact incremental-maintenance tests for relational algebraic aggregates."""

from __future__ import annotations

from collections import Counter

import pandas as pd

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.policy.schema import output_columns


def _bag(frame: pd.DataFrame) -> Counter[tuple[object, ...]]:
    return Counter(tuple(row) for row in frame.itertuples(index=False, name=None))


def _full_view(memory: am.Memory, view_name: str) -> pd.DataFrame:
    query = memory.spec().views[view_name].query
    return LotusAdapter().execute(query, {"log": memory._runtime._state["log"]})


class _GlobalAggregateMemory(am.Memory):
    log = am.Log({"amount": "Numeric amount."})
    metrics = log.agg(
        am.count(output_col="row_count"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="average"),
    )


class _FilteredGlobalAggregateMemory(am.Memory):
    log = am.Log({"kind": "Row kind.", "amount": "Numeric amount."})
    _selected = log.filter(log.col("kind") == "selected")
    metrics = _selected.agg(
        am.count(output_col="row_count"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="average"),
    )


class _GroupedAggregateMemory(am.Memory):
    log = am.Log({"region": "Region.", "amount": "Numeric amount."})
    metrics = log.group_by("region").agg(
        am.count(output_col="row_count"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="average"),
    )


class _ReplacementAggregateMemory(am.Memory):
    log = am.Log({"key": "Key.", "amount": "Numeric amount."})
    _minimums = log.group_by("key").min(column="amount", output_col="amount")
    metrics = _minimums.agg(
        am.count(output_col="row_count"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="average"),
    )


class _GroupMovementMemory(am.Memory):
    log = am.Log({"group": "Ordered group."})
    _current_group = log.min(column="group", output_col="group")
    metrics = _current_group.group_by("group").count(output_col="row_count")


def test_planner_lowers_aggregate_to_hidden_state_and_public_finalizer() -> None:
    policy = _GlobalAggregateMemory.differentiate_policy()
    state_nodes = [
        node for node in policy.nodes.values() if node.execution_kind == "algebraic_state"
    ]

    assert len(state_nodes) == 1
    state = state_nodes[0]
    assert state.query.op == "aggregate_state"
    assert state.maintenance_query is not None
    assert state.maintenance_query.op == "aggregate_state_update"
    assert any(column.startswith("__am_aggregate_") for column in state.output_columns)
    assert len(tuple(state.query.params["numeric_states"])) == 1

    public_node = policy.nodes[policy.view_outputs["metrics"]]
    assert public_node.query.op == "aggregate_finalize"
    assert public_node.execution_kind == "deterministic"
    assert public_node.output_columns == ("row_count", "total", "average")
    assert all(
        not column.startswith("__am_aggregate_")
        for column in public_node.output_columns
    )

    grouped_policy = _GroupedAggregateMemory.differentiate_policy()
    grouped_public = grouped_policy.nodes[grouped_policy.view_outputs["metrics"]]
    assert tuple(grouped_public.query.params["group_keys"]) == ("region",)
    assert tuple(grouped_public.output_columns) == (
        "region",
        "row_count",
        "total",
        "average",
    )
    assert output_columns(grouped_public.query) == grouped_public.output_columns


def test_global_incremental_result_matches_full_recomputation_after_each_batch() -> None:
    memory = _GlobalAggregateMemory()
    batches = (
        pd.DataFrame([{"amount": 10}, {"amount": None}]),
        pd.DataFrame([{"amount": 20}, {"amount": 20}]),
    )

    for batch in batches:
        memory._runtime._engine.apply_delta(batch)
        assert _bag(memory._runtime._state["metrics"]) == _bag(
            _full_view(memory, "metrics")
        )

    assert memory._runtime._state["metrics"].to_dict(orient="records") == [
        {"row_count": 4, "total": 50, "average": 50 / 3}
    ]


def test_global_aggregate_materializes_identity_when_parent_delta_is_empty() -> None:
    memory = _FilteredGlobalAggregateMemory()

    memory.add({"kind": "ignored", "amount": 10})

    assert memory._runtime._state["metrics"].to_dict(orient="records") == [
        {"row_count": 0, "total": None, "average": None}
    ]
    assert _bag(memory._runtime._state["metrics"]) == _bag(
        _full_view(memory, "metrics")
    )


def test_grouped_incremental_result_matches_full_recomputation() -> None:
    memory = _GroupedAggregateMemory()

    for row in (
        {"region": "north", "amount": 10},
        {"region": "south", "amount": 5},
        {"region": "north", "amount": None},
        {"region": "north", "amount": 20},
    ):
        memory.add(row)
        assert _bag(memory._runtime._state["metrics"]) == _bag(
            _full_view(memory, "metrics")
        )


def test_parent_row_replacement_subtracts_old_values_without_full_recompute() -> None:
    memory = _ReplacementAggregateMemory()

    memory.add({"key": "a", "amount": 5})
    memory.add({"key": "b", "amount": 7})
    memory.add({"key": "a", "amount": 3})

    assert memory._runtime._state["metrics"].to_dict(orient="records") == [
        {"row_count": 2, "total": 10, "average": 5}
    ]
    assert _bag(memory._runtime._state["metrics"]) == _bag(
        _full_view(memory, "metrics")
    )


def test_group_disappears_when_deterministic_parent_moves_its_only_row() -> None:
    memory = _GroupMovementMemory()

    memory.add({"group": "b"})
    assert memory._runtime._state["metrics"].to_dict(orient="records") == [
        {"group": "b", "row_count": 1}
    ]

    memory.add({"group": "a"})
    assert memory._runtime._state["metrics"].to_dict(orient="records") == [
        {"group": "a", "row_count": 1}
    ]
    assert _bag(memory._runtime._state["metrics"]) == _bag(
        _full_view(memory, "metrics")
    )


def test_aggregate_snapshot_restore_preserves_hidden_state() -> None:
    original = _GroupedAggregateMemory()
    original.add({"region": "north", "amount": 10})
    original.add({"region": "south", "amount": 5})
    snapshot = original._runtime.snapshot_state()

    restored = _GroupedAggregateMemory()
    restored._runtime.restore_state(snapshot)
    original.add({"region": "north", "amount": 20})
    restored.add({"region": "north", "amount": 20})

    assert _bag(restored._runtime._state["metrics"]) == _bag(
        original._runtime._state["metrics"]
    )
    assert _bag(restored._runtime._state["metrics"]) == _bag(
        _full_view(restored, "metrics")
    )


def test_aggregate_plan_fingerprint_is_stable_and_builtin_defaults_are_pinned() -> None:
    assert (
        _GlobalAggregateMemory.differentiate_policy().fingerprint
        == _GlobalAggregateMemory.differentiate_policy().fingerprint
    )
    assert (
        am.ClaudeMemory.differentiate_policy().fingerprint
        == "625a53555e82e283d04576eacf7e937c590b546a485bcccc6224d3b8c4dae5fe"
    )
    assert (
        am.ZepMemory.differentiate_policy().fingerprint
        == "9d1cb107ef677e9624dffeb22399dd6edd7eb6d058e4a37da26c61c2a8e64059"
    )
    assert (
        am.Mem0Memory.differentiate_policy().fingerprint
        == "d904fdb2fbd64ffdbaef65de51fb7f366c29b7ebbd05ba5194e909725b7f2d28"
    )
