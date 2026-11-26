"""Exact incremental-maintenance tests for relational algebraic aggregates."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.algebraic_aggregates import (
    _cancel_opposite_contributions,
    _direct_layout,
    _finalize_state,
    _update_state,
)
from agent_memory.planner.algebraic_aggregates import aggregate_layout
from agent_memory.policy.schema import output_columns


@pytest.mark.parametrize(
    "initial,inserted,retracted,expected",
    [
        ([7, 9, 7], [7, 3, 7, 5], [7, 9, 7], (4, 22, 5.5)),
        ([7], [7, 7], [7], (2, 14, 7)),
        ([7, 7], [7], [7, 7], (1, 7, 7)),
        ([None, 7], [None, 7], [None, 7], (2, 7, 7)),
        ([None, 7], [], [7], (1, None, None)),
        ([0, 1e308, -1e308, 1e308], [], [1e308, -1e308, 1e308], (1, 0, 0)),
        # Equal opposite contributions must not round away the existing sum.
        ([1], [1e16], [1e16], (1, 1, 1)),
    ],
)
def test_numeric_deltas_preserve_duplicate_and_null_contributions(
    initial: list[float | None],
    inserted: list[float | None],
    retracted: list[float | None],
    expected: tuple[int, float | None, float | None],
) -> None:
    def frame(values: list[float | None]) -> pd.DataFrame:
        return pd.DataFrame({"group": ["g"] * len(values), "amount": values})

    layout = _direct_layout(
        group_keys=("group",),
        aggregates=(
            am.count(output_col="n"),
            am.sum(column="amount", output_col="total"),
            am.avg(column="amount", output_col="mean"),
        ),
    )
    state = _update_state(
        pd.DataFrame(columns=layout["state_columns"]), frame(initial), frame([]), layout
    )
    state = _update_state(state, frame(inserted), frame(retracted), layout)

    assert _finalize_state(state, layout).to_dict(orient="records") == [
        {"group": "g", "n": expected[0], "total": expected[1], "mean": expected[2]}
    ]


def _bag(frame: pd.DataFrame) -> Counter[tuple[object, ...]]:
    return Counter(tuple(row) for row in frame.itertuples(index=False, name=None))


def test_large_finite_sum_and_retractions_preserve_row_order() -> None:
    layout = _direct_layout(
        group_keys=("group",),
        aggregates=(
            am.count(output_col="n"),
            am.sum(column="amount", output_col="total"),
            am.avg(column="amount", output_col="mean"),
        ),
    )
    rows = pd.DataFrame(
        {"group": ["g"] * 3, "amount": [1e308, -1e308, 1e308]}
    )
    empty = rows.iloc[:0]
    state = _update_state(pd.DataFrame(columns=layout["state_columns"]), rows, empty, layout)
    result = _finalize_state(state, layout).iloc[0]
    assert (result.n, result.total, result["mean"]) == (3, 1e308, 1e308 / 3)
    state = _update_state(state, empty, rows.iloc[[0, 2]], layout)
    result = _finalize_state(state, layout).iloc[0]
    assert (result.n, result.total, result["mean"]) == (1, -1e308, -1e308)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("amounts", [[1e16, -1e16, 1.0], [1e308, -1e308, 1e308]])
def test_business_field_replacement_preserves_ordinary_numeric_state(
    grouped: bool, amounts: list[float]
) -> None:
    source = am.Source({"region": "Region", "amount": "Amount", "revision": "Revision"})
    parent = source.group_by("region") if grouped else source
    query = parent.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )
    layout = aggregate_layout(query.expr)
    old = pd.DataFrame({"region": ["north"] * 3, "amount": amounts, "revision": [1] * 3})
    state = _update_state(
        pd.DataFrame(columns=layout["state_columns"]), old, old.iloc[:0], layout
    )

    updated = _update_state(state, old.assign(revision=2), old, layout)

    pd.testing.assert_frame_equal(updated, state, check_exact=True)
    pd.testing.assert_frame_equal(
        _finalize_state(updated, layout), _finalize_state(state, layout), check_exact=True
    )


@pytest.mark.parametrize("group_keys", [("group",), ("group", "amount")])
def test_contribution_cancellation_preserves_surviving_occurrences_and_types(
    group_keys: tuple[str, ...],
) -> None:
    layout = _direct_layout(
        group_keys=group_keys, aggregates=(am.sum(column="amount", output_col="total"),)
    )
    inserted = pd.DataFrame(
        {"group": ["g"] * 8,
         "amount": [7, 3, 7, 1e308, -1e308, 1e308, Decimal("7"), 7.0],
         "business": [["new"]] * 8},
        dtype=object,
        index=[4] * 8,
    )
    retracted = pd.DataFrame(
        {"group": ["g"] * 3, "amount": [7, 7, 9], "business": [["old"]] * 3},
        dtype=object,
        index=[4] * 3,
    )

    remaining_inserted, remaining_retracted = _cancel_opposite_contributions(
        inserted, retracted, layout
    )

    pd.testing.assert_frame_equal(remaining_inserted, inserted.iloc[[1, 3, 4, 5, 6, 7]])
    pd.testing.assert_frame_equal(remaining_retracted, retracted.iloc[[2]])
    assert [type(value) for value in remaining_inserted["amount"]] == [
        int, float, float, float, Decimal, float
    ]


@pytest.mark.parametrize("inserted_count,retracted_count", [(3, 2), (2, 3), (2, 2)])
def test_global_count_cancellation_preserves_unmatched_rows(
    inserted_count: int, retracted_count: int,
) -> None:
    layout = _direct_layout(group_keys=(), aggregates=(am.count(output_col="n"),))
    inserted = pd.DataFrame({"business": [["new"]] * inserted_count})
    retracted = pd.DataFrame({"business": [["old"]] * retracted_count})

    remaining_inserted, remaining_retracted = _cancel_opposite_contributions(
        inserted, retracted, layout
    )

    pd.testing.assert_frame_equal(remaining_inserted, inserted.iloc[retracted_count:])
    pd.testing.assert_frame_equal(remaining_retracted, retracted.iloc[inserted_count:])


@pytest.mark.parametrize(
    "column,value,cancels",
    [("region", "south", False), ("kind", "b", False), ("tax", 2, False),
     ("amount", 7.0, False), ("region", None, False), ("revision", 2, True)],
)
def test_cancellation_uses_all_group_keys_and_numeric_inputs(
    column: str, value: object, cancels: bool
) -> None:
    layout = _direct_layout(
        group_keys=("region", "kind"),
        aggregates=(am.sum(column="amount", output_col="total"),
                    am.avg(column="tax", output_col="mean_tax")),
    )
    record = {"region": "north", "kind": "a", "amount": 7, "tax": 1, "revision": 1}
    old = pd.DataFrame([record], dtype=object)
    new = pd.DataFrame([{**record, column: value}], dtype=object)

    inserted, retracted = _cancel_opposite_contributions(new, old, layout)

    pd.testing.assert_frame_equal(inserted, new.iloc[:0] if cancels else new)
    pd.testing.assert_frame_equal(retracted, old.iloc[:0] if cancels else old)


@pytest.mark.parametrize("column", ["amount", "tax"])
@pytest.mark.parametrize("value", [True, np.bool_(False), "7", object(), [7], {"v": 7}])
def test_cancellation_does_not_hide_invalid_numeric_inputs(column: str, value: object) -> None:
    layout = _direct_layout(
        group_keys=("group",),
        aggregates=(am.sum(column="amount", output_col="total"),
                    am.avg(column="tax", output_col="mean_tax")),
    )
    valid = pd.DataFrame([{"group": "g", "amount": 1, "tax": 1}], dtype=object)
    state = _update_state(
        pd.DataFrame(columns=layout["state_columns"]), valid, valid.iloc[:0], layout
    )
    before = state.copy(deep=True)
    invalid = pd.DataFrame([{"group": "g", "amount": 1, "tax": 1, column: value}], dtype=object)

    with pytest.raises(TypeError, match="requires numeric non-null values"):
        _update_state(state, invalid, invalid.copy(), layout)

    pd.testing.assert_frame_equal(state, before, check_exact=True)


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
        == "79a48859ca8f568fc4654b1874c43f435c437735c3354f101d4a02e112273a12"
    )
    assert (
        am.ZepMemory.differentiate_policy().fingerprint
        == "4d8ae328d44e111a4e761fee4d2242153c09ca430954ffe38d7a18d047be6419"
    )
    assert (
        am.Mem0Memory.differentiate_policy().fingerprint
        == "d904fdb2fbd64ffdbaef65de51fb7f366c29b7ebbd05ba5194e909725b7f2d28"
    )
    # Removing obsolete label parameters changes hashes, not the selected rules.
    for memory_class in (am.ClaudeMemory, am.ZepMemory, am.Mem0Memory):
        assert memory_class.differentiate_policy().grouped_agg_rule == "rule-join-map"
