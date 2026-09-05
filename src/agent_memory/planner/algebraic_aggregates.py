"""Lower exact relational aggregates into hidden state and public finalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_memory.policy.aggregates import (
    AvgAggregateSpec,
    SumAggregateSpec,
    is_algebraic_aggregate_spec,
)
from agent_memory.policy.logical import QueryExpr

_HIDDEN_PREFIX = "__am_aggregate_"


def is_algebraic_aggregate_query(query: QueryExpr) -> bool:
    """Return whether a logical query is an exact count/sum/avg aggregate."""

    aggregates = tuple(query.params.get("aggregates", ()))
    return (
        query.op == "agg"
        and bool(aggregates)
        and all(is_algebraic_aggregate_spec(spec) for spec in aggregates)
    )


def aggregate_layout(query: QueryExpr) -> dict[str, Any]:
    """Return a stable hidden-state layout for one logical aggregate query."""

    if not is_algebraic_aggregate_query(query):
        raise TypeError("aggregate lowering requires only count/sum/avg specs")
    aggregate_input = query.inputs[0]
    if aggregate_input.op == "sem_groupby":
        raise NotImplementedError(
            "sem_groupby numeric algebraic aggregates are not supported"
        )
    group_keys = (
        tuple(str(key) for key in aggregate_input.params["keys"])
        if aggregate_input.op == "group_by"
        else ()
    )
    aggregates = tuple(query.params["aggregates"])
    occupied = set(group_keys)
    row_count_col = _unique_hidden_name("row_count", occupied)
    occupied.add(row_count_col)

    numeric_columns: list[str] = []
    for aggregate in aggregates:
        if isinstance(aggregate, (SumAggregateSpec, AvgAggregateSpec)):
            if aggregate.column not in numeric_columns:
                numeric_columns.append(aggregate.column)
    numeric_states: list[dict[str, str]] = []
    for index, column in enumerate(numeric_columns):
        sum_col = _unique_hidden_name(f"sum_{index}", occupied)
        occupied.add(sum_col)
        non_null_count_col = _unique_hidden_name(
            f"non_null_count_{index}", occupied
        )
        occupied.add(non_null_count_col)
        numeric_states.append(
            {
                "input_col": column,
                "sum_col": sum_col,
                "non_null_count_col": non_null_count_col,
            }
        )
    state_columns = (
        *group_keys,
        row_count_col,
        *(
            state_column
            for numeric_state in numeric_states
            for state_column in (
                numeric_state["sum_col"],
                numeric_state["non_null_count_col"],
            )
        ),
    )
    return {
        "group_keys": group_keys,
        "aggregates": aggregates,
        "row_count_col": row_count_col,
        "numeric_states": tuple(numeric_states),
        "state_columns": state_columns,
    }


def aggregate_state_query(
    source: QueryExpr,
    *,
    layout: Mapping[str, Any],
) -> QueryExpr:
    """Build the complete hidden-state query used as a reference computation."""

    return QueryExpr(op="aggregate_state", inputs=(source,), params=layout)


def aggregate_state_update_query(
    *,
    current_state: QueryExpr,
    inserted_rows: QueryExpr,
    retracted_rows: QueryExpr,
    layout: Mapping[str, Any],
) -> QueryExpr:
    """Build exact signed parent-delta maintenance for hidden aggregate state."""

    return QueryExpr(
        op="aggregate_state_update",
        inputs=(current_state, inserted_rows, retracted_rows),
        params=layout,
    )


def aggregate_finalize_query(
    state: QueryExpr,
    *,
    layout: Mapping[str, Any],
) -> QueryExpr:
    """Build the deterministic public projection from hidden aggregate state."""

    return QueryExpr(op="aggregate_finalize", inputs=(state,), params=layout)


def _unique_hidden_name(suffix: str, occupied: set[str]) -> str:
    candidate = f"{_HIDDEN_PREFIX}{suffix}"
    while candidate in occupied:
        candidate = f"_{candidate}"
    return candidate


__all__ = [
    "aggregate_finalize_query",
    "aggregate_layout",
    "aggregate_state_query",
    "aggregate_state_update_query",
    "is_algebraic_aggregate_query",
]
