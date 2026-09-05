"""Deterministic execution for exact relational aggregate state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd

from agent_memory.policy.aggregates import (
    AggregateSpec,
    AvgAggregateSpec,
    CountAggregateSpec,
    SumAggregateSpec,
    aggregate_output_names,
    is_algebraic_aggregate_spec,
)
from agent_memory.policy.logical import QueryExpr

def is_algebraic_aggregate_query(query: QueryExpr) -> bool:
    """Return whether ``query`` contains only exact numeric aggregates."""

    aggregates = tuple(query.params.get("aggregates", ()))
    return (
        query.op == "agg"
        and bool(aggregates)
        and all(is_algebraic_aggregate_spec(spec) for spec in aggregates)
    )


def execute_algebraic_aggregate(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> pd.DataFrame:
    """Execute one complete global or ordinary grouped aggregate query."""

    if not is_algebraic_aggregate_query(query):
        raise TypeError("algebraic aggregate execution requires count/sum/avg specs")
    source_query = query.inputs[0]
    if source_query.op == "sem_groupby":
        raise NotImplementedError(
            "sem_groupby numeric algebraic aggregates are not supported"
        )
    group_keys = (
        tuple(str(key) for key in source_query.params["keys"])
        if source_query.op == "group_by"
        else ()
    )
    source = execute(source_query, inputs)
    layout = _direct_layout(
        group_keys=group_keys,
        aggregates=tuple(query.params["aggregates"]),
    )
    empty_source = source.iloc[0:0].copy()
    state = _update_state(
        pd.DataFrame(columns=layout["state_columns"]),
        source,
        empty_source,
        layout,
    )
    return _finalize_state(state, layout)


def execute_aggregate_state(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> pd.DataFrame:
    """Build complete hidden aggregate state from one input relation."""

    source = execute(query.inputs[0], inputs)
    return _update_state(
        pd.DataFrame(columns=query.params["state_columns"]),
        source,
        source.iloc[0:0].copy(),
        query.params,
    )


def execute_aggregate_state_update(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> pd.DataFrame:
    """Merge parent insertions and retractions into hidden aggregate state."""

    if len(query.inputs) != 3:
        raise ValueError("aggregate_state_update expects current, inserted, and retracted inputs")
    current, inserted, retracted = (
        execute(input_query, inputs) for input_query in query.inputs
    )
    return _update_state(current, inserted, retracted, query.params)


def execute_aggregate_finalize(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> pd.DataFrame:
    """Project hidden aggregate state into the declared public columns."""

    state = execute(query.inputs[0], inputs)
    return _finalize_state(state, query.params)


def _direct_layout(
    *,
    group_keys: tuple[str, ...],
    aggregates: tuple[AggregateSpec, ...],
) -> dict[str, Any]:
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


def _unique_hidden_name(suffix: str, occupied: set[str]) -> str:
    candidate = f"__am_aggregate_{suffix}"
    while candidate in occupied:
        candidate = f"_{candidate}"
    return candidate


def _update_state(
    current: pd.DataFrame,
    inserted: pd.DataFrame,
    retracted: pd.DataFrame,
    layout: Mapping[str, Any],
) -> pd.DataFrame:
    group_keys = tuple(str(key) for key in layout["group_keys"])
    row_count_col = str(layout["row_count_col"])
    numeric_states = tuple(layout["numeric_states"])
    state_columns = tuple(str(column) for column in layout["state_columns"])
    _require_columns(current, state_columns, relation="aggregate state")
    required_input = (
        *group_keys,
        *(str(state["input_col"]) for state in numeric_states),
    )
    _require_columns(inserted, required_input, relation="inserted rows")
    _require_columns(retracted, required_input, relation="retracted rows")

    states: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in current.to_dict(orient="records"):
        key = _group_key(record, group_keys)
        if key in states:
            raise ValueError("aggregate state contains duplicate group keys")
        states[key] = dict(record)
    if not group_keys and not states:
        states[()] = _empty_accumulator(layout, key_values={})

    _apply_rows(states, inserted, layout, sign=1)
    _apply_rows(states, retracted, layout, sign=-1)

    rows: list[dict[str, Any]] = []
    for key, state in states.items():
        row_count = _require_integral_state(state[row_count_col], column=row_count_col)
        if row_count < 0:
            raise ValueError(f"aggregate state row count became negative for group {key!r}")
        for numeric_state in numeric_states:
            non_null_col = str(numeric_state["non_null_count_col"])
            non_null_count = _require_integral_state(
                state[non_null_col], column=non_null_col
            )
            if non_null_count < 0 or non_null_count > row_count:
                raise ValueError(
                    f"aggregate state non-null count is invalid for group {key!r}"
                )
        if group_keys and row_count == 0:
            continue
        rows.append(state)
    if not group_keys and not rows:
        rows.append(_empty_accumulator(layout, key_values={}))
    return pd.DataFrame(rows, columns=state_columns)


def _apply_rows(
    states: dict[tuple[Any, ...], dict[str, Any]],
    rows: pd.DataFrame,
    layout: Mapping[str, Any],
    *,
    sign: int,
) -> None:
    group_keys = tuple(str(key) for key in layout["group_keys"])
    row_count_col = str(layout["row_count_col"])
    numeric_states = tuple(layout["numeric_states"])
    for record in rows.to_dict(orient="records"):
        key = _group_key(record, group_keys)
        state = states.get(key)
        if state is None:
            state = _empty_accumulator(
                layout,
                key_values={key_name: record[key_name] for key_name in group_keys},
            )
            states[key] = state
        state[row_count_col] = _require_integral_state(
            state[row_count_col], column=row_count_col
        ) + sign
        for numeric_state in numeric_states:
            input_col = str(numeric_state["input_col"])
            sum_col = str(numeric_state["sum_col"])
            non_null_col = str(numeric_state["non_null_count_col"])
            value = record[input_col]
            if _is_null(value):
                continue
            _require_numeric(value, column=input_col)
            state[sum_col] = state[sum_col] + sign * value
            state[non_null_col] = _require_integral_state(
                state[non_null_col], column=non_null_col
            ) + sign


def _empty_accumulator(
    layout: Mapping[str, Any],
    *,
    key_values: Mapping[str, Any],
) -> dict[str, Any]:
    state = dict(key_values)
    state[str(layout["row_count_col"])] = 0
    for numeric_state in layout["numeric_states"]:
        state[str(numeric_state["sum_col"])] = 0
        state[str(numeric_state["non_null_count_col"])] = 0
    return state


def _finalize_state(
    state: pd.DataFrame,
    layout: Mapping[str, Any],
) -> pd.DataFrame:
    group_keys = tuple(str(key) for key in layout["group_keys"])
    aggregates = tuple(layout["aggregates"])
    state_columns = tuple(str(column) for column in layout["state_columns"])
    _require_columns(state, state_columns, relation="aggregate state")
    numeric_by_input = {
        str(numeric_state["input_col"]): numeric_state
        for numeric_state in layout["numeric_states"]
    }
    rows: list[dict[str, Any]] = []
    for state_record in state.to_dict(orient="records"):
        output = {key: state_record[key] for key in group_keys}
        for aggregate in aggregates:
            if isinstance(aggregate, CountAggregateSpec):
                output[aggregate.output_col] = _require_integral_state(
                    state_record[str(layout["row_count_col"])],
                    column=str(layout["row_count_col"]),
                )
                continue
            if isinstance(aggregate, (SumAggregateSpec, AvgAggregateSpec)):
                numeric_state = numeric_by_input[aggregate.column]
                non_null_count = _require_integral_state(
                    state_record[str(numeric_state["non_null_count_col"])],
                    column=str(numeric_state["non_null_count_col"]),
                )
                value = (
                    None
                    if non_null_count == 0
                    else state_record[str(numeric_state["sum_col"])]
                )
                if isinstance(aggregate, AvgAggregateSpec) and value is not None:
                    value = value / non_null_count
                output[aggregate.output_col] = value
                continue
            raise TypeError(f"Unsupported aggregate spec: {type(aggregate).__name__}")
        rows.append(output)
    columns = [*group_keys, *(name for spec in aggregates for name in aggregate_output_names(spec))]
    return pd.DataFrame(rows, columns=columns)


def _group_key(
    record: Mapping[str, Any], group_keys: Sequence[str]
) -> tuple[Any, ...]:
    return tuple(_key_value(record[key]) for key in group_keys)


def _key_value(value: Any) -> Any:
    if _is_null(value):
        return ("null",)
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("aggregate group key values must be hashable") from error
    return (type(value).__module__, type(value).__qualname__, value)


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    relation: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{relation} is missing columns: {missing}")


def _require_numeric(value: Any, *, column: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
        raise TypeError(
            f"aggregate input column {column!r} requires numeric non-null values"
        )


def _require_integral_state(value: Any, *, column: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"aggregate state column {column!r} must contain integers")
    return int(value)


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


__all__ = [
    "execute_aggregate_finalize",
    "execute_aggregate_state",
    "execute_aggregate_state_update",
    "execute_algebraic_aggregate",
    "is_algebraic_aggregate_query",
]
