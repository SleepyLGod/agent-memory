"""Deterministic relational lowering for the LOTUS adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

import pandas as pd

from agent_memory.logical import QueryExpr
from agent_memory.query_schema import output_columns
from agent_memory.window import over_frames


def execute_select(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic column projection."""

    source = execute(query.inputs[0], inputs)
    return source.loc[:, list(query.params["columns"])].copy()


def execute_concat(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute union-all row append semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    _require_matching_columns(left, right, op="concat")
    return pd.concat([left, right], ignore_index=True)


def execute_union(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute exact row-set union semantics."""

    concatenated = execute_concat(query, inputs, execute)
    return concatenated.drop_duplicates(ignore_index=True)


def execute_subtract(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute exact row-set difference semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    _require_matching_columns(left, right, op="subtract")
    if right.empty:
        return left.copy().reset_index(drop=True)

    marker = "_agent_memory_subtract_marker"
    right_unique = right.drop_duplicates()
    merged = left.merge(right_unique.assign(**{marker: True}), how="left")
    result = merged[merged[marker].isna()].drop(columns=[marker])
    return result.loc[:, list(left.columns)].reset_index(drop=True)


def execute_join(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic same-key relational join semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    keys = tuple(str(column) for column in query.params["on"])
    how = str(query.params.get("how", "inner"))
    _require_supported_join_how(how)
    _require_join_keys(left, right, keys)
    _require_non_null_join_keys(left, keys, side="left")
    _require_non_null_join_keys(right, keys, side="right")

    return left.merge(
        right,
        how=how,
        on=list(keys),
        suffixes=(":left", ":right"),
        sort=False,
    ).reset_index(drop=True)


def execute_drop_duplicates(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute exact duplicate-row removal."""

    source = execute(query.inputs[0], inputs)
    return source.drop_duplicates(ignore_index=True)


def execute_array_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic array-of-records aggregation."""

    if query.inputs[0].op == "over":
        return execute_over_array_agg(query, inputs, execute)

    source = execute(query.inputs[0], inputs)
    columns = tuple(str(column) for column in query.params["columns"])
    output_col = str(query.params["output_col"])
    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(f"array_agg input columns not found in DataFrame: {missing}")

    projected = source.loc[:, list(columns)]
    clean = projected.where(pd.notna(projected), None)
    records = clean.to_dict(orient="records")
    value = json.dumps(records, ensure_ascii=False, default=_json_default, allow_nan=False)
    result = pd.DataFrame([{output_col: value}], columns=[output_col])
    for col in source.columns:
        if col in columns or col in result.columns:
            continue
        try:
            is_constant = source[col].nunique(dropna=True) <= 1
        except TypeError:
            is_constant = False
        if is_constant:
            result[col] = source[col].iloc[0] if not source.empty else None
    return result


def execute_over_array_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute row-preserving over-window array aggregation."""

    over_query = query.inputs[0]
    emit_source = execute(over_query.inputs[0], inputs)
    frame_source_query = over_query.params.get("frame_source")
    frame_source = (
        execute(frame_source_query, inputs)
        if isinstance(frame_source_query, QueryExpr)
        else emit_source
    )
    columns = tuple(str(column) for column in query.params["columns"])
    output_col = str(query.params["output_col"])
    emit_columns = tuple(output_columns(over_query.inputs[0])) or tuple(emit_source.columns)
    missing = [column for column in columns if column not in frame_source.columns]
    if missing:
        raise ValueError(f"over array_agg input columns not found in DataFrame: {missing}")
    missing_emit = [column for column in emit_columns if column not in emit_source.columns]
    if missing_emit:
        raise ValueError(f"over array_agg emit columns not found in DataFrame: {missing_emit}")

    rows: list[dict[str, Any]] = []
    for frame in over_frames(emit_source, frame_source, over_query.params):
        projected = frame.frame.loc[:, list(columns)]
        clean = projected.where(pd.notna(projected), None)
        value = json.dumps(
            clean.to_dict(orient="records"),
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        row = frame.emit_row.loc[list(emit_columns)].to_dict()
        row[output_col] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=[*emit_columns, output_col])


def execute_array_cat(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic JSON array-state concatenation."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    column = str(query.params["column"])
    _require_array_column(left, column, side="left")
    _require_array_column(right, column, side="right")
    _require_at_most_one_row(left, op="array_cat", side="left")
    _require_at_most_one_row(right, op="array_cat", side="right")

    if left.empty and right.empty:
        return pd.DataFrame(columns=[column])
    if left.empty:
        _load_array_json(right.iloc[0][column], column=column, side="right")
        return right.loc[:, [column]].copy().reset_index(drop=True)
    if right.empty:
        _load_array_json(left.iloc[0][column], column=column, side="left")
        return left.loc[:, [column]].copy().reset_index(drop=True)

    left_items = _load_array_json(left.iloc[0][column], column=column, side="left")
    right_items = _load_array_json(right.iloc[0][column], column=column, side="right")
    value = json.dumps(
        [*left_items, *right_items],
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )
    return pd.DataFrame([{column: value}], columns=[column])


def _execute_binary_inputs(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> tuple[Any, Any]:
    """Execute and validate a binary relational expression."""

    if len(query.inputs) != 2:
        raise ValueError(f"{query.op} expects exactly two inputs")
    return execute(query.inputs[0], inputs), execute(query.inputs[1], inputs)


def _require_matching_columns(left: Any, right: Any, *, op: str) -> None:
    """Require identical column order for exact row-set operators."""

    if list(left.columns) != list(right.columns):
        raise ValueError(f"{op} requires matching columns")


def _require_supported_join_how(how: str) -> None:
    """Require a pandas relational join mode supported by the public API."""

    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError("join how must be one of: inner, left, right, outer")


def _require_join_keys(left: Any, right: Any, keys: tuple[str, ...]) -> None:
    """Require join key columns to exist on both sides."""

    missing_left = [key for key in keys if key not in left.columns]
    missing_right = [key for key in keys if key not in right.columns]
    if missing_left or missing_right:
        details = []
        if missing_left:
            details.append(f"left missing {missing_left}")
        if missing_right:
            details.append(f"right missing {missing_right}")
        raise ValueError(f"join key columns must exist on both sides: {', '.join(details)}")


def _require_non_null_join_keys(frame: Any, keys: tuple[str, ...], *, side: str) -> None:
    """Reject null join keys to avoid pandas null-null matching surprises."""

    if frame.loc[:, list(keys)].isna().any().any():
        raise ValueError(f"join key columns cannot contain null values on {side} side")


def _require_array_column(frame: Any, column: str, *, side: str) -> None:
    """Require one JSON array column for aggregate-state concatenation."""

    if column not in frame.columns:
        raise ValueError(f"array_cat {side} input is missing column {column!r}")


def _require_at_most_one_row(frame: Any, *, op: str, side: str) -> None:
    """Require a one-row aggregate-state relation, allowing empty state."""

    if len(frame) > 1:
        raise ValueError(f"{op} expects {side} input to contain at most one row")


def _load_array_json(value: Any, *, column: str, side: str) -> list[Any]:
    """Parse one JSON array aggregate-state value."""

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"array_cat {side} value in column {column!r} must be a JSON array"
        ) from error
    if not isinstance(parsed, list):
        raise ValueError(
            f"array_cat {side} value in column {column!r} must be a JSON array"
        )
    return parsed


def _json_default(value: Any) -> Any:
    """Convert pandas/numpy scalar values before string fallback."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
