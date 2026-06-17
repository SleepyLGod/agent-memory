"""Deterministic relational lowering for the LOTUS adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from agent_memory.logical import QueryExpr


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
