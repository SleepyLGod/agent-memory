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
