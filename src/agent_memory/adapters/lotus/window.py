"""Deterministic count-window lowering for the LOTUS adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from agent_memory.logical import QueryExpr
from agent_memory.query_schema import output_columns
from agent_memory.window import WINDOW_SOURCE_INPUT, completed_count_windows


def execute_window_source(inputs: Mapping[str, Any]) -> Any:
    """Bind the current window-local source relation."""

    try:
        return inputs[WINDOW_SOURCE_INPUT]
    except KeyError as error:
        raise KeyError("window_source can only be executed inside process_window") from error


def execute_count_window(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Reject direct count_window execution outside process_window."""

    raise NotImplementedError("count_window must be consumed by process_window")


def execute_process_window(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute a process_window query over all completed count windows."""

    if len(query.inputs) != 2:
        raise ValueError("process_window expects count_window and process body inputs")
    window_query, process_query = query.inputs
    if window_query.op != "count_window" or len(window_query.inputs) != 1:
        raise ValueError("process_window first input must be count_window")

    source = execute(window_query.inputs[0], inputs)
    windows, _next_start = completed_count_windows(source, window_query.params)
    if not windows:
        return pd.DataFrame(
            columns=output_columns(
                process_query,
                window_source_columns=_source_columns(source),
            )
        )

    results = []
    for window in windows:
        window_inputs = dict(inputs)
        window_inputs[WINDOW_SOURCE_INPUT] = window.frame
        results.append(execute(process_query, window_inputs))
    if not results:
        return pd.DataFrame(
            columns=output_columns(
                process_query,
                window_source_columns=_source_columns(source),
            )
        )
    return pd.concat(results, ignore_index=True)


def _source_columns(source: Any) -> tuple[str, ...]:
    """Return string column names for a dataframe-like source."""

    return tuple(str(column) for column in getattr(source, "columns", ()))

