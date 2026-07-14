"""LOTUS-backed sem_flat_map lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.structured import (
    StructuredLMExecutor,
    output_columns,
    parse_structured_array_json,
    resolve_input_cols,
)
from agent_memory.policy.logical import ColumnSpec, QueryExpr


def execute_sem_flat_map(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute sem_flat_map as structured JSON rows-wrapper generation."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    cols = output_columns(query, operator="sem_flat_map")
    input_cols = resolve_input_cols(source, query, operator="sem_flat_map")
    executor = StructuredLMExecutor(source)
    generation = executor(
        input_cols=input_cols,
        output_cols=cols,
        instruction=str(query.params["instruction"]),
        shape="array",
        progress_bar_desc="Flat mapping",
        model_kwargs={},
        structured_max_tokens=context.config.structured_max_tokens,
        structured_parse_retries=context.config.structured_parse_retries,
        semantic_trace_dir=context.config.trace_dir(),
        operator="sem_flat_map",
    )
    return apply_flat_map_outputs(
        source,
        generation.parsed_outputs,
        cols,
        ordinal_col=query.params.get("ordinal_col"),
    )


def parse_structured_flat_map_json(
    raw_output: str,
    output_cols: Sequence[ColumnSpec],
) -> list[dict[str, Any]]:
    """Parse one sem_flat_map JSON rows-wrapper output."""

    return parse_structured_array_json(
        raw_output,
        output_cols,
        operator="sem_flat_map",
    )


def apply_flat_map_outputs(
    source: Any,
    parsed_outputs: Sequence[Sequence[Mapping[str, Any]]],
    output_cols: Sequence[ColumnSpec],
    *,
    ordinal_col: str | None = None,
) -> Any:
    """Explode per-row structured outputs while preserving source columns."""

    output_names = {column.name for column in output_cols}
    if ordinal_col is not None:
        if ordinal_col in source.columns or ordinal_col in output_names:
            raise ValueError(
                "sem_flat_map ordinal_col conflicts with an existing or output "
                f"column: {ordinal_col!r}"
            )
    rows: list[dict[str, Any]] = []
    row_indexes: list[Any] = []
    for (source_index, source_row), emitted_rows in zip(
        source.iterrows(),
        parsed_outputs,
        strict=True,
    ):
        base = source_row.to_dict()
        for ordinal, emitted in enumerate(emitted_rows):
            row = dict(base)
            for column in output_cols:
                row[column.name] = emitted[column.name]
            if ordinal_col is not None:
                row[ordinal_col] = ordinal
            rows.append(row)
            row_indexes.append(source_index)

    columns = list(source.columns)
    for column in output_cols:
        if column.name not in columns:
            columns.append(column.name)
    if ordinal_col is not None:
        columns.append(ordinal_col)
    result = pd.DataFrame(rows, columns=columns)
    result.index = pd.Index(row_indexes, dtype="object")
    return result
