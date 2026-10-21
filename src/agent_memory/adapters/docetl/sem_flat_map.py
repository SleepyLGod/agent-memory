"""The `sem_flat_map` lowering: DocETL's `map` asked for a list column, expanded into 0..n rows here."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from agent_memory.adapters.docetl.columns import (
    declared_outputs,
    describe_outputs,
    output_column_names,
    rows_type,
)
from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.records import require_one_record_per_row, to_records
from agent_memory.adapters.docetl.reserved import INDEX_KEY, ROWS_KEY, strip
from agent_memory.policy.logical import ColumnSpec, QueryExpr

if TYPE_CHECKING:
    from agent_memory.adapters.docetl.context import DocetlExecutionContext


def build_flat_map_config(
    query: QueryExpr,
    *,
    rows_column: str,
    skip_on_error: bool = False,
    litellm_completion_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the DocETL `map` config that asks for a list of output rows."""

    if query.op != "sem_flat_map":
        raise ValueError(f"build_flat_map_config got a {query.op!r} query")

    declared = declared_outputs(query)
    if not declared:
        raise ValueError("sem_flat_map requires at least one output column")
    instruction = str(query.params.get("instruction") or "")
    # Reason: the schema can say "an array of objects", but not how many entries or what each one means,
    #   and an empty result is the case a model most often gets wrong by inventing a row.
    shape = (
        "Return one entry in the list for each row you would emit, and an empty list when there are "
        "none."
    )
    described = describe_outputs(declared)
    prompt = f"{render_prompt(instruction)}\n\n{shape}"
    if described:
        prompt = f"{prompt}\n\n{described}"

    config: dict[str, Any] = {
        "name": query.op,
        "type": "map",
        "prompt": prompt,
        "output": {"schema": {rows_column: rows_type(declared)}},
        "skip_on_error": bool(skip_on_error),
    }
    if litellm_completion_kwargs:
        config["litellm_completion_kwargs"] = dict(litellm_completion_kwargs)
    return config


def _expand(
    records: Sequence[Mapping[str, Any]],
    *,
    index_key: str,
    rows_column: str,
    declared: Sequence[ColumnSpec],
    source_columns: Sequence[str],
    ordinal_col: str | None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Expand each record's list into rows, keeping the record's index value on every emitted row.

    The ordinal's *name* is the policy's (`ordinal_col`); its *value* is the adapter's, assigned here so a
    model is never asked for a row's position among the rows emitted for one input.
    """

    rows: list[dict[str, Any]] = []
    indexes: list[Any] = []
    for record in records:
        emitted = record.get(rows_column) or []
        if not isinstance(emitted, list):
            raise ValueError(
                f"sem_flat_map row {record.get(index_key)!r} returned {type(emitted).__name__} where "
                "the declared schema is a list, so DocETL's validation should have dropped it"
            )
        base = {column: record.get(column) for column in source_columns}
        for ordinal, item in enumerate(emitted):
            row = dict(base)
            for column in declared:
                row[column.name] = item.get(column.name)
            if ordinal_col is not None:
                row[ordinal_col] = ordinal
            rows.append(row)
            indexes.append(record[index_key])
    return rows, indexes


def execute_sem_flat_map(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: DocetlExecutionContext,
) -> Any:
    """Run one `sem_flat_map` on DocETL's `map` operation, then expand its lists into rows."""

    source = execute(query.inputs[0], inputs)
    index_key = INDEX_KEY
    rows_column = ROWS_KEY
    operation = context.build_operation("map")(
        context.handle,
        build_flat_map_config(
            query,
            rows_column=rows_column,
            skip_on_error=context.config.skip_on_error,
            litellm_completion_kwargs=context.config.litellm_completion_kwargs,
        ),
        context.model,
        context.config.max_threads,
    )
    records, _cost = operation.execute(to_records(source, index_key=index_key))
    require_one_record_per_row(source, records, op="sem_flat_map", index_key=index_key)
    ordinal_col = query.params.get("ordinal_col")
    rows, indexes = _expand(
        records,
        index_key=index_key,
        rows_column=rows_column,
        declared=declared_outputs(query),
        source_columns=tuple(source.columns),
        ordinal_col=ordinal_col,
    )

    columns = output_column_names(query, source.columns)
    if ordinal_col is not None:
        columns.append(str(ordinal_col))

    # Reason: the runtime files a semantic row's outputs by the input row's index, and one input row can
    #   now own several outputs, so the index is repeated rather than made unique, and `object` matches
    #   the dtype the runtime allocates for occurrence ids.
    frame = pd.DataFrame(rows, columns=columns)
    frame.index = pd.Index(indexes, dtype="object")
    return strip(frame)
