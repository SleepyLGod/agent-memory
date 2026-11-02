"""The `sem_flat_map` lowering: DocETL's `map` asked for a list column, expanded into 0..n rows here."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from agent_memory.adapters.docetl.columns import (
    declared_outputs,
    describe_outputs,
    rows_type,
    selected_inputs,
)
from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.records import (
    publish_rows,
    require_one_record_per_row,
    source_positions,
    to_records,
)
from agent_memory.adapters.docetl.reserved import INDEX_KEY, ROWS_KEY
from agent_memory.adapters.docetl.specs import op_config
from agent_memory.policy.logical import ColumnSpec, QueryExpr

if TYPE_CHECKING:
    from agent_memory.adapters.docetl.context import DocetlExecutionContext


def build_flat_map_config(
    query: QueryExpr,
    *,
    columns: Sequence[str],
    rows_column: str,
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
    outputs = [column.name for column in declared]
    prompt = f"{render_prompt(instruction, columns=columns, outputs=outputs)}\n\n{shape}"
    if described:
        prompt = f"{prompt}\n\n{described}"

    return op_config(query.op, "map", prompt, {rows_column: rows_type(declared)})


def _expand(
    source: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    *,
    index_key: str,
    rows_column: str,
    declared: Sequence[ColumnSpec],
    ordinal_col: str | None,
) -> tuple[list[int], dict[str, list[Any]]]:
    """Expand each record's list into one row per entry, remembering which source row each came from.

    `positions` has one entry per emitted row, so a record's position repeats when its list is longer
    than one: the source row is named rather than copied, and `publish_rows` selects it. `written` holds
    only what the model produced. The ordinal's *name* is the policy's (`ordinal_col`); its *value* is
    the adapter's, assigned here so a model is never asked for a row's position among the rows emitted
    for one input.
    """

    positions: list[int] = []
    written: dict[str, list[Any]] = {column.name: [] for column in declared}
    if ordinal_col is not None:
        written[str(ordinal_col)] = []
    for position, record in zip(
        source_positions(source, records, index_key=index_key), records, strict=True
    ):
        emitted = record.get(rows_column) or []
        if not isinstance(emitted, list):
            raise ValueError(
                f"sem_flat_map row {record.get(index_key)!r} returned {type(emitted).__name__} where "
                "the declared schema is a list, so DocETL's validation should have dropped it"
            )
        for ordinal, item in enumerate(emitted):
            positions.append(position)
            for column in declared:
                written[column.name].append(item.get(column.name))
            if ordinal_col is not None:
                written[str(ordinal_col)].append(ordinal)
    return positions, written


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
    records = context.run_operation(
        "map",
        build_flat_map_config(
            query,
            columns=selected_inputs(query, source.columns),
            rows_column=rows_column,
        ),
        to_records(source, index_key=index_key),
    )
    # Reason: one input row may legitimately produce zero rows here, which is why the check below
    #   counts records rather than emitted rows: a record holding an empty list answered, while a
    #   missing record is one the engine dropped.
    require_one_record_per_row(source, records, op="sem_flat_map", index_key=index_key)
    positions, written = _expand(
        source,
        records,
        index_key=index_key,
        rows_column=rows_column,
        declared=declared_outputs(query),
        ordinal_col=query.params.get("ordinal_col"),
    )

    # Reason: the runtime files a semantic row's outputs by the input row's index, and one input row can
    #   now own several outputs, so the index is repeated rather than made unique, and `object` matches
    #   the dtype the runtime allocates for occurrence ids.
    return publish_rows(source, positions, written=written, index_dtype="object")
