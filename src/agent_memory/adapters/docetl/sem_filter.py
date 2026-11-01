"""The `sem_filter` lowering: a row-local semantic predicate onto DocETL's `filter` operation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.records import from_records, to_records
from agent_memory.adapters.docetl.reserved import INDEX_KEY, KEEP_KEY
from agent_memory.adapters.docetl.specs import op_config
from agent_memory.policy.logical import QueryExpr

if TYPE_CHECKING:
    from agent_memory.adapters.docetl.context import DocetlExecutionContext

# Based on: `docetl.operations.filter::FilterOperation._handle_result`, which reads exactly one key from
#   each row and pops it, so the name is reserved in `records.py` rather than chosen here.
def build_filter_config(query: QueryExpr, *, columns: Sequence[str]) -> dict[str, Any]:
    """Return the DocETL `filter` config for one row-local `sem_filter` query."""

    if query.op != "sem_filter":
        raise ValueError(f"build_filter_config got a {query.op!r} query")

    prompt = render_prompt(str(query.params.get("instruction") or ""), columns=columns)

    return op_config(query.op, "filter", prompt, {KEEP_KEY: "bool"})


def execute_sem_filter(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: DocetlExecutionContext,
) -> Any:
    """Run one row-local `sem_filter` on DocETL's `filter` operation."""

    source = execute(query.inputs[0], inputs)
    index_key = INDEX_KEY
    kept = context.run_operation(
        "filter",
        build_filter_config(query, columns=tuple(str(column) for column in source.columns)),
        to_records(source, index_key=index_key),
    )
    return from_records(kept, index_key=index_key, source=source, written=())
