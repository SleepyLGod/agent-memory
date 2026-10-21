"""The `sem_filter` lowering: a row-local semantic predicate onto DocETL's `filter` operation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.records import from_records, to_records
from agent_memory.adapters.docetl.reserved import INDEX_KEY, KEEP_KEY, strip
from agent_memory.policy.logical import QueryExpr

if TYPE_CHECKING:
    from agent_memory.adapters.docetl.context import DocetlExecutionContext

# Based on: `docetl.operations.filter::FilterOperation._handle_result`, which reads exactly one key from
#   each row and pops it, so the name is reserved in `records.py` rather than chosen here.
def build_filter_config(
    query: QueryExpr,
    *,
    skip_on_error: bool = False,
    litellm_completion_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the DocETL `filter` config for one row-local `sem_filter` query."""

    if query.op != "sem_filter":
        raise ValueError(f"build_filter_config got a {query.op!r} query")

    prompt = render_prompt(str(query.params.get("instruction") or ""))

    config: dict[str, Any] = {
        "name": query.op,
        "type": "filter",
        "prompt": prompt,
        "output": {"schema": {KEEP_KEY: "bool"}},
        "skip_on_error": bool(skip_on_error),
    }
    if litellm_completion_kwargs:
        config["litellm_completion_kwargs"] = dict(litellm_completion_kwargs)
    return config


def execute_sem_filter(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: DocetlExecutionContext,
) -> Any:
    """Run one row-local `sem_filter` on DocETL's `filter` operation."""

    source = execute(query.inputs[0], inputs)
    index_key = INDEX_KEY
    operation = context.build_operation("filter")(
        context.handle,
        build_filter_config(
            query,
            skip_on_error=context.config.skip_on_error,
            litellm_completion_kwargs=context.config.litellm_completion_kwargs,
        ),
        context.model,
        context.config.max_threads,
    )
    kept, _cost = operation.execute(to_records(source, index_key=index_key))
    # usage is read from the runner's own counters, not from the cost DocETL returns here
    # (`src/agent_memory/adapters/docetl/context.py::DocetlExecutionContext.usage_snapshot`).
    return strip(from_records(kept, index_key=index_key, columns=source.columns))
