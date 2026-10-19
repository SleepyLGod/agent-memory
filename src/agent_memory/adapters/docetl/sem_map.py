"""The `sem_map` lowering: a row-local transform that adds or replaces columns, one row in and one out."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from agent_memory.adapters.docetl.columns import (
    declared_outputs,
    describe_outputs,
    output_column_names,
    scalar_schema,
)
from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.records import (
    from_records,
    require_one_record_per_row,
    to_records,
)
from agent_memory.adapters.docetl.reserved import INDEX_KEY, strip
from agent_memory.policy.logical import QueryExpr

if TYPE_CHECKING:
    from agent_memory.adapters.docetl.context import DocetlExecutionContext


def build_map_config(
    query: QueryExpr,
    *,
    describe: bool | None = None,
    skip_on_error: bool = False,
    litellm_completion_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the DocETL `map` config for one `sem_map` query."""

    if query.op != "sem_map":
        raise ValueError(f"build_map_config got a {query.op!r} query")

    declared = declared_outputs(query)
    prompt = render_prompt(str(query.params.get("instruction") or ""))
    if describe is None:
        describe = len(declared) > 1
    if describe:
        described = describe_outputs(declared)
        if described:
            prompt = f"{prompt}\n\n{described}"

    config: dict[str, Any] = {
        "name": query.op,
        "type": "map",
        "prompt": prompt,
        "output": {"schema": scalar_schema(declared)},
        "skip_on_error": bool(skip_on_error),
    }
    if litellm_completion_kwargs:
        config["litellm_completion_kwargs"] = dict(litellm_completion_kwargs)
    return config


def execute_sem_map(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: DocetlExecutionContext,
) -> Any:
    """Run one `sem_map` on DocETL's `map` operation."""

    source = execute(query.inputs[0], inputs)
    index_key = INDEX_KEY
    operation = context.build_operation("map")(
        context.handle,
        build_map_config(
            query,
            skip_on_error=context.config.skip_on_error,
            litellm_completion_kwargs=context.config.litellm_completion_kwargs,
        ),
        context.model,
        context.config.max_threads,
    )
    mapped, _cost = operation.execute(to_records(source, index_key=index_key))
    require_one_record_per_row(source, mapped, op="sem_map", index_key=index_key)
    return strip(
        from_records(
            mapped,
            index_key=index_key,
            columns=output_column_names(query, source.columns),
        )
    )
