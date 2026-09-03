"""LOTUS sem_map lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.tracing.semantic import write_compact_operator_trace
from agent_memory.adapters.lotus.structured import (
    StructuredLMExecutor,
    examples_dataframe,
    normalize_strategy,
    output_columns as structured_output_columns,
    parse_structured_object_json,
    resolve_input_cols as structured_resolve_input_cols,
    structured_instruction as build_structured_instruction,
    validate_model_kwargs,
)
from agent_memory.policy.logical import ColumnSpec, QueryExpr


def execute_sem_map(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute sem_map through LOTUS native or structured lowering."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    output_cols = output_columns(query)
    if len(output_cols) == 1 and context.config.prompt_batching is None:
        return execute_native_sem_map(query, source, output_cols[0], context.config)
    return execute_structured_sem_map(query, source, output_cols, context.config)


def execute_native_sem_map(
    query: QueryExpr,
    source: Any,
    output_col: ColumnSpec,
    config: LotusExecutionConfig,
) -> Any:
    """Execute single-output sem_map through LOTUS native string output."""

    map_column = temporary_map_column(source)
    mapped = source.sem_map(
        str(query.params["instruction"]),
        **native_sem_map_kwargs(config, suffix=map_column),
    )
    result = apply_sem_map_output(
        source,
        mapped,
        map_column,
        output_col,
    )
    write_compact_operator_trace(
        config.trace_dir(),
        operator="sem_map",
        event_type="operator_result",
        input_frame=source,
        output_frame=result,
        payload={
            "instruction": str(query.params["instruction"]),
            "output_col": output_col.name,
            "native_lotus": True,
        },
    )
    return result


def execute_structured_sem_map(
    query: QueryExpr,
    source: Any,
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
) -> Any:
    """Execute multi-output sem_map with a LOTUS-backed structured contract."""

    input_cols = resolve_input_cols(source, query)
    executor = StructuredLMExecutor(source)
    generation = executor(
        input_cols=input_cols,
        output_cols=tuple(output_cols),
        instruction=str(query.params["instruction"]),
        shape="object",
        system_prompt=config.sem_map_system_prompt,
        examples=config.sem_map_examples,
        strategy=config.sem_map_strategy,
        safe_mode=config.sem_map_safe_mode,
        return_explanations=False,
        progress_bar_desc=config.sem_map_progress_bar_desc,
        model_kwargs=dict(config.sem_map_model_kwargs),
        structured_max_tokens=config.structured_max_tokens,
        structured_parse_retries=config.structured_parse_retries,
        semantic_trace_dir=config.trace_dir(),
        prompt_batching=config.prompt_batching,
    )
    return apply_structured_map_outputs(
        source,
        generation.parsed_outputs,
        output_cols,
    )


def apply_sem_map_output(
    source: Any,
    mapped: Any,
    map_column: str,
    output_col: ColumnSpec,
    *,
    return_explanations: bool = False,
    return_raw_outputs: bool = False,
) -> Any:
    """Copy LOTUS native sem_map string output into the requested column."""

    result = source.copy()
    result[output_col.name] = mapped[map_column]
    if return_explanations:
        explanation_col = "explanation" + map_column
        if explanation_col in mapped.columns:
            result[f"explanation_{output_col.name}"] = mapped[explanation_col]
    if return_raw_outputs:
        raw_output_col = "raw_output" + map_column
        if raw_output_col in mapped.columns:
            result[f"raw_output_{output_col.name}"] = mapped[raw_output_col]
    return result


def temporary_map_column(source: Any) -> str:
    """Return a LOTUS sem_map output column that will not overwrite input."""

    existing = set(str(column) for column in getattr(source, "columns", ()))
    base = "_agent_memory_map"
    candidate = base
    suffix = 1
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def output_columns(query: QueryExpr) -> tuple[ColumnSpec, ...]:
    """Return required output columns for sem_map."""

    return structured_output_columns(query, operator="sem_map")


def single_output_column(query: QueryExpr) -> ColumnSpec:
    """Return the one output column supported by LOTUS native sem_map."""

    output_cols = output_columns(query)
    if len(output_cols) != 1:
        raise NotImplementedError(
            "LotusAdapter currently supports one output column per sem_map; "
            "chain sem_map calls or implement structured map lowering."
        )
    return output_cols[0]


def native_sem_map_kwargs(config: LotusExecutionConfig, *, suffix: str) -> dict[str, Any]:
    """Build kwargs forwarded to LOTUS native df.sem_map."""

    model_kwargs = dict(config.sem_map_model_kwargs)
    kwargs: dict[str, Any] = {
        "suffix": suffix,
        "system_prompt": config.sem_map_system_prompt,
        "examples": examples_dataframe(config.sem_map_examples),
        "strategy": normalize_strategy(config.sem_map_strategy),
        "safe_mode": config.sem_map_safe_mode,
        "return_explanations": False,
        "return_raw_outputs": False,
        "progress_bar_desc": config.sem_map_progress_bar_desc,
    }
    validate_model_kwargs(model_kwargs, reserved=set(kwargs), operator="sem_map")
    kwargs.update(model_kwargs)
    return kwargs


def resolve_input_cols(source: Any, query: QueryExpr) -> tuple[str, ...]:
    """Resolve sem_map input columns for structured lowering."""

    return structured_resolve_input_cols(source, query, operator="sem_map")


def structured_instruction(instruction: str, output_cols: Sequence[ColumnSpec]) -> str:
    """Append the structured output contract to a sem_map instruction."""

    return build_structured_instruction(
        instruction,
        output_cols,
        shape="object",
    )


def parse_structured_map_json(
    raw_output: str,
    output_cols: Sequence[ColumnSpec],
    *,
    require_explanation: bool = False,
) -> dict[str, Any]:
    """Parse and validate one structured sem_map JSON output."""

    parsed, _explanation = parse_structured_object_json(
        raw_output,
        output_cols,
        require_explanation=require_explanation,
        operator="sem_map",
    )
    return parsed


def apply_structured_map_outputs(
    source: Any,
    parsed_outputs: Sequence[Mapping[str, Any]],
    output_cols: Sequence[ColumnSpec],
    *,
    raw_outputs: Sequence[str] | None = None,
    explanations: Sequence[str | None] | None = None,
) -> Any:
    """Apply structured sem_map outputs to a DataFrame copy."""

    result = source.copy()
    for column in output_cols:
        result[column.name] = [parsed[column.name] for parsed in parsed_outputs]
    if raw_outputs is not None:
        result["raw_output_sem_map"] = list(raw_outputs)
    if explanations is not None:
        result["explanation_sem_map"] = list(explanations)
    return result
