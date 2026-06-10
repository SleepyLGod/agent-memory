"""LOTUS sem_topk lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.tracing.semantic import write_compact_operator_trace
from agent_memory.adapters.lotus.structured import normalize_strategy
from agent_memory.logical import QueryExpr


def topk_instruction(source: Any, instruction: str) -> str:
    """Convert plain user queries into LOTUS column-aware expressions."""

    if "{" in instruction and "}" in instruction:
        return instruction

    columns = [str(column) for column in getattr(source, "columns", ())]
    if not columns:
        raise ValueError("sem_topk requires at least one input column")

    row_reference = ", ".join(f"{{{column}}}" for column in columns)
    return f"{row_reference} is relevant to: {instruction}"


def execute_sem_topk(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute LOTUS native semantic top-k retrieval."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    instruction = topk_instruction(source, str(query.params["instruction"]))
    config = context.config
    result = source.sem_topk(
        instruction,
        K=query.params["k"],
        method=config.sem_topk_method,
        strategy=normalize_strategy(config.sem_topk_strategy),
        cascade_threshold=config.sem_topk_cascade_threshold,
        return_stats=config.sem_topk_return_stats,
        safe_mode=config.sem_topk_safe_mode,
        return_explanations=config.sem_topk_return_explanations,
    )
    write_compact_operator_trace(
        config.trace_dir(),
        operator="sem_topk",
        event_type="operator_result",
        input_frame=source,
        output_frame=result,
        payload={
            "instruction": str(query.params["instruction"]),
            "lowered_instruction": instruction,
            "k": query.params["k"],
            "method": config.sem_topk_method,
        },
    )
    return result
