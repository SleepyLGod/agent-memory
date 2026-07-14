"""LOTUS sem_topk lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.structured import normalize_strategy
from agent_memory.adapters.lotus.sem_topk_listwise import execute_listwise_topk
from agent_memory.policy.logical import QueryExpr
from agent_memory.tracing.semantic import write_compact_operator_trace

LOTUS_PAIRWISE_METHODS = {
    "pairwise-naive": "naive",
    "pairwise-quick": "quick",
    "pairwise-heap": "heap",
}


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
    """Execute the configured semantic top-k lowering."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    instruction = topk_instruction(source, str(query.params["instruction"]))
    config = context.config
    if config.sem_topk_method == "listwise":
        listwise_result = execute_listwise_topk(
            source,
            instruction=instruction,
            k=query.params["k"],
            context=context,
        )
        result = listwise_result.frame
        method_payload = {
            "candidate_count": len(source),
            "selected_ids": list(listwise_result.selected_ids),
            "retry_count": listwise_result.retry_count,
        }
    else:
        lotus_method = LOTUS_PAIRWISE_METHODS.get(config.sem_topk_method)
        if lotus_method is None:
            raise ValueError(f"Unsupported sem_topk method {config.sem_topk_method!r}")
        result = source.sem_topk(
            instruction,
            K=query.params["k"],
            method=lotus_method,
            strategy=normalize_strategy(config.sem_topk_strategy),
            cascade_threshold=config.sem_topk_cascade_threshold,
            return_stats=config.sem_topk_return_stats,
            safe_mode=config.sem_topk_safe_mode,
            return_explanations=config.sem_topk_return_explanations,
        )
        method_payload = {"candidate_count": len(source)}
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
            "sem_topk_method": config.sem_topk_method,
            **method_payload,
        },
    )
    return result
