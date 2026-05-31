"""LOTUS sem_filter lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.structured import examples_dataframe, normalize_strategy
from agent_memory.logical import QueryExpr


def execute_sem_filter(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute LOTUS native semantic filtering."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    return source.sem_filter(
        query.params["instruction"],
        **native_sem_filter_kwargs(context.config),
    )


def native_sem_filter_kwargs(config: LotusExecutionConfig) -> dict[str, Any]:
    """Return LOTUS sem_filter kwargs from adapter execution config."""

    return {
        "return_raw_outputs": False,
        "return_explanations": False,
        "return_stats": False,
        "default": config.sem_filter_default,
        "examples": examples_dataframe(config.sem_filter_examples),
        "helper_examples": examples_dataframe(config.sem_filter_helper_examples),
        "strategy": normalize_strategy(config.sem_filter_strategy),
        "cascade_args": sem_filter_cascade_args_from_mapping(
            config.sem_filter_cascade_args
        ),
        "safe_mode": config.sem_filter_safe_mode,
        "progress_bar_desc": config.sem_filter_progress_bar_desc,
        "additional_cot_instructions": config.sem_filter_additional_cot_instructions,
    }


def sem_filter_cascade_args_from_mapping(value: Any) -> Any:
    """Convert row-like cascade args into LOTUS CascadeArgs."""

    if value is None:
        return None
    from lotus.types import CascadeArgs

    if isinstance(value, CascadeArgs):
        return value
    return CascadeArgs(**dict(value))
