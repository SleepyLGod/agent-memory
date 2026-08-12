"""LOTUS sem_filter lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.pair_execution import (
    PairCandidateSelection,
    select_semantic_pair_candidates,
    write_semantic_pair_execution_trace,
)
from agent_memory.tracing.semantic import write_compact_operator_trace
from agent_memory.tracing.semantic import query_digest
from agent_memory.adapters.lotus.structured import examples_dataframe, normalize_strategy
from agent_memory.policy.logical import QueryExpr

QUALIFIED_PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*):([A-Za-z_][A-Za-z0-9_]*)\}(?!\})"
)


def execute_sem_filter(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute LOTUS native semantic filtering."""

    digest = query_digest(query)
    profile = context.config.semantic_pair_profiles.get(digest)
    if profile is not None and profile.mode in {"search-filter", "proxy-only"}:
        if (
            profile.mode == "proxy-only"
            and context.config.sem_filter_cascade_args is not None
        ):
            raise ValueError(
                "sem_filter proxy-only cannot be combined with LOTUS cascade"
            )
        if context.pair_embedding_provider is None:
            raise ValueError(
                f"{profile.mode} requires a pair embedding provider"
            )
    context.configure()
    source = execute(query.inputs[0], inputs)
    selection: PairCandidateSelection | None = None
    oracle_source = source
    if profile is not None and profile.mode in {"search-filter", "proxy-only"}:
        assert context.pair_embedding_provider is not None
        selection = select_semantic_pair_candidates(
            source,
            profile=profile,
            embedding_provider=context.pair_embedding_provider,
        )
        oracle_source = source.iloc[list(selection.selected_positions)].copy()
        write_semantic_pair_execution_trace(
            context.config.trace_dir(),
            operator="sem_filter",
            query_digest_value=digest,
            profile=profile,
            selection=selection,
        )
    lotus_source, instruction, restore_columns = bind_qualified_filter_columns(
        oracle_source,
        str(query.params["instruction"]),
    )
    if profile is not None and profile.mode == "proxy-only":
        result = lotus_source.copy()
    elif selection is not None and lotus_source.empty:
        result = lotus_source.copy()
    else:
        result = lotus_source.sem_filter(
            instruction,
            **native_sem_filter_kwargs(context.config),
        )
    if restore_columns:
        result = result.rename(columns=restore_columns)
    write_compact_operator_trace(
        context.config.trace_dir(),
        operator="sem_filter",
        event_type="operator_result",
        input_frame=source,
        output_frame=result,
        payload={
            "instruction": str(query.params["instruction"]),
            "lowered_instruction": instruction,
            "semantic_pair_profile": (
                "oracle-only" if profile is None else profile.mode
            ),
            "semantic_pair_profile_fingerprint": (
                None if profile is None else profile.fingerprint
            ),
            "candidate_pair_count": (
                None if selection is None else selection.candidate_pair_count
            ),
        },
    )
    return result


def bind_qualified_filter_columns(
    source: pd.DataFrame,
    instruction: str,
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    """Bind alias-qualified columns to names accepted by LOTUS formatting."""

    if not QUALIFIED_PLACEHOLDER_PATTERN.search(instruction):
        return source, instruction, {}
    if not isinstance(source, pd.DataFrame):
        raise TypeError("qualified sem_filter placeholders require a DataFrame input")

    used_names = {str(column) for column in source.columns}
    bindings: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        base, qualifier = match.groups()
        source_column = f"{base}:{qualifier}"
        if source_column not in source.columns:
            raise ValueError(
                f"sem_filter qualified input column not found: {source_column!r}"
            )
        bound_column = bindings.get(source_column)
        if bound_column is None:
            candidate = f"{base}_{qualifier}"
            suffix = 2
            while candidate in used_names:
                candidate = f"{base}_{qualifier}_{suffix}"
                suffix += 1
            bindings[source_column] = candidate
            used_names.add(candidate)
            bound_column = candidate
        return f"{{{bound_column}}}"

    lowered_instruction = QUALIFIED_PLACEHOLDER_PATTERN.sub(replace, instruction)
    bound_source = source.rename(columns=bindings)
    restore_columns = {bound: original for original, bound in bindings.items()}
    return bound_source, lowered_instruction, restore_columns


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
