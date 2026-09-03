"""LOTUS sem_filter lowering."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.pair_execution import (
    PairCandidateSelection,
    select_semantic_pair_candidates,
    write_semantic_pair_execution_trace,
)
from agent_memory.adapters.lotus.sem_filter_batch_prompting import (
    execute_batch_prompted_sem_filter,
    validate_batch_prompting_sem_filter_config,
)
from agent_memory.tracing.semantic import (
    query_digest,
    semantic_trace_scope,
    write_compact_operator_trace,
    write_pair_trace,
)
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
    prompt_batching = context.config.prompt_batching
    batch_prompting = prompt_batching is not None
    if batch_prompting:
        validate_batch_prompting_sem_filter_config(context.config)
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
    prompt_count: int | None = 0 if batch_prompting else None
    prompt_retry_count: int | None = (
        0 if batch_prompting else None
    )
    verified_tuple_count: int | None = (
        len(lotus_source) if batch_prompting else None
    )
    if profile is not None and profile.mode == "proxy-only":
        result = lotus_source.copy()
    elif selection is not None and lotus_source.empty:
        result = lotus_source.copy()
    elif batch_prompting:
        assert prompt_batching is not None
        prompted = execute_batch_prompted_sem_filter(
            lotus_source,
            instruction=instruction,
            context=context,
            prompt_batching=prompt_batching,
        )
        result = prompted.frame
        prompt_count = prompted.prompt_count
        prompt_retry_count = prompted.retry_count
    else:
        result = lotus_source.sem_filter(
            instruction,
            **native_sem_filter_kwargs(context.config),
        )
    if restore_columns:
        result = result.rename(columns=restore_columns)
    with semantic_trace_scope(
        semantic_trace_snapshot_mode=context.config.semantic_trace_snapshot_mode,
    ):
        write_pairwise_sem_filter_trace(
            context.config.trace_dir(),
            source=oracle_source,
            result=result,
            instruction=str(query.params["instruction"]),
            profile=profile,
        )
        trace_payload: dict[str, Any] = {
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
        }
        if prompt_batching is not None:
            trace_payload.update(
                {
                    "prompt_batching": prompt_batching.to_dict(),
                    "prompt_batching_fingerprint": prompt_batching.fingerprint,
                    "verification_prompt_count": prompt_count,
                    "verification_retry_count": prompt_retry_count,
                    "verified_tuple_count": verified_tuple_count,
                }
            )
        write_compact_operator_trace(
            context.config.trace_dir(),
            operator="sem_filter",
            event_type="operator_result",
            input_frame=source,
            output_frame=result,
            payload=trace_payload,
        )
    return result


def write_pairwise_sem_filter_trace(
    trace_dir: object,
    *,
    source: pd.DataFrame,
    result: pd.DataFrame,
    instruction: str,
    profile: Any,
) -> None:
    """Record pair-shaped filter decisions without full relation snapshots."""

    columns = _pairwise_filter_columns(source, instruction, profile)
    if columns is None or source.empty:
        return
    left_id_columns, right_id_columns, left_text_columns, right_text_columns, direction = (
        columns
    )
    result_counts = Counter(_row_signature(row) for _, row in result.iterrows())
    decision_source = (
        "proxy" if profile is not None and profile.mode == "proxy-only" else "oracle"
    )
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        signature = _row_signature(row)
        matched = result_counts[signature] > 0
        if matched:
            result_counts[signature] -= 1
        rows.append(
            {
                "instruction": instruction,
                "direction": direction,
                "decision_source": decision_source,
                "left_id": _endpoint_id(row, left_id_columns),
                "right_id": _endpoint_id(row, right_id_columns),
                "left": _endpoint_text(row, left_text_columns),
                "right": _endpoint_text(row, right_text_columns),
                "parsed_output": matched,
            }
        )
    if any(result_counts.values()):
        raise ValueError("sem_filter output contains rows absent from its oracle input")
    write_pair_trace(trace_dir, operator="sem_filter", rows=rows)


def _pairwise_filter_columns(
    source: pd.DataFrame,
    instruction: str,
    profile: Any,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str] | None:
    if profile is not None:
        return (
            tuple(profile.left_id_columns),
            tuple(profile.right_id_columns),
            tuple(profile.left_text_columns),
            tuple(profile.right_text_columns),
            str(profile.direction),
        )
    placeholders = QUALIFIED_PLACEHOLDER_PATTERN.findall(instruction)
    bases = sorted(
        base
        for base in {base for base, _qualifier in placeholders}
        if {qualifier for candidate, qualifier in placeholders if candidate == base}
        == {"earlier", "later"}
    )
    if not bases:
        return None
    left_text = tuple(f"{base}:earlier" for base in bases)
    right_text = tuple(f"{base}:later" for base in bases)
    if not set((*left_text, *right_text)).issubset(source.columns):
        return None
    left_ids = tuple(
        column
        for column in source.columns
        if str(column).endswith(":earlier")
        and str(column).split(":", 1)[0] in {"_row_id", "_memory_ordinal"}
    )
    right_ids = tuple(
        column
        for column in source.columns
        if str(column).endswith(":later")
        and str(column).split(":", 1)[0] in {"_row_id", "_memory_ordinal"}
    )
    return (
        left_ids or left_text,
        right_ids or right_text,
        left_text,
        right_text,
        "right-to-left",
    )


def _endpoint_id(row: pd.Series, columns: Sequence[str]) -> str:
    return json.dumps(
        [row[column] for column in columns],
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def _endpoint_text(row: pd.Series, columns: Sequence[str]) -> str:
    return "\n".join(
        f"{column.split(':', 1)[0]}: {row[column]}" for column in columns
    )


def _row_signature(row: pd.Series) -> str:
    return json.dumps(
        [(str(column), row[column]) for column in row.index],
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


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
