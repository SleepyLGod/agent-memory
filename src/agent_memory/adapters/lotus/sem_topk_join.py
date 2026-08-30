"""Cardinality-bounded semantic join execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, cast

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
    SemanticPairExecutionProfile,
    select_semantic_pair_candidates,
    write_semantic_pair_execution_trace,
)
from agent_memory.adapters.lotus.sem_topk import LOTUS_PAIRWISE_METHODS
from agent_memory.adapters.lotus.structured import normalize_strategy
from agent_memory.policy.logical import QueryExpr
from agent_memory.tracing.semantic import query_digest, write_pair_trace

LISTWISE_JOIN_MAX_TOKENS = 1024
LISTWISE_JOIN_SYSTEM_PROMPT = (
    "The user will provide one left row, a semantic join condition, and candidate "
    "right rows. Your job is to select the right rows that satisfy the join "
    "condition for that left row. Select at most max_matches candidates. Do not "
    "select merely related candidates or fill the limit when fewer candidates "
    "match. If too many candidates match, select the strongest matches. Return an "
    "empty selected_ids array when none match. Return only a JSON object."
)


@dataclass(frozen=True)
class _ListwiseTask:
    left_id: Any
    prompt: list[dict[str, Any]]
    candidate_positions: Mapping[str, int]


def evaluate_sem_topk_join(
    query: QueryExpr,
    left: pd.DataFrame,
    right: pd.DataFrame,
    context: LotusExecutionContext,
    *,
    profile: SemanticPairExecutionProfile | None,
) -> tuple[list[tuple[Any, Any, str | None]], Mapping[str, Any]]:
    """Evaluate a per-left top-k semantic join with one configured access path."""

    from agent_memory.adapters.lotus.sem_join import join_series
    from agent_memory.adapters.lotus.sem_join import semantic_join_pair_candidates

    k = _positive_k(query.params.get("k"))
    left_series, right_series, left_label, right_label, instruction = join_series(
        left,
        right,
        str(query.params["instruction"]),
    )
    pairs = semantic_join_pair_candidates(
        left_series,
        right_series,
        left_frame=left,
        right_frame=right,
        on=tuple(str(column) for column in query.params.get("on", ())),
    )
    candidates, scores = _apply_pair_profile(
        query,
        pairs,
        context,
        profile=profile,
    )

    method = context.config.sem_join_topk_method
    if profile is not None and profile.mode == "proxy-only":
        join_results = _proxy_topk(candidates, scores=scores, k=k)
        method = "proxy-only"
        retry_count = 0
    elif method == "listwise":
        join_results, retry_count = _listwise_topk(
            candidates,
            instruction=instruction,
            left_label=left_label,
            right_label=right_label,
            k=k,
            context=context,
        )
    else:
        join_results = _pairwise_topk(
            candidates,
            instruction=instruction,
            left_label=left_label,
            right_label=right_label,
            k=k,
            method=method,
            context=context,
        )
        retry_count = 0

    return join_results, {
        "sem_join_topk_method": method,
        "k": k,
        "on": list(query.params.get("on", ())),
        "eligible_pair_count": len(pairs),
        "oracle_candidate_count": len(candidates),
        "selected_pair_count": len(join_results),
        "structured_retry_count": retry_count,
    }


def _apply_pair_profile(
    query: QueryExpr,
    pairs: pd.DataFrame,
    context: LotusExecutionContext,
    *,
    profile: SemanticPairExecutionProfile | None,
) -> tuple[pd.DataFrame, tuple[float, ...]]:
    """Apply an optional candidate profile without changing join semantics by default."""

    if profile is None or profile.mode == "oracle-only":
        return pairs.reset_index(drop=True), ()
    if profile.direction != "left-to-right":
        raise ValueError("top-k sem_join pair profiles must use left-to-right direction")
    if context.pair_embedding_provider is None:
        raise ValueError(f"{profile.mode} requires a pair embedding provider")

    selection = select_semantic_pair_candidates(
        pairs,
        profile=profile,
        embedding_provider=context.pair_embedding_provider,
    )
    write_semantic_pair_execution_trace(
        context.config.trace_dir(),
        operator="sem_join",
        query_digest_value=query_digest(query),
        profile=profile,
        selection=selection,
    )
    positions = list(selection.selected_positions)
    selected = pairs.iloc[positions].reset_index(drop=True)
    selected_scores = tuple(selection.similarity_scores[position] for position in positions)
    return selected, selected_scores


def _listwise_topk(
    candidates: pd.DataFrame,
    *,
    instruction: str,
    left_label: str,
    right_label: str,
    k: int,
    context: LotusExecutionContext,
) -> tuple[list[tuple[Any, Any, str | None]], int]:
    """Resolve zero-to-k right matches for each left row in listwise batches."""

    if candidates.empty:
        return [], 0

    import lotus

    lm = lotus.settings.lm
    if lm is None:
        raise ValueError("listwise top-k sem_join requires a configured language model")
    tasks = _listwise_tasks(
        candidates,
        instruction=instruction,
        left_label=left_label,
        right_label=right_label,
        k=k,
    )
    pending = list(tasks)
    selected_by_left: dict[Any, tuple[int, ...]] = {}
    retry_count = 0
    last_error: ValueError | None = None
    for attempt in range(context.config.structured_parse_retries + 1):
        if not pending:
            break
        lm_call: Any = lm
        output = lm_call(
            [task.prompt for task in pending],
            progress_bar_desc="Listwise join resolution",
            max_tokens=LISTWISE_JOIN_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw_outputs = list(getattr(output, "outputs", ()))
        if len(raw_outputs) != len(pending):
            raise ValueError(
                "listwise top-k sem_join returned an unexpected number of outputs: "
                f"expected {len(pending)}, got {len(raw_outputs)}"
            )
        retry: list[_ListwiseTask] = []
        for task, raw_output in zip(pending, raw_outputs, strict=True):
            raw = str(raw_output)
            try:
                selected_ids = _parse_listwise_ids(
                    raw,
                    valid_ids=set(task.candidate_positions),
                    k=k,
                )
            except ValueError as error:
                last_error = error
                retry.append(task)
                continue
            selected_by_left[task.left_id] = tuple(
                task.candidate_positions[candidate_id] for candidate_id in selected_ids
            )
        if retry:
            retry_count += len(retry)
        pending = retry

    if pending:
        assert last_error is not None
        raise ValueError(
            "listwise top-k sem_join returned invalid structured output after "
            f"{context.config.structured_parse_retries + 1} attempt(s): {last_error}"
        ) from last_error

    selected_positions = {
        position
        for positions in selected_by_left.values()
        for position in positions
    }
    rows = []
    results: list[tuple[Any, Any, str | None]] = []
    for position, row in candidates.iterrows():
        matched = position in selected_positions
        rows.append(
            {
                "operator": "sem_join",
                "instruction": instruction,
                "decision_source": "listwise",
                "left_id": row[PAIR_LEFT_ID_COLUMN],
                "right_id": row[PAIR_RIGHT_ID_COLUMN],
                "left": row[PAIR_LEFT_TEXT_COLUMN],
                "right": row[PAIR_RIGHT_TEXT_COLUMN],
                "parsed_output": matched,
            }
        )
        if matched:
            results.append(
                (
                    row[PAIR_LEFT_ID_COLUMN],
                    row[PAIR_RIGHT_ID_COLUMN],
                    None,
                )
            )
    write_pair_trace(
        context.config.trace_dir(),
        operator="sem_join",
        rows=rows,
    )
    return results, retry_count


def _listwise_tasks(
    candidates: pd.DataFrame,
    *,
    instruction: str,
    left_label: str,
    right_label: str,
    k: int,
) -> list[_ListwiseTask]:
    tasks: list[_ListwiseTask] = []
    for left_id, group in candidates.groupby(PAIR_LEFT_ID_COLUMN, sort=False):
        candidate_positions = {
            f"candidate_{position}": int(position) for position in group.index
        }
        right_candidates = [
            {
                "id": candidate_id,
                right_label: candidates.loc[position, PAIR_RIGHT_TEXT_COLUMN],
            }
            for candidate_id, position in candidate_positions.items()
        ]
        payload = {
            "join_condition": instruction,
            "max_matches": min(k, len(group)),
            "left": {left_label: group.iloc[0][PAIR_LEFT_TEXT_COLUMN]},
            "right_candidates": right_candidates,
            "output_schema": {"selected_ids": ["candidate_id"]},
        }
        tasks.append(
            _ListwiseTask(
                left_id=left_id,
                prompt=[
                    {"role": "system", "content": LISTWISE_JOIN_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, default=str),
                    },
                ],
                candidate_positions=candidate_positions,
            )
        )
    return tasks


def _parse_listwise_ids(
    raw_output: str,
    *,
    valid_ids: set[str],
    k: int,
) -> tuple[str, ...]:
    """Parse a strict zero-to-k listwise join response."""

    if not raw_output.strip():
        raise ValueError("listwise top-k sem_join output is empty")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError("listwise top-k sem_join output is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("listwise top-k sem_join output must be a JSON object")
    selected_ids = payload.get("selected_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(candidate_id, str) for candidate_id in selected_ids
    ):
        raise ValueError("listwise top-k sem_join selected_ids must be a list of strings")
    if len(selected_ids) > k:
        raise ValueError(f"listwise top-k sem_join selected more than k={k} candidates")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("listwise top-k sem_join selected_ids must be unique")
    unknown = sorted(set(selected_ids).difference(valid_ids))
    if unknown:
        raise ValueError(
            f"listwise top-k sem_join selected_ids contain unknown IDs: {unknown}"
        )
    return tuple(selected_ids)


def _pairwise_topk(
    candidates: pd.DataFrame,
    *,
    instruction: str,
    left_label: str,
    right_label: str,
    k: int,
    method: str,
    context: LotusExecutionContext,
) -> list[tuple[Any, Any, str | None]]:
    """Verify candidate pairs, then rank excess positive matches with LOTUS top-k."""

    if candidates.empty:
        return []

    from agent_memory.adapters.lotus.sem_join import verify_semantic_join_candidates

    lotus_method = LOTUS_PAIRWISE_METHODS.get(method)
    if lotus_method is None:
        raise ValueError(f"Unsupported top-k sem_join method {method!r}")
    verified = verify_semantic_join_candidates(
        candidates,
        left_label=left_label,
        right_label=right_label,
        instruction=instruction,
        config=context.config,
    )
    explanations = {(left_id, right_id): explanation for left_id, right_id, explanation in verified}
    positives = candidates[
        [
            (row[PAIR_LEFT_ID_COLUMN], row[PAIR_RIGHT_ID_COLUMN]) in explanations
            for _, row in candidates.iterrows()
        ]
    ]
    results: list[tuple[Any, Any, str | None]] = []
    for left_id, grouped in positives.groupby(PAIR_LEFT_ID_COLUMN, sort=False):
        group = cast(pd.DataFrame, grouped)
        selected = group
        if len(group) > k:
            selected = _rank_pairwise_group(
                group,
                instruction=instruction,
                left_label=left_label,
                right_label=right_label,
                k=k,
                lotus_method=lotus_method,
                context=context,
            )
        for _, row in selected.iterrows():
            right_id = row[PAIR_RIGHT_ID_COLUMN]
            results.append((left_id, right_id, explanations[(left_id, right_id)]))
    return results


def _rank_pairwise_group(
    group: pd.DataFrame,
    *,
    instruction: str,
    left_label: str,
    right_label: str,
    k: int,
    lotus_method: str,
    context: LotusExecutionContext,
) -> pd.DataFrame:
    left_text = str(group.iloc[0][PAIR_LEFT_TEXT_COLUMN])
    source = pd.DataFrame(
        {
            left_label: [left_text] * len(group),
            right_label: group[PAIR_RIGHT_TEXT_COLUMN].tolist(),
            "_pair_position": group.index.tolist(),
        }
    )
    ranking_instruction = (
        f"Rank {{{right_label}}} by how strongly it satisfies this semantic join "
        f"condition with the fixed {{{left_label}}}: {instruction}"
    )
    result = source.sem_topk(
        ranking_instruction,
        K=k,
        method=lotus_method,
        strategy=normalize_strategy(context.config.sem_topk_strategy),
        cascade_threshold=context.config.sem_topk_cascade_threshold,
        return_stats=context.config.sem_topk_return_stats,
        safe_mode=context.config.sem_topk_safe_mode,
        return_explanations=False,
    )
    result_frame = result[0] if isinstance(result, tuple) else result
    positions = [int(position) for position in result_frame["_pair_position"]]
    return group.loc[positions]


def _proxy_topk(
    candidates: pd.DataFrame,
    *,
    scores: Sequence[float],
    k: int,
) -> list[tuple[Any, Any, str | None]]:
    """Use embedding-selected pairs directly and cap each left side by score."""

    if len(candidates) != len(scores):
        raise ValueError("proxy-only top-k sem_join requires one score per candidate")
    ranked = candidates.copy()
    ranked["_similarity_score"] = list(scores)
    results: list[tuple[Any, Any, str | None]] = []
    for _left_id, group in ranked.groupby(PAIR_LEFT_ID_COLUMN, sort=False):
        ordered = group.sort_values(
            by=["_similarity_score", PAIR_RIGHT_ID_COLUMN],
            ascending=[False, True],
            kind="stable",
        ).head(k)
        results.extend(
            (
                row[PAIR_LEFT_ID_COLUMN],
                row[PAIR_RIGHT_ID_COLUMN],
                None,
            )
            for _, row in ordered.iterrows()
        )
    return results


def _positive_k(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("top-k sem_join requires a positive integer k")
    return value
