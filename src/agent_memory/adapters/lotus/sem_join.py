"""LOTUS-backed semantic join lowering."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
    SemanticPairExecutionProfile,
    select_semantic_pair_candidates,
    write_semantic_pair_execution_trace,
)
from agent_memory.tracing.semantic import (
    query_digest,
    write_pair_trace,
    write_trace_event,
)
from agent_memory.adapters.lotus.structured import examples_dataframe, normalize_strategy
from agent_memory.policy.logical import QueryExpr
from agent_memory.storage.embedding import EmbeddingProvider


def execute_sem_join(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute semantic join with LOTUS predicate evaluation."""

    digest = query_digest(query)
    profile = context.config.semantic_pair_profiles.get(digest)
    id_columns = _join_id_columns(query.params.get("id_columns"))
    if query.params.get("k") is not None and context.config.sem_join_cascade_args is not None:
        raise ValueError("top-k sem_join cannot be combined with LOTUS sem_join cascade")
    if query.params.get("on") and context.config.sem_join_cascade_args is not None:
        # LOTUS 1.1.4 cascades rebuild the full Cartesian product and cannot
        # consume exact-key candidate pairs. Keep that backend boundary explicit
        # rather than silently dropping `on` semantics or forking LOTUS here.
        raise ValueError(
            "exact-key sem_join cannot be combined with LOTUS sem_join cascade"
        )
    if profile is not None and profile.mode in {"search-filter", "proxy-only"}:
        if context.config.sem_join_cascade_args is not None:
            raise ValueError(
                f"sem_join {profile.mode} cannot be combined with LOTUS cascade"
            )
        if context.pair_embedding_provider is None:
            raise ValueError(
                f"{profile.mode} requires a pair embedding provider"
            )
    context.configure()
    left = execute(query.inputs[0], inputs)
    right = execute(query.inputs[1], inputs)
    if left.empty or right.empty:
        result = assemble_join_frame(
            left,
            right,
            (),
            how=str(query.params.get("how", "inner")),
            id_columns=id_columns,
        )
        write_sem_join_result_trace(
            context.config.trace_dir(),
            query=query,
            left=left,
            right=right,
            result=result,
            payload={
                "skipped_pairwise": True,
                **(
                    {
                        "sem_join_topk_method": context.config.sem_join_topk_method,
                        "k": int(query.params["k"]),
                    }
                    if query.params.get("k") is not None
                    else {}
                ),
            },
        )
        return result
    topk_payload: Mapping[str, Any] = {}
    if query.params.get("k") is not None:
        from agent_memory.adapters.lotus.sem_topk_join import evaluate_sem_topk_join

        join_results, topk_payload = evaluate_sem_topk_join(
            query,
            left,
            right,
            context,
            profile=profile,
        )
    elif query.params.get("on"):
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
            on=tuple(str(column) for column in query.params["on"]),
        )
        if profile is not None and profile.mode in {"search-filter", "proxy-only"}:
            assert context.pair_embedding_provider is not None
            join_results = evaluate_profiled_semantic_join(
                query,
                left,
                right,
                context.config,
                profile=profile,
                embedding_provider=context.pair_embedding_provider,
                pairs=pairs,
            )
        else:
            join_results = verify_semantic_join_candidates(
                pairs,
                left_label=left_label,
                right_label=right_label,
                instruction=instruction,
                config=context.config,
            )
    elif profile is not None and profile.mode in {"search-filter", "proxy-only"}:
        assert context.pair_embedding_provider is not None
        join_results = evaluate_profiled_semantic_join(
            query,
            left,
            right,
            context.config,
            profile=profile,
            embedding_provider=context.pair_embedding_provider,
        )
    else:
        join_results = evaluate_semantic_join(query, left, right, context.config)
    result = assemble_join_frame(
        left,
        right,
        join_results,
        how=str(query.params.get("how", "inner")),
        id_columns=id_columns,
    )
    write_sem_join_result_trace(
        context.config.trace_dir(),
        query=query,
        left=left,
        right=right,
        result=result,
        payload={
            "skipped_pairwise": False,
            **(
                {
                    "semantic_pair_profile": profile.mode,
                    "semantic_pair_profile_fingerprint": profile.fingerprint,
                }
                if profile is not None
                and profile.mode in {"search-filter", "proxy-only"}
                else {}
            ),
            **topk_payload,
        },
    )
    return result


def write_sem_join_result_trace(
    trace_dir: Any,
    *,
    query: QueryExpr,
    left: pd.DataFrame,
    right: pd.DataFrame,
    result: pd.DataFrame,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """Write sem_join result trace with explicit left/right/output snapshots."""

    event_payload: dict[str, Any] = {
        "query_digest": query_digest(query),
        "how": str(query.params.get("how", "inner")),
        "instruction": str(query.params["instruction"]),
        "left_rows": len(left),
        "left_columns": [str(column) for column in left.columns],
        "right_rows": len(right),
        "right_columns": [str(column) for column in right.columns],
        "output_rows": len(result),
        "output_columns": [str(column) for column in result.columns],
    }
    event_payload.update(dict(payload or {}))
    write_trace_event(
        trace_dir,
        operator="sem_join",
        event_type="operator_result",
        payload=event_payload,
        snapshots={
            "left": left,
            "right": right,
            "output": result,
        },
    )


def evaluate_semantic_join(
    query: QueryExpr,
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: LotusExecutionConfig,
) -> list[tuple[Any, Any, str | None]]:
    """Return LOTUS semantic join matches as left/right ids and explanation."""

    import lotus
    from lotus.sem_ops.sem_join import sem_join, sem_join_cascade

    left_series, right_series, left_label, right_label, instruction = join_series(
        left,
        right,
        str(query.params["instruction"]),
    )
    cascade_args = cascade_args_from_mapping(config.sem_join_cascade_args)
    common_kwargs = {
        "examples_multimodal_data": examples_multimodal_data(
            config.sem_join_examples,
            left_label=left_label,
            right_label=right_label,
        ),
        "examples_answers": example_answers(config.sem_join_examples),
        "cot_reasoning": example_reasoning(config.sem_join_examples),
        "default": config.sem_join_default,
        "strategy": normalize_strategy(config.sem_join_strategy),
        "safe_mode": config.sem_join_safe_mode,
    }

    if cascade_args is not None:
        output = sem_join_cascade(
            left_series,
            right_series,
            list(left_series.index),
            list(right_series.index),
            left_label,
            right_label,
            lotus.settings.lm,
            instruction,
            cascade_args,
            map_instruction=cascade_args.map_instruction,
            map_examples=cascade_args.map_examples,
            **common_kwargs,
        )
    else:
        output = sem_join(
            left_series,
            right_series,
            list(left_series.index),
            list(right_series.index),
            left_label,
            right_label,
            lotus.settings.lm,
            instruction,
            progress_bar_desc=config.sem_join_progress_bar_desc,
            **common_kwargs,
        )
    write_join_pair_trace(
        config.trace_dir(),
        left_series,
        right_series,
        instruction=instruction,
        output=output,
        default=config.sem_join_default,
    )
    return list(output.join_results)


def evaluate_profiled_semantic_join(
    query: QueryExpr,
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: LotusExecutionConfig,
    *,
    profile: SemanticPairExecutionProfile,
    embedding_provider: EmbeddingProvider,
    pairs: pd.DataFrame | None = None,
) -> list[tuple[Any, Any, str | None]]:
    """Execute one profiled semantic join over embedding-selected pairs."""

    left_series, right_series, left_label, right_label, instruction = join_series(
        left,
        right,
        str(query.params["instruction"]),
    )
    if pairs is None:
        pairs = semantic_join_pair_candidates(left_series, right_series)
    selection = select_semantic_pair_candidates(
        pairs,
        profile=profile,
        embedding_provider=embedding_provider,
    )
    write_semantic_pair_execution_trace(
        config.trace_dir(),
        operator="sem_join",
        query_digest_value=query_digest(query),
        profile=profile,
        selection=selection,
    )
    candidates = pairs.iloc[list(selection.selected_positions)].reset_index(drop=True)
    if candidates.empty:
        return []
    if profile.mode == "proxy-only":
        return [
            (
                row[PAIR_LEFT_ID_COLUMN],
                row[PAIR_RIGHT_ID_COLUMN],
                None,
            )
            for _row_index, row in candidates.iterrows()
        ]

    return verify_semantic_join_candidates(
        candidates,
        left_label=left_label,
        right_label=right_label,
        instruction=instruction,
        config=config,
    )


def verify_semantic_join_candidates(
    candidates: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    instruction: str,
    config: LotusExecutionConfig,
) -> list[tuple[Any, Any, str | None]]:
    """Verify canonical semantic-join candidate pairs with the LOTUS oracle."""

    if candidates.empty:
        return []

    import lotus
    from lotus.sem_ops.sem_filter import sem_filter
    from lotus.templates import task_instructions

    oracle_frame = pd.DataFrame(
        {
            left_label: candidates[PAIR_LEFT_TEXT_COLUMN],
            right_label: candidates[PAIR_RIGHT_TEXT_COLUMN],
        }
    )
    docs = task_instructions.df2multimodal_info(
        oracle_frame,
        [left_label, right_label],
    )
    output = sem_filter(
        docs,
        lotus.settings.lm,
        instruction,
        examples_multimodal_data=examples_multimodal_data(
            config.sem_join_examples,
            left_label=left_label,
            right_label=right_label,
        ),
        examples_answers=example_answers(config.sem_join_examples),
        cot_reasoning=example_reasoning(config.sem_join_examples),
        default=config.sem_join_default,
        strategy=normalize_strategy(config.sem_join_strategy),
        safe_mode=config.sem_join_safe_mode,
        progress_bar_desc=config.sem_join_progress_bar_desc,
    )
    outputs = list(output.outputs)
    if len(outputs) != len(candidates):
        raise ValueError(
            "sem_join candidate verification returned an unexpected number of "
            f"outputs: expected {len(candidates)}, got {len(outputs)}"
        )
    raw_outputs = aligned_join_values(output, "raw_outputs", len(candidates), "")
    explanations = aligned_join_values(
        output,
        "explanations",
        len(candidates),
        None,
    )
    write_selected_join_pair_trace(
        config.trace_dir(),
        candidates,
        instruction=instruction,
        outputs=outputs,
        raw_outputs=raw_outputs,
        explanations=explanations,
        default=config.sem_join_default,
    )
    return [
        (
            row[PAIR_LEFT_ID_COLUMN],
            row[PAIR_RIGHT_ID_COLUMN],
            explanations[index],
        )
        for index, (_row_index, row) in enumerate(candidates.iterrows())
        if bool(outputs[index])
    ]


def semantic_join_pair_candidates(
    left: pd.Series,
    right: pd.Series,
    *,
    left_frame: pd.DataFrame | None = None,
    right_frame: pd.DataFrame | None = None,
    on: Sequence[str] = (),
) -> pd.DataFrame:
    """Return semantic join pairs after optional exact-key restriction."""

    if on and (left_frame is None or right_frame is None):
        raise ValueError("exact-key semantic join candidates require both input frames")
    missing = (
        sorted(
            set(on).difference(left_frame.columns).union(
                set(on).difference(right_frame.columns)
            )
        )
        if left_frame is not None and right_frame is not None
        else []
    )
    if missing:
        raise ValueError(f"sem_join on columns not found on both sides: {missing}")

    right_by_key: dict[tuple[object, ...], list[Any]] = defaultdict(list)
    if on:
        assert right_frame is not None
        for right_id in right.index:
            right_by_key[_semantic_join_key(right_frame.loc[right_id], on)].append(
                right_id
            )

    rows: list[dict[str, Any]] = []
    for left_id, left_value in left.items():
        if on:
            assert left_frame is not None
            right_ids: Sequence[Any] = right_by_key.get(
                _semantic_join_key(left_frame.loc[left_id], on),
                (),
            )
        else:
            right_ids = right.index
        rows.extend(
            {
                PAIR_LEFT_ID_COLUMN: left_id,
                PAIR_RIGHT_ID_COLUMN: right_id,
                PAIR_LEFT_TEXT_COLUMN: left_value,
                PAIR_RIGHT_TEXT_COLUMN: right.loc[right_id],
            }
            for right_id in right_ids
        )
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            (
                PAIR_LEFT_ID_COLUMN,
                PAIR_RIGHT_ID_COLUMN,
                PAIR_LEFT_TEXT_COLUMN,
                PAIR_RIGHT_TEXT_COLUMN,
            )
        ),
    )


def _semantic_join_key(
    row: pd.Series,
    columns: Sequence[str],
) -> tuple[object, ...]:
    key = tuple(row[column] for column in columns)
    try:
        hash(key)
    except TypeError as error:
        raise TypeError(
            "sem_join exact-key columns must contain hashable scalar values"
        ) from error
    return key


def aligned_join_values(
    output: Any,
    attribute: str,
    expected: int,
    fill: Any,
) -> list[Any]:
    """Return one optional LOTUS output value per selected join pair."""

    values = list(getattr(output, attribute, ()) or ())
    if len(values) < expected:
        values.extend(fill for _ in range(expected - len(values)))
    return values[:expected]


def write_selected_join_pair_trace(
    trace_dir: Any,
    pairs: pd.DataFrame,
    *,
    instruction: str,
    outputs: Sequence[bool],
    raw_outputs: Sequence[Any],
    explanations: Sequence[Any],
    default: bool,
) -> None:
    """Write one sem_join trace row per oracle-verified candidate pair."""

    rows = [
        {
            "operator": "sem_join",
            "instruction": instruction,
            "left_id": pair[PAIR_LEFT_ID_COLUMN],
            "right_id": pair[PAIR_RIGHT_ID_COLUMN],
            "left": pair[PAIR_LEFT_TEXT_COLUMN],
            "right": pair[PAIR_RIGHT_TEXT_COLUMN],
            "parsed_output": bool(outputs[index]),
            "raw_output": raw_outputs[index],
            "explanation": explanations[index],
            "default": default,
        }
        for index, (_row_index, pair) in enumerate(pairs.iterrows())
    ]
    write_pair_trace(
        trace_dir,
        operator="sem_join",
        rows=rows,
        snapshots={"pairs": pairs},
    )


def write_join_pair_trace(
    trace_dir: Any,
    left_series: pd.Series,
    right_series: pd.Series,
    *,
    instruction: str,
    output: Any,
    default: bool,
) -> None:
    """Write one sem_join trace row per evaluated pair when available."""

    parsed_outputs = list(getattr(output, "filter_outputs", ()))
    if len(parsed_outputs) != len(left_series) * len(right_series):
        return

    raw_outputs = list(getattr(output, "all_raw_outputs", ()))
    explanations = list(getattr(output, "all_explanations", ()))
    rows: list[dict[str, Any]] = []
    index = 0
    for left_id, left_value in left_series.items():
        for right_id, right_value in right_series.items():
            rows.append(
                {
                    "operator": "sem_join",
                    "instruction": instruction,
                    "left_id": left_id,
                    "right_id": right_id,
                    "left": left_value,
                    "right": right_value,
                    "parsed_output": bool(parsed_outputs[index]),
                    "raw_output": raw_outputs[index] if index < len(raw_outputs) else "",
                    "explanation": explanations[index] if index < len(explanations) else "",
                    "default": default,
                }
            )
            index += 1
    write_pair_trace(
        trace_dir,
        operator="sem_join",
        rows=rows,
        snapshots={
            "left": left_series.to_frame(left_series.name or "left"),
            "right": right_series.to_frame(right_series.name or "right"),
        },
    )


def join_series(
    left: pd.DataFrame,
    right: pd.DataFrame,
    instruction: str,
) -> tuple[pd.Series, pd.Series, str, str, str]:
    """Select LOTUS join series, falling back to full-row text when ambiguous."""

    import lotus

    try:
        cols = lotus.nl_expression.parse_cols(instruction)
    except ValueError:
        cols = []

    side_aware_pairs = _find_side_aware_join_column_pairs(cols, left, right)
    if len(side_aware_pairs) == 1:
        left_label, left_column, right_label, right_column = side_aware_pairs[0]
        return (
            left[left_column],
            right[right_column],
            left_label,
            right_label,
            instruction,
        )
    if len(side_aware_pairs) > 1:
        left_label = "left"
        right_label = "right"
        return (
            record_text_series(
                left,
                [
                    left_column
                    for (
                        _left_label,
                        left_column,
                        _right_label,
                        _right_column,
                    ) in side_aware_pairs
                ],
                left_label,
            ),
            record_text_series(
                right,
                [
                    right_column
                    for (
                        _left_label,
                        _left_column,
                        _right_label,
                        right_column,
                    ) in side_aware_pairs
                ],
                right_label,
            ),
            left_label,
            right_label,
            f"{{{left_label}}} and {{{right_label}}} satisfy this semantic join condition: {instruction}",
        )

    left_on, right_on = _find_join_columns(cols, left, right)
    if left_on is not None and right_on is not None:
        return (
            left[left_on[1]],
            right[right_on[1]],
            left_on[0],
            right_on[0],
            instruction,
        )

    left_label = "left"
    right_label = "right"
    return (
        row_text_series(left, left_label),
        row_text_series(right, right_label),
        left_label,
        right_label,
        f"{{{left_label}}} and {{{right_label}}} satisfy this semantic join condition: {instruction}",
    )


def assemble_join_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    join_results: Sequence[tuple[Any, Any, str | None]],
    *,
    how: str,
    return_explanations: bool = False,
    explanation_column: str = "explanation_join",
    id_columns: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble join output and deterministic unmatched rows."""

    how = how.lower()
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"Unsupported sem_join how={how!r}")

    left_columns, right_columns = renamed_columns(left, right)
    output_columns = list(left_columns.values()) + list(right_columns.values())
    if id_columns is not None:
        conflicts = sorted(set(id_columns).intersection(output_columns))
        if conflicts:
            raise ValueError(f"sem_join id columns conflict with output columns: {conflicts}")
        output_columns.extend(id_columns)
    if return_explanations:
        output_columns.append(explanation_column)

    rows: list[dict[str, Any]] = []
    matched_left: set[Any] = set()
    matched_right: set[Any] = set()
    for left_id, right_id, explanation in join_results:
        matched_left.add(left_id)
        matched_right.add(right_id)
        row = join_row(left.loc[left_id], right.loc[right_id], left_columns, right_columns)
        if id_columns is not None:
            row[id_columns[0]] = left_id
            row[id_columns[1]] = right_id
        if return_explanations:
            row[explanation_column] = explanation
        rows.append(row)

    if how in {"left", "outer"}:
        for left_id in left.index:
            if left_id not in matched_left:
                row = left_only_row(left.loc[left_id], left_columns, right_columns)
                if id_columns is not None:
                    row[id_columns[0]] = left_id
                    row[id_columns[1]] = pd.NA
                if return_explanations:
                    row[explanation_column] = pd.NA
                rows.append(row)

    if how in {"right", "outer"}:
        for right_id in right.index:
            if right_id not in matched_right:
                row = right_only_row(right.loc[right_id], left_columns, right_columns)
                if id_columns is not None:
                    row[id_columns[0]] = pd.NA
                    row[id_columns[1]] = right_id
                if return_explanations:
                    row[explanation_column] = pd.NA
                rows.append(row)

    return pd.DataFrame(rows, columns=output_columns)


def _join_id_columns(value: object) -> tuple[str, str] | None:
    """Validate optional internal left/right row-ID output columns."""

    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(not isinstance(column, str) or not column for column in value)
    ):
        raise ValueError("sem_join id_columns must contain two non-empty column names")
    left_id_column, right_id_column = (str(column) for column in value)
    if left_id_column == right_id_column:
        raise ValueError("sem_join id_columns must be distinct")
    return left_id_column, right_id_column


def cascade_args_from_mapping(value: Any) -> Any:
    """Convert row-like cascade args into LOTUS CascadeArgs."""

    if value is None:
        return None
    from lotus.types import CascadeArgs

    return CascadeArgs(**dict(value))


def examples_multimodal_data(
    examples: Any,
    *,
    left_label: str,
    right_label: str,
) -> list[dict[str, Any]] | None:
    """Convert join examples to LOTUS multimodal data."""

    frame = examples_dataframe(examples)
    if frame is None:
        return None
    from lotus.templates import task_instructions

    return task_instructions.df2multimodal_info(frame, [left_label, right_label])


def example_answers(examples: Any) -> list[bool] | None:
    """Return boolean join example answers."""

    if examples is None:
        return None
    return [bool(dict(row)["Answer"]) for row in examples]


def example_reasoning(examples: Any) -> list[str] | None:
    """Return optional join example reasoning."""

    if examples is None:
        return None
    rows = [dict(row) for row in examples]
    if not rows or "Reasoning" not in rows[0]:
        return None
    return [str(row.get("Reasoning", "")) for row in rows]


def row_text_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Serialize complete rows for ambiguous join instructions."""

    return record_text_series(frame, tuple(frame.columns), name)


def record_text_series(
    frame: pd.DataFrame,
    columns: Sequence[Any],
    name: str,
) -> pd.Series:
    """Serialize selected row fields for composite semantic comparisons."""

    return pd.Series(
        [
            "\n".join(f"{column}: {row[column]}" for column in columns)
            for _index, row in frame.iterrows()
        ],
        index=frame.index,
        name=name,
    )


def renamed_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return output column names using LOTUS-style overlap suffixes."""

    overlaps = set(left.columns).intersection(set(right.columns))
    left_columns = {
        column: f"{column}:left" if column in overlaps else str(column)
        for column in left.columns
    }
    right_columns = {
        column: f"{column}:right" if column in overlaps else str(column)
        for column in right.columns
    }
    return left_columns, right_columns


def join_row(
    left_row: pd.Series,
    right_row: pd.Series,
    left_columns: Mapping[str, str],
    right_columns: Mapping[str, str],
) -> dict[str, Any]:
    """Build one matched join row."""

    row = {out: left_row[column] for column, out in left_columns.items()}
    row.update({out: right_row[column] for column, out in right_columns.items()})
    return row


def left_only_row(
    left_row: pd.Series,
    left_columns: Mapping[str, str],
    right_columns: Mapping[str, str],
) -> dict[str, Any]:
    """Build one unmatched-left join row."""

    row = {out: left_row[column] for column, out in left_columns.items()}
    row.update({out: pd.NA for out in right_columns.values()})
    return row


def right_only_row(
    right_row: pd.Series,
    left_columns: Mapping[str, str],
    right_columns: Mapping[str, str],
) -> dict[str, Any]:
    """Build one unmatched-right join row."""

    row = {out: pd.NA for out in left_columns.values()}
    row.update({out: right_row[column] for column, out in right_columns.items()})
    return row


def _find_join_columns(
    parsed_columns: Sequence[str],
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Find the left/right columns named by a LOTUS join instruction."""

    left_on: tuple[str, str] | None = None
    right_on: tuple[str, str] | None = None

    for column in parsed_columns:
        if column.endswith(":left") and column[:-5] in left.columns:
            left_on = (column, column[:-5])
        elif column.endswith(":right") and column[:-6] in right.columns:
            right_on = (column, column[:-6])

    if left_on is None:
        for column in parsed_columns:
            if column in left.columns and column not in right.columns:
                left_on = (column, column)
                break
    if right_on is None:
        for column in parsed_columns:
            if column in right.columns and column not in left.columns:
                right_on = (column, column)
                break
    return left_on, right_on


def _find_side_aware_join_column_pairs(
    parsed_columns: Sequence[str],
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[tuple[str, str, str, str], ...]:
    """Find side-aware left/right column pairs referenced by a join instruction."""

    left_by_name: dict[str, str] = {}
    right_by_name: dict[str, str] = {}
    order: list[str] = []

    for column in parsed_columns:
        side: str | None = None
        base = column
        if column.endswith(":left"):
            side = "left"
            base = column[:-5]
        elif column.endswith(":right"):
            side = "right"
            base = column[:-6]

        if side is None or base in order:
            continue
        order.append(base)

    for column in parsed_columns:
        if column.endswith(":left") and column[:-5] in left.columns:
            left_by_name[column[:-5]] = column
        elif column.endswith(":right") and column[:-6] in right.columns:
            right_by_name[column[:-6]] = column

    pairs: list[tuple[str, str, str, str]] = []
    for base in order:
        left_label = left_by_name.get(base)
        right_label = right_by_name.get(base)
        if left_label is not None and right_label is not None:
            pairs.append((left_label, base, right_label, base))
    return tuple(pairs)
