"""LOTUS-backed semantic join lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.structured import examples_dataframe, normalize_strategy
from agent_memory.logical import QueryExpr


def execute_sem_join(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute semantic join with LOTUS predicate evaluation."""

    context.configure()
    left = execute(query.inputs[0], inputs)
    right = execute(query.inputs[1], inputs)
    join_results = evaluate_semantic_join(query, left, right, context.config)
    return assemble_join_frame(
        left,
        right,
        join_results,
        how=str(query.params.get("how", "inner")),
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
    return list(output.join_results)


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
) -> pd.DataFrame:
    """Assemble join output and deterministic unmatched rows."""

    how = how.lower()
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"Unsupported sem_join how={how!r}")

    left_columns, right_columns = renamed_columns(left, right)
    output_columns = list(left_columns.values()) + list(right_columns.values())
    if return_explanations:
        output_columns.append(explanation_column)

    rows: list[dict[str, Any]] = []
    matched_left: set[Any] = set()
    matched_right: set[Any] = set()
    for left_id, right_id, explanation in join_results:
        matched_left.add(left_id)
        matched_right.add(right_id)
        row = join_row(left.loc[left_id], right.loc[right_id], left_columns, right_columns)
        if return_explanations:
            row[explanation_column] = explanation
        rows.append(row)

    if how in {"left", "outer"}:
        for left_id in left.index:
            if left_id not in matched_left:
                row = left_only_row(left.loc[left_id], left_columns, right_columns)
                if return_explanations:
                    row[explanation_column] = pd.NA
                rows.append(row)

    if how in {"right", "outer"}:
        for right_id in right.index:
            if right_id not in matched_right:
                row = right_only_row(right.loc[right_id], left_columns, right_columns)
                if return_explanations:
                    row[explanation_column] = pd.NA
                rows.append(row)

    return pd.DataFrame(rows, columns=output_columns)


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

    return pd.Series(
        [
            "\n".join(f"{column}: {row[column]}" for column in frame.columns)
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
