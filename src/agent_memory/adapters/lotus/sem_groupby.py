"""LOTUS-backed semantic group assignment lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.sem_join import row_text_series
from agent_memory.adapters.lotus.structured import StructuredLMExecutor
from agent_memory.logical import ColumnSpec, QueryExpr

GROUP_ID_COLUMN = "_agent_memory_group_id"


def execute_sem_groupby(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Assign deterministic semantic group ids to rows."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    input_cols = tuple(str(column) for column in query.params["input_cols"])
    labels = tuple(query.params.get("labels") or ())
    if labels:
        return assign_declared_labels(
            source,
            input_cols=input_cols,
            labels=labels,
            label_col=str(query.params.get("label_col", "_label")),
            instruction=str(query.params["instruction"]),
        )

    unique_rows, row_to_unique = exact_unique_key_rows(source, input_cols)
    matched_pairs = evaluate_group_matches(
        unique_rows,
        input_cols=input_cols,
        instruction=str(query.params["instruction"]),
    )
    result = assign_semantic_group_ids(
        source,
        input_cols=input_cols,
        matched_unique_pairs=matched_pairs,
        row_to_unique=row_to_unique,
    )
    result.attrs["agent_memory_groupby_input_cols"] = input_cols
    return result


def assign_declared_labels(
    source: pd.DataFrame,
    *,
    input_cols: Sequence[str],
    labels: Sequence[ColumnSpec],
    label_col: str,
    instruction: str,
) -> pd.DataFrame:
    """Assign rows to declared closed-world labels with LOTUS structured output."""

    validate_groupby_input_cols(source, input_cols)
    validate_label_col(source, label_col)

    label_names = tuple(label.name for label in labels)
    if len(set(label_names)) != len(label_names):
        raise ValueError(f"sem_groupby labels must be unique: {label_names}")

    if source.empty:
        result = source.copy()
        result[label_col] = []
        result[GROUP_ID_COLUMN] = []
        result.attrs["agent_memory_groupby_input_cols"] = tuple(input_cols)
        result.attrs["agent_memory_groupby_labels"] = label_names
        result.attrs["agent_memory_groupby_label_col"] = label_col
        return result

    label_instruction = labeled_groupby_instruction(instruction, labels, label_col)
    executor = StructuredLMExecutor(source)
    generation = executor(
        input_cols=tuple(input_cols),
        output_cols=(ColumnSpec(label_col, f"One of: {', '.join(label_names)}."),),
        instruction=label_instruction,
        shape="object",
        system_prompt=None,
        examples=None,
        strategy=None,
        safe_mode=False,
        return_explanations=False,
        progress_bar_desc="Grouping labels",
        model_kwargs={},
        operator="sem_groupby",
    )

    label_to_group = {name: index for index, name in enumerate(label_names)}
    assigned_labels = [
        str(output[label_col])
        for output in generation.parsed_outputs
    ]
    invalid = sorted({label for label in assigned_labels if label not in label_to_group})
    if invalid:
        raise ValueError(
            "sem_groupby label output must be one of declared labels; "
            f"got {invalid}. If you want an 'other' bucket, declare an "
            "'other' label explicitly."
        )

    result = source.copy()
    result[label_col] = assigned_labels
    result[GROUP_ID_COLUMN] = [label_to_group[label] for label in assigned_labels]
    result.attrs["agent_memory_groupby_input_cols"] = tuple(input_cols)
    result.attrs["agent_memory_groupby_labels"] = label_names
    result.attrs["agent_memory_groupby_label_col"] = label_col
    return result


def labeled_groupby_instruction(
    instruction: str,
    labels: Sequence[ColumnSpec],
    label_col: str,
) -> str:
    """Append the closed-world label set to a semantic groupby instruction."""

    label_lines = "\n".join(
        f"- {label.name}: {label.description or label.name}"
        for label in labels
    )
    return (
        f"{instruction}\n\n"
        f"Assign each row to exactly one declared label and put it in {label_col}.\n"
        "Declared labels:\n"
        f"{label_lines}\n"
        "Return only one of the declared label names. Do not invent labels. "
        "If an 'other' bucket is needed, it must be declared as a label."
    )


def validate_groupby_input_cols(source: pd.DataFrame, input_cols: Sequence[str]) -> None:
    """Raise when semantic groupby evidence columns are missing."""

    missing = [column for column in input_cols if column not in source.columns]
    if missing:
        raise ValueError(f"sem_groupby input columns not found in DataFrame: {missing}")


def validate_label_col(source: pd.DataFrame, label_col: str) -> None:
    """Raise when the declared label output column cannot be written safely."""

    if not label_col:
        raise ValueError("sem_groupby label_col cannot be empty")
    if label_col in source.columns:
        raise ValueError(
            f"sem_groupby label_col {label_col!r} already exists in the DataFrame"
        )


def assign_semantic_group_ids(
    source: pd.DataFrame,
    *,
    input_cols: Sequence[str],
    matched_unique_pairs: Sequence[tuple[int, int]],
    row_to_unique: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Assign stable group ids from exact keys plus semantic pair matches."""

    unique_rows, inferred_row_to_unique = exact_unique_key_rows(source, input_cols)
    unique_mapping = tuple(row_to_unique) if row_to_unique is not None else inferred_row_to_unique
    union_find = UnionFind(len(unique_rows))
    for left, right in matched_unique_pairs:
        union_find.union(left, right)

    root_to_group: dict[int, int] = {}
    unique_to_group: dict[int, int] = {}
    for unique_index in range(len(unique_rows)):
        root = union_find.find(unique_index)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        unique_to_group[unique_index] = root_to_group[root]

    result = source.copy()
    result[GROUP_ID_COLUMN] = [unique_to_group[index] for index in unique_mapping]
    result.attrs["agent_memory_groupby_input_cols"] = tuple(input_cols)
    return result


def exact_unique_key_rows(
    source: pd.DataFrame,
    input_cols: Sequence[str],
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Collapse exact duplicate input-column values before semantic comparisons."""

    validate_groupby_input_cols(source, input_cols)

    key_to_unique: dict[tuple[Any, ...], int] = {}
    row_to_unique: list[int] = []
    unique_row_indices: list[Any] = []
    for index, row in source.iterrows():
        key = tuple(row[column] for column in input_cols)
        if key not in key_to_unique:
            key_to_unique[key] = len(unique_row_indices)
            unique_row_indices.append(index)
        row_to_unique.append(key_to_unique[key])
    return source.loc[unique_row_indices, :].reset_index(drop=True), tuple(row_to_unique)


def evaluate_group_matches(
    unique_rows: pd.DataFrame,
    *,
    input_cols: Sequence[str],
    instruction: str,
) -> list[tuple[int, int]]:
    """Evaluate candidate row pairs with LOTUS sem_filter."""

    if len(unique_rows) < 2:
        return []

    import lotus
    from lotus.sem_ops.sem_filter import sem_filter
    from lotus.templates import task_instructions

    pairs = semantic_pair_candidates(unique_rows, input_cols)
    docs = task_instructions.df2multimodal_info(pairs, ["left", "right"])
    user_instruction = (
        "{left} and {right} satisfy this semantic grouping condition: "
        f"{instruction}"
    )
    output = sem_filter(
        docs,
        lotus.settings.lm,
        user_instruction,
        progress_bar_desc="Grouping comparisons",
    )
    return [
        (int(row["_left_unique_id"]), int(row["_right_unique_id"]))
        for (_index, row), keep in zip(pairs.iterrows(), output.outputs)
        if keep
    ]


def semantic_pair_candidates(
    unique_rows: pd.DataFrame,
    input_cols: Sequence[str],
) -> pd.DataFrame:
    """Return i < j candidate pairs over exact-unique key rows."""

    row_text = row_text_series(unique_rows.loc[:, list(input_cols)], "row")
    rows: list[dict[str, Any]] = []
    for left in range(len(unique_rows)):
        for right in range(left + 1, len(unique_rows)):
            rows.append(
                {
                    "_left_unique_id": left,
                    "_right_unique_id": right,
                    "left": row_text.iloc[left],
                    "right": row_text.iloc[right],
                }
            )
    return pd.DataFrame(rows)


class UnionFind:
    """Small deterministic union-find for semantic groups."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, value: int) -> int:
        """Return the canonical root for a value."""

        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: int, right: int) -> None:
        """Merge two sets while preserving the smaller root."""

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        root = min(left_root, right_root)
        other = max(left_root, right_root)
        self._parent[other] = root
