"""LOTUS-backed semantic group assignment lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.sem_join import row_text_series
from agent_memory.logical import QueryExpr

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
    key_columns = tuple(str(column) for column in query.params["key"])
    unique_rows, row_to_unique = exact_unique_key_rows(source, key_columns)
    matched_pairs = evaluate_group_matches(
        unique_rows,
        key_columns=key_columns,
        instruction=str(query.params["instruction"]),
    )
    result = assign_semantic_group_ids(
        source,
        key_columns=key_columns,
        matched_unique_pairs=matched_pairs,
        row_to_unique=row_to_unique,
    )
    result.attrs["agent_memory_groupby_key"] = key_columns
    return result


def assign_semantic_group_ids(
    source: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    matched_unique_pairs: Sequence[tuple[int, int]],
    row_to_unique: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Assign stable group ids from exact keys plus semantic pair matches."""

    unique_rows, inferred_row_to_unique = exact_unique_key_rows(source, key_columns)
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
    result.attrs["agent_memory_groupby_key"] = tuple(key_columns)
    return result


def exact_unique_key_rows(
    source: pd.DataFrame,
    key_columns: Sequence[str],
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Collapse exact duplicate keys before semantic pair comparisons."""

    missing = [column for column in key_columns if column not in source.columns]
    if missing:
        raise ValueError(f"sem_groupby key columns not found in DataFrame: {missing}")

    key_to_unique: dict[tuple[Any, ...], int] = {}
    row_to_unique: list[int] = []
    unique_row_indices: list[Any] = []
    for index, row in source.iterrows():
        key = tuple(row[column] for column in key_columns)
        if key not in key_to_unique:
            key_to_unique[key] = len(unique_row_indices)
            unique_row_indices.append(index)
        row_to_unique.append(key_to_unique[key])
    return source.loc[unique_row_indices, :].reset_index(drop=True), tuple(row_to_unique)


def evaluate_group_matches(
    unique_rows: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    instruction: str,
) -> list[tuple[int, int]]:
    """Evaluate candidate row pairs with LOTUS sem_filter."""

    if len(unique_rows) < 2:
        return []

    import lotus
    from lotus.sem_ops.sem_filter import sem_filter
    from lotus.templates import task_instructions

    pairs = semantic_pair_candidates(unique_rows, key_columns)
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
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Return i < j candidate pairs over exact-unique key rows."""

    row_text = row_text_series(unique_rows.loc[:, list(key_columns)], "row")
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
