"""LOTUS-backed semantic group assignment lowering."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

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
from agent_memory.adapters.lotus.sem_join import row_text_series
from agent_memory.adapters.lotus.structured import StructuredLMExecutor
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from agent_memory.storage.embedding import EmbeddingProvider
from agent_memory.tracing.semantic import (
    query_digest,
    write_compact_operator_trace,
    write_pair_trace,
)

GROUP_ID_COLUMN = "_agent_memory_group_id"
PAIRWISE_PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(?::(left|right))?\}(?!\})"
)


def execute_sem_groupby(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Assign deterministic semantic group ids to rows."""

    input_cols = tuple(str(column) for column in query.params["input_cols"])
    partition_by = tuple(str(column) for column in query.params.get("partition_by", ()))
    labels = tuple(query.params.get("labels") or ())
    digest = query_digest(query)
    profile = context.config.semantic_pair_profiles.get(digest)
    if labels and profile is not None:
        raise ValueError(
            "semantic pair profiles apply only to open-ended pairwise sem_groupby"
        )
    if profile is not None and profile.mode in {"search-filter", "proxy-only"}:
        if profile.direction != "symmetric":
            raise ValueError(
                f"pairwise sem_groupby {profile.mode} must be symmetric"
            )
        if context.pair_embedding_provider is None:
            raise ValueError(
                f"{profile.mode} requires a pair embedding provider"
            )
    context.configure()
    source = execute(query.inputs[0], inputs)
    validate_partition_by(source, partition_by)
    if partition_by:
        return execute_partitioned_sem_groupby(
            source,
            input_cols=input_cols,
            partition_by=partition_by,
            labels=labels,
            label_col=str(query.params.get("label_col", "_label")),
            instruction=str(query.params["instruction"]),
            default=context.config.sem_groupby_default,
            pair_batch_size=context.config.sem_groupby_pair_batch_size,
            pair_batch_retries=context.config.sem_groupby_pair_batch_retries,
            trace_dir=context.config.trace_dir(),
            query_digest_value=digest,
            profile=profile,
            embedding_provider=context.pair_embedding_provider,
        )
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
        default=context.config.sem_groupby_default,
        pair_batch_size=context.config.sem_groupby_pair_batch_size,
        pair_batch_retries=context.config.sem_groupby_pair_batch_retries,
        trace_dir=context.config.trace_dir(),
        query_digest_value=digest,
        profile=profile,
        embedding_provider=context.pair_embedding_provider,
    )
    result = assign_semantic_group_ids(
        source,
        input_cols=input_cols,
        matched_unique_pairs=matched_pairs,
        row_to_unique=row_to_unique,
    )
    result.attrs["agent_memory_groupby_input_cols"] = input_cols
    result.attrs["agent_memory_sem_groupby_partition_by"] = partition_by
    write_compact_operator_trace(
        context.config.trace_dir(),
        operator="sem_groupby",
        event_type="operator_result",
        input_frame=source,
        output_frame=result,
        payload={
            "instruction": str(query.params["instruction"]),
            "input_cols": list(input_cols),
        },
    )
    return result


def execute_partitioned_sem_groupby(
    source: pd.DataFrame,
    *,
    input_cols: Sequence[str],
    partition_by: Sequence[str],
    labels: Sequence[ColumnSpec],
    label_col: str,
    instruction: str,
    default: bool,
    pair_batch_size: int | None,
    pair_batch_retries: int,
    trace_dir: Any,
    query_digest_value: str,
    profile: SemanticPairExecutionProfile | None,
    embedding_provider: EmbeddingProvider | None,
) -> pd.DataFrame:
    """Assign semantic group ids independently within deterministic partitions."""

    if source.empty:
        result = source.copy()
        if labels:
            result[label_col] = []
        result[GROUP_ID_COLUMN] = []
        result.attrs["agent_memory_groupby_input_cols"] = tuple(input_cols)
        result.attrs["agent_memory_sem_groupby_partition_by"] = tuple(partition_by)
        return result

    parts: list[pd.DataFrame] = []
    next_group_id = 0
    grouped = source.groupby(list(partition_by), sort=False, dropna=False)
    for _partition_key, partition in grouped:
        if labels:
            result = assign_declared_labels(
                partition,
                input_cols=input_cols,
                labels=labels,
                label_col=label_col,
                instruction=instruction,
            )
        else:
            unique_rows, row_to_unique = exact_unique_key_rows(partition, input_cols)
            matched_pairs = evaluate_group_matches(
                unique_rows,
                input_cols=input_cols,
                instruction=instruction,
                default=default,
                pair_batch_size=pair_batch_size,
                pair_batch_retries=pair_batch_retries,
                trace_dir=trace_dir,
                query_digest_value=query_digest_value,
                profile=profile,
                embedding_provider=embedding_provider,
            )
            result = assign_semantic_group_ids(
                partition,
                input_cols=input_cols,
                matched_unique_pairs=matched_pairs,
                row_to_unique=row_to_unique,
            )
        if not result.empty:
            result[GROUP_ID_COLUMN] = result[GROUP_ID_COLUMN].astype(int) + next_group_id
            next_group_id = int(result[GROUP_ID_COLUMN].max()) + 1
        parts.append(result)

    combined = pd.concat(parts).sort_index().reset_index(drop=True)
    combined.attrs["agent_memory_groupby_input_cols"] = tuple(input_cols)
    combined.attrs["agent_memory_sem_groupby_partition_by"] = tuple(partition_by)
    if labels:
        combined.attrs["agent_memory_groupby_labels"] = tuple(label.name for label in labels)
        combined.attrs["agent_memory_groupby_label_col"] = label_col
    write_compact_operator_trace(
        trace_dir,
        operator="sem_groupby",
        event_type="operator_result",
        input_frame=source,
        output_frame=combined,
        payload={
            "instruction": instruction,
            "input_cols": list(input_cols),
            "partition_by": list(partition_by),
        },
    )
    return combined


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


def validate_partition_by(source: pd.DataFrame, partition_by: Sequence[str]) -> None:
    """Raise when deterministic partition columns are missing."""

    missing = [column for column in partition_by if column not in source.columns]
    if missing:
        raise ValueError(f"sem_groupby partition_by columns not found in DataFrame: {missing}")


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
    default: bool = False,
    pair_batch_size: int | None = None,
    pair_batch_retries: int = 0,
    trace_dir: Any = None,
    query_digest_value: str = "",
    profile: SemanticPairExecutionProfile | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[tuple[int, int]]:
    """Evaluate candidate row pairs with LOTUS sem_filter."""

    if len(unique_rows) < 2:
        return []

    if pair_batch_size is not None and pair_batch_size < 1:
        raise ValueError("sem_groupby pair_batch_size must be positive")
    if pair_batch_retries < 0:
        raise ValueError("sem_groupby pair_batch_retries cannot be negative")

    pairs = semantic_pair_candidates(unique_rows, input_cols)
    if profile is not None and profile.mode in {"search-filter", "proxy-only"}:
        if profile.direction != "symmetric":
            raise ValueError(
                f"pairwise sem_groupby {profile.mode} must be symmetric"
            )
        if embedding_provider is None:
            raise ValueError(
                f"{profile.mode} requires a pair embedding provider"
            )
        selection = select_semantic_pair_candidates(
            groupby_pair_candidate_projection(pairs),
            profile=profile,
            embedding_provider=embedding_provider,
        )
        write_semantic_pair_execution_trace(
            trace_dir,
            operator="sem_groupby",
            query_digest_value=query_digest_value,
            profile=profile,
            selection=selection,
        )
        pairs = pairs.iloc[list(selection.selected_positions)].reset_index(drop=True)
        if pairs.empty:
            return []
        if profile.mode == "proxy-only":
            return [
                (int(row["_left_unique_id"]), int(row["_right_unique_id"]))
                for _index, row in pairs.iterrows()
            ]

    import lotus
    from lotus.sem_ops.sem_filter import sem_filter
    from lotus.templates import task_instructions
    lowered_instruction = lower_pairwise_grouping_instruction(
        instruction,
        input_cols=input_cols,
    )
    user_instruction = (
        "{left} and {right} satisfy this semantic grouping condition: "
        f"{lowered_instruction}"
    )
    batch_size = pair_batch_size or len(pairs)
    parsed_outputs: list[bool] = []
    raw_outputs: list[Any] = []
    explanations: list[Any] = []
    for start in range(0, len(pairs), batch_size):
        pair_batch = pairs.iloc[start : start + batch_size].reset_index(drop=True)
        docs = task_instructions.df2multimodal_info(pair_batch, ["left", "right"])
        output = evaluate_group_match_batch(
            sem_filter,
            docs=docs,
            lm=lotus.settings.lm,
            instruction=user_instruction,
            default=default,
            retries=pair_batch_retries,
        )
        batch_outputs = list(output.outputs)
        if len(batch_outputs) != len(pair_batch):
            raise ValueError(
                "sem_groupby pair batch returned an unexpected number of outputs: "
                f"expected {len(pair_batch)}, got {len(batch_outputs)}"
            )
        parsed_outputs.extend(bool(value) for value in batch_outputs)
        raw_outputs.extend(
            aligned_batch_values(output, "raw_outputs", len(pair_batch), "")
        )
        explanations.extend(
            aligned_batch_values(output, "explanations", len(pair_batch), "")
        )

    write_groupby_pair_trace(
        trace_dir,
        pairs,
        source_instruction=instruction,
        instruction=user_instruction,
        outputs=parsed_outputs,
        raw_outputs=raw_outputs,
        explanations=explanations,
        default=default,
    )
    return [
        (int(row["_left_unique_id"]), int(row["_right_unique_id"]))
        for (_index, row), keep in zip(pairs.iterrows(), parsed_outputs)
        if keep
    ]


def groupby_pair_candidate_projection(pairs: pd.DataFrame) -> pd.DataFrame:
    """Project grouping pairs into the adapter's canonical candidate schema."""

    return pd.DataFrame(
        {
            PAIR_LEFT_ID_COLUMN: pairs["_left_unique_id"],
            PAIR_RIGHT_ID_COLUMN: pairs["_right_unique_id"],
            PAIR_LEFT_TEXT_COLUMN: pairs["left"],
            PAIR_RIGHT_TEXT_COLUMN: pairs["right"],
        }
    )


def evaluate_group_match_batch(
    sem_filter: Callable[..., Any],
    *,
    docs: Sequence[Any],
    lm: Any,
    instruction: str,
    default: bool,
    retries: int,
) -> Any:
    """Evaluate one physical pair batch, retrying only transient provider failures."""

    for attempt in range(retries + 1):
        try:
            return sem_filter(
                docs,
                lm,
                instruction,
                default=default,
                progress_bar_desc="Grouping comparisons",
            )
        except Exception as error:
            if attempt >= retries or not is_retryable_group_match_error(error):
                raise
    raise AssertionError("sem_groupby pair batch retry loop did not return or raise")


def aligned_batch_values(
    output: Any,
    attribute: str,
    expected: int,
    fill: Any,
) -> list[Any]:
    """Return one optional output value per pair without shifting later batches."""

    values = list(getattr(output, attribute, ()) or ())
    if len(values) < expected:
        values.extend(fill for _ in range(expected - len(values)))
    return values[:expected]


def is_retryable_group_match_error(error: Exception) -> bool:
    """Return whether LOTUS surfaced a transient LiteLLM provider failure."""

    from litellm.exceptions import (
        APIConnectionError,
        BadGatewayError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

    return isinstance(
        error,
        (
            APIConnectionError,
            BadGatewayError,
            InternalServerError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        ),
    )


def lower_pairwise_grouping_instruction(
    instruction: str,
    *,
    input_cols: Sequence[str],
) -> str:
    """Lower semantic-key placeholders for a two-row grouping predicate."""

    available = {str(column) for column in input_cols}

    def replace(match: re.Match[str]) -> str:
        column, side = match.groups()
        if side is not None:
            raise ValueError(
                "sem_groupby instructions describe one semantic key and cannot use "
                f"side-qualified placeholder {match.group(0)!r}"
            )
        if column not in available:
            raise ValueError(
                f"unknown sem_groupby input column placeholder: {column!r}; "
                f"available columns are {sorted(available)}"
            )
        return column

    return PAIRWISE_PLACEHOLDER_PATTERN.sub(replace, instruction)


def write_groupby_pair_trace(
    trace_dir: Any,
    pairs: pd.DataFrame,
    *,
    source_instruction: str,
    instruction: str,
    outputs: Sequence[bool],
    raw_outputs: Sequence[Any],
    explanations: Sequence[Any],
    default: bool,
) -> None:
    """Write one sem_groupby trace row per evaluated pair."""

    rows: list[dict[str, Any]] = []
    for index, (_row_index, pair) in enumerate(pairs.iterrows()):
        rows.append(
            {
                "operator": "sem_groupby",
                "source_instruction": source_instruction,
                "instruction": instruction,
                "left_unique_id": int(pair["_left_unique_id"]),
                "right_unique_id": int(pair["_right_unique_id"]),
                "left": pair["left"],
                "right": pair["right"],
                "parsed_output": bool(outputs[index]),
                "raw_output": raw_outputs[index] if index < len(raw_outputs) else "",
                "explanation": explanations[index] if index < len(explanations) else "",
                "default": default,
            }
        )
    write_pair_trace(
        trace_dir,
        operator="sem_groupby",
        rows=rows,
        snapshots={"pairs": pairs},
    )


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
