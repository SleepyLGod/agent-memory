"""Tests for the v0.0 interface layer."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

import agent_memory as am
from agent_memory.policy.aggregates import MinAggregateSpec, normalize_aggregate_specs
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    DEFAULT_STRUCTURED_PARSE_RETRIES,
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.relational import (
    execute_agg,
    execute_array_agg,
    execute_array_cat,
    execute_assign,
    execute_concat,
    execute_drop_duplicates,
    execute_explode,
    execute_filter,
    execute_flatten,
    execute_join,
    execute_min,
    execute_subtract,
    execute_unnest,
    execute_union,
    execute_union_by_name,
)
import agent_memory.adapters.lotus.relational as relational_module
from agent_memory.adapters.lotus.sem_agg import (
    GROUP_ID_COLUMN,
    JSON_OBJECT_RESPONSE_FORMAT,
    aggregate_groups,
    aggregate_input_columns,
    execute_sem_agg,
    lotus_style_sem_agg,
    parse_structured_sem_agg_output,
    structured_aggregate_instruction,
    structured_sem_agg_model_kwargs,
)
from agent_memory.adapters.lotus.sem_map import (
    apply_sem_map_output,
    apply_structured_map_outputs,
    native_sem_map_kwargs,
    normalize_strategy,
    parse_structured_map_json,
    resolve_input_cols as resolve_sem_map_input_cols,
    single_output_column,
    structured_instruction,
    temporary_map_column,
)
from agent_memory.adapters.lotus.sem_flat_map import (
    apply_flat_map_outputs,
    parse_structured_flat_map_json,
)
from agent_memory.adapters.lotus.sem_filter import (
    bind_qualified_filter_columns,
    execute_sem_filter,
    native_sem_filter_kwargs,
)
import agent_memory.adapters.lotus.sem_groupby as sem_groupby_module
from agent_memory.adapters.lotus.sem_groupby import (
    assign_declared_labels,
    assign_semantic_group_ids,
    evaluate_group_matches,
    lower_pairwise_grouping_instruction,
)
from agent_memory.adapters.lotus.sem_join import (
    assemble_join_frame,
    cascade_args_from_mapping,
    evaluate_semantic_join,
    execute_sem_join,
    join_series,
    renamed_columns,
)
import agent_memory.adapters.lotus.sem_topk as sem_topk_module
from agent_memory.adapters.lotus.sem_topk import execute_sem_topk, topk_instruction
from agent_memory.policy.expressions import ColumnExpr, LeastExpr, expr_from_param
import agent_memory.adapters.lotus.structured as structured_module
from agent_memory.adapters.lotus.structured import (
    STRUCTURED_RESERVED_MODEL_KWARGS,
    StructuredLMExecutor,
    StructuredGenerationResult,
    StructuredLMRetryResult,
    execute_structured_lm_with_retries,
    escape_structured_formatter_placeholders,
    parse_structured_object_json,
    reset_structured_retry_stats,
    structured_retry_stats,
    structured_instruction as build_structured_instruction,
    validate_model_kwargs,
)
from agent_memory.adapters.lotus.traced_lm import TracedLM
from agent_memory.datasets.locomo import flatten_locomo_rows
from agent_memory.planner import DifferentialInstructionRewriter, QueryDifferentiator
from agent_memory.planner.rules import DifferentialRules
from agent_memory.policy.logical import ColumnSpec, MemorySpec, MemoryView, QueryExpr, UserQuery
from agent_memory.policy.relation import GroupedRelation, Relation
from agent_memory.policy.schema import output_columns
from agent_memory.runtime import MemoryRuntime
from agent_memory.runtime.window import WINDOW_SOURCE_INPUT, completed_count_windows, over_frames
from agent_memory.tracing.semantic import semantic_trace_scope, write_compact_operator_trace


def trace_events(trace_dir: Path) -> list[dict[str, Any]]:
    """Read trace JSONL events from a test trace directory."""

    events_path = trace_dir / "events.jsonl"
    if not events_path.exists():
        return []
    return [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sem_agg_retry_result(
    raw_output: str,
    *,
    attempts: tuple[str, ...] | None = None,
    invalid: bool = False,
    failure_artifact_paths: tuple[Path, ...] = (),
) -> StructuredLMRetryResult:
    """Build one structured aggregate generation result for focused tests."""

    return StructuredLMRetryResult(
        raw_outputs=(raw_output,),
        raw_output_attempts=(attempts or (raw_output,),),
        invalid_indices=(0,) if invalid else (),
        failure_artifact_paths=failure_artifact_paths,
    )


def trace_artifact(trace_dir: Path, path_value: str) -> Any:
    """Read one JSON trace artifact referenced by an event."""

    path = trace_dir_from_event(trace_dir, path_value)
    return json.loads(path.read_text(encoding="utf-8"))


def trace_dir_from_event(trace_dir: Path, path_value: str) -> Path:
    """Resolve one trace artifact path from an event field."""

    return trace_dir / path_value.removeprefix("trace/")


def test_query_expr_params_are_deeply_frozen() -> None:
    expr = QueryExpr(
        op="test",
        params={
            "nested": {
                "items": [
                    {"name": "topic"},
                    ["memory", "catalog"],
                ]
            },
            "tags": {"durable", "semantic"},
        },
    )

    with pytest.raises(TypeError):
        expr.params["new"] = "value"

    nested = expr.params["nested"]
    with pytest.raises(TypeError):
        nested["items"] = ()

    items = nested["items"]
    assert isinstance(items, tuple)
    with pytest.raises(TypeError):
        items[0]["name"] = "changed"

    assert isinstance(items[1], tuple)
    assert isinstance(expr.params["tags"], tuple)


def test_query_expr_is_hashable_for_planner_memoization() -> None:
    expr = QueryExpr(
        op="sem_map",
        params={
            "input_cols": ["message"],
            "output_cols": {"topic_name": "Candidate topic name."},
        },
    )

    memo = {expr: "rewritten"}

    assert memo[expr] == "rewritten"


def test_query_expr_hash_is_canonical_for_param_key_order() -> None:
    left = QueryExpr(op="sem_map", params={"a": 1, "b": {"c": 2, "d": 3}})
    right = QueryExpr(op="sem_map", params={"b": {"d": 3, "c": 2}, "a": 1})

    assert left == right
    assert hash(left) == hash(right)


def test_message_is_public_minimal_append_input() -> None:
    message = am.Message(
        content="Please remember that I prefer concise docs.",
        role="user",
        timestamp="2026-05-23T00:00:00Z",
        session_id="session-1",
        metadata={"source": "test"},
    )

    assert message.content == "Please remember that I prefer concise docs."
    assert message.role == "user"
    assert message.metadata == {"source": "test"}
    with pytest.raises(TypeError):
        message.metadata["source"] = "changed"
    assert hasattr(am, "Message")


def test_message_is_hashable_with_metadata() -> None:
    left = am.Message(
        content="x",
        metadata={"b": [2, {"nested": "value"}], "a": 1},
    )
    right = am.Message(
        content="x",
        metadata={"a": 1, "b": [2, {"nested": "value"}]},
    )

    assert left == right
    assert hash(left) == hash(right)
    assert {left: "cached"}[right] == "cached"


def test_claude_memory_spec_collects_views_and_private_relations() -> None:
    spec = am.ClaudeMemory.spec()

    assert isinstance(spec, MemorySpec)
    assert spec.log.expr.op == "log"
    assert sorted(spec.views) == ["catalog", "topics"]
    assert sorted(spec.private_relations) == []
    assert sorted(spec.retrieval_queries) == ["default"]
    assert spec.retrieval_queries["default"].op == "select"
    assert "log" not in spec.views
    assert "retrieval_query" not in spec.views


def test_claude_memory_log_schema_is_explicit() -> None:
    spec = am.ClaudeMemory.spec()
    columns = spec.log.expr.params["columns"]

    assert all(isinstance(column, ColumnSpec) for column in columns)
    assert tuple(column.name for column in columns) == (
        "message",
        "role",
        "timestamp",
        "session_id",
        "metadata",
    )


def test_log_system_columns_are_opt_in_and_reserved() -> None:
    plain = am.Log({"message": "Message."})
    metadata = am.Log({"message": "Message."}, system_columns=True)

    assert output_columns(plain.expr) == ("message",)
    assert output_columns(metadata.expr) == (
        "message",
        "_row_id",
        "_added_at",
        "_add_seq",
    )
    assert metadata.expr.params["system_columns"] is True

    with pytest.raises(ValueError, match="reserved system column"):
        am.Log({"_row_id": "Caller-owned id."})


def test_runtime_generates_log_system_columns_and_restores_sequence() -> None:
    class MetadataMemory(am.Memory):
        log = am.Log({"message": "Message."}, system_columns=True)
        rows = log.select(["message", "_row_id", "_added_at", "_add_seq"])

    original = MetadataMemory(adapter=LotusAdapter())
    original.add({"message": "one"})
    original.add({"message": "two"})

    log = original._runtime._state["log"]
    assert list(log["_add_seq"]) == [0, 1]
    assert len(set(log["_row_id"])) == 2
    assert all(str(UUID(value)) == value for value in log["_row_id"])
    assert all(isinstance(value, datetime) for value in log["_added_at"])
    assert all(value.utcoffset().total_seconds() == 0 for value in log["_added_at"])
    assert list(original._runtime._state["rows"]["_add_seq"]) == [0, 1]

    restored = MetadataMemory(adapter=LotusAdapter())
    restored._runtime.restore_state(original._runtime.snapshot_state())
    restored.add({"message": "three"})

    assert list(restored._runtime._state["log"]["_add_seq"]) == [0, 1, 2]


def test_runtime_rejects_caller_owned_log_system_columns() -> None:
    class MetadataMemory(am.Memory):
        log = am.Log({"message": "Message."}, system_columns=True)
        rows = log.select(["message", "_row_id", "_added_at", "_add_seq"])

    memory = MetadataMemory(adapter=LotusAdapter())

    with pytest.raises(ValueError, match="reserved log system columns"):
        memory.add({"message": "spoofed", "_add_seq": 99})


def test_sem_flat_map_preserves_log_system_columns() -> None:
    source = pd.DataFrame(
        [
            {
                "content": "Caroline adopted a dog.",
                "_row_id": "row-1",
                "_added_at": pd.Timestamp("2026-01-01T00:00:00Z"),
                "_add_seq": 0,
            }
        ]
    )

    result = apply_flat_map_outputs(
        source,
        [[{"entity": "Caroline"}]],
        (ColumnSpec("entity"),),
    )

    assert result.loc[0, "_row_id"] == "row-1"
    assert result.loc[0, "_add_seq"] == 0
    assert result.loc[0, "entity"] == "Caroline"


def test_sem_flat_map_ordinal_resets_for_each_input_row() -> None:
    source = pd.DataFrame(
        [
            {"message": "first", "_add_seq": 4},
            {"message": "second", "_add_seq": 5},
            {"message": "empty", "_add_seq": 6},
        ]
    )

    result = apply_flat_map_outputs(
        source,
        [
            [{"entity": "Alice"}, {"entity": "Bob"}],
            [{"entity": "Carol"}],
            [],
        ],
        (ColumnSpec("entity"),),
        ordinal_col="entity_ordinal",
    )

    assert result.to_dict("records") == [
        {
            "message": "first",
            "_add_seq": 4,
            "entity": "Alice",
            "entity_ordinal": 0,
        },
        {
            "message": "first",
            "_add_seq": 4,
            "entity": "Bob",
            "entity_ordinal": 1,
        },
        {
            "message": "second",
            "_add_seq": 5,
            "entity": "Carol",
            "entity_ordinal": 0,
        },
    ]


def test_sem_flat_map_ordinal_is_declared_in_query_schema() -> None:
    log = am.Log({"message": "Message."}, system_columns=True)
    query = log.sem_flat_map(
        output_cols={"entity": "Extracted entity."},
        instruction="Extract entities from {message}.",
        ordinal_col="entity_ordinal",
    )

    assert query.expr.params["ordinal_col"] == "entity_ordinal"
    assert output_columns(query.expr) == (
        "message",
        "_row_id",
        "_added_at",
        "_add_seq",
        "entity",
        "entity_ordinal",
    )

    with pytest.raises(ValueError, match="ordinal_col.*conflicts"):
        log.sem_flat_map(
            output_cols={"entity_ordinal": "Extracted position."},
            instruction="Extract positions from {message}.",
            ordinal_col="entity_ordinal",
        )


def test_memory_spec_requires_a_log_relation() -> None:
    class MissingLogMemory(am.Memory):
        topics = am.Log().select(["message"])

    with pytest.raises(ValueError, match="must declare a Log relation"):
        MissingLogMemory.spec()


def test_memory_spec_rejects_multiple_log_relations() -> None:
    class MultipleLogMemory(am.Memory):
        log = am.Log()
        extra_log = am.Log()

    with pytest.raises(ValueError, match="declares multiple Log relations"):
        MultipleLogMemory.spec()


def test_memory_spec_does_not_merge_parent_memory_class_body() -> None:
    class ExtendedClaudeMemory(am.ClaudeMemory):
        extra = am.ClaudeMemory.log.select(["message"])

    with pytest.raises(ValueError, match="does not merge inherited memory declarations"):
        ExtendedClaudeMemory.spec()


def test_memory_spec_is_cached_after_first_collection() -> None:
    first = am.ClaudeMemory.spec()
    second = am.ClaudeMemory.spec()

    assert first is second


def test_topics_expression_uses_chain_groupby_then_aggregation() -> None:
    spec = am.ClaudeMemory.spec()
    topics_expr = spec.views["topics"].query

    assert topics_expr.op == "select"
    assert topics_expr.params["columns"] == ("name", "description", "type", "body")
    sem_agg_expr = topics_expr.inputs[0]
    assert sem_agg_expr.op == "sem_agg"
    assert sem_agg_expr.params["input_cols"] == (
        "name",
        "description",
        "type",
        "body",
        "message",
        "role",
        "timestamp",
        "session_id",
    )
    assert tuple(col.name for col in sem_agg_expr.params["output_cols"]) == (
        "name",
        "description",
        "type",
        "body",
    )
    sem_groupby_expr = sem_agg_expr.inputs[0]
    assert sem_groupby_expr.op == "sem_groupby"
    assert sem_groupby_expr.params["input_cols"] == ("name", "description", "type")
    projection_expr = sem_groupby_expr.inputs[0]
    assert projection_expr.op == "select"
    sem_flat_map_expr = projection_expr.inputs[0]
    assert sem_flat_map_expr.op == "sem_flat_map"
    assert sem_flat_map_expr.params["input_cols"] == (
        "role",
        "message",
        "timestamp",
        "session_id",
    )
    assert tuple(col.name for col in sem_flat_map_expr.params["output_cols"]) == (
        "name",
        "description",
        "type",
        "body",
    )
    extract_instruction = sem_flat_map_expr.params["instruction"]
    assert "Use the same memory contract" not in extract_instruction
    assert "{{memory name}}" not in extract_instruction
    assert "{{memory content" not in extract_instruction
    for phrase in ("user", "feedback", "project", "reference"):
        assert phrase in extract_instruction
    for phrase in (
        "If the user explicitly asks you to remember something",
        "If they ask you to forget something",
        "There are several discrete types of memory",
        "What NOT to save in memory",
        "Code patterns, conventions, architecture, file paths, or project structure",
        "Git history, recent changes, or who-changed-what",
        "Anything already documented in CLAUDE.md files.",
        "Ephemeral task details: in-progress work, temporary state, current conversation context.",
        "These exclusions apply even when the user explicitly asks you to save.",
        "Write each memory to its own file",
        "Keep the name, description, and type fields in memory files up-to-date with the content",
        "Organize memory semantically by topic, not chronologically",
        "Update or remove memories that turn out to be wrong or outdated",
        "Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        "name: {name}",
        "description: {description}",
        "type: {type}",
        "{body}",
    ):
        assert phrase in extract_instruction

    aggregation_instruction = sem_agg_expr.params["instruction"]
    assert "Use the same consolidation contract" not in aggregation_instruction
    assert "## Phase 4 — Prune and index" not in aggregation_instruction
    assert "Remove pointers" not in aggregation_instruction
    assert "MEMORY.md is an index" not in aggregation_instruction
    for phrase in (
        "You are performing a dream — a reflective pass",
        "Synthesize what you've learned recently into durable, well-organized memories",
        "## Phase 3 — Consolidate",
        "Merging new signal into existing topic",
        "Converting relative dates",
        "Deleting contradicted facts",
        "stale, wrong, or superseded",
        "Resolve contradictions",
        "Output one canonical memory row with {name}, {description}, {type}, and {body}.",
    ):
        assert phrase in aggregation_instruction
    group_instruction = sem_groupby_expr.params["instruction"]
    assert "same future markdown" in group_instruction
    assert "{name}" in group_instruction
    assert "{description}" in group_instruction
    assert "{type}" in group_instruction


def test_grouped_relation_sem_agg_returns_normal_relation() -> None:
    grouped = am.Log().sem_groupby(
        input_cols=["topic_name"],
        instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    )
    aggregated = grouped.sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols={
            "topic_name": "Canonical durable memory topic name.",
            "topic_content": "Merged durable memory content.",
        },
        instruction="Choose a canonical topic name and merge topic content.",
    )

    assert isinstance(grouped, GroupedRelation)
    assert isinstance(aggregated, Relation)
    assert aggregated.expr.op == "sem_agg"
    assert aggregated.expr.inputs[0].op == "sem_groupby"


def test_grouped_relation_agg_accepts_public_aggregate_specs() -> None:
    log = am.Log(
        {
            "episode_id": "Episode id.",
            "name": "Entity name.",
            "summary": "Entity summary.",
        }
    )
    aggregated = log.group_by("episode_id").agg(
        am.array_agg(columns=["name"], output_col="mentions"),
        am.collect_list(column="summary", output_col="raw_summaries"),
        am.sem_agg(
            input_cols=["summary"],
            output_cols={"episode_summary": "Episode-level entity summary."},
            instruction="Summarize entity mentions.",
        ),
    )

    assert aggregated.expr.op == "agg"
    assert aggregated.expr.inputs[0].op == "group_by"
    assert output_columns(aggregated.expr) == (
        "episode_id",
        "mentions",
        "raw_summaries",
        "episode_summary",
    )


def test_min_public_api_supports_global_grouped_and_mixed_aggregation() -> None:
    log = am.Log({"group": "Group.", "value": "Value."})

    global_min = log.min(column="value", output_col="minimum")
    grouped_min = log.group_by("group").min(column="value", output_col="minimum")
    mixed = log.group_by("group").agg(
        am.min(column="value", output_col="minimum"),
        am.collect_list(column="value", output_col="values"),
    )

    assert global_min.expr.op == "min"
    assert output_columns(global_min.expr) == ("minimum",)
    assert grouped_min.expr.op == "min"
    assert output_columns(grouped_min.expr) == ("group", "minimum")
    assert output_columns(mixed.expr) == ("group", "minimum", "values")

    semantic_grouped = log.sem_groupby(
        input_cols=["group"],
        instruction="Rows have the same {group}.",
    )
    with pytest.raises(NotImplementedError, match="sem_groupby.*min"):
        semantic_grouped.min(column="value", output_col="minimum")

    with pytest.raises(ValueError, match="conflicts with group key"):
        log.group_by("group").min(column="value", output_col="group")


def test_min_public_api_supports_composite_columns() -> None:
    log = am.Log(
        {
            "group": "Group.",
            "add_seq": "Append sequence.",
            "ordinal": "Occurrence ordinal.",
        }
    )

    global_min = log.min(
        columns=["add_seq", "ordinal"],
        output_col="occurrence_id",
    )
    grouped_min = log.group_by("group").min(
        columns=["add_seq", "ordinal"],
        output_col="occurrence_id",
    )
    mixed = log.group_by("group").agg(
        am.min(
            columns=["add_seq", "ordinal"],
            output_col="occurrence_id",
        ),
        am.collect_list(column="ordinal", output_col="ordinals"),
    )

    assert global_min.expr.params["columns"] == ("add_seq", "ordinal")
    assert grouped_min.expr.params["columns"] == ("add_seq", "ordinal")
    composite_spec = mixed.expr.params["aggregates"][0]
    assert isinstance(composite_spec, MinAggregateSpec)
    assert composite_spec.columns == ("add_seq", "ordinal")

    with pytest.raises(ValueError, match="exactly one of column or columns"):
        am.min(
            column="add_seq",
            columns=["add_seq", "ordinal"],
            output_col="occurrence_id",
        )
    with pytest.raises(ValueError, match="exactly one of column or columns"):
        am.min(output_col="occurrence_id")
    with pytest.raises(TypeError, match="sequence of column names"):
        am.min(columns="add_seq", output_col="occurrence_id")


def test_least_is_serializable_and_requires_two_operands() -> None:
    log = am.Log({"left": "Left value.", "right": "Right value."})
    expr = am.least(log.col("left"), log.col("right"), 10)

    assert isinstance(expr, LeastExpr)
    assert expr_from_param(expr.to_param()).to_param() == expr.to_param()
    assert expr.to_param()["kind"] == "least"

    with pytest.raises(ValueError, match="at least two operands"):
        am.least(log.col("left"))
    with pytest.raises(TypeError, match="relational expression or scalar literal"):
        am.least(log.col("left"), [1, 2])


def test_sem_groupby_agg_preserves_optional_partition_by_schema() -> None:
    log = am.Log(
        {
            "group_id": "Graph partition.",
            "name": "Entity name.",
            "body": "Evidence body.",
        }
    )
    aggregated = log.sem_groupby(
        input_cols=["name"],
        partition_by="group_id",
        instruction="Rows refer to the same entity.",
    ).agg(
        am.sem_agg(
            input_cols=["name", "body"],
            output_cols={"name": "Canonical entity name."},
            instruction="Choose canonical name.",
        ),
        am.array_agg(columns=["body"], output_col="evidence"),
    )

    sem_groupby_expr = aggregated.expr.inputs[0]
    assert sem_groupby_expr.params["partition_by"] == ("group_id",)
    assert output_columns(aggregated.expr) == ("group_id", "name", "evidence")


def test_sem_groupby_array_agg_public_api_is_not_supported() -> None:
    grouped = am.Log({"name": "Entity name."}).sem_groupby(
        input_cols=["name"],
        instruction="Rows refer to the same entity.",
    )

    with pytest.raises(NotImplementedError, match="sem_groupby.*array_agg"):
        grouped.array_agg(columns=["name"], output_col="names")


def test_sem_groupby_agg_rejects_array_only_specs() -> None:
    grouped = am.Log({"name": "Entity name."}).sem_groupby(
        input_cols=["name"],
        instruction="Rows refer to the same entity.",
    )

    with pytest.raises(NotImplementedError, match="requires at least one sem_agg"):
        grouped.agg(am.collect_list(column="name", output_col="names"))


def test_grouped_agg_rejects_duplicate_aggregate_outputs() -> None:
    grouped = am.Log({"name": "Entity name."}).group_by("name")

    with pytest.raises(ValueError, match="aggregate output columns"):
        grouped.agg(
            am.array_agg(columns=["name"], output_col="summary"),
            am.sem_agg(
                input_cols=["name"],
                output_cols={"summary": "Semantic summary."},
                instruction="Summarize.",
            ),
        )


def test_aggregate_spec_normalization_rejects_duplicate_outputs() -> None:
    with pytest.raises(ValueError, match="aggregate output columns must be unique"):
        normalize_aggregate_specs(
            (
                am.array_agg(columns=["name"], output_col="summary"),
                am.sem_agg(
                    input_cols=["name"],
                    output_cols={"summary": "Semantic summary."},
                    instruction="Summarize.",
                ),
            )
        )


def test_sem_groupby_accepts_declared_labels() -> None:
    grouped = am.Log().sem_groupby(
        input_cols=["title", "abstract"],
        instruction="Assign each paper to the best matching research area.",
        labels={
            "systems": "Systems and databases.",
            "ml": "Machine learning.",
        },
    )

    labels = grouped.expr.params["labels"]

    assert grouped.expr.params["label_col"] == "_label"
    assert tuple(label.name for label in labels) == ("systems", "ml")
    assert tuple(label.description for label in labels) == (
        "Systems and databases.",
        "Machine learning.",
    )


def test_sem_groupby_accepts_explicit_label_column() -> None:
    grouped = am.Log().sem_groupby(
        input_cols=["title"],
        instruction="Assign each paper to a label.",
        labels={"systems": "Systems papers."},
        label_col="paper_area",
    )

    assert grouped.expr.params["label_col"] == "paper_area"


def test_sem_groupby_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="labels cannot be empty"):
        am.Log().sem_groupby(
            input_cols=["title"],
            instruction="Assign each paper to a label.",
            labels={},
        )


def test_semantic_instruction_argument_shapes() -> None:
    log = am.Log()

    filtered = log.sem_filter(instruction="{message} contains a durable memory fact.")
    joined = log.sem_join(
        log,
        instruction="The left row and right row describe the same memory fact.",
        how="inner",
    )
    topk = log.sem_topk("Find relevant rows.", 3)

    assert filtered.expr.op == "sem_filter"
    assert joined.expr.op == "sem_join"
    assert topk.expr.op == "sem_topk"
    assert topk.expr.params["instruction"] == "Find relevant rows."
    assert topk.expr.params["k"] == 3

    with pytest.raises(TypeError):
        log.sem_filter("{message} contains a durable memory fact.")
    with pytest.raises(TypeError):
        log.sem_join(log, "The left row and right row describe the same memory fact.")


def test_join_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"name": "Topic identity.", "body": "Topic body."})

    joined = log.join(log, on="name", how="left")

    assert joined.expr.op == "join"
    assert joined.expr.params == {"on": ("name",), "how": "left"}
    assert joined.expr.inputs == (log.expr, log.expr)

    multi_key = log.join(log, on=["name", "body"])
    assert multi_key.expr.params == {"on": ("name", "body"), "how": "inner"}

    with pytest.raises(ValueError, match="at least one key"):
        log.join(log, on=[])
    with pytest.raises(TypeError):
        log.join("not a relation", on="name")


def test_union_by_name_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"name": "Topic identity.", "body": "Topic body."})

    unioned = log.union_by_name(log)

    assert unioned.expr.op == "union_by_name"
    assert unioned.expr.params == {"allow_missing_columns": True}
    assert unioned.expr.inputs == (log.expr, log.expr)

    strict = log.union_by_name(log, allow_missing_columns=False)
    assert strict.expr.params == {"allow_missing_columns": False}

    with pytest.raises(TypeError, match="allow_missing_columns"):
        log.union_by_name(log, allow_missing_columns="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        log.union_by_name("not a relation")  # type: ignore[arg-type]


def test_count_window_process_window_and_array_agg_query_expr_shape() -> None:
    log = am.Log(
        {
            "timestamp": "Message timestamp.",
            "speaker": "Message speaker.",
            "message": "Message body.",
        }
    )

    blocks = log.count_window(size=2, slide=1).process_window(
        lambda window: window.array_agg(
            columns=("timestamp", "speaker", "message"),
            output_col="conversation_records",
        )
    )

    assert blocks.expr.op == "process_window"
    window_query, process_query = blocks.expr.inputs
    assert window_query.op == "count_window"
    assert window_query.params == {"size": 2, "slide": 1, "trigger": None}
    assert window_query.inputs == (log.expr,)
    assert process_query.op == "array_agg"
    assert process_query.params == {
        "columns": ("timestamp", "speaker", "message"),
        "output_col": "conversation_records",
    }
    window_source = process_query.inputs[0]
    assert window_source.op == "window_source"
    assert window_source.params["columns"] == ("timestamp", "speaker", "message")


def test_process_window_builder_infers_filter_passthrough_columns() -> None:
    log = am.Log({"message": "Message body.", "speaker": "Message speaker."})

    blocks = log.filter(predicate=log.col("speaker") == "A").count_window(
        size=2,
        slide=1,
    ).process_window(
        lambda window: window.array_agg(
            columns=("speaker", "message"),
            output_col="conversation_records",
        )
    )

    process_query = blocks.expr.inputs[1]
    window_source = process_query.inputs[0]
    assert window_source.op == "window_source"
    assert window_source.params["columns"] == ("message", "speaker")


def test_process_window_builder_infers_join_suffix_columns() -> None:
    log = am.Log(
        {
            "id": "Message id.",
            "name": "Name.",
            "body": "Body.",
        }
    )

    joined = log.join(log, on="id")
    blocks = joined.count_window(size=2, slide=1).process_window(
        lambda window: window.array_agg(
            columns=("id", "name:left", "body:left", "name:right", "body:right"),
            output_col="joined_records",
        )
    )

    process_query = blocks.expr.inputs[1]
    window_source = process_query.inputs[0]
    assert window_source.op == "window_source"
    assert window_source.params["columns"] == (
        "id",
        "name:left",
        "body:left",
        "name:right",
        "body:right",
    )


def test_process_window_builder_infers_sem_join_suffix_columns() -> None:
    log = am.Log(
        {
            "name": "Name.",
            "body": "Body.",
        }
    )

    joined = log.sem_join(log, instruction="Rows describe the same memory.")
    blocks = joined.count_window(size=2, slide=1).process_window(
        lambda window: window.array_agg(
            columns=("name:left", "body:left", "name:right", "body:right"),
            output_col="joined_records",
        )
    )

    process_query = blocks.expr.inputs[1]
    window_source = process_query.inputs[0]
    assert window_source.op == "window_source"
    assert window_source.params["columns"] == (
        "name:left",
        "body:left",
        "name:right",
        "body:right",
    )


def test_count_window_and_array_agg_reject_invalid_arguments() -> None:
    log = am.Log({"message": "Message body."})

    with pytest.raises(ValueError, match="size must be positive"):
        log.count_window(size=0)
    with pytest.raises(ValueError, match="slide must be positive"):
        log.count_window(size=2, slide=0)
    with pytest.raises(TypeError, match="size must be an integer"):
        log.count_window(size=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="slide must be an integer"):
        log.count_window(size=2, slide="1")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="trigger=None"):
        log.count_window(size=2, trigger="early")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns cannot be empty"):
        log.array_agg(columns=(), output_col="records")
    with pytest.raises(ValueError, match="output_col cannot be empty"):
        log.array_agg(columns=("message",), output_col="")


def test_over_relation_array_agg_query_expr_shape() -> None:
    log = am.Log({"message": "Message body."})

    contextual = log.over(rows=(-2, -1)).array_agg(
        columns=("message",),
        output_col="previous_messages",
    )

    assert contextual.expr.op == "array_agg"
    over_query = contextual.expr.inputs[0]
    assert over_query.op == "over"
    assert over_query.params == {"rows": (-2, -1)}
    assert contextual.expr.params == {
        "columns": ("message",),
        "output_col": "previous_messages",
    }


def test_over_relation_rejects_invalid_rows() -> None:
    log = am.Log({"message": "Message body."})

    with pytest.raises(TypeError, match="tuple of two integers"):
        log.over(rows=[-2, -1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start must be <= end"):
        log.over(rows=(-1, -2))
    with pytest.raises(NotImplementedError, match="N <= 0"):
        log.over(rows=(-1, 1))


def test_array_cat_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"conversation_records": "JSON array aggregate state."})

    combined = log.array_cat(log, column="conversation_records")

    assert combined.expr.op == "array_cat"
    assert combined.expr.inputs == (log.expr, log.expr)
    assert combined.expr.params == {"column": "conversation_records"}

    with pytest.raises(ValueError, match="column cannot be empty"):
        log.array_cat(log, column="")
    with pytest.raises(TypeError):
        log.array_cat("not a relation", column="conversation_records")  # type: ignore[arg-type]


def test_flatten_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"nested_records": "JSON array of JSON array states."})

    flattened = log.flatten(column="nested_records", output_col="records")

    assert flattened.expr.op == "flatten"
    assert flattened.expr.inputs == (log.expr,)
    assert flattened.expr.params == {"column": "nested_records", "output_col": "records"}
    assert output_columns(flattened.expr) == ("nested_records", "records")

    same_column = log.flatten(column="nested_records")
    assert same_column.expr.params == {"column": "nested_records", "output_col": None}

    with pytest.raises(ValueError, match="column cannot be empty"):
        log.flatten(column="")
    with pytest.raises(ValueError, match="output_col cannot be empty"):
        log.flatten(column="nested_records", output_col="")


def test_explode_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"entity_id": "Entity id.", "mentions": "JSON array of mention records."})

    exploded = log.explode(column="mentions", output_col="_mention")

    assert exploded.expr.op == "explode"
    assert exploded.expr.inputs == (log.expr,)
    assert exploded.expr.params == {"column": "mentions", "output_col": "_mention"}
    assert output_columns(exploded.expr) == ("entity_id", "mentions", "_mention")

    same_column = log.explode(column="mentions")
    assert same_column.expr.params == {"column": "mentions", "output_col": None}
    assert output_columns(same_column.expr) == ("entity_id", "mentions")

    with pytest.raises(ValueError, match="column cannot be empty"):
        log.explode(column="")
    with pytest.raises(ValueError, match="output_col cannot be empty"):
        log.explode(column="mentions", output_col="")
    with pytest.raises(ValueError, match="already exists"):
        output_columns(log.explode(column="mentions", output_col="entity_id").expr)


def test_unnest_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"entity_id": "Entity id.", "_mention": "Mention record."})

    unnested = log.unnest(
        column="_mention",
        fields={"episode_id": "episode_id", "name": "mention_name"},
    )

    assert unnested.expr.op == "unnest"
    assert unnested.expr.inputs == (log.expr,)
    assert unnested.expr.params == {
        "column": "_mention",
        "fields": (("episode_id", "episode_id"), ("name", "mention_name")),
    }
    assert output_columns(unnested.expr) == ("entity_id", "episode_id", "mention_name")

    with pytest.raises(ValueError, match="column cannot be empty"):
        log.unnest(column="", fields={"episode_id": "episode_id"})
    with pytest.raises(ValueError, match="fields cannot be empty"):
        log.unnest(column="_mention", fields={})
    with pytest.raises(ValueError, match="output columns must be unique"):
        log.unnest(column="_mention", fields={"episode_id": "x", "name": "x"})
    with pytest.raises(ValueError, match="conflict"):
        output_columns(
            log.unnest(column="_mention", fields={"episode_id": "entity_id"}).expr
        )


def test_count_window_params_reject_non_integral_values() -> None:
    source = pd.DataFrame({"message": ["one", "two"]})

    with pytest.raises(TypeError, match="size must be an integer"):
        completed_count_windows(source, {"size": 2.5, "slide": 1})
    with pytest.raises(TypeError, match="slide must be an integer"):
        completed_count_windows(source, {"size": 2, "slide": "1"})
    with pytest.raises(TypeError, match="size must be an integer"):
        completed_count_windows(source, {"size": True, "slide": 1})


def test_over_frames_accept_dtype_different_append_suffix() -> None:
    source = pd.DataFrame({"message": pd.Series(["one", "two"], dtype="string")})
    emit = pd.DataFrame({"message": pd.Series(["two"], dtype="object")})

    frames = over_frames(emit, source, {"rows": (-1, -1)})

    assert [frame.emit_position for frame in frames] == [1]
    assert frames[0].frame.to_dict("records") == [{"message": "one"}]


def test_over_frames_reject_non_suffix_emit_rows() -> None:
    source = pd.DataFrame({"message": ["one", "two", "three"]})
    emit = pd.DataFrame({"message": ["one"]})

    with pytest.raises(NotImplementedError, match="append suffix"):
        over_frames(emit, source, {"rows": (-1, -1)})


def test_memory_spec_rejects_unclosed_windowed_relation() -> None:
    with pytest.raises(TypeError, match="process_window"):

        class UnclosedWindowMemory(am.Memory):
            log = am.Log({"message": "Message body."})
            blocks = log.count_window(size=2)

        UnclosedWindowMemory.spec()


def test_memory_spec_rejects_unclosed_over_relation() -> None:
    with pytest.raises(TypeError, match="OverRelation"):

        class UnclosedOverMemory(am.Memory):
            log = am.Log({"message": "Message body."})
            context = log.over(rows=(-2, -1))

        UnclosedOverMemory.spec()


def test_sem_join_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"message": "Raw message."})

    joined = log.sem_join(
        log,
        instruction="{message:left} and {message:right} describe the same memory fact.",
        how="outer",
    )

    params = joined.expr.params

    assert params == {
        "instruction": "{message:left} and {message:right} describe the same memory fact.",
        "how": "outer",
    }
    with pytest.raises(TypeError):
        log.sem_join(
            log,
            instruction="{message:left} and {message:right} describe the same memory fact.",
            cascade_args={"recall_target": 0.95},
        )


def test_sem_filter_query_expr_keeps_only_logical_params() -> None:
    log = am.Log({"message": "Raw message."})

    filtered = log.sem_filter(
        instruction="{message} contains a durable memory fact."
    )

    assert filtered.expr.params == {
        "instruction": "{message} contains a durable memory fact."
    }
    with pytest.raises(TypeError):
        log.sem_filter(
            instruction="{message} contains a durable memory fact.",
            examples=({"message": "Alice likes docs.", "Answer": True},),
        )


def _differentiate_with_defaults(
    query: QueryExpr,
    *,
    is_view_boundary: bool = False,
) -> QueryExpr:
    return DifferentialRules().differentiate(
        query,
        source_input=QueryExpr(op="log"),
        current_view=QueryExpr(op="materialized_view", params={"name": "view"}),
        is_view_boundary=is_view_boundary,
        instruction_rewriter=DifferentialInstructionRewriter(),
    )


def _query_ops(query: QueryExpr) -> tuple[str, ...]:
    return (query.op,) + tuple(
        op for input_query in query.inputs for op in _query_ops(input_query)
    )


def _query_nodes(query: QueryExpr) -> tuple[QueryExpr, ...]:
    return (query,) + tuple(
        node for input_query in query.inputs for node in _query_nodes(input_query)
    )


def _assert_materialized_view(
    query: QueryExpr,
    *,
    name: str,
    columns: tuple[str, ...] | None = None,
) -> None:
    assert query.op == "materialized_view"
    assert query.params["name"] == name
    if columns is not None:
        assert query.params["columns"] == columns


def test_differential_instruction_rewriter_groupby_to_join_rewrites_bare_inputs() -> None:
    instruction = (
        "Rows share a topic when {name} and {description} match. "
        "Keep {name:left} side-aware placeholders."
    )

    rewritten = DifferentialInstructionRewriter().groupby_to_join(
        instruction,
        input_cols=("name", "description"),
    )

    assert "{name:left} and {name:right}" in rewritten
    assert "{description:left} and {description:right}" in rewritten
    assert "Keep {name:left} side-aware placeholders." in rewritten


def test_differential_instruction_rewriter_agg_to_map_rewrites_overlapping_inputs() -> None:
    instruction = (
        "Output {name} and {body}. Use {timestamp} when deciding freshness."
    )

    rewritten = DifferentialInstructionRewriter().agg_to_map(
        instruction,
        input_cols=("name", "body", "timestamp"),
        output_cols=(ColumnSpec("name"), ColumnSpec("body")),
    )

    assert "{name:left} and {name:right}" in rewritten
    assert "{body:left} and {body:right}" in rewritten
    assert "{timestamp:left} and {timestamp:right}" in rewritten


def test_differential_instruction_rewriter_agg_to_map_preserves_output_only_placeholders() -> None:
    instruction = "Produce {summary} from {name}. Keep {name:left} explicit."

    rewritten = DifferentialInstructionRewriter().agg_to_map(
        instruction,
        input_cols=("name",),
        output_cols=(ColumnSpec("summary"),),
    )

    assert "{summary}" in rewritten
    assert "{name:left} and {name:right}" in rewritten
    assert "Keep {name:left} explicit." in rewritten


def test_differential_instruction_rewriter_matches_sem_join_suffix_convention() -> None:
    left = pd.DataFrame({"name": ["docs"], "left_only": ["left"]})
    right = pd.DataFrame({"name": ["documentation"], "right_only": ["right"]})
    left_columns, right_columns = renamed_columns(left, right)

    rewritten = DifferentialInstructionRewriter().agg_to_map(
        "Merge {name} into {summary}.",
        input_cols=("name",),
        output_cols=(ColumnSpec("summary"),),
    )

    assert left_columns["name"] == "name:left"
    assert right_columns["name"] == "name:right"
    assert "{name:left} and {name:right}" in rewritten


def test_differential_instruction_rewriter_rejects_unknown_placeholders() -> None:
    with pytest.raises(ValueError, match="unknown"):
        DifferentialInstructionRewriter().groupby_to_join(
            "Rows share a topic when {unknown} matches.",
            input_cols=("name",),
        )


def test_differential_instruction_rewriter_rewrites_state_reaggregation_placeholders() -> None:
    rewritten = DifferentialInstructionRewriter().state_reaggregation(
        "Merge raw {body} into canonical {summary}.",
        state_cols=("topic", "summary"),
        raw_input_cols=("body", "summary"),
    )

    assert rewritten == "Merge raw body into canonical {summary}."

    side_aware = DifferentialInstructionRewriter().state_reaggregation(
        "Keep {summary:right}.",
        state_cols=("summary:right",),
        raw_input_cols=(),
    )
    assert side_aware == "Keep {summary:right}."

    with pytest.raises(ValueError, match="not present in aggregate state"):
        DifferentialInstructionRewriter().state_reaggregation(
            "Use {body:left}.",
            state_cols=("summary",),
            raw_input_cols=("body",),
        )

    with pytest.raises(ValueError, match="not present in aggregate state"):
        DifferentialInstructionRewriter().state_reaggregation(
            "Use {unknown}.",
            state_cols=("summary",),
            raw_input_cols=("body",),
        )


def test_differential_instruction_rewriter_ignores_non_column_braces() -> None:
    instruction = "Keep {{name}} and {not a column} untouched, rewrite {name}."

    rewritten = DifferentialInstructionRewriter().groupby_to_join(
        instruction,
        input_cols=("name",),
    )

    assert "{{name}}" in rewritten
    assert "{not a column}" in rewritten
    assert "{name:left} and {name:right}" in rewritten


def test_differential_rules_reject_unsupported_operators() -> None:
    grouped = am.Log().sem_groupby(
        input_cols=["topic_name"],
        instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    )
    aggregated = grouped.sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols=["topic_name", "topic_content"],
        instruction="Merge topic rows.",
    )

    with pytest.raises(
        NotImplementedError,
        match=r"sem_agg differential is only supported",
    ):
        _differentiate_with_defaults(aggregated.expr)


def test_differential_rules_reject_malformed_unary_expr() -> None:
    query = QueryExpr(op="sem_filter")

    with pytest.raises(ValueError, match="expects exactly one input"):
        _differentiate_with_defaults(query)


def test_differential_rules_preserve_sem_filter_instruction() -> None:
    class FilterMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        helloworld_tests = log.sem_filter(
            instruction="{message} is a coherent sentence."
        )

    view = FilterMemory.spec().views["helloworld_tests"]
    differentiated = _differentiate_with_defaults(view.query)

    assert differentiated.op == view.query.op
    assert differentiated.inputs[0] == QueryExpr(op="log")
    assert differentiated.params["instruction"] == "{message} is a coherent sentence."


def test_differential_rules_reject_grouped_aggregation_outside_view_boundary() -> None:
    view = am.ClaudeMemory.spec().views["topics"]

    with pytest.raises(
        NotImplementedError,
        match=r"sem_agg differential is only supported",
    ):
        _differentiate_with_defaults(view.query)


def test_differential_query_planner_supports_sem_filter_views() -> None:
    class FilterMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        helloworld_tests = log.sem_filter(
            instruction="{message} is a coherent sentence."
        )

    view = FilterMemory.spec().views["helloworld_tests"]
    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "union"
    _assert_materialized_view(
        differentiated.inputs[0],
        name="helloworld_tests",
        columns=("message",),
    )
    fragment = differentiated.inputs[1]
    assert fragment.op == "sem_filter"
    assert fragment.inputs[0] == QueryExpr(op="log")
    assert fragment.params == view.query.params


def test_differential_query_planner_supports_sem_map_select_views() -> None:
    class MapMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        labels = log.sem_map(
            output_cols={
                "label": "One-word message label.",
                "summary": "Short message summary.",
            },
            instruction="Produce a label and summary for {message}.",
        ).select(["message", "label", "summary"])

    view = MapMemory.spec().views["labels"]
    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "union"
    _assert_materialized_view(
        differentiated.inputs[0],
        name="labels",
        columns=("message", "label", "summary"),
    )
    fragment = differentiated.inputs[1]
    assert fragment.op == "select"
    assert fragment.params == view.query.params
    sem_map_expr = fragment.inputs[0]
    assert sem_map_expr.op == "sem_map"
    assert sem_map_expr.inputs[0] == QueryExpr(op="log")
    assert sem_map_expr.params == view.query.inputs[0].params


def test_differential_query_planner_supports_standalone_sem_agg_views() -> None:
    class SummaryMemory(am.Memory):
        log = am.Log({"summary": "Memory summary.", "evidence": "Raw evidence."})
        summary = log.sem_agg(
            input_cols=["summary"],
            output_cols=["summary"],
            instruction="Merge summaries.",
        )

    view = SummaryMemory.spec().views["summary"]
    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "sem_agg"
    assert differentiated.params == view.query.params
    compressed_input = differentiated.inputs[0]
    assert compressed_input.op == "union"
    current_state, changed_state = compressed_input.inputs
    assert current_state.op == "select"
    assert current_state.params["columns"] == ("summary",)
    _assert_materialized_view(
        current_state.inputs[0],
        name="summary",
        columns=("summary",),
    )
    assert changed_state.op == "select"
    assert changed_state.params["columns"] == ("summary",)
    assert changed_state.inputs[0] == QueryExpr(op="log")


def test_differential_query_planner_supports_select_over_standalone_sem_agg_views() -> None:
    class SummaryMemory(am.Memory):
        log = am.Log({"summary": "Memory summary.", "evidence": "Raw evidence."})
        summary = log.sem_agg(
            input_cols=["summary"],
            output_cols=["summary"],
            instruction="Merge summaries.",
        ).select(["summary"])

    view = SummaryMemory.spec().views["summary"]
    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == ("summary",)
    aggregate = differentiated.inputs[0]
    assert aggregate.op == "sem_agg"
    assert aggregate.params == view.query.inputs[0].params
    compressed_input = aggregate.inputs[0]
    assert compressed_input.op == "union"
    current_state, changed_state = compressed_input.inputs
    assert current_state.op == "select"
    assert current_state.params["columns"] == ("summary",)
    _assert_materialized_view(
        current_state.inputs[0],
        name="summary",
        columns=("summary",),
    )
    assert changed_state.op == "select"
    assert changed_state.params["columns"] == ("summary",)
    assert changed_state.inputs[0] == QueryExpr(op="log")


def test_differential_query_planner_supports_sem_agg_after_row_local_fragment() -> None:
    class FilteredSummaryMemory(am.Memory):
        log = am.Log({"summary": "Memory summary.", "kind": "Message kind."})
        summary = (
            log
            .sem_filter(instruction="{summary} is durable.")
            .sem_agg(
                input_cols=["summary"],
                output_cols=["summary"],
                instruction="Merge durable summaries.",
            )
        )

    view = FilteredSummaryMemory.spec().views["summary"]
    differentiated = QueryDifferentiator().differentiate(view)

    changed_state = differentiated.inputs[0].inputs[1]
    changed_fragment = changed_state.inputs[0]
    assert changed_fragment.op == "sem_filter"
    assert changed_fragment.inputs[0] == QueryExpr(op="log")
    assert changed_state.params["columns"] == ("summary",)


def test_differential_query_planner_rejects_sem_agg_without_input_cols() -> None:
    class SummaryMemory(am.Memory):
        log = am.Log({"summary": "Memory summary."})
        summary = log.sem_agg(
            output_cols=["summary"],
            instruction="Merge summaries.",
        )

    view = SummaryMemory.spec().views["summary"]

    with pytest.raises(NotImplementedError, match="requires explicit input_cols"):
        QueryDifferentiator().differentiate(view)


def test_differential_query_planner_rejects_sem_agg_missing_current_view_columns() -> None:
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="log", params={"columns": (ColumnSpec("summary"),)}),),
        params={
            "input_cols": ("summary", "timestamp"),
            "output_cols": (ColumnSpec("summary"),),
            "instruction": "Merge summaries.",
        },
    )

    with pytest.raises(NotImplementedError, match="current view.*timestamp"):
        DifferentialRules().differentiate(
            query,
            source_input=QueryExpr(op="log"),
            current_view=QueryExpr(
                op="materialized_view",
                params={"name": "summary", "columns": ("summary",)},
            ),
            is_view_boundary=True,
            instruction_rewriter=DifferentialInstructionRewriter(),
        )


def test_differential_query_planner_rejects_sem_agg_missing_changed_fragment_columns() -> None:
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="log", params={"columns": (ColumnSpec("summary"),)}),),
        params={
            "input_cols": ("summary", "timestamp"),
            "output_cols": (ColumnSpec("summary"), ColumnSpec("timestamp")),
            "instruction": "Merge summaries.",
        },
    )

    with pytest.raises(NotImplementedError, match="changed sem_agg input fragment.*timestamp"):
        DifferentialRules().differentiate(
            query,
            source_input=QueryExpr(op="log"),
            current_view=QueryExpr(
                op="materialized_view",
                params={"name": "summary", "columns": ("summary", "timestamp")},
            ),
            is_view_boundary=True,
            instruction_rewriter=DifferentialInstructionRewriter(),
        )


def test_differential_rules_reject_standalone_sem_agg_outside_view_boundary() -> None:
    query = am.Log({"summary": "Memory summary."}).sem_agg(
        input_cols=["summary"],
        output_cols=["summary"],
        instruction="Merge summaries.",
    ).expr

    with pytest.raises(NotImplementedError, match="view boundary"):
        _differentiate_with_defaults(query)


def test_differential_query_planner_rewrites_array_agg_view_to_array_cat() -> None:
    class BlocksMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        blocks = log.array_agg(
            columns=("message",),
            output_col="conversation_records",
        )

    view = BlocksMemory.spec().views["blocks"]

    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "array_cat"
    _assert_materialized_view(
        differentiated.inputs[0],
        name="blocks",
        columns=("conversation_records",),
    )
    changed_aggregate = differentiated.inputs[1]
    assert changed_aggregate.op == "array_agg"
    assert changed_aggregate.inputs[0].op == "log"
    assert changed_aggregate.params == {
        "columns": ("message",),
        "output_col": "conversation_records",
    }
    assert differentiated.params == {"column": "conversation_records"}


def test_differential_query_planner_rewrites_group_by_array_agg_view_to_collect_list_flatten() -> None:
    class EpisodeEntitiesMemory(am.Memory):
        log = am.Log({"episode_id": "Episode id.", "entity": "Entity name."})
        entities = log.group_by("episode_id").array_agg(
            columns=("entity",),
            output_col="entities",
        )

    view = EpisodeEntitiesMemory.spec().views["entities"]

    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "flatten"
    assert differentiated.params == {"column": "entities", "output_col": None}
    aggregate = differentiated.inputs[0]
    assert aggregate.op == "agg"
    collect_spec = aggregate.params["aggregates"][0]
    assert collect_spec.column == "entities"
    assert collect_spec.output_col == "entities"
    assert aggregate.inputs[0].op == "group_by"
    combined = aggregate.inputs[0].inputs[0]
    assert combined.op == "concat"
    _assert_materialized_view(
        combined.inputs[0],
        name="entities",
        columns=("episode_id", "entities"),
    )
    changed_aggregate = combined.inputs[1]
    assert changed_aggregate.op == "array_agg"
    assert changed_aggregate.inputs[0].op == "group_by"


def test_differential_query_planner_rewrites_global_min_view() -> None:
    class EarliestMemory(am.Memory):
        log = am.Log({"valid_at": "Fact validity time."})
        earliest = log.min(column="valid_at", output_col="valid_at")

    differentiated = QueryDifferentiator().differentiate(
        EarliestMemory.spec().views["earliest"]
    )

    assert differentiated.op == "min"
    assert differentiated.params == {
        "columns": ("valid_at",),
        "output_col": "valid_at",
    }
    combined = differentiated.inputs[0]
    assert combined.op == "concat"
    assert combined.inputs[0].op == "min"
    _assert_materialized_view(
        combined.inputs[1],
        name="earliest",
        columns=("valid_at",),
    )


def test_differential_query_planner_supports_group_by_min_rule_families() -> None:
    class EarliestByTopicMemory(am.Memory):
        log = am.Log({"topic": "Topic.", "valid_at": "Fact validity time."})
        earliest = log.group_by("topic").min(
            column="valid_at",
            output_col="valid_at",
        )

    view = EarliestByTopicMemory.spec().views["earliest"]
    regrouped = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group"),
    ).differentiate(view)
    assert regrouped.op == "min"
    assert regrouped.params == {
        "columns": ("valid_at",),
        "output_col": "valid_at",
    }
    assert regrouped.inputs[0].op == "group_by"
    combined = regrouped.inputs[0].inputs[0]
    assert combined.op == "concat"
    assert combined.inputs[0].op == "min"
    _assert_materialized_view(
        combined.inputs[1],
        name="earliest",
        columns=("topic", "valid_at"),
    )

    joined = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-join-map"),
    ).differentiate(view)
    assert joined.op == "select"
    assert joined.params["columns"] == ("topic", "valid_at")
    assigned = joined.inputs[0]
    assert assigned.op == "assign"
    assert assigned.params["assignments"]["valid_at"]["kind"] == "least"
    assert assigned.inputs[0].op == "join"
    assert assigned.inputs[0].params == {"on": ("topic",), "how": "outer"}


def test_differential_query_planner_supports_composite_min_state() -> None:
    class EarliestOccurrenceMemory(am.Memory):
        log = am.Log(
            {
                "topic": "Topic.",
                "add_seq": "Append sequence.",
                "ordinal": "Occurrence ordinal.",
            }
        )
        earliest = log.group_by("topic").min(
            columns=["add_seq", "ordinal"],
            output_col="occurrence_id",
        )

    view = EarliestOccurrenceMemory.spec().views["earliest"]
    regrouped = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group"),
    ).differentiate(view)
    assert regrouped.params == {
        "columns": ("occurrence_id",),
        "output_col": "occurrence_id",
    }
    changed_min = regrouped.inputs[0].inputs[0].inputs[0]
    assert changed_min.params == {
        "columns": ("add_seq", "ordinal"),
        "output_col": "occurrence_id",
    }

    joined = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-join-map"),
    ).differentiate(view)
    assignment = joined.inputs[0].params["assignments"]["occurrence_id"]
    assert assignment["kind"] == "least"


def test_differential_query_planner_supports_group_by_sem_agg_re_group_rule() -> None:
    class TopicSummaryMemory(am.Memory):
        log = am.Log({"topic": "Topic.", "body": "Evidence body."})
        summaries = log.group_by("topic").sem_agg(
            input_cols=["body"],
            output_cols={"summary": "Topic summary."},
            instruction="Summarize {body}.",
        )

    view = TopicSummaryMemory.spec().views["summaries"]
    differentiated = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group"),
    ).differentiate(view)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == ("topic", "summary")
    final_aggregate = differentiated.inputs[0]
    assert final_aggregate.op == "sem_agg"
    assert final_aggregate.params["input_cols"] is None
    assert final_aggregate.inputs[0].op == "group_by"
    combined = final_aggregate.inputs[0].inputs[0]
    assert combined.op == "union_by_name"
    assert combined.inputs[0].op == "sem_agg"
    _assert_materialized_view(
        combined.inputs[1],
        name="summaries",
        columns=("topic", "summary"),
    )


def test_differential_query_planner_supports_group_by_mixed_agg_re_group_rule() -> None:
    class TopicEvidenceMemory(am.Memory):
        log = am.Log(
            {"topic": "Topic.", "body": "Evidence body.", "valid_at": "Valid time."}
        )
        summaries = log.group_by("topic").agg(
            am.sem_agg(
                input_cols=["body"],
                output_cols={"summary": "Topic summary."},
                instruction="Summarize {body}.",
            ),
            am.array_agg(columns=["body"], output_col="evidence"),
            am.min(column="valid_at", output_col="earliest"),
        )

    view = TopicEvidenceMemory.spec().views["summaries"]
    differentiated = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group"),
    ).differentiate(view)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == (
        "topic",
        "summary",
        "evidence",
        "earliest",
    )
    flattened = differentiated.inputs[0]
    assert flattened.op == "flatten"
    assert flattened.params == {"column": "evidence", "output_col": None}
    final_aggregate = flattened.inputs[0]
    assert final_aggregate.op == "agg"
    semantic_spec, collect_spec, min_spec = final_aggregate.params["aggregates"]
    assert semantic_spec.input_cols is None
    assert semantic_spec.instruction == "Summarize body."
    assert collect_spec.column == "evidence"
    assert collect_spec.output_col == "evidence"
    assert isinstance(min_spec, MinAggregateSpec)
    assert min_spec.columns == ("earliest",)
    assert min_spec.output_col == "earliest"
    assert final_aggregate.inputs[0].op == "group_by"
    combined = final_aggregate.inputs[0].inputs[0]
    assert combined.op == "concat"
    assert combined.inputs[1].op == "agg"


def test_differential_query_planner_supports_group_by_mixed_agg_join_map_rule() -> None:
    class TopicEvidenceMemory(am.Memory):
        log = am.Log(
            {"topic": "Topic.", "body": "Evidence body.", "valid_at": "Valid time."}
        )
        summaries = log.group_by("topic").agg(
            am.sem_agg(
                input_cols=["body"],
                output_cols={"summary": "Topic summary."},
                instruction="Summarize {body}.",
            ),
            am.array_agg(columns=["body"], output_col="evidence"),
            am.min(column="valid_at", output_col="earliest"),
        )

    view = TopicEvidenceMemory.spec().views["summaries"]
    differentiated = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-join-map"),
    ).differentiate(view)
    ops = _query_ops(differentiated)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == (
        "topic",
        "summary",
        "evidence",
        "earliest",
    )
    assert "join" in ops
    assert "sem_map" in ops
    assert "assign" in ops
    assert "array_merge_columns" not in ops
    assignments = next(
        node.params["assignments"]
        for node in _query_nodes(differentiated)
        if node.op == "assign"
    )
    assert assignments["evidence"]["kind"] == "array_cat"
    assert assignments["earliest"]["kind"] == "least"


def test_join_map_rejects_partitioned_semantic_grouping() -> None:
    log = am.Log(
        {"group_id": "Partition.", "name": "Name.", "body": "Evidence."}
    )
    direct = log.sem_groupby(
        input_cols=["name"],
        partition_by="group_id",
        instruction="Rows with {name} refer to the same entity.",
    ).sem_agg(
        input_cols=["name", "body"],
        output_cols={"name": "Canonical name.", "summary": "Summary."},
        instruction="Return {name} and summarize {body} as {summary}.",
    ).expr
    mixed = log.sem_groupby(
        input_cols=["name"],
        partition_by="group_id",
        instruction="Rows with {name} refer to the same entity.",
    ).agg(
        am.sem_agg(
            input_cols=["name", "body"],
            output_cols={"name": "Canonical name.", "summary": "Summary."},
            instruction="Return {name} and summarize {body} as {summary}.",
        ),
        am.min(column="body", output_col="first_body"),
    ).expr

    for query in (direct, mixed):
        with pytest.raises(
            NotImplementedError,
            match="partition_by.*rule-join-map",
        ):
            DifferentialRules(grouped_agg_rule="rule-join-map").differentiate(
                query,
                source_input=QueryExpr(op="log"),
                current_view=QueryExpr(
                    op="materialized_view",
                    params={"name": "view", "columns": output_columns(query)},
                ),
                is_view_boundary=True,
            )


def test_differential_rules_reject_array_agg_outside_view_boundary() -> None:
    query = am.Log({"message": "Message body."}).array_agg(
        columns=("message",),
        output_col="conversation_records",
    ).select(["conversation_records"]).expr

    with pytest.raises(NotImplementedError, match="array_agg changed-row differential"):
        _differentiate_with_defaults(query)


def test_differential_rules_support_sem_flat_map_fragments() -> None:
    query = am.Log({"message": "Raw input message."}).sem_flat_map(
        output_cols={"fact": "Extracted fact."},
        instruction="Extract facts from {message}.",
    ).expr

    differentiated = _differentiate_with_defaults(query)

    assert differentiated.op == "sem_flat_map"
    assert differentiated.inputs[0].op == "log"
    assert differentiated.params["instruction"] == "Extract facts from {message}."
    assert tuple(col.name for col in differentiated.params["output_cols"]) == ("fact",)


def test_differential_rules_preserve_sem_flat_map_ordinal_col() -> None:
    query = am.Log({"message": "Raw input message."}).sem_flat_map(
        output_cols={"fact": "Extracted fact."},
        instruction="Extract facts from {message}.",
        ordinal_col="fact_ordinal",
    ).expr

    differentiated = _differentiate_with_defaults(query)

    assert differentiated.op == "sem_flat_map"
    assert differentiated.params["ordinal_col"] == "fact_ordinal"
    assert output_columns(differentiated) == ("fact", "fact_ordinal")


def test_differential_rules_support_explode_and_unnest_fragments() -> None:
    query = (
        am.Log({"entity_id": "Entity id.", "mentions": "Mention records."})
        .explode(column="mentions", output_col="_mention")
        .unnest(
            column="_mention",
            fields={"episode_id": "episode_id", "name": "mention_name"},
        )
        .expr
    )

    differentiated = _differentiate_with_defaults(query)

    assert differentiated.op == "unnest"
    assert differentiated.params == {
        "column": "_mention",
        "fields": (("episode_id", "episode_id"), ("name", "mention_name")),
    }
    assert differentiated.inputs[0].op == "explode"
    assert differentiated.inputs[0].params == {"column": "mentions", "output_col": "_mention"}
    assert differentiated.inputs[0].inputs[0].op == "log"


def test_differential_query_planner_builds_claude_topics_full_next_view() -> None:
    view = am.ClaudeMemory.spec().views["topics"]

    differentiated = QueryDifferentiator().differentiate(view)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == ("name", "description", "type", "body")
    sem_agg_expr = differentiated.inputs[0]
    assert sem_agg_expr.op == "sem_agg"
    assert tuple(col.name for col in sem_agg_expr.params["output_cols"]) == (
        "name",
        "description",
        "type",
        "body",
    )
    sem_groupby_expr = sem_agg_expr.inputs[0]
    assert sem_groupby_expr.op == "sem_groupby"
    assert sem_groupby_expr.params["input_cols"] == (
        "name",
        "description",
        "type",
    )
    union_expr = sem_groupby_expr.inputs[0]
    assert union_expr.op == "union_by_name"
    assert union_expr.params == {"allow_missing_columns": True}
    assert union_expr.inputs[0].op == "select"
    _assert_materialized_view(
        union_expr.inputs[1],
        name="topics",
        columns=("name", "description", "type", "body"),
    )


def test_claude_topics_differentiated_query_no_longer_uses_join_map_merge() -> None:
    view = am.ClaudeMemory.spec().views["topics"]

    differentiated = QueryDifferentiator().differentiate(view)
    ops = _query_ops(differentiated)

    assert "union_by_name" in ops
    assert "sem_join" not in ops
    assert "sem_map" not in ops


def test_default_grouped_agg_rule_stays_compressed() -> None:
    view = am.ClaudeMemory.spec().views["topics"]

    differentiated = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="compressed"),
    ).differentiate(view)
    ops = _query_ops(differentiated)

    assert "union_by_name" in ops
    assert "assign" not in ops
    assert "filter" not in ops
    assert "join" not in ops


def test_changed_aware_grouped_agg_rule_emits_touched_group_shape() -> None:
    view = am.ClaudeMemory.spec().views["topics"]
    differentiated = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="changed-aware"),
    ).differentiate(view)
    ops = _query_ops(differentiated)

    assert differentiated.op == "union"
    assert len(differentiated.inputs) == 2
    kept, updates = differentiated.inputs
    assert kept.op == "select"
    assert kept.params["columns"] == ("name", "description", "type", "body")
    assert kept.inputs[0].op == "join"
    assert kept.inputs[0].params["how"] == "left_anti"
    assert updates.op == "select"
    assert updates.params["columns"] == ("name", "description", "type", "body")
    assert updates.inputs[0].op == "sem_agg"
    assert "assign" in ops
    assert "filter" in ops
    assert "join" in ops
    assert "left_anti" in [
        query.params.get("how")
        for query in _query_nodes(differentiated)
        if query.op == "join"
    ]
    assert "sem_join" not in ops
    assert "sem_map" not in ops


def test_changed_aware_partitioned_grouping_uses_partition_and_group_id() -> None:
    query = (
        am.Log({"group_id": "Partition.", "name": "Name.", "body": "Body."})
        .sem_groupby(
            input_cols=["name"],
            partition_by="group_id",
            instruction="Rows with {name} refer to the same entity.",
        )
        .sem_agg(
            input_cols=["name", "body"],
            output_cols={"name": "Canonical name.", "body": "Merged body."},
            instruction="Return {name} and merge {body}.",
        )
        .expr
    )

    differentiated = DifferentialRules(
        grouped_agg_rule="rule-all-group-optimized"
    ).differentiate(
        query,
        source_input=QueryExpr(op="log"),
        current_view=QueryExpr(
            op="materialized_view",
            params={"name": "entities", "columns": output_columns(query)},
        ),
        is_view_boundary=True,
    )

    touched_joins = [
        node
        for node in _query_nodes(differentiated)
        if node.op == "join" and node.params.get("how") in {"inner", "left_anti"}
    ]
    assert touched_joins
    assert all(
        node.params["on"] == ("group_id", GROUP_ID_COLUMN)
        for node in touched_joins
    )


def test_join_map_grouped_agg_rule_emits_join_map_shape() -> None:
    view = am.ClaudeMemory.spec().views["topics"]
    differentiated = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="join-map"),
    ).differentiate(view)
    ops = _query_ops(differentiated)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == ("name", "description", "type", "body")
    sem_map_expr = differentiated.inputs[0]
    assert sem_map_expr.op == "sem_map"
    sem_join_expr = sem_map_expr.inputs[0]
    assert sem_join_expr.op == "sem_join"
    assert sem_join_expr.params["how"] == "outer"
    assert "sem_join" in ops
    assert "sem_map" in ops
    assert "{name:left} and {name:right}" in sem_join_expr.params["instruction"]
    assert "{body:left} and {body:right}" in sem_map_expr.params["instruction"]


def test_changed_aware_grouped_agg_execution_keeps_old_only_groups() -> None:
    view = am.ClaudeMemory.spec().views["topics"]
    query = QueryDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="changed-aware"),
    ).differentiate(view)
    current = pd.DataFrame(
        [
            {
                "name": "caroline_adoption_goal",
                "description": "Caroline's adoption plan.",
                "type": "profile",
                "body": "Caroline is researching adoption agencies.",
            }
        ]
    )
    changed_log = pd.DataFrame(columns=["role", "message", "timestamp", "session_id"])

    result = LotusAdapter().execute(query, {"log": changed_log, "topics": current})

    assert result.to_dict(orient="records") == current.to_dict(orient="records")
    assert "_changed" not in result.columns
    assert GROUP_ID_COLUMN not in result.columns


def test_differential_rules_reject_unknown_grouped_agg_strategy() -> None:
    with pytest.raises(ValueError, match="grouped_agg_rule"):
        DifferentialRules(grouped_agg_rule="unknown")


def test_claude_topics_join_map_candidate_builder_rewrites_placeholders() -> None:
    view = am.ClaudeMemory.spec().views["topics"]
    current_view = QueryExpr(
        op="materialized_view",
        params={"name": "topics", "columns": ("name", "description", "type", "body")},
    )

    candidate = DifferentialRules()._build_sem_groupby_agg_join_map_candidate(
        view.query,
        changed_group_input=QueryExpr(op="log"),
        current_view=current_view,
        instruction_rewriter=DifferentialInstructionRewriter(),
    )

    assert candidate is not None
    assert candidate.op == "select"
    assert candidate.params["columns"] == ("name", "description", "type", "body")
    sem_map_expr = candidate.inputs[0]
    assert sem_map_expr.op == "sem_map"
    sem_join_expr = sem_map_expr.inputs[0]
    assert sem_join_expr.op == "sem_join"
    assert sem_join_expr.params["how"] == "outer"
    assert "{name:left} and {name:right}" in sem_join_expr.params["instruction"]
    assert "{description:left} and {description:right}" in sem_join_expr.params["instruction"]
    assert "{type:left} and {type:right}" in sem_join_expr.params["instruction"]
    assert "{body:left} and {body:right}" in sem_map_expr.params["instruction"]
    assert tuple(col.name for col in sem_map_expr.params["output_cols"]) == (
        "name",
        "description",
        "type",
        "body",
    )


def test_differential_query_planner_recomputes_views_from_materialized_dependencies() -> None:
    spec = am.ClaudeMemory.spec()
    catalog = spec.views["catalog"]

    differentiated = QueryDifferentiator().differentiate(
        catalog,
        views=spec.views,
    )

    assert differentiated.op == "select"
    assign_expr = differentiated.inputs[0]
    assert assign_expr.op == "assign"
    sem_map_expr = assign_expr.inputs[0]
    assert sem_map_expr.op == "sem_map"
    _assert_materialized_view(
        sem_map_expr.inputs[0],
        name="topics",
        columns=("name", "description", "type", "body"),
    )


def test_differential_query_planner_binds_source_query_dependencies() -> None:
    log = am.Log({"message": "Raw message."})
    source_relation = log.select(["message"])
    query = source_relation.sem_map(
        input_cols=["message"],
        output_cols={"summary": "Summary."},
        instruction="Summarize {message}.",
    ).expr
    source_input = QueryExpr(
        op="materialized_view",
        params={
            "name": "_changed_source",
            "columns": ("message",),
        },
    )

    differentiated = QueryDifferentiator().differentiate(
        MemoryView(name="summaries", query=query),
        views={
            "source_view": MemoryView(
                name="source_view",
                query=source_relation.expr,
            )
        },
        source_query=source_relation.expr,
        source_input=source_input,
    )

    assert differentiated.op == "union"
    _assert_materialized_view(
        differentiated.inputs[0],
        name="summaries",
        columns=("message", "summary"),
    )
    changed = differentiated.inputs[1]
    assert changed.op == "sem_map"
    assert changed.inputs[0] == source_input


def test_differential_rules_reject_mixed_log_and_materialized_view_fragments() -> None:
    query = QueryExpr(
        op="union",
        inputs=(
            QueryExpr(
                op="sem_filter",
                inputs=(QueryExpr(op="log"),),
                params={"instruction": "{message} is memory-worthy."},
            ),
            QueryExpr(
                op="sem_map",
                inputs=(QueryExpr(op="materialized_view", params={"name": "topics"}),),
                params={
                    "output_cols": (ColumnSpec("summary"),),
                    "instruction": "Summarize {body}.",
                },
            ),
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match=r"Mixed log \+ upstream materialized view",
    ):
        _differentiate_with_defaults(query)


def test_differential_rules_reject_generic_inner_sem_join() -> None:
    log = am.Log({"message": "Raw message."})
    query = log.sem_join(
        log,
        instruction="{message:left} and {message:right} describe the same memory.",
        how="inner",
    ).expr

    with pytest.raises(
        NotImplementedError,
        match=r"requires materialized old L/R state",
    ):
        _differentiate_with_defaults(query)


def test_differential_rules_reject_non_inner_sem_join() -> None:
    log = am.Log({"message": "Raw message."})
    query = log.sem_join(
        log,
        instruction="{message:left} and {message:right} describe the same memory.",
        how="outer",
    ).expr

    with pytest.raises(
        NotImplementedError,
        match=r"supports only how='inner'",
    ):
        _differentiate_with_defaults(query)


def test_claude_memory_has_no_private_topic_candidates() -> None:
    spec = am.ClaudeMemory.spec()

    assert "_topic_candidates" not in spec.private_relations


def test_catalog_expression_maps_from_topics() -> None:
    spec = am.ClaudeMemory.spec()
    catalog_expr = spec.views["catalog"].query

    assert catalog_expr.op == "select"
    assert catalog_expr.params["columns"] == ("catalog_title", "name", "hook")
    assign_expr = catalog_expr.inputs[0]
    assert assign_expr.op == "assign"
    catalog_title_expr = assign_expr.params["assignments"]["catalog_title"]
    assert dict(catalog_title_expr) == {
        "kind": "column",
        "name": "name",
        "qualifier": None,
    }
    sem_map_expr = assign_expr.inputs[0]
    assert sem_map_expr.op == "sem_map"
    assert sem_map_expr.params["input_cols"] == (
        "name",
        "description",
        "type",
        "body",
    )
    assert tuple(col.name for col in sem_map_expr.params["output_cols"]) == ("hook",)
    assert single_output_column(sem_map_expr).name == "hook"

    catalog_instruction = " ".join(sem_map_expr.params["instruction"].split())
    assert "plain-text relevance hook" in catalog_instruction
    assert "150 characters" in catalog_instruction
    assert "{name}" in catalog_instruction
    assert "{description}" in catalog_instruction
    assert "{type}" in catalog_instruction
    assert "{body}" in catalog_instruction
    assert "Do not return JSON" in catalog_instruction
    assert "multiple catalog entries" in catalog_instruction


def test_catalog_sem_map_dispatches_to_native_single_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_map as sem_map_module

    catalog_expr = am.ClaudeMemory.spec().views["catalog"].query
    sem_map_expr = catalog_expr.inputs[0].inputs[0]
    source = pd.DataFrame(
        [
            {
                "name": "Fresh Earth",
                "description": "A canonical topic about soil renewal.",
                "type": "project",
                "body": "Track experiments that improve depleted soil.",
            }
        ]
    )
    calls: list[str] = []

    def native(*args: Any) -> pd.DataFrame:
        calls.append("native")
        return source.assign(hook="Useful when discussing soil renewal experiments.")

    def structured(*args: Any) -> pd.DataFrame:
        raise AssertionError("single-output catalog must not use structured sem_map")

    monkeypatch.setattr(sem_map_module, "execute_native_sem_map", native)
    monkeypatch.setattr(sem_map_module, "execute_structured_sem_map", structured)
    context = SimpleNamespace(config=LotusExecutionConfig(), configure=lambda: None)

    result = sem_map_module.execute_sem_map(
        sem_map_expr,
        {},
        lambda query, inputs: source,
        context,
    )

    assert calls == ["native"]
    assert result["hook"].tolist() == [
        "Useful when discussing soil renewal experiments."
    ]


def test_differentiated_policy_compiles_views_and_retrieval_templates() -> None:
    policy = am.ClaudeMemory.differentiate_policy()

    assert policy.spec is am.ClaudeMemory.spec()
    assert set(policy.view_outputs) == {"topics", "catalog"}
    assert policy.execution_order[-1] == policy.view_outputs["catalog"]

    retrieval_query = policy.retrieval_queries["default"]
    assert retrieval_query.op == "select"
    assert retrieval_query.params["columns"] == (
        "name",
        "description",
        "type",
        "body",
    )
    join_query = retrieval_query.inputs[0]
    assert join_query.op == "join"
    assert join_query.params == {"on": ("name", "description", "type"), "how": "inner"}
    topk_query, topics_query = join_query.inputs
    _assert_materialized_view(topics_query, name="topics")
    assert topk_query.op == "sem_topk"
    assert topk_query.params["instruction"] == UserQuery()
    assert topk_query.params["k"] == 5
    manifest_query = topk_query.inputs[0]
    assert manifest_query.op == "select"
    assert manifest_query.params["columns"] == ("name", "description", "type")
    _assert_materialized_view(manifest_query.inputs[0], name="topics")


def test_differentiated_policy_compiles_window_process_node() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log(
            {
                "timestamp": "Message timestamp.",
                "speaker": "Message speaker.",
                "message": "Message body.",
            }
        )
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("timestamp", "speaker", "message"),
                output_col="conversation_records",
            )
        )

    policy = WindowBlockMemory.differentiate_policy()
    node_id = policy.view_outputs["blocks"]
    node = policy.nodes[node_id]

    assert node.execution_kind == "process_window"
    assert node.query.op == "process_window"
    assert node.query.inputs[0].op == "count_window"
    assert node.query.inputs[1].op == "array_agg"
    assert node.output_columns == ("conversation_records",)


def test_differentiated_policy_routes_window_changes_to_downstream_node() -> None:
    class WindowCandidateMemory(am.Memory):
        log = am.Log(
            {
                "timestamp": "Message timestamp.",
                "speaker": "Message speaker.",
                "message": "Message body.",
            }
        )
        candidates = (
            log
            .count_window(size=2, slide=1)
            .process_window(
                lambda window: window.array_agg(
                    columns=("timestamp", "speaker", "message"),
                    output_col="conversation_records",
                )
            )
            .sem_flat_map(
                input_cols=["conversation_records"],
                output_cols={"memory_summary": "Window-local memory summary."},
                instruction="Extract memory summaries from {conversation_records}.",
            )
            .select(["memory_summary"])
        )

    policy = WindowCandidateMemory.differentiate_policy()
    sink = policy.nodes[policy.view_outputs["candidates"]]
    sem_flat_map_node = policy.nodes[sink.input_node_ids[0]]
    process_node = policy.nodes[sem_flat_map_node.input_node_ids[0]]

    assert sink.query.op == "select"
    assert sem_flat_map_node.execution_kind == "semantic_row"
    assert sem_flat_map_node.query.params["input_cols"] == ("conversation_records",)
    assert process_node.execution_kind == "process_window"


def test_differentiated_policy_supports_deterministic_count_window_upstream() -> None:
    class AggregateWindowMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        records = log.array_agg(columns=("message",), output_col="records").count_window(
            size=2,
            slide=1,
        ).process_window(
            lambda window: window.array_agg(
                columns=("records",),
                output_col="window_records",
            )
        )

    policy = AggregateWindowMemory.differentiate_policy()
    sink = policy.nodes[policy.view_outputs["records"]]

    assert sink.execution_kind == "process_window"
    upstream = policy.nodes[sink.input_node_ids[0]]
    assert upstream.query.op == "array_agg"
    assert upstream.execution_kind == "deterministic"


@pytest.mark.parametrize(
    "message",
    [
        "Please remember that I prefer concise docs.",
        am.Message(content="Please remember that I prefer concise docs."),
        {"message": "Please remember that I prefer concise docs."},
    ],
)
def test_memory_add_normalizes_supported_input_forms(message: object) -> None:
    class InputMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        rows = log.select(["message"])

    memory = InputMemory(adapter=LotusAdapter())

    memory.add(message)

    assert memory._runtime._state["rows"].to_dict("records") == [
        {"message": "Please remember that I prefer concise docs."}
    ]


class HelloWorldTestMemory(am.Memory):
    log = am.Log(
        {
            "message": "Raw LOCOMO dialogue utterance.",
            "speaker": "Speaker name or role.",
            "session_id": "LOCOMO session identifier.",
            "turn_id": "Turn/dialogue identifier within the session.",
            "timestamp": "Session or turn timestamp when available.",
        }
    )

    helloworld_tests = (
        log
        .sem_filter(
            instruction="{message} is a coherent LOCOMO dialogue utterance that contains a concrete personal fact, preference, relationship, event, plan, or other memory-worthy information."
        )
        .sem_map(
            output_cols={
                "memory_summary": "One-sentence memory-oriented summary of the utterance, including the relevant speaker when needed.",
            },
            instruction="Produce a concise memory summary for {speaker}'s utterance: {message}.",
        )
        .select(["memory_summary"])
    )

    retrieval_query = helloworld_tests.sem_topk(am.UserQuery(), 2)


def test_lotus_adapter_wraps_plain_topk_query_with_columns() -> None:
    frame = pd.DataFrame({"message": ["hello"]})

    assert (
        topk_instruction(frame, "friendly greetings")
        == "{message} is relevant to: friendly greetings"
    )


def test_lotus_adapter_keeps_column_aware_topk_instruction() -> None:
    frame = pd.DataFrame({"message": ["hello"]})

    assert (
        topk_instruction(frame, "{message} is a friendly greeting")
        == "{message} is a friendly greeting"
    )


def test_lotus_adapter_topk_uses_all_columns_for_plain_queries() -> None:
    frame = pd.DataFrame({"speaker": ["Alice"], "message": ["hello"]})

    assert (
        topk_instruction(frame, "friendly greetings")
        == "{speaker}, {message} is relevant to: friendly greetings"
    )


def test_lotus_adapter_topk_requires_input_columns() -> None:
    frame = pd.DataFrame()

    with pytest.raises(ValueError, match="at least one input column"):
        topk_instruction(frame, "friendly greetings")


def test_sem_topk_query_expr_keeps_only_logical_params() -> None:
    relation = am.Log({"message": "Raw message."}).sem_topk(
        "friendly greetings",
        3,
    )

    params = relation.expr.params

    assert params == {"instruction": "friendly greetings", "k": 3}
    with pytest.raises(TypeError):
        am.Log({"message": "Raw message."}).sem_topk(
            "friendly greetings",
            3,
            method="quick",
        )


def test_sem_topk_accepts_user_query_placeholder() -> None:
    relation = am.Log({"message": "Raw message."}).sem_topk(am.UserQuery(), 3)

    assert relation.expr.params == {"instruction": am.UserQuery(), "k": 3}


def test_lotus_execution_context_passes_backend_lm_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    import lotus.models

    captured: dict[str, Any] = {}

    class FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            captured["lm_kwargs"] = kwargs

    def configure(**kwargs: Any) -> None:
        captured["configure_kwargs"] = kwargs

    monkeypatch.setattr(lotus.models, "LM", FakeLM)
    monkeypatch.setattr(lotus.settings, "configure", configure)

    context = LotusExecutionContext(
        model="deepseek/example",
        config=LotusExecutionConfig(
            lm_num_retries=2,
            lm_timeout=120,
            lm_max_batch_size=4,
            lm_rate_limit=10,
            lm_model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
            lm_enable_cache=False,
        ),
    )
    context.configure()

    assert captured["lm_kwargs"] == {
        "model": "deepseek/example",
        "max_batch_size": 4,
        "num_retries": 2,
        "timeout": 120,
        "rate_limit": 10,
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert isinstance(captured["configure_kwargs"]["lm"], FakeLM)
    assert captured["configure_kwargs"]["enable_cache"] is False


def test_lotus_execution_context_wraps_lm_only_when_trace_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus
    import lotus.models

    captured: dict[str, Any] = {}

    class FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def configure(**kwargs: Any) -> None:
        captured["configure_kwargs"] = kwargs

    monkeypatch.setattr(lotus.models, "LM", FakeLM)
    monkeypatch.setattr(lotus.settings, "configure", configure)

    context = LotusExecutionContext(
        model="deepseek/example",
        config=LotusExecutionConfig(semantic_trace_dir=tmp_path / "trace"),
    )
    context.configure()

    configured_lm = captured["configure_kwargs"]["lm"]
    assert isinstance(configured_lm, TracedLM)
    assert configured_lm.kwargs == {
        "model": "deepseek/example",
        "max_batch_size": 64,
    }


def test_lotus_execution_context_rejects_owned_lm_kwargs() -> None:
    context = LotusExecutionContext(
        model="deepseek/example",
        config=LotusExecutionConfig(lm_model_kwargs={"model": "other"}),
    )

    with pytest.raises(ValueError, match="cannot override.*model"):
        context.configure()


def test_provider_trace_only_exposes_thinking_type_from_extra_body() -> None:
    from agent_memory.adapters.lotus.provider_usage_lm import _safe_provider_kwargs

    safe = _safe_provider_kwargs(
        {
            "extra_body": {
                "thinking": {"type": "enabled", "private": "hidden"},
                "api_key": "secret",
            },
            "max_tokens": 8192,
        }
    )

    assert safe == {
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }


def test_traced_lm_writes_actual_prompt_and_raw_output(tmp_path: Path) -> None:
    class Output:
        outputs = ["Answer: True"]
        logprobs = None

    class FakeLM:
        model = "deepseek/test"
        max_tokens = 512
        max_ctx_len = 128000

        def __init__(self) -> None:
            self.stats = _fake_lm_stats(prompt_tokens=0, completion_tokens=0)
            self.messages: Any = None
            self.kwargs: dict[str, Any] | None = None

        def __call__(self, messages: Any, **kwargs: Any) -> Output:
            self.messages = messages
            self.kwargs = kwargs
            self.stats = _fake_lm_stats(prompt_tokens=7, completion_tokens=3)
            return Output()

        def count_tokens(self, _messages: Any) -> int:
            return 42

        def is_deepseek(self) -> bool:
            return True

    trace_dir = tmp_path / "trace"
    lm = TracedLM(FakeLM(), trace_dir)
    messages = [[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]]

    result = lm(messages, progress_bar_desc="Testing")

    assert result.outputs == ["Answer: True"]
    assert lm.count_tokens(messages[0]) == 42
    assert lm.is_deepseek() is True
    events = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event_type"] == "llm_call"
    assert events[0]["model"] == "deepseek/test"
    assert events[0]["llm_item_index"] == 0
    assert events[0]["usage_physical_total_tokens"] == 10
    assert "-llm-" in events[0]["prompt_path"]
    prompt_path = trace_dir / events[0]["prompt_path"].removeprefix("trace/")
    raw_path = trace_dir / events[0]["raw_output_path"].removeprefix("trace/")
    assert json.loads(prompt_path.read_text(encoding="utf-8")) == messages[0]
    assert json.loads(raw_path.read_text(encoding="utf-8")) == {"output": "Answer: True"}


def test_traced_lm_records_error_and_reraises(tmp_path: Path) -> None:
    class FakeLM:
        model = "deepseek/test"

        def __init__(self) -> None:
            self.stats = _fake_lm_stats(prompt_tokens=0, completion_tokens=0)

        def __call__(self, _messages: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("transport failed")

    trace_dir = tmp_path / "trace"
    lm = TracedLM(FakeLM(), trace_dir)

    with pytest.raises(RuntimeError, match="transport failed"):
        lm([[{"role": "user", "content": "hello"}]])

    events = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event_type"] == "llm_batch_error"
    assert events[0]["error_type"] == "RuntimeError"
    error_path = trace_dir / events[0]["error_output_path"].removeprefix("trace/")
    assert json.loads(error_path.read_text(encoding="utf-8")) == {
        "type": "RuntimeError",
        "message": "transport failed",
    }


def _fake_lm_stats(*, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    """Return a minimal LOTUS LMStats-like object for trace tests."""

    total = prompt_tokens + completion_tokens
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
    )
    return SimpleNamespace(
        physical_usage=usage,
        virtual_usage=usage,
        cache_hits=0,
    )


def test_lotus_execution_config_keeps_retry_defaults_disabled() -> None:
    config = LotusExecutionConfig()

    assert config.lm_num_retries is None
    assert config.lm_timeout is None
    assert config.lm_max_batch_size == 64
    assert config.lm_rate_limit is None
    assert config.lm_model_kwargs == {}
    assert config.lm_enable_cache is None
    assert config.semantic_trace_dir is None
    assert config.structured_parse_retries == DEFAULT_STRUCTURED_PARSE_RETRIES
    assert config.sem_topk_method == "pairwise-naive"


def test_lotus_adapter_forwards_topk_lotus_options() -> None:
    class Source:
        columns = ["message"]

        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        def __len__(self) -> int:
            return 1

        def sem_topk(self, instruction: str, **kwargs: Any) -> str:
            self.instruction = instruction
            self.kwargs = kwargs
            return "ranked"

    class Context:
        def __init__(self, config: LotusExecutionConfig) -> None:
            self.config = config

        def configure(self) -> None:
            pass

    source = Source()
    config = LotusExecutionConfig(
        sem_topk_method="pairwise-naive",
        sem_topk_strategy="ZS_COT",
        sem_topk_cascade_threshold=0.7,
        sem_topk_return_stats=True,
        sem_topk_safe_mode=True,
        sem_topk_return_explanations=True,
    )

    query = QueryExpr(
        op="sem_topk",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "instruction": "friendly greetings",
            "k": 3,
        },
    )

    result = execute_sem_topk(
        query,
        {},
        lambda _query, _inputs: source,
        Context(config=config),
    )

    assert result == "ranked"
    assert source.instruction == "{message} is relevant to: friendly greetings"
    assert source.kwargs is not None
    assert source.kwargs["K"] == 3
    assert source.kwargs["method"] == "naive"
    assert source.kwargs["cascade_threshold"] == 0.7
    assert source.kwargs["return_stats"] is True
    assert source.kwargs["safe_mode"] is True
    assert source.kwargs["return_explanations"] is True
    assert source.kwargs["strategy"].name == "ZS_COT"


@pytest.mark.parametrize(
    ("configured_method", "lotus_method"),
    [
        ("pairwise-naive", "naive"),
        ("pairwise-quick", "quick"),
        ("pairwise-heap", "heap"),
    ],
)
def test_lotus_adapter_maps_canonical_pairwise_topk_methods(
    configured_method: str,
    lotus_method: str,
) -> None:
    class Source:
        columns = ["message"]

        def __len__(self) -> int:
            return 2

        def sem_topk(self, _instruction: str, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ranked"

    class Context:
        config = LotusExecutionConfig(sem_topk_method=configured_method)

        def configure(self) -> None:
            pass

    source = Source()
    query = QueryExpr(
        op="sem_topk",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={"instruction": "plans", "k": 1},
    )

    assert execute_sem_topk(
        query,
        {},
        lambda _query, _inputs: source,
        Context(),
    ) == "ranked"
    assert source.kwargs["method"] == lotus_method


def test_lotus_adapter_dispatches_listwise_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Context:
        config = LotusExecutionConfig(sem_topk_method="listwise")

        def configure(self) -> None:
            pass

    source = pd.DataFrame(
        {
            "name": ["docs", "travel"],
            "description": ["Architecture notes", "Trip plans"],
        }
    )
    ranked = source.iloc[[1]].reset_index(drop=True)
    captured: dict[str, Any] = {}

    def execute_listwise_topk(
        frame: pd.DataFrame,
        *,
        instruction: str,
        k: int,
        context: Any,
    ) -> Any:
        captured.update(
            frame=frame,
            instruction=instruction,
            k=k,
            context=context,
        )
        return SimpleNamespace(
            frame=ranked,
            selected_ids=("row_1",),
            retry_count=0,
        )

    monkeypatch.setattr(
        sem_topk_module,
        "execute_listwise_topk",
        execute_listwise_topk,
        raising=False,
    )
    query = QueryExpr(
        op="sem_topk",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={"instruction": "architecture", "k": 1},
    )

    result = execute_sem_topk(
        query,
        {},
        lambda _query, _inputs: source,
        Context(),
    )

    pd.testing.assert_frame_equal(result, ranked)
    assert captured["frame"] is source
    assert captured["instruction"] == (
        "{name}, {description} is relevant to: architecture"
    )
    assert captured["k"] == 1


class FakeListwiseLM:
    """Return configured raw outputs for listwise top-k tests."""

    max_tokens = 512

    def __init__(self, raw_outputs: list[str]) -> None:
        self._raw_outputs = iter(raw_outputs)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        return SimpleNamespace(outputs=[next(self._raw_outputs)])


class StaticTopKContext:
    """Expose a preconfigured semantic top-k execution config."""

    def __init__(self, config: LotusExecutionConfig) -> None:
        self.config = config

    def configure(self) -> None:
        pass


def execute_listwise_test_query(
    source: pd.DataFrame,
    *,
    k: int,
    config: LotusExecutionConfig,
) -> pd.DataFrame:
    """Execute one listwise top-k query against an in-memory frame."""

    query = QueryExpr(
        op="sem_topk",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "instruction": "{name} and {description} are relevant to architecture",
            "k": k,
        },
    )
    return execute_sem_topk(
        query,
        {},
        lambda _query, _inputs: source,
        StaticTopKContext(config),
    )


def test_listwise_topk_uses_stable_ids_and_preserves_output_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "name": ["melanie", "travel", "melanie"],
            "description": ["Architecture docs", "Summer trip", "API design"],
            "body": ["first", "second", "third"],
        }
    )
    fake_lm = FakeListwiseLM(
        ['{"selected_ids": ["row_2", "row_0"]}']
    )
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)

    result = execute_listwise_test_query(
        source,
        k=2,
        config=LotusExecutionConfig(sem_topk_method="listwise"),
    )

    pd.testing.assert_frame_equal(result, source.iloc[[2, 0]].reset_index(drop=True))
    assert list(result.columns) == ["name", "description", "body"]
    assert len(fake_lm.calls) == 1
    messages, kwargs = fake_lm.calls[0]
    prompt_text = json.dumps(messages, ensure_ascii=False)
    assert all(candidate_id in prompt_text for candidate_id in ("row_0", "row_1", "row_2"))
    assert '"body"' not in prompt_text
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == 1024


def test_listwise_topk_deduplicates_repeated_instruction_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "name": ["docs", "travel"],
            "description": ["Architecture notes", "Trip plans"],
        }
    )
    fake_lm = FakeListwiseLM(['{"selected_ids": ["row_0"]}'])
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)
    query = QueryExpr(
        op="sem_topk",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "instruction": "Compare {name} to {name}; use {description}.",
            "k": 1,
        },
    )

    result = execute_sem_topk(
        query,
        {},
        lambda _query, _inputs: source,
        StaticTopKContext(LotusExecutionConfig(sem_topk_method="listwise")),
    )

    pd.testing.assert_frame_equal(result, source.iloc[[0]].reset_index(drop=True))
    messages, _kwargs = fake_lm.calls[0]
    user_payload = json.loads(messages[0][1]["content"])
    assert list(user_payload["candidates"][0]["row"]) == ["name", "description"]


def test_listwise_topk_retries_invalid_output_and_traces_retry_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "name": ["docs", "travel"],
            "description": ["Architecture notes", "Trip plans"],
        }
    )
    fake_lm = FakeListwiseLM(
        ["", '{"selected_ids": ["row_9"]}', '{"selected_ids": ["row_1"]}']
    )
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)
    trace_dir = tmp_path / "trace"

    result = execute_listwise_test_query(
        source,
        k=1,
        config=LotusExecutionConfig(
            sem_topk_method="listwise",
            structured_parse_retries=2,
            semantic_trace_dir=trace_dir,
        ),
    )

    pd.testing.assert_frame_equal(result, source.iloc[[1]].reset_index(drop=True))
    assert len(fake_lm.calls) == 3
    [event] = trace_events(trace_dir)
    assert event["operator"] == "sem_topk"
    assert event["sem_topk_method"] == "listwise"
    assert event["candidate_count"] == 2
    assert event["selected_ids"] == ["row_1"]
    assert event["retry_count"] == 2


@pytest.mark.parametrize(
    ("raw_output", "message"),
    [
        ('{"selected_ids": ["row_0", "row_0"]}', "unique"),
        ('{"selected_ids": ["row_9", "row_0"]}', "unknown"),
        ('{"selected_ids": ["row_0"]}', "exactly 2"),
        ('{"selected_ids": "row_0"}', "list"),
    ],
)
def test_listwise_topk_rejects_invalid_selection_contract(
    monkeypatch: pytest.MonkeyPatch,
    raw_output: str,
    message: str,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "name": ["docs", "travel"],
            "description": ["Architecture notes", "Trip plans"],
        }
    )
    monkeypatch.setattr(lotus.settings, "lm", FakeListwiseLM([raw_output]))

    with pytest.raises(ValueError, match=message):
        execute_listwise_test_query(
            source,
            k=2,
            config=LotusExecutionConfig(
                sem_topk_method="listwise",
                structured_parse_retries=0,
            ),
        )


def test_listwise_topk_returns_all_rows_when_k_exceeds_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "name": ["docs", "travel"],
            "description": ["Architecture notes", "Trip plans"],
        }
    )
    monkeypatch.setattr(
        lotus.settings,
        "lm",
        FakeListwiseLM(['{"selected_ids": ["row_1", "row_0"]}']),
    )

    result = execute_listwise_test_query(
        source,
        k=5,
        config=LotusExecutionConfig(sem_topk_method="listwise"),
    )

    pd.testing.assert_frame_equal(result, source.iloc[[1, 0]].reset_index(drop=True))


def test_listwise_topk_empty_input_skips_lm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    source = pd.DataFrame(columns=["name", "description"])
    fake_lm = FakeListwiseLM([])
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)

    result = execute_listwise_test_query(
        source,
        k=5,
        config=LotusExecutionConfig(sem_topk_method="listwise"),
    )

    pd.testing.assert_frame_equal(result, source)
    assert fake_lm.calls == []


def test_lotus_adapter_forwards_sem_filter_config_options() -> None:
    class Source:
        def __init__(self) -> None:
            self.instruction: str | None = None
            self.kwargs: dict[str, Any] | None = None

        def sem_filter(self, instruction: str, **kwargs: Any) -> str:
            self.instruction = instruction
            self.kwargs = kwargs
            return "filtered"

    class Context:
        def __init__(self, config: LotusExecutionConfig) -> None:
            self.config = config

        def configure(self) -> None:
            pass

    source = Source()
    config = LotusExecutionConfig(
        sem_filter_examples=(
            {"message": "Alice likes concise docs.", "Answer": True},
        ),
        sem_filter_helper_examples=(
            {"message": "green sleep quickly because table", "Answer": False},
        ),
        sem_filter_strategy="COT",
        sem_filter_default=False,
        sem_filter_cascade_args={"recall_target": 0.95},
        sem_filter_safe_mode=True,
        sem_filter_progress_bar_desc="Filtering memories",
        sem_filter_additional_cot_instructions="Be strict.",
    )
    query = QueryExpr(
        op="sem_filter",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={"instruction": "{message} contains a durable memory fact."},
    )

    result = execute_sem_filter(
        query,
        {},
        lambda _query, _inputs: source,
        Context(config=config),
    )

    assert result == "filtered"
    assert source.instruction == "{message} contains a durable memory fact."
    assert source.kwargs is not None
    assert list(source.kwargs["examples"]["Answer"]) == [True]
    assert list(source.kwargs["helper_examples"]["Answer"]) == [False]
    assert source.kwargs["strategy"].name == "COT"
    assert source.kwargs["default"] is False
    assert source.kwargs["cascade_args"].recall_target == 0.95
    assert source.kwargs["safe_mode"] is True
    assert source.kwargs["progress_bar_desc"] == "Filtering memories"
    assert source.kwargs["additional_cot_instructions"] == "Be strict."
    assert source.kwargs["return_raw_outputs"] is False
    assert source.kwargs["return_explanations"] is False
    assert source.kwargs["return_stats"] is False


def test_native_sem_filter_kwargs_keeps_debug_output_off_by_default() -> None:
    kwargs = native_sem_filter_kwargs(LotusExecutionConfig())

    assert kwargs["return_raw_outputs"] is False
    assert kwargs["return_explanations"] is False
    assert kwargs["return_stats"] is False


def test_sem_filter_binds_alias_qualified_columns_for_lotus_formatting() -> None:
    source = pd.DataFrame(
        {
            "fact:earlier_added": ["Alice lives in Paris."],
            "fact:later_added": ["Alice lives in London."],
            "fact_id:earlier_added": [(1, 0)],
        }
    )

    bound, instruction, restore_columns = bind_qualified_filter_columns(
        source,
        (
            "Does {fact:later_added} contradict "
            "{fact:earlier_added}?"
        ),
    )

    assert instruction == (
        "Does {fact_later_added} contradict {fact_earlier_added}?"
    )
    assert list(bound.columns) == [
        "fact_earlier_added",
        "fact_later_added",
        "fact_id:earlier_added",
    ]
    assert restore_columns == {
        "fact_earlier_added": "fact:earlier_added",
        "fact_later_added": "fact:later_added",
    }


def test_lotus_adapter_default_model_is_deepseek_v4_pro() -> None:
    adapter = LotusAdapter()

    assert adapter.model == "deepseek/deepseek-v4-pro"


def test_helloworld_locomo_loader_flattens_dialogue_rows() -> None:
    dataset = [
        {
            "conversation": {
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {
                        "dia_id": "1",
                        "speaker": "Alice",
                        "text": "I prefer concise design docs.",
                    },
                    {"dia_id": "2", "speaker": "Bob", "text": "   "},
                    {
                        "dia_id": "3",
                        "speaker": "Alice",
                        "text": "Let's meet Sarah tomorrow.",
                    },
                ],
            }
        }
    ]

    rows = flatten_locomo_rows(dataset, sample_limit=1, turn_limit=1)

    assert rows == [
        {
            "message": "I prefer concise design docs.",
            "speaker": "Alice",
            "sample_index": 0,
            "sample_id": "0",
            "session_id": "session_1",
            "turn_id": "1",
            "timestamp": "2024-01-01",
        }
    ]


def test_helloworld_policy_keeps_only_memory_view_columns() -> None:
    view_query = HelloWorldTestMemory.spec().views["helloworld_tests"].query

    assert view_query.op == "select"
    assert view_query.params["columns"] == ("memory_summary",)
    sem_map_query = view_query.inputs[0]
    assert sem_map_query.op == "sem_map"
    sem_filter_query = sem_map_query.inputs[0]
    assert sem_filter_query.op == "sem_filter"


def test_lotus_adapter_uses_non_conflicting_sem_map_temp_column() -> None:
    frame = pd.DataFrame(
        {
            "message": ["hello"],
            "_agent_memory_map": ["existing"],
            "_agent_memory_map_1": ["existing"],
        }
    )

    assert temporary_map_column(frame) == "_agent_memory_map_2"


def test_lotus_adapter_applies_single_output_sem_map_result() -> None:
    source = pd.DataFrame({"message": ["hello", "bye"]})
    mapped = pd.DataFrame(
        {
            "message": ["hello", "bye"],
            "_agent_memory_map": ["A greeting.", "A goodbye."],
        }
    )

    result = apply_sem_map_output(
        source,
        mapped,
        "_agent_memory_map",
        ColumnSpec("summary", "Short message summary."),
    )

    assert list(result.columns) == ["message", "summary"]
    assert list(result["summary"]) == ["A greeting.", "A goodbye."]


def test_single_output_sem_map_helper_rejects_multi_output() -> None:
    query = QueryExpr(
        op="sem_map",
        params={
            "output_cols": (
                ColumnSpec("label", "One-word message label."),
                ColumnSpec("summary", "Short message summary."),
            )
        },
    )

    with pytest.raises(NotImplementedError, match="one output column per sem_map"):
        single_output_column(query)


def test_lotus_adapter_accepts_single_output_sem_map() -> None:
    query = QueryExpr(
        op="sem_map",
        params={"output_cols": (ColumnSpec("summary", "Short message summary."),)},
    )

    assert single_output_column(query).name == "summary"


def test_sem_map_query_expr_keeps_only_logical_params() -> None:
    relation = am.Log({"message": "Raw message."}).sem_map(
        output_cols={"label": "Short label."},
        instruction="Label {message}.",
    )

    params = relation.expr.params

    assert params == {
        "input_cols": None,
        "output_cols": (ColumnSpec("label", "Short label."),),
        "instruction": "Label {message}.",
    }
    with pytest.raises(TypeError):
        am.Log({"message": "Raw message."}).sem_map(
            output_cols={"label": "Short label."},
            instruction="Label {message}.",
            system_prompt="Use terse labels.",
        )
    assert hash(relation.expr)


def test_native_sem_map_kwargs_forwards_adapter_config_options() -> None:
    config = LotusExecutionConfig(
        sem_map_system_prompt="Use terse labels.",
        sem_map_examples=({"message": "hello", "Answer": "greeting"},),
        sem_map_strategy="COT",
        sem_map_safe_mode=True,
        sem_map_progress_bar_desc="Labelling",
        sem_map_model_kwargs={"temperature": 0},
    )

    kwargs = native_sem_map_kwargs(config, suffix="_tmp")

    assert kwargs["suffix"] == "_tmp"
    assert kwargs["system_prompt"] == "Use terse labels."
    assert list(kwargs["examples"]["Answer"]) == ["greeting"]
    assert normalize_strategy("COT").name == "COT"
    assert kwargs["strategy"].name == "COT"
    assert kwargs["safe_mode"] is True
    assert kwargs["return_explanations"] is False
    assert kwargs["return_raw_outputs"] is False
    assert kwargs["progress_bar_desc"] == "Labelling"
    assert kwargs["temperature"] == 0


def test_structured_sem_map_instruction_preserves_instruction_and_schema() -> None:
    instruction = structured_instruction(
        "Produce memory fields for {message}.",
        (
            ColumnSpec("memory_type", "Short memory type."),
            ColumnSpec("summary", "One sentence summary."),
        ),
    )

    assert "Produce memory fields for {message}." in instruction
    assert '"memory_type"' in instruction
    assert '"summary"' in instruction
    assert "valid JSON object" in instruction
    assert "Output shape example" in instruction
    assert '{"memory_type": "string", "summary": "string"}' in instruction


def test_structured_flat_map_instruction_uses_rows_wrapper_shape_hint() -> None:
    instruction = build_structured_instruction(
        "Extract memory facts from {message}.",
        (
            ColumnSpec("memory_fact", "Atomic memory fact."),
            ColumnSpec("fact_type", "Short type label."),
        ),
        shape="array",
    )

    assert "Extract memory facts from {message}." in instruction
    assert 'a "rows" field containing an array' in instruction
    assert '{"rows": [{"memory_fact": "string", "fact_type": "string"}]}' in instruction


def test_structured_sem_map_uses_all_columns_when_instruction_has_no_placeholders() -> None:
    query = QueryExpr(
        op="sem_map",
        params={
            "input_cols": None,
            "output_cols": (ColumnSpec("summary"),),
            "instruction": "Merge rows into one summary.",
        },
    )
    source = pd.DataFrame({"name:left": ["docs"], "name:right": [pd.NA]})

    assert resolve_sem_map_input_cols(source, query) == ("name:left", "name:right")


def test_structured_input_inference_ignores_declared_output_placeholders() -> None:
    query = QueryExpr(
        op="sem_flat_map",
        params={
            "input_cols": None,
            "output_cols": (
                ColumnSpec("name"),
                ColumnSpec("description"),
            ),
            "instruction": (
                "Extract candidates from {message}, filling {name} and {description}."
            ),
        },
    )
    source = pd.DataFrame({"message": ["remember concise docs"]})

    assert resolve_sem_map_input_cols(source, query) == ("message",)


def test_structured_instruction_escapes_non_input_output_placeholders() -> None:
    instruction = escape_structured_formatter_placeholders(
        "Extract candidates from {message}, filling {name} and {description}.",
        input_cols=("message",),
        output_cols=(ColumnSpec("name"), ColumnSpec("description")),
    )

    assert instruction == (
        "Extract candidates from {message}, filling {{name}} and {{description}}."
    )


def test_structured_instruction_escapes_side_aware_placeholders() -> None:
    instruction = escape_structured_formatter_placeholders(
        "Merge {name:left} with {name:right}, filling {name}.",
        input_cols=("name:left", "name:right"),
        output_cols=(ColumnSpec("name"),),
    )

    assert instruction == "Merge {{name:left}} with {{name:right}}, filling {{name}}."


def test_structured_sem_map_parses_and_applies_multiple_outputs() -> None:
    output_cols = (
        ColumnSpec("label", "Short label."),
        ColumnSpec("summary", "Short summary."),
    )
    parsed = [
        parse_structured_map_json(
            '{"label": "greeting", "summary": "Alice says hello.", "extra": "ignored"}',
            output_cols,
        )
    ]
    source = pd.DataFrame({"message": ["hello"]})

    result = apply_structured_map_outputs(
        source,
        parsed,
        output_cols,
        raw_outputs=['{"label": "greeting", "summary": "Alice says hello."}'],
        explanations=[None],
    )

    assert list(result.columns) == [
        "message",
        "label",
        "summary",
        "raw_output_sem_map",
        "explanation_sem_map",
    ]
    assert result.loc[0, "label"] == "greeting"
    assert result.loc[0, "summary"] == "Alice says hello."


def test_structured_sem_map_parses_required_explanation() -> None:
    output, explanation = parse_structured_object_json(
        '{"label": "greeting", "_explanation": "The message greets someone."}',
        (ColumnSpec("label", "Short label."),),
        require_explanation=True,
        operator="sem_map",
    )

    assert output == {"label": "greeting"}
    assert explanation == "The message greets someone."


def test_structured_output_preserves_json_scalar_types() -> None:
    output, explanation = parse_structured_object_json(
        '{"name": "Alice", "ordinal": 2, "active": true, "invalid_at": null}',
        (
            ColumnSpec("name"),
            ColumnSpec("ordinal"),
            ColumnSpec("active"),
            ColumnSpec("invalid_at"),
        ),
        require_explanation=False,
        operator="sem_map",
    )

    assert output == {
        "name": "Alice",
        "ordinal": 2,
        "active": True,
        "invalid_at": None,
    }
    assert explanation is None


@pytest.mark.parametrize("nested", ['["Alice"]', '{"name": "Alice"}'])
def test_structured_output_rejects_nested_field_values(nested: str) -> None:
    with pytest.raises(ValueError, match="JSON scalar"):
        parse_structured_object_json(
            f'{{"value": {nested}}}',
            (ColumnSpec("value"),),
            require_explanation=False,
            operator="sem_map",
        )


def test_structured_instruction_requests_json_scalars_not_only_strings() -> None:
    instruction = build_structured_instruction(
        "Return an ordinal and optional invalidation time.",
        (
            ColumnSpec("ordinal", "Integer position."),
            ColumnSpec("invalid_at", "Timestamp or null."),
        ),
        shape="object",
    )

    assert "JSON scalar" in instruction
    assert "JSON null" in instruction
    assert "string values for every field" not in instruction


def test_structured_sem_map_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_map_json("not json", (ColumnSpec("label"),))


def test_structured_sem_map_rejects_missing_required_output() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        parse_structured_map_json('{"label": "greeting"}', (ColumnSpec("summary"),))


def test_structured_sem_map_requires_explanation_when_requested() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        parse_structured_map_json(
            '{"label": "greeting"}',
            (ColumnSpec("label"),),
            require_explanation=True,
        )


def test_structured_sem_map_rejects_reserved_model_kwargs() -> None:
    with pytest.raises(ValueError, match="response_format"):
        validate_model_kwargs(
            {"response_format": {"type": "json_object"}},
            reserved=STRUCTURED_RESERVED_MODEL_KWARGS,
            operator="sem_map",
        )


def test_structured_sem_map_executor_call_does_not_receive_writeback_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_map as sem_map_module

    captured: dict[str, Any] = {}

    class Executor:
        def __init__(self, source: pd.DataFrame) -> None:
            captured["source_columns"] = tuple(source.columns)

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            captured["kwargs"] = kwargs
            return StructuredGenerationResult(
                parsed_outputs=[{"label": "greeting", "summary": "Alice says hello."}],
                raw_outputs=[
                    '{"label": "greeting", "summary": "Alice says hello.", "_explanation": "The row is a greeting."}'
                ],
                explanations=["The row is a greeting."],
            )

    monkeypatch.setattr(sem_map_module, "StructuredLMExecutor", Executor)
    source = pd.DataFrame({"message": ["hello"]})
    output_cols = (
        ColumnSpec("label", "Short label."),
        ColumnSpec("summary", "Short summary."),
    )
    query = QueryExpr(
        op="sem_map",
        params={
            "instruction": "Classify {message}.",
            "output_cols": output_cols,
        },
    )
    config = LotusExecutionConfig(sem_map_strategy="COT")

    result = sem_map_module.execute_structured_sem_map(query, source, output_cols, config)

    assert captured["source_columns"] == ("message",)
    assert "return_raw_outputs" not in captured["kwargs"]
    assert captured["kwargs"]["return_explanations"] is False
    assert captured["kwargs"]["strategy"] == "COT"
    assert list(result.columns) == ["message", "label", "summary"]


def test_structured_executor_array_shape_uses_json_object_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    class Output:
        outputs = ['{"rows": [{"topic": "docs"}]}']

    class FakeLM:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        def __call__(self, prompts: object, **kwargs: Any) -> Output:
            self.kwargs = kwargs
            return Output()

    fake_lm = FakeLM()
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    source = pd.DataFrame({"message": ["prefers concise docs"]})

    result = StructuredLMExecutor(source)(
        input_cols=("message",),
        output_cols=(ColumnSpec("topic", "Candidate topic."),),
        instruction="Extract topics from {message}.",
        shape="array",
        progress_bar_desc="Flat mapping",
        model_kwargs={},
        operator="sem_flat_map",
    )

    assert fake_lm.kwargs is not None
    assert fake_lm.kwargs["response_format"] == {"type": "json_object"}
    assert fake_lm.kwargs["max_tokens"] >= DEFAULT_STRUCTURED_MAX_TOKENS
    assert result.parsed_outputs == [[{"topic": "docs"}]]


def test_structured_executor_trace_disabled_writes_no_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus

    class Output:
        outputs = ['{"label": "docs"}']

    class FakeLM:
        max_tokens = 512

        def __call__(self, prompts: object, **kwargs: Any) -> Output:
            return Output()

    monkeypatch.setattr(lotus.settings, "lm", FakeLM())
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    StructuredLMExecutor(pd.DataFrame({"message": ["prefers docs"]}))(
        input_cols=("message",),
        output_cols=(ColumnSpec("label"),),
        instruction="Label {message}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={},
        operator="sem_map",
    )

    assert trace_events(tmp_path) == []


def test_structured_executor_trace_writes_input_raw_and_parsed_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus

    class Output:
        outputs = ['{"label": "docs"}']

    class FakeLM:
        max_tokens = 512

        def __call__(self, prompts: object, **kwargs: Any) -> Output:
            return Output()

    monkeypatch.setattr(lotus.settings, "lm", FakeLM())
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    result = StructuredLMExecutor(pd.DataFrame({"message": ["prefers docs"]}))(
        input_cols=("message",),
        output_cols=(ColumnSpec("label", "Short label."),),
        instruction="Label {message}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={},
        semantic_trace_dir=tmp_path,
        operator="sem_map",
    )

    assert result.parsed_outputs == [{"label": "docs"}]
    [event] = trace_events(tmp_path)
    assert event["operator"] == "sem_map"
    assert event["event_type"] == "structured_generation"
    assert event["row_index"] == 0
    assert event["input_preview"] == '{ "message": "prefers docs" }'
    input_snapshot = pd.read_csv(trace_dir_from_event(tmp_path, event["input_snapshot_path"]))
    assert input_snapshot.to_dict("records") == [{"message": "prefers docs"}]
    assert trace_artifact(tmp_path, event["raw_output_path"]) == ['{"label": "docs"}']
    assert trace_artifact(tmp_path, event["parsed_output_path"]) == {"label": "docs"}
    assert event["required_output_cols"] == [
        {"name": "label", "description": "Short label."}
    ]
    assert event["parse_retry_attempts"] == 0
    assert event["parse_error"] == ""


def test_structured_executor_trace_records_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus

    class Output:
        def __init__(self, outputs: list[str]) -> None:
            self.outputs = outputs

    class FakeLM:
        max_tokens = 512

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, prompts: object, **kwargs: Any) -> Output:
            self.calls += 1
            if self.calls == 1:
                return Output([""])
            return Output(['{"label": "docs"}'])

    fake_lm = FakeLM()
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    result = StructuredLMExecutor(pd.DataFrame({"message": ["prefers docs"]}))(
        input_cols=("message",),
        output_cols=(ColumnSpec("label"),),
        instruction="Label {message}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={},
        structured_parse_retries=1,
        semantic_trace_dir=tmp_path,
        operator="sem_map",
    )

    assert result.raw_output_attempts == (("", '{"label": "docs"}'),)
    [event] = trace_events(tmp_path)
    assert trace_artifact(tmp_path, event["raw_output_path"]) == ["", '{"label": "docs"}']
    assert event["parse_retry_attempts"] == 1
    assert trace_artifact(tmp_path, event["parsed_output_path"]) == {"label": "docs"}


def test_structured_executor_trace_writes_output_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus

    class Output:
        outputs = ['{"label": "docs"}']

    class FakeLM:
        max_tokens = 512

        def __call__(self, prompts: object, **kwargs: Any) -> Output:
            return Output()

    monkeypatch.setattr(lotus.settings, "lm", FakeLM())
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    trace_dir = tmp_path / "trace"

    StructuredLMExecutor(pd.DataFrame({"message": ["prefers docs"]}))(
        input_cols=("message",),
        output_cols=(ColumnSpec("label", "Short label."),),
        instruction="Label {message}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={},
        semantic_trace_dir=trace_dir,
        operator="sem_map",
    )

    events = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    assert events[0]["operator"] == "sem_map"
    assert events[0]["event_type"] == "structured_generation"
    assert "parsed_output" not in events[0]
    assert "raw_output" not in events[0]
    assert "prompt_path" not in events[0]
    raw_path = trace_dir / events[0]["raw_output_path"].removeprefix("trace/")
    parsed_path = trace_dir / events[0]["parsed_output_path"].removeprefix("trace/")
    snapshot_path = trace_dir / events[0]["input_snapshot_path"].removeprefix("trace/")
    assert raw_path.exists()
    assert parsed_path.exists()
    assert snapshot_path.exists()
    assert json.loads(parsed_path.read_text(encoding="utf-8")) == {"label": "docs"}
    assert list(tmp_path.glob("*sem_map*structured*.jsonl")) == []


def test_trace_writer_routes_differential_and_view_run_kinds(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    frame = pd.DataFrame({"name": ["docs"]})

    with semantic_trace_scope(run_kind="differential", phase="add"):
        write_compact_operator_trace(
            trace_dir,
            operator="select",
            event_type="operator_result",
            output_frame=frame,
        )
    with semantic_trace_scope(run_kind="view", phase="view_topics"):
        write_compact_operator_trace(
            trace_dir,
            operator="select",
            event_type="operator_result",
            output_frame=frame,
        )

    assert (trace_dir / "differential" / "events.jsonl").exists()
    assert (trace_dir / "view" / "events.jsonl").exists()
    assert (trace_dir / "differential" / "snapshots").is_dir()
    assert (trace_dir / "view" / "snapshots").is_dir()


def test_structured_executor_model_kwargs_can_override_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    class Output:
        outputs = ['{"label": "docs"}']

    class FakeLM:
        max_tokens = 512

        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        def __call__(self, prompts: object, **kwargs: Any) -> Output:
            self.kwargs = kwargs
            return Output()

    fake_lm = FakeLM()
    monkeypatch.setattr(lotus.settings, "lm", fake_lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    StructuredLMExecutor(pd.DataFrame({"message": ["prefers docs"]}))(
        input_cols=("message",),
        output_cols=(ColumnSpec("label"),),
        instruction="Label {message}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={"max_tokens": 2048},
        operator="sem_map",
    )

    assert fake_lm.kwargs is not None
    assert fake_lm.kwargs["max_tokens"] == 2048


def test_structured_lm_retries_only_invalid_json_rows() -> None:
    reset_structured_retry_stats()

    class Output:
        def __init__(self, outputs: list[str]) -> None:
            self.outputs = outputs

    class FakeLM:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def __call__(self, prompts: object, **_kwargs: Any) -> Output:
            self.calls.append(prompts)
            if len(self.calls) == 1:
                return Output(['{"rows": [{"topic": "docs"}]}', ""])
            return Output(['{"rows": [{"topic": "meetings"}]}'])

    fake_lm = FakeLM()

    raw_outputs = execute_structured_lm_with_retries(
        fake_lm,
        ["prompt 1", "prompt 2"],
        lm_kwargs={"progress_bar_desc": "Flat mapping"},
        output_cols=(ColumnSpec("topic"),),
        shape="array",
        require_explanation=False,
        operator="sem_flat_map",
        max_retries=1,
    )

    assert raw_outputs == [
        '{"rows": [{"topic": "docs"}]}',
        '{"rows": [{"topic": "meetings"}]}',
    ]
    assert fake_lm.calls == [["prompt 1", "prompt 2"], ["prompt 2"]]
    stats = structured_retry_stats()
    assert stats.retry_batches == 1
    assert stats.retry_rows == 1
    assert stats.failure_artifacts == 0


def test_structured_lm_retries_nested_field_values() -> None:
    class Output:
        def __init__(self, outputs: list[str]) -> None:
            self.outputs = outputs

    class FakeLM:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _prompts: object, **_kwargs: Any) -> Output:
            self.calls += 1
            if self.calls == 1:
                return Output(['{"ordinal": [0]}'])
            return Output(['{"ordinal": 0}'])

    fake_lm = FakeLM()
    outputs = execute_structured_lm_with_retries(
        fake_lm,
        ["prompt"],
        lm_kwargs={"progress_bar_desc": "Mapping"},
        output_cols=(ColumnSpec("ordinal"),),
        shape="object",
        require_explanation=False,
        operator="sem_map",
        max_retries=1,
    )

    assert outputs == ['{"ordinal": 0}']
    assert fake_lm.calls == 2


def test_structured_lm_retry_failure_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_structured_retry_stats()
    monkeypatch.setattr(structured_module, "STRUCTURED_FAILURE_DIR", tmp_path)

    class Output:
        def __init__(self, outputs: list[str]) -> None:
            self.outputs = outputs

    class FakeLM:
        def __call__(self, prompts: object, **_kwargs: Any) -> Output:
            return Output(["" for _prompt in prompts])

    with pytest.raises(ValueError, match="structured failure artifact") as error:
        execute_structured_lm_with_retries(
            FakeLM(),
            ["bad prompt"],
            lm_kwargs={"progress_bar_desc": "Mapping"},
            output_cols=(ColumnSpec("summary", "One-line summary."),),
            shape="object",
            require_explanation=False,
            operator="sem_map",
            max_retries=1,
        )

    assert str(tmp_path) in str(error.value)
    artifacts = list(tmp_path.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert artifact["operator"] == "sem_map"
    assert artifact["shape"] == "object"
    assert artifact["row_index"] == 0
    assert artifact["expected_output_columns"] == [
        {"name": "summary", "description": "One-line summary."}
    ]
    assert artifact["prompt"] == "bad prompt"
    assert artifact["raw_outputs"] == ["", ""]
    assert "invalid JSON" in artifact["parse_error"]
    stats = structured_retry_stats()
    assert stats.retry_batches == 1
    assert stats.retry_rows == 1
    assert stats.failure_artifacts == 1


def test_sem_flat_map_parses_and_explodes_json_rows_wrapper_outputs() -> None:
    output_cols = (
        ColumnSpec("topic", "Candidate topic."),
        ColumnSpec("summary", "Candidate summary."),
    )
    parsed = [
        parse_structured_flat_map_json(
            '{"rows": [{"topic": "docs", "summary": "Prefers concise docs."}, {"topic": "meetings", "summary": "Plans a meeting."}]}',
            output_cols,
        )
    ]
    source = pd.DataFrame({"message": ["I prefer concise docs and plan a meeting."]})

    result = apply_flat_map_outputs(source, parsed, output_cols)

    assert list(result.columns) == ["message", "topic", "summary"]
    assert list(result["message"]) == [
        "I prefer concise docs and plan a meeting.",
        "I prefer concise docs and plan a meeting.",
    ]
    assert list(result["topic"]) == ["docs", "meetings"]


def test_sem_flat_map_preserves_json_scalar_field_types() -> None:
    output_cols = (
        ColumnSpec("entity_ordinal"),
        ColumnSpec("invalid_at"),
    )

    parsed = parse_structured_flat_map_json(
        '{"rows": [{"entity_ordinal": 1, "invalid_at": null}]}',
        output_cols,
    )

    assert parsed == [{"entity_ordinal": 1, "invalid_at": None}]


def test_sem_flat_map_empty_array_emits_zero_rows_with_columns() -> None:
    output_cols = (ColumnSpec("topic"),)
    parsed = [parse_structured_flat_map_json('{"rows": []}', output_cols)]
    source = pd.DataFrame({"message": ["nothing durable"]})

    result = apply_flat_map_outputs(source, parsed, output_cols)

    assert list(result.columns) == ["message", "topic"]
    assert result.empty


def test_sem_flat_map_rejects_invalid_json_shapes() -> None:
    output_cols = (ColumnSpec("topic"),)

    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_flat_map_json("not json", output_cols)
    with pytest.raises(ValueError, match="non-object JSON wrapper"):
        parse_structured_flat_map_json("[]", output_cols)
    with pytest.raises(ValueError, match="missing required key"):
        parse_structured_flat_map_json('{"topic": "docs"}', output_cols)
    with pytest.raises(ValueError, match="not an array"):
        parse_structured_flat_map_json('{"rows": {"topic": "docs"}}', output_cols)
    with pytest.raises(ValueError, match="not an object"):
        parse_structured_flat_map_json('{"rows": ["docs"]}', output_cols)
    with pytest.raises(ValueError, match="missing required keys"):
        parse_structured_flat_map_json('{"rows": [{"summary": "docs"}]}', output_cols)


def test_lotus_adapter_dispatches_sem_flat_map(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_memory.adapters.lotus.adapter as lotus_adapter_module

    captured: dict[str, Any] = {}

    def execute_sem_flat_map(
        query: QueryExpr,
        inputs: dict[str, Any],
        execute: object,
        context: object,
    ) -> str:
        captured["query"] = query
        captured["inputs"] = inputs
        return "flat"

    monkeypatch.setattr(
        lotus_adapter_module,
        "execute_sem_flat_map",
        execute_sem_flat_map,
    )
    query = QueryExpr(op="sem_flat_map")
    inputs: dict[str, Any] = {}

    assert LotusAdapter().execute(query, inputs) == "flat"
    assert captured == {"query": query, "inputs": inputs}


def test_relational_execution_ops_follow_dataframe_semantics() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame({"message": ["hello", "hello"]}),
        "right": pd.DataFrame({"message": ["hello", "world"]}),
    }
    execute = LotusAdapter().execute

    concatenated = execute_concat(
        QueryExpr(op="concat", inputs=(left, right)),
        inputs,
        execute,
    )
    unioned = execute_union(
        QueryExpr(op="union", inputs=(left, right)),
        inputs,
        execute,
    )
    subtracted = execute_subtract(
        QueryExpr(op="subtract", inputs=(left, right)),
        inputs,
        execute,
    )
    deduped = execute_drop_duplicates(
        QueryExpr(op="drop_duplicates", inputs=(left,)),
        inputs,
        execute,
    )

    assert list(concatenated["message"]) == ["hello", "hello", "hello", "world"]
    assert list(unioned["message"]) == ["hello", "world"]
    assert subtracted.empty
    assert list(deduped["message"]) == ["hello"]


def test_union_by_name_aligns_missing_columns_and_deduplicates() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame(
            {
                "name": ["caroline", "caroline"],
                "body": ["Adoption goal.", "Adoption goal."],
            }
        ),
        "right": pd.DataFrame(
            {
                "body": ["Adoption goal.", "Values LGBTQ+ inclusion."],
                "type": [pd.NA, "user"],
                "name": ["caroline", "caroline_values"],
            }
        ),
    }

    result = execute_union_by_name(
        QueryExpr(
            op="union_by_name",
            inputs=(left, right),
            params={"allow_missing_columns": True},
        ),
        inputs,
        LotusAdapter().execute,
    )

    records = result.astype(object).where(pd.notna(result), None).to_dict(
        orient="records"
    )
    assert list(result.columns) == ["name", "body", "type"]
    assert records == [
        {"name": "caroline", "body": "Adoption goal.", "type": None},
        {"name": "caroline_values", "body": "Values LGBTQ+ inclusion.", "type": "user"},
    ]


def test_row_append_preserves_all_null_columns_without_future_warning() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame(
            {"fact": ["old"], "invalid_at": [None], "score": [None]}
        ),
        "right": pd.DataFrame(
            {
                "fact": ["new"],
                "invalid_at": [pd.Timestamp("2026-01-01")],
                "score": [1.5],
            }
        ),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        concatenated = execute_concat(
            QueryExpr(op="concat", inputs=(left, right)),
            inputs,
            LotusAdapter().execute,
        )
        unioned = execute_union_by_name(
            QueryExpr(op="union_by_name", inputs=(left, right)),
            inputs,
            LotusAdapter().execute,
        )

    for result in (concatenated, unioned):
        assert list(result.columns) == ["fact", "invalid_at", "score"]
        assert result["fact"].tolist() == ["old", "new"]
        assert pd.isna(result.loc[0, "invalid_at"])
        assert result.loc[1, "invalid_at"] == pd.Timestamp("2026-01-01")
        assert pd.isna(result.loc[0, "score"])
        assert result.loc[1, "score"] == 1.5


def test_row_append_ignores_empty_inputs_for_dtype_inference() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame(
            {
                "fact": pd.Series(dtype=object),
                "invalid_at": pd.Series(dtype=object),
            }
        ),
        "right": pd.DataFrame(
            {
                "fact": ["new"],
                "invalid_at": [pd.Timestamp("2026-01-01")],
            }
        ),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        concatenated = execute_concat(
            QueryExpr(op="concat", inputs=(left, right)),
            inputs,
            LotusAdapter().execute,
        )
        unioned = execute_union_by_name(
            QueryExpr(op="union_by_name", inputs=(left, right)),
            inputs,
            LotusAdapter().execute,
        )

    for result in (concatenated, unioned):
        assert result.to_dict(orient="records") == [
            {"fact": "new", "invalid_at": pd.Timestamp("2026-01-01")}
        ]


def test_union_by_name_rejects_missing_columns_when_strict() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame({"name": ["caroline"], "body": ["Adoption goal."]}),
        "right": pd.DataFrame({"name": ["caroline"], "type": ["user"]}),
    }

    with pytest.raises(ValueError, match="same column names"):
        execute_union_by_name(
            QueryExpr(
                op="union_by_name",
                inputs=(left, right),
                params={"allow_missing_columns": False},
            ),
            inputs,
            LotusAdapter().execute,
        )


def test_union_by_name_rejects_non_bool_allow_missing_columns() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame({"name": ["caroline"]}),
        "right": pd.DataFrame({"name": ["caroline"]}),
    }

    with pytest.raises(TypeError, match="allow_missing_columns"):
        execute_union_by_name(
            QueryExpr(
                op="union_by_name",
                inputs=(left, right),
                params={"allow_missing_columns": "yes"},
            ),
            inputs,
            LotusAdapter().execute,
        )


def test_assign_executes_literal_scalar_columns() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    inputs = {"source": pd.DataFrame({"name": ["caroline", "melanie"]})}

    result = execute_assign(
        QueryExpr(
            op="assign",
            inputs=(source,),
            params={"assignments": {"_changed": True, "rank": 1}},
        ),
        inputs,
        LotusAdapter().execute,
    )

    assert result.to_dict(orient="records") == [
        {"name": "caroline", "_changed": True, "rank": 1},
        {"name": "melanie", "_changed": True, "rank": 1},
    ]
    assert output_columns(
        QueryExpr(
            op="assign",
            inputs=(source,),
            params={"assignments": {"_changed": True}},
        )
    ) == ("_changed",)


def test_assign_executes_row_wise_array_cat_expression() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("records:right", "records:left")},
    )
    relation = Relation(source)
    query = relation.assign(
        records=relation.col("records:right").array_cat(relation.col("records:left"))
    ).expr
    inputs = {
        "source": pd.DataFrame(
            {
                "records:right": ['[{"body": "old"}]', pd.NA],
                "records:left": ['[{"body": "new"}]', '[{"body": "only-new"}]'],
            },
            index=[7, 7],
        )
    }

    result = execute_assign(query, inputs, LotusAdapter().execute)

    assert [json.loads(value) for value in result["records"]] == [
        [{"body": "old"}, {"body": "new"}],
        [{"body": "only-new"}],
    ]
    assert result.index.tolist() == [7, 7]


def test_assign_rejects_callable_or_complex_values() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    inputs = {"source": pd.DataFrame({"name": ["caroline"]})}

    with pytest.raises((TypeError, ValueError), match="Expression param|Unsupported expression"):
        execute_assign(
            QueryExpr(
                op="assign",
                inputs=(source,),
                params={"assignments": {"bad": lambda row: row}},
            ),
            inputs,
            LotusAdapter().execute,
        )
    with pytest.raises((TypeError, ValueError), match="Expression param|Unsupported expression"):
        execute_assign(
            QueryExpr(
                op="assign",
                inputs=(source,),
                params={"assignments": {"bad": {"nested": True}}},
            ),
            inputs,
            LotusAdapter().execute,
        )


def test_filter_executes_internal_is_true_predicate() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    inputs = {
        "source": pd.DataFrame(
            {
                "name": ["new", "old", "unknown"],
                "_changed": [True, False, pd.NA],
            }
        )
    }

    result = execute_filter(
        QueryExpr(
            op="filter",
            inputs=(source,),
            params={"predicate": ColumnExpr("_changed").to_param()},
        ),
        inputs,
        LotusAdapter().execute,
    )

    assert result.to_dict(orient="records") == [{"name": "new", "_changed": True}]
    assert output_columns(
        QueryExpr(
            op="filter",
            inputs=(source,),
            params={"predicate": ColumnExpr("_changed").to_param()},
        )
    ) == ()


def test_filter_rejects_unsupported_predicates() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    inputs = {"source": pd.DataFrame({"name": ["new"], "_changed": [True]})}

    with pytest.raises(ValueError, match="Unsupported expression kind"):
        execute_filter(
            QueryExpr(
                op="filter",
                inputs=(source,),
                params={"predicate": {"op": "equals", "column": "_changed"}},
            ),
            inputs,
            LotusAdapter().execute,
        )
    with pytest.raises(ValueError, match="not found"):
        execute_filter(
            QueryExpr(
                op="filter",
                inputs=(source,),
                params={"predicate": ColumnExpr("missing").to_param()},
            ),
            inputs,
            LotusAdapter().execute,
        )


def test_relation_bound_expression_api_query_expr_shape() -> None:
    log = am.Log(
        {
            "fact_id": "Fact id.",
            "source_entity_id": "Source entity.",
            "target_entity_id": "Target entity.",
            "valid_at": "Valid timestamp.",
            "action": "Resolution action.",
        }
    )
    old = log.alias("old")
    new = log.alias("new")

    joined = old.join(
        new,
        on=[
            old.col("fact_id") != new.col("fact_id"),
            old.col("source_entity_id") == new.col("source_entity_id"),
            old.col("target_entity_id") == new.col("target_entity_id"),
            old.col("valid_at") <= new.col("valid_at"),
        ],
    )
    filtered = joined.filter(joined.col("action:old").isin(["contradicts"]))
    assigned = filtered.assign(
        invalid_at=filtered.col("valid_at:new"),
        status="inactive",
    )

    assert old.expr.op == "alias"
    assert old.expr.params["name"] == "old"
    assert joined.expr.op == "join"
    assert len(joined.expr.params["on"]) == 4
    assert filtered.expr.params["predicate"]["kind"] == "boolean"
    assert assigned.expr.params["assignments"]["invalid_at"]["kind"] == "column"
    assert assigned.expr.params["assignments"]["status"]["kind"] == "literal"
    assert output_columns(joined.expr) == (
        "fact_id:old",
        "source_entity_id:old",
        "target_entity_id:old",
        "valid_at:old",
        "action:old",
        "fact_id:new",
        "source_entity_id:new",
        "target_entity_id:new",
        "valid_at:new",
        "action:new",
    )


def test_relation_bound_expression_truthiness_raises() -> None:
    left = ColumnExpr("source_entity_id", qualifier="old")
    right = ColumnExpr("source_entity_id", qualifier="new")
    predicate = left == right

    assert predicate.__class__.__name__ == "ComparisonExpr"
    assert predicate.to_param()["kind"] == "comparison"
    assert ColumnExpr("source_entity_id").to_param() == ColumnExpr("source_entity_id").to_param()

    with pytest.raises(TypeError, match="Relational expressions cannot be used as Python booleans"):
        bool(predicate)

    with pytest.raises(TypeError, match="Relational expressions cannot be used as Python booleans"):
        ColumnExpr("a") in [ColumnExpr("b")]

    with pytest.raises(TypeError, match="Relational expressions cannot be used as Python booleans"):
        (ColumnExpr("a"),) == (ColumnExpr("b"),)


def test_column_expr_is_unhashable() -> None:
    with pytest.raises(TypeError, match="unhashable type"):
        hash(ColumnExpr("name"))


def test_join_on_rejects_mixed_key_names_and_predicates() -> None:
    log = am.Log({"name": "Name.", "body": "Body.", "valid_at": "Valid timestamp."})
    old = log.alias("old")
    new = log.alias("new")

    key_join = log.join(log, on=["name", "body"])
    predicate_join = old.join(new, on=[old.col("valid_at") <= new.col("valid_at")])

    assert key_join.expr.params["on"] == ("name", "body")
    assert predicate_join.expr.params["on"][0]["kind"] == "comparison"

    with pytest.raises(
        ValueError,
        match="join on sequence cannot mix key column names and predicate expressions",
    ):
        old.join(new, on=["name", old.col("valid_at") <= new.col("valid_at")])


def test_filter_and_assign_execute_expression_values() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    action = ColumnExpr("action")
    valid_at = ColumnExpr("valid_at:new")
    inputs = {
        "source": pd.DataFrame(
            {
                "action": ["contradicts", "unrelated", "supersedes"],
                "valid_at:new": ["t2", "t3", "t4"],
            }
        )
    }
    filtered_query = QueryExpr(
        op="filter",
        inputs=(source,),
        params={"predicate": action.isin(["contradicts", "supersedes"]).to_param()},
    )
    assigned_query = QueryExpr(
        op="assign",
        inputs=(filtered_query,),
        params={
            "assignments": {
                "invalid_at": valid_at.to_param(),
                "status": {"kind": "literal", "value": "inactive"},
            }
        },
    )

    result = LotusAdapter().execute(assigned_query, inputs)

    assert result.to_dict(orient="records") == [
        {
            "action": "contradicts",
            "valid_at:new": "t2",
            "invalid_at": "t2",
            "status": "inactive",
        },
        {
            "action": "supersedes",
            "valid_at:new": "t4",
            "invalid_at": "t4",
            "status": "inactive",
        },
    ]


def test_array_agg_executes_to_stable_json_record_array() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="array_agg",
        inputs=(source,),
        params={
            "columns": ("timestamp", "speaker", "message"),
            "output_col": "conversation_records",
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
                "speaker": ["Caroline", "Melanie"],
                "message": ["Hi.", "Hello."],
                "ignored": ["x", "y"],
            }
        )
    }

    result = LotusAdapter().execute(query, inputs)

    assert list(result.columns) == ["conversation_records"]
    expected_json = (
        '[{"timestamp": "2026-01-01T00:00:00Z", "speaker": "Caroline", '
        '"message": "Hi."}, {"timestamp": "2026-01-01T00:01:00Z", '
        '"speaker": "Melanie", "message": "Hello."}]'
    )
    assert result.loc[0, "conversation_records"] == expected_json
    assert json.loads(result.loc[0, "conversation_records"]) == [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "speaker": "Caroline",
            "message": "Hi.",
        },
        {
            "timestamp": "2026-01-01T00:01:00Z",
            "speaker": "Melanie",
            "message": "Hello.",
        },
    ]


def test_array_agg_preserves_json_numeric_and_boolean_types() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="array_agg",
        inputs=(source,),
        params={
            "columns": ("count", "flag"),
            "output_col": "records",
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "count": [np.int64(3)],
                "flag": [np.bool_(True)],
            }
        )
    }

    result = LotusAdapter().execute(query, inputs)
    records = json.loads(result.loc[0, "records"])

    assert records == [{"count": 3, "flag": True}]
    assert isinstance(records[0]["count"], int)
    assert isinstance(records[0]["flag"], bool)


def test_array_agg_paths_serialize_nullable_numeric_values_as_json_null() -> None:
    log = am.Log({"group": "Group key.", "value": "Nullable numeric value."})
    source = pd.DataFrame(
        {
            "group": ["a", "a", "b"],
            "value": [1.5, np.nan, 3.0],
        }
    )
    inputs = {"log": source}

    global_result = LotusAdapter().execute(
        log.array_agg(columns=["value"], output_col="records").expr,
        inputs,
    )
    assert json.loads(global_result.loc[0, "records"]) == [
        {"value": 1.5},
        {"value": None},
        {"value": 3.0},
    ]

    grouped_result = LotusAdapter().execute(
        log.group_by("group").array_agg(
            columns=["value"],
            output_col="records",
        ).expr,
        inputs,
    )
    assert json.loads(grouped_result.loc[0, "records"]) == [
        {"value": 1.5},
        {"value": None},
    ]

    mixed_result = LotusAdapter().execute(
        log.group_by("group").agg(
            am.array_agg(columns=["value"], output_col="records"),
        ).expr,
        inputs,
    )
    assert json.loads(mixed_result.loc[0, "records"]) == [
        {"value": 1.5},
        {"value": None},
    ]

    over_result = LotusAdapter().execute(
        log.over(rows=(-2, -1)).array_agg(
            columns=["value"],
            output_col="previous_values",
        ).expr,
        inputs,
    )
    assert json.loads(over_result.loc[2, "previous_values"]) == [
        {"value": 1.5},
        {"value": None},
    ]


def test_group_by_array_agg_executes_per_exact_key() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="array_agg",
        inputs=(
            QueryExpr(
                op="group_by",
                inputs=(source,),
                params={"keys": ("episode_id",)},
            ),
        ),
        params={"columns": ("entity_id", "name"), "output_col": "entities"},
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "episode_id": ["e1", "e1", "e2"],
                "entity_id": ["c", "m", "c"],
                "name": ["Caroline", "Melanie", "Caroline"],
            }
        )
    }

    result = LotusAdapter().execute(query, inputs)

    assert list(result.columns) == ["episode_id", "entities"]
    assert result.loc[0, "episode_id"] == "e1"
    assert json.loads(result.loc[0, "entities"]) == [
        {"entity_id": "c", "name": "Caroline"},
        {"entity_id": "m", "name": "Melanie"},
    ]
    assert result.loc[1, "episode_id"] == "e2"
    assert json.loads(result.loc[1, "entities"]) == [
        {"entity_id": "c", "name": "Caroline"}
    ]
    assert output_columns(query) == ("episode_id", "entities")

    duplicate_output_col_query = QueryExpr(
        op="array_agg",
        inputs=query.inputs,
        params={"columns": ("entity_id",), "output_col": "episode_id"},
    )
    with pytest.raises(ValueError, match="array_agg output column conflicts with group key"):
        output_columns(duplicate_output_col_query)

    with pytest.raises(ValueError, match="array_agg output column conflicts with group key"):
        LotusAdapter().execute(duplicate_output_col_query, inputs)


def test_group_by_collect_list_executes_per_exact_key() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="agg",
        inputs=(
            QueryExpr(
                op="group_by",
                inputs=(source,),
                params={"keys": ("episode_id",)},
            ),
        ),
        params={"aggregates": (am.collect_list(column="entity_id", output_col="entity_ids"),)},
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "episode_id": ["e1", "e1", "e2"],
                "entity_id": ["c", "m", None],
            }
        )
    }

    result = execute_agg(query, inputs, LotusAdapter().execute, SimpleNamespace(config=LotusExecutionConfig()))

    assert list(result.columns) == ["episode_id", "entity_ids"]
    assert json.loads(result.loc[0, "entity_ids"]) == ["c", "m"]
    assert json.loads(result.loc[1, "entity_ids"]) == [None]
    assert output_columns(query) == ("episode_id", "entity_ids")


def test_min_executes_global_grouped_and_mixed_null_semantics() -> None:
    log = am.Log({"group": "Group.", "value": "Value."})
    global_query = log.min(column="value", output_col="minimum").expr

    result = execute_min(
        global_query,
        {"log": pd.DataFrame({"group": ["a", "a", "b"], "value": [None, 3, 1]})},
        LotusAdapter().execute,
    )
    assert result.to_dict("records") == [{"minimum": 1.0}]

    empty = execute_min(
        global_query,
        {"log": pd.DataFrame(columns=["group", "value"])},
        LotusAdapter().execute,
    )
    assert empty.to_dict("records") == [{"minimum": None}]

    all_null = execute_min(
        global_query,
        {"log": pd.DataFrame({"group": ["a"], "value": [None]})},
        LotusAdapter().execute,
    )
    assert all_null.to_dict("records") == [{"minimum": None}]

    grouped_query = log.group_by("group").min(
        column="value",
        output_col="minimum",
    ).expr
    grouped = execute_min(
        grouped_query,
        {
            "log": pd.DataFrame(
                {"group": ["a", "a", "b"], "value": [None, 2, None]}
            )
        },
        LotusAdapter().execute,
    )
    assert grouped.to_dict("records") == [
        {"group": "a", "minimum": 2.0},
        {"group": "b", "minimum": None},
    ]

    grouped_empty = execute_min(
        grouped_query,
        {"log": pd.DataFrame(columns=["group", "value"])},
        LotusAdapter().execute,
    )
    assert grouped_empty.empty
    assert list(grouped_empty.columns) == ["group", "minimum"]

    mixed_query = log.group_by("group").agg(
        am.min(column="value", output_col="minimum"),
        am.collect_list(column="value", output_col="values"),
    ).expr
    mixed = LotusAdapter().execute(
        mixed_query,
        {"log": pd.DataFrame({"group": ["a", "a"], "value": [None, 2]})},
    )
    assert mixed.loc[0, "minimum"] == 2
    assert json.loads(mixed.loc[0, "values"]) == [None, 2.0]

    with pytest.raises(ValueError, match="min input column not found"):
        execute_min(
            log.min(column="missing", output_col="minimum").expr,
            {"log": pd.DataFrame({"group": ["a"], "value": [1]})},
            LotusAdapter().execute,
        )


def test_composite_min_executes_lexicographically_and_skips_partial_nulls() -> None:
    log = am.Log(
        {
            "group": "Group.",
            "add_seq": "Append sequence.",
            "ordinal": "Occurrence ordinal.",
        }
    )
    source = pd.DataFrame(
        {
            "group": ["a", "a", "a", "b"],
            "add_seq": [12, 8, 8, None],
            "ordinal": [0, 3, 1, 0],
        }
    )

    global_result = execute_min(
        log.min(
            columns=["add_seq", "ordinal"],
            output_col="occurrence_id",
        ).expr,
        {"log": source},
        LotusAdapter().execute,
    )
    assert global_result.to_dict("records") == [{"occurrence_id": (8, 1)}]

    grouped_query = log.group_by("group").min(
        columns=["add_seq", "ordinal"],
        output_col="occurrence_id",
    ).expr
    grouped_result = execute_min(
        grouped_query,
        {"log": source},
        LotusAdapter().execute,
    )
    assert grouped_result.to_dict("records") == [
        {"group": "a", "occurrence_id": (8, 1)},
        {"group": "b", "occurrence_id": None},
    ]

    mixed_query = log.group_by("group").agg(
        am.min(
            columns=["add_seq", "ordinal"],
            output_col="occurrence_id",
        )
    ).expr
    mixed_result = LotusAdapter().execute(mixed_query, {"log": source})
    assert mixed_result.to_dict("records") == [
        {"group": "a", "occurrence_id": (8, 1)},
        {"group": "b", "occurrence_id": None},
    ]


def test_least_assign_executes_row_wise_and_ignores_nulls() -> None:
    log = am.Log({"left": "Left value.", "right": "Right value."})
    query = log.assign(
        earliest=am.least(log.col("left"), log.col("right")),
    ).expr

    result = LotusAdapter().execute(
        query,
        {
            "log": pd.DataFrame(
                {
                    "left": [3, None, None],
                    "right": [2, 5, None],
                },
                index=[7, 7, 8],
            )
        },
    )

    assert result["earliest"].tolist()[:2] == [2.0, 5.0]
    assert pd.isna(result.iloc[2]["earliest"])

    incompatible = am.Log({"left": "Left.", "right": "Right."})
    incompatible_query = incompatible.assign(
        minimum=am.least(incompatible.col("left"), incompatible.col("right")),
    ).expr
    with pytest.raises(TypeError, match="least operands are not mutually comparable"):
        LotusAdapter().execute(
            incompatible_query,
            {"log": pd.DataFrame({"left": [1], "right": ["x"]})},
        )


def test_least_assign_compares_composite_min_state() -> None:
    rows = am.Log({"left": "Left tuple.", "right": "Right tuple."})
    query = rows.assign(
        occurrence_id=am.least(rows.col("left"), rows.col("right")),
    ).expr

    result = LotusAdapter().execute(
        query,
        {
            "log": pd.DataFrame(
                {
                    "left": [(2, 0), None],
                    "right": [(1, 3), (5, 0)],
                }
            )
        },
    )

    assert result["occurrence_id"].tolist() == [(1, 3), (5, 0)]


def test_sem_groupby_array_agg_is_rejected_without_semantic_key_aggregate() -> None:
    source = QueryExpr(
        op="sem_groupby",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
    )
    query = QueryExpr(
        op="array_agg",
        inputs=(source,),
        params={"columns": ("fact", "episode_id"), "output_col": "evidence"},
    )

    def execute(query_expr: QueryExpr, inputs: dict[str, Any]) -> pd.DataFrame:
        assert query_expr is source
        return pd.DataFrame(
            {
                GROUP_ID_COLUMN: [0, 0, 1],
                "fact": ["A", "B", "C"],
                "episode_id": ["e1", "e2", "e3"],
            }
        )

    with pytest.raises(NotImplementedError, match="sem_groupby.*array_agg"):
        execute_array_agg(query, {}, execute)
    assert output_columns(query) == ("evidence",)


def test_sem_groupby_agg_executes_semantic_and_array_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = QueryExpr(
        op="sem_groupby",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={"input_cols": ("name",), "instruction": "Same entity."},
    )
    query = QueryExpr(
        op="agg",
        inputs=(source,),
        params={
            "aggregates": (
                am.sem_agg(
                    input_cols=["name"],
                    output_cols={"name": "Canonical name."},
                    instruction="Choose canonical name.",
                ),
                am.array_agg(
                    columns=["fact", "episode_id"],
                    output_col="evidence",
                ),
                am.min(column="valid_at", output_col="valid_at"),
            )
        },
    )

    def execute(query_expr: QueryExpr, inputs: dict[str, Any]) -> pd.DataFrame:
        assert query_expr is source
        return pd.DataFrame(
            {
                GROUP_ID_COLUMN: [0, 0, 1],
                "name": ["caroline", "Caroline", "melanie"],
                "fact": ["A", "B", "C"],
                "episode_id": ["e1", "e2", "e3"],
                "valid_at": ["2026-01-02", "2026-01-01", None],
            }
        )

    def execute_native_sem_agg_group(
        query_expr: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        config: LotusExecutionConfig,
    ) -> str:
        return str(group.loc[0, input_cols[0]]).title()

    monkeypatch.setattr(
        relational_module,
        "execute_native_sem_agg_group",
        execute_native_sem_agg_group,
    )

    result = execute_agg(query, {}, execute, SimpleNamespace(config=LotusExecutionConfig()))

    assert list(result.columns) == ["name", "evidence", "valid_at"]
    assert result["name"].tolist() == ["Caroline", "Melanie"]
    assert [json.loads(value) for value in result["evidence"]] == [
        [{"fact": "A", "episode_id": "e1"}, {"fact": "B", "episode_id": "e2"}],
        [{"fact": "C", "episode_id": "e3"}],
    ]
    assert result["valid_at"].tolist() == ["2026-01-01", None]
    assert output_columns(query) == ("name", "evidence", "valid_at")


def test_over_array_agg_executes_row_preserving_previous_frame() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="array_agg",
        inputs=(
            QueryExpr(
                op="over",
                inputs=(source,),
                params={"rows": (-2, -1)},
            ),
        ),
        params={
            "columns": ("message",),
            "output_col": "previous_messages",
        },
    )
    inputs = {"source": pd.DataFrame({"message": ["one", "two", "three"]})}

    result = LotusAdapter().execute(query, inputs)

    assert list(result.columns) == ["message", "previous_messages"]
    assert json.loads(result.loc[0, "previous_messages"]) == []
    assert json.loads(result.loc[1, "previous_messages"]) == [{"message": "one"}]
    assert json.loads(result.loc[2, "previous_messages"]) == [
        {"message": "one"},
        {"message": "two"},
    ]


def test_over_sem_agg_empty_frame_outputs_null_without_model_call() -> None:
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="sem_agg",
        inputs=(
            QueryExpr(
                op="over",
                inputs=(source,),
                params={"rows": (-2, -1)},
            ),
        ),
        params={
            "input_cols": ("message",),
            "output_cols": (ColumnSpec("previous_summary"),),
            "instruction": "Summarize previous messages.",
        },
    )
    inputs = {"source": pd.DataFrame({"message": ["one"]})}

    result = LotusAdapter().execute(query, inputs)

    assert list(result.columns) == ["message", "previous_summary"]
    assert pd.isna(result.loc[0, "previous_summary"])


def test_array_cat_executes_json_array_concat() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    query = QueryExpr(
        op="array_cat",
        inputs=(left, right),
        params={"column": "conversation_records"},
    )
    inputs = {
        "left": pd.DataFrame(
            {
                "conversation_records": [
                    '[{"message": "one"}, {"message": "two"}]',
                ]
            }
        ),
        "right": pd.DataFrame(
            {
                "conversation_records": [
                    '[{"message": "three"}]',
                ]
            }
        ),
    }

    result = execute_array_cat(query, inputs, LotusAdapter().execute)

    assert list(result.columns) == ["conversation_records"]
    expected_json = (
        '[{"message": "one"}, {"message": "two"}, {"message": "three"}]'
    )
    assert result.loc[0, "conversation_records"] == expected_json


def test_array_cat_empty_state_returns_other_side() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    query = QueryExpr(
        op="array_cat",
        inputs=(left, right),
        params={"column": "conversation_records"},
    )
    execute = LotusAdapter().execute

    left_empty = {
        "left": pd.DataFrame(columns=["conversation_records"]),
        "right": pd.DataFrame({"conversation_records": ['[{"message": "new"}]']}),
    }
    result = execute_array_cat(query, left_empty, execute)
    assert result.loc[0, "conversation_records"] == '[{"message": "new"}]'

    right_empty = {
        "left": pd.DataFrame({"conversation_records": ['[{"message": "old"}]']}),
        "right": pd.DataFrame(columns=["conversation_records"]),
    }
    result = execute_array_cat(query, right_empty, execute)
    assert result.loc[0, "conversation_records"] == '[{"message": "old"}]'

    both_empty = {
        "left": pd.DataFrame(columns=["conversation_records"]),
        "right": pd.DataFrame(columns=["conversation_records"]),
    }
    result = execute_array_cat(query, both_empty, execute)
    assert list(result.columns) == ["conversation_records"]
    assert result.empty


def test_array_cat_rejects_invalid_aggregate_state() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    query = QueryExpr(
        op="array_cat",
        inputs=(left, right),
        params={"column": "conversation_records"},
    )
    execute = LotusAdapter().execute

    with pytest.raises(ValueError, match="missing column"):
        execute_array_cat(
            query,
            {
                "left": pd.DataFrame({"wrong": ['[]']}),
                "right": pd.DataFrame({"conversation_records": ['[]']}),
            },
            execute,
        )
    with pytest.raises(ValueError, match="at most one row"):
        execute_array_cat(
            query,
            {
                "left": pd.DataFrame({"conversation_records": ["[]", "[]"]}),
                "right": pd.DataFrame({"conversation_records": ["[]"]}),
            },
            execute,
        )
    with pytest.raises(ValueError, match="JSON array"):
        execute_array_cat(
            query,
            {
                "left": pd.DataFrame({"conversation_records": ["not json"]}),
                "right": pd.DataFrame({"conversation_records": ["[]"]}),
            },
            execute,
        )
    with pytest.raises(ValueError, match="JSON array"):
        execute_array_cat(
            query,
            {
                "left": pd.DataFrame({"conversation_records": ['{"message": "one"}']}),
                "right": pd.DataFrame({"conversation_records": ["[]"]}),
            },
            execute,
        )


def test_flatten_executes_json_array_state_flattening() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={
            "name": "source",
            "columns": ("topic", "nested_evidence"),
        },
    )
    query = QueryExpr(
        op="flatten",
        inputs=(source,),
        params={"column": "nested_evidence", "output_col": "evidence"},
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "topic": ["docs", "meetings"],
                "nested_evidence": [
                    json.dumps(['[{"body": "old"}]', '[{"body": "new"}]']),
                    json.dumps([None, '[{"body": "only-new"}]']),
                ],
            }
        )
    }

    result = execute_flatten(query, inputs, LotusAdapter().execute)

    assert output_columns(query) == (
        "topic",
        "nested_evidence",
        "evidence",
    )
    assert [json.loads(value) for value in result["evidence"]] == [
        [{"body": "old"}, {"body": "new"}],
        [{"body": "only-new"}],
    ]


def test_flatten_rejects_overwriting_an_unrelated_existing_column() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("nested_evidence", "evidence")},
    )
    query = QueryExpr(
        op="flatten",
        inputs=(source,),
        params={"column": "nested_evidence", "output_col": "evidence"},
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "nested_evidence": [json.dumps(['[{"body": "new"}]'])],
                "evidence": ["keep-me"],
            }
        )
    }

    with pytest.raises(ValueError, match="flatten output column already exists"):
        output_columns(query)
    with pytest.raises(ValueError, match="flatten output column already exists"):
        execute_flatten(query, inputs, LotusAdapter().execute)


def test_flatten_allows_explicit_in_place_output_column() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("evidence",)},
    )
    query = QueryExpr(
        op="flatten",
        inputs=(source,),
        params={"column": "evidence", "output_col": "evidence"},
    )
    inputs = {
        "source": pd.DataFrame(
            {"evidence": [json.dumps(['[{"body": "old"}]', '[{"body": "new"}]'])]}
        )
    }

    assert output_columns(query) == ("evidence",)
    result = execute_flatten(query, inputs, LotusAdapter().execute)
    assert json.loads(result.loc[0, "evidence"]) == [
        {"body": "old"},
        {"body": "new"},
    ]


def test_explode_executes_json_array_expansion() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("entity_id", "mentions")},
    )
    query = QueryExpr(
        op="explode",
        inputs=(source,),
        params={"column": "mentions", "output_col": "_mention"},
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "entity_id": ["e1", "e2", "e3"],
                "mentions": [
                    json.dumps(
                        [
                            {"episode_id": "m1", "name": "Caroline"},
                            {"episode_id": "m2", "name": "Carol"},
                        ]
                    ),
                    "[]",
                    None,
                ],
            }
        )
    }

    result = execute_explode(query, inputs, LotusAdapter().execute)

    assert output_columns(query) == ("entity_id", "mentions", "_mention")
    assert result["entity_id"].tolist() == ["e1", "e1"]
    assert result["_mention"].tolist() == [
        {"episode_id": "m1", "name": "Caroline"},
        {"episode_id": "m2", "name": "Carol"},
    ]


def test_explode_can_replace_source_column() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("entity_id", "mentions")},
    )
    query = QueryExpr(
        op="explode",
        inputs=(source,),
        params={"column": "mentions", "output_col": None},
    )

    result = execute_explode(
        query,
        {
            "source": pd.DataFrame(
                {"entity_id": ["e1"], "mentions": [json.dumps(["m1", "m2"])]}
            )
        },
        LotusAdapter().execute,
    )

    assert list(result.columns) == ["entity_id", "mentions"]
    assert result["mentions"].tolist() == ["m1", "m2"]


def test_explode_rejects_invalid_array_input() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("entity_id", "mentions")},
    )
    query = QueryExpr(
        op="explode",
        inputs=(source,),
        params={"column": "mentions", "output_col": "_mention"},
    )

    with pytest.raises(ValueError, match="input column not found"):
        execute_explode(
            query,
            {"source": pd.DataFrame({"entity_id": ["e1"]})},
            LotusAdapter().execute,
        )
    with pytest.raises(ValueError, match="already exists"):
        execute_explode(
            QueryExpr(
                op="explode",
                inputs=(source,),
                params={"column": "mentions", "output_col": "entity_id"},
            ),
            {"source": pd.DataFrame({"entity_id": ["e1"], "mentions": ["[]"]})},
            LotusAdapter().execute,
        )
    with pytest.raises(ValueError, match="JSON array"):
        execute_explode(
            query,
            {"source": pd.DataFrame({"entity_id": ["e1"], "mentions": ["not json"]})},
            LotusAdapter().execute,
        )


def test_unnest_executes_json_object_expansion() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("entity_id", "_mention")},
    )
    query = QueryExpr(
        op="unnest",
        inputs=(source,),
        params={
            "column": "_mention",
            "fields": (("episode_id", "episode_id"), ("name", "mention_name")),
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "entity_id": ["e1", "e1"],
                "_mention": [
                    {"episode_id": "m1", "name": "Caroline"},
                    json.dumps({"episode_id": "m2", "name": "Carol"}),
                ],
            }
        )
    }

    result = execute_unnest(query, inputs, LotusAdapter().execute)

    assert output_columns(query) == ("entity_id", "episode_id", "mention_name")
    assert result.to_dict(orient="records") == [
        {"entity_id": "e1", "episode_id": "m1", "mention_name": "Caroline"},
        {"entity_id": "e1", "episode_id": "m2", "mention_name": "Carol"},
    ]


def test_unnest_rejects_invalid_object_input() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("entity_id", "_mention")},
    )
    query = QueryExpr(
        op="unnest",
        inputs=(source,),
        params={"column": "_mention", "fields": (("episode_id", "episode_id"),)},
    )

    with pytest.raises(ValueError, match="input column not found"):
        execute_unnest(
            query,
            {"source": pd.DataFrame({"entity_id": ["e1"]})},
            LotusAdapter().execute,
        )
    with pytest.raises(ValueError, match="JSON object"):
        execute_unnest(
            query,
            {"source": pd.DataFrame({"entity_id": ["e1"], "_mention": ["[]"]})},
            LotusAdapter().execute,
        )
    with pytest.raises(ValueError, match="field 'episode_id' not found"):
        execute_unnest(
            query,
            {"source": pd.DataFrame({"entity_id": ["e1"], "_mention": [{}]})},
            LotusAdapter().execute,
        )


def test_unnest_rejects_duplicate_outputs_in_hand_written_ir() -> None:
    source = QueryExpr(
        op="materialized_view",
        params={"name": "source", "columns": ("_mention",)},
    )
    query = QueryExpr(
        op="unnest",
        inputs=(source,),
        params={
            "column": "_mention",
            "fields": (("episode_id", "identifier"), ("name", "identifier")),
        },
    )

    with pytest.raises(ValueError, match="output columns must be unique"):
        output_columns(query)
    with pytest.raises(ValueError, match="output columns must be unique"):
        execute_unnest(
            query,
            {"source": pd.DataFrame({"_mention": ["not json"]})},
            LotusAdapter().execute,
        )


def test_process_window_full_execution_emits_completed_count_windows() -> None:
    log = am.Log(
        {
            "timestamp": "Message timestamp.",
            "speaker": "Message speaker.",
            "message": "Message body.",
        }
    )
    query = log.count_window(size=2, slide=1).process_window(
        lambda window: window.array_agg(
            columns=("timestamp", "speaker", "message"),
            output_col="conversation_records",
        )
    ).expr
    inputs = {
        "log": pd.DataFrame(
            {
                "timestamp": ["t1", "t2", "t3"],
                "speaker": ["A", "B", "A"],
                "message": ["one", "two", "three"],
            }
        )
    }

    result = LotusAdapter().execute(query, inputs)

    assert list(result.columns) == ["conversation_records"]
    assert len(result) == 2
    assert json.loads(result.loc[0, "conversation_records"]) == [
        {"timestamp": "t1", "speaker": "A", "message": "one"},
        {"timestamp": "t2", "speaker": "B", "message": "two"},
    ]
    assert json.loads(result.loc[1, "conversation_records"]) == [
        {"timestamp": "t2", "speaker": "B", "message": "two"},
        {"timestamp": "t3", "speaker": "A", "message": "three"},
    ]


def test_relational_join_executes_exact_key_merge_semantics() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame(
            {
                "name": ["a", "b", "c"],
                "description": ["left a", "left b", "left c"],
                "rank": [1, 2, 3],
            }
        ),
        "right": pd.DataFrame(
            {
                "name": ["a", "b", "d"],
                "description": ["right a", "right b", "right d"],
                "body": ["A", "B", "D"],
            }
        ),
    }
    execute = LotusAdapter().execute

    inner = execute_join(
        QueryExpr(
            op="join",
            inputs=(left, right),
            params={"on": ("name",), "how": "inner"},
        ),
        inputs,
        execute,
    )
    left_join = execute_join(
        QueryExpr(
            op="join",
            inputs=(left, right),
            params={"on": ("name",), "how": "left"},
        ),
        inputs,
        execute,
    )
    right_join = execute_join(
        QueryExpr(
            op="join",
            inputs=(left, right),
            params={"on": ("name",), "how": "right"},
        ),
        inputs,
        execute,
    )
    outer = execute_join(
        QueryExpr(
            op="join",
            inputs=(left, right),
            params={"on": ("name",), "how": "outer"},
        ),
        inputs,
        execute,
    )
    left_anti = execute_join(
        QueryExpr(
            op="join",
            inputs=(left, right),
            params={"on": ("name",), "how": "left_anti"},
        ),
        inputs,
        execute,
    )

    assert list(inner.columns) == [
        "name",
        "description:left",
        "rank",
        "description:right",
        "body",
    ]
    assert list(inner["name"]) == ["a", "b"]
    assert list(inner["description:left"]) == ["left a", "left b"]
    assert list(inner["description:right"]) == ["right a", "right b"]
    assert list(left_join["name"]) == ["a", "b", "c"]
    assert list(right_join["name"]) == ["a", "b", "d"]
    assert list(outer["name"]) == ["a", "b", "c", "d"]
    assert list(left_anti.columns) == ["name", "description", "rank"]
    assert left_anti.to_dict(orient="records") == [
        {"name": "c", "description": "left c", "rank": 3}
    ]
    assert output_columns(
        QueryExpr(
            op="join",
            inputs=(
                QueryExpr(
                    op="materialized_view",
                    params={"name": "left", "columns": ("name", "description", "rank")},
                ),
                QueryExpr(
                    op="materialized_view",
                    params={"name": "right", "columns": ("name", "body")},
                ),
            ),
            params={"on": ("name",), "how": "left_anti"},
        )
    ) == ("name", "description", "rank")


def test_relational_predicate_join_executes_alias_self_join() -> None:
    left = QueryExpr(
        op="alias",
        inputs=(QueryExpr(op="materialized_view", params={"name": "left"}),),
        params={"name": "old"},
    )
    right = QueryExpr(
        op="alias",
        inputs=(QueryExpr(op="materialized_view", params={"name": "right"}),),
        params={"name": "new"},
    )
    predicates = (
        (ColumnExpr("fact_id", qualifier="old") != ColumnExpr("fact_id", qualifier="new")).to_param(),
        (
            ColumnExpr("source_entity_id", qualifier="old")
            == ColumnExpr("source_entity_id", qualifier="new")
        ).to_param(),
        (
            ColumnExpr("target_entity_id", qualifier="old")
            == ColumnExpr("target_entity_id", qualifier="new")
        ).to_param(),
        (ColumnExpr("valid_at", qualifier="old") <= ColumnExpr("valid_at", qualifier="new")).to_param(),
    )
    query = QueryExpr(
        op="join",
        inputs=(left, right),
        params={"on": predicates, "how": "inner"},
    )
    inputs = {
        "left": pd.DataFrame(
            {
                "fact_id": ["old-1", "old-2"],
                "source_entity_id": ["caroline", "caroline"],
                "target_entity_id": ["home", "home"],
                "valid_at": [1, 5],
            }
        ),
        "right": pd.DataFrame(
            {
                "fact_id": ["new-1", "new-2"],
                "source_entity_id": ["caroline", "caroline"],
                "target_entity_id": ["home", "work"],
                "valid_at": [2, 6],
            }
        ),
    }

    result = execute_join(query, inputs, LotusAdapter().execute)

    assert list(result.columns) == [
        "fact_id:old",
        "source_entity_id:old",
        "target_entity_id:old",
        "valid_at:old",
        "fact_id:new",
        "source_entity_id:new",
        "target_entity_id:new",
        "valid_at:new",
    ]
    assert result.to_dict(orient="records") == [
        {
            "fact_id:old": "old-1",
            "source_entity_id:old": "caroline",
            "target_entity_id:old": "home",
            "valid_at:old": 1,
            "fact_id:new": "new-1",
            "source_entity_id:new": "caroline",
            "target_entity_id:new": "home",
            "valid_at:new": 2,
        }
    ]
    assert output_columns(
        QueryExpr(
            op="join",
            inputs=(
                QueryExpr(
                    op="alias",
                    inputs=(
                        QueryExpr(
                            op="materialized_view",
                            params={
                                "name": "left",
                                "columns": (
                                    "fact_id",
                                    "source_entity_id",
                                    "target_entity_id",
                                    "valid_at",
                                ),
                            },
                        ),
                    ),
                    params={"name": "old"},
                ),
                QueryExpr(
                    op="alias",
                    inputs=(
                        QueryExpr(
                            op="materialized_view",
                            params={
                                "name": "right",
                                "columns": (
                                    "fact_id",
                                    "source_entity_id",
                                    "target_entity_id",
                                    "valid_at",
                                ),
                            },
                        ),
                    ),
                    params={"name": "new"},
                ),
            ),
            params={"on": predicates, "how": "inner"},
        )
    ) == tuple(result.columns)


def test_relational_predicate_join_rejects_non_inner_how() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})

    with pytest.raises(NotImplementedError, match="predicate join"):
        execute_join(
            QueryExpr(
                op="join",
                inputs=(left, right),
                params={
                    "on": ((ColumnExpr("rank") < 3).to_param(),),
                    "how": "left",
                },
            ),
            {
                "left": pd.DataFrame({"rank": [1]}),
                "right": pd.DataFrame({"rank": [2]}),
            },
            LotusAdapter().execute,
        )


def test_relational_predicate_self_join_orders_composite_ids() -> None:
    facts = am.Log({"fact_id": "Composite fact identity."})
    earlier = facts.alias("earlier")
    later = facts.alias("later")
    query = earlier.join(
        later,
        on=[earlier.col("fact_id") < later.col("fact_id")],
    ).expr

    result = LotusAdapter().execute(
        query,
        {"log": pd.DataFrame({"fact_id": [(1, 2), (1, 0), (2, 0)]})},
    )

    assert result[["fact_id:earlier", "fact_id:later"]].to_dict("records") == [
        {"fact_id:earlier": (1, 2), "fact_id:later": (2, 0)},
        {"fact_id:earlier": (1, 0), "fact_id:later": (1, 2)},
        {"fact_id:earlier": (1, 0), "fact_id:later": (2, 0)},
    ]


def test_relational_join_rejects_missing_or_null_keys() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    execute = LotusAdapter().execute

    with pytest.raises(ValueError, match="must exist"):
        execute_join(
            QueryExpr(
                op="join",
                inputs=(left, right),
                params={"on": ("name",), "how": "inner"},
            ),
            {
                "left": pd.DataFrame({"name": ["a"]}),
                "right": pd.DataFrame({"topic": ["a"]}),
            },
            execute,
        )

    with pytest.raises(ValueError, match="cannot contain null"):
        execute_join(
            QueryExpr(
                op="join",
                inputs=(left, right),
                params={"on": ("name",), "how": "inner"},
            ),
            {
                "left": pd.DataFrame({"name": ["a", None]}),
                "right": pd.DataFrame({"name": ["a"]}),
            },
            execute,
        )

    with pytest.raises(ValueError, match="join how"):
        execute_join(
            QueryExpr(
                op="join",
                inputs=(left, right),
                params={"on": ("name",), "how": "cross"},
            ),
            {
                "left": pd.DataFrame({"name": ["a"]}),
                "right": pd.DataFrame({"name": ["a"]}),
            },
            execute,
        )


def test_subtract_requires_matching_columns() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame({"message": ["hello"]}),
        "right": pd.DataFrame({"other": ["hello"]}),
    }

    with pytest.raises(ValueError, match="matching columns"):
        execute_subtract(
            QueryExpr(op="subtract", inputs=(left, right)),
            inputs,
            LotusAdapter().execute,
        )


def test_lotus_adapter_dispatches_relational_execution_ops() -> None:
    left = QueryExpr(op="materialized_view", params={"name": "left"})
    right = QueryExpr(op="materialized_view", params={"name": "right"})
    inputs = {
        "left": pd.DataFrame({"message": ["hello"]}),
        "right": pd.DataFrame({"message": ["world"]}),
    }
    adapter = LotusAdapter()

    result = adapter.execute(QueryExpr(op="concat", inputs=(left, right)), inputs)
    joined = adapter.execute(
        QueryExpr(
            op="join",
            inputs=(left, right),
            params={"on": ("message",), "how": "inner"},
        ),
        {
            "left": pd.DataFrame({"message": ["hello"], "left_value": [1]}),
            "right": pd.DataFrame({"message": ["hello"], "right_value": [2]}),
        },
    )

    assert list(result["message"]) == ["hello", "world"]
    assert joined.to_dict("records") == [
        {"message": "hello", "left_value": 1, "right_value": 2}
    ]


def test_sem_join_assembles_outer_shape_with_overlapping_columns_and_explanations() -> None:
    left = pd.DataFrame(
        {"id": [1, 2], "message": ["prefers concise docs", "likes tea"]},
        index=[10, 20],
    )
    right = pd.DataFrame(
        {"id": [100, 200], "summary": ["short docs", "coffee"]},
        index=[1000, 2000],
    )

    result = assemble_join_frame(
        left,
        right,
        join_results=[(10, 1000, "same preference")],
        how="outer",
        return_explanations=True,
        explanation_column="explanation_join",
    )

    assert list(result.columns) == [
        "id:left",
        "message",
        "id:right",
        "summary",
        "explanation_join",
    ]
    assert result.loc[0, "message"] == "prefers concise docs"
    assert result.loc[0, "summary"] == "short docs"
    assert result.loc[0, "explanation_join"] == "same preference"
    assert result.loc[1, "message"] == "likes tea"
    assert pd.isna(result.loc[1, "summary"])
    assert pd.isna(result.loc[2, "message"])
    assert result.loc[2, "summary"] == "coffee"


def test_sem_join_assembles_inner_left_right_and_outer_rows() -> None:
    left = pd.DataFrame({"message": ["docs", "tea"]}, index=[10, 20])
    right = pd.DataFrame({"category": ["documentation", "coffee"]}, index=[100, 200])
    matches = [(10, 100, None)]

    inner = assemble_join_frame(left, right, matches, how="inner")
    left_join = assemble_join_frame(left, right, matches, how="left")
    right_join = assemble_join_frame(left, right, matches, how="right")
    outer = assemble_join_frame(left, right, matches, how="outer")

    assert len(inner) == 1
    assert len(left_join) == 2
    assert len(right_join) == 2
    assert len(outer) == 3
    assert pd.isna(left_join.loc[1, "category"])
    assert pd.isna(right_join.loc[1, "message"])
    assert pd.isna(outer.loc[1, "category"])
    assert pd.isna(outer.loc[2, "message"])


def test_sem_join_empty_input_uses_unmatched_rows_without_lotus_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    def execute(query: QueryExpr, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return inputs[str(query.params["name"])]

    def evaluate_semantic_join(*args: object) -> list[tuple[int, int, None]]:
        raise AssertionError("semantic join should not run for empty inputs")

    monkeypatch.setattr(
        sem_join_module,
        "evaluate_semantic_join",
        evaluate_semantic_join,
    )
    query = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={"instruction": "same topic", "how": "outer"},
    )

    result = sem_join_module.execute_sem_join(
        query,
        {
            "left": pd.DataFrame({"topic": ["docs"]}),
            "right": pd.DataFrame(columns=["topic"]),
        },
        execute,
        Context(),
    )

    assert list(result.columns) == ["topic:left", "topic:right"]
    assert result.loc[0, "topic:left"] == "docs"
    assert pd.isna(result.loc[0, "topic:right"])


def test_sem_join_series_supports_explicit_and_fallback_formats() -> None:
    left = pd.DataFrame(
        {"message": ["Alice prefers concise docs."], "speaker": ["Alice"]}
    )
    right = pd.DataFrame({"category": ["documentation preference"]})

    left_series, right_series, left_label, right_label, instruction = join_series(
        left,
        right,
        "{message:left} belongs to {category:right}.",
    )

    assert left_label == "message:left"
    assert right_label == "category:right"
    assert instruction == "{message:left} belongs to {category:right}."
    assert left_series.iloc[0] == "Alice prefers concise docs."
    assert right_series.iloc[0] == "documentation preference"

    fallback_left, fallback_right, fallback_left_label, fallback_right_label, fallback_instruction = join_series(
        left,
        right,
        "The left row belongs to the right category.",
    )

    assert fallback_left_label == "left"
    assert fallback_right_label == "right"
    assert "satisfy this semantic join condition" in fallback_instruction
    assert "message: Alice prefers concise docs." in fallback_left.iloc[0]
    assert "category: documentation preference" in fallback_right.iloc[0]


def test_sem_join_series_uses_composite_records_for_multi_column_side_aware_join() -> None:
    left = pd.DataFrame(
        {
            "name": ["adoption_goal"],
            "description": ["Caroline wants to adopt children."],
            "type": ["user"],
            "body": ["left body should not be included"],
        }
    )
    right = pd.DataFrame(
        {
            "name": ["caroline_adoption_journey"],
            "description": ["Caroline is pursuing single-parent adoption."],
            "type": ["user"],
            "body": ["right body should not be included"],
        }
    )

    left_series, right_series, left_label, right_label, instruction = join_series(
        left,
        right,
        (
            "Rows refer to the same topic when {name:left} and {name:right}, "
            "{description:left} and {description:right}, and {type:left} and "
            "{type:right} match."
        ),
    )

    assert left_label == "left"
    assert right_label == "right"
    assert "satisfy this semantic join condition" in instruction
    assert "name: adoption_goal" in left_series.iloc[0]
    assert "description: Caroline wants to adopt children." in left_series.iloc[0]
    assert "type: user" in left_series.iloc[0]
    assert "body:" not in left_series.iloc[0]
    assert "name: caroline_adoption_journey" in right_series.iloc[0]
    assert "description: Caroline is pursuing single-parent adoption." in right_series.iloc[0]
    assert "type: user" in right_series.iloc[0]
    assert "body:" not in right_series.iloc[0]


def test_lotus_sem_join_parse_default_is_false() -> None:
    assert LotusExecutionConfig().sem_join_default is False


def test_sem_join_pairwise_trace_writes_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_join as sem_join_module

    class Output:
        join_results = [(10, 200, None)]
        filter_outputs = [False, True, False, False]
        all_raw_outputs = ["False", "True", "No", "False"]
        all_explanations = [None, "same topic", None, None]

    def sem_join(*args: Any, **kwargs: Any) -> Output:
        return Output()

    monkeypatch.setattr(sem_join_module, "sem_join", sem_join)
    left = pd.DataFrame(
        {"message": ["likes tea", "adoption goal"]},
        index=[10, 20],
    )
    right = pd.DataFrame(
        {"summary": ["coffee preference", "family planning"]},
        index=[100, 200],
    )
    query = QueryExpr(
        op="sem_join",
        params={
            "instruction": "{message:left} and {summary:right} describe the same memory topic."
        },
    )

    matches = evaluate_semantic_join(
        query,
        left,
        right,
        LotusExecutionConfig(semantic_trace_dir=tmp_path),
    )

    assert matches == [(10, 200, None)]
    events = trace_events(tmp_path)
    assert list(event["operator"] for event in events) == ["sem_join"] * 4
    assert list(event["left_id"] for event in events) == [10, 10, 20, 20]
    assert list(event["right_id"] for event in events) == [100, 200, 100, 200]
    assert [trace_artifact(tmp_path, event["parsed_output_path"]) for event in events] == [
        False,
        True,
        False,
        False,
    ]
    assert [trace_artifact(tmp_path, event["raw_output_path"]) for event in events] == [
        "False",
        "True",
        "No",
        "False",
    ]
    assert {event["default"] for event in events} == {False}


def test_sem_join_pairwise_trace_disabled_writes_no_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_join as sem_join_module

    class Output:
        join_results = []
        filter_outputs = [False]
        all_raw_outputs = ["False"]
        all_explanations = [None]

    def sem_join(*args: Any, **kwargs: Any) -> Output:
        return Output()

    monkeypatch.setattr(sem_join_module, "sem_join", sem_join)
    query = QueryExpr(
        op="sem_join",
        params={
            "instruction": "{message:left} and {summary:right} describe the same memory topic."
        },
    )

    matches = evaluate_semantic_join(
        query,
        pd.DataFrame({"message": ["likes tea"]}),
        pd.DataFrame({"summary": ["coffee preference"]}),
        LotusExecutionConfig(),
    )

    assert matches == []
    assert trace_events(tmp_path) == []


def test_sem_join_pairwise_trace_writes_events_and_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus
    import lotus.sem_ops.sem_join as sem_join_module

    class FakeLM:
        pass

    class Output:
        join_results = [(10, 200, None)]
        filter_outputs = [False, True, False, False]
        all_raw_outputs = ["False", "True", "No", "False"]
        all_explanations = [None, "same topic", None, None]

    def sem_join(*args: Any, **kwargs: Any) -> Output:
        return Output()

    monkeypatch.setattr(lotus.settings, "lm", FakeLM())
    monkeypatch.setattr(sem_join_module, "sem_join", sem_join)
    trace_dir = tmp_path / "trace"
    query = QueryExpr(
        op="sem_join",
        params={
            "instruction": "{message:left} and {summary:right} describe the same memory topic."
        },
    )

    matches = evaluate_semantic_join(
        query,
        pd.DataFrame({"message": ["likes tea", "adoption goal"]}, index=[10, 20]),
        pd.DataFrame({"summary": ["coffee preference", "family planning"]}, index=[100, 200]),
        LotusExecutionConfig(semantic_trace_dir=trace_dir),
    )

    assert matches == [(10, 200, None)]
    events = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 4
    assert {event["event_type"] for event in events} == {"pair_decision"}
    assert list(event["operator"] for event in events) == ["sem_join"] * 4
    parsed_path = trace_dir / events[1]["parsed_output_path"].removeprefix("trace/")
    assert json.loads(parsed_path.read_text(encoding="utf-8")) is True
    assert "prompt_path" not in events[0]
    assert (trace_dir / events[0]["left_snapshot_path"].removeprefix("trace/")).exists()
    assert (trace_dir / events[0]["right_snapshot_path"].removeprefix("trace/")).exists()


def test_sem_join_empty_side_trace_writes_left_right_and_output_snapshots(
    tmp_path: Path,
) -> None:
    class Context:
        config = LotusExecutionConfig(semantic_trace_dir=tmp_path)

        def configure(self) -> None:
            pass

    query = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={
            "how": "outer",
            "instruction": "{name:left} and {name:right} describe the same memory topic.",
        },
    )
    inputs = {
        "left": pd.DataFrame({"name": ["adoption"], "body": ["family goal"]}),
        "right": pd.DataFrame({"name": pd.Series(dtype="object"), "body": pd.Series(dtype="object")}),
    }

    result = execute_sem_join(query, inputs, LotusAdapter().execute, Context())

    assert len(result) == 1
    [event] = trace_events(tmp_path)
    assert event["operator"] == "sem_join"
    assert event["event_type"] == "operator_result"
    assert event["skipped_pairwise"] is True
    assert event["left_rows"] == 1
    assert event["right_rows"] == 0
    assert event["output_rows"] == 1
    assert "prompt_path" not in event
    left_snapshot = pd.read_csv(trace_dir_from_event(tmp_path, event["left_snapshot_path"]))
    right_snapshot = pd.read_csv(trace_dir_from_event(tmp_path, event["right_snapshot_path"]))
    output_snapshot = pd.read_csv(trace_dir_from_event(tmp_path, event["output_snapshot_path"]))
    assert left_snapshot.to_dict("records") == [{"name": "adoption", "body": "family goal"}]
    assert list(right_snapshot.columns) == ["name", "body"]
    assert right_snapshot.empty
    assert len(output_snapshot) == 1


def test_sem_join_converts_mapping_to_lotus_cascade_args() -> None:
    cascade_args = cascade_args_from_mapping(
        {"recall_target": 0.95, "precision_target": 0.9}
    )

    assert cascade_args.recall_target == 0.95
    assert cascade_args.precision_target == 0.9


def test_sem_groupby_assigns_stable_group_ids_from_exact_and_semantic_matches() -> None:
    source = pd.DataFrame(
        {
            "name": ["docs", "docs", "meetings", "calls"],
            "description": ["short docs", "short docs", "team sync", "phone"],
        }
    )

    result = assign_semantic_group_ids(
        source,
        input_cols=("name", "description"),
        matched_unique_pairs=[(1, 2)],
    )

    assert list(result[GROUP_ID_COLUMN]) == [0, 0, 1, 1]


def test_sem_groupby_pairwise_default_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    captured: dict[str, Any] = {}

    class Output:
        outputs = [True]

    def sem_filter(*args: Any, **kwargs: Any) -> Output:
        captured["default"] = kwargs["default"]
        captured["progress_bar_desc"] = kwargs["progress_bar_desc"]
        captured["instruction"] = args[2]
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame(
        {
            "name": ["adoption", "inclusive adoption"],
            "description": ["adoption goal", "LGBTQ+ adoption support"],
        }
    )

    matches = evaluate_group_matches(
        source,
        input_cols=("name", "description"),
        instruction="Rows have the same {name} and compatible {description} meaning.",
    )

    assert matches == [(0, 1)]
    assert captured["default"] is False
    assert captured["progress_bar_desc"] == "Grouping comparisons"
    assert "{left}" in captured["instruction"]
    assert "{right}" in captured["instruction"]
    assert "{name}" not in captured["instruction"]
    assert "{description}" not in captured["instruction"]
    assert "same name and compatible description meaning" in captured["instruction"]


def test_sem_groupby_pairwise_instruction_rejects_unknown_column_placeholder() -> None:
    with pytest.raises(ValueError, match="unknown sem_groupby input column"):
        lower_pairwise_grouping_instruction(
            "Rows with the same {missing_column} belong together.",
            input_cols=("name", "description"),
        )


def test_sem_groupby_pairwise_instruction_preserves_escaped_literal_braces() -> None:
    lowered = lower_pairwise_grouping_instruction(
        "Treat {{name}} as literal text, but compare {name}.",
        input_cols=("name",),
    )

    assert lowered == "Treat {{name}} as literal text, but compare name."


def test_sem_groupby_pairwise_trace_writes_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    class Output:
        outputs = [True, False, True]
        raw_outputs = ["True", "False", "True because same preference"]
        explanations = ["same", None, "same"]

    def sem_filter(*args: Any, **kwargs: Any) -> Output:
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame(
        {
            "name": ["adoption", "tea", "adoption agencies"],
            "description": ["family goal", "drink preference", "family planning"],
        }
    )

    matches = evaluate_group_matches(
        source,
        input_cols=("name", "description"),
        instruction="Rows describe the same durable memory topic.",
        trace_dir=tmp_path,
    )

    assert matches == [(0, 1), (1, 2)]
    events = trace_events(tmp_path)
    assert list(event["operator"] for event in events) == [
        "sem_groupby",
        "sem_groupby",
        "sem_groupby",
    ]
    assert list(event["left_unique_id"] for event in events) == [0, 0, 1]
    assert list(event["right_unique_id"] for event in events) == [1, 2, 2]
    assert [trace_artifact(tmp_path, event["parsed_output_path"]) for event in events] == [
        True,
        False,
        True,
    ]
    assert [trace_artifact(tmp_path, event["raw_output_path"]) for event in events] == [
        "True",
        "False",
        "True because same preference",
    ]
    assert {event["default"] for event in events} == {False}


def test_sem_groupby_pairwise_trace_disabled_writes_no_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    class Output:
        outputs = [False]

    def sem_filter(*args: Any, **kwargs: Any) -> Output:
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame(
        {
            "name": ["adoption", "adoption agencies"],
            "description": ["family goal", "family planning"],
        }
    )

    matches = evaluate_group_matches(
        source,
        input_cols=("name", "description"),
        instruction="Rows describe the same durable memory topic.",
    )

    assert matches == []
    assert trace_events(tmp_path) == []


def test_sem_groupby_pairwise_default_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    captured: dict[str, Any] = {}

    class Output:
        outputs = [False]

    def sem_filter(*args: Any, **kwargs: Any) -> Output:
        captured["default"] = kwargs["default"]
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame(
        {
            "name": ["adoption", "inclusive adoption"],
            "description": ["adoption goal", "LGBTQ+ adoption support"],
        }
    )

    matches = evaluate_group_matches(
        source,
        input_cols=("name", "description"),
        instruction="Rows describe the same durable memory topic.",
        default=True,
    )

    assert matches == []
    assert captured["default"] is True


def test_sem_groupby_declared_labels_assign_label_column_and_group_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        {
            "title": ["Fast storage", "Transformer evals"],
            "abstract": ["Database cache design.", "Model benchmark study."],
        }
    )

    class Executor:
        def __init__(self, obj: pd.DataFrame) -> None:
            self.obj = obj

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            assert kwargs["input_cols"] == ("title", "abstract")
            assert kwargs["output_cols"] == (ColumnSpec("_label", "One of: systems, ml."),)
            assert "Do not invent labels" in kwargs["instruction"]
            return StructuredGenerationResult(
                parsed_outputs=[{"_label": "systems"}, {"_label": "ml"}],
                raw_outputs=['{"_label":"systems"}', '{"_label":"ml"}'],
                explanations=[None, None],
            )

    monkeypatch.setattr(sem_groupby_module, "StructuredLMExecutor", Executor)

    result = assign_declared_labels(
        source,
        input_cols=("title", "abstract"),
        labels=(
            ColumnSpec("systems", "Systems and databases."),
            ColumnSpec("ml", "Machine learning."),
        ),
        label_col="_label",
        instruction="Assign each paper to the best matching research area.",
    )

    assert list(result["_label"]) == ["systems", "ml"]
    assert list(result[GROUP_ID_COLUMN]) == [0, 1]
    assert result.attrs["agent_memory_groupby_labels"] == ("systems", "ml")
    assert result.attrs["agent_memory_groupby_label_col"] == "_label"


def test_sem_groupby_declared_labels_reject_invalid_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame({"title": ["Unknown paper"]})

    class Executor:
        def __init__(self, obj: pd.DataFrame) -> None:
            self.obj = obj

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            return StructuredGenerationResult(
                parsed_outputs=[{"_label": "other"}],
                raw_outputs=['{"_label":"other"}'],
                explanations=[None],
            )

    monkeypatch.setattr(sem_groupby_module, "StructuredLMExecutor", Executor)

    with pytest.raises(ValueError, match="declare an 'other' label explicitly"):
        assign_declared_labels(
            source,
            input_cols=("title",),
            labels=(ColumnSpec("systems", "Systems and databases."),),
            label_col="_label",
            instruction="Assign each paper to the best matching research area.",
        )


def test_sem_groupby_declared_labels_reject_label_col_collision() -> None:
    source = pd.DataFrame({"title": ["Fast storage"], "_label": ["existing"]})

    with pytest.raises(ValueError, match="already exists"):
        assign_declared_labels(
            source,
            input_cols=("title",),
            labels=(ColumnSpec("systems", "Systems and databases."),),
            label_col="_label",
            instruction="Assign each paper to the best matching research area.",
        )


def test_sem_agg_resolves_input_columns_and_builds_structured_instruction() -> None:
    source = pd.DataFrame(
        {
            "topic": ["docs", "meetings"],
            "body": ["Prefers concise docs.", "Plans weekly sync."],
            GROUP_ID_COLUMN: [0, 0],
        }
    )
    source.attrs["agent_memory_groupby_input_cols"] = ("topic",)

    assert aggregate_input_columns(source, None) == ("topic", "body")
    query = QueryExpr(
        op="sem_agg",
        params={"instruction": "Merge {body} into durable memory."},
    )
    instruction = structured_aggregate_instruction(
        query,
        ("body",),
        (
            ColumnSpec("topic", "Short topic."),
            ColumnSpec("body", "Durable memory summary."),
        ),
    )

    assert "Return exactly one valid JSON object" in instruction
    assert "- topic: Short topic." in instruction
    assert "- body: Durable memory summary." in instruction
    assert 'Expected JSON shape: {"topic": "string", "body": "string"}' in instruction


def test_sem_agg_instruction_preserves_output_only_placeholders() -> None:
    query = QueryExpr(
        op="sem_agg",
        params={
            "instruction": (
                "Use {content} to return canonical {name} and concise {summary}."
            ),
            "output_cols": (
                ColumnSpec("name", "Canonical name."),
                ColumnSpec("summary", "Concise summary."),
            ),
        },
    )

    instruction = structured_aggregate_instruction(
        query,
        ("content",),
        query.params["output_cols"],
    )

    assert "Use Content to return canonical {name} and concise {summary}." in instruction


def test_lotus_style_sem_agg_passes_response_format_only_on_final_pass() -> None:
    calls: list[dict[str, Any]] = []

    class Model:
        max_ctx_len = 10
        max_tokens = 1

        def count_tokens(self, value: Any) -> int:
            text = str(value)
            if "doc two" in text:
                return 100
            return 1

        def __call__(self, batch: list[Any], **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if len(calls) == 1:
                return SimpleNamespace(outputs=("partial one", "partial two"))
            return SimpleNamespace(outputs=('{"topic": "docs", "body": "summary"}',))

    output = lotus_style_sem_agg(
        ["doc one", "doc two"],
        Model(),
        "Merge documents.",
        [0, 0],
        response_format=JSON_OBJECT_RESPONSE_FORMAT,
        final_model_kwargs={"max_tokens": 1024},
    )

    assert output == '{"topic": "docs", "body": "summary"}'
    assert "response_format" not in calls[0]
    assert calls[1]["response_format"] == JSON_OBJECT_RESPONSE_FORMAT
    assert calls[1]["max_tokens"] == 1024


def test_lotus_style_structured_sem_agg_retries_invalid_final_output() -> None:
    from agent_memory.adapters.lotus.sem_agg import lotus_style_structured_sem_agg

    calls: list[dict[str, Any]] = []
    outputs = iter(("", '{"topic": "docs", "body": "summary"}'))

    class Model:
        max_ctx_len = 4096
        max_tokens = 512

        def count_tokens(self, value: Any) -> int:
            return 1

        def __call__(self, batch: list[Any], **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return SimpleNamespace(outputs=(next(outputs),))

    reset_structured_retry_stats()
    result = lotus_style_structured_sem_agg(
        ["doc one"],
        Model(),
        "Merge documents.",
        [0],
        output_cols=(ColumnSpec("topic"), ColumnSpec("body")),
        max_retries=1,
        final_model_kwargs={"max_tokens": 1024},
    )

    assert result.raw_outputs == ['{"topic": "docs", "body": "summary"}']
    assert result.raw_output_attempts == (
        ("", '{"topic": "docs", "body": "summary"}'),
    )
    assert result.invalid_indices == ()
    assert len(calls) == 2
    assert calls[0]["response_format"] == JSON_OBJECT_RESPONSE_FORMAT
    assert calls[1]["response_format"] == JSON_OBJECT_RESPONSE_FORMAT
    assert structured_retry_stats().retry_batches == 1


def test_lotus_style_structured_sem_agg_retries_only_final_call() -> None:
    from agent_memory.adapters.lotus.sem_agg import lotus_style_structured_sem_agg

    calls: list[tuple[int, dict[str, Any]]] = []

    class Model:
        max_ctx_len = 10
        max_tokens = 1

        def count_tokens(self, value: Any) -> int:
            return 100 if "doc two" in str(value) else 1

        def __call__(self, batch: list[Any], **kwargs: Any) -> Any:
            calls.append((len(batch), dict(kwargs)))
            if len(calls) == 1:
                return SimpleNamespace(outputs=("partial one", "partial two"))
            if len(calls) == 2:
                return SimpleNamespace(outputs=("",))
            return SimpleNamespace(
                outputs=('{"topic": "docs", "body": "summary"}',)
            )

    reset_structured_retry_stats()
    result = lotus_style_structured_sem_agg(
        ["doc one", "doc two"],
        Model(),
        "Merge documents.",
        [0, 0],
        output_cols=(ColumnSpec("topic"), ColumnSpec("body")),
        max_retries=1,
    )

    assert result.raw_output_attempts == (
        ("", '{"topic": "docs", "body": "summary"}'),
    )
    assert [batch_size for batch_size, _kwargs in calls] == [2, 1, 1]
    assert "response_format" not in calls[0][1]
    assert calls[1][1]["response_format"] == JSON_OBJECT_RESPONSE_FORMAT
    assert calls[2][1]["response_format"] == JSON_OBJECT_RESPONSE_FORMAT


def test_lotus_style_structured_sem_agg_does_not_retry_valid_output() -> None:
    from agent_memory.adapters.lotus.sem_agg import lotus_style_structured_sem_agg

    calls: list[dict[str, Any]] = []
    valid_output = '{"topic": "docs", "body": "summary"}'

    class Model:
        max_ctx_len = 4096
        max_tokens = 512

        def count_tokens(self, value: Any) -> int:
            return 1

        def __call__(self, batch: list[Any], **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return SimpleNamespace(outputs=(valid_output,))

    reset_structured_retry_stats()
    result = lotus_style_structured_sem_agg(
        ["doc one"],
        Model(),
        "Merge documents.",
        [0],
        output_cols=(ColumnSpec("topic"), ColumnSpec("body")),
        max_retries=1,
        final_model_kwargs={"max_tokens": 1024},
    )

    assert result.raw_outputs == [valid_output]
    assert result.raw_output_attempts == ((valid_output,),)
    assert result.invalid_indices == ()
    assert len(calls) == 1
    assert structured_retry_stats().retry_batches == 0


def test_lotus_style_structured_sem_agg_records_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent_memory.adapters.lotus.sem_agg import lotus_style_structured_sem_agg

    calls: list[dict[str, Any]] = []
    outputs = iter(("", ""))

    class Model:
        max_ctx_len = 4096
        max_tokens = 512

        def count_tokens(self, value: Any) -> int:
            return 1

        def __call__(self, batch: list[Any], **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return SimpleNamespace(outputs=(next(outputs),))

    reset_structured_retry_stats()
    monkeypatch.setattr(structured_module, "STRUCTURED_FAILURE_DIR", tmp_path)
    result = lotus_style_structured_sem_agg(
        ["doc one"],
        Model(),
        "Merge documents.",
        [0],
        output_cols=(ColumnSpec("topic"), ColumnSpec("body")),
        max_retries=1,
        failure_extra_by_index={0: {"group_index": 3}},
    )

    assert result.raw_outputs == [""]
    assert result.raw_output_attempts == (("", ""),)
    assert result.invalid_indices == (0,)
    assert len(calls) == 2
    assert len(result.failure_artifact_paths) == 1
    artifact = json.loads(
        result.failure_artifact_paths[0].read_text(encoding="utf-8")
    )
    assert artifact["raw_outputs"] == ["", ""]
    assert artifact["group_index"] == 3
    stats = structured_retry_stats()
    assert stats.retry_batches == 1
    assert stats.failure_artifacts == 1


def test_sem_agg_model_kwargs_use_structured_max_tokens_by_default() -> None:
    kwargs = structured_sem_agg_model_kwargs(LotusExecutionConfig())

    assert kwargs["max_tokens"] >= DEFAULT_STRUCTURED_MAX_TOKENS


def test_sem_agg_model_kwargs_can_override_max_tokens() -> None:
    kwargs = structured_sem_agg_model_kwargs(
        LotusExecutionConfig(sem_agg_model_kwargs={"max_tokens": 2048})
    )

    assert kwargs["max_tokens"] == 2048


def test_sem_agg_model_kwargs_cannot_override_response_format() -> None:
    config = LotusExecutionConfig(
        sem_agg_model_kwargs={"response_format": {"type": "text"}},
    )

    with pytest.raises(ValueError, match="cannot override response_format"):
        structured_sem_agg_model_kwargs(config)


def test_sem_agg_model_kwargs_cannot_override_progress_bar_desc() -> None:
    config = LotusExecutionConfig(
        sem_agg_model_kwargs={"progress_bar_desc": "Bad"},
    )

    with pytest.raises(ValueError, match="cannot override progress_bar_desc"):
        structured_sem_agg_model_kwargs(config)


def test_sem_agg_groups_rows_by_internal_group_id() -> None:
    source = pd.DataFrame(
        {
            "body": ["doc one", "doc two", "cooking"],
            GROUP_ID_COLUMN: [0, 0, 1],
        }
    )

    groups = aggregate_groups(source)

    assert len(groups) == 2
    assert [list(group["body"]) for group in groups] == [
        ["doc one", "doc two"],
        ["cooking"],
    ]
    assert all(GROUP_ID_COLUMN not in group.columns for group in groups)


def test_sem_agg_grouped_single_output_returns_one_row_per_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    captured_groups: list[list[str]] = []

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    def execute_native_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        config: LotusExecutionConfig | None = None,
    ) -> str:
        captured_groups.append(list(group["body"]))
        return " + ".join(group["body"])

    monkeypatch.setattr(
        sem_agg_module,
        "execute_native_sem_agg_group",
        execute_native_sem_agg_group,
    )
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("summary"),),
            "instruction": "Merge {body}.",
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "body": ["doc one", "doc two", "cooking"],
                GROUP_ID_COLUMN: [0, 0, 1],
            }
        )
    }

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert captured_groups == [["doc one", "doc two"], ["cooking"]]
    assert list(result["summary"]) == ["doc one + doc two", "cooking"]


def test_group_by_sem_agg_preserves_deterministic_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    def execute_native_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        config: LotusExecutionConfig | None = None,
    ) -> str:
        return " + ".join(group["body"])

    monkeypatch.setattr(
        sem_agg_module,
        "execute_native_sem_agg_group",
        execute_native_sem_agg_group,
    )
    source = QueryExpr(op="materialized_view", params={"name": "source"})
    query = QueryExpr(
        op="sem_agg",
        inputs=(
            QueryExpr(
                op="group_by",
                inputs=(source,),
                params={"keys": ("topic",)},
            ),
        ),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("summary"),),
            "instruction": "Summarize {body}.",
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "topic": ["docs", "docs", "meetings"],
                "body": ["a", "b", "c"],
            }
        )
    }

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert list(result.columns) == ["topic", "summary"]
    assert result.to_dict(orient="records") == [
        {"topic": "docs", "summary": "a + b"},
        {"topic": "meetings", "summary": "c"},
    ]
    assert output_columns(query) == ("topic", "summary")


def test_sem_agg_whole_single_output_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    def execute_native_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        config: LotusExecutionConfig | None = None,
    ) -> str:
        return ",".join(group["body"])

    monkeypatch.setattr(
        sem_agg_module,
        "execute_native_sem_agg_group",
        execute_native_sem_agg_group,
    )
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("summary"),),
            "instruction": "Summarize {body}.",
        },
    )
    inputs = {"source": pd.DataFrame({"body": ["a", "b"]})}

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert list(result["summary"]) == ["a,b"]


def test_sem_agg_grouped_multi_output_returns_one_row_per_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    structured_calls: list[tuple[list[str], tuple[str, ...], tuple[str, ...]]] = []

    def execute_lotus_style_structured_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        output_cols: tuple[ColumnSpec, ...],
        config: LotusExecutionConfig,
        *,
        group_index: int = 0,
    ) -> StructuredLMRetryResult:
        structured_calls.append(
            (
                list(group["body"]),
                tuple(input_cols),
                tuple(column.name for column in output_cols),
            )
        )
        index = len(structured_calls) - 1
        return sem_agg_retry_result((
            '{"topic": "docs", "body": "doc one and doc two"}',
            '{"topic": "cooking", "body": "cooking"}',
        )[index])

    monkeypatch.setattr(
        sem_agg_module,
        "execute_lotus_style_structured_sem_agg_group",
        execute_lotus_style_structured_sem_agg_group,
    )
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("topic"), ColumnSpec("body")),
            "instruction": "Merge {body}.",
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "body": ["doc one", "doc two", "cooking"],
                GROUP_ID_COLUMN: [0, 0, 1],
            }
        )
    }

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert list(result.columns) == ["topic", "body"]
    assert len(result) == 2
    assert list(result["topic"]) == ["docs", "cooking"]
    assert structured_calls == [
        (["doc one", "doc two"], ("body",), ("topic", "body")),
        (["cooking"], ("body",), ("topic", "body")),
    ]


def test_sem_agg_whole_multi_output_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    structured_calls: list[list[str]] = []

    def execute_lotus_style_structured_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        output_cols: tuple[ColumnSpec, ...],
        config: LotusExecutionConfig,
        *,
        group_index: int = 0,
    ) -> StructuredLMRetryResult:
        structured_calls.append(list(group["body"]))
        return sem_agg_retry_result(
            '{"topic": "docs", "body": "doc one and doc two"}'
        )

    monkeypatch.setattr(
        sem_agg_module,
        "execute_lotus_style_structured_sem_agg_group",
        execute_lotus_style_structured_sem_agg_group,
    )
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("topic"), ColumnSpec("body")),
            "instruction": "Merge {body}.",
        },
    )
    inputs = {"source": pd.DataFrame({"body": ["doc one", "doc two"]})}

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert list(result.columns) == ["topic", "body"]
    assert len(result) == 1
    assert result.loc[0, "topic"] == "docs"
    assert structured_calls == [["doc one", "doc two"]]


def test_sem_agg_structured_trace_writes_group_raw_and_parsed_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig(semantic_trace_dir=tmp_path)

        def configure(self) -> None:
            pass

    def execute_lotus_style_structured_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        output_cols: tuple[ColumnSpec, ...],
        config: LotusExecutionConfig,
        *,
        group_index: int = 0,
    ) -> StructuredLMRetryResult:
        return sem_agg_retry_result(
            '{"topic": "docs", "body": "doc one and doc two"}',
            attempts=("", '{"topic": "docs", "body": "doc one and doc two"}'),
        )

    monkeypatch.setattr(
        sem_agg_module,
        "execute_lotus_style_structured_sem_agg_group",
        execute_lotus_style_structured_sem_agg_group,
    )
    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("topic"), ColumnSpec("body")),
            "instruction": "Merge {body}.",
        },
    )
    inputs = {"source": pd.DataFrame({"body": ["doc one", "doc two"]})}

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert list(result["topic"]) == ["docs"]
    [event] = trace_events(tmp_path)
    assert event["operator"] == "sem_agg"
    assert event["event_type"] == "structured_generation"
    assert event["group_index"] == 0
    assert event["input_preview"] == '[ { "body": "doc one" }, { "body": "doc two" } ]'
    group_snapshot = pd.read_csv(trace_dir_from_event(tmp_path, event["group_snapshot_path"]))
    assert group_snapshot.to_dict("records") == [{"body": "doc one"}, {"body": "doc two"}]
    assert trace_artifact(tmp_path, event["raw_output_path"]) == [
        "",
        '{"topic": "docs", "body": "doc one and doc two"}'
    ]
    assert trace_artifact(tmp_path, event["parsed_output_path"]) == {
        "topic": "docs",
        "body": "doc one and doc two",
    }
    assert event["parse_error"] == ""
    assert event["parse_retry_attempts"] == 1


def test_sem_agg_rejects_invalid_structured_json() -> None:
    output_cols = (ColumnSpec("topic"), ColumnSpec("body"))

    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_sem_agg_output("not json", output_cols)

    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_sem_agg_output("", output_cols)

    with pytest.raises(ValueError, match="non-object JSON"):
        parse_structured_sem_agg_output("[]", output_cols)


def test_sem_agg_rejects_missing_structured_key() -> None:
    output_cols = (ColumnSpec("topic"), ColumnSpec("body"))

    with pytest.raises(ValueError, match="missing required keys"):
        parse_structured_sem_agg_output('{"topic": "docs"}', output_cols)


def test_sem_agg_preserves_structured_scalar_types() -> None:
    output_cols = (ColumnSpec("ordinal"), ColumnSpec("invalid_at"))

    assert parse_structured_sem_agg_output(
        {"ordinal": 2, "invalid_at": None},
        output_cols,
    ) == {"ordinal": 2, "invalid_at": None}
    with pytest.raises(ValueError, match="JSON scalar"):
        parse_structured_sem_agg_output(
            {"ordinal": [2], "invalid_at": None},
            output_cols,
        )


def test_sem_agg_structured_failure_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    def execute_lotus_style_structured_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        output_cols: tuple[ColumnSpec, ...],
        config: LotusExecutionConfig,
        *,
        group_index: int = 0,
    ) -> StructuredLMRetryResult:
        return sem_agg_retry_result("", invalid=True)

    monkeypatch.setattr(structured_module, "STRUCTURED_FAILURE_DIR", tmp_path)
    monkeypatch.setattr(
        sem_agg_module,
        "execute_lotus_style_structured_sem_agg_group",
        execute_lotus_style_structured_sem_agg_group,
    )

    query = QueryExpr(
        op="sem_agg",
        inputs=(QueryExpr(op="materialized_view", params={"name": "source"}),),
        params={
            "input_cols": ("body",),
            "output_cols": (ColumnSpec("topic"), ColumnSpec("body")),
            "instruction": "Merge {body}.",
        },
    )
    inputs = {"source": pd.DataFrame({"body": ["doc one", "doc two"]})}

    with pytest.raises(ValueError, match="structured failure artifact") as error:
        execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert str(tmp_path) in str(error.value)
    artifacts = list(tmp_path.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert artifact["operator"] == "sem_agg"
    assert artifact["shape"] == "object"
    assert artifact["group_index"] == 0
    assert artifact["expected_output_columns"] == [
        {"name": "topic", "description": None},
        {"name": "body", "description": None},
    ]
    assert artifact["raw_outputs"] == [""]
    assert "Return exactly one valid JSON object" in artifact["final_instruction"]
    assert artifact["group_row_preview"] == [
        {"body": "doc one"},
        {"body": "doc two"},
    ]


def test_lotus_adapter_dispatches_sem_join_groupby_and_agg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.adapter as lotus_adapter_module

    called: list[str] = []

    def execute_sem_join(*args: Any) -> str:
        called.append("sem_join")
        return "join"

    def execute_sem_groupby(*args: Any) -> str:
        called.append("sem_groupby")
        return "groupby"

    def execute_sem_agg(*args: Any) -> str:
        called.append("sem_agg")
        return "agg"

    monkeypatch.setattr(lotus_adapter_module, "execute_sem_join", execute_sem_join)
    monkeypatch.setattr(lotus_adapter_module, "execute_sem_groupby", execute_sem_groupby)
    monkeypatch.setattr(lotus_adapter_module, "execute_sem_agg", execute_sem_agg)
    adapter = LotusAdapter()

    assert adapter.execute(QueryExpr(op="sem_join"), {}) == "join"
    assert adapter.execute(QueryExpr(op="sem_groupby"), {}) == "groupby"
    assert adapter.execute(QueryExpr(op="sem_agg"), {}) == "agg"
    assert called == ["sem_join", "sem_groupby", "sem_agg"]


def test_runtime_preserves_log_bag_and_recomputes_distinct_view() -> None:
    class DistinctMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        rows = log.drop_duplicates()

    memory = DistinctMemory(adapter=LotusAdapter())
    memory.add({"message": "hello"})
    memory.add({"message": "hello"})
    memory.add({"message": "world"})

    assert list(memory._runtime._state["log"]["message"]) == [
        "hello",
        "hello",
        "world",
    ]
    assert list(memory._runtime._state["rows"]["message"]) == ["hello", "world"]


def test_runtime_executes_q_prime_and_stores_adapter_result() -> None:
    class FilterMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        helloworld_tests = log.sem_filter(
            instruction="{message} is a coherent sentence."
        ).select(["message"])

    class RecordingAdapter(LotusAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[QueryExpr, dict[str, pd.DataFrame]]] = []

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            if query.op == "sem_filter":
                self.calls.append((query, inputs))
                source_name = str(query.inputs[0].params["name"])
                return inputs[source_name].copy()
            return super().execute(query, inputs)

    adapter = RecordingAdapter()
    memory = FilterMemory(adapter=adapter)

    memory.add("hello")

    assert not hasattr(memory._runtime, "_planner")
    query, inputs = adapter.calls[0]
    assert query.op == "sem_filter"
    assert list(next(iter(inputs.values()))["message"]) == ["hello"]
    assert memory._runtime._state["helloworld_tests"].to_dict("records") == [
        {"message": "hello"}
    ]


def test_runtime_executes_standalone_sem_agg_q_prime_and_stores_result() -> None:
    class SummaryMemory(am.Memory):
        log = am.Log({"summary": "Memory summary.", "evidence": "Raw evidence."})
        summary = log.sem_agg(
            input_cols=["summary"],
            output_cols=["summary"],
            instruction="Merge summaries.",
        )

    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[QueryExpr, dict[str, pd.DataFrame]]] = []

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            self.calls.append((query, inputs))
            return pd.DataFrame({"summary": ["next aggregate"]})

    adapter = RecordingAdapter()
    memory = SummaryMemory(adapter=adapter)

    memory.add({"summary": "new summary", "evidence": "raw"})

    query, inputs = adapter.calls[0]
    assert query.op == "sem_agg"
    assert query.inputs[0].op == "union"
    current_source = query.inputs[0].inputs[0].inputs[0]
    current_node_id = memory._runtime.policy.view_outputs["summary"]
    _assert_materialized_view(
        current_source,
        name=current_node_id,
        columns=("summary",),
    )
    assert inputs[current_node_id].empty
    assert list(inputs[current_node_id].columns) == ["summary"]
    assert memory._runtime._state["summary"].to_dict("records") == [
        {"summary": "next aggregate"}
    ]


def test_runtime_maintains_array_agg_view_with_array_cat() -> None:
    class RecordsMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        records = log.array_agg(
            columns=("message",),
            output_col="conversation_records",
        )

    memory = RecordsMemory(adapter=LotusAdapter())

    memory.add({"message": "one"})
    memory.add({"message": "two"})

    records = memory._runtime._state["records"]
    assert list(records.columns) == ["conversation_records"]
    assert len(records) == 1
    assert json.loads(records.loc[0, "conversation_records"]) == [
        {"message": "one"},
        {"message": "two"},
    ]


def test_runtime_maintains_grouped_min_view_incrementally() -> None:
    class EarliestFactMemory(am.Memory):
        log = am.Log({"fact_id": "Fact id.", "valid_at": "Validity time."})
        earliest = log.group_by("fact_id").min(
            column="valid_at",
            output_col="valid_at",
        )

    memory = EarliestFactMemory(adapter=LotusAdapter())

    memory.add({"fact_id": "f1", "valid_at": "2026-01-03"})
    memory.add({"fact_id": "f1", "valid_at": "2026-01-01"})
    memory.add({"fact_id": "f2", "valid_at": None})

    assert memory._runtime._state["earliest"].to_dict("records") == [
        {"fact_id": "f1", "valid_at": "2026-01-01"},
        {"fact_id": "f2", "valid_at": None},
    ]


def test_runtime_maintains_count_window_process_state_incrementally() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log(
            {
                "timestamp": "Message timestamp.",
                "speaker": "Message speaker.",
                "message": "Message body.",
            }
        )
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("timestamp", "speaker", "message"),
                output_col="conversation_records",
            )
        )

    memory = WindowBlockMemory(adapter=LotusAdapter())

    memory.add({"timestamp": "t1", "speaker": "A", "message": "one"})
    assert memory._runtime._state["blocks"].empty
    assert "_blocks_process_window" not in memory._runtime._state

    memory.add({"timestamp": "t2", "speaker": "B", "message": "two"})
    blocks = memory._runtime._state["blocks"]
    assert len(blocks) == 1
    assert json.loads(blocks.loc[0, "conversation_records"]) == [
        {"timestamp": "t1", "speaker": "A", "message": "one"},
        {"timestamp": "t2", "speaker": "B", "message": "two"},
    ]

    memory.add({"timestamp": "t3", "speaker": "A", "message": "three"})
    blocks = memory._runtime._state["blocks"]
    assert len(blocks) == 2
    process_node_id = memory._runtime.policy.view_outputs["blocks"]
    assert memory._runtime._engine._window_next_start[process_node_id] == 2
    assert json.loads(blocks.loc[1, "conversation_records"]) == [
        {"timestamp": "t2", "speaker": "B", "message": "two"},
        {"timestamp": "t3", "speaker": "A", "message": "three"},
    ]


def test_runtime_snapshot_restores_count_window_bookkeeping() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("message",),
                output_col="conversation_records",
            )
        )

    original = WindowBlockMemory(adapter=LotusAdapter())
    original.add({"message": "one"})
    original.add({"message": "two"})
    snapshot = original._runtime.snapshot_state()

    restored = WindowBlockMemory(adapter=LotusAdapter())
    restored._runtime.restore_state(snapshot)
    restored.add({"message": "three"})

    blocks = restored._runtime._state["blocks"]
    assert len(blocks) == 2
    assert json.loads(blocks.loc[0, "conversation_records"]) == [
        {"message": "one"},
        {"message": "two"},
    ]
    assert json.loads(blocks.loc[1, "conversation_records"]) == [
        {"message": "two"},
        {"message": "three"},
    ]


def test_runtime_snapshot_copies_top_level_state_mapping() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("message",),
                output_col="conversation_records",
            )
        )

    memory = WindowBlockMemory(adapter=LotusAdapter())
    memory.add({"message": "one"})
    snapshot = memory._runtime.snapshot_state()

    memory._runtime._state["extra"] = pd.DataFrame([{"message": "later"}])

    assert "extra" not in snapshot["state"]


def test_runtime_maintains_count_window_over_selected_upstream_relation() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log(
            {
                "timestamp": "Message timestamp.",
                "speaker": "Message speaker.",
                "message": "Message body.",
            }
        )
        blocks = log.select(["message"]).count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("message",),
                output_col="conversation_records",
            )
        )

    memory = WindowBlockMemory(adapter=LotusAdapter())

    memory.add({"timestamp": "t1", "speaker": "A", "message": "one"})
    memory.add({"timestamp": "t2", "speaker": "B", "message": "two"})
    memory.add({"timestamp": "t3", "speaker": "A", "message": "three"})

    blocks = memory._runtime._state["blocks"]
    assert len(blocks) == 2
    assert json.loads(blocks.loc[0, "conversation_records"]) == [
        {"message": "one"},
        {"message": "two"},
    ]
    assert json.loads(blocks.loc[1, "conversation_records"]) == [
        {"message": "two"},
        {"message": "three"},
    ]


def test_runtime_maintains_over_array_agg_incrementally() -> None:
    class ContextMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        contextual = log.over(rows=(-2, -1)).array_agg(
            columns=("message",),
            output_col="previous_messages",
        )

    memory = ContextMemory(adapter=LotusAdapter())

    memory.add({"message": "one", "turn_id": "ignored-1"})
    memory.add({"message": "two", "turn_id": "ignored-2"})
    memory.add({"message": "three", "turn_id": "ignored-3"})

    contextual = memory._runtime._state["contextual"]
    assert list(contextual.columns) == ["message", "previous_messages"]
    assert list(contextual["message"]) == ["one", "two", "three"]
    assert json.loads(contextual.loc[0, "previous_messages"]) == []
    assert json.loads(contextual.loc[1, "previous_messages"]) == [{"message": "one"}]
    assert json.loads(contextual.loc[2, "previous_messages"]) == [
        {"message": "one"},
        {"message": "two"},
    ]


def test_runtime_window_update_commits_only_after_public_view_success() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        blocks = (
            log.count_window(size=2, slide=1)
            .process_window(
                lambda window: window.array_agg(
                    columns=("message",),
                    output_col="conversation_records",
                )
            )
            .select(["conversation_records"])
        )

    class FailingAfterProcessAdapter:
        def __init__(self) -> None:
            self.fail_downstream = True
            self.process_calls = 0
            self._delegate = LotusAdapter()

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            if query.op == "array_agg":
                self.process_calls += 1
                return self._delegate.execute(query, inputs)
            if query.op == "select" and self.fail_downstream:
                raise RuntimeError("downstream maintenance failed")
            return self._delegate.execute(query, inputs)

    adapter = FailingAfterProcessAdapter()
    memory = WindowBlockMemory(adapter=adapter)
    sink_node_id = memory._runtime.policy.view_outputs["blocks"]
    process_node_id = memory._runtime.policy.nodes[sink_node_id].input_node_ids[0]

    memory.add({"message": "one"})
    with pytest.raises(RuntimeError, match="downstream maintenance failed"):
        memory.add({"message": "two"})

    assert adapter.process_calls == 1
    assert memory._runtime._engine.node_state[process_node_id].empty
    assert memory._runtime._engine._window_next_start.get(process_node_id, 0) == 0
    assert memory._runtime._state["blocks"].empty
    assert memory._runtime._state["log"].to_dict("records") == [{"message": "one"}]

    adapter.fail_downstream = False
    memory.add({"message": "three"})

    assert adapter.process_calls == 2
    assert memory._runtime._engine._window_next_start[process_node_id] == 1
    private_blocks = memory._runtime._engine.node_state[process_node_id]
    public_blocks = memory._runtime._state["blocks"]
    assert len(private_blocks) == 1
    assert len(public_blocks) == 1
    assert json.loads(public_blocks.loc[0, "conversation_records"]) == [
        {"message": "one"},
        {"message": "three"},
    ]


def test_process_window_node_has_normalized_string_output_columns() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("message",),
                output_col="conversation_records",
            )
        )

    policy = WindowBlockMemory.differentiate_policy()
    node_id = policy.view_outputs["blocks"]
    node = policy.nodes[node_id]
    runtime = MemoryRuntime(policy, adapter=LotusAdapter())
    frame = runtime._engine._empty_node_frame(node_id)

    assert node.output_columns == ("conversation_records",)
    assert list(frame.columns) == ["conversation_records"]


def test_runtime_commits_window_cursor_when_process_output_is_empty() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("message",),
                output_col="conversation_records",
            )
        )

    class EmptyProcessAdapter:
        def __init__(self) -> None:
            self.process_calls = 0
            self.window_sizes: list[int] = []
            self._delegate = LotusAdapter()

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            if query.op == "array_agg":
                self.process_calls += 1
                self.window_sizes.append(len(inputs[WINDOW_SOURCE_INPUT]))
                return pd.DataFrame(columns=["conversation_records"])
            return self._delegate.execute(query, inputs)

    adapter = EmptyProcessAdapter()
    memory = WindowBlockMemory(adapter=adapter)
    process_node_id = memory._runtime.policy.view_outputs["blocks"]

    memory.add({"message": "one"})
    memory.add({"message": "two"})

    assert adapter.process_calls == 1
    assert adapter.window_sizes == [2]
    assert memory._runtime._state["blocks"].empty
    assert memory._runtime._engine.node_state[process_node_id].empty
    assert memory._runtime._engine._window_next_start[process_node_id] == 1

    memory.add({"message": "three"})

    assert adapter.process_calls == 2
    assert adapter.window_sizes == [2, 2]
    assert memory._runtime._state["blocks"].empty
    assert memory._runtime._engine.node_state[process_node_id].empty
    assert memory._runtime._engine._window_next_start[process_node_id] == 2


def test_runtime_count_window_uses_append_sequence_with_timestamp() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log(
            {
                "timestamp": "Message timestamp.",
                "speaker": "Message speaker.",
                "message": "Message body.",
            }
        )
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("timestamp", "speaker", "message"),
                output_col="conversation_records",
            )
        )

    memory = WindowBlockMemory(adapter=LotusAdapter())

    memory.add({"timestamp": "t1", "speaker": "A", "message": "one"})
    memory.add({"timestamp": "t3", "speaker": "B", "message": "three"})
    memory.add({"timestamp": "t2", "speaker": "A", "message": "two"})

    blocks = memory._runtime._state["blocks"]
    assert len(blocks) == 2
    assert json.loads(blocks.loc[0, "conversation_records"]) == [
        {"timestamp": "t1", "speaker": "A", "message": "one"},
        {"timestamp": "t3", "speaker": "B", "message": "three"},
    ]
    assert json.loads(blocks.loc[1, "conversation_records"]) == [
        {"timestamp": "t3", "speaker": "B", "message": "three"},
        {"timestamp": "t2", "speaker": "A", "message": "two"},
    ]


def test_runtime_count_window_uses_append_sequence_without_timestamp() -> None:
    class WindowBlockMemory(am.Memory):
        log = am.Log(
            {
                "speaker": "Message speaker.",
                "message": "Message body.",
            }
        )
        blocks = log.count_window(size=2, slide=1).process_window(
            lambda window: window.array_agg(
                columns=("speaker", "message"),
                output_col="conversation_records",
            )
        )

    memory = WindowBlockMemory(adapter=LotusAdapter())

    memory.add({"speaker": "A", "message": "one"})
    memory.add({"speaker": "B", "message": "two"})
    memory.add({"speaker": "A", "message": "three"})

    blocks = memory._runtime._state["blocks"]
    assert len(blocks) == 2
    assert json.loads(blocks.loc[0, "conversation_records"]) == [
        {"speaker": "A", "message": "one"},
        {"speaker": "B", "message": "two"},
    ]
    assert json.loads(blocks.loc[1, "conversation_records"]) == [
        {"speaker": "B", "message": "two"},
        {"speaker": "A", "message": "three"},
    ]


def test_helloworld_memory_real_lotus_e2e() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    if os.getenv("AGENT_MEMORY_RUN_LOTUS_E2E") != "1":
        pytest.skip("set AGENT_MEMORY_RUN_LOTUS_E2E=1 to run real LOTUS e2e")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for real LOTUS e2e")

    memory = HelloWorldTestMemory(adapter=LotusAdapter())

    memory.add(
        {
            "message": "I prefer concise design documents when we discuss architecture.",
            "speaker": "Alice",
            "session_id": "session_1",
            "turn_id": "1",
            "timestamp": "2024-01-01",
        }
    )
    memory.add(
        {
            "message": "green sleep quickly because table",
            "speaker": "Bob",
            "session_id": "session_1",
            "turn_id": "2",
            "timestamp": "2024-01-01",
        }
    )
    memory.add(
        {
            "message": "I am planning to visit Sarah next weekend.",
            "speaker": "Alice",
            "session_id": "session_1",
            "turn_id": "3",
            "timestamp": "2024-01-01",
        }
    )
    result = memory.query(
        "Which memories are most relevant to a person's preferences, plans, or relationships?"
    )

    assert "log" in memory._runtime._state
    assert "helloworld_tests" in memory._runtime._state
    view = memory._runtime._state["helloworld_tests"]
    assert set(view.columns) == {"memory_summary"}
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_sem_map_memory_real_lotus_e2e() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    if os.getenv("AGENT_MEMORY_RUN_LOTUS_E2E") != "1":
        pytest.skip("set AGENT_MEMORY_RUN_LOTUS_E2E=1 to run real LOTUS e2e")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for real LOTUS e2e")

    class SemMapMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        message_labels = log.sem_map(
            output_cols={
                "label": "Short message label.",
                "summary": "Concise message summary.",
            },
            instruction="Produce a short label and concise summary for {message}.",
        ).select(["message", "label", "summary"])

    memory = SemMapMemory(adapter=LotusAdapter())

    memory.add("Hello, hope you are doing well.")
    memory.add("green sleep quickly because table")

    view = memory._runtime._state["message_labels"]
    assert set(view.columns) == {"message", "label", "summary"}
    assert len(view) == 2
    assert view["label"].notna().all()
    assert view["summary"].notna().all()


def test_sem_join_real_lotus_e2e() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    if os.getenv("AGENT_MEMORY_RUN_LOTUS_E2E") != "1":
        pytest.skip("set AGENT_MEMORY_RUN_LOTUS_E2E=1 to run real LOTUS e2e")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for real LOTUS e2e")

    left = Relation(QueryExpr(op="materialized_view", params={"name": "left"}))
    right = Relation(QueryExpr(op="materialized_view", params={"name": "right"}))
    query = left.sem_join(
        right,
        instruction="{message:left} belongs to {category:right}.",
        how="left",
    ).expr
    inputs = {
        "left": pd.DataFrame(
            {
                "message": [
                    "Alice prefers concise design documents.",
                    "Bob is planning a cooking class.",
                ]
            }
        ),
        "right": pd.DataFrame({"category": ["documentation preference"]}),
    }

    result = LotusAdapter().execute(query, inputs)

    assert "message" in result.columns
    assert "category" in result.columns
    assert len(result) >= len(inputs["left"])


def test_sem_groupby_sem_agg_real_lotus_e2e() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    if os.getenv("AGENT_MEMORY_RUN_LOTUS_E2E") != "1":
        pytest.skip("set AGENT_MEMORY_RUN_LOTUS_E2E=1 to run real LOTUS e2e")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for real LOTUS e2e")

    source = Relation(QueryExpr(op="materialized_view", params={"name": "source"}))
    query = (
        source.sem_groupby(
            input_cols=["name", "description"],
            instruction="Rows refer to the same durable topic when their {name} and {description} describe the same memory.",
        )
        .sem_agg(
            input_cols=["name", "description", "body"],
            output_cols={
                "body": "Merged durable memory body.",
            },
            instruction="Merge grouped memory candidates into one concise durable memory row.",
        )
        .expr
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "name": ["docs", "documentation", "cooking"],
                "description": [
                    "Preference for concise design docs.",
                    "Preference for short architecture documents.",
                    "Interest in cooking classes.",
                ],
                "body": [
                    "Alice prefers concise design documents.",
                    "Alice likes short architecture docs.",
                    "Bob wants to attend a cooking class.",
                ],
            }
        )
    }

    result = LotusAdapter().execute(query, inputs)

    assert list(result.columns) == ["body"]
    assert not result.empty


def test_claude_memory_query_binds_user_query_placeholder() -> None:
    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[QueryExpr, dict[str, pd.DataFrame]]] = []

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> str:
            self.calls.append((query, inputs))
            return "ranked"

    adapter = RecordingAdapter()
    memory = am.ClaudeMemory(adapter=adapter)
    topics = pd.DataFrame(
        {
            "name": ["docs"],
            "description": ["Design docs."],
            "type": ["reference"],
            "body": ["Use the design docs for architecture context."],
        }
    )
    memory._runtime._state["topics"] = topics

    assert memory.query("design docs") == "ranked"

    query, inputs = adapter.calls[0]
    assert query.op == "select"
    join_query = query.inputs[0]
    assert join_query.op == "join"
    topk_query, topics_query = join_query.inputs
    assert topk_query.op == "sem_topk"
    assert topk_query.params["instruction"] == "design docs"
    assert topk_query.params["k"] == 5
    assert topk_query.inputs[0] == QueryExpr(
        op="select",
        inputs=(
            QueryExpr(
                op="materialized_view",
                params={"name": "topics"},
            ),
        ),
        params={"columns": ("name", "description", "type")},
    )
    assert topics_query == QueryExpr(
        op="materialized_view",
        params={"name": "topics"},
    )
    assert inputs["topics"].equals(topics)


def test_memory_subclass_rejects_query_override() -> None:
    with pytest.raises(TypeError, match="retrieval_query"):

        class PlainQueryMemory(am.Memory):
            log = am.Log()

            def query(self, query: str) -> dict[str, str]:
                return {"query": query}


def test_memory_subclass_rejects_invalid_retrieval_query_at_definition_time() -> None:
    with pytest.raises(TypeError, match="retrieval_query must be a Relation"):

        class InvalidRetrievalQueryMemory(am.Memory):
            log = am.Log()
            retrieval_query = "design docs"

    with pytest.raises(TypeError, match="retrieval_query must be a Relation"):

        class EmptyRetrievalQueryMemory(am.Memory):
            log = am.Log()
            retrieval_query = None


def test_base_memory_query_requires_retrieval_query() -> None:
    class MinimalMemory(am.Memory):
        log = am.Log()

    memory = MinimalMemory()

    with pytest.raises(NotImplementedError, match="retrieval query"):
        memory.query("design docs")


def test_runtime_owns_empty_materialized_state_placeholder() -> None:
    memory = am.ClaudeMemory()

    assert memory._runtime._state == {}
    assert not hasattr(memory._runtime, "query")
    assert not hasattr(memory._runtime, "execute_query")
    with pytest.raises(KeyError, match="Missing adapter input 'topics'"):
        memory.query("design docs")


def test_runtime_query_output_columns_support_materialized_view_refs() -> None:
    memory = am.ClaudeMemory()
    query = QueryExpr(op="materialized_view", params={"name": "catalog"})

    assert memory._runtime._query_output_columns(query) == [
        "catalog_title",
        "name",
        "hook",
    ]


def test_runtime_query_output_columns_pass_through_groupby_and_topk() -> None:
    memory = HelloWorldTestMemory()
    view_query = HelloWorldTestMemory.spec().views["helloworld_tests"].query
    groupby_query = QueryExpr(
        op="sem_groupby",
        inputs=(view_query,),
        params={"input_cols": ("memory_summary",), "instruction": "same memory"},
    )
    topk_query = QueryExpr(
        op="sem_topk",
        inputs=(view_query,),
        params={"instruction": "plans", "k": 2},
    )

    assert memory._runtime._query_output_columns(groupby_query) == ["memory_summary"]
    assert memory._runtime._query_output_columns(topk_query) == ["memory_summary"]


def test_runtime_query_output_columns_infers_relational_join_suffixes() -> None:
    memory = am.ClaudeMemory()
    query = QueryExpr(
        op="join",
        inputs=(
            QueryExpr(
                op="materialized_view",
                params={"name": "topics"},
            ),
            QueryExpr(
                op="select",
                inputs=(
                    QueryExpr(
                        op="materialized_view",
                        params={"name": "topics"},
                    ),
                ),
                params={"columns": ("name", "description")},
            ),
        ),
        params={"on": ("name",), "how": "inner"},
    )

    assert memory._runtime._query_output_columns(query) == [
        "name",
        "description:left",
        "type",
        "body",
        "description:right",
    ]


def test_runtime_query_output_columns_reject_unknown_ops() -> None:
    memory = HelloWorldTestMemory()

    with pytest.raises(NotImplementedError, match="Cannot infer output columns"):
        memory._runtime._query_output_columns(QueryExpr(op="unknown"))


def test_relation_is_not_top_level_public_api() -> None:
    assert not hasattr(am, "Relation")
