"""Tests for the v0.0 interface layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    DEFAULT_STRUCTURED_PARSE_RETRIES,
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.relational import (
    execute_concat,
    execute_drop_duplicates,
    execute_join,
    execute_subtract,
    execute_union,
)
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
    execute_sem_filter,
    native_sem_filter_kwargs,
)
import agent_memory.adapters.lotus.sem_groupby as sem_groupby_module
from agent_memory.adapters.lotus.sem_groupby import (
    assign_declared_labels,
    assign_semantic_group_ids,
    evaluate_group_matches,
)
from agent_memory.adapters.lotus.sem_join import (
    assemble_join_frame,
    cascade_args_from_mapping,
    evaluate_semantic_join,
    execute_sem_join,
    join_series,
    renamed_columns,
)
from agent_memory.adapters.lotus.sem_topk import execute_sem_topk, topk_instruction
import agent_memory.adapters.lotus.structured as structured_module
from agent_memory.adapters.lotus.structured import (
    STRUCTURED_RESERVED_MODEL_KWARGS,
    StructuredLMExecutor,
    StructuredGenerationResult,
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
from agent_memory.logical import ColumnSpec, MemorySpec, QueryExpr, UserQuery
from agent_memory.planner import DifferentialInstructionRewriter, DifferentialQueryPlanner
from agent_memory.planner.rules import DifferentialRules
from agent_memory.relation import GroupedRelation, Relation
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
        "metadata",
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
    assert sem_flat_map_expr.params["input_cols"] is None
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
    differentiated = DifferentialQueryPlanner().differentiate(view)

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
    differentiated = DifferentialQueryPlanner().differentiate(view)

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
    differentiated = DifferentialQueryPlanner().differentiate(view)

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
    differentiated = DifferentialQueryPlanner().differentiate(view)

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
    differentiated = DifferentialQueryPlanner().differentiate(view)

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
        DifferentialQueryPlanner().differentiate(view)


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


def test_differential_query_planner_builds_claude_topics_full_next_view() -> None:
    view = am.ClaudeMemory.spec().views["topics"]

    differentiated = DifferentialQueryPlanner().differentiate(view)

    assert differentiated.op == "select"
    assert differentiated.params["columns"] == ("name", "description", "type", "body")
    sem_map_expr = differentiated.inputs[0]
    assert sem_map_expr.op == "sem_map"
    assert tuple(col.name for col in sem_map_expr.params["output_cols"]) == (
        "name",
        "description",
        "type",
        "body",
    )
    sem_join_expr = sem_map_expr.inputs[0]
    assert sem_join_expr.op == "sem_join"
    assert sem_join_expr.params["how"] == "outer"
    join_instruction = sem_join_expr.params["instruction"]
    assert "{name:left} and {name:right}" in join_instruction
    assert "{description:left} and {description:right}" in join_instruction
    assert "{type:left} and {type:right}" in join_instruction
    assert "{name}," not in join_instruction
    _assert_materialized_view(
        sem_join_expr.inputs[1],
        name="topics",
        columns=("name", "description", "type", "body"),
    )
    map_instruction = sem_map_expr.params["instruction"]
    assert "{name:left} and {name:right}" in map_instruction
    assert "{description:left} and {description:right}" in map_instruction
    assert "{type:left} and {type:right}" in map_instruction
    assert "{body:left} and {body:right}" in map_instruction
    changed_aggregate = sem_join_expr.inputs[0]
    assert changed_aggregate.op == "sem_agg"
    assert changed_aggregate.inputs[0].op == "sem_groupby"


def test_claude_topics_join_instruction_lowers_to_composite_records() -> None:
    view = am.ClaudeMemory.spec().views["topics"]
    differentiated = DifferentialQueryPlanner().differentiate(view)
    sem_map_expr = differentiated.inputs[0]
    sem_join_expr = sem_map_expr.inputs[0]

    left = pd.DataFrame(
        {
            "name": ["adoption_goal"],
            "description": ["Caroline wants to adopt children."],
            "type": ["user"],
            "body": ["Caroline is researching adoption agencies."],
        }
    )
    right = pd.DataFrame(
        {
            "name": ["caroline_adoption_journey"],
            "description": ["Caroline is pursuing single-parent adoption."],
            "type": ["user"],
            "body": ["Caroline values LGBTQ+ inclusive adoption agencies."],
        }
    )

    left_series, right_series, left_label, right_label, instruction = join_series(
        left,
        right,
        str(sem_join_expr.params["instruction"]),
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


def test_differential_query_planner_recomputes_views_from_materialized_dependencies() -> None:
    spec = am.ClaudeMemory.spec()
    catalog = spec.views["catalog"]

    differentiated = DifferentialQueryPlanner().differentiate(
        catalog,
        views=spec.views,
    )

    assert differentiated.op == "select"
    sem_map_expr = differentiated.inputs[0]
    assert sem_map_expr.op == "sem_map"
    _assert_materialized_view(
        sem_map_expr.inputs[0],
        name="topics",
    )


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
    sem_map_expr = catalog_expr.inputs[0]
    assert sem_map_expr.op == "sem_map"
    assert sem_map_expr.params["input_cols"] == (
        "name",
        "description",
        "type",
        "body",
    )
    assert tuple(col.name for col in sem_map_expr.params["output_cols"]) == (
        "catalog_title",
        "hook",
    )

    catalog_instruction = " ".join(sem_map_expr.params["instruction"].split())
    assert "MEMORY.md" in catalog_instruction
    assert "index, not a memory" in catalog_instruction
    assert "one line" in catalog_instruction
    assert "150 characters" in catalog_instruction
    assert "{catalog_title}" in catalog_instruction
    assert "{hook}" in catalog_instruction
    assert "Do not generate filesystem paths" in catalog_instruction


def test_differentiated_policy_compiles_views_and_retrieval_templates() -> None:
    policy = am.ClaudeMemory.differentiate_policy()

    assert policy.spec is am.ClaudeMemory.spec()
    assert policy.view_execution_order == ("topics", "catalog")
    assert policy.view_dependencies == {
        "topics": (),
        "catalog": ("topics",),
    }
    assert sorted(policy.view_queries) == ["catalog", "topics"]

    retrieval_query = policy.retrieval_queries["default"]
    assert retrieval_query.op == "select"
    assert retrieval_query.params["columns"] == (
        "name",
        "description:right",
        "type:right",
        "body",
    )
    join_query = retrieval_query.inputs[0]
    assert join_query.op == "join"
    assert join_query.params == {"on": ("name",), "how": "inner"}
    topk_query, topics_query = join_query.inputs
    _assert_materialized_view(topics_query, name="topics")
    assert topk_query.op == "sem_topk"
    assert topk_query.params["instruction"] == UserQuery()
    assert topk_query.params["k"] == 5
    manifest_query = topk_query.inputs[0]
    assert manifest_query.op == "select"
    assert manifest_query.params["columns"] == ("name", "description", "type")
    _assert_materialized_view(manifest_query.inputs[0], name="topics")


@pytest.mark.parametrize(
    "message",
    [
        "Please remember that I prefer concise docs.",
        am.Message(content="Please remember that I prefer concise docs."),
        {"message": "Please remember that I prefer concise docs."},
    ],
)
def test_claude_add_executes_differentiated_queries(message: object) -> None:
    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[QueryExpr, dict[str, pd.DataFrame]]] = []

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            self.calls.append((query, inputs))
            if query.op == "select":
                columns = [str(column) for column in query.params["columns"]]
            else:
                columns = ["value"]
            return pd.DataFrame(
                [{column: f"{column}-{len(self.calls)}" for column in columns}]
            )

    adapter = RecordingAdapter()
    memory = am.ClaudeMemory(adapter=adapter)

    memory.add(message)

    assert [query.op for query, _ in adapter.calls] == ["select", "select"]
    topics_query, topics_inputs = adapter.calls[0]
    catalog_query, catalog_inputs = adapter.calls[1]

    assert topics_query.params["columns"] == ("name", "description", "type", "body")
    assert "log" in topics_inputs
    assert list(topics_inputs["topics"].columns) == [
        "name",
        "description",
        "type",
        "body",
    ]
    assert topics_inputs["topics"].empty

    assert catalog_query.params["columns"] == ("catalog_title", "name", "hook")
    assert catalog_query.inputs[0].inputs[0] == QueryExpr(
        op="materialized_view",
        params={"name": "topics"},
    )
    assert catalog_inputs["topics"].equals(memory._runtime._state["topics"])

    assert list(memory._runtime._state["topics"].columns) == [
        "name",
        "description",
        "type",
        "body",
    ]
    assert list(memory._runtime._state["catalog"].columns) == [
        "catalog_title",
        "name",
        "hook",
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
        ),
    )
    context.configure()

    assert captured["lm_kwargs"] == {
        "model": "deepseek/example",
        "max_batch_size": 4,
        "num_retries": 2,
        "timeout": 120,
        "rate_limit": 10,
    }
    assert isinstance(captured["configure_kwargs"]["lm"], FakeLM)


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
    assert config.semantic_trace_dir is None
    assert config.structured_parse_retries == DEFAULT_STRUCTURED_PARSE_RETRIES


def test_lotus_adapter_forwards_topk_lotus_options() -> None:
    class Source:
        columns = ["message"]

        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

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
        sem_topk_method="naive",
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
    query = QueryExpr(
        op="sem_map",
        params={
            "output_cols": (ColumnSpec("label", "Short label."),),
        },
    )
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
    )

    assert matches == [(0, 1)]
    assert captured == {
        "default": False,
        "progress_bar_desc": "Grouping comparisons",
    }


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

    assert aggregate_input_columns(source, None) == ("body",)
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
    ) -> str:
        structured_calls.append(
            (
                list(group["body"]),
                tuple(input_cols),
                tuple(column.name for column in output_cols),
            )
        )
        index = len(structured_calls) - 1
        return (
            '{"topic": "docs", "body": "doc one and doc two"}',
            '{"topic": "cooking", "body": "cooking"}',
        )[index]

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
    ) -> str:
        structured_calls.append(list(group["body"]))
        return '{"topic": "docs", "body": "doc one and doc two"}'

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
    ) -> str:
        return '{"topic": "docs", "body": "doc one and doc two"}'

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
        '{"topic": "docs", "body": "doc one and doc two"}'
    ]
    assert trace_artifact(tmp_path, event["parsed_output_path"]) == {
        "topic": "docs",
        "body": "doc one and doc two",
    }
    assert event["parse_error"] == ""


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
    ) -> str:
        return ""

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


def test_runtime_log_append_and_view_union_semantics() -> None:
    memory = HelloWorldTestMemory()
    current = pd.DataFrame({"message": ["hello"]})
    duplicate = pd.DataFrame({"message": ["hello"]})
    new_row = pd.DataFrame({"message": ["world"]})

    log_state = memory._runtime._append_log_frame(current, duplicate)
    view_state = memory._runtime._union_view_frame(current, duplicate)
    view_state = memory._runtime._union_view_frame(view_state, new_row)

    assert list(log_state["message"]) == ["hello", "hello"]
    assert list(view_state["message"]) == ["hello", "world"]


def test_runtime_executes_q_prime_and_stores_adapter_result() -> None:
    class FilterMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        helloworld_tests = log.sem_filter(
            instruction="{message} is a coherent sentence."
        ).select(["message"])

    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[QueryExpr, dict[str, pd.DataFrame]]] = []

        def execute(
            self,
            query: QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            self.calls.append((query, inputs))
            return pd.DataFrame({"message": ["next view row"]})

    adapter = RecordingAdapter()
    memory = FilterMemory(adapter=adapter)

    memory.add("hello")

    assert not hasattr(memory._runtime, "_planner")
    query, inputs = adapter.calls[0]
    assert query.op == "union"
    _assert_materialized_view(
        query.inputs[0],
        name="helloworld_tests",
        columns=("message",),
    )
    assert list(inputs["helloworld_tests"].columns) == ["message"]
    assert inputs["helloworld_tests"].empty
    assert memory._runtime._state["helloworld_tests"].to_dict("records") == [
        {"message": "next view row"}
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
    _assert_materialized_view(
        query.inputs[0].inputs[0].inputs[0],
        name="summary",
        columns=("summary",),
    )
    assert inputs["summary"].empty
    assert list(inputs["summary"].columns) == ["summary"]
    assert memory._runtime._state["summary"].to_dict("records") == [
        {"summary": "next aggregate"}
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
