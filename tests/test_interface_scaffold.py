"""Tests for the v0.0 interface layer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.relational import (
    execute_concat,
    execute_drop_duplicates,
    execute_subtract,
    execute_union,
)
from agent_memory.adapters.lotus.sem_agg import (
    GROUP_ID_COLUMN,
    aggregate_context_frame,
    aggregate_groups,
    aggregate_input_columns,
    apply_structured_aggregate_outputs,
    execute_sem_agg,
)
from agent_memory.adapters.lotus.sem_map import (
    apply_sem_map_output,
    apply_structured_map_outputs,
    native_sem_map_kwargs,
    normalize_strategy,
    parse_structured_map_json,
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
)
from agent_memory.adapters.lotus.sem_join import (
    assemble_join_frame,
    cascade_args_from_mapping,
    join_series,
)
from agent_memory.adapters.lotus.sem_topk import execute_sem_topk, topk_instruction
from agent_memory.adapters.lotus.structured import (
    STRUCTURED_RESERVED_MODEL_KWARGS,
    StructuredGenerationResult,
    parse_structured_object_json,
    validate_model_kwargs,
)
from agent_memory.logical import ColumnSpec, MemorySpec, QueryExpr
from agent_memory.planner import DifferentialQueryPlanner
from agent_memory.planner.rules import DifferentialRules
from agent_memory.relation import GroupedRelation, Relation
from examples.helloworld.helloworld_smoke import _flatten_locomo_rows


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
    assert "log" not in spec.views


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

    with pytest.raises(NotImplementedError, match="No differential rule"):
        DifferentialRules().differentiate(aggregated.expr)


def test_differential_rules_reject_malformed_unary_expr() -> None:
    query = QueryExpr(op="sem_filter")

    with pytest.raises(ValueError, match="expects exactly one input"):
        DifferentialRules().differentiate(query)


def test_differential_rules_preserve_sem_filter_instruction() -> None:
    class FilterMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        helloworld_tests = log.sem_filter(
            instruction="{message} is a coherent sentence."
        )

    view = FilterMemory.spec().views["helloworld_tests"]
    differentiated = DifferentialRules().differentiate(view.query)

    assert differentiated == view.query
    assert differentiated.params["instruction"] == "{message} is a coherent sentence."


def test_differential_query_planner_rejects_unsupported_operators() -> None:
    planner = DifferentialQueryPlanner()
    view = am.ClaudeMemory.spec().views["topics"]

    with pytest.raises(NotImplementedError, match="No differential rule"):
        planner.differentiate(view)


def test_differential_query_planner_supports_sem_filter_views() -> None:
    class FilterMemory(am.Memory):
        log = am.Log({"message": "Raw input message."})
        helloworld_tests = log.sem_filter(
            instruction="{message} is a coherent sentence."
        )

    view = FilterMemory.spec().views["helloworld_tests"]

    assert DifferentialQueryPlanner().differentiate(view) == view.query


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

    assert DifferentialQueryPlanner().differentiate(view) == view.query


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


@pytest.mark.parametrize(
    "message",
    [
        "Please remember that I prefer concise docs.",
        am.Message(content="Please remember that I prefer concise docs."),
        {"message": "Please remember that I prefer concise docs."},
    ],
)
def test_claude_add_inputs_reach_unsupported_differential_rule(message: object) -> None:
    memory = am.ClaudeMemory()

    with pytest.raises(NotImplementedError, match="No differential rule"):
        memory.add(message)


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

    def query(self, query: str) -> Relation:
        return self.helloworld_tests.sem_topk(query, 2)


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

    rows = _flatten_locomo_rows(dataset, sample_limit=1, turn_limit=1)

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


def test_sem_flat_map_parses_and_explodes_json_array_outputs() -> None:
    output_cols = (
        ColumnSpec("topic", "Candidate topic."),
        ColumnSpec("summary", "Candidate summary."),
    )
    parsed = [
        parse_structured_flat_map_json(
            '[{"topic": "docs", "summary": "Prefers concise docs."}, {"topic": "meetings", "summary": "Plans a meeting."}]',
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
    parsed = [parse_structured_flat_map_json("[]", output_cols)]
    source = pd.DataFrame({"message": ["nothing durable"]})

    result = apply_flat_map_outputs(source, parsed, output_cols)

    assert list(result.columns) == ["message", "topic"]
    assert result.empty


def test_sem_flat_map_rejects_invalid_json_shapes() -> None:
    output_cols = (ColumnSpec("topic"),)

    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_flat_map_json("not json", output_cols)
    with pytest.raises(ValueError, match="non-array JSON"):
        parse_structured_flat_map_json('{"topic": "docs"}', output_cols)
    with pytest.raises(ValueError, match="not an object"):
        parse_structured_flat_map_json('["docs"]', output_cols)
    with pytest.raises(ValueError, match="missing required keys"):
        parse_structured_flat_map_json('[{"summary": "docs"}]', output_cols)


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

    assert list(result["message"]) == ["hello", "world"]


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


def test_sem_agg_resolves_input_columns_and_applies_structured_outputs() -> None:
    source = pd.DataFrame(
        {
            "topic": ["docs", "meetings"],
            "body": ["Prefers concise docs.", "Plans weekly sync."],
            GROUP_ID_COLUMN: [0, 0],
        }
    )
    source.attrs["agent_memory_groupby_input_cols"] = ("topic",)
    output_cols = (
        ColumnSpec("topic", "Canonical topic."),
        ColumnSpec("body", "Merged body."),
    )

    assert aggregate_input_columns(source, None) == ("body",)

    result = apply_structured_aggregate_outputs(
        [{"topic": "docs", "body": "Prefers concise docs and weekly syncs."}],
        output_cols,
    )

    assert list(result.columns) == ["topic", "body"]
    assert result.loc[0, "topic"] == "docs"
    assert result.loc[0, "body"] == "Prefers concise docs and weekly syncs."


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


def test_sem_agg_context_frame_has_one_row_per_group() -> None:
    source = pd.DataFrame(
        {
            "body": ["Prefers concise docs.", "Likes short docs.", "Plans cooking."],
            GROUP_ID_COLUMN: [0, 0, 1],
        }
    )

    context = aggregate_context_frame(source, ("body",))

    assert len(context) == 2
    assert "Prefers concise docs." in context.loc[0, "context"]
    assert "Likes short docs." in context.loc[0, "context"]
    assert "Plans cooking." in context.loc[1, "context"]


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

    class Executor:
        calls: list[str] = []

        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            assert list(self.frame.columns) == ["context"]
            assert len(self.frame) == 1
            assert kwargs["model_kwargs"]["max_tokens"] >= 1024
            Executor.calls.append(self.frame.loc[0, "context"])
            index = len(Executor.calls) - 1
            parsed = (
                {"topic": "docs", "body": "doc one and doc two"},
                {"topic": "cooking", "body": "cooking"},
            )[index]
            return StructuredGenerationResult(
                raw_outputs=("{}",),
                parsed_outputs=(parsed,),
                explanations=(None,),
            )

    monkeypatch.setattr(sem_agg_module, "StructuredLMExecutor", Executor)
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
    assert len(Executor.calls) == 2


def test_sem_agg_whole_multi_output_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    class Executor:
        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            assert len(self.frame) == 1
            assert kwargs["model_kwargs"]["max_tokens"] >= 1024
            return StructuredGenerationResult(
                raw_outputs=("{}",),
                parsed_outputs=(
                    {"topic": "docs", "body": "doc one and doc two"},
                ),
                explanations=(None,),
            )

    monkeypatch.setattr(sem_agg_module, "StructuredLMExecutor", Executor)
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


def test_sem_agg_multi_output_default_uses_single_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig()

        def configure(self) -> None:
            pass

    class Executor:
        calls: list[str] = []

        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            context = self.frame.loc[0, "context"]
            Executor.calls.append(context)
            return StructuredGenerationResult(
                raw_outputs=("{}",),
                parsed_outputs=(
                    {"topic": "docs", "body": "all rows"},
                ),
                explanations=(None,),
            )

    monkeypatch.setattr(sem_agg_module, "StructuredLMExecutor", Executor)
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
                "body": [f"row {index}" for index in range(5)],
                GROUP_ID_COLUMN: [0, 0, 0, 0, 0],
            }
        )
    }

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert len(Executor.calls) == 1
    assert all(f"row {index}" in Executor.calls[0] for index in range(5))
    assert result.to_dict("records") == [{"topic": "docs", "body": "all rows"}]


def test_sem_agg_multi_output_lotus_hierarchical_strategy_uses_native_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    class Context:
        config = LotusExecutionConfig(sem_agg_structured_strategy="lotus_hierarchical")

        def configure(self) -> None:
            pass

    class Executor:
        calls: list[tuple[str, str]] = []

        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        def __call__(self, **kwargs: Any) -> StructuredGenerationResult:
            context = self.frame.loc[0, "context"]
            instruction = kwargs["instruction"]
            Executor.calls.append((instruction, context))
            index = len(Executor.calls)
            return StructuredGenerationResult(
                raw_outputs=("{}",),
                parsed_outputs=(
                    {"topic": f"final-{index}", "body": f"summary-{index}"},
                ),
                explanations=(None,),
            )

    native_calls: list[tuple[str, tuple[str, ...], list[str]]] = []

    def execute_native_sem_agg_group(
        query: QueryExpr,
        group: pd.DataFrame,
        input_cols: tuple[str, ...],
        config: LotusExecutionConfig | None = None,
    ) -> str:
        native_calls.append(
            (
                str(query.params["instruction"]),
                tuple(input_cols),
                list(group["body"]),
            )
        )
        return "LOTUS native hierarchical intermediate summary."

    monkeypatch.setattr(sem_agg_module, "StructuredLMExecutor", Executor)
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
            "output_cols": (ColumnSpec("topic"), ColumnSpec("body")),
            "instruction": "Merge {body}.",
        },
    )
    inputs = {
        "source": pd.DataFrame(
            {
                "body": [
                    "alpha beta gamma delta epsilon",
                    "zeta eta theta iota kappa",
                    "lambda mu nu xi omicron",
                    "pi rho sigma tau upsilon",
                ]
            }
        )
    }

    result = execute_sem_agg(query, inputs, LotusAdapter().execute, Context())

    assert len(native_calls) == 1
    assert "final output fields later: topic, body" in native_calls[0][0]
    assert native_calls[0][1] == ("body",)
    assert native_calls[0][2] == [
        "alpha beta gamma delta epsilon",
        "zeta eta theta iota kappa",
        "lambda mu nu xi omicron",
        "pi rho sigma tau upsilon",
    ]
    assert Executor.calls == [
        (
            "Merge {body}.\n\nUse the LOTUS hierarchical aggregate in {context} and produce one aggregate object.",
            "LOTUS native hierarchical intermediate summary.",
        )
    ]
    assert result.to_dict("records") == [{"topic": "final-1", "body": "summary-1"}]


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


def test_claude_memory_query_is_policy_owned_placeholder() -> None:
    memory = am.ClaudeMemory()
    captured: dict[str, Any] = {}

    def execute_query(plan: Relation) -> Any:
        captured["plan"] = plan
        raise NotImplementedError("query plan execution")

    memory._runtime.execute_query = execute_query

    with pytest.raises(NotImplementedError, match="query plan execution"):
        memory.query("design docs")

    plan = captured["plan"]
    assert isinstance(plan, Relation)
    assert plan.expr.op == "sem_topk"
    assert plan.expr.params["instruction"] == "design docs"
    assert plan.expr.params["k"] == 5


def test_query_wrapper_returns_plain_python_objects_without_runtime() -> None:
    class PlainQueryMemory(am.Memory):
        log = am.Log()

        def query(self, query: str) -> dict[str, str]:
            return {"query": query}

    memory = PlainQueryMemory()

    def execute_query(plan: Relation) -> Any:
        raise AssertionError("plain query return should not execute a Relation plan")

    memory._runtime.execute_query = execute_query

    assert memory.query("design docs") == {"query": "design docs"}


def test_base_memory_query_requires_policy_override() -> None:
    class MinimalMemory(am.Memory):
        log = am.Log()

    memory = MinimalMemory()

    with pytest.raises(NotImplementedError, match="must override query"):
        memory.query("design docs")


def test_runtime_owns_empty_materialized_state_placeholder() -> None:
    memory = am.ClaudeMemory()

    assert memory._runtime._state == {}
    assert not hasattr(memory._runtime, "query")
    with pytest.raises(KeyError, match="Missing adapter input 'catalog'"):
        memory._runtime.execute_query(memory.catalog.sem_topk("design docs", 5))


def test_relation_is_not_top_level_public_api() -> None:
    assert not hasattr(am, "Relation")
