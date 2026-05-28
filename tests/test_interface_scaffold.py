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
    sem_agg_expr = topics_expr.inputs[0]
    assert sem_agg_expr.op == "sem_agg"
    sem_groupby_expr = sem_agg_expr.inputs[0]
    assert sem_groupby_expr.op == "sem_groupby"
    assert sem_groupby_expr.params["key"] == ("topic_name",)
    projection_expr = sem_groupby_expr.inputs[0]
    assert projection_expr.op == "select"
    sem_flat_map_expr = projection_expr.inputs[0]
    assert sem_flat_map_expr.op == "sem_flat_map"
    assert sem_flat_map_expr.params["input_cols"] is None


def test_grouped_relation_sem_agg_returns_normal_relation() -> None:
    grouped = am.Log().sem_groupby(
        key=["topic_name"],
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


def test_differential_rules_reject_unsupported_operators() -> None:
    grouped = am.Log().sem_groupby(
        key=["topic_name"],
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
    sem_map_expr = catalog_expr.inputs[0]
    assert sem_map_expr.op == "sem_map"
    assert sem_map_expr.params["input_cols"] == ("topic_name", "topic_content")


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
    adapter = LotusAdapter()
    frame = pd.DataFrame({"message": ["hello"]})

    assert (
        adapter._topk_instruction(frame, "friendly greetings")
        == "{message} is relevant to: friendly greetings"
    )


def test_lotus_adapter_keeps_column_aware_topk_instruction() -> None:
    adapter = LotusAdapter()
    frame = pd.DataFrame({"message": ["hello"]})

    assert (
        adapter._topk_instruction(frame, "{message} is a friendly greeting")
        == "{message} is a friendly greeting"
    )


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
    adapter = LotusAdapter()
    frame = pd.DataFrame(
        {
            "message": ["hello"],
            "_agent_memory_map": ["existing"],
            "_agent_memory_map_1": ["existing"],
        }
    )

    assert adapter._temporary_map_column(frame) == "_agent_memory_map_2"


def test_lotus_adapter_applies_single_output_sem_map_result() -> None:
    adapter = LotusAdapter()
    source = pd.DataFrame({"message": ["hello", "bye"]})
    mapped = pd.DataFrame(
        {
            "message": ["hello", "bye"],
            "_agent_memory_map": ["A greeting.", "A goodbye."],
        }
    )

    result = adapter._apply_sem_map_output(
        source,
        mapped,
        "_agent_memory_map",
        ColumnSpec("summary", "Short message summary."),
    )

    assert list(result.columns) == ["message", "summary"]
    assert list(result["summary"]) == ["A greeting.", "A goodbye."]


def test_lotus_adapter_rejects_multi_output_sem_map() -> None:
    adapter = LotusAdapter()
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
        adapter._single_output_column(query)


def test_lotus_adapter_accepts_single_output_sem_map() -> None:
    adapter = LotusAdapter()
    query = QueryExpr(
        op="sem_map",
        params={"output_cols": (ColumnSpec("summary", "Short message summary."),)},
    )

    assert adapter._single_output_column(query).name == "summary"


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
            output_cols={"summary": "Concise message summary."},
            instruction="Produce a concise summary for {message}.",
        ).select(["message", "summary"])

    memory = SemMapMemory(adapter=LotusAdapter())

    memory.add("Hello, hope you are doing well.")
    memory.add("green sleep quickly because table")

    view = memory._runtime._state["message_labels"]
    assert set(view.columns) == {"message", "summary"}
    assert len(view) == 2
    assert view["summary"].notna().all()


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
