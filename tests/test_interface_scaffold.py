"""Tests for the v0.0 interface layer."""

from __future__ import annotations

from typing import Any

import pytest

import agent_memory as am
from agent_memory.logical import ColumnSpec, MemorySpec, MemoryView, QueryExpr
from agent_memory.planner import DifferentialQueryPlanner
from agent_memory.planner.rules import RewriteRule
from agent_memory.relation import GroupedRelation, Relation


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
        instruction="Find candidates related to the same durable memory topic.",
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

    filtered = log.sem_filter(instruction="Keep durable memory facts.")
    joined = log.sem_join(log, instruction="Match related rows.", how="inner")
    topk = log.sem_topk("Find relevant rows.", 3)

    assert filtered.expr.op == "sem_filter"
    assert joined.expr.op == "sem_join"
    assert topk.expr.op == "sem_topk"
    assert topk.expr.params["instruction"] == "Find relevant rows."
    assert topk.expr.params["k"] == 3

    with pytest.raises(TypeError):
        log.sem_filter("Keep durable memory facts.")
    with pytest.raises(TypeError):
        log.sem_join(log, "Match related rows.")


def test_rewrite_rule_supports_pattern_matching() -> None:
    grouped = am.Log().sem_groupby(
        key=["topic_name"],
        instruction="Find candidates related to the same durable memory topic.",
    )
    aggregated = grouped.sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols=["topic_name", "topic_content"],
        instruction="Merge topic rows.",
    )
    view = am.ClaudeMemory.spec().views["topics"]

    class GroupedAggRule:
        def matches(self, query: QueryExpr, view: MemoryView) -> bool:
            return query.op == "sem_agg" and query.inputs[0].op == "sem_groupby"

        def rewrite(self, query: QueryExpr, view: MemoryView) -> QueryExpr:
            return QueryExpr(op="rewritten", inputs=(query,))

    rule: RewriteRule = GroupedAggRule()

    assert rule.matches(aggregated.expr, view)
    rewritten = rule.rewrite(aggregated.expr, view)
    assert rewritten.op == "rewritten"
    assert rewritten.inputs == (aggregated.expr,)


def test_differential_query_planner_boundary_is_explicitly_unimplemented() -> None:
    planner = DifferentialQueryPlanner()
    view = am.ClaudeMemory.spec().views["topics"]

    with pytest.raises(NotImplementedError, match="Differential query planning"):
        planner.differentiate(view)


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
def test_add_inputs_are_explicitly_unimplemented(message: object) -> None:
    memory = am.ClaudeMemory()

    with pytest.raises(NotImplementedError, match="add/log maintenance"):
        memory.add(message)


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
    with pytest.raises(NotImplementedError, match="query plan execution"):
        memory._runtime.execute_query(memory.catalog.sem_topk("design docs", 5))


def test_relation_is_not_top_level_public_api() -> None:
    assert not hasattr(am, "Relation")
