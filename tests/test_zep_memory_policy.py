"""Tests for the Zep/Graphiti-style memory policy declaration."""

from __future__ import annotations

import agent_memory as am
from agent_memory.policy.aggregates import (
    ArrayAggregateSpec,
    MinAggregateSpec,
    SemanticAggregateSpec,
)
from agent_memory.policy.logical import QueryExpr
from agent_memory.memories import ZepMemory, ZepMemoryExtended
from agent_memory.policy.schema import output_columns


def _walk(query: QueryExpr) -> tuple[QueryExpr, ...]:
    """Return query and all nested input queries."""

    nodes = [query]
    for input_query in query.inputs:
        nodes.extend(_walk(input_query))
    return tuple(nodes)


def _ops(query: QueryExpr) -> tuple[str, ...]:
    """Return all operator names in a query tree."""

    return tuple(node.op for node in _walk(query))


def _join_hows(query: QueryExpr) -> tuple[str, ...]:
    """Return join modes used in a query tree."""

    return tuple(
        str(node.params.get("how", "inner"))
        for node in _walk(query)
        if node.op == "join"
    )


def test_zep_memory_exports_public_api() -> None:
    """ZepMemory is part of the public built-in policy surface."""

    assert am.ZepMemory is ZepMemory


def test_zep_memory_spec_exposes_only_baseline_public_views() -> None:
    """Only the Zep logical memory state is exposed as public views."""

    spec = ZepMemory.spec()

    assert tuple(spec.views) == ("episodes", "entities", "facts")
    assert tuple(spec.private_relations) == (
        "_windowed_episodes",
        "_extracted_entities",
        "_episode_entities",
        "_extracted_facts",
        "_deduplicated_facts",
        "_earlier_added_facts",
        "_later_added_facts",
        "_contradictory_fact_pairs",
        "_fact_invalidations",
        "_facts_with_invalidations",
    )
    assert "retrieval_query" not in spec.views


def test_zep_extended_policy_adds_communities_without_changing_core_queries() -> None:
    baseline = ZepMemory.spec()
    extended = ZepMemoryExtended.spec()

    assert am.ZepMemoryExtended is ZepMemoryExtended
    assert tuple(extended.views) == ("episodes", "entities", "facts", "communities")
    for name in ("episodes", "entities", "facts"):
        assert extended.views[name].query == baseline.views[name].query


def test_zep_episodes_view_uses_select_not_map() -> None:
    """Episodes are deterministic log projection, not a new map operator."""

    query = ZepMemory.spec().views["episodes"].query

    assert query.op == "select"
    assigned = query.inputs[0]
    assert assigned.op == "assign"
    assert assigned.inputs[0].op == "log"
    assert assigned.inputs[0].params["system_columns"] is True
    assert assigned.params["assignments"]["episode_id"]["name"] == "_row_id"
    assert assigned.params["assignments"]["created_at"]["name"] == "_added_at"
    assert assigned.params["assignments"]["add_seq"]["name"] == "_add_seq"
    assert "map" not in _ops(query)
    assert output_columns(query) == (
        "episode_id",
        "content",
        "role",
        "speaker",
        "reference_time",
        "source_description",
        "created_at",
        "add_seq",
    )


def test_zep_entities_view_uses_context_extraction_and_semantic_aggregation() -> None:
    """Entities follow over-context extraction into canonical semantic groups."""

    spec = ZepMemory.spec()
    query = spec.views["entities"].query
    ops = _ops(query)

    assert "over" in _ops(spec.private_relations["_windowed_episodes"])
    assert "array_agg" in _ops(spec.private_relations["_windowed_episodes"])
    extracted_entities = spec.private_relations["_extracted_entities"]
    assert extracted_entities.op == "select"
    assert sum(node.op == "sem_flat_map" for node in _walk(extracted_entities)) == 1
    assert "sem_flat_map" in ops
    assert "agg" in ops
    assert "sem_groupby" in ops
    assert output_columns(query) == (
        "entity_id",
        "name",
        "entity_type",
        "summary",
        "mentions",
    )
    extraction = next(node for node in _walk(query) if node.op == "sem_flat_map")
    assert extraction.params["ordinal_col"] == "entity_ordinal"
    assert tuple(column.name for column in extraction.params["output_cols"]) == (
        "name",
    )
    entity_type_assignment = next(
        node
        for node in _walk(extracted_entities)
        if node.op == "assign" and "entity_type" in node.params["assignments"]
    )
    assert entity_type_assignment.params["assignments"]["entity_type"] == {
        "kind": "literal",
        "value": "Entity",
    }
    assert "Always extract the speaker" in extraction.params["instruction"]
    assert "When in doubt, do not extract" in extraction.params["instruction"]
    grouped_agg = next(node for node in _walk(query) if node.op == "agg")
    specs = grouped_agg.params["aggregates"]
    entity_id = next(
        specification
        for specification in specs
        if isinstance(specification, MinAggregateSpec)
        and specification.output_col == "entity_id"
    )
    assert entity_id.columns == ("add_seq", "entity_ordinal")
    semantic = next(
        specification
        for specification in specs
        if isinstance(specification, SemanticAggregateSpec)
    )
    assert tuple(column.name for column in semantic.output_cols) == (
        "name",
        "summary",
    )
    assert semantic.input_cols == ("name", "content", "add_seq")
    assert "add_seq" in semantic.instruction
    final_entity_type_assignment = query.inputs[0]
    assert final_entity_type_assignment.op == "assign"
    assert final_entity_type_assignment.params["assignments"]["entity_type"] == {
        "kind": "literal",
        "value": "Entity",
    }


def test_zep_episode_entity_bridge_uses_deterministic_explode_and_unnest() -> None:
    """Episode/entity mention links are reconstructed without semantic re-matching."""

    query = ZepMemory.spec().private_relations["_episode_entities"]
    ops = _ops(query)

    assert query.op == "select"
    assert "explode" in ops
    assert "unnest" in ops
    assert "sem_join" not in ops
    assert output_columns(query) == (
        "episode_id",
        "entity_ordinal",
        "entity_id",
    )


def test_zep_facts_view_contains_temporal_self_join_paths() -> None:
    """Facts canonicalize duplicates and preserve contradiction lifecycle rows."""

    spec = ZepMemory.spec()
    facts = spec.views["facts"].query
    deduplicated = spec.private_relations["_deduplicated_facts"]
    contradiction_pairs = spec.private_relations["_contradictory_fact_pairs"]
    invalidations = spec.private_relations["_fact_invalidations"]
    facts_with_invalidations = spec.private_relations["_facts_with_invalidations"]

    assert facts.op == "select"
    extracted = spec.private_relations["_extracted_facts"]
    assert "sem_flat_map" in _ops(extracted)
    fact_extraction = next(node for node in _walk(extracted) if node.op == "sem_flat_map")
    assert fact_extraction.params["ordinal_col"] == "fact_ordinal"
    assert tuple(column.name for column in fact_extraction.params["output_cols"]) == (
        "source_entity_ordinal",
        "target_entity_ordinal",
        "relation_type",
        "fact",
        "valid_at",
        "invalid_at",
    )
    assert "fact_id" not in fact_extraction.params["instruction"]
    episode_entities = next(
        node
        for node in _walk(extracted)
        if node.op == "array_agg" and node.params.get("output_col") == "entities"
    )
    assert episode_entities.params["columns"] == (
        "entity_ordinal",
        "name",
        "entity_type",
    )
    assert "summary" not in episode_entities.params["columns"]
    assert "entity_id" not in episode_entities.params["columns"]
    endpoint_joins = [node for node in _walk(extracted) if node.op == "join"]
    assert any(
        node.params["on"] == ("episode_id", "source_entity_ordinal")
        for node in endpoint_joins
    )
    assert any(
        node.params["on"] == ("episode_id", "target_entity_ordinal")
        for node in endpoint_joins
    )
    self_loop_filter = next(
        node
        for node in _walk(extracted)
        if node.op == "filter"
        and node.params["predicate"].get("op") == "ne"
        and node.params["predicate"]["left"].get("name") == "source_entity_id"
        and node.params["predicate"]["right"].get("name") == "target_entity_id"
    )
    assert self_loop_filter.inputs[0].op == "join"
    assert "sem_groupby" in _ops(deduplicated)
    deduplicated_agg = next(node for node in _walk(deduplicated) if node.op == "agg")
    fact_grouping = deduplicated_agg.inputs[0]
    assert fact_grouping.op == "sem_groupby"
    assert fact_grouping.params["input_cols"] == ("relation_type", "fact")
    assert fact_grouping.params["partition_by"] == (
        "source_entity_id",
        "target_entity_id",
    )
    specs = deduplicated_agg.params["aggregates"]
    assert any(isinstance(specification, SemanticAggregateSpec) for specification in specs)
    assert any(isinstance(specification, ArrayAggregateSpec) for specification in specs)
    assert sum(isinstance(specification, MinAggregateSpec) for specification in specs) >= 5
    fact_id = next(
        specification
        for specification in specs
        if isinstance(specification, MinAggregateSpec)
        and specification.output_col == "fact_id"
    )
    assert fact_id.columns == ("add_seq", "fact_ordinal")
    assert not any(
        isinstance(specification, MinAggregateSpec)
        and specification.output_col in {"source_entity_id", "target_entity_id"}
        for specification in specs
    )
    canonical_fact = next(
        specification
        for specification in specs
        if isinstance(specification, SemanticAggregateSpec)
    )
    assert tuple(column.name for column in canonical_fact.output_cols) == (
        "relation_type",
        "fact",
    )
    assert sum(node.op == "sem_filter" for node in _walk(deduplicated)) == 0

    assert contradiction_pairs.op == "sem_filter"
    assert contradiction_pairs.inputs[0].op == "join"
    assert contradiction_pairs.inputs[0].inputs[0].params["name"] == "earlier_added"
    assert contradiction_pairs.inputs[0].inputs[1].params["name"] == "later_added"
    assert "{fact:earlier_added}" in contradiction_pairs.params["instruction"]
    assert "{fact:later_added}" in contradiction_pairs.params["instruction"]
    assert len(contradiction_pairs.inputs[0].params["on"]) == 1
    assert any(
        predicate["op"] == "lt"
        and predicate["left"]["name"] == "fact_id"
        and predicate["right"]["name"] == "fact_id"
        for predicate in contradiction_pairs.inputs[0].params["on"]
    )

    assert invalidations.op == "agg"
    invalidation_events = invalidations.inputs[0].inputs[0]
    assert invalidation_events.op == "union_by_name"
    assert all(
        any(node.op == "filter" for node in _walk(branch))
        for branch in invalidation_events.inputs
    )
    assert all(
        isinstance(specification, MinAggregateSpec)
        for specification in invalidations.params["aggregates"]
    )
    assert output_columns(invalidations) == ("fact_id", "invalid_at", "expired_at")

    assert facts_with_invalidations.op == "join"
    assert facts_with_invalidations.params["how"] == "left"
    assert facts_with_invalidations.inputs[0].params["name"] == "fact"
    assert facts_with_invalidations.inputs[1].params["name"] == "invalidation"

    final_assign = facts.inputs[0]
    assert final_assign.op == "assign"
    assert final_assign.params["assignments"]["invalid_at"]["kind"] == "least"
    assert final_assign.params["assignments"]["expired_at"]["kind"] == "least"
    assert final_assign.inputs[0] == facts_with_invalidations
    assert output_columns(facts) == (
        "fact_id",
        "source_entity_id",
        "target_entity_id",
        "relation_type",
        "fact",
        "valid_at",
        "invalid_at",
        "expired_at",
        "provenance",
        "created_at",
        "add_seq",
    )
    assert "status" not in output_columns(facts)
    assert all(
        "group_id" not in output_columns(node)
        for node in (facts, deduplicated, invalidations)
    )


def test_zep_communities_view_is_semantic_grouping_not_label_propagation() -> None:
    """Community view is declarative semantic grouping, not a fixpoint operator."""

    spec = ZepMemoryExtended.spec()
    query = spec.views["communities"].query
    ops = _ops(query)

    assert "_community_members" not in spec.private_relations
    members = next(node for node in _walk(query) if node.op == "drop_duplicates")
    assert members.inputs[0].op == "union_by_name"
    member_assignments = [
        node.params["assignments"]["entity_id"]["name"]
        for node in _walk(members)
        if node.op == "assign" and "entity_id" in node.params["assignments"]
    ]
    assert set(member_assignments) == {"source_entity_id", "target_entity_id"}
    assert "sem_groupby" in ops
    assert "agg" in ops
    assert "label_propagation" not in ops
    groupby = next(node for node in _walk(query) if node.op == "sem_groupby")
    assert "partition_by" not in groupby.params
    aggregate = next(node for node in _walk(query) if node.op == "agg")
    aggregate_specs = aggregate.params["aggregates"]
    community_id = next(
        specification
        for specification in aggregate_specs
        if isinstance(specification, MinAggregateSpec)
    )
    assert community_id.columns == ("entity_id",)
    assert community_id.output_col == "community_id"
    semantic = next(
        specification
        for specification in aggregate_specs
        if isinstance(specification, SemanticAggregateSpec)
    )
    assert tuple(column.name for column in semantic.output_cols) == ("name", "summary")
    assert output_columns(query) == (
        "community_id",
        "name",
        "summary",
        "members",
    )
