"""Graphiti-compatible storage profile for the baseline Zep policy."""

from __future__ import annotations

from agent_memory.storage import Schema, StatementSet, TableDescriptor

from .policy import ZepMemory


_EPISODE_SCHEMA = (
    Schema.new_builder()
    .column("episode_id", "STRING")
    .column("content", "STRING")
    .column("role", "STRING")
    .column("speaker", "STRING")
    .column("reference_time", "TIMESTAMP(6)")
    .column("source_description", "STRING")
    .column("created_at", "TIMESTAMP_LTZ(6)")
    .column("add_seq", "BIGINT")
    .primary_key("episode_id")
    .build()
)

_ENTITY_SCHEMA = (
    Schema.new_builder()
    .column("entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
    .column("name", "STRING")
    .column("entity_type", "STRING")
    .column("summary", "STRING")
    .column(
        "mentions",
        "ARRAY<ROW<episode_id STRING, entity_ordinal BIGINT, name STRING, "
        "entity_type STRING, content STRING, created_at TIMESTAMP_LTZ(6), "
        "add_seq BIGINT>>",
    )
    .primary_key("entity_id")
    .build()
)

_FACT_SCHEMA = (
    Schema.new_builder()
    .column("fact_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
    .column("source_entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
    .column("target_entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
    .column("relation_type", "STRING")
    .column("fact", "STRING")
    .column("valid_at", "TIMESTAMP(6)")
    .column("invalid_at", "TIMESTAMP(6)")
    .column("expired_at", "TIMESTAMP_LTZ(6)")
    .column(
        "provenance",
        "ARRAY<ROW<episode_id STRING, fact_ordinal BIGINT, "
        "created_at TIMESTAMP_LTZ(6), content STRING>>",
    )
    .column("created_at", "TIMESTAMP_LTZ(6)")
    .column("add_seq", "BIGINT")
    .primary_key("fact_id")
    .build()
)

_MENTION_SCHEMA = (
    Schema.new_builder()
    .column("episode_id", "STRING")
    .column("entity_ordinal", "BIGINT")
    .column("entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
    .primary_key("episode_id", "entity_id")
    .build()
)

_EPISODE_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_EPISODE_SCHEMA)
    .option("kind", "node")
    .option("label", "Episodic")
    .option("key", "episode_id")
    .option("property.episode_id", "uuid")
    .build()
)

_ENTITY_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_ENTITY_SCHEMA)
    .option("kind", "node")
    .option("label", "Entity")
    .option("key", "entity_id")
    .option("property.entity_id", "uuid")
    .build()
)

_FACT_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_FACT_SCHEMA)
    .option("kind", "relationship")
    .option("type", "RELATES_TO")
    .option("key", "fact_id")
    .option("source", "source_entity_id")
    .option("target", "target_entity_id")
    .option("property.fact_id", "uuid")
    .option("property.relation_type", "name")
    .build()
)

_MENTION_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_MENTION_SCHEMA)
    .option("kind", "relationship")
    .option("type", "MENTIONS")
    .option("source", "episode_id")
    .option("target", "entity_id")
    .build()
)


# Connection details stay in StorageDeployment; this profile is immutable.
GRAPHITI_NEO4J_STATEMENTS = (
    StatementSet()
    .add_insert(_EPISODE_TARGET, ZepMemory.episodes)
    .add_insert(_ENTITY_TARGET, ZepMemory.entities)
    .add_insert(_FACT_TARGET, ZepMemory.facts)
    .add_insert(_MENTION_TARGET, ZepMemory._episode_entities)
)


__all__ = ["GRAPHITI_NEO4J_STATEMENTS"]
