"""Graphiti-compatible storage profile for the baseline Zep policy."""

from __future__ import annotations

from agent_memory.storage import Schema, StatementSet, TableDescriptor
from agent_memory.storage.neo4j import (
    EmbeddingSpec,
    Neo4jIdentity,
    Neo4jNestedProperty,
    Neo4jNodeMapping,
    Neo4jRelationshipMapping,
    Neo4jSchema,
)

from .policy import ZepMemory


# Physical embedding and index contract for the Graphiti-compatible profile.
GRAPHITI_BGE_M3 = EmbeddingSpec(
    source_column="name",
    property_name="name_embedding",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    normalize=True,
)

GRAPHITI_NEO4J_SCHEMA = Neo4jSchema(
    queries=(
        "CREATE CONSTRAINT agent_memory_commit_namespace IF NOT EXISTS "
        "FOR (m:_AgentMemoryStorageCommit) REQUIRE m.group_id IS UNIQUE",
        "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)",
        "CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)",
        "CREATE INDEX relation_uuid IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.uuid)",
        "CREATE INDEX mention_uuid IF NOT EXISTS "
        "FOR ()-[e:MENTIONS]-() ON (e.uuid)",
        "CREATE INDEX entity_group_id IF NOT EXISTS "
        "FOR (n:Entity) ON (n.group_id)",
        "CREATE INDEX episode_group_id IF NOT EXISTS "
        "FOR (n:Episodic) ON (n.group_id)",
        "CREATE INDEX relation_group_id IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.group_id)",
        "CREATE INDEX mention_group_id IF NOT EXISTS "
        "FOR ()-[e:MENTIONS]-() ON (e.group_id)",
        "CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        "CREATE INDEX created_at_entity_index IF NOT EXISTS "
        "FOR (n:Entity) ON (n.created_at)",
        "CREATE INDEX created_at_episodic_index IF NOT EXISTS "
        "FOR (n:Episodic) ON (n.created_at)",
        "CREATE INDEX valid_at_episodic_index IF NOT EXISTS "
        "FOR (n:Episodic) ON (n.valid_at)",
        "CREATE INDEX name_edge_index IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.name)",
        "CREATE INDEX created_at_edge_index IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.created_at)",
        "CREATE INDEX expired_at_edge_index IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.expired_at)",
        "CREATE INDEX valid_at_edge_index IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.valid_at)",
        "CREATE INDEX invalid_at_edge_index IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)",
        "CREATE FULLTEXT INDEX episode_content IF NOT EXISTS "
        "FOR (e:Episodic) ON EACH "
        "[e.content, e.source, e.source_description, e.group_id]",
        "CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS "
        "FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]",
        "CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]",
    ),
    node_fulltext_indexes={"Entity": "node_name_and_summary"},
    relationship_fulltext_indexes={"RELATES_TO": "edge_name_and_fact"},
    traversal_relationship_types=("RELATES_TO", "MENTIONS"),
)


# Sink schemas validate logical policy rows before any Neo4j lowering occurs.
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
    .primary_key("episode_id", "entity_ordinal")
    .build()
)

# Typed mappings translate maintained relations into Graphiti graph objects.
_EPISODE_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_EPISODE_SCHEMA)
    .mapping(
        Neo4jNodeMapping(
            label="Episodic",
            identity=Neo4jIdentity(columns=("episode_id",), kind="episode"),
            properties={
                "name": "source_description",
                "source_description": "source_description",
                "content": "content",
                "created_at": "created_at",
                "valid_at": "reference_time",
            },
            constants={"source": "message", "entity_edges": []},
        )
    )
    .build()
)

_ENTITY_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_ENTITY_SCHEMA)
    .mapping(
        Neo4jNodeMapping(
            label="Entity",
            identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
            properties={"name": "name", "summary": "summary"},
            nested_properties=(
                Neo4jNestedProperty(
                    source_column="mentions",
                    field="created_at",
                    property_name="created_at",
                    many=False,
                ),
            ),
            embedding=GRAPHITI_BGE_M3,
        )
    )
    .build()
)

_FACT_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_FACT_SCHEMA)
    .mapping(
        Neo4jRelationshipMapping(
            relationship_type="RELATES_TO",
            identity=Neo4jIdentity(columns=("fact_id",), kind="fact"),
            source=Neo4jIdentity(
                columns=("source_entity_id",), kind="entity"
            ),
            target=Neo4jIdentity(
                columns=("target_entity_id",), kind="entity"
            ),
            properties={
                "name": "relation_type",
                "fact": "fact",
                "created_at": "created_at",
                "expired_at": "expired_at",
                "valid_at": "valid_at",
                "invalid_at": "invalid_at",
            },
            nested_properties=(
                Neo4jNestedProperty(
                    source_column="provenance",
                    field="episode_id",
                    property_name="episodes",
                    many=True,
                    identity_kind="episode",
                ),
            ),
            embedding=EmbeddingSpec(
                source_column="fact",
                property_name="fact_embedding",
                model=GRAPHITI_BGE_M3.model,
                revision=GRAPHITI_BGE_M3.revision,
                dimensions=GRAPHITI_BGE_M3.dimensions,
                normalize=GRAPHITI_BGE_M3.normalize,
            ),
        )
    )
    .build()
)

_MENTION_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_MENTION_SCHEMA)
    .mapping(
        Neo4jRelationshipMapping(
            relationship_type="MENTIONS",
            # One episode may mention the same canonical entity more than once.
            identity=Neo4jIdentity(
                columns=("episode_id", "entity_ordinal"), kind="mention"
            ),
            source=Neo4jIdentity(columns=("episode_id",), kind="episode"),
            target=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        )
    )
    .build()
)


# Bind public views and the private mention bridge without deployment credentials.
GRAPHITI_NEO4J_STATEMENTS = (
    StatementSet()
    .add_insert(_EPISODE_TARGET, ZepMemory.episodes)
    .add_insert(_ENTITY_TARGET, ZepMemory.entities)
    .add_insert(_FACT_TARGET, ZepMemory.facts)
    .add_insert(_MENTION_TARGET, ZepMemory._episode_entities)
)

__all__ = [
    "GRAPHITI_BGE_M3",
    "GRAPHITI_NEO4J_SCHEMA",
    "GRAPHITI_NEO4J_STATEMENTS",
]
