"""Graphiti-compatible Neo4j storage profile for the SimpleMem policy."""

from __future__ import annotations

from agent_memory.storage import Schema, StatementSet, TableDescriptor
from agent_memory.storage.neo4j import (
    EmbeddingSpec,
    Neo4jIdentity,
    Neo4jNodeMapping,
    Neo4jSchema,
)

from .policy import SimpleMemMemory
from .policy_enhanced import SimpleMemMemoryEnhanced


# Physical embedding and index contract for the SimpleMem profile.
SIMPLEMEM_BGE_M3 = EmbeddingSpec(
    source_column="lossless_restatement",
    property_name="fact_embedding",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    normalize=True,
)

SIMPLEMEM_NEO4J_SCHEMA = Neo4jSchema(
    queries=(
        "CREATE CONSTRAINT agent_memory_commit_namespace IF NOT EXISTS "
        "FOR (m:_AgentMemoryStorageCommit) REQUIRE m.group_id IS UNIQUE",
        "CREATE INDEX fact_uuid IF NOT EXISTS FOR (n:Fact) ON (n.uuid)",
        "CREATE INDEX fact_group_id IF NOT EXISTS FOR (n:Fact) ON (n.group_id)",
        "CREATE INDEX fact_restatement_index IF NOT EXISTS "
        "FOR (n:Fact) ON (n.lossless_restatement)",
        "CREATE FULLTEXT INDEX fact_restatement_fulltext IF NOT EXISTS "
        "FOR (n:Fact) ON EACH [n.lossless_restatement, n.group_id]",
    ),
    node_fulltext_indexes={"Fact": "fact_restatement_fulltext"},
    traversal_relationship_types=(),
)

_FACT_SCHEMA = (
    Schema.new_builder()
    .column("lossless_restatement", "STRING")
    .column("keywords", "STRING")
    .column("timestamp", "STRING")
    .column("location", "STRING")
    .column("persons", "STRING")
    .column("entities", "STRING")
    .column("topic", "STRING")
    .primary_key("lossless_restatement")
    .build()
)

_FACT_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_FACT_SCHEMA)
    .mapping(
        Neo4jNodeMapping(
            label="Fact",
            identity=Neo4jIdentity(
                columns=("lossless_restatement",), kind="fact"
            ),
            properties={
                "lossless_restatement": "lossless_restatement",
                "keywords": "keywords",
                "timestamp": "timestamp",
                "location": "location",
                "persons": "persons",
                "entities": "entities",
                "topic": "topic",
            },
            embedding=SIMPLEMEM_BGE_M3,
        )
    )
    .build()
)

# One shared storage profile binds the identical facts view for both policies.
SIMPLEMEM_NEO4J_STATEMENTS = StatementSet().add_insert(
    _FACT_TARGET,
    SimpleMemMemory.facts,
)

SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS = StatementSet().add_insert(
    _FACT_TARGET,
    SimpleMemMemoryEnhanced.facts,
)


__all__ = [
    "SIMPLEMEM_BGE_M3",
    "SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS",
    "SIMPLEMEM_NEO4J_SCHEMA",
    "SIMPLEMEM_NEO4J_STATEMENTS",
]
