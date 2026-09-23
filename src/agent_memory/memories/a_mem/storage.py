"""Neo4j storage profile for the A-Mem note-evolution policy.

Implements equation 3 of the A-Mem design: the embedded text is the verbatim
concatenation of content + keywords + context + tags. The policy emits that
combined text in a single `embedding_text` column (see a_mem/prompts.py), and
this profile hands it to the engine's existing single-column EmbeddingSpec.
No engine code is modified; this module only consumes the public connector API.
"""

from __future__ import annotations

from agent_memory.storage import Schema, StatementSet, TableDescriptor
from agent_memory.storage.neo4j import (
    EmbeddingSpec,
    Neo4jIdentity,
    Neo4jNodeMapping,
    Neo4jRelationshipMapping,
    Neo4jSchema,
)

from .policy import AMem


# Physical embedding and index contract for the A-Mem profile. Reuses the same
# BGE-M3 revision/dimensions as mem0/zep so vectors live in the same space.
AMEM_BGE_M3 = EmbeddingSpec(
    source_column="embedding_text",
    property_name="note_embedding",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    normalize=True,
)


# Sink schema validates logical policy rows before any Neo4j lowering occurs.
_NOTE_SCHEMA = (
    Schema.new_builder()
    .column("_row_id", "BIGINT")
    .column("content", "STRING")
    .column("keywords", "ARRAY<STRING>")
    .column("context", "STRING")
    .column("tags", "ARRAY<STRING>")
    .column("timestamp", "STRING")
    # Verbatim eq.3 concatenation; also stored as a node property so the
    # cross-encoder reranker can read it back as passage text.
    .column("embedding_text", "STRING")
    .primary_key("_row_id")
    .build()
)


_NOTE_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_NOTE_SCHEMA)
    .mapping(
        Neo4jNodeMapping(
            label="Note",
            identity=Neo4jIdentity(columns=("_row_id",), kind="note"),
            properties={
                "content": "content",
                "keywords": "keywords",
                "context": "context",
                "tags": "tags",
                "timestamp": "timestamp",
                "embedding_text": "embedding_text",
            },
            embedding=AMEM_BGE_M3,
        )
    )
    .build()
)


# Link edges live in their own relationship table, not as an array property on
# the Note node, so BFS neighbor expansion can traverse them.
_NOTE_LINK_SCHEMA = (
    Schema.new_builder()
    .column("source_note_id", "STRING")
    .column("target_note_id", "STRING")
    .primary_key("source_note_id", "target_note_id")
    .build()
)


_NOTE_LINK_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(_NOTE_LINK_SCHEMA)
    .mapping(
        Neo4jRelationshipMapping(
            relationship_type="RELATES_TO",
            identity=Neo4jIdentity(
                columns=("source_note_id", "target_note_id"), kind="note_link"
            ),
            source=Neo4jIdentity(columns=("source_note_id",), kind="note"),
            target=Neo4jIdentity(columns=("target_note_id",), kind="note"),
        )
    )
    .build()
)


AMEM_NEO4J_SCHEMA = Neo4jSchema(
    queries=(
        "CREATE CONSTRAINT agent_memory_amem_note_uuid IF NOT EXISTS "
        "FOR (n:Note) REQUIRE n.uuid IS UNIQUE",
        "CREATE INDEX agent_memory_amem_note_embedding IF NOT EXISTS "
        "FOR (n:Note) ON (n.note_embedding)",
        "CREATE INDEX agent_memory_amem_note_link_uuid IF NOT EXISTS "
        "FOR ()-[e:RELATES_TO]-() ON (e.uuid)",
    ),
    traversal_relationship_types=("RELATES_TO",),
)


AMEM_NEO4J_STATEMENTS = (
    StatementSet()
    .add_insert(_NOTE_TARGET, AMem.note)
    .add_insert(_NOTE_LINK_TARGET, AMem._note_links)
)


__all__ = [
    "AMEM_BGE_M3",
    "AMEM_NEO4J_SCHEMA",
    "AMEM_NEO4J_STATEMENTS",
]
