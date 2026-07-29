"""Local Qdrant storage profile for the Mem0 Base policy."""

from __future__ import annotations

from agent_memory.storage import Schema, StatementSet, TableDescriptor
from agent_memory.storage.qdrant import (
    EmbeddingSpec,
    QdrantIdentity,
    QdrantPointMapping,
)

from .policy import Mem0Memory
from .policy_enhanced import Mem0MemoryEnhanced


MEM0_BGE_M3 = EmbeddingSpec(
    source_column="memory",
    property_name="memory_embedding",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    normalize=True,
)

_MEMORY_SCHEMA = (
    Schema.new_builder()
    .column("memory", "STRING")
    .column("attributed_to", "STRING")
    .primary_key("memory")
    .build()
)

_MEMORY_TARGET = (
    TableDescriptor.for_connector("qdrant")
    .schema(_MEMORY_SCHEMA)
    .mapping(
        QdrantPointMapping(
            collection="mem0",
            identity=QdrantIdentity(columns=("memory",), kind="memory"),
            properties={
                "memory": "memory",
                "attributed_to": "attributed_to",
            },
            embedding=MEM0_BGE_M3,
        )
    )
    .build()
)

MEM0_QDRANT_STATEMENTS = StatementSet().add_insert(
    _MEMORY_TARGET,
    Mem0Memory.memories,
)

MEM0_ENHANCED_QDRANT_STATEMENTS = StatementSet().add_insert(
    _MEMORY_TARGET,
    Mem0MemoryEnhanced.memories,
)


__all__ = [
    "MEM0_BGE_M3",
    "MEM0_ENHANCED_QDRANT_STATEMENTS",
    "MEM0_QDRANT_STATEMENTS",
]
