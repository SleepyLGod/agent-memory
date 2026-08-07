"""Mem0 OSS v3-style additive memory policy."""

from __future__ import annotations

from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.retrieval import CosineSimilarity, RetrievalQuery

from .prompts import (
    MEM0_ADDITIVE_EXTRACTION_INSTRUCTION,
    MEM0_SEMANTIC_DUPLICATE_INSTRUCTION,
)

_RECENT_MESSAGE_LIMIT = 10
_MEMORY_ORDINAL_COLUMN = "_memory_ordinal"


class Mem0Memory(Memory):
    """Declarative core memory view inspired by Mem0 OSS v3.

    One runtime and storage namespace represent one user/agent/run scope. Base
    vector retrieval is declared as logical IR and materialized in Qdrant.
    """

    log = Log(
        {
            "role": "Conversation role, normally user or assistant.",
            "content": "One normalized conversation turn.",
            "observation_date": (
                "Date when the message was observed, used to resolve relative time."
            ),
        },
        system_columns=True,
    )

    # Each row remains the current message; the array contains only prior rows.
    _windowed_messages = log.over(
        rows=(-_RECENT_MESSAGE_LIMIT, -1)
    ).array_agg(
        columns=["role", "content", "observation_date"],
        output_col="previous_messages",
    )

    _extracted_memories = _windowed_messages.sem_flat_map(
        input_cols=[
            "role",
            "content",
            "observation_date",
            "previous_messages",
        ],
        output_cols={
            "memory": "One self-contained additive memory statement.",
            "attributed_to": (
                "Source role for the memory: exactly user or assistant."
            ),
        },
        instruction=MEM0_ADDITIVE_EXTRACTION_INSTRUCTION,
        ordinal_col=_MEMORY_ORDINAL_COLUMN,
    )
    _earlier_memories = _extracted_memories.alias("earlier")
    _later_memories = _extracted_memories.alias("later")
    _candidate_memory_pairs = _earlier_memories.join(
        _later_memories,
        on=(
            _earlier_memories.col("_add_seq")
            < _later_memories.col("_add_seq")
        ),
    )
    _duplicate_memory_pairs = _candidate_memory_pairs.sem_filter(
        instruction=MEM0_SEMANTIC_DUPLICATE_INSTRUCTION,
    )
    _duplicate_memory_keys = (
        _duplicate_memory_pairs.assign(
            _row_id=_later_memories.col("_row_id"),
            _memory_ordinal=_later_memories.col(_MEMORY_ORDINAL_COLUMN),
        )
        .select(["_row_id", _MEMORY_ORDINAL_COLUMN])
        .drop_duplicates()
    )

    memories = (
        _extracted_memories.join(
            _duplicate_memory_keys,
            on=["_row_id", _MEMORY_ORDINAL_COLUMN],
            how="left_anti",
        )
        .select(["memory", "attributed_to"])
        .drop_duplicates(subset=["memory"])
    )

    # Mem0 Base parity uses indexed cosine retrieval. An experimental,
    # LLM-ranked alternative is `retrieval_query = memories.sem_topk(
    # UserQuery(), 20)`, which is a different retrieval recipe.
    retrieval_query = RetrievalQuery(
        memories=memories.search(
            UserQuery(),
            methods=[
                CosineSimilarity(candidate_limit=80, min_score=0.1),
            ],
            reranker=None,
            limit=20,
        ).select(
            [
                "record_id",
                "memory",
                "attributed_to",
                "rank",
                "score",
            ]
        )
    )


__all__ = ["Mem0Memory"]
