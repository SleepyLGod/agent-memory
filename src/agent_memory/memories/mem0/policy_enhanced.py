"""Mem0-style additive memory with semantic top-k retrieval."""

from __future__ import annotations

from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery

from .prompts import (
    MEM0_ADDITIVE_EXTRACTION_INSTRUCTION,
    MEM0_SEMANTIC_DUPLICATE_INSTRUCTION,
)

_RECENT_MESSAGE_LIMIT = 10
_MEMORY_ORDINAL_COLUMN = "_memory_ordinal"


class Mem0MemoryEnhanced(Memory):
    """Keep the Mem0 additive view while ranking memories with an LLM."""

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

    # This is an agent-memory semantic alternative, not Native Mem0 retrieval.
    retrieval_query = memories.sem_topk(UserQuery(), 20)


__all__ = ["Mem0MemoryEnhanced"]
