"""Mem0-style additive memory with semantic top-k retrieval."""

from __future__ import annotations

from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery

from .prompts import MEM0_ADDITIVE_EXTRACTION_INSTRUCTION

_RECENT_MESSAGE_LIMIT = 10


class Mem0MemoryEnhanced(Memory):
    """Keep the Mem0 additive view while ranking memories with an LLM."""

    log = Log(
        {
            "role": "Conversation role, normally user or assistant.",
            "content": "One normalized conversation turn.",
            "observation_date": (
                "Date when the message was observed, used to resolve relative time."
            ),
        }
    )

    # Each row remains the current message; the array contains only prior rows.
    _windowed_messages = log.over(
        rows=(-_RECENT_MESSAGE_LIMIT, -1)
    ).array_agg(
        columns=["role", "content", "observation_date"],
        output_col="previous_messages",
    )

    memories = (
        _windowed_messages.sem_flat_map(
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
        )
        .select(["memory", "attributed_to"])
        .drop_duplicates(subset=["memory"])
    )

    # This is an agent-memory semantic alternative, not Native Mem0 retrieval.
    retrieval_query = memories.sem_topk(UserQuery(), 20)


__all__ = ["Mem0MemoryEnhanced"]
