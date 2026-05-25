"""Claude-style memory policy."""

from __future__ import annotations

from typing import Any

from agent_memory.api import Log, Memory


class ClaudeMemory(Memory):
    """Built-in Claude-style memory policy."""

    log = Log(
        {
            "message": "Raw user, assistant, or tool memory event text.",
            "role": "Message role such as user, assistant, system, or tool.",
            "timestamp": "Event timestamp.",
            "session_id": "Conversation or session identifier.",
            "metadata": "Optional structured metadata for the memory event.",
        }
    )

    topics = (
        log
        .sem_flat_map(
            output_cols={
                "topic_name": "Candidate durable memory topic name.",
                "topic_content": "Candidate durable memory content.",
            },
            instruction=(
                "Extract zero or more durable memory topic candidates from "
                "{message}, filling {topic_name} and {topic_content}."
            ),
        )
        .select(["topic_name", "topic_content"])
        .sem_groupby(
            key=["topic_name"],
            instruction="Find candidates related to the same durable memory topic.",
        )
        .sem_agg(
            input_cols=["topic_name", "topic_content"],
            output_cols={
                "topic_name": "Canonical durable memory topic name.",
                "topic_content": "Merged durable memory content.",
            },
            instruction="Choose a canonical topic name and merge topic content.",
        )
        .select(["topic_name", "topic_content"])
    )

    catalog = (
        topics.sem_map(
            input_cols=["topic_name", "topic_content"],
            output_cols={
                "catalog_title": "Title shown in the memory catalog.",
                "path": "Relative path to the topic memory file.",
                "hook": "One-line relevance hook.",
            },
            instruction="Produce one catalog row per topic.",
        ).select(["catalog_title", "path", "hook"])
    )

    def query(self, query: str) -> Any:
        """Run Claude-style retrieval.

        Query is policy-owned semantic retrieval behavior. The current v0.0
        interface records the boundary but does not execute the retrieval plan.
        """

        return self.catalog.sem_topk(query, 5)
