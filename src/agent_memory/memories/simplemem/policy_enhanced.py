"""SimpleMem-style hybrid memory policy."""

from __future__ import annotations

from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.retrieval import (
    BM25,
    CosineSimilarity,
    RRF,
    RetrievalQuery,
)

from .policy import (
    SIMPLEMEM_EXTRACTION_PROMPT,
    WINDOW_SIZE,
    WINDOW_SLIDE,
)


class SimpleMemMemoryEnhanced(Memory):
    """SimpleMem-style hybrid memory policy.

    Declares the same facts view as the base policy but retrieves with the
    full hybrid recipe (BM25 + dense cosine fused by RRF), mirroring the
    native SimpleMem three-view hybrid retriever. The symbolic/structured view
    has no framework analog and is intentionally omitted.
    """

    log = Log(
        {
            "content": "Raw conversation dialogue text.",
            "speaker": "Speaker name or role (e.g., user, assistant, or named person).",
            "timestamp": "Date the dialogue occurred (YYYY-MM-DD).",
        }
    )

    facts = (
        log
        .count_window(size=WINDOW_SIZE, slide=WINDOW_SLIDE)
        .process_window(
            lambda w: w
            .array_agg(
                columns=["content", "speaker", "timestamp"],
                output_col="dialogues",
            )
            .sem_flat_map(
                input_cols=["dialogues"],
                output_cols={
                    "lossless_restatement": "Complete unambiguous restatement with no pronouns and absolute timestamps.",
                    "keywords": "Comma-separated core keywords for exact matching.",
                    "timestamp": "ISO 8601 absolute time or null.",
                    "location": "Location name or null.",
                    "persons": "Comma-separated list of person names mentioned, or null.",
                    "entities": "Comma-separated list of entities (companies, products, etc.), or null.",
                    "topic": "Concise topic phrase.",
                },
                instruction=SIMPLEMEM_EXTRACTION_PROMPT,
            )
            .select([
                "lossless_restatement", "keywords", "timestamp",
                "location", "persons", "entities", "topic",
            ])
        )
        .drop_duplicates(subset=["lossless_restatement"])
    )

    retrieval_query = RetrievalQuery(
        facts=facts.search(
            UserQuery(),
            methods=[
                BM25(),
                CosineSimilarity(candidate_limit=80, min_score=0.1),
            ],
            reranker=RRF(),
            limit=25,
        ).select(
            [
                "record_id",
                "lossless_restatement",
                "keywords",
                "timestamp",
                "location",
                "persons",
                "entities",
                "topic",
                "rank",
                "score",
            ]
        )
    )
