"""SimpleMem-style memory policy."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.retrieval import BM25, CosineSimilarity, RRF


WINDOW_SIZE = 40
WINDOW_SLIDE = 38 

def _derive_observation_date(created_at: str | float | int) -> str:
    if isinstance(created_at, (int, float)):
        dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(created_at))
    return dt.strftime("%Y-%m-%d")


def _now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


SIMPLEMEM_EXTRACTION_PROMPT = r"""
You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format.

Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

[Overlap Awareness]
Dialogue windows may partially overlap with the previous window. Facts that relate exclusively to dialogues appearing in prior windows may have already been captured. Focus on extracting NEW information not previously covered.

[Current Window Dialogues]
{dialogues}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information

[Example]
Dialogues:
[{{"content": "Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product", "speaker": "Alice", "timestamp": "2025-11-15"}}, {{"content": "Okay, I'll prepare the materials", "speaker": "Bob", "timestamp": "2025-11-15"}}]

Output:
{{
  "rows": [
    {{
      "lossless_restatement": "Alice suggested at 2025-11-15 to meet with Bob at Starbucks on 2025-11-16 at 14:00 to discuss the new product.",
      "keywords": "Alice, Bob, Starbucks, new product, meeting",
      "timestamp": "2025-11-16T14:00:00",
      "location": "Starbucks",
      "persons": "Alice, Bob",
      "entities": "new product",
      "topic": "Product discussion meeting arrangement"
    }},
    {{
      "lossless_restatement": "Bob agreed to attend the meeting and committed to prepare relevant materials.",
      "keywords": "Bob, prepare materials, agree",
      "timestamp": null,
      "location": null,
      "persons": "Bob",
      "entities": null,
      "topic": "Meeting preparation confirmation"
    }}
  ]
}}

Now process the above dialogues.
""".strip()


class SimpleMemMemory(Memory):
    """SimpleMem-style memory policy."""

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
            .drop_duplicates()
        )
    )

    retrieval_query = facts.search(
        UserQuery(),
        methods=[BM25(), CosineSimilarity()],
        reranker=RRF(),
        limit=25,
    )
