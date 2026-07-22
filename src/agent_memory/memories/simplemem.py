"""SimpleMem-style memory policy."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery


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
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

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

[Output Format]
Return a JSON object with a "rows" array. Each row is a memory entry:

{{
  "rows": [
    {{
      "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
      "keywords": ["keyword1", "keyword2"],
      "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
      "location": "location name or null",
      "persons": ["name1", "name2"],
      "entities": ["entity1", "entity2"],
      "topic": "topic phrase"
    }}
  ]
}}

[Example]
Dialogues:
[{{"speaker": "Alice", "content": "Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product", "timestamp": "2025-11-15"}}, {{"speaker": "Bob", "content": "Okay, I'll prepare the materials", "timestamp": "2025-11-15"}}]

Output:
{{
  "rows": [
    {{
      "lossless_restatement": "Alice suggested at 2025-11-15 to meet with Bob at Starbucks on 2025-11-16 at 14:00 to discuss the new product.",
      "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
      "timestamp": "2025-11-16T14:00:00",
      "location": "Starbucks",
      "persons": ["Alice", "Bob"],
      "entities": ["new product"],
      "topic": "Product discussion meeting arrangement"
    }},
    {{
      "lossless_restatement": "Bob agreed to attend the meeting and committed to prepare relevant materials.",
      "keywords": ["Bob", "prepare materials", "agree"],
      "timestamp": null,
      "location": null,
      "persons": ["Bob"],
      "entities": [],
      "topic": "Meeting preparation confirmation"
    }}
  ]
}}

Now process the above dialogues. Return ONLY the JSON object, no text, reasoning, explanations, or wrappers.
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
                    "keywords": "JSON array of core keywords for exact matching.",
                    "timestamp": "ISO 8601 absolute time or null.",
                    "location": "Location name or null.",
                    "persons": "JSON array of person names mentioned.",
                    "entities": "JSON array of entities (companies, products, etc.).",
                    "topic": "Concise topic phrase.",
                },
                instruction=SIMPLEMEM_EXTRACTION_PROMPT,
            )
            .select([
                "lossless_restatement", "keywords", "timestamp",
                "location", "persons", "entities", "topic",
            ])
        )
    )

    retrieval_query = facts.sem_topk(UserQuery(), 25)
