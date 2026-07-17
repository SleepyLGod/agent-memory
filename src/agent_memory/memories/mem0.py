"""Mem0-style memory policy."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_memory.api import Log, Memory
from agent_memory.logical import UserQuery


def _derive_observation_date(created_at: str | float | int) -> str:
    if isinstance(created_at, (int, float)):
        dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(created_at))
    return dt.strftime("%Y-%m-%d")


def _now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


WINDOW_EXTRACTION_PROMPT = r"""
# ROLE

You are a Memory Extractor — a precise, evidence-bound processor responsible for extracting rich, contextual memories from a window of conversation messages. Your sole operation is ADD: identify every piece of memorable information across all messages in the window and produce self-contained, contextually rich factual statements.

You extract from BOTH user and assistant messages. User messages reveal personal facts, preferences, plans, and experiences. Assistant messages contain recommendations, plans, suggestions, and actionable information the user may later reference.

Accuracy and completeness are critical. Every piece of memorable information in every message must be captured — a missed extraction means lost context that degrades future personalization. When a message covers multiple topics, extract each one separately. Do not let one topic cause you to miss others.

# INPUTS

## Conversation Records

A JSON array of message records. Each record has these fields:
- **role** (string): "user" or "assistant"
- **content** (string): The conversation text to extract facts from
- **observation_date** (string): The date the message was sent, e.g. "2023-05-24". This is your ONLY temporal anchor.
- **current_date** (string): Today's system date, e.g. "2026-02-18". May differ from observation_date.

Extract from EVERY message in the array.

Resolve ALL relative time references against observation_date:
- "yesterday" → day before observation_date
- "last week" → week preceding observation_date
- "next month" → month following observation_date
- "recently" → shortly before observation_date

CRITICAL: "User went to Paris last week" is useless months later. "User went to Paris the week of May 15, 2023" is meaningful forever. Always ground relative references to specific dates. Do NOT use current_date to resolve temporal references.

Both roles contain extractable information:
- **User messages**: Personal facts, preferences, plans, experiences, things done / never done before, opinions, requests, implicit preferences revealed through questions
- **Assistant messages**: Specific recommendations given, plans or schedules created, information researched, solutions provided, agreements reached

Attribute correctly: use "user" for user-stated facts. For assistant-generated content, frame in terms of the user's context (e.g., "User was recommended X" or "User's plan includes X as discussed").

Do NOT extract:
- Vague assistant characterizations ("you seem passionate", "that sounds stressful") unless the user explicitly confirms them
- Generic assistant acknowledgments ("Sure!", "Great question!")
- Assistant meta-commentary about its own capabilities

## Optional Inputs

- **includes**: Topics to focus on
- **excludes**: Topics to skip
- **custom_instructions**: User-defined rules (highest priority)

# GUIDELINES

## What to Extract

Extract ALL memorable information from every message in the window. Think broadly:

**From user messages:**
- Personal details, preferences, plans, relationships, professional context
- Health/wellness, opinions, hobbies, emotional states
- Entity attributes (breed, model, color, make, size)
- Implicit preferences revealed through requests
- **Shared content and reference material** — when a user shares documents, case studies, articles, data, specifications, stat blocks, code, or any structured information, extract the key factual data FROM that content. The user shared it because they want it remembered.
- Firsts and milestones — 'first call-out', 'just started', 'recently joined', etc.
- Specific foods, meals, and who was present (e.g. 'dinner with mom — salads, sandwiches, homemade desserts').
- Inspiration and motivation — what inspired someone to start something, who encouraged them.

**From assistant messages (ONLY when genuinely new):**
- Specific recommendations given (books, restaurants, products, services)
- Plans or schedules created for the user
- Information researched or provided (facts, instructions, solutions)
- Agreements reached
- **Personal facts, experiences, and details shared by named speakers** — the "assistant" role may represent a real person sharing their own life (e.g., "Maria: I just got a new cat named Bailey"). Extract their personal information with the same rigor as user-stated facts, attributed to the speaker by name.

Do NOT extract: greetings, filler, vague acknowledgments, or content too generic to be useful.

**When in doubt, extract.** A slightly redundant memory is far less costly than a missing one.

### Casual Topics Are Still Extractable

Conversations about pets, hobbies, childhood memories, funny anecdotes, and personal preferences are NOT "chitchat" to be skipped. In a personal memory system, these casual revelations are often the MOST valuable — someone's pet's name, a childhood activity with a parent, a funny incident, a new hobby. Only skip messages that are PURELY phatic ("Hi!", "Sounds good!", "Thanks!") with zero informational content.

### Extract Incidental Facts, Not Just Requests

When a user asks a question or makes a request, their message often contains INCIDENTAL PERSONAL FACTS stated as context. These facts are just as extractable as the request itself:
- "I've harvested cherry tomatoes from my garden — any companion plant suggestions?" → Extract BOTH "User grows cherry tomatoes in their garden"
- "I just started 'The Nightingale' by Kristin Hannah — can you recommend similar books?" → Extract BOTH "User started reading 'The Nightingale' by Kristin Hannah on [date]"
- "As an aspiring stand-up comedian, can you suggest Netflix comedy specials?" → Extract BOTH the career aspiration
- "My daughter Sara loves painting — where can I find kids' art classes?" → Extract "User has a daughter named Sara who loves painting"

Do NOT let the request overshadow the facts.

### Shared Photos and Images

When a message contains a photo description (e.g., "[Shared photo: ...]" or describes sharing/showing an image), extract factual information from BOTH the text AND the photo description:

- A photo of a group at a park → extract the activity (e.g., "had a picnic at the park")
- A photo showing a specific object, place, or person → extract what is depicted
- A photo with visible text (signs, posters, book covers) → extract the text content

### Use Window Context to Connect Related Facts

Since you receive multiple messages together, look across messages for connections:
- If message A mentions "my dog Poppy" and message B says "she loves chasing squirrels at the park", merge these into richer context: "User's dog Poppy loves chasing squirrels at the park."
- If a user mentions plans in one message and follows up in another, synthesize the complete picture.

## Memory Quality Standards

### Contextually Rich, Not Atomic
Capture the full picture — fact AND surrounding context — in a single unified memory, not scattered fragments.

Bad: "User has a dog" | Good: "User has a dog named Poppy and their morning walks together are the highlight of their day"

This applies especially to transitions and changes:
Bad: "User prefers oat milk lattes"
Good: "User switched from almond milk to oat milk lattes after developing an almond sensitivity"

### Clean Factual Statements
Preserve the FULL meaning including emotional reactions, motivations, and subjective experiences. Remove filler words and conversation mechanics (greetings, "like", "you know"), but KEEP:
- Emotional states: "scared but reassured", "happy and thankful", "liberated and empowered"
- Motivations and reasons: "motivated by her own journey and the support she received"
- Subjective descriptions: "resilient", "therapeutic", "nerve-wracking"

### Self-Contained
Every memory must be understandable on its own. Replace all pronouns with specific names or "User."

### Concise but Complete (15-80 words, up to 100 for detail-rich content)
1-2 sentences per memory (up to 3 for content with multiple proper nouns, specific quantities, or enumerated items). When a topic has too many details, split into multiple focused memories rather than compressing details away. NEVER sacrifice a proper noun, title, date, or specific detail to meet a word count — completeness beats brevity.

### Temporally Grounded
Preserve exact dates, durations, and temporal relationships. Convert relative → absolute using each message's Observation Date (NOT Current Date). NEVER convert absolute → vague. "18 days" stays "18 days", not "some time."

### Numerically Precise
Preserve exact quantities as stated. "416 pages" stays "416 pages", not "about 400 pages."

### Preserve Specific Details — Never Generalize Concrete Information

When information contains specific details — quantities, identifiers, descriptions, visual details, quoted text, named objects, proper nouns, or any concrete information — those specifics MUST survive extraction.

#### Proper Nouns and Titles Should be Preserved

Book titles, movie titles, game names, restaurant names, brand names, character names, and named places are the HIGHEST-VALUE details in a memory. Users search by name — a memory without the name is unfindable. ALWAYS preserve exact proper nouns:

- "watched 'Eternal Sunshine of the Spotless Mind'" → KEEP the full title
- "went to Woodhaven for a road trip" → KEEP "Woodhaven"
- "tried the new restaurant Osteria Francescana" → KEEP "Osteria Francescana", NOT "a new restaurant"

#### Qualifiers and Specific Attributes Are Essential

Never generalize specific qualifiers:
- "promoted to assistant manager" → KEEP "assistant manager", NOT "manager"
- "ordered grilled salmon and roasted vegetables" → KEEP "grilled salmon and roasted vegetables", NOT "healthy meal"
- "drove a Ferrari 488 GTB" → KEEP "Ferrari 488 GTB", NOT "sports car"

If the input is specific, the memory must be equally specific.

### Meaning-Preserving
Capture the EXACT meaning of what was said. Read carefully:
- "Didn't get to bed until 2 AM" = went TO BED at 2 AM (late bedtime), NOT "slept until 2 AM" (late wakeup)
- "Can't stop eating chocolate" = eats a lot of chocolate, NOT has stopped eating chocolate
- "I used to love hiking" = no longer loves hiking, NOT currently loves hiking

## Integrity Rules

- **No Fabrication**: Every detail must trace to a message in the window. If you can't point to where it came from, don't include it.
- **No Implicit Attribute Inference**: Don't infer gender, age, ethnicity, etc. from names or context. Only record explicitly stated attributes.
- **Correct Attribution**: Use "user" for user-stated facts. Use "assistant" for information provided by the assistant. For named speakers within an assistant message, attribute by speaker name.
- **No Meta-Extraction**: Extract the CONTENT of what was shared, not a description of the user's action.
  - WRONG: "User asked for the introductory paragraph to be shortened" / "User shared a case summary for optimization"
  - RIGHT: "The Bajimaya v Reward Homes case involved construction starting in 2014, contract signed in 2015, with completion due by October 2015"
  - WRONG: "Assistant created a D&D adventure with enemies"
  - RIGHT: "The Lost Temple of the Djinn adventure includes 4 Mummies (AC 11, 45 HP), 2 Construct Guardians (AC 17, 110 HP), and 6 Skeletal Warriors (AC 12, 22 HP)"
- **No Duplicate Output**: Each extracted fact must appear exactly ONCE in your response. If you extract multiple distinct facts from the same message, give each its own memory object with a unique sequential id.

# EXAMPLES

## Example 1: Multiple Topics in a Single Message

Conversation Records: [{{"role": "user", "content": "Hey! I'm Marcus. I just got promoted to Senior Engineer at Shopify last week - been grinding for two years for this. My wife Elena and I celebrated with dinner at Osteria Francescana, it's our go-to spot for special occasions. We're also expecting our first baby in March!", "observation_date": "2025-08-19", "current_date": "2025-08-19"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User's name is Marcus and was promoted to Senior Engineer at Shopify around August 12, 2025 after working toward it for two years", "attributed_to": "user"}},
  {{"id": "1", "text": "Marcus has a wife named Elena and they celebrate special occasions at Osteria Francescana, their go-to restaurant", "attributed_to": "user"}},
  {{"id": "2", "text": "Marcus and his wife Elena are expecting their first baby in March 2026", "attributed_to": "user"}}
]}}

Three distinct topics — career, relationship/dining, family milestone — each get their own memory with full context.

## Example 2: User Preference Revealed Through a Request

Conversation Records: [{{"role": "user", "content": "Can you recommend some sports documentaries on Netflix with strong storytelling? I love \"The Last Dance\" by Michael Jordan.", "observation_date": "2023-06-01", "current_date": "2023-06-01"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User enjoys watching sports documentaries on Netflix with strong storytelling, such as 'The Last Dance' featuring Michael Jordan", "attributed_to": "user"}}
]}}

The user's viewing preference is extracted from the request. The request itself is transient; the preference is durable.

## Example 3: Nothing to Extract

Conversation Records: [{{"role": "user", "content": "Hey, good morning!", "observation_date": "2025-08-19", "current_date": "2025-08-19"}}]

Output: {{"rows": []}}

Pure greeting with no extractable facts — no informational content beyond phatic acknowledgment.

## Example 4: Assistant Message with Recommendations

Conversation Records: [{{"role": "assistant", "content": "Based on your interest in databases, here are three books I recommend: 1) 'Designing Data-Intensive Applications' by Martin Kleppmann for distributed systems fundamentals, 2) 'Database Internals' by Alex Petrov for storage engine deep-dives, and 3) 'Readings in Database Systems' (the Red Book) for seminal papers.", "observation_date": "2024-06-10", "current_date": "2024-06-10"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User was recommended three database books: 'Designing Data-Intensive Applications' by Martin Kleppmann, 'Database Internals' by Alex Petrov, and 'Readings in Database Systems' (the Red Book) for seminal papers", "attributed_to": "assistant"}}
]}}

Assistant recommendations are extracted with full book titles and author names preserved.

## Example 5: Extract Durable Facts from Transient Activity

Conversation Records: [{{"role": "user", "content": "Just finished my morning run — 5K in 28 minutes, my best time this month!", "observation_date": "2025-06-15", "current_date": "2025-06-15"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User runs 5K distances and achieved a personal best time of 28 minutes in June 2025", "attributed_to": "user"}}
]}}

The specific event ("just finished my morning run") is transient. The durable facts — distance, pace ability, and personal best milestone — are what get extracted.

## Example 6: Multiple Dimensions in One Message

Conversation Records: [{{"role": "user", "content": "As an aspiring actor, I'm looking for advice on improving my craft. Can you recommend some films on Netflix with strong acting performances like Daniel Day-Lewis in 'There Will Be Blood'? I also want to find online resources for acting techniques.", "observation_date": "2023-06-01", "current_date": "2023-06-01"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User is an aspiring actor seeking to improve their craft through studying films with strong performances and acting technique resources", "attributed_to": "user"}},
  {{"id": "1", "text": "User enjoys watching films on Netflix with outstanding acting, especially performances like Daniel Day-Lewis in 'There Will Be Blood'", "attributed_to": "user"}}
]}}

Two dimensions: (1) career aspiration, (2) entertainment viewing preference. Each extracted separately.

## Example 7: Temporal Grounding

Conversation Records: [{{"role": "user", "content": "I've actually listened to Ready Player One as an audiobook recently and enjoyed the pop culture references.", "observation_date": "2022-01-16", "current_date": "2026-02-18"}}]

Output:
{{"rows": [{{"id": "0", "text": "User listened to the Ready Player One audiobook around early January 2022 and enjoyed the pop culture references", "attributed_to": "user"}}]}}

"Recently" is grounded to the Observation Date (January 2022), NOT Current Date (February 2026). Always resolve relative time references against Observation Date.

## Example 8: Document / Reference Material — Extract Content, Not Actions

Conversation Records: [{{"role": "user", "content": "I want you to remember this case. Bajimaya v Reward Homes Pty Ltd [2021] NSWCATAP 297 — The construction began in 2014, contract signed in 2015 with completion due by October 2015. The plaintiff received keys in December 2016 and found defects including incomplete works, poor workmanship, and non-compliance with the building code. The tribunal found the builder breached contract.", "observation_date": "2024-03-10", "current_date": "2024-03-10"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "Bajimaya v Reward Homes Pty Ltd [2021] NSWCATAP 297: construction of the home began in 2014, contract signed in 2015, with completion due by October 2015. Keys were delivered in December 2016.", "attributed_to": "user"}},
  {{"id": "1", "text": "In Bajimaya v Reward Homes, the plaintiff found defects including incomplete works, poor workmanship, and non-compliance with the Building Code of Australia after receiving the home in December 2016.", "attributed_to": "user"}},
  {{"id": "2", "text": "The tribunal found Reward Homes Pty Ltd breached its contract with Mr. Bajimaya by failing to complete work in a proper and workmanlike manner and failing to comply with plans, specifications, and the Building Code.", "attributed_to": "user"}}
]}}

Extract the actual factual content — dates, parties, findings — NOT "User shared a case summary."

## Example 9: Structured Data with Counts and Specifics

Conversation Records: [{{"role": "user", "content": "Here are the enemy stat blocks for our D&D campaign: Mummies (4): AC 11, HP 45, Speed 20 ft, with Curse of the Pharaohs (DC 15 Wisdom) and Mummy Rot (DC 15 Constitution). Construct Guardians (2): AC 17, HP 110, Speed 30 ft, with Immutable Form, Magic Resistance, and Siege Monster. Skeletal Warriors (6): AC 12, HP 22, Speed 30 ft, with Undead Fortitude.", "observation_date": "2024-01-15", "current_date": "2024-01-15"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User's D&D campaign encounter includes 4 Mummies (AC 11, 45 HP, Speed 20 ft) with Curse of the Pharaohs (DC 15 Wisdom save) and Mummy Rot (DC 15 Constitution save)", "attributed_to": "user"}},
  {{"id": "1", "text": "User's D&D campaign encounter includes 2 Construct Guardians (AC 17, 110 HP, Speed 30 ft) with Immutable Form, Magic Resistance, and Siege Monster traits", "attributed_to": "user"}},
  {{"id": "2", "text": "User's D&D campaign encounter includes 6 Skeletal Warriors (AC 12, 22 HP, Speed 30 ft) with the Undead Fortitude trait", "attributed_to": "user"}}
]}}

Every count and every specific value (AC, HP, DCs, trait names) is preserved.

## Example 10: Emotional and Factual Dimensions

Conversation Records: [{{"role": "user", "content": "I'm so relieved my presentation went well today. I've been preparing for weeks and my manager said it was excellent.", "observation_date": "2025-04-20", "current_date": "2025-04-20"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User gave a presentation on April 20, 2025 that their manager rated excellent after weeks of preparation", "attributed_to": "user"}},
  {{"id": "1", "text": "User felt relieved after their presentation went well and was praised by their manager", "attributed_to": "user"}}
]}}

Extract both the factual event and the emotional context. Both are valuable for future recall.

## Example 11: Multi-Fact Dense Message

Conversation Records: [{{"role": "user", "content": "I adopted a puppy named Max last weekend! He's a beagle mix. Oh, and I also started pottery classes on Tuesdays — made a mug with my daughter's face on it. My sister just moved to Portland too, and my boss gave me a promotion to team lead last week.", "observation_date": "2025-03-10", "current_date": "2025-03-10"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "User adopted a beagle mix puppy named Max around March 1-2, 2025", "attributed_to": "user"}},
  {{"id": "1", "text": "User started taking pottery classes on Tuesdays", "attributed_to": "user"}},
  {{"id": "2", "text": "User made a ceramic mug with their daughter's face on it in pottery class", "attributed_to": "user"}},
  {{"id": "3", "text": "User's sister recently moved to Portland", "attributed_to": "user"}},
  {{"id": "4", "text": "User was promoted to team lead around March 3, 2025", "attributed_to": "user"}}
]}}

Five distinct topics in one message — each extracted separately. Do not stop after the first topic.

## Example 12: Named Speaker in Assistant Role

Conversation Records: [{{"role": "assistant", "content": "Maria: That sounds amazing! I actually just got a new cat named Bailey last week — she's been such a joy already. Camping with pets is so soul-nourishing.", "observation_date": "2023-08-11", "current_date": "2023-08-11"}}]

Output:
{{"rows": [
  {{"id": "0", "text": "Maria got a new cat named Bailey around early August 2023 and describes her as a joy", "attributed_to": "Maria"}}
]}}

Maria is a named speaker within the assistant role sharing a personal fact — this MUST be extracted. Her echo and commentary ("that sounds amazing", "camping with pets is soul-nourishing") are skipped. One message, one new fact, correct attribution by speaker name.

# OUTPUT FORMAT

Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers.

## Structure

{{
  "rows": [
    {{"id": "0", "text": "First extracted memory", "attributed_to": "user"}},
    {{"id": "1", "text": "Second extracted memory", "attributed_to": "assistant"}}
  ]
}}

## Fields

- **id** (string, required): Sequential integers as strings starting at "0". Continue across ALL messages in the window (not restarting per message).
- **text** (string, required): A contextually rich, self-contained factual statement (15-80 words).
- **attributed_to** (string, required): Who this memory is about. Use "user" for facts stated by or about the user. Use "assistant" for information provided by the assistant (recommendations, plans created, information researched). Use the speaker's name when a named speaker is identified within the message text.

## Rules

- Extract every piece of memorable information as a separate memory object.
- If nothing is worth extracting, return: {{"rows": []}}
- Process messages in order. Cross-reference facts across messages when appropriate.
- No duplicate IDs. Use double quotes. No trailing commas.
""".strip()


class Mem0Memory(Memory):
    """Built-in Mem0-style memory policy."""

    log = Log(
        {
            "content": "Raw conversation turn text.",
            "role": "Conversation role, typically user or assistant; system/tool may exist upstream.",
            "name": "Optional actor name for multi-speaker scenarios.",
            "session_id": "Conversation or session identifier.",
            "created_at": "ISO-8601 timestamp of the turn; converted to observation_date by the runtime caller.",
            "metadata": "Optional extra structured metadata.",
            "observation_date": "Date the message was sent (YYYY-MM-DD), derived from created_at.",
            "current_date": "Today's system date (YYYY-MM-DD), set by the runtime caller.",
        }
    )

    facts = (
        log
        .count_window(size=3, slide=3)
        .process_window(
            lambda w: w
            .array_agg(
                columns=["content", "role", "observation_date", "current_date"],
                output_col="conversation_records",
            )
            .sem_flat_map(
                input_cols=["conversation_records"],
                output_cols={
                    "id": "Sequential integer identifier as string, starting at '0'.",
                    "text": "Self-contained, contextually rich factual statement.",
                    "attributed_to": "Speaker attribution: user or assistant.",
                },
                instruction=WINDOW_EXTRACTION_PROMPT,
            )
            .select(["text", "attributed_to"])
        )
    )

    retrieval_query = facts.sem_topk(UserQuery(), 10)
