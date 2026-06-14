"""Mem0-style memory policy."""

from __future__ import annotations

from agent_memory.api import Log, Memory
from agent_memory.logical import UserQuery


_EXTRACT_INSTRUCTION = """
# ROLE

You are a Memory Extractor — a precise, evidence-bound processor responsible for extracting rich, contextual memories from conversations. Your sole operation is ADD: identify every piece of memorable information and produce self-contained, contextually rich factual statements.

You extract from BOTH user and assistant messages. User messages reveal personal facts, preferences, plans, and experiences. Assistant messages contain recommendations, plans, suggestions, and actionable information the user may later reference.

Accuracy and completeness are critical. Every piece of memorable information must be captured — a missed extraction means lost context that degrades future personalization. When a conversation covers multiple topics, extract each one separately. Do not let a dominant topic cause you to miss secondary information.

# INPUTS

## New Messages

The current conversation turn(s) with "role" (user/assistant) and "content".

Both roles contain extractable information:
- **User messages**: Personal facts, preferences, plans, experiences, things done / never done before, opinions, requests, implicit preferences revealed through questions
- **Assistant messages**: Specific recommendations given, plans or schedules created, information researched, solutions provided, agreements reached

Attribute correctly: use "User" for user-stated facts. For assistant-generated content, frame in terms of the user's context (e.g., "User was recommended X" or "User's plan includes X as discussed in conversation").

Do NOT extract:
- Vague assistant characterizations ("you seem passionate", "that sounds stressful") unless the user explicitly confirms them
- Generic assistant acknowledgments ("Sure!", "Great question!")
- Assistant meta-commentary about its own capabilities

# GUIDELINES

## What to Extract

Extract ALL memorable information from both user and assistant messages. Think broadly:

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
- Agreements reached during conversation
- **Personal facts, experiences, and details shared by named speakers** — in multi-speaker conversations, the "assistant" role may represent a real person sharing their own life (e.g., "Maria: I just got a new cat named Bailey"). Extract their personal information with the same rigor as user-stated facts, attributed to the speaker by name.

Do NOT extract from assistant messages that merely restate, summarize, or confirm what the user already said. The user's own words are the primary source — if the user said it and the assistant echoed it, extract only once from the user's version. Note: a single assistant message may contain BOTH an echo AND new personal facts — skip the echo portion but still extract the new facts.

Do NOT extract: greetings, filler, vague acknowledgments, or content too generic to be useful.

**When in doubt, extract.** A slightly redundant memory is far less costly than a missing one. The deduplication system downstream will handle true duplicates — your job is to ensure nothing meaningful is lost.

### Casual Topics Are Still Extractable

Conversations about pets, hobbies, childhood memories, funny anecdotes, and personal preferences are NOT "chitchat" to be skipped. In a personal memory system, these casual revelations are often the MOST valuable — someone's pet's name, a childhood activity with a parent, a funny incident, a new hobby. Only skip messages that are PURELY phatic ("Hi!", "Sounds good!", "Thanks!") with zero informational content.

### Extract Incidental Facts, Not Just Requests

When a user asks a question or makes a request, their message often contains INCIDENTAL PERSONAL FACTS stated as context. These facts are just as extractable as the request itself:

- "I've harvested cherry tomatoes from my garden — any companion plant suggestions?" → Extract BOTH "User grows cherry tomatoes in their garden"
- "I just started 'The Nightingale' by Kristin Hannah — can you recommend similar books?" → Extract BOTH "User started reading 'The Nightingale' by Kristin Hannah on [date]"
- "As an aspiring stand-up comedian, can you suggest Netflix comedy specials?" → Extract BOTH the career aspiration
- "My daughter Sara loves painting — where can I find kids' art classes?" → Extract "User has a daughter named Sara who loves painting"

Do NOT let the request overshadow the facts. A question about companion plants is transient; the fact that the user grows cherry tomatoes is a persistent personal detail worth remembering.

**IMPORTANT — Extract ALL dimensions of a conversation.** A single session may contain career facts, entertainment preferences, scheduled plans, and personal opinions. Extract each dimension as a separate memory. Do not let one dominant topic cause you to miss secondary information.

### Shared Photos and Images

When a message contains a photo description (e.g., "[Shared photo: ...]" or describes sharing/showing an image), extract factual information from BOTH the surrounding conversation text AND the photo description. The photo description provides visual context that may contain important details:

- A photo of a group at a park → extract the activity (e.g., "had a picnic at the park")
- A photo showing a specific object, place, or person → extract what is depicted
- A photo with visible text (signs, posters, book covers) → extract the text content

## Memory Quality Standards

### Contextually Rich, Not Atomic
Capture the full picture — fact AND surrounding context — in a single unified memory, not scattered fragments.

Bad: "User has a dog" | Good: "User has a dog named Poppy and their morning walks together are the highlight of their day"

This applies especially to **transitions and changes**. When the user describes changing, switching, replacing, stopping, or trying something new in place of something else, the memory MUST capture the transition — what the new state is AND what it replaces or changes from. The relationship between old and new is critical context. Without it, the system has an isolated new fact with no understanding of what changed.

Bad: "User prefers oat milk lattes"
Good: "User switched from almond milk to oat milk lattes after developing an almond sensitivity"

Bad: "User is taking online Spanish classes on Wednesdays"
Good: "User switched from in-person French classes to online Spanish classes on Wednesdays after relocating"

When the change is explicitly temporary or a trial, capture that too — "for a month", "trying out", "testing" — these signal the old arrangement may resume.

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
Preserve exact dates, durations, and temporal relationships. Convert relative → absolute using the timestamp from the message. NEVER convert absolute → vague. "18 days" stays "18 days", not "some time."

### Numerically Precise
Preserve exact quantities as stated. "416 pages" stays "416 pages", not "about 400 pages."

### Preserve Specific Details — Never Generalize Concrete Information

When information contains specific details — whether quantities, identifiers, descriptions, visual details, quoted text, named objects, proper nouns, or any concrete information — those specifics MUST survive extraction. Replacing a specific detail with a vague category is a critical error.

#### Proper Nouns and Titles Should be Preserved

Book titles, movie titles, game names, song titles, restaurant names, neighborhood names, brand names, character names, and named places are the HIGHEST-VALUE details in a memory. Users search by name — a memory without the name is unfindable. ALWAYS preserve exact proper nouns:

- "watched 'Eternal Sunshine of the Spotless Mind'" → KEEP the full title
- "went to Woodhaven for a road trip" → KEEP "Woodhaven"
- "tried the new restaurant Osteria Francescana" → KEEP "Osteria Francescana", NOT "a new restaurant"
- "reading 'A Court of Thorns and Roses'" → KEEP the title in quotes, NOT "a fantasy book"
- "his favorite character is Aragorn from Lord of the Rings" → KEEP "Aragorn" and "Lord of the Rings"

#### Qualifiers and Specific Attributes Are Essential

Never generalize specific qualifiers. The qualifier is almost always the detail that matters most for recall:

- "promoted to assistant manager" → KEEP "assistant manager", NOT "manager"
- "ordered grilled salmon and roasted vegetables" → KEEP "grilled salmon and roasted vegetables", NOT "healthy meal"
- "started doing aerial yoga" → KEEP "aerial yoga", NOT "yoga" or "a workout class"
- "painted a forest scene in watercolors" → KEEP "a forest scene in watercolors", NOT "started painting"
- "drove a Ferrari 488 GTB" → KEEP "Ferrari 488 GTB", NOT "sports car"
- "scored 3 goals in the semifinal" → KEEP "3 goals in the semifinal", NOT "scored several goals"
- "walks her dogs multiple times a day" → KEEP "multiple times a day", NOT "regularly" or "daily"

If the input is specific, the memory must be equally specific. The concrete details are precisely what distinguishes a useful memory from a useless one. NEVER replace a specific noun, number, title, or description with a vague category or paraphrase — this destroys the information the user actually shared.

### Meaning-Preserving
Capture the EXACT meaning of what was said. Read carefully:
- "Didn't get to bed until 2 AM" = went TO BED at 2 AM (late bedtime), NOT "slept until 2 AM" (late wakeup)
- "Can't stop eating chocolate" = eats a lot of chocolate, NOT has stopped eating chocolate
- "I used to love hiking" = no longer loves hiking, NOT currently loves hiking

Misinterpreting the user's words is worse than not extracting at all.

## Integrity Rules

- **No Fabrication**: Every detail must trace to the inputs. If you can't point to where it came from, don't include it.
- **No Implicit Attribute Inference**: Don't infer gender, age, ethnicity, etc. from names or context. Only record explicitly stated attributes.
- **Correct Attribution**: Distinguish user-stated facts from assistant-provided information. Frame assistant content appropriately.
- **No Echo Extraction**: When an assistant message restates, summarizes, or confirms information the user already provided in the same conversation, do NOT extract it again from the assistant's message. Only extract from assistant messages when they contribute genuinely NEW information not already present in the user's messages — specific recommendations, newly created plans or schedules, researched facts, or solutions the assistant provided that the user did not state themselves. If the user says "I want daily check-ins at 7:30 AM" and the assistant responds "I've set up daily check-ins at 7:30 AM", that is already captured from the user's message — do not extract a second memory from the assistant's echo.
- **No Within-Response Duplication**: Each piece of information must appear exactly ONCE in your output, regardless of how many messages mention it. Before finalizing your output, review your extractions and remove any that are semantically equivalent to another extraction in the same response. Two memories about the same fact phrased differently are redundant — keep the richer one and drop the other.
- **No Meta-Extraction**: Extract the CONTENT of what was shared, not a description of the user's action. When a user shares a document, data, or reference material, extract the actual facts FROM that material.
  - WRONG: "User asked for the introductory paragraph to be shortened" / "User shared a case summary for optimization"
  - RIGHT: "The Bajimaya v Reward Homes case involved construction starting in 2014, contract signed in 2015, with completion due by October 2015" / "The tribunal found Reward Homes breached its contract through poor workmanship, waterproofing defects, and non-compliance with the Building Code of Australia"
  - WRONG: "Assistant created a D&D adventure with enemies"
  - RIGHT: "The Lost Temple of the Djinn adventure includes 4 Mummies (AC 11, 45 HP), 2 Construct Guardians (AC 17, 110 HP), and 6 Skeletal Warriors (AC 12, 22 HP)"
- **No Detail Contamination from Context**: When extracting from New Messages, do NOT import or merge details from other memories into the new extraction UNLESS the new message explicitly references those details. If the New Message says "I had a great meal" and an existing memory says "User's favorite restaurant is Olive Garden," do NOT produce "User had a great meal at Olive Garden" — the new message never mentioned the restaurant. Each extraction must be faithful to its source message only.

# EXAMPLES

## Example 1: Multi-Topic Extraction

New Messages:
[{"role": "user", "content": "Hey! I'm Marcus. I just got promoted to Senior Engineer at Shopify last week - been grinding for two years for this. My wife Elena and I celebrated with dinner at Osteria Francescana, it's our go-to spot for special occasions. We're also expecting our first baby in March!"},
 {"role": "assistant", "content": "Congratulations on everything, Marcus! What exciting times."}]

Output:
[
  {"fact": "User's name is Marcus and was promoted to Senior Engineer at Shopify around August 12, 2025 after working toward it for two years", "attributed_to": "user"},
  {"fact": "Marcus has a wife named Elena and they celebrate special occasions at Osteria Francescana, their go-to restaurant", "attributed_to": "user"},
  {"fact": "Marcus and his wife Elena are expecting their first baby in March 2026", "attributed_to": "user"}
]

Three distinct topics -- career, relationship/dining, family milestone -- each get their own fact with full context.

## Example 2: Extracting from Assistant Recommendations

New Messages:
[{"role": "user", "content": "Can you recommend some sports documentaries on Netflix with strong storytelling? I love The Last Dance by Michael Jordan."},
 {"role": "assistant", "content": "Great taste! Here are some Netflix documentaries known for their storytelling: 1) Formula 1: Drive to Survive (behind the scenes of Formula 1 racing) 2) Athlete A (investigative look at USA Gymnastics) 3) The Battered Bastards of Baseball (independent baseball story). All focus on powerful, narrative-driven sports stories."}]

Output:
[
  {"fact": "User enjoys watching sports documentaries on Netflix with strong storytelling, such as The Last Dance featuring Michael Jordan", "attributed_to": "user"},
  {"fact": "User was recommended the following sports documentaries on Netflix for storytelling: Formula 1: Drive to Survive, Athlete A, and The Battered Bastards of Baseball", "attributed_to": "assistant"}
]

The user's viewing preference is extracted alongside the assistant's specific recommendations. Both are valuable for future personalization.

## Example 3: Nothing to Extract

New Messages:
[{"role": "user", "content": "Hey, good morning!"},
 {"role": "assistant", "content": "Good morning, David! How can I help you today?"}]

Output: []

## Example 4: Extract ALL Dimensions -- Don't Miss Secondary Info

New Messages:
[{"role": "user", "content": "As an aspiring actor, I'm looking for advice on improving my craft. Can you recommend some films on Netflix with strong acting performances like Daniel Day-Lewis in There Will Be Blood? I also want to find online resources for acting techniques."},
 {"role": "assistant", "content": "For Netflix films with great acting, check out Marriage Story and The Irishman. For acting techniques, I'd recommend An Actor Prepares by Stanislavski and the MasterClass by Helen Mirren."}]

Output:
[
  {"fact": "User is an aspiring actor seeking to improve their craft through studying films with strong performances and acting technique resources", "attributed_to": "user"},
  {"fact": "User enjoys watching films on Netflix with outstanding acting, especially performances like Daniel Day-Lewis in There Will Be Blood", "attributed_to": "user"},
  {"fact": "User was recommended Marriage Story and The Irishman for performance study, An Actor Prepares by Stanislavski, and Helen Mirren's MasterClass for acting techniques", "attributed_to": "assistant"}
]

Three dimensions: (1) career aspiration, (2) entertainment viewing preference, (3) specific recommendations. Each extracted separately.

## Example 5: Document / Reference Material -- Extract Content, Not Actions

New Messages:
[{"role": "user", "content": "I want you to remember this case. If you understand, just say acknowledged. Bajimaya v Reward Homes Pty Ltd [2021] NSWCATAP 297 -- The construction began in 2014, contract signed in 2015 with completion due by October 2015. The plaintiff received keys in December 2016 and found defects including incomplete works, poor workmanship, and non-compliance with the building code. The tribunal found the builder breached contract."},
 {"role": "assistant", "content": "Acknowledged."}]

Output:
[
  {"fact": "Bajimaya v Reward Homes Pty Ltd [2021] NSWCATAP 297: construction of the home began in 2014, contract signed in 2015, with completion due by October 2015. Keys were delivered in December 2016.", "attributed_to": "user"},
  {"fact": "In Bajimaya v Reward Homes, the plaintiff found defects including incomplete works, poor workmanship, and non-compliance with the Building Code of Australia after receiving the home in December 2016.", "attributed_to": "user"},
  {"fact": "The tribunal found Reward Homes Pty Ltd breached its contract with Mr. Bajimaya by failing to complete work in a proper and workmanlike manner and failing to comply with plans, specifications, and the Building Code.", "attributed_to": "user"}
]

The user shared reference material to be remembered. Extract the actual factual content -- dates, parties, findings -- NOT "User shared a case summary" or "User asked to remember a case."

Output a flat JSON array of objects with {fact} and {attributed_to} keys.
""".strip()

_GROUP_INSTRUCTION = """
Rows refer to the same durable memory fact when their {fact}
expresses information about the same subject, entity, event, or topic. Use
semantic identity — same person, same event, same preference, same plan —
not surface wording, chronology, or exact string equality. Facts may disagree
or contradict and still belong in the same group; the downstream consolidation
step will resolve the conflict.
""".strip()

_CONSOLIDATE_INSTRUCTION = """
You are a memory consolidation assistant. Your task is to merge a group of semantically related facts into a single canonical factual statement.

The group contains one or more rows that refer to the same subject, entity, event, or topic. Produce one consolidated fact that represents the canonical state of this memory.

Guidelines:
1. Merge complementary details into a single rich, self-contained statement.
2. If facts agree, combine them -- preserve the most complete version.
3. If facts disagree or contradict, prefer the most recent or most specific information based on provided timestamps and evidence.
4. If a fact is stale, wrong, or superseded, drop it from the consolidated output.
5. Preserve specific details -- proper nouns, dates, quantities, and qualifiers must survive.
6. Keep the statement concise (15-80 words) and self-contained.
7. Do NOT output JSON. Do NOT wrap in an object. Output only the plain consolidated fact string.

Output one plain string: the canonical consolidated fact.
""".strip()


class Mem0Memory(Memory):
    """Built-in Mem0-style memory policy."""

    log = Log(
        {
            "message": "Raw user, assistant, or tool memory event text.",
            "role": "Message role such as user, assistant, system, or tool.",
            "timestamp": "Event timestamp.",
            "session_id": "Conversation or session identifier.",
            "metadata": "Optional structured metadata for the memory event.",
        }
    )

    facts = (
        log
        .sem_flat_map(
            output_cols={
                "fact": "Self-contained, contextually rich factual statement.",
                "attributed_to": "Speaker attribution: user or assistant.",
            },
            instruction=_EXTRACT_INSTRUCTION,
        )
        .select(
            [
                "fact",
                "attributed_to",
                "message",
                "role",
                "timestamp",
                "session_id",
                "metadata",
            ]
        )
        .sem_groupby(
            input_cols=["fact"],
            instruction=_GROUP_INSTRUCTION,
        )
        .sem_agg(
            input_cols=[
                "fact",
                "attributed_to",
                "message",
                "role",
                "timestamp",
                "session_id",
                "metadata",
            ],
            output_cols={
                "fact": "Canonical consolidated fact statement.",
            },
            instruction=_CONSOLIDATE_INSTRUCTION,
        )
        .select(["fact"])
    )

    retrieval_query = facts.sem_topk(UserQuery(), 10)
