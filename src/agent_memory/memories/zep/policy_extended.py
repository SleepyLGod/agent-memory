"""Extended Zep/Graphiti-style temporal memory policy with communities."""

from __future__ import annotations

from agent_memory.policy.aggregates import array_agg, min, sem_agg
from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.expressions import least
from agent_memory.policy.retrieval import (
    BFS,
    BM25,
    RRF,
    CosineSimilarity,
    CrossEncoder,
    RetrievalQuery,
)


# Match Graphiti's extraction context depth for sequential episode ingestion.
_EPISODE_WINDOW_LEN = 10


# Adapted from Graphiti graphiti_core/prompts/extract_nodes.py::extract_message.
_ENTITY_EXTRACT_INSTRUCTION = """
You are an entity extraction specialist for conversational messages. Never
extract abstract concepts, feelings, or generic words.

Extract entity nodes explicitly mentioned in the CURRENT MESSAGE {content}.
Return one row per distinct entity with {name}. Use PREVIOUS MESSAGES
{previous_episodes} only to resolve references; never extract an entity that
appears only in previous messages.

The only entity type is Entity: a specific, identifiable entity that does not
fit a more specialized type. It must still be a concrete, meaningful thing that
is specific enough to be uniquely identifiable. GOOD: a named entity not
covered by another type. BAD: luck, ideas, tomorrow, things, them, everybody, a
sense of wonder, or great times. When in doubt, do not extract the entity.

NEVER extract any of the following:
- Pronouns such as you, me, I, he, she, they, we, us, it, them, him, her,
  this, that, or those. Resolve references to explicit names instead.
- Abstract concepts or feelings such as joy, balance, growth, resilience,
  happiness, passion, or motivation.
- Generic common nouns or bare object words such as day, life, people, work,
  stuff, things, food, time, way, tickets, supplies, clothes, keys, or gear.
- Generic media or content nouns unless uniquely identified in the name itself,
  such as photo, pic, picture, image, video, post, or story.
- Generic event or activity nouns unless uniquely identified in the name
  itself, such as event, game, meeting, class, workshop, or competition.
- Broad institutional nouns unless explicitly named or uniquely qualified,
  such as government, school, company, team, or office.
- Ambiguous bare nouns whose meaning depends on sentence context rather than
  the entity name itself.
- Sentence fragments or clauses such as "what you really care about" or
  "results of that effort".
- Adjectives or descriptive fragments such as "amazing", "something
  different", or "new hair color".
- Duplicate references to the same real-world entity. Return each entity at
  most once per message, even when it appears as both speaker and body text.
- Bare relational or kinship terms such as dad, mom, mother, father, sister,
  brother, husband, wife, spouse, son, daughter, uncle, aunt, cousin, grandma,
  grandpa, friend, boss, teacher, neighbor, or roommate, and bare animal or pet
  terms such as dog, cat, pet, puppy, or kitten. Qualify them with the possessor
  when the message supports it, for example "Nisha's dad" or "Jordan's dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor,
  brand, or distinguishing detail, such as "supplies" in "I picked up some
  supplies".

Rules:
1. Always extract the speaker, the text before the first colon, as the first
   entity. If the speaker appears again, return it only once.
2. Extract named entities and specific concrete things only when their names
   can identify them later. Ask whether the entity could have its own database
   entry or is distinguishable from other things of the same category in this
   conversation.
3. Extract brand-named items such as Gamecube, Ford Mustang, or Moen faucet;
   qualified items such as wool coat, red and purple lighting, cracked
   windshield, or dog leash; and objects with a concrete color, material, size,
   model, owner, or use. Do not extract bare heads such as car, coat, game,
   lighting, or windshield.
4. When a named person refers to a relative, pet, or associate with a bare
   term, qualify it using the possessor when possible. Do not return the bare
   term alone.
5. Do not extract relationships, actions, dates, times, or other temporal
   information.
6. Use the most specific form in the message: "road cycling" rather than
   "cycling", "wool coat" rather than "coat", and "dog leash" rather than
   "leash" when context establishes the object type.
7. Use explicit, unambiguous names and full names when available.
   When in doubt, do not extract.

<EXAMPLE>
Message: "Jordan: We just moved to Denver last month. My spouse started a new
role at Lockheed Martin and I enrolled in a ceramics workshop at the Belmont
Arts Center."
Good extractions: Jordan, Denver, Lockheed Martin, Belmont Arts Center, and
ceramics.
Do not extract: spouse, new role, last month, or we.
</EXAMPLE>

<EXAMPLE>
Message: "Nisha: My dad is visiting next week. He loves walking his dogs in
Riverside Park."
Good extractions: Nisha, Nisha's dad, and Riverside Park.
Do not extract: dad, dogs, or next week.
</EXAMPLE>

<EXAMPLE>
Message: "Mary: I forgot Trigger's leash so I couldn't take him on a dog walk.
After that I went road cycling in my new wool coat."
Good extractions: Mary, Trigger, dog leash, road cycling, and wool coat.
Do not extract: leash, cycling, coat, or dog walk.
</EXAMPLE>

<EXAMPLE>
Message: "Nate: My gaming room has red and purple lighting and I mostly play on
a Gamecube. Last week the windshield on my Mustang got cracked."
Good extractions: Nate, gaming room, red and purple lighting, Gamecube,
Mustang, and cracked windshield.
Do not extract: lighting, windshield, or week.
</EXAMPLE>

<EXAMPLE>
Message: "Alex: I shared a pic from the game after the event."
Good extraction: Alex.
Do not extract: pic, game, or event.
</EXAMPLE>

<EXAMPLE>
Message: "Jordan: We won by a tight score. Scoring that last basket felt
incredible."
Good extraction: Jordan.
Do not extract: basket.
</EXAMPLE>
""".strip()


_ENTITY_GROUP_INSTRUCTION = """
Rows refer to the same real-world entity when their {name} describes the same
object, actor, organization, place, or concept.
""".strip()


# Adapted from Graphiti's entity summary update prompt. Candidate retrieval and
# top-1 identity resolution remain outside this declarative approximation.
_ENTITY_SUMMARY_INSTRUCTION = """
Create one canonical entity from the grouped mentions. Return canonical {name}
and a concise, information-dense {summary}. Use only durable facts explicitly
supported by {content}; never infer beyond the evidence.

Choose the canonical name only from the grouped {name} values. Preserve the
original name unless another grouped mention provides a more complete name.
Never derive the entity name from the speaker or other entities in {content}.

Preserve material names, roles, relationships, dates, counts, concrete details,
and changes over time. Prefer newer explicit facts when they conflict with old
facts. Each input row includes add_seq; a larger add_seq means the evidence was
ingested later. Do not mention messages, episodes, prompts, summaries, graphs,
nodes, labels, schemas, or the summarization process. State facts directly
rather than saying they were mentioned. Do not invent preferences, habits,
recurrence, causality, or intent from a single weak observation.
""".strip()


# Adapted from Graphiti graphiti_core/prompts/extract_edges.py::edge.
_FACT_EXTRACT_INSTRUCTION = """
You are an expert fact extractor that extracts factual relationship triples
with relevant date information.

Extract every factual relationship stated in the CURRENT MESSAGE {content}
between the resolved ENTITIES {entities}. Each listed entity has an integer
entity_ordinal. Return one row per relationship with {source_entity_ordinal},
{target_entity_ordinal}, {relation_type}, {fact}, {valid_at}, and {invalid_at}.

Use PREVIOUS MESSAGES {previous_episodes} only to disambiguate references or
support continuity. Use REFERENCE TIME {reference_time} to resolve temporal
expressions in the current message.

Only extract facts that:
- involve two distinct entities from the provided ENTITIES list;
- are clearly stated or unambiguously implied in the CURRENT MESSAGE; and
- can be represented as an edge between those entities.

Extraction rules:
1. {source_entity_ordinal} and {target_entity_ordinal} must be integer ordinals
   from the provided ENTITIES list. An out-of-range ordinal makes the fact
   invalid.
2. The source and target must identify two distinct entities.
   Never emit a self-loop.
3. Prefer facts involving two explicit entities. When a sentence gives a
   specific concrete detail about one entity, such as a brand, item, physical
   description, quantity, location, or named activity, do not drop it. Use a
   second entity from the list to anchor a proper relationship when one exists.
   Skip it only when no second entity can anchor the detail.
   BAD: "Alice feels happy" is a vague single-entity state.
   GOOD: "Alice feels happy about Bob's promotion" relates Alice to Bob's
   promotion.
   GOOD: "Nate plays games on a Gamecube" relates Nate to Gamecube when
   Gamecube is in ENTITIES.
   GOOD: "Alice congratulated Bob" and "Alice lives in Paris" each relate two
   explicit entities.
4. Prefer entity names over pronouns in {fact}. Do not emit semantically
   redundant facts. A later claim with additional concrete detail is a new fact,
   not a duplicate.
   NOT A DUPLICATE: "user plays video games" and "user plays games on a
   Gamecube" differ in specificity.
   DUPLICATE: "user plays games on a Gamecube" and "user plays Gamecube games"
   express the same specific fact.
5. Preserve every supported proper noun, brand, product, model, quantity,
   count, color, material, physical description, item, named location, and
   named activity. Paraphrase the sentence structure without generalizing any
   concrete detail.
   Never generalize Gamecube to gaming console, Ford Mustang to car, wool coat
   to coat, red and purple lighting to lighting, cracked windshield to car
   damage, or three screenplays to several screenplays.
6. Derive {relation_type} from the relationship predicate in
   SCREAMING_SNAKE_CASE, for example WORKS_AT, LIVES_IN, or IS_FRIENDS_WITH.
7. Use ISO 8601 with a Z suffix for temporal bounds. Resolve relative
   expressions against {reference_time}. For an ongoing present-tense fact, set
   {valid_at} to {reference_time}. Set {invalid_at} only when a change or
   termination is expressed. Assume midnight for a date-only value and January
   1 at midnight for a year-only value. Return JSON null when a temporal bound
   is not explicitly stated or resolvable.
   Never infer dates from unrelated events.
""".strip()


# Adapted from Graphiti graphiti_core/prompts/dedupe_edges.py::resolve_edge.
_FACT_GROUP_INSTRUCTION = """
Group rows only when {relation_type} and {fact} express identical factual
information. Never group facts with key differences in numbers, dates, concrete
qualifiers, specificity, or relationship meaning. "Plays video games" and
"plays games on a Gamecube" are not duplicates; "plays games on a Gamecube"
and "plays Gamecube games" are duplicates.
""".strip()


_FACT_CANONICAL_INSTRUCTION = """
Create one canonical fact row from duplicate claims. Return canonical
{relation_type} and faithful {fact}. Preserve every supported concrete detail.
Do not generalize the claim and do not merge contradictory or more-specific
claims into one statement.
""".strip()


# Pairwise form of Graphiti's contradiction decision. Temporal direction is
# applied deterministically after this semantic predicate.
_CONTRADICTORY_FACT_INSTRUCTION = """
Does {fact:later_added} explicitly contradict, supersede, or make
{fact:earlier_added} stop being true? Return false for merely related facts or
facts that can both be true at the same time. Different events, dates, counts,
or qualifiers are not automatically contradictions. A changed role or other
updated value for the same relationship may be a contradiction even when it is
not a duplicate.
""".strip()


_COMMUNITY_GROUP_INSTRUCTION = """
Rows belong to the same high-level community when {name} and {summary} describe
the same coherent cluster of related entities.
""".strip()


_COMMUNITY_SUMMARY_INSTRUCTION = """
Synthesize one concise community from the grouped entity summaries and facts.
Return a short {name} and concise {summary}. The supporting members are retained
deterministically by the query.

This is a declarative approximation of Graphiti community maintenance. It does
not implement Graphiti's label-propagation fixpoint.
""".strip()


class ZepMemoryExtended(Memory):
    """Declarative Zep/Graphiti memory with temporal facts and communities.

    This policy describes logical memory state. It intentionally omits
    Graphiti's hybrid candidate search, top-1 entity resolver, graph storage,
    and exact community label-propagation lowering.
    """

    # Episodes

    log = Log(
        {
            "content": "Raw episode content.",
            "role": "Conversation role or source role.",
            "speaker": "Speaker name when available.",
            "reference_time": "Event reference time as an ISO 8601 timestamp.",
            "source_description": "Optional source description.",
        },
        system_columns=True,
    )
    episodes = log.assign(
        episode_id=log.col("_row_id"),
        created_at=log.col("_added_at"),
        add_seq=log.col("_add_seq"),
    ).select(
        [
            "episode_id",
            "content",
            "role",
            "speaker",
            "reference_time",
            "source_description",
            "created_at",
            "add_seq",
        ]
    )

    # Each row is one episode with its previous-episode context.
    _windowed_episodes = episodes.over(rows=(-_EPISODE_WINDOW_LEN, -1)).array_agg(
        columns=[
            "episode_id",
            "content",
            "role",
            "speaker",
            "reference_time",
            "created_at",
            "add_seq",
        ],
        output_col="previous_episodes",
    )

    # Entities

    # Each row is one raw entity extracted from one episode.
    _extracted_entities = (
        _windowed_episodes.sem_flat_map(
            input_cols=["content", "previous_episodes"],
            output_cols={
                "name": "Extracted entity name.",
            },
            instruction=_ENTITY_EXTRACT_INSTRUCTION,
            ordinal_col="entity_ordinal",
        )
        .assign(entity_type="Entity")
        .select(
            [
                "episode_id",
                "content",
                "created_at",
                "add_seq",
                "entity_ordinal",
                "name",
                "entity_type",
            ]
        )
    )

    entities = (
        _extracted_entities.sem_groupby(
            input_cols=["name"],
            instruction=_ENTITY_GROUP_INSTRUCTION,
            membership="exclusive",
        )
        .agg(
            sem_agg(
                input_cols=["name", "content", "add_seq"],
                output_cols={
                    "name": "Canonical entity name.",
                    "summary": "Concise entity summary.",
                },
                instruction=_ENTITY_SUMMARY_INSTRUCTION,
            ),
            array_agg(
                columns=[
                    "episode_id",
                    "entity_ordinal",
                    "name",
                    "entity_type",
                    "content",
                    "created_at",
                    "add_seq",
                ],
                output_col="mentions",
            ),
            min(
                columns=["add_seq", "entity_ordinal"],
                output_col="entity_id",
            ),
        )
        .assign(entity_type="Entity")
        .select(["entity_id", "name", "entity_type", "summary", "mentions"])
    )

    # Each row is one entity occurrence in one episode.
    _episode_entities = (
        entities.explode(column="mentions")
        .unnest(
            column="mentions",
            fields={
                "episode_id": "episode_id",
                "entity_ordinal": "entity_ordinal",
            },
        )
        .select(
            [
                "episode_id",
                "entity_ordinal",
                "entity_id",
            ]
        )
    )

    # Facts

    # Each row is one extracted fact with canonical source and target entity IDs.
    _extracted_facts = (
        _windowed_episodes.join(
            _extracted_entities.group_by("episode_id").array_agg(
                columns=[
                    "entity_ordinal",
                    "name",
                    "entity_type",
                ],
                output_col="entities",
            ),
            on="episode_id",
        )
        .sem_flat_map(
            input_cols=[
                "content",
                "reference_time",
                "previous_episodes",
                "entities",
            ],
            output_cols={
                "source_entity_ordinal": "Integer source entity ordinal from entities.",
                "target_entity_ordinal": "Integer target entity ordinal from entities.",
                "relation_type": "Relationship predicate label.",
                "fact": "Natural-language fact statement.",
                "valid_at": "When this fact became true, if known.",
                "invalid_at": "When this fact stopped being true, if known.",
            },
            instruction=_FACT_EXTRACT_INSTRUCTION,
            ordinal_col="fact_ordinal",
        )
        .select(
            [
                "episode_id",
                "fact_ordinal",
                "source_entity_ordinal",
                "target_entity_ordinal",
                "relation_type",
                "fact",
                "valid_at",
                "invalid_at",
                "content",
                "created_at",
                "add_seq",
            ]
        )
        .join(
            _episode_entities.assign(
                source_entity_ordinal=_episode_entities.col("entity_ordinal"),
                source_entity_id=_episode_entities.col("entity_id"),
            ).select(
                ["episode_id", "source_entity_ordinal", "source_entity_id"]
            ),
            on=["episode_id", "source_entity_ordinal"],
        )
        .join(
            _episode_entities.assign(
                target_entity_ordinal=_episode_entities.col("entity_ordinal"),
                target_entity_id=_episode_entities.col("entity_id"),
            ).select(
                ["episode_id", "target_entity_ordinal", "target_entity_id"]
            ),
            on=["episode_id", "target_entity_ordinal"],
        )
    )
    # Graphiti drops an edge when both mentions resolve to the same entity.
    _extracted_facts = (
        _extracted_facts.filter(
            _extracted_facts.col("source_entity_id")
            != _extracted_facts.col("target_entity_id")
        )
        .select(
            [
                "episode_id",
                "fact_ordinal",
                "source_entity_ordinal",
                "target_entity_ordinal",
                "source_entity_id",
                "target_entity_id",
                "relation_type",
                "fact",
                "valid_at",
                "invalid_at",
                "content",
                "created_at",
                "add_seq",
            ]
        )
    )

    # Each row is one deduplicated claim with provenance and temporal state.
    _deduplicated_facts = (
        _extracted_facts.filter(_extracted_facts.col("invalid_at").is_null())
        .assign(expired_at=None)
        .union_by_name(
            _extracted_facts.filter(
                _extracted_facts.col("invalid_at").is_not_null()
            ).assign(expired_at=_extracted_facts.col("created_at"))
        )
        .sem_groupby(
            input_cols=["relation_type", "fact"],
            partition_by=["source_entity_id", "target_entity_id"],
            instruction=_FACT_GROUP_INSTRUCTION,
            membership="exclusive",
        )
        .agg(
            sem_agg(
                input_cols=["relation_type", "fact"],
                output_cols={
                    "relation_type": "Canonical relationship predicate.",
                    "fact": "Canonical factual statement.",
                },
                instruction=_FACT_CANONICAL_INSTRUCTION,
            ),
            array_agg(
                columns=[
                    "episode_id",
                    "fact_ordinal",
                    "created_at",
                    "content",
                ],
                output_col="provenance",
            ),
            min(
                columns=["add_seq", "fact_ordinal"],
                output_col="fact_id",
            ),
            min(column="valid_at", output_col="valid_at"),
            min(column="invalid_at", output_col="invalid_at"),
            min(column="expired_at", output_col="expired_at"),
            min(column="created_at", output_col="created_at"),
            min(column="add_seq", output_col="add_seq"),
        )
        .select(
            [
                "fact_id",
                "source_entity_id",
                "target_entity_id",
                "relation_type",
                "fact",
                "valid_at",
                "invalid_at",
                "expired_at",
                "provenance",
                "created_at",
                "add_seq",
            ]
        )
    )

    # These aliases give the two self-join inputs explicit ingestion-order roles.
    _earlier_added_facts = _deduplicated_facts.alias("earlier_added")
    _later_added_facts = _deduplicated_facts.alias("later_added")

    # Each row is one pair of facts judged to contradict each other.
    _contradictory_fact_pairs = _earlier_added_facts.join(
        _later_added_facts,
        on=[
            _earlier_added_facts.col("fact_id")
            < _later_added_facts.col("fact_id"),
        ],
    ).sem_filter(instruction=_CONTRADICTORY_FACT_INSTRUCTION)

    # Each row records the earliest invalidation boundary for one fact.
    _fact_invalidations = (
        _contradictory_fact_pairs.filter(
            _contradictory_fact_pairs.col("valid_at:earlier_added").is_not_null()
            & _contradictory_fact_pairs.col("valid_at:later_added").is_not_null()
            & (
                _contradictory_fact_pairs.col("valid_at:earlier_added")
                < _contradictory_fact_pairs.col("valid_at:later_added")
            )
        )
        .assign(
            fact_id=_contradictory_fact_pairs.col("fact_id:earlier_added"),
            invalid_at=_contradictory_fact_pairs.col("valid_at:later_added"),
            expired_at=_contradictory_fact_pairs.col("created_at:later_added"),
        )
        .select(["fact_id", "invalid_at", "expired_at"])
        .union_by_name(
            _contradictory_fact_pairs.filter(
                _contradictory_fact_pairs.col("valid_at:earlier_added").is_not_null()
                & _contradictory_fact_pairs.col("valid_at:later_added").is_not_null()
                & (
                    _contradictory_fact_pairs.col("valid_at:later_added")
                    < _contradictory_fact_pairs.col("valid_at:earlier_added")
                )
            )
            .assign(
                fact_id=_contradictory_fact_pairs.col("fact_id:later_added"),
                invalid_at=_contradictory_fact_pairs.col("valid_at:earlier_added"),
                expired_at=_contradictory_fact_pairs.col("created_at:later_added"),
            )
            .select(["fact_id", "invalid_at", "expired_at"])
        )
        .group_by("fact_id")
        .agg(
            min(column="invalid_at", output_col="invalid_at"),
            min(column="expired_at", output_col="expired_at"),
        )
    )

    # Each row is a deduplicated fact plus its optional contradiction boundary.
    _facts_with_invalidations = _deduplicated_facts.alias("fact").join(
        _fact_invalidations.alias("invalidation"),
        on="fact_id",
        how="left",
    )

    facts = _facts_with_invalidations.assign(
        invalid_at=least(
            _facts_with_invalidations.col("invalid_at:fact"),
            _facts_with_invalidations.col("invalid_at:invalidation"),
        ),
        expired_at=least(
            _facts_with_invalidations.col("expired_at:fact"),
            _facts_with_invalidations.col("expired_at:invalidation"),
        ),
    ).select(
        [
            "fact_id",
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            "fact",
            "valid_at",
            "invalid_at",
            "expired_at",
            "provenance",
            "created_at",
            "add_seq",
        ]
    )

    # Communities

    communities = (
        facts.assign(entity_id=facts.col("source_entity_id"))
        .select(
            [
                "entity_id",
                "fact_id",
                "relation_type",
                "fact",
                "valid_at",
                "invalid_at",
            ]
        )
        .union_by_name(
            facts.assign(entity_id=facts.col("target_entity_id")).select(
                [
                    "entity_id",
                    "fact_id",
                    "relation_type",
                    "fact",
                    "valid_at",
                    "invalid_at",
                ]
            )
        )
        .drop_duplicates()
        .join(entities, on="entity_id")
        .sem_groupby(
            input_cols=["name", "summary"],
            instruction=_COMMUNITY_GROUP_INSTRUCTION,
            membership="exclusive",
        )
        .agg(
            sem_agg(
                input_cols=[
                    "entity_id",
                    "name",
                    "summary",
                    "fact_id",
                    "relation_type",
                    "fact",
                    "valid_at",
                    "invalid_at",
                ],
                output_cols={
                    "name": "Short community name.",
                    "summary": "Concise community summary.",
                },
                instruction=_COMMUNITY_SUMMARY_INSTRUCTION,
            ),
            array_agg(
                columns=[
                    "entity_id",
                    "name",
                    "summary",
                    "fact_id",
                    "relation_type",
                    "fact",
                    "valid_at",
                    "invalid_at",
                ],
                output_col="members",
            ),
            min(column="entity_id", output_col="community_id"),
        )
        .select(["community_id", "name", "summary", "members"])
    )

    # Retrieval

    _retrieved_entities = entities.search(
        UserQuery(),
        methods=[BM25(), CosineSimilarity()],
        reranker=RRF(),
        limit=20,
    ).select(["record_id", "name", "summary", "rank", "score"])

    retrieval_query = RetrievalQuery(
        entities=_retrieved_entities,
        facts=facts.search(
            UserQuery(),
            methods=[
                BM25(),
                CosineSimilarity(),
                BFS(origins=_retrieved_entities, max_depth=3),
            ],
            reranker=CrossEncoder(model="BAAI/bge-reranker-v2-m3"),
            limit=20,
        ).select(
            [
                "record_id",
                "fact",
                "valid_at",
                "invalid_at",
                "expired_at",
                "rank",
                "score",
            ]
        ),
    )
