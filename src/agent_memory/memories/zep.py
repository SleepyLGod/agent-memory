"""Zep/Graphiti-style temporal memory policy."""

from __future__ import annotations

from agent_memory.policy.aggregates import array_agg, min, sem_agg
from agent_memory.api import Log, Memory
from agent_memory.policy.expressions import least


_EPISODE_WINDOW_LEN = 3


# Adapted from Graphiti graphiti_core/prompts/extract_nodes.py::extract_message.
_ENTITY_EXTRACT_INSTRUCTION = """
Extract every entity explicitly or implicitly mentioned in the current episode
{content}. Return one row per entity with {name} and {entity_type}.

Use {previous_episodes} only to resolve references. Do not extract entities that
appear only in previous episodes. Avoid pronouns as entity names; use explicit,
unambiguous names.
""".strip()


_ENTITY_GROUP_INSTRUCTION = """
Rows refer to the same real-world entity when their {name} and {entity_type}
describe the same object, actor, organization, place, or concept.
""".strip()


# Adapted from Graphiti's entity summary update prompt. Candidate retrieval and
# top-1 identity resolution remain outside this declarative approximation.
_ENTITY_SUMMARY_INSTRUCTION = """
Create one canonical entity from the grouped mentions. Return canonical {name},
best {entity_type}, and concise {summary}. Use the episode {content} only as
evidence and do not invent unsupported details.
""".strip()


# Adapted from Graphiti graphiti_core/prompts/extract_edges.py::edge.
_FACT_EXTRACT_INSTRUCTION = """
Extract every factual relationship stated in the current episode {content}
between the resolved {entities}. Each listed entity has an integer
entity_ordinal. Return one row per relationship with {source_entity_ordinal},
{target_entity_ordinal}, {relation_type}, {fact}, {valid_at}, and {invalid_at}.

Use {previous_episodes} only to resolve references and maintain continuity. Use
{reference_time} to resolve relative temporal expressions. Use ISO 8601
timestamps when temporal bounds can be resolved; otherwise return JSON null.
Both entity ordinals must identify distinct entities in the provided list.
""".strip()


# Adapted from Graphiti graphiti_core/prompts/dedupe_edges.py::resolve_edge.
_FACT_GROUP_INSTRUCTION = """
Rows belong to one group when {relation_type} and {fact} express the same
factual information. Similar claims with key numeric, temporal, relational, or
entity differences are not duplicates.
""".strip()


_FACT_CANONICAL_INSTRUCTION = """
Create one canonical fact row from duplicate claims. Return canonical
{relation_type} and faithful {fact}. Preserve the original meaning; do not merge
contradictory claims into one statement.
""".strip()


# Pairwise form of Graphiti's contradiction decision. Temporal direction is
# applied deterministically after this semantic predicate.
_CONTRADICTORY_FACT_INSTRUCTION = """
Does {fact:later_added} explicitly contradict, supersede, or make
{fact:earlier_added} stop being true? Return false for merely related facts or
facts that can both be true at the same time.
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


class ZepMemory(Memory):
    """Declarative Zep/Graphiti-style memory with temporal facts.

    This policy describes logical memory state. It intentionally omits
    Graphiti's hybrid candidate search, top-1 entity resolver, graph storage,
    and community label-propagation lowering.
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
                "entity_type": "Entity type label.",
            },
            instruction=_ENTITY_EXTRACT_INSTRUCTION,
            ordinal_col="entity_ordinal",
        )
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
            input_cols=["name", "entity_type"],
            instruction=_ENTITY_GROUP_INSTRUCTION,
        )
        .agg(
            sem_agg(
                input_cols=["name", "entity_type", "content"],
                output_cols={
                    "name": "Canonical entity name.",
                    "entity_type": "Canonical entity type.",
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
