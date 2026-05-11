# Conceptual Demos

Conceptually perfect demos of claude code, zep, and mem0 memory systems.

Maximizing the opportunity for optimizations.

### Claude Code Memory

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

import agent_memory as am


##########################
# schema definition
##########################

# Framework note: LogEntry is the default raw log table schema for this demo, not a view
# this can be by default used, or override by policy writers.
class LogEntry(BaseModel):
    """A raw conversation event available to memory update policies."""

    role: str = Field(description="message author, such as user or assistant")
    content: str = Field(description="message text")
    timestamp: str | None = Field(default=None, description="event timestamp if available")


# Policy-writer-specified materialized view schemas:
class Topic(BaseModel):
    """A durable Claude Code-style topic memory file."""

    name: str = Field(description="short stable frontmatter memory name")
    description: str = Field(
        description="one-line frontmatter description used to decide future relevance; be specific"
    )
    type: Literal["user", "feedback", "project", "reference"] = Field(
        description="memory type controlling save/use behavior: user, feedback, project, or reference"
    )
    content: str = Field(
        description="markdown memory body following the selected memory type's structure"
    )


class Catalog(BaseModel):
    """A MEMORY.md catalog entry for durable topic memories."""

    title: str = Field(description="catalog entry title")
    path: str = Field(description="relative path to the topic memory file")
    hook: str = Field(description="one-line hook used in the MEMORY.md catalog entry")

##########################
# memory definition
##########################

class ClaudeCodeMemory(am.Memory):
    """
    Maintain Claude Code-style long-term memory.

    Save durable user, feedback, project, and reference information that is not
    derivable from the current project state. Do not save transient task state,
    code structure, git history, or facts already documented elsewhere.
    """

    STORES = {
        "log": {
            "type": "jsonl",
            "path": ".memory/log.jsonl",
        },
        "topic": {
            "type": "directory",
            "path": ".memory/topics",
            "format": "markdown",
        },
        "catalog": {
            "type": "markdown",
            "path": ".memory/MEMORY.md",
        },
    }

    FRESHNESS = {
        "topic": "10m",
        "catalog": "1m",
    }

    REFRESH_MODE = {
        "topic": "continuous",
        "catalog": "full",
    }

    log = am.Log(LogEntry)

    topic = log.sem_groupby(
        Topic,
        """
        Analyze only the recent messages selected by the runtime for this
        maintenance pass, together with existing topic metadata supplied by the
        runtime, and update durable topic memory files.

        Save durable memory only when it fits one of these Claude Code memory
        types:

        - user: stable information about the user's role, goals,
          responsibilities, knowledge, preferences, or collaboration style that
          should shape future assistance.
        - feedback: guidance about how to approach future work, including
          corrections, things to avoid, and non-obvious approaches the user
          validated. Preserve the reason and how to apply it.
        - project: ongoing work context, goals, initiatives, bugs, incidents,
          constraints, deadlines, or motivations that are not otherwise
          derivable from the code or git history. Convert relative dates to
          absolute dates.
        - reference: pointers to external systems or resources, and why they
          matter, so future sessions know where to look for up-to-date
          information.

        Do not save code patterns, conventions, architecture, file paths,
        project structure, git history, recent changes, who-changed-what,
        debugging recipes, facts already documented in CLAUDE.md, or ephemeral
        in-progress task details. These exclusions still apply when the user
        asks to save an activity log or PR list; preserve only the surprising
        or non-obvious durable signal.

        Check existing memory metadata before writing. Update an existing topic
        memory instead of creating a duplicate. Organize memories semantically by
        topic, not chronologically. Correct or remove memories that become wrong
        or outdated.

        Maintain each topic as one durable memory record with current
        frontmatter-compatible fields: a stable name, a specific one-line
        description for future relevance decisions, a valid type, and a concise
        markdown body following the selected type's structure.

        If the user explicitly asks to remember something, save it immediately
        as the best-fitting type. If the user asks to forget something, remove
        the relevant memory.
        """
    )

    catalog = topic.sem_map(
        Catalog,
        """
        Maintain MEMORY.md as a concise catalog over durable topic memory files.

        MEMORY.md is an index, not a memory body. Each entry should point to one
        topic file and contain one short hook. It has no frontmatter. Never write
        full memory content directly into MEMORY.md.

        Keep the catalog concise because it is loaded into context. Prefer one
        line per entry, roughly under 150 characters. Remove pointers to stale,
        wrong, or superseded topic memories.
        """
    )

    def query(self, query: str):
        return self.topic.sem_topk(
            query,
            """
            Select memory files that are clearly useful for the query.

            Use the candidate memory filename, frontmatter description, memory
            type, freshness, and catalog context. Return at most k memories. If
            unsure whether a memory will help, do not include it. If no memory
            is clearly useful, return an empty result.

            Do not select usage reference or API documentation for tools that
            are already being actively used. Do select memories containing
            warnings, gotchas, known issues, durable project context, user
            preferences, or relevant reference pointers.
            """,
            k=5,
            context=self.catalog,
        )

   
```

### Zep Memory

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

import agent_memory as am

##########################
# schema definition
##########################

# policy writers override the log schema
class LogEntry(BaseModel):
    """One raw event passed into memory."""

    name: str = Field(description="episode name")
    body: str = Field(description="episode body")
    source: Literal["message", "text", "json"] = Field(description="episode source type")
    source_description: str = Field(description="description of the source")
    reference_time: str = Field(description="event time used for temporal extraction")
    group_id: str | None = Field(default=None, description="graph partition")


class Episode(BaseModel):
    """Stored provenance event."""

    uuid: str = Field(description="episode id")
    name: str = Field(description="episode name")
    content: str = Field(description="episode content")
    source: str = Field(description="message, text, or json")
    valid_at: str = Field(description="reference time")
    group_id: str = Field(description="graph partition")


class Entity(BaseModel):
    """Resolved entity node."""

    uuid: str = Field(description="entity id")
    name: str = Field(description="entity name")
    summary: str = Field(description="short entity summary")
    labels: list[str] = Field(description="entity type labels")
    group_id: str = Field(description="graph partition")


class Relation(BaseModel):
    """Temporal fact edge between entities."""

    uuid: str = Field(description="relation id")
    source_entity_id: str = Field(description="source entity id")
    target_entity_id: str = Field(description="target entity id")
    relation_type: str = Field(description="relation predicate")
    fact: str = Field(description="natural-language fact")
    valid_at: str | None = Field(default=None, description="when the fact became true")
    invalid_at: str | None = Field(default=None, description="when the fact stopped being true")
    group_id: str = Field(description="graph partition")


class Community(BaseModel):
    """Optional summary view over related entities."""

    uuid: str = Field(description="community id")
    name: str = Field(description="short community name")
    summary: str = Field(description="community summary")
    group_id: str = Field(description="graph partition")

##########################
# memory definition
##########################  

class ZepMemory(am.Memory):
    """
    Maintain a temporal graph memory from episodes.

    Ingest episodes, extract entities and temporal relations, resolve them
    against existing graph state, preserve provenance, and retrieve context via
    hybrid semantic, keyword, and graph-aware search.
    """

    STORES = {
        "log": {
            "type": "jsonl",
            "path": ".memory/zep/log.jsonl",
        },
        "episode": {
            "type": "graph",
            "uri": "neo4j://localhost:7687",
            "label": "Episodic",
        },
        "entity": {
            "primary": {
                "type": "graph",
                "uri": "neo4j://localhost:7687",
                "label": "Entity",
            },
            "auxiliary": [
                {
                    "type": "vector",
                    "collection": "entity_name_embedding",
                    "field": "name",
                },
            ],
        },
        "relation": {
            "primary": {
                "type": "graph",
                "uri": "neo4j://localhost:7687",
                "relation": "RELATES_TO",
            },
            "auxiliary": [
                {
                    "type": "vector",
                    "collection": "fact_embedding",
                    "field": "fact",
                },
            ],
        },
        "community": {
            "primary": {
                "type": "graph",
                "uri": "neo4j://localhost:7687",
                "label": "Community",
            },
            "auxiliary": [
                {
                    "type": "vector",
                    "collection": "community_name_embedding",
                    "field": "name",
                },
            ],
        },
    }

    FRESHNESS = {
        "episode": "1m",
        "entity": "10m",
        "relation": "10m",
    }

    REFRESH_MODE = {
        "episode": "continuous",
        "entity": "continuous",
        "relation": "continuous",
    }

    log = am.Log(LogEntry)

    episode = (
        log.sem_map(
            Episode,
            """
            Convert each incoming event into a stored episode.

            Preserve name, source type, source description, content, reference
            time, and scope metadata if configured. The episode is the
            provenance record for derived entities and relations.
            """,
        )
    )

    entity = (
        episode
        .sem_map(
            Entity,
            """
            Extract, resolve, and maintain canonical entity nodes from episodes.

            For message episodes, extract the speaker and significant entities
            mentioned explicitly or implicitly in the current episode. Use
            recent previous episodes only for disambiguation. Do not extract
            relationships, actions, dates, times, or pronouns as entities.

            Resolve extracted entities against existing entity memory. Reuse
            canonical identities only when both names refer to the same
            real-world object or concept. Do not merge entities that are merely
            related, similar, or similarly named.

            Preserve canonical entity identity after resolution. Update
            attributes, labels, and summaries with important information from
            the current episode and recent previous episodes when relevant.
            Use configured entity types when provided.
            """,
            context=[episode, episode.window(count=10)],
        )
    )

    relation = (
        episode
        .sem_map(
            Relation,
            """
            Extract, resolve, and maintain temporal factual relations between
            resolved entities.

            Each fact must involve two distinct entities from the resolved
            entity view. Use entity identifiers from the resolved entities.
            Extract facts clearly stated or unambiguously implied by the current
            episode. Include relation type and temporal bounds when the episode
            provides explicit or resolvable time information.

            Reuse existing facts when the new fact expresses identical factual
            information. Keep similar facts separate when they contain key
            differences. If a new fact contradicts existing facts, mark the
            older facts invalid instead of deleting them.
            """,
            context=[episode, entity],
        )
    )

    community = (
        entity
        .sem_groupby(
            Community,
            """
            Optionally maintain community summaries over related entities.

            Update or create community summaries for neighborhoods affected by
            new or updated entities. This corresponds to Graphiti's optional
            community update path.
            """,
            context=relation,
        )
    )

    def query(self, query: str):
        facts = self.relation.sem_topk(
            query,
            "Retrieve relevant temporal facts with hybrid semantic and keyword search.",
            k=10,
            context=[self.entity, self.episode],
        )

        entities = self.entity.sem_topk(
            query,
            "Retrieve relevant entities using hybrid search.",
            k=10,
        )

        communities = self.community.sem_topk(
            query,
            "Retrieve relevant community summaries when available.",
            k=3,
        )

        return {
            "facts": facts,
            "entities": entities,
            "communities": communities,
        }


```

### Mem0 Memory

#### Mem0 V3

```python
from pydantic import BaseModel, Field
import agent_memory as am


# Default raw log row schema. This is source data, not a materialized memory view.
class LogEntry(BaseModel):
    role: str = Field(description="conversation role, such as user or assistant")
    content: str = Field(description="message content")
    name: str | None = Field(default=None, description="optional actor name")

class Entity(BaseModel):
    """Canonical entity used internally for entity-linking retrieval."""

    name: str = Field(description="canonical entity name")
    kind: str | None = Field(default=None, description="optional entity type")

class FactEntityLink(BaseModel):
    """Internal relation linking a memory fact to entities it mentions."""

    memory_id: str = Field(description="source memory fact id")
    entity_id: str = Field(description="linked entity id")
    mention: str = Field(description="entity surface form in the memory text")


class Fact(BaseModel):
    """Atomic long-term memory produced by Mem0 v3-style additive extraction."""

    memory: str = Field(
        description="self-contained long-term memory fact extracted from user or agent messages"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="optional memory categories used for filtering or ranking"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="optional source or application metadata"
    )

class Mem0V3Memory(am.Memory):
    STORES = {
        "log": {
            "type": "jsonl",
            "path": ".memory/mem0-v3/log.jsonl",
        },
        "fact": {
            "primary": {
                "type": "vector",
                "collection": "mem0_v3_facts",
                "field": "memory",
            },
            "auxiliary": [
                {
                    "type": "keyword",
                    "collection": "mem0_v3_facts_bm25",
                    "field": "memory",
                },
            ],
        },
        "_entity": {
            "type": "vector",
            "collection": "mem0_v3_entities",
            "field": "name",
        },
        "_fact_entity": {
            "type": "table",
            "name": "mem0_v3_fact_entities",
        },
    }

    FRESHNESS = {
        "fact": "1m",
        "_entity": "10m",
        "_fact_entity": "10m",
    }

    log = am.Log(LogEntry)

    fact = (
        log.sem_map(
            Fact,
            "Extract ADD-only long-term memories from conversation messages."
        )
    )

    # internal semantic materialized view, not normally shown to user.
    _entity = (
        fact.sem_map(
            Entity,
            "Extract entities from stored memories for entity-linking retrieval."
        )
    )

    # Internal maintained link between memory facts and entities.
    _fact_entity = (
        fact.sem_map(
            FactEntityLink,
            "Link each memory to the canonical entities it mentions.",
            context=_entity,
        )
    )

    def query(self, query: str):
        memories = self.fact.sem_topk(
            query,
            "Retrieve relevant memories; entity view may be used as retrieval context.",
            k=10,
            context=self._entity,
        )
        return {"memories": memories}
```
