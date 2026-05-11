# User Interface Demos

This document is a source-backed design demo. It is not an implementation specification and does not claim that the APIs below exist today.

The proposed authoring surface:

- `am.Memory` for a memory policy.
- `am.Log()` for append-only messages/events.
- Pydantic `BaseModel` schemas for materialized views.
- Chainable `sem_*` operators for semantic transformations.
- Class-level `STORES = {...}` bindings for physical materialization.
- Optional class-level `FRESHNESS = {...}` and `REFRESH_MODE = {...}` targets.

## 1. Source-Backed Summary

### Claude Code

Source basis:

- `claude-code/src/memdir/*`
- `claude-code/src/services/extractMemories/*`
- `claude-code/src/services/autoDream/*`
- `claude-code/src/utils/attachments.ts`
- `claude-code/src/query.ts`

Observed structure:

- Durable memory is file/topic based.
- Each memory file has frontmatter such as `name`, `description`, and `type`.
- `MEMORY.md` is a compact catalog view over memory files.
- Background extraction runs from recent conversation messages and writes or updates topic files.
- Extraction checks existing memory headers before writing, so it prefers updating existing files over creating duplicates.
- Query-time relevant memory surfacing scans memory headers, selects up to a small number of useful files, reads them, and injects them as system-reminder context.
- Auto-dream is a maintenance pass over memory files and recent transcripts. It consolidates, prunes, and keeps the catalog concise.

### Zep / Graphiti

Source basis:

- `zep-graphiti/graphiti_core/graphiti.py`
- `zep-graphiti/graphiti_core/utils/maintenance/node_operations.py`
- `zep-graphiti/graphiti_core/utils/maintenance/edge_operations.py`
- `zep-graphiti/graphiti_core/prompts/*`
- Official docs for terminology: [Adding Episodes](https://help.getzep.com/graphiti/core-concepts/adding-episodes), [Overview](https://help.getzep.com/graphiti/graphiti/overview)

Observed structure:

- Ingestion starts from an episode.
- An episode is stored as provenance for extracted graph state.
- Entity nodes are extracted from the episode using recent previous episodes as context.
- Extracted entities are resolved against existing entities using hybrid search plus semantic duplicate resolution.
- Relation/fact edges are extracted from the episode and resolved against existing edges.
- Duplicate facts are reused; contradictory facts can invalidate earlier facts rather than deleting history.
- Optional community maintenance summarizes entity neighborhoods.
- Retrieval uses hybrid search over facts, entities, episodes, and communities depending on the search recipe.

## 2. Claude Code Memory Demo

This demo tries to express Claude Code's file/topic memory without exposing file operations in the policy body.

### Schemas

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

import agent_memory as am


# Framework note: LogEntry is the default raw log table schema for this demo, not a view
# policy writer can also override that
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
```

### Policy Demo

All code below is proposed demo syntax.

The semantic operators, `STORES`, `FRESHNESS`, and `REFRESH_MODE` are candidate APIs, not implemented guarantees.

```python
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

    # Optional. If omitted, runtime chooses update timing and strategy.
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

    def query(self, query: str, *, k: int = 5):
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
            k=k,
            context=self.catalog,
        )
```

**Explanations**:

- The `topic` rule represents Claude Code's background extraction path: recent log delta plus existing memory metadata updates durable topic files.
- The `sem_groupby(...)` rule intentionally represents extraction, topic grouping, existing-topic update, and consolidation as one policy-level view query.
- The runtime may lower this high-level `sem_groupby(...)` into multiple implementation steps such as `sem_filter(...)`, `sem_map(...)`, lower-level `sem_groupby(...)`, candidate matching, merging, periodic consolidation, and storage writes.
- Claude Code's auto-dream consolidation is therefore modeled as runtime/lowering behavior for the same `topic` view, not as a second canonical `topic = topic.sem_map(...)` policy query.
- `FRESHNESS` and `REFRESH_MODE` are optional maintenance targets. If omitted, the runtime chooses timing and strategy.

### Default And Override Stores

The class-level `stores` block gives this demo a default filesystem materialization:

```python
memory = ClaudeCodeMemory()
```

Users can override the physical layout without changing the logical policy:

```python
memory = ClaudeCodeMemory(
    stores={
        "log": {
            "type": "jsonl",
            "path": "/custom/memory/log.jsonl",
        },
        "topic": {
            "type": "directory",
            "path": "/custom/memory/topics",
            "format": "markdown",
        },
        "catalog": {
            "type": "markdown",
            "path": "/custom/memory/MEMORY.md",
        },
    }
)
```

The important interface question is whether a view such as `catalog` can be treated like any other materialized view while still being usable as explicit retrieval context. This demo assumes yes.

### What This Demo Does Not Model Yet

- `[Store/runtime]` Team memory. Claude Code can maintain private and team memory directories with different scope rules. This demo only models one local memory directory.
- `[API/runtime]` KAIROS daily-log mode. That mode can be understood as a different policy and materialization strategy: write append-only daily logs first, then distill them into topic files and `MEMORY.md` later. This demo models the standard topic-file mode directly.
- `[Runtime]` Exact tool permission gates. Claude Code's extraction subagent is restricted to read tools, read-only Bash, and writes inside the memory directory. This demo expresses semantic behavior, not tool sandbox policy.
- `[UI/runtime]` UI memory-saved notifications. Claude Code can add a system message after memory files are saved or improved. That notification is a transcript/UI
  event, not a durable memory view.
- `[Runtime]` Exact runtime cursor, throttle, and coalescing. `lastMemoryMessageUuid` is the processed-message cursor that makes extraction consider only messages after the last successful run. `turnsSinceLastExtraction` is a turn-count throttle, so extraction does not have to run on every turn. In-progress coalescing prevents overlapping extraction runs from piling up.

## 3. Zep / Graphiti Memory Demo

This demo tries to express Graphiti-style temporal graph memory as materialized views over an append-only log.

### Schemas

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

import agent_memory as am


class ZepLogEvent(BaseModel):
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
```

### Policy Demo

All code below is proposed demo syntax. It is intentionally close to the observed Graphiti add/search flow, but it is not executable today.

```python
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

    log = am.Log(ZepLogEvent)

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

    def query(self, query: str, *, k: int = 10):
        facts = self.relation.sem_topk(
            query,
            "Retrieve relevant temporal facts with hybrid semantic and keyword search.",
            k=k,
            context=[self.entity, self.episode],
        )

        entities = self.entity.sem_topk(
            query,
            "Retrieve relevant entities using hybrid search.",
            k=k,
        )

        communities = self.community.sem_topk(
            query,
            "Retrieve relevant community summaries when available.",
            k=min(k, 3),
        )

        return {
            "facts": facts,
            "entities": entities,
            "communities": communities,
        }
```

### Policy Semantics

In this demo, a top-level assignment inside `am.Memory` defines a materialized view rule. Intermediate `sem_*` calls and `window(...)` calls are transient query stages inside that view rule.

`window(...)` is a bounded relation used as data range or context. It is not a materialized view unless it is assigned to a memory attribute as a top-level view rule.

The high-level `entity` and `relation` rules use stateful `sem_map(...)` to express extraction, candidate retrieval, duplicate resolution, contradiction handling, and MERGE-like effects as one logical view query.

An omitted-right `sem_join(...)` remains an expert/lower-level expansion for existing-state matching. It is not required in the conceptual Zep demo.

`MENTIONS`, `HAS_EPISODE`, and `NEXT_EPISODE` are structural or provenance links maintained by the runtime/store. They are not high-level semantic views in this demo.

### Store Binding Demo

```python
memory = ZepMemory()

custom_memory = ZepMemory(
    stores={
        "log": {
            "type": "jsonl",
            "path": "./logs/zep.jsonl",
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
)
```

Default stores are part of the memory style. Constructor-level `stores={...}` overrides change physical materialization without changing the logical policy.

Graph storage and vector materialization are physical choices attached to logical views, not part of the schema definition.

### What This Demo Does Not Model Yet

- `[API]` Custom entity types and excluded entity types. Graphiti lets callers provide entity type schemas and exclude selected types from extraction. This demo uses one generic `Entity` schema, but this does not require changing the core `Log -> View -> refresh` model. One future option is to keep a polymorphic `Entity` view and pass entity schemas as prompt/schema context:

  ```python
  entity = episode.sem_map(
      Entity,
      "Extract and classify entities using the provided entity type schemas.",
      context=[Person, Company, Product],
  )
  ```

  Another option is to split the entity view into typed materialized views:

  ```python
  person = episode.sem_map(Person, "Extract people from the episode.")
  company = episode.sem_map(Company, "Extract companies from the episode.")
  ```

  The generic `Entity` view is closer to Graphiti's current structure; typed views are a cleaner future alternative for stricter schemas.

- `[API]` Custom edge types and edge type maps. Graphiti can restrict fact types by source/target entity type signatures. This demo uses one generic `Relation` schema, but custom edge types can also be modeled without changing the core view model. One option is to keep a polymorphic `Relation` view and pass relation schemas or source/target constraints as operator context or future structured config. Another option is to split relation types into typed relation views:

  ```python
  works_at = episode.sem_map(
      WorksAt,
      "Extract employment facts between people and companies.",
      context=[person, company],
  )

  uses_product = episode.sem_map(
      UsesProduct,
      "Extract product usage facts involving people, companies, and products.",
      context=[person, company, product],
  )
  ```

  Source/target constraints such as `Person -> Company` allowing `WorksAt`, or `Person -> Product` allowing `UsesProduct`, are API-level schema/prompt constraints, not only runtime behavior.
- `[Runtime/store]` Saga support, including `HAS_EPISODE` and `NEXT_EPISODE` links. These are structural/provenance links for episode grouping and ordering, not high-level semantic views.
- `[Store/runtime]` Raw episode content storage toggles. Whether raw episode content is retained is a storage and privacy policy, not a semantic operator.
- `[Runtime/lowering]` Exact hybrid search, reranking, graph traversal, and candidate retrieval recipes. The policy expresses logical `sem_join(...)` and `sem_topk(...)`; vector, keyword, BFS/k-hop graph expansion, RRF/MMR, graph-distance scoring, and reranking choices belong to runtime/store lowering. Graphiti-style BFS is a candidate retrieval or reranking ingredient, not high-level policy syntax.
- `[Runtime]` Bulk add paths, tracing, concurrency, and queueing behavior. These are execution paths and observability controls, not policy API.
- `[Runtime/scope]` Multi-tenant `group_id` semantics. v0 assumes a single project/user scope; namespace and tenant isolation can be added later.

## 4. Cross-Demo Design Status

The current interface direction can express both systems if the following are true:

- Views can be materialized.
- `am.Log()` can act as the append-only source relation.
- `sem_*` operators can accept Pydantic output schemas.
- `sem_*` operators can accept explicit `context=...`.
- Semantic joins can express both regular relation joins and existing-state joins.
- Store bindings are per logical view and can support one or more physical stores.
- `catalog` is just a normal view and can be passed as retrieval context.
- `FRESHNESS` and `REFRESH_MODE` can optionally constrain maintenance targets without changing the logical query.

API decisions in this document:

- A top-level assignment inside `am.Memory` defines a materialized view rule.
- `window(...)` is transient unless it is assigned to a view as a top-level rule.
- An omitted-right `sem_join(...)` inside a materialized view rule means existing-state join.
- There is no `maintain(...)` policy API in this direction.
- `FRESHNESS` is an optional time-lag target; `REFRESH_MODE` is optional and may be inferred by runtime, Flink-style.
- `turn`, `add`, `query`, `count`, and `session` thresholds are runtime scheduling gates for now, not v0 `FRESHNESS` syntax.
- v0 assumes a single project/user scope. `group_id` is a future namespace/scope extension.

Runtime/lowering issues:

- Delta, cursor, watermark, and checkpoint semantics.
- Candidate retrieval implementation for semantic joins when the policy does not specify it.
- MERGE-like insert/update/delete/invalidation effects.
- Exact scheduler behavior behind `FRESHNESS` / `REFRESH_MODE`.
- Store-specific vector, keyword, graph, hybrid search, and reranking strategies.

Still-open API issues:

- Whether candidate retrieval inside semantic joins ever needs explicit low-level policy syntax, such as `sem_join_lateral(...)`.
- `query()` return shape should default to plain Python objects; framework-specific adapters can wrap this later.

Non-goals for this document:

- It does not implement runtime behavior.
- It does not implement the operator runtime.
- It does not decide the storage engine abstraction.
- It does not claim Claude Code or Zep are fully reproduced.
