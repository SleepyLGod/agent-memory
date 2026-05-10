# Claude Code And Zep User Interface Demos

This document is a source-backed design demo. It is not an implementation specification and does not claim that the APIs below exist today.

The goal is to test whether the current `agent-memory` interface direction can express two realistic memory systems:

- Claude Code-style file/topic memory.
- Zep / Graphiti-style temporal graph memory.

The proposed authoring surface is:

- `am.Memory` for a memory policy.
- `am.Log()` for append-only messages/events.
- Pydantic `BaseModel` schemas for materialized views.
- Chainable `sem_*` operators for semantic transformations.
- Per-view `stores={...}` bindings for physical materialization.

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


# Framework note: LogEntry is the default raw log row schema for this demo.
# This comment is not compiler-visible prompt text. The class docstring and
# field descriptions may be used by semantic operators or optimizers, so they
# should describe the data itself rather than framework metadata.
class LogEntry(BaseModel):
    """A raw conversation event available to memory update policies."""

    role: str = Field(description="message author, such as user or assistant")
    content: str = Field(description="message text")
    timestamp: str | None = Field(default=None, description="event timestamp if available")


# Policy-writer-specified materialized view schemas.
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

The semantic operators, repeated assignment capture, stores, and `refresh(...)` rules are candidate APIs, not implemented guarantees.

```python
class ClaudeCodeMemory(am.Memory):
    """
    Maintain Claude Code-style long-term memory.

    Save durable user, feedback, project, and reference information that is not
    derivable from the current project state. Do not save transient task state,
    code structure, git history, or facts already documented elsewhere.
    """

    stores = {
        "log": am.stores.JSONL(".memory/log.jsonl"),
        "topic": am.stores.Directory(".memory/topics"),
        "catalog": am.stores.Markdown(".memory/MEMORY.md"),
    }

    log = am.Log(LogEntry)

    topic = log.sem_groupby(
        Topic,
        """
        Analyze only the recent messages selected by the runtime for this
        refresh and update durable topic memory files.

        Save durable memory only when it fits one of the Claude Code memory
        types: user, feedback, project, or reference. Do not save code patterns,
        architecture, file paths, project structure, git history, debugging
        recipes, facts already documented in CLAUDE.md, or ephemeral task state.

        Check existing memory metadata before writing. Update an existing topic
        memory instead of creating a duplicate. Organize memories semantically by
        topic, not chronologically. Correct or remove memories that become wrong
        or outdated.

        If the user explicitly asks to remember something, save it immediately
        as the best-fitting type. If the user asks to forget something, remove
        the relevant memory.
        """
    ).refresh(on="turn")

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
    ).refresh(on="source")

    # Repeated assignment to `topic` appends a self-maintenance rule for the
    # same logical view. The previous materialized topic state is the main input.
    # `log` and `catalog` are evidence/context for consolidation, not a second
    # canonical Log -> Topic view definition.
    topic = topic.sem_map(
        Topic,
        """
        Periodically consolidate existing topic memories.

        First orient from the current memory directory: inspect the catalog and
        skim existing topic memories so you improve them rather than creating
        duplicates. Use logs or session transcripts only as narrow evidence when
        they contain signal worth persisting.

        Merge new signal into existing topic memories. Convert relative dates to
        absolute dates. Delete or correct contradicted facts. Resolve conflicts
        between memories. Keep topic memories durable, concise, and organized.
        """,
        context=[log, catalog],
    ).refresh(every="24h", require={"sessions": 5})

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

**Explanitions**:

- The online `topic` rule represents Claude Code's background extraction path: recent log delta plus existing memory metadata updates durable topic files.
- The periodic `topic` rule represents auto-dream consolidation: it operates on the existing topic state and may use log or catalog evidence, but it is not a second canonical `Log -> Topic` view path.
- The online `sem_groupby(...)` rule intentionally represents extraction, topic grouping, and existing-topic update as one policy-level view query. The runtime
  may lower it into multiple implementation steps such as candidate extraction, matching, merging, and storage writes. The user-facing API may not need a
  separate `sem_map(...)` operator unless a custom workflow later wants to split those steps explicitly.

### Default And Override Stores

The class-level `stores` block gives this demo a default filesystem materialization:

```python
memory = ClaudeCodeMemory()
```

Users can override the physical layout without changing the logical policy:

```python
memory = ClaudeCodeMemory(
    stores={
        "log": am.stores.JSONL("/custom/memory/log.jsonl"),
        "topic": am.stores.Directory("/custom/memory/topics"),
        "catalog": am.stores.Markdown("/custom/memory/MEMORY.md"),
    }
)
```

The important interface question is whether a view such as `catalog` can be treated like any other materialized view while still being usable as explicit retrieval context. This demo assumes yes.

### What This Demo Models

- Topic memory files as Claude Code's primary durable memory object.
- `MEMORY.md` as a catalog view over topic memory files, not a memory body.
- Background extraction as a `Log -> Topic` materialized semantic view refresh.
- Query-time relevant memory selection using topic metadata and catalog context.
- Auto-dream-style consolidation as periodic self-maintenance over the same `Topic` view.

### What This Demo Does Not Model Yet

- Team memory. Claude Code can maintain private and team memory directories with different scope rules. This demo only models one local memory directory.
- KAIROS daily-log mode. That mode can be understood as a different refresh and materialization strategy: write append-only daily logs first, then distill them into topic files and `MEMORY.md` later. This demo models the standard topic-file mode directly.
- Exact tool permission gates. Claude Code's extraction subagent is restricted to read tools, read-only Bash, and writes inside the memory directory. This demo expresses semantic behavior, not tool sandbox policy.
- UI memory-saved notifications. Claude Code can add a system message after memory files are saved or improved. That notification is a transcript/UI
  event, not a durable memory view.
- Exact runtime cursor, throttle, and coalescing. `lastMemoryMessageUuid` is the processed-message cursor that makes extraction consider only messages after the last successful run. `turnsSinceLastExtraction` is a turn-count throttle, so extraction does not have to run on every turn. In-progress coalescing prevents overlapping extraction runs from piling up.

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

    stores = {
        "log": am.stores.JSONL(".memory/zep/log.jsonl"),
        "episode": am.stores.Graph("neo4j://localhost:7687", label="Episodic"),
        "entity": [
            am.stores.Graph("neo4j://localhost:7687", label="Entity"),
            am.stores.Vector("entity_name_embedding"),
        ],
        "relation": [
            am.stores.Graph("neo4j://localhost:7687", relation="RELATES_TO"),
            am.stores.Vector("fact_embedding"),
        ],
        "community": [
            am.stores.Graph("neo4j://localhost:7687", label="Community"),
            am.stores.Vector("community_name_embedding"),
        ],
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
        .refresh(on="add")
    )

    entity = (
        episode
        .sem_map(
            Entity,
            """
            Extract entity nodes from the current episode.

            For message episodes, extract the speaker and significant entities
            mentioned explicitly or implicitly in the current episode. Use
            recent previous episodes only for disambiguation. Do not extract
            relationships, actions, dates, times, or pronouns as entities.
            Use configured entity types when provided.
            """,
            context=episode.window(count=10),
        )
        .sem_join(
            """
            Resolve extracted entities against existing entity memory.

            An extracted entity matches an existing entity only if both refer to
            the same real-world object or concept. Do not merge entities that
            are merely related, similar, or similarly named. Reuse the existing
            identity when matched; otherwise keep a new entity.
            """,
            how="left",
            context=episode.window(count=10),
        )
        .sem_map(
            Entity,
            """
            Produce the maintained entity record.

            Preserve the canonical entity identity after resolution. Update
            attributes and summary with important information from the current
            episode and recent previous episodes when relevant.
            """,
            context=[episode, episode.window(count=10)],
        )
        .refresh(on="source")
    )

    relation = (
        episode
        .sem_map(
            Relation,
            """
            Extract factual relationships between resolved entities.

            Each fact must involve two distinct entities from the resolved
            entity view. Use entity identifiers from the resolved entities.
            Extract facts clearly stated or unambiguously implied by the current
            episode. Include relation type and temporal bounds when the episode
            provides explicit or resolvable time information.
            """,
            context=[entity, episode.window(count=10)],
        )
        .sem_join(
            """
            Resolve extracted facts against existing relation memory.

            Reuse existing facts when the new fact expresses identical factual
            information. Keep similar facts separate when they contain key
            differences. If a new fact contradicts existing facts, mark the
            older facts invalid instead of deleting them.
            """,
            how="left",
            context=[episode, entity],
        )
        .refresh(on="source")
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
        .refresh(mode="manual")
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

        return am.pack(
            facts=facts,
            entities=entities,
            communities=communities,
            instruction="Return graph memory context useful for the agent query.",
        )
```

### Policy Semantics

In this demo, a top-level assignment only defines or maintains a materialized view when the right-hand expression ends with  `.refresh(...)`. Intermediate  `sem_map(...)`, `sem_join(...)`, and `window(...)` calls are transient query stages inside that view rule.

`window(...)` is a bounded relation used as data range or context. It is not a materialized view unless it is assigned to a memory attribute and refreshed.

An omitted-right `sem_join(...)` inside a refreshed view rule means an existing-state join: the current stage is joined against the existing state of the assignment target. For example, the `entity` rule lowers to extracted entity rows joined against the existing `entity` view before merge/update.

`MENTIONS`, `HAS_EPISODE`, and `NEXT_EPISODE` are structural or provenance links maintained by the runtime/store. They are not high-level semantic views in this demo.

### Store Binding Demo

```python
memory = ZepMemory()

custom_memory = ZepMemory(
    stores={
        "log": am.stores.JSONL("./logs/zep.jsonl"),
        "episode": am.stores.Graph("neo4j://localhost:7687", label="Episodic"),
        "entity": [
            am.stores.Graph("neo4j://localhost:7687", label="Entity"),
            am.stores.Vector("entity_name_embedding"),
        ],
        "relation": [
            am.stores.Graph("neo4j://localhost:7687", relation="RELATES_TO"),
            am.stores.Vector("fact_embedding"),
        ],
        "community": [
            am.stores.Graph("neo4j://localhost:7687", label="Community"),
            am.stores.Vector("community_name_embedding"),
        ],
    }
)
```

Default stores are part of the memory style. Constructor-level `stores={...}` overrides change physical materialization without changing the logical policy. 

Graph storage and vector materialization are physical choices attached to logical views, not part of the schema definition.

### What This Demo Models

- Episode provenance as a durable `Episode` view.
- Entity extraction, existing-state resolution, and entity summary/attribute updates.
- Relation extraction, duplicate fact resolution, contradiction invalidation, and temporal fact fields.
- Optional community maintenance as a manual view, matching Graphiti's opt-in community update path.
- Query-time retrieval over relation, entity, and community views.

### What This Demo Does Not Model Yet

- Custom entity types and excluded entity types.
- Custom edge types and edge type maps.
- Saga support, including `HAS_EPISODE` and `NEXT_EPISODE` links.
- Raw episode content storage toggles.
- Exact hybrid search, reranking, and candidate retrieval recipes.
- Bulk add paths, tracing, concurrency, and queueing behavior.
- Multi-tenant `group_id` semantics; v0 assumes a single project/user scope.

## 4. Interface Adequacy Checklist

The current interface direction can express both systems if the following are true:

- Views can be materialized.
- `am.Log()` can act as the append-only source relation.
- `sem_*` operators can accept Pydantic output schemas.
- `sem_*` operators can accept explicit `context=...`.
- Semantic joins can express both regular relation joins and existing-state joins.
- Store bindings are per logical view and can support one or more physical stores.
- `catalog` is just a normal view and can be passed as retrieval context.

API decisions in this document:

- `refresh(...)` defines materialized view maintenance timing.
- `window(...)` is transient unless it is assigned to a view and refreshed.
- An omitted-right `sem_join(...)` inside a refreshed view rule means existing-state join.
- There is no `maintain(...)` policy API in this direction; maintenance timing is expressed with `refresh(...)`.
- v0 assumes a single project/user scope. `group_id` is a future namespace/scope extension.

Runtime/lowering issues:

- Delta, cursor, watermark, and checkpoint semantics.
- Candidate retrieval implementation for semantic joins when the policy does not specify it.
- MERGE-like insert/update/delete/invalidation effects.
- Exact refresh scheduler behavior behind `refresh(...)`.
- Store-specific vector, keyword, graph, hybrid search, and reranking strategies.

Still-open API issues:

- Whether candidate retrieval inside semantic joins ever needs explicit low-level policy syntax, such as `sem_join_lateral(...)`.
- `query()` return shape: plain Python objects, `am.pack(...)`, or framework-specific adapters.

Non-goals for this document:

- It does not implement runtime behavior.
- It does not implement the operator runtime.
- It does not decide the storage engine abstraction.
- It does not claim Claude Code or Zep are fully reproduced.
