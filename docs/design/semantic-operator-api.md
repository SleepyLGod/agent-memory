# Semantic Operator API

This document records the current design target for semantic operators in `agent-memory`. It is not an implementation guarantee.

The goal is to keep the authoring surface close to DataFrame, SQL, and Flink concepts while still allowing semantic operators to lower into LLM calls, DSPy programs, retrieval plans, and durable memory maintenance.

## 1. Mental Model

`agent-memory` separates author-facing logical policy from runtime execution.

- `Relation expression`: a lazy dataframe-like expression over logs, views, or intermediate results. It is not durable by itself.
- `Materialized view rule`: a top-level assignment inside an `am.Memory` class body. The assigned name is the logical view name; the right-hand side is the defining semantic query.
- `Freshness / refresh mode config`: optional class-level config for view maintenance targets and strategy. It is not part of the semantic query.
- `Runtime/lowering`: cursor handling, candidate retrieval, MERGE-like update effects, embeddings, vector/BM25/graph search, reranking, and physical writes.
- `Store`: physical materialization for logs and views. It does not define the logical schema or operator prompt.

```python
class ClaudeCodeMemory(am.Memory):
    STORES = {
        "topic": {
            "type": "directory",
            "path": ".memory/topics",
            "format": "markdown",
        },
    }

    FRESHNESS = {
        "topic": "10m",
    }

    REFRESH_MODE = {
        "topic": "continuous",
    }

    log = am.Log(LogEntry)

    topic = log.sem_groupby(
        Topic,
        "Group durable memories into stable topic buckets.",
    )
```

In this example, `log.sem_groupby(...)` is the logical relation expression, `topic` is the materialized view name, `STORES` defines physical materialization, and `FRESHNESS` / `REFRESH_MODE` define optional maintenance targets.

## 2. Operator Text Convention

Semantic operators use a LOTUS-style `user_instruction` argument.

```python
episode = log.sem_map(
    Episode,
    "Convert each raw event into a stored episode with provenance metadata.",
)
```

is equivalent to:

```python
episode = log.sem_map(
    Episode,
    user_instruction="Convert each raw event into a stored episode with provenance metadata.",
)
```

The same convention applies to `sem_filter`, `sem_groupby`, `sem_join`, `sem_agg`, `sem_topk`, and `sem_window`. For `sem_filter`, the instruction is a semantic boolean condition, but the API name remains `user_instruction` to avoid mixing several text-argument names.

## 3. Core Operators

### `sem_map`

Maps each input row, window, or relation fragment into structured output rows.

Structured extraction is a common `sem_map` use case.

```python
episode = log.sem_map(
    Episode,
    "Convert each raw event into a stored episode with provenance metadata.",
)
```

Proposed shape:

```python
relation.sem_map(
    TargetSchema,
    user_instruction: str,
    *,
    context: object | list[object] | None = None,
)
```

When a `sem_map(...)` expression is assigned to a top-level view, it may define a stateful materialized semantic view, not only a pure row-wise map. For example, a Zep-style entity view can be written as one logical map:

```python
entity = episode.sem_map(
    Entity,
    "Extract, canonicalize, and deduplicate entity nodes from episodes.",
)
```

The runtime may lower this high-level view rule into extraction, candidate retrieval, existing-state matching, canonicalization, and MERGE-like update effects. A policy author can later expand those stages explicitly, but the high-level API does not require that expansion.

### `sem_filter`

Keeps rows that satisfy a semantic boolean condition.

```python
durable = log.sem_filter(
    "Keep only durable information worth saving in long-term memory.",
)
```

Proposed shape:

```python
relation.sem_filter(
    user_instruction: str,
    *,
    context: object | list[object] | None = None,
)
```

### `sem_groupby`

Groups, merges, or rolls up source rows into target-schema rows. This is the high-level operator used for Claude-style topic memory.

```python
topic = log.sem_groupby(
    Topic,
    "Group durable memories into semantic topic buckets. Update existing topics when appropriate.",
)
```

Proposed shape:

```python
relation.sem_groupby(
    TargetSchema,
    user_instruction: str,
    *,
    context: object | list[object] | None = None,
)
```

`sem_groupby(...)` is a high-level stateful semantic materialization operator. For Claude-style topic memory, one logical groupby can represent extraction, topic assignment, existing-topic update, consolidation, and storage.

Claude Code's observed implementation is closer to a single stateful extraction agent than to a fixed `filter -> map -> groupby` pipeline: the agent receives recent messages, the existing memory manifest, save rules, and file/index write rules, then updates topic memory files and `MEMORY.md`.

If an expert policy needs to expand this operator, the intermediate schema should represent update candidates rather than final topic records:

```python
topic = (
    log
    .sem_filter("Keep only durable information worth saving.")
    .sem_map(TopicUpdate, "Extract candidate topic memory updates.")
    .sem_groupby(Topic, "Merge updates into stable topic memories.")
)
```

This lower-level form is useful for expert policies or optimization research, but it is not the source-observed Claude Code pipeline and the conceptual policy can stay as a single `sem_groupby(...)`.

### `sem_join`

Joins relation expressions using a semantic match instruction. This is the key operator for Zep/Graphiti-style entity resolution, fact deduplication, and contradiction detection.

#### Regular Semantic Join

Use regular `sem_join` when both sides are explicit relation expressions.

```python
resolved_entities = recent_entities.sem_join(
    entity_history,
    "The new entity and existing entity refer to the same real-world object or concept.",
    how="left",
)
```

Proposed shape:

```python
left.sem_join(
    right_relation,
    user_instruction: str,
    *,
    how: str = "inner",
    context: object | list[object] | None = None,
)
```

Supported `how` values should mirror standard relational join shape:

- `"inner"`: keep only matching pairs.
- `"left"`: keep all left rows and attach matching right rows when found.
- `"right"`: keep all right rows and attach matching left rows when found.
- `"full"`: keep rows from both sides.

#### Existing-State Join

If the right side is omitted, `sem_join(...)` means an existing-state join. This form is only valid inside a top-level materialized view rule, because the assignment target provides the existing target view.

```python
entity = (
    episode
    .sem_map(Entity, "Extract entity nodes from the current episode.")
    .sem_join(
        "Resolve extracted entities against existing entity memory.",
        how="left",
    )
    .sem_map(Entity, "Produce the maintained entity record.")
)
```

The compiler lowers this as:

```text
source = extracted entity rows
target = existing materialized entity view
join = source sem_join target
effect = MERGE-like update into target view
```

This is not a normal self join. It is closer to SQL `MERGE`, where source rows are matched against an existing target table before insert/update/delete effects are applied. In the current design, this is an expert/lower-level expansion of high-level `sem_map(...)` or `sem_groupby(...)`, not something every conceptual demo must expose.

Reference:

- [PostgreSQL MERGE](https://www.postgresql.org/docs/current/sql-merge.html)

#### Future `sem_join_lateral`

Candidate retrieval is not primary v0 policy syntax. The runtime should infer candidate retrieval from stores and lowering rules.

If a future lower-level policy needs explicit per-left-row lookup, it can use a separate lateral/lookup-style operator:

```python
# Future candidate, not v0.
recent_entities.sem_join_lateral(
    entity.lookup(k=10),
    "The two rows refer to the same real-world object or concept.",
    how="left",
)
```

This is analogous to lookup/lateral joins: each left row performs a correlated lookup into a right-side relation before semantic matching.

### `sem_agg`

Aggregates many rows into a summary, profile, or other compact representation.

```python
profile = topic.sem_agg(
    Profile,
    "Distill stable user preferences from the selected topic memories.",
)
```

Proposed shape:

```python
relation.sem_agg(
    TargetSchema,
    user_instruction: str,
    *,
    context: object | list[object] | None = None,
)
```

### `sem_topk`

Retrieves or ranks the top-k rows for a query.

```python
facts = self.relation.sem_topk(
    query,
    "Retrieve temporal facts useful for answering the agent query.",
    k=10,
)
```

Proposed shape:

```python
relation.sem_topk(
    query: str,
    user_instruction: str | None = None,
    *,
    k: int = 10,
    context: object | list[object] | None = None,
)
```

`sem_topk` is a logical retrieval operator. Vector search, keyword search, graph traversal, RRF, MMR, and reranking are store/runtime strategies. They should not be required in the policy body unless the user explicitly wants a lower-level retrieval recipe.

#### Graph Traversal And BFS

`sem_topk` is the author-facing logical retrieval operator. BFS, k-hop traversal, and graph-distance ranking are graph-store/runtime lowering strategies, not separate v0 `sem_*` operators and not `sem_topk` itself.

For a graph-backed relation, the runtime may lower one logical retrieval rule into:

```text
query -> vector/BM25 seeds -> BFS neighborhood expansion -> merge candidates -> rerank -> top k
```

This matches Graphiti-style retrieval, where BFS is a search method alongside BM25 and cosine similarity before reranking. It also matches Neo4j's model: BFS is a graph traversal/pathfinding execution strategy, not a semantic retrieval contract.

If a user needs explicit graph traversal semantics such as k-hop neighborhood, reachability, or shortest path, that should be a future lower-level graph API, not the default v0 `sem_topk` surface.

References:

- [Neo4j GDS BFS](https://neo4j.com/docs/graph-data-science-client/current/api/v2_endpoints/pathfinding_endpoints/)
- [Neo4j shortest path planning](https://neo4j.com/docs/cypher-manual/4.1/execution-plans/shortestpath-planning/)

## 4. Window, Freshness, And Refresh Mode

`window(...)`, `FRESHNESS`, and `REFRESH_MODE` are not semantic operators.

`window(...)` changes the input range or grouping seen by a query.

```python
recent = episode.window(count=10)
hourly = log.window(size="1h", step="5m", unit="time")
```

`sem_window(...)` is different: it performs semantic segmentation and may lower to an LLM/DSPy module.

```python
segments = log.sem_window(
    "Split the conversation when the topic changes or a task completes.",
)
```

Materialized view maintenance targets are expressed as optional class-level config.

```python
class ClaudeCodeMemory(am.Memory):
    FRESHNESS = {
        "topic": "10m",
        "catalog": "1m",
    }

    REFRESH_MODE = {
        "topic": "continuous",
        "catalog": "full",
    }

    topic = log.sem_groupby(Topic, "...")
    catalog = topic.sem_map(CatalogEntry, "...")
```

`FRESHNESS` is a lag target, not a hard trigger. In the current design it follows Flink Materialized Table semantics: the value is a time lag target between base-table changes and the materialized view result. Count-based or turn-based lag is a possible future scheduling constraint, but it is not part of v0 `FRESHNESS`.

`REFRESH_MODE` currently follows Flink terminology:

- `"continuous"`: maintain the view continuously or incrementally when feasible.
- `"full"`: periodically recompute or overwrite the view result.
- omitted: runtime/system infers the mode.

Agent-specific boundaries such as `turn`, `add`, `source`, or `query` are runtime scheduling or dirtying events in the current design. They are not part of the current policy config, because Flink-style `FRESHNESS` and `REFRESH_MODE` describe lag target and maintenance mode, not the semantic source event. For example, Claude Code's `lastMemoryMessageUuid`, `turnsSinceLastExtraction`, in-progress coalescing, and autoDream time/session gates are runtime mechanisms for deciding when to run view maintenance, not logical view-query syntax.

#### Old `refresh(on=...)` Boundary

Earlier drafts used `.refresh(on=...)` as fluent syntax. That mixed several different runtime concepts and should not be v0 policy syntax:

- `on="source"`: update when an upstream source or view changes. This is implicit in a continuous materialized view.
- `on="add"`: update after `memory.add(...)` or episode append. This is an application/runtime event boundary.
- `on="turn"`: update after an agent turn. This is an agent runtime boundary, like Claude Code's extraction hook.
- `on="query"`: update lazily at query time. This is a runtime cache/lazy-update strategy.
- `every="24h"`: periodic background work. In `full` refresh mode this can be implemented as a scheduler derived from freshness.
- `every=3, unit="turn"`: count-based or turn-based threshold. This is closer to a scheduler gate than Flink Materialized Table freshness.
- `require={"sessions": 5}`: additional runtime gate condition, like Claude Code autoDream's session-count gate.

Flink's DataStream window `trigger()` is only a window-level execution analogy. It decides when a window fires, and supports time/count-style triggers. It is not the same as Flink Materialized Table `FRESHNESS`, and it should not become the default v0 materialized memory view API.

An operator-level form such as `.freshness("10m")` can remain future sugar:

```python
topic = log.sem_groupby(Topic, "...").freshness("10m")
```

The primary design keeps freshness separate from the semantic query so the runtime retains optimization freedom.

This follows a standard data-systems separation:

- PostgreSQL materialized views separate `AS query` from the `REFRESH MATERIALIZED VIEW` command.
- Flink Materialized Tables separate query definition from `FRESHNESS` and `REFRESH_MODE`.
- Flink DataStream separates window assignment from trigger/fire behavior.

References:

- [PostgreSQL materialized views](https://www.postgresql.org/docs/current/rules-materializedviews.html)
- [Flink Materialized Table](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/materialized-table/overview/)
- [Flink windows](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/operators/windows/)

## 5. Join Taxonomy

Flink is a useful reference because it separates join families rather than treating all joins as one operation.

- `Regular join`: a standard SQL join between two dynamic tables.
- `Interval join`: joins two streams when their event times fall within a bounded time interval.
- `Temporal join`: joins an input row with the version of another table that was valid at the row's time.
- `Lookup join`: joins streaming input with an external table or service lookup.

Reference:

- [Flink SQL joins](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/sql/queries/joins/)

For `agent-memory`, regular semantic joins are author-facing logical joins. Existing-state joins are a memory-specific shorthand for source-to-target maintenance. Candidate retrieval can be implemented with vector, keyword, graph, or hybrid retrieval during runtime lowering without changing the logical policy.

## 6. Intermediate Expressions Vs Materialized Views

Not every relation expression is a materialized view.

```python
entity = (
    episode
    .sem_map(Entity, "Extract entity nodes from the current episode.")
    .sem_join("Resolve extracted entities against existing entity memory.", how="left")
    .sem_map(Entity, "Produce the maintained entity record.")
)
```

In this example:

- The first `sem_map(...)` output is an intermediate relation expression.
- The `sem_join(...)` output is also intermediate.
- `entity` is the materialized view because it is assigned as a named rule in an `am.Memory` class body.

This matches DataFrame-style chaining: intermediate expressions can be optimized or fused without becoming durable objects.

## 7. Structural Links And Update Effects

Graph memory systems often maintain links that are not user-authored semantic views.

For Zep/Graphiti:

- `RELATES_TO` is semantic memory content. It corresponds to a `Relation` view.
- `MENTIONS` is provenance: an episode mentions an entity.
- `HAS_EPISODE` is saga membership.
- `NEXT_EPISODE` is episode ordering.

`agent-memory` should not force users to author every structural link as a semantic view. If a graph store is used, these links may be stored as graph edges. If a non-graph store is used, they may be represented as join tables, metadata columns, or provenance lists.

The same applies to update effects. A semantic operator may decide that a fact duplicates or contradicts existing facts, but the runtime applies the physical effect:

- insert a new fact,
- reuse an existing fact,
- update a summary,
- mark an old fact invalid,
- attach provenance.

These effects are part of memory view maintenance, not separate user-facing semantic views.

## 8. Existing-State Join Example

The full Zep/Graphiti policy belongs in `user-interface-demos.md`. This API
document only needs the minimal operator pattern.

```python
entity = (
    episode
    .sem_map(
        Entity,
        "Extract entity nodes from the current episode.",
        context=episode.window(count=10),
    )
    .sem_join(
        "Resolve extracted entities against existing entity memory.",
        how="left",
        context=episode.window(count=10),
    )
    .sem_map(
        Entity,
        "Produce the maintained entity record.",
        context=[episode, episode.window(count=10)],
    )
)
```

This example documents the API rule:

- `sem_map(...)` and `sem_join(...)` are transient relation stages inside the
  top-level `entity = ...` materialized view rule.
- The omitted-right `sem_join(...)` joins the current stage against the existing
  state of the assignment target, here `entity`.
- Candidate retrieval and MERGE-like update effects are runtime/lowering
  responsibilities.

## 9. API Vs Runtime Responsibilities

Author-facing API responsibilities:

- declare log and materialized view rules,
- define semantic operator chains,
- provide output schemas and `user_instruction` text,
- attach `context=...` views or relation expressions,
- optionally declare `FRESHNESS` and `REFRESH_MODE` targets.

Runtime/lowering responsibilities:

- track deltas, cursors, watermarks, and checkpoints,
- retrieve candidates for existing-state joins,
- compile semantic operators into LLM/DSPy/retrieval plans,
- choose vector, keyword, graph, hybrid, and reranking strategies,
- apply MERGE-like insert/update/delete/invalidation effects,
- persist records, embeddings, structural links, and provenance.

## 10. Current Non-Goals

The v0 operator API should not expose these as primary policy syntax:

- physical IO operations,
- explicit vector/BM25/graph retrieval recipes,
- tracing and concurrency controls,
- multi-tenant `group_id` / namespace policy,
- debug/update return shapes,
- structural graph links unless the user explicitly chooses a lower-level policy.

For now, assume one local user and one project memory namespace. Multi-user or multi-project isolation can be added later as runtime scope and store configuration.
