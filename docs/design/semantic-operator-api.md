# Semantic Operator API

This document records the current design target for semantic operators in `agent-memory`. It is not an implementation guarantee.

The goal is to keep the authoring surface close to DataFrame, SQL, and Flink concepts while still allowing semantic operators to lower into LLM calls, DSPy programs, retrieval plans, and durable memory maintenance.

## 1. Mental Model

`agent-memory` separates author-facing logical policy from runtime execution.

- `Relation expression`: a lazy dataframe-like expression over logs, views, or intermediate results. It is not durable by itself.
- `Materialized view rule`: a top-level memory assignment whose right-hand side ends with `.refresh(...)`.
- `Runtime/lowering`: cursor handling, candidate retrieval, MERGE-like update effects, embeddings, vector/BM25/graph search, reranking, and physical writes.
- `Store`: physical materialization for logs and views. It does not define the logical schema or operator prompt.

```python
topic = log.sem_groupby(
    Topic,
    "Group durable memories into stable topic buckets.",
).refresh(on="turn")
```

In this example, `log.sem_groupby(...)` is the logical relation expression, `topic` is the materialized view name, and `refresh(on="turn")` is the view maintenance timing.

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
).refresh(on="turn")
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

The runtime may lower one `sem_groupby(...)` into extraction, candidate retrieval, matching, merge, and storage steps. Those are physical maintenance details unless the policy author chooses to write a lower-level workflow.

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

If the right side is omitted, `sem_join(...)` means an existing-state join. This form is only valid inside a refreshed materialized view rule, because the assignment target provides the existing target view.

```python
entity = (
    episode
    .sem_map(Entity, "Extract entity nodes from the current episode.")
    .sem_join(
        "Resolve extracted entities against existing entity memory.",
        how="left",
    )
    .sem_map(Entity, "Produce the maintained entity record.")
    .refresh(on="source")
)
```

The compiler lowers this as:

```text
source = extracted entity rows
target = existing materialized entity view
join = source sem_join target
effect = MERGE-like update into target view
```

This is not a normal self join. It is closer to SQL `MERGE`, where source rows are matched against an existing target table before insert/update/delete effects are applied.

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

## 4. Window And Refresh

`window(...)` and `refresh(...)` are not semantic operators.

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

`refresh(...)` defines when a materialized view is maintained.

```python
topic = log.sem_groupby(Topic, "...").refresh(on="turn")
catalog = topic.sem_map(CatalogEntry, "...").refresh(on="source")
profile = topic.sem_agg(Profile, "...").refresh(every="24h")
draft = log.sem_map(Draft, "...").refresh(mode="manual")
```

This follows a standard data-systems separation:

- PostgreSQL materialized views separate `AS query` from `REFRESH MATERIALIZED VIEW`.
- Flink Materialized Tables separate query definition from freshness and refresh mode.
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
    .refresh(on="source")
)
```

In this example:

- The first `sem_map(...)` output is an intermediate relation expression.
- The `sem_join(...)` output is also intermediate.
- `entity` is the materialized view because it is assigned as a named rule and the right-hand expression has a `refresh(...)` policy.

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

## 8. Zep/Graphiti Mapping

A source-aligned Zep policy should express durable views plus transient operator stages:

```python
episode = log.sem_map(
    Episode,
    "Convert incoming events into stored episodes with source, content, and reference time.",
).refresh(on="add")

entity = (
    episode
    .sem_map(
        Entity,
        "Extract entity nodes from the current episode. Use previous episodes only for disambiguation.",
        context=episode.window(count=10),
    )
    .sem_join(
        "Resolve extracted entities against existing entity memory.",
        how="left",
        context=episode.window(count=10),
    )
    .sem_map(
        Entity,
        "Reuse an existing identity when matched; otherwise create a new entity. Update attributes and summary.",
        context=[episode, episode.window(count=10)],
    )
    .refresh(on="source")
)

relation = (
    episode
    .sem_map(
        Relation,
        "Extract factual relationships between resolved entities, including temporal validity when available.",
        context=[entity, episode.window(count=10)],
    )
    .sem_join(
        "Resolve extracted facts against existing relation memory. Reuse duplicates and invalidate contradicted older facts.",
        how="left",
        context=[episode, entity],
    )
    .refresh(on="source")
)
```

This documents the required operator capabilities:

- chainable intermediate relation expressions,
- semantic joins with relational `how`,
- existing-state joins for materialized view maintenance,
- candidate retrieval handled by runtime/store lowering,
- MERGE-like update effects hidden behind view refresh,
- structural links hidden behind store/runtime configuration.

## 9. API Vs Runtime Responsibilities

Author-facing API responsibilities:

- declare log and materialized view rules,
- define semantic operator chains,
- provide output schemas and `user_instruction` text,
- attach `context=...` views or relation expressions,
- choose logical maintenance timing with `.refresh(...)`.

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
