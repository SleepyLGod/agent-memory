# DataFrame Execution, Memory Views, And Semantic IVM

This note records the current design story around DataFrame execution, semantic query systems, materialized memory views, lazy update, and incremental view maintenance.

Status: the current golden API is DataFrame-first and is defined in `operator-api.md`. This file is rationale/background for that direction.

## 1. DataFrame Execution: Eager, Lazy, And Semantic Query Systems

DataFrame lazy execution and materialized view lazy update are different concepts.

DataFrame lazy execution means the user writes a query plan, but the system does not execute it immediately. When `collect` / `run` is called, the system optimizes and executes the query at once. Execution is deferred until query-time.

A DataFrame can be understood as an already-loaded two-dimensional table. When users apply filtering, selection, aggregation, and similar operations to it, these operations usually run immediately and produce actual results or intermediate results. This is eager execution.

A LazyFrame is more like an execution plan or query DAG. A chain of operations is recorded first and does not run immediately. The system executes the full plan only when `.collect()` is called. This is lazy execution.

Conceptually, DataFrame is the table abstraction, and lazy/eager belongs to the execution engine. In concrete libraries, the lazy/eager distinction often appears as different objects, such as Polars `DataFrame` versus `LazyFrame`.

### System Positions

- Polars is the strongest substrate for deterministic lazy relational query processing. It handles relational operators over structured data and has both eager and lazy modes.
- LOTUS is eager semantic DataFrame processing. LOTUS chose pandas DataFrame accessor as the user interface, so each `sem_*` call executes over the current DataFrame and returns a DataFrame. It does not fully rebuild DataFrame into a logical-plan system. It directly registers semantic operator namespaces on DataFrame.
- Palimpzest is lazy semantic Dataset processing. It implemented its own `Dataset` abstraction: a `Dataset` represents a collection of structured or unstructured data, can be a root `Dataset` or the result of operations, and `sem_filter` / `sem_map` / `sem_join` / `sem_agg` lazily create a new `Dataset`. `run()` executes computation and retrieves the materialized `Dataset`. Root `Dataset` must subclass `IterDataset` / `IndexDataset` / `Context`.

The Palimpzest source code also shows that each operator creates a new `Dataset(sources=[...], operator=LogicalOperator, schema=...)`. For example:

- `sem_filter` creates `FilteredScan`.
- `sem_map` creates `ConvertScan`.
- `sem_join` creates `JoinOp`.

The reason is not only multimodal data. Palimpzest also needs query-processing control that pandas/Polars DataFrame does not own.

The current positioning is:

- Palimpzest is the closest existing substrate for our desired semantic lazy query layer.
- LOTUS is the closest existing substrate for user-facing semantic DataFrame ergonomics.
- Polars is the strongest substrate for deterministic lazy relational query processing.

## 2. Memory Views Are Different From Query-Time Processing

LOTUS and Palimpzest often follow this model: the user submits a dataset query and executes it.

Agent memory follows a different model: the system continuously maintains durable views over a growing log, then agents query those views.

In other words, we need:

- semantic lazy query layer
- durable materialized memory view runtime
- agent memory protocols

For a relation/view such as `topics`, it is first a DataFrame-like object in its logical form. At the same time, it is also a materialized view with lifecycle.

At the current stage, agent memory is also eager in the memory-update sense.

Memory lazy update means the base log has changed, but some materialized memory views are not updated immediately. They are updated later, for example when needed, during low-traffic periods, in batches, or when a relevant query arrives.

The main values are:

- Avoid triggering expensive semantic rollups for every message.
- Batch multiple updates.
- Maintain only the views used by queries.
- Recompute only dirty local regions.
- Use budget to control background maintenance.

For example:

```python
profile = topics.sem_groupby(
    key=["topic_name"],
    instruction="Roll up related topic rows into profile groups.",
    agg=sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols=["profile_name", "profile_content"],
        instruction="Produce one profile row per group.",
    ),
)
```

This lazy query plan describes how a `profile` view is derived from a `topics` view.

The memory runtime decides what happens after `topics` changes:

- Update `profile` immediately.
- Delay `profile` update.
- Update `profile` locally.
- Update `profile` at query-time.

Whether we should use memory lazy update is still uncertain.

## 3. View, Lazy Update, And IVM Are Different

View, lazy update, and incremental view maintenance are separate concepts.

- `view` = a derived relation / derived table / derived memory state
- `lazy update` = the policy for when to maintain this view
- `IVM` = the algorithm for maintaining this view using delta

A view does not imply lazy update. A view also does not imply IVM.

The classic IVM comparison is:

```text
query(Data D): Cost C

No IVM:
query(Data D + δD): Cost C'

With IVM:
maintain query result using δD: Cost δC << C'
```

Lazy update is another dimension:

```text
After δD arrives, do we pay δC immediately, or do we pay δC later?
```

Whether IVM clearly applies to memory is still uncertain.

Memory systems often look naturally incremental, but they are not necessarily strict IVM systems. They usually do not recompute all memory from the entire log every time. Existing memory systems use ad-hoc incremental updates.

Our goal is to formulate memory as semantic materialized views, so we can reason about delta maintenance systematically.

In other words, the story is not:

```text
full recompute vs incremental
```

The story is:

```text
ad-hoc incremental update vs view-defined incremental maintenance
```

A traditional memory update looks like:

```text
message -> LLM decides add/update/delete memories
```

Problems:

- Hard to know which views depend on which memory.
- Hard to batch or defer safely.
- Hard to optimize cost globally.
- Hard to rebuild/check consistency.
- Hard to compare update result with full recompute.

View-defined memory looks like:

```python
topics = log.sem_groupby(key=[...], instruction="...", agg=sem_agg(...))
profile = topics.sem_groupby(key=[...], instruction="...", agg=sem_agg(...))
```

Benefits:

- Clear dependencies.
- Clear materialized views.
- Clear update paths.
- Possible delta propagation.
- Possible background maintenance.
- Possible cost-aware update policy.
- Possible full rebuild fallback.

This story is strong.

## 4. Why We Should Not Directly Use LOTUS Or Palimpzest As The Full Memory Runtime

For LOTUS, Palimpzest, and similar systems, the current conclusion is:

- We should primarily learn from / reuse their API and operator interface.
- We should not assume we can directly use them as the full memory runtime.

The reason is not that they are weak. The reason is that their main target is different.

The main target of Palimpzest and LOTUS is query-time semantic data processing.

The user has a batch of data:

```text
Dataset / DataFrame
```

Then the user writes:

```text
semantic query / semantic analytics pipeline
```

The system executes the query and returns the result.

This is closer to OLAP:

- There is no long-lived materialized memory view.
- Data is static.
- Query is dynamic.
- After the query finishes, the job is done.

In short:

```text
I have a table. Run semantic filters/maps/joins now.
```

Our main target is long-lived agent memory.

There is a growing log:

```text
messages / events
```

The system continuously maintains:

```text
durable semantic views
```

Agents query these views at any time for:

- context injection
- relevant memory
- state update

This is closer to:

```text
streaming + view
```

In this model:

- Data is dynamic.
- Query is static.
- View is dynamic.

In short:

```text
I have a growing log. Maintain durable semantic views over time.
```
