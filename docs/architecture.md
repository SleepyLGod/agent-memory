# Architecture

Agent Memory separates a memory definition from its maintenance and retrieval.
A policy declares relations over incoming messages; the planner derives a
maintenance dataflow; the runtime updates its state and publishes view changes.

```text
Memory class: Log + relational/semantic expressions + retrieval query
    -> policy specification
    -> differentiated maintenance DAG
    -> incremental executor + semantic adapter
    -> materialized views and optional storage sinks
    -> retrieval against the published state
```

## Declare the views

A `Memory` class contains one `Log`, intermediate relations, public views, and
an optional `retrieval_query`. Relations prefixed with `_` are private
intermediates. A relation is an expression, not an eagerly evaluated table.
`Memory.spec()` collects the declaration without calling a model.

The built-in policies illustrate different representations: Claude-style
topics and a catalog, Mem0-style fact records, and Zep-style entities, episodes,
and facts. They are policies implemented in this framework; the native products
have their own execution repositories.

## Compile maintenance

`Memory.differentiate_policy()` builds a `DifferentiatedPolicy`. Its
`execution_order` orders nodes, `view_outputs` identifies public sinks, and
`retrieval_queries` holds the separately compiled retrieval expressions.
Shared dependencies become shared nodes rather than independent view runs.

The planner applies operator-specific rules. Row-local work can process changed
rows; grouped semantic aggregation maintains retained group state. The default
grouped-aggregate strategy is join-map: associate incoming group information
with existing groups, then construct updated group content. See
[grouped aggregation](design/groupby_agg.md) for its input and output contracts.

The compiler validates operator combinations against the supported maintenance
rules. The semantic adapter evaluates matching and synthesis using the selected
model, instructions, and retained state.

## Execute and publish

`Memory.add()` normalizes a message and sends it through the runtime. The
executor retains source, intermediate, and output state, together with row
occurrence identities. A changed aggregate can replace an earlier row, so an
append-only source does not imply append-only intermediate views.

Execution stages new state before publishing it. Downstream nodes consume
insertions and retractions with their multiplicities. If execution fails,
uncommitted state is not published as a successful update. External model calls
already made during that attempt are not undone.

The default is one refresh per added message. `CountRefresh(every=N)` buffers
messages until the count boundary; `flush()` publishes a partial tail.
`query()` does not implicitly flush. This scheduling choice is separate from
logical windows and from packing several model tasks into a prompt.

## Lower semantic operators

The LOTUS adapter evaluates the semantic expressions and ordinary DataFrame
operations required by the plan. Its execution configuration controls model
dispatch, candidate screening, structured responses, prompt batching, and trace
detail. These controls do not require the application to write its own provider
client. Configuration and compatibility checks are documented in the
[configuration reference](configuration.md).

## Store and retrieve

Materialized views can be consumed in memory or bound to storage through a
`StatementSet` and typed table descriptors. Neo4j mappings support graph-backed
Zep retrieval; Qdrant mappings support Mem0 fact retrieval. Storage schemas,
identities, embeddings, and search indexes belong to the storage profile, not
to the model prompt.

A retrieval query reads published views. It may use semantic ranking or
storage-backed search with BM25, cosine similarity, BFS, and reranking.
Retrieval returns context; generating and grading an answer are separate steps
owned by the benchmark harness or calling application.

## Checkpoints and experiments

Runtime snapshots retain execution state and validate the plan fingerprint on
restore. Storage-backed drivers additionally restore the corresponding storage
state. A final view alone is not a replacement for an executor checkpoint.
Only restore trusted local snapshots.

The shared benchmark harness owns input bundles, case isolation, checkpoints,
answering, scoring, and artifacts. Its run contract records the selected
system and execution settings. See the [experiment guide](../tools/evaluation/README.md)
for the supported recovery workflow and comparison checks.

## Source map

| Component | Source |
| --- | --- |
| Public memory API | [`api.py`](../src/agent_memory/api.py) |
| Expressions and schemas | [`policy/`](../src/agent_memory/policy/) |
| Maintenance and retrieval planning | [`planner/`](../src/agent_memory/planner/) |
| State and execution | [`runtime/`](../src/agent_memory/runtime/) |
| Semantic execution | [`adapters/lotus/`](../src/agent_memory/adapters/lotus/) |
| Built-in policies | [`memories/`](../src/agent_memory/memories/) |
| Storage and search | [`storage/`](../src/agent_memory/storage/) |
| Benchmark harness | [`evaluation/`](../src/agent_memory/evaluation/) |
