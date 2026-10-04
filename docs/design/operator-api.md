# Operator API

Declare views with `agent_memory.Relation` expressions. The expressions describe
the result; the planner builds the maintenance dataflow. This reference covers
the API in this checkout. Execution settings live in
[Configuration](../configuration.md); grouped output contracts are detailed in
[Grouped Aggregation](groupby_agg.md).

## Define a memory

This example declares a topic view and a retrieval query. Defining the class or
inspecting its plan does not call a model.

```python
import agent_memory as am

class TopicMemory(am.Memory):
    log = am.Log({"message": "An incoming interaction."})
    _candidates = log.sem_flat_map(
        input_cols=["message"],
        output_cols={"topic": "Topic name.", "body": "Information to retain."},
        instruction="Extract durable topics and their content from {message}.",
    )
    topics = _candidates.sem_groupby(
        input_cols=["topic"],
        instruction="Group rows about the same durable topic.",
        membership="exclusive",
    ).sem_agg(
        input_cols=["topic", "body"],
        output_cols={"topic": "Canonical topic.", "body": "Consolidated content."},
        instruction="Merge related content into one topic without losing details.",
    )
    retrieval_query = topics.sem_topk(am.UserQuery(), 5)

spec = TopicMemory.spec()
plan = TopicMemory.differentiate_policy()
print(plan.execution_order)
print(plan.view_outputs)
```

Use one `Log` per memory. Public relation attributes become views; names
starting with `_` are private intermediate expressions. Close a grouped or
windowed relation with an aggregate or window function before exposing it as
a view. Declare `retrieval_query` instead of overriding `Memory.query()`.

`Memory(adapter=None, storage=None, refresh=None)` accepts the semantic adapter,
optional storage bindings, and an optional refresh policy. The built-in
`ClaudeMemory`, `Mem0Memory`, `Mem0MemoryEnhanced`, `ZepMemory`, and
`ZepMemoryExtended` provide concrete declarations. The benchmark system
selectors are listed separately in the [experiment guide](../../tools/evaluation/README.md).

## Messages and refresh

`add(message)` accepts a string, `Message`, or mapping. `Message` has
`content` and optional `role`, `timestamp`, `session_id`, and `metadata`.
Use mappings for a custom log schema. The runtime normalizes incoming data to
the declared columns.

`Log(columns=None, system_columns=False)` defaults to a single `message`
column. With `system_columns=True`, it adds reserved framework-owned fields:

| Field | Meaning |
| --- | --- |
| `_row_id` | UUID of the source occurrence |
| `_added_at` | UTC ingestion time |
| `_add_seq` | Zero-based append position |

These fields cannot be supplied as user columns or overwritten in input.
Ingestion order is not event time; retain an explicit timestamp for temporal
reasoning. Derived relations preserve these fields only when their expressions
keep the columns.

By default, each `add()` publishes one update. With
`CountRefresh(every=N)`, rows remain pending until the count reaches `N`.
`pending_count` reports buffered rows and `flush()` publishes the tail.
`query()` reads the last published state without flushing. Failed refreshes
leave the batch pending for retry. Refresh scheduling is independent of
logical windows and prompt batching.

## Columns and expressions

Use `relation.col("name")` for column expressions and `alias("name")` to
disambiguate join sides. Column expressions support comparisons, boolean
composition with `&` and `|`, and supported arithmetic. Parenthesize
comparisons; do not use Python `and`/`or` or row lambdas.

`input_cols` selects columns supplied to a semantic operation; omitting it
uses that operator's default input scope. `output_cols` accepts a sequence of
names or a mapping from names to natural-language descriptions. Prefer explicit
inputs and outputs in reusable policies. Instructions can reference columns
with `{column}`; join instructions use `{column:left}` and `{column:right}`
to distinguish the two contexts.

## Relational operators

| Expression | Result |
| --- | --- |
| `r.select(columns)`, `r[columns]` | Selected columns, in the requested order |
| `r.filter(predicate)` | Rows satisfying a deterministic expression |
| `r.assign(**assignments)` | Add or replace columns with expressions or literals |
| `r.limit(n)` | At most `n` rows, without an ordering or subset guarantee |
| `r.alias(name)` | Named relation for qualified column references |
| `r.concat(s)` | Concatenation preserving duplicates |
| `r.union(s)` | Union with exact duplicate removal |
| `r.union_by_name(s, allow_missing_columns=True)` | Align columns by name, then union; missing columns are filled |
| `r.subtract(s)` | Exact set difference, not multiplicity subtraction |
| `r.drop_duplicates(subset=None)` | Exact duplicate removal across all or selected columns |
| `r.join(s, on=keys_or_expression, how="inner")` | Deterministic join |
| `r.group_by(keys)` | Exact-key grouping, followed by an aggregate |

For joins, `on` accepts a key name, sequence of keys, or deterministic
predicate. Key joins support `inner`, `left`, `right`, `outer`, and `left_anti`;
`left_anti` returns left rows without a matching right key. Predicate joins
support `inner`. In key joins, shared key columns appear once, and other
same-named columns receive `:left` and `:right` suffixes, or the corresponding
relation aliases. Predicate joins qualify columns by side.
Aliases make predicate joins explicit:

```python
import agent_memory as am

records = am.Log({"id": "Record identifier.", "value": "Numeric value."})
left = records.alias("left")
right = records.alias("right")
pairs = left.join(right, on=left.col("id") != right.col("id"))
selected = records.filter(records.col("value") > 0).select(["id", "value"])
```

`assign` also supports the exported `least`, `try_cast`, and `case_when`
expressions. These are deterministic row expressions, not LLM tasks.
For detailed expression signatures, see
[expressions.py](../../src/agent_memory/policy/expressions.py).

## Semantic operators

| Expression | Row contract |
| --- | --- |
| `sem_filter(instruction=...)` | Keep or reject each input row |
| `sem_map(input_cols=None, output_cols=..., instruction=...)` | One generated result per input row; preserve other source columns |
| `sem_flat_map(input_cols=None, output_cols=..., instruction=..., ordinal_col=None)` | Zero or more results per input row, retaining source context |
| `sem_join(other, instruction=..., how="inner", on=None, k=None)` | Match rows across two relations |
| `sem_groupby(input_cols=..., instruction=..., partition_by=None, labels=None, label_col="_label", membership=None)` | Group by a semantic criterion; close with an aggregate |
| `sem_agg(input_cols=None, output_cols=None, instruction=...)` | Synthesize an aggregate result over the relation or each group |
| `sem_topk(instruction, k)` | Rank and retain up to `k` rows |

Map and flat-map outputs replace same-named generated columns rather than
duplicating them. An optional flat-map `ordinal_col` records each generated
row's zero-based position within its source row. It must be a new, non-empty
column name distinct from the source and generated columns. A source row
that emits no records contributes no output rows.

Column descriptions in `output_cols` guide generation; they are not Python
dtype declarations. For `sem_agg`, omitted `input_cols` selects all columns
except the internal semantic-group identifier. Omitted `output_cols` reuses
the selected input names for the aggregate result. Explicit names make the
result schema easier to use downstream. Row-frame `sem_agg` requires explicit
`output_cols`.

For a `sem_join` declared directly in a maintained memory view, use
`how="inner"` and `k=None`. An `on` predicate restricts candidates before
semantic matching.

Static adapter execution and compiler-generated internal joins also support
`left`, `right`, and `outer`; unmatched outer-side rows retain their data with
nulls for the opposite side. A positive `k` selects bounded targets per anchor
through the configured resolver. These execution forms are separate from
direct semantic-join view maintenance.

`sem_topk` is used in retrieval, where `UserQuery()` is bound at query time.
Its available ranking algorithms are execution choices. Do not assume a
general incremental top-k maintenance rule for an arbitrary public view.

Semantic grouping requires `input_cols`, a sequence of column names. It can
use exact `partition_by` keys (one name or a sequence) before applying its
instruction. With `labels`, a mapping of label names to descriptions,
grouping assigns one of the declared label
names to `label_col`. Without labels, matching discovers groups.
`membership="exclusive"` constrains join-map matching to at most one existing
target; `None` does not impose that extra restriction. The name
`"overlapping"` is reserved but is not implemented by the current executor.
See the aggregation guide before choosing output columns or combining semantic
and deterministic aggregates.

## Aggregates and arrays

| Operator | Meaning |
| --- | --- |
| `r.count(output_col=...)` | Count all rows |
| `r.sum(column=..., output_col=...)` | Sum non-null numeric values |
| `r.avg(column=..., output_col=...)` | Average non-null numeric values |
| `r.agg(am.count(...), am.sum(...), am.avg(...))` | Several numeric aggregates in one result |
| `r.array_agg(columns=..., output_col=...)` | JSON array of records from selected columns |
| `r.min(column=..., output_col=...)` | Minimum non-null scalar |
| `r.min(columns=..., output_col=...)` | Lexicographically minimum complete tuple |
| `r.array_cat(s, column=...)` | Combine array-aggregate state in that column |
| `r.flatten(column=..., output_col=None)` | Flatten one array nesting level |
| `r.explode(column=..., output_col=None)` | One output row per array element |
| `r.unnest(column=..., fields={...})` | Project object fields into named columns |

`min` requires exactly one of `column` or `columns`; the composite form
ignores rows with a null component. `least(a, b, ...)` instead computes a
minimum within one row.

`array_agg` returns a serialized record array, not a plain list of scalar
values. For a value list inside grouped `.agg(...)`, use
`am.collect_list(column=..., output_col=...)`. Preserve these distinctions when
passing results to `flatten`, `explode`, or `unnest`.
When supplied, a separate flatten/explode output name must not silently
overwrite an unrelated column.

Exact `group_by` supports numeric aggregates as a family; they cannot be mixed
with semantic or array aggregate specs in the same `.agg(...)`.
Semantic groups require a semantic aggregate to generate their canonical keys;
numeric aggregates on semantic groups are not supported.

## Windows

Windows belong to the view definition: they change which rows a computation
sees. They are not `CountRefresh` and are not prompt packing.

### Completed count windows

`r.count_window(size=N, slide=1, trigger=None).process_window(builder)`
runs the builder over each completed window of `N` rows.
Both sizes are positive integers; only `trigger=None` is supported.
Ordering follows source arrival, not a timestamp field.
No output is emitted for an incomplete trailing window.

```python
import agent_memory as am

log = am.Log({"message": "Incoming text.", "timestamp": "Event time."})
blocks = log.count_window(size=10, slide=10).process_window(
    lambda window: window.array_agg(
        columns=["timestamp", "message"], output_col="records"
    )
)
```

`slide < size` overlaps windows, equality makes non-overlapping windows, and
`slide > size` leaves gaps. The builder runs during policy definition to
construct a query; it is not an arbitrary runtime Python UDF.
Its result is an ordinary relation. Operators after `process_window` are
global, not implicitly window-local. The current maintained form supports one
`process_window` boundary per public view, not nested or chained boundaries.

### Row-preserving frames

`r.over(rows=(M, N))` defines an inclusive positional frame with
`M <= N <= 0`. Offset zero is the current row; negative offsets refer to
earlier rows. Frames are clipped at the start of the input.

```python
import agent_memory as am

log = am.Log({"message": "Incoming text."})
context = log.over(rows=(-2, 0)).array_agg(
    columns=["message"], output_col="recent_messages"
)
```

Close a frame with `array_agg(columns=..., output_col=...)` or
`sem_agg(input_cols=None, output_cols=..., instruction=...)`.
The result retains each emit row and attaches the aggregate for its frame.
These are backward-looking row frames, not event-time windows, watermarks,
session windows, or partitioned SQL frames.

## Retrieval

A relation-valued `retrieval_query` can rank materialized rows with
`sem_topk(UserQuery(), k)`. Storage-backed retrieval uses:

```text
relation.search(UserQuery(), methods=[...], reranker=..., limit=20)
```

Available search descriptors are `BM25()`,
`CosineSimilarity(candidate_limit=None, min_score=None)`, and
`BFS(origins=entity_results, max_depth=3)`. Rerank with `RRF()` or
`CrossEncoder(model=...)`. Search results include `record_id`, `rank`,
and `score`; input views must not collide with these names.
Search is query-time work and requires a compatible storage search provider.

Use `RetrievalQuery(entities=..., facts=...)` for named search-result channels;
each value must be a `SearchRelation`.
`Memory.query(text)` returns the bound retrieval result, not an answer.
The application or benchmark harness owns answer generation.

## Validate a policy

Inspect `spec()` and `differentiate_policy()` before calling `add()`.
Compilation validates the maintenance plan without model calls. Use a fake
adapter to check dataflow and output assembly offline.

The [interface smoke](../../examples/claude/interface_smoke.py) prints a
complete built-in policy and its maintenance nodes without model access.

## Execute DataFrame batches

Use `SemanticDataflow` when the application supplies pandas batches directly,
rather than individual `Message` objects:

```python
import pandas as pd
import agent_memory as am

source = am.Source({"value": "Numeric value."})
flow = am.SemanticDataflow(
    source=source,
    views={"positive": source.filter(source.col("value") > 0)},
)
flow.apply(pd.DataFrame({"value": [-1, 2, 3]}))
print(flow.view("positive"))
```

`apply(rows)` appends one source batch and updates the declared views.
`view(name)` returns a copy of the named result. All views must derive from
the declared source. Pass `adapter=...` for semantic execution.
`snapshot_state()` returns state for trusted persistence;
`restore_state(snapshot)` checks compatibility before restoring it.
