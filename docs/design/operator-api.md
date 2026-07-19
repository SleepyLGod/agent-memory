# Operator API

This document records the current target API for `agent-memory` operators. It is
a design target, not an implementation guarantee.

The current direction is DataFrame-first. Logs, memory views, and intermediate
results are dataframe-like relations. Users write a full view definition query
`Q`; the system derives differentiated maintenance queries `Q'` and chooses
runtime plans for cost, latency, and freshness.

## 1. Mental Model

`agent-memory` separates logical query authoring from runtime maintenance.

- `Relation`: a dataframe-like table over log rows, memory rows, or intermediate
  rows.
- `View definition query`: the full logical query `Q` that defines what memory
  should contain.
- `Differential query`: the derived maintenance query `Q'` that updates the view
  from new data without recomputing the full history.
- `Ordinary operator`: deterministic dataframe or relational operation such as
  `select`, `filter`, `assign`, `join`, `concat`, `union`, `union_by_name`,
  `explode`, `unnest`, or `subtract`.
- `Semantic operator`: instruction-driven operation backed by LLMs, embeddings,
  rerankers, DSPy programs, or other semantic execution plans.

Example:

```python
topic_candidates = log.sem_flat_map(
    input_cols=["message"],
    output_cols={
        "topic_name": "Candidate durable memory topic name.",
        "topic_content": "Candidate durable memory content.",
    },
    instruction="Extract zero or more durable memory topic candidates from {message}.",
)

topics = (
    topic_candidates
    .sem_groupby(
        input_cols=["topic_name"],
        instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    )
    .sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols={
            "topic_name": "Canonical durable memory topic name.",
            "topic_content": "Merged durable memory content.",
        },
        instruction="Choose a canonical topic name and merge topic content.",
    )
    .select(["topic_name", "topic_content"])
)

catalog = (
    topics
    .sem_map(
        input_cols=["topic_name", "topic_content"],
        output_cols={
            "catalog_title": "Title shown in the memory catalog.",
            "path": "Relative path to the topic memory file.",
            "hook": "One-line relevance hook for future retrieval.",
        },
        instruction="Produce one catalog row per topic.",
    )
    .select(["catalog_title", "path", "hook"])
)
```

## 2. Ordinary DataFrame Operators

Ordinary operators should follow existing DataFrame naming wherever possible.
They are deterministic and cheap relative to semantic operators, so the runtime
can push, fuse, or reorder them when it is safe.

### `Log` source metadata

`Log(..., system_columns=True)` adds three framework-owned source columns:

- `_row_id`: UUID string for the appended source row.
- `_added_at`: UTC time when runtime appended the row.
- `_add_seq`: zero-based append position in this log.

The default is `system_columns=False`, so existing policies keep their current
schema. These names are reserved: policy schemas and `Memory.add(...)` callers
cannot provide them. Derived relations preserve source metadata like ordinary
columns; they do not receive new row IDs or append times.

```python
log = am.Log({"content": "Episode content."}, system_columns=True)
episodes = log.assign(
    episode_id=log.col("_row_id"),
    created_at=log.col("_added_at"),
    add_seq=log.col("_add_seq"),
)
```

`_add_seq` is ingestion order, not event time. Policies must keep a separate
field such as `reference_time` or `valid_at` when real-world temporal order
matters.

### `select`

Projection over columns.

```python
df.select(["topic_name", "topic_content"])
df[["topic_name", "topic_content"]]
```

`select` and bracket selection are equivalent authoring forms. The operation is
relational projection, but `select` is the public dataframe/table spelling used
by systems such as Spark, Polars, and Flink. SQL also uses `SELECT` for output
columns/expressions; row filtering is `WHERE` / `filter`.

### `filter`

Deterministic row filter.

```python
changed = rows.filter(rows.col("action") != "keep")
expired = rows.filter(rows.col("invalid_at").is_not_null())
```

This is not `sem_filter`. The predicate is a small serializable relational
expression built from `relation.col(...)`; arbitrary Python callables, tuple
predicates, and SQL strings are not part of the public contract.

### `assign`

Deterministic column creation or replacement.

```python
rows = rows.assign(
    invalid_at=rows.col("valid_at:new"),
    status="inactive",
)
```

Use `assign` for cheap, deterministic fields. Use `sem_map` when the new fields
require semantic interpretation. Assignment values are scalar literals, column
expressions, row-wise `array_cat`, or `am.least(...)` expressions.

`am.least(a, b, ...)` is a scalar expression, not an aggregate. It compares two
or more values in the same row, ignores null operands, and returns null only
when every operand is null.

### `concat`

Append rows without exact deduplication.

```python
combined = left.concat(right)
```

`concat` has union-all semantics.

### `drop_duplicates`

Exact duplicate removal.

```python
deduped = rows.drop_duplicates()
```

This is exact row equality, not semantic equality.

### `union`

Relational `UNION`: append rows, then remove exact duplicates.

```python
result = left.union(right)
```

Equivalent definition:

```python
result = left.concat(right).drop_duplicates()
```

`union` is exact relational union. It is not `sem_union`; semantic union/upsert
is backend theory terminology, not a v0 public dataframe operator.

### `union_by_name`

Append rows by column name, optionally filling missing columns with nulls before
exact deduplication.

```python
result = left.union_by_name(right, allow_missing_columns=True)
```

`union_by_name` is useful when two branches have the same logical schema but
different column order, or when one branch lacks optional columns. It is still a
deterministic dataframe operation, not a semantic merge.

### `subtract`

Relational set difference / `EXCEPT`.

```python
remaining = rows.subtract(rows_to_remove)
```

`subtract` removes rows using exact equality or an implementation-defined exact
key. It is not a semantic delete.

### `alias`, `col`, and `join`

Deterministic relational join. Same-named key joins keep the existing compact
form:

```python
joined = selected.join(topics, on="name", how="inner")
```

Self-join or temporal candidate construction uses relation aliases and
relation-bound column expressions:

```python
old = facts.alias("old")
new = facts.alias("new")

pairs = old.join(
    new,
    on=[
        old.col("fact_id") != new.col("fact_id"),
        old.col("source_entity_id") == new.col("source_entity_id"),
        old.col("target_entity_id") == new.col("target_entity_id"),
        old.col("valid_at") <= new.col("valid_at"),
    ],
)
```

`join(on=...)` is pandas-backed exact lookup / merge or deterministic predicate
join. It does not call an LLM. Key joins use the `:left` / `:right` suffix
convention for overlapping non-key columns. Alias predicate joins emit columns
such as `fact_id:old` and `fact_id:new`. This operator is not interchangeable
with `sem_join(...)`: `sem_join` asks an LLM whether two rows semantically
match, while `join(...)` applies deterministic relational predicates.

The expression subset is intentionally small: column references, scalar
literals, comparisons, boolean `&` / `|` / `~`, `.isin(...)`, `.is_null()`,
`.is_not_null()`, row-wise `array_cat`, and `am.least(...)`.

## 3. Aggregate And Window Operators

These operators are part of the public authoring surface, but they use different
receiver types. A `WindowedRelation` or `OverRelation` is not an ordinary
`Relation`; it must be closed by one of its allowed window functions before the
query can continue.

For window semantics and differential rules, see `docs/design/window.md`.

### `array_agg`

Deterministic aggregate that turns relation rows into one JSON array-of-records
column.

```python
records = log.array_agg(
    columns=("timestamp", "speaker", "message"),
    output_col="conversation_records",
)
```

Contract:

- input receiver: `Relation`
- cardinality: many input rows to one output row
- output receiver: ordinary `Relation`
- output columns: one JSON text array-of-records column named by `output_col`
- value format: stable JSON text array of records
- missing scalar values: encoded as JSON `null`; non-standard `NaN` is never
  emitted

`array_agg(columns=...)` preserves row alignment inside each record. It does not
create one independent array per column.

Grouped receivers also support `array_agg(...)`:

```python
episode_entities = resolved_mentions.group_by("episode_id").array_agg(
    columns=("entity_id", "name", "summary"),
    output_col="entities",
)
```

Exact `group_by(...).array_agg(...)` emits group key columns plus the JSON array
column. Direct `sem_groupby(...).array_agg(...)` is not supported because
`array_agg` cannot generate semantic key columns. Use mixed
`sem_groupby(...).agg(sem_agg(...), array_agg(...))` when semantic grouping
also needs array evidence.

### `collect_list`

Grouped deterministic aggregate that collects one existing column into a JSON
array of values.

```python
states = rows.group_by("topic").agg(
    am.collect_list(column="evidence", output_col="evidence"),
)
```

`collect_list` is mainly an implementation-facing aggregate for remerging
already aggregated array state. Policy authors usually want `array_agg(...)`,
which stores records, not one scalar value list.

### `min`

Deterministic aggregate that returns either the minimum non-null value or the
lexicographically minimum complete tuple:

```python
earliest = rows.min(column="valid_at", output_col="valid_at")
earliest_by_fact = rows.group_by("fact_id").min(
    column="invalid_at",
    output_col="invalid_at",
)
entity_identity = rows.group_by("entity_name").min(
    columns=["add_seq", "entity_ordinal"],
    output_col="entity_id",
)
```

`column=...` and `columns=[...]` are mutually exclusive. The composite form
compares tuples in declared column order. A row with a null tuple component is
not eligible; if no complete row remains, the result is null.

`am.min(...)` is the aggregate-spec form for grouped mixed `.agg(...)`. Global
empty input and an all-null group produce null; grouped empty input produces
zero groups. Direct
`sem_groupby(...).min(...)` is unsupported, but `am.min(...)` may appear in a
mixed semantic `.agg(...)` that also contains the required `sem_agg(...)`.

`min` compares values across rows. `am.least(...)` compares values within one
row; they are different operators.

Using `(add_seq, ordinal)` as a grouped minimum gives an append-only logical
occurrence identity. It is stable while earlier occurrences remain in the same
group, but it is not a permanent UUID: full recomputation may change extraction
order, and future group merge/split support needs downstream remapping. A
storage backend may map this logical tuple to a physical UUID later.

### `array_cat`

Deterministic combine operator for one JSON array aggregate-state column.

```python
next_records = current_records.array_cat(delta_records, column="records")
```

Contract:

- input receivers: two ordinary `Relation` values
- supported state shape: the named column contains JSON arrays
- output receiver: ordinary `Relation`
- output columns: the same array column
- primary use: `array_agg` differential maintenance

`array_cat` is not a semantic merge. It concatenates JSON arrays. It currently
has no generic differential rule beyond the `array_agg` view-boundary rule.

Row-wise array concatenation is available inside `assign(...)`:

```python
merged = joined.assign(
    evidence=joined.col("evidence:right").array_cat(joined.col("evidence:left"))
)
```

This form concatenates JSON arrays within each joined row. It is used by grouped
aggregate join-map lowering.

### `flatten`

Flatten one JSON array-of-arrays value into one JSON array value, without
changing row count.

```python
flat = rows.flatten(column="evidence")
```

Example value:

```text
["[{\"x\": 1}]", "[{\"x\": 2}]"] -> [{"x": 1}, {"x": 2}]
```

`flatten` is for aggregate-state remerge. It does not emit more rows. It may
replace its input column in place or write a new column, but it never silently
overwrites another existing column.

### `explode`

Expand one JSON array column into one row per element.

```python
exploded = entities.explode(column="mentions", output_col="_mention")
```

Contract:

- input receiver: ordinary `Relation`
- input value: JSON array or native list
- cardinality: one input row to zero or more output rows
- null or empty arrays: emit zero rows
- with `output_col=None`: replace the source column with each element
- with `output_col="..."`: keep the source array column and append the element
  column

`explode` is deterministic and row-local. It is the dataframe-style collection
expansion operator; it does not unpack object fields.

### `unnest`

Expand one JSON object / struct column into ordinary columns, without changing
row count.

```python
episode_entities = (
    entities
    .select(["entity_id", "name", "summary", "mentions"])
    .explode(column="mentions", output_col="_mention")
    .unnest(
        column="_mention",
        fields={
            "episode_id": "episode_id",
            "entity_ordinal": "entity_ordinal",
            "name": "mention_name",
        },
    )
)
```

`unnest` is deterministic and row-local. It removes the object column and
appends the declared fields. Missing fields, non-object values, and output
column conflicts are errors. Every destination column in `fields` must also be
unique; hand-written or restored logical IR is validated by both schema
inference and execution.

`flatten`, `explode`, and `unnest` are intentionally separate:

- `flatten`: array-of-arrays stays in one row.
- `explode`: array elements become rows.
- `unnest`: object fields become columns.

### `count_window`

Count-window assigner. It returns `WindowedRelation`, not ordinary `Relation`.

```python
blocks = (
    log
    .count_window(size=10, slide=1, trigger=None)
    .process_window(
        lambda w: w.array_agg(
            columns=("timestamp", "speaker", "message"),
            output_col="conversation_records",
        )
    )
)
```

Contract:

- input receiver: `Relation`
- intermediate receiver: `WindowedRelation`
- `size`: positive integer window length
- `slide`: positive integer window start step; default `1`, not the only
  supported value
- closing method: `process_window(lambda w: ...)`
- output receiver after closing: ordinary `Relation`
- ordering: runtime append sequence
- supported trigger: `trigger=None`

`slide < size` creates overlapping windows, `slide == size` creates
non-overlapping windows, and `slide > size` creates gapped windows.

Inside `process_window(...)`, `w` is a window-local ordinary `Relation`.
After `process_window(...)`, downstream operators are global over the process
output relation.

Invalid shape:

```python
log.count_window(size=10).sem_map(...)
```

The window must first be closed:

```python
log.count_window(size=10).process_window(lambda w: w.array_agg(...)).sem_map(...)
```

### `over`

Row-preserving over-window handle. It returns `OverRelation`, not ordinary
`Relation`.

```python
contextual_log = (
    log
    .over(rows=(-10, -1))
    .array_agg(
        columns=("timestamp", "speaker", "message"),
        output_col="previous_messages",
    )
)
```

```python
contextual_log = (
    log
    .over(rows=(-10, -1))
    .sem_agg(
        input_cols=["message"],
        output_cols={"previous_summary": "Summary of previous messages."},
        instruction="Summarize the previous messages in this frame.",
    )
)
```

Contract:

- input receiver: `Relation`
- intermediate receiver: `OverRelation`
- closing methods: `array_agg(...)` or `sem_agg(...)`
- cardinality: row-preserving
- output receiver after closing: ordinary `Relation`
- output columns: original row columns plus frame aggregate columns
- supported frames: append-sequence `rows=(M, N)` with `N <= 0`

`OverRelation.array_agg(...)` differs from `Relation.array_agg(...)`.
The ordinary relation version collapses many rows into one row. The over-window
version keeps each emit row and adds one frame aggregate column.

## 4. Instruction And Column Conventions

Semantic operators use an `instruction` string. The string may refer to columns
with placeholders.

Single relation:

```python
df.sem_filter(instruction="{message} contains durable user preference.")
```

Multi-relation join:

```python
candidates.sem_join(
    topics,
    instruction="{candidates: topic_name} and {topics: topic_name} refer to the same durable memory topic.",
    how="left",
)
```

Column arguments follow two accepted forms:

```python
output_cols=["topic_name", "topic_content"]

output_cols={
    "topic_name": "Candidate durable memory topic name.",
    "topic_content": "Candidate durable memory content.",
}
```

The dictionary form is preferred when column descriptions improve the prompt.

During differentiation, instruction strings are parsed for simple column
placeholders. A single-brace token such as `{topic}` must refer to a declared
input or output column in the current operator scope. If the text should contain
literal braces rather than a column reference, escape it with double braces such
as `{{topic}}`. This strict rule is intentional: it catches misspelled column
placeholders before a backend executes an invalid prompt.

### Instruction Wording Convention

All semantic operators use the parameter name `instruction`, but the expected
content follows the operator semantics:

- `sem_filter`: row-level predicate / claim, such as `"{message} contains a durable user preference."`
- `sem_join`: pairwise predicate / match condition, such as `"{left} and {right} refer to the same topic."`
- `sem_groupby`: grouping / assignment condition, such as `"{topic_name} values refer to the same durable memory topic."`
- `sem_map`: transformation instruction, such as `"Produce catalog fields for this topic."`
- `sem_flat_map`: extraction / generation instruction, such as `"Extract zero or more durable memory topic candidates from {message}."`
- `sem_agg`: aggregation / merge instruction, such as `"Merge topic content into one durable memory."`
- `sem_topk`: ranking query or relevance criterion; adapters may lower plain user query text into backend-specific column-aware expressions.

## 5. Semantic Operators

### `sem_filter`

Semantic row filter.

```python
df.sem_filter(instruction=instruction)
```

Example:

```python
durable = log.sem_filter(
    instruction="{message} contains information worth saving in long-term memory."
)
```

`sem_filter` keeps rows whose content satisfies the semantic condition.
For LOTUS-style lowering, write this as a row-level predicate over columns, such
as `"{message} is coherent"` or `"{message} contains a durable preference"`.
Avoid command wording like `"Find coherent messages"` for `sem_filter`.

### `sem_map`

Semantic add-column operator.

```python
df.sem_map(
    input_cols=None,
    output_cols=[...] | {"col": "description"},
    instruction="...",
)
```

Defaults:

- `input_cols=None` means the whole visible row is available.
- `output_cols` is required.

`sem_map` preserves existing columns and adds or replaces the requested output
columns. Use `select` afterward when the desired result is a new logical table
with only the output columns.
Unlike `sem_filter`, `sem_map` is a transformation. Its instruction should say
what fields to produce, such as `"Produce catalog fields for this topic"`, not
just name the target concept.

LOTUS lowering uses native `df.sem_map(...)` when there is one output column.
For multiple output columns, agent-memory uses structured sem_map lowering:
the original instruction is preserved, the requested `output_cols` become an
explicit JSON output contract, and the backend validates that all requested
keys are present. Decoded fields retain JSON scalar types; nested arrays or
objects are rejected rather than stringified. Backend execution knobs such as
examples, system prompts, reasoning strategies, raw outputs, and explanations
are adapter/runtime configuration, not policy API fields. If an explanation is
part of the logical memory view, declare it explicitly in `output_cols`.

Example:

```python
catalog_enriched = topics.sem_map(
    input_cols=["topic_name", "topic_content"],
    output_cols={
        "catalog_title": "Title shown in the memory catalog.",
        "path": "Relative path to the topic memory file.",
        "hook": "One-line relevance hook.",
    },
    instruction="Produce catalog fields for this topic.",
)

catalog = catalog_enriched.select(["catalog_title", "path", "hook"])
```

### `sem_flat_map`

Semantic add-column operator where one input row may produce zero, one, or many
output rows.

```python
df.sem_flat_map(
    input_cols=None,
    output_cols=[...] | {"col": "description"},
    instruction="...",
    ordinal_col=None,
)
```

`sem_flat_map` preserves source columns for each emitted row and adds the output
columns. Use `select` afterward when only the extracted columns should remain.
Like `sem_map`, its instruction should describe the extraction/transformation;
for example, `"Extract zero or more durable memory topic candidates from {message}"`.
LOTUS lowering expects a JSON object containing a `rows` array for each input
row. Each emitted object must include all declared `output_cols`; an empty array
emits zero rows.

When `ordinal_col` is set, execution adds `0, 1, 2, ...` to accepted emitted
rows, restarting at zero for every source row. The ordinal is deterministic
runtime metadata analogous to SQL `WITH ORDINALITY` or Spark `posexplode`; it
is not requested from the LLM and must not conflict with source/output columns.
Structured fields preserve JSON scalar values (`string`, `number`, `boolean`,
or `null`). Nested array/object field values are invalid and use the existing
bounded parse-retry path.

Example:

```python
topic_candidates = (
    log
    .sem_flat_map(
        input_cols=["message"],
        output_cols={
            "topic_name": "Candidate durable memory topic name.",
            "topic_content": "Candidate durable memory content.",
        },
        instruction="Extract zero or more durable memory topic candidates from {message}.",
        ordinal_col="topic_ordinal",
    )
    .select(["topic_ordinal", "topic_name", "topic_content"])
)
```

### `sem_groupby`

Semantic grouping. In v0, grouped aggregation is expressed by chaining
`.sem_agg(...)` on the returned grouped relation.

```python
df.sem_groupby(
    input_cols=[...],
    instruction="...",
).sem_agg(...)
```

`input_cols` names the columns used as evidence for semantic grouping. The
instruction defines group membership, for example what makes candidate topic
rows belong to the same durable memory topic. `sem_groupby(...)` partitions or
assigns rows; the following `.sem_agg(...)` produces the output row for each
group.

Example:

```python
topics = (
    topic_candidates
    .sem_groupby(
        input_cols=["topic_name"],
        instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    )
    .sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols={
            "topic_name": "Canonical durable memory topic name.",
            "topic_content": "Merged durable memory content.",
        },
        instruction="Choose a canonical topic name and merge topic content.",
    )
)
```

`sem_groupby` does not by itself decide all final output columns. If a visible
grouping field should be canonicalized, include that field in the aggregate
input and output.

By default, `sem_groupby` is open-world grouping: groups are discovered from
the rows using the membership condition.

For closed-world grouping, policy authors may provide explicit `labels`.
The model must assign each row to exactly one declared label; if an `other`
bucket is desired, it must be declared explicitly. The assigned label is written
to `label_col`, which defaults to `"_label"`, and the internal group id remains
available for the following `.sem_agg(...)`.

```python
papers = rows.sem_groupby(
    input_cols=["title", "abstract"],
    instruction="Assign each paper to the best matching research area.",
    labels={
        "systems": "Systems, infrastructure, distributed systems, and databases.",
        "ml": "Machine learning models, training, evaluation, and datasets.",
        "other": "Papers that do not fit the other declared labels.",
    },
)
```

Multi-label and hierarchical labels are not part of the current contract.

`partition_by` optionally adds deterministic partition keys to semantic
grouping. The semantic grouping rule is applied only within each partition, and
partition keys are preserved in the aggregate output.

```python
entities = extracted_entities.sem_groupby(
    input_cols=["name", "entity_type"],
    partition_by="group_id",
    instruction="Rows refer to the same real-world entity.",
)
```

### `group_by(...).agg(...)` and `sem_groupby(...).agg(...)`

Grouped `agg(...)` accepts explicit aggregate specs:

```python
entities = extracted_entities.sem_groupby(
    input_cols=["name", "entity_type"],
    instruction="Rows refer to the same real-world entity.",
).agg(
    am.sem_agg(
        input_cols=["name", "entity_type", "episode_content"],
        output_cols={
            "name": "Canonical entity name.",
            "entity_type": "Canonical entity type.",
            "summary": "Concise entity summary.",
        },
        instruction="Create one canonical entity row.",
    ),
    am.array_agg(
        columns=["episode_id", "entity_ordinal", "name", "entity_type"],
        output_col="mentions",
    ),
    am.min(
        columns=["add_seq", "entity_ordinal"],
        output_col="entity_id",
    ),
)
```

Supported aggregate specs:

- `am.sem_agg(...)`: semantic aggregate function.
- `am.array_agg(columns=..., output_col=...)`: collect grouped rows into one
  JSON array-of-records column.
- `am.collect_list(column=..., output_col=...)`: collect one grouped column into
  a JSON array of values.
- `am.min(column=..., output_col=...)`: deterministic minimum over one grouped
  column.
- `am.min(columns=[...], output_col=...)`: deterministic lexicographic minimum
  over complete grouped tuples.

For deterministic `group_by(K).agg(...)`, output columns are `K + outputs(A*)`.
If aggregate output names overlap deterministic keys, the deterministic key
column wins.

For semantic `sem_groupby(..., partition_by=P).agg(...)`, output columns are
`P + outputs(A*)`. At least one `sem_agg(...)` spec is required, and the
semantic key columns named in `input_cols` must be produced by one or more
`sem_agg.output_cols`. `array_agg` cannot by itself produce canonical semantic
keys.

Direct `sem_groupby(...).array_agg(...)` is intentionally unsupported. Write
mixed `.agg(sem_agg(...), array_agg(...))` instead.

### `sem_agg`

Semantic aggregation.

```python
sem_agg(
    input_cols=None,
    output_cols=None,
    instruction="...",
)
```

Defaults:

- In standalone aggregation, `input_cols=None` means all visible columns.
- In `GroupedRelation.sem_agg(...)`, `input_cols=None` means all visible columns
  except the internal semantic group-id column. Deterministic keys and semantic
  key columns remain visible aggregate state.
- `output_cols=None` means output columns use the same names as `input_cols`.

`input_cols` and `output_cols` are read/write sets, not positional rename lists.
They do not need to be one-to-one.

Operator execution correctness is schema-level: `sem_agg(...)` requires every
declared `input_cols` column to exist in its input DataFrame, ignores extra
columns unless referenced by the lowering, and produces the declared
`output_cols`. Whether an incremental aggregate rule matches full recompute is
a planner / policy accuracy question, not an operator execution precondition.

LOTUS lowering uses lower-level `sem_agg(...)` for aggregation. Single-output
aggregation directly returns the LOTUS aggregate string. Multi-output
aggregation uses an agent-memory compatibility helper that follows the LOTUS
main-branch hierarchical aggregate shape and applies JSON object
`response_format` only on the final LM pass. This is not native support in the
current PyPI LOTUS backend, and it does not monkeypatch LOTUS or pandas
accessors. Execution knobs for this final pass live in adapter config, not in
`sem_agg(...)`; `sem_agg_model_kwargs` cannot override `response_format`, so
the logical query contract remains structured JSON output.

Examples:

```python
sem_agg(
    input_cols=["topic_content"],
    output_cols=["topic_content"],
    instruction="Merge topic content into one durable memory.",
)
```

```python
session_summary = logs.sem_agg(
    input_cols=["message", "timestamp"],
    output_cols={
        "summary": "Concise session summary.",
        "date_range": "Absolute date range covered by the messages.",
    },
    instruction="Summarize the messages and infer the covered date range.",
)
```

The second example reads `message` and `timestamp` together and writes `summary`
and `date_range`. It is not a rename from `message` to `summary`.

### `sem_join`

Semantic join.

```python
left.sem_join(
    right,
    instruction=instruction,
    how="inner",
)
```

Supported join types:

```python
how="inner"
how="left"
how="right"
how="outer"
```

Sugar aliases may also be supported:

```python
left.sem_inner_join(right, instruction=instruction)
left.sem_left_join(right, instruction=instruction)
left.sem_right_join(right, instruction=instruction)
left.sem_outer_join(right, instruction=instruction)
```

The canonical documentation form is `sem_join(..., instruction=..., how=...)`,
because it follows DataFrame style.

Backend execution choices such as cascade, helper models, examples, and
explanation tracing belong to the runtime/adapter layer. They should not appear
in the policy author's logical join definition.

Example:

```python
joined = topic_candidates.sem_join(
    topics,
    instruction="""
    {topic_candidates: topic_name} and {topics: topic_name} refer to the
    same durable memory topic, including corrections,
    contradictions, supersession, or forget/delete targets.
    """,
    how="left",
)
```

`sem_sim_join` is a physical or lower-level candidate retrieval plan, not the
default authoring API. A runtime may lower a semantic join into approximate
nearest-neighbor search, embedding search, BM25, reranking, or hybrid retrieval
before applying the semantic join predicate.

### `sem_topk`

Semantic top-k retrieval.

```python
df.sem_topk(instruction, k)
```

Example:

```python
memories = topics.sem_topk(
    am.UserQuery(),
    5,
)
```

Claude-style retrieval uses `sem_topk` only for relevance selection over a
lightweight topic manifest, then uses deterministic `join(on="name")` to fetch
the full body:

```python
retrieval_query = (
    topics.select(["name", "description", "type"])
    .sem_topk(am.UserQuery(), 5)
    .join(topics, on="name")
    .select(["name", "description:right", "type:right", "body"])
)
```

Backend methods such as `naive`, `quick`, `heap`, `quick-sem`, cascade,
hybrid retrieval, reranking, graph traversal, and BFS are runtime/adapter or
optimizer concerns. When used as a memory retrieval template, the `instruction`
argument is typically `am.UserQuery()`, which runtime binds to the end-user
query text. `k` is the policy author's initial retrieval width.

## 6. Retrieval Queries

### `search` and `RetrievalQuery`

`Relation.search(...)` declares an index-backed, storage-only search and returns
a `SearchRelation`:

```python
_retrieved_entities = entities.search(
    am.UserQuery(),
    methods=[am.BM25(), am.CosineSimilarity()],
    reranker=am.RRF(),
    limit=20,
).select(["record_id", "name", "summary", "rank", "score"])

retrieval_query = am.RetrievalQuery(
    entities=_retrieved_entities,
    facts=facts.search(
        am.UserQuery(),
        methods=[
            am.BM25(),
            am.CosineSimilarity(),
            am.BFS(origins=_retrieved_entities, max_depth=3),
        ],
        reranker=am.CrossEncoder(model="BAAI/bge-reranker-v2-m3"),
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
```

The search result logically preserves readable source columns and adds:

- `record_id`: opaque stable identity supplied by the physical backend;
- `rank`: final one-based result rank;
- `score`: final reranker score.

`RetrievalQuery` is the single retrieval root and contains ordered named
channels. A relation handle such as `_retrieved_entities` is a shared node in
the retrieval DAG, not a memory view or storage sink. `BFS.origins` therefore
accepts a real `SearchRelation`; it does not accept a channel-name string.

The retrieval planner continues to use the ordinary immutable `QueryExpr` IR;
there is no parallel `SearchExpr` tree. It pushes required columns into the
physical search request. The first implementation accepts post-search
`select`, `filter`, `assign`, `alias`, and `drop_duplicates` only when their
input-column dependencies can be derived exactly. Other post-search operators
are rejected at compile time rather than silently fetching or evaluating an
incorrect shape.

Search is retrieval-only:

- it cannot appear in a maintained view or a `StatementSet` sink;
- it requires exactly one matching materialized storage sink;
- it has a separate retrieval fingerprint and is not checkpointed as memory
  state;
- it has no in-memory full-scan or `sem_topk` fallback;
- execution returns `RetrievalResult`, whose channels are DataFrames and whose
  per-channel metrics retain method candidates, BFS origins, reranker scores,
  and physical latency when supplied by the connector.

`sem_topk` remains an ordinary semantic relation operator. It is not used to
lower `search(...)`, and the Graphiti-compatible Zep retrieval path performs no
generative LLM call.

## 7. Differential Maintenance Notes

The API is designed so full view definitions and differential maintenance can be
discussed in the same dataframe language.

Simple operators:

```python
V = D.sem_filter(instruction=instruction)
V_prime = V.union(delta_D.sem_filter(instruction=instruction))

V = D.sem_map(output_cols=[...], instruction="...")
V_prime = V.union(delta_D.sem_map(output_cols=[...], instruction="..."))

V = D.sem_flat_map(output_cols=[...], instruction="...", ordinal_col="ordinal")
V_prime = V.union(delta_D.sem_flat_map(output_cols=[...], instruction="...", ordinal_col="ordinal"))

V = D.explode(column="records", output_col="_record").unnest(
    column="_record",
    fields={"id": "record_id"},
)
V_prime = V.union(
    delta_D.explode(column="records", output_col="_record").unnest(
        column="_record",
        fields={"id": "record_id"},
    )
)
```

Because `sem_map` and `sem_flat_map` preserve existing columns and add output
columns, any projection in the full view definition must also be applied on the
differential branch before `union`:

```python
V = D.sem_map(output_cols=[...], instruction="...").select(V_columns)
V_prime = V.union(
    delta_D
    .sem_map(output_cols=[...], instruction="...")
    .select(V_columns)
)
```

Semantic group-by with aggregation can use a coarse full-next-view maintenance
rule:

```python
V = (
    D
    .sem_groupby(input_cols=[...], instruction=group_instruction)
    .sem_agg(...)
)

delta_groups = (
    delta_D
    .sem_groupby(input_cols=[...], instruction=group_instruction)
    .sem_agg(...)
)

V_prime = (
    delta_groups
    .sem_join(V, instruction=join_instruction_prime, how="outer")
    .sem_map(
        output_cols=V.columns,
        instruction="""
        Produce one next-view row:
        merge matched delta group and existing view row;
        keep unmatched existing view row;
        add unmatched delta group row.
        """,
    )
    .select(V.columns)
)
```

This is a coarse full-view rule. It does not split `delta_minus` and
`delta_plus`, and it does not introduce a separate delta-application operator.
The final `sem_map` is a column-level semantic merge over joined rows, not a
`sem_agg`.

Join follows the usual relational delta shape:

```python
V = L.sem_join(R, instruction=instruction, how="inner")

V_prime = (
    delta_L.sem_join(R, instruction=instruction, how="inner")
    .union(L.sem_join(delta_R, instruction=instruction, how="inner"))
    .union(delta_L.sem_join(delta_R, instruction=instruction, how="inner"))
)
```

For updates, the exact output can be expressed with ordinary set operations:

```python
V_prime = V.subtract(delta_minus).union(delta_plus)
```

Use `concat` instead of `union` only when duplicates are impossible or allowed:

```python
V_prime = V.subtract(delta_minus).concat(delta_plus)
```

`sem_union` may be useful as backend theory terminology for semantic upsert or
semantic union-distinct, but it is not a public v0 operator in this API.

## 8. LOTUS Alignment

This API follows the LOTUS direction closely:

- LOTUS exposes semantic DataFrame operators such as `sem_filter`, `sem_map`,
  `sem_extract`, `sem_agg`, `sem_topk`, `sem_join`, `sem_sim_join`,
  `sem_cluster_by`, and `sem_partition_by`.
- LOTUS supports LazyFrame-style optimized execution, where ordinary dataframe
  operations and semantic operators are part of one lazy plan.
- LOTUS uses column placeholders such as `{column}` and join disambiguation such
  as `{column:left}` / `{column:right}`.

`agent-memory` differs in the pieces needed for memory view maintenance:

- `sem_flat_map` is a first-class operator because one message can produce many
  memory candidates.
- `sem_groupby` is dynamic semantic grouping for durable memory rows, not exact
  column grouping or fixed-size clustering.
- `union` and `subtract` are explicit ordinary relational operators because
  incremental view maintenance needs exact set effects.
- The system-level problem is not only semantic processing, but deriving and
  executing `Q'` from user-authored memory view query `Q`.

References:

- [LOTUS core concepts](https://lotus-ai.readthedocs.io/en/latest/core_concepts.html)
- [LOTUS sem_map](https://lotus-ai.readthedocs.io/en/latest/sem_map.html)
- [LOTUS sem_extract](https://lotus-ai.readthedocs.io/en/latest/sem_extract.html)
- [LOTUS sem_filter](https://lotus-ai.readthedocs.io/en/latest/sem_filter.html)
- [LOTUS sem_agg](https://lotus-ai.readthedocs.io/en/latest/sem_agg.html)
- [LOTUS sem_join](https://lotus-ai.readthedocs.io/en/latest/sem_join.html)
- [LOTUS sem_sim_join](https://lotus-ai.readthedocs.io/en/latest/sem_sim_join.html)
- [LOTUS sem_topk](https://lotus-ai.readthedocs.io/en/latest/sem_topk.html)
- [LOTUS sem_cluster_by](https://lotus-ai.readthedocs.io/en/latest/sem_cluster.html)
- [LOTUS sem_partition_by](https://lotus-ai.readthedocs.io/en/stable/sem_partition.html)
