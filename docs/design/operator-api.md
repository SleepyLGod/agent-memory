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
  `select`, `filter`, `assign`, `concat`, `union`, or `subtract`.
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

### `select`

Projection over columns.

```python
df.select(["topic_name", "topic_content"])
df[["topic_name", "topic_content"]]
```

`select` and bracket selection are equivalent authoring forms. The term
`projection` may appear in theory docs, but `select` is the public dataframe
spelling.

### `filter`

Deterministic row filter.

```python
changed = rows.filter(lambda row: row["action"] != "keep")
```

This is not `sem_filter`. The predicate is ordinary code or a deterministic
expression over existing columns.

### `assign`

Deterministic column creation or replacement.

```python
rows = rows.assign(action=lambda row: classify_action(row))
```

Use `assign` for cheap, deterministic fields. Use `sem_map` when the new fields
require semantic interpretation.

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

### `subtract`

Relational set difference / `EXCEPT`.

```python
remaining = rows.subtract(rows_to_remove)
```

`subtract` removes rows using exact equality or an implementation-defined exact
key. It is not a semantic delete.

### `join`

Deterministic relational join on same-named key columns.

```python
joined = selected.join(topics, on="name", how="inner")
```

`join(on=...)` is pandas-backed exact lookup / merge. It does not call an LLM.
Overlapping non-key columns use the `:left` / `:right` suffix convention. This
operator is not interchangeable with `sem_join(...)`: `sem_join` asks an LLM
whether two rows semantically match, while `join(on=...)` requires exact key
identity.

## 3. Instruction And Column Conventions

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

## 4. Semantic Operators

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
keys are present. Backend execution knobs such as examples, system prompts,
reasoning strategies, raw outputs, and explanations are adapter/runtime
configuration, not policy API fields. If an explanation is part of the logical
memory view, declare it explicitly in `output_cols`.

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
)
```

`sem_flat_map` preserves source columns for each emitted row and adds the output
columns. Use `select` afterward when only the extracted columns should remain.
Like `sem_map`, its instruction should describe the extraction/transformation;
for example, `"Extract zero or more durable memory topic candidates from {message}"`.
LOTUS lowering expects one JSON array of objects per input row. Each object must
include all declared `output_cols`; an empty array emits zero rows.

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
    )
    .select(["topic_name", "topic_content"])
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

Future APIs may add ordinary deterministic grouped aggregates and mixed
aggregate maps. They are not v0 primary syntax:

```python
agg=count()
agg=sum("score")
agg=max("timestamp")
agg=list_collect("evidence")
agg=am.agg.sem_agg(input_cols=["content"], output_cols=["summary"], instruction="Summarize.")
```

A future mapping form could look like:

```python
topics = topic_candidates.sem_groupby(
    input_cols=["topic_name"],
    instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    agg={
        "topic_name": sem_agg(
            input_cols=["topic_name"],
            output_cols=["topic_name"],
            instruction="Choose one canonical topic name.",
        ),
        "topic_content": sem_agg(
            input_cols=["topic_content"],
            output_cols=["topic_content"],
            instruction="Merge topic content.",
        ),
        "last_seen": max("timestamp"),
    },
)
```

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
- In `GroupedRelation.sem_agg(...)`, `input_cols=None` means all non-grouping
  visible columns from the grouped relation.
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

## 5. Differential Maintenance Notes

The API is designed so full view definitions and differential maintenance can be
discussed in the same dataframe language.

Simple operators:

```python
V = D.sem_filter(instruction=instruction)
V_prime = V.union(delta_D.sem_filter(instruction=instruction))

V = D.sem_map(output_cols=[...], instruction="...")
V_prime = V.union(delta_D.sem_map(output_cols=[...], instruction="..."))

V = D.sem_flat_map(output_cols=[...], instruction="...")
V_prime = V.union(delta_D.sem_flat_map(output_cols=[...], instruction="..."))
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

## 6. LOTUS Alignment

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
