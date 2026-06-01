# Groupby Aggregation API Rationale

This note records the v0 decision for grouped semantic aggregation syntax.
It is a focused rationale document, not the final operator API reference.

The current v0 direction is:

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
    .select(["topic_name", "topic_content"])
)
```

## Decision

Use chain-based grouped aggregation as the v0 primary style:

```python
df.sem_groupby(...).sem_agg(...)
```

Do not make `agg=sem_agg(...)` or `agg=[sem_agg(...), min, count]` the v0
primary API.

The reason is conceptual and practical:

- `sem_groupby` is semantic partition / assignment.
- `sem_agg` is semantic aggregation / canonicalization.
- `Relation.sem_agg(...)` means whole-relation aggregation.
- `GroupedRelation.sem_agg(...)` means per-group aggregation.
- The receiver type determines aggregation scope, so the chain has no ambiguity.

This keeps `sem_agg` at the same level as other dataframe-style semantic
operators instead of turning it into a special imported helper.

## Deferred Alternatives

The following alternatives are possible, but they are not v0 primary syntax.

### Top-Level Imports

```python
from agent_memory import sem_agg, count, min

topics = topic_candidates.sem_groupby(
    input_cols=["topic_name"],
    instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    agg=sem_agg(...),
)
```

This is valid Python, but it pollutes the policy-author import surface. Policy
authors should not need to import a separate aggregate helper just to write a
dataframe-style semantic query.

### Namespace Helpers

```python
topics = topic_candidates.sem_groupby(
    input_cols=["topic_name"],
    instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    agg=[
        am.agg.sem_agg(...),
        am.agg.count(),
        am.agg.min("timestamp"),
    ],
)
```

This avoids extra imports, but introduces another namespace and makes
`sem_agg` look different from `df.sem_filter(...)`, `df.sem_map(...)`,
`df.sem_flat_map(...)`, and the rest of the dataframe-style semantic operators.

It may become useful later if v0 grows into mixed aggregate lists such as
`sem_agg`, `count`, `min`, `max`, and `list_collect`, but it should not be the
starting point.

### Inline `agg=sem_agg(...)`

```python
topics = topic_candidates.sem_groupby(
    input_cols=["topic_name"],
    instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    agg=sem_agg(...),
)
```

This is also possible, but it makes `sem_agg` a special helper rather than a
normal dataframe-style semantic operator. It also forces the name `sem_agg` to
come from somewhere: a top-level import, an `am.agg` namespace, or another
object. The chain form avoids that problem.

### Inline Dict Aggregate Spec

```python
topics = topic_candidates.sem_groupby(
    input_cols=["topic_name"],
    instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    agg=[
        "count",
        {
            "type": "sem_agg",
            "input_cols": ["topic_name", "topic_content"],
            "output_cols": {
                "topic_name": "Canonical durable memory topic name.",
                "topic_content": "Merged durable memory content.",
            },
            "instruction": "Choose a canonical topic name and merge topic content.",
        },
    ],
)
```

This form can express mixed aggregates without imports, but it turns the API into
a configuration DSL rather than a dataframe-style operator chain. The string
`"type": "sem_agg"` is weakly typed, heavy to write, and easy to drift away from
the operator model where `sem_agg` is a first-class semantic operator.

This shape may be useful as a low-level serialized plan format later, but it
should not be the policy-author API in v0.

### Receiver-Bound Aggregate Builder Passed Into `agg`

```python
topics = topic_candidates.sem_groupby(
    input_cols=["topic_name"],
    instruction="Rows whose {topic_name} values refer to the same durable memory topic belong in one group.",
    agg=topic_candidates.sem_agg(
        input_cols=["topic_name", "topic_content"],
        output_cols={
            "topic_name": "Canonical durable memory topic name.",
            "topic_content": "Merged durable memory content.",
        },
        instruction="Choose a canonical topic name and merge topic content.",
    ),
)
```

This avoids a top-level import and avoids an `am.agg` namespace, but it is
misleading. Calling `topic_candidates.sem_agg(...)` before `sem_groupby(...)`
looks like whole-relation aggregation has already been requested, while the group
scope does not exist yet. If `sem_agg(...)` is called after `sem_groupby(...)`,
then the API is simply the preferred chain form:

```python
topics = topic_candidates.sem_groupby(...).sem_agg(...)
```

For v0, the project should not support receiver-bound aggregate builders passed
back into `agg=...`.

## Future Option

If a future API needs to mix semantic and deterministic aggregates in the same
grouped query, the project can revisit a namespace or aggregate-list style:

```python
agg=[
    am.agg.sem_agg(...),
    am.agg.count(),
    am.agg.max("timestamp"),
    am.agg.list_collect("evidence"),
]
```

For v0, this is deliberately deferred. The primary goal is a clean,
dataframe-like memory policy surface:

```python
df.sem_groupby(...).sem_agg(...)
```
