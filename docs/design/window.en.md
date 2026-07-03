# Logical Window Design

This is the main window design document for `agent-memory`. It covers window
semantics, the current API, differential rules, limits, and future TODOs. The
full operator contracts for `array_agg` and `array_cat` live in
`docs/design/operator-api.md`.

A window here is part of the view query. It changes the input scope and context
seen by the query. It is not runtime lazy maintenance. Lazy, deferred, or
coalesced maintenance only changes when runtime applies `changed_rows`; it does
not change the logical meaning of the view query.

The current implementation has two window families:

- Count/group window: `count_window(...).process_window(...)`. It forms completed
  windows by row count, then runs a window function on each completed window. The
  output is window-level rows.
- Over window: `over(rows=...).array_agg(...)` and `over(rows=...).sem_agg(...)`.
  It looks up a frame for each emit row, computes one value on that frame, and
  attaches the result to the emit row. The output preserves source row identity.

## Why logical windows exist

These two policies are not the same query:

```text
Run sem_flat_map on each message, then run global sem_groupby / sem_agg.
```

```text
First build a context block from every 10 messages, then run global sem_groupby / sem_agg.
```

LLM operators are not ordinary deterministic row functions. Putting 10 messages
in one context can change candidate extraction, semantic grouping, and aggregate
output. That context boundary belongs in the logical query. It should not be
modeled only by batching multiple `changed_rows` at runtime.

## Terms

This design follows common Flink / SQL terminology, but only implements the
subset needed by the project now.

- Window assigner: decides which rows belong to a window or frame.
- Window function: computes over a window or frame.
- Trigger: decides when a count/group window fires.

The current `count_window(size, slide=1, trigger=None)` supports:

- `size`: positive integer, the number of rows in each completed window.
- `slide`: positive integer, default `1`, the number of rows by which the next
  window start moves.
- `trigger`: currently only `None`, the default completion trigger. A count
  window fires after it has received `size` rows. Non-`None` triggers are future
  TODO.

`slide` does not need to be less than or equal to `size`:

- `slide < size`: overlapping windows.
- `slide == size`: non-overlapping windows.
- `slide > size`: gapped windows; some rows between windows do not belong to any
  window.

The current count window uses `memory.add(...)` append sequence. `timestamp` is
an ordinary field. A window function can include it in output, but it does not
define window boundaries. Event-time windows, watermarks, late data, time
windows, and session windows are out of scope for now.

## Count window

`count_window(...)` returns a `WindowedRelation`, not an ordinary `Relation`.
`WindowedRelation` cannot be used directly as a `MemoryView`, and it cannot
directly receive ordinary relation operators. It must be closed with
`process_window(...)`.

```python
blocks = (
    log
    .count_window(size=10, slide=1)
    .process_window(
        lambda w: w.array_agg(
            columns=("timestamp", "speaker", "message"),
            output_col="conversation_records",
        )
    )
)
```

`process_window(lambda w: ...)` is a policy-definition-time builder. It is not a
runtime Python UDF. `w` is the rows inside one completed window, exposed to the
policy writer as an ordinary `Relation`. Calls such as `w.array_agg(...)`,
`w.sem_flat_map(...)`, and `w.sem_groupby(...).sem_agg(...)` build ordinary
operator expressions.

The output of `process_window(...)` becomes an ordinary `Relation`. Operators
after `process_window(...)` are global relation operators. The window scope does
not implicitly continue downstream.

```python
topics = (
    log
    .count_window(size=10, slide=1)
    .process_window(
        lambda w: w.array_agg(
            columns=("timestamp", "speaker", "message"),
            output_col="conversation_records",
        )
    )
    .sem_flat_map(
        input_cols=["conversation_records"],
        output_cols={...},
        instruction="Extract memory candidates from {conversation_records}.",
    )
    .sem_groupby(...)
    .sem_agg(...)
)
```

Boundary:

```text
inside process_window: window-local
after process_window: global relation
```

The source of a count window can be an ordinary upstream `Relation`, as long as
that upstream query can be maintained by the existing differential rules. For
example:

```python
blocks = (
    log
    .select(["timestamp", "speaker", "message"])
    .count_window(size=10, slide=1)
    .process_window(lambda w: w.array_agg(
        columns=("timestamp", "speaker", "message"),
        output_col="conversation_records",
    ))
)
```

The current implementation supports one `process_window` boundary per public
view. Nested or chained `process_window` boundaries are future TODO.

## Over window

An over window is row-preserving. It does not collapse N rows into one row. It
keeps each input row and adds one or more frame aggregate columns to that row.

Semantics:

```text
for each emit row:
  frame = rows related to this emit row
  value = aggregate(frame)
  output row = emit row + value
```

The SQL shape is similar to:

```sql
SELECT
  timestamp,
  speaker,
  message,
  array_agg(row(timestamp, speaker, message))
    OVER (
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS previous_messages
FROM log;
```

The current API uses receiver style:

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

It also supports frame-level semantic aggregation:

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

`Relation.over(...)` returns an `OverRelation`, not an ordinary `Relation`.
`OverRelation` must be closed by a one-row frame aggregate function. The current
closing methods are `.array_agg(...)` and `.sem_agg(...)`.

`OverRelation.array_agg(...)` and `Relation.array_agg(...)` have different
receivers and different semantics:

```text
Relation.array_agg(...)
  many rows -> one row

OverRelation.array_agg(...)
  source rows -> same number of rows + one frame aggregate value per emit row
```

The current over window supports append sequence, append-only data, and
`rows=(M, N)` with `N <= 0`. This means preceding/current-row frames.
`rows=(-10, -1)` means up to 10 previous rows, excluding the current row.
`rows=(-10, 0)` means up to 10 previous rows plus the current row.

Empty-frame behavior:

- `over(...).array_agg(...)` outputs `[]`.
- `over(...).sem_agg(...)` requires explicit `output_cols`; for an empty frame it
  outputs null values for those declared columns and does not call the LLM.

A following frame such as `rows=(-1, 1)` can change previous rows when a new row
arrives. That needs old-row replacement or `apply_delta`; it is out of scope for
the current implementation.

## IR contract

The `lambda` in `process_window(lambda w: ...)` runs only during policy
collection. It builds a `QueryExpr` subtree and is not saved into the compiled
artifact. The implementation can represent window-local rows with an internal
`window_source` leaf.

Count-window IR shape:

```text
QueryExpr(op="count_window", inputs=(D,), params={"size": ..., "slide": ..., "trigger": None})
QueryExpr(op="process_window", inputs=(count_window_expr, Q1))
```

Over windows do not create a dedicated operator for each frame aggregate. The IR
is an ordinary aggregate operator over an `over` input:

```text
QueryExpr(op="over", inputs=(D,), params={"rows": (M, N)})
QueryExpr(op="array_agg", inputs=(over_expr,), params={...})
QueryExpr(op="sem_agg", inputs=(over_expr,), params={...})
```

## Differential rules

### Count window

Common count-window view shape:

```text
P = D.count_window(size, slide, trigger=None).process_window(Q1)
V = P.Q2
```

Where:

- `D` is the source relation. It can be `log` or a maintainable upstream relation.
- `Q1` is the window-local query.
- `P` is the process-window output relation.
- `Q2` is an ordinary downstream query.

Runtime maintains private window state. After each `add(...)`:

```text
D'      = maintain upstream D
DeltaP = newly completed windows in D' processed by Q1
V'      = differentiate(V = P.Q2, changed_input = DeltaP)
```

`DeltaP` is the process output for newly completed windows. It is not source-level
`DeltaD`. Window completion and cursor advancement belong to count-window
operator/runtime state and are not public operators.

Full execution builds all completed windows from full `D`, runs `Q1` on each
completed window, builds full `P`, and then evaluates `P.Q2`.

### Over window

Over-window form:

```text
T = D.over(rows=(M, N)).OP
```

`OP` is currently `array_agg` or `sem_agg`, both one-row frame aggregate
functions. `T` is row-preserving.

Append-only differential rule:

```text
DeltaT = DeltaD.over(rows=(M, N)).OP
T'     = T.union(DeltaT)
```

`DeltaD.over(...)` is compact notation. It does not mean that the frame is
looked up only inside delta rows. It means:

```text
emit rows: new rows from DeltaD
frame lookup: current full source state D'
```

So a new row's frame can include old rows. Runtime cannot pass only `DeltaD` to
the over function, or previous context will be missing.

## Current limitations

Current scope:

- Count window only supports
  `count_window(size: int, slide: int = 1, trigger=None)`; `size` and `slide`
  must both be positive integers, and `slide=1` is only the default.
- Count window uses append sequence. It does not support `order_by`, event time,
  watermarks, late data, time windows, or session windows.
- Each public view supports one `process_window` boundary.
- `WindowedRelation` has no `.array_agg(...)`, `.sem_agg(...)`, or `.reduce(...)`
  sugar.
- A count-window upstream relation must be maintainable by the existing
  differential rules.
- Over window only supports `rows=(M, N)` with `N <= 0`.
- Over window does not support `partition_by`, `order_by`, following frames,
  multiple over expressions, arbitrary `apply`, or an expression DSL.
- Delete, update, retraction, TTL, old-row replacement, and `apply_delta` are out
  of scope.

## future TODO

- Count-window sugar: lower `log.count_window(...).array_agg(...)` to
  `log.count_window(...).process_window(lambda w: w.array_agg(...))`.
- Trigger variants: count trigger, processing-time trigger, event-time trigger,
  and purging trigger.
- More window assigners: time window, session window, semantic window.
- Multiple `process_window` boundaries: maintain multiple private window process
  states for one public view.
- Over-window extensions: `partition_by`, `order_by`, following frames, multiple
  over expressions, deterministic reduce / aggregate.
- Delta application: following frames, delete/update/retraction, and similar
  cases need old-row replacement or `apply_delta`.
