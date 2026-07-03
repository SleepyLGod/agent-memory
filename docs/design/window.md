# Logical Window Design

本文是 `agent-memory` 的 window 主文档。它只记录 window 语义、当前 API、
differential rules、限制和 future TODO。`array_agg`、`array_cat` 的完整 operator
contract 见 `docs/design/operator-api.md`。

这里的 window 是 view query 语义：它改变 query 看到的输入范围和上下文单位。它不是
runtime lazy maintenance。Lazy / deferred / coalesced maintenance 只决定 runtime
什么时候执行 maintenance，不改变 view query 的逻辑含义。

当前实现有两类 window：

- Count/group window：`count_window(...).process_window(...)`。它按固定 row count
  形成 completed windows，再对每个 completed window 运行 window function。输出是
  window-level rows。
- Over window：`over(rows=...).array_agg(...)` 和 `over(rows=...).sem_agg(...)`。
  它对每个 emit row 查一段 frame，在 frame 上算一个值，再把结果加回当前 row。输出
  仍然保留原 row identity。

## Why logical windows exist

下面两个 policy 不是同一个 query：

```text
每条 message 单独 sem_flat_map，再 global sem_groupby / sem_agg
```

```text
每 10 条 message 先组成一个上下文块，再 global sem_groupby / sem_agg
```

LLM operator 不是普通逐行确定性函数。把 10 条 message 放在同一个上下文里，可能会
改变 candidate extraction、semantic grouping 和 aggregate output。因此这种
"先按 N 条 message 形成上下文" 必须写进 logical query，不能只靠 runtime 把多条
`changed_rows` 攒起来。

## Terms

本文沿用 Flink / SQL 里常见的几个概念，但只实现项目当前需要的子集。

- Window assigner：决定哪些 rows 属于同一个 window 或 frame。
- Window function：在 window 或 frame 上运行的计算。
- Trigger：决定 count/group window 什么时候 fire。

当前 `count_window(size, slide=1, trigger=None)` 支持：

- `size`：正整数，表示每个 completed window 包含多少 rows。
- `slide`：正整数，默认值是 `1`，表示下一个 window start 向后移动多少 rows。
- `trigger`：当前只支持 `None`，也就是 default completion trigger。一个 count
  window 收到 `size` 条 rows 后 fire。非 `None` trigger 是 future TODO。

`slide` 不要求小于等于 `size`：

- `slide < size`：overlapping windows。
- `slide == size`：non-overlapping windows。
- `slide > size`：gapped windows，中间有些 rows 不属于任何 window。

当前 count window 按 `memory.add(...)` 的 append sequence 切分。`timestamp` 只是普通
字段，可以被 window function 收进输出，但不参与窗口边界。event-time window、
watermark、late data、time window 和 session window 都不在当前实现范围内。

## Count window

`count_window(...)` 返回 `WindowedRelation`，不是普通 `Relation`。`WindowedRelation`
不能直接作为 `MemoryView`，也不能直接接普通 relation operators。它必须用
`process_window(...)` 闭合。

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

`process_window(lambda w: ...)` 是 policy-definition-time builder。它不是 runtime
Python UDF。`w` 是当前 completed window 内的 rows，对 policy writer 来说是普通
`Relation`。因此 `w.array_agg(...)`、`w.sem_flat_map(...)`、
`w.sem_groupby(...).sem_agg(...)` 都按普通 operator 语义构造 query。

`process_window(...)` 的输出重新变成普通 `Relation`。后续 operator 是 global
relation operators，不会被 window scope 隐式包住。

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

边界是：

```text
inside process_window: window-local
after process_window: global relation
```

Count window 的 source 可以是 ordinary upstream `Relation`，只要该 upstream query
能通过现有 differential rules 维护 changed rows。例如：

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

当前每个 public view 只支持一个 `process_window` boundary。多个 nested 或 chained
`process_window` boundaries 是 future TODO。

## Over window

Over window 是 row-preserving window。它不把 N 行压成一行，而是保留每条 input row，
并给当前 row 增加一个或多个 frame aggregate columns。

语义：

```text
for each emit row:
  frame = rows related to this emit row
  value = aggregate(frame)
  output row = emit row + value
```

SQL 里类似：

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

当前 API 使用 receiver style：

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

也支持 frame-level semantic aggregate：

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

`Relation.over(...)` 返回 `OverRelation`，不是普通 `Relation`。`OverRelation` 必须
用 one-row frame aggregate function 闭合，当前只支持 `.array_agg(...)` 和
`.sem_agg(...)`。

`OverRelation.array_agg(...)` 和 `Relation.array_agg(...)` 的 receiver 不同，语义也
不同：

```text
Relation.array_agg(...)
  many rows -> one row

OverRelation.array_agg(...)
  source rows -> same number of rows + one frame aggregate value per emit row
```

当前 over window 只支持 append sequence、append-only、`rows=(M, N)` 且 `N <= 0`。
也就是 preceding/current-row frame。`rows=(-10, -1)` 表示最多前 10 条，不包含当前
row。`rows=(-10, 0)` 表示最多前 10 条加当前 row。

Empty frame 的行为：

- `over(...).array_agg(...)` 输出 `[]`。
- `over(...).sem_agg(...)` 要求显式 `output_cols`，empty frame 输出 declared columns
  的 null values，不调用 LLM。

Following frame 例如 `rows=(-1, 1)` 会让新 row 改变旧 row 的 output，需要 old-row
replacement / `apply_delta`。这不在当前实现范围内。

## IR contract

`process_window(lambda w: ...)` 里的 `lambda` 只在 policy collection 阶段执行。它用于
构造 `QueryExpr` subtree，不进入 compiled artifact。实现可以用 internal
`window_source` leaf 表达 window-local rows。

Count-window IR 形态：

```text
QueryExpr(op="count_window", inputs=(D,), params={"size": ..., "slide": ..., "trigger": None})
QueryExpr(op="process_window", inputs=(count_window_expr, Q1))
```

Over window 不为每个 frame aggregate 造新的专用 operator。IR 形态是普通 aggregate
operator 接 `over` input：

```text
QueryExpr(op="over", inputs=(D,), params={"rows": (M, N)})
QueryExpr(op="array_agg", inputs=(over_expr,), params={...})
QueryExpr(op="sem_agg", inputs=(over_expr,), params={...})
```

## Differential rules

### Count window

Count-window view 的常见形态：

```text
P = D.count_window(size, slide, trigger=None).process_window(Q1)
V = P.Q2
```

其中：

- `D` 是 source relation，可以是 `log`，也可以是可维护的 upstream relation。
- `Q1` 是 window-local query。
- `P` 是 process-window output relation。
- `Q2` 是普通 downstream query。

Runtime 维护 private window state。每次 `add(...)` 后：

```text
D'      = maintain upstream D
DeltaP = newly completed windows in D' processed by Q1
V'      = differentiate(V = P.Q2, changed_input = DeltaP)
```

`DeltaP` 是新完成 windows 的 process output，不是 source-level `DeltaD`。Window
completion 和 cursor 推进属于 count-window operator/runtime state，不暴露为 public
operator。

Full execution 语义是从完整 `D` 构造所有 completed windows，对每个 completed window
执行 `Q1`，得到完整 `P`，再执行 `P.Q2`。

### Over window

Over-window form：

```text
T = D.over(rows=(M, N)).OP
```

`OP` 当前是 `array_agg` 或 `sem_agg` 这类 one-row frame aggregate function。`T` 是
row-preserving relation。

Append-only differential rule：

```text
DeltaT = DeltaD.over(rows=(M, N)).OP
T'     = T.union(DeltaT)
```

这里 `DeltaD.over(...)` 是 compact notation，不表示只在 delta rows 内取 frame。它的
含义是：

```text
emit rows: DeltaD 中的新增 rows
frame lookup: 当前完整 source state D'
```

因此新 row 的 frame 可以包含旧 rows。Runtime 不能只把 `DeltaD` 传给 over function，
否则 previous context 会丢失。

## Current limitations

当前实现范围：

- count window 只支持 `count_window(size: int, slide: int = 1, trigger=None)`；
  `size` 和 `slide` 都必须是正整数，`slide=1` 只是默认值。
- count window 按 append sequence 切分，不支持 `order_by`、event-time、watermark、
  late data、time window 或 session window。
- 每个 public view 只支持一个 `process_window` boundary。
- `WindowedRelation` 没有 `.array_agg(...)`、`.sem_agg(...)`、`.reduce(...)` sugar。
- count-window upstream relation 必须能通过现有 differential rules 维护 changed rows。
- over window 只支持 `rows=(M, N)` 且 `N <= 0`。
- over window 不支持 `partition_by`、`order_by`、following frames、多个 over expressions、
  arbitrary `apply` 或 expression DSL。
- delete、update、retraction、TTL、old-row replacement 和 `apply_delta` 都不在当前范围内。

## future TODO

- Count-window sugar：把 `log.count_window(...).array_agg(...)` lower 成
  `log.count_window(...).process_window(lambda w: w.array_agg(...))`。
- Trigger variants：支持 count trigger、processing-time trigger、event-time trigger 和
  purging trigger。
- More window assigners：time window、session window、semantic window。
- Multiple `process_window` boundaries：为一个 public view 维护多个 private window
  process states。
- Over-window extensions：`partition_by`、`order_by`、following frames、multiple over
  expressions、deterministic reduce / aggregate。
- Delta application：支持 following frame、delete/update/retraction 时需要
  old-row replacement 或 `apply_delta`。
