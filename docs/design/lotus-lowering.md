# LOTUS Lowering 设计

本文记录 `agent-memory` 如何把逻辑 `QueryExpr` lower 到 LOTUS-backed
operator execution。它是一份稳定的执行层设计说明，不是实施路线图。

其中 LOTUS 部分讨论的是 `QueryExpr -> LOTUS-backed execution`，不是
`Q -> Q'` 的 differential rewrite。

核心原则：

- `agent-memory` 定义 operator 语义。
- LOTUS 是 execution backend / toolkit，不是 `agent-memory` operator 语义的
  唯一来源。
- 能精确对应 LOTUS 原生 operator 的，就直接 lower 到 LOTUS。
- 语义不精确对应的，不硬套、不 monkeypatch、不偷偷改 prompt 后假装等价。
- Policy author 只写 logical operator API；LOTUS、backend method、cascade、
  examples、safe mode、trace / stats 等执行配置属于 runtime / adapter 层。
- Whole-policy maintenance 已由 `PolicyDifferentiator` 生成的共享
  `DifferentiatedPolicy` 执行。Rules 决定 `Q'` 的语义，runtime 只负责
  topology、change/state、共享 node 和原子提交；详细边界见
  `policy-differentiation-dataflow-runtime.zh.md`。

## 1. Operator Implementation Strategy

Operator 实现要分清两类：

- **LOTUS native exact match**：语义和 LOTUS 原生 operator 精确对应时，直接
  lower 到 pandas accessor。
- **Agent-memory custom lowering**：语义不精确对应时，由 agent-memory 自己
  定义 lowering contract，底层可以复用 LOTUS LM、cache、templates 和
  postprocessors。

当前 execution layer 的实现状态：

- `select`、`log`、`materialized_view`：本地 deterministic execution。
- `concat`、`union`、`subtract`、`drop_duplicates`：本地 deterministic
  dataframe execution。
- `sem_filter`：LOTUS native `df.sem_filter(...)`。
- `sem_topk`：LOTUS native `df.sem_topk(...)`，adapter 可把 plain query
  lower 成 column-aware relevance expression。
- `sem_map`：单输出使用 LOTUS native `df.sem_map(..., suffix=temp_col)`；多
  输出使用 agent-memory structured map lowering，复用 LOTUS LM、batch、
  cache、formatter 和 JSON-style postprocessing pattern。
- `sem_flat_map`：agent-memory custom lowering。每个输入 row 生成 zero or
  more structured output rows，保留 source columns，校验 required output
  columns，malformed output 直接失败。
- `sem_join`：`inner` 使用 LOTUS native semantic join；`left/right/outer`
  在 inner matches 基础上本地补 unmatched rows。
- `sem_groupby`：baseline 使用 pairwise semantic same-topic matching，加
  deterministic union-find group assignment。第一版优先 correctness /
  inspectability，不优先成本。
- `sem_agg`：single-output 复用 LOTUS lower-level `sem_agg`。multi-output
  使用 agent-memory compatibility helper，复刻 LOTUS main-style hierarchical
  aggregate，并只在 final LM pass 应用 JSON object `response_format` 写入
  declared `output_cols`。grouped input 每个 group 产出一行，whole-relation
  aggregation 视为一个 synthetic group。

后续 operator work 不是继续扩 public policy API，而是补 execution parity：
native LOTUS options 进入 adapter/runtime config，custom lowering 补 pruning、
tree fold、audit trace 和大数据量稳定性。

`filter(...)` / `assign(...)` 已收窄成最小 serializable relation-bound
expression subset，通过 `relation.col(...)` 构造 deterministic column
expressions；不支持 arbitrary Python callables、SQL strings 或 tuple predicates。

`LotusAdapter` 应保持 thin dispatch。具体 lowering 拆进
`adapters/lotus/` package，每个 semantic op 一个模块。Schema shaping 只有在它属于
agent-memory operator contract 时才能做，不能为了能跑而偷偷伪装成 LOTUS
原生 operator。

Public `Relation` API 只保存 logical query 参数到 `QueryExpr.params`。例如
`sem_map` 保存 input/output columns 和 instruction，`sem_join` 保存 instruction
和 `how`，`sem_topk` 保存 instruction 和 `k`。LOTUS method、cascade、examples、
safe mode、stats、trace 等不进入 policy class，也不进入 query tree。

## 2. 分层边界

推荐保持如下分层：

```text
Policy API
  log.sem_filter(...).sem_map(...)

QueryExpr
  op="sem_filter", op="sem_map", ...

Differential planner
  Q -> Q'

Execution adapter
  QueryExpr -> backend execution
```

本文只讨论最后一层：`Execution adapter` 如何把 `QueryExpr` lower 到
LOTUS-backed execution。

`QueryDifferentiator` 负责把 view definition query `Q` 变成
differentiated maintenance query `Q'`。它不应该知道 LOTUS 的 pandas accessor
细节。`LotusAdapter` 负责执行已经生成好的 `QueryExpr`，它不应该修改
operator 的逻辑语义。

## 3. 不直接改 LOTUS

不要直接改 LOTUS 源码，也不要覆盖或 monkeypatch `df.sem_map`、
`df.sem_filter` 等 pandas accessor。

原因：

- LOTUS 的 operator 是全局 pandas accessor，覆盖后会污染整个 Python
  进程里的 pandas 行为。
- `agent-memory` 的 operator 语义和 LOTUS 原生 operator 不总是一一对应。
- 后续如果升级 LOTUS，monkeypatch 会变成脆弱的隐式依赖。

正确做法是：在 `agent-memory` 里实现自己的 lowering，底层复用 LOTUS 的
LM、cache、templates、pandas accessors、top-k/filter/join 等能力。

## 4. 推荐模块结构

`LotusAdapter` 不应该膨胀成一个包含所有 operator 细节的大文件。LOTUS
lowering 使用专门的 adapter package：

```text
src/agent_memory/adapters/lotus/
  __init__.py
  adapter.py
  context.py
  sources.py
  relational.py
  sem_filter.py
  sem_map.py
  sem_flat_map.py
  sem_join.py
  sem_groupby.py
  sem_agg.py
  sem_topk.py
```

职责划分：

- `adapter.py`：`LotusAdapter` class 和 `QueryExpr.op` dispatch。
- `context.py`：LOTUS configure helper 和共享 execution context。
- `sources.py`：`log` / `materialized_view` 这种 query leaf 的 runtime input
  binding。
- `relational.py`：`select` 等确定性 relational lowering。
- 每个 `sem_*.py`：只负责一个 operator 的 lowering。

这个 package 不是新的 public adapter surface；public 入口仍然是
`LotusAdapter`。

## 5. Lowering Matrix

| agent_memory op | LOTUS 支持情况 | 状态 | lowering |
|---|---|---|
| `log` / `materialized_view` | runtime inputs | implemented | bind runtime-provided dataframe |
| `select` | pandas 原生 | implemented | 本地 pandas projection |
| `concat` / `union` / `subtract` / `drop_duplicates` | pandas 原生 | implemented | 本地 deterministic dataframe execution |
| `sem_filter` | LOTUS 原生 | implemented | `df.sem_filter(...)` |
| `sem_topk` | LOTUS 原生 | implemented | `df.sem_topk(...)` |
| `sem_map` 单输出 | LOTUS 原生 | implemented | `df.sem_map(..., suffix=temp_col)` |
| `sem_map` 多输出 | 无精确等价 | implemented | agent-memory structured map lowering |
| `sem_flat_map` | 无精确等价 | implemented | agent-memory list-output structured lowering |
| `sem_join` `inner` | LOTUS 原生 | implemented | `df.sem_join(..., how="inner")` |
| `sem_join` `left/right/outer` | LOTUS 签名有 `how`，但实现只支持 `inner` | implemented | inner matches + deterministic unmatched-row completion |
| `sem_groupby` | 无精确等价 | implemented | semantic group assignment / partition lowering |
| `sem_agg` 单输出 | LOTUS 部分支持 | implemented | LOTUS lower-level `sem_agg(...)` where semantics match |
| `sem_agg` 多输出 | 无精确等价 | implemented | agent-memory structured aggregate lowering |
| `filter(...)` / `assign(...)` / predicate `join(...)` | pandas 本地实现 | implemented | minimal relation-bound expression subset via `relation.col(...)` |

## 6. `sem_map`

LOTUS 原生 `sem_map` 是：

```text
one input row -> one string output column
```

`agent-memory sem_map` 的逻辑语义是：

```text
one input row -> same row plus requested output columns
```

因此当前干净实现是：

- `output_cols` 只有一个：lower 到 LOTUS `df.sem_map(...)`。
- `output_cols` 多于一个：使用 agent-memory structured map lowering。

不要把多输出 map 假装成 LOTUS 原生 `sem_map`。Structured map 是
agent-memory 的 operator contract：原始 instruction 保留，`output_cols`
成为显式 JSON output contract，底层复用 LOTUS 的 LM、batch、cache、
formatter 和 postprocessing pattern。
Raw outputs、reasoning、explanations 默认不写入 result DataFrame。它们如果需要
保留，应走 adapter trace / audit log / stats side-channel。如果 explanation 是业务
字段，policy 应该把它显式声明在 `output_cols` 里，而不是通过 backend execution
option 注入结果列。

这不是自动 lower 到 `sem_extract`。`sem_extract` 的语义是抽取字段；
multi-output `sem_map` 的语义仍然是按 instruction 生成 / 改写字段。

## 7. `sem_flat_map`

`sem_flat_map` 的语义是：

```text
one input row -> zero, one, or many output rows
```

这不是 LOTUS 原生 `sem_map`，不能硬套。

当前 lowering 由 `agent-memory` 自己实现：

```text
input DataFrame
  -> per-row LM output as JSON array of objects
  -> parse / validate output rows
  -> explode into DataFrame rows
  -> preserve source columns and add output columns
```

底层可以复用 LOTUS：

- `lotus.settings.lm`
- `lotus.models.LM`
- LOTUS cache / operator cache
- LOTUS prompt formatting helpers
- pandas DataFrame utilities

但 operator contract 属于 `agent-memory`，不是 LOTUS 原生 accessor。

## 8. `sem_join(how=...)`

`agent-memory sem_join` 支持 DataFrame-style `how`：

```python
left.sem_join(right, instruction=..., how="inner")
left.sem_join(right, instruction=..., how="left")
left.sem_join(right, instruction=..., how="right")
left.sem_join(right, instruction=..., how="outer")
```

本地 LOTUS 源码里，`df.sem_join(...)` 的签名包含 `how`，但当前实现对
`how != "inner"` 会 raise `NotImplementedError`。所以：

- `inner` 可以 lower 到 LOTUS native `df.sem_join(...)`。
- `left/right/outer` 不能宣称是 LOTUS native。

当前 lowering：

```text
inner:
  semantic matches from LOTUS df.sem_join(...)

left:
  inner matches
  + unmatched left rows with right-side columns set to null

right:
  inner matches
  + unmatched right rows with left-side columns set to null

outer:
  inner matches
  + unmatched left rows
  + unmatched right rows
```

注意：这只是 join shape 的补全。semantic predicate 本身仍由 LOTUS inner
join 执行。unmatched rows 的补全是 deterministic pandas work。

## 9. `sem_groupby -> sem_agg`

`agent-memory sem_groupby` 是 semantic partition / assignment，不是 pandas
exact groupby。

它回答的问题是：

```text
哪些 rows 应该属于同一个 semantic group？
```

LOTUS 的 `sem_agg(group_by=...)` 只能在已有 exact group key 或 group id 时
做分组聚合。它不能替代 semantic grouping 本身。

因此当前 lowering 分成两步：

```text
rows
  -> semantic group assignment / group_id
  -> grouped semantic aggregation
```

可能复用的 LOTUS 能力：

- `sem_join` / `sem_filter`：pairwise semantic same-group 判断。
- `sem_sim_join` / semantic index：候选 group pruning。
- `sem_cluster_by`：当 embedding clustering 假设足够接近目标语义时使用。
- `sem_partition_by`：当已有明确 partition function 时使用。
- `sem_agg(group_by=...)`：在 group id 已经生成后做聚合。

但这些都不是 `agent-memory sem_groupby` 的直接等价物。lowering 必须明确
说明选择了哪一种物理策略。

## 10. `sem_agg`

`sem_agg` 有两个 receiver scope：

- `Relation.sem_agg(...)`：whole-relation aggregation。
- `GroupedRelation.sem_agg(...)`：per-group aggregation。

receiver type 决定 aggregation scope。lowering 层不能把这两者混掉。

LOTUS lower-level `sem_agg(...)` 可以支持单输出 string aggregation，也支持基于
partition ids 的 tree fold。因此：

- 单输出 whole-relation aggregation：lower 到 LOTUS lower-level `sem_agg(...)`。
- 单输出 grouped aggregation：每个 generated group 独立调用 LOTUS lower-level
  `sem_agg(...)`，输出 `G groups -> G rows`。
- 多输出 aggregation：当前 active lowering 是 agent-memory structured
  hierarchical aggregate。它在 adapter 内部使用 compatibility helper 复刻
  LOTUS main-style tree fold，并只在 final LM pass 应用 JSON object
  `response_format` 生成 declared `output_cols`。这不是当前 PyPI LOTUS 的
  native multi-output capability，也不 monkeypatch LOTUS 或 pandas accessor。

固定 row-count chunking 曾作为实验策略出现过，但它不是 LOTUS 原生
`sem_agg` 的思路，真实 LOCOMO audit 也暴露过 structured JSON 空输出问题。
当前不保留为 runtime strategy，只在 TODO 中作为未来可能的 tree-fold experiment
记录。

不要在 adapter 里临时把 instruction 改成“请返回 JSON”，再拆成多列并假装是
原生 `sem_agg`。Structured aggregate 的 schema/prompt contract 是
`agent-memory` operator lowering 的显式部分。

## 11. 工程原则

Adapter lowering 可以做：

- 选择 LOTUS 原生 operator。
- 调用 LOTUS LM、cache、templates、postprocessors。
- 做 deterministic pandas shape adjustment，例如 projection、rename、
  unmatched join rows。
- 把 plain user query lower 成 LOTUS column-aware top-k expression。

Adapter lowering 不应该做：

- 偷偷改变 policy writer 的 semantic instruction。
- 把一个 agent-memory operator 硬套成语义不同的 LOTUS operator。
- 为了“能跑”静默丢列、丢行、吞掉 parse error。
- monkeypatch LOTUS 或 pandas 全局 accessor。

如果语义 gap 存在，应显式 `NotImplementedError`，并说明未来 lowering 方向。

一句话：**public operator 语义归 `agent-memory`；LOTUS 是可复用的 semantic
execution backend。能精确 lower 就 lower，不能精确 lower 就实现
agent-memory custom lowering，不能为了短期 demo 牺牲语义边界。**
