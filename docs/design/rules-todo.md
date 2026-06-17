# Differential Rules TODO

本文记录 `Q -> Q'` differential rules 的当前共识和未决点。它不是
implementation spec，也不表示这些 rules 已经实现。当前黄金线仍然是
`docs/optimization/incremental-semantic-view-maintenance.tex` 和
`docs/design/operator-api.md`。

## 1. 核心区分

必须区分两个层次：

- **Delta expression**：由 differential rule 推导出来的 changed-output query
  fragment，例如 `DeltaD.sem_filter(...)`。
- **Maintenance query `Q'`**：最终交给 runtime 执行的 logical `QueryExpr`，
  语义是返回下一版 materialized view `V'`。

`Q -> Q'` 是 query rewrite，不看真实 data。Planner 阶段只处理 logical
`QueryExpr`。Runtime 阶段才把真实 `changed_rows` 绑定到 differentiated query
的 source input，把当前 materialized view 绑定成 `V`，然后执行 `Q'`。

本文里的 `DeltaD` / `DeltaQ` 是数学说明用语。代码主路径不要新增
`delta_log`、`delta_source`、`delta_query` 这类名字，因为当前 interface 里没有
单独的 delta schema、delta table、或 delta leaf。当前代码里 differentiated
query 的 source leaf 仍然是 `QueryExpr(op="log")`，runtime 只是在执行 `Q'` 时
把它绑定到本次 `changed_rows`。

因此，primitive operator rule 通常只推导 delta expression；`union`、
`subtract`、upsert、delete、replace 等 state application 语义属于
materialization boundary，不应该无脑塞进每个 primitive operator rule。

例子：

```text
Q  = D.sem_filter(p).select(cols)

delta expression:
  DeltaQ = DeltaD.sem_filter(p).select(cols)

maintenance query:
  Q' = V.union(DeltaQ)
```

## 2. Differential Policy Artifact / DifferentiatedPolicy

`Q -> Q'` 不应该只能发生在 `memory.add(...)` 时。它不看真实 data，所以更适合
作为 policy-level differentiation 阶段的一部分：

```text
Policy Q
  -> differentiate 得到 Q'
  -> offline optimizer 优化 Q'
  -> 保存 DifferentiatedPolicy artifact
  -> end-user runtime load artifact 并执行
```

`DifferentialPolicyCompiler` 负责 whole-policy differentiation。它不是 single
view 的 `DifferentialQueryPlanner`，而是把完整 policy 转成
`DifferentiatedPolicy`：

```text
MemorySpec
  -> DifferentialPolicyCompiler
  -> DifferentiatedPolicy
  -> optional offline optimizer
  -> DifferentiatedPolicy
  -> runtime
```

`DifferentiatedPolicy` 的最小方向是保存：

- `spec: MemorySpec`。
- `view_queries: Mapping[str, QueryExpr]`，每个 public view 的 differentiated
  update query `Q'`。
- `retrieval_queries: Mapping[str, QueryExpr]`，parameterized retrieval query
  templates。
- `view_dependencies: Mapping[str, tuple[str, ...]]`，public view dependency
  graph。
- `view_execution_order: tuple[str, ...]`，runtime 可直接执行的 topological
  order。

不要在 `DifferentiatedPolicy` 里增加 `optimized_view_queries` /
`optimized_retrieval_queries` 这类字段。Optimizer 应输入一个
`DifferentiatedPolicy`，输出一个新的 `DifferentiatedPolicy`，直接改写
`view_queries`、`retrieval_queries`、dependencies 或 execution order。

artifact 可以用 JSON、YAML、或其他机器可执行格式表示。Runtime load artifact 后
只负责把真实 `changed_rows`、当前 materialized views、storage state 绑定进去并
执行；runtime 可以继续做执行期优化，但不应该承担主要 query rewrite 责任。

早期实现曾在 `memory.add(...)` 时调用 planner，适合调试但不是最终架构。
当前第一版已经采用 in-memory `DifferentiatedPolicy`；
JSON/YAML IO、serializer 和 durable artifact save/load 留到后续。

## 3. Declarative Retrieval Query Templates

Retrieval query 也应该进入 differentiated policy，但它进入的是 parameterized
query template，不是已经绑定具体 user text 的 concrete query。

Policy writer 应声明 class-body retrieval query：

```python
class ClaudeMemory(am.Memory):
    ...
    retrieval_query = (
        topics.select(["name", "description", "type"])
        .sem_topk(am.UserQuery(), 5)
        .join(topics, on="name")
        .select(["name", "description:right", "type:right", "body"])
    )
```

`Memory.query(...)` 是 framework 内置的 end-user method，policy writer 不需要
手写：

```python
memory.query("design docs")
```

`UserQuery` 是 runtime-bound placeholder，语义是 end-user retrieval input。它
应该和 column placeholder `{message}` 分开，避免把 user query text 和 dataframe
column reference 混在同一套 string placeholder 里。

Offline 阶段保存：

```text
retrieval_queries["default"] =
  materialized_view("topics")
    .select(["name", "description", "type"])
    .sem_topk(UserQuery("query"), 5)
    .join(materialized_view("topics"), on="name")
    .select(["name", "description:right", "type:right", "body"])
```

Runtime 阶段再绑定：

```text
UserQuery("query") -> "design docs"
```

Retrieval query 不做 view maintenance rule，也不产生 `V'`。Retrieval-time
exact `join(on=...)` 是确定性 lookup / body fetch，不需要 differential rule。
它属于 query-time retrieval template，可以被 offline optimizer 改写 template，
也可以被 online optimizer 在绑定 user text 后做 batching、cascade、cache、
model choice 或 index 相关优化。

## 4. Planner / Rules 职责边界

代码命名约定：

- `DifferentialQueryPlanner.differentiate(view)`：生成 differentiated query
  `Q'`，语义是返回下一版 view `V'`。
- `DifferentialRules.differentiate(query, *, source_input, current_view,
  is_view_boundary, instruction_rewriter)`：执行 operator / pattern rewrite。
- `source_input`：differentiated query 的 source leaf；当前复用
  `QueryExpr(op="log")`，不新增 `delta_input` / `delta_log` operator。
- `current_view`：当前 materialized view 的 logical placeholder，例如
  `QueryExpr(op="materialized_view", params={"name": view.name})`。
- `changed_rows`：runtime 真实 DataFrame，来自 `Memory.add(...)`，不是
  `QueryExpr`。
- `DifferentialInstructionRewriter`：instruction rewrite hook。第一版只做
  deterministic placeholder rewrite，不做 semantic prompt rewrite。

暂不引入 `Context` class。如果以后参数变多，可以考虑
`DifferentiationScope`，但当前先使用显式 keyword 参数，避免过早抽象。

`rules.py` 应负责 operator-level 和 pattern-level rewrite：

- 推导 `select`、`sem_filter`、`sem_map`、`sem_flat_map` 等 row-local delta
  expression。
- 推导 `sem_join` 这类有明确 relational delta 公式的 operator pattern。
- 推导 `sem_groupby(...).sem_agg(...)` 这类 grouped aggregate view-boundary
  pattern。

`differential.py` 应负责 view-level context：

- 知道当前正在维护哪个 `MemoryView`。
- 提供 `current_view` / `V` 这类 materialized-view placeholder。
- 提供 `DeltaD` / changed input placeholder。
- 把 rule 产生的 delta expression assemble 成返回 `V'` 的 maintenance query。
- 注入 instruction transformer。第一版 transformer 只做 deterministic
  placeholder rewrite；以后如果 `sem_groupby -> sem_join` 或 `sem_agg -> sem_map`
  需要 semantic prompt rewrite，可以替换 transformer，而不是重写整个 planner。

`differentiate(...)` 的返回约束：

- `is_view_boundary=False`：只能返回 changed-output fragment。
- `is_view_boundary=True`：可以返回 full `V'` query。
- `sem_groupby(...).sem_agg(...)` 这类 stateful pattern 只能在 view boundary
  返回 full `V'`。如果它出现在普通中间 subtree，第一版应
  `NotImplementedError`，不要假装可以维护未 materialized 的中间 state。

## 5. Row-Local Rules

当前确定的 row-local rules：

```text
Delta(log) = DeltaD

Delta(select(Q)) =
  select(Delta(Q))

Delta(sem_filter(Q, instruction)) =
  sem_filter(Delta(Q), instruction)

Delta(sem_map(Q, input_cols, output_cols, instruction)) =
  sem_map(Delta(Q), input_cols, output_cols, instruction)

Delta(sem_flat_map(Q, input_cols, output_cols, instruction)) =
  sem_flat_map(Delta(Q), input_cols, output_cols, instruction)
```

这些 rule 只产生 changed output。它们不负责 `V.union(...)`。如果这些
operator 的输出是某个 materialized view 的最终输出，planner 再在 view
boundary assemble：

```text
Q' = V.union(DeltaQ)
```

如果这些 operator 只是中间 subquery，则不需要在中间位置 union。未来如果
optimizer 选择 materialize 某个中间 table，该中间 table 自己才形成新的
materialization boundary。

## 6. `sem_join` Rules

`sem_join` 的 inner join delta 公式是确定的：

```text
Q = L.sem_join(R, instruction, how="inner")

DeltaQ =
  DeltaL.sem_join(R, instruction, how="inner")
  .union(L.sem_join(DeltaR, instruction, how="inner"))
  .union(DeltaL.sem_join(DeltaR, instruction, how="inner"))
```

如果 `Q` 是 materialized view `V` 的 definition，则 maintenance query 可以是：

```text
Q' = V.union(DeltaQ)
```

这个 rule 的前提是 planner / runtime 能绑定 old `L` 和 old `R`。也就是说，
`L` 和 `R` 必须是 view、stored table、或已经 materialized 的 intermediate
state。

如果 `L` / `R` 是任意 subquery，当前不要假装可以 generic differential。
未来有两个可选方向：

- **要求中间 materialization**：planner 发现 join 需要 old `L/R`，就要求
  runtime / materializer 保存 `L` 和 `R`。
- **full recompute fallback**：重新计算 `L(D union DeltaD)` 和
  `R(D union DeltaD)`，再执行 join。结果正确，但不是 differential，成本和
  latency tradeoff 必须显式暴露。

第一版 generic `sem_join` differential 只支持 `how="inner"`。`how` 为
`"left"`、`"right"`、或 `"outer"` 时应先 `NotImplementedError`，不要假装
append-only maintenance 已经正确。

原因是 non-inner join 的 old unmatched output 可能会因为新 match 失效。例子：

```text
old L:
  left_row = "Alice likes concise docs"

old R:
  empty

old V = L.sem_join(R, how="left"):
  ("Alice likes concise docs", NULL)

DeltaR:
  right_row = "documentation preference"

如果 DeltaR 匹配 left_row，正确 V' 应该是：
  ("Alice likes concise docs", "documentation preference")

旧输出：
  ("Alice likes concise docs", NULL)

必须删除或改写。单纯 V.union(DeltaQ) 会同时保留 matched row 和旧 unmatched
row，因此是错误的。
```

未来可以考虑两类解决方向：

- **full recompute fallback**：重新计算完整 non-inner join，结果正确但不是
  differential。
- **provenance / action-based maintenance**：记录 row identity、`delta_minus`、
  row action tag、或 `match_type`，从而表达旧 unmatched output 的删除 /
  改写。

这不影响 Claude-style grouped aggregate rule。Claude rule 里的 `outer join` 是
maintenance query `Q'` 内部用来构造 merge input 的一步，随后 `sem_map` 直接
产出下一版 full view `V'`；它不是一个长期 materialized join view 的
append-only maintenance。

## 7. Instruction Placeholder Rewrite

`sem_groupby(...).sem_agg(...) -> sem_join(...).sem_map(...)` 这类 rewrite 会改变
输入 schema。典型情况是 `sem_join` 后重名列会被改成 side-suffixed columns：

```text
name        -> name:left, name:right
description -> description:left, description:right
body        -> body:left, body:right
```

如果原 instruction 仍然包含 `{name}`、`{body}` 这样的 placeholder，而 join 后
DataFrame 已经没有 `name` / `body` column，LOTUS 会严格解析 placeholder 并可能
直接报错。这是 correctness risk，不只是 accuracy risk。

第一版 instruction rewrite 只做 deterministic placeholder rewrite，例如：

```text
{name} -> {name:left} and {name:right}
{body} -> {body:left} and {body:right}
```

这只是保证 differentiated query 可执行，不做 create / keep / merge / delete 的
semantic prompt rewrite。更好的 semantic rewrite 后续再设计，例如显式说明 left
side 是 new candidate、right side 是 existing memory、matched rows 才需要 merge。

## 8. `sem_agg` Rules

`sem_agg` 应该有 maintenance rule，但必须写清楚精度边界。

Full view:

```text
V = D.sem_agg(input_cols, output_cols, instruction)
```

已实现的第一版 standalone compressed-state maintenance 形态是：

```text
DeltaX = differentiate(X)

Q' =
  V.select(input_cols)
  .union(DeltaX.select(input_cols))
  .sem_agg(input_cols, output_cols, instruction)
```

这里的 hard correctness check 只是 schema-level：`V` 和 `DeltaX` 都必须包含
显式声明的 `input_cols`。多余 columns 不影响 operator correctness；缺少
`input_cols` 才是 hard error。

这条 rule 是 compressed-state semantic maintenance，可能不同于：

```text
Q(D union DeltaD)
```

除非 `V` 恰好是 sufficient aggregate state。这里的 sufficient 不是 schema
equality。最终 view 的 columns 和原始输入 columns 不同，不代表一定不
sufficient；关键是维护所需信息是否仍然存在：

```text
required maintenance inputs
  subset of current_view_state + changed_input
```

例如原始输入包含 `name, body, timestamp, evidence`，最终 view 只有 `name, body`。
如果 instruction 只需要 `name/body` 继续 merge，则 current view 可能 sufficient。
如果 instruction 需要 `timestamp/evidence` 判断 freshness、contradiction、delete，
而这些字段已经被 view 丢掉，则 current view 不 sufficient。

第一版不做 semantic sufficient proof，也不做 full recompute fallback。只要
`input_cols` 存在，就可以生成 compressed-state `Q'`；但文档和 audit 必须明确
它是 approximate maintenance，不声明 full-recompute equivalence。

第一版 instruction 可以直接复用。未来如果发现输入 shape 改变导致语义偏移，
再通过 instruction transformer 自动生成 aggregate-to-merge instruction。

## 9. `sem_groupby(...).sem_agg(...)` Grouped Aggregate View-Boundary Rule

`sem_groupby(...).sem_agg(...)` 应该是 pattern rule，不应该作为
`differential.py` 里的特殊业务逻辑。

Full view pattern:

```text
V = (
  X
  .sem_groupby(input_cols=group_cols, instruction=group_instruction)
  .sem_agg(input_cols=agg_input_cols, output_cols=V_columns, instruction=agg_instruction)
)
```

Delta branch:

```text
DeltaT = (
  DeltaX
  .sem_groupby(input_cols=group_cols, instruction=group_instruction)
  .sem_agg(input_cols=agg_input_cols, output_cols=V_columns, instruction=agg_instruction)
)
```

Coarse full-next-view maintenance query:

```text
Q' = (
  DeltaT
  .sem_join(V, instruction=group_to_join(group_instruction), how="outer")
  .sem_map(output_cols=V_columns, instruction=agg_to_map(agg_instruction))
  .select(V_columns)
)
```

第一版 `group_to_join(...)` 和 `agg_to_map(...)` 只做 deterministic placeholder
rewrite，不做 semantic prompt rewrite：

- `sem_groupby -> sem_join` 复用 topic identity / grouping predicate。
- `sem_agg -> sem_map` 复用 aggregate / merge instruction。

这条 rule 产出的是 `V'`，不是 minimal `DeltaV`。它适合 Claude-style topic
memory 的第一版实现，因为我们还没有 `delta_plus` / `delta_minus` /
`apply_delta` / storage upsert semantics。

代码命名不要把这个 generic rule 叫 `consolidation`。`consolidation` 可以作为
Claude prompt 里的业务词，但 planner / rules 层应使用
`grouped aggregate view-boundary rule` 或类似命名。后续实现也应拆小函数，例如：

- `_differentiate_row_local_unary(...)`
- `_differentiate_binary_fragment(...)`
- `_differentiate_sem_join(...)`
- `_differentiate_sem_agg(...)`
- `_differentiate_grouped_aggregate_view(...)`

## 10. `sem_groupby.sem_agg` 的优化空间

`DeltaT.sem_join(V, how="outer")` 之后，不是每一行都必须做 expensive
semantic merge。

可优化情形：

- existing-only row：delta side 为空，结果可以 deterministic keep。
- delta-only row：existing side 为空，结果可以 deterministic create。
- matched row：两边都有内容，才需要 semantic merge / rewrite。

要做这个优化，需要 join output 暴露足够的信息，例如：

- left/right side 是否为空。
- `match_type`：matched、unmatched_delta、unmatched_existing、contradict、
  delete_target、uncertain 等。
- row action tag：keep、create、update、delete、skip。

这些会影响 query shape、runtime apply semantics、storage API 和 audit trace。
当前先记录为 future optimization，不在第一版 rules 中定死。

## 11. Apply Delta / Future State Effects

未来可能有多种维护形态：

- `Q'` 直接返回 full next view `V'`。
- `Q'` 返回 `delta_plus` / `delta_minus`，runtime 用 `subtract` + `union`
  应用。
- `Q'` 返回带 action tag 的 rows，由 runtime / storage 执行 keep、create、
  update、delete、skip。
- 引入 internal-only `apply_delta` operator，但不暴露为 public policy API。

当前 Claude-first implementation 应优先使用 full-next-view `V'` 形态，因为它
最容易审计，也不要求提前设计 storage upsert/delete contract。

## 12. Public View Dependency Binding

如果一个 public view 依赖另一个 public view，planner 应把依赖子树绑定成
materialized view，而不是在 dependent view 里再次 differentiate 上游 view 的
内部 query。

例子：

```text
topics = log.sem_flat_map(...).sem_groupby(...).sem_agg(...)
catalog = topics.sem_map(...).select(...)

catalog 的 differentiated query 应基于：
  materialized_view("topics").sem_map(...).select(...)
```

这条规则是 general 的，不是 Claude special case。但当前第一版有两个限制：

- **exact structural match only**：只有 query subtree 和另一个 public
  `MemoryView.query` 完全相等时才替换；不识别 logically equivalent query。
- **declaration-order dependency**：runtime 当前依赖 view declaration order。
  如果 dependent view 先执行，可能读到 missing / stale materialized state。
- **mixed log + upstream view unsupported**：当前 fast path 只覆盖 pure
  upstream-only dependency，例如 `catalog = topics.sem_map(...)`。如果一个
  dependent view 同时包含 base `log` 和 upstream `materialized_view`，例如
  `union(log.sem_filter(...), topics.sem_map(...))`，就需要 upstream view 在本次
  add 中产生的 changed-view state。v0.0 runtime 目前只暴露 current / next view
  state，不暴露 changed upstream view rows，因此这类 mixed tree 暂不支持
  differential。

未来 `DifferentiatedPolicy` artifact 应保存 public view dependency graph，并在
compile 阶段做 topological ordering 和 cycle detection。Runtime load artifact
后按 compiled order 执行，而不是依赖 class body declaration order。

## 13. 暂不实现的 Rules

- `filter(predicate)` / `assign(...)`：当前仍接受 arbitrary Python value，还没
  收窄成 serializable deterministic expression。
- generic `subtract` differential：需要 negative delta / delete semantics。
- generic arbitrary-subquery `sem_join`：需要 old intermediate materialization
  或 full recompute fallback。
- mixed base log + upstream materialized view dependency：需要 upstream
  changed-view state、中间 materialization，或 full recompute fallback。
- generic `left/right/outer sem_join` delta：旧 unmatched output 的失效和改写
  还没设计。
- standalone `sem_groupby`：它是 grouped intermediate，不是最终 view output。
- `sem_topk` maintenance rule：`sem_topk` 主要是 query-time retrieval /
  ranking template。它不维护 materialized view，也不产生 `V'`；但 declarative
  retrieval query template 应进入 `DifferentiatedPolicy.retrieval_queries`。

这些不是永远不做，而是不能在没有 state / materialization / apply semantics 的
情况下假装已经有正确 generic rule。

## 14. Validation / Error-Message TODO

当前第一版把 correctness 相关的 placeholder rewrite 做在 differentiation 阶段。
后续可以补两个 DX 改进，但不应在本轮规则里硬塞完整 validator：

- **spec-time placeholder validation**：在 `Memory.spec()` / policy compile 阶段
  提前检查 semantic instruction 中的 `{identifier}` placeholder 是否能解析到
  当前 operator scope 的 input / output columns。注意这需要可靠 column-scope
  推导，不能只做字符串扫描。
- **view-name-aware planner errors**：当某个 view 的 grouped aggregate pattern
  缺 output columns、placeholder typo、或 unsupported mixed dependency 报错时，
  错误信息应带上 `MemoryView.name`，例如 `topics: ...`，减少 deep stack debug。

## 15. 实现顺序

建议按以下顺序实现，避免一次性重写 planner/runtime：

1. 更新 planner / rules docstring 和局部变量名，不改行为。
2. 让 `DifferentialRules.differentiate(...)` 接收显式 keyword 参数：
   `source_input`、`current_view`、`is_view_boundary`、`instruction_rewriter`。
3. 实现 row-local changed-output fragment rules：`log`、`select`、
   `sem_filter`、`sem_map`、`sem_flat_map`。
4. 在 `DifferentialQueryPlanner.differentiate(view)` 的 view boundary assemble
   `current_view.union(fragment)`，使 planner 返回完整 `Q'`。
5. 实现 `sem_groupby(...).sem_agg(...)` 的 view-boundary stateful pattern，
   返回 full `V'` query。
6. 更新 runtime：执行 `Q'`，并把结果写入 `state[view.name]`。Runtime 继续用
   `changed_rows` 表示真实输入 batch，不引入 `delta_log` / `delta_source`
   变量。
7. 已实现：增加 in-memory policy-level differentiation surface：
   `Memory.differentiate_policy() -> DifferentiatedPolicy`。
8. 已实现：支持 declarative retrieval query：policy writer 写
   native-like `topics` manifest selection + deterministic `join(on="name")`
   body fetch，runtime 内置 `Memory.query(...)`。
9. 已实现：给 `DifferentialInstructionRewriter` 加 deterministic placeholder
   rewrite，先不做 semantic prompt rewrite。
10. 已实现：给 standalone `sem_agg` 加 approximate compressed-state view rule；
    只做 schema-level `input_cols` check，不做 semantic sufficient proof 或 full
    recompute fallback。
11. 已实现：把 grouped aggregate view-boundary rule 和 standalone aggregate view
    rule 拆成独立 helper，避免 generic rules 层使用 `consolidation` 作为抽象名。
