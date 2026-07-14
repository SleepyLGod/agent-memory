# Policy Differentiation 与共享执行 Runtime

这次改动解决的是一个很具体的问题：一个 memory policy 往往不是一条直线，
而是一组有依赖、也可能共享中间结果的 query。如果继续把每个 public view 当成互不相关的
`Q'` 单独执行，同一段 semantic extraction 可能重复调用，内部 aggregate 的
replacement 也无法正确传给下游。

用户仍然只定义最终状态：

```text
V1 = Q1(D)
V2 = Q2(D, V1)
...
```

`DifferentiatedPolicy` 和 `PolicyExecutor` 只是这份定义的编译结果与执行器，
不是另一套 memory 语义。

## 1. 四层职责

```text
MemorySpec / logical policy
  -> QueryDifferentiator + differential rules
  -> PolicyDifferentiator
  -> future optimizer
  -> PolicyExecutor
  -> ExecutionAdapter
```

各层只负责一件事：

- **Logical policy** 定义 `Q(D)`，也就是最终 view 应该是什么。
- **Differential rules** 定义某个 operator 或 pattern 收到 change 后应该算什么
  `Q'`。matched、left-only、right-only 等语义都属于 rule。
- **Policy differentiation** 把所有 public view root 合成一组有限、无环的执行
  节点，复用结构相同的 query，并记录 public view 对应的输出节点。
- **Policy executor** 按依赖顺序执行节点，保存内部 state，传递节点更新，并在整
  个 step 成功后一次提交。
- **Execution adapter** 忠实执行一个具体 `QueryExpr`，不决定 maintenance 语义。

Future optimizer 的位置在 compiler 和 runtime 之间。它可以换 lowering、共享
index 或选择更高效的 state，但不能把 runtime 变成第二个 rule engine。

## 2. DifferentiatedPolicy

`PolicyDifferentiator(...).differentiate(spec)` 会把所有 public views 编译成一个
immutable `DifferentiatedPolicy`。内部的 `QueryDifferentiator` 负责单棵 query 的
`Q -> Q'`；policy differentiator 负责共享节点和依赖顺序。最终 artifact 直接保存
nodes、execution order、public view outputs、retrieval queries、fingerprint 和
canonical grouped rule，不再嵌套第二层 plan。

每个 `DifferentialNode` 只保存：

- stable `node_id`；
- 已绑定直接 parent node 的局部 `query`；
- `input_node_ids`；
- execution kind；
- output schema；
- stateful semantic node 使用的 `maintenance_query`。

结构相等的 `QueryExpr` 只生成一个 node。同一个 semantic extraction 被两个下游
使用时，每个 add step 只执行一次，两个 child 读取同一份 next state。

Grouping carrier 不会错误地独立物化。比如
`sem_groupby(...).agg(...)`、`group_by(...).sem_agg(...)` 和 window aggregate
会作为完整 pattern 编译成一个 stateful node。

Policy 还保存：

- stable topological order；
- public view name 到 sink node 的 mapping；
- 包含 execution 和 maintenance query 的 fingerprint。

v2 checkpoint 只能恢复到 fingerprint 相同的 plan。不同 rule family 即使来自同
一条 logical query，也会得到不同 fingerprint。

## 3. Node Output Update 和 State

每个 node boundary 同时有两个概念：

- `state`：这个 node 上一次成功提交的完整 relation；
- `NodeOutputUpdate(output_rows, inserted_rows, retracted_rows)`：当前节点的完整
  next output，以及 old state 到 next state 的 multiset difference。

`NodeOutputUpdate` 只存在于 executor 内部，不是 public IR，也不会出现在 policy
writer 的 view query 里。它保留重复行的 multiplicity，不能用 set diff
或无条件 `drop_duplicates()` 代替。

第一版 deterministic lowering 选择最简单的正确实现：只要任一 parent 有
change，就用所有 parent 的 next state 重算当前 deterministic node，再比较
old/next state 产生 change。它不是最省算力的 DBSP lowering，但结果和 full
deterministic query 一致。以后 optimizer 可以把它换成 indexed incremental
join/aggregate，不需要修改 policy。

## 4. Semantic Nodes

### 4.1 Row-local semantic operator

`sem_filter`、`sem_map` 和 `sem_flat_map` 只对 inserted parent occurrences 调用
LLM。Executor 在 `semantic_output_cache` 中保存私有 occurrence correlation：

```text
input occurrence -> zero, one, or many output rows
```

具体实现可以用 DataFrame index 关联输入与输出，但这不是 policy column，不进入
prompt，也不改变 trace schema。`sem_flat_map` 的一对多输出共享同一 input
occurrence。

如果 deterministic parent 用 replacement 表达更新，runtime 会撤回旧 occurrence
缓存的 semantic output，只计算新的 occurrence。已经算过且没有变化的 input
不会再次调用 LLM。

### 4.2 Stateful semantic aggregate

Grouped 或 standalone semantic aggregate 使用现有 differential rule 生成的
`Q'` 维护 insert-only change。Runtime 提供：

- current node state；
- parent inserted rows；
- rule 需要的其他 materialized inputs。

如果 parent update 不含 `retracted_rows`，executor 执行上述 maintenance query。
例如 full query 可能是 `sem_agg(all raw facts)`，incremental query 可以是
`sem_agg(old summary + changed facts)`。这是已经接受的 semantic approximation。

如果 parent update 含有 `retracted_rows`，说明某个内部 input occurrence 的旧版本
已经被 replacement 或 retract-only change 撤回。此时 compressed state 不能证明
旧贡献已经消失，executor 会在 parent 的完整 next state 上执行当前 node 的 logical
query。它只重算当前 semantic node，不重算整个 policy，也不把 internal retraction
暴露成 public delete。

## 5. Window

`process_window` 保存 next-start cursor 和已完成窗口的 output state。只有新完成的
窗口会执行，失败 step 不推进 cursor。

普通 over-window 和 semantic over-window 保留专门 maintenance：只给新增 input
计算对应 frame，同时读取 parent 的完整 next state 作为 context。当前 source 是
顺序 append-only，不包含 watermark、late event 或多 partition 调度。

## 6. 一次 add 的事务边界

一次 `add()` 的执行顺序是：

1. 暂存 next log。
2. 按 policy execution order 依次计算 node next state 和 update。
3. 每个共享 node 最多执行一次。
4. 没有 parent change 的 node 直接跳过。
5. 所有 node 成功后，一次提交 log、public views、private node state、semantic
   output cache 和 window cursor。

任一 node 失败，以上 state 都保持在 add 前。已经发生的 LLM token 消耗和 trace
文件是外部 side effect，不承诺回滚。

Runtime 不能看到 outer join 的 null side 后擅自跳过 `sem_map`。如果
`rule-join-map` 生成了 `outer_join(...).sem_map(...)`，runtime 就必须执行完整
query。right-only 是否应该原样保留，只能修改 rule。

## 7. Checkpoint

- 新实例默认使用 `PolicyExecutor`，snapshot schema 是 v2。
- v2 保存 public state、private node state、semantic output cache、window state、
  occurrence counter 和 plan fingerprint。
- schema v1 checkpoint 自动切换到隔离的 `LegacyViewRuntime`。它使用当前 policy
  保存的 canonical grouped rule 重新生成旧 Q'，并继续产出 v1 snapshot。
- `_state` 仍只暴露 `log + public views`，供 retrieval 和本地分析使用。Private
  node state 不污染 view schema。

## 8. 当前不支持

本轮只处理有限、非递归、单 Log、顺序 append-only policy。以下情况 compile
time 或 runtime 明确拒绝：

- source delete；
- recursive/fixpoint query；
- standalone view-time `sem_join`；
- view-time `sem_topk`；
- generic optimizer 和 indexed physical state。

Differential rule 内部生成的 `sem_join` 不属于 standalone policy node，它是
完整 `Q'` 的一部分，仍由 adapter 执行。

## 9. 概念来源

这套实现借用了三类已有工作的概念，但没有引入对应 runtime 依赖：

- [Flink Planner](https://nightlies.apache.org/flink/flink-docs-stable/api/java/org/apache/flink/table/delegation/Planner.html)
  把 table program 翻译成可执行关系计划；stateful operator 维护 dynamic table
  update。
- [DBSP](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf) 把 incremental
  operator 按 circuit 组合，说明局部 rule 如何形成完整增量程序。
- [Incremental View Maintenance for Collection Programming](https://arxiv.org/abs/1412.4320)
  使用 incremental version / delta query 描述程序变化的组合。

当前实现更接近一个小型、可检查的 Flink-style executor：有依赖顺序、node
update、state 和 atomic step，但 deterministic node 先采用 full next-state recompute。
DBSP-style indexed incremental lowering留给 future optimizer。
