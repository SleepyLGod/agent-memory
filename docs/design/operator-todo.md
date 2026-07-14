# Operator TODO：LOTUS 对齐与优化清单

本文只记录 `QueryExpr -> LOTUS-backed execution` 这一层的 operator 后续工作。
它不讨论 `Q -> ΔQ` differential rules，不讨论 Claude policy，不讨论 storage。

当前状态可以概括为：

- 主要 operator 已经有可运行的 baseline lowering。
- LOTUS 原生等价的路径优先直接调用 LOTUS。
- 语义不等价的路径由 `agent-memory` 自己实现 lowering，并尽量复用 LOTUS 的
  LM、prompt formatter、cache、postprocess pattern 和已有 semantic operator。
- 还没有达到完整 LOTUS parity，尤其是 cascade、strategy、stats、audit trace、
  embedding/index pruning、large-group aggregation 等执行能力。
- Public policy API 保持只表达逻辑语义；backend、LOTUS、优化策略、debug
  输出都属于 runtime / adapter config，不属于 `MyMemory` class 里的 view query。

## 1. 原则

- 先保证 operator 语义正确，再谈 cost / latency 优化。
- 能精确复用 LOTUS 原生 operator 的地方，直接复用。
- 不能精确复用的地方，不 monkeypatch、不假装等价；在 `agent-memory` 里定义
  lowering contract。
- 优化能力如果来自 LOTUS，应该通过 runtime / adapter config 显式配置，而不是
  放进 policy author 的 public operator API。
- `QueryExpr.params` 只记录 logical query 参数，例如 instruction、columns、join
  type、top-k 的 `k`；不要把 backend execution knobs 写进 query tree。
- 每补一个真实执行能力，都要有 offline unit test 和 gated real LOTUS audit。

## 2. 当前 Operator 状态

| operator | 当前状态 | LOTUS 对齐程度 | 后续重点 |
|---|---|---|---|
| `select` | 可用 | pandas 本地实现 | 暂无 |
| `concat` / `union` / `subtract` / `drop_duplicates` | 可用 | pandas 本地实现 | 暂无 |
| `filter(predicate)` / `assign(...)` / predicate `join(...)` | 可用 | pandas 本地实现 | 后续只按真实需求扩 deterministic expression subset |
| `sem_filter` | 可用 | native LOTUS path，基础 adapter config 已接入 | 补 side-channel stats / cascade real audit |
| `sem_map` 单输出 | 可用 | native LOTUS `sem_map` | adapter config 对齐 execution options |
| `sem_map` 多输出 | 可用 | custom structured lowering | 对齐 LOTUS execution ergonomics，不把 debug 写入结果列 |
| `sem_flat_map` | 可用 | custom structured lowering | 后续考虑 batching / retry / cost controls |
| `sem_join inner` | 可用 | lower-level LOTUS `sem_join` | 补真实 cascade audit |
| `sem_join left/right/outer` | 可用 | inner + 本地 unmatched rows | 继续验证 column shape / metadata |
| `sem_groupby` | baseline 可用 | custom pairwise lowering | 加 candidate pruning / indexing |
| `sem_agg` 单输出 | 可用 | lower-level LOTUS `sem_agg` | adapter config 接入/验证 aggregation controls |
| `sem_agg` 多输出 | 可用 | agent-memory structured hierarchical lowering | large-group audit；等 PyPI LOTUS 支持 native response_format 后可重新评估 |
| `sem_topk` | 可用 | native LOTUS `sem_topk(...)`，adapter config 控制 execution method | 继续补 hybrid / index path |

Multi-output `sem_map` 未来可以有两条 optimizer-selectable lowering：

- structured one-call：一次 structured generation 同时写多个 output columns，成本低，
  字段一致性更好，但不是 LOTUS native `df.sem_map` 的原始 one-column shape。
- native per-column：每个 output column 调一次 LOTUS native `df.sem_map`，可以更
  直接复用 LOTUS 原生 execution options，但成本变成 N 倍，字段之间也可能不一致。

`sem_flat_map` 当前要求每个 input row 返回 `{"rows": [...]}` wrapper，以提高
JSON-mode 稳定性，例如：

```json
{
  "rows": [
    {"topic": "support group", "summary": "Caroline attended an LGBTQ support group."},
    {"topic": "self acceptance", "summary": "The group helped Caroline feel accepted."}
  ]
}
```

这属于 lowering format，不改变 `sem_flat_map` 的 logical 语义：一行输入仍然
产生 zero or more output rows。可选 `ordinal_col` 由 executor 在解析成功后按每个
input row 从 0 编号，不进入 prompt。

## 3. `sem_filter` TODO

当前 `sem_filter` 已经直接使用 LOTUS，所以 LOTUS 的核心执行能力在底层是有的。
问题不是 policy API 缺参数。当前基础 adapter config 已经能接入常用 LOTUS
execution options，但 stats / raw / explanation 的 side-channel，以及 cascade
真实 audit 还需要继续做。

LOTUS 已有但当前没有完整暴露的能力包括：

- `examples`：few-shot examples，提高 predicate 判断稳定性。
- `strategy`：例如 COT / ZS_COT。
- raw outputs / explanations：作为 adapter trace、audit log 或 stats side-channel，
  不默认写入 result DataFrame。
- `default`：模型输出无法 parse 时的默认 boolean。
- `safe_mode`：估算成本。
- `cascade_args` / `helper_examples` / `return_stats`：使用 helper model 或
  embedding proxy 做 cascade filtering，减少大模型调用。
- `return_all` / `suffix`：LOTUS dataframe shape/debug 选项。当前不进入
  logical result，因为它们会改变输出 shape 或引入 execution/debug columns。

这里不是 LOTUS 没实现，也不是要把这些 knobs 加到 `Relation.sem_filter(...)`。
这些是 execution policy，应进入 adapter/runtime config。当前实现默认保持
`return_raw_outputs=False`、`return_explanations=False`、`return_stats=False`，
避免 debug 信息混入 logical result DataFrame。

待办：

- 设计 stats / raw / explanation side-channel，不要写入 result DataFrame。
- 增加 real audit：普通 filter、examples、可选 cascade。
- 如果后续 runtime 有统一 trace object，把 `sem_filter` 的 parse failure、
  helper-model decisions 和 stats 统一接进去。

## 4. `sem_topk` TODO

当前 `sem_topk` 不应该在 public API 里暴露 `method` 等 backend 参数。Policy
writer 只写：

```python
df.sem_topk(instruction, k)
```

adapter 再根据 execution config 调用 LOTUS：

```python
df.sem_topk(
    instruction,
    K=k,
    method=method,
    strategy=strategy,
    cascade_threshold=cascade_threshold,
    return_stats=return_stats,
    safe_mode=safe_mode,
    return_explanations=return_explanations,
)
```

LOTUS 支持不同 ranking 方法和执行选项：

- `method="naive"`：全 pairwise，慢但最直接，适合早期 correctness audit。
- `method="quick"`：成本较低，但排序路径更启发式。
- `method="heap"`：heap-based top-k。
- `method="quick-sem"`：使用 semantic index / embedding 辅助优化。
- `strategy`：COT / ZS_COT。
- `cascade_threshold`：小模型 / 大模型 cascade。
- `return_stats`：返回调用次数、token、排序解释等统计。
- `return_explanations`：返回解释；后续应进入 trace / stats side-channel，而不是
  logical result DataFrame。

当前默认用 `naive`，因为它更接近“不省钱但最直观”的 gold execution。
`quick`、`heap`、`quick-sem` 和后续 hybrid search 都是可选执行策略。

待办：

- 真实 audit 继续比较 `naive` 和 `quick` 的语义差异、成本差异。
- 未来设计 agent-memory 自己的 hybrid search，不强行绑定 LOTUS 的单一
  `method`。
- `quick-sem` 需要 semantic index / retrieval model / vector store 配置；不要在
  没有 backend config 的情况下假装已完整支持。
- 如果 `return_stats=True`，不要把 stats 混进 logical result DataFrame；后续应走
  adapter trace / audit side-channel 或 runtime stats object。

## 5. `sem_groupby` TODO

当前 `sem_groupby` 是 baseline：

```text
exact duplicate input-column collapse
-> unique input-column rows pairwise semantic comparison
-> union-find group assignment
```

这个语义清楚、容易审计，但 pairwise 比较是 `O(N^2)`。数据量变大时，不能把所有
row pair 都交给 LLM。

Public API 使用 `input_cols`，不是 relational `key`。这些 columns 是判断 group
membership 的 evidence；它们不是 exact equality key，也不一定直接成为最终输出
列。最终 canonical group fields 仍由后续 `sem_agg(...)` 产生。

后续优化思路是先做 candidate pruning，再用 LLM predicate 精判：

```text
rows
-> embedding / index / cluster / sim-join 找候选 pairs
-> LLM 判断候选 pairs 是否同组
-> union-find 生成 group ids
```

可复用的 LOTUS 工具包括：

- `sem_cluster_by`：候选粗分组，但它不等价于 agent-memory 的 semantic
  grouping contract。
- `sem_sim_join` / vector search / semantic index：找高相似候选 pair。
- `sem_filter` / `sem_join`：对候选 pair 做 LLM predicate 精判。
- cascade：先用便宜 proxy，难例交给大模型。

当前没有配置 embedding model / vector index backend 时，不应该默认启用这些
pruning path。先保留清晰的 pairwise baseline 和未来 candidate-generator
接口位置；等 backend config 明确后，再接 `sem_sim_join`、semantic index 或
cluster-based pruning。

另一个 mode 是 closed-world labeled grouping。Open-world grouping 动态发现
group：

```python
rows.sem_groupby(
    input_cols=["title", "abstract"],
    instruction="Rows belong in one group when they discuss the same research topic.",
)
```

Closed-world grouping 预先给定 labels：

```python
rows.sem_groupby(
    input_cols=["title", "abstract"],
    instruction="Assign each paper to the best matching research area.",
    labels={
        "systems": "Systems, infrastructure, distributed systems, and databases.",
        "ml": "Machine learning models, training, evaluation, and datasets.",
        "hci": "Human-computer interaction and user studies.",
    },
)
```

当前最小实现只支持 single-label closed-world assignment：

- 每行必须输出一个已声明 label。
- 默认 label column 是 `"_label"`，可用 `label_col` 显式覆盖。
- 没有隐式 `other` bucket；如果需要 other，policy author 必须把 `"other"`
  写进 `labels`。
- `_label` 是 visible column，后续 `sem_agg(...)` 可以读取它；内部
  `_agent_memory_group_id` 仍用于 grouped aggregation。

Future work 只记录，不在当前实现中做：multi-label assignment、hierarchical
labels、label alias / canonicalization、unmatched confidence threshold、以及 labeled
grouping 的 candidate pruning / batching 优化。

除 pairwise baseline 外，未来 lowering strategy 可以包括：

- `single_batch`：一次 prompt 让模型输出所有 row 的 group assignments。适合
  小样本 audit，但 JSON 稳定性和 context window 风险更高。
- `chunked_single_batch`：chunk 内分组，再跨 chunk 合并 representatives。需要
  明确 representative 生成和跨 chunk merge 语义。
- `candidate_pairwise`：用 embedding / index / cluster / sim-join 生成候选
  pairs，再用 LLM predicate 精判。这是最贴近 LOTUS 优化能力的优先方向。

待办：

- 设计 `sem_groupby` 的 candidate generator abstraction，但不要提前泛化过度。
- 第一版优化优先做 `candidate_pairwise`，保留 LLM 精判作为最终语义。
- 后续完善 closed-world `labels` mode 的优化和 audit；不要把 multi-label 或
  hierarchy 混进当前 single-label 执行路径。
- audit 要输出候选 pairs、LLM accepted pairs、最终 group ids，方便人工检查。

## 6. `sem_agg` TODO

`sem_agg` 的本质是：

```text
rows -> aggregate row(s) with declared output schema
```

single-output 当前复用 LOTUS lower-level `sem_agg`，因此继承了 LOTUS 的
hierarchical aggregation / tree fold 能力。

multi-output 当前使用 agent-memory structured hierarchical lowering：

```text
group rows
-> LOTUS-main-style hierarchical aggregate
-> final LM pass with JSON object response_format
-> final JSON object with multiple output fields
```

这参考 LOTUS main branch 的方向：hierarchical aggregation 过程中先保持普通文本
aggregate，最后一轮再应用 structured output contract。区别是当前 PyPI LOTUS 还没
暴露 lower-level `sem_agg(response_format=...)`，所以 agent-memory 在 adapter
内部保留一个 compatibility helper，而不是 monkeypatch LOTUS 或覆盖 pandas
accessor。

更完整的 tree fold 路径未来可以是：

```text
group rows
-> chunked structured partial aggregate
-> merge partial aggregate rows
-> final JSON object with multiple output fields
```

当前 active runtime 不保留 `single_batch` / `lotus_hierarchical` backend
strategy。后续如果要恢复多个 lowering 选择，应该作为 optimizer/lowering 选择重新
设计，并用 real audit 证明比当前 structured hierarchical baseline 更稳。

`sem_agg_model_kwargs` 是 adapter execution knob，只用于 multi-output final LM
pass。当前 final pass 默认保证 `max_tokens >= 1024`，然后再合并
`sem_agg_model_kwargs`；调用方可以通过 adapter config 覆盖普通 model kwargs，
但不能覆盖 `response_format`。这个字段不是 public `sem_agg(...)` 参数，也不属于
logical query semantics。

tree fold 在大 group 下可能更稳，但要和 LOTUS native `sem_agg` 的优化明确区分：

- LOTUS native `sem_agg` 会把大 group 分 batch 聚合，再逐层 fold，避免超出
  context window。
- 当前 multi-output path 依赖 final JSON object response_format。DeepSeek /
  LiteLLM 如果结构化输出不稳定，real audit 应该直接失败，不做 retry/fallback
  掩盖问题。
- 未来 tree fold 在 group 很大时可能出现 partial summary drift、JSON 被截断、
  输出不稳定等问题，必须有 trace 和 audit 才能进入 active runtime。
- 暂不把 LOTUS `operator_cache` 直接套到 compatibility helper 上。当前 LOTUS
  decorator 面向 pandas accessor method，依赖 `self._obj` 参与 cache key；
  plain helper 需要单独设计 hash-safe cache key，不能直接照搬。

你说“差距就是 output 行数的多少”只对了一部分。更准确地说：

- output 行数：whole relation 是 1 行；grouped relation 是 `G` 行。
- output 列数：single-output 是 1 列；multi-output 是多列。
- 真正需要继续补齐的是：multi-output 的 tree fold 应该更接近 LOTUS native
  aggregation 的工程成熟度，而不是只停留在基础 chunk + merge。

待办：

- 保留 LOCOMO 小样本 real audit 对 whole/grouped、single/multi output 的覆盖。
- 固定 row-count chunking 暂时不需要做。它不是 LOTUS 原生 `sem_agg` 的
  token-budget hierarchy，而是 agent-memory 曾考虑过的 structured tree-fold
  experiment。当前 runtime 已采用 LOTUS-main-style token-budget hierarchy；额外
  long-context chunking 只作为后续 large-group optimization/audit，不混入本次
  reliability fix。
- 单独增加 large-group optimization audit。当前真实 LOCOMO audit 曾暴露
  experimental tree-fold path 会出现空 JSON / 空 intermediate summary；不要在没有
  trace / retry / cost model 前宣称 large-group path 稳定。
- 为 tree fold 增加 audit trace：每层 partial rows、merge rows、final rows。
- 后续根据真实 CSV 审核决定是否需要不同的 leaf instruction / merge instruction。

## 7. “接入 LOTUS options”是什么意思

这句话不是说重新实现 LOTUS，而是说：

```text
LOTUS 已经支持某个 execution option
但 agent-memory 的 adapter/runtime config 还没有表达它
LotusAdapter 也没有把这个参数传下去
```

例如 LOTUS `sem_filter` 支持 `examples` 和 `cascade_args`，但如果
`LotusExecutionConfig` 或上层 runtime config 没有这些字段，而且 adapter 不转发
它们，那么系统就无法从 agent-memory 使用 LOTUS 的这些能力。

这不代表要把参数加进 `Relation.sem_filter(...)`。Public operator API 的职责是
表达 logical query；execution config 的职责是选择 backend method、cascade、
examples、safe mode、trace 等执行策略。

这类 TODO 的工作一般是：

1. adapter/runtime config 增加必要字段。
2. `QueryExpr.params` 保持只记录 logical query 参数。
3. `LotusAdapter` lowering 把 config 转成 LOTUS 调用参数。
4. offline test 检查 logical params 没被污染、config 参数传递正确。
5. real audit 检查真实效果。

## 8. “优化 custom operator”是什么意思

custom operator 是指 LOTUS 没有精确等价 operator 的部分，例如：

- multi-output `sem_map`
- `sem_flat_map`
- `sem_groupby`
- multi-output `sem_agg`

这些不能假装是 LOTUS native operator。但实现时仍然应该复用 LOTUS 的基础设施：

- LM configuration
- prompt formatter
- operator cache
- safe mode / cost estimation
- examples / strategy / trace / audit stats
- lower-level `sem_filter` / `sem_join` / `sem_agg`
- semantic index / retrieval / cluster 工具

这里的目标不是追求“看起来和 LOTUS API 一样”，而是让 agent-memory 自己的
operator contract 在执行层也有类似 LOTUS 的成本、稳定性和可审计性。

Debug 信息默认不要进入 result DataFrame。比如 raw output、reasoning、
explanation 更适合作为 adapter trace / audit log / stats object。如果 explanation
本身是业务字段，就应该由 policy writer 显式声明在 `output_cols` 里，例如
`"memory_summary_rationale"`，而不是通过 execution option 偷偷写入结果列。

## 9. Python 代码里是否加 TODO

暂时不建议在每个 Python 文件里大量加 `TODO`。原因：

- 当前问题是跨 API、adapter、真实测试的路线问题，不是某个函数里一行能修掉的
  局部问题。
- 过多 inline TODO 会让代码噪音变大，而且容易过期。
- 设计 TODO 集中放在本文档和 `docs/design/TODO.md` 更容易维护。

可以加 inline TODO 的情况：

- 某个函数里确实有明确、局部、下一步就要修的技术债。
- TODO 能指向本文档的具体章节。
- 不加 TODO 会导致后续维护者误以为当前行为就是最终设计。

推荐格式：

```python
# TODO(operator-parity): wire sem_filter examples/cascade through adapter config; see docs/design/operator-todo.md.
```

不要写模糊 TODO，例如：

```python
# TODO: improve this later
```

## 10. Deterministic Aggregate / Source Metadata TODO

当前已实现单列/复合 `min` aggregate 和 row-wise `least` expression。复合
`min(columns=[...])` 返回字典序最小 tuple；`least` 只比较同一 row 的 operands。
二者都不会返回 ordering key 对应的另一份 payload。

待办：

- `min_by(value, order_by)` / `arg_min`：返回最小 ordering value 对应的整行或
  payload。只有出现真实 policy 需求后再确定 public spelling。
- delete / correction semantics：append-only `min` 可以只合并 old/new minima；
  当前 minimum 被撤回时，需要 raw-group recompute 或 ordered auxiliary state。
- concurrent / batch append ordering：当前 `_add_seq` 只定义单进程顺序 add。
  并发、同 batch 多 rows、跨 restore writer coordination 需要单独 runtime
  contract，不能让 policy writer修改 system columns。
