# Zep Native 对齐：先去掉无必要的事实重写

## 已核实的合同差异

核对对象是服务器 Native 128 条实验使用的源码，不是当前 upstream：
`native-zep-deepseek-flash-128e-20260917T131459Z/source/graphiti_core/utils/maintenance/`。

| 环节 | Native | 现有 adaptation | 本轮处理 |
|---|---|---|---|
| 重复事实 | 选择已有 edge，保留文字并追加 episode | semantic aggregation 生成新的 canonical 文字 | 新增独立代表事实 view，不覆盖旧 view |
| 矛盾 | 围绕新 edge，同时返回重复与矛盾候选 ID | canonical facts 的候选关系上执行 predicate | 保留原逻辑，不冒充已对齐 |
| 摘要 | 短内容追加；压缩时还有 episode/previous episodes | fact-summary 只使用事实证据 | 本轮保留，后续需要独立的信息来源合同 |

Native 的具体证据：`edge_operations.py` 的 exact-match fast path 和
`resolve_extracted_edge`；`node_operations.py` 的 `update_entity_summaries`
与 `_process_summary_flight`。因此不能把完整 Native resolver 当成原查询的
无条件等价 fusion，也不能把 fact-only 摘要称为 Native 摘要的完整复刻。

## 本轮实现

`ZepRepresentativeMemory` 在相同语义分组中保留最早到达 occurrence 的
`relation_type` 和 `fact`，不再请求模型综合这两列。排序键是
`(add_seq, fact_ordinal)`，不是模型自行判断的“最早”。provenance 和时间字段
继续走原确定性合并。分组、匹配、矛盾筛选、检索与评分不改。

底层使用小型确定性 `arg_min(order_by=..., columns=...)` 聚合。它连同
排序键一起保留选中行的 payload，因此多批合并仍可选择同一代表，不会把
两行的 relation_type 和 fact 分开拼起来。NULL 排序键和最小键对应冲突
payload 明确失败。重复 occurrence 保留原 provenance 语义，不去重来源。

已有 semantic join-map 负责匹配新旧组，arg_min 负责确定性合并状态。
没有另写状态执行器或将事实判断藏进 runner。仅有 arg_min 的普通 group_by
仍走现有确定性维护；semantic groupby 的代表状态使用现有 semantic-state
维护入口。这里省去的是事实状态生成，不是语义匹配。

程序化入口沿用现有 benchmark：

```python
physical_fusion="zep-representative"
```

配置选择新的 logical view，保留实体 identity fusion、摘要短路及既有可选
节点 batching / 判定复用；事实不再有 target-state synthesis fusion，因为
没有需要模型生成的 target state。旧 `zep-fact-summary` 和 `zep-combined`
不变。新 query 与物理版本有独立 fingerprint，不应恢复旧实验 checkpoint。
候选 profile 必须针对新 plan 重新绑定并预检，不复制旧 query digest。

## 证据边界与下一步

### 128 条对照的启动接线

`tools/zep_combined_smoke.py` 支持显式组合
`--prefix128 --maintenance-work --combined-physical --fact-summary --representative
--node-batch-size 16 --pack-small-groups`。旧入口默认值不变。输入与问题从上一轮
冻结 bundle 复制；Top5 配置按语义 site 重新解析到新计划，预检记录实际 query
digests，并拒绝丢失的筛选或 batching/reuse site。

代表事实物理版本为 `zep-representative-v2`：实体 singleton 改写后保留原
join 的 screening digest，避免因为物理子树改变而绕过 Top5。fact 匹配仍走
top-1 listwise；取消的是 fact synthesis，不是匹配调用。新运行仅可恢复自身
源码和配置身份，不恢复旧 view 的 checkpoint。

离线新增覆盖不同事实与具体细节不被合并、多个旧目标、来源保留、恢复、
真实计划的 profile 绑定及未知 site 拒绝。真实质量与性能必须读取新实验结果，
不能从离线 oracle 推断。

2026-09-18 本轮离线验收：相关执行与 representative 回归 104 项、benchmark
与 Zep harness 回归 82 项通过；修改文件 Ruff、Pyright 和两仓库 diff check
通过。只改变本工作树的接线、回归和文档，原 Replica 与两边 index 不变。

唯一真实运行目录：
`/mnt/data/agent-memory-experiments/zep-representative-top5-packed16-128e-20260918T122808Z`。
通过该目录 `control.sh status` 查看状态；失败后仅显式 `control.sh resume`，
不自动从空状态重跑。输入、69题、候选配置与上一轮逐字节一致；性能与质量
结果尚待该运行完成，不能将本段离线验收当作性能结论。

离线测试使用真实 planner/runtime 和确定性的匹配 oracle，验证重复输入、
当前字段、来源保留、全量/增量代表一致以及 snapshot 恢复；provenance
按 occurrence 比较，不能要求现有 array aggregation 保证到达顺序。

最早代表并不等于 Native 任意情况下选择的 edge：Native 匹配多个旧候选时
依赖其候选与决策顺序。新版本也仍然维护 canonical 候选间的矛盾关系，
不是“只处理新到达 edge”的有序 resolver。它是缩小差距的第一步，
不宣称 Native 等价、真实质量保持或已经追平性能。

初版仅完成离线验证；上述服务器运行已冻结128条、重绑相同候选配置。
运行结果需核对 fact synthesis 是否消失，再比较维护 tokens、延迟和相同
问题上的质量。摘要证据补全与联合 resolver 暂不混入这一步，避免无法归因。
