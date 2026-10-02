# Semantic Physical Fusion：第一轮实现

## 范围

基于 main `93d98d1`，增加默认关闭的物理优化入口，首条注册规则为
`zep-target-state`。不改 Zep/Mem0 的 logical query、join-map rule、retrieval
或候选筛选配置。没有自动搜索、代价模型或新执行器。

这轮只完成离线接线验证。真实 LLM 的 latency、tokens 和答案质量尚未测量，
不能把少一次调用等同于更便宜、更快或语义等价。

## 如何开启

```python
from agent_memory.adapters import LotusAdapter, LotusExecutionConfig
from agent_memory.memories.zep.policy import ZepMemory

adapter = LotusAdapter(config=LotusExecutionConfig(
    physical_fusion="zep-target-state",
    structured_parse_retries=0,
    lm_enable_cache=False,
))
memory = ZepMemory(adapter=adapter)
```

模型、embedding provider 与原有 `semantic_pair_profiles` 仍由调用者显式配置。
不设置 `physical_fusion` 时原行为、plan fingerprint 和执行 fingerprint 不变。
本轮不新增 benchmark CLI；后续实验通过同一 adapter 接线。

## 三层分工

| 层 | 职责 |
|---|---|
| `planner/physical.py` | 对已生成的 differential plan 应用显式注册的改写；验证适用形状，生成新的执行身份。 |
| `LotusAdapter.prepare_policy()` | 在 runtime 开始、storage 准备之前接入物理计划并校验配置。 |
| `adapters/lotus/fusion.py` | 复用候选筛选、结构化调用和 trace；把一次回答还原为原有 dataflow 所需的状态。 |

原 differential plan 的逻辑节点不变，只替换 fact aggregate 的 maintenance query。
关闭优化不改写任何节点。重复准备同一融合计划不会再次改写或改变 fingerprint。
恢复继续复用现有 snapshot，没有另一套状态存储。

## Zep：选择目标与生成目标状态合并

原来的两个模型阶段：

```text
新 fact states + 旧 fact states
-> 原有 partition / embedding 候选筛选
-> 为每条新 fact 选择至多一个旧目标
-> 对每个被选目标合并旧内容与新贡献，生成 canonical fact
-> 原有确定性 provenance、ID、时间处理
-> 后续 contradiction 检测
```

融合后，同一次维护中的候选图交给一个结构化请求。原 join instruction、
consolidation instruction 及各自声明的输入仍在请求中。回答包含：

- 每条有候选的新 fact 的目标 ID，或 `null`。
- 每个被选目标的 `relation_type` 和 `fact`，覆盖分配给它的所有新贡献。

这里“一条新 fact 至多选择一个目标”来自现有 Zep 的 exclusive membership / `k=1`，
不是融合额外施加的限制。多个新 fact 可以选择同一个旧目标；该旧目标只参与一次合并。
原本允许 overlapping membership 的计划不能启用这条规则。

确定性部分继续使用原 maintenance body：匹配者更新；无匹配新 fact 保留自身初始
canonical state；未受影响旧 fact 保留；provenance flatten、ID minimum、时间字段和
输出列由原关系计算处理，不能由模型任意编造。初始 delta 抽取及聚合未被消除。
不融合 entity 阶段，不提前构造 contradiction candidates。

## 有意保留的限制

- 只接受匹配注册 Zep fact 形状的 join-map plan，不按“看到 join 和 aggregate”就随意融合。
- 第一版使用现有 Chat JSON-object 结构化调用、严格结果校验和失败项重试；暂不接
  Responses JSON Schema、prompt batching、pairwise top-k、examples、strategy、
  native cascade 或 safe-mode 等不能忠实保留的组合。
- 原有 embedding 候选筛选可保留；proxy-only 判断不能被悄悄替换成 LLM 判断。
- 一次维护的全部合格候选共同进入请求，不暗中切块或截断。大批次存在上下文容量风险，
  后续先做固定小规模实验；这不是大批次已优化的声明。
- 原 join/consolidation 的指令保留，但物理 prompt 与调用上下文发生变化。
  离线确定性 oracle 只能验证状态组装与接线，不能证明真实模型输出等价。
- 上游有 retraction 时，现有 executor 仍走原 full-query fallback，不谎称该阶段也融合。
- 融合模式和 `zep-target-state-v1` prompt 合同进入执行身份。启闭不一致或版本不一致
  的 checkpoint 不能互相恢复。旧 schema-v1/LegacyViewRuntime 明确拒绝融合恢复。

同一物理请求只沿现有 LM/provider trace 记录一次调用与用量；`fusion_resolution`
另保存逻辑 assignments、canonical states、原始回答及 occurrence 映射，不复制计费。
非法/缺失/重复 ID、候选外目标、缺失目标状态和字段类型错误均明确失败，
不会默认当作“无匹配”，失败更新不提交到 runtime state。

## Mem0：下一条规则，而不是套用 Zep

最新 main 的实际路径是：

```text
抽取 memories
-> earlier/later 按 add_seq 做关系 predicate join
-> 对候选 pair 执行 duplicate sem_filter（可使用现有 embedding 筛选）
-> 提取 later occurrence key 并去重
-> left_anti 排除这些 later keys
-> 按 memory 文本去重输出
```

`relational.py` 的非等值 predicate join 先形成 cross product，再过滤时间关系。
`sem_filter.py` 在接收到 pair 表之后才应用 embedding profile。
因此候选筛选已能减少 LLM 判断，却没有自动消除前面的 pair 物化。
这是代码层的可优化点，不是新跑出的性能归因。

后续可在同一物理优化入口注册专门规则，融合“时间合法候选生成、duplicate 判断、
later-key 投影”。第一版仍按 pair 做 pointwise 判断，不使用 Zep 的单目标选择与
canonical-state 生成。它必须：

- 保持 add_seq 的严格小于关系，不新增同一消息内部的去重规则。
- 保持当前筛选的候选集合、方向、阈值、Top-k 与 tie-break；不能按分块重新 Top-k。
- 保持 occurrence key 和“存在至少一条 duplicate pair 就排除 later key”的语义。
- 若 pair 结果还有其他消费者，则拒绝消除该节点；不能只因为当前 sink 是 key 就丢失公开结果。
- 第一轮只主张减少宽 pair 表及投影的物化工作。保持相同候选与完整判断时，
  不承诺减少 provider calls；提前终止或多个判断合并请求需要另立执行合同和质量对照。

本轮没有实现 Mem0 融合，没有修改其 query、retrieval 或候选筛选配置。

## 后续实验门禁

先批准冻结输入上的 fused/unfused 对照，再跑固定 24-event smoke，最后完整 Sample 0。
保持同一模型、输入、候选筛选与 retrieval，分别记录 latency、tokens、LOCOMO 和 judge。
除了调用数，还需检查目标选择、事实内容、provenance、contradiction 和下游答案变化。
本轮不启动这些真实调用。

## 离线验证记录

新增 `tests/test_physical_fusion.py`，24 项测试通过；最终全量回归
1,750 passed、20 skipped。真实 LOTUS 和外部服务测试关闭。
Ruff 与空白检查通过。新增模块和测试的 Pyright 为零错误；所触及的既有模块
仍有 9 项类型问题，已在干净 `93d98d1` 上复现，未在本轮扩大范围修复。

测试包括真实 planner/runtime 的逐条更新及恢复、多个新 fact 指向同一旧目标、
候选筛选保留、schema/ID 错误、原子失败、融合步骤不重复执行，以及合并前后两个
逻辑结果共用一次 provider 用量记录。初始事实构造和模型回答使用确定性 oracle，
不是实际模型质量或性能测量。
