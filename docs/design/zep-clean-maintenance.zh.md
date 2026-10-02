# Zep：消除重复名称生成与分组等待

本轮只有离线验证，没有调用真实模型或启动服务器实验。保留 representative view、原 instruction、Top5、现有 fusion、listwise16、矛盾 batch16、delta 维护和 retrieval；不实现联合 resolver。

## 改了什么

| 改动 | 启用方式 | 边界 |
|---|---|---|
| 已确认且名称不变的实体直通 | `reuse_unchanged_entity_name=True`，默认 False | 仅注册的 representative 实体 canonical-name 节点；全部匹配已缓存，且每个目标的新旧名称完全一致，才跳过整次请求。不是“同名必定同实体”。 |
| 分组节点 prompt batching | `groupby_prompt_batching={site_id: PromptBatching(max_tasks=16)}`，默认空 | 只作用于指定的 open-ended exclusive 分组节点。复用现有结构化 predicate 执行器，不跨分区、不跨事件。 |
| 单次筛选计算复用 | 内部自动启用 | 范数按文本复用，相似度按有向文本对复用；原浮点计算顺序、阈值、TopK 和并列排序不变。没有跨事件缓存。 |

实体直通只替代模型生成的名称，仍走原 ID、mentions、provenance 和状态更新。任一匹配未知、名称不同或非注册语义字段，保持原请求。无候选的新实体沿用原新建流程。

原 `sem_groupby_pair_batch_size=32` 是 SDK 调度批次，每对仍是独立 prompt。新设置在指定节点替代这个调度设置，让一个 prompt 最多判断16对；其他节点仍用原 pair32。实际任务数可能少于16，不能拿配置上限冒充实际装载量。全局 `prompt_batching` 保持 None。

## 配置与恢复

普通 benchmark 入口、Zep driver、adapter 已传递两个新选项。节点配置按原 semantic site 身份绑定逻辑节点及维护副本，不匹配时预检失败；不支持 declared labels、overlapping 或 proxy-only 分组。全局 batching 与节点 groupby batching 不能混用。

新选项及版本进入执行指纹，旧配置的 checkpoint 不可用于新条件；同一新配置继续使用原 snapshot/restore。实体判定仍使用已有 executor 状态，不新增缓存服务。

后续服务器启动器在原 representative/listwise16 参数上增加：

```text
--groupby-batch-size 16 --reuse-unchanged-entity-name
```

启动器要求既有固定128条 representative 条件，自动绑定实体、事实两处分组 site，写入 manifest 和 condition identity。这里不是启动命令，也没有自动运行实验。

## 观测与验收

- 实体直通写 `unchanged_entity_name` 事件和零调用事实；原 `identity_reuse.state_regenerated` 不再把直接复用误报为重新生成。
- 分组沿用 `prompt_batching` trace 的 `task_count`、`prompt_count`、`chunk_sizes` 和重试记录，分别统计逻辑判断与物理请求。
- 离线检查覆盖相同名称/变化名称、未知匹配、多贡献、当前 metadata、重复行、分区隔离、非法输出、默认路径和恢复指纹。
- 使用真实 planner/runtime 与 fake provider 验证两个分组节点确实打包、实体判断跨恢复复用，以及新 provenance 继续传播；这不证明真实 LLM 质量不变。
- 相似度检查使用原标量公式作独立对照，比较浮点值及阈值边界、并列、方向和重复行选择。

下一步仅在另行批准后跑同128条、69题及52题 judge 的服务器对照。检查维护耗时和错误合并，不以调用减少代替性能或质量结论，不直接扩大到完整 Sample 0。
