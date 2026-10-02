# Zep 组合物理执行（128-event 验证）

这是独立的 `physical_fusion="zep-combined"` 条件，默认关闭。
目标是逼近同模型 Native 的维护工作量，不是宣称相同模型逐次回答等价。
原 view、抽取 instruction、候选筛选、retrieval、评分不变。

## 执行范围

- 实体的同一实体分组判断以及完整有序候选集的匹配决策可复用。
  状态由 executor 持有并随 snapshot 保存；只保存布尔选择，不保存旧行。
  候选、顺序、语义输入或模型配置变化会失效。
- 实体与 fact 共用 target-state fusion：匹配目标和生成当前状态一起执行。
  实体的新贡献仍先生成临时摘要；多个贡献匹配同一目标时共同生成一次状态。
  已复用的匹配不会跳过当前摘要生成。ID、mentions、provenance 仍走原确定性逻辑。
- 只在注册的新 fact canonicalization 节点，对恰有一行的字符串 fact/relation_type
  直接保留输入。多行组仍调用原 aggregate，实体摘要没有这条捷径。
  这是需要真实质量验证的近似策略，不是任意 semantic aggregation 的恒等律。
- 矛盾判断按原 later-added fact 身份分组，每份 prompt 最多四对。
  新 fact 的文本只放一次，各旧候选保留独立任务 ID 和 boolean。
  跨组调度、候选筛选和解析不变。

## 可核对证据

物理改写保留原候选筛选节点身份，避免 subtree 改写导致 profile 未命中。
Trace 分别记录 identity reuse、singleton passthrough、fusion 结果和真实 provider 调用。
新版本指纹与旧实验不同，禁止把旧状态当作新条件恢复。
现有 fusion 保留最少一次格式重试，manifest 单列该事实；不把 provider retries=0
误写成所有层级都零重试。

实验只运行原128条输入一次。质量题由原 benchmark 中非空且全部 evidence IDs
落在此前缀的题确定性选出，不按输出挑题。这不是完整 Sample 0 accuracy。
如无同问题集且同输入范围的 Native/未优化答案，不声称质量门禁已经通过。
恢复、失败调用、trace I/O 和 provider cache 与最终成功路径分开统计。

本地仅做离线验证；真实模型只在服务器新目录执行，不触碰 Native 实验。
