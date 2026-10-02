# Representative fact matching 的节点级打包

本轮只改变物理请求组织，不改变 representative view、匹配 instruction、候选筛选或每个新 fact 的候选域。

- `LotusExecutionConfig.listwise_join_batching` 按 semantic site 选择 top-k listwise join，默认关闭；不打开全局 batching。
- 同一次节点执行中的多个新 fact 共用一份匹配指令，每个任务保留自己的候选列表和 ID，最多 16 个任务一份 prompt。不同 insertion 不合并、不延迟维护。
- 复用现有 prompt batching 执行、严格解析、trace 和重试。漏项、重复 ID 和跨任务候选均报错，不默认当作未匹配。
- 配置进入执行 fingerprint；本条件从空状态开始，只能恢复相同配置的 checkpoint。
- 其余保持 Top5、contradiction packed16、entity fusion、groupby32、aggregate16、fact-based summary、delta 优化、C=64、缓存关闭。并不恢复 fact 文本生成。

现有启动脚本增加 `--listwise-batch-size 16`，仅用于固定的 representative 128-event 条件。预检从实际物理计划定位唯一 fact matching site，找不到或找到多个即停止。实验保持原 69 题和 judge 集合；在服务器新目录执行，使用已有 `run/resume/status`。

离线测试验证 17 个独立任务打包为 16+1、候选隔离、重复行、空输入、其他节点不变、严格错误处理，以及真实 planner/runtime 的增量结果和恢复。Fake provider 只能证明接线；真实语义质量、tokens 和 latency 必须由这轮实验回答。若节点一次只有一个新 fact，batch16 不会凭空减少调用。
