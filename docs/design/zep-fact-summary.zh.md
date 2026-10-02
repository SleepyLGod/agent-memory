# Zep fact-derived summary：独立实验条件

目标是减少不必要的摘要生成，不保证追平 Native。原 `ZepMemory` 保留。

## 依赖与合同

新 `ZepFactSummaryMemory`：实体身份（只规范 name）→ facts → 按两个端点关联的实体摘要。
facts 不再依赖实体摘要；原始消息不能直接作为某个实体的事实追加。无关联 fact 的实体摘要为空。
实体 ID、mentions、事实抽取及矛盾判断复用原逻辑。检索方法和输出字段不变，但摘要内容改变，因此必须重新测质量。

`physical_fusion="zep-fact-summary"` 使用现有 join-map/runtime：

- 实体融合只返回 canonical name，不生成临时或合并摘要；单行 name 直接保留。
- 摘要先拼接已维护摘要与新增关联 facts；不超过 2000 字符直接保存。
- 超过 2000 字符才调用一次现有 aggregate 执行器，直接要求压缩；不再让模型判断长度阈值。v2 对超过 1000 字符的有效回答采用 Native 的句界裁剪策略，无句界时截取前 1000 字符，不为长度额外重试。保留原回答、裁剪结果和前后长度；格式错误、空白输出仍失败。裁剪可能损失尾部信息，不宣称无损；不修改底层 facts。
- 关联两端，同一端点不重复贡献；不同事实的相同文本不擅自去重。
- 当前选择包括保留在 facts 中的历史/失效事实，未承诺删除事实会让历史摘要彻底遗忘。
- 输入撤回沿用已有 runtime 重算回退；不宣称这是 Native 的完整摘要实现或严格全量/增量等价。

新旧 query 和执行 fingerprint 不兼容，必须从空状态跑新条件。压缩有信息损失，质量尚未真实验证。

## 已有条件的小修

`zep-combined-v2` 的固定匹配仅传所选目标状态。未确定匹配仍保留所有筛选后的候选。
匹配缓存键仍包含原候选集合，不能通过裁剪改变缓存身份；旧状态仍由原结果组装保留。

## 实验入口

已有 `tools/zep_combined_smoke.py` 增加 `--fact-summary`，与 `--combined-physical --maintenance-work --prefix128` 配合。
新目录 preflight 后才能 run；需要针对新query核对并绑定 semantic site profiles，不能直接重用旧site IDs或恢复包。
本轮只做离线验证，不调用付费模型、不启动服务器实验。下一轮记录摘要直通/压缩次数、整体 tokens/latency 和相同69题质量。
