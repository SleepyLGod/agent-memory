# Zep fusion smoke：回答格式修复与下一步边界

## 已验证的事实

实验 `zep-target-state-fusion-20260916T151905Z`：同一固定 24-event bundle，
DeepSeek Flash、逐条维护、相同 embedding 候选筛选、LOTUS cache 关闭。
未融合先跑，融合后跑；只有一道问题，没有 Native 新对照。

| 维护指标 | 未融合 | 融合 |
|---|---:|---:|
| Mean insert wall time (s/message) | 22.831 | 15.392 |
| Mean insert tokens/message | 11,313.625 | 11,264.708 |
| Provider calls | 594 | 574 |
| 最终 facts | 31 | 33 |

调用减少不代表同比例 token 节省。抽取阶段没有被融合，但 provider 累计耗时
从 191.384s 变为 75.668s；不能把整轮耗时下降全部归因于 fusion。
不同运行的抽取和状态也有差异，不能把 operator 计数差直接当作同输入消融。

## Answerer：不是 fusion 引入的问题

两组真实回答均照抄了 JSON Schema，没有回答 Caroline 的身份问题。
旧共享 parser 对缺少 `answer` 的 JSON 返回原文；此前测试明确要求 schema echo
进入评分并记零分。这一行为已存在于 fusion 分支的基线，不是新优化产生的。
这证明缺陷早已存在，不证明每一次历史实验都实际触发过。

修复保留普通纯文本兼容，但对 JSON/代码围栏形式的回答严格检查：必须是合法
对象，并包含非空字符串 `answer`。错误结构不再伪装成正常答案；runner 使用
已有有限重试机制，只重试当前回答步骤，不重做维护或 retrieval。
回答 prompt 明确要求返回 schema 的实例，而不是 schema 本身。
parser 身份升级为 v2，回答 prompt 指纹更新，judge 和评分公式不变。

旧实验的两个零分仍保留，不回写 artifacts。它们不能支持融合前后的 memory
质量比较。若补测，可用保存的 retrieval 单独调用新 answerer，并标为新评估条件；
不能声称与旧 Native prompt 字节相同。

## 融合边界

```text
抽取 entities / facts
  -> entity 匹配与维护
  -> fact 初始归组与 canonicalization
  -> 匹配旧 fact + 生成目标 canonical fact  [当前融合]
  -> 用更新后的 facts 构造矛盾候选
  -> 逐 pair 判断矛盾
  -> 确定性计算失效时间、保存图
```

Native Graphiti 的 `resolve_extracted_edge` 在一次请求中返回 duplicate IDs
与 contradicted IDs；重复时选择已有 edge，并不等同于我们的 canonical 文本重写。
当前实现不能称为 Native 等价 fusion。

| 阶段 | 融合组 calls | tokens | 下一步边界 |
|---|---:|---:|---|
| 抽取 sem_flat_map | 48 | 114,154 | 不纳入本轮 target-state fusion |
| 聚合 agg | 122 | 63,327 | 含不同实体/事实阶段，不能整体删除 |
| 矛盾 sem_filter | 303 | 59,871 | 优先考察同一新 fact 的候选集联合判断 |
| 剩余 sem_join | 44 | 12,374 | 保留实体维护等独立任务 |
| sem_groupby | 44 | 6,997 | 初始归组仍然存在 |
| fused target-state | 13 | 13,630 | 已实现并真实执行 |

**建议先做更窄、可检验的候选集联合判断，再决定是否跨阶段融合。**
同一新 fact 的矛盾候选可共享上下文，但必须保留原候选集合、完整 predicate、
方向与每个 pair 的结果身份；空集不调用，失败不能默认当作无矛盾。
这属于物理执行优化，不改变 logical query 或时间规则。

进一步合并“目标选择、canonical 文本、矛盾判断”需要独立设计：现有矛盾候选
筛选依赖 canonicalization 后的文本和状态。如果提前筛选原始 fact，候选可能
不同；如果把所有旧事实放入大 prompt，又改变成本和上下文。不能在没有候选
等价证据时宣称这是纯粹保语义 fusion。单次联合回答也可能改变判断，必须真实验证。

本轮只修复 answerer 并分析边界，没有实现第二条 fusion，也没有发起新的模型调用。
证据位于服务器实验目录的 `metrics/operation_usage.csv`、`provider_usage.csv`、
`answers.jsonl` 及对应 prompt/raw-response trace；源文件与历史结果保持不变。
