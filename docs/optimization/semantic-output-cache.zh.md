# 语义输出缓存：共享空表与非空索引

## 本轮改了什么

只修改通用 executor 的输出记账，不修改 view、IVM rules、Top5、模型请求、prompt、batching、并发、retrieval 或 scorer。

原实现为每个没有输出的 occurrence 保存独立空 DataFrame，并在组装输出时遍历全部记录。现在保留全部处理记录，但共享兼容的空表；另外维护非空输出索引，组装时只访问非空项。

- `RowOutputCache` 是 runtime 内部结构，没有新公共 API、配置或缓存服务。
- 列、dtype、索引结构及名称、duplicate-label 标志必须兼容才能共享。带任意非空 `attrs` 的空表保守保持独立，避免错误合并 metadata。
- 空表模板仅供读取。普通执行与判定复用使用同一记账入口；替换输出而非原地修改共享模板。
- 保存原 mapping 的插入次序；空转非空不会自行移到最后，删除后重插才改变位置。方向、多输出与重复次数不变。
- 每次维护复制记账索引，成功后一起提交。失败不能污染已提交状态。
- Snapshot 仍是 schema-v2、普通字典与 DataFrame；不持久化新索引。旧快照恢复时重建索引并压缩兼容空表，保留原 fingerprint 校验。没有放宽 benchmark 的源码身份恢复合同。

本轮没有消除完整 pair 中间表，也没有消除暂存字典的线性复制；不宣称所有本地开销都已解决。

## 离线验证

新增测试先在原实现上复现：300 个拒绝 occurrence 对应 300 个独立空表，未达到共享要求。

修复后的测试覆盖：多次输入与恢复、旧快照独立空表压缩、空/非空双向切换、删除后重现、输出次序、多输出及 NULL、schema 差异、metadata、派生索引隔离，以及真实 planner/runtime 下的候选替换与判定复用。

Full 和 IVM 原本不保证行序相同。测试分别验证 Full/IVM 的 bag 与 metadata 一致，以及新旧维护记账的输出次序一致，不强行要求与 Full 的生成次序相同。

- 最终全仓 pytest（网络调用被阻止）：**1933 passed, 20 skipped**。
- Ruff：executor、新缓存模块和新增测试通过。
- Pyright：同三个文件 **0 errors, 0 warnings**，显式使用该工作树的 `.venv/bin/python`。
- 两仓库 `git diff --check` 通过。
- 测试有现有 Pydantic deprecated config 警告，以及 fake LM 用量测试中的 LOTUS 费用计算警告；没有将 fake 费用当成真实账单。

## 服务器可信快照微测试

读取已有 `zep-parallel-cache-s0-20260919T185314Z` 的108条与419条快照。通过 SSH 将本次实际缓存模块送入临时 Python 进程，阻止网络连接；只改变内存中的状态副本，不安装新环境、不改历史文件、不调用模型。

转换前后逐 occurrence 核对 key 顺序、空/非空状态、空表 schema，非空输出用 `assert_frame_equal` 验证内容、dtype、索引和顺序。

| 指标 | 108条：之前 → 之后 | 419条：之前 → 之后 |
|---|---:|---:|
| 矛盾节点 occurrence 数 | 13,861 → 13,861 | 185,745 → 185,745 |
| 该节点无输出记录数 | 13,813 → 13,813 | 185,647 → 185,647 |
| 该节点不同 DataFrame 对象数 | 11,348 → 52 | 154,083 → 102 |
| 所有缓存非空输出块数 | 264 → 264 | 932 → 932 |
| 枚举非空输出，中位时间 | 17.86ms → 0.063ms | 204.83ms → 0.169ms |
| 一次全代 GC，中位时间 | 133.28ms → 18.50ms | 1,209.23ms → 36.51ms |
| Python GC 跟踪对象数 | 296,774 → 79,022 | 3,183,042 → 130,876 |
| 同协议重编码 snapshot 字节数 | 7,943,096 → 4,037,637 | 86,530,505 → 33,357,425 |

时间均取三次测量中位数。419条转换后首次枚举为7.58ms，后两次为0.169ms与0.162ms；不隐藏首次成本。

**这些不是端到端 latency。** 枚举计时不包含 `pd.concat` 或其他维护步骤；GC 为手动触发，不知道历史运行实际在何时触发。完整转换加逐项核验耗时分别3.81秒与46.67秒，是离线检查成本，不冒充生产恢复时间。重编码体积使用同一 pickle 协议比较，不是修改了旧文件，也不是进程 RSS；峰值 RSS 不会因释放对象下降，未将其当作当前内存改善指标。

快照 SHA-256：

- 108条：`255092dc2817d66bdea298a0b45a6074a113175f1e76a7a7361156ad7e79b129`
- 419条：`e9b870855fae33351a690627065871eae132081deb588a060d7487166cd4a001`

服务器原路径前缀：
`/mnt/data/agent-memory-experiments/zep-parallel-cache-s0-20260919T185314Z/combined/cases/conv-26-2aac22fc/checkpoints/snapshots/`

本地微测试脚本：`/private/tmp/zep-row-cache-benchmark.py`；原始结果：`/private/tmp/zep-row-cache-108.json`、`/private/tmp/zep-row-cache-419.json`。脚本从本工作树读取实际模块，不是独立重写缓存算法。

## 保留的质量回归材料：本轮未修

以下用例来自同一历史运行的 extraction、retrieval 和 answer traces，只用于后续定位，不进入被测 prompt，也不是本轮准确率改善证据。

| 问题身份 | 已观察到的失败 | 后续应检查的层次 |
|---|---|---|
| `conv-26:q64`，才艺表演月份 | D15:11 实体抽取只有 Caroline；表演事实为 self-loop，被禁止自连接的规则过滤；gold 为2023年9月。 | 实体覆盖与图表示合同，不能直接放开所有 self-loop。 |
| `conv-26:q9`，学校演讲日期 | D3:1 说 last week，抽取已经写成6月9日；gold 是6月9日前一周。 | 抽取的相对时间解析，不是 runtime 将正确日期改坏。 |
| `conv-26:q50`，Pride fest 年份 | D12:15 的 last year 被保留，但 valid_at=null；事实已检索到，仍答无信息；gold 为2022。 | 时间解析和相对时间锚点。 |
| `conv-26:q59`，陶艺盘子日期 | 实际 answer prompt 有8月24日 event_time，回答却为8月23日。 | Answerer，正确信息已经送达。 |
| `conv-26:q63`，公园日期 | 实际 answer prompt 有8月27日 event_time，回答却为8月26日。 | Answerer，不能归为维护丢失。 |

证据文件为该运行 `combined/trace/events.jsonl` 引用的原始请求与回答，以及同 case 的 `retrieval.jsonl`、`answers.jsonl`、`grades.jsonl`。本轮不调整时间格式、实体 prompt、日期规则或回答 prompt，不声称这些质量问题已经解决。

## 后续门禁

离线结果支持将此修复带入下一次同配置服务器实验，但本轮未启动付费调用。下一次仍须单独报告完整插入 latency、tokens、缓存命中、恢复开销和同集合质量，不将本微测试倍数当作全量加速比。

修改仅留在隔离 fusion 工作树，既有 staged 内容与 index 保持不变；原 Replica、历史实验和配置不变。没有 stash、commit 或 push。
