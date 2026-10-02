# Zep: 独立 fact 抽取与实体维护并行

## 范围

默认关闭，只在 `zep-representative` 上开启 `parallel_fact_extraction=True`。
不改 view、prompt、候选筛选或结果含义，不增加 resolver。

```text
抽取实体名称和本条消息内的编号
    +-- 抽取 facts ----------------+
    +-- 实体分组、匹配、确定 ID -----+  两条分支并行
                                  |
                         将 facts 绑定到实体 ID
                                  |
                         原有 fact 维护和摘要
```

编译后的 fact 抽取输入只有原消息、上下文和原始实体列表。
实体 ID 的绑定在后面的关系 join，因此这两条分支没有前后依赖。
注册时校验完整上游编译图；不按 prompt 子串猜测节点。

## 实现边界

- 沿用一个事件循环，只允许一个指定的 row-local `sem_flat_map` 后台执行。
- 主线程继续执行已就绪节点；依赖后台结果的节点等它完成后才执行。
- 后台仅返回抽取结果。executor 缓存、occurrence、节点状态和存储提交都在主线程更新。
- 两条分支使用独立 LM 对象，配置相同，不切换 LOTUS 全局设置。
- 复用原格式、解析、重试及 trace；保留事件上下文，后台 trace 标记执行分支。
- 任一分支报错，本条事件不提交；等待已发出的后台调用退出，不自动重发。
- 不跨事件并发，不增加候选，不改变 batching 或筛选参数。

要求显式指定 `lm_enable_cache=True/False`，且无全局 prompt batching。
现有节点 batching 不受影响。开启缓存时，两条分支各用原生 LOTUS 内存缓存
（最多 1024 项），同时启用 LM 回答缓存和 operator cache。设置一次上下文局部
LM 路由，让 LOTUS 原生 operator cache 读取对应分支的模型、缓存和计数；
不在执行中反复切换全局模型，不复制缓存实现。两路分别记录命中和用量。
缓存不持久化，resume 后冷启动；不保证命中率或避免未提交事件的付费重放。
provider cache 不变，不能把这两类本地缓存说成 provider prompt cache 开关。
物理策略进入执行指纹；新配置不能恢复旧串行配置的 checkpoint。
同配置继续使用原恢复流程，不能据此保证未提交事件的调用绝不重放。

## 使用

现有 `run_agent_memory_bundle` 接受 `parallel_fact_extraction=True`。
现有 `tools/zep_combined_smoke.py` 的 preflight、run、resume 都增加
`--parallel-fact-extraction`，其他已冻结参数保持相同。
同时开启两类 LOTUS 缓存时传 `--lotus-cache-mode memory`，默认仍为 disabled。
manifest、条件身份和恢复校验记录该开关；关闭时不改变原身份。

## 验证与计时

离线使用同步屏障证明实际重叠，而不是用不稳定的 sleep 测速。
覆盖原始顺序、重复行、失败不提交、后台退出、snapshot/restore、
独立 LM 的响应 metadata、用量与 trace 归属，以及 runner 配置传递。

后台 trace I/O 单列，不能把它与主线程 I/O 相加后从 wall time 重复扣除。
真实并行实验应以原始 insertion wall time 为主要延迟指标；旧的扣 trace
指标不是并行关键路径耗时的精确估计。模型时间也应按区间并集统计。

本轮只做离线验证。历史 128 条估算约有 90 秒等待可重叠，不是实测收益，
也不承诺质量、费用或延迟一定追平 Native。
