# Zep：减少重复维护工作

本轮只改物理执行，不改 Zep view query、instruction、候选筛选、retrieval 或评分。没有运行真实模型或服务器实验。

## 三项改动

| 改动 | 现在如何执行 | 边界 |
| --- | --- | --- |
| Predicate 判定复用 | 原筛选完成后，按完整 instruction 和其引用列的实际文本上下文复用 boolean；输出使用当前行及 metadata | 可选，第一版通过 pair-shaped sem_filter site 注册；依赖无法确定时走原路径 |
| Inner join replacement | 根据两侧实际撤回与新增构造受影响连接，按 bag 更新旧结果 | 支持不等式 self-join；仍扫描旧输入及旧结果，不声称只需 O(delta) 工作 |
| 跨组调度 | 每个 later-added fact 保持独立 prompt 分组，最多四对；各组请求一次交给已有调度器 | 不跨组 packing，不新增线程池，也不提高并发配置 |

判定复用保存的是布尔值，不是过期整行；重复行与方向保持，provenance-only 更新仍向下游传播。状态随 executor snapshot 保存，配置和物理版本参与恢复校验。默认关闭，不是 LOTUS cache 或 provider cache。它依赖固定 predicate 假设，不能保证与真实模型反复重采样的结果相同。

关系 join 的撤回计算为 `join(deleted_left, old_right)` 与 `join(surviving_left, deleted_right)`；新增计算为 `join(inserted_left, new_right)` 与 `join(surviving_left, inserted_right)`。这些范围不重复计数，不重新构造无关旧—旧 pairs。

## 配置核对

现有 `tools/zep_combined_smoke.py` 新增可选 `--maintenance-work`，组合启用 predicate reuse、groupby32、aggregate4，并保留 target-state fusion 与 contradiction prompt4。默认条件不变。筛选配置继续读取原 site-profiles 文件，没有改变阈值或 top-k。

groupby32、aggregate4 与 fusion 的组合已通过离线 prepare/runtime 回归。配置上限不等于实际每次能凑满批次，后续需看真实任务分布。历史条件所称的 state narrowing 尚未找到可逐项对应的独立配置，不能宣称已严格复现旧128全部优化；新条件启动前仍需核对历史源码与 manifest。此次没有为追求对齐而擅自删减状态字段。

脚本以 `--prefix128 --maintenance-work` 接受旧实验冻结的128-event、1-question bundle，不截取或重新挑选输入。

2026-09-17核对补充：state narrowing 是编译器实现，不是独立配置。旧128服务器源码的 rules.py 与当前源码都在 mixed aggregate、grouped semantic aggregate 的 state remerge 中使用 `input_cols=state_cols`。该优化已保留，前述证据缺口关闭。旧模型标识为 deepseek-v4-flash，新条件沿用当前 deepseek-flash；结果只能作为历史参考而非严格单因素消融。

## 离线证据

- 完整测试：1,791 passed、20 skipped。定向 Pyright 无错误；没有调用真实 endpoint。
- 40行状态的不等式 self-join replacement 与完整重算 bag 一致，单次受影响连接结果不超过80行，而完整结果为780行。
- 历史冻结的9个候选、3个分组，使用原回答作确定性 replay：metadata 更新后，关闭复用累计18 tasks/6 prompts；开启复用累计9 tasks/3 prompts；结果 bag 相同。历史回答不是人工 gold，此测试不证明准确率。
- 跨组测试从分组串行的两次模型调度变为一次提交三个独立 prompts；没有改变分组内容。这不是实测服务器延迟收益。

## 下一步

先核对旧128与新条件的输入、筛选及状态配置，再批准同128前缀的真实对照。分别检查重复判断、受影响pair规模、实际batch分布、tokens、维护耗时和质量。batch16暂不启用；不得把离线减少的调用直接换算成实测加速或保证达到Native。
