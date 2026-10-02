# Zep 撤回定位、Top5 与日期输入修复

日期：2026-09-20。范围：隔离 fusion 工作树，代码和离线验证；没有调用模型或启动服务器实验。

## 修复了什么

| 问题 | 修复 | 边界 |
| --- | --- | --- |
| 撤回少量 pair 时，反复对整张旧表计算完整行键 | Executor 维护行键到 occurrence 的索引，只定位和更新受影响条目 | 保留方向、重复次数和最早的相同行；不是取消大中间表 |
| Top5 在 replacement 的局部片段里重选，让原先被筛掉的旧 pair 入选 | 仅 metadata 变化时携带原筛选成员资格；语义输入变化时重选完整的受影响候选桶 | 新候选可能挤出旧候选；不能只追加新结果。对称筛选暂时保守刷新完整域 |
| 不同月份的相似事件只凭 “last Friday / last week” 匹配 | Representative fact grouping 及生成的增量匹配均加入已有的 `valid_at` | 这是经批准的语义输入合同变更，不是纯性能优化；未知时间不能证明同一事件 |

前两项属于可复用的运行时处理，不按 Zep 文本或具体人物打补丁。日期 instruction 只用于 `ZepRepresentativeMemory`，未修改其他 memory 变体。代表选择、输出 schema、确定性来源/时间聚合保持原规则。

Top5 的要点：如果完整范围中 A 是 G 的第六名，那么仅给 A 增加来源，不应该在只剩 A 的局部片段里重新把它选为第一名。反过来，真正加入更好的候选 H，则必须考虑它是否把当前第五名挤出。修复同时覆盖这两种情况。

## 已验证的结果

- 真实 planner/runtime 加确定性 embedding/predicate oracle：三种筛选方向的来源更新都不增加模型或 embedding 调用，结果与同配置完整执行的 bag 一致。
- 覆盖候选挤出与晋升、文本变化、重复 occurrence、空集与重现；失败不提交判定、成员资格或连接索引。
- 关系连接随机 bag 回归覆盖 indexed/scan 两条路径、NULL、嵌套单元格、双侧更新；恢复后索引重建正确。
- 日期同时出现在声明式分组与增量匹配左右两侧的输入合同中。没有真实模型实验，不能声称误合并已经消失或 accuracy 已提高。
- 全量离线测试：1,888 passed、20 skipped。相关 Ruff、生产代码及新回归测试 Pyright 通过；已有 LOTUS 弃用及模拟成本计算警告仍存在。

### 撤回定位局部测量

使用已保存的真实 fact 行，只将第一条的 provenance 加入合成来源；不调用模型。每个规模交替运行 scan/indexed 三次，表中为中位数。计时包括连接 delta 构造、索引副本及结果组装，不包括 oracle bag 验证。

| Facts | 已存 pairs | 本步替换 pairs | 原扫描路径 | 已建索引路径 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8,128 | 127 | 0.132 s | 0.019 s |
| 256 | 32,640 | 255 | 0.573 s | 0.042 s |
| 574 | 164,451 | 573 | 2.372 s | 0.126 s |

这不是端到端或服务器 latency。574-fact 状态一次性建索引耗时约 4.78 s；正常执行中随提交增量更新，恢复后首次使用时重建。索引增加内存，字典浅拷贝和 DataFrame 组装仍随状态规模增长，没有解决全部二次规模的中间状态。

## 恢复和下一轮配置

- 索引是派生状态，不增加 snapshot 格式；失败只丢弃本次暂存索引，恢复后从已提交关系重建。
- 有界候选筛选的执行 fingerprint 加入新版本，拒绝旧语义的 checkpoint。日期输入也改变 representative query/site 身份。
- 下一轮必须从空状态开始，并针对新计划重新生成 fact grouping/matching 的节点绑定；旧 digest 不能照抄。预检明确拒绝未命中的绑定。
- 前一轮已经补齐 caption 输入，但 caption、日期上下文及 Top5 正确性尚未经过新的端到端实测。旧实验的低分不能直接当成这些修复后的分数。

## 现在是否需要大 Resolver

暂不扩大融合。先前开销混合了不必要的整表键扫描、错误的局部 Top5 重选，以及真正需要的模型工作；先修前两类，比把错误工作包进更大的 prompt 更清楚。

这些修复也没有消除所有结构性差距：实体和 fact 的分组、匹配、矛盾判断仍有串行依赖，Native 把部分判断放在一次 resolver 中。只有修复后剩余关键路径仍主要由这些串行等待占据，才有证据继续设计联合 resolver。届时仍要保护代表事实、候选和时间合同，不能为了少一次调用改成判断另一个对象。

下一步应在服务器固定输入下测量正确候选域、实际新判断数、各阶段等待及同集合答案质量；不先提高并发，也不承诺这轮已经追平 Native。真正改写事实可能需要刷新候选桶，对称筛选甚至可能增加本地筛选量；不能把正确性修复等同于所有 workload 下的性能提升。

实现入口：`runtime/executor.py`、`adapters/lotus/adapter.py`、`memories/zep/representative.py`。聚焦回归：`tests/test_replacement_screening.py`、`tests/test_maintenance_work.py`、`tests/test_relational_join_ivm.py`、`tests/test_zep_representative.py`。

## 后续复查：混合更新

补充复现：同一次输入给旧事实 A 增加来源，同时插入 H。原先的来源复用要求整批都是 metadata 更新，因此混合输入会让旧候选组也重新筛选。固定八条合成事实的回归中，实际重新筛选了 28 个 pairs，而新增 H 对应的范围只有 7 个。

修复只将复用边界缩小到独立的、单向 Top5 候选组：某组本次只有来源更新，就传递原成员资格及当前 metadata；组内有新候选、文本或身份变化，仍完整重选。对称筛选不能按一个端点独立处理，保留原来的保守范围。

修复后该例筛选 7 个 pairs，固定 oracle 下与全量结果一致。这是候选筛选量下降，不是 75% 的模型调用、tokens 或 latency 降幅：此前 predicate reuse 已能复用部分模型判断。

另外检查实际发给 fake provider 的最终请求，确认 Full 分组包含两个不同月份的日期，增量 listwise 匹配同时包含左右两侧日期；不是只检查逻辑计划中的字段名。本轮未再次修改 view 或 prompt，没有新增优化框架或 resolver。执行 fingerprint 更新至 `complete-candidate-topk-v2`。
