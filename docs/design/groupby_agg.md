# `group_by` / `sem_groupby` aggregate API contract and rules

这份文档分三部分：

- 第一部分定义 grouped aggregate API contract。
- 第二部分记录 paper-wise differential rules，不讨论 implementation-only 的
  `input_cols` 细节。
- 第三部分记录 implementation-wise differential rules，说明实际 lowering 里
  `sem_agg(input_cols=...)`、`collect_list(...)`、`flatten(...)`、`min(...)`
  和 `least(...)` 带来的 rule 变体。

这里的 rules 分成 paper-wise 和 implementation-wise 两层。本文档不承诺当前
implementation 已经支持所有 rules，也不决定 optimizer 如何选择 rule。

## 1. API contract

### 1.1 Relational `group_by(...).agg(...)`

在关系代数、SQL 和主流 DataFrame API 里，`group_by(keys).agg(...)` 的基本
语义是：

```text
many rows per key -> one row per key
output schema = deterministic key columns + aggregate output columns
```

`keys` 不是 aggregate function 的输出。它们是 group identity，group 之后仍然
保留。

例如：

```text
orders(user_id, amount)
  .group_by("user_id")
  .agg(sum("amount") -> "total_amount")

output: user_id, total_amount
```

这条规则对 agent-memory 的 deterministic `group_by` 同样成立。

deterministic key 和 aggregate output 同名时，不存在适用于所有 grouped
aggregate 的通用覆盖规则。具体行为由 closing aggregate form 决定，见后续各节。

### 1.2 Standalone aggregate operators

#### `sem_agg`

`sem_agg(input_cols, output_cols, instruction)` 是 semantic aggregate。单独使用时，
它把输入 relation 的多行变成一个 semantic output row。

```text
input rows -> one output row
output schema = output_cols
```

如果 `input_cols=None`，operator 可以读取可见输入列。若显式给出
`input_cols`，不在 `input_cols` 中的列不参与 aggregation。

`sem_agg` 不会自动保留原始输入列，也不会自动保留 grouping key。

#### `array_agg`

`array_agg(columns, output_col)` 是 deterministic evidence aggregate。它把多行
收集成一个 JSON array-of-records column。

```text
input rows -> one output row with output_col
```

`array_agg` 只保存 evidence rows。它不选择 canonical name，不生成 semantic
identity，也不解释哪些 semantic keys 应该代表一个 group。

#### `collect_list`

`collect_list(column, output_col)` 是 deterministic value-list aggregate。它把同
一组内某个 column 的值收集成一个 JSON list。

```text
input rows -> one output row with output_col
```

它和 `array_agg(columns, output_col)` 的区别是：`array_agg` 收集 row records；
`collect_list` 收集一个 column 的 values。implementation-wise rules 用它来收集
已经存在的 array aggregate state，例如把多行 `evidence` state 收成
`[evidence_state_1, evidence_state_2, ...]`。

#### `flatten`

`flatten(column, output_col=None)` 是 deterministic array-state operator。它把
一行里的 JSON array-of-arrays 展平成一个 JSON array。它不是 `explode`，不会把
array 展开成多行。

```text
input:  column = [[a, b], [c]]
output: column = [a, b, c]
```

如果 `output_col=None`，`flatten` 原地覆盖同名 column；否则写入
`output_col`。后者必须是不存在的新 column，不能静默覆盖其他已有 column。

#### `min` 和 `least`

`min(column, output_col)` 是 deterministic aggregate：它在多行中取一个 column
的最小非 null 值。`min(columns=[c1, c2, ...], output_col)` 则按 column 声明顺序
取字典序最小 tuple。复合 `min` 会忽略任一 component 为 null 的 row；没有完整
tuple 时输出 null。global empty input 和 all-null group 输出 null；grouped empty
input 输出零组。

`least(a, b, ...)` 不是 aggregate。它是 `assign(...)` 可使用的 row expression，
在同一行的两个或多个值中取最小非 null 值。全部为 null 时输出 null。

直接调用 `group_by(keys).min(..., output_col=o)` 时，`o` 不能和 deterministic
keys 重名；重名会直接报错。作为 mixed `group_by(keys).agg(...)` 中的 aggregate
spec 使用时，则遵循该 mixed `.agg(...)` 的 key precedence contract。

### 1.3 `sem_groupby(keys).sem_agg(...)`

这里的 `keys` 指 `sem_groupby(input_cols=keys, ...)` 里的 semantic keys。它们是
判断哪些 rows 属于同一 semantic group 的依据，不是自动输出列。

Contract:

```text
output schema = sem_agg.output_cols
required: semantic keys ⊆ sem_agg.output_cols
```

也就是说，`sem_groupby` 不会自己保留 semantic keys。如果这些 keys 应该出现在
最终 view 里，必须由后面的 `sem_agg` 生成。

例子：

```python
entities = (
    extracted_entities
    .sem_groupby(
        input_cols=["name", "entity_type"],
        instruction="Rows refer to the same real-world entity.",
    )
    .sem_agg(
        input_cols=["name", "entity_type", "episode_content"],
        output_cols={
            "name": "Canonical entity name.",
            "entity_type": "Canonical entity type.",
            "summary": "Entity summary.",
        },
        instruction="Merge entity mentions into one canonical entity row.",
    )
)
```

这里 `name` 和 `entity_type` 是 `sem_agg` 产出的 canonical fields，不是
`sem_groupby` 自动保留下来的原始值。

### 1.4 `group_by(keys).sem_agg(...)`

`group_by` 是 deterministic grouping。它的 keys 必须保留。

Contract:

```text
output schema = deterministic keys + sem_agg.output_cols
```

这个 operator form 允许 `sem_agg.output_cols` 和 deterministic keys 重名，并以
deterministic keys 为准。`sem_agg.input_cols` 可以包含 keys，也可以只包含非 key
columns。

不在 deterministic keys 或 `sem_agg.input_cols` 中的输入列，不参与 `sem_agg`。

例子：

```python
daily_summary = (
    messages
    .group_by(["user_id", "day"])
    .sem_agg(
        input_cols=["message"],
        output_cols={"summary": "Summary of messages for this user on this day."},
        instruction="Summarize the messages.",
    )
)
```

输出 schema：

```text
user_id, day, summary
```

### 1.5 `sem_groupby(keys).array_agg(...)`

暂不支持。

原因是 `array_agg` 只能保存 evidence rows，不能决定 semantic group 的 canonical
key。对于：

```text
name = "Caroline"
name = "caroline"
name = "Carol"
```

如果 `sem_groupby(input_cols=["name"])` 把这些 rows 分到同一组，`array_agg` 无法
决定输出里的 canonical `name` 应该是什么。

如果 policy 需要 semantic keys 出现在输出里，应该使用
`sem_groupby(...).sem_agg(...)`，或未来的 `sem_groupby(...).agg(...)` 中的
`sem_agg` aggregate spec。

### 1.6 `group_by(keys).array_agg(...)`

支持。

Contract:

```text
output schema = deterministic keys + output_col
```

`output_col` 不能和 deterministic keys 重名。

例子：

```python
episode_entities = (
    entity_mentions
    .group_by("episode_id")
    .array_agg(
        columns=["entity_id", "name", "summary"],
        output_col="entities",
    )
)
```

输出 schema：

```text
episode_id, entities
```

### 1.7 `group_by(keys).agg(...)`

`agg(...)` 用来在同一个 deterministic grouping partition 上声明多个 aggregate
functions。

Contract:

```text
output schema = deterministic keys + all aggregate output columns
```

Rules:

- deterministic keys 自动保留。
- aggregate outputs 之间不能重名。
- aggregate output 如果和 deterministic key 重名，以 deterministic key 为准。
  这是 mixed `.agg(...)` 的 contract，和 direct `array_agg(...)`、`min(...)`
  的 collision rejection 不同；这种写法仍不推荐。
- `am.min(column=..., output_col=...)` 或
  `am.min(columns=[...], output_col=...)` 可以和 `sem_agg`、`array_agg`、
  `collect_list` 一起作为 aggregate spec。

例子：

```python
episode_entities = (
    entity_mentions
    .group_by("episode_id")
    .agg(
        array_agg(columns=["entity_id", "name"], output_col="entities"),
        sem_agg(
            input_cols=["name", "summary"],
            output_cols={"episode_entity_summary": "Summary of entities in this episode."},
            instruction="Summarize the entities mentioned in this episode.",
        ),
    )
)
```

输出 schema：

```text
episode_id, entities, episode_entity_summary
```

### 1.8 `sem_groupby(keys).agg(...)`

`agg(...)` 用来在同一个 semantic grouping partition 上声明多个 aggregate
functions。

当前 contract 允许 `sem_agg` 和 `array_agg`、`collect_list`、`min` 混合，但不
允许没有任何 `sem_agg`。

Contract:

```text
output schema = all aggregate output columns
required: semantic keys ⊆ union(all sem_agg.output_cols)
```

Rules:

- `sem_groupby` 不自动输出 semantic keys。
- semantic keys 必须由一个或多个 `sem_agg.output_cols` 覆盖。
- 多个 aggregate output columns 之间不能重名。
- `array_agg` 可以保存 evidence，但不能负责生成 semantic keys。
- 如果 aggregate list 里没有任何 `sem_agg`，直接判错。

例子：

```python
entities = (
    extracted_entities
    .sem_groupby(
        input_cols=["name", "entity_type"],
        instruction="Rows refer to the same real-world entity.",
    )
    .agg(
        sem_agg(
            input_cols=["name", "entity_type", "episode_content"],
            output_cols={
                "name": "Canonical entity name.",
                "entity_type": "Canonical entity type.",
                "summary": "Entity summary.",
            },
            instruction="Merge entity mentions into one canonical entity row.",
        )
    )
)
```

如果要同时保存 evidence，可以在同一个 `.agg(...)` 里加入 `array_agg(...)`，但
semantic keys 仍然必须由 `sem_agg` 输出：

```python
entities = (
    extracted_entities
    .sem_groupby(
        input_cols=["name", "entity_type"],
        instruction="Rows refer to the same real-world entity.",
    )
    .agg(
        sem_agg(
            input_cols=["name", "entity_type", "episode_content"],
            output_cols={
                "name": "Canonical entity name.",
                "entity_type": "Canonical entity type.",
                "summary": "Entity summary.",
            },
            instruction="Merge entity mentions into one canonical entity row.",
        ),
        array_agg(
            columns=["episode_id", "entity_ordinal", "name", "entity_type"],
            output_col="mentions",
        ),
        min(
            columns=["add_seq", "entity_ordinal"],
            output_col="entity_id",
        ),
    )
)
```

### 1.9 `sem_groupby` 的 `partition_by`

`sem_groupby` 支持 deterministic partition keys：

```python
rows.sem_groupby(
    input_cols=["name", "entity_type"],
    partition_by=["group_id"],
    instruction="Rows refer to the same real-world entity.",
)
```

`partition_by` keys 是 relational keys，不是 semantic keys。

当前支持的 `sem_groupby(...).sem_agg(...)` 和
`sem_groupby(...).agg(...)` 遵守这个 contract：

```text
output schema = partition_by keys + semantic aggregate outputs
```

规则：

- `partition_by` keys 自动保留。
- 这两个 operator forms 允许 aggregate output columns 和 `partition_by` keys
  重名，并以 deterministic partition keys 为准。这个行为是为了 schema 稳定，
  但不推荐主动依赖。
- direct `sem_groupby(...).array_agg(...)` 和 `sem_groupby(...).min(...)` 当前不支持，
  因而不定义对应的 collision contract。
- `input_cols` 仍然是 semantic keys。如果最终 view 需要这些 semantic keys，
  它们仍然必须由 `sem_agg` outputs 覆盖。

例子：

```python
entities = (
    extracted_entities
    .sem_groupby(
        input_cols=["name", "entity_type"],
        partition_by=["group_id"],
        instruction="Rows refer to the same entity within the same graph partition.",
    )
    .sem_agg(
        input_cols=["name", "entity_type", "episode_content"],
        output_cols={
            "name": "Canonical entity name.",
            "entity_type": "Canonical entity type.",
            "summary": "Entity summary.",
        },
        instruction="Merge entity mentions into one canonical entity row.",
    )
)
```

输出 schema：

```text
group_id, name, entity_type, summary
```

### 1.10 `sem_groupby` 的 `membership`

`membership` 说明一条 row 最终允许属于几个 semantic groups。它不决定模型调用
是 pairwise 还是 listwise，也不决定是否启用 embedding Search-Filter。

```text
membership=None
  保留旧合同。当前静态执行仍给每条row一个group ID；join-map不额外限制一个
  changed group可以匹配多少个current groups。

membership="exclusive"
  每条row只属于一个semantic group。join-map把这个合同lower为每个changed group
  最多选择一个current group，也就是sem_join(k=1)。没有合适target时仍可新建group。

membership="overlapping"
  一条row可以属于多个semantic groups。API保留了这个名字，但当前静态执行和
  differential lowering尚未实现，调用时明确失败。
```

这里最重要的边界是：`exclusive`是logical membership contract；`k=1`是compiler
为join-map选择的关系表达；pairwise/listwise和Search-Filter则是更下面的physical
execution。三层不能混在一起。

## 2. Paper-wise differential rules

这一节只写paper-wise rules，不展开implementation-only的列投影。简单rule保持一行；
join-map显式写出changed groups和join result，避免把确定性keys、semantic predicate
和target merge压在一条难读的公式里。这里默认`V` row是可继续merge的aggregate
state。

记号：

```text
D   = old input relation
ΔD  = changed input rows
K   = deterministic keys
Ks  = semantic keys
Kr  = partition_by relational keys
θg  = sem_groupby instruction
θa  = sem_agg / sem_map instruction
M   = sem_groupby membership contract
κ(M)= 1 when M is exclusive; omitted when M is unspecified
O   = sem_agg output columns
C   = array_agg input columns
o   = array_agg output column
c   = min input column
m   = min output column
A*  = A1, A2, ..., An
```

本文用下面的缩写表示changed aggregate groups和current view之间的semantic
join-map：

```text
JoinGroups(GΔ, V, Kr, θg, M) =
  GΔ.sem_join(
    V,
    instruction=group_match(θg),
    how="outer",
    on=Kr when Kr is non-empty,
    k=κ(M),
  )
```

`group_match(θg)`是compiler从原`sem_groupby` instruction改写出的“changed group
和current group是否属于同一组”谓词。`Ks`只是这个semantic predicate读取的内容，
绝不能写成`on=Ks`。`sem_join.on`本身支持general deterministic `JoinOn`；但在这条
grouped-aggregate rule里，`on=Kr`有一个更窄、更明确的职责：只用必须精确相等的
`partition_by` keys排除跨partition candidates。

`MergeByTarget`表示：matched rows按选中的current target合并；没有target的changed
group形成新row；未被触及的current row原样保留。如果多个changed groups命中同一个
target，只生成一次合并后的target row。

### 可复用的 direct `sem_join` 增量规则

`rule-join-map`会在自己的maintenance query内部调用`sem_join`，但direct
`sem_join` view的增量维护是一个独立、可复用的binary-state能力。对：

```text
J = L.sem_join(R, on=φ, instruction=θ, how="inner", k=None)
```

append-only rule是：

```text
ΔJ = SemJoin(ΔL, R0, φ, θ)
     bag-union SemJoin(L0, ΔR, φ, θ)
     bag-union SemJoin(ΔL, ΔR, φ, θ)

J1 = J0 bag-union ΔJ
```

这里`φ`可以是`on=None`、同名exact keys或general deterministic `JoinOn`。每一项都
先用`φ`缩小candidate pairs，再执行原semantic predicate `θ`；`ΔL × ΔR`只在第三项
计算一次。实现使用保留重复行的`concat`，不是会去重的`union`。

当前direct differential只支持append-only、`how="inner"`和`k=None`。这不限制
静态`sem_join`，也不改变本章的grouped-aggregate rules：`rule-join-map`仍然可以在
其自身state-merge lowering内部使用outer或top-k semantic join；它不是把该内部
join再次编译成一个独立的direct differential node。

### 2.1 `sem_groupby(...).sem_agg(...)`

```text
V = D.sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(O, θa)
```

rule-re-group:

```text
V' = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(O, θa).union(V).sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(O, θa)
```

rule-join-map:

```text
GΔ = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(O, θa)
J  = JoinGroups(GΔ, V, Kr, θg, M)
V' = MergeByTarget(J, sem_agg(O, θa)).select(Kr + O)
```

说人话：delta先在自己的确定性partition内形成changed groups；changed groups只和
同partition的current groups做semantic匹配；最后只更新命中的旧group，并把未命中的
delta作为新group加入。旧groups彼此不会重新比较。

### 2.2 `group_by(...).sem_agg(...)`

```text
V = D.group_by(K).sem_agg(O, θa)
```

rule-re-group:

```text
V' = ΔD.group_by(K).sem_agg(O, θa).union(V).group_by(K).sem_agg(O, θa)
```

rule-join-map:

```text
V' = ΔD.group_by(K).sem_agg(O, θa).outer_join(V, on=K).sem_map(output_cols=O, instruction=θa).select(K + O)
```

这里 `sem_map(output_cols=O)` 不负责生成 `K`。`K` 来自 deterministic join keys。

### 2.3 `group_by(...).array_agg(...)`

```text
V = D.group_by(K).array_agg(C, o)
```

默认 relational merge rule:

```text
V' = ΔD.group_by(K).array_agg(C, o).full_outer_join(V, on=K).assign(o = array_cat(o:right, o:left)).select(K + o)
```

这里 `o:right` 是旧数组，`o:left` 是新数组。`array_cat(o:right, o:left)` 表示
旧 evidence 在前，新 evidence 在后。

### 2.4 `min(...)` and `group_by(...).min(...)`

以下 `c` 既可以表示单列，也可以表示有序 column tuple。复合 input 先产生一个
tuple state `m`，后续 rule 仍只比较这个 aggregate state。

```text
V = D.min(c, m)
V' = ΔD.min(c, m).union(V).min(m, m)
```

Grouped rule-re-group:

```text
V = D.group_by(K).min(c, m)
V' = ΔD.group_by(K).min(c, m).union(V).group_by(K).min(m, m)
```

Grouped rule-join-map:

```text
V' = ΔD.group_by(K).min(c, m).full_outer_join(V, on=K).assign(m = least(m:left, m:right)).select(K + m)
```

### 2.5 `group_by(...).agg(A1, ..., An)`

```text
V = D.group_by(K).agg(A1, A2, ..., An)
```

rule-re-group:

```text
V' = ΔD.group_by(K).agg(A*).union(V).group_by(K).agg_merge(A*)
```

rule-join-map:

```text
V' = ΔD.group_by(K).agg(A*).full_outer_join(V, on=K).merge(A*).select(K + outputs(A*))
```

### 2.6 `sem_groupby(...).agg(A1, ..., An)`

```text
V = D.sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg(A1, A2, ..., An)
```

rule-re-group:

```text
V' = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg(A*).union(V).sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg_merge(A*)
```

rule-join-map:

```text
GΔ = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg(A*)
J  = JoinGroups(GΔ, V, Kr, θg, M)
V' = MergeByTarget(J, A*).select(Kr + outputs(A*))
```

### 2.7 `merge` and `agg_merge`

`merge` 用在 join-map style rule 里。它把 left / right 两边的 aggregate state
合成下一版 output。

```text
merge(array_agg(C, o)) = assign(o = array_cat(o:right, o:left))
merge(sem_agg(O, θ)) = sem_map(output_cols=O, instruction=θ)
merge(min(c, m)) = assign(m = least(m:left, m:right))
```

`agg_merge` 用在 re-group style rule 里。它处理的输入已经是 aggregate state
rows，不是原始 rows。

```text
agg_merge(array_agg(C, o)) = collect_list(o).flatten() as o
agg_merge(sem_agg(O, θ)) = sem_agg(O, θ)
agg_merge(min(c, m)) = min(m, m)
```

也就是说，普通 `array_agg(C, o)` 是 raw rows 到一个 array state；
`agg_merge(array_agg(C, o))` 是多个 array state 到一个 array state。

### 2.8 `partition_by`

`partition_by` 不创造新的semantic rule family。它把输入先按普通关系key分区，
然后在每个分区内应用同一条semantic rule。也就是说，它把：

```text
sem_groupby(Ks, θg)
```

替换成：

```text
sem_groupby(Ks, partition_by=Kr, θg)
```

输出中额外带上deterministic partition keys `Kr`。在join-map中，`Kr`还会下推成
`sem_join(on=Kr)`，从而保证不同partition的rows根本不会成为semantic candidates。
semantic keys `Ks`仍然只参与`θg`，不要求精确相等。

普通deterministic `group_by(Kr)`的增量维护不是semantic join。delta row可以直接按
`Kr`找到受影响的partition；只有这些partition内部的changed semantic groups需要继续
执行join-map或re-group。未被delta触及的partition原样保留。

## 3. Implementation-wise differential rules

paper-wise rules 里不写 `input_cols`。implementation 里必须写，因为
`sem_agg(input_cols=I, output_cols=O, instruction=θ)` 只读取 `I` 中的列；
`sem_agg(input_cols=None, output_cols=O, instruction=θ)` 则表示读取当前可见
row 的 aggregate state。

这一节只定义 implementation rule family，不声明当前代码都已经实现。

当前已有代码命名可以这样理解：

```text
compressed    = rule-all-group
changed-aware = rule-all-group-optimized
join-map      = rule-join-map
```

### 3.1 Rule families

`rule-join-map`：先把 `ΔD` 聚合成 changed aggregate state，再和 `V` join，
最后用 `sem_map` 或 deterministic merge 生成 `V'`。

`rule-re-group`：先把 `ΔD` 聚合成 changed aggregate state，再和 `V` union，
最后对这些 aggregate state rows 重新 group / aggregate。

`rule-all-group`：直接把 `ΔD` 和 `V` 放在一起重新 group / aggregate。这是
compressed-state approximation，只适合 aggregate 能直接消费 raw rows 和 old
aggregate state 的情况。

`rule-all-group-optimized`：和 `rule-all-group` 语义同类，但 implementation 可以只
重算 touched groups。

### 3.2 `sem_groupby(...).sem_agg(...)`

```text
V = D.sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(input_cols=I, output_cols=O, instruction=θa)
```

rule-join-map:

```text
GΔ = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(input_cols=I, output_cols=O, instruction=θa)
J  = JoinGroups(GΔ, V, Kr, θg, M)
V' = MergeByTarget(J, sem_agg(input_cols=None, output_cols=O, instruction=θa)).select(Kr + O)
```

rule-re-group:

```text
V' = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(input_cols=I, output_cols=O, instruction=θa).union(V).sem_groupby(Ks, partition_by=Kr, membership=M, θg).sem_agg(input_cols=None, output_cols=O, instruction=θa)
```

rule-all-group:

```text
V' = ΔD.union_by_name(V).sem_groupby(Ks, θg).sem_agg(input_cols=I, output_cols=O, instruction=θa)
```

rule-all-group-optimized:

```text
V' = touched_groups(ΔD, V).sem_groupby(Ks, θg).sem_agg(input_cols=I, output_cols=O, instruction=θa).union(untouched_groups(V))
```

`rule-re-group` 和 `rule-all-group` 的区别在最后一步：`rule-re-group` 最后消费的是
aggregate state rows，所以 final `sem_agg` 使用 `input_cols=None`；`rule-all-group`
让同一个 `sem_agg(input_cols=I, ...)` 同时消费 changed raw rows 和 old aggregate
state。

final state reaggregation 的 instruction 只做 deterministic schema repair：state
中仍存在的 `{column}` 保留；已经不存在、但属于原始 `input_cols` 的
`{column}` 去掉花括号。它不做 semantic prompt rewrite。

### 3.3 `group_by(...).sem_agg(...)`

```text
V = D.group_by(K).sem_agg(input_cols=I, output_cols=O, instruction=θa)
```

rule-join-map:

```text
V' = ΔD.group_by(K).sem_agg(input_cols=I, output_cols=O, instruction=θa).outer_join(V, on=K).sem_map(output_cols=O, instruction=θa).select(K + O)
```

rule-re-group:

```text
V' = ΔD.group_by(K).sem_agg(input_cols=I, output_cols=O, instruction=θa).union(V).group_by(K).sem_agg(input_cols=None, output_cols=O, instruction=θa)
```

rule-all-group:

```text
V' = ΔD.union_by_name(V).group_by(K).sem_agg(input_cols=I, output_cols=O, instruction=θa)
```

rule-all-group-optimized:

```text
V' = touched_groups(ΔD, V).group_by(K).sem_agg(input_cols=I, output_cols=O, instruction=θa).union(untouched_groups(V))
```

`K` 是 deterministic key。所有 rule 的最终 output schema 仍然是 `K + O`。

### 3.4 `min(...)` and `group_by(...).min(...)`

Implementation 中单列和复合 input 都规范化为 `columns=(...)`。第一次对 `ΔD`
聚合时读取原始 columns；合并 state 时只对 output column `m` 再做单列 `min` 或
row-wise `least`。

Global:

```text
V' = ΔD.min(c, m).concat(V).min(m, m)
```

Grouped rule-re-group:

```text
V' = ΔD.group_by(K).min(c, m).concat(V).group_by(K).min(m, m)
```

Grouped rule-join-map:

```text
V' = ΔD.group_by(K).min(c, m).full_outer_join(V, on=K).assign(m = least(m:left, m:right)).select(K + m)
```

这些 rules 在 append-only input 下是 exact。delete 或当前 minimum 被撤回时，需要
保存 raw group、ordered auxiliary state，或回退到 group recompute；本轮不定义
negative-delta rule。

### 3.5 `group_by(...).array_agg(...)`

```text
V = D.group_by(K).array_agg(C, o)
```

这里只有 deterministic relational merge rule：

```text
V' = ΔD.group_by(K).array_agg(C, o).full_outer_join(V, on=K).assign(o = array_cat(o:right, o:left)).select(K + o)
```

这里不需要 `rule-join-map` / `rule-re-group` / `rule-all-group` 的 semantic 变体。
`array_agg` 的 merge 行为就是确定性的 `array_cat`。

### 3.6 `group_by(...).agg(A1, ..., An)`

```text
V = D.group_by(K).agg(A1, A2, ..., An)
```

rule-join-map:

```text
V' = ΔD.group_by(K).agg(A*).full_outer_join(V, on=K).merge(A*).select(K + outputs(A*))
```

rule-re-group:

```text
V' = ΔD.group_by(K).agg(A*).union(V).group_by(K).agg_merge(A*)
```

mixed `.agg(A*)` 暂不定义 `rule-all-group`。原因是 `array_agg` 需要 merge array
state，而不是把 old aggregate-state row 当作 raw evidence row 再做普通
`array_agg(C, o)`。

### 3.7 `sem_groupby(...).agg(A1, ..., An)`

```text
V = D.sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg(A1, A2, ..., An)
```

rule-join-map:

```text
GΔ = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg(A*)
J  = JoinGroups(GΔ, V, Kr, θg, M)
V' = MergeByTarget(J, A*).select(Kr + outputs(A*))
```

rule-re-group:

```text
V' = ΔD.sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg(A*)
       .union(V)
       .sem_groupby(Ks, partition_by=Kr, membership=M, θg).agg_merge(A*)
```

和 deterministic `group_by(...).agg(A*)` 一样，mixed semantic `.agg(A*)` 暂不定义
`rule-all-group`。如果 `A*` 里包含 `array_agg`，implementation 必须使用
`collect_list(...).flatten(...)` 来合并 array aggregate state。

### 3.8 Implementation `merge` and `agg_merge`

implementation-wise 的 `merge` 和 paper-wise 一致：

```text
merge(array_agg(C, o)) = assign(o = array_cat(o:right, o:left))
merge(sem_agg(O, θ)) = sem_map(output_cols=O, instruction=θ)
merge(min(c, m)) = assign(m = least(m:left, m:right))
```

implementation-wise 的 `agg_merge` 只是 rule notation，不是 `QueryExpr` operator。
实际 lowering 必须显式生成真实 operator graph，并区分 state rows：

```text
agg_merge(array_agg(C, o)) = collect_list(o).flatten() as o
agg_merge(sem_agg(O, θ)) = sem_agg(input_cols=None, output_cols=O, instruction=θ)
agg_merge(min(c, m)) = min(m, m)
```

也就是说，implementation 里不应该出现 `agg_merge` op，也不应该通过
`agg(mode="state_merge")` 之类的 hidden mode 表达它。

### 3.9 `partition_by`

`partition_by` 不改变implementation rule family，但会改变候选范围和output
identity：

```text
static execution:
  先按Kr拆分DataFrame，再在每个partition内执行sem_groupby

rule-re-group:
  Kr随aggregate state保留，重新分组仍然只发生在各partition内部

rule-join-map:
  compiler把Kr下推为sem_join(on=Kr)
  先做exact-key candidate restriction，再做semantic predicate
```

当前`rule-re-group`和`rule-join-map`都支持`partition_by`。这里不允许“先做全局
semantic outer join，再在结果上filter Kr”，因为那会丢失本应保留的unmatched rows，
改变outer-join语义。

## 4. 本文档不定义什么

本文档不决定：

- optimizer 应该选择哪个 rule family；
- delete / negative delta 时保存 raw group、ordered state，还是 full recompute；
- DAG runtime 应该使用哪种 indexed physical aggregate state。

当前 implementation 已用真实 `QueryExpr` operator graph 实现本文第三章覆盖的
`sem_agg`、`array_agg`、mixed `.agg(...)` 和 `min` lowering；不存在 public 或
internal `agg_merge` operator。Shared `PolicyExecutor` 会在这些 stateful node
boundary 对 insert-only change 执行本章定义的 maintenance query。如果 parent
change 含 internal retraction，executor 会在 parent 的完整 next state 上重算当前
semantic aggregate node，再把 replacement change 传给下游。Source delete 和更
高效的 indexed aggregate state 仍未实现。
