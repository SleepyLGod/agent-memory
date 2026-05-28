# TODO: Semantic IVM Implementation Questions

Status: the current golden line is `docs/optimization/incremental-semantic-view-maintenance.tex` plus  `docs/design/operator-api.md`. This file tracks unresolved implementation questions. It is intentionally question-first, not a decision spec.

## V0.0 -> V0.1 Interface Contracts

- `Memory.spec()` currently collects only the concrete class body via `vars(cls)`.
  Extending a built-in memory by subclassing is intentionally unsupported for
  now. Example:

  ```python
  class MyMemory(am.ClaudeMemory):
      extra = ...
  ```

  This currently does not merge the parent `ClaudeMemory` log, private
  relations, or views. Future work needs explicit inheritance and override
  semantics before supporting this pattern.

- `Memory.add(...)` and `Memory.query(...)` are intentionally asymmetric.
  `add(...)` accepts message/event input and should later normalize platform
  messages into the configured log columns. `query(...)` is policy-owned
  retrieval behavior that builds a semantic query plan. Future work needs to
  decide how query plans are optimized and executed, and whether they need an
  inspectable class-level declaration separate from the Python method body.

- `filter(...)` and `assign(...)` currently accept arbitrary Python values in the
  current v0.0 interface. Before planner execution, they should be restricted to
  serializable deterministic column expressions rather than arbitrary callables
  or closures.

- Output schema / column inference is a planner prerequisite. Some operators
  expose output columns directly, while others preserve or combine input columns.
  We need a consistent schema inference helper before rules rely on expressions
  like `V.columns`.

- `am.Log` currently subclasses `Relation`, which makes `.expr` visible through
  paths such as `Memory.log.expr`. v0.0 keeps this for collector simplicity, but
  v0.1 should decide whether to hide the expression behind `_expr` or promote a
  stable debug/inspection API.

- The first stateful differential rewrite rule must decide how executable `ΔQ`
  references current view state `V` and changed input rows `ΔD`. v0.0 keeps
  `ΔD` as a mathematical/runtime concept rather than modeling it as a current
  `QueryExpr` operator; the current toy row-local `sem_filter`/`sem_map` rules
  work by binding the source log to changed rows at runtime.

- Adapter capability and default adapter injection are still unresolved. Before
  adding non-LOTUS engines, decide what capability contract each execution
  adapter exposes and where the default adapter is injected.

- `LotusAdapter` currently exposes only a `model` field. v0.1 must decide
  whether API keys, base URLs, provider names, temperature, and token limits live
  on the adapter dataclass or are delegated to LOTUS/provider-native
  configuration.

## Open Research Questions

### API context fields

Later we may also add the 'context' fields to the semantic operator APIs

### `sem_join` Match Metadata

- Sometimes computing only the matched pairs is not enough. Do we also need to know whether a pair is `matched`, `contradict`, `invalid`, `supersede`, `forget/delete`, or `uncertain`?
- Should this be represented as one more column such as `match_type` in the  `sem_join` output?
- If yes, who is responsible for producing it: `sem_join`, the rewrittenmaintenance instruction, a following `sem_map`, or runtime?

### Full `V'` vs `Delta V`

- When should the differentiated query compute the full next view `V'` directly?
- When should it compute `Delta V` first and then apply the delta value into  `V`?
- For semantic groupby and aggregation maintenance such as  `sem_join().sem_map()`, can we avoid mapping rows that are unchanged?

### `Delta V+` and `Delta V-`

- Do we need to compute `Delta V+` and `Delta V-` explicitly?
- If yes, are they logical relations in the generated differential query, or only runtime/store effects?
- How should deletes, invalidations, and overwritten rows be represented without adding a public `apply_delta` API too early?

### `sem_map` Execution After Joins

- After a join, does `sem_map` run on all joined rows or only the rows that may change?
- Should a normal deterministic filter happen before `sem_map`?
- Does `sem_map` run as one native batch, one LLM call per row, or a runtime-chosen batching plan?

### Merge / Upsert / Delta Application

- Do we need `merge_into`, `upsert`, or `apply_delta` operators, or can this beexpressed with `filter`, `minus/except`, `union`, `select`, and `sem_map`?
- What is the essence of merge/upsert here: relational set replacement, semantic merge, storage API, or optimizer lowering?
- Is `sem_union` enough for add/update cases, and what handles deletion?

### Differential Instruction Rewriting

- The generated `ΔQ` may need different instructions from the original full query `Q`. Who rewrites those instructions?
- Should groupby/join maintenance instructions explicitly include contradiction, supersession, invalidation, and forget/delete targets?
- How do we inspect and test rewritten instructions?

### Storage Backends

- How should file, relational, vector, and graph stores affect the generated differential plan?
- Which backend capabilities matter: overwrite, append, delete, exact key lookup, vector search, keyword search, graph traversal?
- How do we keep the logical DataFrame query independent from physical storage behavior?

### Schema and DSPy-Style Typed Layer

- Should v0 stay as simple DataFrame columns, or should users define schema classes / DSPy-style typed modules?
- If schemas exist later, do they compile into DataFrame `input_cols`, `output_cols`, column descriptions, and instructions?
- How do we avoid reintroducing schema-first confusion into the current DataFrame-first API?

### Multiple Semantic Execution Engines

- Can the same logical operators run on different semantic processing engines?
- Possible engines include LOTUS-style DataFrame operators, Palimpzest-style lazy datasets, DSPy modules, custom LLM pipelines, vector DB execution, graph DB traversal, or hybrid planners.
- What common contract must each engine satisfy?

### Window, Refresh, and Background Consolidation

- Where do window, refresh, and background consolidation belong: logical query, runtime scheduling, or optimizer-generated repair?
- Especially **window**: For me, window is just an input unit, but window here does not mean "how many rows/info in the log/view are valid", right? SQL also has window things in view query definition, but it seems that their window is more like the rules of info validation. So for us, we may not simply write `.window` things inside view definition, but write it ahead the query. HOWEVER, how could we do if we need the window as operator context???
- Is freshness only time-based, or do count/event/session thresholds matter later?
- Is background consolidation a lazy maintenance strategy, a repair job, or a separate view definition? It could not be the view definition, but we still need to do that? right?
