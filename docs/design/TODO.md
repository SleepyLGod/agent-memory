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

- Catalog identity currently keeps the topic view's `name` column instead of
  asking a semantic operator to generate `topic_name`. Future work should decide
  whether deterministic aliasing belongs in a `rename` / `select_as` operator,
  in storage-side mapping such as `name -> topic_name/path`, or in a more
  general materialization config.

- `am.Log` currently subclasses `Relation`, which makes `.expr` visible through
  paths such as `Memory.log.expr`. v0.0 keeps this for collector simplicity, but
  v0.1 should decide whether to hide the expression behind `_expr` or promote a
  stable debug/inspection API.

- The first stateful differential rewrite rule must decide how executable `ΔQ`
  references current view state `V` and changed input rows `ΔD`. v0.0 keeps
  `ΔD` as a mathematical/runtime concept rather than modeling it as a
  `QueryExpr` operator; current row-local fragment rules work by binding the
  source log to changed rows at runtime.

- The first `sem_groupby(...).sem_agg(...)` view-boundary rule is implemented as
  a Claude-style full-next-view query shape:

  ```text
  ΔC = ΔD.sem_flat_map(...)
  ΔT = ΔC.sem_groupby(...).sem_agg(...)
  M = ΔT.sem_join(V, how="outer", instruction=<same topic identity instruction>)
  V' = M.sem_map(instruction=<same consolidation instruction>)
  ```

  Current limits: it only applies at a public view boundary, returns full `V'`
  rather than minimal `DeltaV`, and is not a generic exact `sem_agg`
  maintenance proof. Deterministic placeholder rewrite is separate from future
  semantic prompt rewriting. If a later design computes `DeltaV` instead, the
  same logical maintenance may become `left join + upsert/delete/skip`.

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
- If yes, who is responsible for producing it: `sem_join`, the maintenance
  instruction, a following `sem_map`, or runtime?
- The Claude-style grouped aggregate rule makes this concrete:
  `changed_topic.outer_join(current_topics)` produces at least three branches:
  `matched` means both sides exist and should be semantically merged,
  `unmatched_delta` means only the changed topic exists and should be created,
  and `unmatched_existing` means only the existing topic exists and should be
  kept. Today these branches are not tagged, so the following `sem_map` must
  infer create/keep/merge behavior from null side columns. That is a quality
  limitation, not an implemented action schema.

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
- For Claude-style topic consolidation, are `skip`, `keep`, `overwrite`,
  `delete`, and `create` represented as `sem_map` output columns, runtime/store
  effects, or a future explicit delta-application operator?

- Claude-style extraction prompts mention forget/remove behavior, but the
  current `topics` view has no action, tombstone, or delete-target schema. If a
  log row says "forget X", is that part of the view query, or a stateful
  maintenance/store operation over existing `V`? A plain `sem_filter` is not
  enough because deletion requires matching the request against materialized
  topics before applying a store update.

### Differential Predicate / Instruction Modification

- The first stateful Claude-style maintenance rule should try direct reuse:
  `sem_groupby -> sem_join` reuses the same topic identity predicate, and
  `sem_agg -> sem_map` reuses the same consolidation instruction.
- Should any predicate or instruction be modified during differentiation at
  all? If yes, what concrete operator/input-output mismatch makes modification
  necessary?
- When is exact reuse acceptable, and when does the changed operator shape
  require extra wording for existing rows, new rows, null sides, delete targets,
  or contradictions?
- If modification is needed later, design an automatic predicate/instruction
  modification component and define how to inspect and test its outputs.

### Storage Backends

- How should file, relational, vector, and graph stores affect the generated differential plan?
- Which backend capabilities matter: overwrite, append, delete, exact key lookup, vector search, keyword search, graph traversal?
- How do we keep the logical DataFrame query independent from physical storage behavior?
- The first `MarkdownStorageBackend` can keep field-to-markdown mapping config
  inside the backend. Future work should decide whether policy authors declare
  materialization config on views, or whether storage remains an external
  runtime configuration.

### Schema and DSPy-Style Typed Layer

- Should v0 stay as simple DataFrame columns, or should users define schema classes / DSPy-style typed modules?
- If schemas exist later, do they compile into DataFrame `input_cols`, `output_cols`, column descriptions, and instructions?
- How do we avoid reintroducing schema-first confusion into the current DataFrame-first API?

### Multiple Semantic Execution Engines

- Can the same logical operators run on different semantic processing engines?
- Possible engines include LOTUS-style DataFrame operators, Palimpzest-style lazy datasets, DSPy modules, custom LLM pipelines, vector DB execution, graph DB traversal, or hybrid planners.
- What common contract must each engine satisfy?

### Backend Reliability and LOTUS Ownership

- Provider transport failures, for example LiteLLM / DeepSeek returning an
  incomplete chunked response, are backend reliability failures. They are not
  Claude policy, `DifferentialRules`, or runtime correctness bugs by
  themselves.
- The preferred owner for generic provider retry, transport recovery, and
  rate-limit handling is LOTUS, LiteLLM, or the provider-native client layer.
  `agent-memory` should not broadly wrap every LOTUS operator with its own
  retry policy unless there is a concrete adapter-level reason.
- A minimal local stopgap may be acceptable only for custom structured lowering
  paths that call `lotus.settings.lm(...)` directly. Such retry must be bounded,
  must not change prompts or schemas, must not produce fake fallback rows, and
  must still raise with observable artifacts when all attempts fail.
- Do not treat semantic parse/default failures as transport failures. For
  example, join/groupby parser failures should use correctness-biased defaults,
  not provider retry.
- Native LOTUS operators should prefer LOTUS-owned retry and observability. A
  broad adapter retry around whole native operators can multiply cost and repeat
  side effects, so it needs a separate design decision.
- `structured_max_tokens=8192` is an adapter execution knob for custom
  structured lowering. It is not part of policy semantics, view definitions, or
  differential rules.
- `sem_join_default=False` and `sem_groupby_default=False` are
  correctness-biased defaults. They reduce false-positive memory merges when
  parsing fails, but may cause conservative topic splits.
- `lotus_style_sem_agg` is a compatibility copy of LOTUS main-style hierarchical
  aggregation for multi-output structured aggregation. Future work should prefer
  an upstream LOTUS hook, or replace the copy when LOTUS exposes an equivalent
  reusable API.

### Window, Refresh, and Background Consolidation

- Where do window, refresh, and background consolidation belong: logical query, runtime scheduling, or optimizer-generated repair?
- Especially **window**: For me, window is just an input unit, but window here does not mean "how many rows/info in the log/view are valid", right? SQL also has window things in view query definition, but it seems that their window is more like the rules of info validation. So for us, we may not simply write `.window` things inside view definition, but write it ahead the query. HOWEVER, how could we do if we need the window as operator context???
- Is freshness only time-based, or do count/event/session thresholds matter later?
- Is background consolidation a lazy maintenance strategy, a repair job, or a separate view definition? It could not be the view definition, but we still need to do that? right?
