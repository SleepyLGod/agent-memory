# TODO: Semantic IVM Implementation Questions

Status: the current golden line is
`docs/optimization/incremental-semantic-view-maintenance.tex` plus
`docs/design/operator-api.md`. This is the single current backlog for unresolved
interface, operator, rule, runtime, and optimizer questions. It is intentionally
question-first, not a decision spec.

## Current Consolidated Backlog

### Operator execution

- Audit LOTUS Cascade with real traces before exposing it as a supported physical
  profile. Keep examples, raw outputs, explanations, helper decisions, and stats
  in a trace side-channel rather than logical result columns.
- Add an indexed or hybrid execution path for `sem_topk` without changing its
  logical ranking contract.
- Evaluate candidate pruning and indexed access for `sem_groupby`; pair batching
  improves stability but does not reduce the number of semantic pairs by itself.
- Validate large-group structured `sem_agg` for context limits, retries, trace,
  and cost before claiming that path is production-stable.
- Add `min_by` / `arg_min` only when a view needs the payload associated with a
  minimum ordering value. Removing a current minimum still requires retraction
  state or recomputation.

### Differential rules and runtime

- Define source deletion and general negative-delta semantics.
- Define old-state binding or an explicit full-recompute fallback for semantic
  joins over arbitrary subqueries.
- Define correct maintenance for materialized left, right, and outer semantic
  joins, where a new match can retract an old unmatched row.
- Support dependencies that mix base-log input with changed upstream views.
- Add recursive/fixpoint planning only with an explicit recursive-view contract.
- Add physical optimization such as indexed join/aggregate state and shared
  arrangements without changing logical or maintenance fingerprints silently.
- Add spec-time placeholder validation and view-aware planner errors once column
  scope inference is reliable.
- Define when a standalone semantic aggregate's current output is sufficient
  maintenance state; schema compatibility alone is not a proof.

### Runtime scheduling

- Define concurrent and multi-writer ordering beyond the current single-process
  append sequence.
- Keep maintenance batching, refresh gates, and background consolidation as
  runtime scheduling decisions unless they change the logical input scope.

## Interface Contracts

- Shared finite dependency maintenance is represented by `DifferentiatedPolicy`
  and executed by `PolicyExecutor`; see
  `policy-differentiation-dataflow-runtime.zh.md`.
  Remaining scheduler work is recursive/fixpoint support, physical optimization,
  source deletion, and advanced concurrent/window scheduling, not basic DAG
  change propagation.

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

- `filter(...)`, `assign(...)`, and predicate `join(...)` now use a minimal
  serializable relation-bound expression subset via `relation.col(...)`.
  Future work should add only proven deterministic expression needs, not
  arbitrary Python callables, SQL strings, or tuple predicates.

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

- Keep query-level windowing separate from runtime maintenance scheduling.
  A view-query window should mean data scope or segmentation, for example
  recent-N messages, time/session windows, or semantic segments. It changes the
  logical input relation seen by the view.
- Coalescing two appended messages before running `Q'` is not a query-level
  window operator. It is lazy / deferred / coalesced maintenance, or
  micro-batch maintenance. It changes the runtime `changed_rows` batch
  granularity, not the logical semantics of the view query.
- Do not model `count=2` coalesced maintenance as `log.window(count=2)`.
  That would incorrectly say the view only sees a two-row window, instead of
  saying runtime delays maintenance until two new rows are available.
- Exact relational IVM should converge to the same final view regardless of
  maintenance batch size. Approximate semantic maintenance may not: LLM
  `sem_groupby`, `sem_agg`, `sem_join`, and merge behavior can change when
  `changed_rows` contains one row versus multiple rows. Treat batch size as an
  experiment variable.
- Future work should decide whether refresh/freshness includes time, count,
  event, turn, session, query-triggered, and manual gates. These belong to
  runtime scheduling unless they change the logical data scope of a view.
- Background consolidation may be a lazy maintenance strategy, a repair job, or
  a separate logical view/update policy. Do not collapse it into `.window(...)`
  without a concrete data-scope semantics.
