# Shared Memory Benchmarks

LongMemEval v1 and MemoryAgentBench use canonical bundles and benchmark-neutral
artifacts for agent-memory policies. Native products run from their own
repositories and are compared only after all runs finish.

- `claude-memory`
- `zep-memory`
- `mem0-memory`
- `mem0-enhanced`

Dataset adapters prepare evidence. Task contracts define prompts and scoring.
System drivers only add events and retrieve context. The runner owns case
isolation, resume, answering, grading, trace, and artifacts.

Claude and Zep runners default to `rule-join-map`. They continue to accept the
planner's other grouped-aggregate rules as explicit compatibility or ablation
conditions. If a selected rule cannot compile a policy, the run fails before
insertion instead of silently changing the requested rule.

Mem0 does not contain a grouped aggregate, so its public benchmark CLI rejects
an explicit `--grouped-agg-rule`. The runner preserves its existing internal
`rule-all-group` identity only for benchmark compatibility and fingerprints;
Mem0 does not execute that grouped-aggregate rule.

## Install

```bash
uv sync --frozen --extra benchmarks --extra zep
```

Mem0 runs use the isolated environment described by the Mem0 design:

```bash
UV_PROJECT_ENVIRONMENT=.venv-mem0 \
  uv sync --frozen --extra benchmarks --extra mem0
```

This does not modify the existing Claude/Zep `.venv`.

## LOCOMO

Prepare the fixed rows 26-28, one-question integration smoke:

```bash
UV_PROJECT_ENVIRONMENT=.venv-mem0 uv run --extra benchmarks --extra mem0 \
  python tools/evaluation/locomo.py prepare \
  --smoke \
  --bundle-dir .memory-test/bundles/locomo-mem0-smoke
```

Run Agent Mem0 against that bundle:

```bash
UV_PROJECT_ENVIRONMENT=.venv-mem0 uv run --extra benchmarks --extra mem0 \
  python tools/evaluation/locomo.py run \
  --bundle-dir .memory-test/bundles/locomo-mem0-smoke \
  --system mem0-memory \
  --output-dir .memory-test/runs/locomo-am-mem0
```

Use `--system mem0-enhanced` for the same additive view with LLM-ranked
retrieval. Its default method is `pairwise-quick`; an explicit
`--sem-topk-method` selects another LOTUS implementation.

The answer is generated once. `grades.jsonl` stores the official LOCOMO score
and, for categories 1-4, the Zep judge result as separate scorer rows.

Real runs require `DEEPSEEK_API_KEY`. Zep runs additionally require the
`AGENT_MEMORY_NEO4J_*` variables. Native Graphiti uses `NEO4J_URI`,
`NEO4J_USER`, and `NEO4J_PASSWORD` and should point to a separate empty Neo4j
5.26.2 instance.

### Zep join-map physical execution

Zep declares exclusive semantic grouping in its policy. With
`rule-join-map`, the compiler compares only changed groups with the current
view and lowers each changed group to a zero-or-one-target semantic join.
Fact groups also carry exact source/target endpoint keys, so semantic matching
never crosses those deterministic partitions.

The policy does not choose how the bounded semantic join is executed. Select
that physical access path at run time:

```bash
uv run --extra benchmarks --extra zep \
  python tools/evaluation/locomo.py run \
  --bundle-dir .memory-test/bundles/locomo-s0 \
  --system zep-memory \
  --output-dir .memory-test/runs/locomo-zep-join-map \
  --grouped-agg-rule rule-join-map \
  --sem-join-topk-method listwise \
  --semantic-pair-profile-config /path/to/zep-site-profiles.json \
  --lotus-cache-mode disabled \
  --embedding-device cuda \
  --semantic-trace-snapshot-mode compact
```

`--sem-join-topk-method` accepts `listwise`, `pairwise-naive`,
`pairwise-quick`, or `pairwise-heap`. It is currently valid only for Zep when
`--grouped-agg-rule` is `join-map` or `rule-join-map`. The policy still contains
the same `sem_join(k=1)` expression; this flag changes only how the adapter
resolves that join.

`--semantic-pair-profile-config` binds Search-Filter or Proxy-Only profiles to
stable semantic predicate site IDs:

```json
{
  "schema_version": 1,
  "bindings": [
    {
      "site_id": "sem_join:<entity-site-digest>",
      "mode": "search-filter",
      "top_k": 15,
      "min_similarity": 0.6
    },
    {
      "site_id": "sem_join:<fact-site-digest>",
      "mode": "search-filter",
      "top_k": 10,
      "min_similarity": null
    },
    {
      "site_id": "sem_filter:<contradiction-site-digest>",
      "mode": "search-filter",
      "top_k": 10,
      "min_similarity": null
    }
  ]
}
```

Replace the placeholders with site IDs inventoried from the exact compiled
policy revision; do not copy digests between revisions by hand. Unknown,
duplicate, or drifted sites fail before model and storage initialization.
Site-level config is mutually exclusive with the global
`--semantic-pair-profile`, `--semantic-pair-top-k`, and
`--semantic-pair-min-similarity` options.

Prompt construction is a separate physical layer from candidate selection.
Enable it for the whole LOTUS execution path with one option:

```bash
--prompt-batch-size all
--prompt-batch-size 8
```

Omitting the option preserves the existing LOTUS prompts. `all` places all
currently ready, independent tasks from one operator invocation into one
prompt, unless the model context requires deterministic chunks. A positive
integer caps each prompt at that many tasks. Search-Filter still runs first, so
only selected candidates enter a prompt. The setting covers semantic filters,
maps, flat maps, pairwise joins, pairwise group comparisons, multi-anchor
top-k joins, and semantic aggregate groups. A direct listwise `sem_topk` is
already one ranking task; delegated LOTUS pairwise ranking keeps its own
algorithm-specific comparison schedule.

This option changes prompt construction, so it is explicit opt-in and enters
the maintenance/checkpoint identity. Operators validate every returned task ID
and output schema. Syntax-only JSON repairs are recorded; missing, duplicate,
unknown, or malformed task results fail instead of being guessed.

Structural validation does not make prompt batching semantics-preserving. A
packed request can return valid JSON while making different semantic decisions
from the same tasks sent in smaller prompts. The current Claude/Mem0/Zep smoke
did not establish a universal best batch size, and larger tested batches caused
material quality loss in at least one policy. Treat every
`--prompt-batch-size` value as a separate experiment condition; do not use
`all` as a general default.

Semantic aggregate provider dispatch remains a separate control:

- `--sem-agg-dispatch provider-batched` keeps every per-group prompt unchanged
  but submits ready prompts together through the LOTUS LM batch interface.

`--sem-agg-dispatch provider-batched` and `--prompt-batch-size` are mutually
exclusive. The former sends several unchanged prompts in one provider call;
the latter puts several tasks inside one prompt. Both are disabled by default.

`--refresh-every N` controls when source rows enter the maintenance DAG. Its
default is `1`, which preserves one eager refresh per event. A value greater
than one buffers normalized rows and publishes each full batch as one atomic
delta; the benchmark flushes the final partial batch after the last event and
before retrieval. That final flush remains inside the last event's insertion
timing, so it is not hidden as unreported cleanup work.

Count refresh is independent of prompt batching and provider request batching.
It changes the physical execution identity and checkpoint contract. Resume
must use the same `--refresh-every` value; intermediate dataset session
boundaries do not force an early refresh.

The remaining flags belong to separate physical layers:

- `--lotus-cache-mode disabled|memory` controls LOTUS's process-local exact
  cache. `memory` is an independent experiment condition and is not the same as
  DeepSeek provider prompt caching.
- `--embedding-device cpu|cuda` selects where the configured embedding model
  runs. It does not choose or change the embedding model.
- `--semantic-trace-snapshot-mode compact|full` controls trace detail.
  `compact` keeps pair decisions and accounting without full intermediate
  DataFrame snapshots; use `full` only for bounded debugging.

These physical settings enter run provenance and checkpoint identity. Do not
restore a checkpoint under a different join resolver, site profile, cache mode,
embedding device, or trace contract.

## LongMemEval v1

Prepare selected complete questions from the pinned cleaned 500-question
dataset:

```bash
uv run --extra benchmarks python tools/evaluation/longmemeval.py prepare \
  --question-ids 852ce960 \
  --bundle-dir .memory-test/bundles/longmemeval-smoke
```

Omit `--question-ids` only when intentionally preparing all 500 cases. Normal
benchmark bundles always contain complete case histories.

Before a paid pilot, `--smoke` prepares one fixed integration-only prefix:

```bash
uv run --extra benchmarks python tools/evaluation/longmemeval.py prepare \
  --smoke \
  --bundle-dir .memory-test/bundles/longmemeval-integration-smoke
```

This is case `8aef76bc` through its first complete session: 8 of 492 events.
The prefix contains that question's annotated evidence, but omits later
distractors. Its answer and grade validate the E2E pipeline only and must not be
reported as LongMemEval accuracy. No arbitrary event-limit option is provided.

Run the ClaudeMemory policy. LongMemEval's native Claude condition is not
launched from this repository:

```bash
uv run --env-file /path/to/.env --extra benchmarks --extra zep \
  python tools/evaluation/longmemeval.py run \
  --bundle-dir .memory-test/bundles/longmemeval-smoke \
  --memory-model deepseek/deepseek-v4-flash \
  --answer-model deepseek/deepseek-v4-flash \
  --judge-model deepseek/deepseek-v4-flash \
  --output-dir .memory-test/runs/longmemeval-claude
```

Use `--system mem0-memory` from `.venv-mem0` for Base cosine retrieval, or
`--system mem0-enhanced` for semantic top-k retrieval. Both reject grouped-rule
options. Base rejects `--sem-topk-method`; Enhanced accepts it and defaults to
`pairwise-quick`.

Successful runs also write `official_hypotheses.jsonl` so the official
evaluator can be run later without repeating memory insertion or answering.

The 30-case matrix inserts each maintenance condition once, then reuses its
read-only checkpoint for two retrieval methods:

```bash
# Maintenance checkpoints
uv run python tools/evaluation/longmemeval.py run \
  --bundle-dir .memory-test/bundles/longmemeval-30 \
  --output-dir .memory-test/runs/JM-maintenance \
  --grouped-agg-rule rule-join-map --maintenance-only

uv run python tools/evaluation/longmemeval.py run \
  --bundle-dir .memory-test/bundles/longmemeval-30 \
  --output-dir .memory-test/runs/RG-maintenance \
  --grouped-agg-rule rule-re-group --maintenance-only

# JM-Q; use listwise and JM-L for the sibling run.
uv run python tools/evaluation/longmemeval.py run \
  --bundle-dir .memory-test/bundles/longmemeval-30 \
  --output-dir .memory-test/runs/JM-Q \
  --grouped-agg-rule rule-join-map \
  --sem-topk-method pairwise-quick \
  --maintenance-checkpoint-output-dir .memory-test/runs/JM-maintenance
```

Repeat the last command for `JM-L`, `RG-Q`, and `RG-L`. A JM checkpoint cannot
be restored by RG, while quick and listwise intentionally share the matching
maintenance state.

Memory, answer, and judge models are separate run contracts. The default judge
uses the official LongMemEval prompts with DeepSeek V4 Flash; its scorer ID
names that model and is not reported as the official GPT-4o metric.

## MemoryAgentBench

The adapter, task registry, scorers, and CLI below are an implementation
foundation. They have offline contract coverage, but the four memory systems
have not all completed the real four-source smoke; do not describe this as a
finished cross-system benchmark.

Prepare one complete case and one question for each of the four official
capability classes:

```bash
uv run --extra benchmarks python tools/evaluation/memory_agent_bench.py prepare \
  --smoke \
  --bundle-dir .memory-test/bundles/mab-smoke
```

`--smoke` selects EventQA 64k, ICL Banking77, DetectiveQA, and
FactConsolidation SH 6k. It preserves the official 4096-token sentence-aligned
chunks. Use `--sources ...` for an explicit source selection; omit both only
when intentionally preparing every pinned source.

Run a built-in policy:

```bash
uv run --env-file /path/to/.env --extra benchmarks --extra zep \
  python tools/evaluation/memory_agent_bench.py run \
  --bundle-dir .memory-test/bundles/mab-smoke \
  --system claude-memory \
  --output-dir .memory-test/runs/mab-claude
```

For Agent Mem0, run the same command from `.venv-mem0` with
`--system mem0-memory` or `--system mem0-enhanced`. Each case owns a separate
embedded Qdrant path; chunks are injected once and all questions reuse that
state. A maintenance-only Mem0 checkpoint can feed both retrieval recipes
because both policies declare the same maintenance identity and storage mapping.

## Native Claude

Native Claude independently reads the same pinned dataset and fixed case IDs:

```bash
bun run tools/native-memory-benchmarks/longmemeval.ts \
  --dataset-path /path/to/longmemeval_s_cleaned.json \
  --output-dir .memory-test/longmemeval-native-30
```

Resume uses only Native Claude's own session-boundary checkpoint:

```bash
bun run tools/native-memory-benchmarks/longmemeval.ts \
  --dataset-path /path/to/longmemeval_s_cleaned.json \
  --output-dir .memory-test/longmemeval-native-30 \
  --resume
```

The native runner performs extraction, per-session consolidation, lenient
retrieval, answering, grading, trace and metrics entirely inside the Claude
Code checkout.

## Compare

Comparison first verifies dataset, normalized input, selected cases/questions,
answer prompt, scorer, and model fingerprints:

```bash
python tools/evaluation/compare_memory_systems.py \
  --run-dir /path/to/native-claude \
  --run-dir /path/to/JM-Q \
  --run-dir /path/to/JM-L \
  --run-dir /path/to/RG-Q \
  --run-dir /path/to/RG-L \
  --output-dir /path/to/comparison
```

Mismatched contracts fail instead of producing a misleading score table.

## Artifacts

Every run writes:

```text
manifest.json
input/cases.jsonl
input/events.jsonl
input/questions.jsonl
cases/<case_id>/retrieval.jsonl
cases/<case_id>/answers.jsonl
cases/<case_id>/grades.jsonl
trace/events.jsonl
trace/prompts/
trace/outputs/
metrics/summary.json
metrics/per_question.csv
metrics/provider_usage.csv
```

Trace phases are `insertion`, `retrieval`, `answering`, and `grading`.
Retrieval candidates are artifacts, not fake LLM calls. Completed cases resume;
failed cases start with a fresh state directory and storage namespace.

Use `uv run python -m pytest` for Python verification. Calling the `pytest`
console script directly does not reliably include this checkout's `tools`
package when tests are selected in isolation.

These are real, potentially expensive benchmarks. LongMemEval's smallest
complete case still contains hundreds of turns, and MAB document chunks can
produce many graph entities. The runner never truncates evidence, skips memory
operators, or reports partial smoke results as benchmark accuracy.

The fixed prefix smoke is an integration acceptance test: it proves insertion,
checkpoint reuse, retrieval, answering, grading, and trace output. It is not a
LongMemEval score and must not be presented as benchmark accuracy.
