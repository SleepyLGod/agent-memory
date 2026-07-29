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

All agent-memory runners accept the planner's complete
`--grouped-agg-rule` choice set. In particular, `rule-re-group`,
`rule-join-map`, and `prefer-join-map` have the same meaning for every policy
and benchmark. A strict strategy remains selectable even when a particular
policy cannot compile it; that run fails before insertion instead of being
hidden or silently changed by the CLI.

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
