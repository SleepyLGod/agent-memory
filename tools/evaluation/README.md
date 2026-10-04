# Memory Benchmarks

Use one workflow across LOCOMO, LongMemEval, and MemoryAgentBench:
**prepare data, select a memory, run, inspect results**.
The runners share case isolation, input bundles, checkpoint handling, and
artifact formats. Dataset-specific task contracts supply the questions,
answer prompts, and scorers.

## 1. Prepare the environment

Run commands from the repository root with Python 3.12+ and uv.

| System selector | Representation and retrieval | Dependencies / storage |
| --- | --- | --- |
| `claude-memory` | Topic views and catalog; semantic top-k retrieval | Benchmark extras; embeddings needed only for selected execution profiles |
| `zep-memory` | Entities, episodes, and facts; graph-backed search and reranking | `zep` extra and a running Neo4j instance |
| `mem0-memory` | Fact view; cosine retrieval | `mem0` extra and case-local Qdrant |
| `mem0-enhanced` | Same fact view; semantic top-k retrieval | `mem0` extra and case-local Qdrant |

These selectors run this repository's policies, not the native products.

```bash
# Claude/Zep
uv sync --frozen --extra benchmarks --extra zep

# Mem0, in a separate environment
UV_PROJECT_ENVIRONMENT=.venv-mem0 uv sync --frozen --extra benchmarks --extra mem0
```

Set `DEEPSEEK_API_KEY` in a private `.env` using
[.env.example](../../.env.example). All `run` commands below make real API
calls. Preparation may download the pinned datasets and tokenization resources
but does not run the memory model. Embedding and reranker weights may also be
downloaded on first use.

### Zep storage

Start an isolated Neo4j deployment before running Zep. Set its URI, credentials,
database, actual image name, and digest in the `AGENT_MEMORY_NEO4J_*` variables
documented in [Configuration](../../docs/configuration.md#storage).
The runner connects to the server; it does not provision it.

For a local Docker deployment, use a dedicated container and volume. Set
`AGENT_MEMORY_NEO4J_PASSWORD` to a private password of at least eight characters
in `.env`, then start Neo4j with the same password:

```bash
# Load your own local .env, using shell-compatible KEY=value assignments.
set -a
source .env
set +a
export NEO4J_AUTH="neo4j/${AGENT_MEMORY_NEO4J_PASSWORD:?Set a Neo4j password in .env}"
docker pull neo4j:5.26.2
docker run -d --name agent-memory-neo4j \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  --env NEO4J_AUTH \
  --volume agent-memory-neo4j-data:/data \
  neo4j:5.26.2
unset NEO4J_AUTH

docker logs --tail 30 agent-memory-neo4j
docker exec agent-memory-neo4j bash -c \
  'cypher-shell -u neo4j -p "${NEO4J_AUTH#neo4j/}" "RETURN 1;"'
docker image inspect neo4j:5.26.2 --format '{{index .RepoDigests 0}}'
```

Allow Neo4j to finish starting before the `RETURN 1` check. Set
`AGENT_MEMORY_NEO4J_URI=bolt://localhost:7687`,
`AGENT_MEMORY_NEO4J_USER=neo4j`, `AGENT_MEMORY_NEO4J_DATABASE=neo4j`, and
`AGENT_MEMORY_NEO4J_IMAGE=neo4j:5.26.2` in `.env`. Set
`AGENT_MEMORY_NEO4J_IMAGE_DIGEST` to the digest returned by Docker. The
container keeps its data in the named volume; subsequent starts use
`docker start agent-memory-neo4j` and the original database password.
Keep native Graphiti and Agent Memory experiments in separate deployments.
Mem0's embedded Qdrant driver does not need a separately running service.

## 2. Prepare a dataset bundle

Each benchmark has a `prepare` subcommand. Reuse the same prepared bundle
across system conditions.

### LOCOMO

```bash
uv run --extra benchmarks python tools/evaluation/locomo.py prepare \
  --sample-index 0 --bundle-dir .memory-test/bundles/locomo-s0
```

This selects Sample 0 without a row limit. `--dataset-path` selects a local
dataset file; the default path is populated from the pinned source when needed.
`--question-numbers` selects questions; `--no-include-adversarial` excludes
adversarial questions. `--start-row` and `--row-limit` create explicit input
subsets.

For integration only, replace the sample selector with `--smoke`; it selects
fixed rows 26-28 and one question. A subset is not a full-sample score.

### LongMemEval

```bash
uv run --extra benchmarks python tools/evaluation/longmemeval.py prepare \
  --question-ids 852ce960 --bundle-dir .memory-test/bundles/longmemeval-selected
```

This prepares the complete history for the selected question. Omit
`--question-ids` to prepare all cases in the pinned cleaned 500-question
dataset. `--dataset-path` selects a local dataset file.

`--smoke` instead prepares a fixed first-session prefix with annotated
evidence. It tests the pipeline, not LongMemEval accuracy over complete
histories.

### MemoryAgentBench

```bash
uv run --extra benchmarks python tools/evaluation/memory_agent_bench.py prepare \
  --smoke --bundle-dir .memory-test/bundles/mab-smoke
```

The smoke selects one complete case and one question for each capability class:
EventQA 64k, ICL Banking77, DetectiveQA, and FactConsolidation SH 6k.
Preparation uses the dataset's 4096-token sentence-aligned chunks.
Use `--sources` for explicit sources; omit both selectors for all pinned
sources. `--dataset-dir`, `--max-cases-per-source`, and
`--max-questions-per-case` control local data and selection.

## 3. Select a system and run

Use the matching runner and bundle. Here are the same basic commands for each
benchmark:

```bash
uv run --env-file .env --extra benchmarks --extra zep \
  python tools/evaluation/locomo.py run \
  --bundle-dir .memory-test/bundles/locomo-s0 \
  --memory-model deepseek/deepseek-flash \
  --answer-model deepseek/deepseek-flash \
  --judge-model deepseek/deepseek-flash \
  --system claude-memory --output-dir .memory-test/runs/locomo-claude

uv run --env-file .env --extra benchmarks --extra zep \
  python tools/evaluation/longmemeval.py run \
  --bundle-dir .memory-test/bundles/longmemeval-selected \
  --memory-model-id deepseek-flash \
  --memory-model deepseek/deepseek-flash \
  --answer-model deepseek/deepseek-flash \
  --judge-model deepseek/deepseek-flash \
  --system claude-memory --output-dir .memory-test/runs/longmemeval-claude

uv run --env-file .env --extra benchmarks --extra zep \
  python tools/evaluation/memory_agent_bench.py run \
  --bundle-dir .memory-test/bundles/mab-smoke \
  --model deepseek/deepseek-flash \
  --system claude-memory --output-dir .memory-test/runs/mab-claude
```

For Zep, change `--system` to `zep-memory` and use a distinct output directory
after configuring Neo4j. For either Mem0 variant, use the same workflow with
the Mem0 environment and selector:

```bash
UV_PROJECT_ENVIRONMENT=.venv-mem0 uv run --env-file .env \
  --extra benchmarks --extra mem0 python tools/evaluation/locomo.py run \
  --bundle-dir .memory-test/bundles/locomo-s0 \
  --memory-model deepseek/deepseek-flash \
  --answer-model deepseek/deepseek-flash \
  --judge-model deepseek/deepseek-flash \
  --system mem0-memory --output-dir .memory-test/runs/locomo-mem0
```

`mem0-enhanced` selects the same maintained fact representation with a
different retrieval recipe. Substitute either Mem0 selector in the other two
runner commands in exactly the same way.

These commands explicitly select `deepseek/deepseek-flash`.
LOCOMO and LongMemEval accept separate memory, answer, and judge model flags;
MemoryAgentBench uses `--model`. Advanced execution parameters are centralized
in [Configuration](../../docs/configuration.md). Use a new output directory for
each changed condition.

### Continue or reuse maintenance

Reissue an unchanged run command to continue its output directory. Completed
cases are skipped; supported drivers restore the latest compatible checkpoint
for unfinished cases. Checkpoint boundaries are defined by the task contract
(for example, session boundaries in LOCOMO), not every successful API call.
Work after the latest durable checkpoint may repeat. Recovery does not promise
zero repeated charges, and process-local caches restart cold.

There is no generic `--resume` flag in these CLIs.
Input, model, and execution identity checks apply when restoring. Do not run
two writers against one output directory or load untrusted snapshots.

LOCOMO and LongMemEval can save a completed maintenance condition with
`--maintenance-only`. A separate run can use
`--maintenance-checkpoint-output-dir <maintenance-output>` to evaluate
compatible retrieval/answering settings without reinserting the history.
Do not combine those two flags. MemoryAgentBench does not expose them.

## 4. Inspect results

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Dataset, model, system, and execution identities |
| `input/*.jsonl` | Normalized cases, events, and questions |
| `cases/<case_id>/retrieval.jsonl` | Retrieved context |
| `cases/<case_id>/answers.jsonl` | Saved model answers |
| `cases/<case_id>/grades.jsonl` | Scorer-specific results |
| `metrics/summary.json` | Aggregate metrics |
| `metrics/per_question.csv` | Question-level results |
| `metrics/provider_usage.csv` | Provider accounting |
| `trace/` | Phase events, prompts, responses, and diagnostic evidence |

Inspect case completion and scorer IDs before interpreting averages.
Insertion, retrieval, answering, and grading are separate phases. Token
workload, cache-hit input, and monetary charges are different quantities;
unknown usage is not a zero-cost call. Traces may contain personal source text,
so keep output directories private.

LOCOMO records its benchmark score and the Zep judge as separate scorer rows;
the judge applies to categories 1-4. LongMemEval uses its official grading
prompts with the selected judge model and also writes
`official_hypotheses.jsonl` for external evaluation. A different judge model is
a different metric condition. MemoryAgentBench uses the scorer appropriate to
each task. Do not pool incompatible scorer denominators.

## Native systems and comparisons

Run Native Claude, Mem0, or Graphiti from their own execution checkouts and
record those revisions. Their launchers and service setup are not installed
by selecting `--system` in this repository. Native Graphiti's `NEO4J_*`
environment variables are separate from Agent Memory's prefixed variables.

When runs export the shared artifact contract, compare them with:

```bash
uv run python tools/evaluation/compare_memory_systems.py \
  --run-dir .memory-test/runs/locomo-claude \
  --run-dir .memory-test/runs/locomo-mem0 \
  --output-dir .memory-test/comparisons/locomo
```

The comparison requires matching dataset, normalized input, questions,
answer contracts, scorers, and model identities. Native outputs must follow
the shared artifact schema to be read by this command.

## Offline checks

```bash
uv run python tools/evaluation/locomo.py run --help
uv run python tools/evaluation/longmemeval.py run --help
uv run python tools/evaluation/memory_agent_bench.py run --help
uv run python -m pytest tests/test_benchmark_cli.py tests/test_benchmark_bundle.py
```

Help and these contract tests validate the CLI and bundle interfaces offline.
