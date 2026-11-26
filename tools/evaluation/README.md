# Memory Benchmarks

## Semantic maintenance experiments

This opt-in experiment is separate from the benchmark runners below. It keeps
their Claude policy and evaluation functions unchanged and adds explicit
maintenance planning. Each condition runs in a separate process; compilation and
restoration never call the planning Agent. No state rule is forced for coverage.

Keep original datasets and generated results outside the repository. With the
pinned raw files in `DATA_DIR/data/`, prepare the small case manifest:

```bash
python -m tools.evaluation.prepare_maintenance_poc --data-dir DATA_DIR
python -m tools.evaluation.maintenance_poc qmsum-specific --data-dir DATA_DIR
```

Both commands are model-free. Prepared Claude input is the first 32 original
messages of LOCOMO sample 0, with five original questions whose annotated evidence
is covered. News classification combines the original hint and category catalog
into Ig; clustering uses its original hint. This tests our maintenance operators,
not a complete reproduction of HOLDUP's algorithms. Raw data is never rewritten.

Real calls require explicit limits and a new output directory:

```bash
python -m tools.evaluation.maintenance_poc qmsum-specific --data-dir DATA_DIR \
  --mode agent --run --source-receipt SOURCE_MANIFEST.json --output RUN_DIR \
  --max-requests 32 --max-reserved-tokens 300000
```

The source receipt lists repository-relative files and their SHA-256 values.
The runner checks it before execution. Use `fallback` or `composable` for controls,
or `baseline` for the original Claude policy. The default backend is `deepseek`,
model `deepseek/deepseek-flash`, thinking disabled, provider/parse retries disabled.
`--model deepseek/<model>` overrides that model. Token reservation is estimated
input plus maximum output, not billable usage. Reaching a budget stops new calls;
the runner does not change the Agent plan, truncate input or increase the budget.
Optional operator validation retries and checkpoint continuation are described
under [maintenance experiment checkpoints](#maintenance-experiment-checkpoints).

### Maintenance experiment conventions

The `maintenance_poc` filename is retained for existing commands. These
conventions bound maintenance experiments, not the public API or the native
benchmark runners below.

- Plan separately with `--plan-only`; execute with the exact saved plans. Keep
  fixed-rule controls in their own mode, without inventing Agent plans for them.
- Declare input/output columns explicitly in new experiment queries, especially
  aggregate descriptors. Existing direct `sem_agg` calls with resolved default
  inputs remain supported. Use unique column names and stable record/key types;
  group keys belong to the program, not the model-generated result fields.
- Select and record pacing and validation retries for each experiment; keep
  comparable conditions consistent. Earlier subscription experiments used serial
  conditions and `--min-interval 10`; ten seconds is not a universal requirement.
  Retain every attempt without editing the prompt or repairing output.
- From the first call, use one shared `--budget-ledger ROUND/budget.json` for the
  whole round, including continuations. Supply cumulative request, reservation,
  and observed completion-token limits. Unknown usage stays unknown.
- Preflight with the actual execution command minus `--run`. Keep the data,
  saved plans, model, and execution settings unchanged on resume; use a current,
  trusted checkpoint and a fresh output directory. Source receipts must cover the
  relevant implementation files, not an empty or deliberately partial selection.
- Use `--maintenance-only` when testing maintenance; do not accidentally append
  retrieval, answering, or scoring. Preserve old runs and label evaluation changes
  separately from maintenance changes.

Runner settings are distinct from those experiment choices:

| Setting | Current runner behavior |
| --- | --- |
| Condition order | Chosen by the launcher; the runner does not schedule a matrix |
| Request spacing | `--min-interval`, default `0`; positive values also serialize multi-prompt calls |
| Model batch size | Fixed to `1` |
| Model cache | Disabled |
| Provider retries | Disabled |
| Output-validation retries | `--validation-retries`, default `0`, allowed range `0..3` |

Live planning followed by resumable execution, continuation without the shared
ledger, legacy checkpoint migration, and cache-enabled retries are outside this
experiment scope.

### Local subscription gateway (optional)

The same runner supports `--backend chatgpt-plan --model MODEL_SLUG`. It sends
`openai/MODEL_SLUG` through LOTUS to `http://127.0.0.1:8787/v1`. `--base-url`
accepts only HTTP numeric loopback addresses with a `/v1` path, no credentials or
query string. Proxies and redirects are disabled for this client. The runner
reads the local gateway key from `OPENAI_API_KEY` only with `--run`; it never reads
subscription credentials or logs the key. Existing gateway authorization is
reused, not copied or refreshed by this runner.

`--saved-plan PLAN.json` binds a previously saved plan and skips the Agent.
It requires `--mode agent` and one plan per target operator. For Claude, repeat
the flag in target order: grouping first, aggregation second. Fixed-rule controls
and an incorrect number of plans are rejected before any model request. The saved
schema and maintenance instruction are not edited. Preflight compiles those exact
plans without reading credentials or invoking a model:

```bash
python -m tools.evaluation.maintenance_poc qmsum-specific --data-dir DATA_DIR \
  --backend chatgpt-plan --model MODEL_SLUG --saved-plan PLAN.json
```

Use `--plan-only --run` with the usual output, receipt and budget flags to call
only the Agent and compile its plans, without ingesting records. Planning still
receives no target-task data examples. Run the executor in a separate process
with `--saved-plan` and its own model; this separates Agent and operator models
without switching LOTUS adapters in one process. Both phases count toward the
experiment budget. Without `--run`, either command remains model-free preflight.

`--plan-target 0` or `--plan-target 1` with `--plan-only` plans one Claude operator
independently; a failed grouping proposal need not prevent measuring aggregation.
The saved file is `plan-0.json` in that target's run directory; `plan_target` records
its original query position. Without this option all targets are planned in order.
For DeepSeek planning, explicitly pass `--thinking enabled`. The default remains
disabled, including operator runs. Subscription runs reject this flag rather than
pretending they control the provider's reasoning configuration.

To compare operators, reuse the same saved plan and vary the execution model.
To compare planners, hold the execution model fixed. Keep planning and execution
directories so each result can be traced to its producing planner and saved plan.

Start the gateway separately using its own documented configuration. Its model
availability, authorization, rate limits and upstream timeout are not configured
by this repository. The current runner uses a 240-second client timeout for the
gateway and 120 seconds for DeepSeek. It does not start a gateway or refresh login.

With `OPENAI_API_KEY` already set to the local gateway key, a bounded execution is:

```bash
python -m tools.evaluation.maintenance_poc qmsum-specific \
  --data-dir DATA_DIR --backend chatgpt-plan --model MODEL_SLUG \
  --saved-plan PLAN.json --run --source-receipt SOURCE_MANIFEST.json \
  --output RUN_DIR --max-requests 3 --max-reserved-tokens 100000 \
  --min-interval 10
```

The limits above are an example, not a promise to finish a task in three requests.
Use a new output directory per condition and fresh source receipts after source
changes. The runner does not automatically switch models on failure.

`run-contract.json` and the HTML report distinguish requested settings from known
gateway omissions. The gateway's existing `lotus` compatibility layer drops and
records unsupported LOTUS parameters such as `temperature` and `max_tokens`;
their constraints are **not** enforced upstream. Thinking uses the provider
default, not "disabled". `--max-output-tokens` remains an input to local token
reservation and structured-call requests, not a gateway output limit. Actual
returned usage is retained in `trace/`; per-request transformations are in gateway
history. JSON-object mode and the existing local schema validation remain in use.

Inspect progress locally, or run the same command through SSH on the experiment host:

```bash
python -m tools.evaluation.maintenance_poc status --runs-dir RUNS_DIR --watch
```

Status reads atomic `progress.json` files without importing the model libraries.
Closing this viewer does not stop experiments launched in a persistent terminal.
The viewer does not restart or retry runs. A missing process is reported separately
from a successful completion. `plan-*.json`, `step-*.json`, `calls/`, and `trace/`
retain plans, results, S, changes, raw model responses and provider usage. Each run
produces an HTML report. Static clustering metrics and existing LOCOMO scores do
not constitute gold labels for intermediate S or historical reassignment.

Offline tests: `pytest tests/test_maintenance_poc.py tests/test_stateful_maintenance.py`.
Set `AGENT_MEMORY_POC_DATA_DIR=DATA_DIR` to include original-data compile checks;
otherwise only those checks are skipped. No original datasets are required by CI.

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
calls. Preparation may download the pinned datasets but does not run the
memory model. Install MemoryAgentBench's sentence-tokenization resource with
the command below. Embedding and reranker weights may also be downloaded on
first use.

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

Install the sentence-tokenization resource before preparing the bundle:

```bash
uv run --extra benchmarks python -m nltk.downloader \
  -d .memory-test/datasets/nltk punkt_tab

uv run --extra benchmarks python tools/evaluation/memory_agent_bench.py prepare \
  --smoke --bundle-dir .memory-test/bundles/mab-smoke
```

The download directory matches the runner's default. For a different location,
use that path for both the downloader's `-d` and preparation's `--nltk-data-dir`.

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
| `cases/<case-directory>/retrieval.jsonl` | Retrieved context |
| `cases/<case-directory>/answers.jsonl` | Saved model answers |
| `cases/<case-directory>/grades.jsonl` | Scorer-specific results |
| `metrics/summary.json` | Aggregate metrics |
| `metrics/per_question.csv` | Question-level results |
| `metrics/provider_usage.csv` | Provider accounting |
| `trace/` | Phase events, prompts, responses, and diagnostic evidence |

Each case directory uses a sanitized case ID followed by an eight-character
hash suffix. Look under `cases/` for the generated directory names.

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

## Maintenance experiment checkpoints

`maintenance_poc` saves a trusted local runtime snapshot after each committed
input step. `--stop-after 16` pauses at the sixteenth step (for Claude, the
sixteenth original message), without running QA against incomplete evidence.
The input manifest still describes the full conversation; do not truncate it.

Continue in a new output directory with the same mode, source receipt, data,
model and request settings, adding `--resume-from <old-run>/checkpoint.json`
and a later `--stop-after`. Restore does not call the Agent or replay committed
inputs. Agent mode requires the same saved plans. Fixed-rule modes (`composable`,
`fallback`, `baseline`) resume without Agent plans and retain their original mode.
Data/plan/config changes reject restoration. Checkpoints use the existing
pickle snapshot format: load only trusted local artifacts, never downloaded files.
Use `--maintenance-only` to stop after ingestion even when the full input has
been processed; otherwise Claude execution continues to retrieval, answering
and evaluation after the final message.

Validation retry counts may change on resume and are recorded in
`resume-validation.json`; data, plans, model and other execution settings must
still match. `--resume-source-changes` accepts only explicitly reviewed before/after
hashes for `tools/evaluation/maintenance_poc.py` and `tests/test_maintenance_poc.py`.
It does not permit operator or runtime changes to bypass checkpoint identity checks.

`budget-live.json` accounts for all dispatched attempts, including failures after
the last commit. Resume inherits those counters rather than resetting the budget.
`--validation-retries 3` permits at most three additional identical requests for
invalid structured output, not a replay of the whole maintenance step. Transport,
authorization, quota and explicitly incomplete-response errors are not retried.
Defaults remain zero retries in the experiment runner; no output repair or
prompt modification is introduced.
`--min-interval` spaces LM calls after request artifacts have been written;
the recorded dispatch time is saved on return or exception for continuation.
This does not promise recovery of that timestamp after a hard process kill.
`--max-completion-tokens` stops new requests
after the observed returned-token threshold (a single response may cross it).
Token reservations remain estimates, not provider-enforced output caps.

For several sequential conditions, pass the same `--budget-ledger LEDGER.json`
and cumulative limits. The ledger is locked during a run; another process cannot
use it concurrently. Without a shared ledger, resume inherits the prior run's
`budget-live.json`. With one, the shared ledger is authoritative; prior counters
are not imported again. Use the same ledger throughout a round if its limits
must include every condition and continuation.

Actual output-token accounting comes from the provider response, using the same
usage normalization as tracing. Known usage is counted even if output parsing
later fails, and does not depend on writing trace files. Local cache hits are not
charged again. Missing output usage stays in `unknown_usage_requests`; a reported
zero is known zero. Unknown usage cannot be included in the observed-token total,
so that threshold is not a guarantee about all billable tokens. Request and token
reservation limits still apply. Existing ledgers are not retroactively corrected.

LongMemEval model inputs omit nested `has_answer` annotations. Original files,
message text, dates and task instructions are preserved.
