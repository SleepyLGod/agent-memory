# Configuration Reference

Start with the [README](../README.md) or [experiment guide](../tools/evaluation/README.md).
This page separates public query semantics, backend execution settings, and
benchmark controls. Defaults below refer to this checkout.

## Models and credentials

| Setting | Default / requirement |
| --- | --- |
| Python `LotusAdapter.model` | `deepseek/deepseek-v4-pro` |
| Shared benchmark models | `deepseek/deepseek-v4-flash` |
| LOCOMO / LongMemEval | Separate `--memory-model`, `--answer-model`, `--judge-model` |
| MemoryAgentBench | `--model` selects the shared model |
| `DEEPSEEK_API_KEY` | Required for DeepSeek API calls; no embedded key |
| Maintenance thinking | Disabled by LOCOMO and MemoryAgentBench CLIs; LongMemEval defaults to `--memory-thinking disabled` |

The framework/provider prefix is part of the model argument. Structured
Responses execution validates its supported DeepSeek model names; changing the
model string alone does not add support for a new provider.

The examples use `deepseek/deepseek-flash` explicitly. For LongMemEval, also set
`--memory-model-id deepseek-flash` to record the matching model identifier.

Use `uv run --env-file .env ...` to load local credentials explicitly.
[.env.example](../.env.example) contains placeholders only. Other provider
credentials, when using LOTUS-compatible models, follow that provider's setup.

## Python execution configuration

```python
from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig

adapter = LotusAdapter(
    model="deepseek/deepseek-flash",
    config=LotusExecutionConfig(
        lm_num_retries=0,
        lm_max_batch_size=64,
        lm_enable_cache=False,
    ),
)
```

This constructs an adapter; memory execution makes the requests. The example
explicitly disables provider retries and local LM cache rather than relying
on SDK defaults.

| Field | Default | Purpose |
| --- | --- | --- |
| `lm_num_retries` | `None` | Provider/SDK retry override; `None` leaves its default |
| `lm_timeout` | `None` | Provider timeout override |
| `lm_max_batch_size` | `64` | LOTUS LM dispatch batch limit; not tasks in a prompt |
| `lm_rate_limit` | `None` | Optional LOTUS rate limit |
| `lm_model_kwargs` | `{}` | Model options, including supported context/output settings |
| `lm_enable_cache` | `None` | LM cache override; distinct from provider prompt cache |
| `structured_max_tokens` | `8192` | Structured-output response limit used by execution helpers |
| `structured_parse_retries` | `3` | Structured parsing retries, separate from provider retries |
| `structured_output_transport` | `chat-json-object` | Structured response transport |
| `prompt_batching` | `None` | Optional `PromptBatching(max_tasks=N)` |
| `semantic_pair_profiles` | `{}` | Candidate/decision profiles bound to semantic sites |
| `semantic_trace_dir` | `None` | Optional semantic trace destination |
| `semantic_trace_snapshot_mode` | `compact` | `compact` or `full` intermediate snapshots |
| `sem_topk_method` | `pairwise-naive` | Semantic retrieval ranking algorithm |
| `sem_join_topk_method` | `listwise` | Bounded semantic-join resolver |
| `sem_groupby_pair_batch_size` | `None` | SDK dispatch of independent pair comparisons |
| `sem_groupby_pair_batch_retries` | `0` | Additional grouped pair-dispatch retries |
| `sem_agg_dispatch` | `sequential` | `sequential` or `provider-batched` independent aggregate prompts |

The complete LOTUS passthrough fields, including per-operator examples,
strategies, safe-mode, and model kwargs, are declared in
[`LotusExecutionConfig`](../src/agent_memory/adapters/lotus/context.py).
These advanced options have operator-specific compatibility checks; a
structured or packed path does not accept every LOTUS prompt customization.

Default non-batched textual predicates use the LOTUS text path.
`chat-json-object` is the default transport for structured operations.
Explicit `responses-json-schema` selects the
single-task structured predicate path when batching is off. With batching on,
the same transport serves multi-task structured requests. It is not necessary
to set a one-task prompt batch to request a single-task schema.

## Shared benchmark controls

The three runner CLIs accept the following controls. Use `<runner>.py run --help`
for the exact parser; do not pass Python dataclass fields as CLI flags.

| Option | Default | Scope |
| --- | --- | --- |
| `--system` | Required in LOCOMO/MAB; Claude in LongMemEval | Four selectors in the experiment guide |
| `--namespace` | Derived from output directory | Storage isolation |
| `--condition-id` | Derived condition identity | Artifact label |
| `--refresh-every` | `1` | Count-triggered maintenance; tail flushed before retrieval |
| `--grouped-agg-rule` | `rule-join-map` for Claude/Zep | Mem0 rejects an explicit grouped-rule option |
| `--sem-topk-method` | Claude: `pairwise-naive`; Mem0 Enhanced: `pairwise-quick` | Not accepted by Mem0 Base or Zep |
| `--sem-join-topk-method` | `listwise` | Zep with join-map only |
| `--sem-groupby-pair-batch-size` | Unset | Independent comparison dispatch |
| `--sem-groupby-pair-batch-retries` | `0` | Dispatch retries |
| `--sem-agg-dispatch` | `sequential` | Independent aggregate dispatch |
| `--prompt-batch-size` | Unset | Positive integer or `all` |
| `--structured-output-transport` | `chat-json-object` | Or `responses-json-schema` with a supported model |
| `--semantic-pair-profile` | `oracle-only` | Or `search-filter`, `proxy-only` |
| `--semantic-pair-top-k` | Unset | Candidate bound for a compatible profile |
| `--semantic-pair-min-similarity` | Unset | Similarity threshold for a compatible profile |
| `--semantic-pair-profile-config` | Unset | JSON bindings for specific semantic sites |
| `--lotus-cache-mode` | `disabled` | `disabled` or process-local `memory` |
| `--embedding-device` | `cpu` | `cpu` or `cuda` for configured embedding work |
| `--semantic-trace-snapshot-mode` | `compact` | `compact` or `full` |

Top-k methods accept `pairwise-naive`, `pairwise-quick`, `pairwise-heap`, or
`listwise` where the selected system supports that option. Grouped rules also
accept `rule-all-group`, `rule-all-group-optimized`, and `rule-re-group`;
`compressed`, `changed-aware`, and `join-map` are supported legacy aliases.
These are explicit alternative conditions, not implicit fallback behavior.

LOCOMO additionally exposes `--lm-max-ctx-len` (unset by default) for the model
context limit. LongMemEval exposes `--max-new-cases` (unset), `--memory-model-id`
(default `deepseek-v4-flash`), and `--memory-thinking`. Mem0 execution requires
thinking disabled.

### Batching and refresh are different

| Control | What is combined |
| --- | --- |
| `CountRefresh` / `--refresh-every` | Source rows published in one maintenance update |
| `PromptBatching` / `--prompt-batch-size` | Several ready semantic tasks inside one prompt |
| `sem_groupby_pair_batch_size` | Independent pair prompts submitted through an SDK batch |
| `sem_agg_dispatch=provider-batched` | Independent aggregate prompts submitted together |

Packing uses only ready tasks at an operator invocation; it does not accumulate
unrelated future events. `all` can still split at the context limit. A batch
cap does not guarantee that every prompt contains that many tasks.

Prompt batching is incompatible with non-sequential `sem_agg_dispatch` and with
an explicit `sem_groupby_pair_batch_size`. Keep those settings at their defaults
when enabling it. SDK dispatch does not imply one provider request or a shared
prompt. Packed prompts change the model's context and should be evaluated as
a distinct condition.

### Candidate profiles

`oracle-only` leaves semantic decisions to the LLM. `search-filter` selects
candidates using embeddings before LLM judgment and requires a top-k or
similarity bound. `proxy-only` uses a similarity threshold instead of LLM pair
judgment and requires `min_similarity`.

For per-site settings, the JSON shape is `schema_version: 1` with a `bindings`
list; each binding has `site_id`, `mode`, `top_k`, and `min_similarity`.
Inventory the exact policy using
`inventory_operator_semantic_pair_sites` in
[`agent_memory_drivers.py`](../src/agent_memory/evaluation/agent_memory_drivers.py)
before selecting IDs. Site bindings are mutually exclusive with global
profile flags. Unknown or duplicate bindings are rejected.

## Storage

Claude's standard benchmark policy uses in-memory views. Mem0's driver creates
case-local embedded Qdrant state; it does not require a remote Qdrant key.
Zep requires a separately started Neo4j instance and the following variables:

| Variable | Requirement |
| --- | --- |
| `AGENT_MEMORY_NEO4J_URI` | Required connection URI |
| `AGENT_MEMORY_NEO4J_PASSWORD` | Required password |
| `AGENT_MEMORY_NEO4J_USER` | Defaults to `neo4j` when omitted |
| `AGENT_MEMORY_NEO4J_DATABASE` | Defaults to `neo4j` when omitted |
| `AGENT_MEMORY_NEO4J_IMAGE` | Required deployment image identifier |
| `AGENT_MEMORY_NEO4J_IMAGE_DIGEST` | Required actual image digest for provenance |

Image metadata records the deployment; setting it does not start a container.
Native Graphiti uses its own `NEO4J_*` configuration, not these prefixed keys.
Do not point unrelated experiments at the same mutable namespace.

Zep and Mem0 storage profiles pin BGE-M3 embeddings. First use may download
model weights. `--embedding-device` changes placement, not the model.
Typed storage bindings and search settings are defined in
[Zep storage](../src/agent_memory/memories/zep/storage.py) and
[Mem0 storage](../src/agent_memory/memories/mem0/storage.py).

## Cache, trace, and recovery

LOTUS's local cache and the provider's prompt cache are different mechanisms.
The benchmark cache setting does not disable provider cache. Process-local
entries do not survive restart. Keep actual provider usage distinct from
logical task counts when comparing runs.

Compact traces omit full intermediate snapshots; they can still contain source
text, prompts, responses, and personal information. Store results privately.
`full` is appropriate for bounded diagnostics, not a prerequisite for scoring.

Reissue the same benchmark command and output directory to use the harness's
checkpoint recovery; these CLIs do not expose a generic `--resume` flag.
Restore validates the input and execution contracts. Work after the last
durable boundary may repeat, so recovery is not a guarantee of zero duplicate
API charges. Never load an untrusted serialized checkpoint.

LOCOMO and LongMemEval offer `--maintenance-only` and
`--maintenance-checkpoint-output-dir` for separating maintenance from retrieval
and answering. Those flags are not exposed by the MemoryAgentBench CLI.
Use a new output directory for a changed condition. See the
[experiment guide](../tools/evaluation/README.md) for scoring and output paths.
