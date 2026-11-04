# Independent semantic grouping and composable maintenance

New `sem_groupby` calls produce a readable membership relation. By default they
use full recomputation on changed input. Composable maintenance is an explicit
rule choice; neither path uses an agent or auxiliary semantic state.

## Phase two: explicit composable rules

Stage two is implemented and locally validated. On 2026-11-04 the complete
offline suite passed 2,074 tests with 19 skipped; Ruff, configured Pyright and
diff checks passed. Expanding Pyright to the new modules, examples and tests
found test-only typing errors; those were corrected and all 67 focused cases
passed again. These files are now included in the configured CI type-check set.
The full run reported the existing LOTUS/Pydantic deprecation and a cost-estimate
warning from the fake-model cache test. Local validation used Python 3.12;
Python 3.13 remains a CI check. No live model was called for this closeout.

```python
rules = DifferentialRules(
    sem_agg_rule="agg-composable",
    sem_groupby_rule="group-composable",
)
```

Pass `rules` to `SemanticDataflow` or `Memory`. Existing defaults and the four
legacy fused grouped-aggregation rules are unchanged. Numeric count/sum/avg
still use signed algebraic state. Mixing relational and semantic descriptors in
one composable aggregate remains unsupported; this new descriptor path requires
semantic descriptors only. Existing legacy mixed-aggregate policies are unchanged.

Aggregation combines each affected group's previous result with its newly
inserted raw rows, keeping their respective field names and duplicates. It does
not first summarize the delta. Unchanged groups make no model calls. A group
with retractions is recomputed from its current input; an empty group disappears.

Grouping first groups the delta, then matches individual new records against
old records within the same partition. Single uses top-1 matching (listwise by
default); multiple uses pairwise matching. The deterministic merge writes matches
into old groups. Unmatched members form new groups according to their local
memberships: if r1 matches G but r2/r3 do not, only r1 enters G and r2/r3 form H.
Shared members never merge old groups, and only duplicate record/group pairs
are removed. Different occurrences of identical text remain different records.

Old group IDs survive insert-only updates. New IDs use the runtime's staged
identity allocator, so rollback and restoration do not require another counter.
A partition containing retractions is fully regrouped with new group identities;
other partitions continue incrementally. This is not an algorithm for semantic
reassignment, nor a source-delete API.

The first batch of each partition uses the existing C contract. Later local
delta grouping uses C=None; the final partition must have no more than C groups.
Exceeding C fails atomically rather than merging groups or silently regrouping.

The rules reuse existing prompts and parsers. They are explicitly selected
execution strategies, not guarantees that every instruction is composable.
Grouping may inspect all historical members for matching, and the runtime still
diffs complete results; this implementation is not overall O(delta).

### Review repairs

Partition lookup uses relational values, not their JSON spelling: `1` and `1.0`
refer to the same partition, and missing keys share one bucket. This applies to
both grouping and aggregation. JSON remains an identity serialization format,
not an equality test. Regressions cover composite keys, missing values, C-limit
rollback, previous aggregate reuse, and downstream fallback when the runtime
emits dtype-related replacement rows.

The configured grouping-to-join instruction rewriter runs during compilation.
Its result is stored in the maintenance plan and fingerprint; execution does not
create a fresh default rewriter. Different conditions cannot silently share a
restored snapshot. Existing legacy rewrite hooks and rules are unchanged.

### Separate Claude experiment

`examples/claude/composable_memory.py` defines `ComposableClaudeMemory`; the built-in
`ClaudeMemory` baseline is not changed. The experiment reuses the baseline's
extraction subtree and instructions, selects independent multiple membership,
and is run with both composable rules. Multiple membership is an explicit
experimental choice, not a claim of equivalence to legacy `membership=None`.

The gated `examples/claude/composable_smoke.py` accepts `--run --case topics` or
`--run --case update`, plus a new `--output` directory. It uses DeepSeek Flash,
zero retries, no local response cache, and full traces. Inputs are synthetic,
not benchmark examples. No external memory store is needed.

On 2026-11-04, the topics case issued 18 provider requests and returned three
topics/catalog entries after three writes. Allergy and vegetarian preference
remained separate topics. The update case issued 9 requests: an Atlas deadline
update matched its existing group, aggregated the previous result with the new
candidate, and retained one topic with the Monday date instead of Friday. Both
plans contained two composable nodes. Retrieval was executed in both cases;
the single-topic update needed no ranking model calls.

These runs do not establish general quality. In the topics case, one existing
LOTUS top-k comparison exhausted the 2048-token budget, returned empty text,
and the installed parser selected its default preference. All three rows were
returned, but that ranking is not validated. In the update case, generated memory
also asserted an external hard commitment absent from the input; this unsupported
detail was carried forward. Neither the retrieval adapter nor the baseline
instructions were patched to hide these observations.

Raw evidence: `/private/tmp/claude-composable-review-20261104/` and
`/private/tmp/claude-composable-update-20261104/`, each with `report.json` and
`trace/events.jsonl`. A successful runner status means execution completed,
not that every generated assertion or ranking was correct.

### Strict top-k follow-up

The smoke runner now explicitly disables thinking; Claude instructions, logical
queries, and the LOTUS ranking prompts/algorithms are unchanged. Pairwise ranking
validates completion status and response count using the shared structured-path
validator, then requires a valid Document 1/2 choice before LOTUS parses it.
ZS_COT requires its explicit final Answer choice. Truncation and invalid choices
raise without retry, guessed selections, output repair, or fallback. Listwise
ranking also checks completeness before its existing configured parse retries.
No new retry policy is introduced.

The scoped LM guard restores the original main/helper models on both success
and failure. It follows LOTUS's existing non-thread-safe global-settings contract;
it does not make concurrent configuration safe. Validated operator cache entries
use a separate namespace so old unchecked rankings cannot bypass validation;
the underlying LM cache and valid operator-cache reuse remain available. Provider
completion metadata is retained even when tracing is disabled.

Offline follow-up: 971 tests passed, four live-only cases skipped; Ruff, configured
Pyright, and diff checks passed. Additional Pyright covered the guard and smoke
runner. Tests exercise actual LOTUS naive/quick/heap sorters with fake physical
provider responses, invalid/truncated/cached answers, response counts, model
restoration, and valid ranking output. No mock result measures model quality.

Two bounded Flash reruns made 28 provider requests (18 topics, 10 update), all
with thinking disabled and normal stop status. All four top-k comparisons
returned legal choices; the three topics comparisons each used three completion
tokens. The update case nevertheless returned False for matching the new Atlas
deadline to its old record, leaving two topics with conflicting dates. This is
an observed semantic matching failure, not a truncated or illegally parsed
ranking. No instruction/query change or deduplication workaround was applied.
One run does not establish that disabling thinking caused the mismatch.

Evidence: `/private/tmp/claude-topk-strict-topics-20261104/` and
`/private/tmp/claude-topk-strict-update-20261104/`. The original runs above are
historical runs, not overwritten reruns. At stage-two closeout, these temporary
directories and the two original Claude trace directories were no longer
available. The counts and observations above record the earlier inspection,
not a fresh audit of accessible artifacts. The gated runner remains available
for an explicitly authorized repeat; no traces were reconstructed or fabricated.

## Default fallback contract

```python
from agent_memory import Source, SemanticDataflow
from agent_memory.planner.rules import DifferentialRules

source = Source({"message": "Customer complaint", "user": "Tenant"})
groups = source.sem_groupby(
    input_cols=["message"], instruction="Group complaints by the issues involved",
    membership="multiple", partition_by="user", C=2,
)
summaries = groups.sem_agg(output_cols=["summary"], instruction="Summarize the complaints")
flow = SemanticDataflow(source=source, views={"groups": groups, "summaries": summaries})
```

Groups are discovered from each complete partition. `labels` and `label_col`
are no longer accepted. This implementation uses one structured call per partition;
it does not silently
truncate or split oversized partitions. The returned memberships are validated
before publication. Invalid, incomplete or duplicate assignments fail the update.

`single` requires exactly one group per record; `multiple` requires at least one.
Shared records do not merge groups. Grouping retains original columns and adds
`_agent_memory_record_id` and `_agent_memory_group_id` only. It does not generate
group names or descriptions; similarly named business columns remain ordinary
input columns. Group IDs are snapshot-local,
not a promise of semantic identity across recomputations. Record IDs distinguish
identical-content occurrences. Metadata is not automatically aggregate input.

Both `groups.sem_agg(...)` and `groups.agg(sem_agg(...))` aggregate by group identity.
That identity is `partition_by + _agent_memory_group_id`. No group explanations
are generated or carried into aggregate results.

`filter`, `select` and `alias` preserve grouping. `select` returns exactly the
requested columns: omitting a partition key or group ID raises an error. Other
columns, including record IDs, may be projected away. Aggregate output columns
must not overwrite group keys. Other operations on the
grouping handle reject undefined scope propagation; an explicit relational
`group_by(...)` may replace the grouping.
`groups.count(...)`, `groups.sum(...)`, `groups.avg(...)` and numeric-only
`groups.agg(...)` use the existing algebraic IVM path, not semantic fallback.
Their group keys are the same identity keys as semantic aggregation. Shared
records contribute once per membership; retractions remove their contributions.
Sum and average over the same column share sum/non-null-count state.

Numeric state contains only group keys, counts and sums, with no description
payload or description counters. Equal opposite contributions cancel by group
keys and numeric inputs before updating sums, even when unrelated business
columns change. Surviving occurrences keep their order and types; invalid
numeric values are not hidden by cancellation. No historical membership scan is needed
downstream: updates consume existing per-group state and deltas. The executor
still copies/finalizes that group-state table; full grouping fallback and result
diffing are not O(delta).

Mixing numeric and semantic aggregate descriptors in a single `.agg(...)` remains
unsupported; use separate aggregations. The semantic branch still uses
`agg-fallback`. This does not add Agent, S or composable semantic maintenance.

Default aggregate input means all remaining business columns, not only the
columns used to assign groups. System columns and the query's generated grouping
metadata are excluded by default; explicitly requested existing columns are
respected. Business columns are not excluded just for sharing an internal-looking
prefix. `output_cols=None` uses the resolved inputs, excluding group keys on the
grouped path.

Standalone aggregation can opt into full recomputation with
`DifferentialRules(sem_agg_rule="agg-fallback")`, passed as `rules=` to `Memory`
or `SemanticDataflow`. Without this option, standalone aggregation retains its
previous maintenance behavior.
The option does not override legacy grouped or window aggregation. The four
legacy grouped rules can coexist with new grouping and standalone fallback in
one policy. Global fallback does not infer grouping from an incidental hidden
group ID column.
If a standalone aggregate leaves both input and output columns implicit, filter
and subsequent fallback aggregation resolve the bound schema during compilation.
Authoring `.count()` or a new `.sem_groupby()` on that result still requires an
explicit aggregate output schema; those APIs inspect columns before compilation.

Existing grouped-aggregation strategies remain available through explicit legacy
membership values (`None` or `"exclusive"`); `"overlapping"` remains unsupported
on that legacy path. Old repository callers explicitly preserve their prior mode.
New calls default to `"single"` and independent grouping. Using membership as a
legacy execution-path selector is temporary compatibility debt: migrate the
Claude/Zep callers and verify the four old rules before removing this selector.
Nonempty `S` remains unsupported. The legacy path rejects nonempty `C`.

## Target group count and model output

On the new path, `C=None` lets the task determine group count. Otherwise `C` must
be a positive integer, not a boolean. It is the target number of groups in each
partition, with at most C nonempty groups; fewer are accepted, not treated as
proof of semantic quality. Every nonempty partition must contain at least C
records, checked across all partitions before any model call. Empty input returns
zero rows with the complete schema. Multiple membership does not require each
record to join C groups. C participates in the plan fingerprint and every fallback.

The operator reuses the LOTUS default system formatter and existing structured
executor. It preserves the user instruction and adds only record numbering,
coverage, membership and target-count constraints. The model returns integer
memberships, not original text or descriptions:

```json
{"rows": [{"record": 0, "group": 0}, {"record": 1, "group": 7}]}
```

Record numbers are partition-local row positions. Group numbers are nonnegative
integers and need not be consecutive. Validation checks types, bounds, coverage,
duplicate pairs, membership cardinality and group count without coercing values
or repairing memberships. The program creates final IDs and constructs output
with batched row selection. Each partition's selected columns are serialized
once. Other operators keep their existing default structured-format examples.
The direct structured executor respects chat JSON and Responses JSON Schema
transport selection without replacing the LOTUS prompt. Explicitly incomplete
provider responses or wrong response counts fail before parsing or publication,
including retry responses; they do not trigger an extra implicit retry.

Updates replace full results and propagate both removals and additions atomically.
Snapshots include membership results. Semantic joins downstream of fallback
are rejected because their incremental executor cannot consume retractions.
Count/over windows also cannot consume fallback retractions. The grouping
contract, resolved inputs and selected rule participate in plan fingerprints;
incompatible snapshots (including pre-repair new-path plans) are rejected rather
than silently migrated.
Source deletion, state-assisted maintenance and agent planning are out of scope.

The regression examples are synthetic contract tests, not a benchmark. Offline
mock tests establish execution and validation behavior, not semantic accuracy.
Regression coverage includes transformed single/multiple membership relations,
both aggregate entry points, mixed legacy/new policies, input capture, historical
membership additions/removals, group disappearance, filtered-empty results,
rollback and restore-then-insert. Local fake LM responses also exercise the real
structured parser and adapters; no live model is needed for these tests.

The combined offline run after the aggregate output-boundary repair passed
1,111 tests with 16 real-model cases excluded.
Ruff and `git diff --check` passed. One existing LOTUS/Pydantic deprecation warning
remains. Target-count tests record C and observed nonempty group counts separately
in the JUnit report; none of these checks establish semantic quality.

## Aggregate output boundary

Leaf and intermediate-summary aggregation retain the LOTUS task instructions,
but replace its schematic Context/Instruction/Answer output example with:

```text
The Context and Instruction sections below provide the input and task.
Return only the result requested by the instruction, in the requested format.
Do not reproduce the input template as part of the result.
```

Single-column sequential aggregation now reuses the repository's existing
LOTUS-style hierarchical executor, also used for structured aggregation. It no
longer calls the installed LOTUS sem_agg directly; neither site-packages nor the
model are monkey-patched. Provider-batched aggregation uses the same template
functions. Existing prompt batching has its own structured-output protocol and
is unchanged. The user instruction, API, grouping identity and maintenance rules
are unchanged; no few-shot example or response-string cleanup is added. The
local executor's existing oversized-document guard also applies to this path.

An offline comparison captures the installed LOTUS and local hierarchical
prompts for the same input. Across leaf and merge levels, the only textual
change is the output-boundary block. Other regressions cover dispatch parity,
structured results, and preserving an explicitly requested Context heading.

A bounded live Flash run used eight synthetic task fixtures, two input batches
per task, plus standalone aggregations: 60 requests without cache or retries.
All 16 grouping snapshots and numeric counts matched the fixture expectations;
40 single-column outputs met the checked format constraints, including five
outputs intentionally starting with Context. Four multi-column results parsed
successfully. Exact-name arrays matched the expected names. The cases cover
complaints with overlap, Chinese project updates, dated preferences, names,
uncertain proposals, deadlines, scoped restrictions, and requested headings.
These observations are not benchmark accuracy or a general guarantee. Live
inputs were short; multi-level aggregation was checked offline, not with a live
long-context workload. Full requests and outputs are in
`/private/tmp/phase1-prompt-integrated-20261102T155654Z/`.

## Local construction check

A single local run with prebuilt, equivalent fake memberships compared the
previous executor with this version. Each input record has two memberships;
checks preserve business values and record/group relationships, ignoring the
intentionally removed generated explanations. Timing includes input serialization,
validation and result construction, with `tracemalloc` enabled. It excludes model
calls, prebuilt responses, runtime diffing and downstream aggregation.

| Memberships | Before (seconds) | After (seconds) | Before peak (MB) | After peak (MB) |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 0.056 | 0.017 | 0.92 | 0.49 |
| 10,000 | 0.528 | 0.114 | 8.07 | 3.95 |
| 100,000 | 5.610 | 1.135 | 79.41 | 38.47 |

Peaks are Python-tracked allocations, not total process RSS. These observations
are not timing assertions or evidence of model quality or whole-system speedup.
