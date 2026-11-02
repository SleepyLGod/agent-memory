# Independent semantic grouping (phase one)

New `sem_groupby` calls produce a readable membership relation. They use full
recomputation on changed input, not an agent or a composable maintenance rule.

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
