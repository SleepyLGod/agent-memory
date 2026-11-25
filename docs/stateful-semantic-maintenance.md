# Stage three: explicit planning and auxiliary state

This is an opt-in execution interface, not a claim that an Agent finds sufficient
state for every task. Public query signatures, original instructions, defaults,
and all four legacy rules remain unchanged. `S` is not a configuration parameter:
non-null public `S` is still rejected.

## Planning, binding, execution

```python
import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.planner.rules import DifferentialRules

source = am.Source({"message": "Original message"})
answer = source.sem_agg(
    output_cols=["answer"],
    instruction="List names mentioned in at least two records.",
)

# Explicit model call; execute only with an approved provider and budget.
adapter = LotusAdapter(model="deepseek/flash")
plan = am.plan_maintenance(answer, adapter=adapter, examples=[])
config = plan.to_dict()  # JSON-serializable; persist with the application query

# Loading a saved configuration does not call the Agent.
bound = am.MaintenancePlan.bind(answer, config)
flow = am.SemanticDataflow(
    source=source, views={"answer": answer}, adapter=adapter,
    rules=DifferentialRules(maintenance_plans=(bound,)),
)
```

The same `rules` argument works with `Memory`. The program binds the original
query and optional `aggregate_index`; the Agent cannot select node IDs.
Unbound operators use their existing rules. Unused or duplicate bindings fail.
Plans are frozen, and their configuration participates in the plan fingerprint.
Compilation, execution and snapshot restoration never call the Agent again.

For an `agg(...)` with several semantic descriptors, bind every descriptor using
`aggregate_index=0, 1, ...`. This version requires the same maintenance rule for
all of them, but each has its own schema, instruction, previous result and S.
Mixed numeric/semantic descriptors, window targets and legacy fused targets are
rejected for this new interface; their existing execution paths are unchanged.

## What the Agent returns

Input: the original task, declared input/output columns and their meanings, partition/group
constraints, available rule behavior, execution limits and optional development
inputs. Column descriptions are not runtime type schemas. For aggregation, output
columns contain only model-generated result fields; group keys are supplied
separately and added back by the program. No task data or gold answers are loaded
automatically; development inputs are empty unless explicitly supplied.

The response has exactly four fields:

| Field | Meaning |
|---|---|
| `rule` | An available fallback, composable or state rule for the target operator |
| `state_schema` | Null, or the typed state schema and optional member binding below |
| `maintenance_instruction` | Null, or field meanings and how to initialize, update and answer |
| `rationale` | Brief reasoning and material assumptions, not a proof |

If no auxiliary state is needed, both state fields must be null.

```json
{
  "value": {"type": "object", "properties": {}, "required": [], "additionalProperties": false},
  "membership": null
}
```

`value` is a JSON Schema, not the live S. Supported state values include nested
records, typed lists and scalars, optionally nullable. Objects declare all fields
as required and disallow undeclared fields. `$ref`, executable code and untyped
blobs are not supported. The accepted schema stays fixed across updates; only its
contents change. Live S is created during the first operator execution.

For `group-state`, `membership` instead names a top-level object array and its
member-ID field, for example `{"objects":"items","record_ids":"members"}`.
An item can represent one or several original records. Evidence citations in
other fields do not grant membership. The binding is part of the frozen plan;
the schema's field contents remain Agent-designed. Entity/property and
edge/property tables are possible representations, not mandatory graph storage.

## Operator contracts and prompts

Agent instructions and design examples live in `planning_prompts.py`;
`planning.py` supplies the query context, supported rules and response schema.
Aggregation and grouping have separate planning instructions and examples, but
use the same Agent implementation. Both prefer composable maintenance when
adequate. State requires a concrete maintenance benefit that justifies reading
and returning it; fallback is the last choice. State instructions ask for needed
information, consolidation of redundancy and removal of no-longer-needed content,
without discarding task-required distinctions. The original Ia/Ig is unchanged.

Both state paths use the existing LOTUS map formatter and structured transport.
There is one response schema and strict local validation. State execution does
not repair outputs or implicitly read historical input. Optional validation
retries use the existing structured executor; the default is zero. Old scalar
parsing is untouched.

The state execution prompt consists of the unchanged Ia/Ig, input-role context,
a short operator-specific state fragment, and the output schema. The Agent's
`maintenance_instruction` is supplied in context, not written into the query.
State-design few-shots appear only in the planning prompt.
One planner selects examples for the target operator: aggregation uses a no-state
example, counts, and relational facts; grouping uses stable no-state assignment
and historical reassignment with state. Each example follows the target's rule
schema. These are illustrative choices, not fixed state templates. The public
`examples` argument supplies development inputs, not these built-in few-shots.

| Rule | Input roles | Model response |
|---|---|---|
| `agg-state` | New records, previous answer, previous S, state-maintenance instruction | `result` plus complete `state` |
| `group-state` | New records with IDs, previous S, previous memberships, member binding and constraints | Complete `state` plus object/group `assignments` |

The aggregation fragment explicitly says that previous answer and state are
retained history, not two additional sets of observations. Aggregation results
retain the existing scalar-column contract; nesting is enabled for S.

The grouping fragment specifies zero-based object positions in the *new* S,
nonnegative temporary group numbers, and complete assignments. Program code
expands declared member links, checks coverage, partition, single/multiple and C,
allocates final IDs, deduplicates membership pairs and joins business columns.
It never asks the model to perform relational merge or allocate framework IDs.

Group identity is assigned deterministically within each partition, independently
of the model's temporary numbers. Unchanged member sets reserve their old IDs
first. Remaining old/new pairs are considered by decreasing number of shared
record IDs; each old ID and each new group can be matched only once. Ties use
old-ID order, then sorted member-ID order (temporary number only for identical
new member sets). Unmatched new groups receive fresh transaction-scoped IDs.
This is a greedy identity policy, not a global maximum-weight assignment or a
claim about semantic accuracy. Overlaps are indexed by record ID, not calculated
by scanning every pair of groups; multiple memberships can still create many
overlapping pairs. No additional identity state is persisted.

An insertion into G therefore keeps G. A split keeps G on the side with more of
G's old records; new arrivals do not increase this overlap score. A merge keeps
the larger contributing old group when it has not already been matched. Shared
members count once per pair, not once per state object. Complex simultaneous
splits/merges use the same one-to-one policy, with unchanged groups reserved first.
Retractions rebuild grouping S, but existing memberships remain available to the
program for identity assignment; they are not reused as semantic state.

Downstream aggregation retains one result/S per group and descriptor. Pure
additions reuse them; any retraction rebuilds that affected group's result/S,
even when its ID survives. Empty groups clear without a model call. A merge does
not combine two auxiliary states: the surviving group receives only newly added
member records, and the other group's state is removed. Existing snapshots keep
their stored IDs and state format; subsequent updates use this identity policy.

Single requires one final membership per record; multiple requires at least
one. A coarse object containing two records cannot split them merely by receiving
two assignments: that assigns *both* records to both groups. The state must retain
separable objects and adequate evidence for a genuine split.

## State and propagation

The compiler creates a hidden state node and a public result projection. Each
hidden row contains one scope, descriptor slot, previous result and S. Grouping
stores membership pairs separately from Agent-designed S. The member mapping
does not require copied business fields, although a plan may retain text in S.
The public projection supplies original business columns from the current input.

Only affected groups/partitions invoke the model. Initial execution and later
updates share the same state rule, regardless of whether S itself is empty/null.
Any upstream retraction rebuilds that scope's result and S together from current
input. Empty scopes clear both without a model call. Source deletion is not added.

The runtime reuses staged node state, occurrence IDs, result differences, atomic
commit and snapshot recovery. S-only changes persist without creating a public
delta. Nested old state is not mutated in place. Changed plan configurations
reject old snapshots rather than silently migrating state. Numeric downstream
count/sum/avg consume the existing signed member deltas.
Legacy v1 snapshots cannot restore a bound plan: rebuilding a legacy executor
would discard its rule/configuration. Unbound legacy policies retain their
existing checkpoint compatibility.

Complete S, state-table traversal, membership expansion and full-result diffing
still have costs. This is not overall O(delta). Oversized inputs or explicitly
incomplete outputs fail; there is no silent truncation or automatic rule switch.
The shared completeness check also covers native text aggregation, including
intermediate summaries and cross-group batches. It rejects reported truncation
and response-count mismatches, without guessing completeness from text. These
failures stop before validation retries or further aggregation.
State-schema validation does not prove that the retained information is sufficient.

## Validation scope

Local validation on 2026-11-05 (Python 3.12): 2,122 tests passed and 5 skipped,
including 49 new state-maintenance cases. Real LOTUS and Neo4j integration files
were excluded and the live-model switch was disabled. Ruff and configured
Pyright passed. Existing LOTUS/Pydantic and fake-model cost-estimation warnings
remain. These are historical results for that revision, not a test count or
live-provider guarantee for the current tree.

`tests/test_stateful_maintenance.py` replaces only the model boundary and exercises
the real formatter, parser, adapter, compiler and runtime. It includes initialization,
empty answers with nonempty S, S-only updates, descriptor/partition isolation,
duplicate records, historical reassignment, numeric retractions, invalid responses,
downstream rollback and restoration. Coarse-member expansion has an explicit
counterexample test. Existing legacy-rule regressions remain in the full suite.

Run `test_agent_state_plan_and_rendered_prompts` with pytest `-s` to export the
actual rendered Agent and operator messages to its printed temporary directory.
These are scripted offline fixtures, not provider traces or model-quality results.
These engineering checks do not measure semantic quality or token savings.
Real-data experiments require a separate model configuration and call budget.
Feedback optimization and automatic schema migration are not implemented.
