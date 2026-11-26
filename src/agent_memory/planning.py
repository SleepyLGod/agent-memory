"""Opt-in Agent planning, separate from compilation and batch execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .maintenance import MaintenancePlan, maintenance_target, object_schema
from .planner.serialization import stable_value
from .planning_prompts import (
    AGG_INSTRUCTION,
    GROUP_INSTRUCTION,
    planning_examples,
    planning_requirements,
)
from .policy.logical import ColumnSpec, QueryExpr
from .policy.relation import Relation
from .policy.schema import output_columns

if TYPE_CHECKING:
    from .adapters.lotus.adapter import LotusAdapter


def available_rules(grouping: bool) -> dict[str, str]:
    """Describe actual supported input roles, not just rule names."""
    if grouping:
        return {
            "group-fallback": "Read all current old and new records in the partition and regroup them under the original criterion. Save no S.",
            "group-composable": "Group newly added records, then match each new record to historical records in the same partition under the original criterion. All historical records in that partition are candidates, using the specified input columns; there is no representative-record selection. Matching compares records, not category definitions or group summaries; supplied categories guide the same-group judgment, not a separate classify-then-group-by-label operation. With single membership, select at most one matching historical record; with multiple membership, retain all positive pairwise matches. A matched record joins the groups of its matched historical records. Only unmatched new records form new groups from their local memberships. Old memberships remain unchanged. Save no S.",
            "group-state": "Read new records, previous S, and previous memberships. Process the complete S of the affected partition and return complete updated S and assignments for all its grouping objects; there is no selective processing of a subset of state objects. Old assignments may change under the original criterion. The program expands assignments through member record IDs and joins the original business columns back into the output. Old input content is not separately reread by the model.",
        }
    return {
        "agg-fallback": "Read all current old and new records in this aggregation and recompute the result under the original instruction. Save no S.",
        "agg-composable": "Read the previous result and newly added records. Apply the original task to these inputs and save the updated result. New records are not aggregated separately first. No S is saved, and historical input records are not reread.",
        "agg-state": "Read the previous result, previous S, and new records. Update S and compute the updated result. Save both. This rule can read information saved in S, including text, but cannot separately reread old input records.",
    }


def _column_descriptions(query: QueryExpr) -> dict[str, str | None]:
    """Expose declared column meanings without inventing runtime data types."""
    inherited = _column_descriptions(query.inputs[0]) if len(query.inputs) == 1 else {}
    declared = {
        col.name: col.description
        for field in ("columns", "output_cols")
        for col in query.params.get(field) or ()
        if isinstance(col, ColumnSpec)
    }
    return {
        name: declared.get(name, inherited.get(name)) for name in output_columns(query)
    }


def plan_maintenance(
    target: Relation | QueryExpr,
    *,
    adapter: LotusAdapter,
    examples: Sequence[Mapping[str, Any]] = (),
    aggregate_index: int = 0,
) -> MaintenancePlan:
    """Explicitly call the planning LM once and bind its validated configuration."""
    from .adapters.lotus.structured import execute_schema_lm

    task = maintenance_target(target, aggregate_index)
    grouping = task.op == "sem_groupby"
    parent = task.inputs[0]
    source = parent.inputs[0] if parent.op == "group_by" else parent
    input_schema = _column_descriptions(source)
    output_schema = (
        _column_descriptions(task)
        if grouping
        else {col.name: col.description for col in task.params["output_cols"]}
    )
    input_descriptions = {
        name: input_schema[name]
        for name in task.params["input_cols"]
        if input_schema[name] is not None
    }
    output_descriptions = {
        name: description
        for name, description in output_schema.items()
        if description is not None
    }
    group_keys = list(parent.params["keys"]) if parent.op == "group_by" else []
    if grouping:
        scope = {
            "Constraints": {
                key: stable_value(task.params.get(key))
                for key in ("partition_by", "membership", "C")
            }
        }
    else:
        scope = {
            "Aggregation scope": "One result per group; groups are maintained independently."
            if group_keys
            else "One result over all input records.",
            "Result responsibility": "The model generates only Output columns. The program preserves and adds the aggregation group keys, if any.",
            **({"Constraints": {"group_keys": group_keys}} if group_keys else {}),
        }
    context = {
        "Operator": task.op,
        "Original task": task.params["instruction"],
        "Input columns": list(task.params["input_cols"]),
        **(
            {"Input column descriptions": input_descriptions}
            if input_descriptions
            else {}
        ),
        "Output columns": list(output_schema),
        **(
            {"Output column descriptions": output_descriptions}
            if output_descriptions
            else {}
        ),
        **scope,
        "Available rules": available_rules(grouping),
        # Formatter example turns repeat the whole task; embed the designs once.
        "Design examples": [
            {"Task": example_task, "Plan": plan}
            for example_task, plan in planning_examples(grouping=grouping)
        ],
        "Development inputs": list(examples),
        "Limits": {
            "max_generated_tokens_per_response": adapter.config.structured_max_tokens
        },
    }
    response_schema = object_schema(
        {
            "rule": {"type": "string", "enum": list(context["Available rules"])},
            "state_schema": {"type": ["object", "null"]},
            "maintenance_instruction": {"type": ["string", "null"]},
            "rationale": {"type": "string"},
        }
    )
    adapter._context.configure()
    config = execute_schema_lm(
        context=context,
        instruction=GROUP_INSTRUCTION if grouping else AGG_INSTRUCTION,
        requirements=planning_requirements(grouping=grouping),
        schema=response_schema,
        config=adapter.config,
        operator="maintenance-planning",
    )
    return MaintenancePlan.bind(target, config, aggregate_index=aggregate_index)
