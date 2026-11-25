"""Typed semantic state execution with deterministic membership expansion."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from agent_memory.maintenance import maintenance_target, object_schema
from agent_memory.planner.serialization import stable_value
from agent_memory.policy.grouping import GROUP_ID, RECORD_ID
from agent_memory.policy.logical import QueryExpr
from .composable import _parts
from .context import LotusExecutionContext
from .membership import validate_memberships
from .relational import _strict_json_records
from .structured import execute_schema_lm

STATE_COLUMNS = ("_scope", "_slot", "_result", "_state")
AGG_STATE_REQUIREMENTS = """Use New inputs, Previous result, and Previous state to produce two outputs:

- result: the user-visible output requested by Instruction.
- state: the complete auxiliary information for future updates,
  initialized or updated according to State maintenance.

Keep each output within the fields declared for it in the response schema.
Previous result and state describe retained history, not new observations."""
GROUP_STATE_REQUIREMENTS = """Use New inputs, Previous state, and Previous memberships to perform Instruction.
Initialize or update state according to State maintenance.
Return complete state and assignments for the declared grouping objects.
The object field is the zero-based position in the updated state's bound object array;
group is a nonnegative local integer. Existing assignments may change.
Evidence references are not memberships. Follow Member binding and Constraints.
Every grouping object must be assigned; after expansion every active record must
have exactly one group for single membership, or at least one for multiple.
Return only assignments and state using the response schema."""


def _continue_group_ids(
    members: Mapping[int, set[str]],
    previous: list[dict[str, str]],
    generation: str,
) -> dict[int, str]:
    """Reserve unchanged groups, then greedily match largest old-member overlaps."""
    if not previous:
        return {group: f"{generation}:{group}" for group in members}
    old: dict[str, set[str]] = defaultdict(set)
    for row in previous:
        old[row["group_id"]].add(row["record_id"])
    identical: dict[frozenset[str], list[str]] = defaultdict(list)
    for gid in sorted(old, reverse=True):
        identical[frozenset(old[gid])].append(gid)
    # Member identities break ties independently of model-assigned local numbers.
    order = sorted(members, key=lambda group: (tuple(sorted(members[group])), group))
    rank = {group: index for index, group in enumerate(order)}
    result: dict[int, str] = {}
    for group in order:
        candidates = identical.get(frozenset(members[group]), [])
        if candidates:
            result[group] = candidates.pop()
    used = set(result.values())
    by_record: dict[str, list[str]] = defaultdict(list)
    for gid, records in old.items():
        if gid not in used:
            for record in records:
                by_record[record].append(gid)
    # Index shared records rather than scanning the Cartesian product of groups.
    overlaps: Counter[tuple[str, int]] = Counter()
    for group in order:
        if group not in result:
            for record in members[group]:
                for gid in by_record.get(record, ()):
                    overlaps[gid, group] += 1
    for gid, group in sorted(
        overlaps, key=lambda pair: (-overlaps[pair], pair[0], rank[pair[1]])
    ):
        if gid not in used and group not in result:
            result[group] = gid
            used.add(gid)
    return {group: result.get(group, f"{generation}:{group}") for group in members}


def expand_assignments(
    state: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    *,
    binding: Mapping[str, str],
    active_ids: list[str],
    previous: list[dict[str, str]],
    membership: str,
    C: int | None,
    generation: str,
) -> list[dict[str, str]]:
    """Expand only declared member links; evidence links never propagate membership."""
    objects = state[binding["objects"]]
    rows = [{"record": row["object"], "group": row["group"]} for row in assignments]
    positions, groups = validate_memberships(rows, len(objects), "multiple", C)
    if not active_ids or len(set(active_ids)) != len(active_ids):
        raise ValueError("group-state requires unique active record identities")
    members: list[tuple[int, str]] = []
    active = set(active_ids)
    for index, obj in enumerate(objects):
        ids = obj[binding["record_ids"]]
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(active):
            raise ValueError(
                "state member binding contains empty, duplicate or unknown references"
            )
        members.extend((index, record) for record in ids)
    mapped = (
        pd.DataFrame({"object": positions, "group": groups})
        .merge(
            pd.DataFrame(members, columns=pd.Index(["object", "record_id"])),
            on="object",
            how="inner",
        )
        .drop_duplicates(["record_id", "group"])
    )
    counts = mapped.groupby("record_id", sort=False).size()
    if set(counts.index) != active or (membership == "single" and (counts != 1).any()):
        raise ValueError("expanded memberships violate record coverage or cardinality")
    group_members: dict[int, set[str]] = defaultdict(set)
    for group, record in mapped[["group", "record_id"]].itertuples(index=False, name=None):
        group_members[group].add(record)
    group_ids = _continue_group_ids(group_members, previous, generation)
    mapped["group_id"] = mapped["group"].map(group_ids.__getitem__)
    return mapped.loc[:, ["record_id", "group_id"]].to_dict("records")


def execute_stateful(
    plan: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> pd.DataFrame:
    """Process affected scopes only, rebuilding both result and state on retraction."""
    query = plan.inputs[0]
    grouping = query.op == "sem_groupby"
    parent = plan.params["parent"]
    old = inputs[plan.params["previous"]]
    current = inputs[parent]
    inserted, retracted = inputs[f"{parent}__inserted"], inputs[f"{parent}__retracted"]
    keys = (
        tuple(query.params["partition_by"])
        if grouping
        else (
            tuple(query.inputs[0].params["keys"])
            if query.inputs[0].op == "group_by"
            else ()
        )
    )
    additions, removals = _parts(inserted, keys), _parts(retracted, keys)
    affected = list(dict.fromkeys([*additions, *removals]))
    full = _parts(current, keys) if grouping or removals else {}
    previous = {
        (tuple(row["_scope"]), row["_slot"]): row for row in old.to_dict("records")
    }
    affected_set = set(affected)
    output = [row for (key, _), row in previous.items() if key not in affected_set]
    if grouping and query.params.get("C") is not None:
        if any(
            0 < len(full[key]) < query.params["C"] for key in affected if key in full
        ):
            raise ValueError(
                "semantic grouping partition has fewer records than target C"
            )
    for scope_index, key in enumerate(affected):
        rebuilding = key in removals
        frame = full.get(key) if rebuilding else additions.get(key)
        if frame is None or frame.empty:
            continue
        active_ids = full[key].index.astype(str).tolist() if grouping else []
        for slot, config in enumerate(plan.params["plans"]):
            before = None if rebuilding else previous.get((key, slot))
            task = maintenance_target(query, slot)
            selected = tuple(task.params["input_cols"])
            state_schema = stable_value(config["state_schema"])
            payload = {
                "New inputs": _strict_json_records(frame, selected),
                "Previous state": None if before is None else before["_state"],
                "State maintenance": config["maintenance_instruction"],
            }
            if grouping:
                payload["New inputs"] = [
                    {"record_id": str(record_id), "values": row}
                    for record_id, row in zip(
                        frame.index, payload["New inputs"], strict=True
                    )
                ]
                payload["Previous memberships"] = (
                    [] if before is None else before["_result"]
                )
                payload["Member binding"] = state_schema["membership"]
                payload["Constraints"] = {
                    "membership": query.params["membership"],
                    "target_C": query.params.get("C"),
                    "scope": "This input contains one partition only.",
                    "C_semantics": "Aim for target_C groups when set; fewer nonempty groups are allowed, but do not exceed it.",
                }
                result_schema = {
                    "type": "array",
                    "items": object_schema(
                        {
                            "object": {"type": "integer", "minimum": 0},
                            "group": {"type": "integer", "minimum": 0},
                        }
                    ),
                }
                response_schema = object_schema(
                    {"assignments": result_schema, "state": state_schema["value"]}
                )
                requirements = GROUP_STATE_REQUIREMENTS
            else:
                payload["Previous result"] = (
                    None if before is None else before["_result"]
                )
                result_schema = object_schema(
                    {
                        col.name: {
                            "type": ["string", "number", "boolean", "null"],
                            "description": col.description or col.name,
                        }
                        for col in task.params["output_cols"]
                    }
                )
                response_schema = object_schema(
                    {"result": result_schema, "state": state_schema["value"]}
                )
                requirements = AGG_STATE_REQUIREMENTS
            context.configure()
            result: Any = None

            def validate_memberships(value: dict[str, Any]) -> None:
                nonlocal result
                result = expand_assignments(
                    value["state"], value["assignments"],
                    binding=state_schema["membership"], active_ids=active_ids,
                    previous=previous.get((key, slot), {}).get("_result", []),
                    membership=query.params["membership"], C=query.params.get("C"),
                    generation=f"{inputs['__group_generation']}:{scope_index}",
                )

            answer = execute_schema_lm(
                context=payload,
                instruction=str(task.params["instruction"]),
                requirements=requirements,
                schema=response_schema,
                config=context.config,
                operator="group-state" if grouping else "agg-state",
                validate_output=validate_memberships if grouping else None,
                max_retries=context.config.maintenance_parse_retries,
            )
            if not grouping:
                result = answer["result"]
            output.append(
                {
                    "_scope": list(key),
                    "_slot": slot,
                    "_result": result,
                    "_state": answer["state"],
                }
            )
    return pd.DataFrame(output, columns=pd.Index(STATE_COLUMNS))


def finalize_stateful(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> pd.DataFrame:
    """Publish only user results, joining original payloads once for grouping."""
    state = execute(query.inputs[0], inputs)
    columns = list(query.params["columns"])
    if state.empty:
        return pd.DataFrame(columns=pd.Index(columns))
    if query.params["grouping"]:
        pairs = pd.DataFrame(
            [pair for result in state["_result"] for pair in result],
            columns=pd.Index(["record_id", "group_id"]),
        ).rename(columns={"record_id": RECORD_ID, "group_id": GROUP_ID})
        source = execute(query.inputs[1], inputs)
        payload = source.assign(**{RECORD_ID: source.index.astype(str)})
        return pairs.merge(
            payload, on=RECORD_ID, how="inner", validate="many_to_one"
        ).loc[:, columns]
    keys = query.params["group_keys"]
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in state.to_dict("records"):
        key = tuple(item["_scope"])
        row = rows.setdefault(key, dict(zip(keys, key, strict=True)))
        row.update(item["_result"])
    return pd.DataFrame(list(rows.values()), columns=pd.Index(columns))
