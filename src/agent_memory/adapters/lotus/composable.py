"""Explicit result-and-delta maintenance without auxiliary semantic state."""

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any

import pandas as pd

from agent_memory.planner.serialization import stable_json
from agent_memory.policy.grouping import GROUP_ID, RECORD_ID
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.schema import output_columns
from .context import LotusExecutionContext
from .membership import execute_memberships
from .sem_agg import (
    aggregate_group_text,
    aggregate_output_columns,
    execute_native_sem_agg_groups,
    execute_structured_sem_agg_groups,
)
from .sem_join import execute_sem_join


def _parts(
    frame: pd.DataFrame, keys: Sequence[str]
) -> dict[tuple[Any, ...], pd.DataFrame]:
    if frame.empty:
        return {}
    if not keys:
        return {(): frame}
    result = {}
    for values, part in frame.groupby(
        list(keys), sort=False, dropna=False, observed=True
    ):
        values = values if isinstance(values, tuple) else (values,)
        # Preserve relational equality (including 1 == 1.0); JSON spellings
        # are serialization, not group identity. Missing keys share one bucket.
        result[tuple(None if bool(pd.isna(v)) else v for v in values)] = part
    return result


def _leaf(name: str, frame: pd.DataFrame) -> QueryExpr:
    return QueryExpr(
        op="materialized_view", params={"name": name, "columns": tuple(frame.columns)}
    )


def execute_composable(
    plan: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> pd.DataFrame:
    """Execute the compiler-selected rule against transaction-local bindings."""
    query = plan.inputs[0]
    parent = str(plan.params["parent"])
    old = inputs[str(plan.params["previous"])]
    current = inputs[parent]
    inserted = inputs[f"{parent}__inserted"]
    retracted = inputs[f"{parent}__retracted"]
    if plan.params["rule"] == "group-composable":
        return _group(
            query,
            old,
            current,
            inserted,
            retracted,
            str(inputs["__group_generation"]),
            context,
        )
    return _aggregate(query, old, current, inserted, retracted, context)


def _aggregate(
    query: QueryExpr,
    old: pd.DataFrame,
    current: pd.DataFrame,
    inserted: pd.DataFrame,
    retracted: pd.DataFrame,
    context: LotusExecutionContext,
) -> pd.DataFrame:
    carrier = query.inputs[0]
    keys = tuple(carrier.params["keys"]) if carrier.op == "group_by" else ()
    previous = _parts(old, keys)
    additions, removals = _parts(inserted, keys), _parts(retracted, keys)
    # Full input is inspected only when a retraction needs recomputation.
    full = _parts(current, keys) if removals else {}
    affected = list(dict.fromkeys([*additions, *removals]))
    groups, prior, identities = [], [], []
    for key in affected:
        frame = full.get(key) if key in removals else additions.get(key)
        if frame is None or frame.empty:
            continue
        groups.append(frame)
        prior.append(None if key in removals else previous.get(key))
        identities.append({name: frame.iloc[0][name] for name in keys})
    columns = list(output_columns(query))
    affected_set = set(affected)
    retained = [frame for key, frame in previous.items() if key not in affected_set]
    if groups:
        context.configure()
        if query.op == "sem_agg":
            tasks = [query]
        else:
            tasks = [
                QueryExpr(
                    op="sem_agg",
                    inputs=query.inputs,
                    params={
                        "instruction": spec.instruction,
                        "input_cols": spec.input_cols,
                        "output_cols": spec.output_cols,
                    },
                )
                for spec in query.params["aggregates"]
            ]
        for task in tasks:
            selected = tuple(task.params["input_cols"])
            outputs = aggregate_output_columns(task, selected)
            documents = []
            for frame, before in zip(groups, prior, strict=True):
                docs = (
                    []
                    if before is None
                    else aggregate_group_text(before, [col.name for col in outputs])
                )
                documents.append(docs + aggregate_group_text(frame, selected))
            if len(outputs) == 1:
                values = execute_native_sem_agg_groups(
                    task, groups, selected, context.config, documents=documents
                )
                for row, value in zip(identities, values, strict=True):
                    row[outputs[0].name] = value
            else:
                values = execute_structured_sem_agg_groups(
                    task, groups, selected, outputs, context.config, documents=documents
                )
                for row, value in zip(identities, values, strict=True):
                    row.update(value)
        retained.append(pd.DataFrame(identities, columns=pd.Index(columns)))
    return (
        pd.concat(retained, ignore_index=True)
        if retained
        else pd.DataFrame(columns=pd.Index(columns))
    )


def merge_memberships(
    old: pd.DataFrame,
    local: pd.DataFrame,
    matches: pd.DataFrame,
    generation: str,
    membership: str,
) -> pd.DataFrame:
    """Insert row-level matches and unmatched local groups, without model calls."""
    pairs = matches.loc[:, [RECORD_ID, GROUP_ID]].drop_duplicates()
    if not set(pairs[RECORD_ID]).issubset(set(local[RECORD_ID])):
        raise ValueError("match references unknown new record")
    if not set(pairs[GROUP_ID]).issubset(set(old[GROUP_ID])):
        raise ValueError("match references unknown old group")
    payload = local.drop(columns=[GROUP_ID]).drop_duplicates(subset=[RECORD_ID])
    matched = payload.merge(pairs, on=RECORD_ID, how="inner", validate="one_to_many")
    unmatched = local.loc[~local[RECORD_ID].isin(pairs[RECORD_ID])].copy()
    mapping = {
        group: json.dumps([generation, group]) for group in unmatched[GROUP_ID].unique()
    }
    unmatched[GROUP_ID] = unmatched[GROUP_ID].map(mapping)
    added = pd.concat([matched, unmatched], ignore_index=True).drop_duplicates(
        [RECORD_ID, GROUP_ID]
    )
    counts = added.groupby(RECORD_ID, sort=False).size()
    if set(counts.index) != set(local[RECORD_ID]) or (
        membership == "single" and (counts != 1).any()
    ):
        raise ValueError("merge violates membership coverage or cardinality")
    if set(mapping.values()) & set(old[GROUP_ID]):
        raise ValueError("new group identity collision")
    return pd.concat([old, added.loc[:, old.columns]], ignore_index=True)


def _group(
    query: QueryExpr,
    old: pd.DataFrame,
    current: pd.DataFrame,
    inserted: pd.DataFrame,
    retracted: pd.DataFrame,
    generation: str,
    context: LotusExecutionContext,
) -> pd.DataFrame:
    keys = tuple(query.params["partition_by"])
    previous = _parts(old, keys)
    additions, removals = _parts(inserted, keys), _parts(retracted, keys)
    full = _parts(current, keys) if removals else {}
    affected = list(dict.fromkeys([*additions, *removals]))
    affected_set = set(affected)
    result = [frame for key, frame in previous.items() if key not in affected_set]
    columns = list(query.params["columns"])
    for key in affected:
        before = previous.get(key, pd.DataFrame(columns=pd.Index(columns)))
        fallback = key in removals
        data = full.get(key) if fallback else additions.get(key)
        if data is None or data.empty:
            continue
        initial = before.empty or fallback
        local_query = QueryExpr(
            op=query.op,
            inputs=query.inputs,
            params={
                **query.params,
                "C": query.params.get("C") if initial else None,
            },
        )
        local = execute_memberships(local_query, data, context)
        namespace = json.dumps([generation, stable_json(key)])
        if initial:
            local[GROUP_ID] = local[GROUP_ID].map(
                lambda value: json.dumps([namespace, value])
            )
            updated = local
        else:
            selected = list(query.params["input_cols"])
            left = data.loc[:, selected].copy()
            left.index = left.index.map(str)
            right_rows = before.drop_duplicates(subset=[RECORD_ID]).set_index(RECORD_ID)
            right = right_rows.loc[:, selected]
            left_id, right_id = "__new_match_id", "__old_match_id"
            while left_id in selected or right_id in selected:
                left_id += "_"
                right_id += "_"
            join = QueryExpr(
                op="sem_join",
                inputs=(_leaf("left", left), _leaf("right", right)),
                params={
                    "instruction": query.params["join_instruction"],
                    "how": "left",
                    "k": 1 if query.params["membership"] == "single" else None,
                    "id_columns": (left_id, right_id),
                },
            )
            matched = execute_sem_join(
                join,
                {"left": left, "right": right},
                lambda q, bindings: bindings[q.params["name"]],
                context,
            )
            matched = matched.dropna(subset=[right_id])
            pairs = matched.loc[:, [left_id, right_id]].merge(
                before.loc[:, [RECORD_ID, GROUP_ID]],
                left_on=right_id,
                right_on=RECORD_ID,
            )
            pairs = pairs.loc[:, [left_id, GROUP_ID]].rename(
                columns={left_id: RECORD_ID}
            )
            updated = merge_memberships(
                before, local, pairs, namespace, str(query.params["membership"])
            )
        C = query.params.get("C")
        if C is not None and updated[GROUP_ID].nunique() > C:
            raise ValueError("composable grouping exceeds target C")
        result.append(updated)
    return (
        pd.concat(result, ignore_index=True)
        if result
        else pd.DataFrame(columns=pd.Index(columns))
    )
