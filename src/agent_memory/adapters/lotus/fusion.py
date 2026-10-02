"""Joint target selection and state synthesis, with deterministic writeback."""

from collections.abc import Callable, Mapping
from dataclasses import replace
import json
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
)
from agent_memory.adapters.lotus.sem_join import (
    assemble_join_frame,
    join_series,
    semantic_join_pair_candidates,
)
from agent_memory.adapters.lotus.sem_topk_join import _apply_pair_profile
from agent_memory.adapters.lotus.structured import execute_structured_lm_retry_result
from agent_memory.planner.physical import replace_query
from agent_memory.planner.rules import JOIN_MAP_BINDING_NAME, JOIN_MAP_RIGHT_ID_COLUMN
from agent_memory.policy.aggregates import SemanticAggregateSpec
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.tracing.semantic import query_digest, write_trace_event

_STATE_BINDING = "__fused_target_states"
_SYSTEM = (
    "Resolve the supplied semantic join and state-consolidation tasks jointly. "
    "For every incoming ID choose one eligible target ID or null. Copy the target "
    "ID exactly from that incoming item's eligible_targets list; never use a target "
    "ID from another incoming item. Select only true "
    "matches, not merely related targets. If several match, choose the strongest. "
    "For each selected target, consolidate its old state and ALL incoming states "
    "assigned to it, exactly once, following the consolidation instruction. "
    "Do not generate states for unselected targets. Do not edit IDs. "
    "Return only the requested JSON object."
)


def parse_resolution(raw: str, eligible: Mapping[str, set[str]], *,
                     state_columns: tuple[str, ...] = ("relation_type", "fact")) -> dict[str, Any]:
    """Validate complete assignments, candidate membership, and canonical states."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("fusion output is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != {"matches", "states"}:
        raise ValueError("fusion output requires only matches and states")
    if not isinstance(value["matches"], list) or not isinstance(value["states"], list):
        raise ValueError("fusion matches and states must be arrays")
    assigned: dict[str, str | None] = {}
    for item in value["matches"]:
        if not isinstance(item, dict) or set(item) != {"left_id", "right_id"}:
            raise ValueError("invalid fusion assignment schema")
        left, right = item["left_id"], item["right_id"]
        if not isinstance(left, str) or left not in eligible or left in assigned:
            raise ValueError("unknown or duplicate incoming ID")
        if right is not None and (
            not isinstance(right, str) or right not in eligible[left]
        ):
            raise ValueError(
                f"target ID {right!r} is not eligible for incoming row {left!r}; "
                f"choose one of {sorted(eligible[left])!r} or null"
            )
        assigned[left] = right
    if set(assigned) != set(eligible):
        raise ValueError("fusion output omitted incoming IDs")
    states: dict[str, dict[str, str]] = {}
    for item in value["states"]:
        if not isinstance(item, dict) or set(item) != {"right_id", *state_columns}:
            raise ValueError("invalid canonical-state schema")
        if any(not isinstance(v, str) for v in item.values()):
            raise ValueError("canonical-state fields must be strings")
        identifier = item["right_id"]
        if identifier in states:
            raise ValueError("duplicate canonical target state")
        states[identifier] = {key: item[key] for key in state_columns}
    if set(states) != {target for target in assigned.values() if target is not None}:
        raise ValueError("canonical states must cover exactly the selected targets")
    return {"matches": assigned, "states": states}


def _unchanged_entity_states(
    eligible: Mapping[str, set[str]],
    fixed: Mapping[str, str | None],
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_ids: Mapping[str, Any],
    right_ids: Mapping[str, Any],
) -> dict[str, dict[str, str]] | None:
    """Copy names only after every screened identity decision is established."""
    if not eligible or set(fixed) != set(eligible):
        return None
    states: dict[str, dict[str, str]] = {}
    for incoming, target in fixed.items():
        if target is None:
            continue
        old = right.loc[right_ids[target], "name"]
        new = left.loc[left_ids[incoming], "name"]
        if not isinstance(old, str) or not isinstance(new, str) or old != new:
            return None
        states[target] = {"name": old}
    return states


def execute_target_state(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> pd.DataFrame:
    """Execute one fused request, retaining relational provenance/temporal logic."""
    import lotus

    joined, body = query.inputs
    aggregate = query.params["aggregate"]
    semantic = next(
        s
        for s in aggregate.params["aggregates"]
        if isinstance(s, SemanticAggregateSpec)
    )
    state_columns = tuple(c.name for c in semantic.output_cols)
    left = execute(joined.inputs[0], inputs)
    right = execute(joined.inputs[1], inputs)
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError("fusion requires unique occurrence indices")
    left_ids = {f"l{i}": idx for i, idx in enumerate(left.index)}
    right_ids = {f"r{i}": idx for i, idx in enumerate(right.index)}
    left_names = {idx: name for name, idx in left_ids.items()}
    right_names = {idx: name for name, idx in right_ids.items()}
    eligible: dict[str, set[str]] = {}
    profile_digest = query.params.get("join_profile_digest", query_digest(joined))
    profile = context.config.semantic_pair_profiles.get(profile_digest)
    if profile is not None and profile.mode not in {"oracle-only", "search-filter"}:
        raise ValueError("target-state fusion cannot replace proxy-only decisions")
    ls, rs = pd.Series(dtype=object), pd.Series(dtype=object)
    if not left.empty and not right.empty:
        ls, rs, _, _, instruction = join_series(
            left, right, str(joined.params["instruction"])
        )
        pairs = semantic_join_pair_candidates(
            ls,
            rs,
            left_frame=left,
            right_frame=right,
            on=tuple(joined.params.get("on", ())),
            query=joined,
        )
        candidates, _ = _apply_pair_profile(joined, pairs, context, profile=profile)
        for _, row in candidates.iterrows():
            eligible.setdefault(left_names[row[PAIR_LEFT_ID_COLUMN]], set()).add(
                right_names[row[PAIR_RIGHT_ID_COLUMN]]
            )
    else:
        instruction = str(joined.params["instruction"])
    from agent_memory.adapters.lotus.identity_reuse import current_identity_decisions
    decisions = current_identity_decisions.get() if query.params.get("identity_reuse") else None
    ordered_targets = {name: sorted(targets) for name, targets in eligible.items()}
    decision_keys = {name: decisions.key("fused-entity-match-v1", instruction,
                        str(ls.loc[left_ids[name]]),
                        [(target, str(rs.loc[right_ids[target]])) for target in targets])
                     for name, targets in ordered_targets.items()} if decisions is not None else {}
    fixed: dict[str, str | None] = {}
    if decisions is not None:
        for name, targets in ordered_targets.items():
            selected = decisions.lookup(decision_keys[name], len(targets))
            if selected is not None:
                if len(selected) > 1:
                    raise ValueError("exclusive fusion cached multiple targets")
                fixed[name] = targets[selected[0]] if selected else None
    resolution: dict[str, Any] = {"matches": {}, "states": {}}
    unchanged_states = None
    if context.config.reuse_unchanged_entity_name and query.params.get("identity_reuse"):
        from agent_memory.memories.zep.fact_summary import IDENTITY_SPEC
        from agent_memory.planner.physical import REPRESENTATIVE_VERSION
        if query.params.get("version") == REPRESENTATIVE_VERSION and semantic == IDENTITY_SPEC:
            unchanged_states = _unchanged_entity_states(eligible, fixed, left, right, left_ids, right_ids)
    state_regenerated = False
    if eligible and len(fixed) == len(eligible) and all(v is None for v in fixed.values()):
        resolution = {"matches": fixed, "states": {}}
    elif unchanged_states is not None:
        resolution = {"matches": fixed, "states": unchanged_states}
        write_trace_event(context.config.trace_dir(), operator="fused_target_state",
                          event_type="unchanged_entity_name", payload={
                              "version": "unchanged-entity-name-v1",
                              "reused_matches": len(fixed), "unchanged_targets": len(unchanged_states),
                              "provider_calls": 0,
                          }, parsed_output=resolution)
    elif eligible:
        context.configure()
        # Fixed decisions need only their chosen state. Unknown assignments
        # still need every screened candidate; never prune their alternatives.
        prompt_targets: dict[str, set[str]] = {}
        for name, targets in eligible.items():
            if name not in fixed:
                prompt_targets[name] = targets
            else:
                target = fixed[name]
                prompt_targets[name] = {target} if target is not None else set()
        # Match context is exactly the ordinary join's text projection. State
        # synthesis receives only the semantic aggregate's declared inputs.
        payload = {
            "version": query.params["version"],
            "join_instruction": instruction,
            "consolidation_instruction": semantic.instruction,
            "incoming": [
                {
                    "id": name,
                    "join_text": str(ls.loc[left_ids[name]]),
                    "state": {
                        c: left.loc[left_ids[name], c]
                        for c in semantic.input_cols or ()
                    },
                    "eligible_targets": sorted(targets),
                }
                for name, targets in prompt_targets.items()
            ],
            "targets": [
                {
                    "id": name,
                    "join_text": str(rs.loc[right_ids[name]]),
                    "state": {
                        c: right.loc[right_ids[name], c]
                        for c in semantic.input_cols or ()
                    },
                }
                for name in sorted(set().union(*prompt_targets.values()))
            ],
            "output_schema": {
                "matches": [
                    {
                        "left_id": "one supplied incoming ID",
                        "right_id": "one ID from that incoming item's eligible_targets, or null",
                    }
                ],
                "states": [
                    {"right_id": "r0", **{c: "string" for c in state_columns}}
                ],
            },
        }
        if fixed:
            payload["fixed_matches"] = fixed
            payload["fixed_match_contract"] = (
                "These identity decisions are already established for identical join inputs. "
                "Copy them unchanged into matches. Still consolidate CURRENT states for each "
                "selected target, including all current incoming contributions."
            )
        prompts = [
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ]
        ]

        def validate(raw: str) -> None:
            parsed = parse_resolution(raw, eligible, state_columns=state_columns)
            if any(parsed["matches"][name] != target for name, target in fixed.items()):
                raise ValueError("fusion changed an established identity decision")

        def build_retry_prompt(prompt: Any, _raw: str, error: str) -> Any:
            if not isinstance(prompt, list):
                raise TypeError("fusion retry requires a message-list prompt")
            return [
                *prompt,
                {
                    "role": "user",
                    "content": (
                        f"The previous response was rejected: {error}. "
                        "Re-read each incoming item's own eligible_targets list. "
                        "Return the complete corrected JSON object and nothing else."
                    ),
                },
            ]

        # A malformed fusion assignment is unsafe to apply, but it is also a
        # recoverable provider-contract failure. Give it one bounded retry even
        # when the generic semantic retry budget is zero; never relax validation.
        result = execute_structured_lm_retry_result(
            lotus.settings.lm,
            prompts,
            lm_kwargs={
                "response_format": {"type": "json_object"},
                "max_tokens": context.config.structured_max_tokens,
                "progress_bar_desc": "Fused target-state",
            },
            output_cols=(),
            shape="object",
            require_explanation=False,
            operator="fused_target_state",
            max_retries=max(1, context.config.structured_parse_retries),
            output_validator=validate,
            retry_prompt_builder=build_retry_prompt,
        )
        if not result.invalid_indices:
            resolution = parse_resolution(result.raw_outputs[0], eligible, state_columns=state_columns)
            state_regenerated = bool(resolution["states"])
        write_trace_event(
            context.config.trace_dir(),
            operator="fused_target_state",
            event_type="fusion_resolution",
            payload={
                "version": query.params["version"],
                "join_query_digest": query_digest(joined),
                "left_occurrences": left_ids,
                "right_occurrences": right_ids,
                "valid": not result.invalid_indices,
                "eligible_pair_count": sum(map(len, eligible.values())),
            },
            raw_output=result.raw_output_attempts,
            parsed_output=resolution if not result.invalid_indices else None,
        )
        if result.invalid_indices:
            raise ValueError(
                f"invalid target-state output; artifacts: {result.failure_artifact_paths}"
            )
    if decisions is not None:
        for name, target in resolution["matches"].items():
            candidates_for_name = ordered_targets[name]
            decisions.store(decision_keys[name], len(candidates_for_name),
                            () if target is None else (candidates_for_name.index(target),))
        write_trace_event(context.config.trace_dir(), operator="fused_target_state",
                          event_type="identity_reuse", payload={
                              "reused": len(fixed), "total": len(eligible),
                              "state_regenerated": state_regenerated,
                          })
    matches = [
        (left_ids[left_id], right_ids[right_id], None)
        for left_id, right_id in resolution["matches"].items()
        if right_id is not None
    ]
    frame = assemble_join_frame(
        left, right, matches, how="outer", id_columns=tuple(joined.params["id_columns"])
    )
    bound = {**inputs, JOIN_MAP_BINDING_NAME: frame}
    deterministic = replace(
        aggregate,
        params={
            "aggregates": tuple(
                s
                for s in aggregate.params["aggregates"]
                if not isinstance(s, SemanticAggregateSpec)
            )
        },
    )
    merged = execute(deterministic, bound)
    for column in state_columns:
        merged[column] = [
            resolution["states"][right_names[idx]][column]
            for idx in merged[JOIN_MAP_RIGHT_ID_COLUMN]
        ]
    reference = QueryExpr(
        op="materialized_view",
        params={"name": _STATE_BINDING, "columns": output_columns(aggregate)},
    )
    return execute(
        replace_query(body, aggregate, reference), {**bound, _STATE_BINDING: merged}
    )
