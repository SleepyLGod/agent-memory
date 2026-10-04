"""Explicit physical rewrites over an already differentiated policy."""

from dataclasses import replace
from hashlib import sha256
from collections.abc import Callable, Iterator

from agent_memory.planner.differential_policy import DifferentiatedPolicy
from agent_memory.planner.rules import JOIN_MAP_BINDING_NAME
from agent_memory.planner.serialization import stable_json
from agent_memory.policy.aggregates import SemanticAggregateSpec
from agent_memory.policy.logical import QueryExpr

FUSION_VERSION = "zep-target-state-v1"
COMBINED_VERSION = "zep-combined-v2"
SUMMARY_VERSION = "zep-fact-summary-v2"
REPRESENTATIVE_VERSION = "zep-representative-v2"
PREDICATE_DECISIONS_INPUT = "__physical_predicate_decisions"


def enable_predicate_reuse(
    policy: DifferentiatedPolicy, node_ids: set[str],
) -> DifferentiatedPolicy:
    """Register fixed textual predicate decisions as physical executor state."""
    if not node_ids:
        return policy
    nodes = dict(policy.nodes)
    for node_id in node_ids:
        node = nodes[node_id]
        if node.query.op != "sem_filter" or node.execution_kind != "semantic_row":
            raise ValueError("predicate reuse requires a row-local sem_filter node")
        nodes[node_id] = replace(node, execution_kind="semantic_predicate")
    fingerprint = sha256(stable_json({
        "base": policy.fingerprint, "predicate-reuse-v1": sorted(node_ids),
    }).encode()).hexdigest()
    return replace(policy, nodes=nodes, fingerprint=fingerprint)


def walk(query: QueryExpr) -> Iterator[QueryExpr]:
    """Visit a query tree, including repeated references."""
    yield query
    for child in query.inputs:
        yield from walk(child)


def replace_query(query: QueryExpr, old: QueryExpr, new: QueryExpr) -> QueryExpr:
    """Substitute a selected subtree without modifying the original plan."""
    if query == old:
        return new
    inputs = tuple(replace_query(q, old, new) for q in query.inputs)
    if all(a is b for a, b in zip(inputs, query.inputs, strict=True)):
        return query
    return replace(query, inputs=inputs)


def _zep_target_state(policy: DifferentiatedPolicy) -> DifferentiatedPolicy:
    from agent_memory.memories.zep.policy import ZepMemory

    template = ZepMemory._deduplicated_facts.expr.inputs[0]
    return _fuse_registered_state(policy, template, FUSION_VERSION, ("relation_type", "fact"))


def _fuse_registered_state(
    policy: DifferentiatedPolicy, template: QueryExpr, version: str,
    state_columns: tuple[str, ...],
) -> DifferentiatedPolicy:
    matches = [
        node
        for node in policy.nodes.values()
        if node.query.op == "agg"
        and node.query.params == template.params
        and node.query.inputs[0].op == "sem_groupby"
        and node.query.inputs[0].params == template.inputs[0].params
    ]
    if policy.grouped_agg_rule != "rule-join-map" or len(matches) != 1:
        raise ValueError(
            "zep-target-state requires exactly one registered Zep fact join-map site"
        )
    node = matches[0]
    query = node.maintenance_query
    if query is not None and query.op == "fused_target_state":
        if query.params.get("version") != version:
            raise ValueError("unsupported target-state fusion version")
        return policy
    if (
        query is None
        or query.op != "let"
        or query.params.get("name") != JOIN_MAP_BINDING_NAME
    ):
        raise ValueError("unsupported Zep join-map maintenance shape")
    joined, body = query.inputs
    if (
        joined.op != "sem_join"
        or joined.params.get("k") != 1
        or joined.params.get("how") != "outer"
    ):
        raise ValueError("target-state fusion requires an exclusive outer join")
    aggregates = [q for q in walk(body) if q.op == "agg"]
    if len(aggregates) != 1:
        raise ValueError("target-state fusion requires one state aggregate")
    aggregate = aggregates[0]
    semantic = [
        s
        for s in aggregate.params["aggregates"]
        if isinstance(s, SemanticAggregateSpec)
    ]
    if len(semantic) != 1 or tuple(c.name for c in semantic[0].output_cols) != state_columns:
        raise ValueError("unsupported target-state semantic output contract")
    fused = QueryExpr(
        op="fused_target_state",
        inputs=query.inputs,
        params={"version": version, "aggregate": aggregate},
    )
    nodes = dict(policy.nodes)
    nodes[node.node_id] = replace(node, maintenance_query=fused)
    fingerprint = sha256(
        stable_json(
            {
                "base": policy.fingerprint,
                "fusion": version,
                "site": node.node_id,
                "query": fused,
            }
        ).encode()
    ).hexdigest()
    return replace(policy, nodes=nodes, fingerprint=fingerprint)


def _zep_combined(policy: DifferentiatedPolicy, *, fact_summary: bool = False) -> DifferentiatedPolicy:
    from agent_memory.memories.zep.policy import ZepMemory

    version = SUMMARY_VERSION if fact_summary else COMBINED_VERSION
    entity_template = next(q for q in walk(ZepMemory.entities.expr) if q.op == "agg")
    entity_columns = ("name", "summary")
    if fact_summary:
        from agent_memory.memories.zep.fact_summary import ZepFactSummaryMemory
        entity_template = next(q for q in walk(ZepFactSummaryMemory._identities.expr) if q.op == "agg")
        entity_columns = ("name",)

    existing = [n.maintenance_query for n in policy.nodes.values()
                if n.maintenance_query is not None and n.maintenance_query.op == "fused_target_state"]
    if len(existing) == 2 and all(q.params.get("version") == version for q in existing):
        if (sum(bool(q.params.get("identity_reuse")) for q in existing) == 1
                and sum(bool(child.params.get("singleton_identity")) for q in existing for child in walk(q)) == 1):
            return policy
    prepared = _fuse_registered_state(
        policy, ZepMemory._deduplicated_facts.expr.inputs[0], version,
        ("relation_type", "fact"),
    )
    prepared = _fuse_registered_state(
        prepared, entity_template, version, entity_columns,
    )
    nodes = dict(prepared.nodes)
    for node_id, node in nodes.items():
        query = node.maintenance_query
        if query is None or query.op != "fused_target_state":
            continue
        if tuple(c.name for s in query.params["aggregate"].params["aggregates"]
                 if isinstance(s, SemanticAggregateSpec) for c in s.output_cols) != ("relation_type", "fact"):
            if fact_summary:
                from agent_memory.tracing.semantic import query_digest
                joined = query.inputs[0]
                delta = next(q for q in walk(joined.inputs[0]) if q.op == "agg")
                query = replace_query(query, delta, replace(delta, params={**delta.params, "identity_singleton": True}))
                query = replace(query, params={**query.params, "join_profile_digest": query_digest(joined)})
            nodes[node_id] = replace(node, maintenance_query=replace(
                query, params={**query.params, "identity_reuse": True}))
            continue
        joined = query.inputs[0]
        delta_aggs = [q for q in walk(joined.inputs[0]) if q.op == "agg"]
        if len(delta_aggs) != 1:
            raise ValueError("singleton fact strategy requires one new-fact aggregate")
        delta = delta_aggs[0]
        marked = replace(delta, params={**delta.params, "singleton_identity": True})
        from agent_memory.tracing.semantic import query_digest
        rewritten = replace_query(query, delta, marked)
        nodes[node_id] = replace(node, maintenance_query=replace(
            rewritten, params={**rewritten.params, "join_profile_digest": query_digest(joined)}))
    fingerprint = sha256(stable_json({"base": prepared.fingerprint,
        "version": version, "singleton_fact_identity": True}).encode()).hexdigest()
    return replace(prepared, nodes=nodes, fingerprint=fingerprint)


def summary_specs() -> tuple[SemanticAggregateSpec, SemanticAggregateSpec]:
    """Exact registered raw and state-merge contracts, never prompt substring matching."""
    from agent_memory.memories.zep.fact_summary import SUMMARY_SPEC
    from agent_memory.planner.rules import DifferentialInstructionRewriter
    return SUMMARY_SPEC, replace(SUMMARY_SPEC, instruction=DifferentialInstructionRewriter().state_reaggregation(
        SUMMARY_SPEC.instruction, state_cols=("summary",), raw_input_cols=("summary",)))


def lower_fact_summary_node(query: QueryExpr) -> QueryExpr:
    """Use the same registered shortcuts for direct and incremental execution."""
    from agent_memory.memories.zep.fact_summary import SUMMARY_SPEC, IDENTITY_SPEC
    from agent_memory.planner.rules import DifferentialInstructionRewriter
    map_instruction = DifferentialInstructionRewriter().agg_to_map(
        SUMMARY_SPEC.instruction, input_cols=("summary",), output_cols=SUMMARY_SPEC.output_cols)
    if query.op == "agg":
        specs = tuple(query.params.get("aggregates", ()))
        if specs in tuple((s,) for s in summary_specs()):
            return replace(query, params={**query.params, "fact_summary": True})
        if query.inputs[0].op == "sem_groupby" and tuple(s for s in specs if isinstance(s, SemanticAggregateSpec)) == (IDENTITY_SPEC,):
            return replace(query, params={**query.params, "identity_singleton": True})
    if query.op == "sem_map" and query.params.get("instruction") == map_instruction and query.params.get("output_cols") == SUMMARY_SPEC.output_cols:
        return replace(query, params={**query.params, "fact_summary_map": True})
    return query


def _zep_fact_summary(policy: DifferentiatedPolicy) -> DifferentiatedPolicy:
    prepared = _zep_combined(policy, fact_summary=True)
    def mark(query: QueryExpr) -> QueryExpr:
        rewritten = replace(query, inputs=tuple(mark(q) for q in query.inputs))
        return lower_fact_summary_node(rewritten)
    nodes = {key: replace(node, query=mark(node.query), maintenance_query=(
        mark(node.maintenance_query) if node.maintenance_query is not None else None))
        for key, node in prepared.nodes.items()}
    return replace(prepared, nodes=nodes)


def _zep_representative(policy: DifferentiatedPolicy) -> DifferentiatedPolicy:
    """Fuse entity identity only; fact representatives need no synthesis call."""
    from agent_memory.memories.zep.representative import ZepRepresentativeMemory
    from agent_memory.policy.aggregates import ArgMinAggregateSpec
    if not any(any(isinstance(s, ArgMinAggregateSpec) for s in node.query.params.get("aggregates", ()))
               for node in policy.nodes.values()):
        raise ValueError("zep-representative requires the representative logical view")
    fused = [n.maintenance_query for n in policy.nodes.values()
             if n.maintenance_query is not None and n.maintenance_query.op == "fused_target_state"]
    if len(fused) == 1 and fused[0].params.get("version") == REPRESENTATIVE_VERSION:
        return policy
    template = next(q for q in walk(ZepRepresentativeMemory._identities.expr) if q.op == "agg")
    prepared = _fuse_registered_state(policy, template, REPRESENTATIVE_VERSION, ("name",))
    def mark(query: QueryExpr) -> QueryExpr:
        return lower_fact_summary_node(replace(query, inputs=tuple(mark(q) for q in query.inputs)))
    nodes = {}
    for key, node in prepared.nodes.items():
        maintenance = node.maintenance_query
        if maintenance is not None:
            from agent_memory.tracing.semantic import query_digest
            profile_digest = query_digest(maintenance.inputs[0]) if maintenance.op == "fused_target_state" else None
            maintenance = mark(maintenance)
            if maintenance.op == "fused_target_state":
                maintenance = replace(maintenance, params={**maintenance.params, "identity_reuse": True,
                                                          "join_profile_digest": profile_digest})
        nodes[key] = replace(node, query=mark(node.query), maintenance_query=maintenance)
    return replace(prepared, nodes=nodes)


_REWRITES: dict[str, Callable[[DifferentiatedPolicy], DifferentiatedPolicy]] = {
    "zep-target-state": _zep_target_state,
    "zep-combined": _zep_combined,
    "zep-fact-summary": _zep_fact_summary,
    "zep-representative": _zep_representative,
}


def optimize_policy(
    policy: DifferentiatedPolicy, *, strategy: str = "disabled"
) -> DifferentiatedPolicy:
    """Apply one registered rewrite; disabled returns the original object."""
    if strategy == "disabled":
        return policy
    if strategy not in _REWRITES:
        raise ValueError(f"unknown physical fusion strategy: {strategy!r}")
    return _REWRITES[strategy](policy)
