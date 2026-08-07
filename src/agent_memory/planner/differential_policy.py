"""Differentiate a logical policy into a shared executable node graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from agent_memory.policy.aggregates import SemanticAggregateSpec
from agent_memory.policy.logical import MemorySpec, MemoryView, QueryExpr
from agent_memory.policy.retrieval import RetrievalQuery
from agent_memory.policy.schema import output_columns
from agent_memory.planner.differential_query import QueryDifferentiator
from agent_memory.planner.retrieval import RetrievalPlan, RetrievalPlanner
from agent_memory.planner.serialization import stable_json, stable_value
from agent_memory.planner.rules import (
    DifferentialInstructionRewriter,
    DifferentialRules,
)
from agent_memory.storage.statements import StatementSet

_GROUP_CARRIERS = {"group_by", "sem_groupby"}
_ROW_LOCAL_SEMANTIC_OPS = {"sem_filter", "sem_map", "sem_flat_map"}


@dataclass(frozen=True)
class DifferentialNode:
    """One locally executable node in a differentiated policy."""

    node_id: str
    query: QueryExpr
    input_node_ids: tuple[str, ...]
    execution_kind: str
    output_columns: tuple[str, ...]
    maintenance_query: QueryExpr | None = None


@dataclass(frozen=True)
class DifferentiatedPolicy:
    """Immutable executable policy produced from one logical memory spec."""

    spec: MemorySpec
    nodes: Mapping[str, DifferentialNode]
    execution_order: tuple[str, ...]
    view_outputs: Mapping[str, str]
    sink_outputs: Mapping[str, str]
    retrieval_queries: Mapping[str, QueryExpr | RetrievalPlan]
    fingerprint: str
    grouped_agg_rule: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "execution_order", tuple(self.execution_order))
        object.__setattr__(self, "view_outputs", MappingProxyType(dict(self.view_outputs)))
        object.__setattr__(self, "sink_outputs", MappingProxyType(dict(self.sink_outputs)))
        object.__setattr__(
            self,
            "retrieval_queries",
            MappingProxyType(dict(self.retrieval_queries)),
        )


class PolicyDifferentiator:
    """Differentiate complete memory policies with one rule configuration."""

    def __init__(
        self,
        rules: DifferentialRules | None = None,
        instruction_rewriter: DifferentialInstructionRewriter | None = None,
    ) -> None:
        self._query_differentiator = QueryDifferentiator(
            rules=rules,
            instruction_rewriter=instruction_rewriter,
        )

    def differentiate(
        self,
        spec: MemorySpec,
        *,
        statements: StatementSet | None = None,
    ) -> DifferentiatedPolicy:
        """Differentiate public views and optional storage sinks into one policy."""

        dependencies = _view_dependencies(spec)
        view_order = _topological_order(tuple(spec.views), dependencies)
        (
            nodes,
            execution_order,
            view_outputs,
            sink_outputs,
            fingerprint,
        ) = _DifferentialPolicyBuilder(self._query_differentiator).build(
            spec,
            view_execution_order=view_order,
            statements=statements,
        )
        retrieval_planner = RetrievalPlanner()
        retrieval_queries: dict[str, QueryExpr | RetrievalPlan] = {}
        for name, query in spec.retrieval_queries.items():
            if isinstance(query, RetrievalQuery):
                retrieval_queries[name] = retrieval_planner.plan(
                    query,
                    statements=statements,
                )
            else:
                retrieval_queries[name] = _bind_materialized_views(query, spec=spec)
        return DifferentiatedPolicy(
            spec=spec,
            nodes=nodes,
            execution_order=execution_order,
            view_outputs=view_outputs,
            sink_outputs=sink_outputs,
            retrieval_queries=retrieval_queries,
            fingerprint=fingerprint,
            grouped_agg_rule=self._query_differentiator.grouped_agg_rule,
        )


class _DifferentialPolicyBuilder:
    """Build structurally shared nodes for one logical memory policy."""

    def __init__(self, query_differentiator: QueryDifferentiator) -> None:
        self._query_differentiator = query_differentiator
        self._nodes: dict[str, DifferentialNode] = {}
        self._node_queries: dict[str, QueryExpr] = {}
        self._query_nodes: dict[QueryExpr, str] = {}
        self._execution_order: list[str] = []
        self._view_nodes: dict[str, str] = {}
        self._sink_nodes: dict[str, str] = {}

    def build(
        self,
        spec: MemorySpec,
        *,
        view_execution_order: tuple[str, ...],
        statements: StatementSet | None,
    ) -> tuple[
        dict[str, DifferentialNode],
        tuple[str, ...],
        dict[str, str],
        dict[str, str],
        str,
    ]:
        """Build public view and storage sink roots into one shared graph."""

        self._nodes = {}
        self._node_queries = {}
        self._query_nodes = {}
        self._execution_order = []
        self._view_nodes = {}
        self._sink_nodes = {}

        for view_name in view_execution_order:
            self._view_nodes[view_name] = self._compile_query(
                spec.views[view_name].query,
                spec=spec,
            )

        if statements is not None:
            for statement in statements.statements:
                self._sink_nodes[statement.statement_id] = self._compile_query(
                    statement.query,
                    spec=spec,
                )

        fingerprint = _plan_fingerprint(
            self._nodes,
            self._execution_order,
            self._view_nodes,
            sink_outputs=self._sink_nodes,
            statements=statements,
        )
        return (
            dict(self._nodes),
            tuple(self._execution_order),
            dict(self._view_nodes),
            dict(self._sink_nodes),
            fingerprint,
        )

    def _compile_query(self, query: QueryExpr, *, spec: MemorySpec) -> str:
        """Compile one query subtree and return its shared node identifier."""

        existing = self._query_nodes.get(query)
        if existing is not None:
            return existing

        if query.op == "materialized_view":
            name = str(query.params["name"])
            if name in self._view_nodes:
                return self._view_nodes[name]
            if name in spec.views:
                raise ValueError(
                    f"Materialized view {name!r} was referenced before its dependency was compiled"
                )
            raise ValueError(f"Unknown materialized view in differentiated policy: {name!r}")

        if query.op == "alias":
            node_id = self._compile_query(query.inputs[0], spec=spec)
            self._query_nodes[query] = node_id
            return node_id

        if query.op == "sem_join":
            raise NotImplementedError(
                "Direct view-time sem_join is not supported by policy differentiation"
            )
        if query.op == "sem_topk":
            raise NotImplementedError(
                "View-time sem_topk is not supported by policy differentiation"
            )
        if query.op == "search":
            raise NotImplementedError(
                "Search is retrieval-only and cannot be maintained as a view or storage sink"
            )
        if query.op in _GROUP_CARRIERS | {"count_window", "over", "window_source"}:
            raise ValueError(
                f"QueryExpr op {query.op!r} is an internal carrier and cannot be materialized alone"
            )

        if query.op == "log":
            node = self._make_node(
                query=query,
                local_query=query,
                input_node_ids=(),
                execution_kind="source",
            )
            return self._register(query, node)

        if query.op == "process_window":
            return self._compile_process_window(query, spec=spec)

        if self._is_grouped_aggregate(query):
            return self._compile_grouped_aggregate(query, spec=spec)

        if self._is_over_aggregate(query):
            return self._compile_over_aggregate(query, spec=spec)

        input_node_ids = tuple(
            self._compile_query(input_query, spec=spec) for input_query in query.inputs
        )
        local_query = self._bind_direct_inputs(query, input_node_ids)
        execution_kind = (
            "semantic_row" if query.op in _ROW_LOCAL_SEMANTIC_OPS else "deterministic"
        )
        if query.op == "sem_agg":
            execution_kind = "semantic_state"
        node = self._make_node(
            query=query,
            local_query=local_query,
            input_node_ids=input_node_ids,
            execution_kind=execution_kind,
            maintenance_query=self._semantic_maintenance_query(
                local_query,
                input_node_ids,
                current_node_id=self._node_id(query),
            )
            if execution_kind == "semantic_state"
            else None,
        )
        return self._register(query, node)

    def _compile_grouped_aggregate(self, query: QueryExpr, *, spec: MemorySpec) -> str:
        """Compile one aggregate together with its grouping carrier."""

        grouped = query.inputs[0]
        source_query = grouped.inputs[0]
        source_node_id = self._compile_query(source_query, spec=spec)
        source_leaf = self._node_leaf(source_node_id)
        local_grouped = QueryExpr(
            op=grouped.op,
            inputs=(source_leaf,),
            params=grouped.params,
        )
        local_query = QueryExpr(
            op=query.op,
            inputs=(local_grouped,),
            params=query.params,
        )
        semantic = query.op == "sem_agg" or (
            query.op == "agg"
            and any(
                isinstance(specification, SemanticAggregateSpec)
                for specification in query.params.get("aggregates", ())
            )
        )
        execution_kind = "semantic_state" if semantic else "deterministic"
        node = self._make_node(
            query=query,
            local_query=local_query,
            input_node_ids=(source_node_id,),
            execution_kind=execution_kind,
            maintenance_query=self._semantic_maintenance_query(
                local_query,
                (source_node_id,),
                current_node_id=self._node_id(query),
            )
            if semantic
            else None,
        )
        return self._register(query, node)

    def _compile_over_aggregate(self, query: QueryExpr, *, spec: MemorySpec) -> str:
        """Compile an over-window function with its frame carrier."""

        over_query = query.inputs[0]
        source_node_id = self._compile_query(over_query.inputs[0], spec=spec)
        local_over = QueryExpr(
            op="over",
            inputs=(self._node_leaf(source_node_id),),
            params=over_query.params,
        )
        local_query = QueryExpr(
            op=query.op,
            inputs=(local_over,),
            params=query.params,
        )
        kind = "semantic_over_window" if query.op == "sem_agg" else "over_window"
        node = self._make_node(
            query=query,
            local_query=local_query,
            input_node_ids=(source_node_id,),
            execution_kind=kind,
        )
        return self._register(query, node)

    def _compile_process_window(self, query: QueryExpr, *, spec: MemorySpec) -> str:
        """Compile count-window processing as one stateful node."""

        window_query, process_query = query.inputs
        if window_query.op != "count_window" or len(window_query.inputs) != 1:
            raise ValueError("process_window requires one count_window input")
        if not self._uses_only_window_source(process_query):
            raise ValueError(
                "process_window callback may only reference its provided window relation"
            )
        source_node_id = self._compile_query(window_query.inputs[0], spec=spec)
        local_window = QueryExpr(
            op="count_window",
            inputs=(self._node_leaf(source_node_id),),
            params=window_query.params,
        )
        local_query = QueryExpr(
            op="process_window",
            inputs=(local_window, query.inputs[1]),
            params=query.params,
        )
        node = self._make_node(
            query=query,
            local_query=local_query,
            input_node_ids=(source_node_id,),
            execution_kind="process_window",
        )
        return self._register(query, node)

    def _uses_only_window_source(self, query: QueryExpr) -> bool:
        """Return whether every relation leaf is the callback window source."""

        if not query.inputs:
            return query.op == "window_source"
        return all(self._uses_only_window_source(item) for item in query.inputs)

    def _semantic_maintenance_query(
        self,
        local_query: QueryExpr,
        input_node_ids: tuple[str, ...],
        *,
        current_node_id: str,
    ) -> QueryExpr:
        """Differentiate a stateful semantic node against its materialized input."""

        if len(input_node_ids) != 1:
            raise NotImplementedError(
                "Stateful semantic policy nodes currently require one input relation"
            )
        input_node_id = input_node_ids[0]
        source = self._node_leaf(input_node_id)
        changed_source = QueryExpr(
            op="materialized_view",
            params={
                "name": f"{input_node_id}__inserted",
                "columns": source.params["columns"],
            },
        )
        return self._query_differentiator.differentiate(
            MemoryView(name=current_node_id, query=local_query),
            views={},
            source_query=source,
            source_input=changed_source,
        )

    def _bind_direct_inputs(
        self,
        query: QueryExpr,
        input_node_ids: tuple[str, ...],
    ) -> QueryExpr:
        """Replace direct query inputs with materialized policy-node leaves."""

        bound_inputs: list[QueryExpr] = []
        for original_input, node_id in zip(
            query.inputs,
            input_node_ids,
            strict=True,
        ):
            materialized = self._node_leaf(node_id)
            if original_input.op == "alias":
                materialized = QueryExpr(
                    op="alias",
                    inputs=(materialized,),
                    params=original_input.params,
                )
            bound_inputs.append(materialized)
        return QueryExpr(
            op=query.op,
            inputs=tuple(bound_inputs),
            params=query.params,
        )

    def _node_leaf(self, node_id: str) -> QueryExpr:
        """Return a materialized input leaf for a compiled node."""

        node = self._nodes[node_id]
        return QueryExpr(
            op="materialized_view",
            params={"name": node_id, "columns": node.output_columns},
        )

    def _make_node(
        self,
        *,
        query: QueryExpr,
        local_query: QueryExpr,
        input_node_ids: tuple[str, ...],
        execution_kind: str,
        maintenance_query: QueryExpr | None = None,
    ) -> DifferentialNode:
        """Build one immutable node from a full and locally bound query."""

        return DifferentialNode(
            node_id=self._node_id(query),
            query=local_query,
            input_node_ids=input_node_ids,
            execution_kind=execution_kind,
            output_columns=output_columns(query),
            maintenance_query=maintenance_query,
        )

    def _register(self, query: QueryExpr, node: DifferentialNode) -> str:
        """Register one node after all of its inputs in topological order."""

        existing_query = self._node_queries.get(node.node_id)
        if existing_query is not None and existing_query != query:
            raise RuntimeError(f"Differential node id collision: {node.node_id}")
        self._nodes[node.node_id] = node
        self._node_queries[node.node_id] = query
        self._query_nodes[query] = node.node_id
        self._execution_order.append(node.node_id)
        return node.node_id

    def _node_id(self, query: QueryExpr) -> str:
        """Return a deterministic identifier for one structural query node."""

        digest = hashlib.sha256(stable_json(query).encode("utf-8")).hexdigest()[:16]
        return f"{query.op}-{digest}"

    def _is_grouped_aggregate(self, query: QueryExpr) -> bool:
        return (
            bool(query.inputs)
            and query.op in {"agg", "array_agg", "min", "sem_agg"}
            and query.inputs[0].op in _GROUP_CARRIERS
        )

    def _is_over_aggregate(self, query: QueryExpr) -> bool:
        return (
            bool(query.inputs)
            and query.op in {"array_agg", "sem_agg"}
            and query.inputs[0].op == "over"
        )


def _view_dependencies(spec: MemorySpec) -> dict[str, tuple[str, ...]]:
    """Return direct public-view dependencies for every public view."""

    return {
        name: _query_dependencies(view.query, current_view=name, spec=spec)
        for name, view in spec.views.items()
    }


def _query_dependencies(
    query: QueryExpr,
    *,
    current_view: str,
    spec: MemorySpec,
) -> tuple[str, ...]:
    """Return public views referenced by one query tree."""

    dependencies: list[str] = []

    def collect(item: QueryExpr) -> None:
        if item.op == "materialized_view":
            name = str(item.params["name"])
            if name != current_view and name in spec.views and name not in dependencies:
                dependencies.append(name)
            return

        for name, view in spec.views.items():
            if name != current_view and item == view.query:
                if name not in dependencies:
                    dependencies.append(name)
                return

        for input_query in item.inputs:
            collect(input_query)

    collect(query)
    return tuple(dependencies)


def _topological_order(
    view_names: tuple[str, ...],
    dependencies: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return stable dependency order or reject a public-view cycle."""

    remaining = {name: set(dependencies[name]) for name in view_names}
    order: list[str] = []
    while remaining:
        ready = [name for name in view_names if name in remaining and not remaining[name]]
        if not ready:
            cycle = ", ".join(sorted(remaining))
            raise ValueError(f"Memory view dependency cycle detected among: {cycle}")
        for name in ready:
            order.append(name)
            del remaining[name]
            for dependency_set in remaining.values():
                dependency_set.discard(name)
    return tuple(order)


def _bind_materialized_views(query: QueryExpr, *, spec: MemorySpec) -> QueryExpr:
    """Replace exact public-view subtrees with materialized view leaves."""

    for name, view in spec.views.items():
        if query == view.query:
            return QueryExpr(op="materialized_view", params={"name": name})
    if not query.inputs:
        return query
    return QueryExpr(
        op=query.op,
        inputs=tuple(_bind_materialized_views(item, spec=spec) for item in query.inputs),
        params=query.params,
    )


def _plan_fingerprint(
    nodes: Mapping[str, DifferentialNode],
    execution_order: list[str],
    view_outputs: Mapping[str, str],
    *,
    sink_outputs: Mapping[str, str],
    statements: StatementSet | None,
) -> str:
    """Return a stable digest for checkpoint compatibility validation."""

    payload = {
        "nodes": [
            {
                "id": node_id,
                "query": stable_value(nodes[node_id].query),
                "maintenance_query": stable_value(nodes[node_id].maintenance_query),
                "inputs": nodes[node_id].input_node_ids,
                "kind": nodes[node_id].execution_kind,
                "outputs": nodes[node_id].output_columns,
            }
            for node_id in execution_order
        ],
        "views": dict(sorted(view_outputs.items())),
    }
    if statements is not None and statements.statements:
        payload["sinks"] = [
            {
                "id": statement.statement_id,
                "node": sink_outputs[statement.statement_id],
                "target": statement.target.to_dict(),
            }
            for statement in statements.statements
        ]
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
