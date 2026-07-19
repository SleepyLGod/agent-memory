"""Compile one declarative retrieval root into a shared execution DAG."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.retrieval import RetrievalQuery
from agent_memory.policy.schema import output_columns
from agent_memory.planner.serialization import stable_json, stable_value
from agent_memory.storage.statements import InsertStatement, StatementSet


_SUPPORTED_POST_SEARCH_OPS = frozenset(
    {"alias", "assign", "drop_duplicates", "filter", "select"}
)


@dataclass(frozen=True)
class RetrievalNode:
    """One locally executable node in a retrieval plan."""

    node_id: str
    query: QueryExpr
    input_node_ids: tuple[str, ...]
    execution_kind: str
    output_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    statement_id: str | None = None


@dataclass(frozen=True)
class RetrievalPlan:
    """Immutable shared DAG for one named-channel retrieval root."""

    nodes: Mapping[str, RetrievalNode]
    execution_order: tuple[str, ...]
    channel_outputs: Mapping[str, str]
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "execution_order", tuple(self.execution_order))
        object.__setattr__(
            self,
            "channel_outputs",
            MappingProxyType(dict(self.channel_outputs)),
        )


class RetrievalPlanner:
    """Compile retrieval-only relation handles without differential rules."""

    def plan(
        self,
        retrieval: RetrievalQuery,
        *,
        statements: StatementSet | None = None,
    ) -> RetrievalPlan:
        """Build a shared retrieval DAG and bind search sources to sinks."""

        if not isinstance(retrieval, RetrievalQuery):
            raise TypeError("retrieval planner requires a RetrievalQuery")
        builder = _RetrievalPlanBuilder(
            statements=statements,
            required_columns=_collect_required_columns(retrieval),
        )
        return builder.build(retrieval)


class _RetrievalPlanBuilder:
    """Mutable builder scoped to one retrieval plan."""

    def __init__(
        self,
        *,
        statements: StatementSet | None,
        required_columns: Mapping[QueryExpr, tuple[str, ...]],
    ) -> None:
        self._statements = statements
        self._required_columns = required_columns
        self._nodes: dict[str, RetrievalNode] = {}
        self._query_nodes: dict[QueryExpr, str] = {}
        self._execution_order: list[str] = []

    def build(self, retrieval: RetrievalQuery) -> RetrievalPlan:
        """Compile ordered channels and return an immutable plan."""

        channel_outputs = {
            name: self._compile(query) for name, query in retrieval.channels.items()
        }
        fingerprint = _retrieval_fingerprint(
            self._nodes,
            self._execution_order,
            channel_outputs,
            self._statements,
        )
        return RetrievalPlan(
            nodes=self._nodes,
            execution_order=tuple(self._execution_order),
            channel_outputs=channel_outputs,
            fingerprint=fingerprint,
        )

    def _compile(self, query: QueryExpr) -> str:
        existing = self._query_nodes.get(query)
        if existing is not None:
            return existing
        if query.op == "search":
            return self._compile_search(query)
        if query.op not in _SUPPORTED_POST_SEARCH_OPS:
            raise NotImplementedError(
                f"post-search operator {query.op!r} is not supported by the "
                "retrieval planner"
            )
        if not query.inputs:
            raise ValueError(
                f"retrieval relation contains an unbound non-search leaf: {query.op!r}"
            )

        input_node_ids = tuple(self._compile(item) for item in query.inputs)
        local_query = QueryExpr(
            op=query.op,
            inputs=tuple(self._node_leaf(node_id) for node_id in input_node_ids),
            params=query.params,
        )
        return self._register(
            query,
            RetrievalNode(
                node_id=_node_id(query),
                query=local_query,
                input_node_ids=input_node_ids,
                execution_kind="relational",
                output_columns=output_columns(query),
                required_columns=output_columns(query),
            ),
        )

    def _compile_search(self, query: QueryExpr) -> str:
        if not query.inputs:
            raise ValueError("search query requires a materialized source")
        source = query.inputs[0]
        statement = self._resolve_statement(source)
        required_columns = self._required_columns[query]
        if statement is not None and statement.target.mapping is not None:
            validate = getattr(statement.target.mapping, "validate_search_columns", None)
            if callable(validate):
                validate(required_columns)
        origin_node_ids = tuple(self._compile(item) for item in query.inputs[1:])
        source_leaf = QueryExpr(
            op="materialized_view",
            params={
                "name": statement.statement_id if statement is not None else "search_source",
                "columns": output_columns(source),
            },
        )
        local_query = QueryExpr(
            op="search",
            inputs=(
                source_leaf,
                *(self._node_leaf(node_id) for node_id in origin_node_ids),
            ),
            params=query.params,
        )
        return self._register(
            query,
            RetrievalNode(
                node_id=_node_id(query),
                query=local_query,
                input_node_ids=origin_node_ids,
                execution_kind="search",
                output_columns=output_columns(query),
                required_columns=required_columns,
                statement_id=None if statement is None else statement.statement_id,
            ),
        )

    def _resolve_statement(self, source: QueryExpr) -> InsertStatement | None:
        if self._statements is None:
            return None
        matches = tuple(
            statement
            for statement in self._statements.statements
            if statement.query == source
        )
        if len(matches) != 1:
            raise ValueError(
                "search source must match exactly one storage sink; "
                f"found {len(matches)} matches"
            )
        return matches[0]

    def _node_leaf(self, node_id: str) -> QueryExpr:
        node = self._nodes[node_id]
        return QueryExpr(
            op="materialized_view",
            params={"name": node_id, "columns": node.output_columns},
        )

    def _register(self, query: QueryExpr, node: RetrievalNode) -> str:
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing.query != node.query:
            raise RuntimeError(f"retrieval node id collision: {node.node_id}")
        self._nodes[node.node_id] = node
        self._query_nodes[query] = node.node_id
        self._execution_order.append(node.node_id)
        return node.node_id


def _node_id(query: QueryExpr) -> str:
    digest = hashlib.sha256(stable_json(query).encode("utf-8")).hexdigest()[:16]
    return f"retrieval-{query.op}-{digest}"


def _retrieval_fingerprint(
    nodes: Mapping[str, RetrievalNode],
    execution_order: list[str],
    channel_outputs: Mapping[str, str],
    statements: StatementSet | None,
) -> str:
    targets = (
        {}
        if statements is None
        else {
            statement.statement_id: statement.target.to_dict()
            for statement in statements.statements
            if any(
                node.statement_id == statement.statement_id for node in nodes.values()
            )
        }
    )
    payload = {
        "nodes": [
            {
                "id": node_id,
                "query": stable_value(nodes[node_id].query),
                "inputs": nodes[node_id].input_node_ids,
                "kind": nodes[node_id].execution_kind,
                "outputs": nodes[node_id].output_columns,
                "required_columns": nodes[node_id].required_columns,
                "statement_id": nodes[node_id].statement_id,
            }
            for node_id in execution_order
        ],
        "channels": dict(channel_outputs),
        "targets": targets,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _collect_required_columns(
    retrieval: RetrievalQuery,
) -> dict[QueryExpr, tuple[str, ...]]:
    """Collect search columns demanded by channel projections and BFS inputs."""

    required: dict[QueryExpr, list[str]] = {}

    def collect(query: QueryExpr, columns: tuple[str, ...]) -> None:
        if query.op == "search":
            values = required.setdefault(query, [])
            for column in columns:
                if column not in values:
                    values.append(column)
            for method in query.params["methods"]:
                if method.kind == "bfs":
                    origin_input = int(method.params["origin_input"])
                    collect(query.inputs[origin_input], ("record_id",))
            return
        if query.op == "select":
            selected = tuple(query.params["columns"])
            missing = sorted(set(columns).difference(selected))
            if missing:
                raise ValueError(
                    f"retrieval projection does not provide required columns: {missing}"
                )
            collect(query.inputs[0], columns)
            return
        if query.op == "filter":
            collect(
                query.inputs[0],
                _ordered_union(columns, _expression_columns(query.params["predicate"])),
            )
            return
        if query.op == "assign":
            assignments = query.params["assignments"]
            input_columns: list[str] = []
            for column in columns:
                dependencies = (
                    _expression_columns(assignments[column])
                    if column in assignments
                    else (column,)
                )
                input_columns.extend(dependencies)
            collect(query.inputs[0], _ordered_union((), tuple(input_columns)))
            return
        if query.op == "alias":
            collect(query.inputs[0], columns)
            return
        if query.op == "drop_duplicates":
            collect(query.inputs[0], output_columns(query.inputs[0]))
            return
        raise NotImplementedError(
            f"post-search operator {query.op!r} is not supported by the "
            "retrieval planner"
        )

    for query in retrieval.channels.values():
        collect(query, output_columns(query))
    return {query: tuple(columns) for query, columns in required.items()}


def _expression_columns(value: Any) -> tuple[str, ...]:
    """Return unqualified source columns referenced by one expression param."""

    columns: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            if item.get("kind") == "column":
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("retrieval expression contains an invalid column")
                if name not in columns:
                    columns.append(name)
                return
            for nested in item.values():
                collect(nested)
            return
        if isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)

    collect(value)
    return tuple(columns)


def _ordered_union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Return a stable union of logical column names."""

    result = list(left)
    for column in right:
        if column not in result:
            result.append(column)
    return tuple(result)


__all__ = ["RetrievalNode", "RetrievalPlan", "RetrievalPlanner"]
