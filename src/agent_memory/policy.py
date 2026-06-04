"""Policy-level differentiation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent_memory.logical import MemorySpec, QueryExpr
from agent_memory.planner import DifferentialQueryPlanner


@dataclass(frozen=True)
class DifferentiatedPolicy:
    """In-memory policy artifact produced by whole-policy differentiation."""

    spec: MemorySpec
    view_queries: Mapping[str, QueryExpr]
    retrieval_queries: Mapping[str, QueryExpr]
    view_dependencies: Mapping[str, tuple[str, ...]]
    view_execution_order: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "view_queries",
            MappingProxyType(dict(self.view_queries)),
        )
        object.__setattr__(
            self,
            "retrieval_queries",
            MappingProxyType(dict(self.retrieval_queries)),
        )
        object.__setattr__(
            self,
            "view_dependencies",
            MappingProxyType(dict(self.view_dependencies)),
        )
        object.__setattr__(
            self,
            "view_execution_order",
            tuple(self.view_execution_order),
        )


class DifferentialPolicyCompiler:
    """Compile a MemorySpec into differentiated view and retrieval queries."""

    def __init__(self, view_planner: DifferentialQueryPlanner | None = None) -> None:
        self._view_planner = (
            view_planner if view_planner is not None else DifferentialQueryPlanner()
        )

    def differentiate(self, spec: MemorySpec) -> DifferentiatedPolicy:
        """Differentiate a full memory policy into an in-memory artifact."""

        view_dependencies = self._view_dependencies(spec)
        view_execution_order = self._topological_order(
            tuple(spec.views),
            view_dependencies,
        )
        view_queries = {
            name: self._view_planner.differentiate(spec.views[name], views=spec.views)
            for name in view_execution_order
        }
        retrieval_queries = {
            name: self._bind_materialized_views(query, spec=spec)
            for name, query in spec.retrieval_queries.items()
        }
        return DifferentiatedPolicy(
            spec=spec,
            view_queries=view_queries,
            retrieval_queries=retrieval_queries,
            view_dependencies=view_dependencies,
            view_execution_order=view_execution_order,
        )

    def _view_dependencies(self, spec: MemorySpec) -> dict[str, tuple[str, ...]]:
        """Collect exact structural public-view dependencies."""

        return {
            name: self._query_dependencies(view.query, current_view=name, spec=spec)
            for name, view in spec.views.items()
        }

    def _query_dependencies(
        self,
        query: QueryExpr,
        *,
        current_view: str,
        spec: MemorySpec,
    ) -> tuple[str, ...]:
        """Return public views referenced by exact query-tree identity."""

        dependencies: list[str] = []
        self._collect_query_dependencies(
            query,
            current_view=current_view,
            spec=spec,
            dependencies=dependencies,
        )
        return tuple(dependencies)

    def _collect_query_dependencies(
        self,
        query: QueryExpr,
        *,
        current_view: str,
        spec: MemorySpec,
        dependencies: list[str],
    ) -> None:
        """Append direct materialized-view dependencies in traversal order."""

        if query.op == "materialized_view":
            name = str(query.params["name"])
            if name != current_view and name in spec.views and name not in dependencies:
                dependencies.append(name)
            return

        for name, view in spec.views.items():
            if name == current_view:
                continue
            if query == view.query:
                if name not in dependencies:
                    dependencies.append(name)
                return

        for input_query in query.inputs:
            self._collect_query_dependencies(
                input_query,
                current_view=current_view,
                spec=spec,
                dependencies=dependencies,
            )

    def _topological_order(
        self,
        view_names: tuple[str, ...],
        dependencies: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        """Return a stable view execution order or raise on cycles."""

        remaining = {name: set(dependencies[name]) for name in view_names}
        order: list[str] = []

        while remaining:
            ready = [
                name
                for name in view_names
                if name in remaining and not remaining[name]
            ]
            if not ready:
                cycle = ", ".join(sorted(remaining))
                raise ValueError(f"Memory view dependency cycle detected among: {cycle}")

            for name in ready:
                order.append(name)
                del remaining[name]
                for deps in remaining.values():
                    deps.discard(name)

        return tuple(order)

    def _bind_materialized_views(self, query: QueryExpr, *, spec: MemorySpec) -> QueryExpr:
        """Replace exact public-view subtrees with materialized view leaves."""

        for name, view in spec.views.items():
            if query == view.query:
                return QueryExpr(op="materialized_view", params={"name": name})

        if not query.inputs:
            return query

        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._bind_materialized_views(input_query, spec=spec)
                for input_query in query.inputs
            ),
            params=query.params,
        )
