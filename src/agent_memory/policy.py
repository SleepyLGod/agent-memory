"""Policy-level differentiation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent_memory.logical import MemorySpec, QueryExpr
from agent_memory.planner import DifferentialQueryPlanner
from agent_memory.query_schema import output_columns


@dataclass(frozen=True)
class WindowProcessPlan:
    """Runtime plan for one count_window(...).process_window(...) boundary."""

    view_name: str
    private_name: str
    changed_name: str
    window_query: QueryExpr
    process_query: QueryExpr
    private_source: QueryExpr
    changed_source: QueryExpr
    upstream_name: str
    upstream_changed_name: str
    upstream_full_query: QueryExpr
    upstream_query: QueryExpr
    upstream_changed_query: QueryExpr
    upstream_source: QueryExpr
    upstream_changed_source: QueryExpr


@dataclass(frozen=True)
class OverWindowPlan:
    """Runtime plan for one over(...).array_agg/sem_agg boundary."""

    view_name: str
    private_name: str
    changed_name: str
    upstream_name: str
    upstream_changed_name: str
    upstream_full_query: QueryExpr
    upstream_query: QueryExpr
    upstream_changed_query: QueryExpr
    over_changed_query: QueryExpr
    private_source: QueryExpr
    changed_source: QueryExpr
    upstream_source: QueryExpr
    upstream_changed_source: QueryExpr


@dataclass(frozen=True)
class DifferentiatedPolicy:
    """In-memory policy artifact produced by whole-policy differentiation."""

    spec: MemorySpec
    view_queries: Mapping[str, QueryExpr]
    retrieval_queries: Mapping[str, QueryExpr]
    view_dependencies: Mapping[str, tuple[str, ...]]
    view_execution_order: tuple[str, ...]
    window_process_plans: Mapping[str, WindowProcessPlan]
    over_window_plans: Mapping[str, OverWindowPlan]

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
            "window_process_plans",
            MappingProxyType(dict(self.window_process_plans)),
        )
        object.__setattr__(
            self,
            "over_window_plans",
            MappingProxyType(dict(self.over_window_plans)),
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
        view_queries: dict[str, QueryExpr] = {}
        window_process_plans: dict[str, WindowProcessPlan] = {}
        over_window_plans: dict[str, OverWindowPlan] = {}
        for name in view_execution_order:
            plan, downstream_query = self._window_process_plan(
                name,
                spec.views[name].query,
                spec=spec,
            )
            if plan is not None:
                window_process_plans[name] = plan
                view_queries[name] = self._view_planner.differentiate_query(
                    view_name=name,
                    query=downstream_query,
                    views=spec.views,
                    source_query=plan.private_source,
                    source_input=plan.changed_source,
                )
                continue

            over_plan, downstream_query = self._over_window_plan(
                name,
                spec.views[name].query,
                spec=spec,
            )
            if over_plan is not None:
                over_window_plans[name] = over_plan
                view_queries[name] = self._view_planner.differentiate_query(
                    view_name=name,
                    query=downstream_query,
                    views=spec.views,
                    source_query=over_plan.private_source,
                    source_input=over_plan.changed_source,
                )
                continue

            view_queries[name] = self._view_planner.differentiate_query(
                view_name=name,
                query=spec.views[name].query,
                views=spec.views,
            )

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
            window_process_plans=window_process_plans,
            over_window_plans=over_window_plans,
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

    def _window_process_plan(
        self,
        view_name: str,
        query: QueryExpr,
        *,
        spec: MemorySpec,
    ) -> tuple[WindowProcessPlan | None, QueryExpr]:
        """Extract a single process_window boundary and downstream query."""

        matches = self._find_process_window_nodes(query)
        if not matches:
            return None, query
        if len(matches) > 1:
            raise NotImplementedError(
                f"{view_name} contains multiple process_window nodes; v0.0 supports one."
            )

        process_window = matches[0]
        window_query, process_query = process_window.inputs
        if window_query.op != "count_window" or len(window_query.inputs) != 1:
            raise ValueError("process_window first input must be count_window")
        upstream_query = window_query.inputs[0]

        private_name = f"_{view_name}_process_window"
        changed_name = f"{private_name}__changed"
        upstream_name = f"{private_name}__source"
        upstream_changed_name = f"{upstream_name}__changed"
        process_columns = self._output_columns(process_window)
        upstream_columns = self._output_columns(upstream_query)
        private_source = QueryExpr(
            op="materialized_view",
            params={"name": private_name, "columns": process_columns},
        )
        changed_source = QueryExpr(
            op="materialized_view",
            params={"name": changed_name, "columns": process_columns},
        )
        upstream_source = QueryExpr(
            op="materialized_view",
            params={"name": upstream_name, "columns": upstream_columns},
        )
        upstream_changed_source = QueryExpr(
            op="materialized_view",
            params={"name": upstream_changed_name, "columns": upstream_columns},
        )
        upstream_changed_query = self._view_planner.differentiate_rows(
            query=upstream_query,
            views=spec.views,
            source_input=QueryExpr(op="log"),
        )
        upstream_next_query = QueryExpr(
            op="concat",
            inputs=(upstream_source, upstream_changed_query),
        )
        downstream_query = self._replace_query(query, process_window, private_source)
        plan = WindowProcessPlan(
            view_name=view_name,
            private_name=private_name,
            changed_name=changed_name,
            window_query=window_query,
            process_query=process_query,
            private_source=private_source,
            changed_source=changed_source,
            upstream_name=upstream_name,
            upstream_changed_name=upstream_changed_name,
            upstream_full_query=upstream_query,
            upstream_query=upstream_next_query,
            upstream_changed_query=upstream_changed_query,
            upstream_source=upstream_source,
            upstream_changed_source=upstream_changed_source,
        )
        return plan, downstream_query

    def _over_window_plan(
        self,
        view_name: str,
        query: QueryExpr,
        *,
        spec: MemorySpec,
    ) -> tuple[OverWindowPlan | None, QueryExpr]:
        """Extract one over-window function boundary and downstream query."""

        matches = self._find_over_window_functions(query)
        if not matches:
            return None, query
        if len(matches) > 1:
            raise NotImplementedError(
                f"{view_name} contains multiple over window functions; v0.0 supports one."
            )

        over_function = matches[0]
        over_query = over_function.inputs[0]
        upstream_query = over_query.inputs[0]
        private_name = f"_{view_name}_over_window"
        changed_name = f"{private_name}__changed"
        upstream_name = f"{private_name}__source"
        upstream_changed_name = f"{upstream_name}__changed"
        private_columns = self._output_columns(over_function)
        upstream_columns = self._output_columns(upstream_query)
        private_source = QueryExpr(
            op="materialized_view",
            params={"name": private_name, "columns": private_columns},
        )
        changed_source = QueryExpr(
            op="materialized_view",
            params={"name": changed_name, "columns": private_columns},
        )
        upstream_source = QueryExpr(
            op="materialized_view",
            params={"name": upstream_name, "columns": upstream_columns},
        )
        upstream_changed_source = QueryExpr(
            op="materialized_view",
            params={"name": upstream_changed_name, "columns": upstream_columns},
        )
        upstream_changed_query = self._view_planner.differentiate_rows(
            query=upstream_query,
            views=spec.views,
            source_input=QueryExpr(op="log"),
        )
        upstream_next_query = QueryExpr(
            op="concat",
            inputs=(upstream_source, upstream_changed_query),
        )
        over_changed_query = self._over_changed_query(
            over_function,
            emit_source=upstream_changed_source,
            frame_source=upstream_source,
        )
        downstream_query = self._replace_query(query, over_function, private_source)
        plan = OverWindowPlan(
            view_name=view_name,
            private_name=private_name,
            changed_name=changed_name,
            upstream_name=upstream_name,
            upstream_changed_name=upstream_changed_name,
            upstream_full_query=upstream_query,
            upstream_query=upstream_next_query,
            upstream_changed_query=upstream_changed_query,
            over_changed_query=over_changed_query,
            private_source=private_source,
            changed_source=changed_source,
            upstream_source=upstream_source,
            upstream_changed_source=upstream_changed_source,
        )
        return plan, downstream_query

    def _find_over_window_functions(self, query: QueryExpr) -> list[QueryExpr]:
        """Return array_agg/sem_agg nodes that consume an over relation."""

        is_over_function = (
            query.op in {"array_agg", "sem_agg"}
            and len(query.inputs) == 1
            and query.inputs[0].op == "over"
        )
        nodes = [query] if is_over_function else []
        for input_query in query.inputs:
            nodes.extend(self._find_over_window_functions(input_query))
        return nodes

    def _over_changed_query(
        self,
        query: QueryExpr,
        *,
        emit_source: QueryExpr,
        frame_source: QueryExpr,
    ) -> QueryExpr:
        """Bind an over-window function to changed emit rows and full frame state."""

        over_query = query.inputs[0]
        # Internal-only binding: public over(...) has one input, but maintenance
        # needs changed emit rows plus the full frame source. A future stricter IR
        # can replace this param with an internal second input or binding wrapper.
        changed_over = QueryExpr(
            op="over",
            inputs=(emit_source,),
            params={**over_query.params, "frame_source": frame_source},
        )
        return QueryExpr(
            op=query.op,
            inputs=(changed_over,),
            params=query.params,
        )

    def _find_process_window_nodes(self, query: QueryExpr) -> list[QueryExpr]:
        """Return process_window nodes in one query tree."""

        nodes = [query] if query.op == "process_window" else []
        for input_query in query.inputs:
            nodes.extend(self._find_process_window_nodes(input_query))
        return nodes

    def _replace_query(
        self,
        query: QueryExpr,
        target: QueryExpr,
        replacement: QueryExpr,
    ) -> QueryExpr:
        """Replace one exact query subtree."""

        if query == target:
            return replacement
        if not query.inputs:
            return query
        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._replace_query(input_query, target, replacement)
                for input_query in query.inputs
            ),
            params=query.params,
        )

    def _output_columns(self, query: QueryExpr) -> tuple[str, ...]:
        """Infer output columns needed for private process sources."""

        return output_columns(query)
