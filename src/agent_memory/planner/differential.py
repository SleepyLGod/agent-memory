"""Differential query planner."""

from __future__ import annotations

from collections.abc import Mapping

from agent_memory.logical import MemoryView, QueryExpr
from agent_memory.planner.rules import (
    DifferentialInstructionRewriter,
    DifferentialRules,
)


class DifferentialQueryPlanner:
    """View-level Q -> Q' planner."""

    def __init__(
        self,
        rules: DifferentialRules | None = None,
        instruction_rewriter: DifferentialInstructionRewriter | None = None,
    ) -> None:
        self._rules = rules if rules is not None else DifferentialRules()
        self._instruction_rewriter = (
            instruction_rewriter
            if instruction_rewriter is not None
            else DifferentialInstructionRewriter()
        )

    def differentiate(
        self,
        view: MemoryView,
        *,
        views: Mapping[str, MemoryView] | None = None,
    ) -> QueryExpr:
        """Generate differentiated query Q' for one memory view."""

        source_input = QueryExpr(op="log")
        current_view = QueryExpr(
            op="materialized_view",
            params={
                "name": view.name,
                "columns": self._output_columns(view.query),
            },
        )
        query = self._bind_materialized_dependencies(
            view.query,
            current_view_name=view.name,
            views=views or {},
        )
        if query != view.query and not self._contains_op(query, "log"):
            return query

        return self._rules.differentiate(
            query,
            source_input=source_input,
            current_view=current_view,
            is_view_boundary=True,
            instruction_rewriter=self._instruction_rewriter,
        )

    def _bind_materialized_dependencies(
        self,
        query: QueryExpr,
        *,
        current_view_name: str,
        views: Mapping[str, MemoryView],
    ) -> QueryExpr:
        """Replace public view dependency subtrees with materialized-view leaves."""

        for name, view in views.items():
            if name != current_view_name and query == view.query:
                return QueryExpr(op="materialized_view", params={"name": name})

        if not query.inputs:
            return query

        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._bind_materialized_dependencies(
                    input_query,
                    current_view_name=current_view_name,
                    views=views,
                )
                for input_query in query.inputs
            ),
            params=query.params,
        )

    def _contains_op(self, query: QueryExpr, op: str) -> bool:
        """Return whether a query tree contains an operator."""

        return query.op == op or any(
            self._contains_op(input_query, op) for input_query in query.inputs
        )

    def _output_columns(self, query: QueryExpr) -> tuple[str, ...]:
        """Infer output columns for materialized-view placeholders."""

        if query.op == "select":
            return tuple(str(column) for column in query.params["columns"])
        if query.op == "log":
            return tuple(column.name for column in query.params.get("columns", ()))
        if query.op == "sem_agg":
            output_cols = query.params.get("output_cols")
            if output_cols is not None:
                return tuple(column.name for column in output_cols)
            input_cols = query.params.get("input_cols")
            if input_cols is not None:
                return tuple(str(column) for column in input_cols)
        if query.op in {"sem_map", "sem_flat_map"}:
            columns = list(self._output_columns(query.inputs[0]))
            for column in query.params.get("output_cols") or ():
                if column.name not in columns:
                    columns.append(column.name)
            return tuple(columns)
        if query.op in {"sem_filter", "sem_groupby", "sem_topk", "drop_duplicates"}:
            return self._output_columns(query.inputs[0])
        if query.op == "join":
            return self._join_output_columns(query)
        if query.op in {"union", "concat", "subtract", "sem_join"}:
            columns: list[str] = []
            for input_query in query.inputs:
                for column in self._output_columns(input_query):
                    if column not in columns:
                        columns.append(column)
            return tuple(columns)
        raise NotImplementedError(
            f"Cannot infer output columns for QueryExpr op {query.op!r}."
        )

    def _join_output_columns(self, query: QueryExpr) -> tuple[str, ...]:
        """Infer pandas merge output columns for same-key relational joins."""

        left_columns = self._output_columns(query.inputs[0])
        right_columns = self._output_columns(query.inputs[1])
        keys = tuple(str(column) for column in query.params["on"])
        overlapping = (
            set(left_columns).intersection(right_columns).difference(keys)
        )

        columns: list[str] = []
        for column in left_columns:
            if column in overlapping:
                columns.append(f"{column}:left")
            else:
                columns.append(column)
        for column in right_columns:
            if column in keys:
                continue
            if column in overlapping:
                columns.append(f"{column}:right")
            else:
                columns.append(column)
        return tuple(columns)
