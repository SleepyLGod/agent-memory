"""Differentiate one logical query into its incremental form."""

from __future__ import annotations

from collections.abc import Mapping

from agent_memory.policy.logical import MemoryView, QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.planner.rules import (
    DifferentialInstructionRewriter,
    DifferentialRules,
)


class QueryDifferentiator:
    """Differentiate logical queries with one reusable rule configuration."""

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

    @property
    def grouped_agg_rule(self) -> str:
        """Return the canonical grouped aggregate rule used by this planner."""

        return self._rules.grouped_agg_rule

    def differentiate(
        self,
        view: MemoryView,
        *,
        views: Mapping[str, MemoryView] | None = None,
        source_query: QueryExpr | None = None,
        source_input: QueryExpr | None = None,
    ) -> QueryExpr:
        """Generate differentiated query Q' for one memory view."""

        view_name = view.name
        query = view.query
        source_input = source_input or QueryExpr(op="log")
        current_view = QueryExpr(
            op="materialized_view",
            params={
                "name": view_name,
                "columns": self._output_columns(query),
            },
        )
        query = self._bind_materialized_dependencies(
            query,
            current_view_name=view_name,
            views=views or {},
        )
        if source_query is not None:
            source_query = self._bind_materialized_dependencies(
                source_query,
                current_view_name=view_name,
                views=views or {},
            )
        contains_source = (
            self._contains_query(query, source_query)
            if source_query is not None
            else self._contains_op(query, "log")
        )
        if not contains_source:
            return query

        return self._rules.differentiate(
            query,
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            is_view_boundary=True,
            instruction_rewriter=self._instruction_rewriter,
        )

    def differentiate_rows(
        self,
        *,
        query: QueryExpr,
        views: Mapping[str, MemoryView] | None = None,
        source_query: QueryExpr | None = None,
        source_input: QueryExpr | None = None,
    ) -> QueryExpr:
        """Generate changed-output rows for one query."""

        source_input = source_input or QueryExpr(op="log")
        current_view = QueryExpr(
            op="materialized_view",
            params={
                "name": "__unused_current_view",
                "columns": self._output_columns(query),
            },
        )
        query = self._bind_materialized_dependencies(
            query,
            current_view_name="__unused_current_view",
            views=views or {},
        )
        if source_query is not None:
            source_query = self._bind_materialized_dependencies(
                source_query,
                current_view_name="__unused_current_view",
                views=views or {},
            )
        return self._rules.differentiate(
            query,
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            is_view_boundary=False,
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
                return QueryExpr(
                    op="materialized_view",
                    params={
                        "name": name,
                        "columns": self._output_columns(view.query),
                    },
                )

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

        return output_columns(query)

    def _contains_query(self, query: QueryExpr, target: QueryExpr | None) -> bool:
        """Return whether a query tree contains one exact subtree."""

        if target is None:
            return False
        return query == target or any(
            self._contains_query(input_query, target) for input_query in query.inputs
        )
