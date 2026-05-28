"""Differential rules for hardcoded Q -> DeltaQ rewrites."""

from __future__ import annotations

from agent_memory.logical import QueryExpr


class DifferentialRules:
    """Operator-level differential rules for the current v0.0 subset."""

    def differentiate(self, query: QueryExpr) -> QueryExpr:
        """Derive DeltaQ for one query expression subtree."""

        match query.op:
            case "log":
                return query
            case "select":
                return self._differentiate_unary(query)
            case "sem_filter":
                return self._differentiate_unary(query)
            case "sem_map":
                return self._differentiate_unary(query)
            case _:
                raise NotImplementedError(
                    f"No differential rule for QueryExpr op {query.op!r}."
                )

    def _differentiate_unary(self, query: QueryExpr) -> QueryExpr:
        """Differentiate a row-local unary operator and preserve its params."""

        if len(query.inputs) != 1:
            raise ValueError(
                f"QueryExpr op {query.op!r} expects exactly one input; got {len(query.inputs)}."
            )

        return QueryExpr(
            op=query.op,
            inputs=(self.differentiate(query.inputs[0]),),
            params=query.params,
        )
