"""Rule interfaces for future hardcoded Q -> DeltaQ rewrites."""

from __future__ import annotations

from typing import Protocol

from agent_memory.logical import MemoryView, QueryExpr


class RewriteRule(Protocol):
    """Protocol for a single differential rewrite rule.

    Rules match expression patterns rather than only root operator names. This
    leaves room for combined rules such as sem_groupby followed by sem_agg.
    """

    # Rules receive the current query subtree and the MemoryView being
    # differentiated. DeltaD is not modeled as a v0 QueryExpr/operator here; it
    # is the runtime changed rows for the relevant input relation when DeltaQ is
    # eventually executed.
    def matches(self, query: QueryExpr, view: MemoryView) -> bool:
        """Return whether this rule can rewrite the expression."""
        ...

    def rewrite(self, query: QueryExpr, view: MemoryView) -> QueryExpr:
        """Rewrite a query subtree for one memory view."""
        ...
