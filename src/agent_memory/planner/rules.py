"""Rule interfaces for future hardcoded Q-to-Q' rewrites."""

from __future__ import annotations

from typing import Protocol

from agent_memory.logical import RelationExpr
from agent_memory.planner.maintenance import RewriteContext


class RewriteRule(Protocol):
    """Protocol for a single maintenance rewrite rule.

    Rules match expression patterns rather than only root operator names. This
    leaves room for combined rules such as sem_groupby followed by sem_agg.
    """

    def matches(self, expr: RelationExpr, context: RewriteContext) -> bool:
        """Return whether this rule can rewrite the expression."""
        ...

    def rewrite(self, expr: RelationExpr, context: RewriteContext) -> RelationExpr:
        """Rewrite an expression under a maintenance context."""
        ...
