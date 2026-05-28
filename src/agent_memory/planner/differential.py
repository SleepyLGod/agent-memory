"""Differential query planner."""

from __future__ import annotations

from agent_memory.logical import MemoryView, QueryExpr
from agent_memory.planner.rules import DifferentialRules


class DifferentialQueryPlanner:
    """View-level Q -> DeltaQ planner."""

    def __init__(self, rules: DifferentialRules | None = None) -> None:
        self._rules = rules if rules is not None else DifferentialRules()

    def differentiate(self, view: MemoryView) -> QueryExpr:
        """Derive a differential query template DeltaQ for one memory view."""

        return self._rules.differentiate(view.query)
