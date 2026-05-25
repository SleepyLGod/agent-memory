"""Memory view maintenance planner shell."""

from __future__ import annotations

from dataclasses import dataclass

from agent_memory.logical import MemoryView, RelationExpr


@dataclass(frozen=True)
class RewriteContext:
    """Context available while deriving a maintenance query."""

    view: MemoryView
    delta: RelationExpr


class MemoryViewMaintenancePlanner:
    """Interface for deriving maintenance queries from memory view definitions.

    This class names the future Q-to-Q' boundary only. The current v0.0 interface does
    not implement rewrite rules or pretend to produce executable plans.
    """

    def plan(self, view: MemoryView, delta: RelationExpr) -> RelationExpr:
        """Derive a maintenance query for a memory view and input delta."""

        raise NotImplementedError(
            "Memory view maintenance planning is not implemented in the current v0.0 interface layer."
        )
