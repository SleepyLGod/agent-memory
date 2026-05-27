"""Differential query planner shell."""

from __future__ import annotations

from agent_memory.logical import MemoryView, QueryExpr


class DifferentialQueryPlanner:
    """Interface for deriving DeltaQ from memory view definition queries.

    This class names the future Q -> DeltaQ boundary only. The current v0.0
    interface does not implement rewrite rules or pretend to produce executable
    plans.
    """

    def differentiate(self, view: MemoryView) -> QueryExpr:
        """Derive a differential query template DeltaQ for one memory view."""

        # view.query is the full view definition query Q. view.name identifies
        # the materialized view V that runtime will maintain. Concrete DeltaD
        # rows are not part of this rewrite interface: they appear only when
        # runtime eventually executes the derived DeltaQ template.
        raise NotImplementedError(
            "Differential query planning is not implemented in the current v0.0 interface layer."
        )
