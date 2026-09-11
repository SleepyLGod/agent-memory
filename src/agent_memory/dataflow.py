"""Public facade for executing declarative semantic dataflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .planner.differential_policy import PolicyDifferentiator
from .policy.logical import MemorySpec, MemoryView
from .policy.relation import Log, Relation
from .runtime.executor import PolicyExecutor

Source = Log


class SemanticDataflow:
    """Compile and execute named semantic views over one source relation."""

    def __init__(
        self,
        *,
        source: Source,
        views: Mapping[str, Relation],
        adapter: Any | None = None,
    ) -> None:
        if not isinstance(source, Log):
            raise TypeError("source must be a Source")
        if not views:
            raise ValueError("views cannot be empty")

        declared_views: dict[str, MemoryView] = {}
        for name, relation in views.items():
            if not isinstance(name, str) or not name:
                raise ValueError("view names must be non-empty strings")
            if not isinstance(relation, Relation):
                raise TypeError("views values must be Relation instances")
            pending = [relation.expr]
            visited: set[int] = set()
            while pending:
                query = pending.pop()
                if id(query) in visited:
                    continue
                visited.add(id(query))
                if query.op == "log" and query is not source.expr:
                    raise ValueError(
                        f"view {name!r} must derive from the declared source"
                    )
                pending.extend(query.inputs)
            declared_views[name] = MemoryView(name=name, query=relation.expr)

        spec = MemorySpec(
            log=source,
            views=declared_views,
            private_relations={},
        )
        self._executor = PolicyExecutor(
            PolicyDifferentiator().differentiate(spec),
            adapter=adapter,
        )

    def apply(self, rows: pd.DataFrame) -> None:
        """Apply one relation-valued source batch to all declared views."""

        self._executor.apply_delta(rows)

    def view(self, name: str) -> pd.DataFrame:
        """Return a defensive copy of one named public view."""

        return self._executor.read_view(name)

    def snapshot_state(self) -> dict[str, Any]:
        """Return the executor's existing snapshot for trusted persistence."""
        return self._executor.snapshot_state()

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot, including the executor's compatibility checks."""
        self._executor.restore_state(snapshot)
