"""Runtime shell for the current memory interface layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_memory.logical import MemorySpec, MemoryView, QueryExpr
from agent_memory.message import Message, MessageInput
from agent_memory.planner import DifferentialQueryPlanner

if TYPE_CHECKING:
    from agent_memory.relation import Relation


class MemoryRuntime:
    """Runtime orchestrator interface for memory instances.

    This v0.0 module implements only row-local sem_filter/sem_map maintenance
    and top-k query execution. It does not implement general storage,
    scheduling, or full semantic view maintenance.
    """

    def __init__(self, spec: MemorySpec, *, adapter: Any | None = None) -> None:
        self.spec = spec
        self.adapter = adapter if adapter is not None else self._default_adapter()
        self._state: dict[str, Any] = {}
        self._planner = DifferentialQueryPlanner()

    def _default_adapter(self) -> Any:
        """Create the default semantic execution backend."""

        from agent_memory.adapters import LotusAdapter

        return LotusAdapter()

    def add(self, message: MessageInput) -> None:
        """Append an end-user message or event to the source log.

        A string is Message(content=...) sugar. Future runtime code will
        normalize Message.content into the log column named "message", preserve
        same-name message fields when available, and maintain derived views. The
        current implementation executes only row-local sem_filter/sem_map views
        when an adapter is provided.
        """

        if self.adapter is None:
            raise NotImplementedError(
                "Memory add/log maintenance requires an execution adapter in the current v0.0 interface layer."
            )

        changed_rows = self._row_frame(self._normalize_message(message))
        differential_queries = self._differential_queries()
        self._append_log_state(changed_rows)
        self._maintain_views(differential_queries, changed_rows)

    def _differential_queries(self) -> dict[str, QueryExpr]:
        """Derive DeltaQ templates for all public views."""

        return {
            view.name: self._planner.differentiate(view)
            for view in self.spec.views.values()
        }

    def _append_log_state(self, changed_rows: Any) -> None:
        """Append changed rows to the runtime-owned source log state."""

        self._state["log"] = self._append_log_frame(
            self._state.get("log"),
            changed_rows,
        )

    def _maintain_views(
        self,
        differential_queries: Mapping[str, QueryExpr],
        changed_rows: Any,
    ) -> None:
        """Execute DeltaQ templates and merge materialized view deltas."""

        for view in self.spec.views.values():
            delta_view = self.adapter.execute(
                differential_queries[view.name],
                {"log": changed_rows},
            )
            self._state[view.name] = self._union_view_frame(
                self._state.get(view.name),
                delta_view,
            )

    def execute_query(self, plan: "Relation") -> Any:
        """Execute a policy-defined semantic query plan.

        The runtime does not choose a retrieval view. Concrete memory policies
        construct the query plan first, then hand it to this future execution
        hook.
        """

        if self.adapter is None:
            raise NotImplementedError(
                "Memory query plan execution requires an execution adapter in the current v0.0 interface layer."
            )

        query = self._bind_materialized_views(plan.expr)
        return self.adapter.execute(query, self._state)

    def _normalize_message(self, message: MessageInput) -> dict[str, Any]:
        """Normalize supported append inputs into one log row."""

        if isinstance(message, str):
            return {"message": message}
        if isinstance(message, Message):
            row: dict[str, Any] = {"message": message.content}
            if message.role is not None:
                row["role"] = message.role
            if message.timestamp is not None:
                row["timestamp"] = message.timestamp
            if message.session_id is not None:
                row["session_id"] = message.session_id
            if message.metadata is not None:
                row["metadata"] = dict(message.metadata)
            return row
        if isinstance(message, Mapping):
            return dict(message)
        raise TypeError("message must be a str, Message, or mapping")

    def _row_frame(self, row: Mapping[str, Any]) -> Any:
        """Build a one-row pandas DataFrame for runtime state."""

        import pandas as pd

        return pd.DataFrame([dict(row)])

    def _append_log_frame(self, current: Any | None, delta: Any) -> Any:
        """Append changed rows to the source log state."""

        return self._concat_frame(current, delta)

    def _union_view_frame(self, current: Any | None, delta: Any) -> Any:
        """Merge changed view rows using exact union semantics."""

        merged = self._concat_frame(current, delta)
        return merged.drop_duplicates(ignore_index=True)

    def _concat_frame(self, current: Any | None, delta: Any) -> Any:
        """Append delta rows to an existing pandas DataFrame-like state."""

        import pandas as pd

        if current is None:
            return delta.copy()
        if getattr(delta, "empty", False):
            return current.copy()
        return pd.concat([current, delta], ignore_index=True)

    def _bind_materialized_views(self, query: QueryExpr) -> QueryExpr:
        """Replace known view-definition subtrees with materialized view refs."""

        matched_view = self._matching_view(query)
        if matched_view is not None:
            return QueryExpr(
                op="materialized_view",
                params={"name": matched_view.name},
            )

        if not query.inputs:
            return query

        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._bind_materialized_views(input_query)
                for input_query in query.inputs
            ),
            params=query.params,
        )

    def _matching_view(self, query: QueryExpr) -> MemoryView | None:
        """Return the materialized view whose definition exactly matches query."""

        for view in self.spec.views.values():
            if view.query == query:
                return view
        return None
