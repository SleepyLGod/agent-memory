"""Runtime shell for the current memory interface layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_memory.logical import MemoryView, QueryExpr, UserQuery
from agent_memory.message import Message, MessageInput
from agent_memory.policy import DifferentiatedPolicy


class MemoryRuntime:
    """Runtime orchestrator interface for memory instances.

    This v0.0 module executes differentiated queries Q' for append-time view
    maintenance and policy-owned top-k query plans. It does not implement
    general storage or scheduling.
    """

    def __init__(self, policy: DifferentiatedPolicy, *, adapter: Any | None = None) -> None:
        self.policy = policy
        self.spec = policy.spec
        self.adapter = adapter if adapter is not None else self._default_adapter()
        self._state: dict[str, Any] = {}

    def _default_adapter(self) -> Any:
        """Create the default semantic execution backend."""

        from agent_memory.adapters import LotusAdapter

        return LotusAdapter()

    def add(self, message: MessageInput) -> None:
        """Append an end-user message or event to the source log.

        A string is Message(content=...) sugar. Future runtime code will
        normalize Message.content into the log column named "message", preserve
        same-name message fields when available, and maintain derived views by
        executing differentiated queries Q'.
        """

        if self.adapter is None:
            raise NotImplementedError(
                "Memory add/log maintenance requires an execution adapter in the "
                "current v0.0 interface layer."
            )

        changed_rows = self._row_frame(self._normalize_message(message))
        self._append_log_state(changed_rows)
        self._maintain_views(self.policy.view_queries, changed_rows)

    def _append_log_state(self, changed_rows: Any) -> None:
        """Append changed rows to the runtime-owned source log state."""

        self._state["log"] = self._append_log_frame(
            self._state.get("log"),
            changed_rows,
        )

    def _maintain_views(
        self,
        differentiated_queries: Mapping[str, QueryExpr],
        changed_rows: Any,
    ) -> None:
        """Execute differentiated queries Q' and store next view states."""

        for view_name in self.policy.view_execution_order:
            view = self.spec.views[view_name]
            next_view = self.adapter.execute(
                differentiated_queries[view.name],
                self._maintenance_inputs(changed_rows),
            )
            self._state[view.name] = next_view

    def execute_retrieval_query(self, name: str, text: str) -> Any:
        """Bind a user query string and execute a compiled retrieval template."""

        if self.adapter is None:
            raise NotImplementedError(
                "Memory retrieval query execution requires an execution adapter in the "
                "current v0.0 interface layer."
            )
        if name not in self.policy.retrieval_queries:
            raise NotImplementedError(
                f"Memory policy does not declare a retrieval query named {name!r}."
            )

        template = self.policy.retrieval_queries[name]
        bound_query = self._bind_user_query(template, text)
        return self.adapter.execute(bound_query, self._state)

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

    def _maintenance_inputs(self, changed_rows: Any) -> dict[str, Any]:
        """Build adapter inputs for executing differentiated queries Q'."""

        return {
            "log": changed_rows,
            **{
                view.name: self._state.get(view.name, self._empty_view_frame(view))
                for view in self.spec.views.values()
            },
        }

    def _empty_view_frame(self, view: MemoryView) -> Any:
        """Build an empty DataFrame with the best-known view output columns."""

        import pandas as pd

        return pd.DataFrame(columns=self._query_output_columns(view.query))

    def _query_output_columns(self, query: QueryExpr) -> list[str]:
        """Infer output columns needed for empty materialized-view inputs."""

        match query.op:
            case "select":
                return [str(column) for column in query.params["columns"]]
            case "log":
                columns = query.params.get("columns", ())
                return [column.name for column in columns]
            case "materialized_view":
                name = str(query.params["name"])
                if name not in self.spec.views:
                    raise KeyError(f"Unknown materialized view {name!r}")
                return self._query_output_columns(self.spec.views[name].query)
            case "sem_agg":
                output_cols = query.params.get("output_cols")
                if output_cols is not None:
                    return [column.name for column in output_cols]
                return self._query_output_columns(query.inputs[0])
            case "sem_map" | "sem_flat_map":
                columns = self._query_output_columns(query.inputs[0])
                output_cols = query.params.get("output_cols") or ()
                for column in output_cols:
                    if column.name not in columns:
                        columns.append(column.name)
                return columns
            case "sem_filter" | "sem_groupby" | "sem_topk" | "drop_duplicates":
                return self._query_output_columns(query.inputs[0])
            case "join":
                return self._join_output_columns(query)
            case "union" | "concat" | "subtract" | "sem_join":
                columns: list[str] = []
                for input_query in query.inputs:
                    for column in self._query_output_columns(input_query):
                        if column not in columns:
                            columns.append(column)
                return columns
            case _:
                raise NotImplementedError(
                    f"Cannot infer output columns for QueryExpr op {query.op!r}."
                )

    def _join_output_columns(self, query: QueryExpr) -> list[str]:
        """Infer pandas merge output columns for same-key relational joins."""

        left_columns = self._query_output_columns(query.inputs[0])
        right_columns = self._query_output_columns(query.inputs[1])
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
        return columns

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

    def _bind_user_query(self, query: QueryExpr, value: str) -> QueryExpr:
        """Replace UserQuery placeholders inside a retrieval template."""

        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._bind_user_query(input_query, value)
                for input_query in query.inputs
            ),
            params={
                key: self._bind_user_query_value(param, value)
                for key, param in query.params.items()
            },
        )

    def _bind_user_query_value(self, value: Any, text: str) -> Any:
        """Bind UserQuery placeholders nested inside expression params."""

        if isinstance(value, UserQuery):
            return text
        if isinstance(value, Mapping):
            return {
                key: self._bind_user_query_value(item, text)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._bind_user_query_value(item, text) for item in value)
        return value
