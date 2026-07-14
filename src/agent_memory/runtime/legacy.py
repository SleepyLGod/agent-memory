"""Legacy per-view executor retained for schema-v1 checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agent_memory.api import Message, MessageInput
from agent_memory.planner.legacy import (
    LegacyPolicyPlan,
    OverWindowPlan,
    WindowProcessPlan,
)
from agent_memory.policy.logical import MemoryView, QueryExpr, UserQuery
from agent_memory.policy.relation import (
    LOG_ADDED_AT_COLUMN,
    LOG_ADD_SEQ_COLUMN,
    LOG_ROW_ID_COLUMN,
    LOG_SYSTEM_COLUMNS,
)
from agent_memory.policy.schema import output_columns
from agent_memory.runtime.window import (
    WINDOW_SOURCE_INPUT,
    completed_count_windows,
)


@dataclass(frozen=True)
class _WindowProcessUpdate:
    """Staged window process state, committed only after public view update."""

    changed_rows: Any
    next_private_frame: Any
    changed_upstream_rows: Any
    next_upstream_frame: Any
    next_start: int
    completed_window_count: int


@dataclass(frozen=True)
class _OverWindowUpdate:
    """Staged over-window state, committed only after public view update."""

    changed_rows: Any
    next_private_frame: Any
    changed_upstream_rows: Any
    next_upstream_frame: Any


class LegacyViewRuntime:
    """Runtime orchestrator interface for memory instances.

    This v0.0 module executes differentiated queries Q' for append-time view
    maintenance and policy-owned top-k query plans. It does not implement
    general storage or scheduling.
    """

    def __init__(self, policy: LegacyPolicyPlan, *, adapter: Any | None = None) -> None:
        self.policy = policy
        self.spec = policy.spec
        self.adapter = adapter if adapter is not None else self._default_adapter()
        self._state: dict[str, Any] = {}
        self._window_next_start: dict[str, int] = {}
        self._upstream_log_count: dict[str, int] = {}

    def snapshot_state(self) -> dict[str, Any]:
        """Return the full mutable runtime state needed for checkpoint resume."""

        return {
            "schema_version": 1,
            "state": dict(self._state),
            "window_next_start": dict(self._window_next_start),
            "upstream_log_count": dict(self._upstream_log_count),
        }

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced by snapshot_state()."""

        if snapshot.get("schema_version") != 1:
            raise ValueError("Unsupported runtime snapshot schema_version")
        state = snapshot.get("state")
        window_next_start = snapshot.get("window_next_start")
        upstream_log_count = snapshot.get("upstream_log_count")
        if not isinstance(state, dict):
            raise ValueError("Runtime snapshot state must be a dict")
        if not isinstance(window_next_start, dict):
            raise ValueError("Runtime snapshot window_next_start must be a dict")
        if not isinstance(upstream_log_count, dict):
            raise ValueError("Runtime snapshot upstream_log_count must be a dict")

        self._state = dict(state)
        self._window_next_start = {
            str(name): int(value) for name, value in window_next_start.items()
        }
        self._upstream_log_count = {
            str(name): int(value) for name, value in upstream_log_count.items()
        }

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
            window_plan = self.policy.window_process_plans.get(view_name)
            if window_plan is not None:
                window_update = self._process_completed_windows(
                    window_plan,
                    changed_rows,
                )
                if getattr(window_update.changed_rows, "empty", False):
                    self._state[view.name] = self._state.get(
                        view.name,
                        self._empty_view_frame(view),
                    )
                    self._state[window_plan.upstream_name] = (
                        window_update.next_upstream_frame
                    )
                    self._upstream_log_count[window_plan.upstream_name] = len(
                        self._state["log"]
                    )
                    if window_update.completed_window_count:
                        self._state[window_plan.private_name] = (
                            window_update.next_private_frame
                        )
                        self._window_next_start[window_plan.private_name] = (
                            window_update.next_start
                        )
                    continue

                next_view = self.adapter.execute(
                    differentiated_queries[view.name],
                    self._maintenance_inputs(
                        changed_rows,
                        extra={
                            window_plan.private_name: window_update.next_private_frame,
                            window_plan.changed_name: window_update.changed_rows,
                        },
                    ),
                )
                self._state[window_plan.private_name] = window_update.next_private_frame
                self._state[window_plan.upstream_name] = window_update.next_upstream_frame
                self._upstream_log_count[window_plan.upstream_name] = len(
                    self._state["log"]
                )
                self._window_next_start[window_plan.private_name] = window_update.next_start
                self._state[view.name] = next_view
                continue

            over_plan = self.policy.over_window_plans.get(view_name)
            if over_plan is not None:
                over_update = self._process_over_window(over_plan, changed_rows)
                if getattr(over_update.changed_rows, "empty", False):
                    self._state[view.name] = self._state.get(
                        view.name,
                        self._empty_view_frame(view),
                    )
                    self._state[over_plan.upstream_name] = over_update.next_upstream_frame
                    self._upstream_log_count[over_plan.upstream_name] = len(
                        self._state["log"]
                    )
                    continue

                next_view = self.adapter.execute(
                    differentiated_queries[view.name],
                    self._maintenance_inputs(
                        changed_rows,
                        extra={
                            over_plan.private_name: over_update.next_private_frame,
                            over_plan.changed_name: over_update.changed_rows,
                        },
                    ),
                )
                self._state[over_plan.private_name] = over_update.next_private_frame
                self._state[over_plan.upstream_name] = over_update.next_upstream_frame
                self._upstream_log_count[over_plan.upstream_name] = len(self._state["log"])
                self._state[view.name] = next_view
                continue

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

        row: dict[str, Any]
        if isinstance(message, str):
            row = {"message": message}
        elif isinstance(message, Message):
            row = {"message": message.content}
            if message.role is not None:
                row["role"] = message.role
            if message.timestamp is not None:
                row["timestamp"] = message.timestamp
            if message.session_id is not None:
                row["session_id"] = message.session_id
            if message.metadata is not None:
                row["metadata"] = dict(message.metadata)
        elif isinstance(message, Mapping):
            row = dict(message)
        else:
            raise TypeError("message must be a str, Message, or mapping")

        reserved = sorted(set(row).intersection(LOG_SYSTEM_COLUMNS))
        if reserved:
            raise ValueError(f"message contains reserved log system columns: {reserved}")
        if self.spec.log.expr.params.get("system_columns", False):
            row.update(
                {
                    LOG_ROW_ID_COLUMN: str(uuid4()),
                    LOG_ADDED_AT_COLUMN: datetime.now(timezone.utc),
                    LOG_ADD_SEQ_COLUMN: len(self._state.get("log", ())),
                }
            )
        return row

    def _row_frame(self, row: Mapping[str, Any]) -> Any:
        """Build a one-row pandas DataFrame for runtime state."""

        import pandas as pd

        return pd.DataFrame([dict(row)])

    def _maintenance_inputs(
        self,
        changed_rows: Any,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build adapter inputs for executing differentiated queries Q'."""

        inputs = {
            "log": changed_rows,
            **{
                view.name: self._state.get(view.name, self._empty_view_frame(view))
                for view in self.spec.views.values()
            }
        }
        for name, frame in self._state.items():
            if name != "log" and name not in inputs:
                inputs[name] = frame
        if extra:
            inputs.update(extra)
        return inputs

    def _empty_view_frame(self, view: MemoryView) -> Any:
        """Build an empty DataFrame with the best-known view output columns."""

        import pandas as pd

        return pd.DataFrame(columns=self._query_output_columns(view.query))

    def _query_output_columns(self, query: QueryExpr) -> list[str]:
        """Infer output columns needed for empty materialized-view inputs."""

        if query.op == "materialized_view" and "columns" not in query.params:
            name = str(query.params["name"])
            if name not in self.spec.views:
                raise KeyError(f"Unknown materialized view {name!r}")
            return self._query_output_columns(self.spec.views[name].query)
        return list(output_columns(self._bind_materialized_output_columns(query)))

    def _process_completed_windows(
        self,
        plan: WindowProcessPlan,
        changed_rows: Any,
    ) -> _WindowProcessUpdate:
        """Run a window process body for newly completed count windows."""

        import pandas as pd

        changed_upstream, next_upstream = self._process_upstream_plan(
            plan.upstream_full_query,
            plan.upstream_query,
            plan.upstream_changed_query,
            plan.upstream_name,
            changed_rows,
        )
        windows, next_start = completed_count_windows(
            next_upstream,
            plan.window_query.params,
            next_start=self._window_next_start.get(plan.private_name, 0),
        )
        current_private = self._state.get(
            plan.private_name,
            self._empty_private_frame(plan),
        )
        if not windows:
            return _WindowProcessUpdate(
                changed_rows=self._empty_private_frame(plan),
                next_private_frame=current_private.copy(),
                changed_upstream_rows=changed_upstream,
                next_upstream_frame=next_upstream,
                next_start=next_start,
                completed_window_count=0,
            )

        results: list[Any] = []
        for window in windows:
            result = self.adapter.execute(
                plan.process_query,
                self._window_process_inputs(window.frame),
            )
            results.append(result)

        changed = (
            pd.concat(results, ignore_index=True)
            if results
            else self._empty_private_frame(plan)
        )
        next_private_frame = self._concat_frame(
            current_private,
            changed,
        )
        return _WindowProcessUpdate(
            changed_rows=changed,
            next_private_frame=next_private_frame,
            changed_upstream_rows=changed_upstream,
            next_upstream_frame=next_upstream,
            next_start=next_start,
            completed_window_count=len(windows),
        )

    def _process_over_window(
        self,
        plan: OverWindowPlan,
        changed_rows: Any,
    ) -> _OverWindowUpdate:
        """Run an over-window function for changed upstream emit rows."""

        changed_upstream, next_upstream = self._process_upstream_plan(
            plan.upstream_full_query,
            plan.upstream_query,
            plan.upstream_changed_query,
            plan.upstream_name,
            changed_rows,
        )
        current_private = self._state.get(
            plan.private_name,
            self._empty_over_private_frame(plan),
        )
        if getattr(changed_upstream, "empty", False):
            return _OverWindowUpdate(
                changed_rows=self._empty_over_private_frame(plan),
                next_private_frame=current_private.copy(),
                changed_upstream_rows=changed_upstream,
                next_upstream_frame=next_upstream,
            )

        changed_over = self.adapter.execute(
            plan.over_changed_query,
            self._maintenance_inputs(
                changed_rows,
                extra={
                    plan.upstream_name: next_upstream,
                    plan.upstream_changed_name: changed_upstream,
                },
            ),
        )
        return _OverWindowUpdate(
            changed_rows=changed_over,
            next_private_frame=self._concat_frame(current_private, changed_over),
            changed_upstream_rows=changed_upstream,
            next_upstream_frame=next_upstream,
        )

    def _process_upstream_plan(
        self,
        upstream_full_query: QueryExpr,
        upstream_query: QueryExpr,
        upstream_changed_query: QueryExpr,
        upstream_name: str,
        changed_rows: Any,
    ) -> tuple[Any, Any]:
        """Stage next upstream private state and changed upstream rows."""

        previous_log_count = self._upstream_log_count.get(upstream_name)
        expected_previous_log_count = len(self._state["log"]) - len(changed_rows)
        if (
            upstream_name not in self._state
            or previous_log_count != expected_previous_log_count
        ):
            current = self._state.get(
                upstream_name,
                self._empty_materialized_frame(upstream_name, upstream_query),
            )
            full_inputs = self._maintenance_inputs(
                changed_rows,
                extra={"log": self._state.get("log", changed_rows)},
            )
            next_upstream = self.adapter.execute(upstream_full_query, full_inputs)
            next_upstream = self._canonical_upstream_frame(current, next_upstream)
            return self._changed_suffix(current, next_upstream), next_upstream

        current = self._state[upstream_name]
        inputs = self._maintenance_inputs(changed_rows, extra={upstream_name: current})
        changed_upstream = self.adapter.execute(upstream_changed_query, inputs)
        changed_upstream = self._canonical_upstream_frame(current, changed_upstream)
        next_upstream = self._concat_frame(current, changed_upstream)
        return changed_upstream, next_upstream

    def _changed_suffix(self, current: Any, next_frame: Any) -> Any:
        """Return appended rows after a recomputed upstream state prefix."""

        next_frame = self._canonical_upstream_frame(current, next_frame)
        if getattr(current, "empty", False):
            return next_frame.copy()
        if len(current) > len(next_frame):
            raise NotImplementedError("upstream recompute produced fewer rows")
        prefix = next_frame.iloc[: len(current)].reset_index(drop=True)
        if not prefix.equals(current.reset_index(drop=True)):
            raise NotImplementedError(
                "upstream recompute is not prefix-preserving; cannot derive changed rows"
            )
        return next_frame.iloc[len(current) :].reset_index(drop=True).copy()

    def _canonical_upstream_frame(self, current: Any, next_frame: Any) -> Any:
        """Align upstream frames to an existing schematized state."""

        current_columns = list(current.columns)
        next_columns = list(next_frame.columns)
        if current_columns:
            if any(column not in next_columns for column in current_columns):
                raise NotImplementedError("upstream recompute changed output columns")
            if current_columns != next_columns:
                return next_frame.loc[:, current_columns].copy()
        return next_frame

    def _window_process_inputs(self, window_frame: Any) -> dict[str, Any]:
        """Build adapter inputs for executing one window process body."""

        inputs = dict(self._state)
        inputs[WINDOW_SOURCE_INPUT] = window_frame
        return inputs

    def _empty_private_frame(self, plan: WindowProcessPlan) -> Any:
        """Build an empty private process relation frame."""

        import pandas as pd

        columns = [
            column.name if hasattr(column, "name") else str(column)
            for column in plan.private_source.params["columns"]
        ]
        return pd.DataFrame(columns=columns)

    def _empty_over_private_frame(self, plan: OverWindowPlan) -> Any:
        """Build an empty private over-window relation frame."""

        import pandas as pd

        columns = [
            column.name if hasattr(column, "name") else str(column)
            for column in plan.private_source.params["columns"]
        ]
        return pd.DataFrame(columns=columns)

    def _empty_query_frame(self, query: QueryExpr) -> Any:
        """Build an empty frame for an arbitrary query's inferred columns."""

        import pandas as pd

        return pd.DataFrame(columns=self._query_output_columns(query))

    def _empty_materialized_frame(self, name: str, query: QueryExpr) -> Any:
        """Build an empty frame for a materialized source referenced by a query."""

        import pandas as pd

        columns: list[str] = []
        for node in self._walk_query(query):
            if node.op == "materialized_view" and node.params.get("name") == name:
                columns = [str(column) for column in node.params.get("columns", ())]
                break
        return pd.DataFrame(columns=columns)

    def _walk_query(self, query: QueryExpr) -> list[QueryExpr]:
        """Return query nodes in preorder."""

        nodes = [query]
        for input_query in query.inputs:
            nodes.extend(self._walk_query(input_query))
        return nodes

    def _bind_materialized_output_columns(self, query: QueryExpr) -> QueryExpr:
        """Populate known public materialized-view columns before schema inference."""

        if query.op == "materialized_view" and "columns" not in query.params:
            name = str(query.params["name"])
            if name in self.spec.views:
                return QueryExpr(
                    op="materialized_view",
                    params={
                        "name": name,
                        "columns": self._query_output_columns(self.spec.views[name].query),
                    },
                )
            return query
        if not query.inputs:
            return query
        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._bind_materialized_output_columns(input_query)
                for input_query in query.inputs
            ),
            params=query.params,
        )

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
