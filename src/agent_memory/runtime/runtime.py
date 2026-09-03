"""Stable memory runtime entry point and checkpoint engine selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from agent_memory.api import CountRefresh, MessageInput
from agent_memory.planner.differential_policy import DifferentiatedPolicy
from agent_memory.planner.legacy import differentiate_legacy_policy
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.relation import LOG_ADD_SEQ_COLUMN, LOG_ROW_ID_COLUMN
from agent_memory.runtime.executor import PolicyExecutor
from agent_memory.runtime.legacy import LegacyViewRuntime
from agent_memory.storage.deployment import StorageDeployment


class MemoryRuntime:
    """Run a policy with the v2 executor or its schema-v1 compatibility engine."""

    def __init__(
        self,
        policy: DifferentiatedPolicy,
        *,
        adapter: Any | None = None,
        storage: StorageDeployment | None = None,
        refresh: CountRefresh | None = None,
    ) -> None:
        self.policy = policy
        self.storage = storage
        self.refresh = refresh
        self._pending_rows: list[dict[str, Any]] = []
        self._engine: PolicyExecutor | LegacyViewRuntime = PolicyExecutor(
            policy,
            adapter=adapter,
            storage=storage,
        )

    @property
    def _state(self) -> dict[str, Any]:
        """Expose source and public views to retrieval and diagnostics."""

        return self._engine._state

    @_state.setter
    def _state(self, state: Mapping[str, Any]) -> None:
        if isinstance(self._engine, PolicyExecutor):
            self._engine.replace_public_state(state)
        else:
            self._engine._state = dict(state)

    def add(self, message: MessageInput) -> None:
        """Append one input through the selected execution engine."""

        if self.refresh is None or self.refresh.every == 1:
            self._engine.add(message)
            return
        if not isinstance(self._engine, PolicyExecutor):
            raise RuntimeError("count refresh requires the schema-v2 policy executor")

        row = self._engine.normalize_source_row(
            message,
            add_seq=self._engine.source_row_count + len(self._pending_rows),
        )
        self._pending_rows.append(row)
        if len(self._pending_rows) >= self.refresh.every:
            self.flush()

    @property
    def pending_count(self) -> int:
        """Return the number of normalized source rows awaiting refresh."""

        return len(self._pending_rows)

    def flush(self) -> None:
        """Apply all pending rows in one atomic policy-execution step."""

        if not self._pending_rows:
            return
        if not isinstance(self._engine, PolicyExecutor):
            raise RuntimeError("count refresh requires the schema-v2 policy executor")

        changed_rows = pd.DataFrame(
            self._pending_rows,
            columns=self._engine.source_columns,
        )
        self._engine.apply_delta(changed_rows)
        self._pending_rows.clear()

    def execute_retrieval_query(self, name: str, text: str) -> Any:
        """Execute one policy-owned retrieval query."""

        return self._engine.execute_retrieval_query(name, text)

    def snapshot_state(self) -> dict[str, Any]:
        """Return a checkpoint in the active engine's schema."""

        engine_state = self._engine.snapshot_state()
        if self.refresh is None or self.refresh.every == 1:
            return engine_state
        if not isinstance(self._engine, PolicyExecutor):
            raise RuntimeError("count refresh requires the schema-v2 policy executor")
        return {
            "schema_version": 3,
            "refresh": {"type": "count", "every": self.refresh.every},
            "pending_rows": pd.DataFrame(
                self._pending_rows,
                columns=self._engine.source_columns,
            ),
            "engine_state": engine_state,
        }

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore v2 directly or route schema-v1 state to the legacy engine."""

        schema_version = snapshot.get("schema_version")
        if schema_version == 3:
            self._restore_count_refresh_state(snapshot)
            return
        if self.refresh is not None and self.refresh.every > 1:
            raise ValueError(
                "count refresh contract requires a schema-v3 runtime checkpoint"
            )
        if schema_version == 1:
            if self.storage is not None:
                raise NotImplementedError(
                    "schema-v1 checkpoints cannot be restored with storage"
                )
            legacy_policy = differentiate_legacy_policy(
                self.policy.spec,
                grouped_agg_rule=self.policy.grouped_agg_rule,
            )
            engine = LegacyViewRuntime(legacy_policy, adapter=self._engine.adapter)
            engine.restore_state(snapshot)
            self._engine = engine
            return
        if schema_version == 2:
            if not isinstance(self._engine, PolicyExecutor):
                self._engine = PolicyExecutor(
                    self.policy,
                    adapter=self._engine.adapter,
                    storage=self.storage,
                )
            self._engine.restore_state(snapshot)
            return
        raise ValueError("Unsupported runtime snapshot schema_version")

    def _restore_count_refresh_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a schema-v3 count-refresh envelope atomically."""

        if self.refresh is None or self.refresh.every == 1:
            raise ValueError(
                "schema-v3 checkpoint requires a matching count refresh contract"
            )
        if not isinstance(self._engine, PolicyExecutor):
            raise ValueError("count refresh requires the schema-v2 policy executor")

        refresh = snapshot.get("refresh")
        expected_refresh = {"type": "count", "every": self.refresh.every}
        if refresh != expected_refresh:
            raise ValueError("runtime snapshot refresh contract does not match")
        engine_state = snapshot.get("engine_state")
        if not isinstance(engine_state, Mapping):
            raise ValueError("schema-v3 checkpoint requires engine_state")
        pending_rows = snapshot.get("pending_rows")
        if not isinstance(pending_rows, pd.DataFrame):
            raise ValueError("schema-v3 checkpoint requires DataFrame pending_rows")
        if list(pending_rows.columns) != list(self._engine.source_columns):
            raise ValueError(
                "runtime snapshot pending columns do not match the log schema"
            )

        pending_records = self._validate_pending_rows(
            pending_rows,
            engine_state=engine_state,
        )
        self._engine.restore_state(engine_state)
        self._pending_rows = pending_records

    def _validate_pending_rows(
        self,
        pending_rows: pd.DataFrame,
        *,
        engine_state: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Validate pending source identities before mutating engine state."""

        state = engine_state.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("schema-v3 engine_state requires state")
        committed_rows = state.get("log")
        if committed_rows is None:
            committed_rows = pd.DataFrame(columns=self._engine.source_columns)
        elif not isinstance(committed_rows, pd.DataFrame):
            raise ValueError("schema-v3 engine_state requires DataFrame log state")

        if LOG_ADD_SEQ_COLUMN in pending_rows.columns:
            expected = list(
                range(len(committed_rows), len(committed_rows) + len(pending_rows))
            )
            if pending_rows[LOG_ADD_SEQ_COLUMN].tolist() != expected:
                raise ValueError("runtime snapshot pending add sequence is invalid")
        if LOG_ROW_ID_COLUMN in pending_rows.columns:
            pending_ids = pending_rows[LOG_ROW_ID_COLUMN]
            if pending_ids.isna().any() or not pending_ids.is_unique:
                raise ValueError("runtime snapshot pending row IDs must be unique")
            if LOG_ROW_ID_COLUMN in committed_rows.columns:
                committed_ids = set(committed_rows[LOG_ROW_ID_COLUMN].dropna())
                if any(row_id in committed_ids for row_id in pending_ids):
                    raise ValueError(
                        "runtime snapshot pending row IDs overlap committed rows"
                    )
        return pending_rows.to_dict(orient="records")

    def _query_output_columns(self, query: QueryExpr) -> list[str]:
        """Infer columns through the selected execution engine."""

        if isinstance(self._engine, PolicyExecutor):
            return self._engine.query_output_columns(query)
        return self._engine._query_output_columns(query)
