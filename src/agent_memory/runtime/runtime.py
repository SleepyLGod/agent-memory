"""Stable memory runtime entry point and checkpoint engine selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_memory.api import MessageInput
from agent_memory.planner.differential_policy import DifferentiatedPolicy
from agent_memory.planner.legacy import differentiate_legacy_policy
from agent_memory.policy.logical import QueryExpr
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
    ) -> None:
        self.policy = policy
        self.storage = storage
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

        self._engine.add(message)

    def execute_retrieval_query(self, name: str, text: str) -> Any:
        """Execute one policy-owned retrieval query."""

        return self._engine.execute_retrieval_query(name, text)

    def snapshot_state(self) -> dict[str, Any]:
        """Return a checkpoint in the active engine's schema."""

        self._require_checkpoint_support()
        return self._engine.snapshot_state()

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore v2 directly or route schema-v1 state to the legacy engine."""

        self._require_checkpoint_support()
        schema_version = snapshot.get("schema_version")
        if schema_version == 1:
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
                self._engine = PolicyExecutor(self.policy, adapter=self._engine.adapter)
            self._engine.restore_state(snapshot)
            return
        raise ValueError("Unsupported runtime snapshot schema_version")

    def _require_checkpoint_support(self) -> None:
        """Reject half-implemented external-storage recovery semantics."""

        if self.storage is not None:
            raise NotImplementedError(
                "storage-bound checkpoint and restore require storage recovery support"
            )

    def _query_output_columns(self, query: QueryExpr) -> list[str]:
        """Infer columns through the selected execution engine."""

        if isinstance(self._engine, PolicyExecutor):
            return self._engine.query_output_columns(query)
        return self._engine._query_output_columns(query)
