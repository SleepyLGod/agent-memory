"""Physical executor for shared differentiated policies."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from agent_memory.api import Message, MessageInput
from agent_memory.planner.differential_policy import DifferentiatedPolicy
from agent_memory.planner.retrieval import RetrievalPlan
from agent_memory.policy.logical import MemoryView, QueryExpr, UserQuery
from agent_memory.policy.retrieval import RetrievalResult
from agent_memory.policy.relation import (
    LOG_ADDED_AT_COLUMN,
    LOG_ADD_SEQ_COLUMN,
    LOG_ROW_ID_COLUMN,
    LOG_SYSTEM_COLUMNS,
)
from agent_memory.policy.schema import output_columns
from agent_memory.runtime.window import WINDOW_SOURCE_INPUT, completed_count_windows
from agent_memory.storage.connector import (
    StorageCommit,
    StorageConflictError,
)
from agent_memory.storage.deployment import StorageDeployment
from agent_memory.storage.search import SearchBatch, SearchRequest


@dataclass(frozen=True)
class NodeOutputUpdate:
    """One node's complete next output and its exact row-level difference."""

    output_rows: pd.DataFrame
    inserted_rows: pd.DataFrame
    retracted_rows: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        """Return whether the node output did not change."""

        return self.inserted_rows.empty and self.retracted_rows.empty

    @classmethod
    def between(cls, old: pd.DataFrame, new: pd.DataFrame) -> "NodeOutputUpdate":
        """Compare two relation states with bag semantics."""

        inserted_rows, retracted_rows = _multiset_difference(old, new)
        return cls(
            output_rows=new,
            inserted_rows=inserted_rows,
            retracted_rows=retracted_rows,
        )


class PolicyExecutor:
    """Execute one differentiated policy step and commit all states atomically."""

    def __init__(
        self,
        policy: DifferentiatedPolicy,
        *,
        adapter: Any | None = None,
        storage: StorageDeployment | None = None,
    ) -> None:
        self.policy = policy
        self.spec = policy.spec
        self.adapter = adapter if adapter is not None else self._default_adapter()
        self.storage = storage
        self._validate_storage_plan()
        self._state: dict[str, pd.DataFrame] = {}
        self._node_state: dict[str, pd.DataFrame] = {}
        self._semantic_output_cache: dict[
            str,
            dict[Hashable, pd.DataFrame],
        ] = {}
        self._window_next_start: dict[str, int] = {}
        self._next_occurrence = 0
        self._storage_commit = (
            None
            if storage is None
            else StorageCommit(
                plan_fingerprint=policy.fingerprint,
                lineage_id=str(uuid4()),
                commit_sequence=0,
                source_row_count=0,
            )
        )
        if storage is not None:
            storage.connector.prepare(storage.statements)

    @property
    def node_state(self) -> Mapping[str, pd.DataFrame]:
        """Expose private node state for diagnostics without mixing it into views."""

        return self._node_state

    def add(self, message: MessageInput) -> None:
        """Append one source row and propagate its change through the shared DAG."""

        row = self._normalize_message(message)
        staged_next_occurrence = self._next_occurrence

        def allocate(node_id: str) -> str:
            nonlocal staged_next_occurrence
            occurrence = f"{node_id}:{staged_next_occurrence}"
            staged_next_occurrence += 1
            return occurrence

        changed_rows = pd.DataFrame([row])
        source_node_ids = tuple(
            node_id
            for node_id in self.policy.execution_order
            if self.policy.nodes[node_id].execution_kind == "source"
        )
        source_prefix = source_node_ids[0] if source_node_ids else "log"
        changed_rows.index = pd.Index([allocate(source_prefix)], dtype="object")
        current_log = (
            self._node_state.get(source_node_ids[0])
            if source_node_ids
            else self._state.get("log")
        )
        next_log = self._concat(current_log, changed_rows)

        staged_node_state = dict(self._node_state)
        staged_cache = {
            node_id: dict(lineage)
            for node_id, lineage in self._semantic_output_cache.items()
        }
        staged_window_start = dict(self._window_next_start)
        updates: dict[str, NodeOutputUpdate] = {}

        for node_id in self.policy.execution_order:
            node = self.policy.nodes[node_id]
            if node.execution_kind == "source":
                staged_node_state[node_id] = next_log
                updates[node_id] = NodeOutputUpdate(
                    output_rows=next_log,
                    inserted_rows=changed_rows,
                    retracted_rows=changed_rows.iloc[0:0].copy(),
                )
                continue

            parent_updates = tuple(updates[parent_id] for parent_id in node.input_node_ids)
            old_state = self._node_state.get(node_id, self._empty_node_frame(node_id))
            if all(update.is_empty for update in parent_updates):
                staged_node_state.setdefault(node_id, old_state)
                updates[node_id] = NodeOutputUpdate.between(old_state, old_state.copy())
                continue

            if node.execution_kind == "deterministic":
                next_state = self._execute_deterministic(node_id, staged_node_state)
            elif node.execution_kind == "semantic_row":
                next_state = self._execute_semantic_row(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                    staged_cache=staged_cache,
                )
            elif node.execution_kind == "semantic_state":
                next_state = self._execute_semantic_state(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                    staged_node_state=staged_node_state,
                )
            elif node.execution_kind == "process_window":
                next_state = self._execute_process_window(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                    parent_state=staged_node_state[node.input_node_ids[0]],
                    staged_window_start=staged_window_start,
                )
            elif node.execution_kind in {"over_window", "semantic_over_window"}:
                next_state = self._execute_over_window(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                    parent_state=staged_node_state[node.input_node_ids[0]],
                )
            else:
                raise RuntimeError(
                    f"Unknown policy node execution kind: {node.execution_kind!r}"
                )

            next_state = self._align_output(node_id, next_state)
            if node.execution_kind != "semantic_row":
                next_state = preserve_occurrence_index(
                    old_state,
                    next_state,
                    allocate=lambda node_id=node_id: allocate(node_id),
                )
            staged_node_state[node_id] = next_state
            updates[node_id] = NodeOutputUpdate.between(old_state, next_state)

        next_public_state: dict[str, pd.DataFrame] = {
            "log": next_log.reset_index(drop=True).copy()
        }
        for view_name, node_id in self.policy.view_outputs.items():
            private_view_state = staged_node_state.get(
                node_id,
                self._empty_view_frame(self.spec.views[view_name]),
            )
            next_public_state[view_name] = private_view_state.reset_index(drop=True).copy()

        next_storage_commit = self._next_storage_commit(
            source_row_count=len(next_log)
        )
        self._write_storage_updates(updates, next_commit=next_storage_commit)
        self._state = next_public_state
        self._node_state = staged_node_state
        self._semantic_output_cache = staged_cache
        self._window_next_start = staged_window_start
        self._next_occurrence = staged_next_occurrence
        self._storage_commit = next_storage_commit

    def snapshot_state(self) -> dict[str, Any]:
        """Return all state required to resume this exact compiled plan."""

        snapshot = {
            "schema_version": 2,
            "plan_fingerprint": self.policy.fingerprint,
            "state": dict(self._state),
            "node_state": dict(self._node_state),
            "semantic_output_cache": {
                node_id: dict(lineage)
                for node_id, lineage in self._semantic_output_cache.items()
            },
            "window_next_start": dict(self._window_next_start),
            "next_occurrence": self._next_occurrence,
        }
        if self.storage is not None:
            commit = self._require_storage_commit()
            if commit.is_initial:
                source_node_ids = tuple(
                    node_id
                    for node_id in self.policy.execution_order
                    if self.policy.nodes[node_id].execution_kind == "source"
                )
                if len(source_node_ids) != 1:
                    raise RuntimeError(
                        "storage checkpoints require exactly one source node"
                    )
                source_node_id = source_node_ids[0]
                empty_log = self._empty_node_frame(source_node_id)
                snapshot["state"] = {**snapshot["state"], "log": empty_log.copy()}
                complete_node_state = dict(snapshot["node_state"])
                complete_node_state.setdefault(source_node_id, empty_log.copy())
                for node_id in self.policy.sink_outputs.values():
                    complete_node_state.setdefault(
                        node_id,
                        self._empty_node_frame(node_id),
                    )
                snapshot["node_state"] = complete_node_state
            physical_commit = self.storage.connector.read_commit(
                namespace=self.storage.namespace
            )
            if physical_commit != self._expected_physical_commit():
                raise StorageConflictError(
                    "storage commit marker does not match the runtime checkpoint"
                )
            snapshot["storage_commit"] = commit.to_dict()
        return snapshot

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a schema-v2 checkpoint for the same compiled policy plan."""

        if snapshot.get("schema_version") != 2:
            raise ValueError("PolicyExecutor requires snapshot schema_version 2")
        if snapshot.get("plan_fingerprint") != self.policy.fingerprint:
            raise ValueError("Runtime snapshot plan fingerprint does not match this policy")

        state = _require_mapping(snapshot, "state")
        node_state = _require_mapping(snapshot, "node_state")
        semantic_output_cache = _require_mapping(snapshot, "semantic_output_cache")
        window_next_start = _require_mapping(snapshot, "window_next_start")
        next_occurrence = snapshot.get("next_occurrence")
        if not isinstance(next_occurrence, int) or next_occurrence < 0:
            raise ValueError("Runtime snapshot next_occurrence must be a non-negative int")

        staged_state = dict(state)
        staged_node_state = dict(node_state)
        staged_semantic_output_cache = {
            str(node_id): dict(_require_nested_mapping(cache, "semantic output cache"))
            for node_id, cache in semantic_output_cache.items()
        }
        staged_window_next_start = {
            str(node_id): int(value) for node_id, value in window_next_start.items()
        }

        staged_storage_commit: StorageCommit | None = None
        if self.storage is not None:
            raw_commit = snapshot.get("storage_commit")
            if not isinstance(raw_commit, Mapping):
                raise ValueError("storage-bound snapshot requires storage_commit")
            staged_storage_commit = StorageCommit.from_dict(raw_commit)
            if staged_storage_commit.plan_fingerprint != self.policy.fingerprint:
                raise ValueError(
                    "storage commit plan fingerprint does not match this policy"
                )
            log_rows = staged_state.get("log")
            if not isinstance(log_rows, pd.DataFrame):
                raise ValueError("storage-bound snapshot requires DataFrame log state")
            if staged_storage_commit.source_row_count != len(log_rows):
                raise ValueError(
                    "storage commit source row count does not match snapshot log"
                )
            physical_commit = self.storage.connector.read_commit(
                namespace=self.storage.namespace
            )
            if physical_commit != self._checkpoint_physical_commit(
                staged_storage_commit
            ):
                self.storage.connector.rebuild(
                    namespace=self.storage.namespace,
                    statements=self.storage.statements,
                    rows_by_statement=self._sink_rows(staged_node_state),
                    expected_commit=physical_commit,
                    next_commit=staged_storage_commit,
                )
        elif "storage_commit" in snapshot:
            raise ValueError(
                "storage-bound snapshot requires a matching storage deployment"
            )

        self._state = staged_state
        self._node_state = staged_node_state
        self._semantic_output_cache = staged_semantic_output_cache
        self._window_next_start = staged_window_next_start
        self._next_occurrence = next_occurrence
        self._storage_commit = staged_storage_commit

    def _validate_storage_plan(self) -> None:
        """Ensure the compiled sink mapping matches the runtime deployment."""

        compiled = tuple(self.policy.sink_outputs)
        deployed = (
            ()
            if self.storage is None
            else tuple(
                statement.statement_id
                for statement in self.storage.statements.statements
            )
        )
        if compiled != deployed:
            raise ValueError(
                "compiled storage sinks do not match the runtime deployment; "
                f"compiled={compiled}, deployed={deployed}"
            )

    def _write_storage_updates(
        self,
        updates: Mapping[str, NodeOutputUpdate],
        *,
        next_commit: StorageCommit | None,
    ) -> None:
        """Write changed sink rows before committing in-memory state."""

        if self.storage is None:
            return
        if next_commit is None:
            raise RuntimeError("storage update requires a next commit")
        writes = [
            (
                statement,
                updates[self.policy.sink_outputs[statement.statement_id]],
            )
            for statement in self.storage.statements.statements
            if not updates[
                self.policy.sink_outputs[statement.statement_id]
            ].is_empty
        ]
        with self.storage.connector.transaction(
            namespace=self.storage.namespace,
            expected_commit=self._expected_physical_commit(),
            next_commit=next_commit,
        ) as transaction:
            for statement, update in writes:
                transaction.write(
                    statement,
                    inserted_rows=update.inserted_rows.reset_index(drop=True).copy(),
                    retracted_rows=update.retracted_rows.reset_index(drop=True).copy(),
                )

    def _next_storage_commit(
        self,
        *,
        source_row_count: int,
    ) -> StorageCommit | None:
        """Return the marker committed with the staged runtime step."""

        if self.storage is None:
            return None
        current = self._require_storage_commit()
        return StorageCommit(
            plan_fingerprint=current.plan_fingerprint,
            lineage_id=current.lineage_id,
            commit_sequence=current.commit_sequence + 1,
            source_row_count=source_row_count,
        )

    def _require_storage_commit(self) -> StorageCommit:
        if self._storage_commit is None:
            raise RuntimeError("storage deployment is missing runtime commit state")
        return self._storage_commit

    def _expected_physical_commit(self) -> StorageCommit | None:
        commit = self._require_storage_commit()
        return self._checkpoint_physical_commit(commit)

    @staticmethod
    def _checkpoint_physical_commit(
        commit: StorageCommit,
    ) -> StorageCommit | None:
        if commit.is_initial:
            return None
        return commit

    def _sink_rows(
        self,
        node_state: Mapping[str, Any],
    ) -> dict[str, pd.DataFrame]:
        storage = self.storage
        if storage is None:
            raise RuntimeError("sink rows require a storage deployment")
        rows: dict[str, pd.DataFrame] = {}
        for statement in storage.statements.statements:
            node_id = self.policy.sink_outputs[statement.statement_id]
            value = node_state.get(node_id)
            if not isinstance(value, pd.DataFrame):
                raise ValueError(
                    f"snapshot is missing sink node state for {statement.statement_id!r}"
                )
            rows[statement.statement_id] = value.reset_index(drop=True).copy()
        return rows

    def replace_public_state(self, state: Mapping[str, Any]) -> None:
        """Replace public state for retrieval-only artifact inspection."""

        self._state = dict(state)
        node_state: dict[str, pd.DataFrame] = {}
        for node_id, node in self.policy.nodes.items():
            if node.execution_kind == "source" and "log" in state:
                node_state[node_id] = state["log"]
        for view_name, node_id in self.policy.view_outputs.items():
            if view_name in state:
                node_state[node_id] = state[view_name]
        self._node_state = node_state
        self._semantic_output_cache = {}
        self._window_next_start = {}

    def execute_retrieval_query(self, name: str, text: str) -> Any:
        """Bind user text and execute one policy-owned retrieval query."""

        if name not in self.policy.retrieval_queries:
            raise NotImplementedError(
                f"Memory policy does not declare a retrieval query named {name!r}."
            )
        retrieval = self.policy.retrieval_queries[name]
        if isinstance(retrieval, RetrievalPlan):
            return self._execute_retrieval_plan(retrieval, text)
        query = self._bind_user_query(retrieval, text)
        return self.adapter.execute(query, self._state)

    def _execute_retrieval_plan(
        self,
        plan: RetrievalPlan,
        text: str,
    ) -> RetrievalResult:
        """Execute one storage-backed retrieval DAG in topological order."""

        if self.storage is None:
            raise NotImplementedError(
                "RetrievalQuery requires a storage backend; in-memory scan fallback "
                "is not supported"
            )
        search = getattr(self.storage.connector, "search", None)
        if not callable(search):
            raise NotImplementedError(
                "Storage connector is not retrieval-capable"
            )

        statements = {
            statement.statement_id: statement
            for statement in self.storage.statements.statements
        }
        outputs: dict[str, pd.DataFrame] = {}
        metrics: dict[str, Mapping[str, Any]] = {}
        for node_id in plan.execution_order:
            node = plan.nodes[node_id]
            if node.execution_kind == "search":
                if node.statement_id is None or node.statement_id not in statements:
                    raise RuntimeError(
                        "Retrieval search node is not bound to the active storage plan"
                    )
                origin_record_ids: list[str] = []
                for input_node_id in node.input_node_ids:
                    frame = outputs[input_node_id]
                    if "record_id" not in frame.columns:
                        raise ValueError(
                            "BFS origin relation must include a record_id column"
                        )
                    for value in frame["record_id"].dropna().tolist():
                        record_id = str(value)
                        if record_id not in origin_record_ids:
                            origin_record_ids.append(record_id)
                statement = statements[node.statement_id]
                batch = search(
                    SearchRequest(
                        statement_id=node.statement_id,
                        target=statement.target,
                        namespace=self.storage.namespace,
                        query=text,
                        methods=tuple(node.query.params["methods"]),
                        reranker=node.query.params["reranker"],
                        limit=int(node.query.params["limit"]),
                        output_columns=node.required_columns,
                        origin_record_ids=tuple(origin_record_ids),
                    )
                )
                if not isinstance(batch, SearchBatch):
                    raise TypeError("storage search must return a SearchBatch")
                missing = set(node.required_columns).difference(batch.rows.columns)
                if missing:
                    raise ValueError(
                        "storage search result is missing logical columns: "
                        f"{sorted(missing)}"
                    )
                outputs[node_id] = batch.rows.loc[:, node.required_columns].copy()
                if batch.metrics:
                    metrics[node_id] = batch.metrics
                continue
            if node.execution_kind != "relational":
                raise RuntimeError(
                    f"Unknown retrieval execution kind: {node.execution_kind!r}"
                )
            inputs = {
                input_node_id: outputs[input_node_id]
                for input_node_id in node.input_node_ids
            }
            outputs[node_id] = self.adapter.execute(node.query, inputs)
            if len(node.input_node_ids) == 1 and node.input_node_ids[0] in metrics:
                metrics[node_id] = metrics[node.input_node_ids[0]]

        return RetrievalResult(
            query=text,
            channels={
                name: outputs[node_id]
                for name, node_id in plan.channel_outputs.items()
            },
            metrics={
                name: metrics[node_id]
                for name, node_id in plan.channel_outputs.items()
                if node_id in metrics
            },
        )

    def query_output_columns(self, query: QueryExpr) -> list[str]:
        """Infer output columns, resolving public materialized-view leaves."""

        if query.op == "materialized_view" and "columns" not in query.params:
            name = str(query.params["name"])
            if name not in self.spec.views:
                raise KeyError(f"Unknown materialized view {name!r}")
            return self.query_output_columns(self.spec.views[name].query)
        return list(output_columns(self._bind_materialized_columns(query)))

    def _execute_deterministic(
        self,
        node_id: str,
        staged_node_state: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        node = self.policy.nodes[node_id]
        inputs = {
            parent_id: staged_node_state[parent_id]
            for parent_id in node.input_node_ids
        }
        return self.adapter.execute(node.query, inputs)

    def _execute_semantic_row(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_update: NodeOutputUpdate,
        staged_cache: dict[str, dict[Hashable, pd.DataFrame]],
    ) -> pd.DataFrame:
        node = self.policy.nodes[node_id]
        if len(node.input_node_ids) != 1:
            raise NotImplementedError("Row-local semantic nodes require one input")

        cache = staged_cache.setdefault(node_id, {})
        for occurrence in set(parent_update.retracted_rows.index):
            cache.pop(occurrence, None)

        inserted = parent_update.inserted_rows
        if not inserted.empty:
            parent_id = node.input_node_ids[0]
            outputs = self.adapter.execute(node.query, {parent_id: inserted})
            outputs = self._align_output(node_id, outputs)
            unknown_indexes = set(outputs.index).difference(inserted.index)
            if unknown_indexes:
                raise RuntimeError(
                    f"{node.query.op} did not preserve semantic input indexes: "
                    f"{sorted(unknown_indexes, key=repr)}"
                )
            empty_output = outputs.iloc[0:0].copy()
            for occurrence in dict.fromkeys(inserted.index):
                mask = outputs.index == occurrence
                cache[occurrence] = (
                    outputs.loc[mask].copy() if mask.any() else empty_output.copy()
                )

        frames = [frame for frame in cache.values() if not frame.empty]
        if not frames:
            return old_state.iloc[0:0].copy()
        return pd.concat(frames, axis=0)

    def _execute_semantic_state(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_update: NodeOutputUpdate,
        staged_node_state: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        node = self.policy.nodes[node_id]
        parent_id = node.input_node_ids[0]
        if not parent_update.retracted_rows.empty:
            return self.adapter.execute(
                node.query,
                {parent_id: staged_node_state[parent_id]},
            )
        if node.maintenance_query is None:
            raise RuntimeError(f"Semantic state node {node_id} has no maintenance query")
        inputs = dict(staged_node_state)
        inputs[node_id] = old_state
        inputs[f"{parent_id}__inserted"] = parent_update.inserted_rows
        return self.adapter.execute(node.maintenance_query, inputs)

    def _execute_process_window(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_update: NodeOutputUpdate,
        parent_state: pd.DataFrame,
        staged_window_start: dict[str, int],
    ) -> pd.DataFrame:
        if not parent_update.retracted_rows.empty:
            raise NotImplementedError("process_window requires append-only input changes")
        node = self.policy.nodes[node_id]
        window_query, process_query = node.query.inputs
        windows, next_start = completed_count_windows(
            parent_state,
            window_query.params,
            next_start=staged_window_start.get(node_id, 0),
        )
        staged_window_start[node_id] = next_start
        if not windows:
            return old_state.copy()
        results = [
            self.adapter.execute(
                process_query,
                {**self._state, WINDOW_SOURCE_INPUT: window.frame},
            )
            for window in windows
        ]
        return self._concat(old_state, pd.concat(results, ignore_index=True))

    def _execute_over_window(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_update: NodeOutputUpdate,
        parent_state: pd.DataFrame,
    ) -> pd.DataFrame:
        if not parent_update.retracted_rows.empty:
            raise NotImplementedError("over window maintenance requires append-only input changes")
        if parent_update.inserted_rows.empty:
            return old_state.copy()

        node = self.policy.nodes[node_id]
        parent_id = node.input_node_ids[0]
        changed_name = f"{parent_id}__inserted"
        over_query = node.query.inputs[0]
        changed_over = QueryExpr(
            op="over",
            inputs=(
                QueryExpr(
                    op="materialized_view",
                    params={
                        "name": changed_name,
                        "columns": self.policy.nodes[parent_id].output_columns,
                    },
                ),
            ),
            params={
                **over_query.params,
                "frame_source": QueryExpr(
                    op="materialized_view",
                    params={
                        "name": parent_id,
                        "columns": self.policy.nodes[parent_id].output_columns,
                    },
                ),
            },
        )
        changed_query = QueryExpr(
            op=node.query.op,
            inputs=(changed_over,),
            params=node.query.params,
        )
        changed_output = self.adapter.execute(
            changed_query,
            {parent_id: parent_state, changed_name: parent_update.inserted_rows},
        )
        return self._concat(old_state, changed_output)

    def _align_output(self, node_id: str, frame: pd.DataFrame) -> pd.DataFrame:
        expected = list(self.policy.nodes[node_id].output_columns)
        actual = list(frame.columns)
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        if missing or extra:
            raise ValueError(
                f"Policy node {node_id} output schema mismatch; "
                f"missing={missing}, extra={extra}"
            )
        return frame.loc[:, expected].copy()

    def _empty_node_frame(self, node_id: str) -> pd.DataFrame:
        return pd.DataFrame(columns=self.policy.nodes[node_id].output_columns)

    def _empty_view_frame(self, view: MemoryView) -> pd.DataFrame:
        return pd.DataFrame(columns=self.query_output_columns(view.query))

    def _normalize_message(self, message: MessageInput) -> dict[str, Any]:
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
        declared_columns = tuple(
            column.name for column in self.spec.log.expr.params.get("columns", ())
        )
        return {column: row.get(column) for column in declared_columns}

    def _bind_user_query(self, query: QueryExpr, text: str) -> QueryExpr:
        return QueryExpr(
            op=query.op,
            inputs=tuple(self._bind_user_query(item, text) for item in query.inputs),
            params={
                key: self._bind_user_query_value(value, text)
                for key, value in query.params.items()
            },
        )

    def _bind_user_query_value(self, value: Any, text: str) -> Any:
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

    def _bind_materialized_columns(self, query: QueryExpr) -> QueryExpr:
        if query.op == "materialized_view" and "columns" not in query.params:
            name = str(query.params["name"])
            if name in self.spec.views:
                return QueryExpr(
                    op="materialized_view",
                    params={"name": name, "columns": self.query_output_columns(self.spec.views[name].query)},
                )
            return query
        return QueryExpr(
            op=query.op,
            inputs=tuple(self._bind_materialized_columns(item) for item in query.inputs),
            params=query.params,
        )

    def _concat(
        self,
        current: pd.DataFrame | None,
        changed: pd.DataFrame,
    ) -> pd.DataFrame:
        if current is None:
            return changed.copy()
        if changed.empty:
            return current.copy()
        if current.empty:
            return changed.copy()
        return pd.concat([current, changed], axis=0)

    def _default_adapter(self) -> Any:
        from agent_memory.adapters import LotusAdapter

        return LotusAdapter()


def _multiset_difference(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inserted and retracted rows between two bag-valued relations."""

    if list(old.columns) != list(new.columns):
        raise ValueError("Relation states must have identical ordered columns")
    old_keys = [_row_key(row) for row in old.itertuples(index=False, name=None)]
    new_keys = [_row_key(row) for row in new.itertuples(index=False, name=None)]
    old_remaining = Counter(old_keys)
    new_remaining = Counter(new_keys)

    inserted_positions: list[int] = []
    for position, key in enumerate(new_keys):
        if old_remaining[key] > 0:
            old_remaining[key] -= 1
        else:
            inserted_positions.append(position)

    retracted_positions: list[int] = []
    for position, key in enumerate(old_keys):
        if new_remaining[key] > 0:
            new_remaining[key] -= 1
        else:
            retracted_positions.append(position)
    return (
        new.iloc[inserted_positions].copy(),
        old.iloc[retracted_positions].copy(),
    )


def preserve_occurrence_index(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    allocate: Callable[[], Hashable],
) -> pd.DataFrame:
    """Reuse indexes for unchanged occurrences and allocate indexes for new rows."""

    if list(old.columns) != list(new.columns):
        raise ValueError("Relation states must have identical ordered columns")
    available: dict[tuple[Any, ...], deque[Hashable]] = defaultdict(deque)
    for index, row in zip(old.index, old.itertuples(index=False, name=None), strict=True):
        available[_row_key(row)].append(index)

    indexes: list[Hashable] = []
    for row in new.itertuples(index=False, name=None):
        key = _row_key(row)
        indexes.append(available[key].popleft() if available[key] else allocate())
    result = new.copy()
    result.index = pd.Index(indexes, dtype="object")
    return result


def _row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return a hashable, type-preserving relation row value."""

    return tuple(_value_key(value) for value in row)


def _value_key(value: Any) -> Any:
    """Normalize nested pandas values for exact bag comparison."""

    if value is None or value is pd.NA:
        return ("null",)
    if isinstance(value, Mapping):
        items = ((_value_key(key), _value_key(item)) for key, item in value.items())
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, list):
        return ("list", tuple(_value_key(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (
            "set",
            tuple(sorted((_value_key(item) for item in value), key=repr)),
        )
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return (type(value).__qualname__, value.isoformat())
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _value_key(scalar)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ("null",)
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(
            f"Unsupported relation cell value for node comparison: {type(value).__name__}"
        ) from error
    return (type(value).__module__, type(value).__qualname__, value)


def _require_mapping(snapshot: Mapping[str, Any], name: str) -> Mapping[Any, Any]:
    value = snapshot.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Runtime snapshot {name} must be a mapping")
    return value


def _require_nested_mapping(value: Any, name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Runtime snapshot {name} must be a mapping")
    return value
