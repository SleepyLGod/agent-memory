"""Physical executor for shared differentiated policies."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from heapq import merge
from typing import Any
from uuid import uuid4

import pandas as pd

from agent_memory.api import Message, MessageInput
from agent_memory.planner.differential_policy import DifferentiatedPolicy
from agent_memory.planner.retrieval import RetrievalPlan
from agent_memory.planner.physical import PREDICATE_DECISIONS_INPUT
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
from agent_memory.runtime.row_outputs import RowOutputCache
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


@dataclass
class _CandidateBucketIndex:
    """Derived membership for joins that retain survivors and append new IDs."""

    columns: tuple[str, ...]
    buckets: dict[tuple[Any, ...], dict[Hashable, int]]
    next_order: int
    row_count: int

    @staticmethod
    def _keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> Iterator[tuple[Any, ...]]:
        # Input is read-only during iteration. Retain the values as well as their
        # IDs to prevent Python ID reuse; bound this per-call memo to 64 entries.
        recent: dict[tuple[int, ...], tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
            identity = tuple(id(value) for value in row)
            entry = recent.get(identity)
            if entry is None:
                key = _row_key(row)
                if len(recent) == 64:
                    recent.clear()
                recent[identity] = (row, key)
            else:
                key = entry[1]
            yield key

    @classmethod
    def build(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> "_CandidateBucketIndex":
        buckets: dict[tuple[Any, ...], dict[Hashable, int]] = {}
        for order, (occurrence, key) in enumerate(zip(frame.index, cls._keys(frame, columns), strict=True)):
            buckets.setdefault(key, {})[occurrence] = order
        return cls(columns, buckets, len(frame), len(frame))

    def updated(self, removed: pd.DataFrame, added: pd.DataFrame) -> "_CandidateBucketIndex":
        if len(removed) == self.row_count:
            return type(self).build(added, self.columns)
        buckets = dict(self.buckets)
        touched: set[tuple[Any, ...]] = set()
        next_order = self.next_order
        for frame, inserting in ((removed, False), (added, True)):
            previous_key = None
            members: dict[Hashable, int] = {}
            for occurrence, key in zip(frame.index, self._keys(frame, self.columns), strict=True):
                # Consecutive members of one bucket need only one lookup of the
                # potentially nested key. The per-call key memo retains it.
                if key is not previous_key:
                    if key not in touched:
                        buckets[key] = dict(buckets.get(key, {}))
                        touched.add(key)
                    members = buckets[key]
                    previous_key = key
                if inserting:
                    members[occurrence] = next_order
                    next_order += 1
                else:
                    del members[occurrence]
        for key in touched:
            if not buckets[key]:
                del buckets[key]
        return type(self)(self.columns, buckets, next_order, self.row_count - len(removed) + len(added))

    def select(self, frame: pd.DataFrame, keys: set[tuple[Any, ...]]) -> pd.DataFrame:
        if len(keys) >= len(self.buckets) and self.buckets.keys() <= keys:
            return frame.iloc[range(len(frame))]
        # Each bucket is already ordered. Merge only the affected memberships.
        streams = [self.buckets[key].items() for key in keys if key in self.buckets]
        occurrences = [occurrence for occurrence, _ in merge(*streams, key=lambda item: item[1])]
        return frame.loc[occurrences] if occurrences else frame.iloc[:0]


_BucketUpdates = dict[
    tuple[str, tuple[str, ...]], tuple[_CandidateBucketIndex, _CandidateBucketIndex]
]


class PolicyExecutor:
    """Execute one differentiated policy step and commit all states atomically."""

    def __init__(
        self,
        policy: DifferentiatedPolicy,
        *,
        adapter: Any | None = None,
        storage: StorageDeployment | None = None,
    ) -> None:
        self.adapter = adapter if adapter is not None else self._default_adapter()
        prepare = getattr(self.adapter, "prepare_policy", None)
        self.policy = policy if prepare is None else prepare(policy)
        policy = self.policy
        self.spec = policy.spec
        self._independent_node_id = getattr(
            self.adapter, "independent_node_id", lambda policy: None,
        )(policy)
        if self._independent_node_id is not None:
            selected = policy.nodes[self._independent_node_id]
            if selected.execution_kind != "semantic_row" or selected.query.op != "sem_flat_map":
                raise ValueError("independent execution requires a row-local sem_flat_map")
        self.storage = storage
        self._validate_storage_plan()
        self._state: dict[str, pd.DataFrame] = {}
        self._node_state: dict[str, pd.DataFrame] = {}
        self._semantic_output_cache: dict[str, RowOutputCache] = {}
        self._window_next_start: dict[str, int] = {}
        self._predicate_decisions: dict[str, dict[str, bool]] = {}
        # Derived from committed rows; rebuild lazily after restore, not in snapshots.
        self._join_row_indexes: dict[str, dict[tuple[Any, ...], tuple[Hashable, ...]]] = {}
        self._candidate_bucket_indexes: dict[tuple[str, tuple[str, ...]], _CandidateBucketIndex] = {}
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

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Return the declared source relation columns in stable order."""

        return tuple(
            column.name for column in self.spec.log.expr.params.get("columns", ())
        )

    @property
    def source_row_count(self) -> int:
        """Return the number of source rows in the committed state."""

        return len(self._state.get("log", ()))

    def read_view(self, name: str) -> pd.DataFrame:
        """Return a defensive copy of one declared public view."""

        if name not in self.spec.views:
            raise KeyError(f"unknown public view: {name!r}")
        value = self._state.get(name)
        if value is None:
            return self._empty_view_frame(self.spec.views[name])
        return _copy_frame_with_objects(value)

    def add(self, message: MessageInput) -> None:
        """Append one source row and propagate its change through the shared DAG."""

        row = self.normalize_source_row(message)
        self.apply_delta(pd.DataFrame([row], columns=pd.Index(self.source_columns)))

    def apply_delta(self, changed_rows: pd.DataFrame) -> None:
        """Propagate one relation-valued source delta and publish it atomically."""

        if self._independent_node_id is None:
            self._apply_delta(changed_rows)
        else:
            # Always drain the worker before returning, including on main-branch failure.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-extraction") as pool:
                self._apply_delta(changed_rows, pool=pool)

    def _apply_delta(
        self, changed_rows: pd.DataFrame, *, pool: ThreadPoolExecutor | None = None,
    ) -> None:
        """Keep state mutation and publication on the calling thread."""

        if not isinstance(changed_rows, pd.DataFrame):
            raise TypeError("changed_rows must be a pandas DataFrame")
        if changed_rows.empty:
            return
        expected_columns = list(self.source_columns)
        if list(changed_rows.columns) != expected_columns:
            raise ValueError(
                "source delta columns must exactly match the declared log schema"
            )
        changed_rows = _copy_frame_with_objects(changed_rows)
        staged_next_occurrence = self._next_occurrence

        def allocate(node_id: str) -> str:
            nonlocal staged_next_occurrence
            occurrence = f"{node_id}:{staged_next_occurrence}"
            staged_next_occurrence += 1
            return occurrence

        source_node_ids = tuple(
            node_id
            for node_id in self.policy.execution_order
            if self.policy.nodes[node_id].execution_kind == "source"
        )
        source_prefix = source_node_ids[0] if source_node_ids else "log"
        changed_rows.index = pd.Index(
            [allocate(source_prefix) for _ in range(len(changed_rows))],
            dtype="object",
        )
        current_log = (
            self._node_state.get(source_node_ids[0])
            if source_node_ids
            else self._state.get("log")
        )
        next_log = self._concat(current_log, changed_rows)

        staged_node_state = dict(self._node_state)
        staged_cache = {
            node_id: lineage.copy()
            for node_id, lineage in self._semantic_output_cache.items()
        }
        staged_window_start = dict(self._window_next_start)
        staged_predicates = {node: dict(decisions) for node, decisions in self._predicate_decisions.items()}
        staged_join_indexes = dict(self._join_row_indexes)
        staged_bucket_updates: _BucketUpdates = {}
        updates: dict[str, NodeOutputUpdate] = {}

        remaining = list(self.policy.execution_order)
        pending: Future[Any] | None = None
        while remaining:
            if pending is not None and pending.done():
                pending.result()  # Surface extraction failure before starting more work.
            node_id = next((candidate for candidate in remaining
                if not (pending is not None and candidate == self._independent_node_id)
                and all(parent in updates for parent in self.policy.nodes[candidate].input_node_ids)), None)
            if node_id is None:
                if pending is None or self._independent_node_id not in remaining:
                    raise RuntimeError("maintenance plan has unresolved dependencies")
                node_id = self._independent_node_id
            node = self.policy.nodes[node_id]
            if node_id == self._independent_node_id and pending is None and pool is not None:
                inserted = updates[node.input_node_ids[0]].inserted_rows
                if not inserted.empty:
                    pending = pool.submit(
                        copy_context().run, self.adapter.execute_independent, node.query,
                        {node.input_node_ids[0]: _copy_frame_with_objects(inserted)},
                    )
                    continue
            remaining.remove(node_id)
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
            needs_global_aggregate_identity = (
                node.execution_kind == "algebraic_state"
                and old_state.empty
                and not tuple(node.query.params["group_keys"])
            )
            if (
                all(update.is_empty for update in parent_updates)
                and not needs_global_aggregate_identity
            ):
                staged_node_state.setdefault(node_id, old_state)
                updates[node_id] = NodeOutputUpdate(
                    old_state, old_state.iloc[:0], old_state.iloc[:0],
                )
                continue

            if node.execution_kind == "relational_state":
                row_index = self._join_row_indexes.get(node_id)
                row_index = _index_join_rows(old_state) if row_index is None else dict(row_index)
                update = self._execute_relational_join_update(
                    node_id, old_state, parent_updates,
                    allocate=lambda node_id=node_id: allocate(node_id),
                    row_index=row_index,
                )
                staged_join_indexes[node_id] = row_index
                staged_node_state[node_id] = update.output_rows
                updates[node_id] = update
                continue

            if node.execution_kind == "deterministic":
                next_state = self._execute_deterministic(node_id, staged_node_state)
            elif node.execution_kind == "algebraic_state":
                next_state = self._execute_algebraic_state(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                )
            elif node.execution_kind == "semantic_binary_state":
                next_state = self._execute_binary_state(
                    node_id,
                    old_state=old_state,
                    parent_updates=parent_updates,
                    staged_node_state=staged_node_state,
                )
            elif node.execution_kind in {"semantic_row", "semantic_predicate"}:
                completed = pending.result() if pending is not None and node_id == self._independent_node_id else None
                next_state = self._execute_semantic_row(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                    staged_cache=staged_cache,
                    staged_bucket_updates=staged_bucket_updates,
                    predicate_decisions=(staged_predicates.setdefault(node_id, {})
                                         if node.execution_kind == "semantic_predicate" else None),
                    execute=(lambda query, inputs: completed)
                    if pending is not None and node_id == self._independent_node_id else None,
                )
            elif node.execution_kind == "semantic_state":
                next_state = self._execute_semantic_state(
                    node_id,
                    old_state=old_state,
                    parent_update=parent_updates[0],
                    staged_node_state=staged_node_state,
                    identity_decisions=(staged_predicates.setdefault(node_id, {})
                                        if node.maintenance_query is not None
                                        and node.maintenance_query.params.get("identity_reuse") else None),
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
            if node.execution_kind not in {"semantic_row", "semantic_predicate"}:
                update = _preserve_occurrences_and_diff(
                    old_state,
                    next_state,
                    allocate=lambda node_id=node_id: allocate(node_id),
                )
            else:
                update = NodeOutputUpdate.between(old_state, next_state)
            staged_node_state[node_id] = update.output_rows
            updates[node_id] = update

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
        self._predicate_decisions = staged_predicates
        self._join_row_indexes = staged_join_indexes
        self._candidate_bucket_indexes.update({key: pair[1] for key, pair in staged_bucket_updates.items()})
        self._window_next_start = staged_window_start
        self._next_occurrence = staged_next_occurrence
        self._storage_commit = next_storage_commit

    def snapshot_state(self) -> dict[str, Any]:
        """Return all state required to resume this exact compiled plan."""

        adapter_execution_fingerprint = _adapter_execution_fingerprint(self.adapter)
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
        if adapter_execution_fingerprint:
            snapshot["adapter_execution_fingerprint"] = (
                adapter_execution_fingerprint
            )
        if any(n.execution_kind == "semantic_predicate" or n.maintenance_query is not None
               and n.maintenance_query.params.get("identity_reuse") for n in self.policy.nodes.values()):
            snapshot["predicate_decisions"] = {node: dict(values) for node, values in self._predicate_decisions.items()}
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
        snapshot_adapter_fingerprint = snapshot.get(
            "adapter_execution_fingerprint", ""
        )
        if not isinstance(snapshot_adapter_fingerprint, str):
            raise ValueError(
                "Runtime snapshot adapter execution fingerprint must be a string"
            )
        if snapshot_adapter_fingerprint != _adapter_execution_fingerprint(
            self.adapter
        ):
            raise ValueError(
                "Runtime snapshot adapter execution fingerprint does not match"
            )

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
            str(node_id): RowOutputCache(_require_nested_mapping(cache, "semantic output cache"))
            for node_id, cache in semantic_output_cache.items()
        }
        staged_window_next_start = {
            str(node_id): int(value) for node_id, value in window_next_start.items()
        }
        reuse_nodes = {key for key, node in self.policy.nodes.items()
                       if node.execution_kind == "semantic_predicate" or node.maintenance_query is not None
                       and node.maintenance_query.params.get("identity_reuse")}
        raw_predicates = _require_mapping(snapshot, "predicate_decisions") if reuse_nodes else {}
        staged_predicates: dict[str, dict[str, bool]] = {}
        for node_id, values in raw_predicates.items():
            if node_id not in reuse_nodes or not isinstance(values, Mapping) or any(
                not isinstance(key, str) or type(value) is not bool for key, value in values.items()
            ):
                raise ValueError("invalid predicate decision state")
            staged_predicates[str(node_id)] = dict(values)

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
        self._predicate_decisions = staged_predicates
        self._join_row_indexes = {}
        self._candidate_bucket_indexes = {}
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

    def _execute_binary_state(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_updates: tuple[NodeOutputUpdate, ...],
        staged_node_state: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Incrementally maintain a binary inner join from old and inserted rows."""

        node = self.policy.nodes[node_id]
        if len(node.input_node_ids) != 2 or len(parent_updates) != 2:
            raise RuntimeError("Binary state nodes require exactly two inputs")
        if any(not update.retracted_rows.empty for update in parent_updates):
            raise NotImplementedError(
                "Semantic binary state maintenance requires append-only input changes"
            )
        if node.maintenance_query is None:
            raise RuntimeError(f"Binary state node {node_id} has no maintenance query")

        inputs: dict[str, pd.DataFrame] = {node_id: old_state}
        for parent_id, update in zip(
            node.input_node_ids,
            parent_updates,
            strict=True,
        ):
            inputs[parent_id] = self._node_state.get(
                parent_id,
                self._empty_node_frame(parent_id),
            )
            inputs[f"{parent_id}__inserted"] = update.inserted_rows
        return self.adapter.execute(node.maintenance_query, inputs)

    def _execute_relational_join_update(
        self, node_id: str, old_state: pd.DataFrame,
        updates: tuple[NodeOutputUpdate, ...],
        *, allocate: Callable[[], Hashable],
        row_index: dict[tuple[Any, ...], tuple[Hashable, ...]] | None = None,
    ) -> NodeOutputUpdate:
        """Propagate exact inner-join changes without re-diffing its full output."""
        node = self.policy.nodes[node_id]
        left, right = updates
        old_left, old_right = (
            self._node_state.get(p, self._empty_node_frame(p))
            for p in node.input_node_ids
        )
        surviving_left = (
            _multiset_difference(left.retracted_rows, old_left)[0]
            if not left.retracted_rows.empty else old_left
        )

        def bind(template: QueryExpr, name: str, frame: pd.DataFrame) -> QueryExpr:
            if template.op == "alias":
                return QueryExpr(op="alias", params=template.params,
                                 inputs=(bind(template.inputs[0], name, frame),))
            return QueryExpr(op="materialized_view", params={
                "name": name, "columns": tuple(frame.columns),
            })

        def join(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            if a.empty or b.empty:
                return old_state.iloc[:0].copy()
            query = QueryExpr(op="join", params=node.query.params, inputs=(
                bind(node.query.inputs[0], "__join_left", a),
                bind(node.query.inputs[1], "__join_right", b),
            ))
            return self._align_output(
                node_id,
                self.adapter.execute(query, {"__join_left": a, "__join_right": b}),
            )

        # Remove each old match once, then add each new match once. Using the
        # surviving left side avoids double-counting simultaneous side changes.
        def concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
            nonempty = [part for part in parts if not part.empty]
            return pd.concat(nonempty, ignore_index=True) if nonempty else old_state.iloc[:0]

        removed = concat([
            join(left.retracted_rows, old_right),
            join(surviving_left, right.retracted_rows),
        ])
        if left.retracted_rows.empty and right.retracted_rows.empty:
            # Preserve the compiled append plan's task order as well as its bag.
            added = concat([
                join(left.inserted_rows, old_right),
                join(old_left, right.inserted_rows),
                join(left.inserted_rows, right.inserted_rows),
            ])
        else:
            added = concat([
                join(left.inserted_rows, right.output_rows),
                join(surviving_left, right.inserted_rows),
            ])
        return _apply_join_delta(old_state, removed, added, allocate=allocate, row_index=row_index)

    def _execute_algebraic_state(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_update: NodeOutputUpdate,
    ) -> pd.DataFrame:
        """Apply exact parent insertions and retractions to aggregate state."""

        node = self.policy.nodes[node_id]
        if len(node.input_node_ids) != 1:
            raise RuntimeError("Algebraic aggregate state requires exactly one input")
        if node.maintenance_query is None:
            raise RuntimeError(
                f"Algebraic aggregate state node {node_id} has no maintenance query"
            )
        parent_id = node.input_node_ids[0]
        return self.adapter.execute(
            node.maintenance_query,
            {
                node_id: old_state,
                f"{parent_id}__inserted": parent_update.inserted_rows,
                f"{parent_id}__retracted": parent_update.retracted_rows,
            },
        )

    def _execute_semantic_row(
        self,
        node_id: str,
        *,
        old_state: pd.DataFrame,
        parent_update: NodeOutputUpdate,
        staged_cache: dict[str, RowOutputCache],
        staged_bucket_updates: _BucketUpdates | None = None,
        predicate_decisions: dict[str, bool] | None = None,
        execute: Callable[[QueryExpr, Mapping[str, Any]], Any] | None = None,
    ) -> pd.DataFrame:
        node = self.policy.nodes[node_id]
        if len(node.input_node_ids) != 1:
            raise NotImplementedError("Row-local semantic nodes require one input")

        cache = staged_cache.get(node_id)
        if not isinstance(cache, RowOutputCache):
            cache = RowOutputCache(cache)
            staged_cache[node_id] = cache
        scope = getattr(self.adapter, "bounded_predicate_scope", lambda query: None)(node.query)
        retracted = parent_update.retracted_rows
        inserted = parent_update.inserted_rows
        if scope is not None:
            dependencies, bucket_columns = scope
            parent_id = node.input_node_ids[0]
            previous = self._node_state.get(parent_id)
            if previous is None:
                previous = parent_update.output_rows.iloc[:0]
            bucket_pair = None
            if (bucket_columns and staged_bucket_updates is not None
                    and self.policy.nodes[parent_id].execution_kind == "relational_state"):
                key = (parent_id, tuple(bucket_columns))
                bucket_pair = staged_bucket_updates.get(key)
                if bucket_pair is None:
                    before = self._candidate_bucket_indexes.get(key)
                    if before is None:
                        before = _CandidateBucketIndex.build(previous, key[1])
                    bucket_pair = (before, before.updated(retracted, inserted))
                    staged_bucket_updates[key] = bucket_pair
            if _carry_predicate_replacements(cache, retracted, inserted, dependencies):
                frames = cache.nonempty_frames()
                return pd.concat(frames) if frames else old_state.iloc[:0].copy()
            if bucket_columns:
                retracted, inserted = _carry_unchanged_predicate_buckets(
                    cache, retracted, inserted, dependencies, bucket_columns,
                )
            if bucket_pair is None:
                retracted, inserted = _affected_candidate_buckets(
                    previous, parent_update.output_rows, retracted, inserted, bucket_columns,
                )
            else:
                keys = {_row_key(row) for frame in (retracted, inserted)
                        for row in frame.loc[:, list(bucket_columns)].itertuples(index=False, name=None)}
                retracted = bucket_pair[0].select(previous, keys)
                inserted = bucket_pair[1].select(parent_update.output_rows, keys)
        for occurrence in set(retracted.index):
            cache.pop(occurrence, None)

        if not inserted.empty:
            parent_id = node.input_node_ids[0]
            inputs: dict[str, Any] = {parent_id: inserted}
            if predicate_decisions is not None:
                inputs[PREDICATE_DECISIONS_INPUT] = predicate_decisions
            outputs = (execute or self.adapter.execute)(node.query, inputs)
            outputs = self._align_output(node_id, outputs)
            unknown_indexes = set(outputs.index).difference(inserted.index)
            if unknown_indexes:
                raise RuntimeError(
                    f"{node.query.op} did not preserve semantic input indexes: "
                    f"{sorted(unknown_indexes, key=repr)}"
                )
            empty_output = cache.empty_like(outputs)
            for occurrence in dict.fromkeys(inserted.index):
                mask = outputs.index == occurrence
                cache[occurrence] = (
                    outputs.loc[mask].copy() if mask.any() else empty_output
                )

        frames = cache.nonempty_frames()
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
        identity_decisions: dict[str, bool] | None = None,
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
        inputs: dict[str, Any] = dict(staged_node_state)
        inputs[node_id] = old_state
        inputs[f"{parent_id}__inserted"] = parent_update.inserted_rows
        if identity_decisions is not None:
            inputs["__physical_identity_decisions"] = identity_decisions
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
        return pd.DataFrame(columns=pd.Index(self.policy.nodes[node_id].output_columns))

    def _empty_view_frame(self, view: MemoryView) -> pd.DataFrame:
        return pd.DataFrame(columns=pd.Index(self.query_output_columns(view.query)))

    def normalize_source_row(
        self,
        message: MessageInput,
        *,
        add_seq: int | None = None,
    ) -> dict[str, Any]:
        """Normalize one accepted message into the declared source schema."""

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
                    LOG_ADD_SEQ_COLUMN: (
                        self.source_row_count if add_seq is None else add_seq
                    ),
                }
            )
        return {column: row.get(column) for column in self.source_columns}

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


def _carry_predicate_replacements(
    cache: RowOutputCache, removed: pd.DataFrame,
    added: pd.DataFrame, dependencies: tuple[str, ...],
) -> bool:
    """Carry membership, not old metadata, when semantic input bags are unchanged."""
    if len(removed) != len(added) or any(i not in cache for i in removed.index):
        return False
    old_keys = [_row_key(row) for row in removed.loc[:, list(dependencies)].itertuples(index=False, name=None)]
    new_keys = [_row_key(row) for row in added.loc[:, list(dependencies)].itertuples(index=False, name=None)]
    if Counter(old_keys) != Counter(new_keys):
        return False
    kept: dict[tuple[Any, ...], deque[bool]] = defaultdict(deque)
    for key, occurrence in zip(old_keys, removed.index, strict=True):
        kept[key].append(not cache[occurrence].empty)
    for occurrence in removed.index:
        cache.pop(occurrence)
    empty = cache.empty_like(added)
    for position, (key, occurrence) in enumerate(zip(new_keys, added.index, strict=True)):
        cache[occurrence] = added.iloc[[position]].copy() if kept[key].popleft() else empty
    return True


def _carry_unchanged_predicate_buckets(
    cache: RowOutputCache, removed: pd.DataFrame,
    added: pd.DataFrame, dependencies: tuple[str, ...], columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carry metadata-only buckets even when other buckets have semantic changes."""
    def buckets(frame: pd.DataFrame) -> dict[tuple[Any, ...], list[int]]:
        result: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for position, row in enumerate(frame.loc[:, list(columns)].itertuples(index=False, name=None)):
            result[_row_key(row)].append(position)
        return result

    old_buckets, new_buckets = buckets(removed), buckets(added)
    carried_old: set[int] = set()
    carried_new: set[int] = set()
    for key, old_positions in old_buckets.items():
        new_positions = new_buckets.get(key, [])
        if _carry_predicate_replacements(
            cache, removed.iloc[old_positions], added.iloc[new_positions], dependencies,
        ):
            carried_old.update(old_positions)
            carried_new.update(new_positions)
    return (
        removed.iloc[[i for i in range(len(removed)) if i not in carried_old]],
        added.iloc[[i for i in range(len(added)) if i not in carried_new]],
    )


def _affected_candidate_buckets(
    previous: pd.DataFrame, current: pd.DataFrame, removed: pd.DataFrame,
    added: pd.DataFrame, columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh complete changed Top-k buckets, including displaced surviving rows."""
    if not columns:
        # Symmetric Top-k is the union of both endpoints' choices. A local subset
        # cannot safely determine membership, so use the full candidate domain.
        return previous, current
    keys = {_row_key(row) for frame in (removed, added)
            for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)}

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        positions = [i for i, row in enumerate(frame.loc[:, list(columns)].itertuples(index=False, name=None))
                     if _row_key(row) in keys]
        return frame.iloc[positions]

    return select(previous), select(current)


def _index_join_rows(frame: pd.DataFrame) -> dict[tuple[Any, ...], tuple[Hashable, ...]]:
    """Index bag occurrences once; subsequent commits update only changed keys."""
    if not frame.index.is_unique:
        raise ValueError("join state requires unique occurrence IDs")
    rows: dict[tuple[Any, ...], list[Hashable]] = defaultdict(list)
    for occurrence, row in zip(frame.index, frame.itertuples(index=False, name=None), strict=True):
        rows[_row_key(row)].append(occurrence)
    return {key: tuple(ids) for key, ids in rows.items()}


def _apply_join_delta(
    old: pd.DataFrame,
    removed: pd.DataFrame,
    added: pd.DataFrame,
    *,
    allocate: Callable[[], Hashable],
    row_index: dict[tuple[Any, ...], tuple[Hashable, ...]] | None = None,
) -> NodeOutputUpdate:
    """Cancel changed bags, preserving old occurrence IDs and indexing new rows."""

    if any(list(frame.columns) != list(old.columns) for frame in (removed, added)):
        raise ValueError("Join delta columns must match the materialized output")
    # A replacement can remove and recreate the same output. Only its net
    # change may propagate to downstream semantic operators.
    if not removed.empty:
        added, removed = _multiset_difference(removed, added)
    positions: list[int] = []
    if not removed.empty:
        remaining = Counter(
            _row_key(row) for row in removed.itertuples(index=False, name=None)
        )
        if row_index is not None:
            occurrences: list[Hashable] = []
            for key, number in remaining.items():
                matches = row_index.get(key, ())
                if len(matches) < number:
                    raise RuntimeError("join retractions exceed materialized output multiplicity")
                occurrences.extend(matches[-number:])
                if len(matches) == number:
                    del row_index[key]
                else:
                    row_index[key] = matches[:-number]
            positions = old.index.get_indexer(occurrences).tolist()
            if -1 in positions:
                raise RuntimeError("join occurrence index does not match materialized state")
            remaining.clear()
        else:
            # Keep the standalone helper's scan path; committed executors supply
            # an index. Both paths retain the earliest equal bag occurrences.
            for reverse_position, row in enumerate(old.iloc[::-1].itertuples(index=False, name=None)):
                key = _row_key(row)
                if remaining.get(key, 0):
                    positions.append(len(old) - 1 - reverse_position)
                    remaining[key] -= 1
                    if remaining[key] == 0:
                        del remaining[key]
                    if not remaining:
                        break
        if remaining:
            raise RuntimeError("join retractions exceed materialized output multiplicity")
    retracted = old.iloc[sorted(positions)]
    retained = old.drop(index=retracted.index) if positions else old
    inserted = added.copy()
    inserted.index = pd.Index([allocate() for _ in range(len(inserted))], dtype="object")
    if row_index is not None:
        for key, new_occurrences in _index_join_rows(inserted).items():
            row_index[key] = row_index.get(key, ()) + new_occurrences
    output = retained if inserted.empty else (
        inserted if retained.empty else pd.concat([retained, inserted])
    )
    return NodeOutputUpdate(output, inserted, retracted)


def _copy_frame_with_objects(frame: pd.DataFrame) -> pd.DataFrame:
    """Detach caller-owned nested values at public input/output boundaries."""

    result = frame.copy(deep=True)
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(deepcopy)
    return result


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


def _preserve_occurrences_and_diff(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    allocate: Callable[[], Hashable],
) -> NodeOutputUpdate:
    """Pair equal occurrences once, retaining the existing bag and ID order."""

    if list(old.columns) != list(new.columns):
        raise ValueError("Relation states must have identical ordered columns")
    available: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    old_indexes: list[Hashable] = list(old.index)
    for position, row in enumerate(old.itertuples(index=False, name=None)):
        available[_row_key(row)].append(position)
    retained = bytearray(len(old))
    indexes: list[Hashable] = []
    inserted: list[int] = []
    for position, row in enumerate(new.itertuples(index=False, name=None)):
        matches = available.get(_row_key(row))
        if matches:
            old_position = matches.popleft()
            retained[old_position] = 1
            indexes.append(old_indexes[old_position])
        else:
            indexes.append(allocate())
            inserted.append(position)
    result = new.copy()
    result.index = pd.Index(indexes, dtype="object")
    retracted = [position for position, keep in enumerate(retained) if not keep]
    return NodeOutputUpdate(result, result.iloc[inserted].copy(), old.iloc[retracted].copy())


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


def _adapter_execution_fingerprint(adapter: Any) -> str:
    """Return an optional adapter-owned maintenance execution identity."""

    value = getattr(adapter, "maintenance_execution_fingerprint", "")
    if not isinstance(value, str):
        raise TypeError("Adapter maintenance execution fingerprint must be a string")
    return value
