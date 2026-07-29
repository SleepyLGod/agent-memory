"""Embedded Qdrant connector with marker-published point versions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pandas as pd

from agent_memory.storage.connector import StorageCommit
from agent_memory.storage.embedding import EmbeddingProvider
from agent_memory.storage.identity import physical_uuid
from agent_memory.storage.search import SearchBatch, SearchRequest
from agent_memory.storage.statements import InsertStatement, StatementSet

from .mapping import QdrantPointMapping
from .recovery import (
    clear_marker,
    publish_marker,
    read_marker,
    require_marker,
)
from .schema import ensure_collection, ensure_control_collection
from .search import execute_search
from .sink import (
    MaterializedPoint,
    PreparedQdrantWrite,
    materialize_retractions,
    materialize_rows,
)


class QdrantConnector:
    """Single-owner embedded Qdrant materialization and search connector."""

    def __init__(
        self,
        *,
        path: str | Path,
        embedding_provider: EmbeddingProvider,
        client: Any | None = None,
        models: Any | None = None,
    ) -> None:
        path_value = str(path)
        if not path_value or "://" in path_value:
            raise ValueError(
                "Qdrant connector requires an embedded local filesystem path"
            )
        if embedding_provider is None:
            raise TypeError("Qdrant connector requires an embedding provider")
        if (client is None) != (models is None):
            raise ValueError("Qdrant client and models test overrides must be paired")
        if client is None:
            try:
                from qdrant_client import QdrantClient, models as qdrant_models
            except ImportError as exc:
                raise ImportError(
                    "local Qdrant storage requires the optional 'mem0' "
                    "dependency extra"
                ) from exc
            # Qdrant local owns a filesystem lock and rejects a second owner.
            client = QdrantClient(path=str(Path(path_value).expanduser()))
            models = qdrant_models
        if client is None or models is None:
            raise RuntimeError("Qdrant client initialization did not complete")
        self.path = path_value
        self.embedding_provider = embedding_provider
        self._client: Any = client
        self._models: Any = models
        self._statement_collections: dict[str, str] = {}

    def prepare(self, statements: StatementSet) -> None:
        """Validate targets and idempotently prepare local collections."""

        mappings = self._validate_statements(statements)
        self._statement_collections = {
            statement.statement_id: mapping.collection
            for statement, mapping in zip(
                statements.statements,
                mappings,
                strict=True,
            )
        }
        ensure_control_collection(self._client, self._models)
        for mapping in mappings:
            ensure_collection(
                self._client,
                self._models,
                collection=mapping.collection,
                dimensions=mapping.embedding.dimensions,
            )

    def read_commit(self, *, namespace: str) -> StorageCommit | None:
        """Read the currently published namespace commit."""

        _validate_namespace(namespace)
        marker = read_marker(self._client, namespace=namespace)
        return None if marker is None else marker.commit

    def transaction(
        self,
        *,
        namespace: str,
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> AbstractContextManager[_QdrantStorageTransaction]:
        """Return a staged transaction whose marker is published last."""

        _validate_namespace(namespace)
        _validate_commit_transition(expected_commit, next_commit)
        return _QdrantStorageTransaction(
            self,
            namespace=namespace,
            expected_commit=expected_commit,
            next_commit=next_commit,
        )

    def rebuild(
        self,
        *,
        namespace: str,
        statements: StatementSet,
        rows_by_statement: Mapping[str, pd.DataFrame],
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> None:
        """Publish a full checkpoint through a fresh materialization."""

        _validate_namespace(namespace)
        mappings = self._validate_statements(statements)
        expected_ids = {statement.statement_id for statement in statements.statements}
        if set(rows_by_statement) != expected_ids:
            raise ValueError(
                "Qdrant rebuild rows must exactly match storage statement IDs"
            )
        require_marker(
            self._client,
            namespace=namespace,
            expected_commit=expected_commit,
        )
        if next_commit.is_initial:
            if any(not rows_by_statement[item].empty for item in expected_ids):
                raise ValueError("initial Qdrant rebuild must contain no sink rows")
            clear_marker(
                self._client,
                self._models,
                namespace=namespace,
                expected_commit=expected_commit,
            )
            return

        materialization_id = physical_uuid(
            namespace,
            "qdrant_materialization",
            (
                "rebuild",
                next_commit.lineage_id,
                next_commit.commit_sequence,
                next_commit.plan_fingerprint,
            ),
        )
        # Every embedding and row validation completes before the first upsert.
        prepared = tuple(
            PreparedQdrantWrite(
                inserted=materialize_rows(
                    mapping,
                    rows_by_statement[statement.statement_id],
                    namespace=namespace,
                    statement_id=statement.statement_id,
                    materialization_id=materialization_id,
                    visible_from=0,
                    embedding_provider=self.embedding_provider,
                ),
                retracted_record_ids=(),
            )
            for statement, mapping in zip(
                statements.statements,
                mappings,
                strict=True,
            )
        )
        require_marker(
            self._client,
            namespace=namespace,
            expected_commit=expected_commit,
        )
        self._upsert_points(prepared)
        publish_marker(
            self._client,
            self._models,
            namespace=namespace,
            expected_commit=expected_commit,
            next_commit=next_commit,
            materialization_id=materialization_id,
        )

    def search(self, request: SearchRequest) -> SearchBatch:
        """Execute one published dense retrieval request."""

        if not isinstance(request, SearchRequest):
            raise TypeError("Qdrant search requires a SearchRequest")
        return execute_search(
            self._client,
            self._models,
            request,
            embedding_provider=self.embedding_provider,
        )

    def close(self) -> None:
        """Release the embedded Qdrant filesystem owner."""

        self._client.close()

    def _validate_statements(
        self,
        statements: StatementSet,
    ) -> tuple[QdrantPointMapping, ...]:
        if not isinstance(statements, StatementSet):
            raise TypeError("Qdrant connector statements must be a StatementSet")
        mappings: list[QdrantPointMapping] = []
        collections: dict[str, int] = {}
        for statement in statements.statements:
            if statement.target.connector != "qdrant":
                raise ValueError("Qdrant connector only accepts qdrant table targets")
            mapping = statement.target.mapping
            if not isinstance(mapping, QdrantPointMapping):
                raise TypeError(
                    "Qdrant table targets require a typed QdrantPointMapping"
                )
            previous_dimensions = collections.setdefault(
                mapping.collection,
                mapping.embedding.dimensions,
            )
            if previous_dimensions != mapping.embedding.dimensions:
                raise ValueError(
                    "Qdrant mappings sharing a collection must use one dimension"
                )
            mappings.append(mapping)
        return tuple(mappings)

    def _upsert_points(self, writes: Sequence[PreparedQdrantWrite]) -> None:
        by_collection: dict[str, list[MaterializedPoint]] = defaultdict(list)
        for write in writes:
            for point in write.inserted:
                by_collection[point.collection].append(point)
        for collection, points in by_collection.items():
            self._client.upsert(
                collection_name=collection,
                points=[
                    self._models.PointStruct(
                        id=point.point_id,
                        vector=list(point.vector),
                        payload=dict(point.payload),
                    )
                    for point in points
                ],
                wait=True,
            )

    def _active_point_ids(
        self,
        *,
        namespace: str,
        statement_id: str,
        materialization_id: str,
        commit_sequence: int,
        record_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not record_ids:
            return ()
        collection = self._statement_collections.get(statement_id)
        if collection is None:
            raise RuntimeError(
                f"Qdrant statement {statement_id!r} was not prepared"
            )
        models = self._models
        points, offset = self._client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    _match(models, "_agent_memory_namespace", namespace),
                    _match(models, "_agent_memory_statement_id", statement_id),
                    _match(
                        models,
                        "_agent_memory_materialization",
                        materialization_id,
                    ),
                    models.FieldCondition(
                        key="_agent_memory_record_id",
                        match=models.MatchAny(any=list(record_ids)),
                    ),
                    models.FieldCondition(
                        key="_agent_memory_visible_from",
                        range=models.Range(lte=commit_sequence),
                    ),
                    models.FieldCondition(
                        key="_agent_memory_visible_until",
                        range=models.Range(gt=commit_sequence),
                    ),
                ]
            ),
            limit=max(256, len(record_ids)),
            with_payload=True,
            with_vectors=False,
        )
        if offset is not None:
            raise RuntimeError(
                "Qdrant active-point lookup exceeded one bounded page"
            )
        by_record: dict[str, list[str]] = defaultdict(list)
        for point in points:
            payload = point.payload
            if not isinstance(payload, Mapping):
                raise ValueError("Qdrant active point payload is missing")
            record_id = payload.get("_agent_memory_record_id")
            if not isinstance(record_id, str):
                raise ValueError("Qdrant active point record ID is missing")
            by_record[record_id].append(str(point.id))
        for record_id in record_ids:
            matches = by_record.get(record_id, [])
            if len(matches) != 1:
                raise RuntimeError(
                    "Qdrant retraction must resolve exactly one active point for "
                    f"{record_id!r}; found {len(matches)}"
                )
        return tuple(by_record[record_id][0] for record_id in record_ids)


class _QdrantStorageTransaction:
    """Collect one runtime step before publishing a Qdrant marker."""

    def __init__(
        self,
        connector: QdrantConnector,
        *,
        namespace: str,
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> None:
        self.connector = connector
        self.namespace = namespace
        self.expected_commit = expected_commit
        self.next_commit = next_commit
        self._writes: list[tuple[InsertStatement, pd.DataFrame, pd.DataFrame]] = []
        self._statement_ids: set[str] = set()

    def __enter__(self) -> _QdrantStorageTransaction:
        return self

    def write(
        self,
        statement: InsertStatement,
        *,
        inserted_rows: pd.DataFrame,
        retracted_rows: pd.DataFrame,
    ) -> None:
        """Stage one sink changelog without touching Qdrant."""

        if not isinstance(statement, InsertStatement):
            raise TypeError("Qdrant transaction statement must be InsertStatement")
        if not isinstance(inserted_rows, pd.DataFrame) or not isinstance(
            retracted_rows,
            pd.DataFrame,
        ):
            raise TypeError("Qdrant transaction rows must be pandas DataFrames")
        if statement.statement_id in self._statement_ids:
            raise ValueError(
                f"Qdrant transaction already contains {statement.statement_id!r}"
            )
        mapping = statement.target.mapping
        if not isinstance(mapping, QdrantPointMapping):
            raise TypeError(
                "Qdrant table targets require a typed QdrantPointMapping"
            )
        self._statement_ids.add(statement.statement_id)
        self._writes.append(
            (statement, inserted_rows.copy(), retracted_rows.copy())
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if exc_type is not None:
            return False

        marker = require_marker(
            self.connector._client,
            namespace=self.namespace,
            expected_commit=self.expected_commit,
        )
        materialization_id = (
            marker.materialization_id
            if marker is not None
            else physical_uuid(
                self.namespace,
                "qdrant_materialization",
                ("lineage", self.next_commit.lineage_id),
            )
        )
        prepared: list[PreparedQdrantWrite] = []
        statement_mappings: list[tuple[InsertStatement, QdrantPointMapping]] = []
        for statement, inserted_rows, retracted_rows in self._writes:
            mapping = statement.target.mapping
            if not isinstance(mapping, QdrantPointMapping):
                raise TypeError(
                    "Qdrant table targets require a typed QdrantPointMapping"
                )
            prepared.append(
                PreparedQdrantWrite(
                    inserted=materialize_rows(
                        mapping,
                        inserted_rows,
                        namespace=self.namespace,
                        statement_id=statement.statement_id,
                        materialization_id=materialization_id,
                        visible_from=self.next_commit.commit_sequence,
                        embedding_provider=self.connector.embedding_provider,
                    ),
                    retracted_record_ids=materialize_retractions(
                        mapping,
                        retracted_rows,
                        namespace=self.namespace,
                    ),
                )
            )
            statement_mappings.append((statement, mapping))

        # Resolve all old versions before the first mutation.
        closures = tuple(
            (
                mapping.collection,
                self.connector._active_point_ids(
                    namespace=self.namespace,
                    statement_id=statement.statement_id,
                    materialization_id=materialization_id,
                    commit_sequence=(
                        0
                        if self.expected_commit is None
                        else self.expected_commit.commit_sequence
                    ),
                    record_ids=write.retracted_record_ids,
                ),
            )
            for write, (statement, mapping) in zip(
                prepared,
                statement_mappings,
                strict=True,
            )
        )
        require_marker(
            self.connector._client,
            namespace=self.namespace,
            expected_commit=self.expected_commit,
        )
        for collection, point_ids in closures:
            if point_ids:
                self.connector._client.set_payload(
                    collection_name=collection,
                    payload={
                        "_agent_memory_visible_until": (
                            self.next_commit.commit_sequence
                        )
                    },
                    points=list(point_ids),
                    wait=True,
                )
        self.connector._upsert_points(prepared)
        publish_marker(
            self.connector._client,
            self.connector._models,
            namespace=self.namespace,
            expected_commit=self.expected_commit,
            next_commit=self.next_commit,
            materialization_id=materialization_id,
        )
        return False


def _match(models: Any, key: str, value: str) -> Any:
    return models.FieldCondition(
        key=key,
        match=models.MatchValue(value=value),
    )


def _validate_namespace(namespace: str) -> None:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("storage namespace must be a non-empty string")


def _validate_commit_transition(
    expected_commit: StorageCommit | None,
    next_commit: StorageCommit,
) -> None:
    if expected_commit is not None and not isinstance(expected_commit, StorageCommit):
        raise TypeError("expected storage commit must be StorageCommit or None")
    if not isinstance(next_commit, StorageCommit):
        raise TypeError("next storage commit must be StorageCommit")
    if expected_commit is not None:
        if next_commit.lineage_id != expected_commit.lineage_id:
            raise ValueError("storage commit lineage cannot change during add")
        if next_commit.plan_fingerprint != expected_commit.plan_fingerprint:
            raise ValueError("storage commit plan fingerprint cannot change during add")
        if next_commit.commit_sequence != expected_commit.commit_sequence + 1:
            raise ValueError("storage commit sequence must advance by exactly one")
        if next_commit.source_row_count < expected_commit.source_row_count:
            raise ValueError("storage source row count cannot decrease during add")


__all__ = ["QdrantConnector"]
