"""Neo4j connector lifecycle, transactions, and namespace rebuilds."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

import pandas as pd

from agent_memory.storage.connector import StorageCommit
from agent_memory.storage.embedding import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from agent_memory.storage.statements import InsertStatement, StatementSet
from agent_memory.storage.search import (
    CrossEncoderProvider,
    SearchBatch,
    SearchRequest,
)

from .mapping import Neo4jNodeMapping, Neo4jRelationshipMapping
from .recovery import (
    clear_namespace,
    compare_and_set_commit,
    delete_commit,
    lock_namespace_for_rebuild,
    read_commit,
    write_commit,
)
from .schema import Neo4jSchema
from .search import execute_search
from .sink import PreparedWrite, apply_writes, materialize_rows


class Neo4jConnector:
    """Transactional Neo4j materialization connector."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        auth: tuple[str, str] | None = None,
        database: str = "neo4j",
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: CrossEncoderProvider | None = None,
        schema: Neo4jSchema,
        driver: Any | None = None,
    ) -> None:
        if not isinstance(database, str) or not database:
            raise ValueError("Neo4j database must be a non-empty string")
        if not isinstance(schema, Neo4jSchema):
            raise TypeError("Neo4j connector schema must be Neo4jSchema")
        if driver is None:
            if not isinstance(uri, str) or not uri:
                raise ValueError("Neo4j connector requires a non-empty URI")
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise ImportError(
                    "Neo4j storage requires the optional 'zep' dependency extra"
                ) from exc
            driver = GraphDatabase.driver(uri, auth=auth)
        self._driver = driver
        self.database = database
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.schema = schema

    def prepare(self, statements: StatementSet) -> None:
        """Validate all targets and idempotently create required schema."""

        self._validate_statements(statements)
        with self._driver.session(database=self.database) as session:
            self.schema.ensure(session)

    def read_commit(self, *, namespace: str) -> StorageCommit | None:
        """Read the current namespace marker."""

        _validate_namespace(namespace)
        with self._driver.session(database=self.database) as session:
            return read_commit(session, namespace=namespace)

    def transaction(
        self,
        *,
        namespace: str,
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> AbstractContextManager[_Neo4jStorageTransaction]:
        """Return a staging transaction that embeds before opening Neo4j."""

        _validate_namespace(namespace)
        _validate_commit_transition(expected_commit, next_commit)
        return _Neo4jStorageTransaction(
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
        """Atomically replace one namespace from full logical sink state."""

        _validate_namespace(namespace)
        self._validate_statements(statements)
        expected_ids = {statement.statement_id for statement in statements.statements}
        if set(rows_by_statement) != expected_ids:
            raise ValueError(
                "Neo4j rebuild rows must exactly match storage statement IDs"
            )
        prepared = tuple(
            self._prepare_write(
                statement,
                namespace=namespace,
                inserted_rows=rows_by_statement[statement.statement_id],
                retracted_rows=rows_by_statement[statement.statement_id].iloc[0:0],
            )
            for statement in statements.statements
        )

        def rebuild_transaction(tx: Any) -> None:
            lock_namespace_for_rebuild(
                tx,
                namespace=namespace,
                expected_commit=expected_commit,
            )
            clear_namespace(tx, namespace=namespace)
            apply_writes(tx, prepared, namespace=namespace)
            if next_commit.is_initial:
                delete_commit(tx, namespace=namespace)
            else:
                write_commit(tx, namespace=namespace, commit=next_commit)

        with self._driver.session(database=self.database) as session:
            session.execute_write(rebuild_transaction)

    def close(self) -> None:
        """Close the underlying Neo4j driver."""

        self._driver.close()

    def server_version(self) -> str:
        """Return the server version reported by the configured database."""

        records, _, _ = self._driver.execute_query(
            """
            CALL dbms.components()
            YIELD versions
            RETURN versions[0] AS version
            """,
            database_=self.database,
        )
        if len(records) != 1 or not isinstance(records[0].get("version"), str):
            raise RuntimeError("Neo4j did not report exactly one server version")
        return records[0]["version"]

    def search(self, request: SearchRequest) -> SearchBatch:
        """Execute one namespace-scoped physical retrieval request."""

        if not isinstance(request, SearchRequest):
            raise TypeError("Neo4j search requires a SearchRequest")
        with self._driver.session(database=self.database) as session:
            return execute_search(
                session,
                request,
                schema=self.schema,
                embedding_provider=self.embedding_provider,
                reranker_provider=self.reranker_provider,
            )

    def _validate_statements(self, statements: StatementSet) -> None:
        if not isinstance(statements, StatementSet):
            raise TypeError("Neo4j connector statements must be a StatementSet")
        for statement in statements.statements:
            if statement.target.connector != "neo4j":
                raise ValueError("Neo4j connector only accepts neo4j table targets")
            mapping = statement.target.mapping
            if not isinstance(mapping, (Neo4jNodeMapping, Neo4jRelationshipMapping)):
                raise TypeError("Neo4j table targets require a typed Neo4j mapping")
            if mapping.embedding is not None and self.embedding_provider is None:
                raise ValueError(
                    "Neo4j embedding mappings require an embedding provider"
                )

    def _prepare_write(
        self,
        statement: InsertStatement,
        *,
        namespace: str,
        inserted_rows: pd.DataFrame,
        retracted_rows: pd.DataFrame,
    ) -> PreparedWrite:
        mapping = statement.target.mapping
        if not isinstance(mapping, (Neo4jNodeMapping, Neo4jRelationshipMapping)):
            raise TypeError("Neo4j table targets require a typed Neo4j mapping")
        return PreparedWrite(
            inserted=tuple(
                materialize_rows(
                    mapping,
                    inserted_rows,
                    namespace=namespace,
                    embedding_provider=self.embedding_provider,
                    include_embeddings=True,
                )
            ),
            retracted=tuple(
                materialize_rows(
                    mapping,
                    retracted_rows,
                    namespace=namespace,
                    embedding_provider=self.embedding_provider,
                    include_embeddings=False,
                )
            ),
        )

class _Neo4jStorageTransaction:
    """Collect logical changelog rows before one physical Neo4j transaction."""

    def __init__(
        self,
        connector: Neo4jConnector,
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

    def __enter__(self) -> _Neo4jStorageTransaction:
        return self

    def write(
        self,
        statement: InsertStatement,
        *,
        inserted_rows: pd.DataFrame,
        retracted_rows: pd.DataFrame,
    ) -> None:
        """Stage one logical sink changelog without touching Neo4j."""

        if not isinstance(statement, InsertStatement):
            raise TypeError("Neo4j transaction statement must be InsertStatement")
        if not isinstance(inserted_rows, pd.DataFrame) or not isinstance(
            retracted_rows, pd.DataFrame
        ):
            raise TypeError("Neo4j transaction rows must be pandas DataFrames")
        if statement.statement_id in self._statement_ids:
            raise ValueError(
                f"Neo4j transaction already contains {statement.statement_id!r}"
            )
        self._statement_ids.add(statement.statement_id)
        self._writes.append(
            (statement, inserted_rows.copy(), retracted_rows.copy())
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if exc_type is not None:
            return False

        prepared = tuple(
            self.connector._prepare_write(
                statement,
                namespace=self.namespace,
                inserted_rows=inserted_rows,
                retracted_rows=retracted_rows,
            )
            for statement, inserted_rows, retracted_rows in self._writes
        )

        def commit_transaction(tx: Any) -> None:
            apply_writes(tx, prepared, namespace=self.namespace)
            compare_and_set_commit(
                tx,
                namespace=self.namespace,
                expected_commit=self.expected_commit,
                next_commit=self.next_commit,
            )

        with self.connector._driver.session(database=self.connector.database) as session:
            session.execute_write(commit_transaction)
        return False


class SentenceTransformerCrossEncoderProvider:
    """CPU sentence-transformers provider for one cross-encoder model."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str = "cpu",
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("cross-encoder model must be a non-empty string")
        if device != "cpu":
            raise ValueError("Zep baseline reranking is fixed to CPU")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "BGE reranking requires the optional 'zep' dependency extra"
            ) from exc
        self.model = model
        self._model = CrossEncoder(model, device=device)

    def rank(
        self,
        *,
        model: str,
        query: str,
        passages: list[str],
    ) -> list[tuple[int, float]]:
        """Score query/passage pairs and return descending results."""

        if model != self.model:
            raise ValueError("reranker request does not match the configured model")
        if not passages:
            return []
        scores = self._model.predict([[query, passage] for passage in passages])
        return sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
            ),
            key=lambda item: item[1],
            reverse=True,
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


__all__ = [
    "Neo4jConnector",
    "SentenceTransformerCrossEncoderProvider",
    "SentenceTransformerEmbeddingProvider",
]
