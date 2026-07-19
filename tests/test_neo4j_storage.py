"""Tests for typed Neo4j storage mappings and physical row encoding."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.planner import PolicyDifferentiator
from agent_memory.storage import Schema, StatementSet, TableDescriptor
from agent_memory.storage import SearchRequest
from agent_memory.storage.neo4j import (
    EmbeddingSpec,
    Neo4jConnector,
    Neo4jIdentity,
    Neo4jNestedProperty,
    Neo4jNodeMapping,
    Neo4jRelationshipMapping,
)
from agent_memory.policy.retrieval import RerankerSpec, SearchMethodSpec
from agent_memory.memories.zep.storage import (
    GRAPHITI_NEO4J_SCHEMA,
    GRAPHITI_NEO4J_STATEMENTS,
)
from agent_memory.storage.connector import StorageCommit, StorageConflictError
from agent_memory.policy.logical import QueryExpr
from agent_memory.storage.statements import InsertStatement
from agent_memory.storage.neo4j.sink import (
    MaterializedNode,
    MaterializedRelationship,
    PreparedWrite,
    apply_writes,
    materialize_rows,
    physical_uuid,
)
from agent_memory.storage.neo4j.search import execute_search


def _entity_schema() -> Schema:
    return (
        Schema.new_builder()
        .column("entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
        .column("name", "STRING")
        .column("summary", "STRING")
        .column("mentions", "ARRAY<ROW<created_at TIMESTAMP_LTZ(6)>>")
        .primary_key("entity_id")
        .build()
    )


def _embedding(*, revision: str = "revision-a") -> EmbeddingSpec:
    return EmbeddingSpec(
        source_column="name",
        property_name="name_embedding",
        model="BAAI/bge-m3",
        revision=revision,
        dimensions=1024,
        normalize=True,
    )


def test_table_descriptor_serializes_typed_connector_mapping() -> None:
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"name": "name", "summary": "summary"},
        nested_properties=(
            Neo4jNestedProperty(
                source_column="mentions",
                field="created_at",
                property_name="created_at",
                many=False,
            ),
        ),
        embedding=_embedding(),
    )

    target = (
        TableDescriptor.for_connector("neo4j")
        .schema(_entity_schema())
        .mapping(mapping)
        .build()
    )

    serialized = json.loads(json.dumps(target.to_dict()))
    assert serialized["mapping"] == mapping.to_dict()
    assert serialized["mapping"]["kind"] == "node"
    assert serialized["mapping"]["embedding"]["revision"] == "revision-a"
    assert hash(target) == hash(target)


def test_table_descriptor_rejects_mapping_for_another_connector() -> None:
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"name": "name"},
    )

    with pytest.raises(ValueError, match="mapping connector"):
        (
            TableDescriptor.for_connector("recording")
            .schema(_entity_schema())
            .mapping(mapping)
            .build()
        )


def test_neo4j_mapping_rejects_unknown_logical_columns() -> None:
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("missing_id",), kind="entity"),
        properties={"name": "missing_name"},
    )

    with pytest.raises(ValueError, match="unknown logical column"):
        (
            TableDescriptor.for_connector("neo4j")
            .schema(_entity_schema())
            .mapping(mapping)
            .build()
        )


def test_neo4j_relationship_mapping_validates_endpoints_and_nested_projection() -> None:
    schema = (
        Schema.new_builder()
        .column("fact_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
        .column("source_entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
        .column("target_entity_id", "ROW<add_seq BIGINT, ordinal BIGINT>")
        .column("fact", "STRING")
        .column("provenance", "ARRAY<ROW<episode_id STRING>>")
        .primary_key("fact_id")
        .build()
    )
    mapping = Neo4jRelationshipMapping(
        relationship_type="RELATES_TO",
        identity=Neo4jIdentity(columns=("fact_id",), kind="fact"),
        source=Neo4jIdentity(columns=("source_entity_id",), kind="entity"),
        target=Neo4jIdentity(columns=("target_entity_id",), kind="entity"),
        properties={"fact": "fact"},
        nested_properties=(
            Neo4jNestedProperty(
                source_column="provenance",
                field="episode_id",
                property_name="episodes",
                many=True,
                identity_kind="episode",
            ),
        ),
    )

    target = (
        TableDescriptor.for_connector("neo4j")
        .schema(schema)
        .mapping(mapping)
        .build()
    )

    assert target.mapping == mapping
    assert target.to_dict()["mapping"]["source"]["kind"] == "entity"
    assert target.to_dict()["mapping"]["nested_properties"][0][
        "identity_kind"
    ] == "episode"


def test_mapping_model_revision_changes_table_identity() -> None:
    first = (
        TableDescriptor.for_connector("neo4j")
        .schema(_entity_schema())
        .mapping(
            Neo4jNodeMapping(
                label="Entity",
                identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
                properties={"name": "name"},
                embedding=_embedding(revision="revision-a"),
            )
        )
        .build()
    )
    second = (
        TableDescriptor.for_connector("neo4j")
        .schema(_entity_schema())
        .mapping(
            Neo4jNodeMapping(
                label="Entity",
                identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
                properties={"name": "name"},
                embedding=_embedding(revision="revision-b"),
            )
        )
        .build()
    )

    assert first != second
    assert first.to_dict() != second.to_dict()


def test_mapping_model_revision_changes_storage_plan_fingerprint() -> None:
    class EntityMemory(am.Memory):
        log = am.Log(
            {
                "entity_id": "Entity id.",
                "name": "Name.",
                "summary": "Summary.",
                "mentions": "Mentions.",
            }
        )
        entities = log.select(["entity_id", "name", "summary", "mentions"])

    def fingerprint(revision: str) -> str:
        target = (
            TableDescriptor.for_connector("neo4j")
            .schema(_entity_schema())
            .mapping(
                Neo4jNodeMapping(
                    label="Entity",
                    identity=Neo4jIdentity(("entity_id",), "entity"),
                    properties={"name": "name", "summary": "summary"},
                    embedding=_embedding(revision=revision),
                )
            )
            .build()
        )
        statements = StatementSet().add_insert(target, EntityMemory.entities)
        return PolicyDifferentiator().differentiate(
            EntityMemory.spec(),
            statements=statements,
        ).fingerprint

    assert fingerprint("revision-a") != fingerprint("revision-b")


def test_mapping_constants_are_deeply_immutable() -> None:
    mapping = Neo4jNodeMapping(
        label="Episodic",
        identity=Neo4jIdentity(columns=("entity_id",), kind="episode"),
        constants={"entity_edges": []},
    )

    assert mapping.constants["entity_edges"] == ()
    assert mapping.to_dict()["constants"] == {"entity_edges": []}


class _EmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[EmbeddingSpec, list[str]]] = []

    def embed(self, spec: EmbeddingSpec, texts: list[str]) -> list[list[float]]:
        self.calls.append((spec, list(texts)))
        return [[float(index)] * spec.dimensions for index, _ in enumerate(texts)]


class _SearchEmbeddingProvider:
    def embed(self, spec: EmbeddingSpec, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _RerankerProvider:
    def rank(
        self,
        *,
        model: str,
        query: str,
        passages: list[str],
    ) -> list[tuple[str, float]]:
        assert model == "BAAI/bge-reranker-v2-m3"
        assert query == "Where does Alice live?"
        scores = {"Alice lives in Paris": 0.9, "Alice likes running": 0.2}
        return sorted(
            ((passage, scores[passage]) for passage in passages),
            key=lambda item: item[1],
            reverse=True,
        )


class _SearchResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def data(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _SearchSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **parameters: Any) -> _SearchResult:
        self.calls.append((query, dict(parameters)))
        if "queryNodes" in query:
            rows = [
                {
                    "record_id": "entity-a",
                    "properties": {"name": "Alice", "summary": "Runner"},
                    "method_score": 3.0,
                },
                {
                    "record_id": "entity-b",
                    "properties": {"name": "Paris", "summary": "City"},
                    "method_score": 2.0,
                },
            ]
        elif "MATCH (record:Entity" in query:
            rows = [
                {
                    "record_id": "entity-b",
                    "properties": {"name": "Paris", "summary": "City"},
                    "method_score": 0.9,
                },
                {
                    "record_id": "entity-a",
                    "properties": {"name": "Alice", "summary": "Runner"},
                    "method_score": 0.8,
                },
            ]
        elif "queryRelationships" in query:
            rows = [
                {
                    "record_id": "fact-running",
                    "properties": {
                        "fact": "Alice likes running",
                        "valid_at": None,
                        "invalid_at": None,
                        "expired_at": None,
                    },
                    "method_score": 2.0,
                }
            ]
        elif "UNWIND $origin_record_ids" in query:
            rows = [
                {
                    "record_id": "fact-running",
                    "properties": {
                        "fact": "Alice likes running",
                        "valid_at": None,
                        "invalid_at": None,
                        "expired_at": None,
                    },
                    "method_score": 1.0,
                }
            ]
        elif "MATCH ()-[record:RELATES_TO]" in query:
            rows = [
                {
                    "record_id": "fact-paris",
                    "properties": {
                        "fact": "Alice lives in Paris",
                        "valid_at": "2026-01-01T00:00:00",
                        "invalid_at": None,
                        "expired_at": None,
                    },
                    "method_score": 0.9,
                }
            ]
        else:
            raise AssertionError(f"unexpected search query: {query}")
        return _SearchResult(rows)


class _SearchDriver:
    def __init__(self) -> None:
        self.search_session = _SearchSession()

    @contextmanager
    def session(self, *, database: str) -> Any:
        assert database == "neo4j"
        yield self.search_session

    def close(self) -> None:
        return None


def _search_target(mapping: Neo4jNodeMapping | Neo4jRelationshipMapping) -> TableDescriptor:
    schema = (
        _entity_schema()
        if isinstance(mapping, Neo4jNodeMapping)
        else (
            Schema.new_builder()
            .column("fact_id", "STRING")
            .column("source_entity_id", "STRING")
            .column("target_entity_id", "STRING")
            .column("fact", "STRING")
            .column("valid_at", "TIMESTAMP(6)")
            .column("invalid_at", "TIMESTAMP(6)")
            .column("expired_at", "TIMESTAMP_LTZ(6)")
            .primary_key("fact_id")
            .build()
        )
    )
    return TableDescriptor.for_connector("neo4j").schema(schema).mapping(mapping).build()


def test_neo4j_entity_search_fuses_bm25_and_cosine_with_stable_rrf() -> None:
    embedding = EmbeddingSpec(
        source_column="name",
        property_name="name_embedding",
        model="test-embedding",
        revision="revision-a",
        dimensions=2,
        normalize=True,
    )
    target = _search_target(
        Neo4jNodeMapping(
            label="Entity",
            identity=Neo4jIdentity(("entity_id",), "entity"),
            properties={"name": "name", "summary": "summary"},
            embedding=embedding,
        )
    )
    session = _SearchSession()

    batch = execute_search(
        session,
        SearchRequest(
            statement_id="sink_0001",
            target=target,
            namespace="sample-0",
            query="Where does Alice live?",
            methods=(
                SearchMethodSpec("bm25"),
                SearchMethodSpec("cosine_similarity"),
            ),
            reranker=RerankerSpec("rrf"),
            limit=20,
            output_columns=("record_id", "name", "summary", "rank", "score"),
        ),
        schema=GRAPHITI_NEO4J_SCHEMA,
        embedding_provider=_SearchEmbeddingProvider(),
        reranker_provider=None,
    )

    assert batch.rows.to_dict("records") == [
        {
            "record_id": "entity-a",
            "name": "Alice",
            "summary": "Runner",
            "rank": 1,
            "score": 1.5,
        },
        {
            "record_id": "entity-b",
            "name": "Paris",
            "summary": "City",
            "rank": 2,
            "score": 1.5,
        },
    ]
    assert all(parameters["namespace"] == "sample-0" for _, parameters in session.calls)
    assert batch.metrics["methods"][0]["candidate_ids"] == ["entity-a", "entity-b"]


def test_neo4j_fact_search_uses_bfs_origins_and_cross_encoder() -> None:
    embedding = EmbeddingSpec(
        source_column="fact",
        property_name="fact_embedding",
        model="test-embedding",
        revision="revision-a",
        dimensions=2,
        normalize=True,
    )
    target = _search_target(
        Neo4jRelationshipMapping(
            relationship_type="RELATES_TO",
            identity=Neo4jIdentity(("fact_id",), "fact"),
            source=Neo4jIdentity(("source_entity_id",), "entity"),
            target=Neo4jIdentity(("target_entity_id",), "entity"),
            properties={
                "fact": "fact",
                "valid_at": "valid_at",
                "invalid_at": "invalid_at",
                "expired_at": "expired_at",
            },
            embedding=embedding,
        )
    )
    session = _SearchSession()

    batch = execute_search(
        session,
        SearchRequest(
            statement_id="sink_0002",
            target=target,
            namespace="sample-0",
            query="Where does Alice live?",
            methods=(
                SearchMethodSpec("bm25"),
                SearchMethodSpec("cosine_similarity"),
                SearchMethodSpec("bfs", {"origin_input": 1, "max_depth": 3}),
            ),
            reranker=RerankerSpec(
                "cross_encoder", {"model": "BAAI/bge-reranker-v2-m3"}
            ),
            limit=20,
            output_columns=(
                "record_id",
                "fact",
                "valid_at",
                "invalid_at",
                "expired_at",
                "rank",
                "score",
            ),
            origin_record_ids=("entity-a",),
        ),
        schema=GRAPHITI_NEO4J_SCHEMA,
        embedding_provider=_SearchEmbeddingProvider(),
        reranker_provider=_RerankerProvider(),
    )

    assert batch.rows.iloc[0].to_dict() == {
        "record_id": "fact-paris",
        "fact": "Alice lives in Paris",
        "valid_at": "2026-01-01T00:00:00",
        "invalid_at": None,
        "expired_at": None,
        "rank": 1,
        "score": 0.9,
    }
    bfs_call = next(call for call in session.calls if "UNWIND $origin_record_ids" in call[0])
    assert bfs_call[1]["origin_record_ids"] == ["entity-a"]
    assert batch.metrics["bfs_origins"] == ["entity-a"]


def test_neo4j_search_rejects_logical_columns_not_readable_from_mapping() -> None:
    target = _search_target(
        Neo4jNodeMapping(
            label="Entity",
            identity=Neo4jIdentity(("entity_id",), "entity"),
            properties={"name": "name", "summary": "summary"},
        )
    )

    with pytest.raises(ValueError, match="cannot read logical column"):
        execute_search(
            _SearchSession(),
            SearchRequest(
                statement_id="sink_0001",
                target=target,
                namespace="sample-0",
                query="Alice",
                methods=(SearchMethodSpec("bm25"),),
                reranker=RerankerSpec("rrf"),
                limit=20,
                output_columns=("record_id", "mentions", "rank", "score"),
            ),
            schema=GRAPHITI_NEO4J_SCHEMA,
            embedding_provider=None,
            reranker_provider=None,
        )


def test_retrieval_planner_validates_neo4j_read_projection() -> None:
    class EntitySearchMemory(am.Memory):
        log = am.Log(
            {
                "entity_id": "Entity identifier.",
                "name": "Entity name.",
                "summary": "Entity summary.",
                "mentions": "Entity mentions.",
            }
        )
        entities = log.select(["entity_id", "name", "summary", "mentions"])
        retrieval_query = am.RetrievalQuery(
            entities=entities.search(
                am.UserQuery(),
                methods=[am.BM25()],
                reranker=am.RRF(),
                limit=20,
            ).select(["record_id", "summary", "rank", "score"])
        )

    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(("entity_id",), "entity"),
        properties={"name": "name"},
    )
    target = _search_target(mapping)

    with pytest.raises(ValueError, match="cannot read logical column"):
        PolicyDifferentiator().differentiate(
            EntitySearchMemory.spec(),
            statements=StatementSet().add_insert(target, EntitySearchMemory.entities),
        )


def test_neo4j_connector_exposes_search_capability() -> None:
    embedding = EmbeddingSpec(
        source_column="name",
        property_name="name_embedding",
        model="test-embedding",
        revision="revision-a",
        dimensions=2,
        normalize=True,
    )
    target = _search_target(
        Neo4jNodeMapping(
            label="Entity",
            identity=Neo4jIdentity(("entity_id",), "entity"),
            properties={"name": "name", "summary": "summary"},
            embedding=embedding,
        )
    )
    driver = _SearchDriver()
    connector = Neo4jConnector(
        driver=driver,
        database="neo4j",
        embedding_provider=_SearchEmbeddingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )

    batch = connector.search(
        SearchRequest(
            statement_id="sink_0001",
            target=target,
            namespace="sample-0",
            query="Where does Alice live?",
            methods=(
                SearchMethodSpec("bm25"),
                SearchMethodSpec("cosine_similarity"),
            ),
            reranker=RerankerSpec("rrf"),
            limit=20,
            output_columns=("record_id", "name", "summary", "rank", "score"),
        )
    )

    assert batch.rows.iloc[0]["record_id"] == "entity-a"
    assert len(driver.search_session.calls) == 2


def test_physical_uuid_is_stable_and_isolates_namespace_and_kind() -> None:
    logical_id = ((12, 0),)

    first = physical_uuid("sample-0", "entity", logical_id)

    assert first == physical_uuid("sample-0", "entity", logical_id)
    assert first != physical_uuid("sample-1", "entity", logical_id)
    assert first != physical_uuid("sample-0", "fact", logical_id)


def test_zep_mentions_keep_distinct_occurrences_for_one_canonical_entity() -> None:
    mapping = GRAPHITI_NEO4J_STATEMENTS.statements[3].target.mapping
    assert isinstance(mapping, Neo4jRelationshipMapping)
    rows = pd.DataFrame(
        [
            {
                "episode_id": "episode-a",
                "entity_ordinal": ordinal,
                "entity_id": (0, 0),
            }
            for ordinal in (0, 1)
        ]
    )

    materialized = materialize_rows(
        mapping,
        rows,
        namespace="sample-0",
        embedding_provider=None,
        include_embeddings=False,
    )

    assert [item.uuid for item in materialized] == [
        physical_uuid("sample-0", "mention", ("episode-a", 0)),
        physical_uuid("sample-0", "mention", ("episode-a", 1)),
    ]
    assert len({item.target_uuid for item in materialized}) == 1


def test_materialize_node_projects_nested_values_and_embedding() -> None:
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"name": "name", "summary": "summary"},
        nested_properties=(
            Neo4jNestedProperty(
                source_column="mentions",
                field="created_at",
                property_name="created_at",
                many=False,
            ),
        ),
        embedding=_embedding(),
    )
    created_at = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)
    rows = pd.DataFrame(
        [
            {
                "entity_id": (12, 0),
                "name": "Alice",
                "summary": "Alice likes running.",
                "mentions": json.dumps(
                    [{"created_at": created_at.isoformat()}]
                ),
            }
        ]
    )
    provider = _EmbeddingProvider()

    materialized = materialize_rows(
        mapping,
        rows,
        namespace="sample-0",
        embedding_provider=provider,
        include_embeddings=True,
    )

    assert materialized == [
        MaterializedNode(
            label="Entity",
            uuid=physical_uuid("sample-0", "entity", ((12, 0),)),
            properties={
                "uuid": physical_uuid("sample-0", "entity", ((12, 0),)),
                "group_id": "sample-0",
                "name": "Alice",
                "summary": "Alice likes running.",
                "created_at": created_at.isoformat(),
                "name_embedding": [0.0] * 1024,
            },
            embedding_property="name_embedding",
        )
    ]
    assert provider.calls == [(_embedding(), ["Alice"])]


def test_materialize_relationship_projects_provenance_episode_ids() -> None:
    mapping = Neo4jRelationshipMapping(
        relationship_type="RELATES_TO",
        identity=Neo4jIdentity(columns=("fact_id",), kind="fact"),
        source=Neo4jIdentity(columns=("source_entity_id",), kind="entity"),
        target=Neo4jIdentity(columns=("target_entity_id",), kind="entity"),
        properties={"name": "relation_type", "fact": "fact"},
        nested_properties=(
            Neo4jNestedProperty(
                source_column="provenance",
                field="episode_id",
                property_name="episodes",
                many=True,
                identity_kind="episode",
            ),
        ),
    )
    rows = pd.DataFrame(
        [
            {
                "fact_id": (20, 1),
                "source_entity_id": (12, 0),
                "target_entity_id": (12, 1),
                "relation_type": "LIKES",
                "fact": "Alice likes running.",
                "provenance": json.dumps(
                    [{"episode_id": "episode-a"}, {"episode_id": "episode-b"}]
                ),
            }
        ]
    )

    materialized = materialize_rows(
        mapping,
        rows,
        namespace="sample-0",
        embedding_provider=None,
        include_embeddings=False,
    )

    fact_uuid = physical_uuid("sample-0", "fact", ((20, 1),))
    source_uuid = physical_uuid("sample-0", "entity", ((12, 0),))
    target_uuid = physical_uuid("sample-0", "entity", ((12, 1),))
    assert materialized == [
        MaterializedRelationship(
            relationship_type="RELATES_TO",
            uuid=fact_uuid,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            properties={
                "uuid": fact_uuid,
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "group_id": "sample-0",
                "name": "LIKES",
                "fact": "Alice likes running.",
                "episodes": [
                    physical_uuid("sample-0", "episode", ("episode-a",)),
                    physical_uuid("sample-0", "episode", ("episode-b",)),
                ],
            },
        )
    ]


def test_retraction_materialization_does_not_call_embedding_provider() -> None:
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"name": "name"},
        embedding=_embedding(),
    )
    provider = _EmbeddingProvider()

    materialized = materialize_rows(
        mapping,
        pd.DataFrame([{"entity_id": (12, 0), "name": "Alice"}]),
        namespace="sample-0",
        embedding_provider=provider,
        include_embeddings=False,
    )

    assert provider.calls == []
    assert "name_embedding" not in materialized[0].properties


def test_materialize_rows_converts_pandas_nat_to_neo4j_null() -> None:
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"expired_at": "expired_at"},
    )

    materialized = materialize_rows(
        mapping,
        pd.DataFrame([{"entity_id": (12, 0), "expired_at": pd.NaT}]),
        namespace="sample-0",
        embedding_provider=None,
        include_embeddings=False,
    )

    assert materialized[0].properties["expired_at"] is None


def test_embedding_dimension_mismatch_is_rejected() -> None:
    class BadProvider:
        def embed(
            self, spec: EmbeddingSpec, texts: list[str]
        ) -> list[list[float]]:
            return [[0.0]]

    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"name": "name"},
        embedding=_embedding(),
    )

    with pytest.raises(ValueError, match="1024 dimensions"):
        materialize_rows(
            mapping,
            pd.DataFrame([{"entity_id": (12, 0), "name": "Alice"}]),
            namespace="sample-0",
            embedding_provider=BadProvider(),
            include_embeddings=True,
        )


class _QueryRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, **parameters: object) -> None:
        self.calls.append((query, dict(parameters)))


def test_same_key_node_replacement_is_upserted_without_delete() -> None:
    node_uuid = physical_uuid("sample-0", "entity", ((12, 0),))
    old = MaterializedNode(
        label="Entity",
        uuid=node_uuid,
        properties={"uuid": node_uuid, "group_id": "sample-0", "name": "Alice"},
    )
    new = MaterializedNode(
        label="Entity",
        uuid=node_uuid,
        properties={
            "uuid": node_uuid,
            "group_id": "sample-0",
            "name": "Alice Smith",
        },
    )
    tx = _QueryRecorder()

    apply_writes(
        tx,
        [PreparedWrite(inserted=(new,), retracted=(old,))],
        namespace="sample-0",
    )

    queries = [query for query, _ in tx.calls]
    assert len(queries) == 1
    assert "MERGE (n:Entity" in queries[0]
    assert "DETACH DELETE" not in queries[0]
    assert tx.calls[0][1]["rows"][0]["properties"]["name"] == "Alice Smith"


def test_relationship_endpoint_replacement_deletes_old_edge_before_upsert() -> None:
    edge_uuid = physical_uuid("sample-0", "fact", ((20, 0),))
    old = MaterializedRelationship(
        relationship_type="RELATES_TO",
        uuid=edge_uuid,
        source_uuid="old-source",
        target_uuid="old-target",
        properties={"uuid": edge_uuid, "group_id": "sample-0"},
    )
    new = MaterializedRelationship(
        relationship_type="RELATES_TO",
        uuid=edge_uuid,
        source_uuid="new-source",
        target_uuid="new-target",
        properties={"uuid": edge_uuid, "group_id": "sample-0"},
    )
    tx = _QueryRecorder()

    apply_writes(
        tx,
        [PreparedWrite(inserted=(new,), retracted=(old,))],
        namespace="sample-0",
    )

    queries = [query for query, _ in tx.calls]
    assert "DELETE r" in queries[0]
    assert "MERGE (source)-[r:RELATES_TO" in queries[1]


def test_relationship_retractions_run_before_node_retractions() -> None:
    relationship = MaterializedRelationship(
        relationship_type="MENTIONS",
        uuid="mention-id",
        source_uuid="episode-id",
        target_uuid="entity-id",
        properties={"uuid": "mention-id", "group_id": "sample-0"},
    )
    node = MaterializedNode(
        label="Entity",
        uuid="entity-id",
        properties={"uuid": "entity-id", "group_id": "sample-0"},
    )
    tx = _QueryRecorder()

    apply_writes(
        tx,
        [
            PreparedWrite(inserted=(), retracted=(node,)),
            PreparedWrite(inserted=(), retracted=(relationship,)),
        ],
        namespace="sample-0",
    )

    queries = [query for query, _ in tx.calls]
    assert "DELETE r" in queries[0]
    assert "DETACH DELETE n" in queries[1]


def test_duplicate_inserted_physical_keys_are_rejected() -> None:
    node = MaterializedNode(
        label="Entity",
        uuid="entity-id",
        properties={"uuid": "entity-id", "group_id": "sample-0"},
    )

    with pytest.raises(ValueError, match="duplicate inserted Neo4j identity"):
        apply_writes(
            _QueryRecorder(),
            [PreparedWrite(inserted=(node, node), retracted=())],
            namespace="sample-0",
        )


def test_graphiti_baseline_schema_contains_only_required_indexes() -> None:
    queries = GRAPHITI_NEO4J_SCHEMA.queries
    text = "\n".join(queries)

    assert "Episodic" in text
    assert "Entity" in text
    assert "RELATES_TO" in text
    assert "MENTIONS" in text
    assert "episode_content" in text
    assert "node_name_and_summary" in text
    assert "edge_name_and_fact" in text
    assert "Community" not in text
    assert "Saga" not in text
    assert "VECTOR INDEX" not in text
    assert len(queries) == len(set(queries))


def test_neo4j_schema_ignores_only_concurrent_equivalent_index_error() -> None:
    class SchemaError(RuntimeError):
        def __init__(self, message: str, *, code: str) -> None:
            super().__init__(message)
            self.code = code

    class SchemaSession:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, query: str, **parameters: Any) -> _Result:
            self.calls += 1
            if self.calls == 1:
                raise SchemaError(
                    "the human-readable message may change",
                    code=(
                        "Neo.ClientError.Schema."
                        "EquivalentSchemaRuleAlreadyExists"
                    ),
                )
            return _Result()

    session = SchemaSession()

    GRAPHITI_NEO4J_SCHEMA.ensure(session)

    assert session.calls == len(GRAPHITI_NEO4J_SCHEMA.queries) + 1


def test_neo4j_schema_does_not_trust_equivalent_error_message_text() -> None:
    class SchemaError(RuntimeError):
        code = "Neo.ClientError.Security.Forbidden"

    class SchemaSession:
        def run(self, query: str, **parameters: Any) -> _Result:
            raise SchemaError(
                "permission denied after EquivalentSchemaRuleAlreadyExists"
            )

    with pytest.raises(SchemaError, match="permission denied"):
        GRAPHITI_NEO4J_SCHEMA.ensure(SchemaSession())


def test_neo4j_schema_does_not_hide_other_creation_errors() -> None:
    class SchemaSession:
        def run(self, query: str, **parameters: Any) -> _Result:
            raise RuntimeError("permission denied")

    with pytest.raises(RuntimeError, match="permission denied"):
        GRAPHITI_NEO4J_SCHEMA.ensure(SchemaSession())


def test_storage_commit_round_trips_as_json_data() -> None:
    commit = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="lineage-a",
        commit_sequence=3,
        source_row_count=7,
    )

    assert StorageCommit.from_dict(commit.to_dict()) == commit
    assert json.loads(json.dumps(commit.to_dict())) == commit.to_dict()


class _Result:
    def __init__(self, record: dict[str, Any] | None = None) -> None:
        self._record = record

    def single(self, *, strict: bool = False) -> dict[str, Any] | None:
        if strict and self._record is None:
            raise ValueError("expected one record")
        return self._record

    def consume(self) -> None:
        return None


class _Neo4jTransaction:
    def __init__(self, database: "_Neo4jDatabase") -> None:
        self.database = database
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.pending_commit = database.commit

    def run(self, query: str, **parameters: Any) -> _Result:
        self.calls.append((query, dict(parameters)))
        if "RETURN m.plan_fingerprint AS plan_fingerprint" in query:
            return _Result(self.database.commit)
        if "AS marker_updated" in query:
            expected = parameters["expected"]
            actual = self.database.commit
            if expected != actual:
                return _Result({"marker_updated": False})
            self.pending_commit = parameters["next"]
            return _Result({"marker_updated": True})
        if "AS marker_locked" in query:
            expected = parameters["expected"]
            actual = self.database.commit
            return _Result({"marker_locked": expected == actual})
        if "MATCH (m:_AgentMemoryStorageCommit" in query and "DELETE m" in query:
            self.pending_commit = None
        if "REMOVE m.rebuild_lock" in query:
            self.pending_commit = parameters["next"]
        return _Result()


class _Neo4jSession:
    def __init__(self, database: "_Neo4jDatabase") -> None:
        self.database = database

    def __enter__(self) -> "_Neo4jSession":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def run(self, query: str, **parameters: Any) -> _Result:
        self.database.schema_calls.append((query, dict(parameters)))
        if "RETURN m.plan_fingerprint AS plan_fingerprint" in query:
            return _Result(self.database.commit)
        return _Result()

    def execute_write(self, callback: Any) -> Any:
        tx = _Neo4jTransaction(self.database)
        try:
            result = callback(tx)
        except Exception:
            self.database.rollbacks += 1
            raise
        self.database.commit = tx.pending_commit
        self.database.transactions.append(tx.calls)
        return result


class _Neo4jDatabase:
    def __init__(self) -> None:
        self.commit: dict[str, Any] | None = None
        self.schema_calls: list[tuple[str, dict[str, Any]]] = []
        self.transactions: list[list[tuple[str, dict[str, Any]]]] = []
        self.rollbacks = 0

    @contextmanager
    def session(self, *, database: str) -> Any:
        assert database == "neo4j"
        yield _Neo4jSession(self)


def _neo4j_entity_statement() -> InsertStatement:
    schema = _entity_schema()
    mapping = Neo4jNodeMapping(
        label="Entity",
        identity=Neo4jIdentity(columns=("entity_id",), kind="entity"),
        properties={"name": "name", "summary": "summary"},
        embedding=_embedding(),
    )
    target = (
        TableDescriptor.for_connector("neo4j")
        .schema(schema)
        .mapping(mapping)
        .build()
    )
    return InsertStatement(
        statement_id="sink_0000",
        target=target,
        query=QueryExpr(op="log"),
    )


def test_neo4j_transaction_materializes_embeddings_before_database_write() -> None:
    database = _Neo4jDatabase()
    provider = _EmbeddingProvider()
    connector = Neo4jConnector(
        driver=database,
        database="neo4j",
        embedding_provider=provider,
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    statement = _neo4j_entity_statement()
    next_commit = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="lineage-a",
        commit_sequence=1,
        source_row_count=1,
    )

    with connector.transaction(
        namespace="sample-0",
        expected_commit=None,
        next_commit=next_commit,
    ) as transaction:
        transaction.write(
            statement,
            inserted_rows=pd.DataFrame(
                [
                    {
                        "entity_id": (12, 0),
                        "name": "Alice",
                        "summary": "Alice likes running.",
                        "mentions": "[]",
                    }
                ]
            ),
            retracted_rows=pd.DataFrame(columns=[
                "entity_id",
                "name",
                "summary",
                "mentions",
            ]),
        )
        assert database.transactions == []

    assert provider.calls == [(_embedding(), ["Alice"])]
    assert database.commit == next_commit.to_dict()
    assert len(database.transactions) == 1
    assert "MERGE (n:Entity" in database.transactions[0][0][0]


def test_neo4j_prepare_creates_schema_idempotently() -> None:
    database = _Neo4jDatabase()
    connector = Neo4jConnector(
        driver=database,
        database="neo4j",
        embedding_provider=_EmbeddingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    statements = StatementSet(statements=(_neo4j_entity_statement(),))

    connector.prepare(statements)
    connector.prepare(statements)

    expected_calls = len(GRAPHITI_NEO4J_SCHEMA.queries) + 1
    assert len(database.schema_calls) == expected_calls * 2
    assert database.schema_calls[expected_calls - 1] == (
        "CALL db.awaitIndexes($timeout)",
        {"timeout": 300},
    )


def test_neo4j_marker_conflict_rolls_back_all_graph_writes() -> None:
    database = _Neo4jDatabase()
    database.commit = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="another-lineage",
        commit_sequence=3,
        source_row_count=3,
    ).to_dict()
    connector = Neo4jConnector(
        driver=database,
        database="neo4j",
        embedding_provider=_EmbeddingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    statement = _neo4j_entity_statement()

    with pytest.raises(StorageConflictError, match="storage commit marker"):
        with connector.transaction(
            namespace="sample-0",
            expected_commit=None,
            next_commit=StorageCommit(
                plan_fingerprint="plan-a",
                lineage_id="lineage-a",
                commit_sequence=1,
                source_row_count=1,
            ),
        ) as transaction:
            transaction.write(
                statement,
                inserted_rows=pd.DataFrame(
                    [
                        {
                            "entity_id": (12, 0),
                            "name": "Alice",
                            "summary": "Alice likes running.",
                            "mentions": "[]",
                        }
                    ]
                ),
                retracted_rows=pd.DataFrame(),
            )

    assert database.rollbacks == 1
    assert database.transactions == []
    assert database.commit["lineage_id"] == "another-lineage"


def test_neo4j_rebuild_replaces_namespace_and_marker_atomically() -> None:
    database = _Neo4jDatabase()
    current = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="lineage-a",
        commit_sequence=3,
        source_row_count=3,
    )
    checkpoint = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="lineage-a",
        commit_sequence=1,
        source_row_count=1,
    )
    database.commit = current.to_dict()
    connector = Neo4jConnector(
        driver=database,
        database="neo4j",
        embedding_provider=_EmbeddingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    statement = _neo4j_entity_statement()
    statements = StatementSet(statements=(statement,))

    connector.rebuild(
        namespace="sample-0",
        statements=statements,
        rows_by_statement={
            statement.statement_id: pd.DataFrame(
                [
                    {
                        "entity_id": (12, 0),
                        "name": "Alice",
                        "summary": "Alice likes running.",
                        "mentions": "[]",
                    }
                ]
            )
        },
        expected_commit=current,
        next_commit=checkpoint,
    )

    queries = [query for query, _ in database.transactions[0]]
    assert "AS marker_locked" in queries[0]
    assert "MATCH ()-[r]->()" in queries[1]
    assert "NOT n:_AgentMemoryStorageCommit" in queries[2]
    assert "MERGE (n:Entity" in queries[3]
    assert "REMOVE m.rebuild_lock" in queries[4]
    assert database.commit == checkpoint.to_dict()


def test_neo4j_empty_rebuild_removes_the_initial_marker() -> None:
    database = _Neo4jDatabase()
    current = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="lineage-a",
        commit_sequence=1,
        source_row_count=1,
    )
    initial = StorageCommit(
        plan_fingerprint="plan-a",
        lineage_id="lineage-a",
        commit_sequence=0,
        source_row_count=0,
    )
    database.commit = current.to_dict()
    connector = Neo4jConnector(
        driver=database,
        database="neo4j",
        embedding_provider=_EmbeddingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    statement = _neo4j_entity_statement()

    connector.rebuild(
        namespace="sample-0",
        statements=StatementSet(statements=(statement,)),
        rows_by_statement={
            statement.statement_id: pd.DataFrame(
                columns=["entity_id", "name", "summary", "mentions"]
            )
        },
        expected_commit=current,
        next_commit=initial,
    )

    queries = [query for query, _ in database.transactions[0]]
    assert "DELETE m" in queries[-1]
    assert database.commit is None


def test_embedding_provider_failure_happens_before_neo4j_transaction() -> None:
    class FailingProvider:
        def embed(
            self, spec: EmbeddingSpec, texts: list[str]
        ) -> list[list[float]]:
            raise RuntimeError("embedding failed")

    database = _Neo4jDatabase()
    connector = Neo4jConnector(
        driver=database,
        database="neo4j",
        embedding_provider=FailingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    statement = _neo4j_entity_statement()

    with pytest.raises(RuntimeError, match="embedding failed"):
        with connector.transaction(
            namespace="sample-0",
            expected_commit=None,
            next_commit=StorageCommit(
                plan_fingerprint="plan-a",
                lineage_id="lineage-a",
                commit_sequence=1,
                source_row_count=1,
            ),
        ) as transaction:
            transaction.write(
                statement,
                inserted_rows=pd.DataFrame(
                    [
                        {
                            "entity_id": (12, 0),
                            "name": "Alice",
                            "summary": "Alice likes running.",
                            "mentions": "[]",
                        }
                    ]
                ),
                retracted_rows=pd.DataFrame(),
            )

    assert database.transactions == []
    assert database.commit is None
