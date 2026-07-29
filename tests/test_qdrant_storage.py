"""Integration tests for the optional embedded Qdrant storage backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.sem_flat_map import apply_flat_map_outputs
from agent_memory.memories.mem0 import Mem0Memory, Mem0MemoryEnhanced
from agent_memory.policy.logical import QueryExpr, UserQuery
from agent_memory.policy.retrieval import (
    CosineSimilarity,
    RetrievalQuery,
    RetrievalResult,
)
from agent_memory.storage import (
    EmbeddingSpec,
    Schema,
    StatementSet,
    StorageCommit,
    StorageDeployment,
    TableDescriptor,
)
from agent_memory.storage.qdrant import (
    QdrantConnector,
    QdrantIdentity,
    QdrantPointMapping,
)
from agent_memory.storage.qdrant.schema import CONTROL_COLLECTION
from agent_memory.storage.qdrant.recovery import publish_marker


qdrant_client = pytest.importorskip("qdrant_client")
qdrant_models = qdrant_client.models


_TEST_EMBEDDING = EmbeddingSpec(
    source_column="memory",
    property_name="memory_embedding",
    model="test-embedding",
    revision="test-revision",
    dimensions=3,
    normalize=True,
)


class _EmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(
        self,
        spec: EmbeddingSpec,
        texts: list[str],
    ) -> list[list[float]]:
        assert spec == _TEST_EMBEDDING
        self.calls.append(list(texts))
        vectors = {
            "alpha": [1.0, 0.0, 0.0],
            "alpha query": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "beta query": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
            "gamma query": [0.0, 0.0, 1.0],
        }
        return [vectors[text] for text in texts]


class _Mem0ExtractionAdapter(LotusAdapter):
    """Extract each test message as one deterministic memory row."""

    def execute(self, query: QueryExpr, inputs: dict[str, Any]) -> Any:
        if query.op != "sem_flat_map":
            return super().execute(query, inputs)
        source = self.execute(query.inputs[0], inputs)
        outputs = [
            [
                {
                    "memory": str(row["content"]),
                    "attributed_to": str(row["role"]),
                }
            ]
            for _, row in source.iterrows()
        ]
        return apply_flat_map_outputs(
            source,
            outputs,
            tuple(query.params["output_cols"]),
        )


class _DenseMemory(am.Memory):
    log = am.Log(
        {
            "memory": "Memory text.",
            "attributed_to": "Source role.",
        }
    )
    memories = log.select(["memory", "attributed_to"])
    retrieval_query = RetrievalQuery(
        memories=memories.search(
            UserQuery(),
            methods=[
                CosineSimilarity(candidate_limit=4, min_score=0.1),
            ],
            reranker=None,
            limit=2,
        ).select(
            [
                "record_id",
                "memory",
                "attributed_to",
                "rank",
                "score",
            ]
        )
    )


def _memory_schema(*, primary_key: str = "memory") -> Schema:
    return (
        Schema.new_builder()
        .column("memory", "STRING")
        .column("attributed_to", "STRING")
        .primary_key(primary_key)
        .build()
    )


def _mapping(*, identity_column: str = "memory") -> QdrantPointMapping:
    return QdrantPointMapping(
        collection="test_memories",
        identity=QdrantIdentity(
            columns=(identity_column,),
            kind="memory",
        ),
        properties={
            "memory": "memory",
            "attributed_to": "attributed_to",
        },
        embedding=_TEST_EMBEDDING,
    )


def _statements() -> StatementSet:
    target = (
        TableDescriptor.for_connector("qdrant")
        .schema(_memory_schema())
        .mapping(_mapping())
        .build()
    )
    return StatementSet().add_insert(target, _DenseMemory.memories)


def _mem0_statements(memory_type: type[am.Memory]) -> StatementSet:
    target = (
        TableDescriptor.for_connector("qdrant")
        .schema(_memory_schema())
        .mapping(_mapping())
        .build()
    )
    return StatementSet().add_insert(target, memory_type.memories)


def _stored_mem0(
    path: Path,
    memory_type: type[am.Memory],
) -> tuple[am.Memory, QdrantConnector]:
    connector = QdrantConnector(
        path=path,
        embedding_provider=_EmbeddingProvider(),
    )
    memory = memory_type(
        adapter=_Mem0ExtractionAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=_mem0_statements(memory_type),
            namespace="shared-mem0-namespace",
        ),
    )
    return memory, connector


def _stored_memory(
    path: Path,
    *,
    provider: _EmbeddingProvider | None = None,
    client: Any | None = None,
) -> tuple[_DenseMemory, QdrantConnector, _EmbeddingProvider]:
    embedding_provider = provider or _EmbeddingProvider()
    connector = QdrantConnector(
        path=path,
        embedding_provider=embedding_provider,
        client=client,
        models=qdrant_models if client is not None else None,
    )
    memory = _DenseMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=_statements(),
            namespace="test-namespace",
        ),
    )
    return memory, connector, embedding_provider


def _records(result: RetrievalResult) -> list[dict[str, Any]]:
    return result.channels["memories"].to_dict("records")


def test_qdrant_mapping_is_typed_serializable_and_schema_checked() -> None:
    mapping = _mapping()
    target = (
        TableDescriptor.for_connector("qdrant")
        .schema(_memory_schema())
        .mapping(mapping)
        .build()
    )

    assert json.loads(json.dumps(target.to_dict()))["mapping"] == mapping.to_dict()
    assert mapping.to_dict()["embedding"]["model"] == "test-embedding"

    with pytest.raises(ValueError, match="primary key"):
        (
            TableDescriptor.for_connector("qdrant")
            .schema(_memory_schema(primary_key="attributed_to"))
            .mapping(mapping)
            .build()
        )


def test_embedded_qdrant_add_search_checkpoint_reopen_and_continue(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qdrant"
    memory, connector, provider = _stored_memory(path)
    memory.add({"memory": "alpha", "attributed_to": "user"})
    memory.add({"memory": "beta", "attributed_to": "assistant"})

    before = memory.query("alpha query")
    snapshot = memory._runtime.snapshot_state()

    assert isinstance(before, RetrievalResult)
    assert [row["memory"] for row in _records(before)] == ["alpha"]
    assert _records(before)[0]["rank"] == 1
    assert before.metrics["memories"]["candidate_limit"] == 4
    assert before.metrics["memories"]["result_ids"] == [
        _records(before)[0]["record_id"]
    ]
    assert provider.calls == [
        ["alpha"],
        ["beta"],
        ["alpha query"],
    ]
    connector.close()

    restored, reopened, restored_provider = _stored_memory(path)
    restored._runtime.restore_state(snapshot)
    after = restored.query("alpha query")
    restored.add({"memory": "gamma", "attributed_to": "user"})
    continued = restored.query("gamma query")

    assert _records(after) == _records(before)
    assert [row["memory"] for row in _records(continued)] == ["gamma"]
    assert restored_provider.calls == [
        ["alpha query"],
        ["gamma"],
        ["gamma query"],
    ]
    reopened.close()


def test_mem0_enhanced_storage_checkpoint_restores_into_base_vector_retrieval(
    tmp_path: Path,
) -> None:
    enhanced, enhanced_connector = _stored_mem0(
        tmp_path / "enhanced-source",
        Mem0MemoryEnhanced,
    )
    enhanced.add(
        {
            "role": "user",
            "content": "alpha",
            "observation_date": "2026-07-28T10:00:00",
        }
    )
    snapshot = enhanced._runtime.snapshot_state()
    enhanced_fingerprint = snapshot["plan_fingerprint"]
    enhanced_connector.close()

    base, base_connector = _stored_mem0(tmp_path / "base-target", Mem0Memory)
    base._runtime.restore_state(snapshot)
    result = base.query("alpha query")

    assert base._runtime.policy.fingerprint == enhanced_fingerprint
    assert [row["memory"] for row in _records(result)] == ["alpha"]
    assert len(base._runtime._state["log"]) == 1
    base_connector.close()


def test_unstored_mem0_enhanced_checkpoint_cannot_restore_into_stored_base(
    tmp_path: Path,
) -> None:
    enhanced = Mem0MemoryEnhanced(adapter=_Mem0ExtractionAdapter())
    enhanced.add(
        {
            "role": "user",
            "content": "alpha",
            "observation_date": "2026-07-28T10:00:00",
        }
    )
    snapshot = enhanced._runtime.snapshot_state()
    base, connector = _stored_mem0(tmp_path / "base", Mem0Memory)

    with pytest.raises(ValueError, match="fingerprint"):
        base._runtime.restore_state(snapshot)

    connector.close()


@pytest.mark.parametrize("marker_state", ["missing", "behind", "ahead"])
def test_checkpoint_restore_rebuilds_a_divergent_qdrant_namespace(
    tmp_path: Path,
    marker_state: str,
) -> None:
    source, source_connector, _ = _stored_memory(tmp_path / "source")
    source.add({"memory": "alpha", "attributed_to": "user"})
    source.add({"memory": "beta", "attributed_to": "assistant"})
    snapshot = source._runtime.snapshot_state()
    checkpoint_commit = StorageCommit.from_dict(snapshot["storage_commit"])
    source_connector.close()

    restored, target_connector, _ = _stored_memory(tmp_path / "target")
    if marker_state != "missing":
        offset = -1 if marker_state == "behind" else 1
        divergent_commit = StorageCommit(
            plan_fingerprint=checkpoint_commit.plan_fingerprint,
            lineage_id=checkpoint_commit.lineage_id,
            commit_sequence=checkpoint_commit.commit_sequence + offset,
            source_row_count=checkpoint_commit.source_row_count + offset,
        )
        publish_marker(
            target_connector._client,
            target_connector._models,
            namespace="test-namespace",
            expected_commit=None,
            next_commit=divergent_commit,
            materialization_id=f"{marker_state}-materialization",
        )

    restored._runtime.restore_state(snapshot)
    result = restored.query("alpha query")

    assert target_connector.read_commit(
        namespace="test-namespace"
    ) == checkpoint_commit
    assert [row["memory"] for row in _records(result)] == ["alpha"]
    target_connector.close()


def test_same_key_replacement_closes_the_old_point_version(tmp_path: Path) -> None:
    class ReplacedMemory(am.Memory):
        log = am.Log(
            {
                "key": "Stable key.",
                "memory": "Memory text.",
                "attributed_to": "Source role.",
            }
        )
        memories = (
            log.group_by("key")
            .min(column="memory", output_col="memory")
            .assign(attributed_to="user")
            .select(["key", "memory", "attributed_to"])
        )

    schema = (
        Schema.new_builder()
        .column("key", "STRING")
        .column("memory", "STRING")
        .column("attributed_to", "STRING")
        .primary_key("key")
        .build()
    )
    mapping = QdrantPointMapping(
        collection="replaced_memories",
        identity=QdrantIdentity(columns=("key",), kind="memory"),
        properties={
            "key": "key",
            "memory": "memory",
            "attributed_to": "attributed_to",
        },
        embedding=_TEST_EMBEDDING,
    )
    target = (
        TableDescriptor.for_connector("qdrant")
        .schema(schema)
        .mapping(mapping)
        .build()
    )
    statements = StatementSet().add_insert(target, ReplacedMemory.memories)
    provider = _EmbeddingProvider()
    connector = QdrantConnector(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
    )
    memory = ReplacedMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=statements,
            namespace="replacement",
        ),
    )

    memory.add({"key": "stable", "memory": "beta", "attributed_to": "user"})
    memory.add({"key": "stable", "memory": "alpha", "attributed_to": "user"})

    statement = statements.statements[0]
    active, _ = connector._client.scroll(
        collection_name="replaced_memories",
        scroll_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="_agent_memory_namespace",
                    match=qdrant_models.MatchValue(value="replacement"),
                ),
                qdrant_models.FieldCondition(
                    key="_agent_memory_statement_id",
                    match=qdrant_models.MatchValue(value=statement.statement_id),
                ),
                qdrant_models.FieldCondition(
                    key="_agent_memory_visible_until",
                    range=qdrant_models.Range(gt=2),
                ),
            ]
        ),
        with_payload=True,
        with_vectors=False,
    )

    assert len(active) == 1
    assert active[0].payload["memory"] == "alpha"
    connector.close()


class _FailingMarkerClient:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.fail_marker = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def upsert(self, *, collection_name: str, **kwargs: Any) -> Any:
        if self.fail_marker and collection_name == CONTROL_COLLECTION:
            raise RuntimeError("marker publication failed")
        return self.client.upsert(collection_name=collection_name, **kwargs)


def test_marker_failure_leaves_half_written_points_invisible(
    tmp_path: Path,
) -> None:
    raw_client = qdrant_client.QdrantClient(path=str(tmp_path / "qdrant"))
    failing_client = _FailingMarkerClient(raw_client)
    memory, connector, _ = _stored_memory(
        tmp_path / "qdrant",
        client=failing_client,
    )

    with pytest.raises(RuntimeError, match="marker publication failed"):
        memory.add({"memory": "alpha", "attributed_to": "user"})

    assert connector.read_commit(namespace="test-namespace") is None
    result = memory.query("alpha query")
    assert isinstance(result, RetrievalResult)
    assert result.channels["memories"].empty
    assert memory._runtime._state == {}

    failing_client.fail_marker = False
    memory.add({"memory": "alpha", "attributed_to": "user"})
    assert [row["memory"] for row in _records(memory.query("alpha query"))] == [
        "alpha"
    ]
    connector.close()
