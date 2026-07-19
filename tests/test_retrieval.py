"""Tests for declarative retrieval queries and search relations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from contextlib import contextmanager
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.planner import PolicyDifferentiator
from agent_memory.planner.retrieval import RetrievalPlan
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.retrieval import (
    BFS,
    BM25,
    RRF,
    CosineSimilarity,
    CrossEncoder,
    RetrievalQuery,
    SearchRelation,
)
from agent_memory.policy.schema import output_columns
from agent_memory.storage import Schema, StatementSet, StorageDeployment, TableDescriptor
from agent_memory.storage.search import SearchBatch, SearchRequest


class SearchMemory(am.Memory):
    """Small two-channel policy used by retrieval planner/runtime tests."""

    log = am.Log({"name": "Entity name.", "fact": "Fact text."})
    entities = log.select(["name"])
    facts = log.select(["fact"])
    _retrieved_entities = entities.search(
        am.UserQuery(),
        methods=[BM25(), CosineSimilarity()],
        reranker=RRF(),
        limit=20,
    ).select(["record_id", "name", "rank", "score"])
    retrieval_query = RetrievalQuery(
        entities=_retrieved_entities,
        facts=facts.search(
            am.UserQuery(),
            methods=[BM25(), BFS(origins=_retrieved_entities, max_depth=3)],
            reranker=CrossEncoder(model="BAAI/bge-reranker-v2-m3"),
            limit=20,
        ).select(["record_id", "fact", "rank", "score"]),
    )


def _entity_search(source: am.Log) -> SearchRelation:
    return source.search(
        am.UserQuery(),
        methods=[BM25(), CosineSimilarity()],
        reranker=RRF(),
        limit=20,
    )


def test_search_builds_retrieval_scoped_relation_with_rank_metadata() -> None:
    source = am.Log({"name": "Entity name.", "summary": "Entity summary."})

    result = _entity_search(source)

    assert isinstance(result, SearchRelation)
    assert result.expr.op == "search"
    assert output_columns(result.expr) == (
        "name",
        "summary",
        "record_id",
        "rank",
        "score",
    )
    assert result.expr.params["limit"] == 20
    assert tuple(method.kind for method in result.expr.params["methods"]) == (
        "bm25",
        "cosine_similarity",
    )
    assert result.expr.params["reranker"].kind == "rrf"


@pytest.mark.parametrize("column", ["record_id", "rank", "score"])
def test_search_schema_rejects_metadata_collisions_in_direct_ir(column: str) -> None:
    """Schema validation must protect QueryExpr paths that bypass Relation.search."""

    source = QueryExpr(
        op="materialized_view",
        params={"name": "entities", "columns": ("name", column)},
    )
    search = QueryExpr(op="search", inputs=(source,))

    with pytest.raises(
        ValueError,
        match=rf"search metadata columns conflict with source columns: \['{column}'\]",
    ):
        output_columns(search)


def test_search_relation_preserves_scope_through_relational_projection() -> None:
    source = am.Log({"name": "Entity name.", "summary": "Entity summary."})

    projected = _entity_search(source).select(
        ["record_id", "name", "summary", "rank", "score"]
    )

    assert isinstance(projected, SearchRelation)
    assert projected.expr.op == "select"
    assert output_columns(projected.expr) == (
        "record_id",
        "name",
        "summary",
        "rank",
        "score",
    )


def test_bfs_uses_a_real_search_relation_as_an_explicit_dependency() -> None:
    entities = am.Log({"name": "Entity name.", "summary": "Summary."})
    facts = am.Log({"fact": "Fact text."})
    entity_hits = _entity_search(entities).select(
        ["record_id", "name", "summary", "rank", "score"]
    )

    fact_hits = facts.search(
        am.UserQuery(),
        methods=[BM25(), CosineSimilarity(), BFS(origins=entity_hits, max_depth=3)],
        reranker=CrossEncoder(model="BAAI/bge-reranker-v2-m3"),
        limit=20,
    )

    assert fact_hits.expr.op == "search"
    assert fact_hits.expr.inputs == (facts.expr, entity_hits.expr)
    bfs = fact_hits.expr.params["methods"][2]
    assert bfs.kind == "bfs"
    assert bfs.params == {"origin_input": 1, "max_depth": 3}


def test_bfs_rejects_string_or_maintenance_relation_origins() -> None:
    source = am.Log({"name": "Entity name."})

    with pytest.raises(TypeError, match="SearchRelation"):
        BFS(origins="entities", max_depth=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SearchRelation"):
        BFS(origins=source, max_depth=3)  # type: ignore[arg-type]


def test_retrieval_descriptors_are_frozen_and_validate_arguments() -> None:
    method = BM25()

    with pytest.raises(FrozenInstanceError):
        method.kind = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-empty"):
        CrossEncoder(model="")
    with pytest.raises(ValueError, match="positive"):
        BFS(origins=_entity_search(am.Log({"name": "Name."})), max_depth=0)


def test_retrieval_query_is_the_only_collected_retrieval_root() -> None:
    spec = SearchMemory.spec()

    assert tuple(spec.views) == ("entities", "facts")
    assert spec.private_relations == {}
    assert tuple(spec.retrieval_queries) == ("default",)
    retrieval = spec.retrieval_queries["default"]
    assert isinstance(retrieval, RetrievalQuery)
    assert tuple(retrieval.channels) == ("entities", "facts")


def test_public_search_relation_must_be_declared_as_retrieval_query() -> None:
    class InvalidMemory(am.Memory):
        log = am.Log({"name": "Name."})
        search_results = _entity_search(log)

    with pytest.raises(TypeError, match="retrieval_query"):
        InvalidMemory.spec()


def test_search_cannot_be_bound_as_a_storage_sink() -> None:
    target = _target("name", "record_id", "rank", "score")
    statements = StatementSet().add_insert(
        target,
        SearchMemory.entities.search(
            am.UserQuery(),
            methods=[BM25()],
            reranker=RRF(),
            limit=5,
        ),
    )

    with pytest.raises(NotImplementedError, match="retrieval-only"):
        PolicyDifferentiator().differentiate(
            SearchMemory.spec(),
            statements=statements,
        )


def _target(*columns: str) -> TableDescriptor:
    builder = Schema.new_builder()
    for column in columns:
        builder.column(column, "STRING")
    return (
        TableDescriptor.for_connector("recording")
        .schema(builder.primary_key(columns[0]).build())
        .build()
    )


def _statements() -> StatementSet:
    return (
        StatementSet()
        .add_insert(_target("name"), SearchMemory.entities)
        .add_insert(_target("fact"), SearchMemory.facts)
    )


def test_retrieval_planner_builds_one_shared_topological_dag() -> None:
    policy = PolicyDifferentiator().differentiate(
        SearchMemory.spec(),
        statements=_statements(),
    )

    retrieval = policy.retrieval_queries["default"]
    assert isinstance(retrieval, RetrievalPlan)
    assert tuple(retrieval.channel_outputs) == ("entities", "facts")
    assert len(retrieval.nodes) == 4
    assert len(retrieval.execution_order) == 4
    search_nodes = [
        node for node in retrieval.nodes.values() if node.execution_kind == "search"
    ]
    assert len(search_nodes) == 2
    entity_search = next(
        node for node in search_nodes if node.statement_id == "sink_0000"
    )
    fact_search = next(
        node for node in search_nodes if node.statement_id == "sink_0001"
    )
    entity_output = retrieval.channel_outputs["entities"]
    assert fact_search.input_node_ids == (entity_output,)
    assert retrieval.execution_order.index(entity_search.node_id) < retrieval.execution_order.index(
        fact_search.node_id
    )
    assert entity_search.required_columns == (
        "record_id",
        "name",
        "rank",
        "score",
    )
    assert fact_search.required_columns == (
        "record_id",
        "fact",
        "rank",
        "score",
    )


def test_retrieval_projection_pushdown_tracks_filter_and_assign_inputs() -> None:
    source = am.Log({"name": "Entity name.", "summary": "Summary."})
    hits = source.search(
        am.UserQuery(),
        methods=[BM25()],
        reranker=RRF(),
        limit=10,
    )
    filtered = hits.filter(hits.col("score") > 0.5)
    projected = filtered.assign(display_name=filtered.col("name")).select(
        ["record_id", "display_name"]
    )
    plan = PolicyDifferentiator().differentiate(
        type(
            "FilteredSearchMemory",
            (am.Memory,),
            {
                "log": source,
                "entities": source.select(["name", "summary"]),
                "retrieval_query": RetrievalQuery(entities=projected),
            },
        ).spec()
    ).retrieval_queries["default"]

    assert isinstance(plan, RetrievalPlan)
    search = next(node for node in plan.nodes.values() if node.execution_kind == "search")
    assert search.required_columns == ("record_id", "name", "score")


def test_retrieval_planner_rejects_unsupported_post_search_operator() -> None:
    source = am.Log({"name": "Entity name."})
    hits = source.search(
        am.UserQuery(),
        methods=[BM25()],
        reranker=RRF(),
        limit=10,
    )

    class JoinedSearchMemory(am.Memory):
        log = source
        entities = source.select(["name"])
        retrieval_query = RetrievalQuery(
            entities=hits.join(hits.alias("replica"), on="record_id")
        )

    with pytest.raises(NotImplementedError, match="post-search operator 'join'"):
        PolicyDifferentiator().differentiate(JoinedSearchMemory.spec())


def test_retrieval_plan_has_a_separate_fingerprint_from_maintenance() -> None:
    without_storage = SearchMemory.differentiate_policy()
    with_storage = PolicyDifferentiator().differentiate(
        SearchMemory.spec(),
        statements=_statements(),
    )

    assert without_storage.fingerprint != ""
    assert with_storage.fingerprint != ""
    assert isinstance(with_storage.retrieval_queries["default"], RetrievalPlan)
    assert with_storage.retrieval_queries["default"].fingerprint


def test_retrieval_planner_rejects_missing_or_ambiguous_sink_binding() -> None:
    with pytest.raises(ValueError, match="exactly one storage sink"):
        PolicyDifferentiator().differentiate(
            SearchMemory.spec(),
            statements=StatementSet().add_insert(_target("name"), SearchMemory.entities),
        )

    duplicated = _statements().add_insert(_target("name"), SearchMemory.entities)
    with pytest.raises(ValueError, match="exactly one storage sink"):
        PolicyDifferentiator().differentiate(
            SearchMemory.spec(),
            statements=duplicated,
        )


class RecordingSearchConnector:
    """Storage/search connector that records physical retrieval requests."""

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def prepare(self, statements: StatementSet) -> None:
        self.statements = statements

    def read_commit(self, *, namespace: str) -> None:
        return None

    @contextmanager
    def transaction(self, **kwargs: Any):
        raise AssertionError("retrieval test must not open a write transaction")
        yield

    def rebuild(self, **kwargs: Any) -> None:
        raise AssertionError("retrieval test must not rebuild storage")

    def search(self, request: SearchRequest) -> SearchBatch:
        self.requests.append(request)
        if request.statement_id == "sink_0000":
            return SearchBatch(
                rows=pd.DataFrame(
                    [
                        {
                            "record_id": "entity-1",
                            "name": "Alice",
                            "rank": 1,
                            "score": 0.8,
                        }
                    ]
                ),
                metrics={"methods": [{"kind": "bm25"}]},
            )
        assert request.origin_record_ids == ("entity-1",)
        return SearchBatch(
            rows=pd.DataFrame(
                [
                    {
                        "record_id": "fact-1",
                        "fact": "Alice lives in Paris",
                        "rank": 1,
                        "score": 0.9,
                    }
                ]
            ),
            metrics={
                "methods": [{"kind": "bfs"}],
                "bfs_origins": ["entity-1"],
            },
        )


def test_runtime_executes_shared_search_once_and_passes_bfs_origins() -> None:
    connector = RecordingSearchConnector()
    memory = SearchMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=_statements(),
            namespace="retrieval-test",
        ),
    )

    result = memory.query("Where does Alice live?")

    assert result.query == "Where does Alice live?"
    assert tuple(result.channels) == ("entities", "facts")
    assert result.channels["entities"].to_dict("records") == [
        {"record_id": "entity-1", "name": "Alice", "rank": 1, "score": 0.8}
    ]
    assert result.channels["facts"].iloc[0]["record_id"] == "fact-1"
    assert len(connector.requests) == 2
    assert connector.requests[0].origin_record_ids == ()
    assert connector.requests[1].origin_record_ids == ("entity-1",)
    assert all(request.query == "Where does Alice live?" for request in connector.requests)
    assert result.metrics["entities"]["methods"] == [{"kind": "bm25"}]
    assert result.metrics["facts"]["bfs_origins"] == ["entity-1"]


def test_storage_backed_retrieval_requires_a_search_capable_backend() -> None:
    class SinkOnlyConnector(RecordingSearchConnector):
        search = None  # type: ignore[assignment]

    memory = SearchMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=SinkOnlyConnector(),
            statements=_statements(),
            namespace="retrieval-test",
        ),
    )

    with pytest.raises(NotImplementedError, match="retrieval-capable"):
        memory.query("Alice")


def test_retrieval_query_without_storage_has_no_scan_fallback() -> None:
    memory = SearchMemory(adapter=LotusAdapter())

    with pytest.raises(NotImplementedError, match="storage backend"):
        memory.query("Alice")
