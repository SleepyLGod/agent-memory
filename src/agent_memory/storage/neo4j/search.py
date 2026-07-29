"""Graphiti-compatible Neo4j lowering for backend-neutral search requests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from time import perf_counter
from typing import Any, cast

import pandas as pd

from agent_memory.policy.retrieval import SearchMethodSpec
from agent_memory.storage.search import (
    CrossEncoderProvider,
    SearchBatch,
    SearchRequest,
)

from .mapping import (
    Neo4jNodeMapping,
    Neo4jRelationshipMapping,
)
from .schema import Neo4jSchema
from .sink import EmbeddingProvider


def execute_search(
    session: Any,
    request: SearchRequest,
    *,
    schema: Neo4jSchema,
    embedding_provider: EmbeddingProvider | None,
    reranker_provider: CrossEncoderProvider | None,
) -> SearchBatch:
    """Execute one mapped search recipe and return logical result rows."""

    mapping = request.target.mapping
    if not isinstance(mapping, (Neo4jNodeMapping, Neo4jRelationshipMapping)):
        raise TypeError("Neo4j search requires a typed Neo4j mapping")
    if request.target.connector != "neo4j":
        raise ValueError("Neo4j search only accepts neo4j table targets")
    readable_properties = _readable_properties(mapping)
    mapping.validate_search_columns(request.output_columns)

    started = perf_counter()
    query_vector = _query_vector(
        request,
        mapping=mapping,
        embedding_provider=embedding_provider,
    )
    # Methods produce candidates; an optional reranker may replace their order.
    rankings: list[list[dict[str, Any]]] = []
    method_metrics: list[dict[str, Any]] = []
    for method in request.methods:
        method_started = perf_counter()
        candidate_limit = _candidate_limit(method, default=request.limit * 2)
        rows = _execute_method(
            session,
            request=request,
            mapping=mapping,
            schema=schema,
            method=method,
            query_vector=query_vector,
            candidate_limit=candidate_limit,
        )
        rankings.append(rows)
        method_metrics.append(
            {
                "kind": method.kind,
                "candidate_ids": [str(row["record_id"]) for row in rows],
                "latency_ms": (perf_counter() - method_started) * 1000,
            }
        )

    candidates = {
        str(row["record_id"]): row
        for ranking in rankings
        for row in ranking
    }
    ordered_ids, scores = _rerank(
        request,
        rankings=rankings,
        candidates=candidates,
        mapping=mapping,
        readable_properties=readable_properties,
        reranker_provider=reranker_provider,
    )
    records = [
        _logical_result(
            candidates[record_id],
            mapping=readable_properties,
            output_columns=request.output_columns,
            rank=rank,
            score=scores[index],
        )
        for index, (rank, record_id) in enumerate(
            zip(range(1, request.limit + 1), ordered_ids[: request.limit], strict=False)
        )
    ]
    reranker_metrics = (
        None
        if request.reranker is None
        else {
            "kind": request.reranker.kind,
            "input_candidate_ids": list(candidates),
            "candidate_ids": ordered_ids[: request.limit],
            "scores": scores[: request.limit],
        }
    )
    metrics = {
        "methods": method_metrics,
        "bfs_origins": list(request.origin_record_ids),
        "reranker": reranker_metrics,
        "latency_ms": (perf_counter() - started) * 1000,
    }
    return SearchBatch(
        rows=pd.DataFrame.from_records(records, columns=request.output_columns),
        metrics=metrics,
    )


def _query_vector(
    request: SearchRequest,
    *,
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    embedding_provider: EmbeddingProvider | None,
) -> list[float] | None:
    if not any(method.kind == "cosine_similarity" for method in request.methods):
        return None
    if mapping.embedding is None:
        raise ValueError("cosine search requires an embedding mapping")
    if embedding_provider is None:
        raise ValueError("cosine search requires an embedding provider")
    vectors = embedding_provider.embed(mapping.embedding, [request.query])
    if len(vectors) != 1 or len(vectors[0]) != mapping.embedding.dimensions:
        raise ValueError("embedding provider returned an invalid query vector")
    return [float(value) for value in vectors[0]]


def _execute_method(
    session: Any,
    *,
    request: SearchRequest,
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    schema: Neo4jSchema,
    method: SearchMethodSpec,
    query_vector: list[float] | None,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    if method.kind == "bm25":
        query, parameters = _fulltext_query(
            request,
            mapping=mapping,
            schema=schema,
            candidate_limit=candidate_limit,
        )
    elif method.kind == "cosine_similarity":
        if query_vector is None:
            raise RuntimeError("cosine search query vector was not prepared")
        query, parameters = _cosine_query(
            request,
            mapping=mapping,
            method=method,
            query_vector=query_vector,
            candidate_limit=candidate_limit,
        )
    elif method.kind == "bfs":
        query, parameters = _bfs_query(
            request,
            mapping=mapping,
            schema=schema,
            method=method,
            candidate_limit=candidate_limit,
        )
    else:
        raise ValueError(f"Neo4j search does not support method {method.kind!r}")
    result = session.run(query, **parameters)
    data = getattr(result, "data", None)
    rows = data() if callable(data) else [dict(record) for record in result]
    return [
        dict(row)
        for row in cast(Iterable[Mapping[str, Any]], rows)
    ]


def _candidate_limit(method: SearchMethodSpec, *, default: int) -> int:
    value = method.params.get("candidate_limit", default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("search candidate_limit must be a positive integer")
    return value


def _fulltext_query(
    request: SearchRequest,
    *,
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    schema: Neo4jSchema,
    candidate_limit: int,
) -> tuple[str, dict[str, Any]]:
    if isinstance(mapping, Neo4jNodeMapping):
        index = schema.fulltext_index(kind="node", graph_type=mapping.label)
        query = f"""
        CALL db.index.fulltext.queryNodes("{index}", $fulltext_query, {{limit: $candidate_limit}})
        YIELD node AS record, score AS method_score
        WHERE record.group_id = $namespace
        RETURN record.uuid AS record_id, properties(record) AS properties, method_score
        ORDER BY method_score DESC
        LIMIT $candidate_limit
        """
    else:
        index = schema.fulltext_index(
            kind="relationship",
            graph_type=mapping.relationship_type,
        )
        query = f"""
        CALL db.index.fulltext.queryRelationships("{index}", $fulltext_query, {{limit: $candidate_limit}})
        YIELD relationship AS record, score AS method_score
        WHERE record.group_id = $namespace AND type(record) = $graph_type
        RETURN record.uuid AS record_id, properties(record) AS properties, method_score
        ORDER BY method_score DESC
        LIMIT $candidate_limit
        """
    return query, {
        "fulltext_query": (
            f'group_id:"{_lucene_escape(request.namespace)}" AND '
            f"({_lucene_escape(request.query)})"
        ),
        "namespace": request.namespace,
        "graph_type": _graph_type(mapping),
        "candidate_limit": candidate_limit,
    }


def _cosine_query(
    request: SearchRequest,
    *,
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    method: SearchMethodSpec,
    query_vector: list[float],
    candidate_limit: int,
) -> tuple[str, dict[str, Any]]:
    if mapping.embedding is None:
        raise ValueError("cosine search requires an embedding mapping")
    embedding_property = mapping.embedding.property_name
    minimum_score = _minimum_score(method, default=0.6)
    score_operator = ">=" if "min_score" in method.params else ">"
    if isinstance(mapping, Neo4jNodeMapping):
        query = f"""
        MATCH (record:{mapping.label})
        WHERE record.group_id = $namespace AND record.{embedding_property} IS NOT NULL
        WITH record, vector.similarity.cosine(record.{embedding_property}, $query_vector) AS method_score
        WHERE method_score {score_operator} $minimum_score
        RETURN record.uuid AS record_id, properties(record) AS properties, method_score
        ORDER BY method_score DESC
        LIMIT $candidate_limit
        """
    else:
        query = f"""
        MATCH ()-[record:{mapping.relationship_type}]->()
        WHERE record.group_id = $namespace AND record.{embedding_property} IS NOT NULL
        WITH record, vector.similarity.cosine(record.{embedding_property}, $query_vector) AS method_score
        WHERE method_score {score_operator} $minimum_score
        RETURN record.uuid AS record_id, properties(record) AS properties, method_score
        ORDER BY method_score DESC
        LIMIT $candidate_limit
        """
    return query, {
        "namespace": request.namespace,
        "query_vector": query_vector,
        "minimum_score": minimum_score,
        "candidate_limit": candidate_limit,
    }


def _minimum_score(method: SearchMethodSpec, *, default: float) -> float:
    value = method.params.get("min_score", default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cosine min_score must be a number")
    normalized = float(value)
    if not isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise ValueError("cosine min_score must be between -1 and 1")
    return normalized


def _bfs_query(
    request: SearchRequest,
    *,
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    schema: Neo4jSchema,
    method: SearchMethodSpec,
    candidate_limit: int,
) -> tuple[str, dict[str, Any]]:
    max_depth = int(method.params["max_depth"])
    if not schema.traversal_relationship_types:
        raise ValueError("Neo4j BFS requires configured traversal relationship types")
    traversal = "|".join(schema.traversal_relationship_types)
    # Origins are physical IDs, while every traversed object remains namespace-scoped.
    if isinstance(mapping, Neo4jNodeMapping):
        query = f"""
        UNWIND $origin_record_ids AS origin_id
        MATCH (origin {{uuid: origin_id}})-[:{traversal}*1..{max_depth}]->(record:{mapping.label})
        WHERE origin.group_id = $namespace AND record.group_id = $namespace
        RETURN DISTINCT record.uuid AS record_id, properties(record) AS properties, 1.0 AS method_score
        LIMIT $candidate_limit
        """
    else:
        query = f"""
        UNWIND $origin_record_ids AS origin_id
        MATCH path = (origin {{uuid: origin_id}})-[:{traversal}*1..{max_depth}]->(:Entity)
        WHERE origin.group_id = $namespace
        UNWIND relationships(path) AS traversed
        MATCH ()-[record:{mapping.relationship_type} {{uuid: traversed.uuid}}]-()
        WHERE record.group_id = $namespace
        RETURN DISTINCT record.uuid AS record_id, properties(record) AS properties, 1.0 AS method_score
        LIMIT $candidate_limit
        """
    return query, {
        "origin_record_ids": list(request.origin_record_ids),
        "namespace": request.namespace,
        "candidate_limit": candidate_limit,
    }


def _rerank(
    request: SearchRequest,
    *,
    rankings: list[list[dict[str, Any]]],
    candidates: Mapping[str, dict[str, Any]],
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    readable_properties: Mapping[str, str],
    reranker_provider: CrossEncoderProvider | None,
) -> tuple[list[str], list[float]]:
    if request.reranker is None:
        if len(rankings) != 1:
            raise ValueError(
                "Neo4j search without a reranker requires exactly one method"
            )
        return (
            [str(row["record_id"]) for row in rankings[0]],
            [float(row["method_score"]) for row in rankings[0]],
        )
    if request.reranker.kind == "rrf":
        return _rrf(
            [[str(row["record_id"]) for row in ranking] for ranking in rankings]
        )
    if request.reranker.kind != "cross_encoder":
        raise ValueError(
            f"Neo4j search does not support reranker {request.reranker.kind!r}"
        )
    if reranker_provider is None:
        raise ValueError("cross-encoder search requires a reranker provider")
    if mapping.embedding is None:
        raise ValueError("cross-encoder search requires an embedding source column")
    source_column = mapping.embedding.source_column
    if source_column not in readable_properties:
        raise ValueError(
            f"cross-encoder source column {source_column!r} is not readable"
        )
    property_name = readable_properties[source_column]
    candidate_ids = list(candidates)
    passages: list[str] = []
    for record_id in candidate_ids:
        passage = candidates[record_id]["properties"].get(property_name)
        if not isinstance(passage, str):
            raise TypeError("cross-encoder passages must be strings")
        passages.append(passage)
    ranked = reranker_provider.rank(
        model=str(request.reranker.params["model"]),
        query=request.query,
        passages=passages,
    )
    ranked_indexes = [index for index, _ in ranked]
    if (
        len(ranked_indexes) != len(candidate_ids)
        or len(set(ranked_indexes)) != len(candidate_ids)
        or any(index < 0 or index >= len(candidate_ids) for index in ranked_indexes)
    ):
        raise ValueError("cross-encoder must rank every candidate index exactly once")
    return (
        [candidate_ids[index] for index in ranked_indexes],
        [float(score) for _, score in ranked],
    )


def _rrf(rankings: Sequence[Sequence[str]]) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for index, record_id in enumerate(ranking):
            scores[record_id] += 1 / (index + 1)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return ordered, [scores[record_id] for record_id in ordered]


def _logical_result(
    candidate: Mapping[str, Any],
    *,
    mapping: Mapping[str, str],
    output_columns: tuple[str, ...],
    rank: int,
    score: float,
) -> dict[str, Any]:
    properties = candidate.get("properties")
    if not isinstance(properties, Mapping):
        raise TypeError("Neo4j search records must include a properties mapping")
    metadata = {
        "record_id": str(candidate["record_id"]),
        "rank": rank,
        "score": float(score),
    }
    result: dict[str, Any] = {}
    for column in output_columns:
        result[column] = (
            metadata[column]
            if column in metadata
            else properties.get(mapping[column])
        )
    return result


def _readable_properties(
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
) -> dict[str, str]:
    return {
        logical_column: property_name
        for property_name, logical_column in mapping.properties.items()
    }


def _graph_type(mapping: Neo4jNodeMapping | Neo4jRelationshipMapping) -> str:
    return mapping.label if isinstance(mapping, Neo4jNodeMapping) else mapping.relationship_type


def _lucene_escape(value: str) -> str:
    special = frozenset("+-&|!(){}[]^\"~*?:\\/")
    return "".join(f"\\{character}" if character in special else character for character in value)


__all__ = ["execute_search"]
