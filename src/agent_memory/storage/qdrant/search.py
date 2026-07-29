"""Dense cosine retrieval over one published Qdrant materialization."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pandas as pd

from agent_memory.storage.embedding import EmbeddingProvider
from agent_memory.storage.search import SearchBatch, SearchRequest

from .mapping import QdrantPointMapping
from .recovery import read_marker


def execute_search(
    client: Any,
    models: Any,
    request: SearchRequest,
    *,
    embedding_provider: EmbeddingProvider,
) -> SearchBatch:
    """Execute Mem0 Base dense retrieval against the active commit marker."""

    started = time.perf_counter()
    mapping = request.target.mapping
    if not isinstance(mapping, QdrantPointMapping):
        raise TypeError("Qdrant search requires a QdrantPointMapping")
    if request.target.connector != "qdrant":
        raise ValueError("Qdrant search only accepts qdrant table targets")
    if request.reranker is not None:
        raise NotImplementedError("Qdrant Base retrieval does not support rerankers")
    if len(request.methods) != 1 or request.methods[0].kind != "cosine_similarity":
        raise NotImplementedError(
            "Qdrant Base retrieval supports exactly one cosine method"
        )
    if request.origin_record_ids:
        raise NotImplementedError("Qdrant Base retrieval does not support BFS origins")

    marker = read_marker(client, namespace=request.namespace)
    if marker is None:
        return SearchBatch(
            rows=pd.DataFrame(columns=pd.Index(request.output_columns)),
            metrics={
                "backend": "qdrant",
                "method": "cosine_similarity",
                "candidate_ids": [],
                "candidate_scores": [],
                "result_ids": [],
                "latency_seconds": time.perf_counter() - started,
            },
        )

    vector = embedding_provider.embed(mapping.embedding, [request.query])
    if len(vector) != 1 or len(vector[0]) != mapping.embedding.dimensions:
        raise ValueError("query embedding has the wrong shape")
    method = request.methods[0]
    candidate_limit = int(method.params.get("candidate_limit", request.limit))
    min_score_value = method.params.get("min_score")
    min_score = None if min_score_value is None else float(min_score_value)
    response = client.query_points(
        collection_name=mapping.collection,
        query=[float(value) for value in vector[0]],
        query_filter=_visibility_filter(
            models,
            namespace=request.namespace,
            statement_id=request.statement_id,
            materialization_id=marker.materialization_id,
            commit_sequence=marker.commit.commit_sequence,
        ),
        limit=candidate_limit,
        with_payload=True,
        with_vectors=False,
    )
    method_candidates = list(response.points)
    candidates = method_candidates
    if min_score is not None:
        candidates = [
            point for point in candidates if float(point.score) >= min_score
        ]
    candidates = candidates[: request.limit]

    payload_by_column = {
        source_column: property_name
        for property_name, source_column in mapping.properties.items()
    }
    rows: list[dict[str, Any]] = []
    for rank, point in enumerate(candidates, start=1):
        payload = point.payload
        if not isinstance(payload, Mapping):
            raise ValueError("Qdrant search result payload is missing")
        row: dict[str, Any] = {}
        for column in request.output_columns:
            if column == "record_id":
                row[column] = _required_payload(
                    payload,
                    "_agent_memory_record_id",
                )
            elif column == "rank":
                row[column] = rank
            elif column == "score":
                row[column] = float(point.score)
            else:
                property_name = payload_by_column.get(column)
                if property_name is None:
                    raise ValueError(
                        f"Qdrant mapping cannot reconstruct column {column!r}"
                    )
                row[column] = _required_payload(payload, property_name)
        rows.append(row)

    return SearchBatch(
        rows=pd.DataFrame(rows, columns=pd.Index(request.output_columns)),
        metrics={
            "backend": "qdrant",
            "method": "cosine_similarity",
            "candidate_limit": candidate_limit,
            "min_score": min_score,
            "candidate_ids": [
                str(_required_payload(point.payload, "_agent_memory_record_id"))
                for point in method_candidates
            ],
            "candidate_scores": [
                float(point.score) for point in method_candidates
            ],
            "result_ids": [
                str(_required_payload(point.payload, "_agent_memory_record_id"))
                for point in candidates
            ],
            "latency_seconds": time.perf_counter() - started,
        },
    )


def _visibility_filter(
    models: Any,
    *,
    namespace: str,
    statement_id: str,
    materialization_id: str,
    commit_sequence: int,
) -> Any:
    return models.Filter(
        must=[
            _match(models, "_agent_memory_namespace", namespace),
            _match(models, "_agent_memory_statement_id", statement_id),
            _match(
                models,
                "_agent_memory_materialization",
                materialization_id,
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
    )


def _match(models: Any, key: str, value: str) -> Any:
    return models.FieldCondition(
        key=key,
        match=models.MatchValue(value=value),
    )


def _required_payload(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Qdrant payload is missing {key!r}")
    return payload[key]


__all__ = ["execute_search"]
