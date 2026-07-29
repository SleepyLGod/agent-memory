"""Immutable descriptors for storage-backed retrieval queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

import pandas as pd

from .logical import QueryExpr
from .relation import SearchRelation


@dataclass(frozen=True)
class SearchMethodSpec:
    """Serializable search-method descriptor stored in a search expression."""

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("search method kind must be a non-empty string")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __hash__(self) -> int:
        return hash((self.kind, tuple(sorted(self.params.items()))))


@dataclass(frozen=True)
class RerankerSpec:
    """Serializable reranker descriptor stored in a search expression."""

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("reranker kind must be a non-empty string")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __hash__(self) -> int:
        return hash((self.kind, tuple(sorted(self.params.items()))))


@dataclass(frozen=True)
class BM25:
    """Full-text BM25 retrieval method."""

    kind: str = field(default="bm25", init=False)


@dataclass(frozen=True)
class CosineSimilarity:
    """Embedding cosine-similarity retrieval method."""

    candidate_limit: int | None = None
    min_score: float | None = None
    kind: str = field(default="cosine_similarity", init=False)

    def __post_init__(self) -> None:
        if self.candidate_limit is not None and (
            isinstance(self.candidate_limit, bool)
            or not isinstance(self.candidate_limit, int)
            or self.candidate_limit <= 0
        ):
            raise ValueError("cosine candidate_limit must be a positive integer")
        if self.min_score is None:
            return
        if isinstance(self.min_score, bool) or not isinstance(
            self.min_score, (int, float)
        ):
            raise TypeError("cosine min_score must be a number")
        normalized_score = float(self.min_score)
        if not isfinite(normalized_score) or not -1.0 <= normalized_score <= 1.0:
            raise ValueError("cosine min_score must be between -1 and 1")
        object.__setattr__(self, "min_score", normalized_score)


@dataclass(frozen=True)
class BFS:
    """Graph breadth-first search from a prior search relation."""

    origins: Any
    max_depth: int = 3
    kind: str = field(default="bfs", init=False)

    def __post_init__(self) -> None:
        from .relation import SearchRelation

        if not isinstance(self.origins, SearchRelation):
            raise TypeError("BFS origins must be a SearchRelation")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth <= 0
        ):
            raise ValueError("BFS max_depth must be a positive integer")


@dataclass(frozen=True)
class RRF:
    """Reciprocal-rank-fusion reranker."""

    kind: str = field(default="rrf", init=False)


@dataclass(frozen=True)
class CrossEncoder:
    """Cross-encoder reranker identified by a pinned model name."""

    model: str
    kind: str = field(default="cross_encoder", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("cross-encoder model must be a non-empty string")


SearchMethod = BM25 | CosineSimilarity | BFS
Reranker = RRF | CrossEncoder


def normalize_search(
    methods: Sequence[SearchMethod],
    reranker: Reranker | None,
) -> tuple[
    tuple[SearchMethodSpec, ...],
    RerankerSpec | None,
    tuple[QueryExpr, ...],
]:
    """Normalize public descriptors and expose search dependencies as inputs."""

    normalized_methods = tuple(methods)
    if not normalized_methods:
        raise ValueError("search methods cannot be empty")
    dependencies: list[QueryExpr] = []
    method_specs: list[SearchMethodSpec] = []
    for method in normalized_methods:
        if isinstance(method, BM25):
            method_specs.append(SearchMethodSpec(kind=method.kind))
        elif isinstance(method, CosineSimilarity):
            params: dict[str, Any] = {}
            if method.candidate_limit is not None:
                params["candidate_limit"] = method.candidate_limit
            if method.min_score is not None:
                params["min_score"] = method.min_score
            method_specs.append(SearchMethodSpec(kind=method.kind, params=params))
        elif isinstance(method, BFS):
            origin = method.origins.expr
            try:
                dependency_index = dependencies.index(origin)
            except ValueError:
                dependencies.append(origin)
                dependency_index = len(dependencies) - 1
            method_specs.append(
                SearchMethodSpec(
                    kind=method.kind,
                    params={
                        "origin_input": dependency_index + 1,
                        "max_depth": method.max_depth,
                    },
                )
            )
        else:
            raise TypeError("search methods must be BM25, CosineSimilarity, or BFS")

    if reranker is None:
        if len(normalized_methods) != 1:
            raise ValueError(
                "search without a reranker requires exactly one method"
            )
        reranker_spec = None
    elif isinstance(reranker, RRF):
        reranker_spec = RerankerSpec(kind=reranker.kind)
    elif isinstance(reranker, CrossEncoder):
        reranker_spec = RerankerSpec(
            kind=reranker.kind,
            params={"model": reranker.model},
        )
    else:
        raise TypeError("search reranker must be RRF or CrossEncoder")
    return tuple(method_specs), reranker_spec, tuple(dependencies)


@dataclass(frozen=True, init=False)
class RetrievalQuery:
    """One immutable retrieval root with ordered named result channels."""

    channels: Mapping[str, QueryExpr]

    def __init__(self, **channels: Any) -> None:
        from .relation import SearchRelation

        if not channels:
            raise ValueError("RetrievalQuery requires at least one channel")
        normalized: dict[str, QueryExpr] = {}
        for name, relation in channels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("retrieval channel names must be non-empty strings")
            if not isinstance(relation, SearchRelation):
                raise TypeError("retrieval channels must be SearchRelation values")
            normalized[name] = relation.expr
        object.__setattr__(self, "channels", MappingProxyType(normalized))


@dataclass(frozen=True)
class RetrievalResult:
    """Ordered DataFrame channels produced by one retrieval execution."""

    query: str
    channels: Mapping[str, pd.DataFrame]
    metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("retrieval result query must be a string")
        normalized: dict[str, pd.DataFrame] = {}
        for name, frame in self.channels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("retrieval result channel names must be non-empty strings")
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("retrieval result channels must be pandas DataFrames")
            normalized[name] = frame.copy()
        object.__setattr__(self, "channels", MappingProxyType(normalized))
        normalized_metrics: dict[str, Mapping[str, Any]] = {}
        for name, metrics in self.metrics.items():
            if name not in normalized:
                raise ValueError(
                    f"retrieval metrics reference unknown channel {name!r}"
                )
            if not isinstance(metrics, Mapping):
                raise TypeError("retrieval channel metrics must be mappings")
            normalized_metrics[name] = MappingProxyType(dict(metrics))
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(normalized_metrics),
        )


__all__ = [
    "BFS",
    "BM25",
    "CosineSimilarity",
    "CrossEncoder",
    "RRF",
    "RetrievalQuery",
    "RetrievalResult",
    "RerankerSpec",
    "SearchMethodSpec",
    "SearchRelation",
]
