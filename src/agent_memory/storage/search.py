"""Backend-neutral contracts for storage-backed retrieval execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import pandas as pd

from agent_memory.policy.retrieval import RerankerSpec, SearchMethodSpec

from .table import TableDescriptor


@dataclass(frozen=True)
class SearchRequest:
    """One physical search request bound to a materialized target."""

    statement_id: str
    target: TableDescriptor
    namespace: str
    query: str
    methods: tuple[SearchMethodSpec, ...]
    reranker: RerankerSpec | None
    limit: int
    output_columns: tuple[str, ...]
    origin_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.statement_id, str) or not self.statement_id:
            raise ValueError("search statement_id must be a non-empty string")
        if not isinstance(self.target, TableDescriptor):
            raise TypeError("search target must be a TableDescriptor")
        for value, name in ((self.namespace, "namespace"), (self.query, "query")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"search {name} must be a non-empty string")
        object.__setattr__(self, "methods", tuple(self.methods))
        if not self.methods or not all(
            isinstance(method, SearchMethodSpec) for method in self.methods
        ):
            raise TypeError("search methods must contain SearchMethodSpec values")
        if self.reranker is not None and not isinstance(
            self.reranker, RerankerSpec
        ):
            raise TypeError("search reranker must be a RerankerSpec")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
            raise ValueError("search limit must be a positive integer")
        columns = tuple(self.output_columns)
        if not columns or any(not isinstance(column, str) or not column for column in columns):
            raise ValueError("search output columns must contain non-empty names")
        if len(set(columns)) != len(columns):
            raise ValueError("search output columns must be unique")
        object.__setattr__(self, "output_columns", columns)
        origins = tuple(self.origin_record_ids)
        if not all(isinstance(record_id, str) and record_id for record_id in origins):
            raise ValueError("search origin record IDs must be non-empty strings")
        object.__setattr__(self, "origin_record_ids", origins)


@dataclass(frozen=True)
class SearchBatch:
    """Physical search rows and backend-specific diagnostic metrics."""

    rows: pd.DataFrame
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, pd.DataFrame):
            raise TypeError("search batch rows must be a pandas DataFrame")
        object.__setattr__(self, "rows", self.rows.copy())
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


class SearchConnector(Protocol):
    """Optional connector capability for physical retrieval."""

    def search(self, request: SearchRequest) -> SearchBatch:
        """Execute one storage-backed search node."""

        ...


class CrossEncoderProvider(Protocol):
    """Runtime provider for one logical cross-encoder descriptor."""

    def rank(
        self,
        *,
        model: str,
        query: str,
        passages: list[str],
    ) -> list[tuple[int, float]]:
        """Return passage indexes and scores in descending relevance order."""

        ...


__all__ = [
    "CrossEncoderProvider",
    "SearchBatch",
    "SearchConnector",
    "SearchRequest",
]
