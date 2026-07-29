"""Logical-row materialization primitives for versioned Qdrant points."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from agent_memory.storage.embedding import EmbeddingProvider
from agent_memory.storage.identity import normalize_storage_value, physical_uuid

from .mapping import QdrantPointMapping


_MAX_VISIBLE_COMMIT = (1 << 63) - 1


@dataclass(frozen=True)
class MaterializedPoint:
    """One immutable point version prepared before Qdrant is mutated."""

    point_id: str
    record_id: str
    collection: str
    vector: tuple[float, ...]
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        for value, name in (
            (self.point_id, "point ID"),
            (self.record_id, "record ID"),
            (self.collection, "collection"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Qdrant {name} must be a non-empty string")
        object.__setattr__(self, "vector", tuple(self.vector))
        if any(not math.isfinite(value) for value in self.vector):
            raise ValueError("Qdrant vectors must contain finite values")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class PreparedQdrantWrite:
    """One statement's point versions and logical retractions."""

    inserted: tuple[MaterializedPoint, ...]
    retracted_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inserted", tuple(self.inserted))
        object.__setattr__(
            self,
            "retracted_record_ids",
            tuple(self.retracted_record_ids),
        )
        if any(not isinstance(item, MaterializedPoint) for item in self.inserted):
            raise TypeError("prepared Qdrant inserts must be MaterializedPoint values")
        if any(
            not isinstance(record_id, str) or not record_id
            for record_id in self.retracted_record_ids
        ):
            raise ValueError("prepared Qdrant retractions require record IDs")


def materialize_rows(
    mapping: QdrantPointMapping,
    rows: pd.DataFrame,
    *,
    namespace: str,
    statement_id: str,
    materialization_id: str,
    visible_from: int,
    embedding_provider: EmbeddingProvider,
) -> tuple[MaterializedPoint, ...]:
    """Encode and embed inserted logical rows without mutating Qdrant."""

    _validate_materialization_inputs(
        mapping,
        rows,
        namespace=namespace,
        statement_id=statement_id,
        materialization_id=materialization_id,
        visible_from=visible_from,
    )
    records = rows.to_dict("records")
    record_ids = tuple(
        _record_id(mapping, row, namespace=namespace) for row in records
    )
    _reject_duplicates(record_ids, name="inserted Qdrant record ID")
    if not records:
        return ()

    texts: list[str] = []
    for row in records:
        value = _required_value(row, mapping.embedding.source_column)
        if not isinstance(value, str):
            raise TypeError("Qdrant embedding source values must be strings")
        texts.append(value)
    vectors = embedding_provider.embed(mapping.embedding, texts)
    if len(vectors) != len(records):
        raise ValueError("embedding provider returned the wrong number of vectors")

    points: list[MaterializedPoint] = []
    for row, record_id, vector in zip(
        records,
        record_ids,
        vectors,
        strict=True,
    ):
        if len(vector) != mapping.embedding.dimensions:
            raise ValueError(
                "embedding provider must return vectors with "
                f"{mapping.embedding.dimensions} dimensions"
            )
        payload = {
            "_agent_memory_namespace": namespace,
            "_agent_memory_statement_id": statement_id,
            "_agent_memory_record_id": record_id,
            "_agent_memory_materialization": materialization_id,
            "_agent_memory_visible_from": visible_from,
            "_agent_memory_visible_until": _MAX_VISIBLE_COMMIT,
        }
        payload.update(
            {
                property_name: _required_value(row, source_column)
                for property_name, source_column in mapping.properties.items()
            }
        )
        points.append(
            MaterializedPoint(
                point_id=physical_uuid(
                    namespace,
                    "qdrant_point_version",
                    (
                        statement_id,
                        materialization_id,
                        record_id,
                        visible_from,
                    ),
                ),
                record_id=record_id,
                collection=mapping.collection,
                vector=tuple(float(value) for value in vector),
                payload=payload,
            )
        )
    return tuple(points)


def materialize_retractions(
    mapping: QdrantPointMapping,
    rows: pd.DataFrame,
    *,
    namespace: str,
) -> tuple[str, ...]:
    """Return stable record IDs for logical rows being retracted."""

    if not isinstance(mapping, QdrantPointMapping):
        raise TypeError("Qdrant retractions require a QdrantPointMapping")
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("Qdrant retractions require a pandas DataFrame")
    record_ids = tuple(
        _record_id(mapping, row, namespace=namespace)
        for row in rows.to_dict("records")
    )
    _reject_duplicates(record_ids, name="retracted Qdrant record ID")
    return record_ids


def _record_id(
    mapping: QdrantPointMapping,
    row: Mapping[str, Any],
    *,
    namespace: str,
) -> str:
    return physical_uuid(
        namespace,
        mapping.identity.kind,
        tuple(_required_value(row, column) for column in mapping.identity.columns),
    )


def _required_value(row: Mapping[str, Any], column: str) -> Any:
    if column not in row:
        raise ValueError(f"logical row is missing column {column!r}")
    return normalize_storage_value(row[column])


def _reject_duplicates(values: Sequence[str], *, name: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {name}: {duplicates}")


def _validate_materialization_inputs(
    mapping: QdrantPointMapping,
    rows: pd.DataFrame,
    *,
    namespace: str,
    statement_id: str,
    materialization_id: str,
    visible_from: int,
) -> None:
    if not isinstance(mapping, QdrantPointMapping):
        raise TypeError("Qdrant materialization requires a QdrantPointMapping")
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("Qdrant materialization requires a pandas DataFrame")
    for value, name in (
        (namespace, "namespace"),
        (statement_id, "statement ID"),
        (materialization_id, "materialization ID"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Qdrant {name} must be a non-empty string")
    if (
        not isinstance(visible_from, int)
        or isinstance(visible_from, bool)
        or visible_from < 0
    ):
        raise ValueError("Qdrant visible commit must be a non-negative integer")


__all__ = [
    "MaterializedPoint",
    "PreparedQdrantWrite",
    "materialize_retractions",
    "materialize_rows",
]
