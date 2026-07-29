"""Immutable mappings from logical relation rows to Qdrant points."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from agent_memory.storage.embedding import EmbeddingSpec
from agent_memory.storage.schema import Schema


_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_PAYLOAD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEARCH_METADATA_COLUMNS = frozenset({"record_id", "rank", "score"})
_RESERVED_PAYLOAD_PREFIX = "_agent_memory_"


def _columns(values: object, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        normalized = (values,)
    else:
        try:
            normalized = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or sequence of strings") from exc
    if not normalized or any(
        not isinstance(value, str) or not value for value in normalized
    ):
        raise ValueError(f"{name} must contain non-empty column names")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate columns")
    return normalized


@dataclass(frozen=True)
class QdrantIdentity:
    """Logical columns deterministically encoded as one stable record ID."""

    columns: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "columns",
            _columns(self.columns, name="Qdrant identity columns"),
        )
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Qdrant identity kind must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serializable representation."""

        return {"columns": list(self.columns), "kind": self.kind}


@dataclass(frozen=True)
class QdrantPointMapping:
    """Typed mapping from one logical relation row to one Qdrant point."""

    connector: ClassVar[str] = "qdrant"

    collection: str
    identity: QdrantIdentity
    embedding: EmbeddingSpec
    properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.collection, str) or not _COLLECTION_NAME.fullmatch(
            self.collection
        ):
            raise ValueError("Qdrant collection must be a safe non-empty name")
        if not isinstance(self.identity, QdrantIdentity):
            raise TypeError("Qdrant mapping identity must be QdrantIdentity")
        normalized_properties: dict[str, str] = {}
        for property_name, source_column in self.properties.items():
            if (
                not isinstance(property_name, str)
                or not _PAYLOAD_NAME.fullmatch(property_name)
                or property_name.startswith(_RESERVED_PAYLOAD_PREFIX)
            ):
                raise ValueError(
                    "Qdrant payload property names must be safe and non-reserved"
                )
            if not isinstance(source_column, str) or not source_column:
                raise ValueError(
                    "Qdrant payload source columns must be non-empty strings"
                )
            normalized_properties[property_name] = source_column
        object.__setattr__(
            self,
            "properties",
            MappingProxyType(normalized_properties),
        )
        if len(set(normalized_properties.values())) != len(
            normalized_properties
        ):
            raise ValueError(
                "Qdrant payload source columns must be unique"
            )
        if not isinstance(self.embedding, EmbeddingSpec):
            raise TypeError("Qdrant mapping embedding must be EmbeddingSpec")
        if self.embedding.property_name in self.properties:
            raise ValueError(
                "Qdrant embedding property cannot also be a payload property"
            )

    def validate(self, schema: Schema) -> None:
        """Validate all logical column references and the primary key."""

        available = {column.name for column in schema.columns}
        referenced = {
            *self.identity.columns,
            *self.properties.values(),
            self.embedding.source_column,
        }
        missing = sorted(referenced.difference(available))
        if missing:
            raise ValueError(
                f"Qdrant mapping references unknown logical column(s): {missing}"
            )
        if schema.primary_key and tuple(schema.primary_key) != self.identity.columns:
            raise ValueError(
                "Qdrant identity columns must match the storage primary key"
            )

    def validate_search_columns(self, columns: tuple[str, ...]) -> None:
        """Reject logical result columns that cannot be reconstructed."""

        readable = set(self.properties.values()) | _SEARCH_METADATA_COLUMNS
        missing = [column for column in columns if column not in readable]
        if missing:
            raise ValueError(
                f"Qdrant mapping cannot read logical column(s): {missing}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""

        return {
            "connector": self.connector,
            "kind": "point",
            "collection": self.collection,
            "identity": self.identity.to_dict(),
            "properties": dict(self.properties),
            "embedding": self.embedding.to_dict(),
        }


__all__ = ["QdrantIdentity", "QdrantPointMapping"]
