"""Immutable mappings from logical relation rows to Neo4j graph objects."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from agent_memory.storage.embedding import EmbeddingSpec
from agent_memory.storage.schema import Schema


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEARCH_METADATA_COLUMNS = frozenset({"record_id", "rank", "score"})


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a valid Neo4j identifier")
    return value


def _columns(values: object, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        normalized = (values,)
    else:
        try:
            normalized = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or sequence of strings") from exc
    if not normalized or any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty column names")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate columns")
    return normalized


@dataclass(frozen=True)
class Neo4jIdentity:
    """Logical columns deterministically encoded as one physical UUID."""

    columns: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", _columns(self.columns, name="identity columns"))
        _identifier(self.kind, name="identity kind")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serializable representation."""

        return {"columns": list(self.columns), "kind": self.kind}


@dataclass(frozen=True)
class Neo4jNestedProperty:
    """Project one field from a logical array-of-records column."""

    source_column: str
    field: str
    property_name: str
    many: bool
    identity_kind: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_column, "nested source column"),
            (self.field, "nested field"),
            (self.property_name, "nested property name"),
        ):
            _identifier(value, name=name)
        if not isinstance(self.many, bool):
            raise TypeError("nested property many must be bool")
        if self.identity_kind is not None:
            _identifier(self.identity_kind, name="nested identity kind")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serializable representation."""

        return {
            "source_column": self.source_column,
            "field": self.field,
            "property_name": self.property_name,
            "many": self.many,
            "identity_kind": self.identity_kind,
        }


@dataclass(frozen=True)
class _Neo4jMapping:
    """Shared immutable mapping state for nodes and relationships."""

    connector: ClassVar[str] = "neo4j"

    identity: Neo4jIdentity
    properties: Mapping[str, str] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    nested_properties: tuple[Neo4jNestedProperty, ...] = ()
    embedding: EmbeddingSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Neo4jIdentity):
            raise TypeError("Neo4j mapping identity must be Neo4jIdentity")
        normalized_properties: dict[str, str] = {}
        for property_name, source_column in self.properties.items():
            normalized_properties[_identifier(property_name, name="property name")] = _identifier(
                source_column, name="property source column"
            )
        normalized_constants: dict[str, Any] = {}
        for property_name, value in self.constants.items():
            normalized_constants[
                _identifier(property_name, name="constant property name")
            ] = _freeze_constant(value)
        object.__setattr__(self, "properties", MappingProxyType(normalized_properties))
        object.__setattr__(self, "constants", MappingProxyType(normalized_constants))
        object.__setattr__(self, "nested_properties", tuple(self.nested_properties))
        if any(not isinstance(item, Neo4jNestedProperty) for item in self.nested_properties):
            raise TypeError("nested properties must be Neo4jNestedProperty values")
        if self.embedding is not None and not isinstance(self.embedding, EmbeddingSpec):
            raise TypeError("Neo4j mapping embedding must be EmbeddingSpec")
        if self.embedding is not None:
            _identifier(
                self.embedding.source_column,
                name="embedding source column",
            )
            _identifier(
                self.embedding.property_name,
                name="embedding property name",
            )

        output_names = [*self.properties, *self.constants]
        output_names.extend(item.property_name for item in self.nested_properties)
        if self.embedding is not None:
            output_names.append(self.embedding.property_name)
        duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Neo4j property names must be unique: {duplicates}")

    def _validate_columns(self, schema: Schema, identities: tuple[Neo4jIdentity, ...]) -> None:
        available = {column.name for column in schema.columns}
        referenced: set[str] = set()
        for identity in identities:
            referenced.update(identity.columns)
        referenced.update(self.properties.values())
        referenced.update(item.source_column for item in self.nested_properties)
        if self.embedding is not None:
            referenced.add(self.embedding.source_column)
        missing = sorted(referenced.difference(available))
        if missing:
            raise ValueError(f"Neo4j mapping references unknown logical column(s): {missing}")
        if schema.primary_key and tuple(schema.primary_key) != self.identity.columns:
            raise ValueError("Neo4j identity columns must match the storage primary key")

    def _base_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "identity": self.identity.to_dict(),
            "properties": dict(self.properties),
            "constants": {
                name: _serialize_constant(value)
                for name, value in self.constants.items()
            },
            "nested_properties": [item.to_dict() for item in self.nested_properties],
            "embedding": None if self.embedding is None else self.embedding.to_dict(),
        }

    def validate_search_columns(self, columns: tuple[str, ...]) -> None:
        """Reject logical result columns that cannot be reconstructed."""

        readable = set(self.properties.values()) | _SEARCH_METADATA_COLUMNS
        missing = [column for column in columns if column not in readable]
        if missing:
            raise ValueError(
                "Neo4j mapping cannot read logical column(s): "
                f"{missing}"
            )


@dataclass(frozen=True)
class Neo4jNodeMapping(_Neo4jMapping):
    """Typed mapping from one logical relation row to one Neo4j node."""

    label: str = "Entity"

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.label, name="node label")

    def validate(self, schema: Schema) -> None:
        """Validate all logical column references."""

        self._validate_columns(schema, (self.identity,))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serializable representation."""

        return {"kind": "node", "label": self.label, **self._base_dict()}


@dataclass(frozen=True)
class Neo4jRelationshipMapping(_Neo4jMapping):
    """Typed mapping from one logical row to one Neo4j relationship."""

    relationship_type: str = "RELATES_TO"
    source: Neo4jIdentity = field(
        default_factory=lambda: Neo4jIdentity(("source_id",), "source")
    )
    target: Neo4jIdentity = field(
        default_factory=lambda: Neo4jIdentity(("target_id",), "target")
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.relationship_type, name="relationship type")
        if not isinstance(self.source, Neo4jIdentity) or not isinstance(
            self.target, Neo4jIdentity
        ):
            raise TypeError("relationship endpoints must be Neo4jIdentity values")

    def validate(self, schema: Schema) -> None:
        """Validate all logical key, endpoint, and property references."""

        self._validate_columns(schema, (self.identity, self.source, self.target))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serializable representation."""

        return {
            "kind": "relationship",
            "relationship_type": self.relationship_type,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            **self._base_dict(),
        }


def _freeze_constant(value: Any) -> Any:
    """Deep-freeze one JSON-compatible physical constant."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_constant(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_constant(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Neo4j constant properties must contain JSON-compatible values"
    )


def _serialize_constant(value: Any) -> Any:
    """Convert a frozen physical constant back into JSON data."""

    if isinstance(value, Mapping):
        return {str(key): _serialize_constant(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_constant(item) for item in value]
    return value
