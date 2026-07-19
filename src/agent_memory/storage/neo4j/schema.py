"""Immutable, idempotent Neo4j schema preparation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EQUIVALENT_SCHEMA_RULE_CODE = (
    "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists"
)


@dataclass(frozen=True)
class Neo4jSchema:
    """Immutable idempotent Neo4j schema statements."""

    queries: tuple[str, ...]
    node_fulltext_indexes: dict[str, str] | None = None
    relationship_fulltext_indexes: dict[str, str] | None = None
    traversal_relationship_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "queries", tuple(self.queries))
        if not self.queries:
            raise ValueError("Neo4j schema requires at least one query")
        if any(not isinstance(query, str) or not query.strip() for query in self.queries):
            raise ValueError("Neo4j schema queries must be non-empty strings")
        if len(set(self.queries)) != len(self.queries):
            raise ValueError("Neo4j schema queries must be unique")
        object.__setattr__(
            self,
            "node_fulltext_indexes",
            MappingProxyType(
                _validate_index_mapping(self.node_fulltext_indexes, name="node")
            ),
        )
        object.__setattr__(
            self,
            "relationship_fulltext_indexes",
            MappingProxyType(
                _validate_index_mapping(
                    self.relationship_fulltext_indexes,
                    name="relationship",
                )
            ),
        )
        traversal = tuple(self.traversal_relationship_types)
        if any(not _is_identifier(value) for value in traversal):
            raise ValueError("Neo4j traversal relationship types must be identifiers")
        if len(set(traversal)) != len(traversal):
            raise ValueError("Neo4j traversal relationship types must be unique")
        object.__setattr__(self, "traversal_relationship_types", traversal)

    def ensure(self, session: Any) -> None:
        """Create all schema objects and wait until indexes are online."""

        for query in self.queries:
            try:
                session.run(query).consume()
            except Exception as exc:
                # Only this machine-readable code represents a harmless schema race.
                if getattr(exc, "code", None) != _EQUIVALENT_SCHEMA_RULE_CODE:
                    raise
        session.run("CALL db.awaitIndexes($timeout)", timeout=300).consume()

    def fulltext_index(self, *, kind: str, graph_type: str) -> str:
        """Return the configured index for one mapped graph type."""

        indexes = (
            self.node_fulltext_indexes
            if kind == "node"
            else self.relationship_fulltext_indexes
            if kind == "relationship"
            else None
        )
        if indexes is None or graph_type not in indexes:
            raise ValueError(
                f"Neo4j schema has no {kind} full-text index for {graph_type!r}"
            )
        return indexes[graph_type]


def _validate_index_mapping(
    values: dict[str, str] | None,
    *,
    name: str,
) -> dict[str, str]:
    normalized = {} if values is None else dict(values)
    if any(not _is_identifier(key) or not _is_identifier(value) for key, value in normalized.items()):
        raise ValueError(
            f"Neo4j {name} full-text mappings must contain identifiers"
        )
    return normalized


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


__all__ = ["Neo4jSchema"]
