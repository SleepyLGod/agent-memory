"""Physical Neo4j row encoding and changelog application primitives."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from .mapping import (
    EmbeddingSpec,
    Neo4jIdentity,
    Neo4jNestedProperty,
    Neo4jNodeMapping,
    Neo4jRelationshipMapping,
)


_AGENT_MEMORY_UUID_NAMESPACE = uuid5(NAMESPACE_URL, "agent-memory")


class EmbeddingProvider(Protocol):
    """Runtime provider for one pinned physical embedding specification."""

    def embed(self, spec: EmbeddingSpec, texts: list[str]) -> list[list[float]]:
        """Embed texts in input order."""

        ...


@dataclass(frozen=True)
class MaterializedNode:
    """One fully encoded Neo4j node write."""

    label: str
    uuid: str
    properties: Mapping[str, Any]
    embedding_property: str | None = None

    def __post_init__(self) -> None:
        _neo4j_identifier(self.label, name="node label")
        _non_empty(self.uuid, name="node uuid")
        if self.embedding_property is not None:
            _neo4j_identifier(
                self.embedding_property, name="node embedding property"
            )
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))


@dataclass(frozen=True)
class MaterializedRelationship:
    """One fully encoded Neo4j relationship write."""

    relationship_type: str
    uuid: str
    source_uuid: str
    target_uuid: str
    properties: Mapping[str, Any]
    embedding_property: str | None = None

    def __post_init__(self) -> None:
        _neo4j_identifier(self.relationship_type, name="relationship type")
        _non_empty(self.uuid, name="relationship uuid")
        _non_empty(self.source_uuid, name="relationship source uuid")
        _non_empty(self.target_uuid, name="relationship target uuid")
        if self.embedding_property is not None:
            _neo4j_identifier(
                self.embedding_property, name="relationship embedding property"
            )
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))


MaterializedGraphObject = MaterializedNode | MaterializedRelationship


@dataclass(frozen=True)
class PreparedWrite:
    """One sink statement encoded before opening a database transaction."""

    inserted: tuple[MaterializedGraphObject, ...]
    retracted: tuple[MaterializedGraphObject, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inserted", tuple(self.inserted))
        object.__setattr__(self, "retracted", tuple(self.retracted))
        if any(
            not isinstance(item, (MaterializedNode, MaterializedRelationship))
            for item in (*self.inserted, *self.retracted)
        ):
            raise TypeError("prepared Neo4j writes must contain graph objects")


def physical_uuid(namespace: str, kind: str, values: Sequence[Any]) -> str:
    """Return a namespace- and kind-isolated deterministic UUIDv5."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("physical UUID namespace must be a non-empty string")
    if not isinstance(kind, str) or not kind:
        raise ValueError("physical UUID kind must be a non-empty string")
    normalized = [_json_identity_value(value) for value in values]
    namespace_uuid = uuid5(_AGENT_MEMORY_UUID_NAMESPACE, namespace)
    identity = json.dumps(
        {"kind": kind, "values": normalized},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return str(uuid5(namespace_uuid, identity))


def materialize_rows(
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    rows: pd.DataFrame,
    *,
    namespace: str,
    embedding_provider: EmbeddingProvider | None,
    include_embeddings: bool,
) -> list[MaterializedGraphObject]:
    """Encode logical rows according to one validated Neo4j mapping."""

    if not isinstance(mapping, (Neo4jNodeMapping, Neo4jRelationshipMapping)):
        raise TypeError("Neo4j row materialization requires a Neo4j mapping")
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("Neo4j row materialization requires a pandas DataFrame")

    materialized = [
        _materialize_row(mapping, row, namespace=namespace)
        for row in rows.to_dict("records")
    ]
    if include_embeddings and mapping.embedding is not None and materialized:
        if embedding_provider is None:
            raise ValueError("Neo4j embedding materialization requires an embedding provider")
        texts = []
        for row in rows.to_dict("records"):
            value = _value(row, mapping.embedding.source_column)
            if not isinstance(value, str):
                raise TypeError("embedding source values must be strings")
            texts.append(value)
        vectors = embedding_provider.embed(mapping.embedding, texts)
        if len(vectors) != len(materialized):
            raise ValueError("embedding provider returned the wrong number of vectors")
        for index, (graph_object, vector) in enumerate(
            zip(materialized, vectors, strict=True)
        ):
            if len(vector) != mapping.embedding.dimensions:
                raise ValueError(
                    "embedding provider must return vectors with "
                    f"{mapping.embedding.dimensions} dimensions"
                )
            properties = dict(graph_object.properties)
            properties[mapping.embedding.property_name] = [
                _finite_float(value) for value in vector
            ]
            if isinstance(graph_object, MaterializedNode):
                replacement: MaterializedGraphObject = MaterializedNode(
                    label=graph_object.label,
                    uuid=graph_object.uuid,
                    properties=properties,
                    embedding_property=mapping.embedding.property_name,
                )
            else:
                replacement = MaterializedRelationship(
                    relationship_type=graph_object.relationship_type,
                    uuid=graph_object.uuid,
                    source_uuid=graph_object.source_uuid,
                    target_uuid=graph_object.target_uuid,
                    properties=properties,
                    embedding_property=mapping.embedding.property_name,
                )
            materialized[index] = replacement
    return materialized


def apply_writes(
    tx: Any,
    writes: Sequence[PreparedWrite],
    *,
    namespace: str,
) -> None:
    """Apply prepared changelog rows in graph dependency order."""

    _non_empty(namespace, name="storage namespace")
    inserted = [item for write in writes for item in write.inserted]
    retracted = [item for write in writes for item in write.retracted]
    _validate_inserted_keys(inserted)
    _validate_namespaces((*inserted, *retracted), namespace=namespace)

    # A same-key retract+insert is an update; only endpoint changes require removal.
    inserted_by_key = {_object_key(item): item for item in inserted}
    relationship_deletes: list[MaterializedRelationship] = []
    node_deletes: list[MaterializedNode] = []
    for old in retracted:
        replacement = inserted_by_key.get(_object_key(old))
        if isinstance(old, MaterializedRelationship):
            if not isinstance(replacement, MaterializedRelationship) or (
                old.source_uuid != replacement.source_uuid
                or old.target_uuid != replacement.target_uuid
            ):
                relationship_deletes.append(old)
        elif not isinstance(replacement, MaterializedNode):
            node_deletes.append(old)

    # Preserve graph dependencies: detach old edges, create nodes, then create edges.
    _delete_relationships(tx, relationship_deletes, namespace=namespace)
    _upsert_nodes(
        tx,
        [item for item in inserted if isinstance(item, MaterializedNode)],
        namespace=namespace,
    )
    _upsert_relationships(
        tx,
        [item for item in inserted if isinstance(item, MaterializedRelationship)],
        namespace=namespace,
    )
    _delete_nodes(tx, node_deletes, namespace=namespace)


def _materialize_row(
    mapping: Neo4jNodeMapping | Neo4jRelationshipMapping,
    row: Mapping[str, Any],
    *,
    namespace: str,
) -> MaterializedGraphObject:
    object_uuid = _identity_uuid(mapping.identity, row, namespace=namespace)
    properties: dict[str, Any] = {
        "uuid": object_uuid,
        "group_id": namespace,
    }
    properties.update(
        {
            property_name: _value(row, source_column)
            for property_name, source_column in mapping.properties.items()
        }
    )
    properties.update(
        {name: _normalize_property(value) for name, value in mapping.constants.items()}
    )
    for nested in mapping.nested_properties:
        properties[nested.property_name] = _nested_value(
            nested,
            row,
            namespace=namespace,
        )

    if isinstance(mapping, Neo4jNodeMapping):
        return MaterializedNode(
            label=mapping.label,
            uuid=object_uuid,
            properties=properties,
            embedding_property=None,
        )

    source_uuid = _identity_uuid(mapping.source, row, namespace=namespace)
    target_uuid = _identity_uuid(mapping.target, row, namespace=namespace)
    properties["source_uuid"] = source_uuid
    properties["target_uuid"] = target_uuid
    return MaterializedRelationship(
        relationship_type=mapping.relationship_type,
        uuid=object_uuid,
        source_uuid=source_uuid,
        target_uuid=target_uuid,
        properties=properties,
        embedding_property=None,
    )


def _identity_uuid(
    identity: Neo4jIdentity,
    row: Mapping[str, Any],
    *,
    namespace: str,
) -> str:
    return physical_uuid(
        namespace,
        identity.kind,
        tuple(_value(row, column) for column in identity.columns),
    )


def _nested_value(
    nested: Neo4jNestedProperty,
    row: Mapping[str, Any],
    *,
    namespace: str,
) -> Any:
    source = _value(row, nested.source_column)
    if source is None:
        return [] if nested.many else None
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"nested property source {nested.source_column!r} must be JSON"
            ) from exc
    if not isinstance(source, (list, tuple)):
        raise TypeError(
            f"nested property source {nested.source_column!r} must be an array"
        )
    values: list[Any] = []
    for item in source:
        if not isinstance(item, Mapping):
            raise TypeError("nested property array elements must be records")
        if nested.field not in item:
            raise ValueError(
                f"nested property field {nested.field!r} is missing from a record"
            )
        value = _normalize_property(item[nested.field])
        if nested.identity_kind is not None and value is not None:
            value = physical_uuid(namespace, nested.identity_kind, (value,))
        values.append(value)
    if nested.many:
        return values
    return values[0] if values else None


def _value(row: Mapping[str, Any], column: str) -> Any:
    if column not in row:
        raise ValueError(f"logical row is missing column {column!r}")
    return _normalize_property(row[column])


def _normalize_property(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_property(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_property(item) for item in value]
    if isinstance(value, list):
        return [_normalize_property(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        return _normalize_property(item())
    return value


def _json_identity_value(value: Any) -> Any:
    value = _normalize_property(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("physical UUID values must be finite")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_identity_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_identity_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    raise TypeError(f"unsupported physical UUID value: {type(value).__name__}")


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("embedding vectors must contain finite values")
    return result


def _object_key(graph_object: MaterializedGraphObject) -> tuple[str, str]:
    kind = "node" if isinstance(graph_object, MaterializedNode) else "relationship"
    return kind, graph_object.uuid


def _validate_inserted_keys(inserted: Sequence[MaterializedGraphObject]) -> None:
    keys = [_object_key(item) for item in inserted]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate inserted Neo4j identity: {duplicates}")


def _validate_namespaces(
    graph_objects: Sequence[MaterializedGraphObject],
    *,
    namespace: str,
) -> None:
    for graph_object in graph_objects:
        if graph_object.properties.get("group_id") != namespace:
            raise ValueError("materialized graph object has the wrong namespace")


def _delete_relationships(
    tx: Any,
    relationships: Sequence[MaterializedRelationship],
    *,
    namespace: str,
) -> None:
    grouped: dict[str, list[str]] = {}
    for relationship in relationships:
        grouped.setdefault(relationship.relationship_type, []).append(relationship.uuid)
    for relationship_type, uuids in grouped.items():
        tx.run(
            f"""
            MATCH ()-[r:{relationship_type}]->()
            WHERE r.uuid IN $uuids AND r.group_id = $namespace
            DELETE r
            """,
            uuids=sorted(set(uuids)),
            namespace=namespace,
        )


def _upsert_nodes(
    tx: Any,
    nodes: Sequence[MaterializedNode],
    *,
    namespace: str,
) -> None:
    grouped: dict[tuple[str, str | None], list[MaterializedNode]] = {}
    for node in nodes:
        grouped.setdefault((node.label, node.embedding_property), []).append(node)
    for (label, embedding_property), values in grouped.items():
        rows = [_node_row(value) for value in values]
        query = f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{uuid: row.uuid}})
            SET n = row.properties
        """
        if embedding_property is not None:
            query += f"""
            WITH n, row
            CALL db.create.setNodeVectorProperty(
                n, "{embedding_property}", row.embedding
            )
            """
        tx.run(query, rows=rows, namespace=namespace)


def _upsert_relationships(
    tx: Any,
    relationships: Sequence[MaterializedRelationship],
    *,
    namespace: str,
) -> None:
    grouped: dict[tuple[str, str | None], list[MaterializedRelationship]] = {}
    for relationship in relationships:
        grouped.setdefault(
            (relationship.relationship_type, relationship.embedding_property), []
        ).append(relationship)
    for (relationship_type, embedding_property), values in grouped.items():
        rows = [_relationship_row(value) for value in values]
        query = f"""
            UNWIND $rows AS row
            MATCH (source {{uuid: row.source_uuid, group_id: $namespace}})
            MATCH (target {{uuid: row.target_uuid, group_id: $namespace}})
            MERGE (source)-[r:{relationship_type} {{uuid: row.uuid}}]->(target)
            SET r = row.properties
        """
        if embedding_property is not None:
            query += f"""
            WITH r, row
            CALL db.create.setRelationshipVectorProperty(
                r, "{embedding_property}", row.embedding
            )
            """
        tx.run(query, rows=rows, namespace=namespace)


def _delete_nodes(
    tx: Any,
    nodes: Sequence[MaterializedNode],
    *,
    namespace: str,
) -> None:
    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(node.label, []).append(node.uuid)
    for label, uuids in grouped.items():
        tx.run(
            f"""
            MATCH (n:{label})
            WHERE n.uuid IN $uuids AND n.group_id = $namespace
            DETACH DELETE n
            """,
            uuids=sorted(set(uuids)),
            namespace=namespace,
        )


def _node_row(node: MaterializedNode) -> dict[str, Any]:
    properties = dict(node.properties)
    embedding = (
        None
        if node.embedding_property is None
        else properties.pop(node.embedding_property)
    )
    return {"uuid": node.uuid, "properties": properties, "embedding": embedding}


def _relationship_row(
    relationship: MaterializedRelationship,
) -> dict[str, Any]:
    properties = dict(relationship.properties)
    embedding = (
        None
        if relationship.embedding_property is None
        else properties.pop(relationship.embedding_property)
    )
    return {
        "uuid": relationship.uuid,
        "source_uuid": relationship.source_uuid,
        "target_uuid": relationship.target_uuid,
        "properties": properties,
        "embedding": embedding,
    }


_NEO4J_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _neo4j_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _NEO4J_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a valid Neo4j identifier")
    return value


def _non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
