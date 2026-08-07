"""Namespace commit markers for versioned Qdrant materializations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_memory.storage.connector import StorageCommit, StorageConflictError
from agent_memory.storage.identity import physical_uuid

from .schema import CONTROL_COLLECTION


@dataclass(frozen=True)
class QdrantMarker:
    """Published storage commit and the materialization visible at that commit."""

    commit: StorageCommit
    materialization_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.commit, StorageCommit):
            raise TypeError("Qdrant marker commit must be StorageCommit")
        if not isinstance(self.materialization_id, str) or not self.materialization_id:
            raise ValueError(
                "Qdrant marker materialization ID must be a non-empty string"
            )


def marker_id(namespace: str) -> str:
    """Return the stable control-point ID for one storage namespace."""

    return physical_uuid(namespace, "qdrant_commit", ("marker",))


def read_marker(client: Any, *, namespace: str) -> QdrantMarker | None:
    """Read and validate one namespace marker."""

    records = client.retrieve(
        collection_name=CONTROL_COLLECTION,
        ids=[marker_id(namespace)],
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    if len(records) != 1:
        raise RuntimeError("Qdrant returned multiple namespace markers")
    payload = records[0].payload
    if not isinstance(payload, dict):
        raise ValueError("Qdrant namespace marker payload is missing")
    if payload.get("namespace") != namespace:
        raise ValueError("Qdrant namespace marker has the wrong namespace")
    raw_commit = payload.get("commit")
    if not isinstance(raw_commit, dict):
        raise ValueError("Qdrant namespace marker commit is missing")
    materialization_id = payload.get("materialization_id")
    if not isinstance(materialization_id, str) or not materialization_id:
        raise ValueError(
            "Qdrant namespace marker materialization ID is missing"
        )
    return QdrantMarker(
        commit=StorageCommit.from_dict(raw_commit),
        materialization_id=materialization_id,
    )


def require_marker(
    client: Any,
    *,
    namespace: str,
    expected_commit: StorageCommit | None,
) -> QdrantMarker | None:
    """Return the current marker or raise on compare-and-set conflict."""

    marker = read_marker(client, namespace=namespace)
    actual_commit = None if marker is None else marker.commit
    if actual_commit != expected_commit:
        raise StorageConflictError(
            "Qdrant namespace marker does not match the expected commit"
        )
    return marker


def publish_marker(
    client: Any,
    models: Any,
    *,
    namespace: str,
    expected_commit: StorageCommit | None,
    next_commit: StorageCommit,
    materialization_id: str,
) -> None:
    """Compare again, then publish the marker as the final write."""

    require_marker(
        client,
        namespace=namespace,
        expected_commit=expected_commit,
    )
    client.upsert(
        collection_name=CONTROL_COLLECTION,
        points=[
            models.PointStruct(
                id=marker_id(namespace),
                vector=[1.0],
                payload={
                    "namespace": namespace,
                    "commit": next_commit.to_dict(),
                    "materialization_id": materialization_id,
                },
            )
        ],
        wait=True,
    )
    marker = read_marker(client, namespace=namespace)
    expected = QdrantMarker(
        commit=next_commit,
        materialization_id=materialization_id,
    )
    if marker != expected:
        raise RuntimeError("Qdrant namespace marker publication was not durable")


def clear_marker(
    client: Any,
    models: Any,
    *,
    namespace: str,
    expected_commit: StorageCommit | None,
) -> None:
    """Delete a marker last when rebuilding to the logical initial state."""

    require_marker(
        client,
        namespace=namespace,
        expected_commit=expected_commit,
    )
    client.delete(
        collection_name=CONTROL_COLLECTION,
        points_selector=models.PointIdsList(points=[marker_id(namespace)]),
        wait=True,
    )
    if read_marker(client, namespace=namespace) is not None:
        raise RuntimeError("Qdrant namespace marker deletion was not durable")


__all__ = [
    "QdrantMarker",
    "clear_marker",
    "marker_id",
    "publish_marker",
    "read_marker",
    "require_marker",
]
