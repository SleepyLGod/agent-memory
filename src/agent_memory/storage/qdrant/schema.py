"""Embedded Qdrant collection preparation and validation."""

from __future__ import annotations

from typing import Any


CONTROL_COLLECTION = "_agent_memory_storage_commits"


def ensure_collection(
    client: Any,
    models: Any,
    *,
    collection: str,
    dimensions: int,
) -> None:
    """Create one cosine collection or validate its existing vector shape."""

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        return
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        raise ValueError("Qdrant named-vector collections are not supported")
    size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    if size != dimensions or _distance_name(distance) != "cosine":
        raise ValueError(
            f"Qdrant collection {collection!r} must use cosine vectors "
            f"with {dimensions} dimensions"
        )


def ensure_control_collection(client: Any, models: Any) -> None:
    """Prepare the internal namespace commit-marker collection."""

    ensure_collection(
        client,
        models,
        collection=CONTROL_COLLECTION,
        dimensions=1,
    )


def _distance_name(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).casefold()


__all__ = ["CONTROL_COLLECTION", "ensure_collection", "ensure_control_collection"]
