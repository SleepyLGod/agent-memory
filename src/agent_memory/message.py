"""Input message objects for memory append APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


# Message is a frozen value object. These helpers prevent nested metadata from
# remaining mutable after construction and give cache/dedup code a stable hash.
def _freeze_metadata(value: Any) -> Any:
    """Recursively freeze message metadata into value-object-safe containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_metadata(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


# Canonical hashes make semantically equal metadata hash the same even when a
# caller provided dict keys in a different insertion order.
def _hashable_metadata(value: Any) -> Any:
    """Convert frozen metadata into a canonical hashable shape."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((key, _hashable_metadata(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_hashable_metadata(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class Message:
    """Minimal message/event input accepted by Memory.add.

    Message is an input boundary, not a user-defined schema system. Future
    agent-platform adapters can translate their native message objects into
    this shape before appending to memory.
    """

    content: str
    role: str | None = None
    timestamp: str | None = None
    session_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def __hash__(self) -> int:
        return hash(
            (
                self.content,
                self.role,
                self.timestamp,
                self.session_id,
                _hashable_metadata(self.metadata),
            )
        )


MessageInput = str | Message | Mapping[str, Any]


__all__ = ["Message", "MessageInput"]
