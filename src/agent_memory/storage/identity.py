"""Stable physical identities shared by storage connectors."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd


_AGENT_MEMORY_UUID_NAMESPACE = uuid5(NAMESPACE_URL, "agent-memory")


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


def normalize_storage_value(value: Any) -> Any:
    """Convert pandas and scalar values into JSON-compatible storage values."""

    if isinstance(value, Mapping):
        return {
            str(key): normalize_storage_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [normalize_storage_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_storage_value(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        return normalize_storage_value(item())
    return value


def _json_identity_value(value: Any) -> Any:
    value = normalize_storage_value(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("physical UUID values must be finite")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, Mapping)):
        return value
    raise TypeError(
        "physical UUID values must be JSON-compatible; "
        f"got {type(value).__name__}"
    )


__all__ = ["normalize_storage_value", "physical_uuid"]
