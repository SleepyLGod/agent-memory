"""Stable serialization helpers shared by logical planners."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from agent_memory.policy.logical import QueryExpr


def stable_json(value: Any) -> str:
    """Serialize immutable planner values without process-specific identity."""

    return json.dumps(
        stable_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_value(value: Any) -> Any:
    """Convert immutable query values into deterministic JSON data."""

    if isinstance(value, QueryExpr):
        return {
            "op": value.op,
            "inputs": [stable_value(item) for item in value.inputs],
            "params": stable_value(value.params),
        }
    if isinstance(value, Mapping):
        return {
            str(key): stable_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [stable_value(item) for item in value]
    if is_dataclass(value):
        return {
            "type": type(value).__qualname__,
            "fields": {
                field.name: stable_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


__all__ = ["stable_json", "stable_value"]
