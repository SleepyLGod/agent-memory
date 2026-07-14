"""LOTUS adapter source binding helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_memory.policy.logical import QueryExpr


def input_frame(inputs: Mapping[str, Any], name: str) -> Any:
    """Return an input DataFrame by runtime state key."""

    try:
        return inputs[name]
    except KeyError as error:
        raise KeyError(f"Missing adapter input {name!r}") from error


def execute_log(inputs: Mapping[str, Any]) -> Any:
    """Bind the logical log source to runtime log state."""

    return input_frame(inputs, "log")


def execute_materialized_view(query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
    """Bind a materialized view source by view name."""

    return input_frame(inputs, str(query.params["name"]))
