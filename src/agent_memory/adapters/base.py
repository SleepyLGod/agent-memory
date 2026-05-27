"""Execution adapter protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from agent_memory.logical import QueryExpr


class ExecutionAdapter(Protocol):
    """Boundary for executing logical query expressions.

    Adapters are runtime backends, not part of policy authoring. The v0.0 interface
    defines this protocol without providing execution behavior.
    """

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute a logical query expression against runtime inputs."""
        ...
