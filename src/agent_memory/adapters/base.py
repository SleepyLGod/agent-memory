"""Execution adapter protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from agent_memory.policy.logical import QueryExpr


class ExecutionAdapter(Protocol):
    """Boundary for executing logical query expressions.

    Adapters are runtime backends, not part of policy authoring. The v0.0 interface
    defines this protocol without providing execution behavior.
    """

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute a logical query expression against runtime inputs.

        Return a frame with exactly the planner's `output_columns`, keep the input index for row-local
        semantics, and never mutate `inputs` — the runtime reuses those frames for other nodes.
        """
        ...
