"""LOTUS execution adapter shell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_memory.logical import RelationExpr


@dataclass
class LotusAdapter:
    """Execution adapter shell for future LOTUS-backed semantic operators.

    The class records adapter configuration only. It does not import LOTUS or
    execute semantic operators in the current v0.0 interface layer.
    """

    model: str | None = None

    def execute(self, expr: RelationExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute a logical expression through LOTUS."""

        raise NotImplementedError(
            "LOTUS execution is not implemented in the current v0.0 interface layer."
        )
