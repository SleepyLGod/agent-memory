"""Executor-owned identity decisions, scoped to registered entity maintenance."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

IDENTITY_DECISIONS_INPUT = "__physical_identity_decisions"


@dataclass
class IdentityDecisions:
    """Store a complete selection mask, never a prior row or occurrence ID."""

    values: dict[str, bool]

    @staticmethod
    def key(*semantic_input: Any) -> str:
        return sha256(json.dumps(semantic_input, ensure_ascii=False, sort_keys=True,
                                 default=str).encode()).hexdigest()

    def lookup(self, key: str, size: int) -> tuple[int, ...] | None:
        keys = [f"identity-v1:{key}:{size}:{i}" for i in range(size)]
        if not keys or any(k not in self.values for k in keys):
            return None
        return tuple(i for i, k in enumerate(keys) if self.values[k])

    def store(self, key: str, size: int, selected: Sequence[int]) -> None:
        if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < size for i in selected):
            raise ValueError("identity selection must contain distinct candidate positions")
        self.values.update({f"identity-v1:{key}:{size}:{i}": i in selected for i in range(size)})


current_identity_decisions: ContextVar[IdentityDecisions | None] = ContextVar(
    "entity_identity_decisions", default=None,
)


@contextmanager
def identity_scope(values: dict[str, bool]) -> Iterator[None]:
    """Keep nested operators on the current executor transaction's decision map."""
    token = current_identity_decisions.set(IdentityDecisions(values))
    try:
        yield
    finally:
        current_identity_decisions.reset(token)
