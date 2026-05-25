"""Runtime shell for the current memory interface layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_memory.logical import MemorySpec
from agent_memory.message import MessageInput

if TYPE_CHECKING:
    from agent_memory.relation import Relation


class MemoryRuntime:
    """Runtime orchestrator interface for memory instances.

    This v0.0 module is an interface layer. It intentionally does not fake
    log ingestion, storage, scheduling, or adapter behavior.
    """

    def __init__(self, spec: MemorySpec, *, adapter: Any | None = None) -> None:
        self.spec = spec
        self.adapter = adapter
        self._state: dict[str, Any] = {}

    def add(self, message: MessageInput) -> None:
        """Append an end-user message or event to the source log.

        A string is Message(content=...) sugar. Future runtime code will
        normalize Message.content into the log column named "message", preserve
        same-name message fields when available, and maintain derived views. The
        current interface layer does not execute.
        """

        raise NotImplementedError(
            "Memory add/log maintenance is not implemented in the current v0.0 interface layer."
        )

    def execute_query(self, plan: "Relation") -> Any:
        """Execute a policy-defined semantic query plan.

        The runtime does not choose a retrieval view. Concrete memory policies
        construct the query plan first, then hand it to this future execution
        hook.
        """

        raise NotImplementedError(
            "Memory query plan execution is not implemented in the current v0.0 interface layer."
        )
