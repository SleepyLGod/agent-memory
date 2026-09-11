"""Optional per-request persistence; no benchmark or storage dependency."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

from agent_memory.tracing.semantic import trace_scope_value


class RequestHook(Protocol):
    """Execute or recover one deterministic request occurrence."""

    def execute(
        self,
        identity: Mapping[str, Any],
        payload: Mapping[str, Any],
        send: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass
class RequestScope:
    hook: RequestHook
    identity: Mapping[str, Any]
    batch: int = 0


_SCOPE: ContextVar[RequestScope | None] = ContextVar("request_hook", default=None)


@contextmanager
def request_scope(hook: RequestHook | None, **identity: Any) -> Iterator[None]:
    """Reset deterministic call positions for one replayable execution unit."""
    token = _SCOPE.set(None if hook is None else RequestScope(hook, identity))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def next_request_batch() -> tuple[RequestHook, dict[str, Any]] | None:
    """Capture scope before dispatching worker threads."""
    scope = _SCOPE.get()
    if scope is None:
        return None
    identity = {**scope.identity, "batch": scope.batch}
    for key in ("semantic_operator", "query_digest"):
        value = trace_scope_value(key, None)
        if value is not None:
            identity[key] = value
    scope.batch += 1
    return scope.hook, identity
