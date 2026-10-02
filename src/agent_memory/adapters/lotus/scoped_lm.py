"""Context-local model routing for LOTUS's global operator-cache lookup."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class ScopedLM:
    """Keep LOTUS settings stable while independent lanes own separate LM state."""

    def __init__(self, model: Any) -> None:
        self._current: ContextVar[Any] = ContextVar("lotus_execution_lm", default=model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current.get(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._current.get()(*args, **kwargs)

    @contextmanager
    def scope(self, model: Any) -> Iterator[None]:
        """Route model calls and native cache/stat accesses within this context."""
        token = self._current.set(model)
        try:
            yield
        finally:
            self._current.reset(token)
