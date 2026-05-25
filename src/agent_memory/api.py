"""Public authoring API for the agent-memory v0.0 interface."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

from .logical import ColumnSpec, MemorySpec, MemoryView, RelationExpr
from .message import MessageInput
from .relation import Relation
from .runtime import MemoryRuntime


class Log(Relation):
    """Base source relation for memory policies.

    A Log is the input table for a memory class. It is collected as the source
    relation, not as a derived MemoryView.
    """

    def __init__(self, columns: Mapping[str, str] | None = None) -> None:
        column_defs = tuple(
            ColumnSpec(name=name, description=description)
            for name, description in (
                columns or {"message": "Raw memory log message."}
            ).items()
        )
        super().__init__(
            RelationExpr(
                op="log",
                params={"columns": column_defs},
            )
        )


class Memory:
    """Base class for declarative memory policies."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Policy query methods may return logical Relation plans; execute those
        # plans implicitly so policy authors never call runtime internals.
        user_query = cls.__dict__.get("query")
        if user_query is None:
            return

        @wraps(user_query)
        def wrapped_query(self: "Memory", *args: Any, **kwargs: Any) -> Any:
            result = user_query(self, *args, **kwargs)
            if isinstance(result, Relation):
                return self._runtime.execute_query(result)
            return result

        cls.query = wrapped_query

    def __init__(self, *, adapter: Any | None = None) -> None:
        self._runtime = MemoryRuntime(self.__class__.spec(), adapter=adapter)

    def add(self, message: MessageInput) -> None:
        """Append an end-user message or event to memory.

        A string is Message(content=...) sugar. Runtime or agent-platform
        adapters will eventually normalize Message objects into the configured
        log shape; Message.content maps to the log column named "message".
        """

        return self._runtime.add(message)

    def query(self, query: str) -> Any:
        """Run this memory's policy-defined retrieval query.

        Base Memory does not choose a view automatically and runtime does not
        route queries. Concrete memory policies must override this method to
        define the semantic retrieval query they want to expose.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must override query(...) to define policy-owned "
            "retrieval behavior."
        )

    @classmethod
    def spec(cls) -> MemorySpec:
        """Return the collected memory specification for this Memory subclass.

        The collected spec is cached on the class. Policy class declarations are
        treated as immutable after first collection, so dynamic class attribute
        edits are intentionally not reflected.
        """

        cached = cls.__dict__.get("_agent_memory_spec")
        if isinstance(cached, MemorySpec):
            return cached

        spec = cls._collect_spec()
        setattr(cls, "_agent_memory_spec", spec)
        return spec

    @classmethod
    def _collect_spec(cls) -> MemorySpec:
        log: Log | None = None
        private_relations: dict[str, Relation] = {}
        views: dict[str, MemoryView] = {}

        for name, value in vars(cls).items():
            if isinstance(value, Log):
                if log is not None:
                    raise ValueError(f"{cls.__name__} declares multiple Log relations")
                log = value
                continue

            if not isinstance(value, Relation):
                continue

            if name.startswith("_"):
                private_relations[name] = value
            else:
                views[name] = MemoryView(name=name, relation=value)

        if log is None:
            inherited_log_owner = next(
                (
                    base
                    for base in cls.__mro__[1:]
                    if any(isinstance(value, Log) for value in vars(base).values())
                ),
                None,
            )
            if inherited_log_owner is not None:
                raise ValueError(
                    f"{cls.__name__} inherits memory declarations from "
                    f"{inherited_log_owner.__name__}, but v0.0 does not merge "
                    "inherited memory declarations. Redeclare 'log' in this "
                    "class body or define a standalone Memory subclass."
                )
            raise ValueError(f"{cls.__name__} must declare a Log relation")

        return MemorySpec(log=log, views=views, private_relations=private_relations)
