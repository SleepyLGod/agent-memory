"""Public authoring API for the agent-memory v0.0 interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .planner.differential_policy import DifferentiatedPolicy, PolicyDifferentiator
from .policy.logical import MemorySpec, MemoryView, QueryExpr
from .policy.relation import Log, OverRelation, Relation, WindowedRelation
from .storage.deployment import StorageDeployment


def _freeze_metadata(value: Any) -> Any:
    """Recursively freeze message metadata into immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_metadata(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _hashable_metadata(value: Any) -> Any:
    """Convert frozen metadata into a canonical hashable shape."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((key, _hashable_metadata(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_hashable_metadata(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class Message:
    """Minimal message or event accepted by ``Memory.add``."""

    content: str
    role: str | None = None
    timestamp: str | None = None
    session_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def __hash__(self) -> int:
        return hash(
            (
                self.content,
                self.role,
                self.timestamp,
                self.session_id,
                _hashable_metadata(self.metadata),
            )
        )


MessageInput = str | Message | Mapping[str, Any]


class Memory:
    """Base class for declarative memory policies."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "query" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare retrieval_query = ... instead of "
                "overriding query(...)."
            )
        if "retrieval_query" in cls.__dict__ and not isinstance(
            cls.__dict__["retrieval_query"],
            Relation,
        ):
            raise TypeError("retrieval_query must be a Relation")

    def __init__(
        self,
        *,
        adapter: Any | None = None,
        storage: StorageDeployment | None = None,
    ) -> None:
        from .runtime import MemoryRuntime

        policy = (
            self.__class__.differentiate_policy()
            if storage is None
            else PolicyDifferentiator().differentiate(
                self.__class__.spec(),
                statements=storage.statements,
            )
        )
        self._runtime = MemoryRuntime(
            policy,
            adapter=adapter,
            storage=storage,
        )

    def add(self, message: MessageInput) -> None:
        """Append an end-user message or event to memory.

        A string is Message(content=...) sugar. Runtime or agent-platform
        adapters will eventually normalize Message objects into the configured
        log shape; Message.content maps to the log column named "message".
        """

        return self._runtime.add(message)

    def query(self, text: str) -> Any:
        """Run the default policy-defined retrieval query."""

        return self._runtime.execute_retrieval_query("default", text)

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
    def differentiate_policy(cls) -> DifferentiatedPolicy:
        """Return the in-memory differentiated policy artifact for this class."""

        cached = cls.__dict__.get("_agent_memory_differentiated_policy")
        if isinstance(cached, DifferentiatedPolicy):
            return cached

        policy = PolicyDifferentiator().differentiate(cls.spec())
        setattr(cls, "_agent_memory_differentiated_policy", policy)
        return policy

    @classmethod
    def _collect_spec(cls) -> MemorySpec:
        log: Log | None = None
        private_relations: dict[str, QueryExpr] = {}
        retrieval_queries: dict[str, QueryExpr] = {}
        views: dict[str, MemoryView] = {}

        for name, value in vars(cls).items():
            if isinstance(value, Log):
                if log is not None:
                    raise ValueError(f"{cls.__name__} declares multiple Log relations")
                log = value
                continue

            if name == "retrieval_query":
                if not isinstance(value, Relation):
                    raise TypeError("retrieval_query must be a Relation")
                retrieval_queries["default"] = value.expr
                continue

            if not isinstance(value, Relation):
                if isinstance(value, WindowedRelation):
                    raise TypeError(
                        f"{name} is a WindowedRelation; call process_window(...) "
                        "before declaring it as a memory relation"
                    )
                if isinstance(value, OverRelation):
                    raise TypeError(
                        f"{name} is an OverRelation; call array_agg(...) or sem_agg(...) "
                        "before declaring it as a memory relation"
                    )
                continue

            if name.startswith("_"):
                private_relations[name] = value.expr
            else:
                views[name] = MemoryView(name=name, query=value.expr)

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

        return MemorySpec(
            log=log,
            views=views,
            private_relations=private_relations,
            retrieval_queries=retrieval_queries,
        )
