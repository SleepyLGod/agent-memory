"""Logical query objects for declarative memory policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .api import Log
    from .retrieval import RetrievalQuery


# These helpers keep logical plans immutable and cache-safe. This is the same
# basic invariant used by query planners and compilers: once a plan node is
# constructed, later code should not be able to mutate its parameters behind the
# planner's back.
def _freeze_param(value: Any) -> Any:
    """Recursively freeze values stored inside logical expression parameters."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_param(item) for key, item in value.items()}
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_param(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_param(item) for item in value)
    return value


# Planner memoization and expression comparison need stable hash keys. This
# canonicalizes nested params so equivalent dicts with different insertion order
# compare and hash the same way.
def _hashable_param(value: Any) -> Any:
    """Convert frozen parameter values into a hashable canonical shape."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((key, _hashable_param(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_param(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_hashable_param(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class ColumnSpec:
    """Description for a dataframe-style output column."""

    name: str
    description: str | None = None


@dataclass(frozen=True)
class QueryExpr:
    """Internal immutable query tree node.

    QueryExpr records a dataframe-style operator tree for a full view
    definition query Q or a derived differential query DeltaQ. It does not
    execute queries, call models, or mutate memory state.
    """

    op: str
    inputs: tuple["QueryExpr", ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "params", _freeze_param(self.params))

    def __hash__(self) -> int:
        return hash((self.op, self.inputs, _hashable_param(self.params)))


@dataclass(frozen=True)
class UserQuery:
    """Runtime-bound placeholder for end-user retrieval query text."""

    name: str = "query"


@dataclass(frozen=True)
class MemoryView:
    """Named derived view declaration V = Q(D) collected from a Memory class."""

    name: str
    query: QueryExpr


@dataclass(frozen=True)
class MemorySpec:
    """Collected declaration for one memory class."""

    log: "Log"
    views: Mapping[str, MemoryView]
    private_relations: Mapping[str, QueryExpr]
    retrieval_queries: Mapping[str, QueryExpr | "RetrievalQuery"] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", MappingProxyType(dict(self.views)))
        object.__setattr__(
            self,
            "private_relations",
            MappingProxyType(dict(self.private_relations)),
        )
        object.__setattr__(
            self,
            "retrieval_queries",
            MappingProxyType(dict(self.retrieval_queries)),
        )
