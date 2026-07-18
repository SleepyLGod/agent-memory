"""Immutable bindings from logical relations to storage targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.relation import Relation
from agent_memory.policy.schema import output_columns

from .table import TableDescriptor


@dataclass(frozen=True)
class InsertStatement:
    """One logical relation bound to one physical target."""

    statement_id: str
    target: TableDescriptor
    query: QueryExpr

    def __post_init__(self) -> None:
        if not isinstance(self.statement_id, str) or not self.statement_id:
            raise ValueError("statement_id must be a non-empty string")
        if not isinstance(self.target, TableDescriptor):
            raise TypeError("statement target must be a TableDescriptor")
        if not isinstance(self.query, QueryExpr):
            raise TypeError("statement query must be a QueryExpr")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "statement_id": self.statement_id,
            "target": self.target.to_dict(),
            "query": _serializable_value(self.query),
        }


@dataclass(frozen=True)
class StatementSet:
    """Immutable set of relation-to-target insert statements."""

    statements: tuple[InsertStatement, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "statements", tuple(self.statements))
        identifiers = tuple(statement.statement_id for statement in self.statements)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("storage statement identifiers must be unique")

    def add_insert(
        self,
        target: TableDescriptor,
        relation: Relation,
    ) -> StatementSet:
        """Return a new statement set with one additional sink binding."""

        if not isinstance(target, TableDescriptor):
            raise TypeError("insert target must be a TableDescriptor")
        if not isinstance(relation, Relation):
            raise TypeError("insert relation must be a Relation")
        relation_columns = output_columns(relation.expr)
        target_columns = tuple(column.name for column in target.schema.columns)
        if relation_columns != target_columns:
            raise ValueError(
                "insert relation columns must exactly match target schema; "
                f"relation={relation_columns}, target={target_columns}"
            )
        used_ids = {statement.statement_id for statement in self.statements}
        statement_index = len(self.statements)
        statement_id = f"sink_{statement_index:04d}"
        while statement_id in used_ids:
            statement_index += 1
            statement_id = f"sink_{statement_index:04d}"
        statement = InsertStatement(
            statement_id=statement_id,
            target=target,
            query=relation.expr,
        )
        return StatementSet(statements=(*self.statements, statement))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "statements": [statement.to_dict() for statement in self.statements]
        }


def _serializable_value(value: Any) -> Any:
    """Convert immutable logical values into deterministic JSON data."""

    if isinstance(value, QueryExpr):
        return {
            "op": value.op,
            "inputs": [_serializable_value(item) for item in value.inputs],
            "params": _serializable_value(value.params),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _serializable_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_serializable_value(item) for item in value]
    if is_dataclass(value):
        return {
            "type": type(value).__qualname__,
            "fields": {
                field.name: _serializable_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"storage statement contains non-serializable value: {type(value).__name__}")
