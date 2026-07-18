"""Immutable physical schema descriptors for storage targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaColumn:
    """One named physical column in a storage schema."""

    name: str
    data_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("schema column name must be a non-empty string")
        if not isinstance(self.data_type, str) or not self.data_type:
            raise ValueError("schema column data_type must be a non-empty string")


@dataclass(frozen=True)
class Schema:
    """Built physical schema for one connector-backed target."""

    columns: tuple[SchemaColumn, ...]
    primary_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "primary_key", tuple(self.primary_key))
        if not self.columns:
            raise ValueError("storage schema requires at least one column")
        names = tuple(column.name for column in self.columns)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"storage schema columns must be unique: {duplicates}")
        missing = sorted(set(self.primary_key).difference(names))
        if missing:
            raise ValueError(f"primary key references unknown column(s): {missing}")
        if len(set(self.primary_key)) != len(self.primary_key):
            raise ValueError("primary key columns must be unique")

    @classmethod
    def new_builder(cls) -> _SchemaBuilder:
        """Return a Flink-style builder for a physical schema."""

        return _SchemaBuilder()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "columns": [
                {"name": column.name, "data_type": column.data_type}
                for column in self.columns
            ],
            "primary_key": list(self.primary_key),
        }


class _SchemaBuilder:
    """Mutable authoring builder whose result is an immutable ``Schema``."""

    def __init__(self) -> None:
        self._columns: list[SchemaColumn] = []
        self._primary_key: tuple[str, ...] = ()

    def column(self, name: str, data_type: str) -> _SchemaBuilder:
        """Append one physical column."""

        if any(column.name == name for column in self._columns):
            raise ValueError(f"schema column already exists: {name!r}")
        self._columns.append(SchemaColumn(name=name, data_type=data_type))
        return self

    def primary_key(self, *columns: str) -> _SchemaBuilder:
        """Declare the target's deterministic primary key columns."""

        if not columns:
            raise ValueError("primary_key requires at least one column")
        if self._primary_key:
            raise ValueError("primary key is already defined")
        names = {column.name for column in self._columns}
        missing = sorted(set(columns).difference(names))
        if missing:
            raise ValueError(f"primary key references unknown column(s): {missing}")
        if len(set(columns)) != len(columns):
            raise ValueError("primary key columns must be unique")
        self._primary_key = tuple(columns)
        return self

    def build(self) -> Schema:
        """Build an immutable schema snapshot."""

        return Schema(columns=tuple(self._columns), primary_key=self._primary_key)
