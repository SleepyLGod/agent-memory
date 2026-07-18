"""Connector-backed storage target descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .schema import Schema


_DEPLOYMENT_OR_SECRET_OPTION_NAMES = frozenset(
    {
        "uri",
        "url",
        "host",
        "port",
        "user",
        "username",
        "password",
        "passwd",
        "token",
        "secret",
        "auth",
        "credential",
        "credentials",
        "api_key",
        "access_key",
        "private_key",
        "tls",
        "ssl",
    }
)


def _validate_option_name(key: object) -> str:
    """Return one public plan-option name or reject deployment configuration."""

    if not isinstance(key, str) or not key:
        raise ValueError("table option names must be non-empty strings")
    parts = tuple(part.casefold().replace("-", "_") for part in key.split("."))
    if parts[0] != "property" and any(
        part in _DEPLOYMENT_OR_SECRET_OPTION_NAMES for part in parts
    ):
        raise ValueError(
            f"table option {key!r} is deployment or secret configuration; "
            "put it on the runtime connector"
        )
    return key


@dataclass(frozen=True)
class TableDescriptor:
    """Immutable description of one physical storage target."""

    connector: str
    schema: Schema
    options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.connector, str) or not self.connector:
            raise ValueError("table connector must be a non-empty string")
        if not isinstance(self.schema, Schema):
            raise TypeError("table schema must be a Schema")
        normalized: dict[str, str] = {}
        for key, value in self.options.items():
            normalized_key = _validate_option_name(key)
            if not isinstance(value, str):
                raise TypeError("table option values must be strings")
            normalized[normalized_key] = value
        object.__setattr__(self, "options", MappingProxyType(normalized))

    @classmethod
    def for_connector(cls, connector: str) -> _TableDescriptorBuilder:
        """Return a Flink-style builder for one connector target."""

        return _TableDescriptorBuilder(connector)

    def __hash__(self) -> int:
        return hash(
            (
                self.connector,
                self.schema,
                tuple(sorted(self.options.items())),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "connector": self.connector,
            "schema": self.schema.to_dict(),
            "options": dict(self.options),
        }


class _TableDescriptorBuilder:
    """Mutable authoring builder whose result is an immutable descriptor."""

    def __init__(self, connector: str) -> None:
        if not isinstance(connector, str) or not connector:
            raise ValueError("table connector must be a non-empty string")
        self._connector = connector
        self._schema: Schema | None = None
        self._options: dict[str, str] = {}

    def schema(self, schema: Schema) -> _TableDescriptorBuilder:
        """Attach the target schema."""

        if not isinstance(schema, Schema):
            raise TypeError("table schema must be a Schema")
        if self._schema is not None:
            raise ValueError("table schema is already defined")
        self._schema = schema
        return self

    def option(self, key: str, value: str) -> _TableDescriptorBuilder:
        """Add one non-secret connector target option."""

        normalized_key = _validate_option_name(key)
        if not isinstance(value, str):
            raise TypeError("table option value must be a string")
        if normalized_key in self._options:
            raise ValueError(f"table option already exists: {normalized_key!r}")
        self._options[normalized_key] = value
        return self

    def build(self) -> TableDescriptor:
        """Build an immutable target descriptor."""

        if self._schema is None:
            raise ValueError("table descriptor requires a schema")
        return TableDescriptor(
            connector=self._connector,
            schema=self._schema,
            options=self._options,
        )
