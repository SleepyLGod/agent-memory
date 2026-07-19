"""Connector-backed storage target descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

from .schema import Schema


@runtime_checkable
class ConnectorMapping(Protocol):
    """Connector-owned, serializable mapping from logical rows to storage."""

    connector: ClassVar[str]

    def validate(self, schema: Schema) -> None:
        """Validate the mapping against the logical sink schema."""

        ...

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""

        ...


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
    mapping: ConnectorMapping | None = None

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
        if self.mapping is not None:
            if not isinstance(self.mapping, ConnectorMapping):
                raise TypeError("table mapping must implement ConnectorMapping")
            if self.mapping.connector != self.connector:
                raise ValueError(
                    "table mapping connector must match the table connector"
                )
            self.mapping.validate(self.schema)

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
                None
                if self.mapping is None
                else _freeze_serializable(self.mapping.to_dict()),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = {
            "connector": self.connector,
            "schema": self.schema.to_dict(),
            "options": dict(self.options),
        }
        if self.mapping is not None:
            result["mapping"] = self.mapping.to_dict()
        return result


class _TableDescriptorBuilder:
    """Mutable authoring builder whose result is an immutable descriptor."""

    def __init__(self, connector: str) -> None:
        if not isinstance(connector, str) or not connector:
            raise ValueError("table connector must be a non-empty string")
        self._connector = connector
        self._schema: Schema | None = None
        self._options: dict[str, str] = {}
        self._mapping: ConnectorMapping | None = None

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

    def mapping(self, mapping: ConnectorMapping) -> _TableDescriptorBuilder:
        """Attach one connector-specific row mapping."""

        if not isinstance(mapping, ConnectorMapping):
            raise TypeError("table mapping must implement ConnectorMapping")
        if self._mapping is not None:
            raise ValueError("table mapping is already defined")
        self._mapping = mapping
        return self

    def build(self) -> TableDescriptor:
        """Build an immutable target descriptor."""

        if self._schema is None:
            raise ValueError("table descriptor requires a schema")
        return TableDescriptor(
            connector=self._connector,
            schema=self._schema,
            options=self._options,
            mapping=self._mapping,
        )


def _freeze_serializable(value: Any) -> Any:
    """Convert JSON data into a stable hashable value."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_serializable(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_serializable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "table mapping contains a non-serializable value: "
        f"{type(value).__name__}"
    )
