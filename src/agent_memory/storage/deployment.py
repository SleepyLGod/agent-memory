"""Runtime binding between a storage plan and a physical connector."""

from __future__ import annotations

from dataclasses import dataclass

from .connector import StorageConnector
from .statements import StatementSet


@dataclass(frozen=True)
class StorageDeployment:
    """Bind immutable storage statements to a connector and namespace."""

    connector: StorageConnector
    statements: StatementSet
    namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.statements, StatementSet):
            raise TypeError("storage deployment statements must be a StatementSet")
        if not self.statements.statements:
            raise ValueError("storage deployment requires at least one statement")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("storage deployment namespace must be a non-empty string")
        if self.connector is None:
            raise TypeError("storage deployment connector cannot be None")
