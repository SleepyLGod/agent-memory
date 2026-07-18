"""Backend-neutral transaction protocols for materialized storage sinks."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import pandas as pd

from .statements import InsertStatement


class StorageTransaction(Protocol):
    """One atomic connector transaction for a memory update step."""

    def write(
        self,
        statement: InsertStatement,
        *,
        inserted_rows: pd.DataFrame,
        retracted_rows: pd.DataFrame,
    ) -> None:
        """Stage one sink's exact inserted and retracted relation rows."""

        ...


class StorageConnector(Protocol):
    """Connector capable of opening namespace-scoped storage transactions."""

    def transaction(
        self,
        *,
        namespace: str,
    ) -> AbstractContextManager[StorageTransaction]:
        """Open a transaction that commits or rolls back on context exit."""

        ...
