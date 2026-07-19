"""Backend-neutral transaction protocols for materialized storage sinks."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from .statements import InsertStatement, StatementSet


class StorageConflictError(RuntimeError):
    """Raised when a physical namespace is not at the expected commit."""


@dataclass(frozen=True)
class StorageCommit:
    """Logical checkpoint marker shared by the runtime and a storage backend."""

    plan_fingerprint: str
    lineage_id: str
    commit_sequence: int
    source_row_count: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.plan_fingerprint, "plan fingerprint"),
            (self.lineage_id, "lineage id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"storage commit {name} must be a non-empty string")
        for value, name in (
            (self.commit_sequence, "commit sequence"),
            (self.source_row_count, "source row count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"storage commit {name} must be a non-negative integer"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint representation."""

        return {
            "plan_fingerprint": self.plan_fingerprint,
            "lineage_id": self.lineage_id,
            "commit_sequence": self.commit_sequence,
            "source_row_count": self.source_row_count,
        }

    @property
    def is_initial(self) -> bool:
        """Return whether this logical commit has no physical marker yet."""

        return self.commit_sequence == 0 and self.source_row_count == 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StorageCommit:
        """Parse and validate one checkpoint or connector marker."""

        if not isinstance(value, Mapping):
            raise TypeError("storage commit must be a mapping")
        expected = {
            "plan_fingerprint",
            "lineage_id",
            "commit_sequence",
            "source_row_count",
        }
        if set(value) != expected:
            raise ValueError(
                "storage commit fields must be exactly " f"{sorted(expected)}"
            )
        return cls(
            plan_fingerprint=value["plan_fingerprint"],
            lineage_id=value["lineage_id"],
            commit_sequence=value["commit_sequence"],
            source_row_count=value["source_row_count"],
        )


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

    def prepare(self, statements: StatementSet) -> None:
        """Validate targets and idempotently prepare physical schema."""

        ...

    def read_commit(self, *, namespace: str) -> StorageCommit | None:
        """Read a marker; the initial empty commit is represented by ``None``."""

        ...

    def transaction(
        self,
        *,
        namespace: str,
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> AbstractContextManager[StorageTransaction]:
        """Stage writes and atomically compare-and-set the namespace marker."""

        ...

    def rebuild(
        self,
        *,
        namespace: str,
        statements: StatementSet,
        rows_by_statement: Mapping[str, pd.DataFrame],
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> None:
        """Replace one namespace from complete logical sink rows."""

        ...
