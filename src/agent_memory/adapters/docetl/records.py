"""Frame ↔ record conversion for DocETL's operators.

DocETL's operators speak ``list[dict]``: no index, and no room for one. The frame index therefore
travels in a reserved key inside each record, stripped again on the way back.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

# Reason: DocETL's operators exchange dicts, not frames, so the index needs a key in the record
#   (`docetl.operations.base::BaseOperation.execute`, the contract every operator implements).
# Required by: the runtime's row-local nodes, which file results by row occurrence
#   (`src/agent_memory/runtime/executor.py::PolicyExecutor._execute_semantic_row`).
# Based on: the LOTUS adapter's join id columns
#   (`src/agent_memory/adapters/lotus/relational.py::_unused_column_name`).
INDEX_KEY_BASE = "__agent_memory_index__"


def unused_index_key(columns: Iterable[str]) -> str:
    """Return an index key that none of `columns` uses, so the carrier cannot overwrite data."""

    taken = set(columns)
    candidate = INDEX_KEY_BASE
    while candidate in taken:
        candidate = f"_{candidate}"
    return candidate


def to_records(frame: pd.DataFrame, *, index_key: str) -> list[dict[str, Any]]:
    """Return `frame` as records, carrying the index under `index_key`.

    `index_key` has to be free, which is what `unused_index_key` is for. A key taken from another
    frame's columns is caught here rather than overwriting a real column.
    """

    if index_key in frame.columns:
        raise ValueError(f"frame already has a column named {index_key!r}")
    if frame.columns.duplicated().any():
        # a dict holds one value per key, so a repeated column name would be dropped silently
        raise ValueError("frame has duplicate column names, which records cannot represent")

    projected = frame.astype(object)
    # prompts and JSON have no NaN or NaT, so missing values become None
    clean = projected.where(pd.notna(projected), None)
    columns = list(frame.columns)
    records: list[dict[str, Any]] = [
        dict(zip(columns, row, strict=True))
        for row in clean.itertuples(index=False, name=None)
    ]
    for record, index in zip(records, frame.index, strict=True):
        record[index_key] = index
    return records


def from_records(
    records: Sequence[dict[str, Any]],
    *,
    index_key: str,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return `records` as a frame, restoring the index they carry under `index_key`.

    `columns` is used only when there are no records, and `index_key` is not consulted at all then.
    """

    if not records:
        # `columns` arrives as a frame's Index, and an Index has no truth value
        return pd.DataFrame(columns=list(columns) if columns is not None else [])

    rows: list[dict[str, Any]] = []
    indexes: list[Any] = []
    for position, record in enumerate(records):
        if index_key not in record:
            raise KeyError(
                f"record {position} is missing {index_key!r}: the operator dropped row identity"
            )
        row = dict(record)
        indexes.append(row.pop(index_key))
        rows.append(row)

    # each column's inferred dtype picks its own missing marker: None, NaN or NaT
    return pd.DataFrame(rows, index=pd.Index(indexes))
