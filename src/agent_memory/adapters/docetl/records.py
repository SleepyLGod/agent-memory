"""Frame ↔ record conversion, and the check that the engine returned a record per row.

DocETL's operators speak `list[dict]`, so the frame index travels in a reserved key and is restored on the
way back; `require_one_record_per_row` is the guarantee that conversion alone cannot state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.docetl.reserved import reject_reserved


def to_records(frame: pd.DataFrame, *, index_key: str) -> list[dict[str, Any]]:
    """Return `frame` as records, carrying the index under `index_key`.

    `index_key` is one of the adapter's reserved names, and a frame carrying any of them is refused rather
    than worked around: an injected key appears in an operation's prompt schema, so it may not vary per frame.
    """

    reject_reserved(frame.columns)
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


def require_one_record_per_row(
    source: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    *,
    op: str,
    index_key: str,
) -> None:
    """Raise if the engine returned a different number of records than rows it was given.

    `map` drops a record whose answer fails validation and reports nothing, so this count is the
    only evidence that a row survived.
    """

    if len(records) == len(source):
        return

    returned = {record.get(index_key) for record in records}
    # a repeated index makes the naming ambiguous, so the count stands alone in that case
    missing = [index for index in source.index if index not in returned]
    detail = f"; missing {missing}" if missing and len(source) == len(set(source.index)) else ""
    raise RuntimeError(
        f"{op} received {len(records)} records for {len(source)} input rows: an answer failed "
        f"validation and its row was dropped{detail}"
    )
