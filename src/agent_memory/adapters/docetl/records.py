"""The frame ↔ record boundary, and the rules T5-T7 that keep row identity and answer intact across it.

DocETL's operators speak `list[dict]`, so the frame index travels in a reserved key and is restored on the
way back. A record is the engine's copy of the input, not the input — a value that passed through a float
comes back rounded — so a record is read for two things only: the row's reserved index key, and the
columns an operator declared. Stated once, cited by label below and from `dtypes.py`:

  T5  the row index is carried like a column: kept under the source index's dtype, and a float-coerced
      integer index is refused, because a label that lost its digits cannot name its row
  T6  a declared output the engine did not produce raises: a missing output is a contract violation,
      not a null
  T7  no records still yields the published columns with the carried dtypes, not a bare column list

T1-T4 belong to a rebuild and are stated in `dtypes.py`; `publish_rows` satisfies T1 here by selecting
rather than rebuilding. `require_one_record_per_row` states the guarantee conversion alone cannot: `map`
drops an answer that fails validation and says nothing, so the count is the only evidence a row survived.
"""

from __future__ import annotations

from collections import deque
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
    source: pd.DataFrame,
    written: Iterable[str],
) -> pd.DataFrame:
    """Return the frame to publish for an operator that answered one record per source row.

    Only `written` is read out of the records; every other column is carried from `source`. `written` is
    the operator's declarations, in order, and they land after the source's — the order
    `columns.output_column_names` publishes. T7: no record means no position, so the result is the
    source's shape with no rows.
    """

    declared = list(written)
    return publish_rows(
        source,
        source_positions(source, records, index_key=index_key),
        written={column: _written(records, column) for column in declared},
    )


def publish_rows(
    source: pd.DataFrame,
    positions: Sequence[int],
    *,
    written: Mapping[str, Sequence[Any]],
    index_dtype: str | None = None,
) -> pd.DataFrame:
    """Return the frame to publish: `source`'s rows at `positions`, plus the columns an operator wrote.

    A carried column is *selected*, never rebuilt, which is how T1 holds here without any dtype work.
    `written` is applied positionally, so a repeated index label cannot misalign it; `index_dtype`
    re-wraps the carried labels for an operator that must repeat one per emitted row.
    """

    frame = source.iloc[list(positions)].copy()
    for column, values in written.items():
        values = list(values)
        # Reason: an empty answer has no rows to infer from, and pandas gives an empty list a float
        #   carrier. `object` is the dtype the record path has always published for that shape.
        frame[column] = values if values else pd.Series(dtype=object)
    if index_dtype is not None:
        frame.index = pd.Index(frame.index, dtype=index_dtype)
    return frame


def source_positions(
    source: pd.DataFrame, records: Sequence[Mapping[str, Any]], *, index_key: str
) -> list[int]:
    """Return each record's position in `source`, in record order.

    T5: the label decides the row, a float label for an integer index is refused, and a repeated label
    fills in source order. Public because `sem_flat_map` names source rows one per emitted entry — a list
    of positions — before handing them to `publish_rows`.
    """

    available: dict[Any, deque[int]] = {}
    for position, label in enumerate(source.index):
        available.setdefault(label, deque()).append(position)

    integer_index = pd.api.types.is_integer_dtype(source.index.dtype)
    positions: list[int] = []
    for position, record in enumerate(records):
        if index_key not in record:
            raise KeyError(
                f"record {position} is missing {index_key!r}: the operator dropped row identity"
            )
        label = record[index_key]
        if integer_index and isinstance(label, float):
            # Requirement: the row index is carried like a column. A float label cannot name the row it
            #   was taken from: `9007199254740993` reads back `9007199254740992`.
            raise ValueError(
                f"{index_key} came back as a float ({label!r}) for an integer index"
            )
        candidates = available.get(label)
        if not candidates:
            raise KeyError(
                f"record {position} carries {label!r}, which is not a row of the source: an operator "
                "may not invent a row"
            )
        positions.append(candidates.popleft())
    return positions


def _written(records: Sequence[Mapping[str, Any]], column: str) -> list[Any]:
    """Return one declared column's values; T6 refuses a record that did not answer."""

    values: list[Any] = []
    for position, record in enumerate(records):
        if column not in record:
            raise KeyError(
                f"record {position} has no {column!r}: the operator declared it and did not return it"
            )
        values.append(record[column])
    return values


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
