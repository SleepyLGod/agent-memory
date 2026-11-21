"""The frame ↔ record boundary, and the rules T5-T7 that keep row identity and answer intact across it.

DocETL's operators speak `list[dict]`, so each record carries its source position in a reserved key.
A record is the engine's copy of the input, not the input — a value that passed through a float
comes back rounded — so a record is read for two things only: the row's reserved index key, and the
columns an operator declared. Stated once, cited by label below and from `dtypes.py`:

  T5  select by source position, not index label: two distinct rows may have the same label
  T6  a declared output the engine did not produce raises: a missing output is a contract violation,
      not a null
  T7  no records still yields the published columns with the carried dtypes, not a bare column list

T1-T4 belong to a rebuild and are stated in `dtypes.py`; `publish_rows` satisfies T1 here by selecting
rather than rebuilding. `require_one_record_per_row` states the guarantee conversion alone cannot: `map`
drops an answer that fails validation and says nothing, so the count is the only evidence a row survived.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any

import pandas as pd

from agent_memory.adapters.docetl.reserved import reject_reserved


def to_records(
    frame: pd.DataFrame, *, index_key: str, selected: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Return `frame` as records, carrying each row's position under `index_key`.

    `index_key` is one of the adapter's reserved names, and a frame carrying any of them is refused rather
    than worked around: an injected key appears in an operation's prompt schema, so it may not vary per frame.
    """

    reject_reserved(frame.columns)
    if index_key in frame.columns:
        raise ValueError(f"frame already has a column named {index_key!r}")
    if frame.columns.duplicated().any():
        # a dict holds one value per key, so a repeated column name would be dropped silently
        raise ValueError("frame has duplicate column names, which records cannot represent")
    if selected is not None:
        frame = frame.loc[:, list(selected)]
    if len(frame.columns) == 0:
        # itertuples(index=False) emits nothing for a frame containing only an index.
        return [{index_key: position} for position in range(len(frame))]

    projected = frame.astype(object)
    # prompts and JSON have no NaN or NaT, so missing values become None
    clean = projected.where(pd.notna(projected), None)
    columns = list(frame.columns)
    records: list[dict[str, Any]] = [
        dict(zip(columns, row, strict=True))
        for row in clean.itertuples(index=False, name=None)
    ]
    for position, record in enumerate(records):
        record[index_key] = position
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
    """Verify the returned source positions, preserving the engine's record order."""

    seen: set[int] = set()
    positions: list[int] = []
    for position, record in enumerate(records):
        if index_key not in record:
            raise KeyError(
                f"record {position} is missing {index_key!r}: the operator dropped row identity"
            )
        value = record[index_key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"record {position} has non-integer source position {value!r}")
        row = int(value)
        if not 0 <= row < len(source):
            raise KeyError(f"record {position} carries unknown source position {row}")
        if row in seen:
            raise ValueError(f"duplicate source position {row}")
        seen.add(row)
        positions.append(row)
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
    missing = [position for position in range(len(source)) if position not in returned]
    detail = f"; missing {missing}" if missing else ""
    raise RuntimeError(
        f"{op} received {len(records)} records for {len(source)} input rows: an answer failed "
        f"validation and its row was dropped{detail}"
    )
