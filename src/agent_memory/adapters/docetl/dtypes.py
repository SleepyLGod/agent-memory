"""The integer-type rules T1-T4, and the loss this module refuses to pass off.

Pandas hands an integer column to a `float64` carrier as soon as that column can hold a missing value —
inside `merge`, `reindex`, or `DataFrame(rows)` — and a `float64` cannot hold an integer past 2**53. The
nullable dtypes hold both. Stated once, cited by label below and from `records.py`:

  T1  a column whose values came from a source frame keeps that source's dtype
  T2  if it gained missing values and the source dtype is a numpy integer, it becomes the matching pandas
      nullable integer (`Int64`, `Int32`, `UInt64`, ...), never `float64`
  T3  a source `object` column stays `object`, and a source float column is never narrowed
  T4  a carried column whose source dtype was an integer is refused if it arrives as `float64`, even when
      every value looks integral: by then the rounding cannot be undone

T5-T7 belong to the record boundary and are stated in `records.py`.

`nullable_integers` satisfies T2 before a merge that will introduce missing values; `carry_integer_dtypes`
enforces T1-T4 after a rebuild. Both belong to a rebuild that cannot select its rows from the source — the
two joins. An operator that can select does so in `records.publish_rows`, where T1 holds by construction
and needs neither.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import pandas as pd

# A table, not a derivation: `to_numpy_dtype` and `numpy_dtype` disagree across pandas releases, and a
#   wrong guess here is silent.
_NULLABLE = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
}


def nullable_integers(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return `frame` with the integer columns among `columns` re-typed to their nullable twin.

    Satisfies T2 before a `merge` that can introduce missing values, so nothing in the merge needs to
    choose `float64`.
    """

    result = frame.copy()
    for column in columns:
        nullable = _NULLABLE.get(str(result[column].dtype))
        if nullable is not None:
            result[column] = result[column].astype(nullable)
    return result


def carry_integer_dtypes(
    result: pd.DataFrame, source_dtypes: Mapping[str, Any], *, op: str
) -> pd.DataFrame:
    """Return `result` with every integer-sourced column under the dtype of the frame it came from.

    Enforces T1-T4, stated at the top of this module. `source_dtypes` maps a published column name to the
    dtype of the column it was carried from; a name the result does not have is ignored, because a merge
    may publish fewer columns than it read.
    """

    carried = result.copy()
    for column, source_dtype in source_dtypes.items():
        if column not in carried.columns:
            continue
        nullable = _NULLABLE.get(str(source_dtype))
        if nullable is None:
            # Reason: a source that is already nullable (`Int64`, `UInt64`, ...) is an integer source
            #   too, and `_NULLABLE` has no entry for it. Its own name is the target it returns to.
            if not pd.api.types.is_integer_dtype(source_dtype):
                continue
            nullable = str(source_dtype)
        series = carried[column]
        if series.dtype == object:
            if not _holds_integers(series):
                raise _refusal(column, source_dtype, series.dtype, op)
            carried[column] = series = series.astype(nullable)
        if pd.api.types.is_float_dtype(series):
            raise _refusal(column, source_dtype, series.dtype, op)
        if not pd.api.types.is_integer_dtype(series):
            continue
        if str(series.dtype) == str(source_dtype):
            continue
        if str(series.dtype) == nullable:
            if not series.isna().any():
                carried[column] = series.astype(source_dtype)
            continue
        series = series.astype(nullable)
        carried[column] = series if series.isna().any() else series.astype(source_dtype)
    return carried


def _holds_integers(series: pd.Series) -> bool:
    """Return whether every present value is an integer.

    A bool is not one — it is a different column that happens to sit in `object` — and a float is not
    one even when its value is whole: that is the corruption arriving in disguise.
    """

    for value in series:
        if isinstance(value, bool):
            return False
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and value != value:
            continue
        if not isinstance(value, Integral):
            return False
    return True


def _refusal(column: str, source_dtype: Any, arrived: Any, op: str) -> ValueError:
    """Describe the loss rather than the dtype, because the dtype is only its symptom."""

    return ValueError(
        f"{op} published {column!r} as {arrived}, but it was carried from {source_dtype}: "
        "an integer that passed through a float cannot be recovered"
    )
