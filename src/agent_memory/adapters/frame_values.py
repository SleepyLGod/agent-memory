"""Column-preserving assembly for deterministic adapter operations."""

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


def numeric_comparison_key(value: Any) -> Any:
    """Avoid NumPy promoting integer/float pairs to float before comparison."""
    if isinstance(value, np.generic) and value.dtype.kind in "iuf":
        return value.item()
    return value


def exact_series(values: list[Any], *, index: pd.Index | None = None) -> pd.Series:
    """Use normal pandas inference unless a float column would round an integer."""
    result = pd.Series(values, index=index)
    if pd.api.types.is_float_dtype(result.dtype) and any(
        isinstance(v, Integral) and int(v) != float(f)
        for v, f in zip(values, result, strict=True)
    ):
        return pd.Series(values, index=index, dtype=object)
    return result


def expand_arrays(
    source: pd.DataFrame, arrays: Iterable[list[Any]], output: str
) -> pd.DataFrame:
    """Repeat original rows by position; infer only the expanded column."""
    positions: list[int] = []
    values: list[Any] = []
    for i, elements in enumerate(arrays):
        positions.extend([i] * len(elements))
        values.extend(elements)
    result = source.iloc[positions].reset_index(drop=True)
    result[output] = exact_series(values)
    return result


def expand_structs(
    source: pd.DataFrame,
    column: str,
    objects: Iterable[dict[str, Any]],
    fields: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Expand named fields without rebuilding unrelated columns from row dicts."""
    columns: dict[str, list[Any]] = {output: [] for _, output in fields}
    for obj in objects:
        for field, output in fields:
            if field not in obj:
                raise ValueError(f"unnest field {field!r} not found in object column {column!r}")
            columns[output].append(obj[field])
    result = source.drop(columns=[column]).reset_index(drop=True)
    for name, values in columns.items():
        result[name] = exact_series(values)
    return result
