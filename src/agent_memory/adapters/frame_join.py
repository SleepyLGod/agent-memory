"""Shared dataframe helpers for relational joins."""

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from agent_memory.policy.schema import require_unique_published_columns


# A table, not a derivation: `to_numpy_dtype` and `numpy_dtype` disagree across pandas releases, and a
#   wrong guess here is silent.
NULLABLE_INTEGER_DTYPES = {
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
    """Use nullable integer columns before introducing missing rows."""

    result = frame.copy()
    for column in columns:
        nullable = NULLABLE_INTEGER_DTYPES.get(str(result[column].dtype))
        if nullable is not None:
            result[column] = result[column].astype(nullable)
    return result


def take_join_rows(source: pd.DataFrame, positions: pd.Index) -> pd.DataFrame:
    """Gather original rows; -1 denotes a missing side of an outer join."""
    if -1 not in positions:
        return source.iloc[positions].reset_index(drop=True)
    framed = nullable_integers(source, source.columns)
    framed.index = pd.RangeIndex(len(source))
    return framed.reindex(positions).reset_index(drop=True)


def mixed_integer_keys(
    left: pd.DataFrame, right: pd.DataFrame, keys: tuple[str, ...]
) -> set[str]:
    """Identify signed/unsigned key pairs from column dtypes, without scanning values."""
    return {
        key for key in keys if {left[key].dtype.kind, right[key].dtype.kind} == {"i", "u"}
    }


def join_mixed_integer_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: tuple[str, ...],
    mixed: set[str],
    how: str,
    suffixes: tuple[str, str],
) -> pd.DataFrame:
    """Match exact comparison keys, then gather the business columns by source position."""

    def key_table(frame: pd.DataFrame, position: str) -> pd.DataFrame:
        # A separate namespace: numbered keys and position tags cannot collide with user columns.
        table = pd.DataFrame(
            {
                i: frame[k].astype(object) if k in mixed else frame[k]
                for i, k in enumerate(keys)
            }
        )
        table[position] = pd.array(range(len(frame)), dtype="Int64")
        return table

    on = list(range(len(keys)))
    lhs, rhs = key_table(left, "L"), key_table(right, "R")
    if how == "left_anti":
        rhs = rhs.drop_duplicates(subset=on)
    pairs = lhs.merge(rhs, on=on, how="left" if how == "left_anti" else how, sort=False)
    if how == "left_anti":
        return left.iloc[
            pairs.loc[pairs["R"].isna(), "L"].to_numpy(dtype=int)
        ].reset_index(drop=True)

    li = pd.Index(pairs["L"].fillna(-1), dtype="int64")
    ri = pd.Index(pairs["R"].fillna(-1), dtype="int64")
    a, b = take_join_rows(left, li), take_join_rows(right, ri)
    for i, key in enumerate(keys):
        if key not in mixed:
            a[key] = pairs[i].array
        elif how == "right" or (-1 in li and -1 not in ri):
            a[key] = b[key].array
        elif -1 in li:
            # A full join may carry both negative integers and values too large for int64.
            a[key] = a[key].astype(object).where(li != -1, b[key].astype(object))
    shared = set(left.columns).intersection(right.columns).difference(keys)
    ls, rs = suffixes
    result = pd.concat(
        [
            a.rename(columns={k: f"{k}{ls}" for k in shared}),
            b.drop(columns=list(keys)).rename(columns={k: f"{k}{rs}" for k in shared}),
        ],
        axis=1,
    )
    require_unique_published_columns(tuple(result.columns), op="join")
    return result


def bind_join_columns(value: Any, sides: tuple[tuple[str, pd.Index], ...]) -> Any:
    """Qualify column references before either pruning pairs or testing the predicate."""

    if isinstance(value, (tuple, list)):
        return tuple(bind_join_columns(item, sides) for item in value)
    if not isinstance(value, Mapping):
        return value
    if value.get("kind") != "column":
        return {k: bind_join_columns(v, sides) for k, v in value.items()}
    name, qualifier = value["name"], value.get("qualifier")
    matches = [
        (name, alias)
        for alias, cols in sides
        if name in cols and qualifier in (None, alias)
    ]
    if not matches and qualifier is None:
        # A literal published name can contain colons in both its column and its alias.
        for alias, cols in sides:
            suffix = f":{alias}"
            if name.endswith(suffix) and (raw := name.removesuffix(suffix)) in cols:
                matches.append((raw, alias))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous join column {name!r}; qualify it with an alias")
    if not matches:
        raise ValueError(
            f"Column {name!r} not found in join inputs for qualifier {qualifier!r}"
        )
    raw, alias = matches[0]
    return {**value, "name": raw, "qualifier": alias}
