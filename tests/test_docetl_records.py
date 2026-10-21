"""Frame ↔ record conversion: a row's identity survives the trip into DocETL and back.

Each case pins one way the conversion could lose, invent or move that identity. No engine, no network:
the runtime's index contract is what breaks when this is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_memory.adapters.docetl.records import from_records, to_records
from agent_memory.adapters.docetl.reserved import INDEX_KEY


def _round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert `frame` to records and back, choosing the index key as a caller would."""

    key = INDEX_KEY
    return from_records(to_records(frame, index_key=key), index_key=key)


def test_index_survives_the_round_trip() -> None:
    source = pd.DataFrame({"message": ["a", "b"]}, index=[7, 9])

    restored = _round_trip(source)

    assert list(restored.index) == [7, 9]
    assert restored["message"].tolist() == ["a", "b"]


def test_duplicate_index_values_survive() -> None:
    """The runtime groups output rows by index occurrence, so duplicates are legitimate."""

    source = pd.DataFrame({"message": ["a", "b", "c"]}, index=[1, 1, 2])

    assert list(_round_trip(source).index) == [1, 1, 2]


def test_a_dropped_row_leaves_a_subset_of_the_input_index() -> None:
    """The runtime's row-local rule: output index ⊆ input index."""

    source = pd.DataFrame({"message": ["a", "b", "c"]}, index=[10, 11, 12])
    key = INDEX_KEY
    kept = [record for record in to_records(source, index_key=key) if record["message"] != "b"]

    restored = from_records(kept, index_key=key)

    assert set(restored.index) <= set(source.index)
    assert restored["message"].tolist() == ["a", "c"]


def test_a_reordered_record_carries_its_own_index() -> None:
    """The reason the index travels in the record: position is not identity."""

    source = pd.DataFrame({"message": ["a", "b", "c"]}, index=[10, 11, 12])
    key = INDEX_KEY

    restored = from_records(
        list(reversed(to_records(source, index_key=key))), index_key=key
    )

    assert list(zip(restored.index, restored["message"])) == [(12, "c"), (11, "b"), (10, "a")]


def test_missing_values_become_none_in_records() -> None:
    """Records are serialised into prompts and JSON, where NaN and NaT have no representation."""

    source = pd.DataFrame(
        {"text": ["a", np.nan], "when": pd.to_datetime(["2020-01-01", None])},
        index=[0, 1],
    )

    records = to_records(source, index_key=INDEX_KEY)

    assert records[1]["text"] is None
    assert records[1]["when"] is None


def test_missing_values_come_back_missing() -> None:
    """On the way back each column's inferred dtype picks the marker: `None` in an object column,
    `NaT` in a datetime one. What matters is that the value stays missing and the row stays present.
    """

    source = pd.DataFrame(
        {"text": ["a", np.nan], "when": pd.to_datetime(["2020-01-01", None])},
        index=[0, 1],
    )

    restored = _round_trip(source)

    assert list(restored.index) == [0, 1]
    assert pd.isna(restored.loc[1, "text"])
    assert pd.isna(restored.loc[1, "when"])


def test_a_record_without_the_index_key_is_refused() -> None:
    """Loud, not positional: a lost index means lost row identity."""

    with pytest.raises(KeyError, match="index"):
        from_records([{"message": "a"}], index_key=INDEX_KEY)


def test_an_empty_frame_converts_to_no_records() -> None:
    """Records carry no schema, so an empty result only gets its columns back if the caller supplies
    them. Handing the runtime a column-less frame would fail its output-columns check.
    """

    empty = pd.DataFrame(columns=["message", "session_id"])
    key = INDEX_KEY

    assert to_records(empty, index_key=key) == []
    restored = from_records([], index_key=key, columns=["message", "session_id"])
    assert list(restored.columns) == ["message", "session_id"]


def test_a_frame_with_duplicate_column_names_is_refused() -> None:
    """A dict holds one value per key, so a repeated column name would be dropped silently."""

    duplicated = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(ValueError, match="duplicate column"):
        to_records(duplicated, index_key=INDEX_KEY)


def test_the_index_key_never_appears_as_a_column() -> None:
    source = pd.DataFrame({"message": ["a"]}, index=[3])

    assert INDEX_KEY not in _round_trip(source).columns


def test_a_frame_using_a_reserved_name_is_refused() -> None:
    """The names are reserved, not dodged: they appear in the model's tool schema, so they cannot be
    chosen per frame. A policy column that wants one is refused instead.
    """

    source = pd.DataFrame({INDEX_KEY: ["policy data"]}, index=[7])

    with pytest.raises(ValueError, match="adapter-reserved"):
        to_records(source, index_key=INDEX_KEY)


def test_a_key_taken_from_another_frame_is_refused() -> None:
    """Guards the caller that passes a non-reserved key which the frame already uses."""

    source = pd.DataFrame({"row_key": ["data"]})

    with pytest.raises(ValueError, match="already has a column"):
        to_records(source, index_key="row_key")
