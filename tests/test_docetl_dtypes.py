"""The integer-type rules a rebuilt frame keeps, and the loss it refuses to pass off as a success.

`BIG` is 2**53 + 1, the first integer `float64` cannot hold; `WIDE` is wider than `int64`, so it lives
in an `object` cell. Every case names the rule it pins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_memory.adapters.docetl.dtypes import carry_integer_dtypes, nullable_integers

BIG = 9007199254740993
WIDE = 10**19


def test_a_complete_integer_column_goes_back_to_the_source_dtype() -> None:
    """T1: a column carried from an `int64` source publishes `int64`, so a join that lost no row
    publishes exactly what it published before the rule existed.
    """

    result = pd.DataFrame({"amount": pd.array([BIG, 5], dtype="Int64")})

    carried = carry_integer_dtypes(result, {"amount": np.dtype("int64")}, op="join")

    assert str(carried["amount"].dtype) == "int64"
    assert carried["amount"].tolist() == [BIG, 5]


def test_an_integer_column_that_gained_a_missing_value_stays_exact() -> None:
    """T2: the missing side becomes `<NA>` in a nullable integer, never `float64`."""

    result = pd.DataFrame({"amount": pd.array([BIG, pd.NA], dtype="Int64")})

    carried = carry_integer_dtypes(result, {"amount": np.dtype("int64")}, op="join")

    assert str(carried["amount"].dtype) == "Int64"
    assert carried["amount"].iloc[0] == BIG
    assert pd.isna(carried["amount"].iloc[1])


def test_an_object_column_of_integers_becomes_a_nullable_integer() -> None:
    """T1 with T2: a rebuild that lands on `object` (which is what a row-by-row assembly produces)
    is normalized rather than left to the caller.
    """

    result = pd.DataFrame({"amount": pd.Series([BIG, pd.NA], dtype=object)})

    carried = carry_integer_dtypes(result, {"amount": np.dtype("int64")}, op="join")

    assert str(carried["amount"].dtype) == "Int64"
    assert carried["amount"].iloc[0] == BIG


def test_an_integer_column_with_no_missing_value_leaves_object_behind() -> None:
    """T1 again, the other side of the branch: completeness picks the numpy dtype back up."""

    result = pd.DataFrame({"amount": pd.Series([BIG, 5], dtype=object)})

    carried = carry_integer_dtypes(result, {"amount": np.dtype("int64")}, op="join")

    assert str(carried["amount"].dtype) == "int64"


def test_a_source_object_column_is_left_alone() -> None:
    """T3: `object` is where an integer wider than `int64` lives, so exactness, not a numeric dtype,
    is the guarantee. Narrowing it would raise instead of preserving it.
    """

    result = pd.DataFrame({"amount": pd.Series([WIDE, 5], dtype=object)})

    carried = carry_integer_dtypes(result, {"amount": np.dtype("O")}, op="join")

    assert str(carried["amount"].dtype) == "object"
    assert carried["amount"].iloc[0] == WIDE


def test_a_source_float_column_is_never_narrowed() -> None:
    """T3: a float source stays a float, however integral its values happen to look."""

    result = pd.DataFrame({"amount": np.array([2.5, 5.0])})

    carried = carry_integer_dtypes(result, {"amount": np.dtype("float64")}, op="join")

    assert str(carried["amount"].dtype) == "float64"
    assert carried["amount"].tolist() == [2.5, 5.0]


def test_a_float_result_for_an_integer_source_is_refused() -> None:
    """T4: the loss happened inside the rebuild, so a conversion cannot undo it. The guard refuses the
    dtype even though both values look integral.
    """

    rounded = float(BIG)  # 9007199254740992.0, and no longer the number the source held
    result = pd.DataFrame({"amount": np.array([rounded, np.nan])})

    with pytest.raises(ValueError, match="amount"):
        carry_integer_dtypes(result, {"amount": np.dtype("int64")}, op="join")


def test_an_object_result_holding_floats_is_refused() -> None:
    """T4 in disguise: `object` hides the coercion, so the guard reads the values, not only the dtype."""

    result = pd.DataFrame({"amount": pd.Series([float(BIG)], dtype=object)})

    with pytest.raises(ValueError, match="amount"):
        carry_integer_dtypes(result, {"amount": np.dtype("int64")}, op="join")


def test_a_column_the_result_does_not_have_is_ignored() -> None:
    """The mapping describes the sources, not the frame: a merge may publish fewer columns than it read."""

    result = pd.DataFrame({"keep": ["a"]})

    carried = carry_integer_dtypes(result, {"gone": np.dtype("int64")}, op="join")

    assert list(carried.columns) == ["keep"]


def test_the_nullable_pre_conversion_keeps_the_value() -> None:
    """The merge fix: converting before pandas can see a missing value is what keeps it exact."""

    frame = pd.DataFrame({"amount": np.array([BIG, 5], dtype="int64")})

    converted = nullable_integers(frame, ["amount"])

    assert str(converted["amount"].dtype) == "Int64"
    assert converted["amount"].tolist() == [BIG, 5]


def test_the_nullable_pre_conversion_does_not_touch_other_columns() -> None:
    """Only integer columns are converted; an `object` or float column keeps its dtype."""

    frame = pd.DataFrame(
        {
            "amount": np.array([BIG], dtype="int64"),
            "label": ["x"],
            "score": np.array([1.5]),
        }
    )

    converted = nullable_integers(frame, ["amount", "label", "score"])

    assert str(converted["label"].dtype) == "object"
    assert str(converted["score"].dtype) == "float64"


def test_each_integer_width_maps_to_its_own_nullable_type() -> None:
    """The mapping is a table, not a constant: an `int32` source comes back `int32`, not `Int64`."""

    result = pd.DataFrame({"small": pd.array([1, pd.NA], dtype="Int32")})

    carried = carry_integer_dtypes(result, {"small": np.dtype("int32")}, op="join")

    assert str(carried["small"].dtype) == "Int32"
    assert carried["small"].iloc[0] == 1


def test_a_nullable_integer_source_is_not_skipped() -> None:
    """A source that is already nullable is still an integer source, so an object rebuild returns to it."""

    result = pd.DataFrame({"amount": pd.Series([BIG, pd.NA], dtype=object)})

    carried = carry_integer_dtypes(result, {"amount": pd.Int64Dtype()}, op="join")

    assert str(carried["amount"].dtype) == "Int64"
    assert carried["amount"].iloc[0] == BIG
    assert pd.isna(carried["amount"].iloc[1])


def test_a_nullable_unsigned_source_keeps_its_width() -> None:
    """The nullable source case uses its own name, so `UInt64` is not narrowed to `Int64`."""

    wide = 10**19

    result = pd.DataFrame({"amount": pd.Series([wide, pd.NA], dtype=object)})

    carried = carry_integer_dtypes(result, {"amount": pd.UInt64Dtype()}, op="join")

    assert str(carried["amount"].dtype) == "UInt64"
    assert carried["amount"].iloc[0] == wide
