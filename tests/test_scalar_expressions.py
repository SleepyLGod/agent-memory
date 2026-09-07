"""Tests for deterministic scalar expressions used by semantic views."""

from __future__ import annotations

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.policy.expressions import CaseWhenExpr, TryCastExpr, expr_from_param


def test_try_cast_and_case_when_execute_inside_assign() -> None:
    source = am.Source({"raw_score": "Untrusted scalar score."})
    parsed_scores = source.assign(
        _parsed_score=am.try_cast(source.col("raw_score"), to="float")
    )
    normalized_scores = parsed_scores.assign(
        review_score=am.case_when(
            (parsed_scores.col("_parsed_score") >= 1)
            & (parsed_scores.col("_parsed_score") <= 5),
            parsed_scores.col("_parsed_score"),
            3.0,
        )
    ).select(["review_score"])
    flow = am.SemanticDataflow(
        source=source,
        views={
            "parsed": parsed_scores.select(["_parsed_score"]),
            "scores": normalized_scores,
        },
        adapter=LotusAdapter(),
    )

    flow.apply(
        pd.DataFrame(
            {"raw_score": ["5", "2.5", "not-a-number", None, "0", "6"]}
        )
    )

    parsed = flow.view("parsed")["_parsed_score"].tolist()
    assert parsed[:2] == [5.0, 2.5]
    assert all(pd.isna(value) for value in parsed[2:4])
    pd.testing.assert_frame_equal(
        flow.view("scores"),
        pd.DataFrame({"review_score": [5.0, 2.5, 3.0, 3.0, 3.0, 3.0]}),
        check_dtype=False,
    )


def test_case_when_only_evaluates_the_selected_branch_rows() -> None:
    source = am.Source(
        {
            "numerator": "Numerator.",
            "denominator": "Denominator.",
        }
    )
    quotient = source.col("numerator") / source.col("denominator")
    result = source.assign(
        guarded_then=am.case_when(
            source.col("denominator") != 0,
            quotient,
            0,
        ),
        guarded_else=am.case_when(
            source.col("denominator") == 0,
            0,
            quotient,
        ),
    )

    frame = LotusAdapter().execute(
        result.expr,
        {
            "log": pd.DataFrame(
                [
                    {"numerator": 10, "denominator": 0},
                    {"numerator": 10, "denominator": 2},
                ],
                index=pd.Index([7, 7]),
            )
        },
    )

    assert frame["guarded_then"].tolist() == [0, 5]
    assert frame["guarded_else"].tolist() == [0, 5]
    assert frame.index.tolist() == [7, 7]


@pytest.mark.parametrize(("condition", "expected"), [(True, 7), (False, 0)])
def test_case_when_accepts_boolean_literal_conditions(
    condition: bool,
    expected: int,
) -> None:
    source = am.Source({"value": "Value."})
    case_expr = am.case_when(condition, source.col("value"), 0)
    query = source.assign(result=case_expr).select(["result"])

    result = LotusAdapter().execute(
        query.expr,
        {"log": pd.DataFrame([{"value": 7}])},
    )

    assert result["result"].tolist() == [expected]
    assert expr_from_param(case_expr.to_param()).to_param() == case_expr.to_param()


@pytest.mark.parametrize("condition", [0, 1, "true"])
def test_case_when_rejects_non_boolean_literal_conditions(condition: object) -> None:
    with pytest.raises(TypeError, match="Expected a boolean relational expression"):
        am.case_when(condition, 1, 0)


def test_try_cast_and_case_when_round_trip_through_query_params() -> None:
    source = am.Source({"raw_score": "Untrusted scalar score."})
    cast_expr = am.try_cast(source.col("raw_score"), to="float")
    case_expr = am.case_when(cast_expr.is_not_null(), cast_expr, 3.0)

    rebuilt_cast = expr_from_param(cast_expr.to_param())
    rebuilt_case = expr_from_param(case_expr.to_param())

    assert isinstance(rebuilt_cast, TryCastExpr)
    assert rebuilt_cast.to_param() == cast_expr.to_param()
    assert isinstance(rebuilt_case, CaseWhenExpr)
    assert rebuilt_case.to_param() == case_expr.to_param()


def test_try_cast_rejects_unsupported_target_types() -> None:
    source = am.Source({"raw_score": "Untrusted scalar score."})

    with pytest.raises(ValueError, match="only supports the 'float' target"):
        am.try_cast(source.col("raw_score"), to="integer")
