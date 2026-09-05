"""Public API and full-query tests for deterministic relational aggregates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.policy.aggregates import (
    AvgAggregateSpec,
    CountAggregateSpec,
    SumAggregateSpec,
)
from agent_memory.policy.expressions import ArithmeticExpr, expr_from_param
from agent_memory.policy.logical import MemorySpec, MemoryView, QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.planner.differential_policy import PolicyDifferentiator


def _execute(relation: object, rows: list[dict[str, object]]) -> pd.DataFrame:
    expr = relation.expr  # type: ignore[attr-defined]
    columns = list(output_columns(expr.inputs[0]))
    source = pd.DataFrame(rows, columns=columns)
    return LotusAdapter().execute(expr, {"log": source})


def test_aggregate_descriptors_and_convenience_methods_build_the_same_query() -> None:
    rows = am.Log({"price": "Numeric price."})

    composed = rows.agg(
        am.count(output_col="row_count"),
        am.sum(column="price", output_col="total_price"),
        am.avg(column="price", output_col="average_price"),
    )

    specs = tuple(composed.expr.params["aggregates"])
    assert isinstance(specs[0], CountAggregateSpec)
    assert isinstance(specs[1], SumAggregateSpec)
    assert isinstance(specs[2], AvgAggregateSpec)
    assert rows.count(output_col="row_count").expr.params["aggregates"] == specs[:1]
    assert rows.sum(column="price", output_col="total_price").expr.params[
        "aggregates"
    ] == specs[1:2]
    assert rows.avg(column="price", output_col="average_price").expr.params[
        "aggregates"
    ] == specs[2:]
    assert output_columns(composed.expr) == (
        "row_count",
        "total_price",
        "average_price",
    )


def test_global_aggregate_uses_count_star_and_ignores_null_numeric_values() -> None:
    rows = am.Log({"price": "Numeric price."})
    aggregate = rows.agg(
        am.count(output_col="row_count"),
        am.sum(column="price", output_col="total_price"),
        am.avg(column="price", output_col="average_price"),
    )

    result = _execute(
        aggregate,
        [{"price": 10}, {"price": None}, {"price": 20}, {"price": 20}],
    )

    assert result.to_dict(orient="records") == [
        {"row_count": 4, "total_price": 50, "average_price": 50 / 3}
    ]


def test_empty_global_aggregate_returns_one_sql_style_row() -> None:
    rows = am.Log({"price": "Numeric price."})
    aggregate = rows.agg(
        am.count(output_col="row_count"),
        am.sum(column="price", output_col="total_price"),
        am.avg(column="price", output_col="average_price"),
    )

    result = _execute(aggregate, [])

    assert result["row_count"].tolist() == [0]
    assert result["total_price"].isna().tolist() == [True]
    assert result["average_price"].isna().tolist() == [True]

    null_only = _execute(aggregate, [{"price": None}, {"price": None}])
    assert null_only["row_count"].tolist() == [2]
    assert null_only["total_price"].isna().tolist() == [True]
    assert null_only["average_price"].isna().tolist() == [True]


def test_grouped_multi_key_aggregate_returns_no_rows_for_empty_input() -> None:
    rows = am.Log(
        {"region": "Region.", "category": "Category.", "price": "Price."}
    )
    aggregate = rows.group_by(["region", "category"]).agg(
        am.count(output_col="row_count"),
        am.sum(column="price", output_col="total_price"),
        am.avg(column="price", output_col="average_price"),
    )

    result = _execute(aggregate, [])

    assert result.empty
    assert list(result.columns) == [
        "region",
        "category",
        "row_count",
        "total_price",
        "average_price",
    ]


def test_grouped_aggregate_preserves_duplicate_rows_and_null_groups() -> None:
    rows = am.Log({"group": "Group.", "price": "Price."})
    aggregate = rows.group_by("group").agg(
        am.count(output_col="row_count"),
        am.sum(column="price", output_col="total_price"),
        am.avg(column="price", output_col="average_price"),
    )

    result = _execute(
        aggregate,
        [
            {"group": "a", "price": 2},
            {"group": "a", "price": 2},
            {"group": "a", "price": None},
            {"group": None, "price": 3},
        ],
    )

    assert result.iloc[0].to_dict() == {
        "group": "a",
        "row_count": 3,
        "total_price": 4,
        "average_price": 2,
    }
    assert pd.isna(result.iloc[1]["group"])
    assert result.iloc[1]["row_count"] == 1
    assert result.iloc[1]["total_price"] == 3
    assert result.iloc[1]["average_price"] == 3


def test_grouped_aggregate_does_not_collide_with_internal_column_names() -> None:
    rows = am.Log(
        {
            "__am_aggregate_row_count": "A caller-owned group key.",
            "price": "Price.",
        }
    )
    aggregate = rows.group_by("__am_aggregate_row_count").count(
        output_col="count"
    )

    result = _execute(
        aggregate,
        [{"__am_aggregate_row_count": "group-a", "price": 1}],
    )

    assert result.to_dict(orient="records") == [
        {"__am_aggregate_row_count": "group-a", "count": 1}
    ]


@pytest.mark.parametrize("value", ["12", True, object()])
def test_sum_and_avg_reject_non_numeric_non_null_values(value: object) -> None:
    rows = am.Log({"price": "Numeric price."})
    aggregate = rows.agg(am.sum(column="price", output_col="total"))

    with pytest.raises(TypeError, match="numeric"):
        _execute(aggregate, [{"price": value}])


def test_aggregate_authoring_rejects_invalid_columns_and_output_collisions() -> None:
    rows = am.Log({"group": "Group.", "price": "Price."})

    with pytest.raises(ValueError, match="input column"):
        rows.sum(column="missing", output_col="total")
    with pytest.raises(ValueError, match="group key"):
        rows.group_by("group").count(output_col="group")
    with pytest.raises(ValueError, match="unique"):
        rows.agg(
            am.count(output_col="result"),
            am.sum(column="price", output_col="result"),
        )


def test_sem_groupby_rejects_numeric_aggregates_without_changing_semantic_agg() -> None:
    rows = am.Log({"text": "Text."})
    grouped = rows.sem_groupby(
        input_cols=["text"], instruction="Group equivalent text."
    )

    with pytest.raises(NotImplementedError, match="numeric algebraic"):
        grouped.count(output_col="row_count")
    with pytest.raises(NotImplementedError, match="numeric algebraic"):
        grouped.agg(am.avg(column="text", output_col="average"))

    existing = grouped.agg(
        am.sem_agg(
            input_cols=["text"],
            output_cols={"text": "Canonical text."},
            instruction="Choose the canonical text.",
        )
    )
    assert existing.expr.op == "agg"


def test_arithmetic_expressions_round_trip_and_execute_both_operand_orders() -> None:
    rows = am.Log({"left": "Left number.", "right": "Right number."})
    expression = (rows.col("left") + 2) * (10 - rows.col("right")) / 4
    rebuilt = expr_from_param(expression.to_param())

    assert isinstance(expression, ArithmeticExpr)
    assert rebuilt.to_param() == expression.to_param()
    result = LotusAdapter().execute(
        rows.assign(
            add=rows.col("left") + rows.col("right"),
            subtract=10 - rows.col("left"),
            multiply=rows.col("left") * 3,
            divide=rows.col("left") / 2,
            reverse_add=2 + rows.col("left"),
            reverse_multiply=3 * rows.col("left"),
            reverse_divide=12 / rows.col("left"),
            composed=expression,
        ).expr,
        {"log": pd.DataFrame([{"left": 6, "right": 2}])},
    )
    assert result.iloc[0].to_dict() == {
        "left": 6,
        "right": 2,
        "add": 8,
        "subtract": 4,
        "multiply": 18,
        "divide": 3,
        "reverse_add": 8,
        "reverse_multiply": 18,
        "reverse_divide": 2,
        "composed": 16,
    }


def test_arithmetic_null_propagates_and_division_by_zero_is_explicit() -> None:
    rows = am.Log({"numerator": "Numerator.", "denominator": "Denominator."})
    divided = rows.assign(
        result=rows.col("numerator") / rows.col("denominator")
    )

    null_result = LotusAdapter().execute(
        divided.expr,
        {"log": pd.DataFrame([{"numerator": None, "denominator": 2}])},
    )
    assert null_result["result"].isna().tolist() == [True]

    with pytest.raises(ZeroDivisionError, match="division by zero"):
        LotusAdapter().execute(
            divided.expr,
            {"log": pd.DataFrame([{"numerator": 1, "denominator": 0}])},
        )


def test_sembench_representative_views_are_expressible_without_special_operators() -> None:
    reviews = am.Log(
        {"review": "Review text.", "movie": "Movie identifier.", "year": "Year."}
    )
    positive = reviews.select(["review", "movie", "year"]).sem_filter(
        instruction="The review expresses a positive opinion: {review}"
    )

    movie_q3 = positive.count(output_col="positive_count")
    positive_count = movie_q3.assign(_scalar_key=1)
    total_count = reviews.count(output_col="total_count").assign(_scalar_key=1)
    movie_q4 = (
        positive_count.join(total_count, on="_scalar_key")
        .assign(
            positive_ratio=positive_count.col("positive_count")
            / total_count.col("total_count")
        )
        .select(["positive_ratio"])
    )
    movie_q8 = (
        reviews.sem_map(
            input_cols=["review"],
            output_cols={"sentiment": "POSITIVE or NEGATIVE."},
            instruction="Classify the review sentiment.",
        )
        .group_by("sentiment")
        .count(output_col="review_count")
    )
    cars_q4 = (
        reviews.assign(age=2026 - reviews.col("year"))
        .avg(column="age", output_col="average_age")
    )

    assert output_columns(movie_q3.expr) == ("positive_count",)
    assert output_columns(movie_q4.expr) == ("positive_ratio",)
    assert output_columns(movie_q8.expr) == ("sentiment", "review_count")
    assert output_columns(cars_q4.expr) == ("average_age",)

    for name, relation in {
        "movie_q3": movie_q3,
        "movie_q4": movie_q4,
        "movie_q8": movie_q8,
        "cars_q4": cars_q4,
    }.items():
        policy = PolicyDifferentiator().differentiate(
            MemorySpec(
                log=reviews,
                views={name: MemoryView(name=name, query=relation.expr)},
                private_relations={},
            )
        )
        assert policy.nodes[policy.view_outputs[name]].output_columns == output_columns(
            relation.expr
        )


def test_multi_aggregate_executes_its_input_only_once() -> None:
    class CountingAdapter(LotusAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.log_executions = 0

        def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
            if query.op == "log":
                self.log_executions += 1
            return super().execute(query, inputs)

    rows = am.Log({"price": "Price."})
    aggregate = rows.agg(
        am.count(output_col="row_count"),
        am.sum(column="price", output_col="total"),
        am.avg(column="price", output_col="average"),
    )
    adapter = CountingAdapter()

    adapter.execute(aggregate.expr, {"log": pd.DataFrame([{"price": 2}])})

    assert adapter.log_executions == 1
