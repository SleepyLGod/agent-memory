"""Unordered LIMIT contracts and local materialized-parent refresh."""

from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.tracing.semantic import query_digest


@pytest.mark.parametrize("n", [0, 1, 2, 10])
@pytest.mark.parametrize("values", [[], [1, 1, 2]])
def test_limit_bounds_preserves_schema_and_bag(n: int, values: list[int]) -> None:
    source = am.Source({"value": "Number."})
    view = source.limit(n)
    rows = pd.DataFrame({"value": pd.Series(values, dtype="Int64")})
    result = LotusAdapter().execute(view.expr, {"log": rows})
    assert len(result) == min(n, len(rows))
    assert result.dtypes.equals(rows.dtypes)
    assert not (Counter(result.value) - Counter(rows.value))
    assert output_columns(view.expr) == ("value",)
    assert view.expr.params == {"n": n}
    assert query_digest(view.expr) == query_digest(source.limit(n).expr)
    assert query_digest(view.expr) != query_digest(source.limit(n + 1).expr)


@pytest.mark.parametrize("n", [True, False, 1.0, "1", None])
def test_limit_rejects_non_integer(n: Any) -> None:
    with pytest.raises(TypeError, match="non-negative integer"):
        am.Source({"value": "Number."}).limit(n)


def test_limit_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        am.Source({"value": "Number."}).limit(-1)


def test_limit_refreshes_replaced_parent_and_empty_parent() -> None:
    source = am.Source({"value": "Number."})
    minimum = source.min(column="value", output_col="value")
    view = minimum.filter(minimum.col("value") > 0).limit(2)
    flow = am.SemanticDataflow(source=source, views={"result": view})
    for value, expected in [(4, [4]), (2, [2]), (-1, [])]:
        flow.apply(pd.DataFrame({"value": [value]}))
        assert flow.view("result").value.tolist() == expected


def test_limit_does_not_reexecute_semantic_parent() -> None:
    class Oracle(LotusAdapter):
        seen: list[int]

        def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
            if query.op == "sem_filter":
                rows = self.execute(query.inputs[0], inputs)
                self.seen.extend(rows.value.tolist())
                return rows.copy()
            return super().execute(query, inputs)

    adapter = Oracle()
    adapter.seen = []
    source = am.Source({"value": "Number."})
    view = source.sem_filter(instruction="Keep {value}.").limit(1)
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=adapter)
    for value in [1, 2, 3]:
        flow.apply(pd.DataFrame({"value": [value]}))
    assert adapter.seen == [1, 2, 3]
    assert len(flow.view("result")) == 1
