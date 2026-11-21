"""Exact integer results for the two dataframe execution adapters."""

import json

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.runtime.window import over_frames


@pytest.mark.parametrize("backend", [LotusAdapter, DocetlAdapter])
@pytest.mark.parametrize("value", [17, 2**53 + 1, 2**64 - 1])
def test_integer_value_through_relational_operators(
    backend: type, value: int
) -> None:
    adapter = backend()
    log = am.Log({"id": "integer", "score": "number"})
    source = pd.DataFrame({"id": [value], "score": [0.25]})
    query = log.over(rows=(0, 0)).array_agg(columns=["id"], output_col="items")
    window = adapter.execute(query.expr, {"log": source})
    assert int(window["id"].iloc[0]) == value
    assert window["id"].dtype == source["id"].dtype
    assert json.loads(window["items"].iat[0]) == [{"id": value}]
    assert window["score"].tolist() == [0.25]

    nested = am.Log({"payload": "array", "label": "text"})
    expanded = nested.explode(column="payload").unnest(
        column="payload", fields={"id": "id"}
    )
    rows = pd.DataFrame(
        {"payload": [[{"id": value}, {"id": None}], []], "label": ["a", "b"]},
        index=pd.Index([9, 9]),
    )
    result = adapter.execute(expanded.expr, {"log": rows})
    assert result["label"].tolist() == ["a", "a"]
    assert int(result["id"].iloc[0]) == value
    assert pd.isna(result["id"].iloc[1])
    rows["payload"] = [[value, None], []]
    flat = adapter.execute(nested.explode(column="payload").expr, {"log": rows})
    assert int(flat["payload"].iloc[0]) == value
    assert pd.isna(flat["payload"].iloc[1])

    nullable = pd.DataFrame({"id": pd.array([value, None], dtype="UInt64")})
    ids = am.Log({"id": "integer"})
    least = ids.assign(result=am.least(ids.col("id"), ids.col("id")))
    result = adapter.execute(least.expr, {"log": nullable})
    assert int(result["result"].iloc[0]) == value
    assert pd.isna(result["result"].iloc[1])


def test_window_rejects_distinct_large_integer_in_emit_suffix() -> None:
    source = pd.DataFrame({"id": [0, 2**53], "score": [0.0, 0.25]})
    wrong_suffix = pd.DataFrame({"id": [2**53 + 1], "score": [0.25]})
    with pytest.raises(NotImplementedError, match="append suffix"):
        over_frames(wrong_suffix, source, {"rows": (-1, 0)})


@pytest.mark.parametrize("backend", [LotusAdapter, DocetlAdapter])
def test_least_compares_large_integer_and_float_in_either_order(backend: type) -> None:
    log = am.Log({"id": "integer", "score": "number"})
    source = pd.DataFrame({"id": [2**53 + 1], "score": [float(2**53)]})
    query = log.assign(
        first=am.least(log.col("id"), log.col("score")),
        second=am.least(log.col("score"), log.col("id")),
    )
    result = backend().execute(query.expr, {"log": source})
    assert int(result["first"].iat[0]) == int(result["second"].iat[0]) == 2**53
