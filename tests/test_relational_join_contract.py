"""Deterministic join and difference contracts, independent of the LLM backend."""

from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.adapters.docetl import relational as docetl
from agent_memory.adapters.lotus import relational as lotus
from agent_memory.policy.expressions import ColumnExpr
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.relation import Relation


@pytest.fixture(params=[LotusAdapter, DocetlAdapter])
def adapter(request: pytest.FixtureRequest) -> LotusAdapter | DocetlAdapter:
    return request.param()


def source(name: str, frame: pd.DataFrame) -> QueryExpr:
    return QueryExpr(
        op="materialized_view", params={"name": name, "columns": tuple(frame)}
    )


@pytest.mark.parametrize("offset, count", [(0, 0), (1, 4)])
def test_duplicate_mixed_integer_keys(
    adapter: LotusAdapter | DocetlAdapter, offset: int, count: int
) -> None:
    big = 2**53
    lf = pd.DataFrame({"id": pd.array([big + 1] * 2, dtype="int64")})
    rf = pd.DataFrame({"id": pd.array([big + offset] * 2, dtype="uint64")})
    left, right = Relation(source("l", lf)), Relation(source("r", rf))
    result = adapter.execute(left.join(right, on="id").expr, {"l": lf, "r": rf})
    assert result["id"].tolist() == [big + 1] * count


@pytest.mark.parametrize(
    "how, expected",
    [
        ("inner", [(7, "a")]),
        ("left", [(-1, "a"), (7, "b"), (7, "a")]),
        ("right", [(7, "a"), (2**64 - 1, "a")]),
        ("outer", [(-1, "a"), (7, "a"), (7, "b"), (2**64 - 1, "a")]),
        ("left_anti", [(-1, "a"), (7, "b")]),
    ],
)
def test_mixed_composite_keys_keep_source_values(
    adapter: LotusAdapter | DocetlAdapter, how: str, expected: list[tuple[int, str]]
) -> None:
    lf = pd.DataFrame(
        {"id": pd.array([-1, 7, 7], dtype="Int64"), "bucket": ["a", "b", "a"]},
        index=[4, 4, 4],
    )
    rf = pd.DataFrame(
        {"id": pd.array([7, 2**64 - 1], dtype="UInt64"), "bucket": ["a", "a"]},
        index=[5, 5],
    )
    q = QueryExpr(
        op="join",
        inputs=(source("l", lf), source("r", rf)),
        params={"on": ("id", "bucket"), "how": how},
    )
    result = adapter.execute(q, {"l": lf, "r": rf})
    assert result.columns.tolist() == ["id", "bucket"]
    assert [
        (int(k), v) for k, v in result.itertuples(index=False, name=None)
    ] == expected


@pytest.mark.parametrize("aliases", [False, True])
def test_predicate_join_binds_columns_and_keeps_exact_pairs(
    adapter: LotusAdapter | DocetlAdapter, aliases: bool
) -> None:
    big = 2**53 + 1
    left = pd.DataFrame({"a": [big, big, 7], "rank": [1, 2, 3]})
    right = pd.DataFrame(
        {"b": pd.Series([big - 1, big, big, "7"], dtype=object), "n": [0, 1, 2, 4]}
    )
    lq, rq = source("l", left), source("r", right)
    if aliases:
        lq = QueryExpr(op="alias", inputs=(lq,), params={"name": "old"})
        rq = QueryExpr(op="alias", inputs=(rq,), params={"name": "new"})
    la, ra = ("old", "new") if aliases else ("left", "right")
    a, b = (
        ColumnExpr("a", la if aliases else None),
        ColumnExpr("b", ra if aliases else None),
    )
    conditions = (b == a, ColumnExpr("rank") <= ColumnExpr("n"))
    query = QueryExpr(
        op="join",
        inputs=(lq, rq),
        params={"on": tuple(c.to_param() for c in conditions)},
    )
    actual = adapter.execute(query, {"l": left, "r": right})
    expected = pd.concat(
        [
            left.iloc[[0, 0, 1]].reset_index(drop=True).add_suffix(f":{la}"),
            right.iloc[[1, 2, 2]].reset_index(drop=True).add_suffix(f":{ra}"),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_predicate_join_does_not_coerce_text_or_scan_all_pairs(
    adapter: LotusAdapter | DocetlAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = pd.DataFrame({"a": [7]}), pd.DataFrame({"b": ["7"]})
    eq = (ColumnExpr("a") == ColumnExpr("b")) & (ColumnExpr("a") > 0)
    query = QueryExpr(
        op="join",
        inputs=(source("l", left), source("r", right)),
        params={"on": (eq.to_param(),)},
    )
    merge = pd.DataFrame.merge

    def guarded(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert kwargs.get("how") != "cross"
        return merge(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", guarded)
    actual = adapter.execute(query, {"l": left, "r": right})
    pd.testing.assert_frame_equal(
        actual,
        pd.concat(
            [left.iloc[:0].add_suffix(":left"), right.iloc[:0].add_suffix(":right")],
            axis=1,
        ),
    )


@pytest.mark.parametrize("backend", [lotus, docetl])
def test_sem_join_candidates_use_matching_source_ids(backend: Any) -> None:
    left = pd.DataFrame({"a": [3, 7, 7]}, index=[40, 50, 60])
    right = pd.DataFrame({"b": [7, "7", 7]}, index=[10, 11, 12])
    on = ((ColumnExpr("a") == ColumnExpr("b")).to_param(),)
    q = QueryExpr(
        op="sem_join", inputs=(source("l", left), source("r", right)), params={"on": on}
    )
    assert backend.predicate_join_row_ids(q, left, right, on=on) == (
        (50, 10),
        (50, 12),
        (60, 10),
        (60, 12),
    )


def test_predicate_join_rejects_ambiguous_bare_column(
    adapter: LotusAdapter | DocetlAdapter,
) -> None:
    frame = pd.DataFrame({"id": [7]})
    condition = ColumnExpr("id") == 7
    query = QueryExpr(
        op="join",
        inputs=(source("l", frame), source("r", frame)),
        params={"on": (condition.to_param(),)},
    )
    with pytest.raises(ValueError, match="(?i)ambiguous.*id"):
        adapter.execute(query, {"l": frame, "r": frame})


@pytest.mark.parametrize("op", ["subtract", "left_anti"])
@pytest.mark.parametrize("empty", [False, True])
def test_difference_preserves_user_marker_columns(
    adapter: LotusAdapter | DocetlAdapter, op: str, empty: bool
) -> None:
    marker = f"_agent_memory_{op}_marker"
    left = pd.DataFrame(
        {
            "key": [7, 8, 7, 7, 8],
            marker: pd.array([None, 4, 9, 9, 4], dtype="Int64"),
            "_" + marker: ["x", "y", "z", "z", "y"],
        }
    )
    right = left.iloc[[] if empty else [0, 0]]
    query = QueryExpr(
        op="subtract" if op == "subtract" else "join",
        inputs=(source("l", left), source("r", right)),
        params={} if op == "subtract" else {"on": ("key",), "how": op},
    )
    actual = adapter.execute(query, {"l": left, "r": right})
    expected = (
        left if empty else left.iloc[[1, 2, 3, 4] if op == "subtract" else [1, 4]]
    )
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True))
