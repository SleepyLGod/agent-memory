"""Offline regressions for compact semantic output bookkeeping."""

from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.runtime.row_outputs import RowOutputCache


class FilterOracle:
    """Keep positive inputs; relational work still uses the real adapter."""

    def __init__(self) -> None:
        self.inputs: list[pd.DataFrame] = []
        self.delegate = LotusAdapter()

    def execute(self, query: Any, inputs: Any) -> pd.DataFrame:
        if query.op == "sem_filter":
            frame = inputs[str(query.inputs[0].params["name"])]
            self.inputs.append(frame.copy(deep=True))
            return frame.loc[frame["value"] > 0].copy()
        return self.delegate.execute(query, inputs)


def build_flow(adapter: FilterOracle) -> Any:
    source = am.Source({"value": "number"})
    return am.SemanticDataflow(
        source=source,
        views={"result": source.sem_filter(instruction="Keep positive {value}.")},
        adapter=adapter,
    )


def test_rejected_occurrences_share_empty_output_across_events_and_restore() -> None:
    adapter = FilterOracle()
    flow = build_flow(adapter)
    for _ in range(3):
        flow.apply(pd.DataFrame({"value": [0] * 100}))
    snapshot = flow.snapshot_state()
    cache = next(iter(snapshot["semantic_output_cache"].values()))
    assert len(cache) == 300
    assert all(frame.empty for frame in cache.values())
    assert len({id(frame) for frame in cache.values()}) == 1
    assert snapshot["schema_version"] == 2
    assert type(cache) is dict
    restored = build_flow(adapter)
    restored.restore_state(snapshot)
    restored.apply(pd.DataFrame({"value": [1, 0, 1]}))
    assert restored.view("result")["value"].tolist() == [1, 1]
    assert [len(frame) for frame in adapter.inputs] == [100, 100, 100, 3]


def test_restore_compacts_legacy_independent_empties_without_changing_snapshot() -> None:
    adapter = FilterOracle()
    flow = build_flow(adapter)
    flow.apply(pd.DataFrame({"value": [0] * 30 + [1]}))
    snapshot = flow.snapshot_state()
    node, cache = next(iter(snapshot["semantic_output_cache"].items()))
    legacy = {key: frame.copy(deep=True) for key, frame in cache.items()}
    snapshot["semantic_output_cache"][node] = legacy
    restored = build_flow(adapter)
    restored.restore_state(snapshot)
    current = restored.snapshot_state()["semantic_output_cache"][node]
    assert len({id(v) for v in legacy.values() if v.empty}) == 30
    assert len({id(v) for v in current.values() if v.empty}) == 1
    for key in legacy:
        pd.testing.assert_frame_equal(legacy[key], current[key])
    assert len(adapter.inputs) == 1
    wrong = {**snapshot, "plan_fingerprint": "different"}
    with pytest.raises(ValueError, match="fingerprint"):
        restored.restore_state(wrong)


def test_membership_transitions_reproduce_dictionary_order_and_multiplicity() -> None:
    empty = pd.DataFrame({"x": pd.Series(dtype="Int64")})
    first = pd.DataFrame({"x": pd.array([1, None, 1], dtype="Int64")}, index=pd.Index(["a"] * 3))
    second = first.assign(x=pd.array([2, None, 2], dtype="Int64"))
    cache = RowOutputCache()
    reference: dict[Any, pd.DataFrame] = {}
    changes = [("a", empty), ("b", first), ("a", second), ("b", empty),
               ("c", first), ("b", second), ("a", None), ("a", first)]
    for key, frame in changes:
        if frame is None:
            del cache[key]
            del reference[key]
        else:
            cache[key] = frame
            reference[key] = frame
        assert list(cache) == list(reference)
        actual = cache.nonempty_frames()
        expected = [v for v in reference.values() if not v.empty]
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            pd.testing.assert_frame_equal(left, right)


def test_forked_bookkeeping_and_output_frames_do_not_mutate_committed_state() -> None:
    empty = pd.DataFrame({"x": pd.Series(dtype="int64")})
    old = pd.DataFrame({"x": [1], "metadata": [["original"]]})
    cache = RowOutputCache({"empty": empty, "kept": old})
    fork = cache.copy()
    fork.pop("kept")
    fork["empty"] = pd.DataFrame({"x": [2], "metadata": [["new"]]})
    fork["new"] = empty.rename(columns={"x": "other"})
    assert list(cache) == ["empty", "kept"]
    assert cache["empty"].empty
    pd.testing.assert_frame_equal(cache["kept"], old)
    assert list(fork) == ["empty", "new"]
    assert fork.nonempty_frames()[0]["metadata"].tolist() == [["new"]]


def test_nonempty_enumeration_does_not_inspect_empty_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = pd.DataFrame({"x": pd.Series(dtype="int64")})
    cache = RowOutputCache({i: empty for i in range(10000)})
    kept = pd.DataFrame({"x": [1]})
    cache["kept"] = kept

    def forbidden(_: pd.DataFrame) -> bool:
        raise AssertionError("enumeration inspected a frame's emptiness")

    monkeypatch.setattr(pd.DataFrame, "empty", property(forbidden))
    result = cache.nonempty_frames()
    assert len(result) == 1 and result[0] is kept


@pytest.mark.parametrize("kind", ["dtype", "column", "index_name", "index_dtype", "category", "flags", "attrs"])
def test_empty_schema_and_metadata_are_not_conflated(kind: str) -> None:
    base = pd.DataFrame({"x": pd.Series(dtype="int64")})
    other = base.copy(deep=True)
    if kind == "dtype":
        other = other.astype({"x": "Int64"})
    elif kind == "column":
        other.columns = pd.Index(["other"])
    elif kind == "index_name":
        other.index.name = "occurrence"
    elif kind == "index_dtype":
        other.index = pd.Index([], dtype="object")
    elif kind == "category":
        other["x"] = pd.Categorical([], categories=["one", "two"], ordered=True)
    elif kind == "flags":
        other.flags.allows_duplicate_labels = False
    else:
        other.attrs["source"] = {"nested": ["value"]}
    cache = RowOutputCache({"a": base, "b": other})
    assert cache["a"] is not cache["b"]
    pd.testing.assert_frame_equal(cache["a"], base)
    pd.testing.assert_frame_equal(cache["b"], other)
    assert cache["b"].attrs == other.attrs


def test_empty_template_does_not_alias_mutable_caller_frame() -> None:
    empty = pd.DataFrame({"x": pd.Series(dtype="int64")})
    cache = RowOutputCache({"a": empty})
    empty.columns = pd.Index(["changed"])
    assert cache["a"].columns.tolist() == ["x"]
    cache["b"] = empty
    assert cache["b"].columns.tolist() == ["changed"]


def test_rejected_predicate_membership_survives_metadata_replacement_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_replacement_screening import build

    _, view, flow, adapter, full, _, _ = build(monkeypatch)
    calls: list[int] = []

    def predicate(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(len(frame))
        return frame.loc[~frame["fact_earlier"].isin(["A", "C"])].copy()

    monkeypatch.setattr(pd.DataFrame, "sem_filter", predicate)
    initial = pd.DataFrame({"id": range(7), "fact": list("ABCDEFG"), "episode": ["old"] * 7})
    flow.apply(initial)
    snapshot = flow.snapshot_state()
    flow.restore_state(snapshot)
    call_count = len(calls)
    extra = initial.iloc[[0, 2]].assign(episode="new source")
    flow.apply(extra)
    assert len(calls) == call_count
    expected = full.execute(view.expr, {"log": pd.concat([initial, extra], ignore_index=True)})
    actual = flow.view("result")
    assert len(actual) == len(expected)
    keys = ["id:earlier", "id:later"]
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
    )
    # Full has its own row order. Maintenance must preserve the legacy cache's
    # insertion order, including rows re-appended after metadata replacement.
    cache = next(iter(flow._executor._semantic_output_cache.values()))
    legacy_order = pd.concat([frame for frame in cache.values() if not frame.empty])
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), legacy_order.reset_index(drop=True))
    assert adapter is flow._executor.adapter
