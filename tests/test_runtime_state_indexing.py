"""Exact occurrence and candidate-order contracts for derived runtime indexes."""

from collections import deque
from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

import agent_memory.runtime.executor as runtime
import agent_memory as am
from agent_memory.planner.physical import walk
from agent_memory.tracing.semantic import query_digest
from test_replacement_screening import build


def test_row_changes_keep_duplicate_identity_and_allocator_order(monkeypatch: pytest.MonkeyPatch) -> None:
    old = pd.DataFrame({"v": ["a", "a", "b", "c", "a"]}, index=pd.Index(["a0", "a1", "b", "c", "a2"]))
    new = pd.DataFrame({"v": ["b", "a", "d", "a", "e"]})
    allocated: list[str] = []
    keys: list[tuple[Any, ...]] = []
    original = runtime._row_key

    def key(row: tuple[Any, ...]) -> tuple[Any, ...]:
        keys.append(row)
        return original(row)

    def allocate() -> str:
        value = f"new{len(allocated)}"
        allocated.append(value)
        return value

    monkeypatch.setattr(runtime, "_row_key", key)
    update = runtime._preserve_occurrences_and_diff(old, new, allocate=allocate)
    assert list(update.output_rows.index) == ["b", "a0", "new0", "a1", "new1"]
    assert list(update.inserted_rows.index) == ["new0", "new1"]
    assert list(update.retracted_rows.index) == ["c", "a2"]
    assert allocated == ["new0", "new1"]
    assert len(keys) == len(old) + len(new)


@pytest.mark.parametrize("empty", [None, "old", "new", "both"])
def test_row_changes_match_existing_complex_cell_contract(empty: str | None) -> None:
    old = pd.DataFrame({"v": [None, {"a": [1, None]}, (2, {3, 4}), pd.NA, float("nan")],
                        "n": pd.array([1, 2, 3, 1, 1], dtype="Int64")})
    old.index = pd.Index(["o0", "o1", "o2", "o3", "o4"], name="old_name")
    new = old.iloc[[2, 0, 1, 1]].copy()
    if empty in {"old", "both"}:
        old = old.iloc[:0]
    if empty in {"new", "both"}:
        new = new.iloc[:0]
    ids = deque(f"n{i}" for i in range(20))
    expected = runtime.preserve_occurrence_index(old, new, allocate=ids.popleft)
    expected_update = runtime.NodeOutputUpdate.between(old, expected)
    ids = deque(f"n{i}" for i in range(20))
    actual = runtime._preserve_occurrences_and_diff(old, new, allocate=ids.popleft)
    for field in ("output_rows", "inserted_rows", "retracted_rows"):
        pd.testing.assert_frame_equal(getattr(actual, field), getattr(expected_update, field))
    with pytest.raises(ValueError, match="ordered columns"):
        runtime._preserve_occurrences_and_diff(old, new[["n", "v"]], allocate=ids.popleft)


def test_row_changes_preserve_occurrence_scalar_types() -> None:
    old = pd.DataFrame({"value": [1, 2]}, index=pd.Index([11, 12]))
    expected = runtime.preserve_occurrence_index(old, old, allocate=lambda: 99)
    actual = runtime._preserve_occurrences_and_diff(old, old, allocate=lambda: 99).output_rows
    assert [(type(i), i) for i in actual.index] == [(type(i), i) for i in expected.index]


def test_bucket_index_copies_only_touched_members_and_preserves_order() -> None:
    old = pd.DataFrame({"bucket": [1, 2, 1, 3], "entity": ["x", "x", "y", "x"],
                        "metadata": [0, 0, 0, 0]}, index=pd.Index(["a", "b", "c", "d"]))
    columns = ("bucket", "entity")
    index = runtime._CandidateBucketIndex.build(old, columns)
    removed = old.loc[["a"]]
    added = pd.DataFrame({"bucket": [1, 2], "entity": ["x", "x"], "metadata": [1, 1]}, index=pd.Index(["e", "f"]))
    current = pd.concat([old.drop(index="a"), added])
    updated = index.updated(removed, added)
    assert updated.buckets[runtime._row_key((3, "x"))] is index.buckets[runtime._row_key((3, "x"))]
    keys = {runtime._row_key((1, "x")), runtime._row_key((2, "x"))}
    assert list(index.select(old, keys).index) == ["a", "b"]
    assert list(updated.select(current, keys).index) == ["b", "e", "f"]
    assert list(updated.select(current, keys)["metadata"]) == [0, 1, 1]
    empty = updated.updated(current, current.iloc[:0])
    assert empty.buckets == {}
    reborn = empty.updated(current.iloc[:0], current)
    pd.testing.assert_frame_equal(reborn.select(current, set(reborn.buckets)), current)


@pytest.mark.parametrize("values", [("a", "b"), (1, 2), (True, 1), ("a", 1), (("fact", 1), ("fact", 2))])
def test_bucket_updates_do_not_refreeze_each_repeated_identity(monkeypatch: pytest.MonkeyPatch, values: tuple[Any, Any]) -> None:
    old = pd.DataFrame({"bucket": list(values) * 100}, index=pd.Index([f"o{i}" for i in range(200)]))
    index = runtime._CandidateBucketIndex.build(old, ("bucket",))
    removed = old.iloc[:100]
    added = removed.copy()
    added.index = pd.Index([f"n{i}" for i in range(100)])
    calls: list[tuple[Any, ...]] = []
    original = runtime._row_key

    def key(row: tuple[Any, ...]) -> tuple[Any, ...]:
        calls.append(row)
        return original(row)

    monkeypatch.setattr(runtime, "_row_key", key)
    updated = index.updated(removed, added)
    assert len(calls) <= 4
    assert updated.row_count == 200


def test_complete_withdrawal_builds_only_new_members(monkeypatch: pytest.MonkeyPatch) -> None:
    old = pd.DataFrame({"bucket": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))
    index = runtime._CandidateBucketIndex.build(old, ("bucket",))
    added = pd.DataFrame({"bucket": [4, 5]}, index=pd.Index(["d", "e"]))
    original = runtime._row_key
    rows: list[tuple[Any, ...]] = []

    def key(row: tuple[Any, ...]) -> tuple[Any, ...]:
        rows.append(row)
        return original(row)

    monkeypatch.setattr(runtime, "_row_key", key)
    updated = index.updated(old, added)
    assert rows == [(4,), (5,)]
    assert updated.row_count == 2
    assert index.row_count == 3


def test_scalar_key_reuse_preserves_type_distinctions_and_bounded_work() -> None:
    class Text(str):
        pass

    values = ["1", Text("1"), 1, True, 1.0, None, pd.NA, {"a": [1]}] + [str(i) for i in range(200)]
    frame = pd.DataFrame({"key": values}, index=pd.Index([f"p{i}" for i in range(len(values))]))
    index = runtime._CandidateBucketIndex.build(frame, ("key",))
    expected: dict = {}
    for occurrence, value in zip(frame.index, values, strict=True):
        expected.setdefault(runtime._row_key((value,)), []).append(occurrence)
    assert {k: list(v) for k, v in index.buckets.items()} == expected


def test_bucket_memo_does_not_alias_temporary_timestamp_objects() -> None:
    frame = pd.DataFrame({"key": pd.date_range("2020-01-01", periods=400, tz="UTC")},
                         index=pd.Index([f"p{i}" for i in range(400)]))
    index = runtime._CandidateBucketIndex.build(frame, ("key",))
    assert len(index.buckets) == 400
    for occurrence, row in zip(frame.index, frame.itertuples(index=False, name=None), strict=True):
        assert list(index.buckets[runtime._row_key(row)]) == [occurrence]


def test_warm_lookup_visits_no_unaffected_bucket_or_frame_rows() -> None:
    frame = pd.DataFrame({"bucket": range(100), "value": range(100)}, index=pd.Index([f"p{i}" for i in range(100)]))
    index = runtime._CandidateBucketIndex.build(frame, ("bucket",))

    class NoScan(dict):
        def __iter__(self) -> Any:
            raise AssertionError("scanned all buckets")

        def items(self) -> Any:
            raise AssertionError("scanned all buckets")

        def values(self) -> Any:
            raise AssertionError("scanned all buckets")

    index.buckets = NoScan(index.buckets)
    pd.testing.assert_frame_equal(index.select(frame, {runtime._row_key((5,))}), frame.iloc[[5]])


def test_full_bucket_coverage_needs_no_member_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"bucket": [2, 1, 2], "value": [0, 1, 2]}, index=pd.Index(["a", "b", "c"]))
    index = runtime._CandidateBucketIndex.build(frame, ("bucket",))

    def unnecessary(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("full coverage must not merge all occurrence IDs")

    monkeypatch.setattr(runtime, "merge", unnecessary)
    selected = index.select(frame, set(index.buckets))
    pd.testing.assert_frame_equal(selected, frame)
    selected.loc["a", "value"] = 99
    assert frame.loc["a", "value"] == 0


def test_bucket_keys_preserve_nulls_nested_values_and_duplicate_occurrences() -> None:
    frame = pd.DataFrame({"key": [None, {"nested": [1, None]}, pd.NA, {"nested": [1, None]}],
                          "value": ["first", "second", "third", "fourth"]}, index=pd.Index(["a", "b", "c", "d"]))
    index = runtime._CandidateBucketIndex.build(frame, ("key",))
    pd.testing.assert_frame_equal(index.select(frame, {runtime._row_key((float("nan"),))}), frame.iloc[[0, 2]])
    removed = frame.iloc[[1]]
    added = frame.iloc[[1]].copy()
    added.index = pd.Index(["e"])
    added["value"] = "current metadata"
    current = pd.concat([frame.drop(index="b"), added])
    updated = index.updated(removed, added)
    chosen = updated.select(current, {runtime._row_key(({"nested": [1, None]},))})
    assert list(chosen.index) == ["d", "e"]
    assert list(chosen["value"]) == ["fourth", "current metadata"]


@pytest.mark.parametrize("direction", ["right-to-left", "left-to-right", "symmetric"])
def test_runtime_index_syncs_metadata_and_rebuilds_after_restore(monkeypatch: pytest.MonkeyPatch, direction: str) -> None:
    _, _, flow, _, _, _, calls = build(monkeypatch, direction)
    initial = pd.DataFrame({"id": range(7), "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    executor = flow._executor
    count = len(calls)
    flow.apply(initial.iloc[[0]].assign(episode="new metadata"))
    assert len(calls) == count
    indexes = executor._candidate_bucket_indexes
    assert bool(indexes) == (direction != "symmetric")
    for (parent, columns), index in indexes.items():
        frame = executor.node_state[parent]
        rebuilt = runtime._CandidateBucketIndex.build(frame, columns)
        assert set(index.buckets) == set(rebuilt.buckets)
        pd.testing.assert_frame_equal(index.select(frame, set(index.buckets)), frame)
        assert all(list(index.buckets[k]) == list(rebuilt.buckets[k]) for k in index.buckets)
    snapshot = flow.snapshot_state()
    assert "candidate_bucket_indexes" not in snapshot
    flow.restore_state(snapshot)
    assert executor._candidate_bucket_indexes == {}
    flow.apply(initial.iloc[[0]].assign(episode="after restore"))
    assert len(calls) == count
    assert bool(executor._candidate_bucket_indexes) == (direction != "symmetric")


@pytest.mark.parametrize("failure", ["model", "storage"])
def test_failed_step_never_commits_bucket_changes(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    _, _, flow, _, _, _, _ = build(monkeypatch)
    initial = pd.DataFrame({"id": [0, 2, 3, 4, 5, 6, 7], "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    executor = flow._executor
    before = {key: deepcopy(index.buckets) for key, index in executor._candidate_bucket_indexes.items()}
    original = getattr(pd.DataFrame, "sem_filter") if failure == "model" else executor._write_storage_updates

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected failure")

    target, name = (pd.DataFrame, "sem_filter") if failure == "model" else (executor, "_write_storage_updates")
    monkeypatch.setattr(target, name, fail)
    extra = pd.DataFrame({"id": [1], "fact": ["H"], "episode": ["new"]})
    with pytest.raises(RuntimeError, match="injected failure"):
        flow.apply(extra)
    assert {k: i.buckets for k, i in executor._candidate_bucket_indexes.items()} == before
    monkeypatch.setattr(target, name, original)
    flow.apply(extra)
    for (parent, _), index in executor._candidate_bucket_indexes.items():
        frame = executor.node_state[parent]
        pd.testing.assert_frame_equal(index.select(frame, set(index.buckets)), frame)


def test_full_runtime_matches_scan_path_including_calls_and_private_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, optimized, _, _, _, _ = build(monkeypatch)
    _, _, reference, _, _, _, calls = build(monkeypatch)
    prompts: list[pd.DataFrame] = []
    oracle = getattr(pd.DataFrame, "sem_filter")

    def observe(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        prompts.append(frame.copy(deep=True))
        return oracle(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sem_filter", observe)
    updates = [
        pd.DataFrame({"id": [0, 2, 3, 4, 5, 6, 7], "fact": list("ABCDEFG"), "episode": ["original"] * 7}),
        pd.DataFrame({"id": [0], "fact": ["A"], "episode": ["metadata"]}),
        pd.DataFrame({"id": [1], "fact": ["H"], "episode": ["stronger"]}),
        pd.DataFrame({"id": [2, 9], "fact": ["B", "H"], "episode": ["again", "new"]}),
    ]
    semantic = runtime.PolicyExecutor._execute_semantic_row

    def scan(self: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        kwargs["staged_bucket_updates"] = None
        return semantic(self, *args, **kwargs)

    def separate(old: pd.DataFrame, new: pd.DataFrame, *, allocate: Any) -> runtime.NodeOutputUpdate:
        return runtime.NodeOutputUpdate.between(old, runtime.preserve_occurrence_index(old, new, allocate=allocate))

    for rows in updates:
        calls.clear()
        prompts.clear()
        optimized.apply(rows)
        expected_calls = list(calls)
        expected_prompts = list(prompts)
        calls.clear()
        prompts.clear()
        with monkeypatch.context() as patch:
            patch.setattr(runtime.PolicyExecutor, "_execute_semantic_row", scan)
            patch.setattr(runtime, "_preserve_occurrences_and_diff", separate)
            reference.apply(rows)
        assert calls == expected_calls
        assert len(prompts) == len(expected_prompts)
        for actual, expected in zip(prompts, expected_prompts, strict=True):
            pd.testing.assert_frame_equal(actual, expected)
        assert optimized._executor._next_occurrence == reference._executor._next_occurrence
        for node_id, frame in optimized._executor.node_state.items():
            pd.testing.assert_frame_equal(frame, reference._executor.node_state[node_id])


def test_two_predicates_share_one_parent_index_update(monkeypatch: pytest.MonkeyPatch) -> None:
    source, view, _, adapter, _, _, _ = build(monkeypatch)
    state = source.group_by(["id", "fact"]).array_agg(columns=["episode"], output_col="provenance")
    left, right = state.alias("earlier"), state.alias("later")
    other = left.join(right, on=left.col("id") < right.col("id")).sem_filter(
        instruction="Again compare {fact:earlier} and {fact:later}.")
    profile = next(iter(adapter.config.semantic_pair_profiles.values()))
    unprepared = am.SemanticDataflow(source=source, views={"first": view, "second": other})._executor.policy
    for node in unprepared.nodes.values():
        for root in (node.query, node.maintenance_query):
            if root is not None:
                for query in walk(root):
                    if query.op == "sem_filter":
                        adapter.config.semantic_pair_profiles[query_digest(query)] = profile
    flow = am.SemanticDataflow(source=source, views={"first": view, "second": other}, adapter=adapter)
    updated = runtime._CandidateBucketIndex.updated
    counts: list[int] = []

    def record(self: Any, removed: pd.DataFrame, added: pd.DataFrame) -> Any:
        counts.append(len(added))
        return updated(self, removed, added)

    monkeypatch.setattr(runtime._CandidateBucketIndex, "updated", record)
    initial = pd.DataFrame({"id": range(7), "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    flow.apply(initial.iloc[[0]].assign(episode="metadata"))
    assert len(counts) == 2
    assert len(flow._executor._candidate_bucket_indexes) == 1
    pd.testing.assert_frame_equal(flow.view("first"), flow.view("second"))
