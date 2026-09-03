"""Focused tests for count-triggered relation-valued refresh."""

from __future__ import annotations

import json
from typing import Any

import agent_memory as am
import pandas as pd
import pytest
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
)
from agent_memory.policy.logical import QueryExpr


class _CountingAdapter(LotusAdapter):
    """Record root operator executions while preserving normal adapter behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.fail_assign = False

    def execute(self, query: QueryExpr, inputs: dict[str, Any]) -> Any:
        self.calls.append(query.op)
        if query.op == "assign" and self.fail_assign:
            raise RuntimeError("injected refresh failure")
        return super().execute(query, inputs)


class _BufferedMemory(am.Memory):
    log = am.Log({"message": "Message body."}, system_columns=True)
    rows = log.assign(published=True).select(
        ["message", "published", "_row_id", "_add_seq"]
    )
    retrieval_query = rows


class _SemanticFilterAdapter(LotusAdapter):
    """Accept every semantic-filter row and record the relation delta size."""

    def __init__(self) -> None:
        super().__init__()
        self.input_sizes: list[int] = []

    def execute(self, query: QueryExpr, inputs: dict[str, Any]) -> Any:
        if query.op == "sem_filter":
            source = inputs[str(query.inputs[0].params["name"])]
            self.input_sizes.append(len(source))
            return source.copy()
        return super().execute(query, inputs)


class _SemanticFilterMemory(am.Memory):
    log = am.Log({"message": "Message body."})
    rows = log.sem_filter(instruction="Keep {message}.")


class _SemanticSelfJoinMemory(am.Memory):
    log = am.Log({"key": "Exact partition.", "value": "Semantic value."})
    _left = log.alias("left")
    _right = log.alias("right")
    pairs = _left.sem_join(
        _right,
        on="key",
        instruction="{value:left} and {value:right} mean the same thing.",
    )


class _WindowMemory(am.Memory):
    log = am.Log({"message": "Message body."})
    blocks = log.count_window(size=2, slide=1).process_window(
        lambda window: window.array_agg(
            columns=("message",),
            output_col="messages",
        )
    )


def test_count_refresh_publishes_one_relation_delta_at_threshold() -> None:
    adapter = _CountingAdapter()
    memory = _BufferedMemory(adapter=adapter, refresh=am.CountRefresh(every=3))

    memory.add({"message": "one"})
    memory.add({"message": "two"})

    assert memory.pending_count == 2
    assert memory._runtime._state == {}
    assert "assign" not in adapter.calls

    memory.add({"message": "three"})

    assert memory.pending_count == 0
    assert adapter.calls.count("assign") == 1
    assert memory._runtime._state["rows"]["message"].tolist() == [
        "one",
        "two",
        "three",
    ]
    assert memory._runtime._state["rows"]["_add_seq"].tolist() == [0, 1, 2]
    assert memory._runtime._state["rows"]["_row_id"].is_unique


def test_query_reads_committed_view_and_flush_publishes_the_tail() -> None:
    memory = _BufferedMemory(refresh=am.CountRefresh(every=3))
    for message in ("one", "two", "three"):
        memory.add({"message": message})
    memory.add({"message": "four"})

    result = memory.query("unused")

    assert result["message"].tolist() == ["one", "two", "three"]
    assert memory.pending_count == 1

    memory.flush()

    assert memory.pending_count == 0
    assert memory._runtime._state["rows"]["message"].tolist() == [
        "one",
        "two",
        "three",
        "four",
    ]


def test_failed_refresh_retains_pending_rows_and_publishes_nothing() -> None:
    adapter = _CountingAdapter()
    adapter.fail_assign = True
    memory = _BufferedMemory(adapter=adapter, refresh=am.CountRefresh(every=2))

    memory.add({"message": "one"})
    pending_ids = [memory._runtime._pending_rows[0]["_row_id"]]
    with pytest.raises(RuntimeError, match="injected refresh failure"):
        memory.add({"message": "two"})
    pending_ids.append(memory._runtime._pending_rows[1]["_row_id"])

    assert memory.pending_count == 2
    assert memory._runtime._state == {}
    assert memory._runtime._engine.node_state == {}

    adapter.fail_assign = False
    memory.flush()

    assert memory.pending_count == 0
    assert memory._runtime._state["rows"]["_row_id"].tolist() == pending_ids


def test_count_refresh_snapshot_restores_pending_rows_with_matching_config() -> None:
    original = _BufferedMemory(refresh=am.CountRefresh(every=3))
    original.add({"message": "one"})
    original.add({"message": "two"})

    snapshot = original._runtime.snapshot_state()
    restored = _BufferedMemory(refresh=am.CountRefresh(every=3))
    restored._runtime.restore_state(snapshot)

    assert snapshot["schema_version"] == 3
    assert snapshot["refresh"] == {"type": "count", "every": 3}
    assert restored.pending_count == 2

    restored.flush()

    assert restored._runtime._state["rows"]["message"].tolist() == ["one", "two"]
    assert restored._runtime._state["rows"]["_row_id"].tolist() == snapshot[
        "pending_rows"
    ]["_row_id"].tolist()


def test_count_refresh_restore_rejects_a_different_refresh_contract() -> None:
    original = _BufferedMemory(refresh=am.CountRefresh(every=3))
    original.add({"message": "one"})
    snapshot = original._runtime.snapshot_state()
    restored = _BufferedMemory(refresh=am.CountRefresh(every=4))

    with pytest.raises(ValueError, match="refresh contract"):
        restored._runtime.restore_state(snapshot)


def test_deferred_runtime_rejects_an_eager_checkpoint() -> None:
    eager = _BufferedMemory()
    eager.add({"message": "one"})
    deferred = _BufferedMemory(refresh=am.CountRefresh(every=3))

    with pytest.raises(ValueError, match="schema-v3"):
        deferred._runtime.restore_state(eager._runtime.snapshot_state())


def test_eager_runtime_rejects_a_deferred_checkpoint() -> None:
    deferred = _BufferedMemory(refresh=am.CountRefresh(every=3))
    deferred.add({"message": "one"})
    eager = _BufferedMemory()

    with pytest.raises(ValueError, match="matching count refresh contract"):
        eager._runtime.restore_state(deferred._runtime.snapshot_state())


def test_count_refresh_restore_rejects_invalid_pending_sequence() -> None:
    original = _BufferedMemory(refresh=am.CountRefresh(every=3))
    original.add({"message": "one"})
    snapshot = original._runtime.snapshot_state()
    snapshot["pending_rows"].loc[0, "_add_seq"] = 7
    restored = _BufferedMemory(refresh=am.CountRefresh(every=3))

    with pytest.raises(ValueError, match="pending add sequence"):
        restored._runtime.restore_state(snapshot)

    assert restored.pending_count == 0
    assert restored._runtime._state == {}


def test_count_refresh_restore_rejects_duplicate_pending_row_ids() -> None:
    original = _BufferedMemory(refresh=am.CountRefresh(every=3))
    original.add({"message": "one"})
    original.add({"message": "two"})
    snapshot = original._runtime.snapshot_state()
    snapshot["pending_rows"].loc[1, "_row_id"] = snapshot["pending_rows"].loc[
        0, "_row_id"
    ]
    restored = _BufferedMemory(refresh=am.CountRefresh(every=3))

    with pytest.raises(ValueError, match="row IDs must be unique"):
        restored._runtime.restore_state(snapshot)


def test_count_refresh_every_one_is_exactly_eager() -> None:
    adapter = _CountingAdapter()
    memory = _BufferedMemory(adapter=adapter, refresh=am.CountRefresh(every=1))

    memory.add({"message": "one"})

    assert memory.pending_count == 0
    assert adapter.calls.count("assign") == 1
    assert memory._runtime.snapshot_state()["schema_version"] == 2


def test_eager_runtime_keeps_schema_v2_checkpoint_contract() -> None:
    memory = _BufferedMemory()
    memory.add({"message": "one"})

    assert memory.pending_count == 0
    assert memory._runtime.snapshot_state()["schema_version"] == 2


def test_relation_delta_reaches_semantic_row_operator_in_one_execution() -> None:
    adapter = _SemanticFilterAdapter()
    memory = _SemanticFilterMemory(
        adapter=adapter,
        refresh=am.CountRefresh(every=3),
    )

    for message in ("one", "two", "three"):
        memory.add({"message": message})

    assert adapter.input_sizes == [3]
    assert memory._runtime._state["rows"]["message"].tolist() == [
        "one",
        "two",
        "three",
    ]


def test_relation_delta_drives_semantic_self_join_once_without_missing_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    candidate_sizes: list[int] = []

    def verify(candidates: pd.DataFrame, **_kwargs: object) -> list[tuple[Any, Any, None]]:
        if not candidates.empty:
            candidate_sizes.append(len(candidates))
        return [
            (
                row[PAIR_LEFT_ID_COLUMN],
                row[PAIR_RIGHT_ID_COLUMN],
                None,
            )
            for _index, row in candidates.iterrows()
        ]

    monkeypatch.setattr(sem_join_module, "verify_semantic_join_candidates", verify)
    memory = _SemanticSelfJoinMemory(refresh=am.CountRefresh(every=2))

    memory.add({"key": "all", "value": "one"})
    memory.add({"key": "all", "value": "two"})

    assert candidate_sizes == [4]
    assert len(memory._runtime._state["pairs"]) == 4


def test_relation_delta_completes_all_ready_count_windows() -> None:
    memory = _WindowMemory(refresh=am.CountRefresh(every=3))

    for message in ("one", "two", "three"):
        memory.add({"message": message})

    blocks = memory._runtime._state["blocks"]
    assert len(blocks) == 2
    assert json.loads(blocks.loc[0, "messages"]) == [
        {"message": "one"},
        {"message": "two"},
    ]
    assert json.loads(blocks.loc[1, "messages"]) == [
        {"message": "two"},
        {"message": "three"},
    ]


@pytest.mark.parametrize("every", [0, -1])
def test_count_refresh_rejects_non_positive_thresholds(every: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        am.CountRefresh(every=every)


@pytest.mark.parametrize("every", [True, 1.5, "2"])
def test_count_refresh_rejects_non_integer_thresholds(every: Any) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        am.CountRefresh(every=every)
