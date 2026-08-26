from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from neo4j.time import DateTime
import pytest

from agent_memory.evaluation.attempt_metrics import (
    classify_attempt_rows,
    partition_attempt_rows,
)
from agent_memory.evaluation.question_results import AtomicQuestionStore
from agent_memory.evaluation.recovery import (
    ArtifactContractError,
    UnitAttemptExhausted,
    UnitAttemptStore,
    is_retryable_unit_error,
)
from agent_memory.evaluation.trace_metrics import summarize_provider_calls


def test_unit_attempt_store_allows_twenty_failures_and_rejects_twenty_first(
    tmp_path: Path,
) -> None:
    store = UnitAttemptStore(tmp_path)

    for attempt in range(1, 21):
        unit_attempt = store.begin(
            phase="insertion",
            unit_id="event-1",
            execution_attempt=attempt,
        )
        assert unit_attempt == attempt
        exhausted = store.finish(
            phase="insertion",
            unit_id="event-1",
            execution_attempt=attempt,
            unit_attempt=unit_attempt,
            status="failed",
            error=RuntimeError("retryable"),
        )
        assert exhausted is (attempt == 20)

    with pytest.raises(UnitAttemptExhausted, match="exhausted 20 attempts"):
        store.begin(
            phase="insertion",
            unit_id="event-1",
            execution_attempt=21,
        )


def test_contract_failure_does_not_consume_unit_attempt(tmp_path: Path) -> None:
    store = UnitAttemptStore(tmp_path)

    first = store.begin(
        phase="question",
        unit_id="q1",
        execution_attempt=1,
    )
    assert first == 1
    contract_error = ArtifactContractError("invalid scorer contract")
    assert isinstance(contract_error, ValueError)

    second = store.begin(
        phase="question",
        unit_id="q1",
        execution_attempt=2,
    )
    assert second == 1
    assert store.retryable_failure_count() == 0


def test_atomic_question_store_serializes_read_only_nested_mappings(
    tmp_path: Path,
) -> None:
    store = AtomicQuestionStore(tmp_path)
    store.publish(
        question_id="q1",
        contract_fingerprint="contract",
        retrieval=MappingProxyType(
            {
                "channels": MappingProxyType(
                    {"memories": (MappingProxyType({"memory": "tea"}),)}
                )
            }
        ),
        answer={"answer": "tea"},
        grades=({"scorer_id": "exact", "score": 1.0},),
        execution_attempt=1,
        unit_attempt=1,
    )

    store.rebuild_jsonl(("q1",))

    assert (tmp_path / "retrieval.jsonl").read_text(encoding="utf-8") == (
        '{"channels": {"memories": [{"memory": "tea"}]}}\n'
    )


def test_atomic_question_store_serializes_neo4j_temporal_values(
    tmp_path: Path,
) -> None:
    store = AtomicQuestionStore(tmp_path)
    store.publish(
        question_id="q1",
        contract_fingerprint="contract",
        retrieval={
            "channels": {
                "facts": [
                    {
                        "valid_at": DateTime(2026, 8, 26, 13, 14, 15, 0),
                    }
                ]
            }
        },
        answer={"answer": "tea"},
        grades=({"scorer_id": "exact", "score": 1.0},),
        execution_attempt=1,
        unit_attempt=1,
    )

    store.rebuild_jsonl(("q1",))

    assert (tmp_path / "retrieval.jsonl").read_text(encoding="utf-8") == (
        '{"channels": {"facts": [{"valid_at": '
        '"2026-08-26T13:14:15.000000000"}]}}\n'
    )


def test_only_transient_provider_failures_are_retryable() -> None:
    class AuthenticationError(RuntimeError):
        status_code = 401

    class ServiceUnavailableError(RuntimeError):
        status_code = 503

    assert not is_retryable_unit_error(AuthenticationError("invalid key"))
    assert not is_retryable_unit_error(RuntimeError("invalid output"))
    assert is_retryable_unit_error(TimeoutError("timed out"))
    assert is_retryable_unit_error(ServiceUnavailableError("service busy"))


def test_attempt_metrics_use_durable_lineage_and_preserve_unknown_cost() -> None:
    actual = [
        {
            "case_id": "case-1",
            "phase": "insertion",
            "event_id": "event-1",
            "execution_attempt": 1,
            "unit_attempt": 1,
            "latency_ms": 5.0,
            "usage_available": True,
            "prompt_tokens": 10,
            "cache_hit_tokens": 2,
            "cache_miss_tokens": 8,
            "completion_tokens": 3,
            "reasoning_tokens": 0,
            "known_cost_usd": 0.1,
            "estimated_cost_usd": 0.1,
        },
        {
            "case_id": "case-1",
            "phase": "insertion",
            "event_id": "event-1",
            "execution_attempt": 2,
            "unit_attempt": 2,
            "latency_ms": 7.0,
            "usage_available": False,
            "prompt_tokens": None,
            "cache_hit_tokens": None,
            "cache_miss_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "known_cost_usd": 0.0,
            "estimated_cost_usd": None,
        },
    ]
    classified = classify_attempt_rows(
        actual,
        durable_units={
            ("case-1", "insertion", "event-1", 2, 2),
        },
        authoritative_cases={"case-1"},
    )
    final, recovery = partition_attempt_rows(classified)

    assert [row["execution_attempt"] for row in final] == [2]
    assert [row["execution_attempt"] for row in recovery] == [1]
    actual_summary = summarize_provider_calls(classified)
    final_summary = summarize_provider_calls(final)
    recovery_summary = summarize_provider_calls(recovery)
    assert actual_summary["known_cost_usd"] == 0.1
    assert actual_summary["estimated_cost_usd"] is None
    assert actual_summary["usage_complete"] is False
    assert final_summary["known_cost_usd"] == 0.0
    assert final_summary["estimated_cost_usd"] is None
    assert recovery_summary["estimated_cost_usd"] == 0.1


def test_failed_provider_attempt_in_durable_unit_is_recovery_overhead() -> None:
    rows = classify_attempt_rows(
        [
            {
                "case_id": "case-1",
                "phase": "answering",
                "question_id": "q1",
                "execution_attempt": 2,
                "unit_attempt": 1,
                "status": "error",
            },
            {
                "case_id": "case-1",
                "phase": "answering",
                "question_id": "q1",
                "execution_attempt": 2,
                "unit_attempt": 1,
                "status": "success",
            },
        ],
        durable_units={("case-1", "question", "q1", 2, 1)},
        authoritative_cases={"case-1"},
    )

    assert [row["successful_path"] for row in rows] == [False, True]
