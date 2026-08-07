from __future__ import annotations

import json

import pytest

from agent_memory.evaluation.bundle import BenchmarkBundle, read_bundle, write_bundle
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
)


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-1",
        task_id="longmemeval",
        events=(
            BenchmarkEvent(
                sample_id="case-1",
                event_id="event-1",
                speaker="user",
                text="Remember this.",
                session_id="session-1",
                timestamp="2026-01-01",
                metadata={"turn_index": 0},
            ),
        ),
        questions=(
            BenchmarkQuestion(
                question_id="question-1",
                sample_id="case-1",
                question="What should you remember?",
                gold_answer="this",
                evidence_event_ids=("event-1",),
                category="single-session-user",
                metadata={"question_date": "2026-01-02"},
            ),
        ),
        metadata={"has_answer": True},
    )


def test_case_rejects_rows_from_another_case() -> None:
    event = _case().events[0]
    with pytest.raises(ValueError, match="events must belong"):
        BenchmarkCase(
            case_id="other",
            task_id="longmemeval",
            events=(event,),
            questions=_case().questions,
        )


def test_bundle_round_trip_preserves_order_and_fingerprints(tmp_path) -> None:
    bundle = BenchmarkBundle(
        benchmark_id="longmemeval-v1",
        dataset_revision="revision",
        dataset_sha256="sha256",
        cases=(_case(),),
        metadata={"selection": ["question-1"]},
    )

    write_bundle(bundle, tmp_path)
    restored = read_bundle(tmp_path)

    assert restored == bundle
    assert restored.fingerprint == bundle.fingerprint
    assert restored.policy_input_fingerprint == bundle.policy_input_fingerprint
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["case_count"] == 1
    assert manifest["event_count"] == 1
    assert manifest["question_count"] == 1


def test_policy_input_fingerprint_excludes_gold_and_evidence() -> None:
    first = BenchmarkBundle(
        benchmark_id="longmemeval-v1",
        dataset_revision="revision",
        dataset_sha256="sha256",
        cases=(_case(),),
    )
    changed_question = BenchmarkQuestion(
        **{
            **_case().questions[0].__dict__,
            "gold_answer": "changed",
            "evidence_event_ids": (),
        }
    )
    changed_case = BenchmarkCase(
        **{**_case().__dict__, "questions": (changed_question,)}
    )
    second = BenchmarkBundle(
        benchmark_id="longmemeval-v1",
        dataset_revision="revision",
        dataset_sha256="sha256",
        cases=(changed_case,),
    )

    assert first.policy_input_fingerprint == second.policy_input_fingerprint
    assert first.fingerprint != second.fingerprint


def test_write_bundle_refuses_to_overwrite_artifacts(tmp_path) -> None:
    (tmp_path / "keep.txt").write_text("keep")
    bundle = BenchmarkBundle(
        benchmark_id="longmemeval-v1",
        dataset_revision="revision",
        dataset_sha256="sha256",
        cases=(_case(),),
    )

    with pytest.raises(FileExistsError, match="not empty"):
        write_bundle(bundle, tmp_path)
