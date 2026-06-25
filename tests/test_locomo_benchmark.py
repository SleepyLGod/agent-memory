"""Tests for the isolated LOCOMO benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent_memory.benchmarks.diagnostics import build_llm_anomaly_rows
from agent_memory.benchmarks.types import BenchmarkEvent, BenchmarkQuestion
from agent_memory.benchmarks.locomo import (
    eligible_questions,
    event_to_claude_log_row,
    load_locomo_sample,
    normalize_locomo_sample,
    select_events,
)
from agent_memory.benchmarks.metrics import (
    contains_answer,
    duplicate_name_count,
    duplicate_name_extra_rows,
    exact_match,
    locomo_answer_score,
    locomo_f1_score,
    locomo_multi_answer_f1,
    question_metric_row,
    retrieval_hit,
    summarize_question_metrics,
    token_f1,
)
from examples.benchmarks.locomo_benchmark import (
    ANSWER_SYSTEM_PROMPT,
    BENCHMARK_CONTRACT,
    POLICY_CONTRACT,
    checkpoint_contract_digest,
    checkpoint_manifest_path,
    checkpoint_snapshots_dir,
    current_checkpoint_snapshot_dir,
    load_checkpoint,
    run_questions,
    save_checkpoint,
    summary_frame,
    write_failure_metadata,
    write_recovery_metadata,
    write_run_artifacts,
    write_jsonl_atomic,
)


def locomo_fixture() -> dict[str, Any]:
    """Return a tiny LOCOMO-like sample."""

    return {
        "sample_id": "conv-test",
        "conversation": {
            "session_1_date_time": "2026-01-01",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I started researching adoption agencies.",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:2",
                    "text": "That sounds like a big step.",
                },
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:3",
                    "text": "",
                },
            ],
        },
        "qa": [
            {
                "question": "What did Caroline research?",
                "answer": "adoption agencies",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": "Who encouraged Caroline?",
                "answer": "Melanie",
                "evidence": ["D1:2"],
                "category": 2,
            },
            {
                "question": "What happens in the future?",
                "answer": "unknown",
                "evidence": ["D9:1"],
                "category": 3,
            },
        ],
    }


def write_fixture(path: Path) -> Path:
    """Write the tiny LOCOMO-like fixture to disk."""

    path.write_text(json.dumps([locomo_fixture()]), encoding="utf-8")
    return path


def test_locomo_adapter_preserves_events_and_questions(tmp_path: Path) -> None:
    dataset_path = write_fixture(tmp_path / "locomo.json")

    sample = load_locomo_sample(dataset_path, sample_index=0)

    assert sample.sample_id == "conv-test"
    assert tuple(event.event_id for event in sample.events) == ("D1:1", "D1:2")
    assert sample.events[0].speaker == "Caroline"
    assert sample.events[0].text == "I started researching adoption agencies."
    assert sample.events[0].timestamp == "2026-01-01"
    assert sample.questions[0].question_id == "conv-test:q1"
    assert sample.questions[0].gold_answer == "adoption agencies"
    assert sample.questions[0].evidence_event_ids == ("D1:1",)
    assert sample.questions[0].category == "1"


def test_normalize_locomo_sample_uses_fallback_sample_id() -> None:
    fixture = dict(locomo_fixture())
    fixture.pop("sample_id")

    sample = normalize_locomo_sample(fixture, sample_index=4)

    assert sample.sample_id == "sample-4"
    assert all(event.sample_id == "sample-4" for event in sample.events)
    assert all(question.sample_id == "sample-4" for question in sample.questions)


def test_normalize_locomo_sample_skips_empty_evidence_ids() -> None:
    fixture = locomo_fixture()
    fixture["qa"].append(
        {
            "question": "Which evidence id remains?",
            "answer": "adoption agencies",
            "evidence": [None, "", "  ", "D1:1"],
            "category": 2,
        }
    )

    sample = normalize_locomo_sample(fixture)

    assert sample.questions[-1].evidence_event_ids == ("D1:1",)


def test_eligible_questions_require_ingested_evidence() -> None:
    sample = normalize_locomo_sample(locomo_fixture())

    selected = eligible_questions(
        sample.questions,
        ingested_event_ids=["D1:1"],
        question_limit=None,
    )

    assert tuple(question.question for question in selected) == (
        "What did Caroline research?",
    )


def test_eligible_questions_respects_question_limit() -> None:
    sample = normalize_locomo_sample(locomo_fixture())

    selected = eligible_questions(
        sample.questions,
        ingested_event_ids=["D1:1", "D1:2"],
        question_limit=1,
    )

    assert len(selected) == 1
    assert selected[0].question_id == "conv-test:q1"


def test_select_events_applies_row_limit() -> None:
    sample = normalize_locomo_sample(locomo_fixture())

    selected = select_events(sample.events, row_limit=1)

    assert tuple(event.event_id for event in selected) == ("D1:1",)


def test_event_to_claude_log_row_matches_current_schema() -> None:
    sample = normalize_locomo_sample(locomo_fixture())

    row = event_to_claude_log_row(sample.events[0])

    assert row == {
        "message": "I started researching adoption agencies.",
        "role": "Caroline",
        "timestamp": "2026-01-01",
        "session_id": "session_1",
    }


def test_text_metrics_handle_exact_contains_and_f1() -> None:
    assert exact_match("The adoption agencies", "adoption agencies")
    assert contains_answer("Caroline researched adoption agencies yesterday.", "adoption agencies")
    assert retrieval_hit("body: adoption agencies", "adoption agencies")
    assert token_f1("adoption agencies", "adoption agencies") == 1.0
    assert token_f1("adoption", "adoption agencies") > 0.0


def test_locomo_category_one_uses_multi_answer_partial_f1() -> None:
    score = locomo_answer_score(
        "Psychology, counseling",
        "psychology, counseling certification",
        1,
    )

    assert 0.8 < score < 1.0
    assert score == locomo_multi_answer_f1(
        "Psychology, counseling",
        "psychology, counseling certification",
    )


def test_locomo_categories_two_and_four_use_stemmed_f1() -> None:
    assert locomo_answer_score("running", "runs", 2) == 1.0
    assert locomo_answer_score("helped with childcare", "help with child care", 4) > 0


def test_locomo_category_three_uses_answer_before_semicolon() -> None:
    assert locomo_answer_score("psychology", "psychology; counseling certification", 3) == 1.0


def test_locomo_category_five_checks_no_info_answers() -> None:
    assert locomo_answer_score("No information available in the memory.", "anything", 5) == 1.0
    assert locomo_answer_score("I don't know.", "anything", 5) == 0.0
    assert locomo_answer_score("Caroline went yesterday.", "anything", 5) == 0.0


def test_answer_prompt_uses_locomo_category_five_no_info_phrase() -> None:
    assert "No information available" in ANSWER_SYSTEM_PROMPT
    assert "I don't know" not in ANSWER_SYSTEM_PROMPT


def test_locomo_smoke_date_case_gets_partial_f1_not_contains() -> None:
    prediction = "8 May 2023."
    gold = "7 May 2023"

    assert not exact_match(prediction, gold)
    assert not contains_answer(prediction, gold)
    assert locomo_f1_score(prediction, gold) == 2 / 3


def test_duplicate_name_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"name": "caroline", "body": "a"},
            {"name": "caroline", "body": "b"},
            {"name": "melanie", "body": "c"},
        ]
    )

    assert duplicate_name_count(frame) == 1
    assert duplicate_name_extra_rows(frame) == 1


def test_question_metric_row_handles_empty_retrieval() -> None:
    row = question_metric_row(
        question_id="q1",
        question="What did Caroline research?",
        gold_answer="adoption agencies",
        retrieved_frame=pd.DataFrame(),
    )

    assert row["retrieved_row_count"] == 0
    assert row["proxy_answer_string_hit"] is False
    assert "generated_answer" not in row
    assert "locomo_answer_score" not in row


def test_question_metric_row_adds_locomo_answer_score_when_answering() -> None:
    row = question_metric_row(
        question_id="q1",
        question="When did Caroline go?",
        gold_answer="7 May 2023",
        retrieved_frame=pd.DataFrame(),
        generated_answer="8 May 2023.",
        category=2,
    )

    assert row["locomo_answer_score"] == round(2 / 3, 6)
    assert row["answer_exact_match"] is False


def test_question_metric_row_scores_empty_generated_answer() -> None:
    row = question_metric_row(
        question_id="q1",
        question="When did Caroline go?",
        gold_answer="7 May 2023",
        retrieved_frame=pd.DataFrame(),
        generated_answer="",
        category=2,
    )

    assert row["generated_answer"] == ""
    assert row["answer_f1"] == 0.0
    assert row["locomo_answer_score"] == 0.0


def test_proxy_answer_string_hit_does_not_imply_answer_score() -> None:
    row = question_metric_row(
        question_id="q1",
        question="What did the race raise awareness for?",
        gold_answer="mental health",
        retrieved_frame=pd.DataFrame(
            [{"body": "Caroline wants to work in mental health."}]
        ),
        generated_answer="I don't know.",
        category=4,
    )

    assert row["proxy_answer_string_hit"] is True
    assert row["locomo_answer_score"] == 0.0


def test_summarize_question_metrics_handles_no_eligible_questions() -> None:
    summary = summarize_question_metrics([])

    assert summary["questions_evaluated"] == 0
    assert summary["proxy_answer_string_hit_rate"] == ""


def test_summarize_question_metrics_reports_locomo_category_means() -> None:
    rows = [
        {
            "category": "2",
            "proxy_answer_string_hit": True,
            "generated_answer": "",
            "answer_exact_match": False,
            "answer_contains_gold": False,
            "answer_f1": 0.0,
            "locomo_answer_score": 0.0,
        },
        {
            "category": "5",
            "proxy_answer_string_hit": False,
            "generated_answer": "No information available.",
            "answer_exact_match": False,
            "answer_contains_gold": False,
            "answer_f1": 0.0,
            "locomo_answer_score": 1.0,
        },
    ]

    summary = summarize_question_metrics(rows)

    assert summary["locomo_answer_score_mean"] == 0.5
    assert summary["answer_f1_mean"] == 0.0
    assert summary["category_2_count"] == 1
    assert summary["category_2_locomo_answer_score_mean"] == 0.0
    assert summary["category_5_locomo_answer_score_mean"] == 1.0


def test_summary_frame_records_input_rendering_contract() -> None:
    frame = summary_frame(
        run_mode="answer",
        model="test-model",
        sample_index=0,
        events=(),
        questions=(),
        memory=FakeBenchmarkMemory(),
        question_metrics=(),
        question_results=(),
        step_metrics=(),
    )
    row = frame.iloc[0]

    assert row["input_rendering"] == "message_with_event_context"
    assert bool(row["bookkeeping_metadata_excluded_from_semantic_input"]) is True


class FakeBenchmarkMemory:
    """Minimal memory object for runner helper tests."""

    def __init__(self) -> None:
        self._runtime = type(
            "Runtime",
            (),
            {
                "_state": {
                    "log": pd.DataFrame([{"message": "I researched adoption agencies."}]),
                    "topics": pd.DataFrame(
                        [{"name": "adoption", "body": "Caroline researched adoption agencies."}]
                    ),
                    "catalog": pd.DataFrame(
                        [{"catalog_title": "Adoption", "name": "adoption", "hook": "adoption"}]
                    ),
                }
            },
        )()
        self.calls = 0

    def query(self, question: str) -> pd.DataFrame:
        """Return one retrieval result, then fail on the next query."""

        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("query failed")
        return pd.DataFrame(
            [{"name": "adoption", "body": "Caroline researched adoption agencies."}]
        )


def benchmark_questions() -> tuple[BenchmarkQuestion, BenchmarkQuestion]:
    """Return two tiny benchmark questions."""

    return (
        BenchmarkQuestion(
            question_id="q1",
            sample_id="sample",
            question="What did Caroline research?",
            gold_answer="adoption agencies",
            evidence_event_ids=("D1:1",),
            category="2",
        ),
        BenchmarkQuestion(
            question_id="q2",
            sample_id="sample",
            question="Who helped?",
            gold_answer="Melanie",
            evidence_event_ids=("D1:2",),
            category="2",
        ),
    )


def benchmark_events() -> tuple[BenchmarkEvent, BenchmarkEvent]:
    """Return two tiny benchmark events."""

    return (
        BenchmarkEvent(
            sample_id="sample",
            event_id="D1:1",
            speaker="Caroline",
            text="I researched adoption agencies.",
        ),
        BenchmarkEvent(
            sample_id="sample",
            event_id="D1:2",
            speaker="Melanie",
            text="I helped.",
        ),
    )


def test_run_questions_preserves_completed_rows_when_later_question_fails() -> None:
    result_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    with pytest.raises(RuntimeError, match="query failed"):
        run_questions(
            FakeBenchmarkMemory(),
            benchmark_questions(),
            answer=False,
            result_rows=result_rows,
            metric_rows=metric_rows,
        )

    assert len(result_rows) == 1
    assert len(metric_rows) == 1
    assert result_rows[0]["question_id"] == "q1"
    assert result_rows[0]["proxy_answer_string_hit"] is True


def test_run_questions_can_resume_from_completed_question_rows() -> None:
    result_rows = [{"question_id": "q1", "retrieved_text": "adoption agencies"}]
    metric_rows = [{"question_id": "q1", "proxy_answer_string_hit": True}]
    memory = FakeBenchmarkMemory()

    run_questions(
        memory,
        benchmark_questions(),
        answer=False,
        result_rows=result_rows,
        metric_rows=metric_rows,
        start_index=1,
    )

    assert memory.calls == 1
    assert [row["question_id"] for row in result_rows] == ["q1", "q2"]
    assert [row["question_id"] for row in metric_rows] == ["q1", "q2"]


def test_checkpoint_round_trips_runtime_state_and_completed_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    memory = FakeBenchmarkMemory()
    events = benchmark_events()
    questions = benchmark_questions()
    step_metrics = [{"phase": "add", "event_id": "D1:1"}]
    result_rows = [{"question_id": "q1", "retrieved_text": "adoption agencies"}]
    metric_rows = [{"question_id": "q1", "proxy_answer_string_hit": True}]

    save_checkpoint(
        output_dir=output_dir,
        memory=memory,
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        trace_enabled=False,
        events=events,
        questions=questions,
        step_metrics=step_metrics,
        result_rows=result_rows,
        metric_rows=metric_rows,
    )
    state, loaded_steps, loaded_results, loaded_metrics = load_checkpoint(
        output_dir=output_dir,
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        trace_enabled=False,
        events=events,
        questions=questions,
    )

    pd.testing.assert_frame_equal(state["topics"], memory._runtime._state["topics"])
    assert loaded_steps == step_metrics
    assert loaded_results == result_rows
    assert loaded_metrics == metric_rows


def test_load_checkpoint_ignores_unreferenced_partial_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    step_metrics = [{"phase": "add", "event_id": "D1:1"}]
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=step_metrics,
        result_rows=[],
        metric_rows=[],
    )
    partial_snapshot = checkpoint_snapshots_dir(output_dir) / "partial-write"
    partial_snapshot.mkdir(parents=True)
    write_jsonl_atomic(
        partial_snapshot / "step_metrics.jsonl",
        [
            {"phase": "add", "event_id": "D1:1"},
            {"phase": "add", "event_id": "D1:2"},
        ],
    )

    _, loaded_steps, loaded_results, loaded_metrics = load_checkpoint(
        output_dir=output_dir,
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        trace_enabled=False,
        events=events,
        questions=questions,
    )

    assert loaded_steps == step_metrics
    assert loaded_results == []
    assert loaded_metrics == []


def test_load_checkpoint_rejects_mismatched_arguments(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="model"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="other-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_trace_mode_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
        trace_enabled=False,
    )

    with pytest.raises(SystemExit, match="trace_enabled"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=True,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_changed_input_content_with_same_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )
    changed_events = (
        BenchmarkEvent(
            sample_id="sample",
            event_id="D1:1",
            speaker="Caroline",
            text="I researched something else.",
        ),
        events[1],
    )

    with pytest.raises(SystemExit, match="input_digest"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=changed_events,
            questions=questions,
        )


def test_load_checkpoint_rejects_contract_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )
    manifest_path = checkpoint_manifest_path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_contract"] = f"old-{BENCHMARK_CONTRACT}"
    manifest["policy_contract"] = f"old-{POLICY_CONTRACT}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="benchmark_contract, policy_contract"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_checkpoint_contract_digest_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )
    manifest_path = checkpoint_manifest_path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint_contract_digest"] == checkpoint_contract_digest()
    manifest["checkpoint_contract_digest"] = "old-digest"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="checkpoint_contract_digest"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_progress_rows_that_do_not_match_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[{"phase": "add", "event_id": "D1:1"}],
        result_rows=[{"question_id": "q1", "retrieved_text": "adoption agencies"}],
        metric_rows=[{"question_id": "q1", "proxy_answer_string_hit": True}],
    )
    checkpoint_directory = current_checkpoint_snapshot_dir(output_dir)
    write_jsonl_atomic(checkpoint_directory / "step_metrics.jsonl", [])

    with pytest.raises(SystemExit, match="Checkpoint step metrics do not match manifest"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_same_length_wrong_event_prefix(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[{"phase": "add", "event_id": "D1:1"}],
        result_rows=[],
        metric_rows=[],
    )
    checkpoint_directory = current_checkpoint_snapshot_dir(output_dir)
    write_jsonl_atomic(checkpoint_directory / "step_metrics.jsonl", [{"event_id": "D1:2"}])

    with pytest.raises(SystemExit, match="event ids"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_same_length_wrong_question_prefix(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[{"question_id": "q1", "retrieved_text": "adoption agencies"}],
        metric_rows=[{"question_id": "q1", "proxy_answer_string_hit": True}],
    )
    checkpoint_directory = current_checkpoint_snapshot_dir(output_dir)
    write_jsonl_atomic(
        checkpoint_directory / "result_rows.jsonl",
        [{"question_id": "q2", "retrieved_text": "Melanie"}],
    )

    with pytest.raises(SystemExit, match="question ids"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_result_rows_that_do_not_match_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[{"question_id": "q1", "retrieved_text": "adoption agencies"}],
        metric_rows=[{"question_id": "q1", "proxy_answer_string_hit": True}],
    )
    checkpoint_directory = current_checkpoint_snapshot_dir(output_dir)
    write_jsonl_atomic(checkpoint_directory / "result_rows.jsonl", [])

    with pytest.raises(SystemExit, match="Checkpoint result rows do not match manifest"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_metric_rows_that_do_not_match_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[{"question_id": "q1", "retrieved_text": "adoption agencies"}],
        metric_rows=[{"question_id": "q1", "proxy_answer_string_hit": True}],
    )
    checkpoint_directory = current_checkpoint_snapshot_dir(output_dir)
    write_jsonl_atomic(checkpoint_directory / "metric_rows.jsonl", [])

    with pytest.raises(SystemExit, match="Checkpoint metric rows do not match result rows"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_same_length_wrong_metric_prefix(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[{"question_id": "q1", "retrieved_text": "adoption agencies"}],
        metric_rows=[{"question_id": "q1", "proxy_answer_string_hit": True}],
    )
    checkpoint_directory = current_checkpoint_snapshot_dir(output_dir)
    write_jsonl_atomic(
        checkpoint_directory / "metric_rows.jsonl",
        [{"question_id": "q2", "proxy_answer_string_hit": False}],
    )

    with pytest.raises(SystemExit, match="metric row question ids"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_trace_exclusion_keeps_failed_attempt_out_of_final_anomalies(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trace_id": "success-before-failure",
                        "event_type": "operator_result",
                        "operator": "sem_map",
                    }
                ),
                json.dumps(
                    {
                        "trace_id": "failed-attempt",
                        "event_type": "llm_batch_error",
                        "operator": "sem_topk",
                        "error_type": "InternalServerError",
                        "error_message": "SSL EOF",
                    }
                ),
                json.dumps(
                    {
                        "trace_id": "success-after-resume",
                        "event_type": "llm_call",
                        "operator": "sem_topk",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    anomalies = build_llm_anomaly_rows(
        trace_dir=trace_dir,
        excluded_event_ranges=[(1, 2)],
    )

    assert [row["trace_id"] for row in anomalies] == ["success-after-resume"]
    assert anomalies[0]["issue"] == "missing_raw_output"


def test_trace_exclusion_range_preserves_checkpoint_boundary(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trace_id": "checkpoint-boundary",
                        "event_type": "llm_batch_error",
                        "operator": "sem_topk",
                        "error_type": "ShouldRemainVisible",
                        "error_message": "event 1 is inside the checkpoint boundary",
                    }
                ),
                json.dumps(
                    {
                        "trace_id": "failed-after-checkpoint",
                        "event_type": "llm_batch_error",
                        "operator": "sem_topk",
                        "error_type": "ShouldBeExcluded",
                        "error_message": "event 2 happened after the checkpoint",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    anomalies = build_llm_anomaly_rows(
        trace_dir=trace_dir,
        excluded_event_ranges=[(1, 2)],
    )

    assert [row["trace_id"] for row in anomalies] == ["checkpoint-boundary"]


def test_recovery_metadata_records_excluded_trace_ranges(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    path = write_recovery_metadata(
        output_dir=output_dir,
        excluded_trace_event_ranges=[(3, 5), (5, 7)],
        error=RuntimeError("transport failed"),
        trace_dir=None,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["excluded_trace_event_ranges"] == [
        {"start": 3, "end": 7, "count": 4}
    ]
    assert payload["excluded_trace_event_count"] == 4
    assert payload["last_error_type"] == "RuntimeError"


def test_failed_run_helpers_write_partial_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    trace_dir = output_dir / "trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "phase": "retrieval",
                "question_id": "q2",
                "operator": "sem_topk",
                "event_type": "llm_batch_error",
                "error_type": "InternalServerError",
                "error_message": "SSL EOF",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    questions = benchmark_questions()
    events = benchmark_events()

    written = write_run_artifacts(
        output_dir=output_dir,
        run_mode="answer",
        model="test-model",
        sample_index=0,
        events=events,
        questions=questions,
        memory=FakeBenchmarkMemory(),
        result_rows=[{"question_id": "q1", "retrieved_text": "adoption agencies"}],
        metric_rows=[{"question_id": "q1", "proxy_answer_string_hit": True}],
        step_metrics=[{"event_id": "D1:1"}],
        trace_dir=trace_dir,
        llm_anomaly_rows=[
            {
                "phase": "retrieval",
                "operator": "sem_topk",
                "event_type": "llm_batch_error",
                "trace_id": "t1",
                "question_id": "q2",
                "event_id": "",
                "issue": "llm_batch_error",
                "prompt_path": "",
                "raw_output_path": "",
                "raw_output_preview": "",
                "error_type": "InternalServerError",
                "error_message": "SSL EOF",
                "model": "",
            }
        ],
        ingested_event_ids=("D1:1",),
        include_summary=False,
    )
    failure_path = write_failure_metadata(
        output_dir=output_dir,
        error=RuntimeError("query failed"),
        events=events,
        questions=questions,
        step_metrics=[{"event_id": "D1:1"}, {"event_id": "D1:2"}],
        result_rows=[{"question_id": "q1"}],
        trace_dir=trace_dir,
    )

    assert (output_dir / "memory" / "topics.csv").exists()
    assert (output_dir / "retrieval" / "results.csv").exists()
    assert (output_dir / "metrics" / "questions.csv").exists()
    assert (output_dir / "diagnostics" / "cause_trace.csv").exists()
    assert (output_dir / "diagnostics" / "llm_anomalies.csv").exists()
    assert written["trace"] == trace_dir
    cause_trace = pd.read_csv(output_dir / "diagnostics" / "cause_trace.csv")
    q2_trace = cause_trace[cause_trace["question_id"] == "q2"].iloc[0]
    assert q2_trace["source_status"] == "not_ingested"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error_type"] == "RuntimeError"
    assert failure["failed_phase"] == "question"
    assert failure["failed_question_id"] == "q2"
    assert failure["completed_questions"] == 1
