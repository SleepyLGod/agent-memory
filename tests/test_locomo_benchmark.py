"""Tests for the isolated LOCOMO benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import agent_memory.evaluation.claude_memory.locomo as claude_memory_locomo_module
from agent_memory.evaluation.claude_memory.bindings import event_to_claude_log_row
from agent_memory.evaluation.diagnostics import build_llm_anomaly_rows
from agent_memory.evaluation.types import BenchmarkEvent, BenchmarkQuestion
from agent_memory.evaluation.locomo import (
    eligible_questions,
    load_locomo_sample,
    normalize_locomo_sample,
    select_events,
)
from agent_memory.evaluation.metrics import (
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
from agent_memory.evaluation.claude_memory.locomo import (
    ANSWER_MAX_TOKENS,
    ANSWER_SYSTEM_PROMPT,
    BENCHMARK_CONTRACT,
    ClaudeMemoryLocomoRunConfig,
    POLICY_CONTRACT,
    checkpoint_contract_digest,
    checkpoint_manifest_path,
    checkpoint_snapshots_dir,
    csv_safe_frame,
    current_checkpoint_snapshot_dir,
    events_frame,
    load_checkpoint as _load_checkpoint,
    load_artifact_runtime_state,
    load_external_runtime_state,
    reset_output_dir,
    run_claude_memory_locomo,
    run_questions,
    save_checkpoint,
    summary_frame,
    validate_output_paths,
    write_csv,
    write_failure_metadata,
    write_recovery_metadata,
    write_run_artifacts,
    write_jsonl_atomic,
    write_state_jsonl,
)


def load_checkpoint(
    **kwargs: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a trusted local checkpoint in tests."""

    return _load_checkpoint(**kwargs, trusted_checkpoint=True)


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


def test_eligible_questions_zero_limit_selects_no_questions() -> None:
    sample = normalize_locomo_sample(locomo_fixture())

    selected = eligible_questions(
        sample.questions,
        ingested_event_ids=["D1:1", "D1:2"],
        question_limit=0,
    )

    assert selected == ()


def test_eligible_questions_rejects_negative_question_limit() -> None:
    sample = normalize_locomo_sample(locomo_fixture())

    with pytest.raises(ValueError, match="question_limit must be non-negative"):
        eligible_questions(
            sample.questions,
            ingested_event_ids=["D1:1", "D1:2"],
            question_limit=-1,
        )


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
    assert exact_match("agencies adoption", "adoption agencies")
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


def test_question_metric_row_skips_locomo_score_for_invalid_category() -> None:
    row = question_metric_row(
        question_id="q1",
        question="When did Caroline go?",
        gold_answer="7 May 2023",
        retrieved_frame=pd.DataFrame(),
        generated_answer="8 May 2023.",
        category="",
    )

    assert row["category"] == ""
    assert row["answer_f1"] == round(2 / 3, 6)
    assert "locomo_answer_score" not in row

    dirty = question_metric_row(
        question_id="q2",
        question="When did Caroline go?",
        gold_answer="7 May 2023",
        retrieved_frame=pd.DataFrame(),
        generated_answer="8 May 2023.",
        category="not-a-category",
    )
    assert dirty["category"] == "not-a-category"
    assert "locomo_answer_score" not in dirty


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
        maintenance_mode="ingest",
        source_run_dir="",
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


def test_summary_frame_records_sem_topk_method() -> None:
    frame = summary_frame(
        run_mode="answer",
        model="test-model",
        grouped_agg_rule="join-map",
        sem_topk_method="listwise",
        sample_index=0,
        maintenance_mode="external-state",
        source_run_dir="/tmp/source",
        events=(),
        questions=(),
        memory=FakeBenchmarkMemory(),
        question_metrics=(),
        question_results=(),
        step_metrics=(),
    )

    row = frame.iloc[0]
    assert row["grouped_agg_rule"] == "join-map"
    assert row["sem_topk_method"] == "listwise"


class FakeBenchmarkMemory:
    """Minimal memory object for runner helper tests."""

    def __init__(self) -> None:
        self._runtime = FakeBenchmarkRuntime()
        self.calls = 0

    def query(self, question: str) -> pd.DataFrame:
        """Return one retrieval result, then fail on the next query."""

        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("query failed")
        return pd.DataFrame(
            [{"name": "adoption", "body": "Caroline researched adoption agencies."}]
        )


class FakeBenchmarkRuntime:
    """Minimal runtime snapshot surface for checkpoint helper tests."""

    def __init__(self) -> None:
        self._state = {
            "log": pd.DataFrame([{"message": "I researched adoption agencies."}]),
            "topics": pd.DataFrame(
                [{"name": "adoption", "body": "Caroline researched adoption agencies."}]
            ),
            "catalog": pd.DataFrame(
                [{"catalog_title": "Adoption", "name": "adoption", "hook": "adoption"}]
            ),
        }
        self._window_next_start = {"_blocks_process_window": 2}
        self._upstream_log_count = {"_blocks_upstream": 3}

    def snapshot_state(self) -> dict[str, Any]:
        """Return the same shape as MemoryRuntime.snapshot_state()."""

        return {
            "schema_version": 1,
            "state": dict(self._state),
            "window_next_start": dict(self._window_next_start),
            "upstream_log_count": dict(self._upstream_log_count),
        }


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


def run_config(tmp_path: Path, **overrides: Any) -> ClaudeMemoryLocomoRunConfig:
    """Return a minimal runner config for argument-validation tests."""

    values: dict[str, Any] = {
        "sample_index": 0,
        "row_limit": 0,
        "question_limit": 0,
        "model": "test-model",
        "output_dir": tmp_path / "output",
        "locomo_cache_path": tmp_path / "locomo.json",
    }
    values.update(overrides)
    return ClaudeMemoryLocomoRunConfig(**values)


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
    snapshot, loaded_steps, loaded_results, loaded_metrics = load_checkpoint(
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

    pd.testing.assert_frame_equal(
        snapshot["state"]["topics"],
        memory._runtime._state["topics"],
    )
    assert snapshot["window_next_start"] == memory._runtime._window_next_start
    assert snapshot["upstream_log_count"] == memory._runtime._upstream_log_count
    assert loaded_steps == step_metrics
    assert loaded_results == result_rows
    assert loaded_metrics == metric_rows


def test_load_checkpoint_requires_trusted_checkpoint_gate(tmp_path: Path) -> None:
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
        trace_enabled=False,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="trust-existing-output-dir"):
        _load_checkpoint(
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


def test_checkpoint_contract_digest_includes_answer_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    original = checkpoint_contract_digest()

    monkeypatch.setattr(
        claude_memory_locomo_module,
        "ANSWER_MAX_TOKENS",
        ANSWER_MAX_TOKENS + 1,
    )

    assert checkpoint_contract_digest() != original


def test_validate_output_paths_rejects_unsafe_destructive_targets(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    output_run = tmp_path / "output"

    validate_output_paths(output_dir=output_run.resolve(), source_run_dir=source_run.resolve())

    with pytest.raises(SystemExit, match="broad path"):
        validate_output_paths(output_dir=Path.cwd().resolve(), source_run_dir=None)
    with pytest.raises(SystemExit, match="nested"):
        validate_output_paths(
            output_dir=tmp_path.resolve(),
            source_run_dir=(tmp_path / "source").resolve(),
        )
    with pytest.raises(SystemExit, match="nested"):
        validate_output_paths(
            output_dir=(tmp_path / "child").resolve(),
            source_run_dir=tmp_path.resolve(),
        )


def test_reset_output_dir_requires_run_owned_marker_for_non_empty_dirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(SystemExit, match="without .agent-memory-locomo-run.json"):
        reset_output_dir(output_dir, safe=True)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    reset_output_dir(empty_dir, safe=True)
    assert (empty_dir / ".agent-memory-locomo-run.json").exists()

    (empty_dir / "artifact.txt").write_text("owned", encoding="utf-8")
    reset_output_dir(empty_dir, safe=True)
    assert (empty_dir / ".agent-memory-locomo-run.json").exists()
    assert not (empty_dir / "artifact.txt").exists()


def test_runner_rejects_restore_artifact_state_invalid_flag_combinations(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --existing-output-dir"):
        run_claude_memory_locomo(run_config(tmp_path, restore_artifact_csv_state=True))

    with pytest.raises(SystemExit, match="restore-csv-state is no longer supported"):
        run_claude_memory_locomo(
            run_config(
                tmp_path,
                existing_output_dir=tmp_path / "source",
                trust_existing_output_dir=True,
                restore_csv_state=True,
            )
        )

    with pytest.raises(SystemExit, match="trust-existing-output-dir"):
        run_claude_memory_locomo(
            run_config(
                tmp_path,
                existing_output_dir=tmp_path / "source",
                restore_artifact_csv_state=True,
            )
        )

    with pytest.raises(SystemExit, match="cannot be used with --resume"):
        run_claude_memory_locomo(
            run_config(
                tmp_path,
                existing_output_dir=tmp_path / "source",
                trust_existing_output_dir=True,
                restore_artifact_csv_state=True,
                resume=True,
            )
        )


def test_runner_rejects_invalid_continuation_flag_combinations(tmp_path: Path) -> None:
    source = tmp_path / "source"

    with pytest.raises(SystemExit, match="mutually exclusive"):
        run_claude_memory_locomo(
            run_config(
                tmp_path,
                existing_output_dir=source,
                continue_from_output_dir=source,
                source_checkpoint_id="checkpoint-1",
                trust_existing_output_dir=True,
            )
        )

    with pytest.raises(SystemExit, match="requires --continue-from-output-dir"):
        run_claude_memory_locomo(
            run_config(tmp_path, source_checkpoint_id="checkpoint-1")
        )

    with pytest.raises(SystemExit, match="requires --source-checkpoint-id"):
        run_claude_memory_locomo(
            run_config(
                tmp_path,
                continue_from_output_dir=source,
                trust_existing_output_dir=True,
            )
        )

    with pytest.raises(SystemExit, match="omit external source arguments"):
        run_claude_memory_locomo(
            run_config(
                tmp_path,
                continue_from_output_dir=source,
                source_checkpoint_id="checkpoint-1",
                trust_existing_output_dir=True,
                resume=True,
            )
        )


def test_write_csv_escapes_spreadsheet_formula_prefixes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "formula": "=SUM(1,2)",
                "plus": "+x",
                "minus": "-x",
                "at": "@x",
                "leading_space_formula": " =SUM(1,2)",
                "leading_newline_formula": "\n=SUM(1,2)",
                "leading_tab_formula": "\t@cmd",
                "leading_mixed_formula": " \r\n+cmd",
                "leading_space_text": " plain",
                "normal": "plain",
                "number": 3,
                "boolean": True,
            }
        ]
    )

    path = write_csv("safe", frame, tmp_path)
    restored = pd.read_csv(path, keep_default_na=False)

    assert restored.loc[0, "formula"] == "'=SUM(1,2)"
    assert restored.loc[0, "plus"] == "'+x"
    assert restored.loc[0, "minus"] == "'-x"
    assert restored.loc[0, "at"] == "'@x"
    assert restored.loc[0, "leading_space_formula"] == "' =SUM(1,2)"
    assert restored.loc[0, "leading_newline_formula"] == "'\n=SUM(1,2)"
    assert restored.loc[0, "leading_tab_formula"] == "'\t@cmd"
    assert restored.loc[0, "leading_mixed_formula"] == "' \r\n+cmd"
    assert restored.loc[0, "leading_space_text"] == " plain"
    assert restored.loc[0, "normal"] == "plain"
    assert restored.loc[0, "number"] == 3
    assert bool(restored.loc[0, "boolean"]) is True

    safe = csv_safe_frame(frame)
    assert frame.loc[0, "formula"] == "=SUM(1,2)"
    assert safe.loc[0, "formula"] == "'=SUM(1,2)"

    state_path = write_state_jsonl("raw", frame, tmp_path / "state")
    raw_rows = [
        json.loads(line)
        for line in state_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert raw_rows[0]["formula"] == "=SUM(1,2)"


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


def test_grouped_agg_rule_choices_match_the_planner() -> None:
    from agent_memory.planner.rules import GROUPED_AGG_RULES

    assert claude_memory_locomo_module.GROUPED_AGG_RULES == GROUPED_AGG_RULES


def test_sem_topk_method_choices_use_canonical_names() -> None:
    assert claude_memory_locomo_module.SEM_TOPK_METHODS == (
        "pairwise-naive",
        "pairwise-quick",
        "pairwise-heap",
        "listwise",
    )


def test_load_checkpoint_rejects_cross_sem_topk_method_resume(
    tmp_path: Path,
) -> None:
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
        sem_topk_method="pairwise-naive",
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="sem_topk_method"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            sem_topk_method="listwise",
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_legacy_checkpoint_without_sem_topk_method_means_pairwise_naive(
    tmp_path: Path,
) -> None:
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
        sem_topk_method="pairwise-naive",
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )
    manifest_path = checkpoint_manifest_path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("sem_topk_method")
    manifest.pop("sem_topk_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_checkpoint(
        output_dir=output_dir,
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        sem_topk_method="pairwise-naive",
        trace_enabled=False,
        events=events,
        questions=questions,
    )
    with pytest.raises(SystemExit, match="sem_topk_method"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            sem_topk_method="listwise",
            trace_enabled=False,
            events=events,
            questions=questions,
        )


@pytest.mark.parametrize("grouped_agg_rule", ["compressed", "changed-aware"])
def test_load_checkpoint_accepts_existing_grouped_agg_rules(
    tmp_path: Path,
    grouped_agg_rule: str,
) -> None:
    output_dir = tmp_path / grouped_agg_rule
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
        grouped_agg_rule=grouped_agg_rule,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    snapshot, loaded_steps, loaded_results, loaded_metrics = load_checkpoint(
        output_dir=output_dir,
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        grouped_agg_rule=grouped_agg_rule,
        trace_enabled=False,
        events=events,
        questions=questions,
    )

    assert set(snapshot) >= {"state", "window_next_start", "upstream_log_count"}
    assert loaded_steps == []
    assert loaded_results == []
    assert loaded_metrics == []


@pytest.mark.parametrize("stored_rule", ["compressed", "changed-aware"])
def test_load_checkpoint_rejects_cross_grouped_agg_rule_resume(
    tmp_path: Path,
    stored_rule: str,
) -> None:
    output_dir = tmp_path / stored_rule
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
        grouped_agg_rule=stored_rule,
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="grouped_agg_rule"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            grouped_agg_rule="join-map",
            trace_enabled=False,
            events=events,
            questions=questions,
        )


def test_load_checkpoint_rejects_maintenance_mode_mismatch(tmp_path: Path) -> None:
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
        maintenance_mode="external-state",
        source_run_dir="/tmp/source-run",
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="maintenance_mode, source_run_dir"):
        load_checkpoint(
            output_dir=output_dir,
            sample_index=0,
            row_limit=2,
            question_limit=2,
            model="test-model",
            answer=False,
            maintenance_mode="ingest",
            source_run_dir="",
            trace_enabled=False,
            events=events,
            questions=questions,
        )


@pytest.mark.parametrize("runtime_schema_version", [1, 2])
def test_external_state_resume_uses_target_checkpoint_origin_and_skips_ingest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_schema_version: int,
) -> None:
    output_dir = tmp_path / f"resume-v{runtime_schema_version}"
    missing_source_dir = (tmp_path / "source-no-longer-available").resolve()
    events = benchmark_events()
    questions = benchmark_questions()[:1]

    class ResumeRuntime(FakeBenchmarkRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.restored_snapshot: dict[str, Any] | None = None

        def snapshot_state(self) -> dict[str, Any]:
            return {
                "schema_version": runtime_schema_version,
                "test_marker": f"runtime-v{runtime_schema_version}",
            }

        def restore_state(self, snapshot: dict[str, Any]) -> None:
            self.restored_snapshot = snapshot

    class ResumeMemory(FakeBenchmarkMemory):
        def __init__(self) -> None:
            super().__init__()
            self._runtime = ResumeRuntime()

    save_checkpoint(
        output_dir=output_dir,
        memory=ResumeMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=1,
        model="test-model",
        answer=False,
        maintenance_mode="external-state",
        source_run_dir=str(missing_source_dir),
        events=events,
        questions=questions,
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )

    resumed_memory = ResumeMemory()
    monkeypatch.setattr(
        claude_memory_locomo_module,
        "selected_benchmark_data",
        lambda **_kwargs: (tmp_path / "locomo.json", events, questions),
    )
    monkeypatch.setattr(
        claude_memory_locomo_module,
        "create_memory",
        lambda **_kwargs: resumed_memory,
    )

    def reject_ingest(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("external-state resume must not ingest source events again")

    monkeypatch.setattr(
        claude_memory_locomo_module,
        "run_memory_ingest",
        reject_ingest,
    )

    def finish_question(
        _memory: Any,
        selected_questions: Any,
        *,
        answer: bool,
        result_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
        start_index: int,
        checkpoint_callback: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert answer is False
        assert start_index == 0
        question = selected_questions[0]
        result_rows.append({"question_id": question.question_id})
        metric_rows.append({"question_id": question.question_id})
        checkpoint_callback()
        return result_rows, metric_rows

    monkeypatch.setattr(
        claude_memory_locomo_module,
        "run_questions",
        finish_question,
    )
    monkeypatch.setattr(
        claude_memory_locomo_module,
        "write_run_artifacts",
        lambda **_kwargs: {},
    )

    run_claude_memory_locomo(
        run_config(
            tmp_path,
            output_dir=output_dir,
            row_limit=2,
            question_limit=1,
            resume=True,
            trust_existing_output_dir=True,
        )
    )

    assert resumed_memory._runtime.restored_snapshot == {
        "schema_version": runtime_schema_version,
        "test_marker": f"runtime-v{runtime_schema_version}",
    }
    manifest = json.loads(checkpoint_manifest_path(output_dir).read_text(encoding="utf-8"))
    assert manifest["maintenance_mode"] == "external-state"
    assert manifest["source_run_dir"] == str(missing_source_dir)
    assert manifest["completed_events"] == 0
    assert manifest["completed_questions"] == 1


def test_checkpoint_run_origin_defaults_old_manifest_to_ingest(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    save_checkpoint(
        output_dir=output_dir,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=0,
        question_limit=0,
        model="test-model",
        answer=False,
        events=(),
        questions=(),
        step_metrics=[],
        result_rows=[],
        metric_rows=[],
    )
    manifest_path = checkpoint_manifest_path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("maintenance_mode")
    manifest.pop("source_run_dir")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert claude_memory_locomo_module.checkpoint_run_origin(output_dir) == (
        "ingest",
        "",
    )


def test_load_external_runtime_state_requires_source_event_coverage(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=source_run,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=1,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events[:1],
        questions=questions,
        step_metrics=[{"phase": "add", "event_id": "D1:1"}],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="event ids do not cover selected events"):
        load_external_runtime_state(source_run, events=events)


def test_load_continuation_checkpoint_reads_a_specific_snapshot_without_mutating_source(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    memory = FakeBenchmarkMemory()
    save_checkpoint(
        output_dir=source_run,
        memory=memory,
        sample_index=0,
        row_limit=2,
        question_limit=0,
        model="test-model",
        answer=False,
        grouped_agg_rule="join-map",
        sem_topk_method="pairwise-naive",
        events=events,
        questions=(),
        step_metrics=[{"phase": "add", "event_id": "D1:1"}],
        result_rows=[],
        metric_rows=[],
    )
    checkpoint_id = json.loads(
        (source_run / "checkpoint" / "current.json").read_text(encoding="utf-8")
    )["checkpoint_id"]
    before = {
        path.relative_to(source_run): path.read_bytes()
        for path in source_run.rglob("*")
        if path.is_file()
    }

    snapshot, step_metrics = claude_memory_locomo_module.load_continuation_checkpoint(
        source_run,
        checkpoint_id=checkpoint_id,
        sample_index=0,
        model="test-model",
        answer=False,
        grouped_agg_rule="join-map",
        sem_topk_method="pairwise-naive",
        events=events,
        trusted_checkpoint=True,
    )

    assert snapshot["schema_version"] == 1
    assert [row["event_id"] for row in step_metrics] == ["D1:1"]
    assert before == {
        path.relative_to(source_run): path.read_bytes()
        for path in source_run.rglob("*")
        if path.is_file()
    }


def test_load_continuation_checkpoint_accepts_legacy_manifest_without_rule(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    save_checkpoint(
        output_dir=source_run,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=0,
        model="test-model",
        answer=False,
        grouped_agg_rule="join-map",
        events=events,
        questions=(),
        step_metrics=[{"phase": "add", "event_id": "D1:1"}],
        result_rows=[],
        metric_rows=[],
    )
    checkpoint_id = json.loads(
        (source_run / "checkpoint" / "current.json").read_text(encoding="utf-8")
    )["checkpoint_id"]
    manifest_path = (
        source_run / "checkpoint" / "snapshots" / checkpoint_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("grouped_agg_rule")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    snapshot, _ = claude_memory_locomo_module.load_continuation_checkpoint(
        source_run,
        checkpoint_id=checkpoint_id,
        sample_index=0,
        model="test-model",
        answer=False,
        grouped_agg_rule="join-map",
        sem_topk_method="pairwise-naive",
        events=events,
        trusted_checkpoint=True,
    )

    assert snapshot["schema_version"] == 1


def test_load_continuation_checkpoint_rejects_changed_completed_event(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    save_checkpoint(
        output_dir=source_run,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=0,
        model="test-model",
        answer=False,
        grouped_agg_rule="join-map",
        events=events,
        questions=(),
        step_metrics=[{"phase": "add", "event_id": "D1:1"}],
        result_rows=[],
        metric_rows=[],
    )
    checkpoint_id = json.loads(
        (source_run / "checkpoint" / "current.json").read_text(encoding="utf-8")
    )["checkpoint_id"]
    changed_events = (
        BenchmarkEvent(
            sample_id=events[0].sample_id,
            event_id=events[0].event_id,
            speaker=events[0].speaker,
            text="Changed completed event.",
            session_id=events[0].session_id,
            timestamp=events[0].timestamp,
        ),
        events[1],
    )

    with pytest.raises(SystemExit, match="completed event content"):
        claude_memory_locomo_module.load_continuation_checkpoint(
            source_run,
            checkpoint_id=checkpoint_id,
            sample_index=0,
            model="test-model",
            answer=False,
            grouped_agg_rule="join-map",
            sem_topk_method="pairwise-naive",
            events=changed_events,
            trusted_checkpoint=True,
        )


def write_artifact_state_source(
    source_run: Path,
    *,
    events: tuple[BenchmarkEvent, ...],
    log: pd.DataFrame | None = None,
    topics: pd.DataFrame | None = None,
    catalog: pd.DataFrame | None = None,
) -> None:
    """Write a minimal raw artifact-state source run for restore tests."""

    write_csv("events", events_frame(events), source_run / "input")
    write_state_jsonl(
        "events",
        pd.DataFrame(
            [
                {
                    "sample_id": event.sample_id,
                    "event_id": event.event_id,
                    "speaker": event.speaker,
                    "text": event.text,
                    "session_id": event.session_id,
                    "timestamp": event.timestamp,
                }
                for event in events
            ]
        ),
        source_run / "state" / "input",
    )
    memory_dir = source_run / "memory"
    state_memory_dir = source_run / "state" / "memory"
    log_frame = (
        log
        if log is not None
        else pd.DataFrame([{"message": "I researched adoption agencies."}])
    )
    write_csv("log", log_frame, memory_dir)
    write_state_jsonl("log", log_frame, state_memory_dir)
    topics_frame = (
        topics
        if topics is not None
        else pd.DataFrame(
            [
                {
                    "name": "adoption",
                    "description": "Adoption research.",
                    "type": "user",
                    "body": "Caroline researched adoption agencies.",
                }
            ]
        )
    )
    write_csv(
        "topics",
        topics_frame,
        memory_dir,
    )
    write_state_jsonl("topics", topics_frame, state_memory_dir)
    catalog_frame = (
        catalog
        if catalog is not None
        else pd.DataFrame(
            [
                {
                    "catalog_title": "Adoption",
                    "name": "adoption",
                    "hook": "Caroline researched adoption agencies.",
                }
            ]
        )
    )
    write_csv(
        "catalog",
        catalog_frame,
        memory_dir,
    )
    write_state_jsonl("catalog", catalog_frame, state_memory_dir)


def test_load_artifact_runtime_state_restores_public_memory_tables(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    write_artifact_state_source(source_run, events=events)

    state = load_artifact_runtime_state(
        source_run,
        events=events,
        trusted_checkpoint=True,
    )

    assert set(state) == {"log", "topics", "catalog"}
    assert list(state["catalog"].columns) == ["catalog_title", "name", "hook"]
    assert state["catalog"].loc[0, "name"] == "adoption"


def test_load_artifact_runtime_state_restores_empty_public_memory_tables(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    write_artifact_state_source(
        source_run,
        events=events,
        topics=pd.DataFrame(columns=["name", "description", "type", "body"]),
        catalog=pd.DataFrame(columns=["catalog_title", "name", "hook"]),
    )

    state = load_artifact_runtime_state(
        source_run,
        events=events,
        trusted_checkpoint=True,
    )

    assert state["topics"].empty
    assert list(state["topics"].columns) == ["name", "description", "type", "body"]
    assert state["catalog"].empty
    assert list(state["catalog"].columns) == ["catalog_title", "name", "hook"]


def test_load_artifact_runtime_state_restores_empty_event_and_log_schemas(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source"
    write_artifact_state_source(
        source_run,
        events=(),
        log=pd.DataFrame(columns=["message", "role", "timestamp", "session_id"]),
        topics=pd.DataFrame(columns=["name", "description", "type", "body"]),
        catalog=pd.DataFrame(columns=["catalog_title", "name", "hook"]),
    )

    state = load_artifact_runtime_state(
        source_run,
        events=(),
        trusted_checkpoint=True,
    )

    assert state["log"].empty
    assert list(state["log"].columns) == ["message", "role", "timestamp", "session_id"]
    assert state["topics"].empty
    assert list(state["topics"].columns) == ["name", "description", "type", "body"]
    assert state["catalog"].empty
    assert list(state["catalog"].columns) == ["catalog_title", "name", "hook"]


def test_load_artifact_runtime_state_rejects_empty_source_events_for_nonempty_selection(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source"
    write_artifact_state_source(
        source_run,
        events=(),
        topics=pd.DataFrame(columns=["name", "description", "type", "body"]),
        catalog=pd.DataFrame(columns=["catalog_title", "name", "hook"]),
    )

    with pytest.raises(SystemExit, match="do not cover selected events"):
        load_artifact_runtime_state(
            source_run,
            events=benchmark_events(),
            trusted_checkpoint=True,
        )


def test_load_artifact_runtime_state_requires_trusted_checkpoint_gate(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    write_artifact_state_source(source_run, events=events)

    with pytest.raises(SystemExit, match="trust-existing-output-dir"):
        load_artifact_runtime_state(source_run, events=events)


def test_load_artifact_runtime_state_rejects_changed_event_content(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    write_artifact_state_source(source_run, events=events)
    changed_events = (
        BenchmarkEvent(
            sample_id=events[0].sample_id,
            event_id=events[0].event_id,
            speaker=events[0].speaker,
            text="Same id, different content.",
            session_id=events[0].session_id,
            timestamp=events[0].timestamp,
        ),
        events[1],
    )

    with pytest.raises(SystemExit, match="event content"):
        load_artifact_runtime_state(
            source_run,
            events=changed_events,
            trusted_checkpoint=True,
        )


def test_load_artifact_runtime_state_requires_catalog_and_retrieval_columns(tmp_path: Path) -> None:
    source_run = tmp_path / "missing-catalog"
    events = benchmark_events()
    write_state_jsonl(
        "events",
        pd.DataFrame(
            [
                {
                    "sample_id": event.sample_id,
                    "event_id": event.event_id,
                    "speaker": event.speaker,
                    "text": event.text,
                    "session_id": event.session_id,
                    "timestamp": event.timestamp,
                }
                for event in events
            ]
        ),
        source_run / "state" / "input",
    )
    write_state_jsonl(
        "topics",
        pd.DataFrame(
            [
                {
                    "name": "adoption",
                    "description": "Adoption research.",
                    "type": "user",
                    "body": "Caroline researched adoption agencies.",
                }
            ]
        ),
        source_run / "state" / "memory",
    )

    with pytest.raises(SystemExit, match="catalog.jsonl"):
        load_artifact_runtime_state(
            source_run,
            events=events,
            trusted_checkpoint=True,
        )

    bad_catalog_run = tmp_path / "bad-catalog"
    write_artifact_state_source(
        bad_catalog_run,
        events=events,
        catalog=pd.DataFrame([{"name": "adoption", "hook": "missing title"}]),
    )
    with pytest.raises(SystemExit, match="missing required columns"):
        load_artifact_runtime_state(
            bad_catalog_run,
            events=events,
            trusted_checkpoint=True,
        )


def test_load_external_runtime_state_rejects_same_id_changed_event_content(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=source_run,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[
            {"phase": "add", "event_id": "D1:1"},
            {"phase": "add", "event_id": "D1:2"},
        ],
        result_rows=[],
        metric_rows=[],
    )
    changed_events = (
        BenchmarkEvent(
            sample_id=events[0].sample_id,
            event_id=events[0].event_id,
            speaker=events[0].speaker,
            text="Same id, different content.",
            session_id=events[0].session_id,
            timestamp=events[0].timestamp,
        ),
        events[1],
    )

    with pytest.raises(SystemExit, match="event content"):
        load_external_runtime_state(
            source_run,
            events=changed_events,
            trusted_checkpoint=True,
        )


def test_load_external_runtime_state_loads_valid_prefix_source(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    questions = benchmark_questions()
    memory = FakeBenchmarkMemory()
    save_checkpoint(
        output_dir=source_run,
        memory=memory,
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[
            {"phase": "add", "event_id": "D1:1"},
            {"phase": "add", "event_id": "D1:2"},
        ],
        result_rows=[],
        metric_rows=[],
    )

    snapshot = load_external_runtime_state(
        source_run,
        events=events,
        trusted_checkpoint=True,
    )

    pd.testing.assert_frame_equal(
        snapshot["state"]["topics"],
        memory._runtime._state["topics"],
    )
    assert snapshot["window_next_start"] == memory._runtime._window_next_start
    assert snapshot["upstream_log_count"] == memory._runtime._upstream_log_count


def test_load_external_runtime_state_rejects_extra_completed_events(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=source_run,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[
            {"phase": "add", "event_id": "D1:1"},
            {"phase": "add", "event_id": "D1:2"},
        ],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="completed event boundary"):
        load_external_runtime_state(
            source_run,
            events=events[:1],
            trusted_checkpoint=True,
        )


def test_load_external_runtime_state_requires_trusted_checkpoint_gate(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    events = benchmark_events()
    questions = benchmark_questions()
    save_checkpoint(
        output_dir=source_run,
        memory=FakeBenchmarkMemory(),
        sample_index=0,
        row_limit=2,
        question_limit=2,
        model="test-model",
        answer=False,
        events=events,
        questions=questions,
        step_metrics=[
            {"phase": "add", "event_id": "D1:1"},
            {"phase": "add", "event_id": "D1:2"},
        ],
        result_rows=[],
        metric_rows=[],
    )

    with pytest.raises(SystemExit, match="trust-existing-output-dir"):
        load_external_runtime_state(source_run, events=events)


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
        sem_topk_method="listwise",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["excluded_trace_event_ranges"] == [
        {"start": 3, "end": 7, "count": 4}
    ]
    assert payload["excluded_trace_event_count"] == 4
    assert payload["last_error_type"] == "RuntimeError"
    assert payload["sem_topk_method"] == "listwise"
    assert payload["sem_topk_contract"] == "listwise:v1"


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
        maintenance_mode="ingest",
        source_run_dir="",
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
        sem_topk_method="listwise",
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
    assert failure["sem_topk_method"] == "listwise"
    assert failure["sem_topk_contract"] == "listwise:v1"
