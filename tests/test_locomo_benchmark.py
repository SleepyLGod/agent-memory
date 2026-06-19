"""Tests for the isolated LOCOMO benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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
        "metadata": {
            "benchmark": "locomo",
            "sample_id": "conv-test",
            "event_id": "D1:1",
            "speaker": "Caroline",
        },
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
    assert locomo_answer_score("Caroline went yesterday.", "anything", 5) == 0.0


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
    assert row["gold_answer_in_retrieved_text"] is False
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


def test_summarize_question_metrics_handles_no_eligible_questions() -> None:
    summary = summarize_question_metrics([])

    assert summary["questions_evaluated"] == 0
    assert summary["retrieval_gold_answer_hit_rate"] == ""


def test_summarize_question_metrics_reports_locomo_category_means() -> None:
    rows = [
        {
            "category": "2",
            "gold_answer_in_retrieved_text": True,
            "generated_answer": "",
            "answer_exact_match": False,
            "answer_contains_gold": False,
            "answer_f1": 0.0,
            "locomo_answer_score": 0.0,
        },
        {
            "category": "5",
            "gold_answer_in_retrieved_text": False,
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
