from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from agent_memory.evaluation.artifacts import BenchmarkArtifactStore
from agent_memory.evaluation.longmemeval import (
    LONGMEMEVAL_CLAUDE_PILOT_30_IDS,
    LONGMEMEVAL_SMOKE_CASE_ID,
    LONGMEMEVAL_SMOKE_EVENT_COUNT,
    LONGMEMEVAL_SMOKE_SESSION_ID,
    build_answer_prompt,
    build_judge_prompt,
    hypothesis_record,
    longmemeval_smoke_bundle,
    longmemeval_task_contract,
    normalize_longmemeval,
    parse_judge_response,
    write_official_hypotheses,
)
from agent_memory.evaluation.longmemeval.dataset import (
    LONGMEMEVAL_CLEANED_REVISION,
    LONGMEMEVAL_CLEANED_SHA256,
    parse_longmemeval_timestamp,
)


def _record(
    *,
    question_id: str = "q-1",
    question_type: str = "single-session-user",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "What did I buy?",
        "answer": "A bicycle",
        "question_date": "January 4, 2026 9:00 AM",
        "haystack_session_ids": ["later", "earlier"],
        "haystack_dates": [
            "January 3, 2026 9:00 AM",
            "2026/01/02 (Fri) 08:30",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I bought a bicycle.", "has_answer": True},
                {"role": "assistant", "content": "Enjoy it!"},
            ],
            [{"role": "user", "content": "I went shopping."}],
        ],
        "answer_session_ids": ["later"],
    }


def test_pin_is_exact_and_not_main() -> None:
    assert len(LONGMEMEVAL_CLEANED_REVISION) == 40
    assert len(LONGMEMEVAL_CLEANED_SHA256) == 64


def test_official_hypotheses_export_completed_answers(tmp_path: Path) -> None:
    bundle = normalize_longmemeval([_record()])
    case = bundle.cases[0]
    case_dir = BenchmarkArtifactStore(tmp_path).case_dir(case.case_id)
    case_dir.mkdir(parents=True)
    (case_dir / "answers.jsonl").write_text(
        '{"question_id":"q-1","answer":"A bicycle"}\n',
        encoding="utf-8",
    )

    path = write_official_hypotheses(bundle, tmp_path)

    assert path.read_text(encoding="utf-8") == (
        '{"hypothesis": "A bicycle", "question_id": "q-1"}\n'
    )


def test_official_hypotheses_reject_incomplete_run(tmp_path: Path) -> None:
    bundle = normalize_longmemeval([_record()])

    with pytest.raises(ValueError, match="has no answers"):
        write_official_hypotheses(bundle, tmp_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026/01/02 (Fri) 08:30", datetime(2026, 1, 2, 8, 30)),
        ("January 4, 2026 9:00 AM", datetime(2026, 1, 4, 9, 0)),
        ("January 4, 2026 9:00 PM", datetime(2026, 1, 4, 21, 0)),
    ],
)
def test_timestamp_parser_is_locale_independent(value: str, expected: datetime) -> None:
    assert parse_longmemeval_timestamp(value) == expected


def test_normalization_sorts_sessions_and_does_not_leak_gold_labels() -> None:
    bundle = normalize_longmemeval([_record()])
    case = bundle.cases[0]

    assert [event.session_id for event in case.events] == ["earlier", "later", "later"]
    assert [event.speaker for event in case.events] == ["user", "user", "assistant"]
    assert all("has_answer" not in event.metadata for event in case.events)
    assert all("answer" not in event.metadata for event in case.events)
    assert case.questions[0].evidence_event_ids == ("q-1:s0:t0",)
    assert "A bicycle" not in str(bundle._payload(include_labels=False))


def test_question_date_is_explicit_in_retrieval_answer_prompt() -> None:
    question = normalize_longmemeval([_record()]).cases[0].questions[0]

    prompt = build_answer_prompt(question, "retrieved context")

    assert "Current Date: January 4, 2026 9:00 AM" in prompt
    assert "Question: What did I buy?" in prompt
    assert "retrieved context" in prompt


@pytest.mark.parametrize(
    ("question_type", "needle"),
    [
        ("single-session-user", "subset of the information"),
        ("single-session-assistant", "subset of the information"),
        ("multi-session", "subset of the information"),
        ("temporal-reasoning", "off-by-one errors"),
        ("knowledge-update", "previous information"),
        ("single-session-preference", "Rubric:"),
    ],
)
def test_official_question_type_prompts(question_type: str, needle: str) -> None:
    question = normalize_longmemeval(
        [_record(question_type=question_type)]
    ).cases[0].questions[0]

    assert needle in build_judge_prompt(question, "model answer")


def test_abstention_prompt_and_official_yes_logic() -> None:
    question = normalize_longmemeval(
        [_record(question_id="q-1_abs")]
    ).cases[0].questions[0]

    assert "unanswerable question" in build_judge_prompt(question, "I do not know")
    assert parse_judge_response("YES") is True
    assert parse_judge_response("Yesterday") is True
    assert parse_judge_response("no") is False


def test_official_hypothesis_shape() -> None:
    question = normalize_longmemeval([_record()]).cases[0].questions[0]
    assert hypothesis_record(question, "A bicycle") == {
        "question_id": "q-1",
        "hypothesis": "A bicycle",
    }


def test_runner_contract_includes_date_and_custom_judge_label() -> None:
    question = normalize_longmemeval([_record()]).cases[0].questions[0]
    contract = longmemeval_task_contract(
        judge_model_id="deepseek/deepseek-v4-flash"
    )

    assert contract.task_id == "longmemeval-v1"
    assert contract.scorer_id == (
        "longmemeval_judge:deepseek/deepseek-v4-flash"
    )
    assert "Current Date:" in contract.retrieval_query(question)
    assert contract.judge_plan is not None
    answer_prompt = contract.answer_prompt(question, "retrieved context")
    assert answer_prompt.thinking_enabled is False
    steps = contract.judge_plan(question, "A bicycle")
    assert len(steps) == 1
    assert steps[0].prompt.temperature == 0
    assert steps[0].prompt.max_tokens == 10
    assert steps[0].prompt.thinking_enabled is False


def test_judge_result_names_the_actual_model_without_claiming_official_score() -> None:
    question = normalize_longmemeval([_record()]).cases[0].questions[0]
    contract = longmemeval_task_contract(judge_model_id="custom/judge")
    assert contract.judge_reducer is not None

    result = contract.judge_reducer(question, "A bicycle", (True,))

    assert result.scorer_id == "longmemeval_judge:custom/judge"
    assert result.details == {
        "judge_model_id": "custom/judge",
        "official_evaluator_contract": True,
        "official_metric_model": False,
    }


def test_unknown_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown LongMemEval"):
        normalize_longmemeval([_record()], question_ids=["missing"])


def test_claude_pilot_has_unique_fixed_cases_and_two_abstentions() -> None:
    assert len(LONGMEMEVAL_CLAUDE_PILOT_30_IDS) == 30
    assert len(set(LONGMEMEVAL_CLAUDE_PILOT_30_IDS)) == 30
    assert sum(case_id.endswith("_abs") for case_id in LONGMEMEVAL_CLAUDE_PILOT_30_IDS) == 2


def test_smoke_bundle_keeps_one_complete_session_without_label_leakage() -> None:
    record = _record(question_id=LONGMEMEVAL_SMOKE_CASE_ID)
    record["haystack_session_ids"] = [LONGMEMEVAL_SMOKE_SESSION_ID, "later"]
    record["haystack_dates"] = [
        "January 2, 2026 8:30 AM",
        "January 3, 2026 9:00 AM",
    ]
    record["haystack_sessions"] = [
        [
            {
                "role": "assistant" if index % 2 else "user",
                "content": f"turn {index}",
                **({"has_answer": True} if index == 1 else {}),
            }
            for index in range(LONGMEMEVAL_SMOKE_EVENT_COUNT)
        ],
        [
            {"role": "user", "content": f"later distractor {index}"}
            for index in range(484)
        ],
    ]
    record["answer_session_ids"] = [LONGMEMEVAL_SMOKE_SESSION_ID]
    full = normalize_longmemeval([record])

    smoke = longmemeval_smoke_bundle(full)

    case = smoke.cases[0]
    assert case.case_id == LONGMEMEVAL_SMOKE_CASE_ID
    assert len(case.events) == LONGMEMEVAL_SMOKE_EVENT_COUNT
    assert {event.session_id for event in case.events} == {
        LONGMEMEVAL_SMOKE_SESSION_ID
    }
    assert smoke.metadata == {
        "source": "xiaowu0162/longmemeval-cleaned",
        "run_mode": "integration-smoke",
        "source_case_event_count": 492,
        "included_event_count": LONGMEMEVAL_SMOKE_EVENT_COUNT,
        "included_session_ids": [LONGMEMEVAL_SMOKE_SESSION_ID],
    }
    assert "A bicycle" not in str(smoke._payload(include_labels=False))


def test_smoke_bundle_rejects_missing_evidence_or_unexpected_source_shape() -> None:
    record = _record(question_id=LONGMEMEVAL_SMOKE_CASE_ID)
    record["haystack_session_ids"] = [LONGMEMEVAL_SMOKE_SESSION_ID]
    record["haystack_dates"] = ["January 2, 2026 8:30 AM"]
    record["haystack_sessions"] = [[
        {"role": "user", "content": f"turn {index}"}
        for index in range(LONGMEMEVAL_SMOKE_EVENT_COUNT)
    ], [
        {"role": "user", "content": f"later turn {index}"}
        for index in range(484)
    ]]
    record["haystack_session_ids"] = [LONGMEMEVAL_SMOKE_SESSION_ID, "later"]
    record["haystack_dates"] = [
        "January 2, 2026 8:30 AM",
        "January 3, 2026 8:30 AM",
    ]
    record["answer_session_ids"] = []
    full = normalize_longmemeval([record])

    with pytest.raises(ValueError, match="evidence"):
        longmemeval_smoke_bundle(full)

    sessions = cast(list[list[dict[str, object]]], record["haystack_sessions"])
    sessions[0][0]["has_answer"] = True
    record["haystack_sessions"] = [sessions[0]]
    record["haystack_session_ids"] = [LONGMEMEVAL_SMOKE_SESSION_ID]
    record["haystack_dates"] = ["January 2, 2026 8:30 AM"]
    full = normalize_longmemeval([record])
    with pytest.raises(ValueError, match="source event count"):
        longmemeval_smoke_bundle(full)
