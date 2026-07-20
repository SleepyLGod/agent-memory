"""Tests for the formal Zep LOCOMO benchmark contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import agent_memory.evaluation.zep.locomo as zep_locomo_module
from agent_memory.evaluation.locomo import normalize_locomo_sample
from agent_memory.evaluation.types import BenchmarkQuestion
from agent_memory.evaluation.zep.answering import (
    ANSWER_PROMPT_DIGEST,
    ZEP_JUDGE_PROMPT_DIGEST,
    AnswerRecord,
    format_retrieval_context,
    generate_answer,
    generate_zep_judge,
)
from agent_memory.evaluation.zep.artifacts import ArtifactStore
from agent_memory.evaluation.zep.locomo import (
    EvaluationResult,
    LOCOMO_SHA256,
    RetrievalRecord,
    ZepLocomoRunConfig,
    build_manifest,
    evaluate_questions,
    event_to_zep_log_row,
    run_ingestion,
    run_zep_locomo,
    select_questions_for_events,
    select_event_range,
    validate_pinned_dataset,
)
from agent_memory.evaluation.zep.scoring import OfficialGrade, ZepJudgeGrade
from agent_memory.policy.retrieval import RetrievalResult

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/evaluation/zep_locomo.py"


def _fixture() -> dict[str, object]:
    return {
        "sample_id": "conv-test",
        "conversation": {
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I started researching adoption agencies.",
                    "blip_caption": "an adoption agency brochure",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:2",
                    "text": "That sounds like a big step.",
                },
            ],
        },
        "qa": [
            {
                "question": "What did Caroline research?",
                "answer": "adoption agencies",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What did Caroline buy yesterday?",
                "adversarial_answer": "No information available.",
                "evidence": [],
                "category": 5,
            },
        ],
    }


def test_cli_has_explicit_smoke_and_full_benchmark_defaults(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("zep_locomo_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    smoke = module.parse_args(["smoke", "--output-dir", str(tmp_path / "smoke")])
    benchmark = module.parse_args(
        ["benchmark", "--output-dir", str(tmp_path / "benchmark")]
    )

    assert (smoke.start_row, smoke.row_limit) == (26, 3)
    assert (smoke.question_start, smoke.question_limit) == (4, 1)
    assert (benchmark.start_row, benchmark.row_limit) == (1, None)
    assert (benchmark.question_start, benchmark.question_limit) == (1, None)
    assert benchmark.include_adversarial is True


def test_cli_accepts_exact_question_numbers_and_rejects_range_mix(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("zep_locomo_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "smoke",
            "--question-numbers",
            "4",
            "86",
            "154",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert args.question_numbers == (4, 86, 154)

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "benchmark",
                "--question-numbers",
                "4",
                "--question-start",
                "4",
                "--output-dir",
                str(tmp_path / "invalid"),
            ]
        )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="What did Caroline research?",
        channels={
            "entities": pd.DataFrame.from_records(
                [
                    {
                        "record_id": "entity-1",
                        "name": "Caroline",
                        "summary": "Caroline is researching adoption.",
                        "rank": 1,
                        "score": 1.0,
                    }
                ]
            ),
            "facts": pd.DataFrame.from_records(
                [
                    {
                        "record_id": "fact-1",
                        "fact": "Caroline researched adoption agencies.",
                        "valid_at": "2023-05-08T13:56:00",
                        "invalid_at": None,
                        "expired_at": None,
                        "rank": 1,
                        "score": 1.0,
                    }
                ]
            ),
        },
        metrics={
            "entities": {"latency_ms": 1.0},
            "facts": {"latency_ms": 2.0, "bfs_origins": ["entity-1"]},
        },
    )


def test_normalization_preserves_zep_event_metadata() -> None:
    sample = normalize_locomo_sample(_fixture(), sample_index=0)

    event = sample.events[0]
    assert event.metadata == {
        "sample_index": 0,
        "row_number": 1,
        "session_number": 1,
        "blip_caption": "an adoption agency brochure",
    }
    assert sample.questions[0].metadata["question_number"] == 1
    assert sample.questions[1].metadata["adversarial_answer"] == (
        "No information available."
    )


def test_zep_event_mapping_preserves_caption_and_local_reference_time() -> None:
    event = normalize_locomo_sample(_fixture(), sample_index=0).events[0]

    row = event_to_zep_log_row(event)

    assert row == {
        "content": (
            "Caroline: I started researching adoption agencies.\n"
            "(description of attached image: an adoption agency brochure)"
        ),
        "role": "Caroline",
        "speaker": "Caroline",
        "reference_time": "2023-05-08T13:56:00",
        "source_description": "LOCOMO sample 0 session 1",
    }


def test_event_range_is_one_based_and_contiguous() -> None:
    events = normalize_locomo_sample(_fixture(), sample_index=0).events

    selected = select_event_range(events, start_row=2, row_limit=1)

    assert tuple(event.event_id for event in selected) == ("D1:2",)


def test_question_selection_uses_original_number_and_evidence_boundary() -> None:
    sample = normalize_locomo_sample(_fixture(), sample_index=0)

    selected = select_questions_for_events(
        sample.questions,
        sample.events,
        start_question=1,
        question_limit=1,
    )

    assert tuple(question.question_id for question in selected) == ("conv-test:q1",)


def test_question_selection_accepts_exact_numbers_and_rejects_missing_evidence() -> None:
    sample = normalize_locomo_sample(_fixture(), sample_index=0)

    selected = select_questions_for_events(
        sample.questions,
        sample.events,
        question_numbers=(2, 1),
    )
    assert tuple(question.question_id for question in selected) == (
        "conv-test:q2",
        "conv-test:q1",
    )

    with pytest.raises(ValueError, match="question 1.*D1:1"):
        select_questions_for_events(
            sample.questions,
            sample.events[1:],
            question_numbers=(1,),
        )


def test_pinned_dataset_validation_rejects_wrong_bytes(tmp_path: Path) -> None:
    dataset = tmp_path / "locomo.json"
    dataset.write_text("[]", encoding="utf-8")

    try:
        validate_pinned_dataset(dataset)
    except ValueError as error:
        assert LOCOMO_SHA256 in str(error)
    else:
        raise AssertionError("wrong LOCOMO bytes should be rejected")


def test_answer_and_judge_contracts_match_locked_native_digests() -> None:
    assert ANSWER_PROMPT_DIGEST == (
        "dbd2f002e7cfafb4552a18d5c17e591cb6dd7d5eb44c762afd34094736e7fb6b"
    )
    assert ZEP_JUDGE_PROMPT_DIGEST == (
        "e74e7fbe755a002e5a8a05b3c661ed159a13a15c75c1953841019fc72d3cf83a"
    )


def test_manifest_locks_dual_scorers_and_native_retrieval_contract(
    tmp_path: Path,
) -> None:
    config = ZepLocomoRunConfig(
        dataset_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "run",
        namespace="zep-test",
    )

    manifest = build_manifest(
        config,
        namespace="zep-test",
        dataset_sha256=LOCOMO_SHA256,
        policy_input_sha256="policy-input",
        agent_memory_commit="abc123",
        agent_memory_dirty=True,
    )

    assert manifest["scoring"] == {
        "official_categories": [1, 2, 3, 4, 5],
        "zep_judge_categories": [1, 2, 3, 4],
        "shared_answer_stream": True,
    }
    assert manifest["retrieval"] == {
        "entity_methods": ["bm25", "cosine_similarity"],
        "entity_reranker": "rrf",
        "entity_limit": 20,
        "fact_methods": ["bm25", "cosine_similarity", "bfs"],
        "fact_reranker": "cross_encoder",
        "fact_limit": 20,
        "bfs_max_depth": 3,
        "generative_llm": False,
    }
    assert manifest["prompts"]["answer_sha256"] == ANSWER_PROMPT_DIGEST
    assert manifest["prompts"]["zep_judge_sha256"] == ZEP_JUDGE_PROMPT_DIGEST


def test_manifest_can_exclude_adversarial_from_official_scoring(tmp_path: Path) -> None:
    config = ZepLocomoRunConfig(
        dataset_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "run",
        include_adversarial=False,
    )

    manifest = build_manifest(
        config,
        namespace="zep-test",
        dataset_sha256=LOCOMO_SHA256,
        policy_input_sha256="policy-input",
        agent_memory_commit="abc123",
        agent_memory_dirty=False,
    )

    assert manifest["scoring"]["official_categories"] == [1, 2, 3, 4]


def test_retrieval_context_matches_published_zep_shape() -> None:
    context = format_retrieval_context(_retrieval())

    assert "<FACTS>" in context
    assert (
        "Caroline researched adoption agencies. "
        "(event_time: 2023-05-08T13:56:00)" in context
    )
    assert "Caroline: Caroline is researching adoption." in context


def test_questions_generate_one_answer_and_share_it_across_both_scorers() -> None:
    sample = normalize_locomo_sample(_fixture(), sample_index=0)
    memory = SimpleNamespace(query=lambda question: _retrieval())
    answer_calls: list[str] = []
    judge_calls: list[tuple[str, str]] = []

    def answerer(
        question: BenchmarkQuestion,
        retrieval: RetrievalResult,
    ) -> AnswerRecord:
        answer_calls.append(question.question_id)
        answer = (
            "adoption agencies"
            if question.category != "5"
            else "No information available."
        )
        return AnswerRecord.from_question(
            question,
            retrieval,
            answer=answer,
            latency_ms=1.0,
        )

    def judge(question: BenchmarkQuestion, answer: str) -> ZepJudgeGrade:
        judge_calls.append((question.question_id, answer))
        return ZepJudgeGrade(
            question_id=question.question_id,
            category=int(question.category),
            label="CORRECT",
            is_correct=True,
            reasoning="The answer matches.",
            latency_ms=1.0,
        )

    result = evaluate_questions(
        memory,
        sample.questions,
        answerer=answerer,
        zep_judge=judge,
    )

    assert answer_calls == ["conv-test:q1", "conv-test:q2"]
    assert judge_calls == [("conv-test:q1", "adoption agencies")]
    assert [grade.score for grade in result.official_grades] == [1.0, 1.0]
    assert [grade.question_id for grade in result.zep_judge_grades] == [
        "conv-test:q1"
    ]
    assert result.answers[0].answer == judge_calls[0][1]
    assert result.answers[1].gold_answer is None


class _SequencedLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(outputs=[next(self.outputs)])


def test_answer_semantic_validation_retries_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    lm = _SequencedLM(['{"answer": ""}', '{"answer": "adoption agencies"}'])
    monkeypatch.setattr(lotus.settings, "lm", lm)
    question = normalize_locomo_sample(_fixture(), sample_index=0).questions[0]

    answer = generate_answer(question, _retrieval())

    assert answer.answer == "adoption agencies"
    assert lm.calls == 2


def test_judge_semantic_validation_retries_invalid_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    lm = _SequencedLM(
        [
            '{"label": "MAYBE", "reasoning": "uncertain"}',
            '{"label": "correct", "reasoning": "The answer matches."}',
        ]
    )
    monkeypatch.setattr(lotus.settings, "lm", lm)
    question = normalize_locomo_sample(_fixture(), sample_index=0).questions[0]

    grade = generate_zep_judge(question, "adoption agencies")

    assert grade.label == "CORRECT"
    assert lm.calls == 2


def test_answer_semantic_validation_exhausts_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    lm = _SequencedLM(['{"answer": ""}'] * 4)
    monkeypatch.setattr(lotus.settings, "lm", lm)
    question = normalize_locomo_sample(_fixture(), sample_index=0).questions[0]

    with pytest.raises(ValueError, match="non-empty answer"):
        generate_answer(question, _retrieval())
    assert lm.calls == 4


def test_ingestion_calls_memory_add_with_zep_rows_and_records_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = normalize_locomo_sample(_fixture(), sample_index=0)
    added: list[dict[str, str]] = []
    state = {
        "episodes": pd.DataFrame([{"episode_id": "episode-1"}]),
        "entities": pd.DataFrame([{"entity_id": "entity-1"}]),
        "facts": pd.DataFrame(),
    }
    memory = SimpleNamespace(
        add=lambda row: added.append(row),
        _runtime=SimpleNamespace(_state=state),
    )
    snapshots = iter(
        [
            {field: 0 for field in zep_locomo_module.USAGE_FIELDS},
            {field: 1 for field in zep_locomo_module.USAGE_FIELDS},
        ]
        * len(sample.events)
    )
    monkeypatch.setattr(
        zep_locomo_module,
        "_usage_snapshot",
        lambda: next(snapshots),
    )

    metrics = run_ingestion(memory, sample.events)

    assert added[0]["content"].startswith("Caroline: I started researching")
    assert len(metrics) == 2
    assert metrics[0]["event_id"] == "D1:1"
    assert metrics[0]["episodes_rows"] == 1
    assert metrics[0]["physical_total_tokens"] == 1


def test_artifact_store_writes_required_dual_score_outputs(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "run")
    sample = normalize_locomo_sample(_fixture(), sample_index=0)

    store.write_manifest({"system": "agent-memory-zep"})
    store.write_inputs(sample.events, sample.questions)
    store.write_status(status="completed", phase="complete")
    store.write_ingestion_metrics([])
    store.write_retrievals([])
    store.write_retrieval_metrics([])
    store.write_answers([])
    store.write_grades([], [])
    store.write_metrics()

    expected = {
        "manifest.json",
        "status.json",
        "input/events.jsonl",
        "input/questions.jsonl",
        "insertion/metrics.csv",
        "retrieval/results.jsonl",
        "retrieval/metrics.csv",
        "answers/results.jsonl",
        "grades/official.jsonl",
        "grades/official_summary.csv",
        "grades/zep_judge.jsonl",
        "grades/zep_judge_summary.csv",
        "metrics/llm_calls.csv",
        "metrics/summary.json",
    }
    actual = {
        str(path.relative_to(store.output_dir))
        for path in store.output_dir.rglob("*")
        if path.is_file()
    }
    assert expected <= actual
    assert json.loads((store.output_dir / "status.json").read_text())["status"] == (
        "completed"
    )


def test_artifact_metrics_use_provider_cache_and_reasoning_usage(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "run")
    usage_path = store.output_dir / "trace/outputs/usage.json"
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text(
        json.dumps({"completion_tokens_details": {"reasoning_tokens": 20}}),
        encoding="utf-8",
    )
    events = [
        {
            "trace_id": "llm-1",
            "event_type": "llm_call",
            "llm_item_index": 0,
            "phase": "answering",
            "operator": "locomo.answer",
            "prompt_name": "locomo.answer",
            "model": "deepseek/deepseek-v4-flash",
            "latency_sec": 1.25,
            "usage_physical_prompt_tokens": 300,
            "usage_physical_completion_tokens": 50,
            "usage_physical_total_tokens": 350,
        },
        {
            "trace_id": "usage-1",
            "event_type": "provider_usage",
            "phase": "answering",
            "provider_prompt_tokens": 300,
            "provider_completion_tokens": 50,
            "provider_prompt_cache_hit_tokens": 100,
            "provider_prompt_cache_miss_tokens": 200,
            "provider_raw_usage_path": "trace/outputs/usage.json",
        },
    ]
    (store.trace_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    store.write_metrics()

    summary = json.loads(
        (store.output_dir / "metrics/summary.json").read_text(encoding="utf-8")
    )
    answering = summary["phases"]["answering"]
    assert answering["llm_batch_count"] == 1
    assert answering["llm_latency_ms"] == 1250.0
    assert answering["cache_hit_tokens"] == 100
    assert answering["cache_miss_tokens"] == 200
    assert answering["reasoning_tokens"] == 20
    assert answering["estimated_cost_usd"] == 0.00004228


def test_formal_runner_writes_dual_scores_and_closes_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "locomo.json"
    dataset.write_text(json.dumps([_fixture()]), encoding="utf-8")
    output_dir = tmp_path / "run"
    config = ZepLocomoRunConfig(
        dataset_path=dataset,
        output_dir=output_dir,
        start_row=1,
        row_limit=2,
        question_start=1,
        question_limit=1,
        namespace="zep-test",
    )
    sample = normalize_locomo_sample(_fixture(), sample_index=0)
    question = sample.questions[0]
    retrieval = _retrieval()
    retrieval_record = RetrievalRecord.from_result(
        question,
        retrieval,
        latency_ms=1.0,
    )
    answer = AnswerRecord.from_question(
        question,
        retrieval,
        answer="adoption agencies",
        latency_ms=1.0,
    )
    result = EvaluationResult(
        retrievals=(retrieval_record,),
        answers=(answer,),
        official_grades=(
            OfficialGrade(
                question_id=question.question_id,
                category=2,
                score=1.0,
                latency_ms=0.1,
            ),
        ),
        zep_judge_grades=(
            ZepJudgeGrade(
                question_id=question.question_id,
                category=2,
                label="CORRECT",
                is_correct=True,
                reasoning="The answer matches.",
                latency_ms=1.0,
            ),
        ),
    )
    connector = SimpleNamespace(close=Mock())
    storage = SimpleNamespace(connector=connector)
    memory = SimpleNamespace()
    monkeypatch.setattr(zep_locomo_module, "require_environment", lambda: None)
    monkeypatch.setattr(
        zep_locomo_module,
        "validate_pinned_dataset",
        lambda path: LOCOMO_SHA256,
    )
    monkeypatch.setattr(
        zep_locomo_module,
        "_git_state",
        lambda: ("abc123", False),
    )
    monkeypatch.setattr(
        zep_locomo_module,
        "create_runtime",
        lambda **kwargs: (storage, memory, object()),
    )
    monkeypatch.setattr(
        zep_locomo_module,
        "run_ingestion",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        zep_locomo_module,
        "checkpoint_round_trip",
        lambda *args, **kwargs: (
            memory,
            b"checkpoint",
            {"round_trip_verified": True},
            {"result_order_verified": True},
        ),
    )
    monkeypatch.setattr(
        zep_locomo_module,
        "evaluate_questions",
        lambda *args, **kwargs: result,
    )

    completed = run_zep_locomo(config)

    assert completed == output_dir
    assert json.loads((output_dir / "status.json").read_text())["status"] == (
        "completed"
    )
    assert len((output_dir / "answers/results.jsonl").read_text().splitlines()) == 1
    assert len((output_dir / "grades/official.jsonl").read_text().splitlines()) == 1
    assert len((output_dir / "grades/zep_judge.jsonl").read_text().splitlines()) == 1
    connector.close.assert_called_once_with()
