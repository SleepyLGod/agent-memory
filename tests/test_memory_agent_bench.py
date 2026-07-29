from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory.evaluation.memory_agent_bench import (
    MEMORY_AGENT_BENCH_REVISION,
    MEMORY_AGENT_BENCH_SPLITS,
    MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256,
    MEMORY_AGENT_TASKS,
    SMOKE_SOURCES,
    chunk_text_into_sentences,
    exact_match,
    normalize_memory_agent_bench,
    parse_output,
    recall_at_k,
    score_deterministic,
    substring_exact_match,
)
from agent_memory.evaluation.memory_agent_bench.contracts import (
    memory_agent_bench_task_contracts,
)


class _WordEncoder:
    def encode(self, text: str, *, allowed_special: set[str]) -> list[str]:
        del allowed_special
        return text.split()


def _split_rows(source: str, *, questions: list[str] | None = None) -> dict[str, list[dict[str, object]]]:
    rows = {split: [] for split in MEMORY_AGENT_BENCH_SPLITS}
    task = MEMORY_AGENT_TASKS[source]
    selected_questions = questions or ["Question one?", "Question two?"]
    rows[task.split].append(
        {
            "context": "First sentence. Second sentence. Third sentence.",
            "questions": selected_questions,
            "answers": [["one"], ["two"]][: len(selected_questions)],
            "metadata": {
                "source": source,
                "qa_pair_ids": [f"{source}-q{i}" for i in range(len(selected_questions))],
                "question_dates": None,
                "question_types": None,
                "question_ids": None,
            },
        }
    )
    return rows


def _merge_split_rows(*items: dict[str, list[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    merged = {split: [] for split in MEMORY_AGENT_BENCH_SPLITS}
    for item in items:
        for split, rows in item.items():
            merged[split].extend(rows)
    return merged


def _chunker(text: str) -> tuple[str, ...]:
    return chunk_text_into_sentences(
        text,
        chunk_size=4,
        sentence_tokenizer=lambda value: [part.strip() + "." for part in value.split(".") if part.strip()],
        token_encoder=_WordEncoder(),
    )


def test_dataset_pin_and_complete_source_registry() -> None:
    assert len(MEMORY_AGENT_BENCH_REVISION) == 40
    assert sum(contract["rows"] for contract in MEMORY_AGENT_BENCH_SPLITS.values()) == 146
    assert len(MEMORY_AGENT_TASKS) == 22
    assert set(SMOKE_SOURCES) <= set(MEMORY_AGENT_TASKS)
    assert MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256 == (
        "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"
    )


def test_official_chunking_preserves_sentence_boundaries() -> None:
    chunks = _chunker("One two. Three four. Five six.")
    assert chunks == ("One two. Three four.", "Five six.")


def test_mab_contract_uses_event_checkpoints_and_scores_memory_errors_zero() -> None:
    contracts = memory_agent_bench_task_contracts()

    assert contracts
    assert {contract.checkpoint_boundary for contract in contracts.values()} == {
        "event"
    }
    assert {contract.memory_system_error_score for contract in contracts.values()} == {
        0.0
    }


def test_missing_official_sentence_model_has_clear_setup_error() -> None:
    def missing(_: str) -> list[str]:
        raise LookupError("punkt_tab")

    with pytest.raises(RuntimeError, match="punkt_tab"):
        chunk_text_into_sentences(
            "Sentence.", sentence_tokenizer=missing, token_encoder=_WordEncoder()
        )


def test_inject_once_query_many_normalization() -> None:
    source = "icl_banking77_5900shot_balance"
    bundle = normalize_memory_agent_bench(
        _split_rows(source),
        sources=[source],
        chunker=_chunker,
    )
    case = bundle.cases[0]

    assert case.task_id == f"memory-agent-bench:{source}"
    assert len(case.events) == 2
    assert len(case.questions) == 2
    assert case.events[0].text.startswith("Dialogue between User and Assistant")
    assert case.questions[0].question_id == f"{source}-q0"
    assert all(not question.evidence_event_ids for question in case.questions)


def test_four_smoke_sources_cover_all_competencies() -> None:
    rows = _merge_split_rows(*(_split_rows(source, questions=["Question?"]) for source in SMOKE_SOURCES))
    bundle = normalize_memory_agent_bench(
        rows,
        sources=SMOKE_SOURCES,
        max_cases_per_source=1,
        max_questions_per_case=1,
        chunker=_chunker,
    )
    assert {case.metadata["source"] for case in bundle.cases} == set(SMOKE_SOURCES)
    assert {case.questions[0].category for case in bundle.cases} == {
        "accurate_retrieval",
        "test_time_learning",
        "long_range_understanding",
        "conflict_resolution",
    }


def test_unknown_source_never_falls_through() -> None:
    rows = _split_rows("eventqa_65536")
    rows["Accurate_Retrieval"][0]["metadata"] = {"source": "unknown"}
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_memory_agent_bench(rows, sources=None, chunker=_chunker)


def test_official_deterministic_scorers() -> None:
    assert substring_exact_match("The answer is Paris!", ["Paris"])
    assert exact_match("Answer: label 28", "label 28") is False
    assert exact_match("label: 28", "28") is False
    assert parse_output("Answer: Paris\nExplanation") == "Paris"
    assert recall_at_k(["A", "B", "C", "D", "E", "F"], ["B", "F"], 5) == 0.5

    event_task = MEMORY_AGENT_TASKS["eventqa_65536"]
    assert score_deterministic(event_task, "Answer: Paris", ["Paris"]) == 1.0
    exact_task = MEMORY_AGENT_TASKS["detective_qa"]
    assert score_deterministic(exact_task, "Answer: C. Suicide", ["C. Suicide"]) == 1.0


def test_llm_and_recommendation_scorers_require_their_explicit_path() -> None:
    with pytest.raises(ValueError, match="requires LLM scorer"):
        score_deterministic(
            MEMORY_AGENT_TASKS["longmemeval_s*"], "answer", ["answer"]
        )
    with pytest.raises(ValueError, match="entity2id"):
        score_deterministic(
            MEMORY_AGENT_TASKS["recsys_redial_full"], "answer", ["answer"]
        )


def test_task_prompt_is_shared_and_explicitly_contains_retrieval_context() -> None:
    task = MEMORY_AGENT_TASKS["factconsolidation_sh_6k"]
    prompt = task.format_answer_prompt("Who is current?", "Fact 2 is newer")
    assert prompt.startswith("Retrieved Memory:\nFact 2 is newer")
    assert "larger serial number" in prompt


def test_source_registry_rejects_wrong_split() -> None:
    rows = _split_rows("eventqa_65536")
    row = rows["Accurate_Retrieval"].pop()
    rows["Conflict_Resolution"].append(row)
    with pytest.raises(ValueError, match="wrong split"):
        normalize_memory_agent_bench(rows, sources=["eventqa_65536"], chunker=_chunker)


def test_source_registry_drift_names_missing_and_unexpected_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent_memory.evaluation.memory_agent_bench import dataset

    expected_sources = set(MEMORY_AGENT_TASKS)
    actual_sources = expected_sources - {"eventqa_65536"} | {"unexpected_source"}

    class _Frame:
        def __len__(self) -> int:
            return 1

        def to_dict(self, *, orient: str):
            assert orient == "records"
            return [
                {"metadata": {"source": source}}
                for source in sorted(actual_sources)
            ]

    for split, contract in MEMORY_AGENT_BENCH_SPLITS.items():
        path = tmp_path / f"{split}.parquet"
        path.write_bytes(b"fixture")
        monkeypatch.setitem(contract, "sha256", dataset._sha256_file(path))
        monkeypatch.setitem(contract, "rows", 1)
    monkeypatch.setattr("pandas.read_parquet", lambda _: _Frame())

    with pytest.raises(ValueError) as error:
        dataset.load_memory_agent_bench(tmp_path)

    assert "missing=['eventqa_65536']" in str(error.value)
    assert "unexpected=['unexpected_source']" in str(error.value)


def test_contract_registry_covers_every_official_source() -> None:
    contracts = memory_agent_bench_task_contracts()

    assert set(contracts) == {
        task.contract_id for task in MEMORY_AGENT_TASKS.values()
    }
    assert all(contract.task_id == contract_id for contract_id, contract in contracts.items())


def test_contract_answer_prompt_uses_official_task_instruction_and_context() -> None:
    source = "factconsolidation_sh_6k"
    case = normalize_memory_agent_bench(
        _split_rows(source, questions=["Who is current?"]),
        sources=[source],
        chunker=_chunker,
    ).cases[0]
    contract = memory_agent_bench_task_contracts()[case.task_id]

    prompt = contract.answer_prompt(case.questions[0], "Fact 2 is newer")

    assert prompt.messages[0] == {
        "role": "system",
        "content": MEMORY_AGENT_TASKS[source].system_prompt,
    }
    assert "Retrieved Memory:\nFact 2 is newer" in prompt.messages[1]["content"]
    assert "larger serial number" in prompt.messages[1]["content"]
    assert prompt.thinking_enabled is False


def test_infbench_contract_keeps_three_official_judge_calls() -> None:
    source = "infbench_sum_eng_shots2"
    rows = _split_rows(source, questions=["Summarize the story."])
    split_rows = rows["Long_Range_Understanding"]
    assert isinstance(split_rows, list)
    row = split_rows[0]
    assert isinstance(row, dict)
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    metadata["keypoints"] = [
        "The protagonist leaves home.",
        "The protagonist returns.",
    ]
    case = normalize_memory_agent_bench(
        rows,
        sources=[source],
        chunker=_chunker,
    ).cases[0]
    question = case.questions[0]
    contract = memory_agent_bench_task_contracts()[case.task_id]

    assert question.metadata["keypoints"] == [
        "The protagonist leaves home.",
        "The protagonist returns.",
    ]
    assert contract.judge_plan is not None
    steps = contract.judge_plan(question, "A fluent summary.")
    assert [step.prompt.prompt_name for step in steps] == [
        "memory_agent_bench.infbench.fluency",
        "memory_agent_bench.infbench.recall",
        "memory_agent_bench.infbench.precision",
    ]
    assert all(step.prompt.temperature == 0 for step in steps)
    assert all(step.prompt.thinking_enabled is False for step in steps)
    assert "The protagonist leaves home." in steps[1].prompt.messages[0]["content"]
    assert question.gold_answer[0] in steps[2].prompt.messages[0]["content"]


def test_infbench_reducer_matches_official_formula() -> None:
    source = "infbench_sum_eng_shots2"
    rows = _split_rows(source, questions=["Summarize the story."])
    split_rows = rows["Long_Range_Understanding"]
    assert isinstance(split_rows, list)
    row = split_rows[0]
    assert isinstance(row, dict)
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    metadata["keypoints"] = ["a", "b"]
    case = normalize_memory_agent_bench(
        rows,
        sources=[source],
        chunker=_chunker,
    ).cases[0]
    contract = memory_agent_bench_task_contracts()[case.task_id]
    assert contract.judge_reducer is not None

    grade = contract.judge_reducer(
        case.questions[0],
        "summary",
        (
            {"fluency": 1},
            {"recall": 1},
            {"precision": 3, "sentence_count": 4},
        ),
    )

    assert grade.scorer_id == "infbench_v4_flash_judge"
    assert grade.score == pytest.approx(0.6)
    assert grade.details == {
        "official_metric_model": False,
        "fluency": 1,
        "recall_found": 1,
        "recall_total": 2,
        "recall": 0.5,
        "precision_found": 3,
        "precision_total": 4,
        "precision": 0.75,
    }


def test_recsys_contract_uses_official_movie_mapping_for_recall_at_5() -> None:
    source = "recsys_redial_full"
    case = normalize_memory_agent_bench(
        _split_rows(source, questions=["Recommend movies"]),
        sources=[source],
        chunker=_chunker,
    ).cases[0]
    question = replace(case.questions[0], gold_answer=["1", "2"])
    contracts = memory_agent_bench_task_contracts(
        movie_entity_mapping={
            "<http://dbpedia.org/resource/Paris_(2008_film)>": 1,
            "<http://dbpedia.org/resource/Moonlight_(2016_film)>": 2,
            "<http://dbpedia.org/resource/Arrival_(film)>": 3,
        }
    )
    contract = contracts[case.task_id]
    assert contract.deterministic_scorer is not None

    grade = contract.deterministic_scorer(
        question,
        "The recommendations are:\n1. Paris\n2. Arrival\n3. Moonlight",
    )

    assert grade.scorer_id == "recall_at_5"
    assert grade.score == 1.0
    assert grade.details["predicted_movies"][:3] == [
        "Paris",
        "Arrival",
        "Moonlight",
    ]


def test_memory_agent_bench_smoke_bundle_records_run_mode() -> None:
    bundle = normalize_memory_agent_bench(
        _split_rows("eventqa_65536", questions=["What happened?"]),
        sources=["eventqa_65536"],
        max_cases_per_source=1,
        max_questions_per_case=1,
        run_mode="integration-smoke",
        chunker=_chunker,
    )

    assert bundle.metadata["run_mode"] == "integration-smoke"
