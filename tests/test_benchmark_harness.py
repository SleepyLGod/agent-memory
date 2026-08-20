from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import pytest

from agent_memory.evaluation.artifacts import BenchmarkArtifactStore
from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.harness import (
    BenchmarkRunner,
    GradeContract,
    GradeResult,
    JudgeStep,
    MemorySystemContract,
    ModelPrompt,
    ModelResponse,
    RetrievalOutput,
    TaskContract,
)
from agent_memory.evaluation.locomo_contracts import (
    LOCOMO_ANSWER_PARSER_ID,
    locomo_task_contract,
)
from agent_memory.evaluation.recovery import UnitAttemptStore
from agent_memory.evaluation.types import BenchmarkCase, BenchmarkEvent, BenchmarkQuestion
from agent_memory.tracing.semantic import active_trace_scope, write_trace_event


def _bundle() -> BenchmarkBundle:
    case = BenchmarkCase(
        case_id="case-1",
        task_id="task-1",
        events=(
            BenchmarkEvent(
                sample_id="case-1",
                event_id="event-1",
                speaker="user",
                text="remember one",
            ),
            BenchmarkEvent(
                sample_id="case-1",
                event_id="event-2",
                speaker="user",
                text="remember two",
            ),
        ),
        questions=(
            BenchmarkQuestion("q1", "case-1", "first?", "one", (), "test"),
            BenchmarkQuestion("q2", "case-1", "second?", "two", (), "test"),
        ),
    )
    return BenchmarkBundle("benchmark", "revision", "sha", (case,))


def _prompt(question: BenchmarkQuestion, context: str) -> ModelPrompt:
    return ModelPrompt(
        prompt_name="answer",
        messages=({"role": "user", "content": f"{context}\n{question.question}"},),
        prompt_digest="answer-digest",
    )


def _parse_answer(value: str) -> str:
    answer = value.strip()
    if not answer:
        raise ValueError("answer is empty")
    return answer


def _contract() -> TaskContract:
    return TaskContract(
        task_id="task-1",
        answer_prompt=_prompt,
        answer_parser=_parse_answer,
        retrieval_query=lambda question: f"query:{question.question}",
        answer_prompt_digest="answer-digest",
        scorer_id="exact",
        scorer_digest="score-digest",
        deterministic_scorer=lambda question, answer: GradeResult(
            scorer_id="exact", score=float(answer == question.gold_answer)
        ),
    )


def _system_contract() -> MemorySystemContract:
    return MemorySystemContract(
        system_id="fake",
        memory_model_id="memory-model",
        memory_provider_model_id="provider-memory-model",
        input_adapter_id="fake-input:v1",
        retrieval_recipe_id="fake-retrieval:v1",
        checkpoint_enabled=True,
    )


def test_framework_cache_mode_changes_only_maintenance_identity() -> None:
    disabled = _system_contract()
    enabled = replace(disabled, framework_cache_mode="lotus-memory:1024")

    assert disabled.maintenance_fingerprint != enabled.maintenance_fingerprint
    assert disabled.retrieval_recipe_id == enabled.retrieval_recipe_id


def test_checkpoint_restore_rejects_changed_framework_cache_mode(
    tmp_path: Path,
) -> None:
    bundle = _session_bundle()
    disabled = _system_contract()
    maintenance_dir = tmp_path / "maintenance"
    BenchmarkRunner(
        system_contract=disabled,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(maintenance_dir),
        maintenance_only=True,
    ).run(bundle)
    enabled = replace(disabled, framework_cache_mode="lotus-memory:1024")
    runner = BenchmarkRunner(
        system_contract=enabled,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path / "retrieval"),
        maintenance_checkpoint_source=BenchmarkArtifactStore(maintenance_dir),
    )

    with pytest.raises(ValueError, match="maintenance_fingerprint"):
        runner.run(bundle)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"answer":"Speyer"}', "Speyer"),
        ('```json\n{"answer":"Speyer"}\n```', "Speyer"),
        ("Speyer", "Speyer"),
        ('{"wrong_field":"schema echo"}', '{"wrong_field":"schema echo"}'),
    ],
)
def test_locomo_answer_parser_accepts_structured_or_raw_text(
    response: str,
    expected: str,
) -> None:
    contract = locomo_task_contract()

    assert contract.answer_parser(response) == expected
    assert contract.answer_parser_id == LOCOMO_ANSWER_PARSER_ID


def test_locomo_answer_parser_rejects_empty_and_fingerprints_contract() -> None:
    contract = locomo_task_contract()

    with pytest.raises(ValueError, match="non-empty"):
        contract.answer_parser(" \n")

    previous = replace(contract, answer_parser_id="")
    assert contract.answer_prompt_digest == previous.answer_prompt_digest
    assert contract.fingerprint != previous.fingerprint


def test_locomo_schema_echo_is_scored_as_zero_without_special_case() -> None:
    contract = locomo_task_contract()
    question = BenchmarkQuestion(
        question_id="q167",
        sample_id="case",
        question="What is not in memory?",
        gold_answer="No information available",
        evidence_event_ids=(),
        category="5",
    )
    schema_echo = '{"type":"object","properties":{"answer":{"type":"string"}}}'
    answer = contract.answer_parser(schema_echo)
    assert contract.deterministic_scorer is not None

    assert answer == schema_echo
    assert contract.deterministic_scorer(question, answer).score == 0.0


@dataclass
class _Driver:
    state_dir: Path
    system_id: str = "fake"
    events: list[str] | None = None
    queries: list[str] | None = None
    closed: bool = False
    fail_retrieval: bool = False
    fail_event_id: str | None = None
    fail_finish: bool = False
    restored_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.events = []
        self.queries = []

    def add(self, event: BenchmarkEvent) -> dict[str, int]:
        assert self.events is not None
        if event.event_id == self.fail_event_id:
            raise RuntimeError(f"failed to add {event.event_id}")
        self.events.append(event.event_id)
        return {"row_count": len(self.events)}

    def retrieve(self, request):
        assert self.queries is not None
        self.queries.append(request.query_text)
        if self.fail_retrieval:
            raise RuntimeError("retrieval failed")
        answer = "one" if request.question_id == "q1" else "two"
        return RetrievalOutput(
            context=answer,
            channels={"rows": ({"answer": answer},)},
            metrics={"candidate_count": 1},
        )

    def finish_session(self, session_id: str) -> dict[str, str]:
        if self.fail_finish:
            raise RuntimeError("consolidation failed")
        return {"session_id": session_id}

    def close(self) -> None:
        self.closed = True

    def save_state(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        event_ids = list(self.events or ())
        (directory / "driver.json").write_text(
            json.dumps({"event_ids": event_ids}),
            encoding="utf-8",
        )
        return {"format": "fake-driver:v1", "event_count": len(event_ids)}

    def restore_state(
        self,
        directory: Path,
        completed_events: tuple[BenchmarkEvent, ...],
    ) -> None:
        state = json.loads((directory / "driver.json").read_text(encoding="utf-8"))
        restored = tuple(state["event_ids"])
        assert restored == tuple(event.event_id for event in completed_events)
        self.events = list(restored)
        self.restored_event_ids = restored


class _Model:
    model_id = "fake-model"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or ["one", "two"])
        self.calls: list[tuple[str, int]] = []

    def complete(self, prompt: ModelPrompt, *, attempt: int) -> ModelResponse:
        self.calls.append((prompt.prompt_name, attempt))
        text = self.responses.pop(0)
        return ModelResponse(
            model=self.model_id,
            text=text,
            raw_response={"text": text},
            usage={
                "prompt_tokens": 2,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 2,
                "completion_tokens": 1,
                "reasoning_tokens": 0,
            },
            latency_ms=1.5,
        )


def test_runner_injects_once_queries_many_and_writes_contract(tmp_path) -> None:
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        assert case_id == "case-1"
        del trace_dir
        driver = _Driver(state_dir)
        drivers.append(driver)
        return driver

    model = _Model()
    artifacts = BenchmarkArtifactStore(tmp_path)
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=artifacts,
    )
    runner.run(_bundle())

    assert drivers[0].events == ["event-1", "event-2"]
    assert drivers[0].queries == ["query:first?", "query:second?"]
    assert drivers[0].closed
    assert model.calls == [("answer", 1), ("answer", 1)]
    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset_revision"] == "revision"
    assert manifest["case_ids"] == ["case-1"]
    assert manifest["question_ids"] == ["q1", "q2"]
    assert manifest["answer_prompt_digests"] == {"task-1": "answer-digest"}
    assert manifest["scorer_contracts"] == {
        "task-1": {
            "scorer_id": "exact",
            "scorer_digest": "score-digest",
            "additional_graders": [],
        }
    }
    assert summary["completed_cases"] == 1
    assert summary["failed_cases"] == 0
    assert summary["mean_score"] == 1.0
    assert summary["provider_call_count"] == 2
    assert summary["provider_error_count"] == 0
    assert summary["question_count"] == 2
    assert summary["estimated_cost_usd"] == 1.12e-06
    assert summary["driver_setup_error_count"] == 0
    assert summary["driver_setup_wall_latency"]["mean_ms"] is not None
    assert summary["embedding_call_count"] == 0
    assert summary["embedding_error_count"] == 0
    assert summary["framework_cache_usage"] == {
        "observed_operation_count": 0,
        "error_count": 0,
        "lm_cache_hits": 0,
        "operator_cache_hits": 0,
        "physical_prompt_tokens": 0,
        "physical_completion_tokens": 0,
        "physical_total_tokens": 0,
        "virtual_prompt_tokens": 0,
        "virtual_completion_tokens": 0,
        "virtual_total_tokens": 0,
    }
    assert summary["phases"]["answering"]["prompt_tokens"] == 4
    for filename in (
        "overview.csv",
        "per_event.csv",
        "per_session.csv",
        "per_question.csv",
        "provider_usage.csv",
        "embedding_usage.csv",
        "framework_cache_usage.csv",
        "operation_usage.csv",
        "per_case.csv",
        "state_shape.csv",
        "reliability.csv",
        "checkpoint_metrics.csv",
    ):
        assert (tmp_path / "metrics" / filename).is_file()
    assert len((tmp_path / "metrics" / "per_event.csv").read_text().splitlines()) == 3
    assert len(
        (tmp_path / "metrics" / "checkpoint_metrics.csv").read_text().splitlines()
    ) == 2
    with (tmp_path / "metrics" / "per_case.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        case_metric = next(csv.DictReader(stream))
    assert float(case_metric["estimated_cost_usd"]) == 1.12e-06
    assert float(case_metric["cost_per_question_usd"]) == 5.6e-07


def test_runner_answers_once_and_records_primary_and_secondary_graders(
    tmp_path: Path,
) -> None:
    case = replace(_bundle().cases[0], questions=(_bundle().cases[0].questions[0],))
    bundle = replace(_bundle(), cases=(case,))
    contract = replace(
        _contract(),
        additional_graders=(
            GradeContract(
                scorer_id="secondary",
                scorer_digest="secondary-digest",
                deterministic_scorer=lambda question, answer: GradeResult(
                    scorer_id="secondary",
                    score=float(answer == question.gold_answer),
                ),
            ),
        ),
    )
    model = _Model(["one"])

    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": contract},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    ).run(bundle)

    assert model.calls == [("answer", 1)]
    case_dir = BenchmarkArtifactStore(tmp_path).case_dir("case-1")
    grades = [
        json.loads(line)
        for line in (case_dir / "grades.jsonl").read_text().splitlines()
    ]
    assert [(row["scorer_id"], row["primary"]) for row in grades] == [
        ("exact", True),
        ("secondary", False),
    ]
    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    assert summary["question_count"] == 1
    assert summary["mean_score"] == 1.0
    assert summary["scores_by_scorer"] == {
        "exact": {"grade_count": 1, "mean_score": 1.0},
        "secondary": {"grade_count": 1, "mean_score": 1.0},
    }

    grade_path = case_dir / "grades.jsonl"
    grade_rows = [json.loads(line) for line in grade_path.read_text().splitlines()]
    grade_rows[0]["latency_ms"] = 0.7
    grade_rows[1]["latency_ms"] = 860.0
    grade_path.write_text(
        "".join(json.dumps(row) + "\n" for row in grade_rows),
        encoding="utf-8",
    )
    BenchmarkArtifactStore(tmp_path).finalize_metrics()

    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    assert "grading_wall_latency" not in summary
    assert summary["grading_wall_latency_by_scorer"] == {
        "exact": {
            "mean_ms": 0.7,
            "median_ms": 0.7,
            "p95_ms": 0.7,
        },
        "secondary": {
            "mean_ms": 860.0,
            "median_ms": 860.0,
            "p95_ms": 860.0,
        },
    }
    with (tmp_path / "metrics" / "per_question.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        question_row = next(csv.DictReader(stream))
    assert question_row["primary_scorer_id"] == "exact"
    assert question_row["primary_grading_latency_ms"] == "0.7"
    assert "scorer_id" not in question_row
    assert "grading_latency_ms" not in question_row
    with (tmp_path / "metrics" / "per_case.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        case_row = next(csv.DictReader(stream))
    assert "grading_wall_latency_ms" not in case_row
    with (tmp_path / "metrics" / "overview.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        overview_row = next(csv.DictReader(stream))
    assert not any(key.startswith("grading_") for key in overview_row)


def test_driver_setup_failure_records_zero_side_effect_boundary(tmp_path) -> None:
    contract = replace(
        _system_contract(),
        system_id="zep-memory",
        maintenance_rule="rule-join-map",
    )

    def fail_compilation(case_id, state_dir, trace_dir):
        del case_id, state_dir, trace_dir
        raise NotImplementedError(
            "sem_groupby partition_by is not supported by rule-join-map"
        )

    artifacts = BenchmarkArtifactStore(tmp_path)
    runner = BenchmarkRunner(
        system_contract=contract,
        contracts={"task-1": _contract()},
        driver_factory=fail_compilation,
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=artifacts,
    )

    with pytest.raises(NotImplementedError, match="partition_by.*rule-join-map"):
        runner.run(_bundle())

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    case_dir = artifacts.case_dir("case-1")
    status = json.loads((case_dir / "status.json").read_text())
    assert manifest["system_id"] == "zep-memory"
    assert manifest["maintenance_rule"] == "rule-join-map"
    assert status["failure_phase"] == "driver_setup"
    assert status["requested_strategy"] == "rule-join-map"
    assert status["policy_id"] == "zep-memory"
    assert status["add_count"] == 0
    assert status["storage_transaction_count"] == 0
    assert not (case_dir / "checkpoints/current.json").exists()
    events = [
        json.loads(line)
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
    ]
    setup = next(row for row in events if row["event_type"] == "driver_setup_result")
    assert setup["status"] == "error"
    assert setup["error_type"] == "NotImplementedError"


def test_embedding_usage_metrics_remain_separate_from_provider_costs(
    tmp_path: Path,
) -> None:
    class EmbeddingDriver(_Driver):
        trace_dir: Path

        def add(self, event: BenchmarkEvent) -> dict[str, int]:
            write_trace_event(
                self.trace_dir,
                operator="embedding",
                event_type="embedding_call",
                payload={
                    "status": "success",
                    "model": "BAAI/bge-m3",
                    "revision": "revision",
                    "batch_size": 1,
                    "dimensions": 1024,
                    "result_count": 1,
                    "result_dimensions": 1024,
                    "normalize": True,
                    "device": "cpu",
                    "latency_ms": 2.5,
                    "input_path": "trace/prompts/input.json",
                },
            )
            return super().add(event)

    def factory(case_id: str, state_dir: Path, trace_dir: Path) -> _Driver:
        del case_id
        driver = EmbeddingDriver(state_dir)
        driver.trace_dir = trace_dir
        return driver

    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    ).run(_bundle())

    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    assert summary["embedding_call_count"] == 2
    assert summary["embedding_error_count"] == 0
    assert summary["embedding_latency_sum_ms"] == 5.0
    assert summary["embedding_phases"]["insertion"]["call_count"] == 2
    assert summary["provider_call_count"] == 2
    with (tmp_path / "metrics" / "embedding_usage.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        embedding_rows = list(csv.DictReader(stream))
    assert len(embedding_rows) == 2
    with (tmp_path / "metrics" / "per_event.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        event_rows = list(csv.DictReader(stream))
    assert [row["embedding_call_count"] for row in event_rows] == ["1", "1"]
    assert all(row["estimated_cost_usd"] == "0.0" for row in event_rows)


def test_insertion_metrics_separate_measured_semantic_trace_io(
    tmp_path: Path,
) -> None:
    class TraceWritingDriver(_Driver):
        trace_dir: Path

        def add(self, event: BenchmarkEvent) -> dict[str, int]:
            write_trace_event(
                self.trace_dir,
                operator="sem_filter",
                event_type="pair_decision",
                payload={"decision": True, "evidence": "x" * 256},
            )
            return super().add(event)

    def factory(case_id: str, state_dir: Path, trace_dir: Path) -> _Driver:
        del case_id
        driver = TraceWritingDriver(state_dir)
        driver.trace_dir = trace_dir
        return driver

    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    ).run(_bundle())

    with (tmp_path / "metrics" / "per_event.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        event_rows = list(csv.DictReader(stream))
    with (tmp_path / "metrics" / "per_case.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        case_row = next(csv.DictReader(stream))
    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())

    assert len(event_rows) == 2
    for row in event_rows:
        wall_ms = float(row["wall_latency_ms"])
        trace_ms = float(row["semantic_trace_io_latency_ms"])
        adjusted_ms = float(row["insertion_latency_excluding_trace_io_ms"])
        assert int(row["semantic_trace_bytes_written"]) > 0
        assert 0 <= trace_ms <= wall_ms
        assert adjusted_ms == pytest.approx(max(0.0, wall_ms - trace_ms), abs=0.002)
    assert int(case_row["semantic_trace_bytes_written"]) == sum(
        int(row["semantic_trace_bytes_written"]) for row in event_rows
    )
    assert summary["semantic_trace_bytes_written"] == int(
        case_row["semantic_trace_bytes_written"]
    )
    assert summary["semantic_trace_io_wall_latency"]["mean_ms"] is not None
    assert summary["insertion_wall_latency_excluding_trace_io"]["mean_ms"] is not None


def test_operation_usage_separates_calls_batches_and_provider_responses(
    tmp_path: Path,
) -> None:
    class ProviderTracingDriver(_Driver):
        trace_dir: Path

        def retrieve(self, request):
            for batch_id, size in (("batch-1", 3), ("batch-2", 2), ("batch-3", 1)):
                for item_index in range(size):
                    write_trace_event(
                        self.trace_dir,
                        operator="sem_topk",
                        event_type="provider_usage",
                        payload={
                            "operator_call_id": "logical-1",
                            "provider_batch_id": batch_id,
                            "provider_item_index": item_index,
                            "provider_usage_available": True,
                            "provider_prompt_tokens": 10,
                            "provider_prompt_cache_hit_tokens": 6,
                            "provider_prompt_cache_miss_tokens": 4,
                            "provider_completion_tokens": 2,
                            "model": "deepseek-v4-flash",
                        },
                    )
            return super().retrieve(request)

    def factory(case_id: str, state_dir: Path, trace_dir: Path) -> _Driver:
        del case_id
        driver = ProviderTracingDriver(state_dir)
        driver.trace_dir = trace_dir
        return driver

    case = replace(_bundle().cases[0], questions=(_bundle().cases[0].questions[0],))
    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    ).run(replace(_bundle(), cases=(case,)))

    with (tmp_path / "metrics" / "operation_usage.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    row = next(
        item
        for item in rows
        if item["phase"] == "retrieval" and item["operator"] == "sem_topk"
    )
    assert row["logical_call_count"] == "1"
    assert row["physical_batch_count"] == "3"
    assert row["provider_call_count"] == "6"
    assert row["physical_item_count"] == "6"
    assert row["items_per_batch"] == "2.0"
    assert row["retry_provider_call_count"] == "0"
    assert row["retry_batch_count"] == "0"
    assert row["prompt_tokens"] == "60"
    assert row["completion_tokens"] == "12"
    assert row["estimated_cost_usd"] == "6.8208e-06"


def test_session_metrics_separate_insertion_and_consolidation_provider_usage(
    tmp_path: Path,
) -> None:
    class SessionTracingDriver(_Driver):
        trace_dir: Path

        def _trace_usage(self, batch_id: str) -> None:
            write_trace_event(
                self.trace_dir,
                operator="memory",
                event_type="provider_usage",
                payload={
                    "provider_batch_id": batch_id,
                    "provider_usage_available": True,
                    "provider_prompt_tokens": 10,
                    "provider_prompt_cache_hit_tokens": 6,
                    "provider_prompt_cache_miss_tokens": 4,
                    "provider_completion_tokens": 2,
                    "model": "deepseek-v4-flash",
                },
            )

        def add(self, event: BenchmarkEvent) -> dict[str, int]:
            self._trace_usage(f"insert-{event.event_id}")
            return super().add(event)

        def finish_session(self, session_id: str) -> dict[str, str]:
            self._trace_usage(f"consolidate-{session_id}")
            return super().finish_session(session_id)

    def factory(case_id: str, state_dir: Path, trace_dir: Path) -> _Driver:
        del case_id
        driver = SessionTracingDriver(state_dir)
        driver.trace_dir = trace_dir
        return driver

    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    ).run(_bundle())

    with (tmp_path / "metrics" / "per_event.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        event_rows = list(csv.DictReader(stream))
    with (tmp_path / "metrics" / "per_session.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        session_rows = list(csv.DictReader(stream))

    assert [row["provider_call_count"] for row in event_rows] == ["1", "1"]
    assert len(session_rows) == 1
    assert session_rows[0]["insertion_provider_call_count"] == "2"
    assert session_rows[0]["consolidation_provider_call_count"] == "1"
    assert session_rows[0]["provider_call_count"] == "3"


def test_default_condition_id_uses_maintenance_and_retrieval_contract() -> None:
    quick = replace(
        _system_contract(),
        maintenance_rule="rule-join-map",
        retrieval_recipe_id="semantic-topk:pairwise-quick",
    )
    listwise = replace(
        quick,
        retrieval_recipe_id="semantic-topk:listwise",
    )
    regroup = replace(quick, maintenance_rule="rule-re-group")

    assert quick.effective_condition_id != listwise.effective_condition_id
    assert quick.effective_condition_id != regroup.effective_condition_id
    assert replace(quick, condition_id="JM-Q").effective_condition_id == "JM-Q"


def test_completed_case_resume_does_not_recreate_driver_or_model(tmp_path) -> None:
    factory_calls = 0

    def factory(case_id, state_dir, trace_dir):
        nonlocal factory_calls
        assert case_id == "case-1"
        del trace_dir
        factory_calls += 1
        return _Driver(state_dir)

    model = _Model()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )
    runner.run(_bundle())
    runner.run(_bundle())
    assert factory_calls == 1
    assert len(model.calls) == 2


def test_bundle_run_mode_is_written_to_manifest(tmp_path) -> None:
    bundle = replace(_bundle(), metadata={"run_mode": "integration-smoke"})
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(bundle)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_mode"] == "integration-smoke"


def test_resume_rejects_changed_semantic_trace_snapshot_mode(tmp_path: Path) -> None:
    def provenance(mode: str) -> dict[str, object]:
        return {
            "source": {"commit": "source", "dirty": False},
            "runtime": {
                "python": "3.12",
                "lotus_execution": {
                    "semantic_trace_snapshot_mode": mode,
                },
            },
        }

    first = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
        runtime_provenance=provenance("compact"),
    )
    first.run(_bundle())

    changed = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
        runtime_provenance=provenance("full"),
    )
    with pytest.raises(ValueError, match="different run contract"):
        changed.run(_bundle())


def test_retrieval_system_error_scores_zero_and_completed_case_is_skipped(
    tmp_path: Path,
) -> None:
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        assert case_id == "case-1"
        del trace_dir
        driver = _Driver(state_dir, fail_retrieval=True)
        drivers.append(driver)
        return driver

    model = _Model()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={
            "task-1": replace(_contract(), memory_system_error_score=0.0)
        },
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )
    runner.run(_bundle())
    runner.run(_bundle())

    assert len(drivers) == 1
    assert drivers[0].closed
    assert drivers[0].queries == ["query:first?"]
    assert model.calls == []

    case_dir = tmp_path / "cases" / "case-1-ba225b98"
    status = json.loads((case_dir / "status.json").read_text())
    retrievals = [
        json.loads(line)
        for line in (case_dir / "retrieval.jsonl").read_text().splitlines()
    ]
    answers = [
        json.loads(line)
        for line in (case_dir / "answers.jsonl").read_text().splitlines()
    ]
    grades = [
        json.loads(line)
        for line in (case_dir / "grades.jsonl").read_text().splitlines()
    ]
    assert status["status"] == "completed"
    assert [row["status"] for row in retrievals] == [
        "system_error",
        "system_error",
    ]
    assert [row["answer"] for row in answers] == ["", ""]
    assert [row["score"] for row in grades] == [0.0, 0.0]
    assert [row["label"] for row in grades] == ["system_error", "system_error"]
    assert all(
        row["details"]["grade_source"] == "benchmark_failure_policy"
        for row in grades
    )

    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    assert summary["completed_cases"] == 1
    assert summary["failed_cases"] == 0
    assert summary["question_count"] == 2
    assert summary["mean_score"] == 0.0
    assert summary["retrieval_system_error_count"] == 2
    assert summary["memory_system_error_case_count"] == 1
    assert summary["memory_system_error_question_count"] == 2
    with (tmp_path / "metrics" / "per_question.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        question_rows = list(csv.DictReader(handle))
    assert [row["retrieval_status"] for row in question_rows] == [
        "system_error",
        "system_error",
    ]
    assert [row["retrieval_error_type"] for row in question_rows] == [
        "RuntimeError",
        "RuntimeError",
    ]
    with (tmp_path / "metrics" / "reliability.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reliability = list(csv.DictReader(handle))
    assert reliability[0]["retrieval_system_error_count"] == "2"
    assert reliability[0]["memory_system_error_case_count"] == "1"
    assert reliability[0]["memory_system_error_question_count"] == "2"


def test_retryable_retrieval_error_retries_before_zero_score_policy(
    tmp_path: Path,
) -> None:
    drivers: list[_Driver] = []

    class TimeoutRetrieval(_Driver):
        def retrieve(self, request):
            assert self.queries is not None
            self.queries.append(request.query_text)
            raise TimeoutError("temporary retrieval timeout")

    def factory(case_id, state_dir, trace_dir):
        del case_id, trace_dir
        driver = TimeoutRetrieval(state_dir) if not drivers else _Driver(state_dir)
        drivers.append(driver)
        return driver

    model = _Model()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={
            "task-1": replace(_contract(), memory_system_error_score=0.0)
        },
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(TimeoutError, match="temporary retrieval timeout"):
        runner.run(_bundle())
    runner.run(_bundle())

    assert len(drivers) == 2
    assert drivers[1].restored_event_ids == ("event-1", "event-2")
    grades = [
        json.loads(line)
        for line in (
            tmp_path / "cases" / "case-1-ba225b98" / "grades.jsonl"
        ).read_text().splitlines()
    ]
    assert [row["score"] for row in grades] == [1.0, 1.0]
    assert all(row["label"] != "system_error" for row in grades)


def test_retrieval_system_error_does_not_stop_later_cases(tmp_path: Path) -> None:
    first = replace(_bundle().cases[0], questions=(_bundle().cases[0].questions[0],))
    second = BenchmarkCase(
        case_id="case-2",
        task_id="task-1",
        events=(
            BenchmarkEvent(
                sample_id="case-2",
                event_id="event-3",
                speaker="user",
                text="remember three",
            ),
        ),
        questions=(
            BenchmarkQuestion("q3", "case-2", "second?", "two", (), "test"),
        ),
    )
    bundle = BenchmarkBundle("benchmark", "revision", "sha", (first, second))
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        del trace_dir
        driver = _Driver(state_dir, fail_retrieval=case_id == "case-1")
        drivers.append(driver)
        return driver

    model = _Model(["two"])
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={
            "task-1": replace(_contract(), memory_system_error_score=0.0)
        },
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(bundle)

    assert len(drivers) == 2
    assert all(driver.closed for driver in drivers)
    assert model.calls == [("answer", 1)]
    statuses = [
        json.loads(path.read_text())["status"]
        for path in sorted((tmp_path / "cases").glob("*/status.json"))
    ]
    assert statuses == ["completed", "completed"]
    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    assert summary["completed_cases"] == 2
    assert summary["failed_cases"] == 0
    assert summary["question_count"] == 2
    assert summary["retrieval_system_error_count"] == 1
    assert summary["mean_score"] == 0.5


def test_case_scoped_retrieval_failure_skips_all_answering(
    tmp_path: Path,
) -> None:
    class SecondRetrievalFails(_Driver):
        def retrieve(self, request):
            assert self.queries is not None
            if len(self.queries) == 1:
                raise RuntimeError("second retrieval failed")
            return super().retrieve(request)

    driver: SecondRetrievalFails | None = None

    def factory(case_id, state_dir, trace_dir):
        nonlocal driver
        del case_id, trace_dir
        driver = SecondRetrievalFails(state_dir)
        return driver

    model = _Model()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={
            "task-1": replace(_contract(), memory_system_error_score=0.0)
        },
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(_bundle())

    assert driver is not None
    assert driver.queries == ["query:first?"]
    assert model.calls == []
    case_dir = tmp_path / "cases" / "case-1-ba225b98"
    grades = [
        json.loads(line)
        for line in (case_dir / "grades.jsonl").read_text().splitlines()
    ]
    assert [row["score"] for row in grades] == [0.0, 0.0]
    assert {row["details"]["failed_phase"] for row in grades} == {"retrieval"}


def test_declared_insertion_system_error_scores_case_zero_and_completes(
    tmp_path: Path,
) -> None:
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        del case_id, trace_dir
        driver = _Driver(state_dir, fail_event_id="event-2")
        drivers.append(driver)
        return driver

    contract = replace(
        _contract(),
        checkpoint_boundary="event",
        memory_system_error_score=0.0,
    )
    model = _Model()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": contract},
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(_bundle())
    runner.run(_bundle())

    assert len(drivers) == 1
    assert drivers[0].events == ["event-1"]
    assert drivers[0].queries == []
    assert model.calls == []
    case_dir = tmp_path / "cases" / "case-1-ba225b98"
    status = json.loads((case_dir / "status.json").read_text())
    grades = [
        json.loads(line)
        for line in (case_dir / "grades.jsonl").read_text().splitlines()
    ]
    assert status["status"] == "completed"
    assert [row["score"] for row in grades] == [0.0, 0.0]
    assert {row["details"]["failed_phase"] for row in grades} == {"insertion"}
    current = json.loads(
        (case_dir / "checkpoints" / "current.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (
            case_dir
            / "checkpoints"
            / "snapshots"
            / current["checkpoint_id"]
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["completed_event_ids"] == ["event-1"]


def test_event_checkpoint_boundary_is_independent_of_session_boundary(
    tmp_path: Path,
) -> None:
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        del case_id, trace_dir
        driver = _Driver(
            state_dir,
            fail_event_id="event-2" if not drivers else None,
        )
        drivers.append(driver)
        return driver

    contract = replace(_contract(), checkpoint_boundary="event")
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": contract},
        driver_factory=factory,
        answer_model=_Model(["one", "two"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(RuntimeError, match="failed to add event-2"):
        runner.run(_bundle())
    runner.run(_bundle())

    assert drivers[1].restored_event_ids == ("event-1",)
    assert drivers[1].events == ["event-1", "event-2"]


def test_declared_consolidation_error_scores_zero_without_answering(
    tmp_path: Path,
) -> None:
    model = _Model()
    contract = replace(_contract(), memory_system_error_score=0.0)
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": contract},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(
            state_dir,
            fail_finish=True,
        ),
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(_bundle())

    grades = [
        json.loads(line)
        for line in (
            tmp_path / "cases" / "case-1-ba225b98" / "grades.jsonl"
        ).read_text().splitlines()
    ]
    assert {row["details"]["failed_phase"] for row in grades} == {
        "consolidation"
    }
    assert model.calls == []


def test_control_signal_is_never_converted_to_system_error(tmp_path: Path) -> None:
    class InterruptingDriver(_Driver):
        def add(self, event: BenchmarkEvent) -> dict[str, int]:
            del event
            raise KeyboardInterrupt

    contract = replace(_contract(), memory_system_error_score=0.0)
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": contract},
        driver_factory=lambda case_id, state_dir, trace_dir: InterruptingDriver(
            state_dir
        ),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(KeyboardInterrupt):
        runner.run(_bundle())


def test_invalid_answer_retries_only_current_model_step(tmp_path) -> None:
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        assert case_id == "case-1"
        del trace_dir
        driver = _Driver(state_dir)
        drivers.append(driver)
        return driver

    model = _Model(["", "one", "two"])
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )
    runner.run(_bundle())
    assert drivers[0].events == ["event-1", "event-2"]
    assert drivers[0].queries == ["query:first?", "query:second?"]
    assert model.calls == [("answer", 1), ("answer", 2), ("answer", 1)]


def test_runner_uses_separate_answer_and_judge_models(tmp_path: Path) -> None:
    answer_model = _Model(["one", "two"])
    judge_model = _Model(["yes", "yes"])
    answer_model.model_id = "answer-model"
    judge_model.model_id = "judge-model"

    def judge_plan(question: BenchmarkQuestion, answer: str):
        del question, answer
        return (
            JudgeStep(
                prompt=ModelPrompt(
                    prompt_name="judge",
                    messages=({"role": "user", "content": "judge"},),
                    prompt_digest="judge-digest",
                ),
                parse=lambda value: value.strip() == "yes",
            ),
        )

    judged_contract = TaskContract(
        task_id="task-1",
        answer_prompt=_prompt,
        answer_parser=_parse_answer,
        retrieval_query=lambda question: f"query:{question.question}",
        answer_prompt_digest="answer-digest",
        scorer_id="judge",
        scorer_digest="judge-digest",
        judge_plan=judge_plan,
        judge_reducer=lambda question, answer, results: GradeResult(
            scorer_id="judge",
            score=float(results[0] is True),
        ),
    )
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": judged_contract},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=answer_model,
        judge_model=judge_model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(_bundle())

    assert answer_model.calls == [("answer", 1), ("answer", 1)]
    assert judge_model.calls == [("judge", 1), ("judge", 1)]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["memory_model_id"] == "memory-model"
    assert manifest["answer_model_id"] == "answer-model"
    assert manifest["judge_model_id"] == "judge-model"


def test_runner_scopes_driver_work_by_benchmark_phase(tmp_path: Path) -> None:
    scopes: list[dict[str, object]] = []

    class ScopedDriver(_Driver):
        def add(self, event: BenchmarkEvent) -> dict[str, int]:
            scopes.append(active_trace_scope())
            return super().add(event)

        def retrieve(self, request):
            scopes.append(active_trace_scope())
            return super().retrieve(request)

    model = _Model()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: ScopedDriver(state_dir),
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )
    runner.run(_bundle())

    assert [scope["phase"] for scope in scopes] == [
        "insertion",
        "insertion",
        "retrieval",
        "retrieval",
    ]
    assert all(scope["case_id"] == "case-1" for scope in scopes)


def test_model_failure_is_traced_before_case_failure(tmp_path: Path) -> None:
    class FailingModel:
        model_id = "failing-model"

        def complete(self, prompt: ModelPrompt, *, attempt: int) -> ModelResponse:
            del prompt, attempt
            raise RuntimeError("provider unavailable")

    model = FailingModel()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        runner.run(_bundle())

    events = [
        json.loads(line)
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
    ]
    error = next(row for row in events if row["event_type"] == "llm_batch_error")
    assert error["phase"] == "answering"
    assert error["question_id"] == "q1"
    assert error["error_type"] == "RuntimeError"


def test_retryable_provider_failure_retries_only_the_current_step(tmp_path: Path) -> None:
    class RetryableModel(_Model):
        def __init__(self) -> None:
            super().__init__(["one", "two"])
            self.attempts = 0

        def complete(self, prompt: ModelPrompt, *, attempt: int) -> ModelResponse:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary provider failure")
            return super().complete(prompt, attempt=attempt)

        @staticmethod
        def is_retryable_error(error: BaseException) -> bool:
            return isinstance(error, RuntimeError)

    model = RetryableModel()
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(_bundle())

    assert model.attempts == 3
    events = [
        json.loads(line)
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
    ]
    attempts = [
        row["attempt"]
        for row in events
        if row["event_type"] in {"llm_batch_error", "llm_call"}
        and row.get("phase") == "answering"
    ]
    assert attempts[:2] == [1, 2]


def test_task_contract_requires_one_scoring_path() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TaskContract(
            task_id="invalid",
            answer_prompt=_prompt,
            answer_parser=_parse_answer,
            retrieval_query=lambda question: question.question,
            answer_prompt_digest="answer",
            scorer_id="none",
            scorer_digest="none",
        )


def _session_bundle() -> BenchmarkBundle:
    case = BenchmarkCase(
        case_id="case-1",
        task_id="task-1",
        events=tuple(
            BenchmarkEvent(
                sample_id="case-1",
                event_id=f"event-{index}",
                speaker="user",
                text=f"remember {index}",
                session_id="session-1" if index <= 2 else "session-2",
            )
            for index in range(1, 5)
        ),
        questions=(BenchmarkQuestion("q1", "case-1", "first?", "one", ()),),
    )
    return BenchmarkBundle("benchmark", "revision", "sha", (case,))


def test_session_checkpoint_resumes_without_readding_completed_session(
    tmp_path: Path,
) -> None:
    drivers: list[_Driver] = []

    def factory(case_id, state_dir, trace_dir):
        del case_id, trace_dir
        driver = _Driver(
            state_dir,
            fail_event_id="event-4" if not drivers else None,
        )
        drivers.append(driver)
        return driver

    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(RuntimeError, match="failed to add event-4"):
        runner.run(_session_bundle())

    current = json.loads(
        (tmp_path / "cases" / "case-1-ba225b98" / "checkpoints" / "current.json")
        .read_text(encoding="utf-8")
    )
    checkpoint_dir = (
        tmp_path
        / "cases"
        / "case-1-ba225b98"
        / "checkpoints"
        / "snapshots"
        / current["checkpoint_id"]
    )
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    assert manifest["completed_event_ids"] == ["event-1", "event-2"]
    assert manifest["completed_session_id"] == "session-1"

    runner.run(_session_bundle())

    assert drivers[1].restored_event_ids == ("event-1", "event-2")
    assert drivers[1].events == ["event-1", "event-2", "event-3", "event-4"]
    insertion_rows = [
        json.loads(line)
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
        if '"event_type": "insertion_result"' in line
    ]
    assert [row["event_id"] for row in insertion_rows].count("event-1") == 1
    assert [row["event_id"] for row in insertion_rows].count("event-2") == 1
    assert [row["event_id"] for row in insertion_rows].count("event-3") == 2


def test_checkpoint_manifest_reconciles_ledger_after_publish_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drivers: list[_Driver] = []
    original = UnitAttemptStore.reconcile_many
    crashed = False

    def crash_after_checkpoint(self, lineages):
        nonlocal crashed
        values = tuple(lineages)
        if values and not crashed:
            crashed = True
            raise SystemExit("simulated crash after checkpoint publish")
        return original(self, values)

    monkeypatch.setattr(UnitAttemptStore, "reconcile_many", crash_after_checkpoint)
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: drivers.append(
            _Driver(state_dir)
        )
        or drivers[-1],
        answer_model=_Model(["one", "two"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(SystemExit, match="checkpoint publish"):
        runner.run(_bundle())

    monkeypatch.setattr(UnitAttemptStore, "reconcile_many", original)
    runner.run(_bundle())

    assert drivers[1].restored_event_ids == ("event-1", "event-2")
    insertion_rows = [
        json.loads(line)
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
        if '"event_type": "insertion_result"' in line
    ]
    assert [row["event_id"] for row in insertion_rows] == [
        "event-1",
        "event-2",
    ]


def test_atomic_question_result_reconciles_without_repeating_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drivers: list[_Driver] = []
    model = _Model(["one", "two"])
    original = UnitAttemptStore.reconcile_success
    crashed = False

    def crash_after_question(self, lineage):
        nonlocal crashed
        if lineage[:2] == ("question", "q1") and not crashed:
            crashed = True
            raise SystemExit("simulated crash after question publish")
        return original(self, lineage)

    monkeypatch.setattr(
        UnitAttemptStore,
        "reconcile_success",
        crash_after_question,
    )
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: drivers.append(
            _Driver(state_dir)
        )
        or drivers[-1],
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(SystemExit, match="question publish"):
        runner.run(_bundle())

    monkeypatch.setattr(UnitAttemptStore, "reconcile_success", original)
    runner.run(_bundle())

    assert drivers[1].queries == ["query:second?"]
    assert model.calls == [("answer", 1), ("answer", 1)]


def test_runner_does_not_checkpoint_a_system_without_checkpoint_support(
    tmp_path: Path,
) -> None:
    class NoCheckpointDriver(_Driver):
        def save_state(self, directory: Path) -> dict[str, Any]:
            del directory
            raise AssertionError("save_state must not be called")

    contract = MemorySystemContract(
        system_id="fake",
        memory_model_id="memory-model",
        memory_provider_model_id="provider-memory-model",
        input_adapter_id="fake-input:v1",
        retrieval_recipe_id="fake-retrieval:v1",
        checkpoint_enabled=False,
    )
    runner = BenchmarkRunner(
        system_contract=contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: NoCheckpointDriver(
            state_dir
        ),
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    runner.run(_session_bundle())

    assert not (
        tmp_path / "cases" / "case-1-ba225b98" / "checkpoints"
    ).exists()


def test_maintenance_checkpoint_requires_declared_checkpoint_support(
    tmp_path: Path,
) -> None:
    contract = MemorySystemContract(
        system_id="fake",
        memory_model_id="memory-model",
        memory_provider_model_id="provider-memory-model",
        input_adapter_id="fake-input:v1",
        retrieval_recipe_id="fake-retrieval:v1",
        checkpoint_enabled=False,
    )

    with pytest.raises(ValueError, match="checkpoint support"):
        BenchmarkRunner(
            system_contract=contract,
            contracts={"task-1": _contract()},
            driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
            answer_model=_Model(),
            judge_model=_Model(),
            artifacts=BenchmarkArtifactStore(tmp_path),
            maintenance_only=True,
        )


def test_checkpoint_pointer_is_not_published_when_driver_save_fails(
    tmp_path: Path,
) -> None:
    class FailingCheckpointDriver(_Driver):
        saves = 0

        def save_state(self, directory: Path) -> dict[str, Any]:
            self.saves += 1
            if self.saves == 2:
                raise RuntimeError("checkpoint save failed")
            return super().save_state(directory)

    driver = FailingCheckpointDriver(tmp_path / "state")
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: driver,
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )

    with pytest.raises(RuntimeError, match="checkpoint save failed"):
        runner.run(_session_bundle())

    pointer = json.loads(
        (tmp_path / "cases" / "case-1-ba225b98" / "checkpoints" / "current.json")
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            tmp_path
            / "cases"
            / "case-1-ba225b98"
            / "checkpoints"
            / "snapshots"
            / pointer["checkpoint_id"]
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["completed_event_ids"] == ["event-1", "event-2"]


def test_session_checkpoint_rejects_changed_event_prefix(tmp_path: Path) -> None:
    driver = _Driver(tmp_path / "state", fail_event_id="event-4")
    runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: driver,
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path),
    )
    with pytest.raises(RuntimeError):
        runner.run(_session_bundle())

    original = _session_bundle()
    case = original.cases[0]
    changed = BenchmarkBundle(
        original.benchmark_id,
        original.dataset_revision,
        original.dataset_sha256,
        (
            BenchmarkCase(
                case.case_id,
                case.task_id,
                (
                    BenchmarkEvent(
                        sample_id="case-1",
                        event_id="event-1",
                        speaker="user",
                        text="changed evidence",
                        session_id="session-1",
                    ),
                    *case.events[1:],
                ),
                case.questions,
            ),
        ),
    )

    with pytest.raises(ValueError, match="different run contract|event prefix"):
        runner.run(changed)


def test_final_maintenance_checkpoint_is_reused_read_only_for_retrieval(
    tmp_path: Path,
) -> None:
    bundle = _session_bundle()
    source_dir = tmp_path / "maintenance"
    source_drivers: list[_Driver] = []
    source_runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: source_drivers.append(
            _Driver(state_dir)
        )
        or source_drivers[-1],
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(source_dir),
        maintenance_only=True,
    )
    source_runner.run(bundle)

    assert source_drivers[0].events == [
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    ]
    assert source_drivers[0].queries == []
    pointer_path = (
        source_dir
        / "cases"
        / "case-1-ba225b98"
        / "checkpoints"
        / "current.json"
    )
    pointer_before = pointer_path.read_bytes()

    retrieval_drivers: list[_Driver] = []
    retrieval_runner = BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: retrieval_drivers.append(
            _Driver(state_dir)
        )
        or retrieval_drivers[-1],
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path / "retrieval"),
        maintenance_checkpoint_source=BenchmarkArtifactStore(source_dir),
    )
    retrieval_runner.run(bundle)

    assert retrieval_drivers[0].restored_event_ids == (
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    )
    assert retrieval_drivers[0].queries == ["query:first?"]
    assert pointer_path.read_bytes() == pointer_before


def test_checkpoint_restore_propagates_trace_phase_to_driver(
    tmp_path: Path,
) -> None:
    bundle = _session_bundle()
    maintenance_dir = tmp_path / "maintenance"
    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(state_dir),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(maintenance_dir),
        maintenance_only=True,
    ).run(bundle)

    class RestoreTracingDriver(_Driver):
        trace_dir: Path

        def restore_state(
            self,
            directory: Path,
            completed_events: tuple[BenchmarkEvent, ...],
        ) -> None:
            write_trace_event(
                self.trace_dir,
                operator="embedding",
                event_type="embedding_call",
                payload={
                    "status": "success",
                    "model": "BAAI/bge-m3",
                    "revision": "revision",
                    "batch_size": 1,
                    "dimensions": 1024,
                    "latency_ms": 1.0,
                    "input_path": "trace/prompts/restore.json",
                },
            )
            super().restore_state(directory, completed_events)

    def factory(case_id: str, state_dir: Path, trace_dir: Path) -> _Driver:
        del case_id
        driver = RestoreTracingDriver(state_dir)
        driver.trace_dir = trace_dir
        return driver

    output_dir = tmp_path / "retrieval"
    BenchmarkRunner(
        system_contract=_system_contract(),
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(output_dir),
        maintenance_checkpoint_source=BenchmarkArtifactStore(maintenance_dir),
    ).run(bundle)

    with (output_dir / "metrics" / "embedding_usage.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["phase"] == "checkpoint"
    assert rows[0]["operation"] == "restore"


def test_explicit_maintenance_policy_id_allows_cross_system_checkpoint_reuse(
    tmp_path: Path,
) -> None:
    bundle = _session_bundle()
    base_contract = replace(
        _system_contract(),
        system_id="mem0-memory",
        condition_id="AM-Mem0-Maintenance",
        maintenance_policy_id="mem0-memory",
        input_adapter_id="benchmark-event-to-mem0-message:v1",
        retrieval_recipe_id="mem0-base-bge-m3-cosine:v1",
        maintenance_rule="mem0-additive-view:v1",
    )
    enhanced_contract = replace(
        base_contract,
        system_id="mem0-enhanced",
        condition_id="AM-Mem0-Enhanced",
        retrieval_recipe_id="mem0-enhanced-sem-topk:v1:pairwise-quick",
    )
    assert base_contract.maintenance_fingerprint == (
        enhanced_contract.maintenance_fingerprint
    )

    maintenance_dir = tmp_path / "maintenance"
    BenchmarkRunner(
        system_contract=base_contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(
            state_dir,
            system_id="mem0-memory",
        ),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(maintenance_dir),
        maintenance_only=True,
    ).run(bundle)

    retrieval_drivers: list[_Driver] = []
    BenchmarkRunner(
        system_contract=enhanced_contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: (
            retrieval_drivers.append(
                _Driver(state_dir, system_id="mem0-enhanced")
            )
            or retrieval_drivers[-1]
        ),
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path / "enhanced"),
        maintenance_checkpoint_source=BenchmarkArtifactStore(maintenance_dir),
    ).run(bundle)

    assert retrieval_drivers[0].restored_event_ids == (
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    )
    assert retrieval_drivers[0].queries == ["query:first?"]


def test_cross_system_checkpoint_without_shared_maintenance_identity_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = _session_bundle()
    source_contract = replace(_system_contract(), system_id="source")
    source_dir = tmp_path / "source"
    BenchmarkRunner(
        system_contract=source_contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(
            state_dir,
            system_id="source",
        ),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(source_dir),
        maintenance_only=True,
    ).run(bundle)

    target_contract = replace(source_contract, system_id="target")
    runner = BenchmarkRunner(
        system_contract=target_contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: _Driver(
            state_dir,
            system_id="target",
        ),
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path / "target"),
        maintenance_checkpoint_source=BenchmarkArtifactStore(source_dir),
    )

    with pytest.raises(ValueError, match="maintenance_policy_id"):
        runner.run(bundle)


def test_zep_maintenance_checkpoint_restore_does_not_replay_events(
    tmp_path: Path,
) -> None:
    bundle = _session_bundle()
    contract = replace(
        _system_contract(),
        system_id="zep-memory",
        input_adapter_id="benchmark-event-to-zep-log:v1",
        retrieval_recipe_id="zep-memory-entity-rrf-fact-bfs-cross-encoder:v1",
        maintenance_rule="rule-re-group",
    )
    maintenance_dir = tmp_path / "maintenance"
    maintenance_drivers: list[_Driver] = []
    BenchmarkRunner(
        system_contract=contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: (
            maintenance_drivers.append(
                _Driver(state_dir, system_id="zep-memory")
            )
            or maintenance_drivers[-1]
        ),
        answer_model=_Model(),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(maintenance_dir),
        maintenance_only=True,
    ).run(bundle)

    retrieval_drivers: list[_Driver] = []
    BenchmarkRunner(
        system_contract=contract,
        contracts={"task-1": _contract()},
        driver_factory=lambda case_id, state_dir, trace_dir: (
            retrieval_drivers.append(_Driver(state_dir, system_id="zep-memory"))
            or retrieval_drivers[-1]
        ),
        answer_model=_Model(["one"]),
        judge_model=_Model(),
        artifacts=BenchmarkArtifactStore(tmp_path / "retrieval"),
        maintenance_checkpoint_source=BenchmarkArtifactStore(maintenance_dir),
    ).run(bundle)

    assert maintenance_drivers[0].events == [
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    ]
    assert retrieval_drivers[0].restored_event_ids == (
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    )
    assert retrieval_drivers[0].events == [
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    ]
    assert retrieval_drivers[0].queries == ["query:first?"]


def test_maintenance_fingerprint_separates_rules_but_not_retrieval() -> None:
    join_map = replace(
        _system_contract(),
        maintenance_rule="rule-join-map",
        retrieval_recipe_id="claude-memory:pairwise-quick",
    )
    join_map_listwise = replace(
        join_map,
        retrieval_recipe_id="claude-memory:listwise",
    )
    re_group = replace(join_map, maintenance_rule="rule-re-group")
    preferred = replace(join_map, maintenance_rule="prefer-join-map")

    assert join_map.maintenance_fingerprint == join_map_listwise.maintenance_fingerprint
    assert join_map.maintenance_fingerprint != re_group.maintenance_fingerprint
    assert preferred.maintenance_fingerprint != join_map.maintenance_fingerprint
    assert preferred.maintenance_fingerprint != re_group.maintenance_fingerprint
