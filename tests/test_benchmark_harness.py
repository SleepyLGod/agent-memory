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
    GradeResult,
    JudgeStep,
    MemorySystemContract,
    ModelPrompt,
    ModelResponse,
    RetrievalOutput,
    TaskContract,
)
from agent_memory.evaluation.types import BenchmarkCase, BenchmarkEvent, BenchmarkQuestion
from agent_memory.tracing.semantic import active_trace_scope


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


@dataclass
class _Driver:
    state_dir: Path
    system_id: str = "fake"
    events: list[str] | None = None
    queries: list[str] | None = None
    closed: bool = False
    fail_retrieval: bool = False
    fail_event_id: str | None = None
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
    assert drivers[0].closed
    assert model.calls == [("answer", 1), ("answer", 1)]
    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset_revision"] == "revision"
    assert manifest["case_ids"] == ["case-1"]
    assert manifest["question_ids"] == ["q1", "q2"]
    assert manifest["answer_prompt_digests"] == {"task-1": "answer-digest"}
    assert manifest["scorer_contracts"] == {
        "task-1": {"scorer_id": "exact", "scorer_digest": "score-digest"}
    }
    assert summary["completed_cases"] == 1
    assert summary["failed_cases"] == 0
    assert summary["mean_score"] == 1.0
    assert summary["provider_call_count"] == 2
    assert summary["provider_error_count"] == 0
    assert summary["question_count"] == 2
    assert summary["estimated_cost_usd"] == 1.12e-06
    assert summary["phases"]["answering"]["prompt_tokens"] == 4
    for filename in (
        "overview.csv",
        "per_event.csv",
        "per_session.csv",
        "per_question.csv",
        "provider_usage.csv",
        "operator_usage.csv",
        "state_shape.csv",
        "reliability.csv",
        "checkpoint_metrics.csv",
    ):
        assert (tmp_path / "metrics" / filename).is_file()
    assert len((tmp_path / "metrics" / "per_event.csv").read_text().splitlines()) == 3
    assert len(
        (tmp_path / "metrics" / "checkpoint_metrics.csv").read_text().splitlines()
    ) == 2


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
        contracts={"task-1": _contract()},
        driver_factory=factory,
        answer_model=model,
        judge_model=model,
        artifacts=BenchmarkArtifactStore(tmp_path),
    )
    runner.run(_bundle())
    runner.run(_bundle())

    assert len(drivers) == 1
    assert drivers[0].closed
    assert drivers[0].queries == ["query:first?", "query:second?"]
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
        contracts={"task-1": _contract()},
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

    assert join_map.maintenance_fingerprint == join_map_listwise.maintenance_fingerprint
    assert join_map.maintenance_fingerprint != re_group.maintenance_fingerprint
