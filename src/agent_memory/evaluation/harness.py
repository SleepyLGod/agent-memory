"""Benchmark-neutral case runner and execution contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from agent_memory.evaluation.artifacts import BenchmarkArtifactStore
from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
    RetrievalRequest,
)
from agent_memory.tracing.semantic import semantic_trace_scope


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelPrompt:
    """One answer or judge provider request before retry metadata is added."""

    prompt_name: str
    messages: tuple[Mapping[str, str], ...]
    prompt_digest: str
    temperature: float = 0
    max_tokens: int = 8192
    thinking_enabled: bool = True


@dataclass(frozen=True)
class ModelResponse:
    """One real provider response with raw usage evidence."""

    model: str
    text: str
    raw_response: Any
    usage: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = 0


@dataclass(frozen=True)
class MemorySystemContract:
    """Versioned memory input and retrieval contract for one system."""

    system_id: str
    memory_model_id: str
    memory_provider_model_id: str
    input_adapter_id: str
    retrieval_recipe_id: str
    condition_id: str = ""
    maintenance_rule: str = ""
    thinking_enabled: bool | None = None
    consolidation_mode: str = "none"
    parser_mode: str = ""
    framework_cache_mode: str = "unspecified"
    checkpoint_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "system_id",
            "memory_model_id",
            "memory_provider_model_id",
            "input_adapter_id",
            "retrieval_recipe_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.checkpoint_enabled, bool):
            raise TypeError("checkpoint_enabled must be a bool")
        if self.condition_id and not isinstance(self.condition_id, str):
            raise TypeError("condition_id must be a string")
        if self.thinking_enabled is not None and not isinstance(
            self.thinking_enabled, bool
        ):
            raise TypeError("thinking_enabled must be a bool or None")
        for field_name in (
            "maintenance_rule",
            "consolidation_mode",
            "parser_mode",
            "framework_cache_mode",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")

    @property
    def effective_condition_id(self) -> str:
        """Return the unique experiment identity used in reports."""

        return self.condition_id or self.system_id

    @property
    def maintenance_fingerprint(self) -> str:
        """Fingerprint only settings that can affect materialized memory state."""

        return _digest(
            {
                "system_id": self.system_id,
                "memory_model_id": self.memory_model_id,
                "memory_provider_model_id": self.memory_provider_model_id,
                "input_adapter_id": self.input_adapter_id,
                "maintenance_rule": self.maintenance_rule,
                "thinking_enabled": self.thinking_enabled,
                "consolidation_mode": self.consolidation_mode,
                "framework_cache_mode": self.framework_cache_mode,
            }
        )

    @property
    def input_adapter_digest(self) -> str:
        """Return the stable digest of the versioned input adapter contract."""

        return sha256(self.input_adapter_id.encode("utf-8")).hexdigest()

    @property
    def retrieval_recipe_digest(self) -> str:
        """Return the stable digest of the versioned retrieval recipe contract."""

        return sha256(self.retrieval_recipe_id.encode("utf-8")).hexdigest()


class BenchmarkModel(Protocol):
    """Provider boundary used for answering and LLM-based scoring."""

    model_id: str

    def complete(self, prompt: ModelPrompt, *, attempt: int) -> ModelResponse:
        """Execute one provider attempt."""

        ...


@dataclass(frozen=True)
class RetrievalOutput:
    """System retrieval result consumed by the shared answerer."""

    context: str
    channels: Mapping[str, Sequence[Mapping[str, Any]]]
    metrics: Mapping[str, Any] = field(default_factory=dict)


class MemorySystemDriver(Protocol):
    """One isolated case-specific memory system instance."""

    system_id: str

    def add(self, event: BenchmarkEvent) -> Mapping[str, Any]:
        """Append one normalized event and return compact metrics."""

        ...

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        """Retrieve context for one question without answering it."""

        ...

    def finish_session(self, session_id: str) -> Mapping[str, Any]:
        """Finish maintenance that must run exactly once per input session."""

        ...

    def save_state(self, directory: Path) -> Mapping[str, Any]:
        """Write opaque driver state into a new checkpoint directory."""

        ...

    def restore_state(
        self,
        directory: Path,
        completed_events: tuple[BenchmarkEvent, ...],
    ) -> None:
        """Restore opaque state and the completed canonical input prefix."""

        ...

    def close(self) -> None:
        """Release case-local resources."""

        ...


@dataclass(frozen=True)
class GradeResult:
    """One normalized deterministic or LLM-based grade."""

    scorer_id: str
    score: float
    label: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeStep:
    """One independently retryable request in an LLM scoring plan."""

    prompt: ModelPrompt
    parse: Callable[[str], Any]


@dataclass(frozen=True)
class TaskContract:
    """Benchmark task behavior independent of any memory system."""

    task_id: str
    answer_prompt: Callable[[BenchmarkQuestion, str], ModelPrompt]
    answer_parser: Callable[[str], str]
    retrieval_query: Callable[[BenchmarkQuestion], str]
    answer_prompt_digest: str
    scorer_id: str
    scorer_digest: str
    deterministic_scorer: Callable[[BenchmarkQuestion, str], GradeResult] | None = None
    judge_plan: Callable[[BenchmarkQuestion, str], tuple[JudgeStep, ...]] | None = None
    judge_reducer: Callable[
        [BenchmarkQuestion, str, tuple[Any, ...]], GradeResult
    ] | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        deterministic = self.deterministic_scorer is not None
        judged = self.judge_plan is not None or self.judge_reducer is not None
        if deterministic == judged:
            raise ValueError(
                "TaskContract requires exactly one deterministic or LLM judge path"
            )
        if judged and (self.judge_plan is None or self.judge_reducer is None):
            raise ValueError("LLM judge contracts require both plan and reducer")

    @property
    def fingerprint(self) -> str:
        """Fingerprint the declared prompt/scorer contract, not Python callables."""

        return _digest(
            {
                "task_id": self.task_id,
                "answer_prompt_digest": self.answer_prompt_digest,
                "scorer_id": self.scorer_id,
                "scorer_digest": self.scorer_digest,
            }
        )


DriverFactory = Callable[[str, Path, Path], MemorySystemDriver]


class BenchmarkRunner:
    """Run isolated cases without benchmark- or system-specific branches."""

    def __init__(
        self,
        *,
        system_contract: MemorySystemContract,
        contracts: Mapping[str, TaskContract],
        driver_factory: DriverFactory,
        answer_model: BenchmarkModel,
        judge_model: BenchmarkModel,
        artifacts: BenchmarkArtifactStore,
        parse_attempts: int = 4,
        maintenance_only: bool = False,
        maintenance_checkpoint_source: BenchmarkArtifactStore | None = None,
        max_new_cases: int | None = None,
    ) -> None:
        if parse_attempts < 1:
            raise ValueError("parse_attempts must be positive")
        if not isinstance(system_contract, MemorySystemContract):
            raise TypeError("system_contract must be a MemorySystemContract")
        if max_new_cases is not None and max_new_cases < 1:
            raise ValueError("max_new_cases must be positive")
        if (
            maintenance_only or maintenance_checkpoint_source is not None
        ) and not system_contract.checkpoint_enabled:
            raise ValueError(
                "maintenance checkpoint execution requires declared checkpoint support"
            )
        self.system_contract = system_contract
        self.system_id = system_contract.system_id
        self.contracts = dict(contracts)
        self.driver_factory = driver_factory
        self.answer_model = answer_model
        self.judge_model = judge_model
        self.artifacts = artifacts
        self.parse_attempts = parse_attempts
        self.maintenance_only = maintenance_only
        self.maintenance_checkpoint_source = maintenance_checkpoint_source
        self.max_new_cases = max_new_cases

    def run(self, bundle: BenchmarkBundle) -> None:
        """Run incomplete cases and rebuild aggregate metrics."""

        required_task_ids = {case.task_id for case in bundle.cases}
        missing = required_task_ids - set(self.contracts)
        if missing:
            raise ValueError(f"missing task contracts: {sorted(missing)}")
        selected_contracts = {
            task_id: self.contracts[task_id].fingerprint
            for task_id in sorted(required_task_ids)
        }
        answer_prompt_digests = {
            task_id: self.contracts[task_id].answer_prompt_digest
            for task_id in sorted(required_task_ids)
        }
        scorer_contracts = {
            task_id: {
                "scorer_id": self.contracts[task_id].scorer_id,
                "scorer_digest": self.contracts[task_id].scorer_digest,
            }
            for task_id in sorted(required_task_ids)
        }
        bundle_run_mode = bundle.metadata.get("run_mode")
        if bundle_run_mode is not None and not isinstance(bundle_run_mode, str):
            raise TypeError("benchmark bundle run_mode metadata must be a string")
        run_mode = bundle_run_mode or (
            "maintenance" if self.maintenance_only else "full"
        )
        if self.maintenance_only and bundle_run_mode:
            run_mode = f"{bundle_run_mode}-maintenance"
        self.artifacts.initialize(
            bundle=bundle,
            system_contract=self.system_contract,
            answer_model_id=self.answer_model.model_id,
            judge_model_id=self.judge_model.model_id,
            contract_fingerprints=selected_contracts,
            answer_prompt_digests=answer_prompt_digests,
            scorer_contracts=scorer_contracts,
            run_mode=run_mode,
            maintenance_checkpoint_source=(
                str(self.maintenance_checkpoint_source.output_dir.resolve())
                if self.maintenance_checkpoint_source is not None
                else None
            ),
        )

        first_error: BaseException | None = None
        new_cases = 0
        for case in bundle.cases:
            contract = self.contracts[case.task_id]
            already_complete = (
                self.artifacts.maintenance_completed(case, contract.fingerprint)
                if self.maintenance_only
                else self.artifacts.completed(case, contract.fingerprint)
            )
            if already_complete:
                continue
            if self.max_new_cases is not None and new_cases >= self.max_new_cases:
                break
            try:
                self._run_case(bundle, case, contract)
                new_cases += 1
            except BaseException as error:
                self.artifacts.fail_case(case, contract.fingerprint, error)
                first_error = error
                break
        self.artifacts.finalize_metrics()
        if first_error is not None:
            raise first_error

    def _run_case(
        self,
        bundle: BenchmarkBundle,
        case: BenchmarkCase,
        contract: TaskContract,
    ) -> None:
        state_dir = self.artifacts.start_case(case, contract.fingerprint)
        attempt = int(state_dir.name.removeprefix("attempt-"))
        # Drivers receive only an opaque case identity, never questions or gold labels.
        driver = self.driver_factory(case.case_id, state_dir, self.artifacts.trace_dir)
        if driver.system_id != self.system_id:
            raise ValueError(
                f"driver system_id {driver.system_id!r} does not match run "
                f"system_id {self.system_id!r}"
            )
        primary_error: BaseException | None = None
        try:
            checkpoint = None
            if self.system_contract.checkpoint_enabled:
                checkpoint = self.artifacts.load_checkpoint(
                    bundle=bundle,
                    case=case,
                    system_contract=self.system_contract,
                )
            if checkpoint is None and self.maintenance_checkpoint_source is not None:
                checkpoint = self.maintenance_checkpoint_source.load_checkpoint(
                    bundle=bundle,
                    case=case,
                    system_contract=self.system_contract,
                )
                if checkpoint is None:
                    raise ValueError(
                        f"maintenance checkpoint source has no state for {case.case_id!r}"
                    )
                if len(checkpoint.completed_events) != len(case.events):
                    raise ValueError(
                        "maintenance checkpoint source must contain the full case input"
                    )
            completed_count = 0
            if checkpoint is not None:
                restore_started = perf_counter()
                driver.restore_state(
                    checkpoint.directory / "driver",
                    checkpoint.completed_events,
                )
                completed_count = len(checkpoint.completed_events)
                self.artifacts.trace_stage(
                    phase="checkpoint",
                    case_id=case.case_id,
                    payload={
                        "operation": "restore",
                        "checkpoint_id": checkpoint.directory.name,
                        "completed_event_count": completed_count,
                        "latency_ms": round(
                            (perf_counter() - restore_started) * 1000, 3
                        ),
                    },
                )

            for event_index, event in enumerate(
                case.events[completed_count:],
                start=completed_count,
            ):
                started = perf_counter()
                with semantic_trace_scope(
                    phase="insertion",
                    case_id=case.case_id,
                    event_id=event.event_id,
                    attempt=attempt,
                ):
                    metrics = driver.add(event)
                self.artifacts.trace_stage(
                    phase="insertion",
                    case_id=case.case_id,
                    payload={
                        "event_id": event.event_id,
                        "session_id": event.session_id,
                        "attempt": attempt,
                        "latency_ms": round((perf_counter() - started) * 1000, 3),
                        **dict(metrics),
                    },
                )
                next_event = (
                    case.events[event_index + 1]
                    if event_index + 1 < len(case.events)
                    else None
                )
                if self.system_contract.checkpoint_enabled and (
                    next_event is None or next_event.session_id != event.session_id
                ):
                    finish_started = perf_counter()
                    finish_metrics = driver.finish_session(event.session_id)
                    if finish_metrics:
                        self.artifacts.trace_stage(
                            phase="consolidation",
                            case_id=case.case_id,
                            payload={
                                "session_id": event.session_id,
                                "completed_event_count": event_index + 1,
                                "latency_ms": round(
                                    (perf_counter() - finish_started) * 1000, 3
                                ),
                                **dict(finish_metrics),
                            },
                        )
                    checkpoint_started = perf_counter()
                    checkpoint = self.artifacts.save_checkpoint(
                        bundle=bundle,
                        case=case,
                        system_contract=self.system_contract,
                        completed_events=case.events[: event_index + 1],
                        driver=driver,
                    )
                    self.artifacts.trace_stage(
                        phase="checkpoint",
                        case_id=case.case_id,
                        payload={
                            "operation": "save",
                            "session_id": event.session_id,
                            "checkpoint_id": checkpoint.directory.name,
                            "completed_event_count": event_index + 1,
                            "checkpoint_bytes": sum(
                                path.stat().st_size
                                for path in checkpoint.directory.rglob("*")
                                if path.is_file()
                            ),
                            "latency_ms": round(
                                (perf_counter() - checkpoint_started) * 1000, 3
                            ),
                        },
                    )
            if self.maintenance_only:
                self.artifacts.complete_maintenance(case, contract.fingerprint)
                return
            for question in case.questions:
                self._run_question(case, question, contract, driver)
            self.artifacts.complete_case(case, contract.fingerprint)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                driver.close()
            except BaseException:
                if primary_error is None:
                    raise

    def _run_question(
        self,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        contract: TaskContract,
        driver: MemorySystemDriver,
    ) -> None:
        query_text = contract.retrieval_query(question)
        started = perf_counter()
        try:
            with semantic_trace_scope(
                phase="retrieval",
                case_id=case.case_id,
                question_id=question.question_id,
            ):
                retrieval = driver.retrieve(
                    RetrievalRequest(
                        question_id=question.question_id,
                        query_text=query_text,
                    )
                )
        except Exception as error:
            self._record_retrieval_system_error(
                case=case,
                question=question,
                contract=contract,
                query_text=query_text,
                error=error,
                latency_ms=(perf_counter() - started) * 1000,
            )
            return
        retrieval_latency = (perf_counter() - started) * 1000
        self.artifacts.append_case_row(
            case.case_id,
            "retrieval",
            {
                "question_id": question.question_id,
                "query": query_text,
                "context": retrieval.context,
                "channels": retrieval.channels,
                "metrics": retrieval.metrics,
                "latency_ms": round(retrieval_latency, 3),
            },
        )
        self.artifacts.trace_stage(
            phase="retrieval",
            case_id=case.case_id,
            payload={
                "question_id": question.question_id,
                "latency_ms": round(retrieval_latency, 3),
                "channels": {
                    name: len(rows) for name, rows in retrieval.channels.items()
                },
            },
        )

        answer_prompt = contract.answer_prompt(question, retrieval.context)
        answer_started = perf_counter()
        answer = self._complete_and_parse(
            case=case,
            question=question,
            prompt=answer_prompt,
            phase="answering",
            parse=contract.answer_parser,
            model=self.answer_model,
        )
        answer_latency = (perf_counter() - answer_started) * 1000
        self.artifacts.append_case_row(
            case.case_id,
            "answers",
            {
                "question_id": question.question_id,
                "answer": answer,
                "latency_ms": round(answer_latency, 3),
            },
        )

        grade_started = perf_counter()
        if contract.deterministic_scorer is not None:
            grade = contract.deterministic_scorer(question, answer)
        else:
            assert contract.judge_plan is not None
            assert contract.judge_reducer is not None
            parsed_steps = tuple(
                self._complete_and_parse(
                    case=case,
                    question=question,
                    prompt=step.prompt,
                    phase="grading",
                    parse=step.parse,
                    model=self.judge_model,
                )
                for step in contract.judge_plan(question, answer)
            )
            grade = contract.judge_reducer(question, answer, parsed_steps)
        self.artifacts.append_case_row(
            case.case_id,
            "grades",
            {
                "question_id": question.question_id,
                "scorer_id": grade.scorer_id,
                "score": grade.score,
                "label": grade.label,
                "details": grade.details,
                "latency_ms": round((perf_counter() - grade_started) * 1000, 3),
            },
        )

    def _record_retrieval_system_error(
        self,
        *,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        contract: TaskContract,
        query_text: str,
        error: Exception,
        latency_ms: float,
    ) -> None:
        """Record a terminal zero score when the tested retrieval system fails."""

        error_type = type(error).__name__
        error_message = str(error)
        rounded_latency = round(latency_ms, 3)
        self.artifacts.append_case_row(
            case.case_id,
            "retrieval",
            {
                "question_id": question.question_id,
                "query": query_text,
                "status": "system_error",
                "context": "",
                "channels": {},
                "metrics": {
                    "status": "system_error",
                    "error_type": error_type,
                    "error": error_message,
                },
                "error_type": error_type,
                "error": error_message,
                "latency_ms": rounded_latency,
            },
        )
        self.artifacts.append_case_row(
            case.case_id,
            "answers",
            {
                "question_id": question.question_id,
                "status": "skipped_due_to_retrieval_error",
                "answer": "",
                "latency_ms": None,
            },
        )
        self.artifacts.append_case_row(
            case.case_id,
            "grades",
            {
                "question_id": question.question_id,
                "scorer_id": contract.scorer_id,
                "score": 0.0,
                "label": "system_error",
                "details": {
                    "grade_source": "benchmark_failure_policy",
                    "failed_phase": "retrieval",
                    "error_type": error_type,
                    "error": error_message,
                },
                "latency_ms": None,
            },
        )
        self.artifacts.trace_stage(
            phase="retrieval",
            case_id=case.case_id,
            payload={
                "question_id": question.question_id,
                "status": "system_error",
                "error_type": error_type,
                "error": error_message,
                "latency_ms": rounded_latency,
                "channels": {},
            },
        )

    def _complete_and_parse(
        self,
        *,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        prompt: ModelPrompt,
        phase: str,
        parse: Callable[[str], Any],
        model: BenchmarkModel,
    ) -> Any:
        last_error: ValueError | None = None
        for attempt in range(1, self.parse_attempts + 1):
            started = perf_counter()
            try:
                response = model.complete(prompt, attempt=attempt)
            except BaseException as error:
                self.artifacts.trace_model_error(
                    phase=phase,
                    case_id=case.case_id,
                    question_id=question.question_id,
                    prompt_name=prompt.prompt_name,
                    prompt_digest=prompt.prompt_digest,
                    attempt=attempt,
                    model=model.model_id,
                    messages=prompt.messages,
                    kwargs={
                        "temperature": prompt.temperature,
                        "max_tokens": prompt.max_tokens,
                        "thinking": {
                            "type": "enabled" if prompt.thinking_enabled else "disabled"
                        },
                    },
                    latency_ms=(perf_counter() - started) * 1000,
                    error=error,
                )
                retryable = getattr(model, "is_retryable_error", None)
                if (
                    attempt < self.parse_attempts
                    and callable(retryable)
                    and retryable(error)
                ):
                    continue
                raise
            self.artifacts.trace_model_call(
                phase=phase,
                case_id=case.case_id,
                question_id=question.question_id,
                prompt_name=prompt.prompt_name,
                prompt_digest=prompt.prompt_digest,
                attempt=attempt,
                model=response.model,
                messages=prompt.messages,
                kwargs={
                    "temperature": prompt.temperature,
                    "max_tokens": prompt.max_tokens,
                    "thinking": {
                        "type": "enabled" if prompt.thinking_enabled else "disabled"
                    },
                },
                raw_output=response.raw_response,
                latency_ms=response.latency_ms,
                usage=response.usage,
            )
            try:
                return parse(response.text)
            except ValueError as error:
                last_error = error
        assert last_error is not None
        raise last_error


__all__ = [
    "BenchmarkModel",
    "BenchmarkRunner",
    "GradeResult",
    "JudgeStep",
    "MemorySystemContract",
    "MemorySystemDriver",
    "ModelPrompt",
    "ModelResponse",
    "RetrievalOutput",
    "TaskContract",
]
