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
from agent_memory.evaluation.recovery import (
    ArtifactContractError,
    AttemptLineage,
    UnitAttemptExhausted,
    is_retryable_unit_error,
)
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
    RetrievalRequest,
)
from agent_memory.tracing.semantic import (
    measure_semantic_trace_io,
    semantic_trace_scope,
)


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
    maintenance_policy_id: str = ""
    maintenance_rule: str = ""
    maintenance_execution_id: str = ""
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
        if self.maintenance_policy_id and not isinstance(
            self.maintenance_policy_id, str
        ):
            raise TypeError("maintenance_policy_id must be a string")
        if self.thinking_enabled is not None and not isinstance(
            self.thinking_enabled, bool
        ):
            raise TypeError("thinking_enabled must be a bool or None")
        for field_name in (
            "maintenance_rule",
            "maintenance_execution_id",
            "consolidation_mode",
            "parser_mode",
            "framework_cache_mode",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")

    @property
    def effective_condition_id(self) -> str:
        """Return the unique experiment identity used in reports."""

        if self.condition_id:
            identity = self.condition_id
        else:
            maintenance = self.maintenance_rule or "none"
            identity = (
                f"{self.system_id}|maintenance={maintenance}"
                f"|retrieval={self.retrieval_recipe_id}"
            )
        if self.maintenance_execution_id:
            return f"{identity}|execution={self.maintenance_execution_id}"
        return identity

    @property
    def effective_maintenance_policy_id(self) -> str:
        """Return the explicit compatibility identity for maintenance state."""

        return self.maintenance_policy_id or self.system_id

    @property
    def maintenance_fingerprint(self) -> str:
        """Fingerprint only settings that can affect materialized memory state."""

        contract = {
            # Keep this key for stable fingerprints of existing contracts.
            "system_id": self.effective_maintenance_policy_id,
            "memory_model_id": self.memory_model_id,
            "memory_provider_model_id": self.memory_provider_model_id,
            "input_adapter_id": self.input_adapter_id,
            "maintenance_rule": self.maintenance_rule,
            "thinking_enabled": self.thinking_enabled,
            "consolidation_mode": self.consolidation_mode,
            "framework_cache_mode": self.framework_cache_mode,
        }
        if self.maintenance_execution_id:
            contract["maintenance_execution_id"] = self.maintenance_execution_id
        return _digest(contract)

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
class GradeContract:
    """One independently recorded scorer applied to a generated answer."""

    scorer_id: str
    scorer_digest: str
    deterministic_scorer: Callable[[BenchmarkQuestion, str], GradeResult] | None = None
    judge_plan: Callable[[BenchmarkQuestion, str], tuple[JudgeStep, ...]] | None = None
    judge_reducer: Callable[
        [BenchmarkQuestion, str, tuple[Any, ...]], GradeResult
    ] | None = None
    applies_to: Callable[[BenchmarkQuestion], bool] | None = None

    def __post_init__(self) -> None:
        if not self.scorer_id or not self.scorer_digest:
            raise ValueError("grade scorer ID and digest must be non-empty")
        deterministic = self.deterministic_scorer is not None
        judged = self.judge_plan is not None or self.judge_reducer is not None
        if deterministic == judged:
            raise ValueError(
                "GradeContract requires exactly one deterministic or LLM judge path"
            )
        if judged and (self.judge_plan is None or self.judge_reducer is None):
            raise ValueError("LLM grade contracts require both plan and reducer")

    def applies(self, question: BenchmarkQuestion) -> bool:
        """Return whether this scorer evaluates the question."""

        return self.applies_to is None or self.applies_to(question)

    @property
    def fingerprint_payload(self) -> Mapping[str, str]:
        """Return the stable declared scorer contract."""

        return {
            "scorer_id": self.scorer_id,
            "scorer_digest": self.scorer_digest,
        }


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
    checkpoint_boundary: str = "session"
    memory_system_error_score: float | None = None
    deterministic_scorer: Callable[[BenchmarkQuestion, str], GradeResult] | None = None
    judge_plan: Callable[[BenchmarkQuestion, str], tuple[JudgeStep, ...]] | None = None
    judge_reducer: Callable[
        [BenchmarkQuestion, str, tuple[Any, ...]], GradeResult
    ] | None = None
    additional_graders: tuple[GradeContract, ...] = ()
    answer_parser_id: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not isinstance(self.answer_parser_id, str):
            raise TypeError("answer_parser_id must be a string")
        if self.checkpoint_boundary not in {"session", "event"}:
            raise ValueError("checkpoint_boundary must be 'session' or 'event'")
        if self.memory_system_error_score is not None and not 0 <= self.memory_system_error_score <= 1:
            raise ValueError("memory_system_error_score must be between 0 and 1")
        deterministic = self.deterministic_scorer is not None
        judged = self.judge_plan is not None or self.judge_reducer is not None
        if deterministic == judged:
            raise ValueError(
                "TaskContract requires exactly one deterministic or LLM judge path"
            )
        if judged and (self.judge_plan is None or self.judge_reducer is None):
            raise ValueError("LLM judge contracts require both plan and reducer")
        if any(not isinstance(grader, GradeContract) for grader in self.additional_graders):
            raise TypeError("additional graders must be GradeContract values")
        scorer_ids = [grader.scorer_id for grader in self.graders]
        if len(scorer_ids) != len(set(scorer_ids)):
            raise ValueError("task scorer IDs must be unique")

    @property
    def graders(self) -> tuple[GradeContract, ...]:
        """Return the primary scorer followed by optional secondary scorers."""

        primary = GradeContract(
            scorer_id=self.scorer_id,
            scorer_digest=self.scorer_digest,
            deterministic_scorer=self.deterministic_scorer,
            judge_plan=self.judge_plan,
            judge_reducer=self.judge_reducer,
        )
        return (primary, *self.additional_graders)

    @property
    def fingerprint(self) -> str:
        """Fingerprint the declared prompt/scorer contract, not Python callables."""

        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "answer_prompt_digest": self.answer_prompt_digest,
            "scorer_id": self.scorer_id,
            "scorer_digest": self.scorer_digest,
            "additional_graders": [
                dict(grader.fingerprint_payload)
                for grader in self.additional_graders
            ],
            "checkpoint_boundary": self.checkpoint_boundary,
            "memory_system_error_score": self.memory_system_error_score,
        }
        if self.answer_parser_id:
            payload["answer_parser_id"] = self.answer_parser_id
        return _digest(payload)


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
        runtime_provenance: Mapping[str, Any] | None = None,
        storage_provenance: Mapping[str, Any] | None = None,
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
        self.runtime_provenance = dict(runtime_provenance or {})
        self.storage_provenance = (
            dict(storage_provenance) if storage_provenance is not None else None
        )
        self._execution_attempt = 0
        self._unit_attempt = 0

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
        answer_parser_contracts = {
            task_id: self.contracts[task_id].answer_parser_id
            for task_id in sorted(required_task_ids)
            if self.contracts[task_id].answer_parser_id
        }
        scorer_contracts = {
            task_id: {
                "scorer_id": self.contracts[task_id].scorer_id,
                "scorer_digest": self.contracts[task_id].scorer_digest,
                "additional_graders": [
                    dict(grader.fingerprint_payload)
                    for grader in self.contracts[task_id].additional_graders
                ],
            }
            for task_id in sorted(required_task_ids)
        }
        bundle_run_mode = bundle.metadata.get("run_mode")
        if bundle_run_mode is not None and (
            not isinstance(bundle_run_mode, str) or not bundle_run_mode
        ):
            raise TypeError(
                "benchmark bundle run_mode metadata must be a non-empty string"
            )
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
            answer_parser_contracts=answer_parser_contracts or None,
            scorer_contracts=scorer_contracts,
            runtime_provenance=self.runtime_provenance,
            storage_provenance=self.storage_provenance,
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
                if isinstance(error, UnitAttemptExhausted):
                    self.artifacts.attention_case(
                        case,
                        contract.fingerprint,
                        error,
                    )
                else:
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
        self._execution_attempt = attempt
        self._unit_attempt = 0
        # Drivers receive only an opaque case identity, never questions or gold labels.
        setup_started = perf_counter()
        try:
            driver = self.driver_factory(
                case.case_id,
                state_dir,
                self.artifacts.trace_dir,
            )
        except BaseException as error:
            setup_latency_ms = round((perf_counter() - setup_started) * 1000, 3)
            self.artifacts.trace_stage(
                phase="driver_setup",
                case_id=case.case_id,
                payload={
                    "attempt": attempt,
                    "status": "error",
                    "latency_ms": setup_latency_ms,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            self.artifacts.fail_case(
                case,
                contract.fingerprint,
                error,
                details={
                    "failure_phase": "driver_setup",
                    "requested_strategy": self.system_contract.maintenance_rule,
                    "policy_id": self.system_id,
                    "add_count": 0,
                    "storage_transaction_count": 0,
                },
            )
            raise
        self.artifacts.trace_stage(
            phase="driver_setup",
            case_id=case.case_id,
            payload={
                "attempt": attempt,
                "status": "success",
                "latency_ms": round((perf_counter() - setup_started) * 1000, 3),
            },
        )
        if driver.system_id != self.system_id:
            raise ValueError(
                f"driver system_id {driver.system_id!r} does not match run "
                f"system_id {self.system_id!r}"
            )
        primary_error: BaseException | None = None
        try:
            checkpoint = None
            checkpoint_is_local = False
            if self.system_contract.checkpoint_enabled:
                checkpoint = self.artifacts.load_checkpoint(
                    bundle=bundle,
                    case=case,
                    system_contract=self.system_contract,
                )
                checkpoint_is_local = checkpoint is not None
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
            committed_attempts: list[AttemptLineage] = []
            pending_attempts: list[AttemptLineage] = []
            attempt_store = self.artifacts.unit_attempt_store(case.case_id)
            if checkpoint is not None:
                restore_started = perf_counter()
                with semantic_trace_scope(
                    phase="checkpoint",
                    case_id=case.case_id,
                    attempt=attempt,
                    operation="restore",
                ):
                    driver.restore_state(
                        checkpoint.directory / "driver",
                        checkpoint.completed_events,
                )
                completed_count = len(checkpoint.completed_events)
                committed_attempts.extend(checkpoint.event_attempts)
                if checkpoint_is_local:
                    attempt_store.reconcile_many(checkpoint.event_attempts)
                self.artifacts.trace_stage(
                    phase="checkpoint",
                    case_id=case.case_id,
                    payload={
                        "operation": "restore",
                        "checkpoint_id": checkpoint.directory.name,
                        "completed_event_count": completed_count,
                        "execution_attempt": attempt,
                        "unit_attempt": 0,
                        "latency_ms": round(
                            (perf_counter() - restore_started) * 1000, 3
                        ),
                    },
                )

            for event_index, event in enumerate(
                case.events[completed_count:],
                start=completed_count,
            ):
                unit_attempt = self.artifacts.begin_unit_attempt(
                    case=case,
                    phase="insertion",
                    unit_id=event.event_id,
                    execution_attempt=attempt,
                )
                self._unit_attempt = unit_attempt
                started = perf_counter()
                operation_phase = "insertion"
                insertion_recorded = False
                trace_io_latency_ms = 0.0
                trace_bytes_written = 0
                trace_io = None
                try:
                    with measure_semantic_trace_io() as trace_io:
                        with semantic_trace_scope(
                            phase="insertion",
                            case_id=case.case_id,
                            event_id=event.event_id,
                            session_id=event.session_id,
                            attempt=attempt,
                            execution_attempt=attempt,
                            unit_attempt=unit_attempt,
                        ):
                            metrics = driver.add(event)
                    trace_io_latency_ms = round(trace_io.latency_ms, 3)
                    trace_bytes_written = trace_io.bytes_written
                    insertion_latency_ms = round(
                        (perf_counter() - started) * 1000,
                        3,
                    )
                    self.artifacts.trace_stage(
                        phase="insertion",
                        case_id=case.case_id,
                        payload={
                            "event_id": event.event_id,
                            "session_id": event.session_id,
                            "attempt": attempt,
                            "execution_attempt": attempt,
                            "unit_attempt": unit_attempt,
                            "latency_ms": insertion_latency_ms,
                            "semantic_trace_io_latency_ms": trace_io_latency_ms,
                            "semantic_trace_bytes_written": trace_bytes_written,
                            "insertion_latency_excluding_trace_io_ms": round(
                                max(
                                    0.0,
                                    insertion_latency_ms - trace_io_latency_ms,
                                ),
                                3,
                            ),
                            **dict(metrics),
                        },
                    )
                    insertion_recorded = True
                    lineage: AttemptLineage = (
                        "insertion",
                        event.event_id,
                        attempt,
                        unit_attempt,
                    )
                    pending_attempts.append(lineage)
                    next_event = (
                        case.events[event_index + 1]
                        if event_index + 1 < len(case.events)
                        else None
                    )
                    session_finished = (
                        next_event is None
                        or next_event.session_id != event.session_id
                    )
                    if session_finished:
                        finish_started = perf_counter()
                        operation_phase = "consolidation"
                        with semantic_trace_scope(
                            phase="consolidation",
                            case_id=case.case_id,
                            event_id=event.event_id,
                            session_id=event.session_id,
                            attempt=attempt,
                            execution_attempt=attempt,
                            unit_attempt=unit_attempt,
                        ):
                            finish_metrics = driver.finish_session(
                                event.session_id
                            )
                        if finish_metrics:
                            self.artifacts.trace_stage(
                                phase="consolidation",
                                case_id=case.case_id,
                                payload={
                                    "event_id": event.event_id,
                                    "session_id": event.session_id,
                                    "completed_event_count": event_index + 1,
                                    "execution_attempt": attempt,
                                    "unit_attempt": unit_attempt,
                                    "latency_ms": round(
                                        (perf_counter() - finish_started) * 1000,
                                        3,
                                    ),
                                    **dict(finish_metrics),
                                },
                            )
                    checkpoint_boundary = (
                        self.system_contract.checkpoint_enabled
                        and (
                            contract.checkpoint_boundary == "event"
                            or session_finished
                        )
                    )
                    if checkpoint_boundary:
                        checkpoint_started = perf_counter()
                        operation_phase = "checkpoint"
                        checkpoint = self.artifacts.save_checkpoint(
                            bundle=bundle,
                            case=case,
                            system_contract=self.system_contract,
                            completed_events=case.events[: event_index + 1],
                            event_attempts=tuple(
                                committed_attempts + pending_attempts
                            ),
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
                                "execution_attempt": attempt,
                                "unit_attempt": unit_attempt,
                                "checkpoint_bytes": sum(
                                    path.stat().st_size
                                    for path in checkpoint.directory.rglob("*")
                                    if path.is_file()
                                ),
                                "latency_ms": round(
                                    (perf_counter() - checkpoint_started) * 1000,
                                    3,
                                ),
                            },
                        )
                        attempt_store.reconcile_many(pending_attempts)
                        committed_attempts.extend(pending_attempts)
                        pending_attempts.clear()
                    elif not self.system_contract.checkpoint_enabled:
                        attempt_store.reconcile_many(pending_attempts)
                        committed_attempts.extend(pending_attempts)
                        pending_attempts.clear()
                except ArtifactContractError:
                    raise
                except Exception as error:
                    if trace_io is not None:
                        trace_io_latency_ms = round(trace_io.latency_ms, 3)
                        trace_bytes_written = trace_io.bytes_written
                    retryable = is_retryable_unit_error(error)
                    if not insertion_recorded:
                        insertion_latency_ms = round(
                            (perf_counter() - started) * 1000,
                            3,
                        )
                        self.artifacts.trace_stage(
                            phase="insertion",
                            case_id=case.case_id,
                            payload={
                                "event_id": event.event_id,
                                "session_id": event.session_id,
                                "attempt": attempt,
                                "execution_attempt": attempt,
                                "unit_attempt": unit_attempt,
                                "status": "error",
                                "latency_ms": insertion_latency_ms,
                                "insertion_latency_excluding_trace_io_ms": round(
                                    max(
                                        0.0,
                                        insertion_latency_ms
                                        - trace_io_latency_ms,
                                    ),
                                    3,
                                ),
                                "semantic_trace_io_latency_ms": (
                                    trace_io_latency_ms
                                ),
                                "semantic_trace_bytes_written": (
                                    trace_bytes_written
                                ),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )
                    elif operation_phase in {"consolidation", "checkpoint"}:
                        self.artifacts.trace_stage(
                            phase=operation_phase,
                            case_id=case.case_id,
                            payload={
                                "event_id": event.event_id,
                                "session_id": event.session_id,
                                "operation": (
                                    "save"
                                    if operation_phase == "checkpoint"
                                    else "finish_session"
                                ),
                                "execution_attempt": attempt,
                                "unit_attempt": unit_attempt,
                                "status": "error",
                                "latency_ms": None,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )
                    exhausted = False
                    if retryable:
                        exhausted = self.artifacts.finish_unit_attempt(
                            case=case,
                            phase="insertion",
                            unit_id=event.event_id,
                            execution_attempt=attempt,
                            unit_attempt=unit_attempt,
                            status="failed",
                            error=error,
                        )
                        if exhausted and contract.memory_system_error_score is None:
                            raise UnitAttemptExhausted(
                                f"insertion:{event.event_id} exhausted "
                                f"{unit_attempt} attempts"
                            ) from error
                        if not exhausted:
                            raise
                    if contract.memory_system_error_score is None:
                        raise
                    self._record_case_system_error(
                        case=case,
                        contract=contract,
                        phase=operation_phase,
                        error=error,
                        latency_ms=(perf_counter() - started) * 1000,
                        event_id=event.event_id,
                    )
                    self.artifacts.complete_case(case, contract.fingerprint)
                    return
            if self.maintenance_only:
                self.artifacts.complete_maintenance(case, contract.fingerprint)
                return
            if contract.memory_system_error_score is not None:
                prepared_retrievals: list[
                    tuple[BenchmarkQuestion, str, RetrievalOutput, float, AttemptLineage]
                ] = []
                for question in case.questions:
                    unit_attempt = self.artifacts.begin_unit_attempt(
                        case=case,
                        phase="question",
                        unit_id=question.question_id,
                        execution_attempt=attempt,
                    )
                    self._unit_attempt = unit_attempt
                    started = perf_counter()
                    try:
                        prepared_retrievals.append(
                            (
                                question,
                                *self._retrieve_question(case, question, contract, driver),
                                ("question", question.question_id, attempt, unit_attempt),
                            )
                        )
                    except ArtifactContractError:
                        raise
                    except Exception as error:
                        if is_retryable_unit_error(error):
                            exhausted = self.artifacts.finish_unit_attempt(
                                case=case,
                                phase="question",
                                unit_id=question.question_id,
                                execution_attempt=attempt,
                                unit_attempt=unit_attempt,
                                status="failed",
                                error=error,
                            )
                            if not exhausted:
                                raise
                        self._record_case_system_error(
                            case=case,
                            contract=contract,
                            phase="retrieval",
                            error=error,
                            latency_ms=(perf_counter() - started) * 1000,
                            event_id=None,
                        )
                        self.artifacts.complete_case(case, contract.fingerprint)
                        return
                for (
                    question,
                    query_text,
                    retrieval,
                    latency_ms,
                    question_lineage,
                ) in prepared_retrievals:
                    self._run_question(
                        case,
                        question,
                        contract,
                        driver,
                        prepared_retrieval=(query_text, retrieval, latency_ms),
                    )
                    attempt_store.reconcile_success(question_lineage)
            else:
                for question in case.questions:
                    scorer_ids = [
                        grader.scorer_id
                        for grader in contract.graders
                        if grader.applies(question)
                    ]
                    if self.artifacts.question_completed(
                        case=case,
                        question=question,
                        contract_fingerprint=contract.fingerprint,
                        scorer_ids=scorer_ids,
                    ):
                        question_lineage = self.artifacts.question_store(
                            case.case_id
                        ).lineage(question.question_id)
                        if question_lineage is not None:
                            attempt_store.reconcile_success(question_lineage)
                        continue
                    unit_attempt = self.artifacts.begin_unit_attempt(
                        case=case,
                        phase="question",
                        unit_id=question.question_id,
                        execution_attempt=attempt,
                    )
                    self._unit_attempt = unit_attempt
                    try:
                        with semantic_trace_scope(
                            case_id=case.case_id,
                            question_id=question.question_id,
                            execution_attempt=attempt,
                            unit_attempt=unit_attempt,
                        ):
                            self._run_question(
                                case,
                                question,
                                contract,
                                driver,
                            )
                    except ArtifactContractError:
                        raise
                    except Exception as error:
                        if is_retryable_unit_error(error):
                            exhausted = self.artifacts.finish_unit_attempt(
                                case=case,
                                phase="question",
                                unit_id=question.question_id,
                                execution_attempt=attempt,
                                unit_attempt=unit_attempt,
                                status="failed",
                                error=error,
                            )
                            if exhausted:
                                raise UnitAttemptExhausted(
                                    f"question:{question.question_id} exhausted "
                                    f"{unit_attempt} attempts"
                                ) from error
                        raise
                    question_lineage = self.artifacts.question_store(
                        case.case_id
                    ).lineage(question.question_id)
                    if question_lineage is None:
                        raise ArtifactContractError(
                            "question completed without atomic result evidence"
                        )
                    attempt_store.reconcile_success(question_lineage)
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
        *,
        prepared_retrieval: tuple[str, RetrievalOutput, float] | None = None,
    ) -> None:
        if prepared_retrieval is None:
            started = perf_counter()
            try:
                query_text, retrieval, retrieval_latency = self._retrieve_question(
                    case, question, contract, driver
                )
            except Exception as error:
                if contract.memory_system_error_score is None:
                    raise
                self._record_retrieval_system_error(
                    case=case,
                    question=question,
                    contract=contract,
                    query_text=contract.retrieval_query(question),
                    error=error,
                    latency_ms=(perf_counter() - started) * 1000,
                )
                return
        else:
            query_text, retrieval, retrieval_latency = prepared_retrieval
        retrieval_row = self._record_retrieval_success(
            case=case,
            question=question,
            query_text=query_text,
            retrieval=retrieval,
            latency_ms=retrieval_latency,
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
        answer_row = {
            "question_id": question.question_id,
            "answer": answer,
            "latency_ms": round(answer_latency, 3),
        }

        grade_rows: list[dict[str, Any]] = []
        for grader_index, grader_contract in enumerate(contract.graders):
            if not grader_contract.applies(question):
                continue
            grade_started = perf_counter()
            if grader_contract.deterministic_scorer is not None:
                grade = grader_contract.deterministic_scorer(question, answer)
            else:
                assert grader_contract.judge_plan is not None
                assert grader_contract.judge_reducer is not None
                parsed_steps = tuple(
                    self._complete_and_parse(
                        case=case,
                        question=question,
                        prompt=step.prompt,
                        phase="grading",
                        parse=step.parse,
                        model=self.judge_model,
                    )
                    for step in grader_contract.judge_plan(question, answer)
                )
                grade = grader_contract.judge_reducer(
                    question,
                    answer,
                    parsed_steps,
                )
            if grade.scorer_id != grader_contract.scorer_id:
                raise ArtifactContractError(
                    "grade result scorer_id does not match its declared contract"
                )
            grade_rows.append(
                {
                    "question_id": question.question_id,
                    "scorer_id": grade.scorer_id,
                    "score": grade.score,
                    "label": grade.label,
                    "details": grade.details,
                    "primary": grader_index == 0,
                    "latency_ms": round(
                        (perf_counter() - grade_started) * 1000,
                        3,
                    ),
                },
            )
        self.artifacts.publish_question(
            case=case,
            question=question,
            contract_fingerprint=contract.fingerprint,
            retrieval=retrieval_row,
            answer=answer_row,
            grades=grade_rows,
            execution_attempt=self._execution_attempt,
            unit_attempt=self._unit_attempt,
        )

    def _retrieve_question(
        self,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        contract: TaskContract,
        driver: MemorySystemDriver,
    ) -> tuple[str, RetrievalOutput, float]:
        """Execute one retrieval without applying benchmark failure policy."""

        query_text = contract.retrieval_query(question)
        started = perf_counter()
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
        return query_text, retrieval, (perf_counter() - started) * 1000

    def _record_retrieval_success(
        self,
        *,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        query_text: str,
        retrieval: RetrievalOutput,
        latency_ms: float,
    ) -> dict[str, Any]:
        row = {
            "question_id": question.question_id,
            "query": query_text,
            "context": retrieval.context,
            "channels": retrieval.channels,
            "metrics": retrieval.metrics,
            "latency_ms": round(latency_ms, 3),
        }
        self.artifacts.trace_stage(
            phase="retrieval",
            case_id=case.case_id,
            payload={
                "question_id": question.question_id,
                "latency_ms": round(latency_ms, 3),
                "channels": {
                    name: len(rows) for name, rows in retrieval.channels.items()
                },
            },
        )
        return row

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
        retrieval_row = {
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
        }
        answer_row = {
            "question_id": question.question_id,
            "status": "skipped_due_to_retrieval_error",
            "answer": "",
            "latency_ms": None,
        }
        grade_rows: list[dict[str, Any]] = []
        for grader_index, grader in enumerate(contract.graders):
            if not grader.applies(question):
                continue
            grade_rows.append(
                {
                    "question_id": question.question_id,
                    "scorer_id": grader.scorer_id,
                    "score": 0.0,
                    "label": "system_error",
                    "details": {
                        "grade_source": "benchmark_failure_policy",
                        "failed_phase": "retrieval",
                        "error_type": error_type,
                        "error": error_message,
                    },
                    "primary": grader_index == 0,
                    "latency_ms": None,
                },
            )
        self.artifacts.publish_question(
            case=case,
            question=question,
            contract_fingerprint=contract.fingerprint,
            retrieval=retrieval_row,
            answer=answer_row,
            grades=grade_rows,
            execution_attempt=self._execution_attempt,
            unit_attempt=self._unit_attempt,
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

    def _record_case_system_error(
        self,
        *,
        case: BenchmarkCase,
        contract: TaskContract,
        phase: str,
        error: Exception,
        latency_ms: float,
        event_id: str | None,
    ) -> None:
        """Score all questions zero when a declared memory operation fails."""

        assert contract.memory_system_error_score is not None
        error_type = type(error).__name__
        error_message = str(error)
        rounded_latency = round(latency_ms, 3)
        self.artifacts.trace_stage(
            phase=phase,
            case_id=case.case_id,
            payload={
                "status": "system_error",
                "event_id": event_id,
                "error_type": error_type,
                "error": error_message,
                "latency_ms": rounded_latency,
            },
        )
        for question in case.questions:
            query_text = contract.retrieval_query(question)
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
                        "failed_phase": phase,
                        "error_type": error_type,
                        "error": error_message,
                    },
                    "error_type": error_type,
                    "error": error_message,
                    "latency_ms": None,
                },
            )
            self.artifacts.append_case_row(
                case.case_id,
                "answers",
                {
                    "question_id": question.question_id,
                    "status": f"skipped_due_to_{phase}_error",
                    "answer": "",
                    "latency_ms": None,
                },
            )
            for grader_index, grader in enumerate(contract.graders):
                if not grader.applies(question):
                    continue
                self.artifacts.append_case_row(
                    case.case_id,
                    "grades",
                    {
                        "question_id": question.question_id,
                        "scorer_id": grader.scorer_id,
                        "score": contract.memory_system_error_score,
                        "label": "system_error",
                        "details": {
                            "grade_source": "benchmark_failure_policy",
                            "failed_phase": phase,
                            "event_id": event_id,
                            "error_type": error_type,
                            "error": error_message,
                        },
                        "primary": grader_index == 0,
                        "latency_ms": None,
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
    "GradeContract",
    "JudgeStep",
    "MemorySystemContract",
    "MemorySystemDriver",
    "ModelPrompt",
    "ModelResponse",
    "RetrievalOutput",
    "TaskContract",
]
