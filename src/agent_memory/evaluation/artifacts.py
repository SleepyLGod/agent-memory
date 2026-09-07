"""Benchmark-neutral artifact and resume storage."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from statistics import mean, median
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from uuid import uuid4

from agent_memory.evaluation.attempt_metrics import (
    DurableUnit,
    classify_attempt_rows,
)
from agent_memory.evaluation.bundle import BenchmarkBundle, write_bundle
from agent_memory.evaluation.pricing import PricingSnapshot
from agent_memory.evaluation.provenance import collect_runtime_provenance
from agent_memory.evaluation.question_results import AtomicQuestionStore
from agent_memory.evaluation.recovery import (
    ArtifactContractError,
    AttemptLineage,
    MAX_UNIT_ATTEMPTS,
    UnitAttemptExhausted,
    UnitAttemptStore,
)
from agent_memory.evaluation.trace_metrics import (
    normalize_framework_cache_usage,
    normalize_provider_calls,
    summarize_framework_cache_usage,
    summarize_provider_calls,
)
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
)
from agent_memory.tracing.semantic import (
    semantic_trace_scope,
    write_llm_call_trace,
    write_trace_event,
)

if TYPE_CHECKING:
    from agent_memory.evaluation.harness import MemorySystemContract, MemorySystemDriver


@dataclass(frozen=True)
class BenchmarkCheckpoint:
    """One validated session-boundary checkpoint owned by a system driver."""

    directory: Path
    completed_events: tuple[BenchmarkEvent, ...]
    event_attempts: tuple[AttemptLineage, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _event_fingerprint(event: BenchmarkEvent) -> str:
    payload = json.dumps(
        _json_safe(asdict(event)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_durable_units(case_dir: Path, case_id: str) -> set[DurableUnit]:
    """Read durable event lineages from the atomically published checkpoint."""

    pointer_path = case_dir / "checkpoints" / "current.json"
    if not pointer_path.is_file():
        return set()
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ArtifactContractError(
            "checkpoint current pointer is missing checkpoint_id"
        )
    manifest_path = (
        case_dir
        / "checkpoints"
        / "snapshots"
        / checkpoint_id
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempts = manifest.get("completed_event_attempts")
    if not isinstance(attempts, list):
        raise ArtifactContractError(
            "checkpoint is missing completed event attempt evidence"
        )
    durable: set[DurableUnit] = set()
    for value in attempts:
        if not isinstance(value, Mapping):
            raise ArtifactContractError(
                "checkpoint event attempt must be an object"
            )
        event_id = value.get("event_id")
        execution_attempt = value.get("execution_attempt")
        unit_attempt = value.get("unit_attempt")
        if (
            not isinstance(event_id, str)
            or not event_id
            or not isinstance(execution_attempt, int)
            or execution_attempt < 1
            or not isinstance(unit_attempt, int)
            or unit_attempt < 0
        ):
            raise ArtifactContractError(
                "checkpoint event attempt lineage is invalid"
            )
        durable.add(
            (
                case_id,
                "insertion",
                event_id,
                execution_attempt,
                unit_attempt,
            )
        )
    return durable


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _case_directory_name(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._-") or "case"
    digest = sha256(case_id.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:80]}-{digest}"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_ms": round(mean(values), 3) if values else None,
        "median_ms": round(median(values), 3) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 3) if values else None,
    }


def _normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(value.casefold().split())


def _token_f1(prediction: Any, reference: Any) -> float:
    predicted = _normalized_text(prediction).split()
    expected = _normalized_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    remaining = list(expected)
    overlap = 0
    for token in predicted:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return round(2 * precision * recall / (precision + recall), 6)


def _provider_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [row.get("estimated_cost_usd") for row in rows]
    known_costs = [float(cost) for cost in costs if cost is not None]
    return {
        "provider_call_count": len(rows),
        "provider_error_count": sum(row.get("status") == "error" for row in rows),
        "provider_latency_sum_ms": round(
            sum(float(row.get("latency_ms") or 0) for row in rows), 3
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "cache_hit_tokens": sum(
            int(row.get("cache_hit_tokens") or 0) for row in rows
        ),
        "cache_miss_tokens": sum(
            int(row.get("cache_miss_tokens") or 0) for row in rows
        ),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in rows
        ),
        "reasoning_tokens": (
            sum(int(row["reasoning_tokens"]) for row in rows)
            if rows and all(row.get("reasoning_tokens") is not None for row in rows)
            else None
        ),
        "usage_complete": all(bool(row.get("usage_available")) for row in rows),
        "known_cost_usd": round(sum(known_costs), 12),
        "estimated_cost_usd": (
            round(sum(known_costs), 12)
            if costs and len(known_costs) == len(costs)
            else (0.0 if not costs else None)
        ),
    }


class BenchmarkArtifactStore:
    """Write one resumable benchmark run with opaque driver checkpoints."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.trace_dir = output_dir / "trace"

    def initialize(
        self,
        *,
        bundle: BenchmarkBundle,
        system_contract: MemorySystemContract,
        answer_model_id: str,
        judge_model_id: str,
        contract_fingerprints: Mapping[str, str],
        answer_prompt_digests: Mapping[str, str],
        scorer_contracts: Mapping[str, Mapping[str, Any]],
        answer_parser_contracts: Mapping[str, str] | None = None,
        runtime_provenance: Mapping[str, Any] | None = None,
        storage_provenance: Mapping[str, Any] | None = None,
        run_mode: str = "full",
        maintenance_checkpoint_source: str | None = None,
    ) -> None:
        """Create or validate the immutable run contract and normalized input."""

        provenance = runtime_provenance or collect_runtime_provenance(
            Path(__file__).resolve().parents[3],
            lockfile="uv.lock",
            dependencies=("agent-memory",),
        )
        expected = {
            "schema_version": 3,
            "benchmark_id": bundle.benchmark_id,
            "dataset_revision": bundle.dataset_revision,
            "dataset_sha256": bundle.dataset_sha256,
            "bundle_fingerprint": bundle.fingerprint,
            "policy_input_fingerprint": bundle.policy_input_fingerprint,
            "case_ids": [case.case_id for case in bundle.cases],
            "question_ids": [
                question.question_id
                for case in bundle.cases
                for question in case.questions
            ],
            "system_id": system_contract.system_id,
            "condition_id": system_contract.effective_condition_id,
            "memory_model_id": system_contract.memory_model_id,
            "memory_provider_model_id": system_contract.memory_provider_model_id,
            "input_adapter_id": system_contract.input_adapter_id,
            "input_adapter_digest": system_contract.input_adapter_digest,
            "retrieval_recipe_id": system_contract.retrieval_recipe_id,
            "retrieval_recipe_digest": system_contract.retrieval_recipe_digest,
            "maintenance_policy_id": (
                system_contract.effective_maintenance_policy_id
            ),
            "maintenance_rule": system_contract.maintenance_rule,
            "maintenance_execution_id": system_contract.maintenance_execution_id,
            "maintenance_fingerprint": system_contract.maintenance_fingerprint,
            "thinking_enabled": system_contract.thinking_enabled,
            "consolidation_mode": system_contract.consolidation_mode,
            "parser_mode": system_contract.parser_mode,
            "framework_cache_mode": system_contract.framework_cache_mode,
            "checkpoint_enabled": system_contract.checkpoint_enabled,
            "answer_model_id": answer_model_id,
            "answer_provider_model_id": answer_model_id,
            "answer_thinking_enabled": False,
            "judge_model_id": judge_model_id,
            "judge_provider_model_id": judge_model_id,
            "judge_thinking_enabled": False,
            "contract_fingerprints": dict(sorted(contract_fingerprints.items())),
            "answer_prompt_digests": dict(sorted(answer_prompt_digests.items())),
            "scorer_contracts": {
                task_id: dict(scorer_contracts[task_id])
                for task_id in sorted(scorer_contracts)
            },
            "source_provenance": dict(provenance["source"]),
            "runtime_provenance": dict(provenance["runtime"]),
            "storage_provenance": (
                dict(storage_provenance) if storage_provenance is not None else None
            ),
            "run_mode": run_mode,
            "maintenance_checkpoint_source": maintenance_checkpoint_source,
        }
        if answer_parser_contracts:
            expected["answer_parser_contracts"] = dict(
                sorted(answer_parser_contracts.items())
            )
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError("existing benchmark output has a different run contract")
            return

        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"benchmark output directory is not empty: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("cases", "trace/prompts", "trace/outputs", "metrics"):
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)
        write_bundle(bundle, self.output_dir / "input")
        _write_json(manifest_path, expected)

    def case_dir(self, case_id: str) -> Path:
        """Return the portable artifact directory for one logical case ID."""

        return self.output_dir / "cases" / _case_directory_name(case_id)

    def completed(self, case: BenchmarkCase, contract_fingerprint: str) -> bool:
        """Return whether a matching case attempt completed successfully."""

        path = self.case_dir(case.case_id) / "status.json"
        if not path.exists():
            return False
        status = json.loads(path.read_text(encoding="utf-8"))
        return (
            status.get("status") == "completed"
            and status.get("case_id") == case.case_id
            and status.get("contract_fingerprint") == contract_fingerprint
        )

    def start_case(self, case: BenchmarkCase, contract_fingerprint: str) -> Path:
        """Start a fresh attempt while preserving the prior diagnostic status."""

        case_dir = self.case_dir(case.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        status_path = case_dir / "status.json"
        attempt = 1
        if status_path.exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            attempt = int(previous.get("attempt", 0)) + 1
        AtomicQuestionStore(case_dir).rebuild_jsonl(
            [question.question_id for question in case.questions]
        )
        _write_json(
            case_dir / "case.json",
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "event_count": len(case.events),
                "question_count": len(case.questions),
                "metadata": dict(case.metadata),
            },
        )
        _write_json(
            status_path,
            {
                "status": "running",
                "case_id": case.case_id,
                "attempt": attempt,
                "contract_fingerprint": contract_fingerprint,
            },
        )
        state_dir = case_dir / "state" / f"attempt-{attempt:04d}"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def begin_unit_attempt(
        self,
        *,
        case: BenchmarkCase,
        phase: str,
        unit_id: str,
        execution_attempt: int,
    ) -> int:
        """Start one persisted event or question attempt."""

        return self.unit_attempt_store(case.case_id).begin(
            phase=phase,
            unit_id=unit_id,
            execution_attempt=execution_attempt,
        )

    def finish_unit_attempt(
        self,
        *,
        case: BenchmarkCase,
        phase: str,
        unit_id: str,
        execution_attempt: int,
        unit_attempt: int,
        status: str,
        error: BaseException | None = None,
    ) -> bool:
        """Finish one unit attempt and return whether its budget is exhausted."""

        return self.unit_attempt_store(case.case_id).finish(
            phase=phase,
            unit_id=unit_id,
            execution_attempt=execution_attempt,
            unit_attempt=unit_attempt,
            status=status,
            error=error,
        )

    def question_completed(
        self,
        *,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        contract_fingerprint: str,
        scorer_ids: Iterable[str],
    ) -> bool:
        """Return whether one atomic question result matches its contract."""

        return self.question_store(case.case_id).completed(
            question_id=question.question_id,
            contract_fingerprint=contract_fingerprint,
            scorer_ids=list(scorer_ids),
        )

    def publish_question(
        self,
        *,
        case: BenchmarkCase,
        question: BenchmarkQuestion,
        contract_fingerprint: str,
        retrieval: Mapping[str, Any],
        answer: Mapping[str, Any],
        grades: Iterable[Mapping[str, Any]],
        execution_attempt: int,
        unit_attempt: int,
    ) -> None:
        """Atomically publish one complete retrieval, answer, and grade result."""

        grade_rows = [dict(row) for row in grades]
        store = self.question_store(case.case_id)
        store.publish(
            question_id=question.question_id,
            contract_fingerprint=contract_fingerprint,
            retrieval=retrieval,
            answer=answer,
            grades=grade_rows,
            execution_attempt=execution_attempt,
            unit_attempt=unit_attempt,
        )
        store.rebuild_jsonl(
            [item.question_id for item in case.questions]
        )

    def unit_attempt_store(self, case_id: str) -> UnitAttemptStore:
        """Return the case-local unit attempt ledger."""

        return UnitAttemptStore(self.case_dir(case_id) / "control")

    def question_store(self, case_id: str) -> AtomicQuestionStore:
        """Return the case-local atomic question store."""

        return AtomicQuestionStore(self.case_dir(case_id))

    def attention_case(
        self,
        case: BenchmarkCase,
        contract_fingerprint: str,
        error: UnitAttemptExhausted,
    ) -> None:
        """Mark a case stopped after one unit exhausts its attempt budget."""

        status_path = self.case_dir(case.case_id) / "status.json"
        current = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {"attempt": 0}
        )
        _write_json(
            status_path,
            {
                **current,
                "status": "attention_required",
                "case_id": case.case_id,
                "contract_fingerprint": contract_fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def save_checkpoint(
        self,
        *,
        bundle: BenchmarkBundle,
        case: BenchmarkCase,
        system_contract: MemorySystemContract,
        completed_events: tuple[BenchmarkEvent, ...],
        event_attempts: tuple[AttemptLineage, ...],
        driver: MemorySystemDriver,
    ) -> BenchmarkCheckpoint:
        """Publish driver state only after one complete input session succeeds."""

        if not completed_events:
            raise ArtifactContractError(
                "checkpoint requires at least one completed event"
            )
        completed_ids = tuple(event.event_id for event in completed_events)
        expected_prefix = tuple(event.event_id for event in case.events[: len(completed_ids)])
        if completed_ids != expected_prefix:
            raise ArtifactContractError(
                "checkpoint events must be a prefix of the case input"
            )
        if len(event_attempts) != len(completed_events):
            raise ArtifactContractError(
                "checkpoint event attempts must match the completed event prefix"
            )
        for event, lineage in zip(completed_events, event_attempts, strict=True):
            phase, unit_id, execution_attempt, unit_attempt = lineage
            if (
                phase != "insertion"
                or unit_id != event.event_id
                or execution_attempt < 1
                or unit_attempt < 0
            ):
                raise ArtifactContractError(
                    "checkpoint event attempt does not match its completed event"
                )

        checkpoint_root = self.case_dir(case.case_id) / "checkpoints"
        checkpoint_id = (
            f"events-{len(completed_events):06d}-"
            f"{_event_fingerprint(completed_events[-1])[:12]}"
        )
        staging = checkpoint_root / "staging" / f"{checkpoint_id}-{uuid4().hex}"
        driver_metadata = driver.save_state(staging / "driver")
        manifest = {
            "schema_version": 3,
            "checkpoint_id": checkpoint_id,
            "case_id": case.case_id,
            "bundle_fingerprint": bundle.fingerprint,
            "policy_input_fingerprint": bundle.policy_input_fingerprint,
            "system_id": system_contract.system_id,
            "maintenance_policy_id": (
                system_contract.effective_maintenance_policy_id
            ),
            "memory_model_id": system_contract.memory_model_id,
            "memory_provider_model_id": system_contract.memory_provider_model_id,
            "input_adapter_id": system_contract.input_adapter_id,
            "maintenance_rule": system_contract.maintenance_rule,
            "maintenance_execution_id": system_contract.maintenance_execution_id,
            "maintenance_fingerprint": system_contract.maintenance_fingerprint,
            "thinking_enabled": system_contract.thinking_enabled,
            "consolidation_mode": system_contract.consolidation_mode,
            "framework_cache_mode": system_contract.framework_cache_mode,
            "completed_session_id": completed_events[-1].session_id,
            "completed_event_ids": list(completed_ids),
            "completed_event_fingerprints": [
                _event_fingerprint(event) for event in completed_events
            ],
            "completed_event_attempts": [
                {
                    "event_id": event.event_id,
                    "execution_attempt": lineage[2],
                    "unit_attempt": lineage[3],
                }
                for event, lineage in zip(
                    completed_events,
                    event_attempts,
                    strict=True,
                )
            ],
            "driver_state": dict(driver_metadata),
        }
        _write_json(staging / "manifest.json", manifest)
        snapshot = checkpoint_root / "snapshots" / checkpoint_id
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            existing = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            if existing != manifest:
                raise ArtifactContractError(
                    "checkpoint ID collision contains different state"
                )
        else:
            os.replace(staging, snapshot)
        _write_json_atomic(
            checkpoint_root / "current.json",
            {"schema_version": 1, "checkpoint_id": checkpoint_id},
        )
        return BenchmarkCheckpoint(snapshot, completed_events, event_attempts)

    def load_checkpoint(
        self,
        *,
        bundle: BenchmarkBundle,
        case: BenchmarkCase,
        system_contract: MemorySystemContract,
    ) -> BenchmarkCheckpoint | None:
        """Validate and return the latest checkpoint for an incomplete case."""

        checkpoint_root = self.case_dir(case.case_id) / "checkpoints"
        pointer_path = checkpoint_root / "current.json"
        if not pointer_path.exists():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint_id = pointer.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ArtifactContractError(
                "checkpoint current pointer is missing checkpoint_id"
            )
        snapshot = checkpoint_root / "snapshots" / checkpoint_id
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_maintenance_policy_id = manifest.get(
            "maintenance_policy_id",
            manifest.get("system_id"),
        )
        if (
            checkpoint_maintenance_policy_id
            != system_contract.effective_maintenance_policy_id
        ):
            raise ArtifactContractError(
                "checkpoint does not match current run contract: "
                "maintenance_policy_id"
            )
        expected_contract = {
            "case_id": case.case_id,
            "bundle_fingerprint": bundle.fingerprint,
            "policy_input_fingerprint": bundle.policy_input_fingerprint,
            "memory_model_id": system_contract.memory_model_id,
            "memory_provider_model_id": system_contract.memory_provider_model_id,
            "input_adapter_id": system_contract.input_adapter_id,
            "maintenance_rule": system_contract.maintenance_rule,
            "maintenance_fingerprint": system_contract.maintenance_fingerprint,
        }
        mismatched = [
            name
            for name, expected in expected_contract.items()
            if manifest.get(name) != expected
        ]
        if mismatched:
            raise ArtifactContractError(
                "checkpoint does not match current run contract: "
                + ", ".join(mismatched)
            )
        event_ids = manifest.get("completed_event_ids")
        fingerprints = manifest.get("completed_event_fingerprints")
        attempts = manifest.get("completed_event_attempts")
        if (
            not isinstance(event_ids, list)
            or not isinstance(fingerprints, list)
            or not isinstance(attempts, list)
        ):
            raise ArtifactContractError(
                "checkpoint is missing completed event prefix evidence"
            )
        completed = case.events[: len(event_ids)]
        if [event.event_id for event in completed] != event_ids or [
            _event_fingerprint(event) for event in completed
        ] != fingerprints:
            raise ArtifactContractError(
                "checkpoint completed event prefix does not match input"
            )
        if len(attempts) != len(completed):
            raise ArtifactContractError(
                "checkpoint event attempts do not match completed events"
            )
        event_attempts: list[AttemptLineage] = []
        for event, value in zip(completed, attempts, strict=True):
            if not isinstance(value, Mapping) or value.get("event_id") != event.event_id:
                raise ArtifactContractError(
                    "checkpoint event attempt identity does not match input"
                )
            execution_attempt = value.get("execution_attempt")
            unit_attempt = value.get("unit_attempt")
            if (
                not isinstance(execution_attempt, int)
                or execution_attempt < 1
                or not isinstance(unit_attempt, int)
                or unit_attempt < 0
            ):
                raise ArtifactContractError(
                    "checkpoint event attempt lineage is invalid"
                )
            event_attempts.append(
                (
                    "insertion",
                    event.event_id,
                    execution_attempt,
                    unit_attempt,
                )
            )
        return BenchmarkCheckpoint(
            snapshot,
            tuple(completed),
            tuple(event_attempts),
        )

    def complete_case(self, case: BenchmarkCase, contract_fingerprint: str) -> None:
        """Mark one case complete only after all questions were graded."""

        status_path = self.case_dir(case.case_id) / "status.json"
        current = json.loads(status_path.read_text(encoding="utf-8"))
        _write_json(
            status_path,
            {
                **current,
                "status": "completed",
                "contract_fingerprint": contract_fingerprint,
            },
        )

    def complete_maintenance(
        self,
        case: BenchmarkCase,
        contract_fingerprint: str,
    ) -> None:
        """Mark a case whose final checkpoint is ready for read-only reuse."""

        status_path = self.case_dir(case.case_id) / "status.json"
        current = json.loads(status_path.read_text(encoding="utf-8"))
        _write_json(
            status_path,
            {
                **current,
                "status": "maintenance_completed",
                "contract_fingerprint": contract_fingerprint,
            },
        )

    def maintenance_completed(
        self,
        case: BenchmarkCase,
        contract_fingerprint: str,
    ) -> bool:
        """Return whether a matching final maintenance checkpoint exists."""

        path = self.case_dir(case.case_id) / "status.json"
        if not path.exists():
            return False
        status = json.loads(path.read_text(encoding="utf-8"))
        return (
            status.get("status") == "maintenance_completed"
            and status.get("case_id") == case.case_id
            and status.get("contract_fingerprint") == contract_fingerprint
        )

    def fail_case(
        self,
        case: BenchmarkCase,
        contract_fingerprint: str,
        error: BaseException,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist enough failure evidence to rerun the case from fresh state."""

        status_path = self.case_dir(case.case_id) / "status.json"
        current = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {"attempt": 0}
        )
        _write_json(
            status_path,
            {
                **current,
                **dict(details or {}),
                "status": "failed",
                "case_id": case.case_id,
                "contract_fingerprint": contract_fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def append_case_row(self, case_id: str, kind: str, row: Mapping[str, Any]) -> None:
        """Append one retrieval, answer, or grade row for a case."""

        if kind not in {"retrieval", "answers", "grades"}:
            raise ValueError(f"unsupported case artifact kind {kind!r}")
        _append_jsonl(self.case_dir(case_id) / f"{kind}.jsonl", row)

    def trace_stage(
        self,
        *,
        phase: str,
        case_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Write a compact non-LLM insertion or retrieval event."""

        with semantic_trace_scope(phase=phase, case_id=case_id):
            write_trace_event(
                self.trace_dir,
                operator="benchmark_runner",
                event_type=f"{phase}_result",
                payload=payload,
            )

    def trace_model_call(
        self,
        *,
        phase: str,
        case_id: str,
        question_id: str,
        prompt_name: str,
        prompt_digest: str,
        attempt: int,
        model: str,
        messages: Iterable[Mapping[str, str]],
        kwargs: Mapping[str, Any],
        raw_output: Any,
        latency_ms: float,
        usage: Mapping[str, Any],
    ) -> None:
        """Record one real answer or grading provider attempt."""

        with semantic_trace_scope(
            phase=phase,
            case_id=case_id,
            question_id=question_id,
            prompt_name=prompt_name,
            prompt_digest=prompt_digest,
            attempt=attempt,
        ):
            write_llm_call_trace(
                self.trace_dir,
                model=model,
                messages=[list(messages)],
                kwargs=kwargs,
                outputs=[raw_output],
                latency_sec=latency_ms / 1000,
                usage_delta=usage,
            )

    def trace_model_error(
        self,
        *,
        phase: str,
        case_id: str,
        question_id: str,
        prompt_name: str,
        prompt_digest: str,
        attempt: int,
        model: str,
        messages: Iterable[Mapping[str, str]],
        kwargs: Mapping[str, Any],
        latency_ms: float,
        error: BaseException,
    ) -> None:
        """Record a failed physical answer or grading provider attempt."""

        with semantic_trace_scope(
            phase=phase,
            case_id=case_id,
            question_id=question_id,
            prompt_name=prompt_name,
            prompt_digest=prompt_digest,
            attempt=attempt,
        ):
            write_llm_call_trace(
                self.trace_dir,
                model=model,
                messages=[list(messages)],
                kwargs=kwargs,
                latency_sec=latency_ms / 1000,
                error=error,
            )

    def finalize_metrics(self) -> None:
        """Rebuild run summaries from completed case artifacts and trace evidence."""

        question_rows: list[dict[str, Any]] = []
        grade_rows: list[dict[str, Any]] = []
        completed_cases = 0
        failed_cases = 0
        attention_required_cases = 0
        empty_retrievals = 0
        retrieval_system_errors = 0
        memory_system_error_questions = 0
        memory_system_error_cases: set[str] = set()
        input_question_rows = _read_jsonl(
            self.output_dir / "input" / "questions.jsonl"
        )
        input_questions = {
            row["question_id"]: row for row in input_question_rows
        }
        question_ids_by_case: dict[str, list[str]] = {}
        for row in input_question_rows:
            question_ids_by_case.setdefault(str(row["case_id"]), []).append(
                str(row["question_id"])
            )
        for case_dir in sorted((self.output_dir / "cases").glob("*")):
            status_path = case_dir / "status.json"
            if not status_path.exists():
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed":
                completed_cases += 1
            elif status.get("status") == "failed":
                failed_cases += 1
                continue
            elif status.get("status") == "attention_required":
                attention_required_cases += 1
            retrievals = {
                row["question_id"]: row
                for row in _read_jsonl(case_dir / "retrieval.jsonl")
            }
            answers = {
                row["question_id"]: row
                for row in _read_jsonl(case_dir / "answers.jsonl")
            }
            grades_by_question: dict[str, list[dict[str, Any]]] = {}
            for row in _read_jsonl(case_dir / "grades.jsonl"):
                grades_by_question.setdefault(str(row["question_id"]), []).append(row)
            for question_id in sorted(
                set(retrievals) | set(answers) | set(grades_by_question)
            ):
                retrieval = retrievals.get(question_id, {})
                answer = answers.get(question_id, {})
                question_grades = grades_by_question.get(question_id, [])
                primary_grades = [
                    row for row in question_grades if row.get("primary", True)
                ]
                if len(primary_grades) > 1:
                    raise ValueError(
                        f"question {question_id!r} has multiple primary grades"
                    )
                grade = primary_grades[0] if primary_grades else {}
                retrieval_status = str(retrieval.get("status") or "success")
                if retrieval_status == "system_error":
                    retrieval_system_errors += 1
                grade_details = grade.get("details", {})
                failed_phase = (
                    str(grade_details.get("failed_phase") or "")
                    if isinstance(grade_details, Mapping)
                    else ""
                )
                if grade.get("label") == "system_error":
                    memory_system_error_questions += 1
                    case_id = status.get("case_id")
                    if isinstance(case_id, str) and case_id:
                        memory_system_error_cases.add(case_id)
                reference = input_questions.get(question_id, {}).get("gold_answer", "")
                answer_text = answer.get("answer", "")
                channels = retrieval.get("channels", {})
                returned_count = (
                    sum(len(rows) for rows in channels.values())
                    if isinstance(channels, Mapping)
                    else 0
                )
                if not str(retrieval.get("context") or "").strip():
                    empty_retrievals += 1
                normalized_answer = _normalized_text(answer_text)
                normalized_reference = _normalized_text(reference)
                completion = self.question_store(
                    str(status.get("case_id") or "")
                ).completion_metadata(
                    question_id
                )
                question_rows.append(
                    {
                        "case_id": status.get("case_id"),
                        "question_id": question_id,
                        "execution_attempt": completion.get(
                            "execution_attempt",
                            "",
                        ),
                        "unit_attempt": completion.get("unit_attempt", ""),
                        "retrieval_status": retrieval_status,
                        "retrieval_error_type": retrieval.get("error_type", ""),
                        "retrieval_error": retrieval.get("error", ""),
                        "system_error_phase": failed_phase,
                        "retrieval_latency_ms": retrieval.get("latency_ms", ""),
                        "retrieval_returned_count": returned_count,
                        "answer_latency_ms": answer.get("latency_ms", ""),
                        "primary_grading_latency_ms": grade.get("latency_ms", ""),
                        "answer": answer_text,
                        "primary_scorer_id": grade.get("scorer_id", ""),
                        "primary_score": grade.get("score", ""),
                        "exact_match": int(normalized_answer == normalized_reference),
                        "contains_match": int(
                            bool(normalized_reference)
                            and normalized_reference in normalized_answer
                        ),
                        "token_f1": _token_f1(answer_text, reference),
                    }
                )
                for grade_row in question_grades:
                    grade_rows.append(
                        {
                            "case_id": status.get("case_id"),
                            "question_id": question_id,
                            "scorer_id": grade_row.get("scorer_id", ""),
                            "score": grade_row.get("score", ""),
                            "label": grade_row.get("label", ""),
                            "primary": bool(grade_row.get("primary", True)),
                            "grading_latency_ms": grade_row.get("latency_ms", ""),
                        }
                    )

        trace_rows = _read_jsonl(self.trace_dir / "events.jsonl")
        framework_cache_rows = normalize_framework_cache_usage(trace_rows)
        embedding_rows = [
            {
                "trace_id": row.get("trace_id", ""),
                "case_id": row.get("case_id", ""),
                "event_id": row.get("event_id", ""),
                "question_id": row.get("question_id", ""),
                "phase": row.get("phase", ""),
                "operation": row.get("operation", ""),
                "attempt": row.get("attempt", ""),
                "execution_attempt": row.get("execution_attempt", ""),
                "unit_attempt": row.get("unit_attempt", ""),
                "status": row.get("status", ""),
                "model": row.get("model", ""),
                "revision": row.get("revision", ""),
                "source_column": row.get("source_column", ""),
                "property_name": row.get("property_name", ""),
                "batch_size": row.get("batch_size", ""),
                "dimensions": row.get("dimensions", ""),
                "result_count": row.get("result_count", ""),
                "result_dimensions": row.get("result_dimensions", ""),
                "normalize": row.get("normalize", ""),
                "device": row.get("device", ""),
                "latency_ms": row.get("latency_ms", ""),
                "input_path": row.get("input_path", ""),
                "error_type": row.get("error_type", ""),
                "error_message": row.get("error_message", ""),
            }
            for row in trace_rows
            if row.get("event_type") == "embedding_call"
        ]
        embedding_by_event: dict[
            tuple[str, str, int, int],
            list[dict[str, Any]],
        ] = {}
        embedding_by_question: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in embedding_rows:
            case_id = str(row.get("case_id") or "")
            event_id = str(row.get("event_id") or "")
            question_id = str(row.get("question_id") or "")
            execution_attempt = int(row.get("execution_attempt") or 0)
            unit_attempt = int(row.get("unit_attempt") or 0)
            if event_id:
                embedding_by_event.setdefault(
                    (case_id, event_id, execution_attempt, unit_attempt),
                    [],
                ).append(row)
            if question_id:
                embedding_by_question.setdefault(
                    (case_id, question_id),
                    [],
                ).append(row)
        for row in question_rows:
            calls = embedding_by_question.get(
                (str(row.get("case_id") or ""), str(row.get("question_id") or "")),
                [],
            )
            row["embedding_call_count"] = len(calls)
            row["embedding_error_count"] = sum(
                call.get("status") == "error" for call in calls
            )
            row["embedding_latency_sum_ms"] = round(
                sum(float(call.get("latency_ms") or 0) for call in calls),
                3,
            )
        self._write_csv(self.output_dir / "metrics" / "per_grade.csv", grade_rows)
        pricing = PricingSnapshot.deepseek_2026_07_17()
        provider_rows = normalize_provider_calls(
            trace_rows,
            output_dir=self.output_dir,
            pricing=pricing,
        )
        durable_units: set[DurableUnit] = set()
        authoritative_cases: set[str] = set()
        for case_dir in sorted((self.output_dir / "cases").glob("*")):
            status_path = case_dir / "status.json"
            if not status_path.is_file():
                continue
            case_id = str(
                json.loads(status_path.read_text(encoding="utf-8")).get(
                    "case_id"
                )
                or ""
            )
            attempt_store = UnitAttemptStore(case_dir / "control")
            if attempt_store.state_path.is_file():
                authoritative_cases.add(case_id)
            durable_units.update(
                _checkpoint_durable_units(case_dir, case_id)
            )
            question_store = AtomicQuestionStore(case_dir)
            for question_id in question_ids_by_case.get(case_id, []):
                lineage = question_store.lineage(question_id)
                if lineage is None:
                    continue
                durable_units.add(
                    (
                        case_id,
                        *lineage,
                    )
                )
        provider_rows = classify_attempt_rows(
            provider_rows,
            durable_units=durable_units,
            authoritative_cases=authoritative_cases,
        )
        embedding_rows = classify_attempt_rows(
            embedding_rows,
            durable_units=durable_units,
            authoritative_cases=authoritative_cases,
        )
        framework_cache_rows = classify_attempt_rows(
            framework_cache_rows,
            durable_units=durable_units,
            authoritative_cases=authoritative_cases,
        )
        self._write_csv(
            self.output_dir / "metrics" / "embedding_usage.csv",
            embedding_rows,
        )
        self._write_csv(
            self.output_dir / "metrics" / "framework_cache_usage.csv",
            framework_cache_rows,
        )
        for question_row in question_rows:
            question_provider_rows = [
                row
                for row in provider_rows
                if row.get("case_id") == question_row.get("case_id")
                and row.get("question_id") == question_row.get("question_id")
            ]
            for phase in ("retrieval", "answering", "grading"):
                phase_rows = [
                    row
                    for row in question_provider_rows
                    if row.get("phase") == phase
                ]
                actual = summarize_provider_calls(phase_rows)
                final = summarize_provider_calls(
                    [row for row in phase_rows if row.get("successful_path")]
                )
                recovery = summarize_provider_calls(
                    [row for row in phase_rows if not row.get("successful_path")]
                )
                question_row[f"{phase}_provider_call_count"] = actual[
                    "provider_call_count"
                ]
                question_row[f"{phase}_provider_latency_ms"] = actual[
                    "latency_ms"
                ]
                question_row[f"{phase}_prompt_tokens"] = actual[
                    "prompt_tokens"
                ]
                question_row[f"{phase}_cache_hit_tokens"] = actual[
                    "cache_hit_tokens"
                ]
                question_row[f"{phase}_cache_miss_tokens"] = actual[
                    "cache_miss_tokens"
                ]
                question_row[f"{phase}_completion_tokens"] = actual[
                    "completion_tokens"
                ]
                question_row[f"{phase}_known_cost_usd"] = actual[
                    "known_cost_usd"
                ]
                question_row[f"{phase}_estimated_cost_usd"] = actual[
                    "estimated_cost_usd"
                ]
                question_row[f"{phase}_usage_complete"] = actual[
                    "usage_complete"
                ]
                question_row[f"{phase}_final_known_cost_usd"] = final[
                    "known_cost_usd"
                ]
                question_row[f"{phase}_final_estimated_cost_usd"] = final[
                    "estimated_cost_usd"
                ]
                question_row[f"{phase}_recovery_known_cost_usd"] = recovery[
                    "known_cost_usd"
                ]
        self._write_csv(
            self.output_dir / "metrics" / "per_question.csv",
            question_rows,
        )
        self._write_csv(
            self.output_dir / "metrics" / "provider_usage.csv", provider_rows
        )
        provider_by_event: dict[
            tuple[str, str, int, int],
            list[dict[str, Any]],
        ] = {}
        for row in provider_rows:
            key = (
                str(row.get("case_id") or ""),
                str(row.get("event_id") or ""),
                int(row.get("execution_attempt") or 0),
                int(row.get("unit_attempt") or 0),
            )
            provider_by_event.setdefault(key, []).append(row)

        insertion_events = [
            row for row in trace_rows if row.get("event_type") == "insertion_result"
        ]
        per_event_rows: list[dict[str, Any]] = []
        event_occurrences: dict[tuple[str, str], int] = {}
        previous_shape: dict[str, dict[str, int]] = {}
        for event in insertion_events:
            case_id = str(event.get("case_id") or "")
            event_id = str(event.get("event_id") or "")
            occurrence_key = (case_id, event_id)
            occurrence = event_occurrences.get(occurrence_key, 0) + 1
            event_occurrences[occurrence_key] = occurrence
            attempt = int(event.get("attempt") or 1)
            execution_attempt = int(event.get("execution_attempt") or attempt)
            unit_attempt = int(event.get("unit_attempt") or 0)
            final_path = (
                case_id not in authoritative_cases
                or (
                    case_id,
                    "insertion",
                    event_id,
                    execution_attempt,
                    unit_attempt,
                )
                in durable_units
            )
            calls = provider_by_event.get(
                (case_id, event_id, execution_attempt, unit_attempt),
                [],
            )
            calls = [
                call for call in calls if call.get("phase") == "insertion"
            ]
            embedding_calls = embedding_by_event.get(
                (case_id, event_id, execution_attempt, unit_attempt),
                [],
            )
            embedding_calls = [
                call
                for call in embedding_calls
                if call.get("phase") == "insertion"
            ]
            shape = {
                key: int(value)
                for key, value in event.items()
                if (
                    key.endswith("_rows")
                    or key.startswith("topic_")
                    or key == "catalog_rows"
                )
                and isinstance(value, int)
            }
            deltas = {
                f"delta_{key}": value
                - previous_shape.get(case_id, {}).get(key, 0)
                for key, value in shape.items()
            }
            previous_shape[case_id] = shape
            per_event_rows.append(
                {
                    "case_id": case_id,
                    "event_id": event_id,
                    "session_id": event.get("session_id", ""),
                    "occurrence": occurrence,
                    "attempt": attempt,
                    "execution_attempt": execution_attempt,
                    "unit_attempt": unit_attempt,
                    "successful_path": final_path,
                    "replayed": occurrence > 1,
                    "wall_latency_ms": event.get("latency_ms", ""),
                    "semantic_trace_io_latency_ms": event.get(
                        "semantic_trace_io_latency_ms", ""
                    ),
                    "semantic_trace_bytes_written": event.get(
                        "semantic_trace_bytes_written", ""
                    ),
                    "insertion_latency_excluding_trace_io_ms": event.get(
                        "insertion_latency_excluding_trace_io_ms", ""
                    ),
                    "status": event.get("status", "success"),
                    "error_type": event.get("error_type", ""),
                    "error": event.get("error", ""),
                    **_provider_rollup(calls),
                    "embedding_call_count": len(embedding_calls),
                    "embedding_error_count": sum(
                        call.get("status") == "error"
                        for call in embedding_calls
                    ),
                    "embedding_latency_sum_ms": round(
                        sum(
                            float(call.get("latency_ms") or 0)
                            for call in embedding_calls
                        ),
                        3,
                    ),
                    **shape,
                    **deltas,
                }
            )
        self._write_csv(self.output_dir / "metrics" / "per_event.csv", per_event_rows)

        consolidation_by_session = {
            (str(row.get("case_id") or ""), str(row.get("session_id") or "")): row
            for row in trace_rows
            if row.get("event_type") == "consolidation_result"
        }
        events_by_session: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in per_event_rows:
            key = (str(row["case_id"]), str(row.get("session_id") or ""))
            events_by_session.setdefault(key, []).append(row)
        consolidation_rows = []
        for (case_id, session_id), rows in sorted(events_by_session.items()):
            consolidation = consolidation_by_session.get((case_id, session_id), {})
            final_rows = [row for row in rows if row.get("successful_path")]
            recovery_rows = [
                row for row in rows if not row.get("successful_path")
            ]
            session_provider_rows = [
                row
                for row in provider_rows
                if row.get("case_id") == case_id
                and row.get("session_id") == session_id
                and row.get("phase") in {"insertion", "consolidation"}
            ]
            insertion_provider_rows = [
                row
                for row in session_provider_rows
                if row.get("phase") == "insertion"
            ]
            consolidation_provider_rows = [
                row
                for row in session_provider_rows
                if row.get("phase") == "consolidation"
            ]
            session_usage = summarize_provider_calls(session_provider_rows)
            final_session_usage = summarize_provider_calls(
                [
                    row
                    for row in session_provider_rows
                    if row.get("successful_path")
                ]
            )
            recovery_session_usage = summarize_provider_calls(
                [
                    row
                    for row in session_provider_rows
                    if not row.get("successful_path")
                ]
            )
            insertion_usage = summarize_provider_calls(
                insertion_provider_rows
            )
            consolidation_usage = summarize_provider_calls(
                consolidation_provider_rows
            )
            final_consolidation_usage = summarize_provider_calls(
                [
                    row
                    for row in consolidation_provider_rows
                    if row.get("successful_path")
                ]
            )
            recovery_consolidation_usage = summarize_provider_calls(
                [
                    row
                    for row in consolidation_provider_rows
                    if not row.get("successful_path")
                ]
            )
            session_checkpoint_events = [
                row
                for row in trace_rows
                if row.get("event_type") == "checkpoint_result"
                and row.get("session_id") == session_id
            ]
            consolidation_rows.append(
                {
                    "case_id": case_id,
                    "session_id": session_id,
                    "event_count": len(final_rows),
                    "attempt_event_count": len(rows),
                    "insertion_wall_latency_ms": round(
                        sum(float(row.get("wall_latency_ms") or 0) for row in rows),
                        3,
                    ),
                    "final_insertion_wall_latency_ms": round(
                        sum(
                            float(row.get("wall_latency_ms") or 0)
                            for row in final_rows
                        ),
                        3,
                    ),
                    "recovery_insertion_wall_latency_ms": round(
                        sum(
                            float(row.get("wall_latency_ms") or 0)
                            for row in recovery_rows
                        ),
                        3,
                    ),
                    "semantic_trace_io_latency_ms": (
                        round(
                            sum(
                                float(row["semantic_trace_io_latency_ms"])
                                for row in rows
                            ),
                            3,
                        )
                        if rows
                        and all(
                            row.get("semantic_trace_io_latency_ms")
                            not in {"", None}
                            for row in rows
                        )
                        else None
                    ),
                    "semantic_trace_bytes_written": (
                        sum(
                            int(row["semantic_trace_bytes_written"])
                            for row in rows
                        )
                        if rows
                        and all(
                            row.get("semantic_trace_bytes_written")
                            not in {"", None}
                            for row in rows
                        )
                        else None
                    ),
                    "insertion_latency_excluding_trace_io_ms": (
                        round(
                            sum(
                                float(
                                    row[
                                        "insertion_latency_excluding_trace_io_ms"
                                    ]
                                )
                                for row in rows
                            ),
                            3,
                        )
                        if rows
                        and all(
                            row.get("insertion_latency_excluding_trace_io_ms")
                            not in {"", None}
                            for row in rows
                        )
                        else None
                    ),
                    "consolidation_wall_latency_ms": consolidation.get(
                        "latency_ms", 0
                    ),
                    "checkpoint_wall_latency_ms": round(
                        sum(
                            float(row.get("latency_ms") or 0)
                            for row in session_checkpoint_events
                        ),
                        3,
                    ),
                    "insertion_provider_call_count": insertion_usage[
                        "provider_call_count"
                    ],
                    "insertion_provider_latency_ms": insertion_usage[
                        "latency_ms"
                    ],
                    "consolidation_provider_call_count": consolidation_usage[
                        "provider_call_count"
                    ],
                    "consolidation_provider_latency_ms": consolidation_usage[
                        "latency_ms"
                    ],
                    "provider_call_count": session_usage[
                        "provider_call_count"
                    ],
                    "provider_latency_sum_ms": session_usage["latency_ms"],
                    "prompt_tokens": session_usage["prompt_tokens"],
                    "cache_hit_tokens": session_usage["cache_hit_tokens"],
                    "cache_miss_tokens": session_usage["cache_miss_tokens"],
                    "completion_tokens": session_usage["completion_tokens"],
                    "reasoning_tokens": session_usage["reasoning_tokens"],
                    "estimated_cost_usd": session_usage[
                        "estimated_cost_usd"
                    ],
                    "known_cost_usd": session_usage["known_cost_usd"],
                    "usage_complete": session_usage["usage_complete"],
                    "final_known_cost_usd": final_session_usage[
                        "known_cost_usd"
                    ],
                    "final_estimated_cost_usd": final_session_usage[
                        "estimated_cost_usd"
                    ],
                    "recovery_known_cost_usd": recovery_session_usage[
                        "known_cost_usd"
                    ],
                    "recovery_estimated_cost_usd": recovery_session_usage[
                        "estimated_cost_usd"
                    ],
                    "consolidation_known_cost_usd": consolidation_usage[
                        "known_cost_usd"
                    ],
                    "consolidation_estimated_cost_usd": consolidation_usage[
                        "estimated_cost_usd"
                    ],
                    "consolidation_usage_complete": consolidation_usage[
                        "usage_complete"
                    ],
                    "final_consolidation_known_cost_usd": (
                        final_consolidation_usage["known_cost_usd"]
                    ),
                    "recovery_consolidation_known_cost_usd": (
                        recovery_consolidation_usage["known_cost_usd"]
                    ),
                }
            )
        self._write_csv(
            self.output_dir / "metrics" / "per_session.csv", consolidation_rows
        )

        checkpoint_rows = [
            {
                "case_id": row.get("case_id", ""),
                "operation": row.get("operation", ""),
                "session_id": row.get("session_id", ""),
                "checkpoint_id": row.get("checkpoint_id", ""),
                "execution_attempt": row.get("execution_attempt", ""),
                "unit_attempt": row.get("unit_attempt", ""),
                "completed_event_count": row.get("completed_event_count", ""),
                "checkpoint_bytes": row.get("checkpoint_bytes", ""),
                "wall_latency_ms": row.get("latency_ms", ""),
                "status": row.get("status", "success"),
                "error_type": row.get("error_type", ""),
                "error": row.get("error", ""),
            }
            for row in trace_rows
            if row.get("event_type") == "checkpoint_result"
        ]
        self._write_csv(
            self.output_dir / "metrics" / "checkpoint_metrics.csv", checkpoint_rows
        )

        operator_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in provider_rows:
            key = (str(row.get("phase") or "unknown"), str(row.get("operator") or ""))
            operator_groups.setdefault(key, []).append(row)
        operator_rows = []
        for (phase, operator), rows in sorted(operator_groups.items()):
            logical_calls = {
                str(row.get("logical_call_id") or row.get("trace_id") or "")
                for row in rows
            }
            framework_batches = {
                str(row.get("framework_batch_id") or row.get("trace_id") or "")
                for row in rows
            }
            retry_rows = [row for row in rows if (row.get("attempt") or 1) > 1]
            retry_batches = {
                str(row.get("framework_batch_id") or row.get("trace_id") or "")
                for row in retry_rows
            }
            batch_count = len(framework_batches)
            operator_rows.append(
                {
                    "phase": phase,
                    "operator": operator,
                    "logical_call_count": len(logical_calls),
                    "physical_batch_count": batch_count,
                    "physical_item_count": len(rows),
                    "items_per_batch": round(len(rows) / max(1, batch_count), 3),
                    "retry_provider_call_count": len(retry_rows),
                    "retry_batch_count": len(retry_batches),
                    **_provider_rollup(rows),
                }
            )
        self._write_csv(
            self.output_dir / "metrics" / "operation_usage.csv", operator_rows
        )

        latest_state: dict[str, dict[str, Any]] = {}
        for row in per_event_rows:
            latest_state[str(row["case_id"])] = row
        state_rows = [
            {
                "case_id": case_id,
                **{
                    key: value
                    for key, value in row.items()
                    if key.endswith("_rows")
                    or key.startswith("topic_")
                    or key == "catalog_rows"
                },
            }
            for case_id, row in sorted(latest_state.items())
        ]
        self._write_csv(self.output_dir / "metrics" / "state_shape.csv", state_rows)

        scores = [
            float(row["primary_score"])
            for row in question_rows
            if row["primary_score"] != ""
        ]
        scores_by_scorer: dict[str, list[float]] = {}
        for row in grade_rows:
            if row["score"] in {"", None}:
                continue
            scores_by_scorer.setdefault(str(row["scorer_id"]), []).append(
                float(row["score"])
            )
        provider_summary = summarize_provider_calls(provider_rows)
        final_provider_rows = [
            row for row in provider_rows if row.get("successful_path")
        ]
        recovery_provider_rows = [
            row for row in provider_rows if not row.get("successful_path")
        ]
        final_provider_summary = summarize_provider_calls(final_provider_rows)
        recovery_provider_summary = summarize_provider_calls(
            recovery_provider_rows
        )
        framework_cache_summary = summarize_framework_cache_usage(
            framework_cache_rows
        )
        insertion_latencies = [
            float(row["wall_latency_ms"])
            for row in per_event_rows
            if row.get("wall_latency_ms") not in {"", None}
        ]
        trace_io_metrics_available = bool(per_event_rows) and all(
            row.get("semantic_trace_io_latency_ms") not in {"", None}
            and row.get("semantic_trace_bytes_written") not in {"", None}
            and row.get("insertion_latency_excluding_trace_io_ms")
            not in {"", None}
            for row in per_event_rows
        )
        semantic_trace_io_latencies = (
            [float(row["semantic_trace_io_latency_ms"]) for row in per_event_rows]
            if trace_io_metrics_available
            else []
        )
        insertion_latencies_excluding_trace_io = (
            [
                float(row["insertion_latency_excluding_trace_io_ms"])
                for row in per_event_rows
            ]
            if trace_io_metrics_available
            else []
        )
        retrieval_latencies = [
            float(row["retrieval_latency_ms"])
            for row in question_rows
            if row.get("retrieval_latency_ms") not in {"", None}
        ]
        answer_latencies = [
            float(row["answer_latency_ms"])
            for row in question_rows
            if row.get("answer_latency_ms") not in {"", None}
        ]
        grading_latencies_by_scorer: dict[str, list[float]] = {}
        for row in grade_rows:
            latency = row.get("grading_latency_ms")
            if latency is None or latency == "":
                continue
            if not isinstance(latency, int | float | str):
                raise TypeError("grading latency must be numeric")
            grading_latencies_by_scorer.setdefault(
                str(row["scorer_id"]), []
            ).append(float(latency))
        consolidation_latencies = [
            float(row["consolidation_wall_latency_ms"])
            for row in consolidation_rows
            if row.get("consolidation_wall_latency_ms") not in {"", None, 0}
        ]
        checkpoint_latencies = [
            float(row["wall_latency_ms"])
            for row in checkpoint_rows
            if row.get("wall_latency_ms") not in {"", None}
        ]
        embedding_latencies = [
            float(row["latency_ms"])
            for row in embedding_rows
            if row.get("latency_ms") not in {"", None}
        ]
        embedding_rows_by_phase: dict[str, list[dict[str, Any]]] = {}
        for row in embedding_rows:
            embedding_rows_by_phase.setdefault(
                str(row.get("phase") or "unknown"),
                [],
            ).append(row)
        embedding_phases = {
            phase: {
                "call_count": len(rows),
                "error_count": sum(row.get("status") == "error" for row in rows),
                "latency_sum_ms": round(
                    sum(float(row.get("latency_ms") or 0) for row in rows),
                    3,
                ),
                "latency": _latency_stats(
                    [
                        float(row["latency_ms"])
                        for row in rows
                        if row.get("latency_ms") not in {"", None}
                    ]
                ),
            }
            for phase, rows in sorted(embedding_rows_by_phase.items())
        }
        driver_setup_rows = [
            row
            for row in trace_rows
            if row.get("event_type") == "driver_setup_result"
        ]
        driver_setup_latencies = [
            float(row["latency_ms"])
            for row in driver_setup_rows
            if row.get("latency_ms") not in {"", None}
        ]
        driver_setup_by_case = {
            str(row.get("case_id") or ""): row for row in driver_setup_rows
        }
        manifest = json.loads(
            (self.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        case_rows: list[dict[str, Any]] = []
        case_ids = sorted(
            {
                str(row.get("case_id") or "")
                for row in (*per_event_rows, *question_rows, *provider_rows)
                if row.get("case_id")
            }
        )
        for case_id in case_ids:
            case_questions = [
                row for row in question_rows if row.get("case_id") == case_id
            ]
            case_events = [row for row in per_event_rows if row.get("case_id") == case_id]
            case_provider = [
                row for row in provider_rows if row.get("case_id") == case_id
            ]
            case_final_provider = [
                row for row in case_provider if row.get("successful_path")
            ]
            case_recovery_provider = [
                row for row in case_provider if not row.get("successful_path")
            ]
            case_provider_rollup = _provider_rollup(case_provider)
            case_final_rollup = _provider_rollup(case_final_provider)
            case_recovery_rollup = _provider_rollup(case_recovery_provider)
            case_cost = case_provider_rollup["estimated_cost_usd"]
            case_scores = [
                float(row["primary_score"])
                for row in case_questions
                if row.get("primary_score") not in {"", None}
            ]
            phase_costs: dict[str, float | None] = {}
            for phase in (
                "insertion",
                "consolidation",
                "retrieval",
                "answering",
                "grading",
            ):
                values = [
                    row.get("estimated_cost_usd")
                    for row in case_provider
                    if row.get("phase") == phase
                ]
                known_values = [float(value) for value in values if value is not None]
                phase_costs[f"{phase}_cost_usd"] = (
                    round(sum(known_values), 12)
                    if values and len(known_values) == len(values)
                    else (0.0 if not values else None)
                )
            case_rows.append(
                {
                    "case_id": case_id,
                    "event_count": len(case_events),
                    "question_count": len(case_questions),
                    "mean_score": (
                        sum(case_scores) / len(case_scores) if case_scores else None
                    ),
                    "system_error_question_count": sum(
                        bool(row.get("system_error_phase")) for row in case_questions
                    ),
                    "driver_setup_wall_latency_ms": driver_setup_by_case.get(
                        case_id,
                        {},
                    ).get("latency_ms", ""),
                    "insertion_wall_latency_ms": round(
                        sum(float(row.get("wall_latency_ms") or 0) for row in case_events),
                        3,
                    ),
                    "semantic_trace_io_latency_ms": (
                        round(
                            sum(
                                float(row["semantic_trace_io_latency_ms"])
                                for row in case_events
                            ),
                            3,
                        )
                        if case_events
                        and all(
                            row.get("semantic_trace_io_latency_ms")
                            not in {"", None}
                            for row in case_events
                        )
                        else None
                    ),
                    "semantic_trace_bytes_written": (
                        sum(
                            int(row["semantic_trace_bytes_written"])
                            for row in case_events
                        )
                        if case_events
                        and all(
                            row.get("semantic_trace_bytes_written")
                            not in {"", None}
                            for row in case_events
                        )
                        else None
                    ),
                    "insertion_latency_excluding_trace_io_ms": (
                        round(
                            sum(
                                float(
                                    row[
                                        "insertion_latency_excluding_trace_io_ms"
                                    ]
                                )
                                for row in case_events
                            ),
                            3,
                        )
                        if case_events
                        and all(
                            row.get("insertion_latency_excluding_trace_io_ms")
                            not in {"", None}
                            for row in case_events
                        )
                        else None
                    ),
                    "retrieval_wall_latency_ms": round(
                        sum(
                            float(row.get("retrieval_latency_ms") or 0)
                            for row in case_questions
                        ),
                        3,
                    ),
                    "answering_wall_latency_ms": round(
                        sum(
                            float(row.get("answer_latency_ms") or 0)
                            for row in case_questions
                        ),
                        3,
                    ),
                    **case_provider_rollup,
                    "actual_known_cost_usd": case_provider_rollup[
                        "known_cost_usd"
                    ],
                    "actual_estimated_cost_usd": case_provider_rollup[
                        "estimated_cost_usd"
                    ],
                    "final_known_cost_usd": case_final_rollup[
                        "known_cost_usd"
                    ],
                    "final_estimated_cost_usd": case_final_rollup[
                        "estimated_cost_usd"
                    ],
                    "recovery_known_cost_usd": case_recovery_rollup[
                        "known_cost_usd"
                    ],
                    "recovery_estimated_cost_usd": case_recovery_rollup[
                        "estimated_cost_usd"
                    ],
                    "cost_per_question_usd": (
                        float(case_cost) / len(case_questions)
                        if isinstance(case_cost, int | float)
                        and case_questions
                        else None
                    ),
                    **phase_costs,
                }
            )
        self._write_csv(self.output_dir / "metrics" / "per_case.csv", case_rows)
        summary = {
            "condition_id": manifest.get("condition_id"),
            "completed_cases": completed_cases,
            "failed_cases": failed_cases,
            "attention_required_cases": attention_required_cases,
            "question_count": len(question_rows),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "scores_by_scorer": {
                scorer_id: {
                    "grade_count": len(values),
                    "mean_score": sum(values) / len(values),
                }
                for scorer_id, values in sorted(scores_by_scorer.items())
            },
            "empty_retrieval_count": empty_retrievals,
            "retrieval_system_error_count": retrieval_system_errors,
            "memory_system_error_case_count": len(memory_system_error_cases),
            "memory_system_error_question_count": memory_system_error_questions,
            "framework_cache_mode": manifest.get("framework_cache_mode"),
            "framework_cache_usage": framework_cache_summary,
            "insertion_wall_latency": _latency_stats(insertion_latencies),
            "semantic_trace_io_wall_latency": _latency_stats(
                semantic_trace_io_latencies
            ),
            "semantic_trace_io_latency_sum_ms": (
                round(sum(semantic_trace_io_latencies), 3)
                if trace_io_metrics_available
                else None
            ),
            "semantic_trace_bytes_written": (
                sum(
                    int(row["semantic_trace_bytes_written"])
                    for row in per_event_rows
                )
                if trace_io_metrics_available
                else None
            ),
            "insertion_wall_latency_excluding_trace_io": _latency_stats(
                insertion_latencies_excluding_trace_io
            ),
            "retrieval_wall_latency": _latency_stats(retrieval_latencies),
            "answering_wall_latency": _latency_stats(answer_latencies),
            "grading_wall_latency_by_scorer": {
                scorer_id: _latency_stats(latencies)
                for scorer_id, latencies in sorted(
                    grading_latencies_by_scorer.items()
                )
            },
            "consolidation_wall_latency": _latency_stats(
                consolidation_latencies
            ),
            "checkpoint_wall_latency": _latency_stats(checkpoint_latencies),
            "driver_setup_wall_latency": _latency_stats(driver_setup_latencies),
            "driver_setup_error_count": sum(
                row.get("status") == "error" for row in driver_setup_rows
            ),
            "embedding_call_count": len(embedding_rows),
            "embedding_error_count": sum(
                row.get("status") == "error" for row in embedding_rows
            ),
            "embedding_latency_sum_ms": round(sum(embedding_latencies), 3),
            "embedding_wall_latency": _latency_stats(embedding_latencies),
            "embedding_phases": embedding_phases,
            "pricing": pricing.to_dict(),
            "actual_provider_usage": provider_summary,
            "final_successful_provider_usage": final_provider_summary,
            "recovery_overhead_provider_usage": recovery_provider_summary,
            "final_successful_provider_usage_by_phase": {
                phase: summarize_provider_calls(
                    [
                        row
                        for row in final_provider_rows
                        if row.get("phase") == phase
                    ]
                )
                for phase in (
                    "insertion",
                    "consolidation",
                    "retrieval",
                    "answering",
                    "grading",
                )
            },
            "recovery_overhead_provider_usage_by_phase": {
                phase: summarize_provider_calls(
                    [
                        row
                        for row in recovery_provider_rows
                        if row.get("phase") == phase
                    ]
                )
                for phase in (
                    "insertion",
                    "consolidation",
                    "retrieval",
                    "answering",
                    "grading",
                )
            },
            **provider_summary,
        }
        _write_json(
            self.output_dir / "metrics" / "summary.json",
            summary,
        )
        self._write_csv(
            self.output_dir / "metrics" / "overview.csv",
            [
                {
                    "condition_id": manifest.get("condition_id"),
                    "completed_cases": completed_cases,
                    "failed_cases": failed_cases,
                    "attention_required_cases": attention_required_cases,
                    "question_count": len(question_rows),
                    "mean_score": summary["mean_score"],
                    "retrieval_system_error_count": retrieval_system_errors,
                    "memory_system_error_case_count": len(
                        memory_system_error_cases
                    ),
                    "memory_system_error_question_count": (
                        memory_system_error_questions
                    ),
                    "provider_call_count": provider_summary["provider_call_count"],
                    "prompt_tokens": provider_summary["prompt_tokens"],
                    "cache_hit_tokens": provider_summary["cache_hit_tokens"],
                    "cache_miss_tokens": provider_summary["cache_miss_tokens"],
                    "provider_cache_hit_ratio": (
                        provider_summary["cache_hit_tokens"]
                        / (
                            provider_summary["cache_hit_tokens"]
                            + provider_summary["cache_miss_tokens"]
                        )
                        if provider_summary["cache_hit_tokens"]
                        + provider_summary["cache_miss_tokens"]
                        > 0
                        else None
                    ),
                    "framework_cache_mode": manifest.get("framework_cache_mode"),
                    "framework_lm_cache_hits": framework_cache_summary[
                        "lm_cache_hits"
                    ],
                    "framework_operator_cache_hits": framework_cache_summary[
                        "operator_cache_hits"
                    ],
                    "framework_physical_total_tokens": framework_cache_summary[
                        "physical_total_tokens"
                    ],
                    "framework_virtual_total_tokens": framework_cache_summary[
                        "virtual_total_tokens"
                    ],
                    "completion_tokens": provider_summary["completion_tokens"],
                    "reasoning_tokens": provider_summary["reasoning_tokens"],
                    "estimated_cost_usd": provider_summary["estimated_cost_usd"],
                    "known_cost_usd": provider_summary["known_cost_usd"],
                    "usage_complete": provider_summary["usage_complete"],
                    "final_estimated_cost_usd": final_provider_summary[
                        "estimated_cost_usd"
                    ],
                    "final_known_cost_usd": final_provider_summary[
                        "known_cost_usd"
                    ],
                    "recovery_estimated_cost_usd": recovery_provider_summary[
                        "estimated_cost_usd"
                    ],
                    "recovery_known_cost_usd": recovery_provider_summary[
                        "known_cost_usd"
                    ],
                    "driver_setup_mean_latency_ms": _latency_stats(
                        driver_setup_latencies
                    )["mean_ms"],
                    "driver_setup_median_latency_ms": _latency_stats(
                        driver_setup_latencies
                    )["median_ms"],
                    "driver_setup_p95_latency_ms": _latency_stats(
                        driver_setup_latencies
                    )["p95_ms"],
                    "driver_setup_error_count": summary[
                        "driver_setup_error_count"
                    ],
                    "embedding_call_count": summary["embedding_call_count"],
                    "embedding_error_count": summary["embedding_error_count"],
                    "embedding_latency_sum_ms": summary[
                        "embedding_latency_sum_ms"
                    ],
                    "insertion_mean_latency_ms": _latency_stats(
                        insertion_latencies
                    )["mean_ms"],
                    "insertion_p95_latency_ms": _latency_stats(
                        insertion_latencies
                    )["p95_ms"],
                    "insertion_median_latency_ms": _latency_stats(
                        insertion_latencies
                    )["median_ms"],
                    "insertion_excluding_semantic_trace_mean_ms": summary[
                        "insertion_wall_latency_excluding_trace_io"
                    ]["mean_ms"],
                    "insertion_excluding_semantic_trace_median_ms": summary[
                        "insertion_wall_latency_excluding_trace_io"
                    ]["median_ms"],
                    "insertion_excluding_semantic_trace_p95_ms": summary[
                        "insertion_wall_latency_excluding_trace_io"
                    ]["p95_ms"],
                    "insertion_excluding_semantic_trace_max_ms": summary[
                        "insertion_wall_latency_excluding_trace_io"
                    ]["max_ms"],
                    "retrieval_mean_latency_ms": _latency_stats(
                        retrieval_latencies
                    )["mean_ms"],
                    "retrieval_median_latency_ms": _latency_stats(
                        retrieval_latencies
                    )["median_ms"],
                    "retrieval_p95_latency_ms": _latency_stats(
                        retrieval_latencies
                    )["p95_ms"],
                    "answering_mean_latency_ms": _latency_stats(
                        answer_latencies
                    )["mean_ms"],
                    "answering_median_latency_ms": _latency_stats(
                        answer_latencies
                    )["median_ms"],
                    "answering_p95_latency_ms": _latency_stats(
                        answer_latencies
                    )["p95_ms"],
                }
            ],
        )
        self._write_csv(
            self.output_dir / "metrics" / "reliability.csv",
            [
                {
                    "condition_id": manifest.get("condition_id"),
                    "completed_cases": completed_cases,
                    "failed_cases": failed_cases,
                    "attention_required_cases": attention_required_cases,
                    "restore_count": sum(
                        row.get("operation") == "restore" for row in checkpoint_rows
                    ),
                    "provider_error_count": provider_summary[
                        "provider_error_count"
                    ],
                    "driver_setup_error_count": summary[
                        "driver_setup_error_count"
                    ],
                    "embedding_error_count": summary["embedding_error_count"],
                    "usage_complete": provider_summary["usage_complete"],
                    "empty_retrieval_count": empty_retrievals,
                    "retrieval_system_error_count": retrieval_system_errors,
                    "memory_system_error_case_count": len(
                        memory_system_error_cases
                    ),
                    "memory_system_error_question_count": (
                        memory_system_error_questions
                    ),
                    "replay_event_count": sum(
                        bool(row.get("replayed")) for row in per_event_rows
                    ),
                    "unit_attempt_failure_count": sum(
                        UnitAttemptStore(
                            case_dir / "control"
                        ).retryable_failure_count()
                        for case_dir in (self.output_dir / "cases").glob("*")
                    ),
                    "recovery_provider_call_count": recovery_provider_summary[
                        "provider_call_count"
                    ],
                }
            ],
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            if not fields:
                return
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


__all__ = [
    "ArtifactContractError",
    "BenchmarkArtifactStore",
    "MAX_UNIT_ATTEMPTS",
    "UnitAttemptExhausted",
]
