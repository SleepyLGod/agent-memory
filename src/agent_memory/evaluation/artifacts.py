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

from agent_memory.evaluation.bundle import BenchmarkBundle, write_bundle
from agent_memory.evaluation.pricing import PricingSnapshot
from agent_memory.evaluation.provenance import collect_runtime_provenance
from agent_memory.evaluation.trace_metrics import (
    normalize_provider_calls,
    summarize_provider_calls,
)
from agent_memory.evaluation.types import BenchmarkCase, BenchmarkEvent
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
        for name in ("retrieval.jsonl", "answers.jsonl", "grades.jsonl"):
            (case_dir / name).write_text("", encoding="utf-8")
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

    def save_checkpoint(
        self,
        *,
        bundle: BenchmarkBundle,
        case: BenchmarkCase,
        system_contract: MemorySystemContract,
        completed_events: tuple[BenchmarkEvent, ...],
        driver: MemorySystemDriver,
    ) -> BenchmarkCheckpoint:
        """Publish driver state only after one complete input session succeeds."""

        if not completed_events:
            raise ValueError("checkpoint requires at least one completed event")
        completed_ids = tuple(event.event_id for event in completed_events)
        expected_prefix = tuple(event.event_id for event in case.events[: len(completed_ids)])
        if completed_ids != expected_prefix:
            raise ValueError("checkpoint events must be a prefix of the case input")

        checkpoint_root = self.case_dir(case.case_id) / "checkpoints"
        checkpoint_id = (
            f"events-{len(completed_events):06d}-"
            f"{_event_fingerprint(completed_events[-1])[:12]}"
        )
        staging = checkpoint_root / "staging" / f"{checkpoint_id}-{uuid4().hex}"
        driver_metadata = driver.save_state(staging / "driver")
        manifest = {
            "schema_version": 2,
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
            "maintenance_fingerprint": system_contract.maintenance_fingerprint,
            "thinking_enabled": system_contract.thinking_enabled,
            "consolidation_mode": system_contract.consolidation_mode,
            "framework_cache_mode": system_contract.framework_cache_mode,
            "completed_session_id": completed_events[-1].session_id,
            "completed_event_ids": list(completed_ids),
            "completed_event_fingerprints": [
                _event_fingerprint(event) for event in completed_events
            ],
            "driver_state": dict(driver_metadata),
        }
        _write_json(staging / "manifest.json", manifest)
        snapshot = checkpoint_root / "snapshots" / checkpoint_id
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            existing = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("checkpoint ID collision contains different state")
        else:
            os.replace(staging, snapshot)
        _write_json_atomic(
            checkpoint_root / "current.json",
            {"schema_version": 1, "checkpoint_id": checkpoint_id},
        )
        return BenchmarkCheckpoint(snapshot, completed_events)

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
            raise ValueError("checkpoint current pointer is missing checkpoint_id")
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
            raise ValueError(
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
            raise ValueError(
                "checkpoint does not match current run contract: "
                + ", ".join(mismatched)
            )
        event_ids = manifest.get("completed_event_ids")
        fingerprints = manifest.get("completed_event_fingerprints")
        if not isinstance(event_ids, list) or not isinstance(fingerprints, list):
            raise ValueError("checkpoint is missing completed event prefix evidence")
        completed = case.events[: len(event_ids)]
        if [event.event_id for event in completed] != event_ids or [
            _event_fingerprint(event) for event in completed
        ] != fingerprints:
            raise ValueError("checkpoint completed event prefix does not match input")
        return BenchmarkCheckpoint(snapshot, tuple(completed))

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
        empty_retrievals = 0
        retrieval_system_errors = 0
        memory_system_error_questions = 0
        memory_system_error_cases: set[str] = set()
        input_questions = {
            row["question_id"]: row
            for row in _read_jsonl(self.output_dir / "input" / "questions.jsonl")
        }
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
                question_rows.append(
                    {
                        "case_id": status.get("case_id"),
                        "question_id": question_id,
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
        embedding_rows = [
            {
                "trace_id": row.get("trace_id", ""),
                "case_id": row.get("case_id", ""),
                "event_id": row.get("event_id", ""),
                "question_id": row.get("question_id", ""),
                "phase": row.get("phase", ""),
                "operation": row.get("operation", ""),
                "attempt": row.get("attempt", ""),
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
        self._write_csv(
            self.output_dir / "metrics" / "embedding_usage.csv",
            embedding_rows,
        )
        embedding_by_event: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        embedding_by_question: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in embedding_rows:
            case_id = str(row.get("case_id") or "")
            event_id = str(row.get("event_id") or "")
            question_id = str(row.get("question_id") or "")
            attempt = int(row.get("attempt") or 1)
            if event_id:
                embedding_by_event.setdefault(
                    (case_id, event_id, attempt),
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
        self._write_csv(self.output_dir / "metrics" / "per_question.csv", question_rows)
        self._write_csv(self.output_dir / "metrics" / "per_grade.csv", grade_rows)
        pricing = PricingSnapshot.deepseek_2026_07_17()
        provider_rows = normalize_provider_calls(
            trace_rows,
            output_dir=self.output_dir,
            pricing=pricing,
        )
        self._write_csv(
            self.output_dir / "metrics" / "provider_usage.csv", provider_rows
        )
        provider_by_event: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for row in provider_rows:
            key = (
                str(row.get("case_id") or ""),
                str(row.get("event_id") or ""),
                int(row.get("attempt") or 1),
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
            calls = provider_by_event.get((case_id, event_id, attempt), [])
            embedding_calls = embedding_by_event.get(
                (case_id, event_id, attempt),
                [],
            )
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
                    "replayed": occurrence > 1,
                    "wall_latency_ms": event.get("latency_ms", ""),
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
            session_costs = [
                float(row["estimated_cost_usd"])
                for row in rows
                if row.get("estimated_cost_usd") is not None
            ]
            session_reasoning = [
                int(row["reasoning_tokens"])
                for row in rows
                if row.get("reasoning_tokens") is not None
            ]
            consolidation_rows.append(
                {
                    "case_id": case_id,
                    "session_id": session_id,
                    "event_count": len(rows),
                    "insertion_wall_latency_ms": round(
                        sum(float(row.get("wall_latency_ms") or 0) for row in rows),
                        3,
                    ),
                    "consolidation_wall_latency_ms": consolidation.get(
                        "latency_ms", 0
                    ),
                    "provider_call_count": sum(
                        int(row.get("provider_call_count") or 0) for row in rows
                    ),
                    "provider_latency_sum_ms": round(
                        sum(
                            float(row.get("provider_latency_sum_ms") or 0)
                            for row in rows
                        ),
                        3,
                    ),
                    "prompt_tokens": sum(
                        int(row.get("prompt_tokens") or 0) for row in rows
                    ),
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
                        sum(session_reasoning)
                        if len(session_reasoning) == len(rows)
                        else None
                    ),
                    "estimated_cost_usd": (
                        round(sum(session_costs), 12)
                        if len(session_costs) == len(rows)
                        else None
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
                "completed_event_count": row.get("completed_event_count", ""),
                "checkpoint_bytes": row.get("checkpoint_bytes", ""),
                "wall_latency_ms": row.get("latency_ms", ""),
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
        insertion_latencies = [
            float(row["wall_latency_ms"])
            for row in per_event_rows
            if row.get("wall_latency_ms") not in {"", None}
        ]
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
            case_provider_rollup = _provider_rollup(case_provider)
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
            "insertion_wall_latency": _latency_stats(insertion_latencies),
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
                    "completion_tokens": provider_summary["completion_tokens"],
                    "reasoning_tokens": provider_summary["reasoning_tokens"],
                    "estimated_cost_usd": provider_summary["estimated_cost_usd"],
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


__all__ = ["BenchmarkArtifactStore"]
