"""Artifacts for the formal agent-memory Zep LOCOMO benchmark."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from agent_memory.evaluation.types import BenchmarkEvent, BenchmarkQuestion
from agent_memory.evaluation.zep.answering import AnswerRecord
from agent_memory.evaluation.zep.scoring import OfficialGrade, ZepJudgeGrade

INGESTION_METRIC_COLUMNS = (
    "event_id",
    "row_number",
    "latency_ms",
    "episodes_rows",
    "entities_rows",
    "facts_rows",
    "physical_prompt_tokens",
    "physical_completion_tokens",
    "physical_total_tokens",
    "virtual_prompt_tokens",
    "virtual_completion_tokens",
    "virtual_total_tokens",
    "cache_hits",
)

RETRIEVAL_METRIC_COLUMNS = (
    "question_id",
    "category",
    "latency_ms",
    "entity_count",
    "fact_count",
    "entity_latency_ms",
    "fact_latency_ms",
    "bfs_origins",
)

LLM_CALL_METRIC_COLUMNS = (
    "trace_id",
    "phase",
    "operator",
    "prompt_name",
    "status",
    "model",
    "latency_ms",
    "physical_prompt_tokens",
    "physical_completion_tokens",
    "physical_total_tokens",
    "prompt_path",
    "raw_output_path",
)


@dataclass(frozen=True)
class PricingSnapshot:
    """Immutable DeepSeek pricing used for reproducible cost estimates."""

    effective_date: str
    source_url: str
    cache_hit_input_per_million_usd: Decimal
    cache_miss_input_per_million_usd: Decimal
    output_per_million_usd: Decimal

    @classmethod
    def deepseek_2026_07_17(cls) -> "PricingSnapshot":
        """Return the pricing snapshot shared with the native baseline."""

        return cls(
            effective_date="2026-07-17",
            source_url="https://api-docs.deepseek.com/quick_start/pricing",
            cache_hit_input_per_million_usd=Decimal("0.0028"),
            cache_miss_input_per_million_usd=Decimal("0.14"),
            output_per_million_usd=Decimal("0.28"),
        )

    def estimate_cost_usd(
        self,
        *,
        cache_hit_input_tokens: int | None,
        cache_miss_input_tokens: int | None,
        output_tokens: int | None,
    ) -> Decimal | None:
        """Estimate cost only when the provider returned every required counter."""

        if (
            cache_hit_input_tokens is None
            or cache_miss_input_tokens is None
            or output_tokens is None
        ):
            return None
        million = Decimal(1_000_000)
        return (
            Decimal(cache_hit_input_tokens)
            * self.cache_hit_input_per_million_usd
            + Decimal(cache_miss_input_tokens)
            * self.cache_miss_input_per_million_usd
            + Decimal(output_tokens) * self.output_per_million_usd
        ) / million

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe pricing manifest."""

        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }


class ArtifactStore:
    """Write one isolated, inspectable Zep LOCOMO run."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    @classmethod
    def create(cls, output_dir: Path) -> "ArtifactStore":
        """Create a new artifact root without overwriting an existing run."""

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "input",
            "insertion",
            "retrieval",
            "answers",
            "grades",
            "trace",
            "checkpoint",
            "diagnostics",
            "metrics",
        ):
            (output_dir / name).mkdir(exist_ok=True)
        return cls(output_dir)

    @property
    def trace_dir(self) -> Path:
        """Return the provider and semantic trace directory."""

        return self.output_dir / "trace"

    def write_manifest(self, payload: Mapping[str, Any]) -> Path:
        """Write the immutable run contract."""

        return _write_json(self.output_dir / "manifest.json", payload)

    def write_status(
        self,
        *,
        status: str,
        phase: str,
        error: BaseException | None = None,
    ) -> Path:
        """Write the current terminal or in-progress run status."""

        payload: dict[str, Any] = {"status": status, "phase": phase}
        if error is not None:
            payload.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        return _write_json(self.output_dir / "status.json", payload)

    def write_failure(self, *, phase: str, error: BaseException) -> Path:
        """Write a durable failure marker without swallowing the exception."""

        return _write_json(
            self.output_dir / "diagnostics" / "failure.json",
            {
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def write_inputs(
        self,
        events: Sequence[BenchmarkEvent],
        questions: Sequence[BenchmarkQuestion],
    ) -> dict[str, Path]:
        """Write the exact normalized events and eligible questions."""

        return {
            "events": _write_jsonl(
                self.output_dir / "input" / "events.jsonl",
                [_event_payload(event) for event in events],
            ),
            "questions": _write_jsonl(
                self.output_dir / "input" / "questions.jsonl",
                [_question_payload(question) for question in questions],
            ),
        }

    def write_ingestion_metrics(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> Path:
        """Write one measured row per completed insertion."""

        return _write_csv(
            self.output_dir / "insertion" / "metrics.csv",
            rows,
            INGESTION_METRIC_COLUMNS,
        )

    def write_retrievals(self, rows: Sequence[Any]) -> Path:
        """Write one complete two-channel retrieval artifact per question."""

        return _write_jsonl(
            self.output_dir / "retrieval" / "results.jsonl",
            [_payload(row) for row in rows],
        )

    def write_retrieval_metrics(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> Path:
        """Write compact retrieval latency and BFS evidence."""

        return _write_csv(
            self.output_dir / "retrieval" / "metrics.csv",
            rows,
            RETRIEVAL_METRIC_COLUMNS,
        )

    def write_answers(self, rows: Sequence[AnswerRecord]) -> Path:
        """Write the single generated answer stream shared by both scorers."""

        return _write_jsonl(
            self.output_dir / "answers" / "results.jsonl",
            [row.to_dict() for row in rows],
        )

    def write_grades(
        self,
        official: Sequence[OfficialGrade],
        zep_judge: Sequence[ZepJudgeGrade],
    ) -> dict[str, Path]:
        """Write deterministic and LLM-judge grades with independent summaries."""

        official_summary = _official_summary(official)
        judge_summary = _judge_summary(zep_judge)
        written = {
            "official": _write_jsonl(
                self.output_dir / "grades" / "official.jsonl",
                [grade.to_dict() for grade in official],
            ),
            "official_summary": _write_csv(
                self.output_dir / "grades" / "official_summary.csv",
                [official_summary],
                tuple(official_summary),
            ),
            "zep_judge": _write_jsonl(
                self.output_dir / "grades" / "zep_judge.jsonl",
                [grade.to_dict() for grade in zep_judge],
            ),
            "zep_judge_summary": _write_csv(
                self.output_dir / "grades" / "zep_judge_summary.csv",
                [judge_summary],
                tuple(judge_summary),
            ),
        }
        written["summary"] = _write_json(
            self.output_dir / "grades" / "summary.json",
            {"official": official_summary, "zep_judge": judge_summary},
        )
        return written

    def write_checkpoint(
        self,
        *,
        state: bytes,
        metadata: Mapping[str, Any],
        retrieval_validation: Mapping[str, Any],
    ) -> dict[str, Path]:
        """Write runtime state and checkpoint/retrieval round-trip evidence."""

        state_path = self.output_dir / "checkpoint" / "state.pkl"
        state_path.write_bytes(state)
        return {
            "state": state_path,
            "metadata": _write_json(
                self.output_dir / "checkpoint" / "metadata.json",
                metadata,
            ),
            "retrieval_validation": _write_json(
                self.output_dir / "checkpoint" / "retrieval_validation.json",
                retrieval_validation,
            ),
        }

    def write_metrics(
        self,
        *,
        pricing: PricingSnapshot | None = None,
    ) -> dict[str, Path]:
        """Derive phase usage, latency, and cost from real provider traces."""

        pricing = pricing or PricingSnapshot.deepseek_2026_07_17()
        events = _read_trace_events(self.trace_dir / "events.jsonl")
        llm_calls = [
            _llm_call_row(event)
            for event in events
            if event.get("event_type") in {"llm_call", "llm_batch_error"}
            and int(event.get("llm_item_index", 0)) == 0
        ]
        provider_events = [
            event for event in events if event.get("event_type") == "provider_usage"
        ]
        phases = {
            phase: _phase_metrics(
                phase,
                llm_calls=llm_calls,
                provider_events=provider_events,
                output_dir=self.output_dir,
                pricing=pricing,
            )
            for phase in ("insertion", "answering", "grading")
        }
        return {
            "llm_calls": _write_csv(
                self.output_dir / "metrics" / "llm_calls.csv",
                llm_calls,
                LLM_CALL_METRIC_COLUMNS,
            ),
            "summary": _write_json(
                self.output_dir / "metrics" / "summary.json",
                {"pricing": pricing.to_dict(), "phases": phases},
            ),
        }


def _event_payload(event: BenchmarkEvent) -> dict[str, Any]:
    return {
        "sample_id": event.sample_id,
        "event_id": event.event_id,
        "speaker": event.speaker,
        "text": event.text,
        "session_id": event.session_id,
        "timestamp": event.timestamp,
        "metadata": dict(event.metadata),
    }


def _question_payload(question: BenchmarkQuestion) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "sample_id": question.sample_id,
        "question": question.question,
        "gold_answer": question.gold_answer,
        "evidence_event_ids": list(question.evidence_event_ids),
        "category": question.category,
        "metadata": dict(question.metadata),
    }


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported artifact row: {type(value).__name__}")


def _official_summary(grades: Sequence[OfficialGrade]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len(grades),
        "mean_score": (
            round(sum(grade.score for grade in grades) / len(grades), 6)
            if grades
            else ""
        ),
        "categories": ";".join(str(value) for value in sorted({g.category for g in grades})),
    }
    for category in range(1, 6):
        selected = [grade for grade in grades if grade.category == category]
        summary[f"category_{category}_count"] = len(selected)
        summary[f"category_{category}_mean_score"] = (
            round(sum(grade.score for grade in selected) / len(selected), 6)
            if selected
            else ""
        )
    return summary


def _judge_summary(grades: Sequence[ZepJudgeGrade]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len(grades),
        "accuracy": (
            round(sum(grade.is_correct for grade in grades) / len(grades), 6)
            if grades
            else ""
        ),
        "categories": ";".join(str(value) for value in sorted({g.category for g in grades})),
    }
    for category in range(1, 5):
        selected = [grade for grade in grades if grade.category == category]
        summary[f"category_{category}_count"] = len(selected)
        summary[f"category_{category}_accuracy"] = (
            round(sum(grade.is_correct for grade in selected) / len(selected), 6)
            if selected
            else ""
        )
    return summary


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Trace event {line_number} is not a JSON object")
        events.append(value)
    return events


def _llm_call_row(event: Mapping[str, Any]) -> dict[str, Any]:
    latency = event.get("latency_sec")
    latency_ms = (
        round(float(latency) * 1000, 3)
        if isinstance(latency, (int, float, str)) and str(latency)
        else ""
    )
    return {
        "trace_id": event.get("trace_id", ""),
        "phase": event.get("phase", ""),
        "operator": event.get("operator", ""),
        "prompt_name": event.get("prompt_name", ""),
        "status": "error" if event.get("event_type") == "llm_batch_error" else "success",
        "model": event.get("model", ""),
        "latency_ms": latency_ms,
        "physical_prompt_tokens": event.get("usage_physical_prompt_tokens", ""),
        "physical_completion_tokens": event.get(
            "usage_physical_completion_tokens", ""
        ),
        "physical_total_tokens": event.get("usage_physical_total_tokens", ""),
        "prompt_path": event.get("prompt_path", ""),
        "raw_output_path": event.get("raw_output_path", ""),
    }


def _phase_metrics(
    phase: str,
    *,
    llm_calls: Sequence[Mapping[str, Any]],
    provider_events: Sequence[Mapping[str, Any]],
    output_dir: Path,
    pricing: PricingSnapshot,
) -> dict[str, Any]:
    calls = [row for row in llm_calls if row.get("phase") == phase]
    usage_events = [event for event in provider_events if event.get("phase") == phase]
    costs: list[Decimal | None] = []
    reasoning_tokens = 0
    for event in usage_events:
        raw_usage = _read_provider_usage(output_dir, event)
        reasoning_tokens += int(
            ((raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            or 0
        )
        costs.append(
            pricing.estimate_cost_usd(
                cache_hit_input_tokens=_optional_int(
                    event.get("provider_prompt_cache_hit_tokens")
                ),
                cache_miss_input_tokens=_optional_int(
                    event.get("provider_prompt_cache_miss_tokens")
                ),
                output_tokens=_optional_int(event.get("provider_completion_tokens")),
            )
        )
    return {
        "llm_batch_count": len(calls),
        "llm_error_count": sum(row.get("status") == "error" for row in calls),
        "provider_response_count": len(usage_events),
        "llm_latency_ms": round(
            sum(float(row.get("latency_ms") or 0) for row in calls), 3
        ),
        "prompt_tokens": sum(
            int(event.get("provider_prompt_tokens") or 0) for event in usage_events
        ),
        "cache_hit_tokens": sum(
            int(event.get("provider_prompt_cache_hit_tokens") or 0)
            for event in usage_events
        ),
        "cache_miss_tokens": sum(
            int(event.get("provider_prompt_cache_miss_tokens") or 0)
            for event in usage_events
        ),
        "completion_tokens": sum(
            int(event.get("provider_completion_tokens") or 0)
            for event in usage_events
        ),
        "reasoning_tokens": reasoning_tokens,
        "estimated_cost_usd": (
            float(round(sum((cost for cost in costs if cost is not None), Decimal(0)), 12))
            if costs and all(cost is not None for cost in costs)
            else (0.0 if not costs else None)
        ),
    }


def _read_provider_usage(
    output_dir: Path,
    event: Mapping[str, Any],
) -> Mapping[str, Any]:
    path_value = event.get("provider_raw_usage_path")
    if not isinstance(path_value, str) or not path_value:
        return {}
    value = json.loads((output_dir / path_value).read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    return path


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, type):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, type):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    runtime_value = cast(Any, value)
    if hasattr(runtime_value, "item"):
        return _json_safe(runtime_value.item())
    if hasattr(runtime_value, "isoformat"):
        return runtime_value.isoformat()
    return _json_default(runtime_value)


__all__ = ["ArtifactStore", "PricingSnapshot"]
