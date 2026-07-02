"""Importable ClaudeMemory LOCOMO evaluation runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time
from typing import Any

import pandas as pd

import agent_memory as am
from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.datasets.locomo import DEFAULT_LOCOMO_URL, ensure_locomo_dataset
from agent_memory.evaluation.claude_memory.bindings import event_to_claude_log_row
from agent_memory.evaluation.diagnostics import (
    LLM_ANOMALY_COLUMNS,
    PROVIDER_USAGE_COLUMNS,
    PROVIDER_USAGE_SUMMARY_COLUMNS,
    build_cause_trace_rows,
    build_llm_anomaly_rows,
    build_provider_usage_rows,
    build_provider_usage_summary_rows,
)
from agent_memory.evaluation.locomo import (
    eligible_questions,
    load_locomo_sample,
    select_events,
)
from agent_memory.evaluation.metrics import (
    duplicate_name_count,
    duplicate_name_extra_rows,
    frame_text,
    question_metric_row,
    summarize_question_metrics,
)
from agent_memory.evaluation.types import BenchmarkEvent, BenchmarkQuestion
from agent_memory.tracing.semantic import semantic_trace_scope

ANSWER_MAX_TOKENS = 256
ANSWER_SYSTEM_PROMPT = (
    "Answer the benchmark question using only the retrieved memory context. "
    "If the context is insufficient, answer 'No information available.'."
)
CHECKPOINT_SCHEMA_VERSION = 1
RUN_MARKER_SCHEMA_VERSION = 1
RUN_MARKER_FILENAME = ".agent-memory-locomo-run.json"
BENCHMARK_CONTRACT = "message_with_event_context:v1"
POLICY_CONTRACT = "claude_memory_policy:v1"
SCORER_CONTRACT = "locomo_official_compatible_category_logic:v1"
USAGE_FIELDS = (
    "physical_prompt_tokens",
    "physical_completion_tokens",
    "physical_total_tokens",
    "virtual_prompt_tokens",
    "virtual_completion_tokens",
    "virtual_total_tokens",
    "cache_hits",
)
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")
CSV_FORMULA_LEADING_CHARS = " \t\r\n"


@dataclass(frozen=True)
class ClaudeMemoryLocomoRunConfig:
    """Configuration for one ClaudeMemory LOCOMO evaluation run."""

    sample_index: int
    row_limit: int
    question_limit: int
    model: str
    output_dir: Path
    locomo_cache_path: Path
    existing_output_dir: Path | None = None
    trust_existing_output_dir: bool = False
    trace: bool = False
    answer: bool = False
    resume: bool = False
    restore_csv_state: bool = False
    restore_artifact_csv_state: bool = False


def reset_output_dir(output_dir: Path, *, safe: bool = False) -> None:
    """Create a clean benchmark output directory."""

    if not safe:
        raise ValueError("reset_output_dir requires prior output path safety validation")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"Unsafe --output-dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not is_run_owned_output_dir(output_dir):
            raise SystemExit(
                "Refusing to delete non-empty --output-dir without "
                f"{RUN_MARKER_FILENAME}: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_marker(output_dir)


def run_marker_path(output_dir: Path) -> Path:
    """Return the marker path proving a directory is owned by this runner."""

    return output_dir / RUN_MARKER_FILENAME


def write_run_marker(output_dir: Path) -> None:
    """Write a marker that permits later safe cleanup of this run directory."""

    payload = {
        "schema_version": RUN_MARKER_SCHEMA_VERSION,
        "tool": "agent-memory.claude-memory-locomo",
    }
    write_text_atomic(run_marker_path(output_dir), json.dumps(payload, indent=2))


def is_run_owned_output_dir(output_dir: Path) -> bool:
    """Return whether output_dir has a valid runner ownership marker."""

    marker_path = run_marker_path(output_dir)
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("schema_version") == RUN_MARKER_SCHEMA_VERSION
        and payload.get("tool") == "agent-memory.claude-memory-locomo"
    )


def csv_safe_value(value: Any) -> Any:
    """Return a spreadsheet-safe value for CSV artifact cells."""

    if isinstance(value, str) and value.lstrip(CSV_FORMULA_LEADING_CHARS).startswith(
        CSV_FORMULA_PREFIXES
    ):
        return f"'{value}"
    return value


def csv_safe_frame(frame: Any) -> Any:
    """Return a copy with formula-like strings escaped for CSV artifacts."""

    if not isinstance(frame, pd.DataFrame):
        return frame
    safe = frame.copy()
    for column in safe.select_dtypes(include=("object", "string")).columns:
        safe[column] = safe[column].map(csv_safe_value)
    return safe


def write_csv(name: str, frame: Any, output_dir: Path) -> Path:
    """Write a DataFrame-like object to CSV and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    csv_safe_frame(frame).to_csv(csv_path, index=False)
    return csv_path


def write_state_jsonl(name: str, frame: Any, output_dir: Path) -> Path:
    """Write raw machine-readable state rows to JSONL and return its path."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("write_state_jsonl requires a pandas DataFrame")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{name}.jsonl"
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    write_jsonl_atomic(jsonl_path, records)
    return jsonl_path


def checkpoint_dir(output_dir: Path) -> Path:
    """Return the benchmark checkpoint directory for one run."""

    return output_dir / "checkpoint"


def checkpoint_snapshots_dir(output_dir: Path) -> Path:
    """Return the directory containing immutable checkpoint snapshots."""

    return checkpoint_dir(output_dir) / "snapshots"


def checkpoint_current_path(output_dir: Path) -> Path:
    """Return the atomic pointer to the current checkpoint snapshot."""

    return checkpoint_dir(output_dir) / "current.json"


def checkpoint_manifest_path(output_dir: Path) -> Path:
    """Return the manifest path for the current checkpoint snapshot."""

    return current_checkpoint_snapshot_dir(output_dir) / "manifest.json"


def current_checkpoint_snapshot_dir(output_dir: Path) -> Path:
    """Return the snapshot directory pointed to by current.json."""

    pointer_path = checkpoint_current_path(output_dir)
    if not pointer_path.exists():
        raise SystemExit(f"--resume requires an existing checkpoint: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise SystemExit("Checkpoint current pointer is missing checkpoint_id")
    return checkpoint_snapshots_dir(output_dir) / checkpoint_id


def write_text_atomic(path: Path, content: str) -> None:
    """Write text via replace so readers never see a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def write_bytes_atomic(path: Path, content: bytes) -> None:
    """Write bytes via replace so readers never see a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL rows atomically."""

    content = "".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows)
    write_text_atomic(path, content)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, returning an empty list for an absent file."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def path_contains(parent: Path, child: Path) -> bool:
    """Return whether parent is child or one of child's ancestors."""

    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_paths(*, output_dir: Path, source_run_dir: Path | None) -> None:
    """Reject destructive output paths and source/output overlaps."""

    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    root = Path(output_dir.anchor).resolve()
    unsafe_paths = {cwd, home, root}
    if output_dir in unsafe_paths:
        raise SystemExit(f"Unsafe --output-dir would delete a broad path: {output_dir}")
    if source_run_dir is None:
        return
    if path_contains(output_dir, source_run_dir) or path_contains(source_run_dir, output_dir):
        raise SystemExit(
            "--output-dir and --existing-output-dir must not be the same path "
            "or nested inside each other"
        )


def stable_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for JSON-compatible benchmark metadata."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_input_digest(
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
) -> str:
    """Return a digest of benchmark input content, not just event/question ids."""

    return stable_digest(
        {
            "events": [
                {
                    "sample_id": event.sample_id,
                    "event_id": event.event_id,
                    "speaker": event.speaker,
                    "text": event.text,
                    "session_id": event.session_id,
                    "timestamp": event.timestamp,
                }
                for event in events
            ],
            "questions": [
                {
                    "question_id": question.question_id,
                    "sample_id": question.sample_id,
                    "question": question.question,
                    "gold_answer": question.gold_answer,
                    "evidence_event_ids": list(question.evidence_event_ids),
                    "category": question.category,
                }
                for question in questions
            ],
        }
    )


def benchmark_event_fingerprints(
    events: Sequence[BenchmarkEvent],
) -> tuple[dict[str, Any], ...]:
    """Return event content fingerprints for checkpoint prefix validation."""

    return tuple(
        {
            "sample_id": event.sample_id,
            "event_id": event.event_id,
            "speaker": event.speaker,
            "text": event.text,
            "session_id": event.session_id,
            "timestamp": event.timestamp,
        }
        for event in events
    )


def checkpoint_contract_digest() -> str:
    """Return the current benchmark/checkpoint contract digest."""

    return stable_digest(
        {
            "answer_max_tokens": ANSWER_MAX_TOKENS,
            "answer_system_prompt": ANSWER_SYSTEM_PROMPT,
            "benchmark_contract": BENCHMARK_CONTRACT,
            "policy_contract": POLICY_CONTRACT,
            "scorer_contract": SCORER_CONTRACT,
        }
    )


def trace_event_count(trace_dir: Path | None) -> int:
    """Return the current number of trace event rows."""

    if trace_dir is None:
        return 0
    events_path = trace_dir / "events.jsonl"
    if not events_path.exists():
        return 0
    return sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip())


def recovery_path(output_dir: Path) -> Path:
    """Return the recovery diagnostics path."""

    return output_dir / "diagnostics" / "recovery.json"


def read_recovery_trace_ranges(output_dir: Path) -> list[tuple[int, int]]:
    """Return trace ranges that should be excluded from final successful metrics."""

    path = recovery_path(output_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    ranges: list[tuple[int, int]] = []
    for item in payload.get("excluded_trace_event_ranges", []):
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    return ranges


def merge_trace_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted non-overlapping trace ranges."""

    sorted_ranges = sorted((start, end) for start, end in ranges if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in sorted_ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def trace_exclusion_ranges(
    *,
    output_dir: Path,
    trace_dir: Path | None,
    checkpoint_trace_event_count: int,
) -> list[tuple[int, int]]:
    """Return trace ranges from failed attempts that final metrics should ignore."""

    ranges = read_recovery_trace_ranges(output_dir)
    current_trace_event_count = trace_event_count(trace_dir)
    if current_trace_event_count > checkpoint_trace_event_count:
        ranges.append((checkpoint_trace_event_count, current_trace_event_count))
    return merge_trace_ranges(ranges)


def usage_snapshot() -> dict[str, float | int]:
    """Return current LOTUS token/cache usage counters."""

    import lotus

    snapshot: dict[str, float | int] = {
        "physical_prompt_tokens": 0,
        "physical_completion_tokens": 0,
        "physical_total_tokens": 0,
        "virtual_prompt_tokens": 0,
        "virtual_completion_tokens": 0,
        "virtual_total_tokens": 0,
        "cache_hits": 0,
    }
    lm = lotus.settings.lm
    if lm is None:
        return snapshot
    stats = lm.stats
    snapshot.update(
        {
            "physical_prompt_tokens": stats.physical_usage.prompt_tokens,
            "physical_completion_tokens": stats.physical_usage.completion_tokens,
            "physical_total_tokens": stats.physical_usage.total_tokens,
            "virtual_prompt_tokens": stats.virtual_usage.prompt_tokens,
            "virtual_completion_tokens": stats.virtual_usage.completion_tokens,
            "virtual_total_tokens": stats.virtual_usage.total_tokens,
            "cache_hits": stats.cache_hits,
        }
    )
    return snapshot


def usage_delta(
    before: Mapping[str, float | int],
    after: Mapping[str, float | int],
) -> dict[str, float | int]:
    """Return usage counter deltas."""

    return {field: after[field] - before[field] for field in USAGE_FIELDS}


def selected_benchmark_data(
    *,
    sample_index: int,
    row_limit: int,
    question_limit: int,
    locomo_cache_path: Path,
) -> tuple[Path, tuple[BenchmarkEvent, ...], tuple[BenchmarkQuestion, ...]]:
    """Load and select one LOCOMO benchmark slice."""

    if row_limit < 0:
        raise SystemExit("--row-limit must be non-negative")
    if question_limit < 0:
        raise SystemExit("--question-limit must be non-negative")

    dataset_path = ensure_locomo_dataset(locomo_cache_path, url=DEFAULT_LOCOMO_URL)
    sample = load_locomo_sample(dataset_path, sample_index=sample_index)
    events = select_events(sample.events, row_limit=row_limit)
    questions = eligible_questions(
        sample.questions,
        ingested_event_ids=[event.event_id for event in events],
        question_limit=question_limit,
    )
    return dataset_path, events, questions


def checkpoint_manifest(
    *,
    sample_index: int,
    row_limit: int,
    question_limit: int,
    model: str,
    answer: bool,
    maintenance_mode: str = "ingest",
    source_run_dir: str = "",
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    step_metrics: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    trace_dir: Path | None,
    trace_enabled: bool,
) -> dict[str, Any]:
    """Build the checkpoint manifest for the current successful boundary."""

    completed_events = len(step_metrics)
    completed_questions = len(result_rows)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark_contract": BENCHMARK_CONTRACT,
        "policy_contract": POLICY_CONTRACT,
        "scorer_contract": SCORER_CONTRACT,
        "checkpoint_contract_digest": checkpoint_contract_digest(),
        "input_digest": benchmark_input_digest(events, questions),
        "trace_enabled": trace_enabled,
        "sample_index": sample_index,
        "row_limit": row_limit,
        "question_limit": question_limit,
        "model": model,
        "answer": answer,
        "maintenance_mode": maintenance_mode,
        "source_run_dir": source_run_dir,
        "event_ids": [event.event_id for event in events],
        "event_fingerprints": list(benchmark_event_fingerprints(events)),
        "question_ids": [question.question_id for question in questions],
        "completed_events": completed_events,
        "completed_questions": completed_questions,
        "trace_event_count": trace_event_count(trace_dir),
        "last_event_id": "" if completed_events == 0 else events[completed_events - 1].event_id,
        "last_question_id": ""
        if completed_questions == 0
        else questions[completed_questions - 1].question_id,
    }


def save_checkpoint(
    *,
    output_dir: Path,
    memory: am.ClaudeMemory,
    sample_index: int,
    row_limit: int,
    question_limit: int,
    model: str,
    answer: bool,
    maintenance_mode: str = "ingest",
    source_run_dir: str = "",
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    step_metrics: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    trace_dir: Path | None = None,
    trace_enabled: bool = False,
) -> None:
    """Persist checkpoint state after a fully successful step."""

    checkpoint_id = (
        f"events-{len(step_metrics):06d}-questions-{len(result_rows):06d}-{time.time_ns()}"
    )
    directory = checkpoint_snapshots_dir(output_dir) / checkpoint_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = checkpoint_manifest(
        sample_index=sample_index,
        row_limit=row_limit,
        question_limit=question_limit,
        model=model,
        answer=answer,
        maintenance_mode=maintenance_mode,
        source_run_dir=source_run_dir,
        events=events,
        questions=questions,
        step_metrics=step_metrics,
        result_rows=result_rows,
        trace_dir=trace_dir,
        trace_enabled=trace_enabled,
    )
    write_bytes_atomic(directory / "state.pkl", pickle.dumps(memory._runtime._state))
    write_jsonl_atomic(directory / "step_metrics.jsonl", step_metrics)
    write_jsonl_atomic(directory / "result_rows.jsonl", result_rows)
    write_jsonl_atomic(directory / "metric_rows.jsonl", metric_rows)
    write_text_atomic(directory / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    write_text_atomic(
        checkpoint_current_path(output_dir),
        json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def load_checkpoint(
    *,
    output_dir: Path,
    sample_index: int,
    row_limit: int,
    question_limit: int,
    model: str,
    answer: bool,
    maintenance_mode: str = "ingest",
    source_run_dir: str = "",
    trace_enabled: bool,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    trusted_checkpoint: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate one benchmark checkpoint."""

    directory = current_checkpoint_snapshot_dir(output_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Checkpoint snapshot is missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark_contract": BENCHMARK_CONTRACT,
        "policy_contract": POLICY_CONTRACT,
        "scorer_contract": SCORER_CONTRACT,
        "checkpoint_contract_digest": checkpoint_contract_digest(),
        "input_digest": benchmark_input_digest(events, questions),
        "trace_enabled": trace_enabled,
        "sample_index": sample_index,
        "row_limit": row_limit,
        "question_limit": question_limit,
        "model": model,
        "answer": answer,
        "maintenance_mode": maintenance_mode,
        "source_run_dir": source_run_dir,
        "event_ids": [event.event_id for event in events],
        "event_fingerprints": list(benchmark_event_fingerprints(events)),
        "question_ids": [question.question_id for question in questions],
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        joined = ", ".join(mismatches)
        raise SystemExit(f"Checkpoint does not match current benchmark arguments: {joined}")

    state_path = directory / "state.pkl"
    if not state_path.exists():
        raise SystemExit(f"Checkpoint is missing runtime state: {state_path}")
    if not trusted_checkpoint:
        raise SystemExit(
            "Loading --resume requires --trust-existing-output-dir because "
            "checkpoint state.pkl uses Python pickle. Only use trusted local "
            "benchmark outputs."
        )
    state = pickle.loads(state_path.read_bytes())
    if not isinstance(state, dict):
        raise SystemExit("Checkpoint runtime state must be a dict")
    step_metrics = read_jsonl(directory / "step_metrics.jsonl")
    result_rows = read_jsonl(directory / "result_rows.jsonl")
    metric_rows = read_jsonl(directory / "metric_rows.jsonl")
    validate_checkpoint_progress(
        manifest=manifest,
        events=events,
        questions=questions,
        step_metrics=step_metrics,
        result_rows=result_rows,
        metric_rows=metric_rows,
    )
    return state, step_metrics, result_rows, metric_rows


def load_external_runtime_state(
    source_run_dir: Path,
    *,
    events: Sequence[BenchmarkEvent],
    trusted_checkpoint: bool = False,
) -> dict[str, Any]:
    """Load runtime state from another benchmark run checkpoint."""

    pointer_path = checkpoint_current_path(source_run_dir)
    if not pointer_path.exists():
        raise SystemExit(
            "--existing-output-dir requires a source checkpoint: "
            f"{pointer_path}"
        )
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid source checkpoint pointer: {pointer_path}") from error
    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise SystemExit("Source checkpoint current pointer is missing checkpoint_id")
    snapshot_dir = checkpoint_snapshots_dir(source_run_dir) / checkpoint_id
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Source checkpoint is missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid source checkpoint manifest: {manifest_path}") from error
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark_contract": BENCHMARK_CONTRACT,
        "policy_contract": POLICY_CONTRACT,
        "scorer_contract": SCORER_CONTRACT,
        "checkpoint_contract_digest": checkpoint_contract_digest(),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        joined = ", ".join(mismatches)
        raise SystemExit(f"Source checkpoint does not match benchmark contracts: {joined}")
    source_event_ids = manifest.get("event_ids")
    if not isinstance(source_event_ids, list):
        raise SystemExit("Source checkpoint manifest is missing event_ids")
    selected_event_ids = [event.event_id for event in events]
    if source_event_ids[: len(selected_event_ids)] != selected_event_ids:
        raise SystemExit("Source checkpoint event ids do not cover selected events")
    source_event_fingerprints = manifest.get("event_fingerprints")
    if not isinstance(source_event_fingerprints, list):
        raise SystemExit("Source checkpoint manifest is missing event_fingerprints")
    selected_event_fingerprints = list(benchmark_event_fingerprints(events))
    if source_event_fingerprints[: len(selected_event_fingerprints)] != selected_event_fingerprints:
        raise SystemExit("Source checkpoint event content does not cover selected events")
    try:
        completed_events = int(manifest.get("completed_events", -1))
    except (TypeError, ValueError) as error:
        raise SystemExit("Source checkpoint completed_events is invalid") from error
    if completed_events != len(selected_event_ids):
        raise SystemExit(
            "Source checkpoint completed event boundary must match selected events"
        )
    state_path = snapshot_dir / "state.pkl"
    if not state_path.exists():
        raise SystemExit(f"Source checkpoint is missing runtime state: {state_path}")
    if not trusted_checkpoint:
        raise SystemExit(
            "Loading --existing-output-dir requires --trust-existing-output-dir "
            "because checkpoint state.pkl uses Python pickle. Only use trusted "
            "local benchmark outputs."
        )
    state = pickle.loads(state_path.read_bytes())
    if not isinstance(state, dict):
        raise SystemExit("Source checkpoint runtime state must be a dict")
    return state


def load_artifact_runtime_state(
    source_run_dir: Path,
    *,
    events: Sequence[BenchmarkEvent],
    trusted_checkpoint: bool = False,
) -> dict[str, Any]:
    """Restore public ClaudeMemory state from raw machine-readable artifacts."""

    if not trusted_checkpoint:
        raise SystemExit(
            "Loading --restore-artifact-csv-state requires --trust-existing-output-dir. "
            "Only use trusted local benchmark outputs."
        )
    source_events_path = source_run_dir / "state" / "input" / "events.jsonl"
    if not source_events_path.exists():
        raise SystemExit(
            "--restore-artifact-csv-state requires source events state: "
            f"{source_events_path}"
        )
    event_columns = ("sample_id", "event_id", "speaker", "session_id", "timestamp", "text")
    source_events = read_state_jsonl(
        source_events_path,
        required=True,
        empty_columns=event_columns,
    )
    validate_artifact_event_prefix(source_events, events)

    memory_dir = source_run_dir / "state" / "memory"
    log_columns = ("message", "role", "timestamp", "session_id")
    log = read_state_jsonl(
        memory_dir / "log.jsonl",
        required=False,
        empty_columns=log_columns,
    )
    topic_columns = ("name", "description", "type", "body")
    topics = read_state_jsonl(
        memory_dir / "topics.jsonl",
        required=True,
        empty_columns=topic_columns,
    )
    require_columns(
        topics,
        set(topic_columns),
        path=memory_dir / "topics.jsonl",
    )
    catalog_columns = ("catalog_title", "name", "hook")
    catalog = read_state_jsonl(
        memory_dir / "catalog.jsonl",
        required=True,
        empty_columns=catalog_columns,
    )
    require_columns(
        catalog,
        set(catalog_columns),
        path=memory_dir / "catalog.jsonl",
    )
    return {
        "log": log,
        "topics": topics,
        "catalog": catalog,
    }


def read_state_jsonl(
    path: Path,
    *,
    required: bool,
    empty_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Read one raw JSONL state artifact, returning an empty DataFrame when optional."""

    if not path.exists():
        if required:
            raise SystemExit(f"--restore-artifact-csv-state requires state JSONL: {path}")
        return pd.DataFrame()
    rows = read_jsonl(path)
    if not rows and empty_columns:
        return pd.DataFrame(columns=list(empty_columns))
    return pd.DataFrame(rows)


def require_columns(frame: pd.DataFrame, columns: set[str], *, path: Path) -> None:
    """Require restored artifact state columns."""

    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise SystemExit(f"Restored artifact state {path} is missing required columns: {missing}")


def validate_artifact_event_prefix(
    source_events: pd.DataFrame,
    events: Sequence[BenchmarkEvent],
) -> None:
    """Validate restored source events match the selected input prefix."""

    required = ("sample_id", "event_id", "speaker", "session_id", "timestamp", "text")
    require_columns(source_events, set(required), path=Path("input/events.csv"))
    if len(source_events) < len(events):
        raise SystemExit("Restored CSV events do not cover selected events")
    if not events:
        return
    expected = pd.DataFrame(list(benchmark_event_fingerprints(events))).loc[:, list(required)]
    actual = source_events.loc[: len(events) - 1, list(required)].reset_index(drop=True)
    if actual.to_dict(orient="records") != expected.to_dict(orient="records"):
        raise SystemExit("Restored CSV event content does not cover selected events")


def validate_checkpoint_progress(
    *,
    manifest: Mapping[str, Any],
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    step_metrics: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate checkpoint progress rows against the selected input prefix."""

    if len(step_metrics) != int(manifest.get("completed_events", -1)):
        raise SystemExit("Checkpoint step metrics do not match manifest")
    if len(result_rows) != int(manifest.get("completed_questions", -1)):
        raise SystemExit("Checkpoint result rows do not match manifest")
    if len(metric_rows) != len(result_rows):
        raise SystemExit("Checkpoint metric rows do not match result rows")
    if len(step_metrics) > len(events):
        raise SystemExit("Checkpoint step metrics exceed selected events")
    if len(result_rows) > len(questions):
        raise SystemExit("Checkpoint result rows exceed selected questions")

    for index, row in enumerate(step_metrics):
        expected_event_id = events[index].event_id
        if row.get("event_id") != expected_event_id:
            raise SystemExit("Checkpoint step metrics event ids do not match selected events")

    for index, row in enumerate(result_rows):
        expected_question_id = questions[index].question_id
        if row.get("question_id") != expected_question_id:
            raise SystemExit("Checkpoint result row question ids do not match selected questions")

    for index, row in enumerate(metric_rows):
        expected_question_id = result_rows[index].get("question_id")
        if row.get("question_id") != expected_question_id:
            raise SystemExit("Checkpoint metric row question ids do not match result rows")


def checkpoint_trace_event_count(output_dir: Path) -> int:
    """Return the trace event boundary recorded in the latest checkpoint."""

    current_path = checkpoint_current_path(output_dir)
    if not current_path.exists():
        return 0
    manifest_path = checkpoint_manifest_path(output_dir)
    if not manifest_path.exists():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    try:
        return int(manifest.get("trace_event_count", 0))
    except (TypeError, ValueError):
        return 0


def events_frame(events: Sequence[BenchmarkEvent]) -> pd.DataFrame:
    """Return selected benchmark events as CSV-ready rows."""

    return pd.DataFrame(
        [
            {
                "sample_id": event.sample_id,
                "event_id": event.event_id,
                "speaker": event.speaker,
                "session_id": event.session_id,
                "timestamp": event.timestamp,
                "text": event.text,
            }
            for event in events
        ]
    )


def questions_frame(questions: Sequence[BenchmarkQuestion]) -> pd.DataFrame:
    """Return selected benchmark questions as CSV-ready rows."""

    return pd.DataFrame(
        [
            {
                "question_id": question.question_id,
                "sample_id": question.sample_id,
                "question": question.question,
                "gold_answer": question.gold_answer,
                "evidence_event_ids": ";".join(question.evidence_event_ids),
                "category": question.category,
            }
            for question in questions
        ]
    )


def run_memory_ingest(
    memory: am.ClaudeMemory,
    events: Sequence[BenchmarkEvent],
    *,
    step_metrics: list[dict[str, Any]] | None = None,
    start_index: int = 0,
    checkpoint_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Append selected benchmark events into ClaudeMemory."""

    if step_metrics is None:
        step_metrics = []
    for index, event in enumerate(events[start_index:], start=start_index + 1):
        row = event_to_claude_log_row(event)
        before = usage_snapshot()
        start = time.perf_counter()
        with semantic_trace_scope(
            run_kind="benchmark",
            phase="add",
            add_index=index,
            sample_id=event.sample_id,
            event_id=event.event_id,
        ):
            memory.add(row)
        after = usage_snapshot()
        step_metrics.append(
            {
                "phase": "add",
                "event_id": event.event_id,
                "latency_sec": round(time.perf_counter() - start, 4),
                **usage_delta(before, after),
                **memory_row_counts(memory),
            }
        )
        if checkpoint_callback is not None:
            checkpoint_callback()
    return step_metrics


def create_memory(*, model: str, trace_dir: Path | None) -> am.ClaudeMemory:
    """Create the ClaudeMemory instance used by one benchmark run."""

    return am.ClaudeMemory(
        adapter=LotusAdapter(
            model=model,
            config=LotusExecutionConfig(semantic_trace_dir=trace_dir),
        )
    )


def memory_row_counts(memory: am.ClaudeMemory) -> dict[str, int]:
    """Return current public memory table sizes."""

    state = memory._runtime._state
    return {
        "log_rows": len(state.get("log", [])),
        "topics_rows": len(state.get("topics", [])),
        "catalog_rows": len(state.get("catalog", [])),
    }


def run_questions(
    memory: am.ClaudeMemory,
    questions: Sequence[BenchmarkQuestion],
    *,
    answer: bool,
    result_rows: list[dict[str, Any]] | None = None,
    metric_rows: list[dict[str, Any]] | None = None,
    start_index: int = 0,
    checkpoint_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run retrieval and optional answer generation for selected questions."""

    if result_rows is None:
        result_rows = []
    if metric_rows is None:
        metric_rows = []
    for question in questions[start_index:]:
        before = usage_snapshot()
        retrieval_start = time.perf_counter()
        with semantic_trace_scope(
            run_kind="benchmark",
            phase="retrieval",
            question_id=question.question_id,
        ):
            retrieved = memory.query(question.question)
        retrieval_latency = time.perf_counter() - retrieval_start
        after_retrieval = usage_snapshot()

        generated_answer: str | None = None
        answer_latency = 0.0
        answer_usage = {field: 0 for field in USAGE_FIELDS}
        if answer:
            before_answer = usage_snapshot()
            answer_start = time.perf_counter()
            generated_answer = generate_answer(
                question.question,
                retrieved,
                question_id=question.question_id,
            )
            answer_latency = time.perf_counter() - answer_start
            answer_usage = usage_delta(before_answer, usage_snapshot())

        metric_row = question_metric_row(
            question_id=question.question_id,
            question=question.question,
            gold_answer=question.gold_answer,
            retrieved_frame=retrieved,
            generated_answer=generated_answer,
            category=question.category,
        )
        retrieval_usage = usage_delta(before, after_retrieval)
        result_rows.append(
            {
                **metric_row,
                "category": question.category,
                "evidence_event_ids": ";".join(question.evidence_event_ids),
                "retrieved_names": retrieved_names(retrieved),
                "retrieval_latency_sec": round(retrieval_latency, 4),
                "answer_latency_sec": round(answer_latency, 4),
                **{f"retrieval_{key}": value for key, value in retrieval_usage.items()},
                **{f"answer_{key}": value for key, value in answer_usage.items()},
            }
        )
        metric_rows.append(metric_row)
        if checkpoint_callback is not None:
            checkpoint_callback()
    return result_rows, metric_rows


def generate_answer(question: str, retrieved: Any, *, question_id: str) -> str:
    """Generate one answer from retrieved memory rows using the configured LOTUS LM."""

    import lotus

    if lotus.settings.lm is None:
        raise RuntimeError("LOTUS LM is not configured before answer generation")

    context = frame_text(retrieved)
    messages = [
        [
            {
                "role": "system",
                "content": ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved memory context:\n{context or '(empty)'}\n\n"
                    "Answer with a short factual phrase or sentence."
                ),
            },
        ]
    ]
    with semantic_trace_scope(
        semantic_operator="answer",
        phase="answer",
        question_id=question_id,
    ):
        output = lotus.settings.lm(
            messages,
            show_progress_bar=False,
            progress_bar_desc="Answering",
            max_tokens=ANSWER_MAX_TOKENS,
        )
    outputs = list(getattr(output, "outputs", ()))
    return str(outputs[0]).strip() if outputs else ""


def retrieved_names(retrieved: Any) -> str:
    """Return retrieved topic names as a semicolon-delimited string."""

    if not hasattr(retrieved, "columns") or "name" not in retrieved.columns:
        return ""
    return ";".join(str(value) for value in retrieved["name"].dropna())


def summary_frame(
    *,
    run_mode: str,
    model: str,
    sample_index: int,
    maintenance_mode: str,
    source_run_dir: str,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    memory: am.ClaudeMemory,
    question_metrics: Sequence[Mapping[str, Any]],
    question_results: Sequence[Mapping[str, Any]],
    step_metrics: Sequence[Mapping[str, Any]],
    llm_anomaly_rows: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build one-row benchmark summary metrics."""

    topics = memory._runtime._state.get("topics", pd.DataFrame())
    catalog = memory._runtime._state.get("catalog", pd.DataFrame())
    question_summary = summarize_question_metrics(question_metrics)
    step_frame = pd.DataFrame(step_metrics)
    result_frame = pd.DataFrame(question_results)
    ingest_latency = float(step_frame["latency_sec"].sum()) if not step_frame.empty else 0.0
    retrieval_latency = (
        float(result_frame["retrieval_latency_sec"].sum())
        if "retrieval_latency_sec" in result_frame
        else 0.0
    )
    answer_latency = (
        float(result_frame["answer_latency_sec"].sum())
        if "answer_latency_sec" in result_frame
        else 0.0
    )
    row = {
        "run_mode": run_mode,
        "input_rendering": "message_with_event_context",
        "bookkeeping_metadata_excluded_from_semantic_input": True,
        "qa_accuracy_available": run_mode == "answer",
        "official_score_available": False,
        "strict_evidence_recall_available": False,
        "maintenance_mode": maintenance_mode,
        "source_run_dir": source_run_dir,
        "model": model,
        "sample_index": sample_index,
        "events_ingested": len(events),
        "eligible_questions": len(questions),
        "topics_rows": len(topics),
        "catalog_rows": len(catalog),
        "duplicate_topic_name_count": duplicate_name_count(topics),
        "duplicate_topic_name_extra_rows": duplicate_name_extra_rows(topics),
        "latency_sec": round(ingest_latency + retrieval_latency + answer_latency, 4),
        "ingest_latency_sec": round(ingest_latency, 4),
        "retrieval_latency_sec": round(retrieval_latency, 4),
        "answer_latency_sec": round(answer_latency, 4),
        "llm_anomaly_count": "" if llm_anomaly_rows is None else len(llm_anomaly_rows),
        "llm_empty_output_count": (
            ""
            if llm_anomaly_rows is None
            else sum(row.get("issue") == "empty_output" for row in llm_anomaly_rows)
        ),
        **question_summary,
    }
    for field in USAGE_FIELDS:
        ingest_value = int(step_frame[field].sum()) if field in step_frame else 0
        retrieval_field = f"retrieval_{field}"
        answer_field = f"answer_{field}"
        retrieval_value = (
            int(result_frame[retrieval_field].sum())
            if retrieval_field in result_frame
            else 0
        )
        answer_value = (
            int(result_frame[answer_field].sum())
            if answer_field in result_frame
            else 0
        )
        row[field] = ingest_value + retrieval_value + answer_value
    return pd.DataFrame([row])


def write_memory_tables(memory: am.ClaudeMemory, output_dir: Path) -> dict[str, Path]:
    """Write runtime memory state display CSVs and raw restore JSONL."""

    state = memory._runtime._state
    log = state.get("log", pd.DataFrame())
    topics = state.get("topics", pd.DataFrame())
    catalog = state.get("catalog", pd.DataFrame())
    return {
        "memory/log": write_csv("log", log, output_dir / "memory"),
        "memory/topics": write_csv("topics", topics, output_dir / "memory"),
        "memory/catalog": write_csv("catalog", catalog, output_dir / "memory"),
        "state/memory/log": write_state_jsonl("log", log, output_dir / "state" / "memory"),
        "state/memory/topics": write_state_jsonl(
            "topics",
            topics,
            output_dir / "state" / "memory",
        ),
        "state/memory/catalog": write_state_jsonl(
            "catalog",
            catalog,
            output_dir / "state" / "memory",
        ),
    }


def write_run_artifacts(
    *,
    output_dir: Path,
    run_mode: str,
    model: str,
    sample_index: int,
    maintenance_mode: str,
    source_run_dir: str,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    memory: am.ClaudeMemory | None,
    result_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    step_metrics: Sequence[Mapping[str, Any]],
    trace_dir: Path | None,
    llm_anomaly_rows: Sequence[Mapping[str, Any]] | None,
    ingested_event_ids: Iterable[str],
    include_summary: bool,
    excluded_trace_event_ranges: Sequence[tuple[int, int]] = (),
) -> dict[str, Path]:
    """Write all artifacts that are available for the current run state."""

    written: dict[str, Path] = {}
    if memory is not None:
        written.update(write_memory_tables(memory, output_dir))
    if result_rows:
        written["retrieval/results"] = write_csv(
            "results",
            pd.DataFrame(result_rows),
            output_dir / "retrieval",
        )
    if metric_rows:
        written["metrics/questions"] = write_csv(
            "questions",
            pd.DataFrame(metric_rows),
            output_dir / "metrics",
        )
    if include_summary and memory is not None:
        written["metrics/summary"] = write_csv(
            "summary",
            summary_frame(
                run_mode=run_mode,
                model=model,
                sample_index=sample_index,
                maintenance_mode=maintenance_mode,
                source_run_dir=source_run_dir,
                events=events,
                questions=questions,
                memory=memory,
                question_metrics=metric_rows,
                question_results=result_rows,
                step_metrics=step_metrics,
                llm_anomaly_rows=llm_anomaly_rows,
            ),
            output_dir / "metrics",
        )
    if trace_dir is not None:
        written.update(
            write_trace_diagnostics(
                output_dir=output_dir,
                trace_dir=trace_dir,
                questions=questions,
                ingested_event_ids=ingested_event_ids,
                llm_anomaly_rows=llm_anomaly_rows or [],
                excluded_trace_event_ranges=excluded_trace_event_ranges,
            )
        )
        written["trace"] = trace_dir
    return written


def write_trace_diagnostics(
    *,
    output_dir: Path,
    trace_dir: Path,
    questions: Sequence[BenchmarkQuestion],
    ingested_event_ids: Iterable[str],
    llm_anomaly_rows: Sequence[Mapping[str, Any]],
    excluded_trace_event_ranges: Sequence[tuple[int, int]] = (),
) -> dict[str, Path]:
    """Write trace-derived diagnostic CSV artifacts."""

    provider_usage_rows = build_provider_usage_rows(
        trace_dir=trace_dir,
        excluded_event_ranges=excluded_trace_event_ranges,
    )
    return {
        "diagnostics/cause_trace": write_csv(
            "cause_trace",
            pd.DataFrame(
                build_cause_trace_rows(
                    questions=questions,
                    ingested_event_ids=set(ingested_event_ids),
                    trace_dir=trace_dir,
                    excluded_event_ranges=excluded_trace_event_ranges,
                )
            ),
            output_dir / "diagnostics",
        ),
        "diagnostics/llm_anomalies": write_csv(
            "llm_anomalies",
            pd.DataFrame(llm_anomaly_rows, columns=LLM_ANOMALY_COLUMNS),
            output_dir / "diagnostics",
        ),
        "diagnostics/provider_usage": write_csv(
            "provider_usage",
            pd.DataFrame(provider_usage_rows, columns=PROVIDER_USAGE_COLUMNS),
            output_dir / "diagnostics",
        ),
        "metrics/provider_usage_summary": write_csv(
            "provider_usage_summary",
            pd.DataFrame(
                build_provider_usage_summary_rows(provider_usage_rows),
                columns=PROVIDER_USAGE_SUMMARY_COLUMNS,
            ),
            output_dir / "metrics",
        ),
    }


def write_failure_metadata(
    *,
    output_dir: Path,
    error: BaseException,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    step_metrics: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    trace_dir: Path | None,
    completed_event_count: int | None = None,
) -> Path:
    """Write one JSON failure summary without swallowing the original error."""

    completed_events = (
        len(step_metrics) if completed_event_count is None else completed_event_count
    )
    completed_questions = len(result_rows)
    failed_event_id = ""
    failed_question_id = ""
    failed_phase = "unknown"
    if completed_events < len(events):
        failed_phase = "add"
        failed_event_id = events[completed_events].event_id
    elif completed_questions < len(questions):
        failed_phase = "question"
        failed_question_id = questions[completed_questions].question_id

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    failure_path = diagnostics_dir / "failure.json"
    failure = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failed_phase": failed_phase,
        "failed_event_id": failed_event_id,
        "failed_question_id": failed_question_id,
        "completed_events": completed_events,
        "total_events": len(events),
        "completed_questions": completed_questions,
        "total_questions": len(questions),
        "output_dir": str(output_dir),
        "trace_dir": "" if trace_dir is None else str(trace_dir),
    }
    failure_path.write_text(
        json.dumps(failure, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return failure_path


def write_recovery_metadata(
    *,
    output_dir: Path,
    excluded_trace_event_ranges: Sequence[tuple[int, int]],
    error: BaseException | None = None,
    trace_dir: Path | None = None,
) -> Path:
    """Write recovery-only diagnostics that are excluded from final benchmark scoring."""

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = recovery_path(output_dir)
    ranges = merge_trace_ranges(excluded_trace_event_ranges)
    payload: dict[str, Any] = {
        "excluded_trace_event_ranges": [
            {"start": start, "end": end, "count": end - start}
            for start, end in ranges
        ],
        "excluded_trace_event_count": sum(end - start for start, end in ranges),
        "trace_event_count": trace_event_count(trace_dir),
    }
    if error is not None:
        payload.update(
            {
                "last_error_type": type(error).__name__,
                "last_error_message": str(error),
            }
        )
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def completed_event_ids(step_metrics: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return event ids for successfully completed add steps."""

    return tuple(
        str(row["event_id"])
        for row in step_metrics
        if row.get("phase") == "add" and row.get("event_id")
    )


def run_claude_memory_locomo(config: ClaudeMemoryLocomoRunConfig) -> dict[str, Path]:
    """Run one ClaudeMemory LOCOMO evaluation slice and write CSV artifacts."""

    output_dir = config.output_dir.resolve()
    source_run_dir = (
        config.existing_output_dir.resolve()
        if config.existing_output_dir is not None
        else None
    )
    validate_output_paths(output_dir=output_dir, source_run_dir=source_run_dir)
    if config.restore_csv_state:
        raise SystemExit(
            "--restore-csv-state is no longer supported because display CSVs "
            "are spreadsheet-escaped. Use --restore-artifact-csv-state."
        )
    if config.restore_artifact_csv_state and source_run_dir is None:
        raise SystemExit("--restore-artifact-csv-state requires --existing-output-dir")
    if config.restore_artifact_csv_state and config.resume:
        raise SystemExit("--restore-artifact-csv-state cannot be used with --resume")
    if source_run_dir is not None and not config.trust_existing_output_dir:
        raise SystemExit(
            "--existing-output-dir requires --trust-existing-output-dir because "
            "local benchmark outputs may contain checkpoint pickle or artifact state."
        )
    if config.resume and not config.trust_existing_output_dir:
        raise SystemExit(
            "--resume requires --trust-existing-output-dir because checkpoint "
            "state.pkl uses Python pickle. Only use trusted local benchmark outputs."
        )
    if config.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        reset_output_dir(output_dir, safe=True)

    dataset_path, events, questions = selected_benchmark_data(
        sample_index=config.sample_index,
        row_limit=config.row_limit,
        question_limit=config.question_limit,
        locomo_cache_path=config.locomo_cache_path,
    )
    trace_dir = output_dir / "trace" if config.trace else None
    run_mode = "answer" if config.answer else "retrieval_only_diagnostic"
    maintenance_mode = "external-state" if source_run_dir is not None else "ingest"
    source_run_dir_str = "" if source_run_dir is None else str(source_run_dir)

    print("LOCOMO benchmark")
    print(f"dataset: {dataset_path}")
    print(f"sample_index: {config.sample_index}")
    print(f"events: {len(events)}")
    print(f"eligible_questions: {len(questions)}")
    print(f"run_mode: {run_mode}")
    print(f"maintenance_mode: {maintenance_mode}")
    if source_run_dir is not None:
        print(f"source_run_dir: {source_run_dir}")
    print(f"model: {config.model}")
    print(f"output_dir: {output_dir}")

    written: dict[str, Path] = {
        "input/events": write_csv("events", events_frame(events), output_dir / "input"),
        "input/questions": write_csv(
            "questions",
            questions_frame(questions),
            output_dir / "input",
        ),
        "state/input/events": write_state_jsonl(
            "events",
            pd.DataFrame(list(benchmark_event_fingerprints(events))),
            output_dir / "state" / "input",
        ),
    }
    memory: am.ClaudeMemory | None = None
    step_metrics: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    excluded_trace_event_ranges: list[tuple[int, int]] = []
    try:
        memory = create_memory(model=config.model, trace_dir=trace_dir)
        if config.resume:
            state, step_metrics, result_rows, metric_rows = load_checkpoint(
                output_dir=output_dir,
                sample_index=config.sample_index,
                row_limit=config.row_limit,
                question_limit=config.question_limit,
                model=config.model,
                answer=config.answer,
                maintenance_mode=maintenance_mode,
                source_run_dir=source_run_dir_str,
                trace_enabled=config.trace,
                events=events,
                questions=questions,
                trusted_checkpoint=config.trust_existing_output_dir,
            )
            memory._runtime._state = state
            checkpoint_trace_count = checkpoint_trace_event_count(output_dir)
            excluded_trace_event_ranges = trace_exclusion_ranges(
                output_dir=output_dir,
                trace_dir=trace_dir,
                checkpoint_trace_event_count=checkpoint_trace_count,
            )
            if excluded_trace_event_ranges:
                write_recovery_metadata(
                    output_dir=output_dir,
                    excluded_trace_event_ranges=excluded_trace_event_ranges,
                    trace_dir=trace_dir,
                )
            print(
                "resuming checkpoint: "
                f"completed_events={len(step_metrics)}, "
                f"completed_questions={len(result_rows)}"
            )
        elif source_run_dir is not None and config.restore_artifact_csv_state:
            memory._runtime._state = load_artifact_runtime_state(
                source_run_dir,
                events=events,
                trusted_checkpoint=config.trust_existing_output_dir,
            )
            counts = memory_row_counts(memory)
            print(
                "restored artifact memory state: "
                f"topics_rows={counts['topics_rows']}, "
                f"catalog_rows={counts['catalog_rows']}"
            )
        elif source_run_dir is not None:
            memory._runtime._state = load_external_runtime_state(
                source_run_dir,
                events=events,
                trusted_checkpoint=config.trust_existing_output_dir,
            )
            counts = memory_row_counts(memory)
            print(
                "restored external memory state: "
                f"topics_rows={counts['topics_rows']}, "
                f"catalog_rows={counts['catalog_rows']}"
            )

        def checkpoint() -> None:
            save_checkpoint(
                output_dir=output_dir,
                memory=memory,
                sample_index=config.sample_index,
                row_limit=config.row_limit,
                question_limit=config.question_limit,
                model=config.model,
                answer=config.answer,
                maintenance_mode=maintenance_mode,
                source_run_dir=source_run_dir_str,
                events=events,
                questions=questions,
                step_metrics=step_metrics,
                result_rows=result_rows,
                metric_rows=metric_rows,
                trace_dir=trace_dir,
                trace_enabled=config.trace,
            )

        if source_run_dir is None:
            run_memory_ingest(
                memory,
                events,
                step_metrics=step_metrics,
                start_index=len(step_metrics),
                checkpoint_callback=checkpoint,
            )
        else:
            print("query-only mode: skipping memory ingest")
        result_rows, metric_rows = run_questions(
            memory,
            questions,
            answer=config.answer,
            result_rows=result_rows,
            metric_rows=metric_rows,
            start_index=len(result_rows),
            checkpoint_callback=checkpoint,
        )
    except Exception as error:
        try:
            llm_anomaly_rows = (
                build_llm_anomaly_rows(
                    trace_dir=trace_dir,
                    excluded_event_ranges=excluded_trace_event_ranges,
                )
                if trace_dir is not None
                else None
            )
            written.update(
                write_run_artifacts(
                    output_dir=output_dir,
                    run_mode=run_mode,
                    model=config.model,
                    sample_index=config.sample_index,
                    maintenance_mode=maintenance_mode,
                    source_run_dir=source_run_dir_str,
                    events=events,
                    questions=questions,
                    memory=memory,
                    result_rows=result_rows,
                    metric_rows=metric_rows,
                    step_metrics=step_metrics,
                    trace_dir=trace_dir,
                    llm_anomaly_rows=llm_anomaly_rows,
                    ingested_event_ids=(
                        tuple(event.event_id for event in events)
                        if source_run_dir is not None
                        else completed_event_ids(step_metrics)
                    ),
                    include_summary=False,
                    excluded_trace_event_ranges=excluded_trace_event_ranges,
                )
            )
            written["diagnostics/failure"] = write_failure_metadata(
                output_dir=output_dir,
                error=error,
                events=events,
                questions=questions,
                step_metrics=step_metrics,
                result_rows=result_rows,
                trace_dir=trace_dir,
                completed_event_count=(
                    len(events) if source_run_dir is not None else len(step_metrics)
                ),
            )
            if trace_dir is not None:
                failure_trace_ranges = trace_exclusion_ranges(
                    output_dir=output_dir,
                    trace_dir=trace_dir,
                    checkpoint_trace_event_count=checkpoint_trace_event_count(output_dir),
                )
                written["diagnostics/recovery"] = write_recovery_metadata(
                    output_dir=output_dir,
                    excluded_trace_event_ranges=failure_trace_ranges,
                    error=error,
                    trace_dir=trace_dir,
                )
        except Exception as artifact_error:
            print(f"warning: failed to write partial artifacts: {artifact_error}")
        print("\nwrote partial benchmark artifacts before failure:")
        for name, path_value in written.items():
            print(f"- {name}: {path_value}")
        raise

    llm_anomaly_rows = (
        build_llm_anomaly_rows(
            trace_dir=trace_dir,
            excluded_event_ranges=excluded_trace_event_ranges,
        )
        if trace_dir is not None
        else None
    )
    written.update(
        write_run_artifacts(
            output_dir=output_dir,
            run_mode=run_mode,
            model=config.model,
            sample_index=config.sample_index,
            maintenance_mode=maintenance_mode,
            source_run_dir=source_run_dir_str,
            events=events,
            questions=questions,
            memory=memory,
            result_rows=result_rows,
            metric_rows=metric_rows,
            step_metrics=step_metrics,
            trace_dir=trace_dir,
            llm_anomaly_rows=llm_anomaly_rows,
            ingested_event_ids=(
                tuple(event.event_id for event in events)
                if source_run_dir is not None
                else completed_event_ids(step_metrics)
            ),
            include_summary=True,
            excluded_trace_event_ranges=excluded_trace_event_ranges,
        )
    )

    print("\nwrote benchmark artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")
    return written
