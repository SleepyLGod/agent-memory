"""Benchmark diagnostics derived from semantic trace artifacts."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import Any

from .types import BenchmarkQuestion

LLM_ANOMALY_COLUMNS = (
    "phase",
    "operator",
    "event_type",
    "trace_id",
    "question_id",
    "event_id",
    "issue",
    "prompt_path",
    "raw_output_path",
    "raw_output_preview",
    "error_type",
    "error_message",
    "model",
)


def build_cause_trace_rows(
    *,
    questions: Sequence[BenchmarkQuestion],
    ingested_event_ids: Collection[str],
    trace_dir: Path,
    excluded_event_ranges: Sequence[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    """Build per-evidence dataflow diagnostics from existing trace artifacts."""

    trace_events = load_trace_events(trace_dir, excluded_event_ranges=excluded_event_ranges)
    events_by_source = _add_events_by_source_event(trace_events)
    ingested = set(ingested_event_ids)
    rows: list[dict[str, Any]] = []
    for question in questions:
        if not question.evidence_event_ids:
            rows.append(_base_row(question.question_id, "", "no_evidence_id", "no_evidence_id"))
            continue
        for evidence_event_id in question.evidence_event_ids:
            rows.append(
                _cause_trace_row(
                    question_id=question.question_id,
                    evidence_event_id=evidence_event_id,
                    ingested=evidence_event_id in ingested,
                    events=events_by_source.get(evidence_event_id, ()),
                    trace_dir=trace_dir,
                )
            )
    return rows


def build_llm_anomaly_rows(
    *,
    trace_dir: Path,
    excluded_event_ranges: Sequence[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    """Build LLM boundary anomaly diagnostics from existing trace artifacts."""

    rows: list[dict[str, Any]] = []
    for event in load_trace_events(trace_dir, excluded_event_ranges=excluded_event_ranges):
        event_type = event.get("event_type")
        if event_type == "llm_batch_error":
            rows.append(_llm_anomaly_row(event, issue="llm_batch_error", raw_output=event))
            continue
        if event_type != "llm_call":
            continue
        raw_path = event.get("raw_output_path")
        if not isinstance(raw_path, str) or not raw_path:
            rows.append(_llm_anomaly_row(event, issue="missing_raw_output"))
            continue
        raw_output = _read_json_artifact(trace_dir, raw_path)
        issue = _llm_output_issue(raw_output)
        if issue:
            rows.append(
                _llm_anomaly_row(
                    event,
                    issue=issue,
                    raw_output_path=raw_path,
                    raw_output=raw_output,
                )
            )
    return rows


def load_trace_events(
    trace_dir: Path,
    excluded_event_ranges: Sequence[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    """Load JSONL semantic trace events from ``trace_dir``."""

    events_path = trace_dir / "events.jsonl"
    if not events_path.exists():
        return []
    excluded_ranges = tuple(excluded_event_ranges)
    events: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as handle:
        for event_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if _is_excluded_event_number(event_number, excluded_ranges):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _is_excluded_event_number(
    event_number: int,
    excluded_event_ranges: Sequence[tuple[int, int]],
) -> bool:
    """Return whether one 1-based event number falls inside an excluded trace range.

    Ranges use checkpoint-boundary semantics: ``(start, end]``. For example,
    ``(32, 33)`` excludes only event 33, preserving the successful checkpoint
    boundary at event 32.
    """

    return any(start < event_number <= end for start, end in excluded_event_ranges)


def _llm_output_issue(raw_output: Any) -> str:
    """Return a conservative anomaly label for one raw LLM output artifact."""

    if raw_output is None:
        return "unreadable_raw_output"
    output_text = _raw_output_text(raw_output)
    if output_text is not None and not output_text.strip():
        return "empty_output"
    return ""


def _raw_output_text(raw_output: Any) -> str | None:
    """Return the primary text output from a TracedLM raw output artifact."""

    if isinstance(raw_output, Mapping) and "output" in raw_output:
        output = raw_output["output"]
    else:
        output = raw_output
    if isinstance(output, str):
        return output
    return None


def _llm_anomaly_row(
    event: Mapping[str, Any],
    *,
    issue: str,
    raw_output_path: str = "",
    raw_output: Any = None,
) -> dict[str, Any]:
    """Return one CSV-ready LLM anomaly row."""

    return {
        "phase": event.get("phase", ""),
        "operator": event.get("operator", ""),
        "event_type": event.get("event_type", ""),
        "trace_id": event.get("trace_id", ""),
        "question_id": event.get("question_id", ""),
        "event_id": event.get("event_id", ""),
        "issue": issue,
        "prompt_path": event.get("prompt_path", ""),
        "raw_output_path": raw_output_path,
        "raw_output_preview": _preview(raw_output),
        "error_type": event.get("error_type", ""),
        "error_message": event.get("error_message", event.get("error", "")),
        "model": event.get("model", ""),
    }


def trace_output_row_count(event: Mapping[str, Any], trace_dir: Path) -> int | None:
    """Return an observed output row count for one trace event when available."""

    output_rows = _int_or_none(event.get("output_rows"))
    if output_rows is not None:
        return output_rows

    parsed_path = event.get("parsed_output_path")
    if isinstance(parsed_path, str) and parsed_path:
        parsed = _read_json_artifact(trace_dir, parsed_path)
        count = _structured_output_count(parsed)
        if count is not None:
            return count

    snapshot_path = event.get("output_snapshot_path")
    if isinstance(snapshot_path, str) and snapshot_path:
        return _csv_data_row_count(_resolve_trace_path(trace_dir, snapshot_path))

    return None


def _cause_trace_row(
    *,
    question_id: str,
    evidence_event_id: str,
    ingested: bool,
    events: Sequence[Mapping[str, Any]],
    trace_dir: Path,
) -> dict[str, Any]:
    """Build one cause trace row for one evidence event."""

    if not ingested:
        return _base_row(question_id, evidence_event_id, "not_ingested", "not_ingested")
    if not events:
        return _base_row(
            question_id,
            evidence_event_id,
            "ingested",
            "missing_trace",
            notes="The evidence event was ingested, but no add-phase trace events were found.",
        )

    observed: list[tuple[Mapping[str, Any], int]] = []
    for event in events:
        count = trace_output_row_count(event, trace_dir)
        if count is not None:
            observed.append((event, count))

    for index, (event, count) in enumerate(observed):
        if count == 0:
            first_positive = next(
                ((item, item_count) for item, item_count in observed[:index] if item_count > 0),
                None,
            )
            first_nonzero_operator = ""
            first_nonzero_trace_id = ""
            first_nonzero_output_rows: int | str = ""
            if first_positive is not None:
                first_event, first_count = first_positive
                first_nonzero_operator = str(first_event.get("operator", ""))
                first_nonzero_trace_id = str(first_event.get("trace_id", ""))
                first_nonzero_output_rows = first_count
            return _base_row(
                question_id,
                evidence_event_id,
                "ingested",
                "stopped_at_zero_output",
                first_nonzero_operator=first_nonzero_operator,
                first_nonzero_trace_id=first_nonzero_trace_id,
                first_nonzero_output_rows=first_nonzero_output_rows,
                last_observed_operator=str(event.get("operator", "")),
                last_observed_trace_id=str(event.get("trace_id", "")),
                first_zero_output_operator=str(event.get("operator", "")),
                first_zero_output_trace_id=str(event.get("trace_id", "")),
                operator_output_rows=count,
                notes="The evidence event was ingested, but this operator emitted zero rows.",
            )

    positive = [(event, count) for event, count in observed if count > 0]
    if positive:
        first_event, first_count = positive[0]
        last_event, last_count = positive[-1]
        return _base_row(
            question_id,
            evidence_event_id,
            "ingested",
            "observed_with_output",
            first_nonzero_operator=str(first_event.get("operator", "")),
            first_nonzero_trace_id=str(first_event.get("trace_id", "")),
            first_nonzero_output_rows=first_count,
            last_observed_operator=str(last_event.get("operator", "")),
            last_observed_trace_id=str(last_event.get("trace_id", "")),
            operator_output_rows=last_count,
            notes=(
                "The evidence event was observed in trace events that emitted rows; "
                "first_nonzero_operator marks the earliest row-producing boundary."
            ),
        )

    return _base_row(
        question_id,
        evidence_event_id,
        "ingested",
        "unknown",
        notes="The evidence event was traced, but no output row count could be inferred.",
    )


def _add_events_by_source_event(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group add-phase trace events by benchmark source event id."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        if event.get("phase") != "add":
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        if not _is_countable_trace_event(event):
            continue
        grouped.setdefault(event_id, []).append(event)
    return grouped


def _is_countable_trace_event(event: Mapping[str, Any]) -> bool:
    """Return whether an event can describe row flow through an operator."""

    if "operator" not in event:
        return False
    if "output_rows" in event or "parsed_output_path" in event or "output_snapshot_path" in event:
        return True
    return False


def _base_row(
    question_id: str,
    evidence_event_id: str,
    source_status: str,
    trace_status: str,
    *,
    first_nonzero_operator: str = "",
    first_nonzero_trace_id: str = "",
    first_nonzero_output_rows: int | str = "",
    last_observed_operator: str = "",
    last_observed_trace_id: str = "",
    first_zero_output_operator: str = "",
    first_zero_output_trace_id: str = "",
    operator_output_rows: int | str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Return one CSV-ready cause trace row."""

    return {
        "question_id": question_id,
        "evidence_event_id": evidence_event_id,
        "source_status": source_status,
        "trace_status": trace_status,
        "first_nonzero_operator": first_nonzero_operator,
        "first_nonzero_trace_id": first_nonzero_trace_id,
        "first_nonzero_output_rows": first_nonzero_output_rows,
        "last_observed_operator": last_observed_operator,
        "last_observed_trace_id": last_observed_trace_id,
        "first_zero_output_operator": first_zero_output_operator,
        "first_zero_output_trace_id": first_zero_output_trace_id,
        "operator_output_rows": operator_output_rows,
        "notes": notes,
    }


def _read_json_artifact(trace_dir: Path, artifact_path: str) -> Any:
    """Read a JSON trace artifact path recorded in an event."""

    path = _resolve_trace_path(trace_dir, artifact_path)
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_trace_path(trace_dir: Path, artifact_path: str) -> Path:
    """Resolve a trace artifact path stored relative to the run directory."""

    path = Path(artifact_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == trace_dir.name:
        return trace_dir.parent / path
    return trace_dir / path


def _structured_output_count(value: Any) -> int | None:
    """Infer row count from a structured parsed output artifact."""

    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        rows = value.get("rows")
        if isinstance(rows, list):
            return len(rows)
        return 1
    return None


def _csv_data_row_count(path: Path) -> int | None:
    """Return the number of data rows in a CSV snapshot."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except OSError:
        return None
    if not rows:
        return 0
    return max(len(rows) - 1, 0)


def _int_or_none(value: Any) -> int | None:
    """Convert an arbitrary value to int when safe."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _preview(value: Any, *, limit: int = 240) -> str:
    """Return a compact one-line preview for CSV diagnostics."""

    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
