"""Backend-neutral trace writers for semantic execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

import pandas as pd

TRACE_EVENTS_FILE = "events.jsonl"
TRACE_METRICS_FILE = "metrics.csv"
TRACE_PROMPTS_DIR = "prompts"
TRACE_OUTPUTS_DIR = "outputs"
TRACE_SNAPSHOTS_DIR = "snapshots"
TRACE_RUN_KINDS = {"differential", "view"}
TEXT_PREVIEW_CHARS = 240
ARTIFACT_PAYLOAD_KEYS = {
    "formatted_instruction",
    "final_instruction",
    "raw_output",
    "raw_output_attempts",
    "parsed_output",
}
PREVIEW_PAYLOAD_KEYS = {
    "instruction",
    "input_preview",
    "group_preview",
}

_TRACE_SCOPE: ContextVar[dict[str, Any]] = ContextVar(
    "agent_memory_semantic_trace_scope",
    default={},
)
_TRACE_WRITE_LOCK = Lock()


@contextmanager
def semantic_trace_scope(**metadata: Any) -> Iterator[None]:
    """Attach runtime metadata to semantic trace events in the current context."""

    parent = dict(_TRACE_SCOPE.get())
    child = dict(parent)
    child.update({key: value for key, value in metadata.items() if value is not None})
    token = _TRACE_SCOPE.set(child)
    try:
        yield
    finally:
        _TRACE_SCOPE.reset(token)


def active_trace_scope() -> dict[str, Any]:
    """Return a copy of the current semantic trace metadata."""

    return dict(_TRACE_SCOPE.get())


def trace_scope_value(key: str, default: Any = None) -> Any:
    """Return one active semantic trace scope value."""

    return _TRACE_SCOPE.get().get(key, default)


def write_trace_event(
    trace_dir: Path | str | None,
    *,
    operator: str,
    event_type: str,
    payload: Mapping[str, Any] | None = None,
    raw_output: Any = None,
    parsed_output: Any = None,
    snapshots: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Write one semantic trace event and return the event row."""

    if trace_dir is None:
        return None

    root = _event_root(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    trace_id = _trace_id(operator)
    event: dict[str, Any] = {
        "trace_id": trace_id,
        "timestamp": _timestamp(),
        "operator": operator,
        "event_type": event_type,
    }
    event.update(active_trace_scope())
    if payload:
        event.update(_compact_payload(payload))

    if raw_output is not None:
        event["raw_output_path"] = _write_json_artifact(
            root / TRACE_OUTPUTS_DIR,
            trace_id,
            "raw.json",
            raw_output,
        )
    if parsed_output is not None:
        event["parsed_output_path"] = _write_json_artifact(
            root / TRACE_OUTPUTS_DIR,
            trace_id,
            "parsed.json",
            parsed_output,
        )
    for name, snapshot in (snapshots or {}).items():
        event[f"{name}_snapshot_path"] = _write_snapshot(
            root / TRACE_SNAPSHOTS_DIR,
            trace_id,
            name,
            snapshot,
        )

    _append_event(root, event)
    return event


def write_llm_call_trace(
    trace_dir: Path | str | None,
    *,
    model: str,
    messages: Sequence[Any],
    kwargs: Mapping[str, Any],
    outputs: Sequence[Any] | None = None,
    latency_sec: float | None = None,
    usage_delta: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> list[dict[str, Any]]:
    """Write actual LOTUS LM request/response trace events."""

    if trace_dir is None:
        return []

    root = _event_root(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    scope = active_trace_scope()
    operator = str(scope.get("operator", scope.get("semantic_operator", "llm")))
    trace_operator = "llm" if operator == "llm" else f"{operator}-llm"
    batch_id = _trace_id(trace_operator)
    batch_size = len(messages)
    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        trace_id = f"{batch_id}-{index:04d}"
        event: dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": _timestamp(),
            "operator": operator,
            "event_type": "llm_batch_error" if error is not None else "llm_call",
            "llm_batch_id": batch_id,
            "llm_item_index": index,
            "llm_batch_size": batch_size,
            "model": model,
            "latency_sec": "" if latency_sec is None else round(float(latency_sec), 6),
        }
        event.update(scope)
        event.update(_compact_payload({"llm_kwargs": _json_safe(kwargs)}))
        if usage_delta and index == 0:
            event["usage_scope"] = "batch"
            event.update({f"usage_{key}": value for key, value in usage_delta.items()})

        event["prompt_path"] = _write_json_artifact(
            root / TRACE_PROMPTS_DIR,
            trace_id,
            "prompt.json",
            message,
        )
        if outputs is not None and index < len(outputs):
            event["raw_output_path"] = _write_json_artifact(
                root / TRACE_OUTPUTS_DIR,
                trace_id,
                "raw.json",
                outputs[index],
            )
        if error is not None:
            event["error_type"] = type(error).__name__
            event["error_message"] = _preview(str(error))
            event["error_output_path"] = _write_json_artifact(
                root / TRACE_OUTPUTS_DIR,
                trace_id,
                "error.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
        _append_event(root, event)
        rows.append(event)
    return rows


def write_pair_trace(
    trace_dir: Path | str | None,
    *,
    operator: str,
    rows: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write pairwise semantic decision trace events."""

    if trace_dir is None or not rows:
        return []

    events: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        event = write_trace_event(
            trace_dir,
            operator=operator,
            event_type="pair_decision",
            payload={**dict(row), "pair_index": index},
            raw_output=row.get("raw_output"),
            parsed_output=row.get("parsed_output"),
            snapshots=snapshots if index == 0 else None,
        )
        if event is not None:
            events.append(event)
    return events


def write_structured_generation_trace(
    trace_dir: Path | str | None,
    *,
    operator: str,
    rows: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write structured generation trace events."""

    if trace_dir is None or not rows:
        return []

    events: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw_output = row.get("raw_output_attempts") or row.get("raw_output")
        event = write_trace_event(
            trace_dir,
            operator=operator,
            event_type="structured_generation",
            payload={**dict(row), "structured_index": index},
            raw_output=raw_output,
            parsed_output=row.get("parsed_output"),
            snapshots=snapshots if index == 0 else None,
        )
        if event is not None:
            events.append(event)
    return events


def write_compact_operator_trace(
    trace_dir: Path | str | None,
    *,
    operator: str,
    event_type: str,
    input_frame: Any = None,
    output_frame: Any = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Write a compact trace event for operators without row-level raw outputs."""

    if trace_dir is None:
        return None

    event_payload: dict[str, Any] = dict(payload or {})
    if input_frame is not None:
        event_payload.update(_frame_shape_payload(input_frame, prefix="input"))
    if output_frame is not None:
        event_payload.update(_frame_shape_payload(output_frame, prefix="output"))

    snapshots: dict[str, Any] = {}
    if input_frame is not None:
        snapshots["input"] = input_frame
    if output_frame is not None:
        snapshots["output"] = output_frame

    return write_trace_event(
        trace_dir,
        operator=operator,
        event_type=event_type,
        payload=event_payload,
        snapshots=snapshots,
    )


def append_trace_metrics(
    trace_dir: Path | str | None,
    rows: Sequence[Mapping[str, Any]],
) -> Path | None:
    """Append demo-level metrics rows under the trace tree."""

    if trace_dir is None or not rows:
        return None

    fieldnames = sorted({key for row in rows for key in row.keys()})
    paths: list[Path] = []
    grouped_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        run_kind = str(row.get("run_kind", ""))
        key = run_kind if run_kind in TRACE_RUN_KINDS else ""
        grouped_rows.setdefault(key, []).append(row)

    for run_kind, selected_rows in grouped_rows.items():
        path = _run_root(trace_dir, run_kind) / TRACE_METRICS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in selected_rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        paths.append(path)
    return paths[0] if paths else None


def query_digest(query: Any) -> str:
    """Return a short deterministic digest for a query-like value."""

    text = repr(query)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _append_event(root: Path, event: Mapping[str, Any]) -> None:
    """Append one JSONL trace event without interleaving concurrent writers."""

    events_path = root / TRACE_EVENTS_FILE
    with _TRACE_WRITE_LOCK:
        with events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str))
            file.write("\n")


def _event_root(trace_dir: Path | str) -> Path:
    """Return the trace root for the active run kind."""

    run_kind = str(trace_scope_value("run_kind", ""))
    if run_kind in TRACE_RUN_KINDS:
        return Path(trace_dir) / run_kind
    return Path(trace_dir)


def _run_root(trace_dir: Path | str, run_kind: str) -> Path:
    """Return the trace root for a metrics row run kind."""

    if run_kind in TRACE_RUN_KINDS:
        return Path(trace_dir) / run_kind
    return Path(trace_dir)


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep events compact while preserving previews for artifact-backed fields."""

    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ARTIFACT_PAYLOAD_KEYS:
            continue
        elif key == "instruction":
            compact[f"{key}_preview"] = _preview(value)
        elif key in {"input_preview", "group_preview"}:
            compact[key] = _preview(value)
        elif key == "llm_kwargs":
            compact[key] = value
        else:
            compact[key] = value
    return compact


def _json_safe(value: Any) -> Any:
    """Convert common nested values into JSON-safe debug metadata."""

    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_json_safe(item) for item in value]
        return str(value)


def _preview(value: Any) -> str:
    """Return a compact one-line preview for event metadata."""

    text = " ".join(_to_text(value).split())
    if len(text) <= TEXT_PREVIEW_CHARS:
        return text
    return f"{text[: TEXT_PREVIEW_CHARS - 3]}..."


def _frame_shape_payload(frame: Any, *, prefix: str) -> dict[str, Any]:
    """Return row/column metadata for a DataFrame-like object."""

    columns = [str(column) for column in getattr(frame, "columns", ())]
    try:
        rows = len(frame)
    except TypeError:
        rows = ""
    return {
        f"{prefix}_rows": rows,
        f"{prefix}_columns": columns,
    }


def _write_json_artifact(directory: Path, trace_id: str, name: str, value: Any) -> str:
    """Write a JSON artifact and return a relative trace path."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace_id}-{name}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return _relative_trace_path(path)


def _write_snapshot(directory: Path, trace_id: str, name: str, value: Any) -> str:
    """Write a snapshot artifact and return a relative trace path."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace_id}-{name}.csv"
    if hasattr(value, "to_csv"):
        value.to_csv(path, index=False)
    else:
        path.write_text(_to_text(value), encoding="utf-8")
    return _relative_trace_path(path)


def _to_text(value: Any) -> str:
    """Serialize prompt-like values as readable text."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _relative_trace_path(path: Path) -> str:
    """Return a path relative to the trace root when possible."""

    parts = path.parts
    if "trace" in parts:
        index = parts.index("trace")
        return str(Path(*parts[index:]))
    return str(path)


def _trace_id(operator: str) -> str:
    """Return a sortable trace id."""

    return f"{_timestamp()}-{operator}-{uuid4().hex[:8]}"


def _timestamp() -> str:
    """Return a compact UTC timestamp for trace artifact filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
