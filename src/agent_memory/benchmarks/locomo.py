"""LOCOMO benchmark normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent_memory.benchmarks.types import BenchmarkEvent, BenchmarkQuestion


@dataclass(frozen=True)
class LocomoBenchmarkSample:
    """One LOCOMO sample normalized into events and questions."""

    sample_id: str
    events: tuple[BenchmarkEvent, ...]
    questions: tuple[BenchmarkQuestion, ...]


def load_locomo_sample(dataset_path: Path, *, sample_index: int = 0) -> LocomoBenchmarkSample:
    """Load one LOCOMO sample from a JSON dataset path."""

    dataset = _load_dataset(dataset_path)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} is out of range for {len(dataset)} LOCOMO samples"
        )
    sample = dataset[sample_index]
    if not isinstance(sample, Mapping):
        raise TypeError(f"LOCOMO sample {sample_index} must be a JSON object")
    return normalize_locomo_sample(sample, sample_index=sample_index)


def normalize_locomo_sample(
    sample: Mapping[str, Any],
    *,
    sample_index: int = 0,
) -> LocomoBenchmarkSample:
    """Normalize one official LOCOMO sample into benchmark events and questions."""

    sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
    events = tuple(_conversation_events(sample, sample_id=sample_id))
    questions = tuple(_questions(sample, sample_id=sample_id))
    return LocomoBenchmarkSample(sample_id=sample_id, events=events, questions=questions)


def select_events(
    events: Sequence[BenchmarkEvent],
    *,
    row_limit: int | None,
) -> tuple[BenchmarkEvent, ...]:
    """Return the event prefix used for a benchmark run."""

    if row_limit is None:
        return tuple(events)
    if row_limit < 0:
        raise ValueError("row_limit must be non-negative")
    return tuple(events[:row_limit])


def eligible_questions(
    questions: Sequence[BenchmarkQuestion],
    *,
    ingested_event_ids: Iterable[str],
    question_limit: int | None = None,
) -> tuple[BenchmarkQuestion, ...]:
    """Return questions whose evidence is already present in ingested events."""

    ingested = set(ingested_event_ids)
    selected: list[BenchmarkQuestion] = []
    for question in questions:
        if all(event_id in ingested for event_id in question.evidence_event_ids):
            selected.append(question)
            if question_limit is not None and len(selected) >= question_limit:
                break
    return tuple(selected)


def event_to_claude_log_row(event: BenchmarkEvent) -> dict[str, Any]:
    """Render one LOCOMO event into the current ClaudeMemory log schema."""

    return {
        "message": event.text,
        "role": event.speaker,
        "timestamp": event.timestamp,
        "session_id": event.session_id,
        "metadata": {
            "benchmark": "locomo",
            "sample_id": event.sample_id,
            "event_id": event.event_id,
            "speaker": event.speaker,
            **dict(event.metadata),
        },
    }


def _load_dataset(dataset_path: Path) -> list[Any]:
    """Read a LOCOMO JSON file as a list of samples."""

    with dataset_path.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def _conversation_events(
    sample: Mapping[str, Any],
    *,
    sample_id: str,
) -> Iterable[BenchmarkEvent]:
    """Yield normalized conversation events from common LOCOMO JSON shapes."""

    conversation = sample.get("conversation")
    for session_id, timestamp, turns in _iter_sessions(conversation):
        for turn_index, turn in enumerate(turns, start=1):
            event = _turn_event(
                turn,
                sample_id=sample_id,
                session_id=session_id,
                timestamp=timestamp,
                fallback_index=turn_index,
            )
            if event is not None:
                yield event


def _iter_sessions(conversation: Any) -> Iterable[tuple[str, str, Iterable[Any]]]:
    """Yield session id, timestamp, and turns from LOCOMO conversation data."""

    if isinstance(conversation, Mapping):
        for key, value in conversation.items():
            if not isinstance(value, list):
                continue
            timestamp = str(conversation.get(f"{key}_date_time", ""))
            yield str(key), timestamp, value
        return

    if isinstance(conversation, list):
        for index, item in enumerate(conversation):
            if isinstance(item, Mapping) and isinstance(item.get("dialogue"), list):
                session_id = str(item.get("session_id", f"session_{index + 1}"))
                timestamp = str(item.get("session_date_time", item.get("timestamp", "")))
                yield session_id, timestamp, item["dialogue"]
            elif isinstance(item, Mapping):
                yield "session_1", "", [item]


def _turn_event(
    turn: Any,
    *,
    sample_id: str,
    session_id: str,
    timestamp: str,
    fallback_index: int,
) -> BenchmarkEvent | None:
    """Convert one LOCOMO dialogue turn into a benchmark event."""

    if not isinstance(turn, Mapping):
        return None
    text = str(turn.get("text") or turn.get("message") or turn.get("content") or "").strip()
    if not text:
        return None
    event_id = str(turn.get("dia_id") or turn.get("turn_id") or f"{session_id}:{fallback_index}")
    return BenchmarkEvent(
        sample_id=sample_id,
        event_id=event_id,
        speaker=str(turn.get("speaker", "")),
        text=text,
        session_id=session_id,
        timestamp=str(turn.get("timestamp", timestamp)),
    )


def _questions(
    sample: Mapping[str, Any],
    *,
    sample_id: str,
) -> Iterable[BenchmarkQuestion]:
    """Yield normalized LOCOMO QA rows."""

    qa_rows = sample.get("qa") or ()
    if not isinstance(qa_rows, Sequence) or isinstance(qa_rows, (str, bytes, bytearray)):
        return
    for index, row in enumerate(qa_rows, start=1):
        if not isinstance(row, Mapping):
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        evidence = row.get("evidence") or ()
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
            evidence_ids = tuple(
                evidence_id
                for item in evidence
                if item is not None
                for evidence_id in (str(item).strip(),)
                if evidence_id
            )
        else:
            evidence_ids = ()
        yield BenchmarkQuestion(
            question_id=f"{sample_id}:q{index}",
            sample_id=sample_id,
            question=question,
            gold_answer=row.get("answer", ""),
            evidence_event_ids=evidence_ids,
            category=str(row.get("category", "")),
        )
