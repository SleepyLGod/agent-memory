"""LOCOMO evaluation dataset normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
)

LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    f"{LOCOMO_COMMIT}/data/locomo10.json"
)


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


def ensure_locomo_dataset(path: Path) -> Path:
    """Download and verify the pinned LOCOMO dataset."""

    if path.exists():
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != LOCOMO_SHA256:
            raise ValueError(
                f"LOCOMO SHA-256 mismatch: expected {LOCOMO_SHA256}, got {actual}"
            )
        return path
    with urlopen(LOCOMO_URL, timeout=60) as response:
        content = response.read()
    actual = sha256(content).hexdigest()
    if actual != LOCOMO_SHA256:
        raise ValueError(
            f"LOCOMO SHA-256 mismatch: expected {LOCOMO_SHA256}, got {actual}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def locomo_bundle(
    dataset_path: Path,
    *,
    sample_index: int = 0,
    start_row: int = 1,
    row_limit: int | None = None,
    question_numbers: Sequence[int] | None = None,
    include_adversarial: bool = True,
    run_mode: str | None = None,
) -> BenchmarkBundle:
    """Build one pinned, evidence-complete LOCOMO case bundle."""

    ensure_locomo_dataset(dataset_path)
    sample = load_locomo_sample(dataset_path, sample_index=sample_index)
    if start_row < 1:
        raise ValueError("start_row must be one-based and at least 1")
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be positive")
    start = start_row - 1
    events = tuple(
        sample.events[start:]
        if row_limit is None
        else sample.events[start : start + row_limit]
    )
    if not events or (row_limit is not None and len(events) != row_limit):
        raise ValueError("LOCOMO event selection is incomplete")

    normalized_numbers = tuple(question_numbers or ())
    if any(number < 1 for number in normalized_numbers):
        raise ValueError("question numbers must be positive")
    if len(normalized_numbers) != len(set(normalized_numbers)):
        raise ValueError("question numbers must be unique")
    by_number = {
        int(question.metadata.get("question_number", 0)): question
        for question in sample.questions
        if include_adversarial or int(question.category) != 5
    }
    if normalized_numbers:
        missing = [number for number in normalized_numbers if number not in by_number]
        if missing:
            raise ValueError(f"LOCOMO questions are unavailable: {missing}")
        questions = tuple(by_number[number] for number in normalized_numbers)
    else:
        questions = tuple(by_number[number] for number in sorted(by_number))

    event_ids = {event.event_id for event in events}
    for question in questions:
        missing_evidence = set(question.evidence_event_ids) - event_ids
        if missing_evidence:
            number = question.metadata.get("question_number")
            raise ValueError(
                f"LOCOMO question {number} requires un-ingested evidence: "
                + ", ".join(sorted(missing_evidence))
            )
    if not questions:
        raise ValueError("LOCOMO selection contains no questions")

    return BenchmarkBundle(
        benchmark_id="locomo",
        dataset_revision=LOCOMO_COMMIT,
        dataset_sha256=LOCOMO_SHA256,
        cases=(
            BenchmarkCase(
                case_id=sample.sample_id,
                task_id="locomo",
                events=events,
                questions=questions,
                metadata={
                    "sample_index": sample_index,
                    "start_row": start_row,
                    "row_limit": row_limit,
                },
            ),
        ),
        metadata={
            "source": "snap-research/locomo",
            **({"run_mode": run_mode} if run_mode is not None else {}),
        },
    )


def normalize_locomo_sample(
    sample: Mapping[str, Any],
    *,
    sample_index: int = 0,
) -> LocomoBenchmarkSample:
    """Normalize one official LOCOMO sample into benchmark events and questions."""

    sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
    events = tuple(
        _conversation_events(
            sample,
            sample_id=sample_id,
            sample_index=sample_index,
        )
    )
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

    if question_limit is not None:
        if question_limit < 0:
            raise ValueError("question_limit must be non-negative")
        if question_limit == 0:
            return ()

    ingested = set(ingested_event_ids)
    selected: list[BenchmarkQuestion] = []
    for question in questions:
        if all(event_id in ingested for event_id in question.evidence_event_ids):
            selected.append(question)
            if question_limit is not None and len(selected) >= question_limit:
                break
    return tuple(selected)


def _load_dataset(dataset_path: Path) -> list[Any]:
    """Read a LOCOMO JSON file as a list of samples."""

    with dataset_path.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def _conversation_events(
    sample: Mapping[str, Any],
    *,
    sample_id: str,
    sample_index: int,
) -> Iterable[BenchmarkEvent]:
    """Yield normalized conversation events from common LOCOMO JSON shapes."""

    conversation = sample.get("conversation")
    row_number = 0
    for session_id, timestamp, turns in _iter_sessions(conversation):
        for turn_index, turn in enumerate(turns, start=1):
            row_number += 1
            event = _turn_event(
                turn,
                sample_id=sample_id,
                sample_index=sample_index,
                session_id=session_id,
                timestamp=timestamp,
                fallback_index=turn_index,
                row_number=row_number,
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
    sample_index: int,
    session_id: str,
    timestamp: str,
    fallback_index: int,
    row_number: int,
) -> BenchmarkEvent | None:
    """Convert one LOCOMO dialogue turn into a benchmark event."""

    if not isinstance(turn, Mapping):
        return None
    text = str(turn.get("text") or turn.get("message") or turn.get("content") or "").strip()
    if not text:
        return None
    event_id = str(turn.get("dia_id") or turn.get("turn_id") or f"{session_id}:{fallback_index}")
    caption = turn.get("blip_caption")
    metadata: dict[str, Any] = {
        "sample_index": sample_index,
        "row_number": row_number,
        "session_number": _session_number(session_id),
    }
    if isinstance(caption, str) and caption.strip():
        metadata["blip_caption"] = caption.strip()
    return BenchmarkEvent(
        sample_id=sample_id,
        event_id=event_id,
        speaker=str(turn.get("speaker", "")),
        text=text,
        session_id=session_id,
        timestamp=str(turn.get("timestamp", timestamp)),
        metadata=metadata,
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
        evidence: object = row.get("evidence") or ()
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
            metadata={
                "question_number": index,
                "adversarial_answer": row.get("adversarial_answer"),
            },
        )


def _session_number(session_id: str) -> int:
    """Return the numeric LOCOMO session suffix when available."""

    suffix = session_id.removeprefix("session_")
    return int(suffix) if suffix.isdigit() else 0
