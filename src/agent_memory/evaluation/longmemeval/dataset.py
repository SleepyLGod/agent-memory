"""Pinned LongMemEval v1 cleaned dataset normalization."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import urlopen

from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
)

LONGMEMEVAL_CLEANED_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LONGMEMEVAL_CLEANED_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
LONGMEMEVAL_CLEANED_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{LONGMEMEVAL_CLEANED_REVISION}/longmemeval_s_cleaned.json"
)

# Five length-stratified cases per official question type. Two replacements are
# abstention cases at comparable lengths; selection uses no answer or evidence.
LONGMEMEVAL_CLAUDE_PILOT_30_IDS = (
    "852ce960",
    "2133c1b5_abs",
    "01493427",
    "830ce83f",
    "cf22b7bf",
    "ba358f49",
    "87f22b4a",
    "gpt4_372c3eed_abs",
    "81507db6",
    "37f165cf",
    "71a3fd6b",
    "2bf43736",
    "70b3e69b",
    "c7cf7dfd",
    "5809eb10",
    "d6233ab6",
    "54026fce",
    "caf03d32",
    "8a2466db",
    "1c0ddc50",
    "8550ddae",
    "86b68151",
    "29f2956b",
    "3f1e9474",
    "58bf7951",
    "d01c6aa8",
    "gpt4_e061b84f",
    "gpt4_8279ba02",
    "gpt4_e414231e",
    "71017276",
)

LONGMEMEVAL_SMOKE_CASE_ID = "8aef76bc"
LONGMEMEVAL_SMOKE_SESSION_ID = "answer_ultrachat_563222"
LONGMEMEVAL_SMOKE_SOURCE_EVENT_COUNT = 492
LONGMEMEVAL_SMOKE_EVENT_COUNT = 8

_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_longmemeval(destination: Path) -> Path:
    """Download the pinned cleaned file and reject any checksum drift."""

    if destination.exists():
        actual = _sha256_file(destination)
        if actual != LONGMEMEVAL_CLEANED_SHA256:
            raise ValueError(
                f"LongMemEval checksum mismatch: expected "
                f"{LONGMEMEVAL_CLEANED_SHA256}, got {actual}"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(LONGMEMEVAL_CLEANED_URL) as response, partial.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    actual = _sha256_file(partial)
    if actual != LONGMEMEVAL_CLEANED_SHA256:
        raise ValueError(
            f"LongMemEval checksum mismatch: expected "
            f"{LONGMEMEVAL_CLEANED_SHA256}, got {actual}"
        )
    partial.replace(destination)
    return destination


def _parse_clock(hour: str | None, minute: str | None, meridiem: str | None) -> tuple[int, int]:
    parsed_hour = int(hour or 0)
    parsed_minute = int(minute or 0)
    if parsed_minute > 59:
        raise ValueError("minute must be between 0 and 59")
    if meridiem:
        if parsed_hour < 1 or parsed_hour > 12:
            raise ValueError("12-hour clock hour must be between 1 and 12")
        parsed_hour %= 12
        if meridiem.lower() == "pm":
            parsed_hour += 12
    elif parsed_hour > 23:
        raise ValueError("hour must be between 0 and 23")
    return parsed_hour, parsed_minute


def parse_longmemeval_timestamp(value: str) -> datetime:
    """Parse pinned English/numeric timestamps without using process locale."""

    normalized = re.sub(r"\s*\([^)]*\)\s*", " ", value.strip())
    numeric = re.fullmatch(
        r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
        r"(?:[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})"
        r"(?:\s*(?P<meridiem>am|pm))?)?",
        normalized,
        flags=re.IGNORECASE,
    )
    if numeric:
        hour, minute = _parse_clock(
            numeric.group("hour"),
            numeric.group("minute"),
            numeric.group("meridiem"),
        )
        return datetime(
            int(numeric.group("year")),
            int(numeric.group("month")),
            int(numeric.group("day")),
            hour,
            minute,
        )

    named = re.fullmatch(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
        r"(?:\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})"
        r"(?:\s*(?P<meridiem>am|pm))?)?",
        normalized,
        flags=re.IGNORECASE,
    )
    if named:
        month_name = named.group("month").lower()
        if month_name not in _MONTHS:
            raise ValueError(f"unknown English month in timestamp {value!r}")
        hour, minute = _parse_clock(
            named.group("hour"), named.group("minute"), named.group("meridiem")
        )
        return datetime(
            int(named.group("year")),
            _MONTHS[month_name],
            int(named.group("day")),
            hour,
            minute,
        )

    raise ValueError(f"invalid LongMemEval timestamp {value!r}")


def _require_sequence(record: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = record.get(field)
    if not isinstance(value, list):
        raise TypeError(f"LongMemEval field {field!r} must be a list")
    return value


def _normalize_record(record: Mapping[str, Any], source_index: int) -> BenchmarkCase:
    question_id = record.get("question_id")
    question_type = record.get("question_type")
    question = record.get("question")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("LongMemEval question_id must be a non-empty string")
    if not isinstance(question_type, str) or not question_type:
        raise ValueError("LongMemEval question_type must be a non-empty string")
    if not isinstance(question, str) or not question:
        raise ValueError("LongMemEval question must be a non-empty string")

    dates = _require_sequence(record, "haystack_dates")
    session_ids = _require_sequence(record, "haystack_session_ids")
    sessions = _require_sequence(record, "haystack_sessions")
    if not (len(dates) == len(session_ids) == len(sessions)):
        raise ValueError(f"LongMemEval case {question_id!r} has misaligned sessions")

    ordered_sessions: list[tuple[datetime, int, str, str, Sequence[Any]]] = []
    for index, (date, session_id, turns) in enumerate(
        zip(dates, session_ids, sessions, strict=True)
    ):
        if not isinstance(date, str) or not isinstance(session_id, str):
            raise TypeError("LongMemEval session dates and IDs must be strings")
        if not isinstance(turns, list):
            raise TypeError("LongMemEval sessions must contain turn lists")
        ordered_sessions.append(
            (parse_longmemeval_timestamp(date), index, date, session_id, turns)
        )
    ordered_sessions.sort(key=lambda item: (item[0], item[1]))

    answer_session_ids = set(_require_sequence(record, "answer_session_ids"))
    events: list[BenchmarkEvent] = []
    explicit_evidence_ids: list[str] = []
    fallback_evidence_ids: list[str] = []
    for _, original_index, source_date, session_id, turns in ordered_sessions:
        timestamp = parse_longmemeval_timestamp(source_date).isoformat()
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise TypeError("LongMemEval turns must be JSON objects")
            role = turn.get("role")
            content = turn.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("LongMemEval turns require user/assistant role and text")
            event_id = f"{question_id}:s{original_index}:t{turn_index}"
            events.append(
                BenchmarkEvent(
                    sample_id=question_id,
                    event_id=event_id,
                    speaker=role,
                    text=content,
                    session_id=session_id,
                    timestamp=timestamp,
                    metadata={
                        "session_index": original_index,
                        "turn_index": turn_index,
                    },
                )
            )
            if turn.get("has_answer") is True:
                explicit_evidence_ids.append(event_id)
            if session_id in answer_session_ids:
                fallback_evidence_ids.append(event_id)

    question_date = record.get("question_date")
    if not isinstance(question_date, str) or not question_date:
        raise ValueError("LongMemEval question_date must be a non-empty string")
    parse_longmemeval_timestamp(question_date)
    evidence_ids = explicit_evidence_ids or fallback_evidence_ids
    normalized_question = BenchmarkQuestion(
        question_id=question_id,
        sample_id=question_id,
        question=question,
        gold_answer=record.get("answer"),
        evidence_event_ids=tuple(evidence_ids),
        category="abstention" if question_id.endswith("_abs") else question_type,
        metadata={
            "question_date": question_date,
            "question_type": question_type,
        },
    )
    return BenchmarkCase(
        case_id=question_id,
        task_id="longmemeval-v1",
        events=tuple(events),
        questions=(normalized_question,),
        metadata={"source_index": source_index},
    )


def normalize_longmemeval(
    records: Iterable[Mapping[str, Any]],
    *,
    question_ids: Sequence[str] | None = None,
) -> BenchmarkBundle:
    """Normalize selected LongMemEval questions into isolated benchmark cases."""

    if question_ids is not None and len(question_ids) != len(set(question_ids)):
        raise ValueError("LongMemEval question IDs must be unique")
    selected = set(question_ids) if question_ids is not None else None
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for source_index, record in enumerate(records):
        question_id = record.get("question_id")
        if selected is not None and question_id not in selected:
            continue
        case = _normalize_record(record, source_index)
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate LongMemEval question ID {case.case_id!r}")
        seen_ids.add(case.case_id)
        cases.append(case)

    if selected is not None and seen_ids != selected:
        missing = sorted(selected - seen_ids)
        raise ValueError(f"unknown LongMemEval question IDs: {missing}")
    if not cases:
        raise ValueError("LongMemEval selection contains no cases")
    if question_ids is not None:
        cases_by_id = {case.case_id: case for case in cases}
        cases = [cases_by_id[question_id] for question_id in question_ids]
    return BenchmarkBundle(
        benchmark_id="longmemeval-v1-cleaned-s",
        dataset_revision=LONGMEMEVAL_CLEANED_REVISION,
        dataset_sha256=LONGMEMEVAL_CLEANED_SHA256,
        cases=tuple(cases),
        metadata={"source": "xiaowu0162/longmemeval-cleaned"},
    )


def load_longmemeval(
    path: Path,
    *,
    question_ids: Sequence[str] | None = None,
) -> BenchmarkBundle:
    """Verify, load, and normalize the pinned cleaned LongMemEval file."""

    actual = _sha256_file(path)
    if actual != LONGMEMEVAL_CLEANED_SHA256:
        raise ValueError(
            f"LongMemEval checksum mismatch: expected "
            f"{LONGMEMEVAL_CLEANED_SHA256}, got {actual}"
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("pinned LongMemEval cleaned dataset must contain 500 cases")
    return normalize_longmemeval(records, question_ids=question_ids)


def longmemeval_smoke_bundle(bundle: BenchmarkBundle) -> BenchmarkBundle:
    """Return the fixed evidence-complete session prefix used for E2E smoke."""

    if len(bundle.cases) != 1 or bundle.cases[0].case_id != LONGMEMEVAL_SMOKE_CASE_ID:
        raise ValueError(
            "LongMemEval smoke requires only case "
            f"{LONGMEMEVAL_SMOKE_CASE_ID!r}"
        )
    case = bundle.cases[0]
    if len(case.events) != LONGMEMEVAL_SMOKE_SOURCE_EVENT_COUNT:
        raise ValueError(
            "LongMemEval smoke source event count changed: expected "
            f"{LONGMEMEVAL_SMOKE_SOURCE_EVENT_COUNT}, got {len(case.events)}"
        )
    prefix: list[BenchmarkEvent] = []
    for event in case.events:
        if event.session_id != LONGMEMEVAL_SMOKE_SESSION_ID:
            break
        prefix.append(event)
    if len(prefix) != LONGMEMEVAL_SMOKE_EVENT_COUNT:
        raise ValueError(
            "LongMemEval smoke session shape changed: expected "
            f"{LONGMEMEVAL_SMOKE_EVENT_COUNT} events, got {len(prefix)}"
        )
    evidence_ids = set(case.questions[0].evidence_event_ids)
    prefix_ids = {event.event_id for event in prefix}
    if not evidence_ids or not evidence_ids.issubset(prefix_ids):
        raise ValueError("LongMemEval smoke prefix no longer contains all evidence")
    return replace(
        bundle,
        cases=(replace(case, events=tuple(prefix)),),
        metadata={
            **dict(bundle.metadata),
            "run_mode": "integration-smoke",
            "source_case_event_count": len(case.events),
            "included_event_count": len(prefix),
            "included_session_ids": [LONGMEMEVAL_SMOKE_SESSION_ID],
        },
    )


__all__ = [
    "LONGMEMEVAL_CLAUDE_PILOT_30_IDS",
    "LONGMEMEVAL_CLEANED_REVISION",
    "LONGMEMEVAL_CLEANED_SHA256",
    "LONGMEMEVAL_CLEANED_URL",
    "LONGMEMEVAL_SMOKE_CASE_ID",
    "LONGMEMEVAL_SMOKE_EVENT_COUNT",
    "LONGMEMEVAL_SMOKE_SESSION_ID",
    "download_longmemeval",
    "load_longmemeval",
    "longmemeval_smoke_bundle",
    "normalize_longmemeval",
    "parse_longmemeval_timestamp",
]
