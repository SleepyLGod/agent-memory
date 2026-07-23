"""Shared benchmark data types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkEvent:
    """One normalized benchmark input event."""

    sample_id: str
    event_id: str
    speaker: str
    text: str
    session_id: str = ""
    timestamp: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One normalized benchmark question and its gold metadata."""

    question_id: str
    sample_id: str
    question: str
    gold_answer: Any
    evidence_event_ids: tuple[str, ...]
    category: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    """One isolated memory history followed by one or more questions."""

    case_id: str
    task_id: str
    events: tuple[BenchmarkEvent, ...]
    questions: tuple[BenchmarkQuestion, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.events:
            raise ValueError("a benchmark case must contain at least one event")
        if not self.questions:
            raise ValueError("a benchmark case must contain at least one question")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"case {self.case_id!r} contains duplicate event IDs")
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(f"case {self.case_id!r} contains duplicate question IDs")

        if any(event.sample_id != self.case_id for event in self.events):
            raise ValueError("all benchmark events must belong to their case_id")
        if any(question.sample_id != self.case_id for question in self.questions):
            raise ValueError("all benchmark questions must belong to their case_id")


@dataclass(frozen=True)
class RetrievalRequest:
    """Runtime retrieval request derived from a benchmark question."""

    question_id: str
    query_text: str


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark question result row before CSV serialization."""

    question_id: str
    question: str
    gold_answer: Any
    retrieved_row_count: int
    retrieved_text: str
    generated_answer: str | None = None
