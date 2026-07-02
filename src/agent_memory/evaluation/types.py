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
