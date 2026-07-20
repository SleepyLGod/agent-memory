"""Official-compatible LOCOMO and Zep judge grade records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from agent_memory.evaluation.metrics import locomo_answer_score
from agent_memory.evaluation.types import BenchmarkQuestion


@dataclass(frozen=True)
class OfficialGrade:
    """One deterministic category-specific LOCOMO grade."""

    question_id: str
    category: int
    score: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload."""

        return asdict(self)


@dataclass(frozen=True)
class ZepJudgeGrade:
    """One corrected Zep judge result for a category 1-4 question."""

    question_id: str
    category: int
    label: str
    is_correct: bool
    reasoning: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload."""

        return asdict(self)


def score_official_answer(
    question: BenchmarkQuestion,
    prediction: str,
) -> OfficialGrade:
    """Apply the shared deterministic LOCOMO scorer to one generated answer."""

    started = perf_counter()
    score = locomo_answer_score(
        prediction,
        question.gold_answer,
        question.category,
    )
    return OfficialGrade(
        question_id=question.question_id,
        category=int(question.category),
        score=score,
        latency_ms=round((perf_counter() - started) * 1000, 3),
    )


__all__ = ["OfficialGrade", "ZepJudgeGrade", "score_official_answer"]
