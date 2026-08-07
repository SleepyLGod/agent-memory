"""Deterministic benchmark metric helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re
import string
from statistics import mean
from typing import Any

from nltk.stem import PorterStemmer
import pandas as pd

_ARTICLES = {"a", "an", "the", "and"}
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
_PORTER_STEMMER = PorterStemmer()


def normalize_text(value: Any) -> str:
    """Normalize text for lightweight QA comparison."""

    text = str(value).lower()
    text = text.translate(_PUNCT_TRANSLATION)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def answer_texts(value: Any) -> tuple[str, ...]:
    """Return one or more gold answer strings from LOCOMO-style answer values."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def exact_match(prediction: str, gold_answer: Any) -> bool:
    """Return LOCOMO-style unordered token-set exact match.

    This intentionally mirrors LOCOMO's helper behavior instead of conventional
    normalized string equality. The primary QA score remains locomo_answer_score.
    """

    prediction_tokens = set(normalize_text(prediction).split())
    return any(
        prediction_tokens == set(normalize_text(answer).split())
        for answer in answer_texts(gold_answer)
    )


def contains_answer(text: str, gold_answer: Any) -> bool:
    """Return whether text contains any normalized gold answer."""

    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    return any(
        bool(normalized_answer) and normalized_answer in normalized_text
        for normalized_answer in (normalize_text(answer) for answer in answer_texts(gold_answer))
    )


def token_f1(prediction: str, gold_answer: Any) -> float:
    """Return the best token-overlap F1 against LOCOMO-style gold answers."""

    return max((locomo_f1_score(prediction, answer) for answer in answer_texts(gold_answer)), default=0.0)


def locomo_answer_score(
    prediction: str,
    gold_answer: Any,
    category: str | int,
) -> float:
    """Return the LOCOMO official-compatible answer score for one QA item."""

    category_id = int(category)
    answer = _gold_answer_csv_value(gold_answer)
    if category_id == 3:
        answer = answer.split(";")[0].strip()

    if category_id in {2, 3, 4}:
        return locomo_f1_score(prediction, answer)
    if category_id == 1:
        return locomo_multi_answer_f1(prediction, answer)
    if category_id == 5:
        output = prediction.lower()
        return float(
            "no information available" in output
            or "not mentioned" in output
        )
    raise ValueError(f"Unsupported LOCOMO question category {category!r}")


def locomo_f1_score(prediction: str, gold_answer: Any) -> float:
    """Return LOCOMO-style Porter-stemmed token F1."""

    prediction_tokens = _stemmed_tokens(prediction)
    gold_tokens = _stemmed_tokens(gold_answer)
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def locomo_multi_answer_f1(prediction: str, gold_answer: str) -> float:
    """Return LOCOMO category-1 partial F1 over comma-separated sub-answers."""

    predictions = [part.strip() for part in prediction.split(",")]
    gold_answers = [part.strip() for part in gold_answer.split(",")]
    return mean(
        max(locomo_f1_score(predicted, gold) for predicted in predictions)
        for gold in gold_answers
    )


def retrieval_hit(retrieved_text: str, gold_answer: Any) -> bool:
    """Return the proxy retrieval hit used by retrieval-only diagnostics."""

    return contains_answer(retrieved_text, gold_answer)


def duplicate_name_count(frame: Any, *, column: str = "name") -> int:
    """Return the number of distinct names that appear more than once."""

    if not hasattr(frame, "columns") or column not in frame.columns:
        return 0
    names = [str(value).strip() for value in frame[column].dropna() if str(value).strip()]
    counts = Counter(names)
    return sum(1 for count in counts.values() if count > 1)


def duplicate_name_extra_rows(frame: Any, *, column: str = "name") -> int:
    """Return duplicate rows beyond the first row for each repeated name."""

    if not hasattr(frame, "columns") or column not in frame.columns:
        return 0
    names = [str(value).strip() for value in frame[column].dropna() if str(value).strip()]
    counts = Counter(names)
    return sum(count - 1 for count in counts.values() if count > 1)


def frame_text(frame: Any, *, columns: Sequence[str] | None = None) -> str:
    """Render selected DataFrame columns into plain text for diagnostics."""

    if not hasattr(frame, "empty") or frame.empty:
        return ""
    selected = list(columns or getattr(frame, "columns", ()))
    existing = [column for column in selected if column in frame.columns]
    if not existing:
        return ""
    parts: list[str] = []
    for _index, row in frame[existing].iterrows():
        values = [f"{column}: {row[column]}" for column in existing if not pd.isna(row[column])]
        parts.append(" | ".join(values))
    return "\n".join(parts)


def question_metric_row(
    *,
    question_id: str,
    question: str,
    gold_answer: Any,
    retrieved_frame: Any,
    generated_answer: str | None = None,
    category: str | int | None = None,
) -> dict[str, Any]:
    """Build one deterministic per-question metric row."""

    retrieved_text = frame_text(retrieved_frame)
    row: dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "gold_answer": _gold_answer_csv_value(gold_answer),
        "retrieved_row_count": len(retrieved_frame) if hasattr(retrieved_frame, "__len__") else 0,
        "retrieved_text": retrieved_text,
        "proxy_answer_string_hit": retrieval_hit(retrieved_text, gold_answer),
    }
    if category is not None:
        row["category"] = category
    if generated_answer is not None:
        row.update(
            {
                "generated_answer": generated_answer,
                "answer_exact_match": exact_match(generated_answer, gold_answer),
                "answer_contains_gold": contains_answer(generated_answer, gold_answer),
                "answer_f1": round(token_f1(generated_answer, gold_answer), 6),
            }
        )
        category_id = _locomo_category_id(category)
        if category_id is not None:
            row["locomo_answer_score"] = round(
                locomo_answer_score(generated_answer, gold_answer, category_id),
                6,
            )
    return row


def summarize_question_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-question benchmark metric rows."""

    total = len(rows)
    if total == 0:
        return {
            "questions_evaluated": 0,
            "proxy_answer_string_hit_rate": "",
            "answer_exact_match_rate": "",
            "answer_contains_rate": "",
            "answer_f1_mean": "",
            "locomo_answer_score_mean": "",
        }
    retrieval_hits = sum(bool(row.get("proxy_answer_string_hit")) for row in rows)
    answer_rows = [row for row in rows if "generated_answer" in row]
    summary: dict[str, Any] = {
        "questions_evaluated": total,
        "proxy_answer_string_hit_rate": round(retrieval_hits / total, 6),
    }
    if answer_rows:
        summary.update(
            {
                "answer_exact_match_rate": round(
                    sum(bool(row.get("answer_exact_match")) for row in answer_rows) / len(answer_rows),
                    6,
                ),
                "answer_contains_rate": round(
                    sum(bool(row.get("answer_contains_gold")) for row in answer_rows) / len(answer_rows),
                    6,
                ),
                "answer_f1_mean": round(
                    sum(float(row.get("answer_f1", 0.0)) for row in answer_rows) / len(answer_rows),
                    6,
                ),
                "locomo_answer_score_mean": _mean_available(
                    answer_rows,
                    "locomo_answer_score",
                ),
            }
        )
        summary.update(_category_score_summary(answer_rows))
    else:
        summary.update(
            {
                "answer_exact_match_rate": "",
                "answer_contains_rate": "",
                "answer_f1_mean": "",
                "locomo_answer_score_mean": "",
            }
        )
    return summary


def _stemmed_tokens(value: Any) -> list[str]:
    """Return official-style normalized and Porter-stemmed tokens."""

    return [_PORTER_STEMMER.stem(token) for token in normalize_text(value).split()]


def _locomo_category_id(category: str | int | None) -> int | None:
    """Return a valid LOCOMO category id, or None for absent/dirty values."""

    try:
        category_id = int(category)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if category_id not in {1, 2, 3, 4, 5}:
        return None
    return category_id


def _mean_available(rows: Sequence[Mapping[str, Any]], key: str) -> float | str:
    """Return mean of rows containing a numeric metric key."""

    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    if not values:
        return ""
    return round(sum(values) / len(values), 6)


def _category_score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return per-category LOCOMO answer score counts and means."""

    summary: dict[str, Any] = {}
    categories = sorted({str(row.get("category", "")) for row in rows if row.get("category", "") != ""})
    for category in categories:
        selected = [row for row in rows if str(row.get("category", "")) == category]
        summary[f"category_{category}_count"] = len(selected)
        summary[f"category_{category}_locomo_answer_score_mean"] = _mean_available(
            selected,
            "locomo_answer_score",
        )
    return summary


def _gold_answer_csv_value(gold_answer: Any) -> str:
    """Return a stable CSV value for a gold answer."""

    if isinstance(gold_answer, Sequence) and not isinstance(gold_answer, (str, bytes, bytearray)):
        return "; ".join(str(item) for item in gold_answer)
    text = str(gold_answer)
    return re.sub(r"\s+", " ", text).strip()
