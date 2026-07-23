"""MemoryAgentBench deterministic scorer contracts."""

from __future__ import annotations

import re
import string
from typing import Any, Iterable, Mapping, Sequence

from .tasks import MemoryAgentTask


def normalize_answer(value: str) -> str:
    """Apply the official DRQA-style normalization."""

    text = value.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _flatten_answers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_answers(item))
        return flattened
    return [str(value)]


def exact_match(prediction: str, ground_truth: Any) -> bool:
    """Return official normalized exact match over all accepted answers."""

    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(answer) for answer in _flatten_answers(ground_truth))


def substring_exact_match(prediction: str, ground_truth: Any) -> bool:
    """Return whether any normalized accepted answer occurs in the prediction."""

    normalized = normalize_answer(prediction)
    return any(normalize_answer(answer) in normalized for answer in _flatten_answers(ground_truth))


def parse_output(output: str, answer_prefix: str = "Answer:") -> str:
    """Reproduce the official first-line answer extraction."""

    patterns = (
        re.compile(f"(?:{re.escape(answer_prefix)})(.*)(?:\\n|$)", re.IGNORECASE),
        re.compile(r"^(.*)(?:\n|$)"),
    )
    for pattern in patterns:
        if match := pattern.search(output):
            answer = match.group(1).strip()
            return re.sub(
                f"^{re.escape(answer_prefix)}", "", answer, flags=re.IGNORECASE
            ).strip()
    return ""


def recall_at_k(predicted_items: Sequence[str], ground_truth_items: Iterable[str], k: int) -> float:
    """Compute official recommendation recall for canonicalized movie names."""

    if k <= 0:
        raise ValueError("k must be positive")
    ground_truth = tuple(ground_truth_items)
    if not ground_truth:
        raise ValueError("recall requires at least one ground-truth item")
    top_k = set(predicted_items[:k])
    return sum(item in top_k for item in ground_truth) / len(ground_truth)


def _movie_name(value: str) -> str:
    filename = value.split("/")[-1]
    name = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    return " ".join(re.sub(r"\([^()]*\)", "", name).split())


def _recommendation_names(output: str) -> list[str]:
    try:
        _, recommendations = output.split("1.", maxsplit=1)
    except ValueError:
        recommendations = output.replace(",", "\n")
    names = []
    for line in recommendations.splitlines():
        name = re.sub(r"^(?:\d+[.、)]?\s*[-—–]?\s*)?", "", line.strip())
        name = " ".join(re.sub(r"\([^()]*\)", "", name).split())
        if name:
            names.append(name)
    return names


def score_movie_recommendations(
    prediction: str,
    ground_truth: Any,
    entity_mapping: Mapping[str, int],
) -> tuple[float, list[str], list[str]]:
    """Reproduce official ReDial nearest-name matching and Recall@5."""

    try:
        import editdistance
    except ImportError as exc:
        raise RuntimeError(
            "install agent-memory[benchmarks] for ReDial scoring"
        ) from exc
    id_to_name = {
        entity_id: _movie_name(resource)
        for resource, entity_id in entity_mapping.items()
    }
    candidates = sorted(set(id_to_name.values()))
    if not candidates:
        raise ValueError("movie entity mapping must contain candidates")
    predicted_movies = [
        min(
            candidates,
            key=lambda candidate: (
                editdistance.eval(name.lower(), candidate.lower()),
                candidate,
            ),
        )
        for name in _recommendation_names(prediction)
    ]
    ground_truth_ids = [int(item) for item in _flatten_answers(ground_truth)]
    try:
        ground_truth_movies = [id_to_name[item] for item in ground_truth_ids]
    except KeyError as exc:
        raise ValueError(f"unknown ReDial movie ID {exc.args[0]}") from exc
    return (
        recall_at_k(predicted_movies, ground_truth_movies, 5),
        predicted_movies,
        ground_truth_movies
    )


def score_deterministic(task: MemoryAgentTask, prediction: str, ground_truth: Any) -> float:
    """Score tasks whose official metric does not call an LLM judge."""

    parsed = parse_output(prediction)
    candidates = (prediction, parsed)
    if task.scorer == "exact_match":
        return float(any(exact_match(value, ground_truth) for value in candidates))
    if task.scorer == "substring_exact_match":
        return float(
            any(substring_exact_match(value, ground_truth) for value in candidates)
        )
    if task.scorer == "recall_at_5":
        raise ValueError("Recall@5 requires canonical movie names and entity2id mapping")
    raise ValueError(f"task {task.source!r} requires LLM scorer {task.scorer!r}")


__all__ = [
    "exact_match",
    "normalize_answer",
    "parse_output",
    "recall_at_k",
    "score_deterministic",
    "score_movie_recommendations",
    "substring_exact_match",
]
