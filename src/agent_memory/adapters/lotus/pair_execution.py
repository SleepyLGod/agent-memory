"""Physical candidate generation for pair-shaped semantic operators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from time import perf_counter

import pandas as pd

from agent_memory.storage.embedding import EmbeddingProvider, EmbeddingSpec

SEMANTIC_PAIR_EXECUTION_MODES = ("oracle-only", "search-filter")
SEMANTIC_PAIR_DIRECTIONS = ("symmetric", "left-to-right", "right-to-left")


@dataclass(frozen=True)
class SemanticPairExecutionProfile:
    """One explicit physical implementation for a pair predicate."""

    mode: str
    direction: str
    left_id_columns: tuple[str, ...]
    right_id_columns: tuple[str, ...]
    left_text_columns: tuple[str, ...]
    right_text_columns: tuple[str, ...]
    embedding: EmbeddingSpec | None = None
    top_k: int | None = None
    min_similarity: float | None = None

    def __post_init__(self) -> None:
        """Validate the physical profile without inspecting policy names."""

        if self.mode not in SEMANTIC_PAIR_EXECUTION_MODES:
            raise ValueError(
                "semantic pair mode must be one of: "
                + ", ".join(SEMANTIC_PAIR_EXECUTION_MODES)
            )
        if self.direction not in SEMANTIC_PAIR_DIRECTIONS:
            raise ValueError(
                "semantic pair direction must be one of: "
                + ", ".join(SEMANTIC_PAIR_DIRECTIONS)
            )
        for columns, name in (
            (self.left_id_columns, "left ID columns"),
            (self.right_id_columns, "right ID columns"),
            (self.left_text_columns, "left text columns"),
            (self.right_text_columns, "right text columns"),
        ):
            if not columns or not all(isinstance(column, str) and column for column in columns):
                raise ValueError(f"semantic pair {name} must be non-empty strings")
            if len(set(columns)) != len(columns):
                raise ValueError(f"semantic pair {name} must be unique")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValueError("semantic pair top_k must be a positive integer")
        if self.min_similarity is not None and (
            not isinstance(self.min_similarity, (int, float))
            or isinstance(self.min_similarity, bool)
            or not math.isfinite(float(self.min_similarity))
        ):
            raise ValueError("semantic pair min_similarity must be finite")
        if self.mode == "oracle-only":
            if self.embedding is not None or self.top_k is not None or self.min_similarity is not None:
                raise ValueError("oracle-only semantic pair profiles cannot select candidates")
        elif self.embedding is None:
            raise ValueError("search-filter semantic pair profiles require an embedding")
        elif self.top_k is None and self.min_similarity is None:
            raise ValueError("search-filter requires top_k or min_similarity")

    def to_dict(self) -> dict[str, object]:
        """Return the stable physical profile contract."""

        return {
            "mode": self.mode,
            "direction": self.direction,
            "left_id_columns": list(self.left_id_columns),
            "right_id_columns": list(self.right_id_columns),
            "left_text_columns": list(self.left_text_columns),
            "right_text_columns": list(self.right_text_columns),
            "embedding": None if self.embedding is None else self.embedding.to_dict(),
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable identity for checkpoint and trace provenance."""

        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairCandidateSelection:
    """Candidate rows selected before the oracle predicate."""

    selected_positions: tuple[int, ...]
    similarity_scores: tuple[float, ...]
    total_pair_count: int
    candidate_pair_count: int
    pair_reduction: float
    embedding_latency_ms: float


def select_semantic_pair_candidates(
    source: pd.DataFrame,
    *,
    profile: SemanticPairExecutionProfile,
    embedding_provider: EmbeddingProvider,
) -> PairCandidateSelection:
    """Select pair rows with an embedding threshold and optional directed top-k."""

    if profile.mode != "search-filter" or profile.embedding is None:
        raise ValueError("candidate selection requires a search-filter profile")
    _require_columns(source, profile)
    if source.empty:
        return PairCandidateSelection(
            selected_positions=(),
            similarity_scores=(),
            total_pair_count=0,
            candidate_pair_count=0,
            pair_reduction=0.0,
            embedding_latency_ms=0.0,
        )

    left_ids: list[str] = []
    right_ids: list[str] = []
    pair_texts: list[tuple[str, str]] = []
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    left_texts_by_id: dict[str, str] = {}
    right_texts_by_id: dict[str, str] = {}
    symmetric_texts_by_id: dict[str, str] = {}
    for _, row in source.iterrows():
        left_text = _endpoint_text(row, profile.left_text_columns)
        right_text = _endpoint_text(row, profile.right_text_columns)
        left_id = _endpoint_id(row, profile.left_id_columns)
        right_id = _endpoint_id(row, profile.right_id_columns)
        _record_endpoint_text(left_texts_by_id, left_id, left_text)
        _record_endpoint_text(right_texts_by_id, right_id, right_text)
        if profile.direction == "symmetric":
            _record_endpoint_text(symmetric_texts_by_id, left_id, left_text)
            _record_endpoint_text(symmetric_texts_by_id, right_id, right_text)
        left_ids.append(left_id)
        right_ids.append(right_id)
        pair_texts.append((left_text, right_text))
        for text in (left_text, right_text):
            if text not in seen_texts:
                seen_texts.add(text)
                unique_texts.append(text)

    started = perf_counter()
    vectors = embedding_provider.embed(profile.embedding, unique_texts)
    embedding_latency_ms = (perf_counter() - started) * 1000
    vectors_by_text = _validated_vectors(unique_texts, vectors, profile.embedding)
    scores = tuple(
        _cosine(vectors_by_text[left], vectors_by_text[right])
        for left, right in pair_texts
    )
    selected = _select_positions(
        scores,
        left_ids=left_ids,
        right_ids=right_ids,
        direction=profile.direction,
        top_k=profile.top_k,
        min_similarity=profile.min_similarity,
    )
    total = len(source)
    return PairCandidateSelection(
        selected_positions=selected,
        similarity_scores=scores,
        total_pair_count=total,
        candidate_pair_count=len(selected),
        pair_reduction=0.0 if total == 0 else 1.0 - (len(selected) / total),
        embedding_latency_ms=embedding_latency_ms,
    )


def semantic_pair_profiles_fingerprint(
    profiles: Mapping[str, SemanticPairExecutionProfile],
) -> str:
    """Fingerprint a query-addressed set of physical pair profiles."""

    if not profiles:
        return ""
    payload = {
        query_digest: profile.to_dict()
        for query_digest, profile in sorted(profiles.items())
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_columns(
    source: pd.DataFrame,
    profile: SemanticPairExecutionProfile,
) -> None:
    required = (
        *profile.left_id_columns,
        *profile.right_id_columns,
        *profile.left_text_columns,
        *profile.right_text_columns,
    )
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"semantic pair source columns not found: {missing}")


def _endpoint_text(row: pd.Series, columns: Sequence[str]) -> str:
    return "\n".join(
        f"{column.split(':', 1)[0]}: {row[column]}" for column in columns
    )


def _endpoint_id(row: pd.Series, columns: Sequence[str]) -> str:
    payload = [row[column] for column in columns]
    return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


def _record_endpoint_text(known: dict[str, str], endpoint_id: str, text: str) -> None:
    previous = known.setdefault(endpoint_id, text)
    if previous != text:
        raise ValueError("semantic pair endpoint ID maps to multiple text values")


def _validated_vectors(
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    spec: EmbeddingSpec,
) -> dict[str, tuple[float, ...]]:
    if len(vectors) != len(texts):
        raise ValueError("embedding provider returned an unexpected vector count")
    result: dict[str, tuple[float, ...]] = {}
    for text, vector in zip(texts, vectors, strict=True):
        values = tuple(float(value) for value in vector)
        if len(values) != spec.dimensions:
            raise ValueError("embedding provider returned an unexpected dimension")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding provider returned a non-finite value")
        if not any(value != 0 for value in values):
            raise ValueError("embedding provider returned a zero vector")
        result[text] = values
    return result


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


def _select_positions(
    scores: Sequence[float],
    *,
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    direction: str,
    top_k: int | None,
    min_similarity: float | None,
) -> tuple[int, ...]:
    pair_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, pair in enumerate(zip(left_ids, right_ids, strict=True)):
        pair_positions[pair].append(index)
    representatives = [positions[0] for positions in pair_positions.values()]
    eligible = {
        index
        for index in representatives
        if min_similarity is None or scores[index] >= min_similarity
    }
    if top_k is None:
        selected_pairs = {
            (left_ids[index], right_ids[index]) for index in eligible
        }
        return tuple(
            index
            for index, pair in enumerate(zip(left_ids, right_ids, strict=True))
            if pair in selected_pairs
        )

    buckets: dict[str, list[int]] = defaultdict(list)
    for index in eligible:
        if direction == "left-to-right":
            buckets[f"left:{left_ids[index]}"].append(index)
        elif direction == "right-to-left":
            buckets[f"right:{right_ids[index]}"].append(index)
        else:
            buckets[f"node:{left_ids[index]}"].append(index)
            buckets[f"node:{right_ids[index]}"].append(index)
    selected: set[int] = set()
    for indices in buckets.values():
        indices.sort(
            key=lambda index: (
                -scores[index],
                _endpoint_pair_id(left_ids[index], right_ids[index]),
            )
        )
        selected.update(indices[:top_k])
    selected_pairs = {
        (left_ids[index], right_ids[index]) for index in selected
    }
    return tuple(
        index
        for index, pair in enumerate(zip(left_ids, right_ids, strict=True))
        if pair in selected_pairs
    )


def _endpoint_pair_id(left_id: str, right_id: str) -> str:
    """Return a row-order-independent tie-break identity for one endpoint pair."""

    return json.dumps([left_id, right_id], ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "PairCandidateSelection",
    "SEMANTIC_PAIR_DIRECTIONS",
    "SEMANTIC_PAIR_EXECUTION_MODES",
    "SemanticPairExecutionProfile",
    "select_semantic_pair_candidates",
    "semantic_pair_profiles_fingerprint",
]
