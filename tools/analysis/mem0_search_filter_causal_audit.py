"""Audit Mem0 Search-Filter quality changes from immutable benchmark artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import random
import re
import tarfile
from typing import Any, Protocol

from tools.analysis.semantic_pair_candidates import (
    CandidateStrategy,
    PairGroup,
    PairScorer,
    SentenceTransformerCosineScorer,
    TarSource,
    _endpoint_pair_id,
    _select_candidates,
)


class AuditError(ValueError):
    """Raised when causal-audit evidence is missing or inconsistent."""


@dataclass(frozen=True)
class MaintenanceEvidence:
    """Immutable maintenance evidence for one benchmark condition."""

    name: str
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    operation_usage: tuple[Mapping[str, str], ...]
    provider_usage: tuple[Mapping[str, str], ...]
    memories: tuple[str, ...] | None
    evidence_sha256: str


@dataclass(frozen=True)
class RetrievalEvidence:
    """Immutable retrieval, answer, and grade evidence for one condition."""

    name: str
    manifest: Mapping[str, Any]
    questions: tuple[Mapping[str, Any], ...]
    retrievals: tuple[Mapping[str, Any], ...]
    answers: tuple[Mapping[str, Any], ...]
    grades: tuple[Mapping[str, Any], ...]
    evidence_sha256: str


@dataclass(frozen=True)
class RunArtifacts:
    """Combined maintenance and retrieval evidence for one condition."""

    name: str
    manifest: Mapping[str, Any]
    maintenance_manifest: Mapping[str, Any]
    questions: tuple[Mapping[str, Any], ...]
    retrievals: tuple[Mapping[str, Any], ...]
    answers: tuple[Mapping[str, Any], ...]
    grades: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    operation_usage: tuple[Mapping[str, str], ...]
    provider_usage: tuple[Mapping[str, str], ...]
    memories: tuple[str, ...] | None
    evidence_sha256: str


class RunReader(Protocol):
    """Read named bytes from an immutable run root."""

    def read(self, relative_path: str) -> bytes:
        """Return one required artifact."""

        ...

    def find_one(self, pattern: re.Pattern[str]) -> tuple[str, bytes]:
        """Return the unique artifact whose relative path matches pattern."""

        ...


class DirectoryRunReader:
    """Read one benchmark run directly from a directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def read(self, relative_path: str) -> bytes:
        target = (self.root / _safe_relative_path(relative_path)).resolve()
        if not target.is_relative_to(self.root):
            raise AuditError(f"artifact escapes run root: {relative_path}")
        try:
            return target.read_bytes()
        except OSError as error:
            raise AuditError(f"cannot read required artifact: {target}") from error

    def find_one(self, pattern: re.Pattern[str]) -> tuple[str, bytes]:
        matches = [
            path
            for path in self.root.rglob("*")
            if path.is_file() and pattern.fullmatch(path.relative_to(self.root).as_posix())
        ]
        if len(matches) != 1:
            raise AuditError(
                f"expected one artifact matching {pattern.pattern!r} under "
                f"{self.root}, found {len(matches)}"
            )
        path = matches[0]
        return path.relative_to(self.root).as_posix(), path.read_bytes()


class CollectedTarRunReader:
    """Read one run from bytes collected during a single tar stream."""

    def __init__(self, *, root: str, members: Mapping[str, bytes]) -> None:
        self.root = root.rstrip("/")
        self.members = dict(members)

    def read(self, relative_path: str) -> bytes:
        relative = _safe_relative_path(relative_path).as_posix()
        try:
            return self.members[relative]
        except KeyError as error:
            raise AuditError(
                f"missing tar artifact: {self.root}/{relative}"
            ) from error

    def find_one(self, pattern: re.Pattern[str]) -> tuple[str, bytes]:
        matches = [
            (name, value)
            for name, value in self.members.items()
            if pattern.fullmatch(name)
        ]
        if len(matches) != 1:
            raise AuditError(
                f"expected one tar artifact matching {pattern.pattern!r} under "
                f"{self.root}, found {len(matches)}"
            )
        return matches[0]


_MAINTENANCE_FIXED_MEMBERS = (
    "manifest.json",
    "metrics/summary.json",
    "metrics/operation_usage.csv",
    "metrics/provider_usage.csv",
)
_RETRIEVAL_FIXED_MEMBERS = ("manifest.json", "input/questions.jsonl")
_CASE_MEMBER_NAMES = frozenset(
    {"retrieval.jsonl", "answers.jsonl", "grades.jsonl"}
)
_FINAL_RUNTIME_PATTERN = re.compile(
    r"cases/[^/]+/checkpoints/snapshots/events-000419-[^/]+/driver/runtime\.pkl"
)


def collect_tar_run_readers(
    archive_path: Path,
    roots: Mapping[str, str],
) -> dict[str, CollectedTarRunReader]:
    """Collect only small required evidence for several runs in one tar pass."""

    archive = archive_path.resolve()
    normalized_roots = {name: root.strip("/") for name, root in roots.items()}
    collected: dict[str, dict[str, bytes]] = {name: {} for name in roots}
    try:
        stream = tarfile.open(archive, mode="r|")
    except (OSError, tarfile.TarError) as error:
        raise AuditError(f"cannot stream tar archive: {archive}") from error
    with stream:
        for member in stream:
            if not member.isfile():
                continue
            for name, root in normalized_roots.items():
                prefix = f"{root}/"
                if not member.name.startswith(prefix):
                    continue
                relative = member.name.removeprefix(prefix)
                if not _wanted_tar_member(relative, source_name=name):
                    break
                handle = stream.extractfile(member)
                if handle is None:
                    raise AuditError(f"cannot read tar member: {member.name}")
                if relative in collected[name]:
                    raise AuditError(f"duplicate tar member: {member.name}")
                collected[name][relative] = handle.read()
                break
    return {
        name: CollectedTarRunReader(root=normalized_roots[name], members=members)
        for name, members in collected.items()
    }


def _wanted_tar_member(relative: str, *, source_name: str) -> bool:
    if source_name.endswith("-maintenance"):
        return relative in _MAINTENANCE_FIXED_MEMBERS or bool(
            source_name != "native-maintenance"
            and _FINAL_RUNTIME_PATTERN.fullmatch(relative)
        )
    if relative in _RETRIEVAL_FIXED_MEMBERS:
        return True
    path = PurePosixPath(relative)
    return (
        len(path.parts) == 3
        and path.parts[0] == "cases"
        and path.parts[2] in _CASE_MEMBER_NAMES
    )


def load_maintenance(
    reader: RunReader, *, name: str, load_memory_view: bool = True
) -> MaintenanceEvidence:
    """Parse maintenance-only evidence from one run root."""

    raw: dict[str, bytes] = {
        path: reader.read(path) for path in _MAINTENANCE_FIXED_MEMBERS
    }
    runtime_path: str | None = None
    runtime_bytes: bytes | None = None
    if load_memory_view:
        runtime_path, runtime_bytes = reader.find_one(_FINAL_RUNTIME_PATTERN)
        raw[runtime_path] = runtime_bytes
    return MaintenanceEvidence(
        name=name,
        manifest=_json_object(raw["manifest.json"], source=f"{name}:manifest.json"),
        summary=_json_object(
            raw["metrics/summary.json"], source=f"{name}:metrics/summary.json"
        ),
        operation_usage=_csv_rows(
            raw["metrics/operation_usage.csv"],
            source=f"{name}:metrics/operation_usage.csv",
        ),
        provider_usage=_csv_rows(
            raw["metrics/provider_usage.csv"],
            source=f"{name}:metrics/provider_usage.csv",
        ),
        memories=(
            _runtime_memories(runtime_bytes, source=f"{name}:{runtime_path}")
            if runtime_bytes is not None and runtime_path is not None
            else None
        ),
        evidence_sha256=_evidence_digest(raw),
    )


def load_retrieval(reader: RunReader, *, name: str) -> RetrievalEvidence:
    """Parse retrieval, answer, and grade evidence from one run root."""

    raw: dict[str, bytes] = {
        path: reader.read(path) for path in _RETRIEVAL_FIXED_MEMBERS
    }
    case_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for filename in sorted(_CASE_MEMBER_NAMES):
        relative, value = reader.find_one(
            re.compile(rf"cases/[^/]+/{re.escape(filename)}")
        )
        raw[relative] = value
        case_rows[filename] = _jsonl(value, source=f"{name}:{relative}")
    questions = _jsonl(
        raw["input/questions.jsonl"], source=f"{name}:input/questions.jsonl"
    )
    question_ids = _unique_rows(questions, "question_id", source=name)
    for filename, rows in case_rows.items():
        if filename == "grades.jsonl":
            grade_groups = _grades_by_scorer(rows)
            grade_ids = set().union(*(set(group) for group in grade_groups.values()))
            if not grade_ids <= question_ids or not any(
                set(group) == question_ids for group in grade_groups.values()
            ):
                raise AuditError(
                    f"grade coverage mismatch in {name}:{filename}"
                )
            continue
        row_ids = _unique_rows(rows, "question_id", source=f"{name}:{filename}")
        if row_ids != question_ids:
            missing = sorted(question_ids - row_ids)[:3]
            extra = sorted(row_ids - question_ids)[:3]
            raise AuditError(
                f"question coverage mismatch in {name}:{filename}; "
                f"missing={missing}, extra={extra}"
            )
    return RetrievalEvidence(
        name=name,
        manifest=_json_object(raw["manifest.json"], source=f"{name}:manifest.json"),
        questions=questions,
        retrievals=case_rows["retrieval.jsonl"],
        answers=case_rows["answers.jsonl"],
        grades=case_rows["grades.jsonl"],
        evidence_sha256=_evidence_digest(raw),
    )


def _evidence_digest(values: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path, value in sorted(values.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value)
    return digest.hexdigest()


def analyze_conditions(
    native: RunArtifacts | None,
    oracle: RunArtifacts,
    optimized: RunArtifacts,
    *,
    pair_audit: Mapping[str, Any],
    native_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one deterministic three-condition causal-audit report."""

    compared_runs = (oracle, optimized) if native is None else (native, oracle, optimized)
    _validate_question_contract(compared_runs)
    conditions: dict[str, Any] = {
        run.name: _condition_summary(run) for run in compared_runs
    }
    retained_memories = set(
        _string_sequence(pair_audit.get("counterfactual_retained_later_memories"))
    )
    comparisons = {
        "optimized_minus_oracle": _compare_runs(
            oracle,
            optimized,
            attributed_memories=retained_memories,
        ),
    }
    if native is not None:
        comparisons["oracle_minus_native"] = _compare_runs(native, oracle)
        comparisons["optimized_minus_native"] = _compare_runs(native, optimized)
    elif native_reference is not None:
        conditions["native"] = dict(native_reference)
        comparisons["native_pairwise_comparison"] = {
            "available": False,
            "reason": "formal Native per-question artifacts are absent from the archive",
        }
    else:
        raise AuditError("native artifacts or a frozen Native reference are required")
    if oracle.memories is None or optimized.memories is None:
        raise AuditError("AM Oracle and Search-Filter memory views are required")
    oracle_memories = set(oracle.memories)
    optimized_memories = set(optimized.memories)
    comparisons["optimized_minus_oracle"]["memory_text_diff"] = {
        "oracle_only_count": len(oracle_memories - optimized_memories),
        "optimized_only_count": len(optimized_memories - oracle_memories),
        "exact_intersection_count": len(oracle_memories & optimized_memories),
        "counterfactual_retained_count": len(retained_memories),
        "counterfactual_retained_present_in_optimized_count": len(
            retained_memories & optimized_memories
        ),
        "counterfactual_retained_present_in_oracle_count": len(
            retained_memories & oracle_memories
        ),
    }
    return {
        "schema_version": 1,
        "analysis_kind": "mem0-search-filter-causal-audit",
        "conditions": conditions,
        "contract_comparison": _contract_comparison(compared_runs),
        "pair_audit": dict(pair_audit),
        "comparisons": comparisons,
        "interpretation_inputs": _interpretation_inputs(
            comparisons["optimized_minus_oracle"], pair_audit
        ),
    }


def analyze_top_k_pairs(
    groups: Iterator[PairGroup],
    scorer: PairScorer,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Audit directed Top-k false negatives and counterfactual retained rows."""

    accumulator = _TopKPairAudit(top_k=top_k, scorer_metadata=scorer.metadata)
    for group in groups:
        scores = tuple(float(value) for value in scorer.score(group))
        accumulator.add(group, scores)
    return accumulator.report()


def analyze_top_k_pair_parity(
    groups: Iterator[PairGroup],
    primary: PairScorer,
    comparison: PairScorer,
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare two scorers over exactly the same streamed pair groups."""

    first = _TopKPairAudit(top_k=top_k, scorer_metadata=primary.metadata)
    second = _TopKPairAudit(top_k=top_k, scorer_metadata=comparison.metadata)
    for group in groups:
        first.add(group, tuple(float(value) for value in primary.score(group)))
        second.add(group, tuple(float(value) for value in comparison.score(group)))
    first_report = first.report()
    second_report = second.report()
    comparison_report = {
        "primary_scorer": first_report["scorer"],
        "comparison_scorer": second_report["scorer"],
        "selected_pair_set_equal": (
            first_report["selected_pair_ids_sha256"]
            == second_report["selected_pair_ids_sha256"]
        ),
        "missed_positive_set_equal": (
            first_report["missed_positive_pair_ids_sha256"]
            == second_report["missed_positive_pair_ids_sha256"]
        ),
        "primary_selected_pair_count": first_report["selected_pair_count"],
        "comparison_selected_pair_count": second_report["selected_pair_count"],
        "primary_missed_positive_count": first_report[
            "missed_oracle_positive_pair_count"
        ],
        "comparison_missed_positive_count": second_report[
            "missed_oracle_positive_pair_count"
        ],
    }
    return first_report, comparison_report


class _TopKPairAudit:
    def __init__(self, *, top_k: int, scorer_metadata: Mapping[str, Any]) -> None:
        self.strategy = CandidateStrategy(top_k=top_k)
        self.scorer_metadata = scorer_metadata
        self.pair_count = 0
        self.positive_count = 0
        self.selected_count = 0
        self.selected_positive_count = 0
        self.missed: list[dict[str, Any]] = []
        self.positive_later_ids: set[str] = set()
        self.selected_positive_later_ids: set[str] = set()
        self.later_text: dict[str, str] = {}
        self.selected_digest = hashlib.sha256()
        self.missed_digest = hashlib.sha256()

    def add(self, group: PairGroup, scores: Sequence[float]) -> None:
        if group.operator != "sem_filter" or group.direction != "right-to-left":
            raise AuditError(
                "Mem0 pair audit requires right-to-left sem_filter groups"
            )
        if len(scores) != len(group.pairs) or not all(map(math.isfinite, scores)):
            raise AuditError(f"invalid pair scores for group {group.group_id}")
        selected = _select_candidates(group, scores, self.strategy)
        ranks = _directed_ranks(group, scores)
        self.pair_count += len(group.pairs)
        self.selected_count += len(selected)
        for index in sorted(selected):
            self.selected_digest.update(
                f"{group.group_id}\0{group.pairs[index].pair_id}\n".encode()
            )
        for index, pair in enumerate(group.pairs):
            self.later_text.setdefault(pair.right_id, pair.right)
            if self.later_text[pair.right_id] != pair.right:
                raise AuditError(f"later endpoint text changed: {pair.right_id}")
            if not pair.baseline_match:
                continue
            self.positive_count += 1
            self.positive_later_ids.add(pair.right_id)
            if index in selected:
                self.selected_positive_count += 1
                self.selected_positive_later_ids.add(pair.right_id)
                continue
            self.missed_digest.update(
                f"{group.group_id}\0{pair.pair_id}\n".encode()
            )
            self.missed.append(
                {
                    "group_id": group.group_id,
                    "event_id": group.event_id,
                    "pair_id": pair.pair_id,
                    "earlier_id": pair.left_id,
                    "later_id": pair.right_id,
                    "earlier_memory": pair.left,
                    "later_memory": pair.right,
                    "cosine_score": scores[index],
                    "candidate_rank_for_later": ranks[index],
                    "oracle_duplicate": True,
                    "researcher_adjudication": None,
                }
            )

    def report(self) -> dict[str, Any]:
        retained_later_ids = (
            self.positive_later_ids - self.selected_positive_later_ids
        )
        return {
            "strategy": self.strategy.strategy_id,
            "scorer": dict(self.scorer_metadata),
            "pair_count": self.pair_count,
            "selected_pair_count": self.selected_count,
            "selected_pair_ids_sha256": self.selected_digest.hexdigest(),
            "pair_reduction": (
                1.0 - self.selected_count / self.pair_count
                if self.pair_count
                else 0.0
            ),
            "oracle_positive_pair_count": self.positive_count,
            "selected_oracle_positive_pair_count": self.selected_positive_count,
            "oracle_positive_recall": (
                self.selected_positive_count / self.positive_count
                if self.positive_count
                else None
            ),
            "missed_oracle_positive_pair_count": len(self.missed),
            "missed_positive_pair_ids_sha256": self.missed_digest.hexdigest(),
            "missed_oracle_positive_pairs": self.missed,
            "oracle_deleted_later_count": len(self.positive_later_ids),
            "counterfactual_retained_later_count": len(retained_later_ids),
            "counterfactual_retained_later_memories": sorted(
                self.later_text[value] for value in retained_later_ids
            ),
        }


def _directed_ranks(group: PairGroup, scores: Sequence[float]) -> dict[int, int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(group.pairs):
        buckets[pair.right_id].append(index)
    ranks: dict[int, int] = {}
    for indices in buckets.values():
        indices.sort(
            key=lambda index: (
                -float(scores[index]),
                _endpoint_pair_id(
                    group.pairs[index].left_id, group.pairs[index].right_id
                ),
            )
        )
        for rank, index in enumerate(indices, start=1):
            ranks[index] = rank
    return ranks


def _condition_summary(run: RunArtifacts) -> dict[str, Any]:
    questions = {str(row["question_id"]): row for row in run.questions}
    grades = _grades_by_scorer(run.grades)
    return {
        "evidence_sha256": run.evidence_sha256,
        "question_count": len(questions),
        "memory_view_available": run.memories is not None,
        "memory_count": len(run.memories) if run.memories is not None else None,
        "exact_memory_text_count": (
            len(set(run.memories)) if run.memories is not None else None
        ),
        "scores": {
            scorer: _score_summary(rows, questions)
            for scorer, rows in sorted(grades.items())
        },
        "insertion_usage": _insertion_usage(run),
    }


def _compare_runs(
    a: RunArtifacts,
    b: RunArtifacts,
    *,
    attributed_memories: set[str] | None = None,
) -> dict[str, Any]:
    questions = {str(row["question_id"]): row for row in a.questions}
    retrieval_a = _rows_by_question(a.retrievals, source=a.name)
    retrieval_b = _rows_by_question(b.retrievals, source=b.name)
    answers_a = _rows_by_question(a.answers, source=a.name)
    answers_b = _rows_by_question(b.answers, source=b.name)
    grades_a = _grades_by_scorer(a.grades)
    grades_b = _grades_by_scorer(b.grades)
    scorer_ids = sorted(set(grades_a) | set(grades_b))
    if set(grades_a) != set(grades_b):
        raise AuditError(f"scorer mismatch: {a.name} vs {b.name}")
    b_only_memories = (
        set(b.memories) - set(a.memories)
        if a.memories is not None and b.memories is not None
        else set()
    )
    attributed = attributed_memories or set()
    question_rows: list[dict[str, Any]] = []
    for question_id in sorted(questions, key=_question_sort_key):
        context_a = _canonical_context(retrieval_a[question_id])
        context_b = _canonical_context(retrieval_b[question_id])
        answer_a = _canonical_text(answers_a[question_id].get("answer"))
        answer_b = _canonical_text(answers_b[question_id].get("answer"))
        retrieved_b_texts = set(_retrieved_memory_texts(retrieval_b[question_id]))
        retrieved_a_texts = set(_retrieved_memory_texts(retrieval_a[question_id]))
        scores = {
            scorer: {
                "a": float(grades_a[scorer][question_id]["score"]),
                "b": float(grades_b[scorer][question_id]["score"]),
                "delta_b_minus_a": float(grades_b[scorer][question_id]["score"])
                - float(grades_a[scorer][question_id]["score"]),
            }
            for scorer in scorer_ids
            if question_id in grades_a[scorer] and question_id in grades_b[scorer]
        }
        question_rows.append(
            {
                "question_id": question_id,
                "category": str(questions[question_id].get("category") or ""),
                "question": str(questions[question_id].get("question") or ""),
                "context_same": context_a == context_b,
                "answer_same": answer_a == answer_b,
                "a_answer": answer_a,
                "b_answer": answer_b,
                "a_retrieved_memories": sorted(retrieved_a_texts),
                "b_retrieved_memories": sorted(retrieved_b_texts),
                "b_only_memory_retrieved": bool(
                    retrieved_b_texts & b_only_memories
                ),
                "b_only_retrieved_memories": sorted(
                    retrieved_b_texts & b_only_memories
                ),
                "attributed_memory_retrieved": bool(
                    retrieved_b_texts & attributed
                ),
                "attributed_retrieved_memories": sorted(
                    retrieved_b_texts & attributed
                ),
                "scores": scores,
            }
        )
    return {
        "a": a.name,
        "b": b.name,
        "scorers": {
            scorer: {
                **_paired_score_summary(
                    [
                        row["scores"][scorer]["delta_b_minus_a"]
                        for row in question_rows
                        if scorer in row["scores"]
                    ]
                ),
                "change_decomposition": _score_change_decomposition(
                    question_rows, scorer=scorer
                ),
            }
            for scorer in scorer_ids
        },
        "context_answer_decomposition": _context_answer_decomposition(question_rows),
        "questions": question_rows,
    }


def _context_answer_decomposition(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row["attributed_memory_retrieved"]:
            label = "search_filter_retained_memory_retrieved"
        elif row["context_same"] and row["answer_same"]:
            label = "same_context_same_answer"
        elif row["context_same"]:
            label = "same_context_different_answer"
        elif row["b_only_memory_retrieved"]:
            label = "different_context_with_b_only_memory"
        else:
            label = "different_context_without_exact_b_only_memory"
        counts[label] += 1
    return dict(sorted(counts.items()))


def _score_change_decomposition(
    rows: Sequence[Mapping[str, Any]], *, scorer: str
) -> dict[str, Mapping[str, int | float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row["scores"].get(scorer)
        if not isinstance(score, Mapping):
            continue
        if row["attributed_memory_retrieved"]:
            label = "search_filter_retained_memory_retrieved"
        elif not row["context_same"]:
            label = "retrieval_context_changed_without_attribution"
        elif not row["answer_same"]:
            label = "same_context_answer_changed"
        elif float(score["delta_b_minus_a"]) != 0.0:
            label = "same_answer_grade_changed"
        else:
            label = "no_observed_change"
        grouped[label].append(float(score["delta_b_minus_a"]))
    denominator = sum(len(values) for values in grouped.values())
    return {
        label: {
            "question_count": len(values),
            "score_delta_sum": sum(values),
            "overall_mean_contribution": (
                sum(values) / denominator if denominator else 0.0
            ),
        }
        for label, values in sorted(grouped.items())
    }


def _paired_score_summary(deltas: Sequence[float]) -> dict[str, Any]:
    if not deltas:
        return {
            "question_count": 0,
            "mean_delta_b_minus_a": None,
            "wins_b": 0,
            "ties": 0,
            "losses_b": 0,
            "bootstrap_95_ci": [None, None],
        }
    epsilon = 1e-12
    return {
        "question_count": len(deltas),
        "mean_delta_b_minus_a": sum(deltas) / len(deltas),
        "wins_b": sum(value > epsilon for value in deltas),
        "ties": sum(abs(value) <= epsilon for value in deltas),
        "losses_b": sum(value < -epsilon for value in deltas),
        "bootstrap_95_ci": list(_bootstrap_mean_interval(deltas)),
    }


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(0)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return means[int(samples * 0.025)], means[int(samples * 0.975) - 1]


def _score_summary(
    rows: Mapping[str, Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for question_id, row in rows.items():
        by_category[str(questions[question_id].get("category") or "")].append(
            float(row["score"])
        )
    all_scores = [score for scores in by_category.values() for score in scores]
    return {
        "question_count": len(all_scores),
        "mean": sum(all_scores) / len(all_scores) if all_scores else None,
        "by_category": {
            category: {
                "question_count": len(scores),
                "mean": sum(scores) / len(scores),
            }
            for category, scores in sorted(by_category.items())
        },
    }


def _insertion_usage(run: RunArtifacts) -> dict[str, Any]:
    summary_usage = run.summary.get("final_successful_provider_usage")
    if not isinstance(summary_usage, Mapping):
        summary_usage = run.summary.get("actual_provider_usage")
    if not isinstance(summary_usage, Mapping):
        raise AuditError(f"missing provider usage summary: {run.name}")
    phases = summary_usage.get("phases")
    insertion = phases.get("insertion") if isinstance(phases, Mapping) else None
    if not isinstance(insertion, Mapping):
        insertion = summary_usage
    events = _completed_event_count(run)
    total_tokens = _number(insertion.get("total_tokens"), default=0.0)
    return {
        "completed_events": events,
        "provider_call_count": int(_number(insertion.get("provider_call_count"), default=0)),
        "prompt_tokens": int(_number(insertion.get("prompt_tokens"), default=0)),
        "cache_hit_tokens": int(_number(insertion.get("cache_hit_tokens"), default=0)),
        "cache_miss_tokens": int(_number(insertion.get("cache_miss_tokens"), default=0)),
        "completion_tokens": int(_number(insertion.get("completion_tokens"), default=0)),
        "total_tokens": int(total_tokens),
        "tokens_per_event": total_tokens / events if events else None,
        "known_cost_usd": insertion.get("known_cost_usd"),
        "operations": _operation_usage(run.operation_usage),
    }


def _completed_event_count(run: RunArtifacts) -> int:
    for key in ("completed_event_count", "event_count"):
        value = run.summary.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise AuditError(f"missing completed event count: {run.name}")


def _operation_usage(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        operation = row.get("operation") or row.get("operator") or row.get("op")
        if not operation:
            continue
        result.append(
            {
                "operation": operation,
                "logical_call_count": _optional_number(row.get("logical_call_count")),
                "provider_call_count": _optional_number(row.get("provider_call_count")),
                "total_tokens": _optional_number(row.get("total_tokens")),
                "known_cost_usd": _optional_number(row.get("known_cost_usd")),
            }
        )
    return result


def _contract_comparison(runs: Sequence[RunArtifacts]) -> dict[str, Any]:
    fields = (
        "policy_input_fingerprint",
        "memory_provider_model_id",
        "thinking_enabled",
        "answer_prompt_digests",
        "official_scorer_digest",
        "zep_judge_digest",
        "retrieval_recipe_id",
        "retrieval_recipe_digest",
    )
    rows: dict[str, Any] = {}
    for field in fields:
        values = {
            run.name: run.manifest.get(field)
            for run in runs
        }
        rows[field] = {"values": values, "all_equal": len(_json_values(values)) == 1}
    rows["known_maintenance_design_difference"] = {
        "native": "message-to-existing-memory top-10 inside one extraction call",
        "agent_memory": "fact extraction followed by fact-to-fact semantic dedup",
    }
    return rows


def _interpretation_inputs(
    comparison: Mapping[str, Any], pair_audit: Mapping[str, Any]
) -> dict[str, Any]:
    decomposition = comparison["context_answer_decomposition"]
    return {
        "same_context_different_answer_count": decomposition.get(
            "same_context_different_answer", 0
        ),
        "search_filter_retained_memory_retrieved_count": decomposition.get(
            "search_filter_retained_memory_retrieved", 0
        ),
        "missed_oracle_positive_pair_count": pair_audit.get(
            "missed_oracle_positive_pair_count"
        ),
        "counterfactual_retained_later_count": pair_audit.get(
            "counterfactual_retained_later_count"
        ),
        "decision": "requires-researcher-adjudication",
    }


def _validate_question_contract(runs: Sequence[RunArtifacts]) -> None:
    baseline = {
        str(row["question_id"]): (
            str(row.get("question") or ""),
            str(row.get("category") or ""),
            json.dumps(row.get("gold_answer"), ensure_ascii=False, sort_keys=True),
        )
        for row in runs[0].questions
    }
    for run in runs[1:]:
        current = {
            str(row["question_id"]): (
                str(row.get("question") or ""),
                str(row.get("category") or ""),
                json.dumps(row.get("gold_answer"), ensure_ascii=False, sort_keys=True),
            )
            for row in run.questions
        }
        if current != baseline:
            raise AuditError(f"question contract differs for {run.name}")


def _runtime_memories(value: bytes, *, source: str) -> tuple[str, ...]:
    try:
        snapshot = pickle.loads(value)
        state = snapshot["state"]
        frame = state["memories"]
        memories = tuple(str(item) for item in frame["memory"].tolist())
    except Exception as error:
        raise AuditError(f"cannot read runtime memories: {source}") from error
    if len(memories) != len(set(memories)):
        raise AuditError(f"runtime memory view contains exact duplicates: {source}")
    return memories


def _canonical_context(row: Mapping[str, Any]) -> tuple[str, ...]:
    memories = _retrieved_memory_texts(row)
    if memories:
        return tuple(_canonical_text(value) for value in memories)
    context = row.get("context")
    return tuple(_canonical_text(context).split("\n"))


def _retrieved_memory_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    channels = row.get("channels")
    if not isinstance(channels, Mapping):
        return ()
    memories = channels.get("memories")
    if not isinstance(memories, list):
        return ()
    result: list[str] = []
    for item in memories:
        if isinstance(item, Mapping) and isinstance(item.get("memory"), str):
            result.append(item["memory"])
    return tuple(result)


def _canonical_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _grades_by_scorer(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        scorer = str(row.get("scorer_id") or "")
        question_id = str(row.get("question_id") or "")
        if not scorer or not question_id or isinstance(row.get("score"), bool):
            raise AuditError("grade row is missing scorer_id, question_id, or score")
        if question_id in result[scorer]:
            raise AuditError(f"duplicate grade: {scorer}:{question_id}")
        result[scorer][question_id] = row
    return dict(result)


def _rows_by_question(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in result:
            raise AuditError(f"invalid or duplicate question row in {source}")
        result[question_id] = row
    return result


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], key: str, *, source: str
) -> set[str]:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise AuditError(f"invalid or duplicate {key} in {source}")
    return set(values)


def _json_object(value: bytes, *, source: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON object: {source}") from error
    if not isinstance(parsed, Mapping):
        raise AuditError(f"expected JSON object: {source}")
    return parsed


def _jsonl(value: bytes, *, source: str) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditError(f"invalid UTF-8 JSONL: {source}") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSONL: {source}:{line_number}") from error
        if not isinstance(row, Mapping):
            raise AuditError(f"expected JSON object: {source}:{line_number}")
        rows.append(row)
    return tuple(rows)


def _csv_rows(value: bytes, *, source: str) -> tuple[Mapping[str, str], ...]:
    try:
        text = value.decode("utf-8")
        return tuple(dict(row) for row in csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise AuditError(f"invalid CSV: {source}") from error


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise AuditError(f"unsafe artifact path: {value}")
    return path


def _number(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"expected numeric value, got {value!r}")
    return float(value)


def _optional_number(value: object) -> int | float | None:
    if value in (None, ""):
        return None
    try:
        number = float(str(value))
    except ValueError as error:
        raise AuditError(f"expected optional numeric CSV value, got {value!r}") from error
    return int(number) if number.is_integer() else number


def _json_values(values: Mapping[str, Any]) -> set[str]:
    return {
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in values.values()
    }


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditError("expected a sequence of counterfactual memory texts")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise AuditError("counterfactual memory texts must be non-empty")
    return result


def _question_sort_key(question_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = question_id.rpartition("q")
    return prefix, int(suffix) if separator and suffix.isdigit() else 0, question_id


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"audit report already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"audit report parent does not exist: {target.parent}")
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the causal-audit CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--native-maintenance-root")
    parser.add_argument("--native-retrieval-root")
    parser.add_argument("--native-reference", type=Path)
    parser.add_argument("--oracle-maintenance-root", required=True)
    parser.add_argument("--oracle-retrieval-root", required=True)
    parser.add_argument("--optimized-maintenance-root", type=Path, required=True)
    parser.add_argument("--optimized-retrieval-root", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--parity-embedding-device")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one read-only Mem0 causal audit."""

    args = build_parser().parse_args(argv)
    native_roots_supplied = bool(
        args.native_maintenance_root or args.native_retrieval_root
    )
    if native_roots_supplied != bool(
        args.native_maintenance_root and args.native_retrieval_root
    ):
        raise AuditError(
            "native maintenance and retrieval roots must be provided together"
        )
    if native_roots_supplied == bool(args.native_reference):
        raise AuditError(
            "provide either Native artifact roots or one frozen Native reference"
        )
    roots = {
        "oracle-maintenance": args.oracle_maintenance_root,
        "oracle-retrieval": args.oracle_retrieval_root,
    }
    if native_roots_supplied:
        roots.update(
            {
                "native-maintenance": args.native_maintenance_root,
                "native-retrieval": args.native_retrieval_root,
            }
        )
    tar_readers = collect_tar_run_readers(args.archive, roots)
    native: RunArtifacts | None = None
    native_reference: Mapping[str, Any] | None = None
    if native_roots_supplied:
        native_maintenance = load_maintenance(
            tar_readers["native-maintenance"],
            name="native-maintenance",
            load_memory_view=False,
        )
        native_retrieval = load_retrieval(
            tar_readers["native-retrieval"], name="native"
        )
        native = _combine_maintenance_and_retrieval(
            native_maintenance, native_retrieval, name="native"
        )
    else:
        if args.native_reference is None:
            raise AuditError("missing Native reference")
        native_reference = _read_native_reference(args.native_reference)
    oracle_maintenance = load_maintenance(
        tar_readers["oracle-maintenance"], name="oracle-maintenance"
    )
    oracle_retrieval = load_retrieval(
        tar_readers["oracle-retrieval"], name="oracle"
    )
    optimized_maintenance = load_maintenance(
        DirectoryRunReader(args.optimized_maintenance_root),
        name="optimized-maintenance",
    )
    optimized_retrieval = load_retrieval(
        DirectoryRunReader(args.optimized_retrieval_root), name="optimized"
    )
    oracle = _combine_maintenance_and_retrieval(
        oracle_maintenance, oracle_retrieval, name="oracle"
    )
    optimized = _combine_maintenance_and_retrieval(
        optimized_maintenance, optimized_retrieval, name="optimized"
    )
    scorer = SentenceTransformerCosineScorer(
        model=args.embedding_model,
        revision=args.embedding_revision,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    pair_source = TarSource(
        args.archive,
        args.oracle_maintenance_root,
        read_workers=1,
    )
    pair_parity: Mapping[str, Any] | None = None
    if args.parity_embedding_device:
        parity_scorer = SentenceTransformerCosineScorer(
            model=args.embedding_model,
            revision=args.embedding_revision,
            device=args.parity_embedding_device,
            batch_size=args.embedding_batch_size,
        )
        pair_audit, pair_parity = analyze_top_k_pair_parity(
            pair_source.iter_groups(phase="insertion"),
            scorer,
            parity_scorer,
            top_k=args.top_k,
        )
    else:
        pair_audit = analyze_top_k_pairs(
            pair_source.iter_groups(phase="insertion"), scorer, top_k=args.top_k
        )
    report = analyze_conditions(
        native,
        oracle,
        optimized,
        pair_audit=pair_audit,
        native_reference=native_reference,
    )
    report["pair_device_parity"] = pair_parity
    report["sources"] = {
        "archive": str(args.archive.resolve()),
        "native_maintenance_root": args.native_maintenance_root,
        "native_retrieval_root": args.native_retrieval_root,
        "native_reference": (
            str(args.native_reference.resolve()) if args.native_reference else None
        ),
        "oracle_maintenance_root": args.oracle_maintenance_root,
        "oracle_retrieval_root": args.oracle_retrieval_root,
        "optimized_maintenance_root": str(args.optimized_maintenance_root.resolve()),
        "optimized_retrieval_root": str(args.optimized_retrieval_root.resolve()),
        "pair_trace": pair_source.description,
        "pair_source_stats": vars(pair_source.stats),
    }
    _write_report(args.output, report)
    return 0


def _read_native_reference(path: Path) -> Mapping[str, Any]:
    value = path.resolve().read_bytes()
    parsed = dict(_json_object(value, source=str(path)))
    parsed["evidence_sha256"] = hashlib.sha256(value).hexdigest()
    parsed["evidence_kind"] = "frozen-report-reference"
    return parsed


def _combine_maintenance_and_retrieval(
    maintenance: MaintenanceEvidence,
    retrieval: RetrievalEvidence,
    *,
    name: str,
) -> RunArtifacts:
    return RunArtifacts(
        name=name,
        manifest=retrieval.manifest,
        maintenance_manifest=maintenance.manifest,
        questions=retrieval.questions,
        retrievals=retrieval.retrievals,
        answers=retrieval.answers,
        grades=retrieval.grades,
        summary=maintenance.summary,
        operation_usage=maintenance.operation_usage,
        provider_usage=maintenance.provider_usage,
        memories=maintenance.memories,
        evidence_sha256=hashlib.sha256(
            f"{maintenance.evidence_sha256}:{retrieval.evidence_sha256}".encode()
        ).hexdigest(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
