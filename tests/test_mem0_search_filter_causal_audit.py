"""Tests for the read-only Mem0 Search-Filter causal audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import tarfile
from typing import Any

import pandas as pd
import pytest

from tools.analysis.mem0_search_filter_causal_audit import (
    AuditError,
    DirectoryRunReader,
    RunArtifacts,
    _combine_maintenance_and_retrieval,
    _compare_runs,
    _completed_event_count,
    analyze_top_k_pairs,
    analyze_top_k_pair_parity,
    collect_tar_run_readers,
    load_maintenance,
    load_retrieval,
)
from tools.analysis.semantic_pair_candidates import PairGroup, PairRecord


class _FakeScorer:
    def __init__(self, scores: Mapping[str, Sequence[float]]) -> None:
        self.scores = scores

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"kind": "fake"}

    def score(self, group: PairGroup) -> Sequence[float]:
        return self.scores[group.group_id]


def test_directory_and_tar_condition_loading_match(tmp_path: Path) -> None:
    maintenance = tmp_path / "directory-maintenance"
    retrieval = tmp_path / "directory-retrieval"
    _write_maintenance(maintenance, memories=("memory one", "memory two"))
    _write_retrieval(retrieval)

    directory_run = _combine_maintenance_and_retrieval(
        load_maintenance(DirectoryRunReader(maintenance), name="maintenance"),
        load_retrieval(DirectoryRunReader(retrieval), name="retrieval"),
        name="condition",
    )

    archive = tmp_path / "artifacts.tar"
    _write_tar(
        archive,
        {
            "runs/condition-maintenance": maintenance,
            "runs/condition-retrieval": retrieval,
        },
    )
    readers = collect_tar_run_readers(
        archive,
        {
            "condition-maintenance": "runs/condition-maintenance",
            "condition-retrieval": "runs/condition-retrieval",
        },
    )
    tar_run = _combine_maintenance_and_retrieval(
        load_maintenance(
            readers["condition-maintenance"], name="maintenance"
        ),
        load_retrieval(readers["condition-retrieval"], name="retrieval"),
        name="condition",
    )

    assert tar_run == directory_run


def test_top_k_pair_audit_reports_missed_positive_and_retained_memory() -> None:
    group = PairGroup(
        group_id="group-1",
        operator="sem_filter",
        direction="right-to-left",
        source="fixture",
        case_id="case-1",
        session_id="session-1",
        event_id="event-1",
        query_digest="query-1",
        pairs=(
            _pair("a", "new-1", "earlier A", "later one", False),
            _pair("b", "new-1", "earlier B", "later one", True),
            _pair("c", "new-2", "earlier C", "later two", True),
            _pair("d", "new-2", "earlier D", "later two", False),
        ),
    )
    report = analyze_top_k_pairs(
        iter((group,)),
        _FakeScorer({"group-1": (0.9, 0.8, 0.7, 0.6)}),
        top_k=1,
    )

    assert report["pair_count"] == 4
    assert report["selected_pair_count"] == 2
    assert report["selected_oracle_positive_pair_count"] == 1
    assert report["missed_oracle_positive_pair_count"] == 1
    assert report["counterfactual_retained_later_memories"] == ["later one"]
    assert report["missed_oracle_positive_pairs"][0][
        "candidate_rank_for_later"
    ] == 2


def test_pair_device_parity_compares_candidate_identity_not_only_counts() -> None:
    group = PairGroup(
        group_id="group-1",
        operator="sem_filter",
        direction="right-to-left",
        source="fixture",
        case_id="case-1",
        session_id="session-1",
        event_id="event-1",
        query_digest="query-1",
        pairs=(
            _pair("a", "new", "earlier A", "later", False),
            _pair("b", "new", "earlier B", "later", True),
        ),
    )
    _, same = analyze_top_k_pair_parity(
        iter((group,)),
        _FakeScorer({"group-1": (0.9, 0.8)}),
        _FakeScorer({"group-1": (0.9, 0.8)}),
        top_k=1,
    )
    _, changed = analyze_top_k_pair_parity(
        iter((group,)),
        _FakeScorer({"group-1": (0.9, 0.8)}),
        _FakeScorer({"group-1": (0.8, 0.9)}),
        top_k=1,
    )

    assert same["selected_pair_set_equal"] is True
    assert same["missed_positive_set_equal"] is True
    assert changed["selected_pair_set_equal"] is False
    assert changed["missed_positive_set_equal"] is False


def test_question_decomposition_separates_memory_and_generation_changes() -> None:
    baseline = _run(
        name="oracle",
        memories=("shared",),
        retrieval_memories={"case:q1": ("shared",), "case:q2": ("shared",)},
        answers={"case:q1": "old one", "case:q2": "old two"},
        scores={"case:q1": 0.0, "case:q2": 0.0},
    )
    optimized = _run(
        name="optimized",
        memories=("shared", "retained support"),
        retrieval_memories={
            "case:q1": ("shared",),
            "case:q2": ("retained support",),
        },
        answers={"case:q1": "new one", "case:q2": "new two"},
        scores={"case:q1": 1.0, "case:q2": 1.0},
    )

    comparison = _compare_runs(baseline, optimized)

    assert comparison["context_answer_decomposition"] == {
        "different_context_with_b_only_memory": 1,
        "same_context_different_answer": 1,
    }
    assert comparison["scorers"]["official"]["wins_b"] == 2
    assert comparison["questions"][1]["b_only_retrieved_memories"] == [
        "retained support"
    ]


def test_question_contract_mismatch_is_rejected() -> None:
    baseline = _run(
        name="oracle",
        memories=("shared",),
        retrieval_memories={"case:q1": ("shared",), "case:q2": ("shared",)},
        answers={"case:q1": "one", "case:q2": "two"},
        scores={"case:q1": 0.0, "case:q2": 0.0},
    )
    changed = _run(
        name="optimized",
        memories=("shared",),
        retrieval_memories={"case:q1": ("shared",), "case:q2": ("shared",)},
        answers={"case:q1": "one", "case:q2": "two"},
        scores={"case:q1": 0.0, "case:q2": 0.0},
        second_question="different question",
    )

    from tools.analysis.mem0_search_filter_causal_audit import analyze_conditions

    with pytest.raises(AuditError, match="question contract differs"):
        analyze_conditions(baseline, baseline, changed, pair_audit={})


def test_completed_event_count_requires_explicit_summary_evidence() -> None:
    run = _run(
        name="condition",
        memories=("shared",),
        retrieval_memories={"case:q1": ("shared",), "case:q2": ("shared",)},
        answers={"case:q1": "one", "case:q2": "two"},
        scores={"case:q1": 1.0, "case:q2": 1.0},
    )

    assert _completed_event_count(run) == 2
    assert _completed_event_count(replace(run, summary={"event_count": 3})) == 3
    with pytest.raises(AuditError, match="missing completed event count: condition"):
        _completed_event_count(replace(run, summary={}))


def _pair(
    left_id: str,
    right_id: str,
    left: str,
    right: str,
    baseline_match: bool,
) -> PairRecord:
    return PairRecord(
        pair_id=f"{left_id}:{right_id}",
        left_id=left_id,
        right_id=right_id,
        left=left,
        right=right,
        baseline_match=baseline_match,
    )


def _write_maintenance(root: Path, *, memories: tuple[str, ...]) -> None:
    (root / "metrics").mkdir(parents=True)
    checkpoint = (
        root
        / "cases/case-1/checkpoints/snapshots/events-000419-fixture/driver"
    )
    checkpoint.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"condition_id": "fixture"}), encoding="utf-8"
    )
    (root / "metrics/summary.json").write_text(
        json.dumps(_summary()), encoding="utf-8"
    )
    (root / "metrics/operation_usage.csv").write_text(
        "phase,operator,logical_call_count,provider_call_count,total_tokens,known_cost_usd\n"
        "insertion,sem_flat_map,2,2,20,0.01\n",
        encoding="utf-8",
    )
    (root / "metrics/provider_usage.csv").write_text(
        "phase,operator,total_tokens,known_cost_usd\n"
        "insertion,sem_flat_map,20,0.01\n",
        encoding="utf-8",
    )
    snapshot = {"state": {"memories": pd.DataFrame({"memory": memories})}}
    (checkpoint / "runtime.pkl").write_bytes(pickle.dumps(snapshot))


def _write_retrieval(root: Path) -> None:
    (root / "input").mkdir(parents=True)
    case = root / "cases/case-1"
    case.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"condition_id": "fixture"}), encoding="utf-8"
    )
    questions = (
        {"question_id": "case:q1", "question": "one?", "category": "1"},
        {"question_id": "case:q2", "question": "two?", "category": "2"},
    )
    _write_jsonl(root / "input/questions.jsonl", questions)
    _write_jsonl(
        case / "retrieval.jsonl",
        tuple(
            {
                "question_id": row["question_id"],
                "channels": {"memories": [{"memory": "memory one"}]},
            }
            for row in questions
        ),
    )
    _write_jsonl(
        case / "answers.jsonl",
        tuple(
            {"question_id": row["question_id"], "answer": "answer"}
            for row in questions
        ),
    )
    _write_jsonl(
        case / "grades.jsonl",
        tuple(
            {
                "question_id": row["question_id"],
                "scorer_id": "official",
                "score": 1.0,
            }
            for row in questions
        )
        + (
            {
                "question_id": "case:q1",
                "scorer_id": "judge",
                "score": 1.0,
            },
        ),
    )


def _write_tar(archive: Path, roots: Mapping[str, Path]) -> None:
    with tarfile.open(archive, mode="w") as output:
        for member_root, source_root in roots.items():
            for path in sorted(source_root.rglob("*")):
                if not path.is_file():
                    continue
                value = path.read_bytes()
                info = tarfile.TarInfo(
                    f"{member_root}/{path.relative_to(source_root).as_posix()}"
                )
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _summary() -> Mapping[str, Any]:
    return {
        "completed_event_count": 2,
        "final_successful_provider_usage": {
            "phases": {
                "insertion": {
                    "provider_call_count": 2,
                    "prompt_tokens": 18,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 18,
                    "completion_tokens": 2,
                    "total_tokens": 20,
                    "known_cost_usd": 0.01,
                }
            }
        },
    }


def _run(
    *,
    name: str,
    memories: tuple[str, ...],
    retrieval_memories: Mapping[str, tuple[str, ...]],
    answers: Mapping[str, str],
    scores: Mapping[str, float],
    second_question: str = "question two",
) -> RunArtifacts:
    questions = (
        {"question_id": "case:q1", "question": "question one", "category": "1"},
        {"question_id": "case:q2", "question": second_question, "category": "2"},
    )
    return RunArtifacts(
        name=name,
        manifest={},
        maintenance_manifest={},
        questions=questions,
        retrievals=tuple(
            {
                "question_id": question_id,
                "channels": {
                    "memories": [{"memory": value} for value in values]
                },
            }
            for question_id, values in retrieval_memories.items()
        ),
        answers=tuple(
            {"question_id": question_id, "answer": answer}
            for question_id, answer in answers.items()
        ),
        grades=tuple(
            {
                "question_id": question_id,
                "scorer_id": "official",
                "score": score,
            }
            for question_id, score in scores.items()
        ),
        summary=_summary(),
        operation_usage=(),
        provider_usage=(),
        memories=memories,
        evidence_sha256=f"digest-{name}",
    )
