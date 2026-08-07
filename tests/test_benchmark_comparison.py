from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agent_memory.evaluation.comparison import compare_benchmark_runs
from tools.evaluation import compare_memory_systems


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run(
    root: Path,
    *,
    system_id: str,
    condition_id: str | None = None,
    judge_model_id: str = "judge",
    score: float = 1.0,
    label: str = "",
    run_mode: str = "integration-smoke",
    dirty: bool = False,
    frozen_source: bool = False,
    storage_provenance: dict[str, object] | None = None,
) -> Path:
    source_provenance: dict[str, object] = {
        "commit": f"commit-{system_id}",
        "dirty": dirty,
    }
    if frozen_source:
        source_provenance.update(
            {
                "source_snapshot_sha256": "a" * 64,
                "evidence_sha256": "b" * 64,
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 2,
            "benchmark_id": "benchmark",
            "dataset_revision": "revision",
            "dataset_sha256": "sha",
            "bundle_fingerprint": "bundle",
            "policy_input_fingerprint": "input",
            "case_ids": ["case-1"],
            "question_ids": ["q1"],
            "system_id": system_id,
            "condition_id": condition_id or system_id,
            "memory_model_id": "memory",
            "thinking_enabled": False,
            "memory_provider_model_id": f"provider-{system_id}",
            "input_adapter_id": f"input-{system_id}:v1",
            "input_adapter_digest": f"input-digest-{system_id}",
            "retrieval_recipe_id": f"retrieval-{system_id}:v1",
            "retrieval_recipe_digest": f"retrieval-digest-{system_id}",
            "answer_model_id": "answer",
            "answer_thinking_enabled": False,
            "judge_model_id": judge_model_id,
            "judge_thinking_enabled": False,
            "run_mode": run_mode,
            "source_provenance": source_provenance,
            "runtime_provenance": {
                "language": "python",
                "language_version": "3.13",
                "lockfile": "uv.lock",
                "lockfile_sha256": f"lock-{system_id}",
                "dependencies": {"benchmark-runtime": "1.0.0"},
            },
            "storage_provenance": storage_provenance,
            "contract_fingerprints": {"task": "contract"},
            "answer_prompt_digests": {"task": "answer"},
            "answer_parser_contracts": {"task": "parser"},
            "scorer_contracts": {
                "task": {"scorer_id": "exact", "scorer_digest": "score"}
            },
        },
    )
    case_dir = root / "cases" / "case-1"
    _write_json(
        case_dir / "status.json",
        {"status": "completed", "case_id": "case-1"},
    )
    _write_jsonl(
        case_dir / "retrieval.jsonl",
        [{"question_id": "q1", "latency_ms": 2.0}],
    )
    _write_jsonl(
        case_dir / "answers.jsonl",
        [{"question_id": "q1", "answer": f"answer-{system_id}", "latency_ms": 3.0}],
    )
    _write_jsonl(
        case_dir / "grades.jsonl",
        [
            {
                "question_id": "q1",
                "scorer_id": "exact",
                "score": score,
                "label": label,
            }
        ],
    )
    _write_json(
        root / "metrics" / "summary.json",
        {"mean_score": score, "estimated_cost_usd": 0.25},
    )
    return root


def test_comparison_requires_matching_benchmark_contracts(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(
        tmp_path / "second",
        system_id="second",
        judge_model_id="other",
    )

    with pytest.raises(ValueError, match="judge_model_id"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


def test_comparison_writes_question_and_system_rows(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(tmp_path / "second", system_id="second")

    output = compare_benchmark_runs((first, second), tmp_path / "comparison")

    with (output / "per_question.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {(row["condition_id"], row["answer"]) for row in rows} == {
        ("first", "answer-first"),
        ("second", "answer-second"),
    }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["conditions"] == ["first", "second"]
    assert summary["contract"]["bundle_fingerprint"] == "bundle"
    assert summary["condition_contracts"]["first"]["input_adapter_id"] == (
        "input-first:v1"
    )
    assert summary["actual_paid_experiment_cost_usd"] == 0.5
    assert summary["metrics"]["first"]["logical_total_cost_usd"] == 0.25
    with (output / "paired_deltas.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        paired = list(csv.DictReader(handle))
    assert paired == [
        {
            "bootstrap_95_ci_lower": "0.0",
            "bootstrap_95_ci_upper": "0.0",
            "comparison_type": "other_pair",
            "condition_a": "first",
            "condition_b": "second",
            "losses_b": "0",
                "mean_score_delta_b_minus_a": "0.0",
                "question_count": "1",
                "scorer_id": "exact",
                "ties": "1",
                "wins_b": "0",
            }
    ]


def test_comparison_rejects_matching_partial_result_prefixes(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(tmp_path / "second", system_id="second")
    for run_dir in (first, second):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["question_ids"] = ["q1", "q2"]
        _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="question set is incomplete"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


def test_comparison_includes_retrieval_system_error_as_zero(tmp_path: Path) -> None:
    failed = _run(
        tmp_path / "failed",
        system_id="failed",
        score=0.0,
        label="system_error",
    )
    successful = _run(tmp_path / "successful", system_id="successful")

    output = compare_benchmark_runs((failed, successful), tmp_path / "comparison")

    with (output / "per_question.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failed_row = next(row for row in rows if row["condition_id"] == "failed")
    assert failed_row["score"] == "0.0"
    assert failed_row["label"] == "system_error"
    with (output / "paired_deltas.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        paired = list(csv.DictReader(handle))
    assert paired[0]["wins_b"] == "1"
    assert paired[0]["mean_score_delta_b_minus_a"] == "1.0"


def test_comparison_cli_accepts_repeated_run_directories(tmp_path: Path) -> None:
    args = compare_memory_systems.parse_args(
        [
            "--run-dir",
            str(tmp_path / "first"),
            "--run-dir",
            str(tmp_path / "second"),
            "--output-dir",
            str(tmp_path / "comparison"),
        ]
    )

    assert args.run_dirs == (tmp_path / "first", tmp_path / "second")


def test_comparison_counts_shared_maintenance_once(tmp_path: Path) -> None:
    maintenance = tmp_path / "maintenance"
    _write_json(
        maintenance / "metrics" / "summary.json",
        {"estimated_cost_usd": 0.1},
    )
    first = _run(tmp_path / "first", system_id="JM-Q")
    second = _run(tmp_path / "second", system_id="JM-L")
    third = _run(tmp_path / "third", system_id="AM-Mem0-Base")
    for run in (first, second, third):
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["maintenance_checkpoint_source"] = str(maintenance)
        _write_json(run / "manifest.json", manifest)

    output = compare_benchmark_runs((first, second, third), tmp_path / "comparison")
    summary = json.loads((output / "summary.json").read_text())

    assert summary["metrics"]["JM-Q"]["logical_total_cost_usd"] == 0.35
    assert summary["metrics"]["JM-L"]["logical_total_cost_usd"] == 0.35
    assert summary["metrics"]["AM-Mem0-Base"]["logical_total_cost_usd"] == 0.35
    assert summary["actual_paid_experiment_cost_usd"] == 0.85


def test_comparison_scopes_full_parent_run_to_insertion_state(
    tmp_path: Path,
) -> None:
    storage: dict[str, object] = {
        "connector": "qdrant",
        "mode": "embedded-local-single-owner",
        "driver_version": "1.12.1",
        "embedding_runtime_version": "3.4.1",
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "revision",
        "dimensions": 1024,
        "device": "cpu",
        "bm25_enabled": False,
        "entity_boost_enabled": False,
        "reranker_enabled": False,
    }
    parent = _run(tmp_path / "parent", system_id="native-parent")
    _write_json(
        parent / "metrics" / "summary.json",
        {
            "estimated_cost_usd": 0.7,
            "prompt_tokens": 70,
            "completion_tokens": 7,
            "phases": {
                "insertion": {
                    "estimated_cost_usd": 0.1,
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "provider_call_count": 3,
                }
            },
        },
    )
    native = _run(
        tmp_path / "native",
        system_id="native-mem0",
        condition_id="Native-Mem0-Base",
        storage_provenance=storage,
    )
    manifest = json.loads((native / "manifest.json").read_text())
    manifest["maintenance_checkpoint_source"] = {
        "path": str(parent),
        "cost_scope": "insertion",
    }
    _write_json(native / "manifest.json", manifest)
    peer = _run(
        tmp_path / "peer",
        system_id="mem0-memory",
        condition_id="AM-Mem0-Base",
        storage_provenance=storage,
    )

    output = compare_benchmark_runs((native, peer), tmp_path / "comparison")
    summary = json.loads((output / "summary.json").read_text())

    native_metrics = summary["metrics"]["Native-Mem0-Base"]
    assert native_metrics["shared_maintenance_cost_usd"] == 0.1
    assert native_metrics["logical_total_cost_usd"] == 0.35
    assert native_metrics["logical_prompt_tokens"] == 10
    assert native_metrics["logical_completion_tokens"] == 1
    assert summary["actual_paid_experiment_cost_usd"] == 1.2


def test_comparison_classifies_native_pair_from_system_contract(
    tmp_path: Path,
) -> None:
    storage: dict[str, object] = {
        "connector": "qdrant",
        "mode": "embedded-local-single-owner",
        "driver_version": "1.12.1",
        "embedding_runtime_version": "3.4.1",
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "revision",
        "dimensions": 1024,
        "device": "cpu",
        "bm25_enabled": False,
        "entity_boost_enabled": False,
        "reranker_enabled": False,
    }
    native = _run(
        tmp_path / "native",
        system_id="native-mem0",
        condition_id="Native-Mem0-Base",
        storage_provenance=storage,
    )
    agent = _run(
        tmp_path / "agent",
        system_id="mem0-memory",
        condition_id="AM-Mem0-Base",
        storage_provenance=storage,
    )

    output = compare_benchmark_runs((native, agent), tmp_path / "comparison")

    with (output / "paired_deltas.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["comparison_type"] == "native_vs_agent"


def test_mem0_four_condition_comparison_preserves_cost_boundaries(
    tmp_path: Path,
) -> None:
    storage: dict[str, object] = {
        "connector": "qdrant",
        "mode": "embedded-local-single-owner",
        "driver_version": "1.12.1",
        "embedding_runtime_version": "3.4.1",
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "revision",
        "dimensions": 1024,
        "device": "cpu",
        "bm25_enabled": False,
        "entity_boost_enabled": False,
        "reranker_enabled": False,
    }
    maintenance = tmp_path / "maintenance"
    _write_json(
        maintenance / "metrics" / "summary.json",
        {
            "estimated_cost_usd": 0.1,
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "phases": {"insertion": {"provider_call_count": 3}},
        },
    )
    native = _run(
        tmp_path / "native",
        system_id="native-mem0",
        condition_id="Native-Mem0-Base",
        storage_provenance=storage,
    )
    base = _run(
        tmp_path / "base",
        system_id="mem0-memory",
        condition_id="AM-Mem0-Base",
        storage_provenance=storage,
    )
    quick = _run(
        tmp_path / "quick",
        system_id="mem0-enhanced",
        condition_id="AM-Mem0-pairwise-quick",
        storage_provenance=storage,
    )
    listwise = _run(
        tmp_path / "listwise",
        system_id="mem0-enhanced",
        condition_id="AM-Mem0-listwise",
        storage_provenance=storage,
    )
    _write_json(
        native / "metrics" / "summary.json",
        {
            "estimated_cost_usd": 0.4,
            "prompt_tokens": 40,
            "completion_tokens": 4,
            "phases": {"insertion": {"provider_call_count": 2}},
            "actual_provider_usage": {"estimated_cost_usd": 0.4},
            "final_successful_provider_usage": {"estimated_cost_usd": 0.3},
            "recovery_overhead_provider_usage": {"estimated_cost_usd": 0.1},
        },
    )
    for run in (base, quick, listwise):
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["maintenance_checkpoint_source"] = str(maintenance)
        _write_json(run / "manifest.json", manifest)
        _write_json(
            run / "metrics" / "summary.json",
            {
                "estimated_cost_usd": 0.2,
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "phases": {"insertion": {"provider_call_count": 0}},
                "actual_provider_usage": {"estimated_cost_usd": 0.2},
                "final_successful_provider_usage": {
                    "estimated_cost_usd": 0.15
                },
                "recovery_overhead_provider_usage": {
                    "estimated_cost_usd": 0.05
                },
            },
        )

    output = compare_benchmark_runs(
        (native, base, quick, listwise),
        tmp_path / "comparison",
    )
    summary = json.loads((output / "summary.json").read_text())

    assert summary["actual_paid_experiment_cost_usd"] == 1.1
    assert summary["metrics"]["Native-Mem0-Base"][
        "logical_insertion_metrics"
    ]["provider_call_count"] == 2
    for condition in (
        "AM-Mem0-Base",
        "AM-Mem0-pairwise-quick",
        "AM-Mem0-listwise",
    ):
        metrics = summary["metrics"][condition]
        assert metrics["logical_total_cost_usd"] == 0.3
        assert metrics["logical_insertion_metrics"]["provider_call_count"] == 3
        assert metrics["actual_provider_usage"]["estimated_cost_usd"] == 0.2
        assert metrics["final_successful_provider_usage"][
            "estimated_cost_usd"
        ] == 0.15
        assert metrics["recovery_overhead_provider_usage"][
            "estimated_cost_usd"
        ] == 0.05


def test_comparison_keeps_total_cost_unknown_when_usage_is_incomplete(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(tmp_path / "second", system_id="second")
    _write_json(
        first / "metrics" / "summary.json",
        {"estimated_cost_usd": None, "known_cost_usd": 0.1},
    )

    output = compare_benchmark_runs((first, second), tmp_path / "comparison")
    summary = json.loads((output / "summary.json").read_text())

    assert summary["actual_paid_experiment_cost_usd"] is None
    assert summary["metrics"]["first"]["logical_total_cost_usd"] is None
    assert summary["metrics"]["first"]["known_cost_usd"] == 0.1


def test_comparison_rejects_missing_runtime_provenance(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(tmp_path / "second", system_id="second")
    manifest = json.loads((second / "manifest.json").read_text())
    manifest.pop("runtime_provenance")
    _write_json(second / "manifest.json", manifest)

    with pytest.raises(ValueError, match="runtime_provenance"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


def test_comparison_rejects_missing_dependency_versions(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(tmp_path / "second", system_id="second")
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["runtime_provenance"]["dependencies"] = {}
    _write_json(second / "manifest.json", manifest)

    with pytest.raises(ValueError, match="dependency versions"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


def test_comparison_rejects_formal_run_without_source_evidence(
    tmp_path: Path,
) -> None:
    first = _run(
        tmp_path / "first",
        system_id="first",
        run_mode="full",
        dirty=True,
    )
    second = _run(
        tmp_path / "second",
        system_id="second",
        run_mode="full",
        frozen_source=True,
    )

    with pytest.raises(ValueError, match="validated source evidence"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


def test_comparison_rejects_thinking_mismatch(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", system_id="first")
    second = _run(tmp_path / "second", system_id="second")
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["thinking_enabled"] = True
    _write_json(second / "manifest.json", manifest)

    with pytest.raises(ValueError, match="thinking_enabled"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


def test_comparison_rejects_graph_storage_mismatch(tmp_path: Path) -> None:
    storage: dict[str, object] = {
        "connector": "neo4j",
        "image": "neo4j:5.26.2",
        "image_digest": "sha256:image",
        "server_version": "5.26.2",
        "driver_version": "6.1.0",
    }
    first = _run(
        tmp_path / "first",
        system_id="native-graphiti",
        storage_provenance=storage,
    )
    second = _run(
        tmp_path / "second",
        system_id="zep-memory",
        storage_provenance={**storage, "driver_version": "6.2.0"},
    )

    with pytest.raises(ValueError, match="storage_provenance"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")


@pytest.mark.parametrize("second_system_id", ["mem0-memory", "mem0-enhanced"])
def test_comparison_rejects_mem0_qdrant_or_embedding_mismatch(
    tmp_path: Path,
    second_system_id: str,
) -> None:
    storage = {
        "connector": "qdrant",
        "mode": "embedded-local-single-owner",
        "driver_version": "1.12.1",
        "embedding_runtime_version": "3.4.1",
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "revision",
        "dimensions": 1024,
        "device": "cpu",
        "bm25_enabled": False,
        "entity_boost_enabled": False,
        "reranker_enabled": False,
    }
    first = _run(
        tmp_path / "first",
        system_id="native-mem0",
        storage_provenance=storage,
    )
    second = _run(
        tmp_path / "second",
        system_id=second_system_id,
        storage_provenance={**storage, "embedding_revision": "different"},
    )

    with pytest.raises(ValueError, match="storage_provenance"):
        compare_benchmark_runs((first, second), tmp_path / "comparison")
