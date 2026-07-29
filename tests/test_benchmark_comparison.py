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
    judge_model_id: str = "judge",
    score: float = 1.0,
    label: str = "",
    run_mode: str = "integration-smoke",
    dirty: bool = False,
    storage_provenance: dict[str, object] | None = None,
) -> Path:
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
            "condition_id": system_id,
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
            "source_provenance": {"commit": f"commit-{system_id}", "dirty": dirty},
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
            "ties": "1",
            "wins_b": "0",
        }
    ]


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
    for run in (first, second):
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["maintenance_checkpoint_source"] = str(maintenance)
        _write_json(run / "manifest.json", manifest)

    output = compare_benchmark_runs((first, second), tmp_path / "comparison")
    summary = json.loads((output / "summary.json").read_text())

    assert summary["metrics"]["JM-Q"]["logical_total_cost_usd"] == 0.35
    assert summary["metrics"]["JM-L"]["logical_total_cost_usd"] == 0.35
    assert summary["actual_paid_experiment_cost_usd"] == 0.6


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


def test_comparison_rejects_dirty_formal_run(tmp_path: Path) -> None:
    first = _run(
        tmp_path / "first",
        system_id="first",
        run_mode="full",
        dirty=True,
    )
    second = _run(tmp_path / "second", system_id="second", run_mode="full")

    with pytest.raises(ValueError, match="dirty source"):
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
    storage = {
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
