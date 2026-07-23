"""Cross-system comparison guarded by immutable benchmark contracts."""

from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

_CONTRACT_FIELDS = (
    "benchmark_id",
    "dataset_revision",
    "dataset_sha256",
    "bundle_fingerprint",
    "policy_input_fingerprint",
    "case_ids",
    "question_ids",
    "memory_model_id",
    "answer_model_id",
    "judge_model_id",
    "contract_fingerprints",
    "answer_prompt_digests",
    "scorer_contracts",
)


def compare_benchmark_runs(
    run_dirs: tuple[Path, ...],
    output_dir: Path,
) -> Path:
    """Validate comparable run contracts, then write a long-form comparison."""

    if len(run_dirs) < 2:
        raise ValueError("comparison requires at least two benchmark runs")
    manifests = [_read_object(path / "manifest.json") for path in run_dirs]
    conditions = [str(manifest.get("condition_id") or "") for manifest in manifests]
    if any(not condition for condition in conditions) or len(conditions) != len(
        set(conditions)
    ):
        raise ValueError("comparison requires unique non-empty condition_id values")

    reference = manifests[0]
    for field in _CONTRACT_FIELDS:
        expected = reference.get(field)
        for run_dir, manifest in zip(run_dirs[1:], manifests[1:], strict=True):
            if manifest.get(field) != expected:
                raise ValueError(
                    f"benchmark comparison mismatch for {field}: {run_dir}"
                )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_sets = [
        _read_run_results(path, condition_id)
        for path, condition_id in zip(run_dirs, conditions, strict=True)
    ]
    expected_questions = set(result_sets[0])
    for run_dir, results in zip(run_dirs[1:], result_sets[1:], strict=True):
        if set(results) != expected_questions:
            raise ValueError(f"benchmark result question set differs: {run_dir}")

    rows = [
        result_sets[system_index][question_key]
        for question_key in sorted(expected_questions)
        for system_index in range(len(result_sets))
    ]
    _write_csv(output_dir / "per_question.csv", rows)
    _write_csv(
        output_dir / "paired_deltas.csv",
        _paired_delta_rows(result_sets, conditions),
    )
    summaries = {
        condition_id: _logical_condition_metrics(run_dir)
        for run_dir, condition_id in zip(run_dirs, conditions, strict=True)
    }
    actual_paid_costs = [
        _read_object(run_dir / "metrics" / "summary.json").get(
            "estimated_cost_usd"
        )
        for run_dir in run_dirs
    ]
    maintenance_sources = {
        str(manifest.get("maintenance_checkpoint_source"))
        for manifest in manifests
        if manifest.get("maintenance_checkpoint_source")
    }
    actual_paid_costs.extend(
        _read_object(Path(source) / "metrics" / "summary.json").get(
            "estimated_cost_usd"
        )
        for source in sorted(maintenance_sources)
    )
    _write_json(
        output_dir / "summary.json",
        {
            "conditions": conditions,
            "contract": {field: reference.get(field) for field in _CONTRACT_FIELDS},
            "condition_contracts": {
                condition_id: {
                    "system_id": manifest.get("system_id"),
                    "memory_provider_model_id": manifest.get(
                        "memory_provider_model_id"
                    ),
                    "answer_provider_model_id": manifest.get(
                        "answer_provider_model_id"
                    ),
                    "judge_provider_model_id": manifest.get(
                        "judge_provider_model_id"
                    ),
                    "input_adapter_id": manifest.get("input_adapter_id"),
                    "input_adapter_digest": manifest.get("input_adapter_digest"),
                    "retrieval_recipe_id": manifest.get("retrieval_recipe_id"),
                    "retrieval_recipe_digest": manifest.get(
                        "retrieval_recipe_digest"
                    ),
                }
                for condition_id, manifest in zip(
                    conditions, manifests, strict=True
                )
            },
            "metrics": summaries,
            "actual_paid_experiment_cost_usd": _sum_complete(
                actual_paid_costs
            ),
        },
    )
    return output_dir


def _logical_condition_metrics(run_dir: Path) -> dict[str, Any]:
    summary = _read_object(run_dir / "metrics" / "summary.json")
    manifest = _read_object(run_dir / "manifest.json")
    source_value = manifest.get("maintenance_checkpoint_source")
    if not source_value:
        return {
            **summary,
            "actual_run_cost_usd": summary.get("estimated_cost_usd"),
            "shared_maintenance_cost_usd": 0.0,
            "logical_total_cost_usd": summary.get("estimated_cost_usd"),
            "logical_prompt_tokens": summary.get("prompt_tokens"),
            "logical_completion_tokens": summary.get("completion_tokens"),
            "logical_insertion_metrics": _phase_metrics(summary, "insertion"),
        }
    source_summary = _read_object(
        Path(str(source_value)) / "metrics" / "summary.json"
    )
    return {
        **summary,
        "actual_run_cost_usd": summary.get("estimated_cost_usd"),
        "shared_maintenance_cost_usd": source_summary.get("estimated_cost_usd"),
        "logical_total_cost_usd": _sum_complete(
            [
                summary.get("estimated_cost_usd"),
                source_summary.get("estimated_cost_usd"),
            ]
        ),
        "logical_prompt_tokens": int(summary.get("prompt_tokens") or 0)
        + int(source_summary.get("prompt_tokens") or 0),
        "logical_completion_tokens": int(summary.get("completion_tokens") or 0)
        + int(source_summary.get("completion_tokens") or 0),
        "logical_insertion_metrics": _phase_metrics(
            source_summary, "insertion"
        ),
    }


def _sum_complete(values: list[Any]) -> float | None:
    if any(value is None for value in values):
        return None
    return round(sum(float(value) for value in values), 12)


def _phase_metrics(summary: dict[str, Any], phase: str) -> dict[str, Any]:
    phases = summary.get("phases")
    if not isinstance(phases, dict):
        return {}
    value = phases.get(phase)
    return value if isinstance(value, dict) else {}


def _paired_delta_rows(
    result_sets: list[dict[tuple[str, str], dict[str, Any]]],
    conditions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(conditions)), 2):
        left = result_sets[left_index]
        right = result_sets[right_index]
        deltas = [
            float(right[key]["score"]) - float(left[key]["score"])
            for key in sorted(left)
        ]
        lower, upper = _bootstrap_mean_interval(deltas)
        left_condition = conditions[left_index]
        right_condition = conditions[right_index]
        rows.append(
            {
                "condition_a": left_condition,
                "condition_b": right_condition,
                "comparison_type": _comparison_type(
                    left_condition, right_condition
                ),
                "question_count": len(deltas),
                "wins_b": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses_b": sum(delta < 0 for delta in deltas),
                "mean_score_delta_b_minus_a": mean(deltas) if deltas else None,
                "bootstrap_95_ci_lower": lower,
                "bootstrap_95_ci_upper": upper,
            }
        )
    return rows


def _bootstrap_mean_interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    random = Random(0)
    samples = sorted(
        mean(random.choice(values) for _ in values) for _ in range(2_000)
    )
    return samples[int(0.025 * (len(samples) - 1))], samples[
        int(0.975 * (len(samples) - 1))
    ]


def _comparison_type(left: str, right: str) -> str:
    pair = frozenset((left, right))
    if pair == {"JM-Q", "JM-L"}:
        return "retrieval_within_join_map"
    if pair == {"RG-Q", "RG-L"}:
        return "retrieval_within_re_group"
    if pair == {"JM-Q", "RG-Q"}:
        return "maintenance_with_pairwise_quick"
    if pair == {"JM-L", "RG-L"}:
        return "maintenance_with_listwise"
    if "native" in pair:
        return "native_vs_agent"
    return "other_pair"


def _read_run_results(
    run_dir: Path,
    condition_id: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for case_dir in sorted((run_dir / "cases").glob("*")):
        status_path = case_dir / "status.json"
        if not status_path.is_file():
            continue
        status = _read_object(status_path)
        case_id = str(status.get("case_id") or "")
        if status.get("status") != "completed":
            raise ValueError(f"benchmark run contains incomplete case {case_id!r}")
        retrievals = _rows_by_question(case_dir / "retrieval.jsonl")
        answers = _rows_by_question(case_dir / "answers.jsonl")
        grades = _rows_by_question(case_dir / "grades.jsonl")
        question_ids = set(retrievals) | set(answers) | set(grades)
        for question_id in question_ids:
            key = case_id, question_id
            if key in results:
                raise ValueError(f"duplicate benchmark result {key!r}")
            retrieval = retrievals.get(question_id, {})
            answer = answers.get(question_id, {})
            grade = grades.get(question_id, {})
            results[key] = {
                "case_id": case_id,
                "question_id": question_id,
                "condition_id": condition_id,
                "answer": _csv_value(answer.get("answer")),
                "scorer_id": grade.get("scorer_id", ""),
                "score": grade.get("score", ""),
                "label": grade.get("label", ""),
                "retrieval_latency_ms": retrieval.get("latency_ms", ""),
                "answer_latency_ms": answer.get("latency_ms", ""),
            }
    return results


def _rows_by_question(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path}:{line_number} requires question_id")
        if question_id in rows:
            raise ValueError(f"{path} contains duplicate question_id {question_id!r}")
        rows[question_id] = row
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else ()
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["compare_benchmark_runs"]
