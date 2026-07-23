from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evaluation import longmemeval, memory_agent_bench


def test_longmemeval_cli_separates_bundle_preparation_from_system_run(tmp_path) -> None:
    prepared = longmemeval.parse_args(
        [
            "prepare",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--question-ids",
            "q1",
            "q2",
        ]
    )
    assert prepared.command == "prepare"
    assert prepared.question_ids == ("q1", "q2")

    run = longmemeval.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert run.command == "run"
    assert run.bundle_dir == tmp_path / "bundle"
    assert run.memory_model_id == "deepseek-v4-flash"
    assert run.answer_model == "deepseek/deepseek-v4-flash"
    assert run.judge_model == "deepseek/deepseek-v4-flash"
    assert run.grouped_agg_rule == "rule-all-group"
    assert run.sem_topk_method == "pairwise-naive"
    assert run.memory_thinking == "disabled"

    matrix = longmemeval.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "matrix"),
            "--grouped-agg-rule",
            "rule-re-group",
            "--sem-topk-method",
            "listwise",
            "--memory-thinking",
            "disabled",
            "--maintenance-checkpoint-output-dir",
            str(tmp_path / "maintenance"),
        ]
    )
    assert matrix.grouped_agg_rule == "rule-re-group"
    assert matrix.sem_topk_method == "listwise"
    assert matrix.memory_thinking == "disabled"
    assert matrix.maintenance_checkpoint_output_dir == tmp_path / "maintenance"
    assert longmemeval._condition_id(matrix) == "RG-L"

    maintenance = longmemeval.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "maintenance-output"),
            "--grouped-agg-rule",
            "rule-join-map",
            "--maintenance-only",
        ]
    )
    assert longmemeval._condition_id(maintenance) == "JM-M"


def test_memory_agent_bench_cli_has_explicit_four_source_smoke(tmp_path) -> None:
    args = memory_agent_bench.parse_args(
        [
            "prepare",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--smoke",
        ]
    )

    assert args.command == "prepare"
    assert args.smoke is True
    assert args.sources is None
    assert args.nltk_data_dir.name == "nltk"


def test_agent_cli_rejects_unknown_system(tmp_path) -> None:
    with pytest.raises(SystemExit):
        longmemeval.parse_args(
            [
                "run",
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--system",
                "unknown",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_longmemeval_cli_exposes_fixed_pilot_selection(tmp_path) -> None:
    args = longmemeval.parse_args(
        ["prepare", "--bundle-dir", str(tmp_path / "bundle"), "--pilot-30"]
    )

    assert args.pilot_30 is True


def test_longmemeval_cli_exposes_fixed_integration_smoke(tmp_path) -> None:
    args = longmemeval.parse_args(
        ["prepare", "--bundle-dir", str(tmp_path / "bundle"), "--smoke"]
    )

    assert args.smoke is True
    assert args.pilot_30 is False
    assert args.question_ids is None


def test_longmemeval_maintenance_run_does_not_export_hypotheses(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "maintenance"
    monkeypatch.setattr(
        longmemeval,
        "read_bundle",
        lambda path: SimpleNamespace(benchmark_id="longmemeval-v1-cleaned-s"),
    )
    monkeypatch.setattr(
        longmemeval,
        "run_agent_memory_bundle",
        lambda **kwargs: output_dir,
    )
    monkeypatch.setattr(
        longmemeval,
        "write_official_hypotheses",
        lambda *args: pytest.fail("maintenance must not export hypotheses"),
    )

    result = longmemeval.main(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(output_dir),
            "--maintenance-only",
        ]
    )

    assert result == output_dir
