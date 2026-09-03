from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from agent_memory.adapters.lotus.prompt_batching import PromptBatching

from agent_memory.planner.rules import GROUPED_AGG_RULES
from tools.evaluation import locomo, longmemeval, memory_agent_bench


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
@pytest.mark.parametrize("method", ("pairwise-quick", "listwise"))
def test_agent_benchmark_clis_select_sem_join_topk_access_path(
    module, method, tmp_path
) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
            "--grouped-agg-rule",
            "rule-join-map",
            "--sem-join-topk-method",
            method,
        ]
    )

    assert args.sem_join_topk_method == method


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_reject_sem_join_topk_without_zep_join_map(
    module, tmp_path
) -> None:
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "run",
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--output-dir",
                str(tmp_path / "output"),
                "--system",
                "zep-memory",
                "--grouped-agg-rule",
                "rule-re-group",
                "--sem-join-topk-method",
                "listwise",
            ]
        )


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_share_sem_groupby_execution_options(
    module, tmp_path
) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "claude-memory",
            "--sem-groupby-pair-batch-size",
            "12",
            "--sem-groupby-pair-batch-retries",
            "2",
        ]
    )

    assert args.sem_groupby_pair_batch_size == 12
    assert args.sem_groupby_pair_batch_retries == 2
    assert args.semantic_trace_snapshot_mode == "compact"


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
@pytest.mark.parametrize(
    ("value", "expected"),
    (("all", PromptBatching()), ("4", PromptBatching(max_tasks=4))),
)
def test_agent_benchmark_clis_share_prompt_batching_option(
    module, value, expected, tmp_path
) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
            "--prompt-batch-size",
            value,
        ]
    )

    assert args.prompt_batch_size == expected


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_disable_prompt_batching_by_default(
    module, tmp_path
) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
        ]
    )

    assert args.prompt_batch_size is None


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_accept_full_semantic_trace_snapshots(
    module, tmp_path
) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "mem0-memory",
            "--semantic-trace-snapshot-mode",
            "full",
        ]
    )

    assert args.semantic_trace_snapshot_mode == "full"


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
@pytest.mark.parametrize("profile", ("search-filter", "proxy-only"))
def test_agent_benchmark_clis_share_semantic_pair_execution_options(
    module, profile, tmp_path
) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "mem0-memory",
            "--semantic-pair-profile",
            profile,
            "--semantic-pair-top-k",
            "10",
            "--semantic-pair-min-similarity",
            "0.6",
            "--embedding-device",
            "cuda",
        ]
    )

    assert args.semantic_pair_profile == profile
    assert args.semantic_pair_top_k == 10
    assert args.semantic_pair_min_similarity == pytest.approx(0.6)
    assert args.embedding_device == "cuda"


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_share_site_profiles_and_cache_mode(
    module, tmp_path
) -> None:
    profile_path = tmp_path / "profiles.json"
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
            "--semantic-pair-profile-config",
            str(profile_path),
            "--lotus-cache-mode",
            "memory",
        ]
    )

    assert args.semantic_pair_profile_config == profile_path
    assert args.lotus_cache_mode == "memory"


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_share_count_refresh_option(module, tmp_path) -> None:
    args = module.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
            "--refresh-every",
            "4",
        ]
    )

    assert args.refresh_every == 4


@pytest.mark.parametrize("module", (locomo, longmemeval, memory_agent_bench))
def test_agent_benchmark_clis_reject_site_and_global_profiles(
    module, tmp_path
) -> None:
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "run",
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--output-dir",
                str(tmp_path / "output"),
                "--system",
                "zep-memory",
                "--semantic-pair-profile-config",
                str(tmp_path / "profiles.json"),
                "--semantic-pair-profile",
                "oracle-only",
            ]
        )


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
    assert run.system == "claude-memory"
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


def test_longmemeval_cli_selects_zep_memory_without_claude_options(tmp_path) -> None:
    args = longmemeval.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
            "--namespace",
            "longmemeval-zep-smoke",
            "--memory-thinking",
            "enabled",
        ]
    )

    assert args.system == "zep-memory"
    assert args.namespace == "longmemeval-zep-smoke"
    assert args.memory_thinking == "enabled"
    assert args.grouped_agg_rule == "rule-all-group"
    assert longmemeval._condition_id(args) == "zep-memory-rule-all-group"


@pytest.mark.parametrize(
    "option_args",
    (
        ("--sem-topk-method", "listwise"),
        ("--sem-topk-method=listwise",),
    ),
)
def test_longmemeval_cli_rejects_explicit_claude_options_for_zep(
    tmp_path, option_args, capsys
) -> None:
    with pytest.raises(SystemExit):
        longmemeval.parse_args(
            [
                "run",
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--output-dir",
                str(tmp_path / "output"),
                "--system",
                "zep-memory",
                "--namespace",
                "longmemeval-zep-smoke",
                *option_args,
            ]
        )
    assert (
        "only valid with --system claude-memory or --system mem0-enhanced"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("system_id", ("claude-memory", "zep-memory"))
@pytest.mark.parametrize(
    "grouped_agg_rule",
    ("rule-re-group", "rule-join-map"),
)
def test_longmemeval_cli_accepts_planner_strategies_for_all_agent_policies(
    tmp_path, system_id, grouped_agg_rule
) -> None:
    args = longmemeval.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            system_id,
            "--grouped-agg-rule",
            grouped_agg_rule,
        ]
    )

    assert args.grouped_agg_rule == grouped_agg_rule
    assert grouped_agg_rule in longmemeval._condition_id(args)


@pytest.mark.parametrize(
    "abbreviated_option",
    (
        "--grouped-agg-r=rule-re-group",
        "--sem-topk-m=listwise",
    ),
)
def test_longmemeval_cli_rejects_abbreviated_options(
    tmp_path, abbreviated_option, capsys
) -> None:
    with pytest.raises(SystemExit):
        longmemeval.parse_args(
            [
                "run",
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--output-dir",
                str(tmp_path / "output"),
                "--system",
                "zep-memory",
                abbreviated_option,
            ]
        )
    assert "unrecognized arguments" in capsys.readouterr().err


def test_longmemeval_cli_reads_process_argv_at_call_time(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "longmemeval",
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
        ],
    )

    args = longmemeval.parse_args()

    assert args.command == "run"
    assert args.system == "zep-memory"
    assert args.bundle_dir == tmp_path / "bundle"


def test_longmemeval_cli_allows_derived_namespace_for_zep(tmp_path) -> None:
    args = longmemeval.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
        ]
    )

    assert args.namespace is None
    assert longmemeval._condition_id(args) == "zep-memory-rule-all-group"


def test_longmemeval_zep_run_forwards_existing_runner_contract(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    captured = {}
    bundle = SimpleNamespace(benchmark_id="longmemeval-v1-cleaned-s", cases=())
    monkeypatch.setattr(longmemeval, "read_bundle", lambda path: bundle)
    monkeypatch.setattr(
        longmemeval,
        "run_agent_memory_bundle",
        lambda **kwargs: captured.update(kwargs) or output_dir,
    )
    monkeypatch.setattr(longmemeval, "write_official_hypotheses", lambda *args: None)

    result = longmemeval.main(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(output_dir),
            "--system",
            "zep-memory",
            "--namespace",
            "longmemeval-zep-smoke",
            "--condition-id",
            "ZEP-SMOKE",
            "--memory-thinking",
            "enabled",
            "--sem-groupby-pair-batch-size",
            "12",
            "--sem-groupby-pair-batch-retries",
            "2",
            "--maintenance-checkpoint-output-dir",
            str(tmp_path / "maintenance"),
        ]
    )

    assert result == output_dir
    assert captured["system_id"] == "zep-memory"
    assert captured["base_namespace"] == "longmemeval-zep-smoke"
    assert captured["condition_id"] == "ZEP-SMOKE"
    assert captured["memory_thinking_enabled"] is True
    assert captured["sem_groupby_pair_batch_size"] == 12
    assert captured["sem_groupby_pair_batch_retries"] == 2
    assert captured["maintenance_checkpoint_output_dir"] == tmp_path / "maintenance"


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


def test_memory_agent_bench_cli_accepts_explicit_condition_id(tmp_path) -> None:
    args = memory_agent_bench.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "claude-memory",
            "--condition-id",
            "MAB-JM-Q",
        ]
    )

    assert args.condition_id == "MAB-JM-Q"


@pytest.mark.parametrize("grouped_agg_rule", GROUPED_AGG_RULES)
def test_memory_agent_bench_cli_accepts_every_planner_strategy(
    tmp_path, grouped_agg_rule
) -> None:
    args = memory_agent_bench.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "output"),
            "--system",
            "zep-memory",
            "--grouped-agg-rule",
            grouped_agg_rule,
        ]
    )

    assert args.grouped_agg_rule == grouped_agg_rule


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
        lambda path: SimpleNamespace(
            benchmark_id="longmemeval-v1-cleaned-s", cases=()
        ),
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


def test_longmemeval_partial_run_does_not_export_hypotheses(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "partial"
    bundle = SimpleNamespace(
        benchmark_id="longmemeval-v1-cleaned-s",
        cases=(SimpleNamespace(case_id="case-1"),),
    )
    monkeypatch.setattr(longmemeval, "read_bundle", lambda path: bundle)
    monkeypatch.setattr(
        longmemeval,
        "run_agent_memory_bundle",
        lambda **kwargs: output_dir,
    )
    monkeypatch.setattr(
        longmemeval,
        "write_official_hypotheses",
        lambda *args: pytest.fail("partial run must not export hypotheses"),
    )

    result = longmemeval.main(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(output_dir),
            "--max-new-cases",
            "1",
        ]
    )

    assert result == output_dir
