from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import pytest

from tools.analysis.prompt_batching_scale_report import generate_report
from tools.evaluation import prompt_batching_scale as controller


EXPECTED_LEVELS = (
    ("original", 1, None),
    ("packed-1", 1, 1),
    ("packed-2", 2, 2),
    ("packed-4", 4, 4),
    ("packed-8", 8, 8),
    ("packed-16", 16, 16),
    ("packed-32", 32, 32),
    ("packed-64", 64, 64),
    ("packed-128", 128, 128),
    ("packed-all", 128, "all"),
)


def test_condition_matrix_runs_two_identical_forward_passes() -> None:
    matrix = controller.condition_matrix()

    assert len(matrix) == 20
    for pass_index in (1, 2):
        rows = [condition for condition in matrix if condition.pass_index == pass_index]
        assert tuple(
            (row.label, row.refresh_every, row.prompt_batch) for row in rows
        ) == EXPECTED_LEVELS
        assert [row.run_id for row in rows] == [
            f"pass-{pass_index}-{label}" for label, _refresh, _prompt in EXPECTED_LEVELS
        ]


@pytest.mark.parametrize(
    ("label", "refresh", "prompt", "expected_prompt"),
    (
        ("original", 1, None, None),
        ("packed-1", 1, 1, "1"),
        ("packed-128", 128, 128, "128"),
        ("packed-all", 128, "all", "all"),
    ),
)
def test_condition_command_preserves_the_fixed_experiment_contract(
    label: str,
    refresh: int,
    prompt: int | str | None,
    expected_prompt: str | None,
    tmp_path: Path,
) -> None:
    condition = controller.ConditionSpec(1, label, refresh, prompt)
    command = controller.build_condition_command(
        condition,
        source=tmp_path / "source",
        venv=tmp_path / "venv",
        bundle=tmp_path / "bundle",
        output=tmp_path / "output",
    )

    assert command[command.index("--system") + 1] == "claude-memory"
    assert command[command.index("--grouped-agg-rule") + 1] == "rule-join-map"
    assert command[command.index("--sem-topk-method") + 1] == "listwise"
    assert command[command.index("--semantic-pair-profile") + 1] == "search-filter"
    assert command[command.index("--semantic-pair-top-k") + 1] == "20"
    assert command[command.index("--semantic-pair-min-similarity") + 1] == "0.5"
    assert command[command.index("--embedding-device") + 1] == "cuda"
    assert command[command.index("--lotus-cache-mode") + 1] == "disabled"
    assert command[command.index("--refresh-every") + 1] == str(refresh)
    if expected_prompt is None:
        assert "--prompt-batch-size" not in command
    else:
        assert command[command.index("--prompt-batch-size") + 1] == expected_prompt


def test_deepseek_cost_uses_request_time_periods() -> None:
    assert controller._is_peak(datetime(2026, 9, 4, 9, 30))
    assert not controller._is_peak(datetime(2026, 9, 4, 13, 30))
    assert not controller._is_peak(datetime(2026, 9, 5, 10, 30))
    usage = {
        "off_peak": {"cache_hit": 1_000_000, "cache_miss": 1_000_000, "completion": 1_000_000},
        "peak": {"cache_hit": 1_000_000, "cache_miss": 1_000_000, "completion": 1_000_000},
    }

    assert controller._usage_cost_cny(usage) == Decimal("18.150000")


def test_initialize_root_allows_only_prepared_source_package(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "package").mkdir()

    controller._validate_initialization_root(tmp_path)

    (tmp_path / "conditions").mkdir()
    with pytest.raises(FileExistsError, match="unexpected entries"):
        controller._validate_initialization_root(tmp_path)


def test_completed_questions_uses_current_artifact_layout(tmp_path: Path) -> None:
    current = tmp_path / "cases" / "case-1" / "question-results" / "q1"
    legacy = tmp_path / "cases" / "case-1" / "questions" / "q2"
    current.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (current / "complete.json").write_text("{}")
    (legacy / "complete.json").write_text("{}")

    assert controller._completed_questions(tmp_path) == 1


def test_monitor_counts_structured_output_repairs(tmp_path: Path) -> None:
    trace = tmp_path / "events.jsonl"
    trace.write_text(
        json.dumps(
            {
                "operator": "sem_flat_map",
                "event_type": "prompt_batching",
                "query_digest": "site",
                "task_count": 1,
                "prompt_count": 1,
                "retry_count": 0,
                "chunk_sizes": [1],
                "structured_output_repair_count": 1,
                "syntax_repair_count": 0,
            }
        )
        + "\n"
    )

    state = controller._update_monitor(trace, tmp_path / "monitor-state.json")

    assert state["prompt_batching"]["sem_flat_map:site"]["repairs"] == 1


def _contract(tmp_path: Path, conditions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment": "claude-prompt-batching-scale-128e",
        "source": {"directory": str(tmp_path / "source")},
        "runtime": {
            "venv": str(tmp_path / "venv"),
            "deps_dir": str(tmp_path / "deps"),
            "hf_home": str(tmp_path / "hf-home"),
        },
        "conditions": conditions,
    }


def test_condition_environment_uses_source_before_dependency_overlay(
    tmp_path: Path,
) -> None:
    paths = controller.ExperimentPaths.from_contract(tmp_path, _contract(tmp_path, []))

    environment = controller._condition_environment(paths)

    assert environment["PYTHONPATH"].split(":") == [
        str((tmp_path / "source" / "src").resolve()),
        str((tmp_path / "deps").resolve()),
    ]


class _FailedProcess:
    returncode = 1
    pid = 123

    def poll(self) -> int:
        return 1


def test_first_condition_failure_stops_all_remaining_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = controller.ConditionSpec(1, "original", 1, None)
    later = controller.ConditionSpec(1, "packed-1", 1, 1)
    second_pass = controller.ConditionSpec(2, "original", 1, None)
    control = tmp_path / "control"
    control.mkdir()
    (tmp_path / "conditions").mkdir()
    (control / "experiment-contract.json").write_text(
        json.dumps(_contract(tmp_path, [row.to_dict() for row in (first, later, second_pass)]))
    )
    (control / "experiment-state").write_text("prepared\n")
    monkeypatch.setattr(controller, "_before_condition", lambda _paths: None)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *_args, **_kwargs: _FailedProcess())

    with pytest.raises(RuntimeError, match="condition failed: pass-1-original"):
        controller.run_experiment(tmp_path)

    assert (tmp_path / "conditions" / first.run_id / "state").read_text().strip() == "failed"
    assert not (tmp_path / "conditions" / later.run_id).exists()
    assert not (tmp_path / "conditions" / second_pass.run_id).exists()


class _SuccessfulProcess:
    returncode = 0
    pid = 456

    def poll(self) -> int:
        return 0


@pytest.mark.parametrize(
    "failure_site",
    ["pid", "_update_monitor", "_total_cost", "_directory_bytes", "_disk_available_gib", "interrupt"],
)
@pytest.mark.parametrize("stop_write_failure", [None, "log", "state"])
def test_supervision_failure_stops_child_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_site: str,
    stop_write_failure: str | None,
) -> None:
    conditions = [
        controller.ConditionSpec(1, "original", 1, None),
        controller.ConditionSpec(2, "original", 1, None),
    ]
    control = tmp_path / "control"
    control.mkdir()
    (control / "experiment-contract.json").write_text(
        json.dumps(_contract(tmp_path, [c.to_dict() for c in conditions]))
    )
    (control / "experiment-state").write_text("prepared\n")
    directory = tmp_path / "conditions" / conditions[0].run_id
    trace = directory / "output" / "trace" / "events.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}\n")
    error = (
        KeyboardInterrupt("interrupted")
        if failure_site == "interrupt"
        else OSError("monitor failed")
    )
    process = Mock(pid=456, returncode=None)
    process.poll.side_effect = lambda: process.returncode

    def wait(**_kwargs: Any) -> int:
        process.returncode = 0
        return 0

    process.wait.side_effect = wait
    killpg = Mock()
    monkeypatch.setattr(controller.os, "killpg", killpg)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(controller, "_before_condition", lambda _p: None)
    monkeypatch.setattr(
        controller, "_update_monitor",
        lambda *_a: {"reasoning_tokens": 0, "provider_contract_violations": []},
    )
    monkeypatch.setattr(controller, "_total_cost", lambda _p: 0)
    monkeypatch.setattr(controller, "_directory_bytes", lambda _p: 0)
    monkeypatch.setattr(controller, "_disk_available_gib", lambda _p: 200)
    if failure_site == "pid":
        original_write = controller._atomic_text

        def write(path: Path, text: str) -> None:
            if path.name == "runner.pid":
                raise error
            original_write(path, text)

        monkeypatch.setattr(controller, "_atomic_text", write)
    elif failure_site == "interrupt":
        monkeypatch.setattr(controller.time, "sleep", Mock(side_effect=error))
    else:
        monkeypatch.setattr(controller, failure_site, Mock(side_effect=error))
    if stop_write_failure == "log":
        original_log = controller._log

        def log(paths: Any, message: str) -> None:
            if message.startswith("SAFETY_STOP"):
                raise OSError("stop log failed")
            original_log(paths, message)

        monkeypatch.setattr(controller, "_log", log)
    elif stop_write_failure == "state":
        original_set_state = controller._set_state

        def set_state(paths: Any, key: str, value: str) -> None:
            if value == "safety-stopped":
                raise OSError("stop state failed")
            original_set_state(paths, key, value)

        monkeypatch.setattr(controller, "_set_state", set_state)

    with pytest.raises(type(error)) as raised:
        controller.run_experiment(tmp_path)

    assert raised.value is error
    killpg.assert_called_once_with(process.pid, controller.signal.SIGINT)
    process.wait.assert_called_once_with(timeout=60)
    assert not (tmp_path / "conditions" / conditions[1].run_id).exists()
    if stop_write_failure is not None:
        assert any(f"stop {stop_write_failure} failed" in note for note in error.__notes__)


def test_safe_stop_preserves_timeout_escalation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = controller.ExperimentPaths.from_contract(tmp_path, _contract(tmp_path, []))
    process = Mock(pid=456)
    process.wait.side_effect = [controller.subprocess.TimeoutExpired("worker", 60), 0]
    killpg = Mock()
    monkeypatch.setattr(controller.os, "killpg", killpg)
    monkeypatch.setattr(controller, "_log", Mock())
    monkeypatch.setattr(controller, "_set_state", Mock())

    controller._safe_stop(paths, process, reason="test")

    assert killpg.call_args_list == [
        call(process.pid, controller.signal.SIGINT),
        call(process.pid, controller.signal.SIGTERM),
    ]
    assert process.wait.call_args_list == [call(timeout=60), call(timeout=30)]


def test_second_pass_starts_only_after_the_first_pass_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conditions = [
        controller.ConditionSpec(1, "original", 1, None),
        controller.ConditionSpec(1, "packed-1", 1, 1),
        controller.ConditionSpec(2, "original", 1, None),
        controller.ConditionSpec(2, "packed-1", 1, 1),
    ]
    control = tmp_path / "control"
    control.mkdir()
    (tmp_path / "conditions").mkdir()
    (control / "experiment-contract.json").write_text(
        json.dumps(_contract(tmp_path, [row.to_dict() for row in conditions]))
    )
    (control / "experiment-state").write_text("prepared\n")
    observed: list[str] = []
    monkeypatch.setattr(controller, "_before_condition", lambda _paths: None)
    monkeypatch.setattr(
        controller.subprocess, "Popen", lambda *_args, **_kwargs: _SuccessfulProcess()
    )

    def validate(
        _paths: controller.ExperimentPaths, condition: controller.ConditionSpec
    ) -> dict[str, float]:
        observed.append(condition.run_id)
        return {"cost_cny": 0.0}

    monkeypatch.setattr(controller, "_validate_condition", validate)
    monkeypatch.setattr(controller, "summarize", lambda _root: {})

    controller.run_experiment(tmp_path)

    assert observed == [row.run_id for row in conditions]
    assert (control / "experiment-state").read_text().strip() == "completed"


def _validation(condition: controller.ConditionSpec) -> dict[str, Any]:
    scale = float(condition.refresh_every)
    return {
        **condition.to_dict(),
        "provider_call_count": int(100 / scale),
        "physical_tokens": int(10_000 / scale),
        "cache_hit_tokens": 100,
        "cache_miss_tokens": 200,
        "completion_tokens": 300,
        "cost_cny": 1 / scale,
        "insertion_excluding_trace_mean_ms": 1000 / scale,
        "insertion_excluding_trace_median_ms": 900 / scale,
        "insertion_excluding_trace_p95_ms": 1500 / scale,
        "insertion_excluding_trace_max_ms": 2000 / scale,
        "condition_end_to_end_wall_seconds": 300 / scale,
        "retrieval_mean_ms": 50,
        "official_locomo_score": 0.55,
        "zep_judge_score": 0.7,
        "candidate_pairs": {
            "sem_join:site": {"eligible": 20, "selected": 10}
        },
        "operator_usage": {
            "sem_join:site": {"responses": 2, "tokens": 500}
        },
        "prompt_batching": {
            "sem_join:site": {
                "tasks": 10,
                "prompts": 2,
                "max_chunk": 5,
                "retries": 0,
                "repairs": 0,
            }
        },
        "final_state": {
            "topics": {"row_count": 5, "multiset_digest": "topic-digest"},
            "catalog": {"row_count": 5, "multiset_digest": "catalog-digest"},
        },
        "artifact_bytes": 1234,
    }


def test_report_is_bounded_and_preserves_both_passes(tmp_path: Path) -> None:
    conditions = [
        controller.ConditionSpec(1, "original", 1, None),
        controller.ConditionSpec(2, "original", 1, None),
    ]
    control = tmp_path / "control"
    control.mkdir()
    (control / "experiment-contract.json").write_text(
        json.dumps(_contract(tmp_path, [row.to_dict() for row in conditions]))
    )
    for condition in conditions:
        directory = tmp_path / "conditions" / condition.run_id
        directory.mkdir(parents=True)
        (directory / "validation.json").write_text(
            json.dumps(_validation(condition))
        )

    summary = generate_report(tmp_path)

    assert summary["completed_conditions"] == 2
    assert summary["complete"] is False
    assert summary["cross_pass"]["original"]["topic_state_equal"] is True
    assert {path.name for path in (tmp_path / "analysis").iterdir()} == {
        "conditions.csv",
        "site_metrics.csv",
        "summary.json",
        "efficiency.svg",
        "quality.svg",
        "pareto.svg",
    }
    assert "pass-1-original" in (tmp_path / "analysis" / "conditions.csv").read_text()
    assert "pass-2-original" in (tmp_path / "analysis" / "conditions.csv").read_text()
    assert (tmp_path / "analysis" / "efficiency.svg").read_text().startswith("<svg")


def _completed_origin(root: Path) -> dict[str, Any]:
    condition = controller.condition_matrix()[0]
    contract = _contract(root, [condition.to_dict()])
    contract.update(input={"bundle_fingerprint": "input"}, fixed_execution={"model": "model"})
    controller._atomic_json(root / "control/experiment-contract.json", contract)
    controller._atomic_text(root / "control/experiment-state", "failed")
    controller._atomic_json(root / "control/source-evidence.json", {"source": "old"})
    directory = root / "conditions" / condition.run_id
    controller._atomic_text(directory / "state", "completed")
    controller._atomic_json(directory / "validation.json", _validation(condition))
    controller._atomic_json(directory / "output/manifest.json", {"source": "old"})
    controller._atomic_json(directory / "output/metrics/summary.json", {
        "completed_cases": 1, "failed_cases": 0, "question_count": 71,
        "memory_system_error_question_count": 0, "retrieval_system_error_count": 0,
        "scores_by_scorer": {
            controller.OFFICIAL_SCORER_ID: {"grade_count": 71},
            controller.JUDGE_SCORER_ID: {"grade_count": 54},
        },
    })
    controller._atomic_json(directory / "monitor-state.json", {"insertion_results": 128})
    for number in range(71):
        controller._atomic_json(directory / f"output/cases/case/question-results/{number}/complete.json", {})
    trace = directory / "output/trace/events.jsonl"
    controller._atomic_text(trace, "")
    # Failed attempts must contribute to the carried budget as well.
    controller._atomic_text(root / "conditions/pass-1-packed-1/output/trace/events.jsonl", json.dumps({
        "event_type": "provider_usage", "timestamp": "20260906T080000000000Z",
        "provider_prompt_cache_hit_tokens": 0, "provider_prompt_cache_miss_tokens": 1_000_000,
        "provider_completion_tokens": 0,
    }) + "\n")
    return contract


def test_continuation_preserves_origin_and_failed_cost(tmp_path: Path) -> None:
    origin = tmp_path / "old"
    contract = _completed_origin(origin)
    before = {str(p): p.read_bytes() for p in origin.rglob("*") if p.is_file()}
    continuation = controller._capture_continuation(origin, ("pass-1-original",), contract)
    assert Decimal(continuation["prior_cost_cny"]) == Decimal("1.5")
    assert {str(p): p.read_bytes() for p in origin.rglob("*") if p.is_file()} == before
    new = tmp_path / "new"
    contract = {**contract, "continuation": continuation}
    controller._atomic_json(new / "control/experiment-contract.json", contract)
    paths = controller.ExperimentPaths.from_contract(new, contract)
    controller._atomic_json(new / "conditions/pass-1-packed-16/monitor-state.json", {"cost_cny": 2})
    assert controller._total_cost(paths) == Decimal("3.5")
    controller._validate_continuation(contract)
    summary = controller.summarize(new)
    assert summary["runs"][0]["result_origin"]["status"] == "reused-completed"
    assert summary["runs"][0]["result_origin"]["source"] == contract["source"]
    report = generate_report(new)
    assert report["result_origins"]["pass-1-original"]["status"] == "reused-completed"
    assert not (new / "conditions/pass-1-original").exists()
    (origin / "conditions/pass-1-original/validation.json").write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        controller._validate_continuation(contract)


@pytest.mark.parametrize("damage", ("state", "questions", "summary", "input", "duplicate", "unknown"))
def test_continuation_rejects_invalid_reuse(tmp_path: Path, damage: str) -> None:
    contract = _completed_origin(tmp_path)
    directory = tmp_path / "conditions/pass-1-original"
    ids = ("pass-1-original",)
    if damage == "state":
        (directory / "state").write_text("failed")
    elif damage == "questions":
        controller._atomic_json(directory / "output/cases/case/question-results/extra/complete.json", {})
    elif damage == "summary":
        (directory / "output/metrics/summary.json").write_text("{}")
    elif damage == "input":
        contract = {**contract, "input": {"bundle_fingerprint": "different"}}
    elif damage == "duplicate":
        ids = ("pass-1-original", "pass-1-original")
    else:
        ids = ("pass-1-unknown",)
    with pytest.raises((RuntimeError, ValueError)):
        controller._capture_continuation(tmp_path, ids, contract)


def test_reused_condition_is_skipped_but_failure_blocks_second_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = tmp_path / "old"
    old_contract = _completed_origin(old)
    root = tmp_path / "new"
    conditions = [controller.condition_matrix()[i] for i in (0, 5, 10)]
    contract = _contract(root, [c.to_dict() for c in conditions])
    contract["continuation"] = controller._capture_continuation(old, ("pass-1-original",), old_contract)
    controller._atomic_json(root / "control/experiment-contract.json", contract)
    controller._atomic_text(root / "control/experiment-state", "prepared")
    monkeypatch.setattr(controller, "_before_condition", lambda _paths: None)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *_args, **_kwargs: _FailedProcess())
    with pytest.raises(RuntimeError, match="condition failed: pass-1-packed-16"):
        controller.run_experiment(root)
    assert not (root / "conditions/pass-1-original").exists()
    assert not (root / "conditions/pass-2-original").exists()


def test_monitor_includes_answer_judge_without_counting_lotus_mirrors(tmp_path: Path) -> None:
    direct = {
        "event_type": "llm_call", "operator": "llm", "phase": "answering",
        "timestamp": "20260906T080000000000Z", "usage_scope": "batch", "llm_batch_size": 1,
        "usage_prompt_cache_hit_tokens": 0, "usage_prompt_cache_miss_tokens": 1_000_000,
        "usage_completion_tokens": 0, "llm_kwargs": {"thinking": {"type": "disabled"}},
    }
    trace = tmp_path / "events.jsonl"
    trace.write_text("\n".join(json.dumps(e) for e in (
        direct, {**direct, "phase": "grading"}, {**direct, "operator": "sem_map", "phase": "insertion"},
    )) + "\n")
    state = controller._update_monitor(trace, tmp_path / "monitor.json")
    assert state["cost_cny"] == 3
    assert state["provider_responses"] == 2
    assert state["provider_contract_violations"] == []
    assert controller._update_monitor(trace, tmp_path / "monitor.json")["cost_cny"] == 3


def test_successful_continuation_enters_second_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = tmp_path / "old"
    previous = _completed_origin(old)
    root = tmp_path / "new"
    conditions = [controller.condition_matrix()[i] for i in (0, 5, 10)]
    contract = _contract(root, [c.to_dict() for c in conditions])
    contract["continuation"] = controller._capture_continuation(old, ("pass-1-original",), previous)
    controller._atomic_json(root / "control/experiment-contract.json", contract)
    controller._atomic_text(root / "control/experiment-state", "prepared")
    monkeypatch.setattr(controller, "_before_condition", lambda _paths: None)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *_args, **_kwargs: _SuccessfulProcess())
    observed = []

    def validate(_paths: controller.ExperimentPaths, condition: controller.ConditionSpec) -> dict[str, float]:
        observed.append(condition.run_id)
        return {"cost_cny": 0}

    monkeypatch.setattr(controller, "_validate_condition", validate)
    controller.run_experiment(root)
    assert observed == ["pass-1-packed-16", "pass-2-original"]
