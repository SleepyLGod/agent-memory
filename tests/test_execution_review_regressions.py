"""Offline regressions for execution setup and experiment failure handling."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy
import sys
from threading import Event, Lock
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_filter_batch_prompting import execute_batch_prompted_predicate


def test_concurrent_configuration_installs_one_shared_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus

    context = LotusExecutionContext("deepseek/deepseek-flash", LotusExecutionConfig(
        physical_fusion="zep-representative", parallel_fact_extraction=True, lm_enable_cache=False,
    ))
    first_started, release_first, second_started, duplicate_build = (Event() for _ in range(4))
    counter_lock = Lock()
    models: list[object] = []

    def new_lm(self: LotusExecutionContext) -> object:
        model = object()
        with counter_lock:
            models.append(model)
            first = len(models) == 1
        if first:
            first_started.set()
            assert release_first.wait(5)
        else:
            duplicate_build.set()
        return model

    def second_configure() -> None:
        second_started.set()
        context.configure()

    monkeypatch.setattr(LotusExecutionContext, "new_lm", new_lm)
    monkeypatch.setattr(lotus.settings, "lm", None)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(context.configure)
        assert first_started.wait(5)
        second = pool.submit(second_configure)
        assert second_started.wait(5)
        duplicate_build.wait(0.2)
        release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert len(models) == 1
    assert lotus.settings.lm is context._scoped_lm
    child = context.fork()
    assert child._scoped_lm is lotus.settings.lm
    assert child._lm is not context._lm


def test_configuration_failure_does_not_publish_partial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus

    context = LotusExecutionContext("deepseek/deepseek-flash", LotusExecutionConfig(
        physical_fusion="zep-representative", parallel_fact_extraction=True, lm_enable_cache=False,
    ))
    monkeypatch.setattr(LotusExecutionContext, "new_lm", lambda self: object())
    attempts: list[dict[str, Any]] = []

    def configure(**kwargs: Any) -> None:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("configuration failed")

    monkeypatch.setattr(lotus.settings, "configure", configure)
    with pytest.raises(RuntimeError, match="configuration failed"):
        context.configure()
    assert context._lm is None and context._scoped_lm is None
    assert not context._configured
    context.configure()
    assert context._configured and context._scoped_lm is attempts[-1]["lm"]


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_groupby_only_batching_preserves_completion_metadata(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str,
) -> None:
    import lotus
    import lotus.models
    from litellm import ModelResponse
    from lotus.models import LM

    class FakeLM(LM):
        def _process_uncached_messages(self, uncached_data: Any, all_kwargs: Any,
                                       show_progress_bar: bool, progress_bar_desc: str) -> Any:
            return [ModelResponse(model="deepseek/deepseek-flash", choices=[{
                "index": 0, "message": {"role": "assistant", "content": json.dumps({
                    "decisions": [{"row_id": "row_0", "keep": True}],
                })}, "finish_reason": finish_reason,
            }], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                for _ in uncached_data]

    monkeypatch.setattr(lotus.models, "LM", FakeLM)
    monkeypatch.setattr(lotus.settings, "lm", None)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    batching = PromptBatching(max_tasks=16)
    context = LotusExecutionContext("deepseek/deepseek-flash", LotusExecutionConfig(
        groupby_prompt_batching={"sem_groupby:test": batching},
        structured_parse_retries=0, lm_enable_cache=False,
    ))
    context.configure()

    def execute() -> Any:
        return execute_batch_prompted_predicate(
            pd.DataFrame({"left": ["a"], "right": ["a"]}),
            instruction="Do {left} and {right} match?", prompt_batching=batching,
            structured_parse_retries=0, structured_max_tokens=128,
            progress_bar_desc="Offline grouping", operator="sem_groupby",
        )

    if finish_reason == "length":
        with pytest.raises(ValueError, match="incomplete structured response"):
            execute()
    else:
        assert execute().decisions == (True,)


@pytest.mark.parametrize("state", ["missing", "empty", "preflighted"])
def test_followup_cli_precondition_failure_preserves_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str,
) -> None:
    import tools.zep_frozen_followup as probe

    output = tmp_path / "run"
    if state != "missing":
        output.mkdir()
    status = output / "status.json"
    if state == "preflighted":
        status.write_text(json.dumps({"stage": "preflight_passed"}))
    monkeypatch.setattr(sys, "argv", [probe.__file__, "run", "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        runpy.run_path(str(probe.__file__), run_name="__main__")
    if state == "preflighted":
        assert json.loads(status.read_text()) == {"stage": "preflight_passed"}
    else:
        assert not status.exists()
        assert output.exists() == (state == "empty")


@pytest.mark.parametrize("stage", ["completed", "answer_replay", "contradictions", "failed"])
def test_followup_cli_rejected_replay_preserves_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str,
) -> None:
    import tools.zep_frozen_followup as probe

    status = tmp_path / "status.json"
    original = json.dumps({"stage": stage, "details": "original execution"}, indent=2)
    status.write_text(original)
    monkeypatch.setattr(sys, "argv", [probe.__file__, "run", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="refusing replay"):
        runpy.run_path(str(probe.__file__), run_name="__main__")
    assert status.read_text() == original


def test_followup_execution_failure_records_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import tools.zep_frozen_followup as probe

    probe.save(tmp_path / "status.json", {"stage": "preflight_passed"})
    probe.save(tmp_path / "frozen.json", [])
    probe.save(tmp_path / "answer-inputs.json", {})
    probe.save(tmp_path / "manifest.json", {
        "input_hashes": {}, "script_sha256": probe.digest(Path(probe.__file__)),
        "frozen_sha256": probe.digest(tmp_path / "frozen.json"),
        "answer-inputs_sha256": probe.digest(tmp_path / "answer-inputs.json"),
    })

    def fail_replay(output: Path) -> None:
        assert json.loads((output / "status.json").read_text())["stage"] == "answer_replay"
        raise RuntimeError("offline injected failure")

    monkeypatch.setattr(probe, "replay_answers", fail_replay)
    with pytest.raises(RuntimeError, match="offline injected failure"):
        probe.run(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text()) == {
        "stage": "failed", "error": "offline injected failure", "type": "RuntimeError",
    }
