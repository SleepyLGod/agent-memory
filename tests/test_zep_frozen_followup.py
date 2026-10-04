"""Offline boundary checks for the fixed experiment."""
from types import SimpleNamespace
from typing import Any, cast
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.zep_frozen_followup import evaluate_group, freeze, validate_native_outputs
from tools.zep_frozen_followup import synthetic_preflight, run
from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_filter_batch_prompting import execute_batch_prompted_sem_filter


def events() -> list[dict[str, Any]]:
    result = []
    for g in range(3):
        call = f"call-{g}"
        result.append({"operator": "sem_filter", "event_type": "operator_result", "operator_call_id": call,
                       "lowered_instruction": "{fact_later_added} contradicts {fact_earlier_added}"})
        for p in range(12):
            result.append({"operator": "sem_filter", "event_type": "pair_decision", "trace_id": f"{g}-{p}",
                           "operator_call_id": call, "event_id": str(g), "right_id": "new", "left_id": "old",
                           "direction": "right-to-left", "decision_source": "oracle", "left": "fact: identical",
                           "right": "fact: repeated", "pair_index": p, "decision": False})
    return result


def test_freeze_preserves_duplicate_occurrences_order_and_caps() -> None:
    groups = freeze(events())
    assert [len(g["rows"]) for g in groups] == [10, 10, 10]
    assert [r["occurrence"] for r in groups[0]["rows"]] == [f"0-{i}" for i in range(10)]
    changed = events()
    for e in changed:
        if "decision" in e:
            e["decision"] = True
    assert [[r["occurrence"] for r in g["rows"]] for g in freeze(changed)] == [
        [r["occurrence"] for r in g["rows"]] for g in groups]


@pytest.mark.parametrize("damage", ["identity", "instruction", "direction"])
def test_freeze_fails_closed(damage: str) -> None:
    data = events()
    if damage == "identity":
        data[2]["trace_id"] = data[1]["trace_id"]
    elif damage == "instruction":
        data[0]["lowered_instruction"] = "{missing}"
    else:
        data[1]["direction"] = "left-to-right"
    with pytest.raises(ValueError):
        freeze(data)


def test_empty_group_makes_no_call() -> None:
    assert evaluate_group(None, {"rows": []}) == []


@pytest.mark.parametrize("output", ["Maybe", "", "True and False", "{}"])
def test_native_defaults_are_rejected(tmp_path: Path, output: str) -> None:
    (tmp_path / "raw.json").write_text(json.dumps({"output": output}))
    with pytest.raises(ValueError, match="unambiguous"):
        validate_native_outputs([{"event_type": "llm_call", "raw_output_path": "raw.json"}], tmp_path, [False])


def test_existing_batch_path_preserves_occurrences(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus

    class Model:
        max_tokens = 512
        cache = None

        def __call__(self, messages: Any, **kwargs: Any) -> Any:
            assert len(messages) == 1
            return SimpleNamespace(outputs=[json.dumps({"decisions": [
                {"row_id": f"row_{i}", "keep": i != 1} for i in range(3)]})])

    monkeypatch.setattr(lotus.settings, "lm", Model())
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    frame = pd.DataFrame({"fact_earlier_added": ["a", "b", "a"],
                          "fact_later_added": ["c"] * 3, "occurrence": ["0", "1", "2"]})
    result = execute_batch_prompted_sem_filter(frame, instruction="{fact_later_added} contradicts {fact_earlier_added}",
        context=cast(LotusExecutionContext, SimpleNamespace(config=LotusExecutionConfig(structured_parse_retries=0))),
        prompt_batching=PromptBatching(max_tasks=10))
    assert result.decisions == (True, False, True)
    assert result.frame["occurrence"].tolist() == ["0", "2"]


def test_synthetic_freeze_and_label_isolation(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures/zep_contradiction_contrasts.json"
    output = tmp_path / "run"
    synthetic_preflight(fixture, output)
    groups = json.loads((output / "frozen.json").read_text())
    labels = json.loads((output / "assessment.json").read_text())
    assert len(groups) == 6 and len(labels) == 24
    assert sum(row["expected"] for row in labels) == 6
    class Adapter:
        def execute(self, query: Any, views: Any) -> Any:
            frame = views["frozen"]
            assert list(frame.columns) == ["occurrence", "fact_earlier_added", "fact_later_added"]
            assert "expected" not in query.params["instruction"]
            assert frame["fact_later_added"].nunique() == 1
            return frame.iloc[[0, 2]]
    for group in groups:
        assert evaluate_group(Adapter(), group) == [True, False, True, False]
        assert all(set(row) == {"occurrence", "fact_earlier_added", "fact_later_added"} for row in group["rows"])
    with pytest.raises(FileExistsError):
        synthetic_preflight(fixture, output)
    (output / "frozen.json").write_text("[]")
    with pytest.raises(ValueError, match="evidence changed"):
        run(output)


def test_synthetic_failure_stops_without_answer_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.zep_frozen_followup as probe
    fixture = Path(__file__).parent / "fixtures/zep_contradiction_contrasts.json"
    output = tmp_path / "run"
    synthetic_preflight(fixture, output)
    calls = []
    def fail(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(probe, "LotusAdapter", fail)
    monkeypatch.setattr(probe, "replay_answers", lambda _: pytest.fail("unexpected answer replay"))
    with pytest.raises(RuntimeError, match="unavailable"):
        run(output)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="refusing replay"):
        run(output)


@pytest.mark.parametrize("artifact", ["frozen", "answer-inputs"])
@pytest.mark.parametrize("change", ["none", "content", "missing_digest"])
def test_historical_frozen_artifacts_checked_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str, change: str,
) -> None:
    import tools.zep_frozen_followup as probe
    import agent_memory.evaluation.provenance as provenance

    monkeypatch.setattr(provenance, "build_source_evidence", lambda _: {})
    old = tmp_path / "old"
    trace_events = events()
    for event in list(trace_events):
        if event["event_type"] != "pair_decision":
            continue
        prompt = f"prompts/{event['trace_id']}.json"
        probe.save(old / "fused" / prompt, [{"content": "identical repeated"}])
        trace_events.append({"event_type": "llm_call", "operator": "sem_filter",
                             "operator_call_id": event["operator_call_id"],
                             "llm_item_index": event["pair_index"], "prompt_path": prompt})
    trace = old / "fused/trace/events.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(event) + "\n" for event in trace_events))
    for mode in ("unfused", "fused"):
        for relative, row in (
            ("input/questions.jsonl", {"question_id": "q1"}),
            ("cases/conv-26-2aac22fc/retrieval.jsonl", {"question_id": "q1", "context": "original"}),
        ):
            path = old / mode / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row) + "\n")
    output = tmp_path / "run"
    probe.preflight(old, output)
    manifest = json.loads((output / "manifest.json").read_text())
    for name in ("frozen", "answer-inputs"):
        assert manifest[f"{name}_sha256"] == probe.digest(output / f"{name}.json")
    if change == "content":
        with (output / f"{artifact}.json").open("a") as stream:
            stream.write("\n")
    elif change == "missing_digest":
        manifest.pop(f"{artifact}_sha256")
        probe.save(output / "manifest.json", manifest)
    assert all(probe.digest(Path(path)) == value for path, value in manifest["input_hashes"].items())

    def stop_before_provider(_: Path) -> None:
        raise RuntimeError("offline execution reached")

    monkeypatch.setattr(probe, "replay_answers", stop_before_provider)
    monkeypatch.setattr(probe, "LotusAdapter", lambda **_: pytest.fail("unexpected provider setup"))
    if change == "none":
        with pytest.raises(RuntimeError, match="offline execution reached"):
            run(output)
    else:
        with pytest.raises(ValueError, match=f"evidence changed or digest missing: {artifact}"):
            run(output)
        assert json.loads((output / "status.json").read_text()) == {"stage": "preflight_passed"}
