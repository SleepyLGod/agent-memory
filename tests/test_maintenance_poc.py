"""Offline runner regressions. Synthetic responses are not benchmark results."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import agent_memory as am
from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.context import (
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.maintenance import object_schema
from agent_memory.adapters.lotus.provider_usage_lm import (
    ProviderCallUsage,
    provider_usage_tracing_lm_class,
)

from tools.evaluation import maintenance_poc as runner


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected remote access fails instead of invoking a real provider."""
    import socket

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Offline test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace only the LM call, retaining real formatting, parsing and runtime."""
    import lotus

    class FakeLM:
        max_ctx_len = 100_000
        max_tokens = 512

        def __init__(self) -> None:
            self.calls: list = []
            self.handler = lambda payload: {}

        def count_tokens(self, messages: Any) -> int:
            return len(json.dumps(messages)) // 4

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            self.calls.append(deepcopy(prompts))
            payload = json.JSONDecoder().raw_decode(
                prompts[0][-1]["content"].split("Context:\n", 1)[1]
            )[0]
            return SimpleNamespace(
                outputs=[json.dumps(self.handler(payload))], response_metadata=None
            )

    fake = FakeLM()
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    monkeypatch.setattr(lotus.settings, "lm", fake)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    return fake


def configuration(group: bool = False) -> dict:
    """Test fixture only, never used by the real runner to design S."""
    value = object_schema({"count": {"type": "integer"}})
    membership = None
    if group:
        value = object_schema(
            {
                "items": {
                    "type": "array",
                    "items": object_schema(
                        {
                            "record_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "topic": {"type": "string"},
                        }
                    ),
                }
            }
        )
        membership = {"objects": "items", "record_ids": "record_ids"}
    return {
        "rule": "group-state" if group else "agg-state",
        "state_schema": {"value": value, "membership": membership},
        "maintenance_instruction": "Test-only state",
        "rationale": "Offline fixture",
    }


def fixture_case(group: bool = False) -> tuple[dict, dict]:
    """Not a real dataset; test data must not enter the sample manifest."""
    key = "Description" if group else "text"
    return (
        {
            "id": "offline-fixture",
            "kind": "groupby" if group else "agg",
            "batch_ranges": [[0, 1], [1, 2]],
        },
        {
            "instruction": "Original task {unchanged}",
            "context": None,
            "records": [{key: "a"}, {key: "b"}],
        },
    )


def test_agent_choice_is_bound_and_state_is_recorded(
    model: Any, tmp_path: Path
) -> None:
    contexts = []

    def response(payload: dict) -> dict:
        contexts.append(payload)
        if "Available rules" in payload:
            assert payload["Development inputs"] == []
            return configuration()
        previous = payload["Previous state"]
        total = (previous["count"] if previous else 0) + len(payload["New inputs"])
        return {"result": {"answer": ""}, "state": {"count": total}}

    model.handler = response
    case, payload = fixture_case()
    progress = runner.Progress(tmp_path, case["id"], "agent", 2)
    result = runner.execute_case(
        case,
        payload,
        "agent",
        LotusAdapter(),
        tmp_path,
        data_dir=tmp_path,
        progress=progress,
    )
    assert result["status"] == "completed"
    assert result["execution_kinds"].count("semantic_stateful") == 1
    assert len(model.calls) == 3  # Plan once, then two state updates.
    assert contexts[0]["Original task"] == payload["instruction"]
    step = json.loads((tmp_path / "step-0002.json").read_text())
    assert step["changes"]["answer"] == {"inserted": [], "retracted": []}
    assert next(iter(step["state_nodes"].values()))[0]["_state"]["count"] == 2
    assert json.loads((tmp_path / "plan-0.json").read_text()) == configuration()
    assert runner.read_status(tmp_path)["completed_steps"] == 2
    assert runner.read_status(tmp_path)["status"] == "completed"


def test_group_state_reassignment_is_observed(model: Any, tmp_path: Path) -> None:
    def response(payload: dict) -> dict:
        if "Available rules" in payload:
            return configuration(group=True)
        old = payload["Previous state"]
        items = deepcopy(old["items"]) if old else []
        items.extend(
            {"record_ids": [r["record_id"]], "topic": r["values"]["Description"]}
            for r in payload["New inputs"]
        )
        # The fake changes its partition on batch two, to test capture of withdrawals.
        assignments = [
            {"object": i, "group": i if old else 0} for i in range(len(items))
        ]
        return {"state": {"items": items}, "assignments": assignments}

    model.handler = response
    case, payload = fixture_case(group=True)
    result = runner.execute_case(
        case, payload, "agent", LotusAdapter(), tmp_path, data_dir=tmp_path
    )
    assert result["status"] == "completed"
    step = json.loads((tmp_path / "step-0002.json").read_text())
    assert len(step["outputs"]["answer"]) == 2
    assert len(step["state_nodes"]) == 1
    assert len(model.calls) == 3


def test_invalid_output_is_reported_not_retried(model: Any, tmp_path: Path) -> None:
    model.handler = lambda payload: (
        configuration() if "Available rules" in payload else {"state": {"count": 1}}
    )
    case, payload = fixture_case()
    adapter = LotusAdapter(config=LotusExecutionConfig(structured_parse_retries=0))
    with pytest.raises(Exception):
        runner.execute_case(
            case, payload, "agent", adapter, tmp_path, data_dir=tmp_path
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["stage"] == "maintenance"
    assert summary["completed_steps"] == 0
    assert not list(tmp_path.glob("step-*.json"))
    assert len(model.calls) == 2


@pytest.mark.parametrize("mode", ["fallback", "composable"])
def test_controls_do_not_plan(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        am, "plan_maintenance", lambda *a, **k: pytest.fail("Control invoked planning")
    )
    rules = runner.rules_for(mode, [], runner.NoExecution(), None)
    assert rules.sem_agg_rule == f"agg-{mode}"


def test_claude_two_targets_bound_without_query_rewrite(
    model: Any, tmp_path: Path
) -> None:
    model.handler = lambda p: configuration(group=p["Operator"] == "sem_groupby")
    case = {"kind": "policy"}
    owner, targets, _ = runner.build(case, {}, "agent")
    original = owner.topics.expr
    rules = runner.rules_for("agent", targets, LotusAdapter(), tmp_path)
    memory = runner.make_engine(case, owner, targets, runner.NoExecution(), rules)
    assert owner.topics.expr is original
    assert len(model.calls) == 2
    assert (
        sum(
            n.execution_kind == "semantic_stateful"
            for n in runner.executor_of(memory).policy.nodes.values()
        )
        == 2
    )
    for index, target in enumerate(targets):
        context = json.JSONDecoder().raw_decode(
            model.calls[index][0][-1]["content"].split("Context:\n", 1)[1]
        )[0]
        assert context["Original task"] == target.expr.params["instruction"]
    # The baseline remains separate and is not rebound or rewritten.
    baseline, baseline_targets, _ = runner.build(case, {}, "baseline")
    assert baseline is am.ClaudeMemory
    assert baseline_targets == []


@pytest.mark.parametrize("group", [False, True])
def test_checkpoint_restores_state_and_only_consumes_next_records(
    model: Any, tmp_path: Path, group: bool,
) -> None:
    case, payload = fixture_case(group)
    key = "Description" if group else "text"
    payload["records"] = [{key: str(i)} for i in range(32)]
    case["batch_ranges"] = [[i, i + 1] for i in range(32)]

    def respond(p: dict) -> dict:
        old = p["Previous state"]
        if group:
            items = deepcopy(old["items"]) if old else []
            items.extend({"topic": "one", "record_ids": [r["record_id"]]}
                         for r in p["New inputs"])
            return {"state": {"items": items},
                    "assignments": [{"object": i, "group": 0} for i in range(len(items))]}
        count = (old["count"] if old else 0) + len(p["New inputs"])
        return {"state": {"count": count}, "result": {"answer": str(count)}}

    model.handler = respond
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    args = {"data_dir": tmp_path, "saved_plans": [configuration(group)]}
    paused = runner.execute_case(case, payload, "agent", LotusAdapter(), first,
                                 stop_after=16, **args)
    assert paused["status"] == "paused" and len(model.calls) == 16
    done = runner.execute_case(case, payload, "agent", LotusAdapter(), second,
                               resume_from=first / "checkpoint.json", **args)
    assert done["status"] == "completed" and len(model.calls) == 32
    assert not (second / "step-0016.json").exists()
    last = json.loads((second / "step-0032.json").read_text())
    state = next(iter(last["state_nodes"].values()))[0]["_state"]
    assert len(state["items"]) == 32 if group else state["count"] == 32


def test_resume_rejects_changed_inputs_before_model(model: Any, tmp_path: Path) -> None:
    case, payload = fixture_case()
    model.handler = lambda p: {"result": {"answer": ""}, "state": {"count": 1}}
    runner.execute_case(case, payload, "agent", LotusAdapter(), tmp_path,
                        data_dir=tmp_path, saved_plans=[configuration()], stop_after=1)
    before = len(model.calls)
    altered = deepcopy(payload)
    altered["records"][1]["text"] = "changed"
    out = tmp_path / "resume"
    out.mkdir()
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.execute_case(case, altered, "agent", LotusAdapter(), out,
                            data_dir=tmp_path, saved_plans=[configuration()],
                            resume_from=tmp_path / "checkpoint.json")
    assert len(model.calls) == before


@pytest.mark.parametrize("mode", ["composable", "fallback", "baseline"])
def test_fixed_rule_resume_preserves_mode_without_planning(
    mode: str, model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    from lotus.types import LMStats

    monkeypatch.setattr(am, "plan_maintenance", lambda *a, **k: pytest.fail("Called Agent"))
    case, payload = fixture_case()
    payload["instruction"] = "Summarize {text}"
    if mode == "baseline":
        case["kind"] = "policy"
        payload = {"records": [
            {"sample_id": "s", "event_id": str(i), "speaker": "user",
             "text": f"message {i}", "timestamp": "2024-01-01"} for i in range(2)
        ]}

    class LM:
        stats = LMStats()

        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            model.calls.extend(deepcopy(prompts))
            value = json.dumps({"rows": []}) if mode == "baseline" else "summary"
            return SimpleNamespace(outputs=[value for _ in prompts], response_metadata=None)

    monkeypatch.setattr(lotus.settings, "lm", LM())
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    paused = runner.execute_case(case, payload, mode, LotusAdapter(), first,
                                 data_dir=tmp_path, stop_after=1, maintenance_only=True)
    assert paused["status"] == "paused"
    before = len(model.calls)
    for change in ("data", "model", "mode"):
        altered = deepcopy(payload)
        options: dict[str, Any] = {}
        resume_mode = mode
        if change == "data":
            altered["records"][0]["text"] = "changed"
        elif change == "model":
            options["model"] = "different-model"
        else:
            resume_mode = "composable" if mode != "composable" else "fallback"
        rejected = tmp_path / change
        rejected.mkdir()
        with pytest.raises(ValueError, match="identity mismatch"):
            runner.execute_case(case, altered, resume_mode, LotusAdapter(), rejected,
                                data_dir=tmp_path, maintenance_only=True,
                                resume_from=first / "checkpoint.json", **options)
        assert len(model.calls) == before
    done = runner.execute_case(case, payload, mode, LotusAdapter(), second,
                               data_dir=tmp_path, maintenance_only=True,
                               resume_from=first / "checkpoint.json")
    assert done["status"] == "completed" and done["completed_steps"] == 2
    assert done["plan_fingerprint"] == paused["plan_fingerprint"]
    assert len(model.calls) == before + 1
    assert not (second / "step-0001.json").exists()
    assert not list(second.glob("plan-*.json"))
    checkpoint = json.loads((second / "checkpoint.json").read_text())
    assert checkpoint["identity_inputs"]["mode"] == mode
    assert checkpoint["identity_inputs"]["plans"] is None
    if mode == "baseline":
        metrics = json.loads((second / "ingest-metrics.json").read_text())
        assert [row["event_id"] for row in metrics] == ["0", "1"]
    else:
        step = json.loads((second / "step-0002.json").read_text())
        assert step["outputs"]["answer"] == [{"answer": "summary"}]


@pytest.mark.parametrize("recover", [False, True])
def test_state_validation_retries_identical_request_without_partial_commit(
    model: Any, tmp_path: Path, recover: bool,
) -> None:
    case, payload = fixture_case()
    def respond(p: dict) -> dict:
        result: dict[str, Any] = {"answer": ""}
        if not recover or len(model.calls) <= 3:
            result["evidence"] = None
        return {"result": result, "state": {"count": 1}}
    model.handler = respond
    adapter = LotusAdapter(config=LotusExecutionConfig(maintenance_parse_retries=3))
    def run() -> dict:
        return runner.execute_case(case, payload, "agent", adapter, tmp_path,
                                   data_dir=tmp_path, saved_plans=[configuration()], stop_after=1)
    if recover:
        assert run()["status"] == "paused"
    else:
        with pytest.raises(Exception, match="evidence"):
            run()
        assert not (tmp_path / "checkpoint.json").exists()
    assert len(model.calls) == 4
    assert all(call == model.calls[0] for call in model.calls)


def test_lme_excludes_annotations_without_changing_text() -> None:
    case, payload = fixture_case()
    case["id"] = "lme-real-format-fixture"
    payload["records"] = [{"messages": [{"content": "has_answer stays in text", "has_answer": True}]}]
    original = deepcopy(payload)
    _, _, frame = runner.build(case, payload, "agent")
    assert json.loads(frame.iloc[0]["messages"]) == [{"content": "has_answer stays in text"}]
    assert payload == original


def test_claude_16_then_resume_restores_both_state_nodes(
    model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    from lotus.types import LMStats
    from agent_memory.evaluation.claude_memory import locomo

    candidate = {"name": "work", "description": "work", "type": "user", "body": "engineer"}
    class Model:
        stats = LMStats()
        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            text = prompts[0][-1]["content"]
            if kwargs.get("progress_bar_desc") in ("group-state", "agg-state"):
                return model(prompts, **kwargs)
            model.calls.append(deepcopy(prompts))
            value = {"rows": [candidate]} if "Extract zero or more" in text else {"hook": "work"}
            return SimpleNamespace(outputs=[json.dumps(value) for _ in prompts], response_metadata=None)

    def respond(p: dict) -> dict:
        old = p["Previous state"]
        if "Previous memberships" in p:
            items = deepcopy(old["items"]) if old else []
            items.extend({"record_ids": [r["record_id"]], "topic": "work"}
                         for r in p["New inputs"])
            return {"state": {"items": items}, "assignments": [
                {"object": i, "group": 0} for i in range(len(items))]}
        return {"state": {"count": (old["count"] if old else 0) + len(p["New inputs"])},
                "result": candidate}

    model.handler = respond
    monkeypatch.setattr(lotus.settings, "lm", Model())
    monkeypatch.setattr(locomo, "run_questions", lambda *a, **k: pytest.fail("QA on partial input"))
    case = {"id": "offline-claude", "kind": "policy", "batch_ranges": [[i, i + 1] for i in range(32)]}
    payload = {"records": [{"sample_id": "s", "event_id": str(i), "speaker": "user",
                             "text": f"I am an engineer {i}", "timestamp": "2024-01-01"}
                            for i in range(32)]}
    plans = [configuration(True), configuration()]
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    one = runner.execute_case(case, payload, "agent", LotusAdapter(), first,
                              data_dir=tmp_path, saved_plans=plans, stop_after=16)
    assert one["status"] == "paused"
    two = runner.execute_case(case, payload, "agent", LotusAdapter(), second,
                              data_dir=tmp_path, saved_plans=plans, stop_after=32,
                              maintenance_only=True,
                              resume_from=first / "checkpoint.json")
    assert two["completed_steps"] == 32 and two["status"] == "completed"
    metrics = json.loads((second / "ingest-metrics.json").read_text())
    assert [m["event_id"] for m in metrics] == [str(i) for i in range(32)]
    assert not (second / "step-0016.json").exists()
    states = json.loads((second / "step-0017.json").read_text())["state_nodes"]
    assert len(states) == 2
    assert sorted(len(r[0]["_state"].get("items", [])) for r in states.values()) == [0, 17]


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_changes_retries_only_after_verifying_original_identity(
    model: Any, tmp_path: Path, legacy: bool,
) -> None:
    case, payload = fixture_case()
    model.handler = lambda p: {
        "result": {"answer": ""},
        "state": {"count": (p["Previous state"] or {}).get("count", 0) + 1},
    }
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = "tools/evaluation/maintenance_poc.py"
    old_execution = {"source": {name: "old", "src/operator.py": "same"},
                     "model": {"model": "test"}, "validation_retries": 0}
    runner.execute_case(case, payload, "agent", LotusAdapter(), first, data_dir=tmp_path,
                        saved_plans=[configuration()], stop_after=1, execution_identity=old_execution)
    checkpoint = json.loads((first / "checkpoint.json").read_text())
    if legacy:
        inputs = checkpoint.pop("identity_inputs")
        checkpoint["identity"] = runner.identity_hash(inputs)
        runner.write_json(first / "checkpoint.json", checkpoint)
        runner.write_json(first / "run-contract.json", {
            "source": {"files": old_execution["source"]}, "model": "test", "retries": 0})
    execution = deepcopy(old_execution)
    execution["source"][name] = "new"
    execution["validation_retries"] = 3
    result = runner.execute_case(
        case, payload, "agent", LotusAdapter(), second, data_dir=tmp_path,
        saved_plans=[configuration()], resume_from=first / "checkpoint.json",
        execution_identity=execution,
        resume_source_changes={name: {"before": "old", "after": "new"}},
    )
    assert result["completed_steps"] == 2
    assert len(model.calls) == 2
    assert json.loads((second / "resume-validation.json").read_text())["previous_retries"] == 0
    states = json.loads((second / "step-0002.json").read_text())["state_nodes"]
    assert next(iter(states.values()))[0]["_state"] == {"count": 2}


@pytest.mark.parametrize("changed", ["payload", "plans", "model", "operator", "unreviewed"])
def test_resume_does_not_relax_semantic_identity(changed: str, tmp_path: Path) -> None:
    name = "tools/evaluation/maintenance_poc.py"
    old = {"payload": "original", "plans": ["saved"], "model": "same",
           "execution": {"source": {name: "old", "src/operator.py": "same"},
                         "model": {}, "validation_retries": 0}}
    checkpoint = {"identity_inputs": old, "identity": runner.identity_hash(runner.stable_identity(old))}
    current = deepcopy(old)
    changes = {}
    if changed == "operator":
        current["execution"]["source"]["src/operator.py"] = "new"
        changes = {"src/operator.py": {"before": "same", "after": "new"}}
    elif changed == "unreviewed":
        current["execution"]["source"][name] = "new"
    else:
        current[changed] = "different"
    with pytest.raises(ValueError):
        runner.validate_resume(checkpoint, current, tmp_path, changes)


def test_shared_budget_carries_attempts_across_conditions(tmp_path: Path) -> None:
    class LM:
        last_call_usage = ProviderCallUsage()

        def count_tokens(self, messages: Any) -> int:
            return 1

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            self.last_call_usage = ProviderCallUsage(completion_tokens=[7])
            return SimpleNamespace(outputs=["{}"], response_metadata=None)

    ledger = tmp_path / "round.json"
    for name in ("luna", "ds"):
        directory = tmp_path / name / "calls"
        directory.mkdir(parents=True)
        budget = runner.BudgetLM(LM(), directory, 2, 100, 10, ledger=ledger)
        budget([[{"content": "same"}]])
    saved = json.loads(ledger.read_text())
    assert saved["requests"] == 2 and saved["completion_tokens"] == 14
    with pytest.raises(runner.BudgetExceeded):
        budget([[{"content": "same"}]])
    assert saved["unknown_usage_requests"] == 0


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 3}, {"completion_tokens": 0}, {"completion_tokens": 7}])
@pytest.mark.parametrize("failure", [None, "parse", "transport"])
def test_budget_uses_provider_usage_even_when_parsing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    usage: dict[str, int] | None, failure: str | None,
) -> None:
    import lotus
    from litellm import ModelResponse
    from lotus.cache import InMemoryCache
    from lotus.models import LM

    error = ValueError("output parsing failed") if failure == "parse" else TimeoutError("no response")
    response: Any = ModelResponse(choices=[{"message": {"content": "answer"}, "finish_reason": "stop"}])
    response.usage = usage

    def complete(*args: Any, **kwargs: Any) -> list[Any]:
        if failure == "transport":
            raise error
        return [response]

    def parse(self: Any, value: Any) -> str:
        if failure == "parse":
            raise error
        return "answer"

    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    monkeypatch.setattr("lotus.models.lm.batch_completion", complete)
    monkeypatch.setattr(LM, "_get_top_choice", parse)
    # LOTUS's cumulative counters are deliberately irrelevant to the budget.
    monkeypatch.setattr(LM, "_update_stats", lambda *args, **kwargs: None)
    lm = provider_usage_tracing_lm_class(LM)(
        model="gpt-4o-mini", cache=InMemoryCache(max_size=8), trace_dir=None,
    )
    monkeypatch.setattr(lm, "count_tokens", lambda messages: 1)
    calls = tmp_path / "calls"
    calls.mkdir()
    ledger = tmp_path / "round.json"
    budget = runner.BudgetLM(lm, calls, 3, 100, 10, ledger=ledger)
    if failure:
        with pytest.raises(type(error)) as caught:
            budget([[{"role": "user", "content": "one"}]])
        assert caught.value is error
    else:
        assert budget([[{"role": "user", "content": "one"}]]).outputs == ["answer"]

    tokens = usage.get("completion_tokens") if usage is not None and failure != "transport" else None
    assert budget.completion_tokens == (tokens or 0)
    assert budget.unknown_usage_requests == int(tokens is None)
    restored = runner.BudgetLM(lm, calls, 3, 100, 10, ledger=ledger)
    assert restored.accounting() == budget.accounting()
    # A later response with no usage must not reuse the first response's usage.
    failure = None
    response.usage = None
    restored([[{"role": "user", "content": "two"}]])
    assert restored.completion_tokens == (tokens or 0)
    assert restored.unknown_usage_requests == int(tokens is None) + 1
    failure = "transport"
    with pytest.raises(type(error)) as caught:
        restored([[{"role": "user", "content": "three"}]])
    assert caught.value is error
    assert restored.completion_tokens == (tokens or 0)
    assert restored.unknown_usage_requests == int(tokens is None) + 2


def test_budget_does_not_charge_cached_provider_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    from litellm import ModelResponse
    from lotus.cache import InMemoryCache
    from lotus.models import LM

    sent: list[Any] = []

    def complete(model: str, batch: list[Any], **kwargs: Any) -> list[Any]:
        sent.extend(batch)
        return [ModelResponse(
            choices=[{"message": {"content": "answer"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        ) for _ in batch]

    monkeypatch.setattr(lotus.settings, "enable_cache", True)
    monkeypatch.setattr("lotus.models.lm.batch_completion", complete)
    monkeypatch.setattr("lotus.models.lm.completion_cost", lambda **kwargs: 0.0)
    lm = provider_usage_tracing_lm_class(LM)(model="gpt-4o-mini", cache=InMemoryCache(max_size=8))
    monkeypatch.setattr(lm, "count_tokens", lambda messages: 1)
    calls = tmp_path / "calls"
    calls.mkdir()
    budget = runner.BudgetLM(lm, calls, 6, 100, 10)
    first = [{"role": "user", "content": "one"}]
    second = [{"role": "user", "content": "two"}]
    budget([first])
    budget([first, second])
    budget([second])
    assert len(sent) == 2
    assert budget.requests == 4
    assert budget.completion_tokens == 14
    assert budget.unknown_usage_requests == 0
    assert lm.last_call_usage == ProviderCallUsage(cache_hits=1)
    error = ValueError("parse failed after provider returned")

    def fail_parse(self: Any, response: Any) -> str:
        raise error

    monkeypatch.setattr(LM, "_get_top_choice", fail_parse)
    with pytest.raises(ValueError) as caught:
        budget([second, [{"role": "user", "content": "three"}]])
    assert caught.value is error
    assert budget.completion_tokens == 21
    assert budget.unknown_usage_requests == 0
    assert lm.last_call_usage == ProviderCallUsage(completion_tokens=[7], cache_hits=1)


def test_shared_budget_retains_unknown_usage_after_transport_error(tmp_path: Path) -> None:
    class LM:
        def count_tokens(self, messages: Any) -> int:
            return 1

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            raise TimeoutError("no completed response")

    directory = tmp_path / "calls"
    directory.mkdir()
    ledger = tmp_path / "round.json"
    budget = runner.BudgetLM(LM(), directory, 2, 100, 10, ledger=ledger)
    with pytest.raises(TimeoutError):
        budget([[{"content": "same"}]])
    restored = runner.BudgetLM(LM(), directory, 2, 100, 10, ledger=ledger)
    assert restored.requests == 1 and restored.unknown_usage_requests == 1


def test_budget_restores_failed_attempts_not_just_committed_steps(tmp_path: Path) -> None:
    class LM:
        def count_tokens(self, messages: Any) -> int:
            return 5

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            raise TimeoutError("test timeout")

    calls = tmp_path / "calls"
    calls.mkdir()
    budget = runner.BudgetLM(LM(), calls, 1, 100, 10)
    with pytest.raises(TimeoutError):
        budget([[{"content": "original"}]])
    recorded = json.loads((tmp_path / "budget-live.json").read_text())
    next_budget = runner.BudgetLM(LM(), calls, 1, 100, 10, initial=recorded)
    with pytest.raises(runner.BudgetExceeded):
        next_budget([[{"content": "original"}]])
    assert next_budget.requests == 1


def test_pacing_serializes_multiple_prompts_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    clock = [100.0]
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    class LM:
        def count_tokens(self, messages: Any) -> int:
            return 1

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            calls.append((deepcopy(prompts), clock[0]))
            return SimpleNamespace(outputs=[prompts[0][0]["content"]],
                                   logprobs=None, response_metadata=[{"finish_reason": "stop"}])

    directory = tmp_path / "calls"
    directory.mkdir()
    budget = runner.BudgetLM(LM(), directory, 2, 100, 10, min_interval=10)
    prompts = [[{"role": "user", "content": value}] for value in ("one", "two")]
    result = budget(prompts)
    assert [item[0][0] for item in calls] == prompts
    assert calls[1][1] - calls[0][1] == 10
    assert result.outputs == ["one", "two"]
    assert result.response_metadata == [{"finish_reason": "stop"}] * 2
    assert budget.requests == 2
    assert len(list(directory.glob("*.request.json"))) == 2


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_pacing_uses_dispatch_time_after_slow_artifact_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore: bool, failure: bool,
) -> None:
    clock = [100.0]
    dispatches: list[float] = []
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    write_json = runner.write_json

    def slow_write(path: Path, value: Any) -> None:
        if path.name.endswith(".request.json") and not dispatches:
            clock[0] += 7
        write_json(path, value)

    monkeypatch.setattr(runner, "write_json", slow_write)

    class LM:
        last_call_usage = ProviderCallUsage()

        def count_tokens(self, messages: Any) -> int:
            return 1

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            dispatches.append(clock[0])
            if failure and len(dispatches) == 1:
                raise TimeoutError("test timeout")
            self.last_call_usage = ProviderCallUsage(completion_tokens=[2])
            return SimpleNamespace(outputs=["ok"], response_metadata=None)

    ledger = tmp_path / "budget.json"
    directory = tmp_path / "first" / "calls"
    directory.mkdir(parents=True)
    budget = runner.BudgetLM(LM(), directory, 2, 100, 10, min_interval=10, ledger=ledger)
    if failure:
        with pytest.raises(TimeoutError, match="test timeout"):
            budget([[{"content": "first"}]])
    else:
        budget([[{"content": "first"}]])
    if restore:
        directory = tmp_path / "second" / "calls"
        directory.mkdir(parents=True)
        budget = runner.BudgetLM(LM(), directory, 2, 100, 10, min_interval=10, ledger=ledger)
    budget([[{"content": "second"}]])
    assert dispatches == [107.0, 117.0]
    saved = json.loads(ledger.read_text())
    assert saved["requests"] == 2
    assert saved["completion_tokens"] == (2 if failure else 4)
    assert saved["unknown_usage_requests"] == int(failure)


def test_failed_next_step_keeps_last_checkpoint(model: Any, tmp_path: Path) -> None:
    case, payload = fixture_case()
    def respond(p: dict) -> dict:
        return {"result": {"answer": ""}, "state": {"count": 1 if len(model.calls) == 1 else "invalid"}}
    model.handler = respond
    with pytest.raises(Exception):
        runner.execute_case(case, payload, "agent", LotusAdapter(), tmp_path,
                            data_dir=tmp_path, saved_plans=[configuration()])
    assert json.loads((tmp_path / "checkpoint.json").read_text())["completed_steps"] == 1
    assert not (tmp_path / "checkpoint-0002.pkl").exists()


def test_group_state_reference_retry_does_not_change_prompt(model: Any, tmp_path: Path) -> None:
    case, payload = fixture_case(True)
    def respond(p: dict) -> dict:
        return {"state": {"items": [{"topic": "test", "record_ids": [p["New inputs"][0]["record_id"]]}]},
                "assignments": [{"object": 1 if len(model.calls) == 1 else 0, "group": 0}]}
    model.handler = respond
    runner.execute_case(case, payload, "agent",
                        LotusAdapter(config=LotusExecutionConfig(maintenance_parse_retries=3)), tmp_path,
                        data_dir=tmp_path, saved_plans=[configuration(True)], stop_after=1)
    assert len(model.calls) == 2 and model.calls[0] == model.calls[1]


@pytest.mark.parametrize("limit", ["requests", "tokens"])
def test_budget_stops_before_provider(tmp_path: Path, limit: str) -> None:
    called = []

    class LM:
        def count_tokens(self, prompt: Any) -> int:
            return 10

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            called.extend(prompts)
            return SimpleNamespace(outputs=["ok"] * len(prompts))

    lm = runner.BudgetLM(
        LM(),
        tmp_path,
        1 if limit == "requests" else 3,
        100 if limit == "requests" else 15,
        5,
    )
    lm([[{"role": "user", "content": "one"}]])
    with pytest.raises(runner.BudgetExceeded):
        lm([[{"role": "user", "content": "two"}]])
    assert len(called) == 1
    assert lm.requests == 1


def test_dataset_formatting_keeps_original_instruction_and_content() -> None:
    payload = {
        "instruction": "Question",
        "context": {"question_date": "original date"},
        "records": [{"messages": [{"role": "user", "content": "original"}]}],
    }
    _, targets, frame = runner.build({"kind": "agg"}, payload, "agent")
    assert targets[0].expr.params["instruction"] == "Question"
    assert json.loads(frame.iloc[0]["messages"]) == payload["records"][0]["messages"]
    assert frame.iloc[0]["question_date"] == "original date"
    payload["context"] = {"messages": "cannot overwrite"}
    with pytest.raises(ValueError, match="overwrite"):
        runner.build({"kind": "agg"}, payload, "agent")


def test_execution_requires_explicit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_case.py", "qmsum-general", "--run"])
    monkeypatch.setattr(
        LotusExecutionContext,
        "configure",
        lambda self: pytest.fail("Unapproved run configured LM"),
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "case_id",
    [
        "qmsum-general",
        "qmsum-specific",
        "lme-gpt4_2655b836",
        "lme-6a1eabeb",
        "ag-news-classification",
        "ag-news-clustering",
        "claude-locomo-s0-32",
    ],
)
@pytest.mark.skipif(
    not os.environ.get("AGENT_MEMORY_POC_DATA_DIR"),
    reason="Original datasets are external; set AGENT_MEMORY_POC_DATA_DIR for data preflight",
)
def test_actual_cases_compile_without_models(
    case_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "maintenance_poc.py",
            case_id,
            "--data-dir",
            os.environ["AGENT_MEMORY_POC_DATA_DIR"],
        ],
    )
    monkeypatch.setattr(
        LotusExecutionContext,
        "configure",
        lambda self: pytest.fail("Dry run configured a model"),
    )
    runner.main()


def test_record_step_captures_actual_retraction(tmp_path: Path) -> None:
    frame = pd.DataFrame({"value": ["new"]})
    node = SimpleNamespace(execution_kind="semantic_stateful")
    executor = SimpleNamespace(
        policy=SimpleNamespace(nodes={"n": node}),
        snapshot_state=lambda: {
            "state": {"topics": frame, "catalog": frame},
            "node_state": {"n": pd.DataFrame({"_state": [{"count": 1}]})},
        },
    )
    engine = SimpleNamespace(_runtime=SimpleNamespace(_engine=executor))
    old = pd.DataFrame({"value": ["old"]})
    runner.record_step(engine, tmp_path, 1, {"topics": old, "catalog": old}, 0)
    step = json.loads((tmp_path / "step-0001.json").read_text())
    assert step["changes"]["topics"] == {
        "inserted": [{"value": "new"}],
        "retracted": [{"value": "old"}],
    }


def test_group_evaluation_ignores_arbitrary_group_names(tmp_path: Path) -> None:
    from agent_memory.policy.grouping import GROUP_ID

    gold = tmp_path / "gold.json"
    runner.write_json(gold, [{"id": "a", "Class": "X"}, {"id": "b", "Class": "Y"}])
    output = pd.DataFrame({"id": ["b", "a"], GROUP_ID: ["g99", "g12"]})
    assert runner.evaluate_grouping(output, gold)["adjusted_rand_index"] == 1
    with pytest.raises(ValueError, match="different records"):
        runner.evaluate_grouping(output.iloc[:1], gold)


def test_status_does_not_import_model_stack(tmp_path: Path) -> None:
    runner.Progress(tmp_path, "fixture", "agent", 32)
    code = (
        "import sys; from pathlib import Path; "
        "from tools.evaluation.maintenance_poc import show_status; "
        f"show_status(Path({str(tmp_path)!r})); "
        "assert not any(x in sys.modules for x in ('lotus','litellm','pandas','agent_memory'))"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_dead_process_is_not_reported_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.Progress(tmp_path, "fixture", "agent", 32)

    def dead(pid: int, signal: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", dead)
    assert (
        runner.read_status(tmp_path)["status"] == "process-exited-without-final-status"
    )


def test_deepseek_settings_are_unchanged() -> None:
    name, options, contract = runner.model_settings("deepseek", None, None, 8192)
    assert name == runner.MODEL
    assert options == {
        "max_tokens": 8192,
        "temperature": 0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert contract["known_omissions"] == []
    assert contract["output_tokens_per_request"] == 8192


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "http://127.0.0.1.evil.test/v1",
        "https://127.0.0.1/v1",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/v1?key=secret",
        "http://127.0.0.1:0/v1",
        "http://127.0.0.1:bad/v1",
        "http://127.0.0.1/other",
    ],
)
def test_gateway_rejects_nonlocal_or_credentialed_url(url: str) -> None:
    with pytest.raises(ValueError) as error:
        runner.model_settings("chatgpt-plan", "gpt-6-astra", url, 8192)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("name", [None, "", "openai/gpt-6-astra", " a "])
def test_gateway_requires_explicit_model(name: str | None) -> None:
    with pytest.raises(ValueError, match="model"):
        runner.model_settings("chatgpt-plan", name, None, 8192)


def prepare_cli_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    """Create synthetic files for CLI wiring tests, not experiment evidence."""
    case, payload = fixture_case()
    case.update(
        status="READY_DATA_ONLY",
        inputs="input.json",
        gold="gold.json",
        source="raw.json",
    )
    (tmp_path / "data").mkdir()
    runner.write_json(tmp_path / "data/raw.json", payload)
    runner.write_json(tmp_path / "input.json", payload)
    runner.write_json(tmp_path / "gold.json", {})
    runner.write_json(
        tmp_path / "manifest.json",
        {
            "cases": [case],
            "sources": {
                "raw.json": {"sha256": runner.digest(tmp_path / "data/raw.json")}
            },
        },
    )
    plan = tmp_path / "saved.json"
    runner.write_json(plan, configuration())
    return case, payload, plan


def test_saved_plan_preflight_never_reads_credentials_or_configures_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case, payload, plan = prepare_cli_fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "poc",
            case["id"],
            "--data-dir",
            str(tmp_path),
            "--saved-plan",
            str(plan),
            "--backend",
            "chatgpt-plan",
            "--model",
            "gpt-6-astra",
        ],
    )
    original_get = os.environ.get

    def get(key: str, default: Any = None) -> Any:
        assert key not in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}
        return original_get(key, default)

    monkeypatch.setattr(os.environ, "get", get)
    monkeypatch.setattr(
        LotusExecutionContext, "configure", lambda self: pytest.fail("Configured model")
    )
    monkeypatch.setattr(
        am, "plan_maintenance", lambda *a, **k: pytest.fail("Called Agent")
    )
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == 0
    assert result["compile_checks"][0]["control"] == "saved-plan"
    assert result["original_instructions"] == [payload["instruction"]]
    assert result["model_settings"]["output_tokens_per_request"] is None


@pytest.mark.parametrize("mode", ["composable", "fallback", "baseline"])
def test_fixed_rule_resume_cli_accepts_no_agent_plan(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case, _, _ = prepare_cli_fixture(tmp_path)
    if mode == "baseline":
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["cases"][0]["kind"] = "policy"
        runner.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(sys, "argv", [
        "poc", case["id"], "--data-dir", str(tmp_path), "--mode", mode,
        "--resume-from", str(tmp_path / "checkpoint.json"),
    ])
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: pytest.fail("Configured model"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


def test_agent_resume_still_requires_saved_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, payload = fixture_case()
    with pytest.raises(ValueError, match="saved plans"):
        runner.execute_case(case, payload, "agent", runner.NoExecution(), tmp_path,
                            data_dir=tmp_path, resume_from=tmp_path / "checkpoint.json")
    monkeypatch.setattr(sys, "argv", [
        "poc", case["id"], "--mode", "agent",
        "--resume-from", str(tmp_path / "checkpoint.json"),
    ])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "mode,count", [("fallback", 1), ("composable", 1), ("agent", 2), ("baseline", 0)]
)
def test_saved_plan_rejects_ambiguous_or_fixed_rule_targets(
    mode: str, count: int
) -> None:
    case, payload = fixture_case()
    _, targets, _ = runner.build(case, payload, "agent")
    with pytest.raises(ValueError, match="one plan per target"):
        runner.rules_for(
            mode,
            targets * count,
            runner.NoExecution(),
            None,
            saved_plans=[configuration()],
        )


@pytest.mark.parametrize("policy", [False, True])
def test_plan_only_then_bind_in_separate_execution(
    policy: bool, model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, payload = fixture_case()
    if policy:
        case["kind"] = "policy"
    _, targets, _ = runner.build(case, payload, "agent")
    configurations = [
        configuration(group=target.expr.op == "sem_groupby") for target in targets
    ]
    responses = iter(configurations)
    model.handler = lambda context: next(responses)
    summary = runner.execute_case(
        case,
        payload,
        "agent",
        LotusAdapter(),
        tmp_path,
        data_dir=tmp_path,
        plan_only=True,
    )
    assert summary["status"] == "completed"
    assert summary["stage"] == "planned"
    assert summary["completed_steps"] == 0
    assert len(model.calls) == len(targets) == (2 if policy else 1)
    assert not list(tmp_path.glob("step-*.json"))
    saved = [
        json.loads((tmp_path / f"plan-{i}.json").read_text())
        for i in range(len(targets))
    ]
    assert saved == configurations
    monkeypatch.setattr(
        am, "plan_maintenance", lambda *a, **k: pytest.fail("Agent called")
    )
    rules = runner.rules_for(
        "agent", targets, runner.NoExecution(), None, saved_plans=saved
    )
    owner, rebound, _ = runner.build(case, payload, "agent")
    runner.make_engine(case, owner, rebound, runner.NoExecution(), rules)
    assert len(model.calls) == len(targets)


@pytest.mark.parametrize(
    "mode,saved", [("fallback", None), ("agent", [configuration()])]
)
def test_plan_only_rejects_conflicting_modes_before_model(
    mode: str, saved: list[dict] | None, model: Any, tmp_path: Path
) -> None:
    case, payload = fixture_case()
    with pytest.raises(ValueError, match="--plan-only requires"):
        runner.execute_case(
            case,
            payload,
            mode,
            LotusAdapter(),
            tmp_path,
            data_dir=tmp_path,
            plan_only=True,
            saved_plans=saved,
        )
    assert model.calls == []


def test_claude_saved_plans_reject_swapped_operator_types(tmp_path: Path) -> None:
    case, payload = fixture_case()
    case["kind"] = "policy"
    _, targets, _ = runner.build(case, payload, "agent")
    with pytest.raises(ValueError):
        runner.rules_for(
            "agent",
            targets,
            runner.NoExecution(),
            tmp_path,
            saved_plans=[configuration(), configuration(group=True)],
        )
    assert not list(tmp_path.glob("plan-*.json"))


@pytest.mark.parametrize("target_index", [0, 1])
def test_claude_target_can_plan_independently(
    target_index: int, model: Any, tmp_path: Path
) -> None:
    case, payload = fixture_case()
    case["kind"] = "policy"
    _, targets, _ = runner.build(case, payload, "agent")
    model.handler = lambda context: configuration(group=target_index == 0)
    summary = runner.execute_case(
        case,
        payload,
        "agent",
        LotusAdapter(),
        tmp_path,
        data_dir=tmp_path,
        plan_only=True,
        target_index=target_index,
    )
    assert summary["status"] == "completed"
    assert summary["stage"] == "planned"
    assert summary["plan_target"] == target_index
    assert len(model.calls) == 1
    content = model.calls[0][0][-1]["content"]
    context = json.JSONDecoder().raw_decode(content.split("Context:\n", 1)[1])[0]
    assert context["Original task"] == targets[target_index].expr.params["instruction"]
    assert context["Development inputs"] == []
    assert not list(tmp_path.glob("step-*.json"))


@pytest.mark.parametrize("only,index", [(False, 0), (True, -1), (True, 2)])
def test_invalid_planning_target_fails_before_model(
    only: bool, index: int, model: Any, tmp_path: Path
) -> None:
    case, payload = fixture_case()
    with pytest.raises(ValueError, match="--plan-target requires"):
        runner.execute_case(
            case,
            payload,
            "agent",
            LotusAdapter(),
            tmp_path,
            data_dir=tmp_path,
            plan_only=only,
            target_index=index,
        )
    assert model.calls == []


def test_thinking_is_explicit_and_backend_specific() -> None:
    _, options, contract = runner.model_settings(
        "deepseek", None, None, 8192, thinking="enabled"
    )
    assert options["extra_body"]["thinking"] == {"type": "enabled"}
    assert contract["thinking"] == "enabled"
    with pytest.raises(ValueError, match="only supported"):
        runner.model_settings(
            "chatgpt-plan", "gpt-6-luna", None, 8192, thinking="enabled"
        )


@pytest.mark.parametrize(
    "model_name", ["gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"]
)
@pytest.mark.parametrize(
    "failure", [None, "invalid-json", "extra-field", "timeout", "quota"]
)
def test_gateway_cli_real_client_with_offline_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    failure: str | None,
) -> None:
    """Exercise LOTUS, LiteLLM, SDK and strict parsing, without opening a socket."""
    import httpx
    from lotus.models import LM

    case, payload, plan = prepare_cli_fixture(tmp_path)
    output = tmp_path / model_name
    key = "offline-local-key-not-for-records"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(
        am, "plan_maintenance", lambda *a, **k: pytest.fail("Called Agent")
    )
    monkeypatch.setattr(runner, "verify_source", lambda path: {"files": {}})
    monkeypatch.setattr(LM, "count_tokens", lambda self, messages: 10)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url == httpx.URL("http://127.0.0.1:8787/v1/chat/completions")
        assert request.headers["authorization"] == f"Bearer {key}"
        assert body["model"] == model_name
        assert body["response_format"] == {"type": "json_object"}
        assert "thinking" not in body and "reasoning_effort" not in body
        assert body["max_tokens"] == 8192
        assert request.extensions["timeout"]["read"] == 240
        assert payload["instruction"] in body["messages"][-1]["content"]
        assert "Test-only state" in body["messages"][-1]["content"]
        if failure == "timeout":
            raise httpx.ReadTimeout("Offline timeout", request=request)
        if failure == "quota":
            return httpx.Response(
                429,
                json={
                    "error": {"message": "Offline quota", "type": "insufficient_quota"}
                },
            )
        value = {"result": {"answer": ""}, "state": {"count": len(requests)}}
        if failure == "extra-field":
            value["result"]["evidence"] = None
        content = "{not-json" if failure == "invalid-json" else json.dumps(value)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )

    client_class = httpx.Client

    class FakeTransportClient(client_class):
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"trust_env": False, "follow_redirects": False}
            super().__init__(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "Client", FakeTransportClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "poc",
            case["id"],
            "--data-dir",
            str(tmp_path),
            "--saved-plan",
            str(plan),
            "--backend",
            "chatgpt-plan",
            "--model",
            model_name,
            "--run",
            "--output",
            str(output),
            "--source-receipt",
            str(tmp_path / "unused.json"),
            "--max-requests",
            "3",
            "--max-reserved-tokens",
            "100000",
        ],
    )
    if failure:
        with pytest.raises(Exception):
            runner.main()
        assert len(requests) == 1
        assert not list(output.glob("step-*.json"))
    else:
        runner.main()
        assert len(requests) == 2
        assert json.loads((output / "plan-0.json").read_text()) == configuration()
        step = json.loads((output / "step-0002.json").read_text())
        assert next(iter(step["state_nodes"].values()))[0]["_state"]["count"] == 2
        usage = list((output / "trace/outputs").glob("*-provider-usage.json"))
        assert len(usage) == 2
        assert [json.loads(path.read_text())["total_tokens"] for path in usage] == [
            30,
            30,
        ]
    contract = json.loads((output / "run-contract.json").read_text())
    assert contract["thinking"] == "provider-default; not controlled"
    assert contract["output_tokens_per_request"] is None
    assert contract["known_omissions"] == ["max_tokens", "temperature"]
    assert runner.read_status(output)["model"] == f"openai/{model_name}"
    assert runner.read_status(output)["status"] == (
        "failed" if failure else "completed"
    )
    if failure:
        expected = {
            "invalid-json": "Expecting property name enclosed in double quotes",
            "extra-field": "evidence",
            "timeout": "Timeout",
            "quota": "Offline quota",
        }[failure]
        assert expected in runner.read_status(output)["error"]
    assert json.loads((output / "budget.json").read_text())["requests"] == len(requests)
    for artifact in output.rglob("*"):
        if artifact.is_file():
            assert key.encode() not in artifact.read_bytes()
