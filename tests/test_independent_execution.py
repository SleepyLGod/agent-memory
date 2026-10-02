"""Offline checks for one independent extraction branch and atomic publication."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, current_thread
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.memories.zep.representative import ZepRepresentativeMemory
from agent_memory.runtime.executor import PolicyExecutor


def test_registered_branch_has_no_entity_resolution_dependency() -> None:
    adapter = LotusAdapter(config=LotusExecutionConfig(
        physical_fusion="zep-representative", parallel_fact_extraction=True,
        lm_enable_cache=False,
    ))
    policy = adapter.prepare_policy(ZepRepresentativeMemory.differentiate_policy())
    selected = adapter.independent_node_id(policy)
    assert selected is not None
    assert policy.nodes[selected].query.op == "sem_flat_map"
    assert "entities" in policy.nodes[selected].query.params["input_cols"]


class ForkMemory(am.Memory):
    log = am.Log({"text": "Input"})
    extracted = log.sem_flat_map(
        input_cols=["text"], output_cols={"fact": "Fact"}, instruction="Extract {text} into {fact}",
    )
    resolved = log.sem_map(
        input_cols=["text"], output_cols={"entity": "Entity"}, instruction="Resolve {text} into {entity}",
    )
    result = extracted.join(resolved.select(["text", "entity"]), on="text")


class ForkAdapter(LotusAdapter):
    """Require both branches to be in flight without relying on sleep timing."""

    def __init__(self, *, fail: str | None = None, parallel: bool = True) -> None:
        super().__init__()
        self.fail = fail
        self.parallel = parallel
        self.started = Event()
        self.released = Event()
        self.finished = Event()
        self.calls: list[str] = []

    def independent_node_id(self, policy: Any) -> str | None:
        return next(n.node_id for n in policy.nodes.values() if n.query.op == "sem_flat_map") if self.parallel else None

    def execute_independent(self, query: Any, inputs: Any) -> Any:
        self.calls.append("fact")
        self.started.set()
        try:
            assert self.released.wait(5), "independent branch was not scheduled"
            if self.fail == "fact":
                raise ValueError("fact failed")
            source = super().execute(query.inputs[0], inputs)
            return source.assign(fact=source.text)
        finally:
            self.finished.set()

    def execute(self, query: Any, inputs: Any) -> Any:
        if query.op == "sem_flat_map":
            assert not self.parallel, "deferred extraction was executed twice"
            self.calls.append("fact")
            source = super().execute(query.inputs[0], inputs)
            return source.assign(fact=source.text)
        if query.op == "sem_map":
            assert current_thread().name == "MainThread"
            self.calls.append("entity")
            if self.parallel:
                assert self.started.wait(5)
                self.released.set()
            if self.fail == "entity":
                raise ValueError("entity failed")
            source = super().execute(query.inputs[0], inputs)
            return source.assign(entity=source.text)
        return super().execute(query, inputs)


def test_parallel_branch_matches_serial_and_restores() -> None:
    adapter = ForkAdapter()
    executor = PolicyExecutor(ForkMemory.differentiate_policy(), adapter=adapter)
    serial = PolicyExecutor(ForkMemory.differentiate_policy(), adapter=ForkAdapter(parallel=False))
    for _ in range(2):
        executor.add({"text": "same"})
        serial.add({"text": "same"})
    assert adapter.calls.count("fact") == 2
    assert adapter.calls.count("entity") == 2
    assert adapter.finished.is_set()
    # Four join rows, not two: duplicate occurrences must survive.
    result = executor.read_view("result")
    assert len(result) == 4
    assert result[["text", "fact", "entity"]].equals(serial.read_view("result")[["text", "fact", "entity"]])
    restored = PolicyExecutor(ForkMemory.differentiate_policy(), adapter=ForkAdapter())
    restored.restore_state(executor.snapshot_state())
    pd.testing.assert_frame_equal(restored.read_view("result"), result)
    restored.add({"text": "new"})
    assert len(restored.read_view("result")) == 5


@pytest.mark.parametrize("branch", ["fact", "entity"])
def test_either_failure_drains_worker_and_does_not_commit(branch: str) -> None:
    adapter = ForkAdapter(fail=branch)
    executor = PolicyExecutor(ForkMemory.differentiate_policy(), adapter=adapter)
    before = executor.snapshot_state()
    with pytest.raises(ValueError, match=f"{branch} failed"):
        executor.add({"text": "x"})
    assert adapter.finished.is_set()
    assert executor.source_row_count == 0
    assert executor.snapshot_state() == before


def test_config_and_restore_boundaries() -> None:
    base = LotusExecutionConfig(physical_fusion="zep-representative", lm_enable_cache=False)
    policy = ZepRepresentativeMemory.differentiate_policy()
    serial = PolicyExecutor(policy, adapter=LotusAdapter(config=base))
    parallel = PolicyExecutor(policy, adapter=LotusAdapter(config=replace(base, parallel_fact_extraction=True)))
    with pytest.raises(ValueError, match="fingerprint"):
        parallel.restore_state(serial.snapshot_state())
    restored = PolicyExecutor(policy, adapter=LotusAdapter(config=replace(base, parallel_fact_extraction=True)))
    restored.restore_state(parallel.snapshot_state())
    cached = PolicyExecutor(policy, adapter=LotusAdapter(config=replace(base,
        parallel_fact_extraction=True, lm_enable_cache=True)))
    with pytest.raises(ValueError, match="fingerprint"):
        cached.restore_state(parallel.snapshot_state())
    for kwargs in ({"lm_enable_cache": None}, {"physical_fusion": "disabled"}):
        with pytest.raises(ValueError, match="requires"):
            replace(base, parallel_fact_extraction=True, **kwargs)
    assert LotusAdapter(config=base).independent_node_id(policy) is None
    with pytest.raises(ValueError, match="registered"):
        parallel.adapter.independent_node_id(ForkMemory.differentiate_policy())
    selected = parallel._independent_node_id
    assert selected is not None
    parent_id = policy.nodes[selected].input_node_ids[0]
    changed = dict(policy.nodes)
    changed[parent_id] = replace(changed[parent_id], query=replace(changed[parent_id].query, params={"changed": True}))
    with pytest.raises(ValueError, match="input contract"):
        parallel.adapter.independent_node_id(replace(policy, nodes=changed))


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_private_lm_keeps_global_model_usage_metadata_and_trace_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_enabled: bool) -> None:
    import lotus
    import lotus.models
    from litellm import ModelResponse
    from lotus.models import LM
    from agent_memory.tracing.semantic import semantic_trace_scope

    barrier = Barrier(2, timeout=5)

    class FakeLM(LM):
        def _process_uncached_messages(self, uncached_data: Any, all_kwargs: Any,
                                       show_progress_bar: bool, progress_bar_desc: str) -> Any:
            if not uncached_data:
                return []
            barrier.wait()
            responses = []
            for prompt, _ in uncached_data:
                word = "worker" if "worker-input" in str(prompt) else "main"
                tokens = 17 if word == "worker" else 11
                responses.append(ModelResponse(model="deepseek/deepseek-flash", choices=[{
                    "index": 0, "message": {"role": "assistant", "content": json.dumps({"rows": [{"fact": word}]})},
                    "finish_reason": "stop",
                }], usage={"prompt_tokens": tokens, "completion_tokens": 3, "total_tokens": tokens + 3}))
            return responses

    monkeypatch.setattr(lotus.models, "LM", FakeLM)
    # Restore global settings after the test, including the prior model.
    monkeypatch.setattr(lotus.settings, "lm", None)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    adapter = LotusAdapter(model="deepseek/deepseek-flash", config=LotusExecutionConfig(
        physical_fusion="zep-representative", parallel_fact_extraction=True,
        lm_enable_cache=cache_enabled, semantic_trace_dir=tmp_path, structured_parse_retries=0,
    ))
    adapter._context.configure()
    global_lm = lotus.settings.lm
    assert global_lm is not None
    query = am.Source({"text": "Input"}).sem_flat_map(input_cols=["text"],
        output_cols={"fact": "Fact"}, instruction="Extract {text} into {fact}").expr
    with semantic_trace_scope(event_id="event-1", phase="insertion"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, adapter.execute_independent, query,
                                 {"log": pd.DataFrame({"text": ["worker-input"]})})
            main = adapter.execute(query, {"log": pd.DataFrame({"text": ["main-input"]})})
            worker = future.result()
    assert main.fact.tolist() == ["main"] and worker.fact.tolist() == ["worker"]
    assert lotus.settings.lm is global_lm
    assert adapter._independent_context is not None
    worker_lm = adapter._independent_context._lm
    assert worker_lm is not None
    assert worker_lm is not global_lm
    assert global_lm.stats.physical_usage.total_tokens == 14
    assert worker_lm.stats.physical_usage.total_tokens == 20
    if cache_enabled:
        # Exact operator reuse bypasses the LM. Metadata-only differences miss
        # the operator cache but preserve the prompt and hit the native LM cache.
        for frame in (pd.DataFrame({"text": ["worker-input"]}),
                      pd.DataFrame({"text": ["worker-input"], "metadata": [1]})):
            adapter.execute_independent(query, {"log": frame})
        assert worker_lm.stats.operator_cache_hits == 1
        assert worker_lm.stats.cache_hits == 1
        assert worker_lm.stats.physical_usage.total_tokens == 20
        assert worker_lm.stats.virtual_usage.total_tokens == 60
        assert global_lm.stats.operator_cache_hits == 0
        assert global_lm.stats.cache_hits == 0
        assert global_lm.stats.virtual_usage.total_tokens == 14
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    calls = [e for e in events if e["event_type"] == "llm_call"]
    real_calls = [c for c in calls if c["usage_physical_total_tokens"]]
    assert len(real_calls) == 2 and {c["event_id"] for c in real_calls} == {"event-1"}
    assert sorted(c["usage_physical_total_tokens"] for c in real_calls) == [14, 20]
    assert len({c["operator_call_id"] for c in real_calls}) == 2
    assert sum("execution_lane" in c for c in real_calls) == 1
    usage = [e for e in events if e["event_type"] == "provider_usage"]
    assert sorted(e["provider_total_tokens"] for e in usage) == [14, 20]
    assert all(e["provider_finish_reason"] == "stop" for e in usage)
    independent = [e for e in events if e["event_type"] == "independent_execution_result"]
    assert len(independent) == (3 if cache_enabled else 1) and independent[0]["trace_bytes_written"] > 0
    if cache_enabled:
        cached = [e for e in events if e["event_type"] == "framework_cache_usage"]
        assert sum(e["physical_total_tokens"] for e in cached) == 34
        assert sum(e["virtual_total_tokens"] for e in cached) == 74
        assert sum(e["lm_cache_hits"] for e in cached) == 1
        assert sum(e["operator_cache_hits"] for e in cached) == 1
        assert all("execution_lane" in e for e in cached if e["lm_cache_hits"] or e["operator_cache_hits"])


def test_scoped_model_resets_after_failure() -> None:
    from types import SimpleNamespace
    from agent_memory.adapters.lotus.scoped_lm import ScopedLM

    main = SimpleNamespace(cache="main")
    worker = SimpleNamespace(cache="worker")
    model = ScopedLM(main)
    with pytest.raises(ValueError, match="failed"):
        with model.scope(worker):
            assert model.cache == "worker"
            raise ValueError("failed")
    assert model.cache == "main"
