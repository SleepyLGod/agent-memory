from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from agent_memory.evaluation.agent_memory_drivers import (
    ClaudeMemoryDriverFactory,
    ClaudeMemoryDriver,
    ZepMemoryDriverFactory,
    ZepMemoryDriver,
    event_to_zep_log_row,
)
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkQuestion,
    BenchmarkEvent,
    RetrievalRequest,
)
from agent_memory.policy.retrieval import RetrievalResult


class _Memory:
    def __init__(self, result) -> None:
        self.rows = []
        self.queries = []
        self.result = result
        self._runtime: _CheckpointRuntime = _CheckpointRuntime()

    def add(self, row) -> None:
        self.rows.append(row)

    def query(self, text: str):
        self.queries.append(text)
        return self.result


class _CheckpointRuntime:
    def __init__(self) -> None:
        self.state = {"schema_version": 2, "value": "before"}
        self.restored = None

    def snapshot_state(self):
        return dict(self.state)

    def restore_state(self, snapshot) -> None:
        self.restored = snapshot
        self.state = dict(snapshot)


def _event() -> BenchmarkEvent:
    return BenchmarkEvent(
        sample_id="case-1",
        event_id="event-1",
        speaker="user",
        text="I moved to Paris.",
        session_id="session-1",
        timestamp="2025-01-02T03:04:00",
    )


def test_claude_driver_uses_native_policy_retrieval() -> None:
    memory = _Memory(
        pd.DataFrame(
            [
                {
                    "name": "home",
                    "description": "Current home",
                    "type": "user",
                    "body": "The user lives in Paris.",
                }
            ]
        )
    )
    driver = ClaudeMemoryDriver(memory)

    driver.add(_event())
    result = driver.retrieve(RetrievalRequest("q1", "Where do I live?"))

    assert memory.rows == [
        {
            "message": "I moved to Paris.",
            "role": "user",
            "timestamp": "2025-01-02T03:04:00",
            "session_id": "session-1",
        }
    ]
    assert memory.queries == ["Where do I live?"]
    assert result.channels["memory"][0]["name"] == "home"
    assert "The user lives in Paris." in result.context


def test_zep_event_mapping_is_benchmark_neutral() -> None:
    assert event_to_zep_log_row(_event()) == {
        "content": "user: I moved to Paris.",
        "role": "user",
        "speaker": "user",
        "reference_time": "2025-01-02T03:04:00",
        "source_description": "case-1 / event-1",
    }


def test_zep_driver_preserves_native_entity_and_fact_channels(tmp_path) -> None:
    retrieval = RetrievalResult(
        query="Where do I live?",
        channels={
            "entities": pd.DataFrame(
                [{"record_id": "entity-1", "name": "Paris", "summary": "A city"}]
            ),
            "facts": pd.DataFrame(
                [
                    {
                        "record_id": "fact-1",
                        "fact": "The user lives in Paris",
                        "valid_at": "2025-01-02T03:04:00",
                    }
                ]
            ),
        },
        metrics={"entities": {"latency_ms": 2}, "facts": {"latency_ms": 3}},
    )
    memory = _Memory(retrieval)
    driver = ZepMemoryDriver(memory, trace_dir=tmp_path)

    driver.add(_event())
    result = driver.retrieve(RetrievalRequest("q1", "Where do I live?"))

    assert memory.rows == [event_to_zep_log_row(_event())]
    assert result.channels["entities"][0]["record_id"] == "entity-1"
    assert result.channels["facts"][0]["record_id"] == "fact-1"
    assert "The user lives in Paris" in result.context
    assert result.metrics == {
        **retrieval.metrics,
        "generative_llm_calls": 0,
    }


def test_zep_driver_rejects_generative_retrieval(tmp_path) -> None:
    class GenerativeMemory(_Memory):
        def query(self, text: str):
            trace_dir = tmp_path
            (trace_dir / "events.jsonl").write_text(
                '{"event_type":"llm_call"}\n',
                encoding="utf-8",
            )
            return super().query(text)

    memory = GenerativeMemory(
        RetrievalResult(
            query="question",
            channels={"entities": pd.DataFrame(), "facts": pd.DataFrame()},
        )
    )
    driver = ZepMemoryDriver(memory, trace_dir=tmp_path)

    with pytest.raises(RuntimeError, match="generative LLM call"):
        driver.retrieve(RetrievalRequest("q1", "question"))


def test_agent_driver_round_trips_opaque_runtime_state(tmp_path: Path) -> None:
    memory = _Memory(pd.DataFrame())
    driver = ClaudeMemoryDriver(memory)

    metadata = driver.save_state(tmp_path / "checkpoint")
    memory._runtime.state["value"] = "changed"
    driver.restore_state(tmp_path / "checkpoint", (_event(),))

    assert metadata == {
        "format": "agent-memory-runtime-pickle:v1",
        "schema_version": 2,
    }
    assert memory._runtime.restored == {"schema_version": 2, "value": "before"}


def test_case_factories_create_isolated_policy_instances(monkeypatch, tmp_path) -> None:
    import agent_memory as am
    import agent_memory.adapters.lotus as lotus_module
    import agent_memory.planner as planner_module
    import agent_memory.runtime as runtime_module

    created = []

    class FakeMemory:
        def __init__(self, *, adapter, storage=None) -> None:
            self.adapter = adapter
            self.storage = storage
            created.append(self)

        @classmethod
        def spec(cls):
            return "fake-spec"

    class FakeAdapter:
        def __init__(self, *, model, config) -> None:
            self.model = model
            self.config = config

    class FakeConnector:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    class FakePolicyDifferentiator:
        def __init__(self, *, rules) -> None:
            self.rules = rules

        def differentiate(self, spec):
            return (spec, self.rules.grouped_agg_rule)

    class FakeRuntime:
        def __init__(self, policy, *, adapter) -> None:
            self.policy = policy
            self.adapter = adapter

    monkeypatch.setattr(am, "ClaudeMemory", FakeMemory)
    monkeypatch.setattr(am, "ZepMemory", FakeMemory)
    monkeypatch.setattr(lotus_module, "LotusAdapter", FakeAdapter)
    monkeypatch.setattr(planner_module, "PolicyDifferentiator", FakePolicyDifferentiator)
    monkeypatch.setattr(runtime_module, "MemoryRuntime", FakeRuntime)
    case = BenchmarkCase(
        case_id="case/one",
        task_id="task",
        events=(replace(_event(), sample_id="case/one"),),
        questions=(BenchmarkQuestion("q", "case/one", "?", "a", ()),),
    )

    claude_factory = ClaudeMemoryDriverFactory(
        model_id="model",
        grouped_agg_rule="rule-re-group",
        sem_topk_method="listwise",
        thinking_enabled=False,
    )
    claude = claude_factory(
        case.case_id,
        tmp_path / "attempt-0001",
        tmp_path / "trace",
    )
    assert isinstance(claude, ClaudeMemoryDriver)
    assert created[-1].storage is None
    assert created[-1].adapter.config.lm_num_retries == 2
    assert created[-1].adapter.config.structured_max_tokens == 32_768
    assert created[-1].adapter.config.sem_topk_method == "listwise"
    assert created[-1].adapter.config.lm_model_kwargs == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }

    connector = FakeConnector()
    zep_factory = ZepMemoryDriverFactory(
        connector=connector,
        base_namespace="benchmark-run",
        model_id="model",
        thinking_enabled=False,
    )
    zep = zep_factory(
        case.case_id,
        tmp_path / "attempt-0002",
        tmp_path / "trace",
    )
    assert isinstance(zep, ZepMemoryDriver)
    storage = created[-1].storage
    assert storage is not None
    assert storage.namespace.startswith("benchmark-run-")
    assert storage.namespace.endswith("-attempt-0002")
    assert created[-1].adapter.config.lm_num_retries == 2
    assert created[-1].adapter.config.structured_max_tokens == 32_768
    assert created[-1].adapter.config.lm_model_kwargs == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    zep_factory.close()
    assert connector.closed == 1
