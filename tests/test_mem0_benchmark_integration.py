"""Mem0 benchmark input, driver, and CLI contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_memory.evaluation.agent_memory_drivers import (
    Mem0MemoryDriver,
    Mem0MemoryDriverFactory,
    Mem0MemoryEnhancedDriver,
    build_mem0_semantic_pair_profiles,
    event_to_mem0_log_row,
)
from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.embedding_trace import TracingEmbeddingProvider
from agent_memory.evaluation.harness import MemorySystemContract
from agent_memory.evaluation.run import AGENT_MEMORY_SYSTEMS, run_agent_memory_bundle
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkQuestion,
    RetrievalRequest,
)
from agent_memory.policy.retrieval import RetrievalResult
from agent_memory.storage.embedding import EmbeddingSpec
from agent_memory.memories.mem0.storage import MEM0_BGE_M3
from agent_memory.tracing.semantic import semantic_trace_scope
from tools.evaluation import locomo, longmemeval, memory_agent_bench


def test_mem0_search_filter_targets_the_shared_duplicate_query() -> None:
    import agent_memory as am
    from agent_memory.memories.mem0.storage import MEM0_QDRANT_STATEMENTS
    from agent_memory.planner import PolicyDifferentiator
    from agent_memory.tracing.semantic import query_digest

    base = build_mem0_semantic_pair_profiles(
        am.Mem0Memory,
        mode="search-filter",
        embedding=MEM0_BGE_M3,
        top_k=None,
        min_similarity=0.6,
    )
    enhanced = build_mem0_semantic_pair_profiles(
        am.Mem0MemoryEnhanced,
        mode="search-filter",
        embedding=MEM0_BGE_M3,
        top_k=None,
        min_similarity=0.6,
    )

    assert base == enhanced
    assert len(base) == 1
    profile = next(iter(base.values()))
    assert profile.mode == "search-filter"
    assert profile.direction == "right-to-left"
    assert profile.left_text_columns == ("memory:earlier",)
    assert profile.right_text_columns == ("memory:later",)
    assert profile.min_similarity == pytest.approx(0.6)
    assert profile.embedding_device == "cpu"
    policy = PolicyDifferentiator().differentiate(
        am.Mem0Memory.spec(),
        statements=MEM0_QDRANT_STATEMENTS,
    )
    executed_filter_digests = {
        query_digest(node.query)
        for node in policy.nodes.values()
        if node.query.op == "sem_filter"
    }
    assert set(base) == executed_filter_digests


def test_oracle_only_mem0_has_no_physical_pair_override() -> None:
    import agent_memory as am

    assert (
        build_mem0_semantic_pair_profiles(
            am.Mem0Memory,
            mode="oracle-only",
            embedding=MEM0_BGE_M3,
            top_k=None,
            min_similarity=None,
        )
        == {}
    )


def test_mem0_factory_reuses_one_traced_embedding_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent_memory as am
    import agent_memory.storage.qdrant as qdrant_module

    class FakeEmbeddingProvider:
        def __init__(self, spec, *, device: str, dependency_extra: str) -> None:
            self.spec = spec
            self.device = device
            self.dependency_extra = dependency_extra

        def embed(self, spec, texts):
            del spec, texts
            raise AssertionError("factory construction must not embed")

    class FakeConnector:
        def __init__(self, *, path: Path, embedding_provider) -> None:
            self.path = path
            self.embedding_provider = embedding_provider
            self.closed = False

        def prepare(self, statements) -> None:
            self.statements = statements

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        qdrant_module,
        "SentenceTransformerEmbeddingProvider",
        FakeEmbeddingProvider,
    )
    monkeypatch.setattr(qdrant_module, "QdrantConnector", FakeConnector)
    profiles = build_mem0_semantic_pair_profiles(
        am.Mem0Memory,
        mode="search-filter",
        embedding=MEM0_BGE_M3,
        embedding_device="cuda",
        top_k=None,
        min_similarity=0.6,
    )
    factory = Mem0MemoryDriverFactory(
        base_namespace="test-mem0",
        semantic_pair_profiles=profiles,
        embedding_device="cuda",
    )

    driver = factory("case-1", tmp_path / "state", tmp_path / "trace")

    runtime = driver._memory._runtime
    assert runtime.storage.connector.embedding_provider is (
        runtime._engine.adapter.pair_embedding_provider
    )
    assert runtime._engine.adapter.config.semantic_pair_profiles == profiles
    assert runtime.storage.connector.embedding_provider.device == "cuda"
    driver.close()


def test_mem0_event_mapping_preserves_roles_names_dates_and_captions() -> None:
    assistant = BenchmarkEvent(
        sample_id="case",
        event_id="e1",
        speaker="assistant",
        text="Hello",
        timestamp="2026-07-26T10:00:00",
    )
    named = BenchmarkEvent(
        sample_id="case",
        event_id="e2",
        speaker="Caroline",
        text="I visited Speyer.",
        timestamp="2026-07-26T11:00:00",
        metadata={"blip_caption": "Cathedral by the river"},
    )

    assert event_to_mem0_log_row(assistant) == {
        "role": "assistant",
        "content": "Hello",
        "observation_date": "2026-07-26T10:00:00",
    }
    assert event_to_mem0_log_row(named) == {
        "role": "user",
        "content": (
            "Caroline: I visited Speyer.\n"
            "(description of attached image: Cathedral by the river)"
        ),
        "observation_date": "2026-07-26T11:00:00",
    }


def test_mem0_driver_retrieval_is_non_generative_and_closes_connector(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "record_id": "memory-1",
                "memory": "Caroline visited Speyer.",
                "attributed_to": "user",
                "rank": 1,
                "score": 0.91,
            }
        ]
    )
    memory = SimpleNamespace(
        query=lambda query: RetrievalResult(
            query=query,
            channels={"memories": frame},
            metrics={"memories": {"candidate_ids": ["memory-1"]}},
        )
    )

    class Connector:
        closes = 0

        def close(self) -> None:
            self.closes += 1

    connector = Connector()
    driver = Mem0MemoryDriver(memory, connector=connector, trace_dir=tmp_path)

    output = driver.retrieve(RetrievalRequest("q1", "Where did Caroline visit?"))
    driver.close()
    driver.close()

    assert output.context == "- [user] Caroline visited Speyer."
    assert output.channels["memories"][0]["record_id"] == "memory-1"
    assert output.metrics["generative_llm_calls"] == 0
    assert connector.closes == 1


def test_mem0_enhanced_driver_records_generative_semantic_retrieval(
    tmp_path: Path,
) -> None:
    class Memory:
        def query(self, query: str) -> pd.DataFrame:
            assert query == "Where did Caroline visit?"
            (tmp_path / "events.jsonl").write_text(
                '{"event_type":"provider_usage","phase":"retrieval"}\n',
                encoding="utf-8",
            )
            return pd.DataFrame(
                [
                    {
                        "memory": "Caroline visited Speyer.",
                        "attributed_to": "user",
                    }
                ]
            )

    class Connector:
        def close(self) -> None:
            pass

    driver = Mem0MemoryEnhancedDriver(
        Memory(),
        connector=Connector(),
        trace_dir=tmp_path,
    )

    output = driver.retrieve(RetrievalRequest("q1", "Where did Caroline visit?"))

    assert output.context == "- [user] Caroline visited Speyer."
    assert output.channels["memories"] == (
        {
            "memory": "Caroline visited Speyer.",
            "attributed_to": "user",
        },
    )
    assert output.metrics == {"row_count": 1, "provider_usage_events": 1}


def test_all_agent_benchmark_clis_expose_mem0_without_policy_specific_options(
    tmp_path: Path,
) -> None:
    assert "mem0-memory" in AGENT_MEMORY_SYSTEMS
    assert "mem0-enhanced" in AGENT_MEMORY_SYSTEMS
    common = [
        "run",
        "--bundle-dir",
        str(tmp_path / "bundle"),
        "--output-dir",
        str(tmp_path / "output"),
        "--system",
        "mem0-memory",
    ]
    assert longmemeval.parse_args(common).system == "mem0-memory"
    assert memory_agent_bench.parse_args(common).system == "mem0-memory"
    assert locomo.parse_args(common).system == "mem0-memory"

    with pytest.raises(SystemExit):
        longmemeval.parse_args([*common, "--sem-topk-method", "listwise"])
    with pytest.raises(SystemExit):
        memory_agent_bench.parse_args(
            [*common, "--grouped-agg-rule", "rule-re-group"]
        )
    with pytest.raises(SystemExit):
        locomo.parse_args([*common, "--grouped-agg-rule", "rule-re-group"])

    enhanced = [*common[:-1], "mem0-enhanced"]
    assert longmemeval.parse_args(enhanced).sem_topk_method == "pairwise-quick"
    assert memory_agent_bench.parse_args(enhanced).sem_topk_method == "pairwise-quick"
    assert locomo.parse_args(enhanced).sem_topk_method == "pairwise-quick"
    assert (
        longmemeval.parse_args(
            [*enhanced, "--sem-topk-method", "listwise"]
        ).sem_topk_method
        == "listwise"
    )
    with pytest.raises(SystemExit):
        memory_agent_bench.parse_args(
            [*enhanced, "--grouped-agg-rule", "rule-re-group"]
        )


def test_locomo_cli_exposes_shared_maintenance_without_combining_source(
    tmp_path: Path,
) -> None:
    maintenance = locomo.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "maintenance"),
            "--system",
            "mem0-memory",
            "--condition-id",
            "AM-Mem0-Maintenance",
            "--maintenance-only",
        ]
    )
    assert maintenance.condition_id == "AM-Mem0-Maintenance"
    assert maintenance.maintenance_only is True
    assert maintenance.maintenance_checkpoint_output_dir is None

    branch = locomo.parse_args(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "enhanced"),
            "--system",
            "mem0-enhanced",
            "--condition-id",
            "AM-Mem0-Enhanced-L",
            "--sem-topk-method",
            "listwise",
            "--maintenance-checkpoint-output-dir",
            str(tmp_path / "maintenance"),
        ]
    )
    assert branch.condition_id == "AM-Mem0-Enhanced-L"
    assert branch.sem_topk_method == "listwise"
    assert branch.maintenance_checkpoint_output_dir == tmp_path / "maintenance"

    with pytest.raises(SystemExit):
        locomo.parse_args(
            [
                "run",
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--output-dir",
                str(tmp_path / "invalid"),
                "--system",
                "mem0-memory",
                "--maintenance-only",
                "--maintenance-checkpoint-output-dir",
                str(tmp_path / "maintenance"),
            ]
        )


def test_locomo_cli_forwards_shared_maintenance_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        locomo,
        "read_bundle",
        lambda path: SimpleNamespace(benchmark_id="locomo"),
    )
    monkeypatch.setattr(
        locomo,
        "run_agent_memory_bundle",
        lambda **kwargs: captured.update(kwargs) or output_dir,
    )

    result = locomo.main(
        [
            "run",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(output_dir),
            "--system",
            "mem0-enhanced",
            "--condition-id",
            "AM-Mem0-Enhanced-Q",
            "--sem-topk-method",
            "pairwise-quick",
            "--maintenance-checkpoint-output-dir",
            str(tmp_path / "maintenance"),
        ]
    )

    assert result == output_dir
    assert captured["condition_id"] == "AM-Mem0-Enhanced-Q"
    assert captured["maintenance_only"] is False
    assert captured["maintenance_checkpoint_output_dir"] == tmp_path / "maintenance"


def test_benchmark_embedding_trace_records_inputs_without_vectors(
    tmp_path: Path,
) -> None:
    class Provider:
        device = "cuda"
        torch_version = "2.7.0+cu118"
        torch_cuda_version = "11.8"

        def embed(
            self,
            spec: EmbeddingSpec,
            texts: list[str],
        ) -> list[list[float]]:
            assert spec.dimensions == 2
            assert texts == ["first", "second"]
            return [[1.0, 0.0], [0.0, 1.0]]

    trace_dir = tmp_path / "trace"
    wrapper = TracingEmbeddingProvider(Provider(), trace_dir=trace_dir)
    spec = EmbeddingSpec("memory", "embedding", "model", "revision", 2, True)
    with semantic_trace_scope(
        phase="retrieval",
        case_id="case-1",
        question_id="q1",
    ):
        vectors = wrapper.embed(spec, ["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    event = json.loads((trace_dir / "events.jsonl").read_text())
    assert event["event_type"] == "embedding_call"
    assert event["phase"] == "retrieval"
    assert event["question_id"] == "q1"
    assert event["batch_size"] == 2
    assert event["dimensions"] == 2
    assert event["result_dimensions"] == 2
    assert event["device"] == "cuda"
    assert event["torch_version"] == "2.7.0+cu118"
    assert event["torch_cuda_version"] == "11.8"
    input_payload = json.loads((tmp_path / event["input_path"]).read_text())
    assert input_payload == {
        "embedding": spec.to_dict(),
        "texts": ["first", "second"],
    }
    assert "vectors" not in input_payload


def test_benchmark_embedding_trace_records_failure_and_reraises(
    tmp_path: Path,
) -> None:
    class Provider:
        def embed(
            self,
            spec: EmbeddingSpec,
            texts: list[str],
        ) -> list[list[float]]:
            del spec, texts
            raise RuntimeError("embedding failed")

    wrapper = TracingEmbeddingProvider(Provider(), trace_dir=tmp_path / "trace")
    spec = EmbeddingSpec("memory", "embedding", "model", "revision", 2, True)

    with pytest.raises(RuntimeError, match="embedding failed"):
        wrapper.embed(spec, ["first"])

    event = json.loads((tmp_path / "trace" / "events.jsonl").read_text())
    assert event["status"] == "error"
    assert event["error_type"] == "RuntimeError"
    assert event["error_message"] == "embedding failed"


def test_longmemeval_mem0_system_errors_are_scored_and_do_not_abort() -> None:
    from agent_memory.evaluation.longmemeval import longmemeval_task_contract

    contract = longmemeval_task_contract()

    assert contract.memory_system_error_score == 0.0


def test_mem0_enhanced_runner_uses_shared_maintenance_and_quick_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent_memory.evaluation.run as run_module

    captured: dict[str, object] = {}

    class FakeFactory:
        def __init__(self, **kwargs: object) -> None:
            captured["factory"] = kwargs

        def runtime_provenance(self) -> dict[str, object]:
            return {"connector": "qdrant"}

        def close(self) -> None:
            captured["closed"] = True

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            captured["runner"] = kwargs

        def run(self, bundle: BenchmarkBundle) -> None:
            captured["bundle"] = bundle

    monkeypatch.setattr(run_module, "_require_environment", lambda system_id: None)
    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: {"source": {}, "runtime": {}},
    )
    monkeypatch.setattr(run_module, "validate_run_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "Mem0MemoryEnhancedDriverFactory", FakeFactory)
    monkeypatch.setattr(run_module, "BenchmarkRunner", FakeRunner)

    event = BenchmarkEvent(
        sample_id="case-1",
        event_id="event-1",
        speaker="user",
        text="I visited Speyer.",
    )
    bundle = BenchmarkBundle(
        "longmemeval-v1-cleaned-s",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="longmemeval-v1",
                events=(event,),
                questions=(
                    BenchmarkQuestion("q1", "case-1", "Where?", "Speyer", ()),
                ),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    run_agent_memory_bundle(
        bundle=bundle,
        contracts={},
        system_id="mem0-enhanced",
        output_dir=tmp_path / "output",
        memory_thinking_enabled=False,
    )

    factory = captured["factory"]
    assert isinstance(factory, dict)
    assert isinstance(factory["base_namespace"], str)
    assert factory["base_namespace"].startswith("longmemeval-v1-cleaned-s-")
    assert factory["model_id"] == "deepseek/deepseek-v4-flash"
    assert factory["sem_topk_method"] == "pairwise-quick"
    assert factory["thinking_enabled"] is False
    runner = captured["runner"]
    assert isinstance(runner, dict)
    system_contract = runner["system_contract"]
    assert isinstance(system_contract, MemorySystemContract)
    assert system_contract.condition_id == "AM-Mem0-Enhanced"
    assert system_contract.effective_maintenance_policy_id == "mem0-memory"
    assert (
        system_contract.retrieval_recipe_id
        == "mem0-enhanced-sem-topk:v1:pairwise-quick"
    )
    assert system_contract.maintenance_rule == "mem0-additive-view:v1"
    assert captured["closed"] is True


def test_mem0_search_filter_runner_records_physical_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent_memory.evaluation.run as run_module

    captured: dict[str, object] = {}

    class FakeFactory:
        def __init__(self, **kwargs: object) -> None:
            captured["factory"] = kwargs

        def runtime_provenance(self) -> dict[str, object]:
            return {"connector": "qdrant"}

        def close(self) -> None:
            pass

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            captured["runner"] = kwargs

        def run(self, bundle: BenchmarkBundle) -> None:
            del bundle

    monkeypatch.setattr(run_module, "_require_environment", lambda system_id: None)
    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: {"source": {}, "runtime": {}},
    )
    monkeypatch.setattr(run_module, "validate_run_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "Mem0MemoryDriverFactory", FakeFactory)
    monkeypatch.setattr(run_module, "BenchmarkRunner", FakeRunner)
    event = BenchmarkEvent(
        sample_id="case-1",
        event_id="event-1",
        speaker="user",
        text="I visited Speyer.",
    )
    bundle = BenchmarkBundle(
        "locomo",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="locomo",
                events=(event,),
                questions=(
                    BenchmarkQuestion("q1", "case-1", "Where?", "Speyer", ()),
                ),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    run_agent_memory_bundle(
        bundle=bundle,
        contracts={},
        system_id="mem0-memory",
        output_dir=tmp_path / "output",
        memory_thinking_enabled=False,
        semantic_pair_profile="search-filter",
        semantic_pair_min_similarity=0.6,
        embedding_device="cuda",
    )

    factory = captured["factory"]
    assert isinstance(factory, dict)
    profiles = factory["semantic_pair_profiles"]
    assert isinstance(profiles, dict)
    assert len(profiles) == 1
    profile = next(iter(profiles.values()))
    assert profile.embedding_device == "cuda"
    assert factory["embedding_device"] == "cuda"
    runner = captured["runner"]
    assert isinstance(runner, dict)
    contract = runner["system_contract"]
    assert isinstance(contract, MemorySystemContract)
    assert contract.maintenance_execution_id.startswith(
        "semantic-pair-search-filter:"
    )
    assert "execution=semantic-pair-search-filter:" in contract.effective_condition_id
    execution = runner["runtime_provenance"]["runtime"]["lotus_execution"]
    assert execution["semantic_pair_profile"] == "search-filter"
    assert execution["semantic_pair_min_similarity"] == pytest.approx(0.6)
    assert execution["embedding_device"] == "cuda"
    assert execution["semantic_pair_execution_fingerprint"]

    run_agent_memory_bundle(
        bundle=bundle,
        contracts={},
        system_id="mem0-memory",
        output_dir=tmp_path / "oracle-gpu",
        memory_thinking_enabled=False,
        embedding_device="cuda",
    )

    oracle_runner = captured["runner"]
    assert isinstance(oracle_runner, dict)
    oracle_contract = oracle_runner["system_contract"]
    assert isinstance(oracle_contract, MemorySystemContract)
    assert oracle_contract.maintenance_execution_id == "embedding-device:cuda"
    assert "execution=embedding-device:cuda" in (
        oracle_contract.effective_condition_id
    )
