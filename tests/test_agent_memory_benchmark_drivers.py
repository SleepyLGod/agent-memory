from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
    SemanticPairExecutionProfile,
)
from agent_memory.evaluation.agent_memory_drivers import (
    ClaudeMemoryDriverFactory,
    ClaudeMemoryDriver,
    SEMANTIC_PAIR_BGE_M3,
    ZepMemoryDriverFactory,
    ZepMemoryDriver,
    build_operator_semantic_pair_profiles,
    build_site_semantic_pair_profiles,
    event_to_zep_log_row,
    inventory_operator_semantic_pair_sites,
)
from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.embedding_trace import TracingEmbeddingProvider
from agent_memory.evaluation.run import run_agent_memory_bundle
from agent_memory.evaluation.semantic_pair_config import SemanticPairSiteBinding
from agent_memory.evaluation.types import (
    BenchmarkCase,
    BenchmarkQuestion,
    BenchmarkEvent,
    RetrievalRequest,
)
from agent_memory.policy.retrieval import RetrievalResult
from agent_memory.storage.embedding import EmbeddingSpec
from agent_memory.tracing.semantic import semantic_trace_scope


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


def _search_filter_profile(
    *, device: str = "cpu"
) -> SemanticPairExecutionProfile:
    return SemanticPairExecutionProfile(
        mode="search-filter",
        direction="left-to-right",
        left_id_columns=(PAIR_LEFT_ID_COLUMN,),
        right_id_columns=(PAIR_RIGHT_ID_COLUMN,),
        left_text_columns=(PAIR_LEFT_TEXT_COLUMN,),
        right_text_columns=(PAIR_RIGHT_TEXT_COLUMN,),
        embedding=SEMANTIC_PAIR_BGE_M3,
        embedding_device=device,
        top_k=20,
        min_similarity=0.5,
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
        def __init__(self, *, model, config, pair_embedding_provider=None) -> None:
            self.model = model
            self.config = config
            self.pair_embedding_provider = pair_embedding_provider

    class FakeConnector:
        def __init__(self) -> None:
            self.closed = 0
            self.embedding_provider = object()

        def close(self) -> None:
            self.closed += 1

    class FakePolicyDifferentiator:
        def __init__(self, *, rules) -> None:
            self.rules = rules

        def differentiate(self, spec, *, statements=None):
            return (spec, self.rules.grouped_agg_rule, statements)

    class FakeRuntime:
        def __init__(self, policy, *, adapter, storage=None) -> None:
            self.policy = policy
            self.adapter = adapter
            self.storage = storage

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
        sem_groupby_pair_batch_size=12,
        sem_groupby_pair_batch_retries=2,
        semantic_trace_snapshot_mode="full",
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
    assert created[-1].adapter.config.sem_groupby_pair_batch_size == 12
    assert created[-1].adapter.config.sem_groupby_pair_batch_retries == 2
    assert created[-1].adapter.config.semantic_trace_snapshot_mode == "full"
    assert created[-1].adapter.config.lm_model_kwargs == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }

    connector = FakeConnector()
    original_embedding_provider = connector.embedding_provider
    zep_factory = ZepMemoryDriverFactory(
        connector=connector,
        base_namespace="benchmark-run",
        model_id="model",
        grouped_agg_rule="prefer-join-map",
        sem_groupby_pair_batch_size=12,
        sem_groupby_pair_batch_retries=2,
        semantic_pair_profiles={"query": _search_filter_profile()},
        thinking_enabled=False,
        neo4j_image="neo4j:5.26.2",
        neo4j_image_digest="sha256:image",
    )
    zep = zep_factory(
        case.case_id,
        tmp_path / "attempt-0002",
        tmp_path / "trace",
    )
    assert isinstance(zep, ZepMemoryDriver)
    assert created[-1].storage is None
    storage = created[-1]._runtime.storage
    assert storage is not None
    assert storage.namespace.startswith("benchmark-run-")
    assert storage.namespace.endswith("-attempt-0002")
    assert created[-1].adapter.config.lm_num_retries == 2
    assert created[-1].adapter.config.structured_max_tokens == 32_768
    assert created[-1].adapter.config.sem_groupby_pair_batch_size == 12
    assert created[-1].adapter.config.sem_groupby_pair_batch_retries == 2
    assert created[-1].adapter.config.semantic_trace_snapshot_mode == "compact"
    assert created[-1].adapter.config.lm_model_kwargs == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert created[-1]._runtime.policy == (
        am.ZepMemory.spec(),
        "prefer-join-map",
        storage.statements,
    )
    assert isinstance(connector.embedding_provider, TracingEmbeddingProvider)
    assert connector.embedding_provider._provider is original_embedding_provider
    assert created[-1].adapter.pair_embedding_provider is (
        connector.embedding_provider
    )
    assert created[-1].adapter.config.semantic_pair_profiles == {
        "query": _search_filter_profile()
    }

    first_tracing_provider = connector.embedding_provider
    zep_factory(
        case.case_id,
        tmp_path / "attempt-0003",
        tmp_path / "trace-2",
    )
    assert isinstance(connector.embedding_provider, TracingEmbeddingProvider)
    assert connector.embedding_provider is not first_tracing_provider
    assert connector.embedding_provider._provider is original_embedding_provider
    zep_factory.close()
    assert connector.closed == 1


def test_claude_factory_creates_traced_pair_provider_only_for_profile(
    monkeypatch, tmp_path
) -> None:
    import agent_memory.storage as storage_module

    captured = {}

    class FakeEmbeddingProvider:
        def __init__(
            self,
            spec: EmbeddingSpec,
            *,
            device: str,
            dependency_extra: str,
        ) -> None:
            captured.update(
                spec=spec,
                device=device,
                dependency_extra=dependency_extra,
            )
            self.device = device

        def embed(
            self,
            spec: EmbeddingSpec,
            texts: list[str],
        ) -> list[list[float]]:
            del spec
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        storage_module,
        "SentenceTransformerEmbeddingProvider",
        FakeEmbeddingProvider,
    )
    profiles = {"query": _search_filter_profile(device="cuda")}
    driver = ClaudeMemoryDriverFactory(
        grouped_agg_rule="rule-join-map",
        semantic_pair_profiles=profiles,
        thinking_enabled=False,
    )(
        "case-1",
        tmp_path / "attempt-0001",
        tmp_path / "trace",
    )

    adapter = driver._memory._runtime._engine.adapter
    assert captured == {
        "spec": SEMANTIC_PAIR_BGE_M3,
        "device": "cuda",
        "dependency_extra": "zep or mem0",
    }
    assert isinstance(adapter.pair_embedding_provider, TracingEmbeddingProvider)
    assert adapter.config.semantic_pair_profiles == profiles

    oracle = ClaudeMemoryDriverFactory(thinking_enabled=False)(
        "case-2",
        tmp_path / "attempt-0002",
        tmp_path / "trace-2",
    )
    assert oracle._memory._runtime._engine.adapter.pair_embedding_provider is None


def test_embedding_trace_decorator_records_success_and_error(tmp_path: Path) -> None:
    spec = EmbeddingSpec(
        source_column="text",
        property_name="embedding",
        model="test-model",
        revision="test-revision",
        dimensions=2,
        normalize=True,
    )

    class Provider:
        def __init__(self) -> None:
            self.fail = False

        def embed(
            self,
            spec: EmbeddingSpec,
            texts: list[str],
        ) -> list[list[float]]:
            assert spec.model == "test-model"
            if self.fail:
                raise RuntimeError("embedding failed")
            return [[1.0, 0.0] for _ in texts]

    provider = Provider()
    traced = TracingEmbeddingProvider(provider, trace_dir=tmp_path / "trace")
    with semantic_trace_scope(
        phase="retrieval",
        case_id="case-1",
        question_id="q1",
    ):
        assert traced.embed(spec, ["one", "two"]) == [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
        provider.fail = True
        with pytest.raises(RuntimeError, match="embedding failed"):
            traced.embed(spec, ["three"])

    rows = [
        json.loads(line)
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in rows] == ["success", "error"]
    assert all(row["phase"] == "retrieval" for row in rows)
    assert all(row["case_id"] == "case-1" for row in rows)
    assert all(row["question_id"] == "q1" for row in rows)
    assert rows[0]["batch_size"] == 2
    assert rows[0]["dimensions"] == 2
    assert rows[0]["result_count"] == 2
    assert rows[0]["result_dimensions"] == 2
    assert rows[1]["error_type"] == "RuntimeError"


def test_zep_factory_reports_physical_storage_provenance(monkeypatch) -> None:
    import agent_memory.evaluation.agent_memory_drivers as drivers_module

    class Connector:
        def server_version(self) -> str:
            return "5.26.2"

    monkeypatch.setattr(drivers_module, "version", lambda _package: "6.1.0")
    factory = ZepMemoryDriverFactory(
        connector=Connector(),
        base_namespace="benchmark-run",
        neo4j_image="neo4j:5.26.2",
        neo4j_image_digest="sha256:image",
    )

    assert factory.runtime_provenance() == {
        "connector": "neo4j",
        "image": "neo4j:5.26.2",
        "image_digest": "sha256:image",
        "server_version": "5.26.2",
        "driver_version": "6.1.0",
        "embedding_device": "cpu",
        "embedding_runtime_version": "6.1.0",
    }


def test_zep_search_filter_rejects_missing_or_mismatched_provider() -> None:
    class Connector:
        def __init__(self) -> None:
            self.embedding_provider: object | None = None

    with pytest.raises(ValueError, match="requires an embedding provider"):
        ZepMemoryDriverFactory(
            connector=Connector(),
            base_namespace="benchmark-run",
            semantic_pair_profiles={"query": _search_filter_profile()},
            neo4j_image="neo4j:5.26.2",
            neo4j_image_digest="sha256:image",
        )

    connector = Connector()
    connector.embedding_provider = type("Provider", (), {"device": "cpu"})()
    with pytest.raises(ValueError, match="devices do not match"):
        ZepMemoryDriverFactory(
            connector=connector,
            base_namespace="benchmark-run",
            semantic_pair_profiles={
                "query": _search_filter_profile(device="cuda")
            },
            embedding_device="cuda",
            neo4j_image="neo4j:5.26.2",
            neo4j_image_digest="sha256:image",
        )


def test_zep_run_configures_existing_factory_and_checkpoint_flow(
    monkeypatch, tmp_path
) -> None:
    import agent_memory.evaluation.run as run_module

    captured = {}

    class FakeFactory:
        closed = False

        @classmethod
        def from_environment(cls, **kwargs):
            captured["factory"] = kwargs
            return cls()

        def close(self) -> None:
            self.closed = True
            captured["closed"] = True

        def runtime_provenance(self):
            return {
                "connector": "neo4j",
                "image": "neo4j:5.26.2",
                "image_digest": "sha256:image",
                "server_version": "5.26.2",
                "driver_version": "6.1.0",
            }

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["runner"] = kwargs

        def run(self, bundle) -> None:
            captured["bundle"] = bundle

    monkeypatch.setattr(run_module, "_require_environment", lambda system_id: None)
    monkeypatch.setattr(run_module, "ZepMemoryDriverFactory", FakeFactory)
    monkeypatch.setattr(run_module, "BenchmarkRunner", FakeRunner)
    bundle = BenchmarkBundle(
        "longmemeval-v1-cleaned-s",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="longmemeval-v1",
                events=(_event(),),
                questions=(
                    BenchmarkQuestion("q1", "case-1", "?", "answer", ()),
                ),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )
    maintenance_dir = tmp_path / "maintenance"
    output_dir = tmp_path / "output"

    result = run_agent_memory_bundle(
        bundle=bundle,
        contracts={},
        system_id="zep-memory",
        output_dir=output_dir,
        memory_provider_model_id="provider-model",
        grouped_agg_rule="prefer-join-map",
        sem_groupby_pair_batch_size=12,
        sem_groupby_pair_batch_retries=2,
        memory_thinking_enabled=False,
        condition_id="ZEP-SMOKE",
        maintenance_checkpoint_output_dir=maintenance_dir,
    )

    namespace_digest = sha256(
        str(output_dir.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    assert result == output_dir
    assert captured["factory"] == {
        "base_namespace": f"longmemeval-v1-cleaned-s-{namespace_digest}",
        "model_id": "provider-model",
        "grouped_agg_rule": "prefer-join-map",
        "sem_groupby_pair_batch_size": 12,
            "sem_groupby_pair_batch_retries": 2,
            "semantic_pair_profiles": {},
            "embedding_device": "cpu",
            "semantic_trace_snapshot_mode": "compact",
            "lotus_cache_mode": "disabled",
            "thinking_enabled": False,
        }
    system_contract = captured["runner"]["system_contract"]
    assert system_contract.system_id == "zep-memory"
    assert system_contract.condition_id == "ZEP-SMOKE"
    assert system_contract.maintenance_rule == "prefer-join-map"
    assert system_contract.thinking_enabled is False
    assert captured["runner"]["runtime_provenance"]["runtime"][
        "lotus_execution"
    ] == {
        "sem_groupby_pair_batch_size": 12,
        "sem_groupby_pair_batch_retries": 2,
        "semantic_pair_profile": "oracle-only",
        "semantic_pair_top_k": None,
        "semantic_pair_min_similarity": None,
        "embedding_device": "cpu",
        "semantic_trace_snapshot_mode": "compact",
        "semantic_pair_execution_fingerprint": None,
        "semantic_pair_query_profiles": {},
    }
    assert (
        captured["runner"]["maintenance_checkpoint_source"].output_dir
        == maintenance_dir
    )
    assert captured["closed"] is True


def test_zep_run_closes_factory_when_storage_provenance_fails(
    monkeypatch, tmp_path
) -> None:
    import agent_memory.evaluation.run as run_module

    captured = {"closed": False}

    class FakeFactory:
        @classmethod
        def from_environment(cls, **kwargs):
            del kwargs
            return cls()

        def runtime_provenance(self):
            raise RuntimeError("Neo4j provenance failed")

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(run_module, "_require_environment", lambda system_id: None)
    monkeypatch.setattr(run_module, "ZepMemoryDriverFactory", FakeFactory)
    bundle = BenchmarkBundle(
        "longmemeval-v1-cleaned-s",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="longmemeval-v1",
                events=(_event(),),
                questions=(
                    BenchmarkQuestion("q1", "case-1", "?", "answer", ()),
                ),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    with pytest.raises(RuntimeError, match="Neo4j provenance failed"):
        run_agent_memory_bundle(
            bundle=bundle,
            contracts={},
            system_id="zep-memory",
            output_dir=tmp_path / "output",
        )

    assert captured["closed"] is True


@pytest.mark.parametrize("mode", ("search-filter", "proxy-only"))
def test_operator_profiles_target_claude_join_and_zep_groupby(mode: str) -> None:
    import agent_memory as am
    from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
    from agent_memory.planner import DifferentialRules, PolicyDifferentiator

    claude = PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-join-map")
    ).differentiate(am.ClaudeMemory.spec())
    claude_profiles = build_operator_semantic_pair_profiles(
        claude,
        mode=mode,
        operators=("sem_join",),
        embedding=SEMANTIC_PAIR_BGE_M3,
        embedding_device="cuda",
        top_k=20,
        min_similarity=0.5,
    )
    assert len(claude_profiles) == 1
    assert {profile.direction for profile in claude_profiles.values()} == {
        "left-to-right"
    }
    assert {profile.mode for profile in claude_profiles.values()} == {mode}

    zep = PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group")
    ).differentiate(
        am.ZepMemory.spec(),
        statements=GRAPHITI_NEO4J_STATEMENTS,
    )
    zep_profiles = build_operator_semantic_pair_profiles(
        zep,
        mode=mode,
        operators=("sem_groupby",),
        embedding=SEMANTIC_PAIR_BGE_M3,
        embedding_device="cuda",
        top_k=None,
        min_similarity=0.5,
    )
    assert len(zep_profiles) == 6
    assert {profile.direction for profile in zep_profiles.values()} == {"symmetric"}
    assert {profile.mode for profile in zep_profiles.values()} == {mode}


def test_zep_differential_groupby_queries_form_two_predicate_sites() -> None:
    import agent_memory as am
    from agent_memory.memories.zep.storage import (
        GRAPHITI_BGE_M3,
        GRAPHITI_NEO4J_STATEMENTS,
    )
    from agent_memory.planner import DifferentialRules, PolicyDifferentiator

    zep = PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group")
    ).differentiate(
        am.ZepMemory.spec(),
        statements=GRAPHITI_NEO4J_STATEMENTS,
    )
    sites = inventory_operator_semantic_pair_sites(
        zep,
        operators=("sem_groupby",),
    )

    assert len(sites) == 2
    assert sorted(len(site.query_digests) for site in sites.values()) == [3, 3]
    entity_site = next(site for site in sites.values() if not site.partition_by)
    fact_site = next(site for site in sites.values() if site.partition_by)
    assert entity_site.semantic_columns == ("name",)
    assert fact_site.semantic_columns == ("relation_type", "fact")
    assert fact_site.partition_by == ("source_entity_id", "target_entity_id")

    profiles, resolved_sites = build_site_semantic_pair_profiles(
        zep,
        bindings=(
            SemanticPairSiteBinding(
                site_id=entity_site.site_id,
                mode="search-filter",
                top_k=15,
                min_similarity=0.6,
            ),
            SemanticPairSiteBinding(
                site_id=fact_site.site_id,
                mode="search-filter",
                top_k=10,
                min_similarity=None,
            ),
        ),
        operators=("sem_groupby",),
        embedding=GRAPHITI_BGE_M3,
        embedding_device="cuda",
    )

    assert resolved_sites == sites
    assert len(profiles) == 6
    assert {
        (profile.top_k, profile.min_similarity)
        for digest, profile in profiles.items()
        if digest in entity_site.query_digests
    } == {(15, 0.6)}
    assert {
        (profile.top_k, profile.min_similarity)
        for digest, profile in profiles.items()
        if digest in fact_site.query_digests
    } == {(10, None)}


def test_site_profiles_reject_stale_site_before_external_setup() -> None:
    import agent_memory as am
    from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
    from agent_memory.planner import DifferentialRules, PolicyDifferentiator

    zep = PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="rule-re-group")
    ).differentiate(
        am.ZepMemory.spec(),
        statements=GRAPHITI_NEO4J_STATEMENTS,
    )

    with pytest.raises(ValueError, match="not found in policy"):
        build_site_semantic_pair_profiles(
            zep,
            bindings=(
                SemanticPairSiteBinding(
                    site_id="sem_groupby:stale",
                    mode="search-filter",
                    top_k=10,
                    min_similarity=None,
                ),
            ),
            operators=("sem_groupby",),
            embedding=SEMANTIC_PAIR_BGE_M3,
        )


def test_run_rejects_stale_site_before_provenance_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent_memory.evaluation.run as run_module

    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bindings": [
                    {
                        "site_id": "sem_groupby:stale",
                        "mode": "search-filter",
                        "top_k": 10,
                        "min_similarity": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provenance must not run before site validation")
        ),
    )
    bundle = BenchmarkBundle(
        "locomo",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="locomo",
                events=(_event(),),
                questions=(BenchmarkQuestion("q1", "case-1", "?", "answer", ()),),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    with pytest.raises(ValueError, match="not found in policy"):
        run_agent_memory_bundle(
            bundle=bundle,
            contracts={},
            system_id="zep-memory",
            output_dir=tmp_path / "output",
            grouped_agg_rule="rule-re-group",
            semantic_pair_profile_config=profile_path,
            memory_thinking_enabled=False,
        )


@pytest.mark.parametrize(
    ("grouped_agg_rule", "expected_profile_count", "expected_site_count"),
    (
        ("rule-re-group", 6, 2),
        ("prefer-join-map", 4, 3),
    ),
)
def test_run_resolves_site_config_into_manifest_and_query_profiles(
    grouped_agg_rule: str,
    expected_profile_count: int,
    expected_site_count: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent_memory as am
    import agent_memory.evaluation.run as run_module
    from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
    from agent_memory.planner import DifferentialRules, PolicyDifferentiator

    policy = PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
    ).differentiate(
        am.ZepMemory.spec(),
        statements=GRAPHITI_NEO4J_STATEMENTS,
    )
    sites = inventory_operator_semantic_pair_sites(
        policy,
        operators=("sem_join", "sem_groupby"),
    )
    bindings = [
        {
            "site_id": site.site_id,
            "mode": "search-filter",
            "top_k": 10 if site.partition_by else 15,
            "min_similarity": None if site.partition_by else 0.6,
        }
        for site in sites.values()
        if grouped_agg_rule == "rule-re-group"
        or site.operator == "sem_join"
        or site.partition_by
    ]
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps({"schema_version": 1, "bindings": bindings}),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeFactory:
        @classmethod
        def from_environment(cls, **kwargs):
            captured["factory"] = kwargs
            return cls()

        def runtime_provenance(self):
            return {
                "connector": "neo4j",
                "image": "neo4j:5.26.2",
                "image_digest": "sha256:image",
                "server_version": "5.26.2",
                "driver_version": "6.1.0",
                "embedding_device": "cuda",
            }

        def close(self) -> None:
            pass

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["runner"] = kwargs

        def run(self, bundle) -> None:
            captured["bundle"] = bundle

    monkeypatch.setattr(run_module, "_require_environment", lambda _system: None)
    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: {"source": {}, "runtime": {}},
    )
    monkeypatch.setattr(
        run_module,
        "validate_run_provenance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(run_module, "ZepMemoryDriverFactory", FakeFactory)
    monkeypatch.setattr(run_module, "BenchmarkRunner", FakeRunner)
    bundle = BenchmarkBundle(
        "locomo",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="locomo",
                events=(_event(),),
                questions=(BenchmarkQuestion("q1", "case-1", "?", "answer", ()),),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    run_agent_memory_bundle(
        bundle=bundle,
        contracts={},
        system_id="zep-memory",
        output_dir=tmp_path / "output",
        grouped_agg_rule=grouped_agg_rule,
        semantic_pair_profile_config=profile_path,
        embedding_device="cuda",
        lotus_cache_mode="memory",
        memory_thinking_enabled=False,
    )

    factory = captured["factory"]
    assert len(factory["semantic_pair_profiles"]) == expected_profile_count
    bound_query_digests = set(factory["semantic_pair_profiles"])
    expected_bound_digests = {
        query_digest
        for site in sites.values()
        if grouped_agg_rule == "rule-re-group"
        or site.operator == "sem_join"
        or site.partition_by
        for query_digest in site.query_digests
    }
    assert bound_query_digests == expected_bound_digests
    runner = captured["runner"]
    contract = runner["system_contract"]
    assert contract.framework_cache_mode == "lotus-memory:1024"
    execution = runner["runtime_provenance"]["runtime"]["lotus_execution"]
    assert execution["semantic_pair_profile"] == "oracle-only"
    assert len(execution["semantic_pair_site_inventory"]) == expected_site_count
    assert execution["semantic_pair_profile_config"]["source_sha256"] == sha256(
        profile_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("system_id", "grouped_agg_rule", "top_k", "expected_count"),
    (
        ("claude-memory", "rule-join-map", 20, 1),
        ("zep-memory", "rule-re-group", None, 6),
    ),
)
def test_run_wires_search_filter_profiles_before_external_execution(
    system_id,
    grouped_agg_rule,
    top_k,
    expected_count,
    monkeypatch,
    tmp_path,
) -> None:
    import agent_memory.evaluation.run as run_module

    captured = {}

    class FakeFactory:
        def __init__(self, **kwargs) -> None:
            captured["factory"] = kwargs

        @classmethod
        def from_environment(cls, **kwargs):
            return cls(**kwargs)

        def runtime_provenance(self):
            return {
                "connector": "neo4j",
                "image": "neo4j:5.26.2",
                "image_digest": "sha256:image",
                "server_version": "5.26.2",
                "driver_version": "6.1.0",
                "embedding_device": "cuda",
                "embedding_runtime_version": "3.4.1",
            }

        def close(self) -> None:
            captured["closed"] = True

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["runner"] = kwargs

        def run(self, bundle) -> None:
            captured["bundle"] = bundle

    monkeypatch.setattr(run_module, "_require_environment", lambda _system: None)
    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: {"source": {}, "runtime": {}},
    )
    monkeypatch.setattr(
        run_module,
        "validate_run_provenance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(run_module, "BenchmarkRunner", FakeRunner)
    if system_id == "claude-memory":
        monkeypatch.setattr(run_module, "ClaudeMemoryDriverFactory", FakeFactory)
    else:
        monkeypatch.setattr(run_module, "ZepMemoryDriverFactory", FakeFactory)
    bundle = BenchmarkBundle(
        "locomo",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="locomo",
                events=(_event(),),
                questions=(BenchmarkQuestion("q1", "case-1", "?", "answer", ()),),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    run_agent_memory_bundle(
        bundle=bundle,
        contracts={},
        system_id=system_id,
        output_dir=tmp_path / "output",
        grouped_agg_rule=grouped_agg_rule,
        semantic_pair_profile="search-filter",
        semantic_pair_top_k=top_k,
        semantic_pair_min_similarity=0.5,
        lotus_cache_mode="memory",
        embedding_device="cuda",
        memory_thinking_enabled=False,
    )

    profiles = captured["factory"]["semantic_pair_profiles"]
    assert len(profiles) == expected_count
    assert {profile.embedding_device for profile in profiles.values()} == {"cuda"}
    assert captured["factory"]["lotus_cache_mode"] == "memory"
    if system_id == "zep-memory":
        assert captured["factory"]["embedding_device"] == "cuda"
        assert captured["closed"] is True
    contract = captured["runner"]["system_contract"]
    assert contract.maintenance_execution_id.startswith(
        "semantic-pair-search-filter:"
    )
    assert contract.maintenance_execution_id.endswith(
        "|lotus-cache:lotus-memory:1024"
    )
    assert contract.framework_cache_mode == "lotus-memory:1024"
    execution = captured["runner"]["runtime_provenance"]["runtime"][
        "lotus_execution"
    ]
    assert len(execution["semantic_pair_query_profiles"]) == expected_count
    assert execution["embedding_device"] == "cuda"
    assert execution["lotus_cache_mode"] == "memory"


def test_claude_search_filter_rejects_rule_without_join_before_external_setup(
    monkeypatch, tmp_path
) -> None:
    import agent_memory.evaluation.run as run_module

    def unexpected_provenance(*args, **kwargs):
        del args, kwargs
        raise AssertionError("provenance must not run before profile validation")

    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        unexpected_provenance,
    )
    bundle = BenchmarkBundle(
        "longmemeval-v1-cleaned-s",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="longmemeval-v1",
                events=(_event(),),
                questions=(
                    BenchmarkQuestion("q1", "case-1", "?", "answer", ()),
                ),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    with pytest.raises(ValueError, match="no eligible sem_join queries"):
        run_agent_memory_bundle(
            bundle=bundle,
            contracts={},
            system_id="claude-memory",
            output_dir=tmp_path / "output",
            semantic_pair_profile="search-filter",
            semantic_pair_min_similarity=0.6,
        )


def test_claude_oracle_rejects_cuda_before_external_setup(
    monkeypatch, tmp_path
) -> None:
    import agent_memory.evaluation.run as run_module

    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provenance must not run before device validation")
        ),
    )
    bundle = BenchmarkBundle(
        "longmemeval-v1-cleaned-s",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="longmemeval-v1",
                events=(_event(),),
                questions=(BenchmarkQuestion("q1", "case-1", "?", "answer", ()),),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    with pytest.raises(ValueError, match="require.*search-filter"):
        run_agent_memory_bundle(
            bundle=bundle,
            contracts={},
            system_id="claude-memory",
            output_dir=tmp_path / "output",
            embedding_device="cuda",
        )


def test_proxy_only_requires_threshold_before_external_setup(
    monkeypatch, tmp_path
) -> None:
    import agent_memory.evaluation.run as run_module

    monkeypatch.setattr(
        run_module,
        "collect_runtime_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provenance must not run before profile validation")
        ),
    )
    bundle = BenchmarkBundle(
        "longmemeval-v1-cleaned-s",
        "revision",
        "sha256",
        (
            BenchmarkCase(
                case_id="case-1",
                task_id="longmemeval-v1",
                events=(_event(),),
                questions=(BenchmarkQuestion("q1", "case-1", "?", "answer", ()),),
            ),
        ),
        {"run_mode": "integration-smoke"},
    )

    with pytest.raises(ValueError, match="proxy-only requires min_similarity"):
        run_agent_memory_bundle(
            bundle=bundle,
            contracts={},
            system_id="claude-memory",
            output_dir=tmp_path / "output",
            semantic_pair_profile="proxy-only",
            semantic_pair_top_k=10,
        )


def test_strict_zep_join_map_fails_before_storage_side_effects(tmp_path) -> None:
    class CountingConnector:
        def __init__(self) -> None:
            self.prepare_count = 0
            self.transaction_count = 0

        def prepare(self, statements) -> None:
            del statements
            self.prepare_count += 1

        def transaction(self, **kwargs):
            del kwargs
            self.transaction_count += 1
            raise AssertionError("storage transaction must not begin during compilation")

        def close(self) -> None:
            pass

    connector = CountingConnector()
    factory = ZepMemoryDriverFactory(
        connector=connector,
        base_namespace="strict-join-map",
        grouped_agg_rule="rule-join-map",
        neo4j_image="neo4j:5.26.2",
        neo4j_image_digest="sha256:image",
    )

    with pytest.raises(NotImplementedError, match="partition_by.*rule-join-map"):
        factory(
            "case-1",
            tmp_path / "attempt-0001",
            tmp_path / "trace",
        )

    assert connector.prepare_count == 0
    assert connector.transaction_count == 0
