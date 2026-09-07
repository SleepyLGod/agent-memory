"""Provider-free benchmark transport wiring and checkpoint identity tests."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.evaluation import agent_memory_drivers as drivers
from agent_memory.evaluation import run as run_module
from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.types import BenchmarkCase, BenchmarkEvent, BenchmarkQuestion
from tools.evaluation import locomo, longmemeval, memory_agent_bench


DEFAULT_TRANSPORT = "chat-json-object"
RESPONSES_TRANSPORT = "responses-json-schema"
CLI_MODULES = (locomo, longmemeval, memory_agent_bench)
FACTORIES = (
    drivers.ClaudeMemoryDriverFactory,
    drivers.ZepMemoryDriverFactory,
    drivers.Mem0MemoryDriverFactory,
    drivers.Mem0MemoryEnhancedDriverFactory,
)


def _bundle(benchmark_id: str = "locomo") -> BenchmarkBundle:
    return BenchmarkBundle(
        benchmark_id,
        "revision",
        "sha256",
        (BenchmarkCase(
            "case", "task",
            (BenchmarkEvent("case", "event", "user", "I like tea."),),
            (BenchmarkQuestion("question", "case", "What drink?", "tea", ("event",)),),
        ),),
        {"run_mode": "integration-smoke"},
    )


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "run", "--bundle-dir", str(tmp_path / "bundle"),
        "--output-dir", str(tmp_path / "output"),
        "--system", "claude-memory",
    ]


def _factory_options(factory: type[Any]) -> dict[str, Any]:
    if factory is drivers.ClaudeMemoryDriverFactory:
        return {}
    options: dict[str, Any] = {"base_namespace": "transport-test"}
    if factory is drivers.ZepMemoryDriverFactory:
        options.update(
            connector=SimpleNamespace(embedding_provider=None),
            neo4j_image="neo4j:5.26.2",
            neo4j_image_digest="sha256:test",
        )
    return options


@pytest.mark.parametrize("module", CLI_MODULES)
@pytest.mark.parametrize("transport", (None, DEFAULT_TRANSPORT, RESPONSES_TRANSPORT))
def test_cli_forwards_transport_without_changing_default_calls(
    module: ModuleType,
    transport: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each CLI forwards opt-in transport while keeping old runner signatures valid."""

    captured: dict[str, Any] = {}

    def run(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return kwargs["output_dir"]

    benchmark_id = {
        locomo: "locomo",
        longmemeval: "longmemeval-v1-cleaned-s",
        memory_agent_bench: "memory-agent-bench",
    }[module]
    monkeypatch.setattr(module, "read_bundle", lambda _path: _bundle(benchmark_id))
    monkeypatch.setattr(module, "run_agent_memory_bundle", run)
    if module is memory_agent_bench:
        monkeypatch.setattr(module, "download_movie_entity_mapping", lambda path: path)
        monkeypatch.setattr(module, "load_movie_entity_mapping", lambda _path: {})
        monkeypatch.setattr(module, "memory_agent_bench_task_contracts", lambda **kw: {})
    args = _cli_args(tmp_path)
    if module is longmemeval:
        args.append("--maintenance-only")
    if transport is not None:
        args.extend(("--structured-output-transport", transport))

    parsed = module.parse_args(args)
    assert parsed.structured_output_transport == (transport or DEFAULT_TRANSPORT)
    assert module.main(args) == tmp_path / "output"
    if transport == RESPONSES_TRANSPORT:
        assert captured["structured_output_transport"] == RESPONSES_TRANSPORT
    else:
        assert "structured_output_transport" not in captured


@pytest.mark.parametrize("module", CLI_MODULES)
@pytest.mark.parametrize(
    ("transport", "model"),
    (("invalid", "deepseek/deepseek-v4-flash"),
     (RESPONSES_TRANSPORT, "openai/gpt-4o"),
     (RESPONSES_TRANSPORT, "deepseek/deepseek-chat")),
)
def test_cli_rejects_transport_before_loading_inputs(
    module: ModuleType,
    transport: str,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid transport/provider combinations fail before bundle or dataset I/O."""

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid transport reached input loading")

    monkeypatch.setattr(module, "read_bundle", unexpected)
    model_flag = "--model" if module is memory_agent_bench else "--memory-model"
    with pytest.raises(SystemExit) as error:
        module.main([
            *_cli_args(tmp_path), "--structured-output-transport", transport,
            model_flag, model,
        ])
    assert error.value.code == 2


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("transport", (None, DEFAULT_TRANSPORT, RESPONSES_TRANSPORT))
def test_factory_passes_transport_into_execution_config(
    factory: type[Any],
    transport: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All four factory paths deliver the selected transport to the adapter config."""

    import agent_memory.adapters.lotus as lotus_module
    import agent_memory.storage.qdrant as qdrant_module

    class ConfigCaptured(Exception):
        """Stop before any adapter or runtime can initialize a model."""

    captured: dict[str, Any] = {}

    def adapter(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ConfigCaptured

    monkeypatch.setattr(lotus_module, "LotusAdapter", adapter)
    monkeypatch.setattr(
        qdrant_module, "SentenceTransformerEmbeddingProvider",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        qdrant_module, "QdrantConnector",
        lambda **kwargs: SimpleNamespace(close=lambda: None),
    )
    options = _factory_options(factory)
    if transport is not None:
        options["structured_output_transport"] = transport
    with pytest.raises(ConfigCaptured):
        factory(**options)("case", tmp_path / "state", tmp_path / "trace")
    assert captured["config"].structured_output_transport == (
        transport or DEFAULT_TRANSPORT
    )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("transport", "model", "message"),
    (("invalid", "deepseek/deepseek-v4-flash", "structured_output_transport"),
     (RESPONSES_TRANSPORT, "openai/gpt-4o", "DeepSeek Responses model"),
     (RESPONSES_TRANSPORT, "deepseek/deepseek-chat", "DeepSeek Responses model")),
)
def test_factory_validates_transport_at_construction(
    factory: type[Any], transport: str, model: str, message: str,
) -> None:
    """Bad execution modes must not be deferred until case-local storage setup."""

    with pytest.raises(ValueError, match=message):
        factory(
            **_factory_options(factory), model_id=model,
            structured_output_transport=transport,
        )


@pytest.mark.parametrize("transport", (DEFAULT_TRANSPORT, RESPONSES_TRANSPORT))
def test_zep_from_environment_forwards_transport(
    transport: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment constructor retains transport when handing off its connector."""

    import agent_memory.storage.neo4j as neo4j_module

    for name, value in {
        "AGENT_MEMORY_NEO4J_IMAGE": "neo4j:5.26.2",
        "AGENT_MEMORY_NEO4J_IMAGE_DIGEST": "sha256:test",
        "AGENT_MEMORY_NEO4J_URI": "bolt://unused",
        "AGENT_MEMORY_NEO4J_PASSWORD": "unused",
    }.items():
        monkeypatch.setenv(name, value)
    for name in (
        "SentenceTransformerEmbeddingProvider", "SentenceTransformerCrossEncoderProvider",
    ):
        monkeypatch.setattr(neo4j_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(neo4j_module, "Neo4jConnector", lambda **kw: SimpleNamespace(**kw))

    factory = drivers.ZepMemoryDriverFactory.from_environment(
        base_namespace="transport-test", structured_output_transport=transport,
    )
    assert factory.structured_output_transport == transport


@pytest.mark.parametrize(
    ("transport", "model", "message"),
    (("invalid", "deepseek/deepseek-v4-flash", "structured_output_transport"),
     (RESPONSES_TRANSPORT, "openai/gpt-4o", "DeepSeek Responses model")),
)
def test_zep_from_environment_validates_before_external_initialization(
    transport: str, model: str, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport errors take precedence even when deployment credentials are absent."""

    import agent_memory.storage.neo4j as neo4j_module

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid transport initialized external storage or models")

    monkeypatch.delenv("AGENT_MEMORY_NEO4J_IMAGE", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_NEO4J_IMAGE_DIGEST", raising=False)
    for name in (
        "Neo4jConnector", "SentenceTransformerEmbeddingProvider",
        "SentenceTransformerCrossEncoderProvider",
    ):
        monkeypatch.setattr(neo4j_module, name, unexpected)
    with pytest.raises(ValueError, match=message):
        drivers.ZepMemoryDriverFactory.from_environment(
            base_namespace="transport-test", model_id=model,
            structured_output_transport=transport,
        )


@pytest.mark.parametrize("system_id", run_module.AGENT_MEMORY_SYSTEMS)
@pytest.mark.parametrize("transport", (None, DEFAULT_TRANSPORT, RESPONSES_TRANSPORT))
@pytest.mark.parametrize("batching", (None, PromptBatching(max_tasks=8)))
def test_run_records_only_enabled_execution_contracts(
    system_id: str,
    transport: str | None,
    batching: PromptBatching | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest provenance and maintenance identity isolate opt-in behavior only."""

    captured: dict[str, Any] = {}

    class FakeFactory:
        """Capture factory inputs without opening benchmark storage."""

        def __init__(self, **kwargs: Any) -> None:
            if transport != RESPONSES_TRANSPORT:
                assert "structured_output_transport" not in kwargs
            captured["factory"] = kwargs

        @classmethod
        def from_environment(cls, **kwargs: Any) -> FakeFactory:
            """Capture the Zep constructor path as well."""
            return cls(**kwargs)

        def runtime_provenance(self) -> dict[str, Any]:
            """Avoid provider-specific version and storage queries."""
            return {}

    class FakeRunner:
        """Capture the manifest inputs without executing benchmark cases."""

        def __init__(self, **kwargs: Any) -> None:
            captured["runner"] = kwargs

        def run(self, bundle: BenchmarkBundle) -> None:
            """No model calls are needed to inspect execution identity."""

    for factory in FACTORIES:
        monkeypatch.setattr(run_module, factory.__name__, FakeFactory)
    monkeypatch.setattr(run_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(run_module, "LiteLLMBenchmarkModel", lambda **kw: None)
    monkeypatch.setattr(run_module, "_require_environment", lambda _system: None)
    monkeypatch.setattr(
        run_module, "collect_runtime_provenance",
        lambda *args, **kwargs: {"source": {}, "runtime": {}},
    )
    monkeypatch.setattr(
        run_module, "validate_run_provenance", lambda *args, **kwargs: None,
    )
    options: dict[str, Any] = {} if transport is None else {"structured_output_transport": transport}
    run_module.run_agent_memory_bundle(
        bundle=_bundle(), contracts={}, system_id=system_id,
        output_dir=tmp_path / "output", memory_thinking_enabled=False,
        prompt_batching=batching, **options,
    )

    provenance = captured["runner"]["runtime_provenance"]["runtime"]["lotus_execution"]
    identity = captured["runner"]["system_contract"].maintenance_execution_id
    parts = ["sem-join-topk:listwise"] if system_id == "zep-memory" else []
    if batching is not None:
        assert provenance["prompt_batching"] == {"max_tasks": 8}
        assert provenance["json_repair_version"] == "bounded-json-v2"
        parts.append(f"prompt-batching:{batching.fingerprint}")
    else:
        assert "prompt_batching" not in provenance
        assert "json_repair_version" not in provenance
    if transport == RESPONSES_TRANSPORT:
        assert captured["factory"]["structured_output_transport"] == transport
        assert provenance["structured_output_transport"] == transport
        parts.append("structured-output-transport:responses-json-schema")
    else:
        assert "structured_output_transport" not in provenance
    assert identity == "|".join(parts)


@pytest.mark.parametrize("system_id", run_module.AGENT_MEMORY_SYSTEMS)
@pytest.mark.parametrize(
    ("transport", "model", "message"),
    (("invalid", "deepseek/deepseek-v4-flash", "structured_output_transport"),
     (RESPONSES_TRANSPORT, "openai/gpt-4o", "DeepSeek Responses model")),
)
def test_run_validates_transport_before_environment_or_factories(
    system_id: str, transport: str, model: str, message: str,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Invalid modes cannot reach provenance collection, model or storage setup."""

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid transport reached external initialization")

    for name in (
        "_require_environment", "collect_runtime_provenance", "LiteLLMBenchmarkModel",
        *(factory.__name__ for factory in FACTORIES),
    ):
        monkeypatch.setattr(run_module, name, unexpected)
    with pytest.raises(ValueError, match=message):
        run_module.run_agent_memory_bundle(
            bundle=_bundle(), contracts={}, system_id=system_id,
            output_dir=tmp_path / "output", memory_provider_model_id=model,
            structured_output_transport=transport,
        )
