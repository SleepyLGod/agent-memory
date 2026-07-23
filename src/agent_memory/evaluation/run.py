"""Agent-memory system selection around the benchmark-neutral runner."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path
import re

from .agent_memory_drivers import (
    ClaudeMemoryDriverFactory,
    ZepMemoryDriverFactory,
)
from .artifacts import BenchmarkArtifactStore
from .bundle import BenchmarkBundle
from .harness import BenchmarkRunner, MemorySystemContract, TaskContract
from .models import LiteLLMBenchmarkModel

AGENT_MEMORY_SYSTEMS = ("claude-memory", "zep-memory")
DEFAULT_MEMORY_MODEL_ID = "deepseek-v4-flash"
DEFAULT_PROVIDER_MODEL_ID = "deepseek/deepseek-v4-flash"

_BUILT_IN_CONTRACTS = {
    "claude-memory": (
        "benchmark-event-to-claude-log:v1",
        "claude-memory-declared-sem-topk:v1",
    ),
    "zep-memory": (
        "benchmark-event-to-zep-log:v1",
        "zep-memory-entity-rrf-fact-bfs-cross-encoder:v1",
    ),
}


def _namespace(benchmark_id: str, output_dir: Path) -> str:
    label = re.sub(r"[^a-z0-9]+", "-", benchmark_id.lower()).strip("-")
    digest = sha256(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{label}-{digest}"


def _require_environment(system_id: str) -> None:
    required = ["DEEPSEEK_API_KEY"]
    if system_id == "zep-memory":
        required.extend(
            ("AGENT_MEMORY_NEO4J_URI", "AGENT_MEMORY_NEO4J_PASSWORD")
        )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"{system_id} benchmark requires environment variables: "
            + ", ".join(missing)
        )


def run_agent_memory_bundle(
    *,
    bundle: BenchmarkBundle,
    contracts: Mapping[str, TaskContract],
    system_id: str,
    output_dir: Path,
    memory_model_id: str = DEFAULT_MEMORY_MODEL_ID,
    memory_provider_model_id: str = DEFAULT_PROVIDER_MODEL_ID,
    answer_model_id: str = DEFAULT_PROVIDER_MODEL_ID,
    judge_model_id: str = DEFAULT_PROVIDER_MODEL_ID,
    base_namespace: str | None = None,
    grouped_agg_rule: str = "rule-all-group",
    sem_topk_method: str = "pairwise-naive",
    memory_thinking_enabled: bool = True,
    condition_id: str = "",
    maintenance_only: bool = False,
    maintenance_checkpoint_output_dir: Path | None = None,
    max_new_cases: int | None = None,
) -> Path:
    """Execute one canonical bundle with a selected built-in memory policy."""

    if system_id not in AGENT_MEMORY_SYSTEMS:
        raise ValueError(f"unsupported agent-memory benchmark system {system_id!r}")
    _require_environment(system_id)
    if (
        maintenance_checkpoint_output_dir is not None
        and maintenance_checkpoint_output_dir.resolve() == output_dir.resolve()
    ):
        raise ValueError("maintenance checkpoint source and output must be different")
    input_adapter_id, retrieval_recipe_id = _BUILT_IN_CONTRACTS[system_id]
    system_contract = MemorySystemContract(
        system_id=system_id,
        memory_model_id=memory_model_id,
        memory_provider_model_id=memory_provider_model_id,
        input_adapter_id=input_adapter_id,
        retrieval_recipe_id=(
            f"{retrieval_recipe_id}:{sem_topk_method}"
            if system_id == "claude-memory"
            else retrieval_recipe_id
        ),
        condition_id=condition_id,
        maintenance_rule=(
            grouped_agg_rule if system_id == "claude-memory" else "zep-view-policy"
        ),
        thinking_enabled=memory_thinking_enabled,
        consolidation_mode="none",
        framework_cache_mode="disabled",
        checkpoint_enabled=True,
    )
    if system_id == "claude-memory":
        driver_factory = ClaudeMemoryDriverFactory(
            model_id=memory_provider_model_id,
            grouped_agg_rule=grouped_agg_rule,
            sem_topk_method=sem_topk_method,
            thinking_enabled=memory_thinking_enabled,
        )
    else:
        driver_factory = ZepMemoryDriverFactory.from_environment(
            base_namespace=base_namespace or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            thinking_enabled=memory_thinking_enabled,
        )
    try:
        BenchmarkRunner(
            system_contract=system_contract,
            contracts=contracts,
            driver_factory=driver_factory,
            answer_model=LiteLLMBenchmarkModel(model_id=answer_model_id),
            judge_model=LiteLLMBenchmarkModel(model_id=judge_model_id),
            artifacts=BenchmarkArtifactStore(output_dir),
            maintenance_only=maintenance_only,
            maintenance_checkpoint_source=(
                BenchmarkArtifactStore(maintenance_checkpoint_output_dir)
                if maintenance_checkpoint_output_dir is not None
                else None
            ),
            max_new_cases=max_new_cases,
        ).run(bundle)
    finally:
        close = getattr(driver_factory, "close", None)
        if callable(close):
            close()
    return output_dir


__all__ = [
    "AGENT_MEMORY_SYSTEMS",
    "run_agent_memory_bundle",
]
