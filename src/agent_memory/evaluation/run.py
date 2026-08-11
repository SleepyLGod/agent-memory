"""Agent-memory system selection around the benchmark-neutral runner."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import math
import os
from pathlib import Path
import re

from .agent_memory_drivers import (
    ClaudeMemoryDriverFactory,
    Mem0MemoryDriverFactory,
    Mem0MemoryEnhancedDriverFactory,
    ZepMemoryDriverFactory,
    build_mem0_semantic_pair_profiles,
)
from agent_memory.adapters.lotus.pair_execution import (
    SEMANTIC_PAIR_EXECUTION_MODES,
    semantic_pair_profiles_fingerprint,
)
from .artifacts import BenchmarkArtifactStore
from .bundle import BenchmarkBundle
from .harness import BenchmarkRunner, MemorySystemContract, TaskContract
from .models import LiteLLMBenchmarkModel
from .provenance import collect_runtime_provenance, validate_run_provenance

AGENT_MEMORY_SYSTEMS = (
    "claude-memory",
    "zep-memory",
    "mem0-memory",
    "mem0-enhanced",
)
SEMANTIC_PAIR_PROFILES = SEMANTIC_PAIR_EXECUTION_MODES
DEFAULT_MEMORY_MODEL_ID = "deepseek-v4-flash"
DEFAULT_PROVIDER_MODEL_ID = "deepseek/deepseek-v4-flash"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_BUILT_IN_CONTRACTS = {
    "claude-memory": (
        "benchmark-event-to-claude-log:v1",
        "claude-memory-declared-sem-topk:v1",
    ),
    "zep-memory": (
        "benchmark-event-to-zep-log:v1",
        "zep-memory-entity-rrf-fact-bfs-cross-encoder:v1",
    ),
    "mem0-memory": (
        "benchmark-event-to-mem0-message:v1",
        "mem0-base-bge-m3-cosine:v1",
    ),
    "mem0-enhanced": (
        "benchmark-event-to-mem0-message:v1",
        "mem0-enhanced-sem-topk:v1",
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
    sem_topk_method: str | None = None,
    sem_groupby_pair_batch_size: int | None = None,
    sem_groupby_pair_batch_retries: int = 0,
    semantic_pair_profile: str = "oracle-only",
    semantic_pair_top_k: int | None = None,
    semantic_pair_min_similarity: float | None = None,
    memory_thinking_enabled: bool = True,
    condition_id: str = "",
    maintenance_only: bool = False,
    maintenance_checkpoint_output_dir: Path | None = None,
    max_new_cases: int | None = None,
) -> Path:
    """Execute one canonical bundle with a selected built-in memory policy."""

    if system_id not in AGENT_MEMORY_SYSTEMS:
        raise ValueError(f"unsupported agent-memory benchmark system {system_id!r}")
    if sem_groupby_pair_batch_size is not None and sem_groupby_pair_batch_size < 1:
        raise ValueError("sem_groupby_pair_batch_size must be positive")
    if sem_groupby_pair_batch_retries < 0:
        raise ValueError("sem_groupby_pair_batch_retries cannot be negative")
    if semantic_pair_profile not in SEMANTIC_PAIR_PROFILES:
        raise ValueError(
            "semantic_pair_profile must be one of: "
            + ", ".join(SEMANTIC_PAIR_PROFILES)
        )
    if semantic_pair_top_k is not None and (
        isinstance(semantic_pair_top_k, bool) or semantic_pair_top_k < 1
    ):
        raise ValueError("semantic_pair_top_k must be a positive integer")
    if semantic_pair_min_similarity is not None and (
        isinstance(semantic_pair_min_similarity, bool)
        or not isinstance(semantic_pair_min_similarity, (int, float))
        or not math.isfinite(float(semantic_pair_min_similarity))
    ):
        raise ValueError("semantic_pair_min_similarity must be finite")
    if semantic_pair_profile == "oracle-only" and (
        semantic_pair_top_k is not None or semantic_pair_min_similarity is not None
    ):
        raise ValueError("oracle-only does not accept semantic pair bounds")
    if semantic_pair_profile == "search-filter":
        if system_id not in {"mem0-memory", "mem0-enhanced"}:
            raise ValueError(
                "search-filter is currently supported only for Mem0 systems"
            )
        if semantic_pair_top_k is None and semantic_pair_min_similarity is None:
            raise ValueError("search-filter requires top_k or min_similarity")
    if sem_topk_method is None:
        sem_topk_method = (
            "pairwise-quick" if system_id == "mem0-enhanced" else "pairwise-naive"
        )
    semantic_pair_profiles = {}
    if system_id in {"mem0-memory", "mem0-enhanced"}:
        import agent_memory as am
        from agent_memory.memories.mem0.storage import MEM0_BGE_M3

        memory_type = (
            am.Mem0Memory if system_id == "mem0-memory" else am.Mem0MemoryEnhanced
        )
        semantic_pair_profiles = build_mem0_semantic_pair_profiles(
            memory_type,
            mode=semantic_pair_profile,
            embedding=MEM0_BGE_M3,
            top_k=semantic_pair_top_k,
            min_similarity=semantic_pair_min_similarity,
        )
    semantic_pair_execution_fingerprint = semantic_pair_profiles_fingerprint(
        semantic_pair_profiles
    )
    maintenance_execution_id = (
        f"semantic-pair-search-filter:{semantic_pair_execution_fingerprint}"
        if semantic_pair_execution_fingerprint
        else ""
    )
    bundle_run_mode = bundle.metadata.get("run_mode")
    if bundle_run_mode is not None and (
        not isinstance(bundle_run_mode, str) or not bundle_run_mode
    ):
        raise TypeError(
            "benchmark bundle run_mode metadata must be a non-empty string"
        )
    run_mode = bundle_run_mode or ("maintenance" if maintenance_only else "full")
    if maintenance_only and bundle_run_mode:
        run_mode = f"{bundle_run_mode}-maintenance"
    runtime_provenance = collect_runtime_provenance(
        PROJECT_ROOT,
        lockfile="uv.lock",
        dependencies=(
            "agent-memory",
            "lotus-ai",
            "pandas",
            *(
                ("neo4j", "sentence-transformers")
                if system_id == "zep-memory"
                else (
                    ("qdrant-client", "sentence-transformers")
                    if system_id in {"mem0-memory", "mem0-enhanced"}
                    else ()
                )
            ),
        ),
    )
    runtime_provenance["runtime"]["lotus_execution"] = {
        "sem_groupby_pair_batch_size": sem_groupby_pair_batch_size,
        "sem_groupby_pair_batch_retries": sem_groupby_pair_batch_retries,
        "semantic_pair_profile": semantic_pair_profile,
        "semantic_pair_top_k": semantic_pair_top_k,
        "semantic_pair_min_similarity": semantic_pair_min_similarity,
        "semantic_pair_execution_fingerprint": (
            semantic_pair_execution_fingerprint or None
        ),
        "semantic_pair_query_profiles": {
            digest: profile.to_dict()
            for digest, profile in sorted(semantic_pair_profiles.items())
        },
    }
    validate_run_provenance(runtime_provenance, run_mode=run_mode)
    _require_environment(system_id)
    if (
        maintenance_checkpoint_output_dir is not None
        and maintenance_checkpoint_output_dir.resolve() == output_dir.resolve()
    ):
        raise ValueError("maintenance checkpoint source and output must be different")
    input_adapter_id, retrieval_recipe_id = _BUILT_IN_CONTRACTS[system_id]
    if system_id in {"mem0-memory", "mem0-enhanced"}:
        if grouped_agg_rule != "rule-all-group":
            raise ValueError(f"{system_id} does not use grouped aggregate rules")
        if system_id == "mem0-memory" and sem_topk_method != "pairwise-naive":
            raise ValueError("mem0-memory Base retrieval does not use sem_topk")
        if memory_thinking_enabled:
            raise ValueError(f"{system_id} benchmark requires thinking disabled")
    system_contract = MemorySystemContract(
        system_id=system_id,
        memory_model_id=memory_model_id,
        memory_provider_model_id=memory_provider_model_id,
        input_adapter_id=input_adapter_id,
        retrieval_recipe_id=(
            f"{retrieval_recipe_id}:{sem_topk_method}"
            if system_id in {"claude-memory", "mem0-enhanced"}
            else retrieval_recipe_id
        ),
        condition_id=condition_id
        or (
            "AM-Mem0-Maintenance"
            if maintenance_only
            and system_id in {"mem0-memory", "mem0-enhanced"}
            else "AM-Mem0-Base"
            if system_id == "mem0-memory"
            else "AM-Mem0-Enhanced"
            if system_id == "mem0-enhanced"
            else ""
        ),
        maintenance_policy_id=(
            "mem0-memory"
            if system_id in {"mem0-memory", "mem0-enhanced"}
            else ""
        ),
        maintenance_rule=(
            "mem0-additive-view:v1"
            if system_id in {"mem0-memory", "mem0-enhanced"}
            else grouped_agg_rule
        ),
        maintenance_execution_id=maintenance_execution_id,
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
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            thinking_enabled=memory_thinking_enabled,
        )
    elif system_id == "zep-memory":
        driver_factory = ZepMemoryDriverFactory.from_environment(
            base_namespace=base_namespace or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            grouped_agg_rule=grouped_agg_rule,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            thinking_enabled=memory_thinking_enabled,
        )
    elif system_id == "mem0-memory":
        driver_factory = Mem0MemoryDriverFactory(
            base_namespace=base_namespace or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            thinking_enabled=memory_thinking_enabled,
        )
    else:
        driver_factory = Mem0MemoryEnhancedDriverFactory(
            base_namespace=base_namespace or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            sem_topk_method=sem_topk_method,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            thinking_enabled=memory_thinking_enabled,
        )
    try:
        storage_provenance = (
            driver_factory.runtime_provenance()
            if isinstance(
                driver_factory,
                (
                    ZepMemoryDriverFactory,
                    Mem0MemoryDriverFactory,
                    Mem0MemoryEnhancedDriverFactory,
                ),
            )
            else None
        )
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
            runtime_provenance=runtime_provenance,
            storage_provenance=storage_provenance,
        ).run(bundle)
    finally:
        close = getattr(driver_factory, "close", None)
        if callable(close):
            close()
    return output_dir


__all__ = [
    "AGENT_MEMORY_SYSTEMS",
    "SEMANTIC_PAIR_PROFILES",
    "run_agent_memory_bundle",
]
