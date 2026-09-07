"""Agent-memory system selection around the benchmark-neutral runner."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import math
import os
from pathlib import Path
import re
from typing import Any

from .agent_memory_drivers import (
    ClaudeMemoryDriverFactory,
    LOTUS_CACHE_MODES,
    Mem0MemoryDriverFactory,
    Mem0MemoryEnhancedDriverFactory,
    SEMANTIC_PAIR_BGE_M3,
    ZepMemoryDriverFactory,
    build_mem0_semantic_pair_profiles,
    build_operator_semantic_pair_profiles,
    build_site_semantic_pair_profiles,
)
from agent_memory.adapters.lotus.pair_execution import (
    SEMANTIC_PAIR_EXECUTION_MODES,
    semantic_pair_profiles_fingerprint,
)
from agent_memory.adapters.lotus.context import (
    LOTUS_MEMORY_CACHE_ID,
    LOTUS_MEMORY_CACHE_MAX_SIZE,
    SEM_AGG_DISPATCH_METHODS,
    SEM_JOIN_TOPK_METHODS,
)
from agent_memory.adapters.lotus.json_output import JSON_REPAIR_VERSION
from agent_memory.adapters.lotus.prompt_batching import (
    PromptBatching,
    validate_structured_output_transport,
)
from agent_memory.planner import DEFAULT_GROUPED_AGG_RULE
from .artifacts import BenchmarkArtifactStore
from .bundle import BenchmarkBundle
from .harness import BenchmarkRunner, MemorySystemContract, TaskContract
from .models import LiteLLMBenchmarkModel
from .provenance import collect_runtime_provenance, validate_run_provenance
from .semantic_pair_config import load_semantic_pair_profile_config

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


def resolve_grouped_agg_rule(
    system_id: str,
    grouped_agg_rule: str | None,
) -> str:
    """Resolve the grouped-aggregate default for one built-in system."""

    if grouped_agg_rule is not None:
        return grouped_agg_rule
    if system_id in {"mem0-memory", "mem0-enhanced"}:
        return "rule-all-group"
    return DEFAULT_GROUPED_AGG_RULE


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
    grouped_agg_rule: str | None = None,
    sem_topk_method: str | None = None,
    sem_join_topk_method: str | None = None,
    sem_groupby_pair_batch_size: int | None = None,
    sem_groupby_pair_batch_retries: int = 0,
    sem_agg_dispatch: str = "sequential",
    prompt_batching: PromptBatching | None = None,
    structured_output_transport: str = "chat-json-object",
    semantic_pair_profile: str = "oracle-only",
    semantic_pair_top_k: int | None = None,
    semantic_pair_min_similarity: float | None = None,
    semantic_pair_profile_config: Path | None = None,
    lotus_cache_mode: str = "disabled",
    embedding_device: str = "cpu",
    semantic_trace_snapshot_mode: str = "compact",
    refresh_every: int = 1,
    memory_thinking_enabled: bool = True,
    condition_id: str = "",
    maintenance_only: bool = False,
    maintenance_checkpoint_output_dir: Path | None = None,
    max_new_cases: int | None = None,
) -> Path:
    """Execute one canonical bundle with a selected built-in memory policy."""

    if system_id not in AGENT_MEMORY_SYSTEMS:
        raise ValueError(f"unsupported agent-memory benchmark system {system_id!r}")
    validate_structured_output_transport(
        structured_output_transport, model=memory_provider_model_id
    )
    grouped_agg_rule = resolve_grouped_agg_rule(system_id, grouped_agg_rule)
    if (
        isinstance(refresh_every, bool)
        or not isinstance(refresh_every, int)
        or refresh_every < 1
    ):
        raise ValueError("refresh_every must be a positive integer")
    if sem_groupby_pair_batch_size is not None and sem_groupby_pair_batch_size < 1:
        raise ValueError("sem_groupby_pair_batch_size must be positive")
    if sem_groupby_pair_batch_retries < 0:
        raise ValueError("sem_groupby_pair_batch_retries cannot be negative")
    if sem_agg_dispatch not in SEM_AGG_DISPATCH_METHODS:
        raise ValueError(
            "sem_agg_dispatch must be one of: "
            + ", ".join(SEM_AGG_DISPATCH_METHODS)
        )
    if prompt_batching is not None and not isinstance(
        prompt_batching, PromptBatching
    ):
        raise TypeError("prompt_batching must be PromptBatching or None")
    if prompt_batching is not None and sem_agg_dispatch != "sequential":
        raise ValueError(
            "prompt_batching cannot be combined with sem_agg_dispatch"
        )
    if prompt_batching is not None and sem_groupby_pair_batch_size is not None:
        raise ValueError(
            "prompt_batching cannot be combined with "
            "sem_groupby_pair_batch_size"
        )
    if (
        sem_join_topk_method is not None
        and sem_join_topk_method not in SEM_JOIN_TOPK_METHODS
    ):
        raise ValueError(
            "sem_join_topk_method must be one of: "
            + ", ".join(SEM_JOIN_TOPK_METHODS)
        )
    if semantic_pair_profile not in SEMANTIC_PAIR_PROFILES:
        raise ValueError(
            "semantic_pair_profile must be one of: "
            + ", ".join(SEMANTIC_PAIR_PROFILES)
        )
    if lotus_cache_mode not in LOTUS_CACHE_MODES:
        raise ValueError(
            "lotus_cache_mode must be one of: " + ", ".join(LOTUS_CACHE_MODES)
        )
    if semantic_pair_profile_config is not None and (
        semantic_pair_profile != "oracle-only"
        or semantic_pair_top_k is not None
        or semantic_pair_min_similarity is not None
    ):
        raise ValueError(
            "semantic_pair_profile_config is mutually exclusive with global "
            "semantic pair profile bounds"
        )
    if embedding_device not in {"cpu", "cuda"}:
        raise ValueError("embedding_device must be 'cpu' or 'cuda'")
    if semantic_trace_snapshot_mode not in {"compact", "full"}:
        raise ValueError(
            "semantic_trace_snapshot_mode must be 'compact' or 'full'"
        )
    if (
        system_id == "claude-memory"
        and semantic_pair_profile == "oracle-only"
        and semantic_pair_profile_config is None
        and embedding_device != "cpu"
    ):
        raise ValueError(
            "Claude non-CPU embeddings require search-filter or proxy-only"
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
        if semantic_pair_top_k is None and semantic_pair_min_similarity is None:
            raise ValueError("search-filter requires top_k or min_similarity")
    if (
        semantic_pair_profile == "proxy-only"
        and semantic_pair_min_similarity is None
    ):
        raise ValueError("proxy-only requires min_similarity")
    if sem_topk_method is None:
        sem_topk_method = (
            "pairwise-quick" if system_id == "mem0-enhanced" else "pairwise-naive"
        )
    uses_sem_join_topk = system_id == "zep-memory" and grouped_agg_rule in {
        "join-map",
        "rule-join-map",
    }
    if sem_join_topk_method is not None and not uses_sem_join_topk:
        raise ValueError(
            "sem_join_topk_method requires Zep with a join-map grouped aggregate rule"
        )
    resolved_sem_join_topk_method = sem_join_topk_method or "listwise"
    site_profile_config = (
        load_semantic_pair_profile_config(semantic_pair_profile_config)
        if semantic_pair_profile_config is not None
        else None
    )
    semantic_pair_profiles = {}
    semantic_pair_sites = {}
    if site_profile_config is not None:
        if system_id not in {"claude-memory", "zep-memory"}:
            raise ValueError(
                "semantic pair site configs currently support Claude and Zep policies"
            )
        import agent_memory as am
        from agent_memory.planner import DifferentialRules, PolicyDifferentiator

        if system_id == "claude-memory":
            policy = PolicyDifferentiator(
                rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
            ).differentiate(am.ClaudeMemory.spec())
            operators = ("sem_join",)
            embedding = SEMANTIC_PAIR_BGE_M3
        else:
            from agent_memory.memories.zep.storage import (
                GRAPHITI_BGE_M3,
                GRAPHITI_NEO4J_STATEMENTS,
            )

            policy = PolicyDifferentiator(
                rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
            ).differentiate(
                am.ZepMemory.spec(),
                statements=GRAPHITI_NEO4J_STATEMENTS,
            )
            operators = ("sem_filter", "sem_join", "sem_groupby")
            embedding = GRAPHITI_BGE_M3
        semantic_pair_profiles, semantic_pair_sites = (
            build_site_semantic_pair_profiles(
                policy,
                bindings=site_profile_config.bindings,
                operators=operators,
                embedding=embedding,
                embedding_device=embedding_device,
            )
        )
    elif (
        semantic_pair_profile in {"search-filter", "proxy-only"}
        and system_id == "claude-memory"
    ):
        import agent_memory as am
        from agent_memory.planner import DifferentialRules, PolicyDifferentiator

        policy = PolicyDifferentiator(
            rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
        ).differentiate(am.ClaudeMemory.spec())
        semantic_pair_profiles = build_operator_semantic_pair_profiles(
            policy,
            mode=semantic_pair_profile,
            operators=("sem_join",),
            embedding=SEMANTIC_PAIR_BGE_M3,
            embedding_device=embedding_device,
            top_k=semantic_pair_top_k,
            min_similarity=semantic_pair_min_similarity,
        )
    elif (
        semantic_pair_profile in {"search-filter", "proxy-only"}
        and system_id == "zep-memory"
    ):
        import agent_memory as am
        from agent_memory.memories.zep.storage import (
            GRAPHITI_BGE_M3,
            GRAPHITI_NEO4J_STATEMENTS,
        )
        from agent_memory.planner import DifferentialRules, PolicyDifferentiator

        policy = PolicyDifferentiator(
            rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
        ).differentiate(
            am.ZepMemory.spec(),
            statements=GRAPHITI_NEO4J_STATEMENTS,
        )
        semantic_pair_profiles = build_operator_semantic_pair_profiles(
            policy,
            mode=semantic_pair_profile,
            operators=("sem_groupby",),
            embedding=GRAPHITI_BGE_M3,
            embedding_device=embedding_device,
            top_k=semantic_pair_top_k,
            min_similarity=semantic_pair_min_similarity,
        )
    elif system_id in {"mem0-memory", "mem0-enhanced"}:
        import agent_memory as am
        from agent_memory.memories.mem0.storage import MEM0_BGE_M3

        memory_type = (
            am.Mem0Memory if system_id == "mem0-memory" else am.Mem0MemoryEnhanced
        )
        semantic_pair_profiles = build_mem0_semantic_pair_profiles(
            memory_type,
            mode=semantic_pair_profile,
            embedding=MEM0_BGE_M3,
            embedding_device=embedding_device,
            top_k=semantic_pair_top_k,
            min_similarity=semantic_pair_min_similarity,
        )
    semantic_pair_execution_fingerprint = semantic_pair_profiles_fingerprint(
        semantic_pair_profiles
    )
    semantic_pair_execution_id = (
        f"semantic-pair-{'site-config' if site_profile_config else semantic_pair_profile}:"
        f"{semantic_pair_execution_fingerprint}"
        if semantic_pair_execution_fingerprint
        else f"embedding-device:{embedding_device}"
        if system_id in {"zep-memory", "mem0-memory", "mem0-enhanced"}
        and embedding_device != "cpu"
        else ""
    )
    maintenance_execution_parts = [
        part
        for part in (
            semantic_pair_execution_id,
            (
                f"sem-join-topk:{resolved_sem_join_topk_method}"
                if uses_sem_join_topk
                else ""
            ),
            (
                f"sem-agg-dispatch:{sem_agg_dispatch}"
                if sem_agg_dispatch != "sequential"
                else ""
            ),
            (
                f"prompt-batching:{prompt_batching.fingerprint}"
                if prompt_batching is not None
                else ""
            ),
            (
                f"structured-output-transport:{structured_output_transport}"
                if structured_output_transport != "chat-json-object"
                else ""
            ),
            f"refresh:count:{refresh_every}" if refresh_every > 1 else "",
            f"lotus-cache:{LOTUS_MEMORY_CACHE_ID}"
            if lotus_cache_mode == "memory"
            else "",
        )
        if part
    ]
    maintenance_execution_id = "|".join(maintenance_execution_parts)
    framework_cache_mode = (
        LOTUS_MEMORY_CACHE_ID if lotus_cache_mode == "memory" else "disabled"
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
                ("neo4j", "sentence-transformers", "torch")
                if system_id == "zep-memory"
                else (
                    ("qdrant-client", "sentence-transformers", "torch")
                    if system_id in {"mem0-memory", "mem0-enhanced"}
                    else (
                        ("sentence-transformers", "torch")
                        if semantic_pair_profiles
                        else ()
                    )
                )
            ),
        ),
    )
    if refresh_every > 1:
        runtime_provenance["runtime"]["refresh"] = {
            "type": "count",
            "every": refresh_every,
        }
    lotus_execution_provenance = {
        "sem_groupby_pair_batch_size": sem_groupby_pair_batch_size,
        "sem_groupby_pair_batch_retries": sem_groupby_pair_batch_retries,
        "sem_join_topk_method": (
            resolved_sem_join_topk_method if uses_sem_join_topk else None
        ),
        "semantic_pair_profile": semantic_pair_profile,
        "semantic_pair_top_k": semantic_pair_top_k,
        "semantic_pair_min_similarity": semantic_pair_min_similarity,
        "embedding_device": embedding_device,
        "semantic_trace_snapshot_mode": semantic_trace_snapshot_mode,
        "semantic_pair_execution_fingerprint": (
            semantic_pair_execution_fingerprint or None
        ),
        "semantic_pair_query_profiles": {
            digest: profile.to_dict()
            for digest, profile in sorted(semantic_pair_profiles.items())
        },
    }
    if sem_agg_dispatch != "sequential":
        lotus_execution_provenance.update(
            {
                "sem_agg_dispatch": sem_agg_dispatch,
            }
        )
    if prompt_batching is not None:
        lotus_execution_provenance["prompt_batching"] = (
            prompt_batching.to_dict()
        )
        lotus_execution_provenance["json_repair_version"] = JSON_REPAIR_VERSION
    if structured_output_transport != "chat-json-object":
        lotus_execution_provenance["structured_output_transport"] = (
            structured_output_transport
        )
    if site_profile_config is not None:
        lotus_execution_provenance.update(
            {
                "semantic_pair_profile_config": site_profile_config.to_dict(),
                "semantic_pair_site_inventory": {
                    site_id: site.to_dict()
                    for site_id, site in sorted(semantic_pair_sites.items())
                },
            }
        )
    if lotus_cache_mode == "memory":
        lotus_execution_provenance.update(
            {
                "lotus_cache_mode": "memory",
                "lotus_cache_max_entries": LOTUS_MEMORY_CACHE_MAX_SIZE,
            }
        )
    runtime_provenance["runtime"]["lotus_execution"] = (
        lotus_execution_provenance
    )
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
        framework_cache_mode=framework_cache_mode,
        checkpoint_enabled=True,
    )
    sem_agg_execution_options = (
        {
            "sem_agg_dispatch": sem_agg_dispatch,
        }
        if sem_agg_dispatch != "sequential"
        else {}
    )
    prompt_batching_options = (
        {"prompt_batching": prompt_batching}
        if prompt_batching is not None
        else {}
    )
    structured_transport_options = (
        {"structured_output_transport": structured_output_transport}
        if structured_output_transport != "chat-json-object"
        else {}
    )
    refresh_execution_options: dict[str, Any] = {}
    if refresh_every > 1:
        refresh_execution_options["refresh_every"] = refresh_every
    if system_id == "claude-memory":
        driver_factory = ClaudeMemoryDriverFactory(
            model_id=memory_provider_model_id,
            grouped_agg_rule=grouped_agg_rule,
            sem_topk_method=sem_topk_method,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            semantic_trace_snapshot_mode=semantic_trace_snapshot_mode,
            lotus_cache_mode=lotus_cache_mode,
            thinking_enabled=memory_thinking_enabled,
            **sem_agg_execution_options,
            **prompt_batching_options,
            **structured_transport_options,
            **refresh_execution_options,
        )
    elif system_id == "zep-memory":
        driver_factory = ZepMemoryDriverFactory.from_environment(
            base_namespace=base_namespace
            or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            grouped_agg_rule=grouped_agg_rule,
            sem_join_topk_method=resolved_sem_join_topk_method,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            embedding_device=embedding_device,
            semantic_trace_snapshot_mode=semantic_trace_snapshot_mode,
            lotus_cache_mode=lotus_cache_mode,
            thinking_enabled=memory_thinking_enabled,
            **sem_agg_execution_options,
            **prompt_batching_options,
            **structured_transport_options,
            **refresh_execution_options,
        )
    elif system_id == "mem0-memory":
        driver_factory = Mem0MemoryDriverFactory(
            base_namespace=base_namespace
            or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            embedding_device=embedding_device,
            semantic_trace_snapshot_mode=semantic_trace_snapshot_mode,
            lotus_cache_mode=lotus_cache_mode,
            thinking_enabled=memory_thinking_enabled,
            **sem_agg_execution_options,
            **prompt_batching_options,
            **structured_transport_options,
            **refresh_execution_options,
        )
    else:
        driver_factory = Mem0MemoryEnhancedDriverFactory(
            base_namespace=base_namespace
            or _namespace(bundle.benchmark_id, output_dir),
            model_id=memory_provider_model_id,
            sem_topk_method=sem_topk_method,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            embedding_device=embedding_device,
            semantic_trace_snapshot_mode=semantic_trace_snapshot_mode,
            lotus_cache_mode=lotus_cache_mode,
            thinking_enabled=memory_thinking_enabled,
            **sem_agg_execution_options,
            **prompt_batching_options,
            **structured_transport_options,
            **refresh_execution_options,
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
