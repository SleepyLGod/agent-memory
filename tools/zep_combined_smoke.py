"""One frozen combined condition using the existing benchmark runner."""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time
from typing import Any

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.site_batching import PairFilterBatching
from agent_memory.evaluation.bundle import read_bundle
from agent_memory.evaluation.locomo_contracts import locomo_task_contract
from agent_memory.evaluation.provenance import build_source_evidence
from agent_memory.evaluation.run import run_agent_memory_bundle
from agent_memory.memories.zep.policy import ZepMemory
from agent_memory.planner import PolicyDifferentiator

MODEL = "deepseek/deepseek-flash"


def save(path: Path, data: Any) -> None:
    """Atomically record progress in this run only."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str))
    temp.replace(path)


def hashes(root: Path) -> dict[str, str]:
    """Bind the experiment to its copied source and fixed inputs."""
    paths = [root / "site-profiles.json", Path(__file__)]
    paths.extend(p for p in (root / "bundle").rglob("*") if p.is_file())
    paths.extend((root / "source/src").rglob("*.py"))
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def execute(root: Path, mode: str, *, full_sample0: bool = False, maintenance_work: bool = False, prefix128: bool = False, combined_physical: bool = False, fact_summary: bool = False, representative: bool = False, node_batch_size: int = 4, pack_small_groups: bool = False, listwise_batch_size: int | None = None, groupby_batch_size: int | None = None, reuse_unchanged_entity_name: bool = False, parallel_fact_extraction: bool = False, lotus_cache_mode: str = "disabled") -> None:
    """Preflight, start, or resume one condition without replaying its prefix."""
    import agent_memory
    from agent_memory.memories.zep.fact_summary import zep_memory_type, zep_storage_statements
    if lotus_cache_mode not in {"disabled", "memory"}:
        raise ValueError("lotus_cache_mode must be disabled or memory")
    if isinstance(node_batch_size, bool) or node_batch_size not in (4, 16):
        raise ValueError("node_batch_size must be 4 or 16")
    if representative and not fact_summary:
        raise ValueError("representative requires --fact-summary")
    if listwise_batch_size is not None and (not representative or type(listwise_batch_size) is not int or listwise_batch_size != 16):
        raise ValueError("listwise batching requires representative and the fixed batch16 condition")
    if groupby_batch_size is not None and (not representative or type(groupby_batch_size) is not int or groupby_batch_size != 16):
        raise ValueError("groupby batching requires representative and the fixed batch16 condition")
    if reuse_unchanged_entity_name and not representative:
        raise ValueError("unchanged names require representative")
    if parallel_fact_extraction and not representative:
        raise ValueError("parallel fact extraction requires representative")
    if (representative or node_batch_size != 4 or pack_small_groups) and not (fact_summary and combined_physical and maintenance_work and (prefix128 or full_sample0)):
        raise ValueError("representative/batch16/packing requires a frozen fact-summary condition")
    if fact_summary and not combined_physical:
        raise ValueError("fact-summary requires --combined-physical")
    if full_sample0 and prefix128:
        raise ValueError("choose full Sample 0 or the frozen 128-event prefix")
    if combined_physical and (not (prefix128 or full_sample0) or not maintenance_work):
        raise ValueError("combined physical experiment requires a frozen maintenance condition")
    bundle = read_bundle(root / "bundle")
    config = {semantic_pair_site_id(ZepMemory._contradictory_fact_pairs.expr):
              PairFilterBatching(("fact_id:later_added",), PromptBatching(max_tasks=node_batch_size),
                                shared_columns=("fact_later_added",) if combined_physical else (),
                                pack_small_groups=pack_small_groups)}
    fusion = "zep-combined" if combined_physical else "zep-target-state"
    if fact_summary:
        fusion = "zep-fact-summary"
    if representative:
        fusion = "zep-representative"
    memory_type = zep_memory_type(fusion)
    groupby_config: dict[str, PromptBatching] = {}
    if groupby_batch_size is not None:
        from agent_memory.planner.physical import optimize_policy, walk
        logical = PolicyDifferentiator().differentiate(memory_type.spec(), statements=zep_storage_statements(fusion))
        physical = optimize_policy(logical, strategy=fusion)
        sites = {semantic_pair_site_id(q) for node in physical.nodes.values()
                 for root_query in (node.query, node.maintenance_query) if root_query is not None
                 for q in walk(root_query) if q.op == "sem_groupby"}
        if len(sites) != 2:
            raise ValueError("representative must have exactly two entity/fact grouping sites")
        groupby_config = {site: PromptBatching(max_tasks=groupby_batch_size) for site in sites}
    listwise_config: dict[str, PromptBatching] = {}
    if listwise_batch_size is not None:
        from agent_memory.planner.physical import optimize_policy, walk
        logical = PolicyDifferentiator().differentiate(memory_type.spec(), statements=zep_storage_statements(fusion))
        physical = optimize_policy(logical, strategy=fusion)
        sites = {semantic_pair_site_id(q) for node in physical.nodes.values()
                 if node.maintenance_query is not None and node.maintenance_query.op != "fused_target_state"
                 for q in walk(node.maintenance_query) if q.op == "sem_join"}
        if len(sites) != 1:
            raise ValueError("representative must have exactly one executable fact matching site")
        listwise_config = {site: PromptBatching(max_tasks=listwise_batch_size) for site in sites}
    aggregate_batching = PromptBatching(max_tasks=node_batch_size) if full_sample0 or maintenance_work else None
    reuse_sites = tuple(config) if maintenance_work else ()
    groupby_size = 32 if maintenance_work else None
    expected = (419, 198) if full_sample0 else ((128, 1) if prefix128 else (24, 1))
    if combined_physical and not full_sample0:
        case = bundle.cases[0]
        event_ids = {event.event_id for event in case.events}
        if not case.questions or any(not q.evidence_event_ids or not set(q.evidence_event_ids) <= event_ids for q in case.questions):
            raise ValueError("quality questions require complete evidence in the 128-event prefix")
        expected = (128, len(case.questions))
    condition = "zep-combined-aggregate4-sample0" if full_sample0 else "zep-combined-24e"
    if full_sample0 and combined_physical:
        condition = "zep-combined-sample0"
    if prefix128:
        condition = "zep-combined-128e"
    if maintenance_work:
        condition += "-maintenance-work-v1"
    if combined_physical:
        condition += "-combined-physical-v1"
    if fact_summary:
        condition += "-fact-summary-v1"
    if representative:
        condition += "-representative-v1"
    if node_batch_size != 4:
        condition += f"-node-batch{node_batch_size}"
    if pack_small_groups:
        condition += "-packed-groups-v1"
    if listwise_batch_size is not None:
        condition += f"-listwise-batch{listwise_batch_size}-v1"
    if groupby_batch_size is not None:
        condition += f"-groupby-prompt{groupby_batch_size}-v1"
    if reuse_unchanged_entity_name:
        condition += "-unchanged-entity-name-v1"
    if parallel_fact_extraction:
        condition += "-parallel-fact-extraction-v2"
    if lotus_cache_mode == "memory":
        condition += "-lotus-memory-cache"
    if mode == "preflight":
        if (root / "status.json").exists():
            raise FileExistsError("preflight already exists")
        if len(bundle.cases) != 1 or (len(bundle.cases[0].events), len(bundle.cases[0].questions)) != expected:
            raise ValueError(f"expected frozen bundle with events/questions={expected}")
        if not Path(agent_memory.__file__).resolve().is_relative_to(root / "source"):
            raise ValueError("wrong AM source imported")
        logical = PolicyDifferentiator().differentiate(memory_type.spec(), statements=zep_storage_statements(fusion))
        extra_config: dict[str, Any] = {"lm_enable_cache": lotus_cache_mode == "memory"}
        site_digests: dict[str, list[str]] = {}
        profile_count = 0
        if combined_physical:
            from agent_memory.evaluation.agent_memory_drivers import build_site_semantic_pair_profiles, BENCHMARK_STRUCTURED_MAX_TOKENS
            from agent_memory.evaluation.semantic_pair_config import load_semantic_pair_profile_config
            from agent_memory.memories.zep.storage import GRAPHITI_BGE_M3
            profiles, sites = build_site_semantic_pair_profiles(logical,
                bindings=load_semantic_pair_profile_config(root / "site-profiles.json").bindings,
                operators=("sem_filter", "sem_join", "sem_groupby"), embedding=GRAPHITI_BGE_M3,
                embedding_device="cuda")
            missing = set(config).union(reuse_sites) - set(sites)
            if missing:
                raise ValueError(f"batching/reuse sites not found in policy: {sorted(missing)}")
            site_digests = {s: list(site.query_digests) for s, site in sites.items()}
            profile_count = len(profiles)
            extra_config = {"semantic_pair_profiles": profiles, "lm_num_retries": 0,
                "structured_parse_retries": 0, "structured_max_tokens": BENCHMARK_STRUCTURED_MAX_TOKENS,
                "lm_model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}},
                "lm_enable_cache": lotus_cache_mode == "memory"}
        adapter = LotusAdapter(model=MODEL, config=LotusExecutionConfig(physical_fusion=fusion, pair_filter_batching=config,
            listwise_join_batching=listwise_config,
            groupby_prompt_batching=groupby_config, reuse_unchanged_entity_name=reuse_unchanged_entity_name,
            parallel_fact_extraction=parallel_fact_extraction,
            sem_agg_prompt_batching=aggregate_batching, predicate_reuse_sites=reuse_sites,
            sem_groupby_pair_batch_size=groupby_size, sem_join_topk_method="listwise", **extra_config))
        policy = adapter.prepare_policy(logical)
        from agent_memory.planner.physical import walk
        from agent_memory.policy.aggregates import ArgMinAggregateSpec
        joins = [q for n in policy.nodes.values() if n.maintenance_query is not None
                 for q in walk(n.maintenance_query) if q.op == "sem_join"]
        if representative:
            fused = [n for n in policy.nodes.values() if n.maintenance_query is not None
                     and n.maintenance_query.op == "fused_target_state"]
            if len(fused) != 1 or not any(any(isinstance(s, ArgMinAggregateSpec) for s in n.query.params.get("aggregates", ())) for n in policy.nodes.values()):
                raise ValueError("representative requires entity-only fusion and deterministic fact state")
            if not joins or any(q.params.get("k") != 1 for q in joins):
                raise ValueError("representative fact matching must retain top-1 listwise execution")
            from agent_memory.tracing.semantic import query_digest
            bound_profiles = extra_config["semantic_pair_profiles"]
            fused_query = fused[0].maintenance_query
            assert fused_query is not None
            required_digests = {fused_query.params["join_profile_digest"]}
            required_digests.update(query_digest(q) for n in policy.nodes.values()
                if n.maintenance_query is not None and n.maintenance_query.op != "fused_target_state"
                for q in walk(n.maintenance_query) if q.op == "sem_join")
            if not required_digests <= set(bound_profiles):
                raise ValueError("physical join rewrite lost its candidate screening profile")
        save(root / "source-evidence.json", build_source_evidence(root / "source"))
        save(root / "manifest.json", {"bundle_fingerprint": bundle.fingerprint, "hashes": hashes(root),
            "source_path": agent_memory.__file__, "model": MODEL, "thinking": False,
            "cache": lotus_cache_mode == "memory", "lotus_cache_mode": lotus_cache_mode,
            "provider_retries": 0, "structured_parse_retries": 0, "answer_parse_attempts": 1,
            "physical_fusion": fusion, "prompt_batching": None,
            "node_batch_size": node_batch_size, "pack_small_groups": pack_small_groups,
            **({"listwise_join_batching": {site: setting.to_dict() for site, setting in listwise_config.items()}} if listwise_config else {}),
            **({"groupby_prompt_batching": {site: setting.to_dict() for site, setting in groupby_config.items()}} if groupby_config else {}),
            **({"reuse_unchanged_entity_name": True} if reuse_unchanged_entity_name else {}),
            **({"parallel_fact_extraction": True} if parallel_fact_extraction else {}),
            "lm_max_batch_size": adapter.config.lm_max_batch_size,
            "sem_join_topk_method": "listwise",
            "site_query_digests": site_digests,
            "matched_profile_query_count": profile_count,
            "fact_state": "earliest representative" if representative else "semantic synthesis",
            "logical_variant": memory_type.__name__,
            "summary_input": "incident facts" if fact_summary else "entity mentions",
            "fusion_minimum_parse_retries": 1,
            "approximate_singleton_fact_identity": combined_physical and not representative,
            "quality_scope": "full Sample 0" if full_sample0 else ("evidence-contained prefix questions; not full Sample 0 accuracy" if combined_physical else "historical fixed questions"),
            "condition": condition, "event_count": expected[0], "question_count": expected[1],
            "sem_agg_prompt_batching": aggregate_batching.to_dict() if aggregate_batching else None,
            "predicate_reuse_sites": reuse_sites, "sem_groupby_pair_batch_size": groupby_size,
            "pair_filter_batching": {s: c.to_dict() for s, c in config.items()},
            "execution_fingerprint": adapter.maintenance_execution_fingerprint,
            "plan_fingerprint": policy.fingerprint,
            "dependencies": {n: version(n) for n in ("lotus-ai", "litellm", "pandas", "neo4j")},
            "note": "one new combined condition; historical comparison, not paired ablation"})
        save(root / "status.json", {"stage": "preflight_passed"})
        return
    status = json.loads((root / "status.json").read_text())
    if mode == "run" and status["stage"] != "preflight_passed":
        raise ValueError("refusing replay of a started experiment")
    if mode == "resume":
        if status["stage"] not in {"failed", "interrupted"}:
            raise ValueError("resume requires a failed or interrupted run")
        checkpoints = list((root / "combined" / "cases").glob("*/checkpoints/current.json"))
        if not checkpoints:
            raise ValueError("resume requires a durable checkpoint")
    manifest = json.loads((root / "manifest.json").read_text())
    if hashes(root) != manifest["hashes"]:
        raise ValueError("source or input changed since preflight")
    if manifest.get("condition") != condition:
        raise ValueError("execution condition changed since preflight")
    save(
        root / "status.json",
        {
            "stage": "running",
            "mode": mode,
            "started_at": time.time(),
            "resuming": mode == "resume",
        },
    )
    try:
        run_agent_memory_bundle(
            bundle=bundle, contracts={"locomo": locomo_task_contract(judge_model_id=MODEL)},
            system_id="zep-memory", output_dir=root / "combined", base_namespace=root.name,
            memory_model_id=MODEL, memory_provider_model_id=MODEL, answer_model_id=MODEL, judge_model_id=MODEL,
            grouped_agg_rule="rule-join-map", sem_join_topk_method="listwise", physical_fusion=fusion,
            pair_filter_batching=config, memory_num_retries=0, structured_parse_retries=0, parse_attempts=1,
            listwise_join_batching=listwise_config,
            groupby_prompt_batching=groupby_config, reuse_unchanged_entity_name=reuse_unchanged_entity_name,
            parallel_fact_extraction=parallel_fact_extraction,
            sem_agg_prompt_batching=aggregate_batching,
            predicate_reuse_sites=reuse_sites, sem_groupby_pair_batch_size=groupby_size,
            semantic_pair_profile_config=root / "site-profiles.json", embedding_device="cuda",
            lotus_cache_mode=lotus_cache_mode, memory_thinking_enabled=False, condition_id=condition,
        )
    except KeyboardInterrupt:
        save(root / "status.json", {"stage": "interrupted", "error": "execution interrupted; not automatically resumed"})
        raise
    except Exception as error:
        save(root / "status.json", {"stage": "failed", "error": str(error), "type": type(error).__name__})
        raise
    save(root / "status.json", {"stage": "completed", "finished_at": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "resume"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-sample0", action="store_true")
    parser.add_argument("--maintenance-work", action="store_true")
    parser.add_argument("--prefix128", action="store_true")
    parser.add_argument("--combined-physical", action="store_true")
    parser.add_argument("--fact-summary", action="store_true")
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--node-batch-size", type=int, default=4)
    parser.add_argument("--pack-small-groups", action="store_true")
    parser.add_argument("--listwise-batch-size", type=int)
    parser.add_argument("--groupby-batch-size", type=int)
    parser.add_argument("--reuse-unchanged-entity-name", action="store_true")
    parser.add_argument("--parallel-fact-extraction", action="store_true")
    parser.add_argument("--lotus-cache-mode", choices=("disabled", "memory"), default="disabled")
    args = parser.parse_args()
    execute(args.root.resolve(), args.mode, full_sample0=args.full_sample0, maintenance_work=args.maintenance_work, prefix128=args.prefix128, combined_physical=args.combined_physical, fact_summary=args.fact_summary, representative=args.representative, node_batch_size=args.node_batch_size, pack_small_groups=args.pack_small_groups, listwise_batch_size=args.listwise_batch_size, groupby_batch_size=args.groupby_batch_size, reuse_unchanged_entity_name=args.reuse_unchanged_entity_name, parallel_fact_extraction=args.parallel_fact_extraction, lotus_cache_mode=args.lotus_cache_mode)
