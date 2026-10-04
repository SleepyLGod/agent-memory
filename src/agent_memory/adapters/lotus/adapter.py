"""LOTUS execution adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sized
from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any
from uuid import uuid4

from agent_memory.adapters.lotus.context import (
    LOTUS_MEMORY_CACHE_ID,
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.algebraic_aggregates import (
    execute_aggregate_finalize,
    execute_aggregate_state,
    execute_aggregate_state_update,
    execute_algebraic_aggregate,
    is_algebraic_aggregate_query,
)
from agent_memory.adapters.lotus.pair_execution import (
    semantic_pair_profiles_fingerprint,
)
from agent_memory.adapters.lotus.relational import (
    execute_agg,
    execute_alias,
    execute_array_agg,
    execute_array_cat,
    execute_assign,
    execute_concat,
    execute_drop_duplicates,
    execute_explode,
    execute_filter,
    execute_flatten,
    execute_group_by,
    execute_join,
    execute_let,
    execute_limit,
    execute_min,
    execute_select,
    execute_subtract,
    execute_unnest,
    execute_union,
    execute_union_by_name,
)
from agent_memory.adapters.lotus.sem_agg import execute_sem_agg
from agent_memory.adapters.lotus.sem_filter import execute_sem_filter
from agent_memory.adapters.lotus.sem_flat_map import execute_sem_flat_map
from agent_memory.adapters.lotus.sem_groupby import execute_sem_groupby
from agent_memory.adapters.lotus.sem_join import execute_sem_join
from agent_memory.adapters.lotus.sem_map import execute_sem_map
from agent_memory.adapters.lotus.sem_topk import execute_sem_topk
from agent_memory.tracing.semantic import (
    measure_semantic_trace_io,
    query_digest,
    semantic_trace_scope,
    write_compact_operator_trace,
    write_trace_event,
)
from agent_memory.adapters.lotus.sources import execute_log, execute_materialized_view
from agent_memory.adapters.lotus.window import (
    execute_count_window,
    execute_process_window,
    execute_window_source,
)
from agent_memory.policy.logical import QueryExpr
from agent_memory.storage.embedding import EmbeddingProvider
from agent_memory.planner.differential_policy import DifferentiatedPolicy
from agent_memory.planner.physical import COMBINED_VERSION, FUSION_VERSION, SUMMARY_VERSION, REPRESENTATIVE_VERSION, optimize_policy, walk
from agent_memory.adapters.lotus.fusion import execute_target_state

_FUSION_VERSIONS = {"zep-target-state": FUSION_VERSION, "zep-combined": COMBINED_VERSION,
                    "zep-fact-summary": SUMMARY_VERSION, "zep-representative": REPRESENTATIVE_VERSION}

DEFAULT_LOTUS_MODEL = "deepseek/deepseek-v4-pro"


@dataclass
class LotusAdapter:
    """Execution adapter for LOTUS-backed semantic operators."""

    model: str = DEFAULT_LOTUS_MODEL
    config: LotusExecutionConfig = field(default_factory=LotusExecutionConfig)
    pair_embedding_provider: EmbeddingProvider | None = None
    _context: LotusExecutionContext = field(init=False, repr=False)
    _independent_context: LotusExecutionContext | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize shared LOTUS execution context."""

        self._context = LotusExecutionContext(
            model=self.model,
            config=self.config,
            pair_embedding_provider=self.pair_embedding_provider,
        )

    @property
    def supports_legacy_restore(self) -> bool:
        """Physical rewrites require the node-state executor and its identity."""
        return self.config.physical_fusion == "disabled" and not self.config.pair_filter_batching and not self.config.listwise_join_batching and not self.config.groupby_prompt_batching and self.config.sem_agg_prompt_batching is None and not self.config.predicate_reuse_sites and not any(p.top_k is not None and p.mode != "oracle-only" for p in self.config.semantic_pair_profiles.values())

    def bounded_predicate_scope(self, query: QueryExpr) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """Declare non-row-local filter dependencies and complete Top-k buckets."""
        profile = self.config.semantic_pair_profiles.get(query_digest(query))
        if query.op != "sem_filter" or profile is None or profile.mode == "oracle-only" or profile.top_k is None:
            return None
        from agent_memory.policy.schema import output_columns

        columns = output_columns(query.inputs[0])
        referenced = tuple(dict.fromkeys(re.findall(r"\{([^{}]+)\}", str(query.params["instruction"]))))
        # Unknown dependencies disable the metadata-only shortcut, not screening.
        semantic = referenced if referenced and set(referenced) <= set(columns) else tuple(columns)
        dependencies = tuple(dict.fromkeys((*semantic, *profile.left_id_columns,
            *profile.right_id_columns, *profile.left_text_columns, *profile.right_text_columns)))
        buckets = (profile.right_id_columns if profile.direction == "right-to-left"
                   else profile.left_id_columns if profile.direction == "left-to-right" else ())
        return dependencies, buckets

    @property
    def maintenance_execution_fingerprint(self) -> str:
        """Identify physical settings that can change maintained state."""

        base = self._maintenance_execution_fingerprint()
        if self.config.parallel_fact_extraction:
            base = sha256(f"{base}\nindependent-fact-extraction-v2:cache={self.config.lm_enable_cache}".encode()).hexdigest()
        if any(p.top_k is not None and p.mode != "oracle-only" for p in self.config.semantic_pair_profiles.values()):
            base = sha256(f"{base}\ncomplete-candidate-topk-v2".encode()).hexdigest()
        if self.config.reuse_unchanged_entity_name:
            base = sha256(f"{base}\nunchanged-entity-name-v1".encode()).hexdigest()
        if self.config.groupby_prompt_batching:
            from agent_memory.planner.serialization import stable_json
            base = sha256(stable_json({"base": base, "groupby-prompt-batching-v1": {
                site: setting.to_dict() for site, setting in sorted(self.config.groupby_prompt_batching.items())
            }}).encode()).hexdigest()
        if self.config.listwise_join_batching:
            from agent_memory.planner.serialization import stable_json
            base = sha256(stable_json({"base": base, "listwise-join-batching-v1": {
                site: setting.to_dict() for site, setting in sorted(self.config.listwise_join_batching.items())
            }}).encode()).hexdigest()
        if self.config.predicate_reuse_sites:
            base = sha256(f"{base}\npredicate-reuse-v1:{sorted(self.config.predicate_reuse_sites)}".encode()).hexdigest()
        if self.config.sem_agg_prompt_batching is not None:
            base = sha256(f"{base}\nsem-agg-prompt-batching-v1:{self.config.sem_agg_prompt_batching.fingerprint}".encode()).hexdigest()
        if self.config.pair_filter_batching:
            from agent_memory.planner.serialization import stable_json
            base = sha256(stable_json({"base": base, "pair_filter_batching": {
                site: config.to_dict() for site, config in sorted(self.config.pair_filter_batching.items())
            }}).encode()).hexdigest()
        if self.config.physical_fusion == "disabled":
            return base
        version = _FUSION_VERSIONS[self.config.physical_fusion]
        if self.config.physical_fusion in {"zep-combined", "zep-fact-summary", "zep-representative"}:
            from agent_memory.planner.serialization import stable_json
            base = sha256(stable_json({"base": base, "model": self.model,
                "lm_model_kwargs": dict(self.config.lm_model_kwargs),
                "structured_max_tokens": self.config.structured_max_tokens,
                "structured_parse_retries": self.config.structured_parse_retries,
                "sem_groupby_default": self.config.sem_groupby_default,
                "sem_groupby_pair_batch_size": self.config.sem_groupby_pair_batch_size,
            }).encode()).hexdigest()
        return sha256(f"{base}\n{version}".encode()).hexdigest()

    def prepare_policy(self, policy: DifferentiatedPolicy) -> DifferentiatedPolicy:
        """Validate and lower explicitly enabled physical execution strategies."""
        prepared = optimize_policy(policy, strategy=self.config.physical_fusion)
        if self.config.groupby_prompt_batching:
            from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
            from agent_memory.adapters.lotus.sem_groupby import validate_site_groupby_batching
            found = set()
            roots = [node.query for node in prepared.nodes.values()]
            roots.extend(node.maintenance_query for node in prepared.nodes.values() if node.maintenance_query is not None)
            for root in roots:
                for query in walk(root):
                    if query.op != "sem_groupby":
                        continue
                    site = semantic_pair_site_id(query)
                    if site in self.config.groupby_prompt_batching:
                        validate_site_groupby_batching(query, self.config)
                        found.add(site)
            if found != set(self.config.groupby_prompt_batching):
                raise ValueError("configured groupby batching site not found in executable plan")
        if self.config.listwise_join_batching:
            from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
            found = set()
            pending = [node.query for node in prepared.nodes.values()]
            pending.extend(node.maintenance_query for node in prepared.nodes.values() if node.maintenance_query is not None)
            while pending:
                query = pending.pop()
                if query.op == "fused_target_state":
                    continue
                pending.extend(query.inputs)
                if query.op != "sem_join":
                    continue
                try:
                    site = semantic_pair_site_id(query)
                except ValueError:
                    continue
                if site not in self.config.listwise_join_batching:
                    continue
                profile = self.config.semantic_pair_profiles.get(query_digest(query))
                if not query.params.get("k") or (profile is not None and profile.mode == "proxy-only"):
                    raise ValueError("site listwise batching requires a top-k oracle join")
                found.add(site)
            if found != set(self.config.listwise_join_batching):
                raise ValueError("configured listwise batching site not found in executable plan")
        if self.config.predicate_reuse_sites:
            from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
            from agent_memory.planner.physical import enable_predicate_reuse
            sites: dict[str, set[str]] = {}
            for node_id, node in prepared.nodes.items():
                if node.query.op != "sem_filter":
                    continue
                try:
                    site = semantic_pair_site_id(node.query)
                except ValueError:
                    continue
                sites.setdefault(site, set()).add(node_id)
            if not set(self.config.predicate_reuse_sites) <= sites.keys():
                raise ValueError("configured predicate reuse site not found in plan")
            prepared = enable_predicate_reuse(prepared, {
                node_id for site in self.config.predicate_reuse_sites for node_id in sites[site]
            })
        if self.config.pair_filter_batching:
            from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id, semantic_pair_site_physical_contract
            from agent_memory.policy.schema import output_columns
            found = set()
            queries = [n.query for n in prepared.nodes.values()]
            queries.extend(n.maintenance_query for n in prepared.nodes.values() if n.maintenance_query is not None)
            for root in queries:
                for query in walk(root):
                    if query.op != "sem_filter":
                        continue
                    try:
                        site = semantic_pair_site_id(query)
                    except ValueError:
                        continue
                    setting = self.config.pair_filter_batching.get(site)
                    if setting is None:
                        continue
                    semantic_pair_site_physical_contract(query)
                    columns = output_columns(query.inputs[0])
                    if not set(setting.group_by) <= set(columns) or any(
                        not col.split(":", 1)[0].endswith("_id") or ":later" not in col for col in setting.group_by
                    ):
                        raise ValueError("site batching requires stable later endpoint ID columns")
                    profile = self.config.semantic_pair_profiles.get(query_digest(query))
                    if profile is not None and profile.mode == "proxy-only":
                        raise ValueError("site batching cannot replace proxy-only decisions")
                    found.add(site)
            if found != set(self.config.pair_filter_batching):
                raise ValueError("configured pair-filter batching site not found in plan")
        for node in prepared.nodes.values():
            if node.maintenance_query is None:
                continue
            for query in walk(node.maintenance_query):
                if query.op != "fused_target_state":
                    continue
                if self.config.physical_fusion not in _FUSION_VERSIONS:
                    raise ValueError("fused plan requires matching physical_fusion configuration")
                profile = self.config.semantic_pair_profiles.get(query.params.get("join_profile_digest", query_digest(query.inputs[0])))
                if profile is not None and profile.mode not in {"oracle-only", "search-filter"}:
                    raise ValueError("target-state fusion cannot replace proxy-only decisions")
        self.independent_node_id(prepared)
        return prepared

    def independent_node_id(self, policy: DifferentiatedPolicy) -> str | None:
        """Select only the registered Zep extraction with raw entity inputs."""
        if not self.config.parallel_fact_extraction:
            return None
        from agent_memory.memories.zep.representative import ZepRepresentativeMemory

        template = next(q for q in walk(ZepRepresentativeMemory.facts.expr)
                        if q.op == "sem_flat_map" and "entities" in q.params["input_cols"])
        candidates = [n for n in policy.nodes.values()
                      if n.execution_kind == "semantic_row" and n.query.op == template.op
                      and n.query.params == template.params]
        if len(candidates) != 1:
            raise ValueError("parallel fact extraction requires exactly one registered extraction node")
        node = candidates[0]
        # Check the compiled upstream graph, not just prompt text or output columns.
        reference = ZepRepresentativeMemory.differentiate_policy()
        pending = [node.node_id]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            actual, expected = policy.nodes[current], reference.nodes.get(current)
            if expected is None or actual.query != expected.query or actual.input_node_ids != expected.input_node_ids:
                raise ValueError("parallel fact extraction input contract differs from registered Zep view")
            pending.extend(actual.input_node_ids)
        return node.node_id

    def execute_independent(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute one extraction using private request metadata and usage counters."""
        if not self.config.parallel_fact_extraction or query.op != "sem_flat_map":
            raise ValueError("independent execution is not enabled for this operator")
        if self._independent_context is None:
            self._independent_context = self._context.fork()
        context = self._independent_context
        if context._scoped_lm is None:
            raise RuntimeError("independent LM context was not configured")
        with context._scoped_lm.scope(context._lm), measure_semantic_trace_io() as trace_io, semantic_trace_scope(
            semantic_operator=query.op, operator_call_id=f"{query.op}-{query_digest(query)}-{uuid4().hex[:8]}",
            query_digest=query_digest(query), execution_lane="independent-fact-extraction-v2",
            semantic_trace_snapshot_mode=self.config.semantic_trace_snapshot_mode,
        ):
            try:
                result = execute_sem_flat_map(
                    query, inputs, self.execute, context, model=context._lm,
                )
            except Exception as exc:
                if self.config.lm_enable_cache is True:
                    try:
                        self._write_framework_cache_usage(query,
                            usage=context.consume_cache_usage_delta(), status="error", error=exc)
                    except Exception as trace_error:
                        exc.add_note(f"LOTUS cache usage trace failed: {trace_error}")
                raise
            if self.config.lm_enable_cache is True:
                self._write_framework_cache_usage(query,
                    usage=context.consume_cache_usage_delta(), status="success", output=result)
            # Worker I/O can overlap main-thread work; do not subtract it again
            # from event wall time or mutate the caller's measurement object.
            write_trace_event(self.config.trace_dir(), operator=query.op,
                event_type="independent_execution_result", payload={
                    "trace_io_latency_ms": trace_io.latency_ms,
                    "trace_bytes_written": trace_io.bytes_written,
                })
            return result

    def _maintenance_execution_fingerprint(self) -> str:
        """Keep the pre-fusion default fingerprint byte-for-byte stable."""

        pair_fingerprint = semantic_pair_profiles_fingerprint(
            self.config.semantic_pair_profiles
        )
        context_limit = self.config.lm_model_kwargs.get("max_ctx_len")
        sem_agg_configured = self.config.sem_agg_dispatch != "sequential"
        transport_configured = (
            self.config.structured_output_transport != "chat-json-object"
        )
        if (
            sem_agg_configured
            or self.config.prompt_batching is not None
            or transport_configured
            or context_limit is not None
        ):
            base_fingerprint = self._base_maintenance_execution_fingerprint(
                pair_fingerprint
            )
            parts = [f"base_execution_fingerprint={base_fingerprint}"]
            if context_limit is not None:
                parts.append(f"lm_max_ctx_len={context_limit}")
            if sem_agg_configured:
                parts.append(f"sem_agg_dispatch={self.config.sem_agg_dispatch}")
            if transport_configured:
                parts.append(
                    f"structured_output_transport={self.config.structured_output_transport}"
                )
            if self.config.prompt_batching is not None:
                parts.append(
                    "prompt_batching="
                    f"{self.config.prompt_batching.fingerprint}"
                )
            payload = "\n".join(parts)
            return sha256(payload.encode("utf-8")).hexdigest()
        return self._base_maintenance_execution_fingerprint(pair_fingerprint)

    def _base_maintenance_execution_fingerprint(self, pair_fingerprint: str) -> str:
        """Preserve the pre-prompting physical fingerprint contract."""

        if self.config.sem_join_topk_method == "listwise":
            return pair_fingerprint
        payload = (
            f"semantic_pair_profiles={pair_fingerprint}\n"
            f"sem_join_topk_method={self.config.sem_join_topk_method}"
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute a logical query expression through LOTUS."""
        from agent_memory.adapters.lotus.identity_reuse import IDENTITY_DECISIONS_INPUT, identity_scope
        if IDENTITY_DECISIONS_INPUT in inputs:
            scoped = dict(inputs)
            values = scoped.pop(IDENTITY_DECISIONS_INPUT)
            with identity_scope(values):
                return self._execute(query, scoped)
        return self._execute(query, inputs)

    def _execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Dispatch within the optional executor-owned identity scope."""
        if self.config.physical_fusion in {"zep-fact-summary", "zep-representative"} and query.op in {"agg", "sem_map"}:
            from agent_memory.planner.physical import lower_fact_summary_node
            query = lower_fact_summary_node(query)

        match query.op:
            case "fused_target_state":
                if self.config.physical_fusion not in _FUSION_VERSIONS:
                    raise ValueError("fused plan requires matching physical_fusion configuration")
                version = _FUSION_VERSIONS[self.config.physical_fusion]
                if query.params.get("version") != version:
                    raise ValueError("unsupported target-state fusion version")
                return self._execute_traced_semantic(query, inputs, execute_target_state)
            case "log":
                return execute_log(inputs)
            case "materialized_view":
                return execute_materialized_view(query, inputs)
            case "window_source":
                return execute_window_source(inputs)
            case "select":
                return self._execute_traced_relational(query, inputs, execute_select)
            case "limit":
                return self._execute_traced_relational(query, inputs, execute_limit)
            case "alias":
                return self._execute_traced_relational(query, inputs, execute_alias)
            case "assign":
                return self._execute_traced_relational(query, inputs, execute_assign)
            case "filter":
                return self._execute_traced_relational(query, inputs, execute_filter)
            case "group_by":
                return self._execute_traced_relational(query, inputs, execute_group_by)
            case "let":
                return self._execute_traced_relational(query, inputs, execute_let)
            case "array_agg":
                return self._execute_traced_relational(query, inputs, execute_array_agg)
            case "array_cat":
                return self._execute_traced_relational(query, inputs, execute_array_cat)
            case "min":
                return self._execute_traced_relational(query, inputs, execute_min)
            case "flatten":
                return self._execute_traced_relational(query, inputs, execute_flatten)
            case "explode":
                return self._execute_traced_relational(query, inputs, execute_explode)
            case "unnest":
                return self._execute_traced_relational(query, inputs, execute_unnest)
            case "agg":
                if is_algebraic_aggregate_query(query):
                    return self._execute_traced_relational(
                        query, inputs, execute_algebraic_aggregate
                    )
                return self._execute_traced_semantic(query, inputs, execute_agg)
            case "aggregate_state":
                return self._execute_traced_relational(
                    query, inputs, execute_aggregate_state
                )
            case "aggregate_state_update":
                return self._execute_traced_relational(
                    query, inputs, execute_aggregate_state_update
                )
            case "aggregate_finalize":
                return self._execute_traced_relational(
                    query, inputs, execute_aggregate_finalize
                )
            case "concat":
                return self._execute_traced_relational(query, inputs, execute_concat)
            case "union":
                return self._execute_traced_relational(query, inputs, execute_union)
            case "union_by_name":
                return self._execute_traced_relational(query, inputs, execute_union_by_name)
            case "subtract":
                return self._execute_traced_relational(query, inputs, execute_subtract)
            case "join":
                return self._execute_traced_relational(query, inputs, execute_join)
            case "drop_duplicates":
                return self._execute_traced_relational(query, inputs, execute_drop_duplicates)
            case "count_window":
                return self._execute_traced_relational(query, inputs, execute_count_window)
            case "process_window":
                return self._execute_traced_relational(query, inputs, execute_process_window)
            case "sem_filter":
                return self._execute_traced_semantic(query, inputs, execute_sem_filter)
            case "sem_flat_map":
                return self._execute_traced_semantic(query, inputs, execute_sem_flat_map)
            case "sem_join":
                return self._execute_traced_semantic(query, inputs, execute_sem_join)
            case "sem_groupby":
                return self._execute_traced_semantic(query, inputs, execute_sem_groupby)
            case "sem_agg":
                return self._execute_traced_semantic(query, inputs, execute_sem_agg)
            case "sem_map":
                if query.params.get("fact_summary_map"):
                    from agent_memory.adapters.lotus.relational import execute_fact_summary_map
                    return self._execute_traced_semantic(query, inputs, execute_fact_summary_map)
                return self._execute_traced_semantic(query, inputs, execute_sem_map)
            case "sem_topk":
                return self._execute_traced_semantic(query, inputs, execute_sem_topk)
            case _:
                raise NotImplementedError(
                    f"LOTUS adapter does not support QueryExpr op {query.op!r}."
                )

    def _execute_traced_relational(
        self,
        query: QueryExpr,
        inputs: Mapping[str, Any],
        executor: Any,
    ) -> Any:
        """Execute a deterministic relational op and emit compact trace metadata."""

        with semantic_trace_scope(
            semantic_trace_snapshot_mode=self.config.semantic_trace_snapshot_mode,
        ):
            result = executor(query, inputs, self.execute)
            write_compact_operator_trace(
                self.config.trace_dir(),
                operator=query.op,
                event_type="operator_result",
                output_frame=result,
                payload={
                    "query_digest": query_digest(query),
                    "params": dict(query.params),
                    "input_count": len(query.inputs),
                },
            )
        return result

    def _execute_traced_semantic(
        self,
        query: QueryExpr,
        inputs: Mapping[str, Any],
        executor: Any,
    ) -> Any:
        """Execute a semantic op under a trace scope shared with LLM calls."""

        digest = query_digest(query)
        operator_call_id = f"{query.op}-{digest}-{uuid4().hex[:8]}"
        with semantic_trace_scope(
            semantic_operator=query.op,
            operator_call_id=operator_call_id,
            query_digest=digest,
            semantic_trace_snapshot_mode=self.config.semantic_trace_snapshot_mode,
        ):
            cache_enabled = self.config.lm_enable_cache is True
            if cache_enabled:
                self._context.configure()
            try:
                result = executor(query, inputs, self.execute, self._context)
            except BaseException as error:
                if cache_enabled:
                    try:
                        self._write_framework_cache_usage(
                            query,
                            usage=self._context.consume_cache_usage_delta(),
                            status="error",
                            error=error,
                        )
                    except Exception as trace_error:
                        error.add_note(
                            "LOTUS cache usage trace failed: "
                            f"{type(trace_error).__name__}: {trace_error}"
                        )
                raise
            if cache_enabled:
                self._write_framework_cache_usage(
                    query,
                    usage=self._context.consume_cache_usage_delta(),
                    status="success",
                    output=result,
                )
            return result

    def _write_framework_cache_usage(
        self,
        query: QueryExpr,
        *,
        usage: Mapping[str, int],
        status: str,
        output: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Record one cache delta without changing provider accounting."""

        payload: dict[str, Any] = {
            "query_digest": query_digest(query),
            "cache_mode": LOTUS_MEMORY_CACHE_ID,
            "status": status,
            **usage,
        }
        output_frame = output[0] if isinstance(output, tuple) and output else output
        if isinstance(output_frame, Sized):
            payload["output_row_count"] = len(output_frame)
        if error is not None:
            payload.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        write_trace_event(
            self.config.trace_dir(),
            operator=query.op,
            event_type="framework_cache_usage",
            payload=payload,
        )
