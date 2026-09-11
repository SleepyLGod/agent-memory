"""LOTUS execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
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

DEFAULT_LOTUS_MODEL = "deepseek/deepseek-v4-pro"


@dataclass
class LotusAdapter:
    """Execution adapter for LOTUS-backed semantic operators."""

    model: str = DEFAULT_LOTUS_MODEL
    config: LotusExecutionConfig = field(default_factory=LotusExecutionConfig)
    pair_embedding_provider: EmbeddingProvider | None = None
    _context: LotusExecutionContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize shared LOTUS execution context."""

        self._context = LotusExecutionContext(
            model=self.model,
            config=self.config,
            pair_embedding_provider=self.pair_embedding_provider,
        )

    @property
    def maintenance_execution_fingerprint(self) -> str:
        """Identify physical settings that can change maintained state."""

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

        match query.op:
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
        if hasattr(output_frame, "__len__"):
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
