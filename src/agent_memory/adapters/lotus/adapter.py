"""LOTUS execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.relational import (
    execute_array_agg,
    execute_array_cat,
    execute_concat,
    execute_drop_duplicates,
    execute_select,
    execute_subtract,
    execute_union,
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
)
from agent_memory.adapters.lotus.sources import execute_log, execute_materialized_view
from agent_memory.adapters.lotus.window import (
    execute_count_window,
    execute_process_window,
    execute_window_source,
)
from agent_memory.logical import QueryExpr

DEFAULT_LOTUS_MODEL = "deepseek/deepseek-v4-pro"


@dataclass
class LotusAdapter:
    """Execution adapter for LOTUS-backed semantic operators."""

    model: str = DEFAULT_LOTUS_MODEL
    config: LotusExecutionConfig = field(default_factory=LotusExecutionConfig)
    _context: LotusExecutionContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize shared LOTUS execution context."""

        self._context = LotusExecutionContext(model=self.model, config=self.config)

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
            case "array_agg":
                return self._execute_traced_relational(query, inputs, execute_array_agg)
            case "array_cat":
                return self._execute_traced_relational(query, inputs, execute_array_cat)
            case "concat":
                return self._execute_traced_relational(query, inputs, execute_concat)
            case "union":
                return self._execute_traced_relational(query, inputs, execute_union)
            case "subtract":
                return self._execute_traced_relational(query, inputs, execute_subtract)
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
        ):
            return executor(query, inputs, self.execute, self._context)
