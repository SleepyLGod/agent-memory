"""DocETL execution adapter: the runtime's logical query expressions, run on DocETL."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from agent_memory.adapters.docetl.context import (
    DocetlExecutionConfig,
    DocetlExecutionContext,
    default_docetl_model,
)
from agent_memory.adapters.docetl.sem_filter import execute_sem_filter
from agent_memory.adapters.docetl.sources import execute_log, execute_materialized_view
from agent_memory.adapters.docetl.window import (
    execute_count_window,
    execute_process_window,
    execute_window_source,
)
from agent_memory.policy.logical import QueryExpr
from agent_memory.storage.embedding import EmbeddingProvider
from agent_memory.tracing.semantic import (
    query_digest,
    semantic_trace_scope,
    write_compact_operator_trace,
)

Handler = Callable[[QueryExpr, Mapping[str, Any]], Any]


@dataclass
class DocetlAdapter:
    """Execution adapter that lowers each operator onto a DocETL operation class."""

    model: str = field(default_factory=default_docetl_model)
    config: DocetlExecutionConfig = field(default_factory=DocetlExecutionConfig)
    pair_embedding_provider: EmbeddingProvider | None = None
    _context: DocetlExecutionContext = field(init=False, repr=False)
    _handlers: dict[str, Handler] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the (lazy) engine context and the operator registry."""

        self._context = DocetlExecutionContext(model=self.model, config=self.config)
        self._handlers = self._build_handlers()

    @property
    def maintenance_execution_fingerprint(self) -> str:
        """Identify the physical settings that can change maintained state.

        Invariants: non-empty, and **backend-distinct** so a LOTUS-produced checkpoint can
        never be restored under DocETL. `backend=docetl` is what guarantees that.
        """

        payload = "\n".join(
            (
                "backend=docetl",
                f"model={self.model}",
                f"skip_on_error={self.config.skip_on_error}",
                f"enable_cache={self.config.enable_cache}",
                f"lm_api_base={self.config.lm_api_base or ''}",
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Dispatch to the registered lowering; the frame contract is `ExecutionAdapter.execute`'s."""

        handler = self._handlers.get(query.op)
        if handler is None:
            raise NotImplementedError(
                f"DocETL adapter does not implement QueryExpr op {query.op!r} yet "
                f"(supported: {sorted(self._handlers)})."
            )
        return self._execute_traced(query, inputs, handler)

    def supported_ops(self) -> frozenset[str]:
        """Return the operators this adapter can currently execute.

        Introspected from the dispatch registry rather than declared separately, so it
        cannot drift from the code.
        """

        return frozenset(self._handlers)

    def _build_handlers(self) -> dict[str, Handler]:
        """Return the operator registry.

        The backend-neutral operators are duplicated from `adapters/lotus/` rather than shared,
        so the two adapters can change without touching each other. Engine-backed operators are
        registered one at a time as their lowering lands, so an operator missing here raises
        rather than returns something unvalidated.
        """

        return {
            "log": lambda query, inputs: execute_log(inputs),
            "materialized_view": execute_materialized_view,
            "window_source": lambda query, inputs: execute_window_source(inputs),
            "count_window": lambda query, inputs: execute_count_window(
                query, inputs, self.execute
            ),
            "process_window": lambda query, inputs: execute_process_window(
                query, inputs, self.execute
            ),
            "sem_filter": lambda query, inputs: execute_sem_filter(
                query, inputs, self.execute, self._context
            ),
        }

    def _execute_traced(
        self,
        query: QueryExpr,
        inputs: Mapping[str, Any],
        handler: Handler,
    ) -> Any:
        """Execute one operator and emit its trace event."""

        with semantic_trace_scope(
            semantic_trace_snapshot_mode=self.config.semantic_trace_snapshot_mode,
        ):
            result = handler(query, inputs)
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
