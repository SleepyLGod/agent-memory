"""LOTUS execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.relational import (
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
from agent_memory.adapters.lotus.sources import execute_log, execute_materialized_view
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
            case "select":
                return execute_select(query, inputs, self.execute)
            case "concat":
                return execute_concat(query, inputs, self.execute)
            case "union":
                return execute_union(query, inputs, self.execute)
            case "subtract":
                return execute_subtract(query, inputs, self.execute)
            case "drop_duplicates":
                return execute_drop_duplicates(query, inputs, self.execute)
            case "sem_filter":
                return execute_sem_filter(query, inputs, self.execute, self._context)
            case "sem_flat_map":
                return execute_sem_flat_map(query, inputs, self.execute, self._context)
            case "sem_join":
                return execute_sem_join(query, inputs, self.execute, self._context)
            case "sem_groupby":
                return execute_sem_groupby(query, inputs, self.execute, self._context)
            case "sem_agg":
                return execute_sem_agg(query, inputs, self.execute, self._context)
            case "sem_map":
                return execute_sem_map(query, inputs, self.execute, self._context)
            case "sem_topk":
                return execute_sem_topk(query, inputs, self.execute, self._context)
            case _:
                raise NotImplementedError(
                    f"LOTUS adapter does not support QueryExpr op {query.op!r}."
                )
