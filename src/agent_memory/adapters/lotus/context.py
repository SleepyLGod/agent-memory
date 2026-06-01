"""Shared LOTUS execution context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LotusExecutionConfig:
    """Backend execution knobs that are not part of policy query semantics."""

    sem_filter_examples: Sequence[Mapping[str, Any]] | None = None
    sem_filter_helper_examples: Sequence[Mapping[str, Any]] | None = None
    sem_filter_strategy: Any | None = None
    sem_filter_default: bool = True
    sem_filter_cascade_args: Mapping[str, Any] | None = None
    sem_filter_safe_mode: bool = False
    sem_filter_progress_bar_desc: str = "Filtering"
    sem_filter_additional_cot_instructions: str = ""

    sem_topk_method: str = "naive"
    sem_topk_strategy: Any | None = None
    sem_topk_cascade_threshold: float | None = None
    sem_topk_return_stats: bool = False
    sem_topk_safe_mode: bool = False
    sem_topk_return_explanations: bool = False

    sem_map_system_prompt: str | None = None
    sem_map_examples: Sequence[Mapping[str, Any]] | None = None
    sem_map_strategy: Any | None = None
    sem_map_safe_mode: bool = False
    sem_map_progress_bar_desc: str = "Mapping"
    sem_map_model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    sem_join_examples: Sequence[Mapping[str, Any]] | None = None
    sem_join_strategy: Any | None = None
    sem_join_default: bool = True
    sem_join_cascade_args: Mapping[str, Any] | None = None
    sem_join_safe_mode: bool = False
    sem_join_progress_bar_desc: str = "Join comparisons"

    sem_agg_structured_strategy: str = "single_batch"
    sem_agg_safe_mode: bool = False
    sem_agg_progress_bar_desc: str = "Aggregating"
    sem_agg_model_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LotusExecutionContext:
    """Configure LOTUS once for a selected LiteLLM-compatible model."""

    model: str
    config: LotusExecutionConfig = field(default_factory=LotusExecutionConfig)
    _configured: bool = field(default=False, init=False, repr=False)

    def configure(self) -> None:
        """Configure LOTUS before invoking semantic dataframe operators."""

        if self._configured:
            return

        import lotus
        from lotus.models import LM

        lotus.settings.configure(lm=LM(model=self.model))
        self._configured = True
