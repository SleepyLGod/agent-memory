"""Shared LOTUS execution context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_memory.adapters.lotus.provider_usage_lm import provider_usage_tracing_lm_class
from agent_memory.adapters.lotus.traced_lm import TracedLM

DEFAULT_STRUCTURED_MAX_TOKENS = 8192
DEFAULT_STRUCTURED_PARSE_RETRIES = 3


@dataclass(frozen=True)
class LotusExecutionConfig:
    """Backend execution knobs that are not part of policy query semantics."""

    lm_num_retries: int | None = None
    lm_timeout: float | int | None = None
    lm_max_batch_size: int = 64
    lm_rate_limit: int | None = None
    structured_max_tokens: int = DEFAULT_STRUCTURED_MAX_TOKENS
    structured_parse_retries: int = DEFAULT_STRUCTURED_PARSE_RETRIES
    semantic_trace_dir: Path | str | None = None

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
    sem_join_default: bool = False
    sem_join_cascade_args: Mapping[str, Any] | None = None
    sem_join_safe_mode: bool = False
    sem_join_progress_bar_desc: str = "Join comparisons"

    sem_groupby_default: bool = False

    sem_agg_safe_mode: bool = False
    sem_agg_progress_bar_desc: str = "Aggregating"
    sem_agg_model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def trace_dir(self) -> Path | str | None:
        """Return the configured semantic trace directory."""

        return self.semantic_trace_dir


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

        lm_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_batch_size": self.config.lm_max_batch_size,
        }
        if self.config.lm_num_retries is not None:
            lm_kwargs["num_retries"] = self.config.lm_num_retries
        if self.config.lm_timeout is not None:
            lm_kwargs["timeout"] = self.config.lm_timeout
        if self.config.lm_rate_limit is not None:
            lm_kwargs["rate_limit"] = self.config.lm_rate_limit

        trace_dir = self.config.trace_dir()
        base_lm = (
            provider_usage_tracing_lm_class(LM)(**lm_kwargs, trace_dir=trace_dir)
            if trace_dir is not None
            else LM(**lm_kwargs)
        )
        lm = TracedLM(base_lm, trace_dir) if trace_dir is not None else base_lm
        lotus.settings.configure(lm=lm)
        self._configured = True
