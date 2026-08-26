"""Shared LOTUS execution context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_memory.adapters.lotus.pair_execution import (
    SemanticPairExecutionProfile,
)
from agent_memory.adapters.lotus.provider_usage_lm import provider_usage_tracing_lm_class
from agent_memory.adapters.lotus.traced_lm import TracedLM
from agent_memory.storage.embedding import EmbeddingProvider

DEFAULT_STRUCTURED_MAX_TOKENS = 8192
DEFAULT_STRUCTURED_PARSE_RETRIES = 3
LOTUS_MEMORY_CACHE_MAX_SIZE = 1024
LOTUS_MEMORY_CACHE_ID = f"lotus-memory:{LOTUS_MEMORY_CACHE_MAX_SIZE}"
SEM_TOPK_METHODS = (
    "pairwise-naive",
    "pairwise-quick",
    "pairwise-heap",
    "listwise",
)
_LM_OWNED_KWARGS = {
    "cache",
    "max_batch_size",
    "model",
    "num_retries",
    "rate_limit",
    "timeout",
}


@dataclass(frozen=True)
class LotusExecutionConfig:
    """Backend execution knobs that are not part of policy query semantics."""

    lm_num_retries: int | None = None
    lm_timeout: float | int | None = None
    lm_max_batch_size: int = 64
    lm_rate_limit: int | None = None
    lm_model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    lm_enable_cache: bool | None = None
    structured_max_tokens: int = DEFAULT_STRUCTURED_MAX_TOKENS
    structured_parse_retries: int = DEFAULT_STRUCTURED_PARSE_RETRIES
    semantic_trace_dir: Path | str | None = None
    semantic_trace_snapshot_mode: str = "compact"
    semantic_pair_profiles: Mapping[str, SemanticPairExecutionProfile] = field(
        default_factory=dict
    )

    sem_filter_examples: Sequence[Mapping[str, Any]] | None = None
    sem_filter_helper_examples: Sequence[Mapping[str, Any]] | None = None
    sem_filter_strategy: Any | None = None
    sem_filter_default: bool = True
    sem_filter_cascade_args: Mapping[str, Any] | None = None
    sem_filter_safe_mode: bool = False
    sem_filter_progress_bar_desc: str = "Filtering"
    sem_filter_additional_cot_instructions: str = ""

    sem_topk_method: str = "pairwise-naive"
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
    sem_groupby_pair_batch_size: int | None = None
    sem_groupby_pair_batch_retries: int = 0

    sem_agg_safe_mode: bool = False
    sem_agg_progress_bar_desc: str = "Aggregating"
    sem_agg_model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate bounded semantic execution settings."""

        if (
            self.sem_groupby_pair_batch_size is not None
            and self.sem_groupby_pair_batch_size < 1
        ):
            raise ValueError("sem_groupby_pair_batch_size must be positive")
        if self.sem_groupby_pair_batch_retries < 0:
            raise ValueError("sem_groupby_pair_batch_retries cannot be negative")
        if self.semantic_trace_snapshot_mode not in {"compact", "full"}:
            raise ValueError(
                "semantic_trace_snapshot_mode must be 'compact' or 'full'"
            )
        for query_digest, profile in self.semantic_pair_profiles.items():
            if not isinstance(query_digest, str) or not query_digest:
                raise ValueError("semantic pair profile keys must be query digests")
            if not isinstance(profile, SemanticPairExecutionProfile):
                raise TypeError(
                    "semantic pair profile values must be "
                    "SemanticPairExecutionProfile"
                )

    def trace_dir(self) -> Path | str | None:
        """Return the configured semantic trace directory."""

        return self.semantic_trace_dir


@dataclass
class LotusExecutionContext:
    """Configure LOTUS once for a selected LiteLLM-compatible model."""

    model: str
    config: LotusExecutionConfig = field(default_factory=LotusExecutionConfig)
    pair_embedding_provider: EmbeddingProvider | None = None
    _configured: bool = field(default=False, init=False, repr=False)
    _lm: Any | None = field(default=None, init=False, repr=False)
    _reported_cache_usage: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def configure(self) -> None:
        """Configure LOTUS before invoking semantic dataframe operators."""

        if self._configured:
            return

        import lotus
        from lotus.cache import InMemoryCache
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
        conflicting = sorted(_LM_OWNED_KWARGS & self.config.lm_model_kwargs.keys())
        if conflicting:
            raise ValueError(
                "lm_model_kwargs cannot override execution-owned options: "
                + ", ".join(conflicting)
            )
        lm_kwargs.update(self.config.lm_model_kwargs)
        if self.config.lm_enable_cache is True:
            lm_kwargs["cache"] = InMemoryCache(
                max_size=LOTUS_MEMORY_CACHE_MAX_SIZE
            )

        trace_dir = self.config.trace_dir()
        base_lm = (
            provider_usage_tracing_lm_class(LM)(**lm_kwargs, trace_dir=trace_dir)
            if trace_dir is not None
            else LM(**lm_kwargs)
        )
        lm = TracedLM(base_lm, trace_dir) if trace_dir is not None else base_lm
        self._lm = lm
        settings_kwargs: dict[str, Any] = {"lm": lm}
        if self.config.lm_enable_cache is not None:
            settings_kwargs["enable_cache"] = self.config.lm_enable_cache
        lotus.settings.configure(**settings_kwargs)
        self._configured = True

    def cache_usage_snapshot(self) -> dict[str, int]:
        """Return physical, virtual, and framework-cache counters for this LM."""

        stats = getattr(self._lm, "stats", None)
        if stats is None:
            return {
                "lm_cache_hits": 0,
                "operator_cache_hits": 0,
                "physical_prompt_tokens": 0,
                "physical_completion_tokens": 0,
                "physical_total_tokens": 0,
                "virtual_prompt_tokens": 0,
                "virtual_completion_tokens": 0,
                "virtual_total_tokens": 0,
            }
        return {
            "lm_cache_hits": int(getattr(stats, "cache_hits", 0)),
            "operator_cache_hits": int(
                getattr(stats, "operator_cache_hits", 0)
            ),
            "physical_prompt_tokens": int(stats.physical_usage.prompt_tokens),
            "physical_completion_tokens": int(
                stats.physical_usage.completion_tokens
            ),
            "physical_total_tokens": int(stats.physical_usage.total_tokens),
            "virtual_prompt_tokens": int(stats.virtual_usage.prompt_tokens),
            "virtual_completion_tokens": int(stats.virtual_usage.completion_tokens),
            "virtual_total_tokens": int(stats.virtual_usage.total_tokens),
        }

    def consume_cache_usage_delta(self) -> dict[str, int]:
        """Return counters not already attributed to an earlier trace event."""

        current = self.cache_usage_snapshot()
        delta = {
            key: value - self._reported_cache_usage.get(key, 0)
            for key, value in current.items()
        }
        decreased = {key: value for key, value in delta.items() if value < 0}
        if decreased:
            raise RuntimeError(f"LOTUS cache counters decreased: {decreased}")
        self._reported_cache_usage = current
        return delta
