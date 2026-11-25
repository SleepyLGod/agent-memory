"""Helpers for LOTUS LM subclasses that trace raw provider usage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from agent_memory.tracing.semantic import normalized_provider_usage, write_provider_usage_trace

LOGGER = logging.getLogger(__name__)

SAFE_PROVIDER_KWARGS = {
    "max_tokens",
    "num_retries",
    "response_format",
    "stream",
    "temperature",
    "timeout",
    "top_p",
}


@dataclass
class ProviderCallUsage:
    """Actual uncached response usage and confirmed local cache hits for one call."""

    completion_tokens: list[int | None] = field(default_factory=list)
    cache_hits: int = 0


class ProviderUsageTracingMixin:
    """Mixin that preserves provider usage payloads in trace artifacts."""

    def __init__(
        self,
        *args: Any,
        trace_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        self._provider_usage_trace_dir = trace_dir
        self._completion_metadata: list[dict[str, Any]] = []
        self.last_call_usage = ProviderCallUsage()
        self._call_prompt_count: int | None = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Keep completion metadata aligned with outputs, including cache hits."""

        self._completion_metadata = []
        self.last_call_usage = ProviderCallUsage()
        self._call_prompt_count = len(args[0] if args else kwargs["messages"])
        parent: Any = super()
        try:
            output = parent.__call__(*args, **kwargs)
        finally:
            self._call_prompt_count = None
        output.response_metadata = tuple(self._completion_metadata)
        return output

    def _get_top_choice(self, response: Any) -> str:
        """Preserve the finish state discarded by LOTUS's text-only LMOutput."""

        parent: Any = super()
        text = parent._get_top_choice(response)
        metadata = getattr(response, "provider_response_metadata", {})
        choices = getattr(response, "choices", ())
        self._completion_metadata.append(
            {
                "finish_reason": metadata.get(
                    "finish_reason",
                    getattr(choices[0], "finish_reason", None) if choices else None,
                ),
                "status": metadata.get("status"),
                "incomplete_reason": metadata.get("incomplete_reason"),
            }
        )
        return text

    def _process_uncached_messages(
        self,
        uncached_data: list[tuple[list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        show_progress_bar: bool,
        progress_bar_desc: str,
    ) -> list[Any]:
        """Call LOTUS normally, then trace raw provider usage from responses."""

        parent: Any = super()
        if self._call_prompt_count is not None:
            self.last_call_usage.cache_hits = self._call_prompt_count - len(uncached_data)
        offset = len(self.last_call_usage.completion_tokens)
        self.last_call_usage.completion_tokens.extend([None] * len(uncached_data))
        responses = parent._process_uncached_messages(
            uncached_data,
            all_kwargs,
            show_progress_bar,
            progress_bar_desc,
        )
        for index, response in enumerate(responses):
            tokens = normalized_provider_usage(getattr(response, "usage", None))[
                "provider_completion_tokens"
            ]
            if type(tokens) is int:
                self.last_call_usage.completion_tokens[offset + index] = tokens
        try:
            write_provider_usage_trace(
                self._provider_usage_trace_dir,
                model=str(getattr(self, "model", "")),
                responses=responses,
                request_metadata={
                    "provider_batch_size": len(uncached_data),
                    "provider_kwargs": _safe_provider_kwargs(all_kwargs),
                },
            )
        except Exception as error:  # pragma: no cover - trace failures are non-semantic.
            LOGGER.warning("Failed to write provider usage trace: %s", error)
        return responses


def _safe_provider_kwargs(all_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-sensitive request tuning kwargs for trace metadata."""

    safe = {key: all_kwargs[key] for key in SAFE_PROVIDER_KWARGS if key in all_kwargs}
    extra_body = all_kwargs.get("extra_body")
    if isinstance(extra_body, Mapping):
        thinking = extra_body.get("thinking")
        if isinstance(thinking, Mapping) and isinstance(thinking.get("type"), str):
            safe["thinking"] = {"type": thinking["type"]}
    return safe


def provider_usage_tracing_lm_class(base_lm_class: type[Any]) -> type[Any]:
    """Return a trace-enabled subclass of the active LOTUS LM class."""

    class ProviderUsageTracingLM(ProviderUsageTracingMixin, base_lm_class):
        pass

    return ProviderUsageTracingLM
