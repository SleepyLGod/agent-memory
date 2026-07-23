"""Real provider boundary shared by benchmark answerers and judges."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from .harness import ModelPrompt, ModelResponse


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError(f"provider {name} must be mapping-like")


def _usage(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    usage = _mapping(value, name="usage")
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping):
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            usage["reasoning_tokens"] = reasoning_tokens
    return usage


class LiteLLMBenchmarkModel:
    """Call one LiteLLM-compatible model without application-level caching."""

    def __init__(
        self,
        *,
        model_id: str = "deepseek/deepseek-v4-flash",
        completion: Callable[..., Any] | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("benchmark model_id must be non-empty")
        if completion is None:
            from litellm import completion as litellm_completion

            completion = litellm_completion
        self.model_id = model_id
        self._completion = completion

    def complete(self, prompt: ModelPrompt, *, attempt: int) -> ModelResponse:
        """Execute one attempt and preserve the provider's raw response evidence."""

        del attempt
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [dict(message) for message in prompt.messages],
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_tokens,
            "caching": False,
            "num_retries": 0,
        }
        kwargs["extra_body"] = {
            "thinking": {
                "type": "enabled" if prompt.thinking_enabled else "disabled"
            }
        }
        started = perf_counter()
        response = self._completion(**kwargs)
        latency_ms = (perf_counter() - started) * 1000
        raw = _mapping(response, name="response")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("provider response requires at least one choice")
        choice = _mapping(choices[0], name="choice")
        message = _mapping(choice.get("message"), name="message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("provider response requires text content")
        model = raw.get("model")
        return ModelResponse(
            model=model if isinstance(model, str) and model else self.model_id,
            text=content,
            raw_response=raw,
            usage=_usage(raw.get("usage")),
            latency_ms=latency_ms,
        )

    @staticmethod
    def is_retryable_error(error: BaseException) -> bool:
        """Return whether one physical provider failure is transient."""

        from litellm.exceptions import (
            APIConnectionError,
            BadGatewayError,
            InternalServerError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        )

        return isinstance(
            error,
            (
                APIConnectionError,
                BadGatewayError,
                InternalServerError,
                RateLimitError,
                ServiceUnavailableError,
                Timeout,
            ),
        )


__all__ = ["LiteLLMBenchmarkModel"]
