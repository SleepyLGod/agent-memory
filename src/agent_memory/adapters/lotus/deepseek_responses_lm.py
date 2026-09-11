"""Native DeepSeek Responses transport for LOTUS JSON-schema requests.

Wrap this factory's result with provider_usage_tracing_lm_class, not vice versa.
The uncached hook returns ordered ModelResponse/OpenAIError items before LOTUS
accounts usage. Every item carries provider_response_metadata with transport,
status, incomplete_reason, finish_reason, and raw_usage. Native response failures
also expose response (the normalized partial ModelResponse), native_response
(the original SDK Response), model, and usage when reported. Normalization
errors retain those fields with finish_reason="normalization_error" and the
original exception as their cause; SDK errors retain their type and identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import math
import os
import time
from typing import Any

from litellm import ModelResponse
from litellm.types.utils import Usage
from openai import OpenAI, OpenAIError
from openai.types.responses import Response
from tqdm import tqdm

__all__ = ["deepseek_responses_lm_class", "validate_deepseek_responses_model"]

_MODELS = {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro"}
_PARAMS = {
    "response_format",
    "max_tokens",
    "temperature",
    "top_p",
    "timeout",
    "num_retries",
    "extra_body",
    "reasoning_effort",
    "stream",
    "api_key",
    "api_base",
    "user",
}
_EFFORTS = {
    "none": "none",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "high",
    "max": "max",
}
_TRANSPORT = "responses-json-schema"


def validate_deepseek_responses_model(model: str) -> None:
    """Reject non-native providers and unsupported native DeepSeek model names."""
    if not isinstance(model, str) or model.removeprefix("deepseek/") not in _MODELS:
        raise ValueError(
            f"Unsupported DeepSeek Responses model: {model!r}; use "
            f"{', '.join(sorted(_MODELS))}, optionally prefixed with deepseek/"
        )


def _reasoning_effort(kwargs: dict[str, Any]) -> str:
    extra = kwargs.get("extra_body", {})
    if not isinstance(extra, Mapping) or extra.keys() - {"thinking"}:
        raise ValueError("Responses extra_body supports only thinking.type")
    thinking = extra.get("thinking", {})
    if not isinstance(thinking, Mapping) or thinking.keys() - {"type"}:
        raise ValueError("Responses thinking supports only type")
    mode = thinking.get("type")
    if "thinking" in extra and mode not in {"enabled", "disabled"}:
        raise ValueError("thinking.type must be enabled or disabled")
    effort = kwargs.get("reasoning_effort", "none" if mode == "disabled" else "high")
    if not isinstance(effort, str) or effort not in _EFFORTS:
        raise ValueError("Unsupported reasoning_effort for DeepSeek Responses")
    if (mode == "disabled" and effort != "none") or (
        mode == "enabled" and effort == "none"
    ):
        raise ValueError("thinking.type conflicts with reasoning_effort")
    return _EFFORTS[effort]


def _request_options(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    unsupported = kwargs.keys() - _PARAMS
    if unsupported:
        raise ValueError(
            f"Unsupported DeepSeek Responses parameters: {', '.join(sorted(unsupported))}"
        )
    if kwargs.get("stream", False) is not False:
        raise ValueError("DeepSeek Responses adapter supports only stream=False")
    response_format = kwargs["response_format"]
    schema = response_format.get("json_schema")
    if (
        response_format.keys() - {"type", "json_schema"}
        or not isinstance(schema, Mapping)
        or schema.keys() - {"name", "schema", "strict", "description"}
        or not isinstance(schema.get("name"), str)
        or not schema["name"]
        or not isinstance(schema.get("schema"), Mapping)
    ):
        raise ValueError(
            "response_format requires json_schema.name and json_schema.schema"
        )
    if "strict" in schema and not isinstance(schema["strict"], bool):
        raise ValueError("json_schema.strict must be a boolean")
    if "description" in schema and not isinstance(schema["description"], str):
        raise ValueError("json_schema.description must be a string")

    retries = kwargs.get("num_retries", 2)
    if type(retries) is not int or not 0 <= retries <= 10:
        raise ValueError("num_retries must be an integer between 0 and 10")
    payload: dict[str, Any] = {
        "text": {"format": {"type": "json_schema", **schema}},
        "reasoning": {"effort": _reasoning_effort(kwargs)},
        "stream": False,
    }
    if "max_tokens" in kwargs:
        tokens = kwargs["max_tokens"]
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        payload["max_output_tokens"] = tokens
    for key, maximum in (("temperature", 2), ("top_p", 1), ("timeout", None)):
        if key not in kwargs:
            continue
        value = kwargs[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (value <= 0 if key == "timeout" else value < 0)
            or (maximum is not None and value > maximum)
        ):
            raise ValueError(f"Unsupported {key} value for DeepSeek Responses")
        if key != "timeout":
            payload[key] = value
    if "user" in kwargs:
        if not isinstance(kwargs["user"], str):
            raise ValueError("user must be a string")
        payload["user"] = kwargs["user"]
    api_key = kwargs.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Set DEEPSEEK_API_KEY or pass api_key for DeepSeek Responses")
    base_url = kwargs.get("api_base", "https://api.deepseek.com")
    if not isinstance(base_url, str) or not base_url.startswith(
        ("https://", "http://")
    ):
        raise ValueError("api_base must be an HTTP(S) base URL")
    client_options = {"api_key": api_key, "base_url": base_url, "max_retries": retries}
    if "timeout" in kwargs:
        client_options["timeout"] = kwargs["timeout"]
    return payload, client_options


def _normalize_usage(raw: dict[str, Any] | None) -> Usage | None:
    if raw is None:
        return None
    details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    kwargs: dict[str, Any] = {
        "prompt_tokens": raw["input_tokens"],
        "completion_tokens": raw["output_tokens"],
        "total_tokens": raw["total_tokens"],
    }
    cached = details.get("cached_tokens")
    if cached is not None:
        kwargs.update(
            prompt_cache_hit_tokens=cached,
            prompt_cache_miss_tokens=raw["input_tokens"] - cached,
            prompt_tokens_details={"cached_tokens": cached},
        )
    if output_details.get("reasoning_tokens") is not None:
        kwargs["completion_tokens_details"] = {
            "reasoning_tokens": output_details["reasoning_tokens"]
        }
    return Usage(**kwargs)


class _ResponseFailure(OpenAIError):
    """A non-success native response that still incurred provider usage."""

    def __init__(
        self, response: ModelResponse, detail: Any, native_response: Response
    ) -> None:
        dynamic_response: Any = response
        metadata: dict[str, Any] = dynamic_response.provider_response_metadata
        super().__init__(f"DeepSeek Responses {metadata['finish_reason']}: {detail}")
        self.response = response
        self.native_response = native_response
        self.model = response.model
        self.provider_response_metadata = metadata
        if hasattr(response, "usage"):
            self.usage = dynamic_response.usage


def _response_envelope(response: Response, finish: str, content: str) -> ModelResponse:
    # Read accounting fields independently of output-item serialization/parsing.
    raw_usage = (
        response.usage.model_dump(mode="json") if response.usage is not None else None
    )
    return ModelResponse(
        id=response.id,
        created=int(response.created_at),
        model=response.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        usage=_normalize_usage(raw_usage),
        provider_response_metadata={
            "transport": _TRANSPORT,
            "status": response.status,
            "incomplete_reason": (
                response.incomplete_details.reason
                if response.incomplete_details is not None
                else None
            ),
            "finish_reason": finish,
            "raw_usage": raw_usage,
        },
    )


def _normalize_response(response: Response) -> ModelResponse | OpenAIError:
    raw = response.model_dump(mode="json")
    status = raw.get("status")
    reason = (raw.get("incomplete_details") or {}).get("reason")
    text: list[str] = []
    refusal: list[str] = []
    invalid_output = False
    for item in raw.get("output", []):
        if item["type"] == "reasoning":
            continue
        if item["type"] != "message":
            invalid_output = True
            continue
        if item.get("status") != "completed":
            invalid_output = True
        for part in item["content"]:
            if part["type"] == "output_text":
                text.append(part["text"])
            elif part["type"] == "refusal":
                refusal.append(part["refusal"])
            else:
                invalid_output = True
    content = "".join(text)
    if status == "incomplete":
        finish = {
            "max_output_tokens": "length",
            "content_filter": "content_filter",
        }.get(reason or "", "incomplete")
    elif status == "failed" or raw.get("error"):
        finish = "error"
    elif status != "completed":
        finish = status or "unknown"
    elif refusal:
        finish = "content_filter"
    elif invalid_output:
        finish = "unsupported_output"
    elif not content.strip():
        finish = "empty_output"
    else:
        finish = "stop"
    normalized = _response_envelope(response, finish, content)
    if finish != "stop":
        return _ResponseFailure(
            normalized, raw.get("error") or refusal or reason or status, response
        )
    return normalized


class _DeepSeekResponsesMixin:
    """Replace only the JSON-schema network path of LOTUS 1.1.4."""

    model: str
    max_batch_size: int
    rate_limit: int | None

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        validate_deepseek_responses_model(model)
        parent: Any = super()
        parent.__init__(model, *args, **kwargs)

    def _process_uncached_messages(
        self,
        uncached_data: list[tuple[list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        show_progress_bar: bool,
        progress_bar_desc: str,
    ) -> list[Any]:
        response_format = all_kwargs.get("response_format")
        if (
            not isinstance(response_format, Mapping)
            or response_format.get("type") != "json_schema"
        ):
            parent: Any = super()
            return parent._process_uncached_messages(
                uncached_data, all_kwargs, show_progress_bar, progress_bar_desc
            )
        if not uncached_data:
            return []
        validate_deepseek_responses_model(self.model)
        payload, client_options = _request_options(all_kwargs)
        batch = [messages for messages, _ in uncached_data]
        for messages in batch:
            if not messages or any(
                not isinstance(message, Mapping)
                or message.keys() != {"role", "content"}
                or message["role"] not in {"system", "developer", "user", "assistant"}
                or not isinstance(message["content"], str)
                for message in messages
            ):
                raise ValueError(
                    "DeepSeek Responses supports text-only role/content messages"
                )

        with (
            OpenAI(**client_options) as client,
            ThreadPoolExecutor(max_workers=self.max_batch_size) as executor,
            tqdm(
                total=len(batch), desc=progress_bar_desc, disable=not show_progress_bar
            ) as pbar,
        ):
            responses_api: Any = client.responses

            def request(messages: list[dict[str, str]]) -> ModelResponse | OpenAIError:
                try:
                    native_response: Response = responses_api.create(
                        model=self.model.removeprefix("deepseek/"),
                        input=messages,
                        **payload,
                    )
                except Exception as error:
                    # Like LiteLLM batch_completion, keep failures in input order.
                    failure: Any = (
                        error
                        if isinstance(error, OpenAIError)
                        else OpenAIError(str(error))
                    )
                    if failure is not error:
                        failure.__cause__ = error
                    failure.provider_response_metadata = {
                        "transport": _TRANSPORT,
                        "status": "failed",
                        "incomplete_reason": None,
                        "finish_reason": "error",
                        "raw_usage": None,
                    }
                    return failure
                try:
                    return _normalize_response(native_response)
                except Exception as error:
                    try:
                        envelope = _response_envelope(
                            native_response, "normalization_error", ""
                        )
                    except Exception as accounting_error:
                        error.add_note(
                            "Response accounting unavailable: "
                            f"{type(accounting_error).__name__}: {accounting_error}"
                        )
                        # Do not read malformed native fields again in this fallback.
                        envelope = ModelResponse(
                            model=self.model,
                            created=0,
                            choices=[],
                            usage=None,
                            provider_response_metadata={
                                "transport": _TRANSPORT,
                                "status": None,
                                "incomplete_reason": None,
                                "finish_reason": "normalization_error",
                                "raw_usage": None,
                            },
                        )
                    normalization_failure = _ResponseFailure(
                        envelope, error, native_response
                    )
                    normalization_failure.__cause__ = error
                    return normalization_failure

            responses: list[Any] = []
            batch_size = (
                self.max_batch_size if self.rate_limit is not None else len(batch)
            )
            for start in range(0, len(batch), batch_size):
                started = time.monotonic()
                sub_batch = batch[start : start + batch_size]
                responses.extend(executor.map(request, sub_batch))
                pbar.update(len(sub_batch))
                elapsed = time.monotonic() - started
                if self.rate_limit is not None and start + batch_size < len(batch):
                    delay = len(sub_batch) * 60 / self.rate_limit - elapsed
                    if delay > 0:
                        time.sleep(delay)
            return responses

    def _cache_response(self, response: Any, hash: str) -> None:
        if (
            isinstance(response, OpenAIError)
            and getattr(response, "provider_response_metadata", {}).get("transport")
            == _TRANSPORT
        ):
            # LOTUS must finish accounting the batch before _get_top_choice raises.
            return
        parent: Any = super()
        parent._cache_response(response, hash)

    def _update_stats(self, response: Any, is_cached: bool = False) -> None:
        if isinstance(response, _ResponseFailure):
            response = response.response
        parent: Any = super()
        parent._update_stats(response, is_cached=is_cached)


def deepseek_responses_lm_class(base: type[Any]) -> type[Any]:
    """Return a LOTUS subclass routing only json_schema calls to native Responses."""

    class DeepSeekResponsesLM(_DeepSeekResponsesMixin, base):
        pass

    return DeepSeekResponsesLM
