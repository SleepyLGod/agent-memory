"""Offline contract tests for the native DeepSeek Responses LOTUS adapter."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
from threading import Event, Lock
from typing import Any

import httpx
import lotus
from litellm import ModelResponse
from lotus.cache import InMemoryCache
from lotus.models import LM
from openai import APIConnectionError, OpenAI, OpenAIError
from openai.types.responses import Response
import pytest

from agent_memory.adapters.lotus import deepseek_responses_lm as adapter


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
MODEL = "deepseek/deepseek-v4-flash"


def _native(text: str = '{"answer":1}', **overrides: Any) -> Response:
    data = {
        "id": "resp-test",
        "object": "response",
        "created_at": 123,
        "model": "deepseek-v4-flash-actual",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "output": [
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "temperature": 0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 8},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 19,
        },
    }
    data.update(overrides)
    return Response.model_validate(data)


class FakeClient:
    """Record SDK boundary calls without making network requests."""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.responses = self
        self.calls: list[dict[str, Any]] = []
        self.handler: Callable[[dict[str, Any]], Response] = lambda _: _native()
        self.closed = False

    def create(self, **payload: Any) -> Response:
        """Return the configured native response or raise its failure."""
        self.calls.append(deepcopy(payload))
        return self.handler(payload)

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.closed = True


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> list[FakeClient]:
    made: list[FakeClient] = []

    def create(**options: Any) -> FakeClient:
        client = FakeClient(**options)
        made.append(client)
        return client

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Tests must not make network requests")

    monkeypatch.setattr(adapter, "OpenAI", create)
    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    monkeypatch.setattr("lotus.models.lm.completion_cost", lambda **_: 0.0)
    return made


def _lm(**kwargs: Any) -> Any:
    return adapter.deepseek_responses_lm_class(LM)(
        model=MODEL, cache=InMemoryCache(max_size=64), **kwargs
    )


def _messages(count: int = 1) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": str(i)}] for i in range(count)]


def _process(lm: Any, count: int = 1, **kwargs: Any) -> list[Any]:
    return lm._process_uncached_messages(
        [(messages, str(i)) for i, messages in enumerate(_messages(count))],
        {**lm.kwargs, "response_format": SCHEMA, **kwargs},
        False,
        "Test responses",
    )


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4-flash",
        "deepseek/deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek/deepseek-v4-pro",
    ],
)
def test_supported_model_names(model: str) -> None:
    assert adapter.validate_deepseek_responses_model(model) is None


@pytest.mark.parametrize(
    "model",
    [
        "openai/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
        "deepseek/deepseek-chat",
        "deepseek-reasoner",
        "gpt-4o",
        "",
        "deepseek-v5",
    ],
)
def test_invalid_model_rejected_before_parent_resources(model: str) -> None:
    class ResourceBase:
        def __init__(self, **kwargs: Any) -> None:
            pytest.fail("Parent resources initialized for unsupported model")

    with pytest.raises(ValueError, match="model"):
        adapter.validate_deepseek_responses_model(model)
    with pytest.raises(ValueError, match="model"):
        adapter.deepseek_responses_lm_class(ResourceBase)(model=model)


def test_schema_payload_and_nonthinking_translation(clients: list[FakeClient]) -> None:
    schema = deepcopy(SCHEMA)
    kwargs = {
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.8,
        "timeout": 9.5,
        "num_retries": 4,
        "extra_body": {"thinking": {"type": "disabled"}},
        "stream": False,
    }
    lm = _lm(**kwargs)
    messages = [
        [
            {"role": "system", "content": "Answer with JSON."},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": '{"answer":0}'},
            {"role": "user", "content": "two"},
        ]
    ]
    result = lm(messages, response_format=schema, show_progress_bar=False)

    assert result.outputs == ['{"answer":1}']
    assert clients[0].options == {
        "base_url": "https://api.deepseek.com",
        "api_key": "fake-deepseek-key",
        "max_retries": 4,
        "timeout": 9.5,
    }
    assert clients[0].calls == [
        {
            "model": "deepseek-v4-flash",
            "input": messages[0],
            "text": {"format": {"type": "json_schema", **SCHEMA["json_schema"]}},
            "reasoning": {"effort": "none"},
            "max_output_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.8,
            "stream": False,
        }
    ]
    assert clients[0].closed
    assert schema == SCHEMA
    assert lm.kwargs == kwargs


@pytest.mark.parametrize(
    ("thinking", "effort", "expected"),
    [
        (None, None, "high"),
        ("enabled", None, "high"),
        ("enabled", "low", "low"),
        ("enabled", "medium", "high"),
        ("enabled", "high", "high"),
        ("enabled", "xhigh", "high"),
        ("enabled", "max", "max"),
        ("disabled", None, "none"),
        (None, "none", "none"),
    ],
)
def test_explicit_reasoning_mapping(
    clients: list[FakeClient], thinking: str | None, effort: str | None, expected: str
) -> None:
    kwargs: dict[str, Any] = {}
    if thinking is not None:
        kwargs["extra_body"] = {"thinking": {"type": thinking}}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    _process(_lm(**kwargs))
    assert clients[0].calls[0]["reasoning"] == {"effort": expected}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tools": []},
        {"tool_choice": "none"},
        {"stream": True},
        {"stream": 1},
        {"logprobs": True},
        {"top_logprobs": 3},
        {"n": 2},
        {"seed": 2},
        {"stop": ["END"]},
        {"frequency_penalty": 0},
        {"made_up": True},
        {"extra_body": {"other": 1}},
        {"extra_body": {"thinking": {"type": "maybe"}}},
        {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 100}}},
        {"extra_body": {"thinking": {"type": "disabled"}}, "reasoning_effort": "high"},
        {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "none"},
        {"reasoning_effort": "minimal"},
        {"num_retries": -1},
        {"num_retries": 11},
        {"num_retries": 1.5},
        {"num_retries": True},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": 3},
        {"top_p": -0.1},
        {"timeout": 0},
        {"timeout": float("inf")},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "missing"},
            }
        },
    ],
)
def test_unsupported_parameters_fail_before_client_creation(
    clients: list[FakeClient], kwargs: dict[str, Any]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _process(_lm(), **kwargs)
    assert clients == []


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "x"},
        {"role": "user", "content": [{"type": "image_url", "image_url": "x"}]},
        {"role": "assistant", "content": "x", "tool_calls": []},
    ],
)
def test_entire_input_is_validated_before_any_request(
    clients: list[FakeClient], message: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="text-only"):
        _lm()(
            _messages() + [[message]], response_format=SCHEMA, show_progress_bar=False
        )
    assert clients == []


def test_client_overrides_and_zero_retries(clients: list[FakeClient]) -> None:
    _process(
        _lm(),
        api_key="override-key",
        api_base="https://gateway.example/v1",
        num_retries=0,
    )
    assert clients[0].options["api_key"] == "override-key"
    assert clients[0].options["base_url"] == "https://gateway.example/v1"
    assert clients[0].options["max_retries"] == 0
    assert "api_key" not in clients[0].calls[0]


def test_missing_deepseek_key_does_not_use_openai_key(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _process(_lm())
    assert clients == []


def test_normalizes_usage_actual_model_and_metadata(clients: list[FakeClient]) -> None:
    response = _process(_lm())[0]
    assert isinstance(response, ModelResponse)
    assert response.model == "deepseek-v4-flash-actual"
    assert response.id == "resp-test"
    assert response.created == 123
    assert response.choices[0].message.content == '{"answer":1}'
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 19
    assert response.usage.prompt_cache_hit_tokens == 8
    assert response.usage.prompt_cache_miss_tokens == 4
    assert response.usage.prompt_tokens_details.cached_tokens == 8
    assert response.usage.completion_tokens_details.reasoning_tokens == 3
    assert response.provider_response_metadata == {
        "transport": "responses-json-schema",
        "status": "completed",
        "incomplete_reason": None,
        "finish_reason": "stop",
        "raw_usage": _native().usage.model_dump(mode="json"),
    }
    assert (
        json.loads(response.model_dump_json())["provider_response_metadata"]
        == response.provider_response_metadata
    )


def test_concurrency_is_bounded_and_output_order_is_preserved(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    second_started = Event()
    lock = Lock()
    active = 0
    peak = 0
    finished: list[int] = []

    def handler(payload: dict[str, Any]) -> Response:
        nonlocal active, peak
        index = int(payload["input"][0]["content"])
        with lock:
            active += 1
            peak = max(peak, active)
        if index == 0:
            assert second_started.wait(timeout=5)
        if index == 1:
            second_started.set()
        with lock:
            finished.append(index)
            active -= 1
        return _native(str(index))

    client.handler = handler
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    results = _process(_lm(max_batch_size=2), count=6)
    assert [r.choices[0].message.content for r in results] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert peak == 2
    assert sorted(finished) == list(range(6))


@pytest.mark.parametrize(
    ("elapsed", "expected_sleeps"), [(0.25, [1.75, 1.75]), (3, [])]
)
def test_rate_limit_wait_matches_lotus_batches(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
    elapsed: float,
    expected_sleeps: list[float],
) -> None:
    ticks = iter([0, elapsed, 10, 10 + elapsed, 20, 20 + elapsed])
    sleeps: list[float] = []
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(adapter.time, "sleep", sleeps.append)
    results = _process(_lm(max_batch_size=2, rate_limit=60), count=5)
    assert len(results) == 5
    assert sleeps == expected_sleeps


def test_rate_limit_caps_batch_size(clients: list[FakeClient]) -> None:
    lm = _lm(max_batch_size=10, rate_limit=2)
    assert lm.max_batch_size == 2


@pytest.mark.parametrize(
    ("overrides", "finish_reason"),
    [
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            "length",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            "content_filter",
        ),
        ({"status": "incomplete"}, "incomplete"),
        (
            {
                "status": "failed",
                "error": {"code": "server_error", "message": "failed"},
            },
            "error",
        ),
        ({"status": "in_progress"}, "in_progress"),
        ({"output": []}, "empty_output"),
        (
            {
                "output": [
                    {
                        "id": "m",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": "No."}],
                    }
                ]
            },
            "content_filter",
        ),
    ],
)
def test_non_success_is_an_exception_with_usage_not_empty_success(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    finish_reason: str,
) -> None:
    client = FakeClient()
    client.handler = lambda _: _native(**overrides)
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    response = _process(_lm())[0]
    assert isinstance(response, OpenAIError)
    assert response.usage.total_tokens == 19
    assert response.model == "deepseek-v4-flash-actual"
    assert response.provider_response_metadata["finish_reason"] == finish_reason
    assert response.provider_response_metadata["status"] == overrides.get(
        "status", "completed"
    )
    assert response.provider_response_metadata["raw_usage"]["input_tokens"] == 12
    lm = _lm()
    with pytest.raises(OpenAIError):
        lm(_messages(), response_format=SCHEMA, show_progress_bar=False)
    assert lm.stats.physical_usage.total_tokens == 19


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("native_failure", [False, True])
def test_partial_failures_preserve_all_usage_and_successful_cache_entries(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
    cache_enabled: bool,
    failure_index: int,
    native_failure: bool,
) -> None:
    client = FakeClient()
    network_error = APIConnectionError(
        request=httpx.Request("POST", "https://api.deepseek.com/responses")
    )

    def handler(payload: dict[str, Any]) -> Response:
        index = int(payload["input"][0]["content"])
        if index == failure_index:
            if native_failure:
                return _native(
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                )
            raise network_error
        return _native(str(index))

    client.handler = handler
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    monkeypatch.setattr(lotus.settings, "enable_cache", cache_enabled)
    lm = _lm(num_retries=2)
    results = _process(lm, count=3)
    assert isinstance(results[failure_index], OpenAIError)
    assert len(results) == 3
    assert len(client.calls) == 3  # No outer retry loop.
    assert (
        results[failure_index].provider_response_metadata["transport"]
        == "responses-json-schema"
    )
    assert lm.stats.physical_usage.total_tokens == 0  # Parent owns accounting.
    client.calls.clear()
    with pytest.raises(OpenAIError) as caught:
        lm(_messages(3), response_format=SCHEMA, show_progress_bar=False)
    if not native_failure:
        assert caught.value is network_error
    assert len(client.calls) == 3
    assert lm.stats.physical_usage.total_tokens == (57 if native_failure else 38)
    assert lm.stats.virtual_usage.total_tokens == (57 if native_failure else 38)
    client.handler = lambda p: _native(p["input"][0]["content"])
    client.calls.clear()
    assert lm(
        _messages(3), response_format=SCHEMA, show_progress_bar=False
    ).outputs == ["0", "1", "2"]
    assert len(client.calls) == (1 if cache_enabled else 3)
    assert lm.stats.cache_hits == (2 if cache_enabled else 0)


@pytest.mark.parametrize(
    "response_format", [None, {"type": "json_object"}, {"type": "text"}]
)
def test_non_schema_delegates_native_chat_without_sdk_client(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch, response_format: Any
) -> None:
    calls: list[dict[str, Any]] = []

    def chat(model: str, batch: Any, **kwargs: Any) -> list[ModelResponse]:
        calls.append({"model": model, "batch": batch, **kwargs})
        return [
            ModelResponse(
                choices=[{"message": {"role": "assistant", "content": "chat"}}]
            )
        ]

    monkeypatch.setattr("lotus.models.lm.batch_completion", chat)
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    result = _lm()(
        _messages(),
        response_format=response_format,
        stop=["END"],
        show_progress_bar=False,
    )
    assert result.outputs == ["chat"]
    assert clients == []
    assert calls[0]["response_format"] == response_format
    assert calls[0]["stop"] == ["END"]
    assert calls[0]["model"] == MODEL


def test_cache_hits_keep_virtual_usage_without_new_client(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lotus.settings, "enable_cache", True)
    lm = _lm()
    assert lm(_messages(), response_format=SCHEMA, show_progress_bar=False).outputs == [
        '{"answer":1}'
    ]
    assert lm(_messages(), response_format=SCHEMA, show_progress_bar=False).outputs == [
        '{"answer":1}'
    ]
    assert len(clients) == 1
    assert lm.stats.physical_usage.total_tokens == 19
    assert lm.stats.virtual_usage.total_tokens == 38
    assert lm.stats.cache_hits == 1
    altered_schema = deepcopy(SCHEMA)
    altered_schema["json_schema"]["name"] = "different"
    lm(_messages(), response_format=altered_schema, show_progress_bar=False)
    assert len(clients) == 2


def test_empty_batch_does_not_initialize_client(clients: list[FakeClient]) -> None:
    assert _process(_lm(), count=0) == []
    assert clients == []


def test_schema_without_optional_fields_and_default_retry_budget(
    clients: list[FakeClient],
) -> None:
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }
    _process(_lm(), response_format=schema)
    assert clients[0].calls[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "answer",
            "schema": {"type": "object"},
        }
    }
    assert clients[0].options["max_retries"] == 2


def test_reasoning_is_not_mixed_into_text_and_usage_extensions_survive(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _native().model_dump(mode="json")
    raw["output"].insert(
        0,
        {
            "id": "r",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "private reasoning"}],
        },
    )
    raw["output"][1]["content"] = [
        {"type": "output_text", "text": '{"answer":', "annotations": []},
        {"type": "output_text", "text": "1}", "annotations": []},
    ]
    raw["usage"]["provider_counter"] = 42
    raw["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
    client = FakeClient()
    client.handler = lambda _: Response.model_validate(raw)
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    response = _process(_lm())[0]
    assert response.choices[0].message.content == '{"answer":1}'
    assert response.usage.completion_tokens_details.reasoning_tokens == 0
    assert response.provider_response_metadata["raw_usage"]["provider_counter"] == 42


def test_missing_usage_is_not_fabricated(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    client.handler = lambda _: _native(usage=None, status="failed")
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    response = _process(_lm())[0]
    assert isinstance(response, OpenAIError)
    assert getattr(response, "usage", None) is None
    assert response.provider_response_metadata["raw_usage"] is None
    lm = _lm()
    with pytest.raises(OpenAIError):
        lm(_messages(), response_format=SCHEMA, show_progress_bar=False)
    assert lm.stats.physical_usage.total_tokens == 0


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("bad_field", ["created_at", "usage"])
def test_malformed_accounting_preserves_order_and_other_usage(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch,
    bad_field: str, cache_enabled: bool,
) -> None:
    native = _native().model_copy(update={
        bad_field: None if bad_field == "created_at" else {"invalid": "usage"},
    })
    error = ValueError("original normalization failure")
    original_normalizer = adapter._normalize_response

    def normalize(response: Response) -> Any:
        if response is native:
            raise error
        return original_normalizer(response)

    client = FakeClient()
    client.handler = lambda p: native if p["input"][0]["content"] == "1" else _native()
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    monkeypatch.setattr(adapter, "_normalize_response", normalize)
    monkeypatch.setattr(lotus.settings, "enable_cache", cache_enabled)
    lm = _lm()
    responses = _process(lm, count=3)
    assert isinstance(responses[0], ModelResponse)
    assert isinstance(responses[2], ModelResponse)
    failure = responses[1]
    assert isinstance(failure, OpenAIError)
    assert failure.native_response is native
    assert failure.__cause__ is error
    assert any("accounting" in note for note in error.__notes__)
    assert failure.provider_response_metadata["finish_reason"] == "normalization_error"
    assert failure.provider_response_metadata["raw_usage"] is None
    assert getattr(failure, "usage", None) is None
    with pytest.raises(OpenAIError, match="original normalization failure"):
        lm(_messages(3), response_format=SCHEMA, show_progress_bar=False)
    assert lm.stats.physical_usage.total_tokens == 38
    assert lm.stats.virtual_usage.total_tokens == 38


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("usage_available", [False, True])
@pytest.mark.parametrize("failure_stage", ["normalizer", "output_serialization"])
def test_normalizer_failure_preserves_native_response_and_batch_usage(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
    cache_enabled: bool,
    usage_available: bool,
    failure_stage: str,
) -> None:
    native = _native(
        id="resp-normalization-failure",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        usage=_native().usage if usage_available else None,
    )
    client = FakeClient()
    client.handler = lambda p: native if p["input"][0]["content"] == "0" else _native()
    original_normalizer = adapter._normalize_response
    normalization_error = ValueError("Unsupported native item")

    def normalize(response: Response) -> Any:
        if response is native:
            raise normalization_error
        return original_normalizer(response)

    if failure_stage == "normalizer":
        monkeypatch.setattr(adapter, "_normalize_response", normalize)
    else:
        original_dump = Response.model_dump

        def dump(response: Response, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if response is native:
                raise normalization_error
            return original_dump(response, *args, **kwargs)

        monkeypatch.setattr(Response, "model_dump", dump)
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    monkeypatch.setattr(lotus.settings, "enable_cache", cache_enabled)
    lm = _lm()
    responses = _process(lm, count=3)
    failure = responses[0]
    assert isinstance(failure, OpenAIError)
    assert failure.native_response is native
    assert failure.__cause__ is normalization_error
    assert failure.model == "deepseek-v4-flash-actual"
    assert isinstance(failure.response, ModelResponse)
    assert failure.provider_response_metadata == {
        "transport": "responses-json-schema",
        "status": "incomplete",
        "incomplete_reason": "max_output_tokens",
        "finish_reason": "normalization_error",
        "raw_usage": native.usage.model_dump(mode="json") if usage_available else None,
    }
    if usage_available:
        assert failure.usage.total_tokens == 19
        assert failure.usage.prompt_cache_hit_tokens == 8
        assert failure.usage.prompt_cache_miss_tokens == 4
        assert failure.usage.completion_tokens_details.reasoning_tokens == 3
    else:
        assert not hasattr(failure, "usage")
    assert lm.stats.physical_usage.total_tokens == 0
    client.calls.clear()
    with pytest.raises(OpenAIError, match="Unsupported native item"):
        lm(_messages(3), response_format=SCHEMA, show_progress_bar=False)
    assert len(client.calls) == 3
    assert lm.stats.physical_usage.total_tokens == (57 if usage_available else 38)
    assert lm.stats.virtual_usage.total_tokens == (57 if usage_available else 38)
    client.calls.clear()
    client.handler = lambda _: _native()
    assert (
        len(lm(_messages(3), response_format=SCHEMA, show_progress_bar=False).outputs)
        == 3
    )
    assert len(client.calls) == (1 if cache_enabled else 3)


def test_unsupported_native_output_keeps_original_response(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native(
        output=[
            {
                "type": "function_call",
                "id": "fc-test",
                "call_id": "call-test",
                "name": "unsupported",
                "arguments": "{}",
                "status": "completed",
            }
        ]
    )
    client = FakeClient()
    client.handler = lambda _: native
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    failure = _process(_lm())[0]
    assert isinstance(failure, OpenAIError)
    assert failure.native_response is native
    assert failure.usage.total_tokens == 19
    assert failure.provider_response_metadata["finish_reason"] == "unsupported_output"


def test_outer_hook_receives_ordered_failure_metadata_before_parent_accounting(
    clients: list[FakeClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    class OuterHook(adapter.deepseek_responses_lm_class(LM)):
        def _process_uncached_messages(self, *args: Any, **kwargs: Any) -> list[Any]:
            responses = super()._process_uncached_messages(*args, **kwargs)
            assert self.stats.physical_usage.total_tokens == 0
            seen.extend(responses)
            return responses

    client = FakeClient()
    client.handler = lambda p: _native(
        p["input"][0]["content"],
        status="incomplete" if p["input"][0]["content"] == "0" else "completed",
        incomplete_details={"reason": "max_output_tokens"}
        if p["input"][0]["content"] == "0"
        else None,
    )
    monkeypatch.setattr(adapter, "OpenAI", lambda **_: client)
    monkeypatch.setattr(lotus.settings, "enable_cache", True)
    lm = OuterHook(model=MODEL, cache=InMemoryCache(max_size=8))
    with pytest.raises(OpenAIError):
        lm(_messages(2), response_format=SCHEMA, show_progress_bar=False)
    assert [r.provider_response_metadata["finish_reason"] for r in seen] == [
        "length",
        "stop",
    ]
    assert (
        seen[0].provider_response_metadata["incomplete_reason"] == "max_output_tokens"
    )
    assert lm.stats.physical_usage.total_tokens == 38


@pytest.mark.parametrize("always_fail", [False, True])
def test_installed_sdk_serializes_responses_and_owns_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
    always_fail: bool,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if always_fail or len(requests) == 1:
            return httpx.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "retry"}},
            )
        return httpx.Response(200, json=_native().model_dump(mode="json"))

    def sdk_client(**kwargs: Any) -> OpenAI:
        return OpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr(adapter, "OpenAI", sdk_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    result = _process(
        _lm(num_retries=1, extra_body={"thinking": {"type": "disabled"}})
    )[0]
    assert len(requests) == 2
    assert str(requests[0].url) == "https://api.deepseek.com/responses"
    assert requests[0].headers["authorization"] == "Bearer fake-deepseek-key"
    payload = json.loads(requests[0].content)
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_output_tokens"] == 512
    assert payload["text"]["format"]["schema"] == SCHEMA["json_schema"]["schema"]
    assert "max_tokens" not in payload
    assert "num_retries" not in payload
    if always_fail:
        assert isinstance(result, OpenAIError)
        assert result.provider_response_metadata["status"] == "failed"
    else:
        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == '{"answer":1}'
