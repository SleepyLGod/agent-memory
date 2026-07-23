from __future__ import annotations

import pytest

from agent_memory.evaluation.harness import ModelPrompt
from agent_memory.evaluation.models import LiteLLMBenchmarkModel


def _prompt() -> ModelPrompt:
    return ModelPrompt(
        prompt_name="test.answer",
        messages=({"role": "user", "content": "Answer this"},),
        prompt_digest="digest",
        temperature=0,
        max_tokens=123,
    )


def test_litellm_model_preserves_raw_usage_and_disables_cache() -> None:
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return {
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "Paris",
                        "reasoning_content": "The memory says Paris.",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }

    model = LiteLLMBenchmarkModel(
        model_id="deepseek/deepseek-v4-flash",
        completion=completion,
    )
    response = model.complete(_prompt(), attempt=1)

    assert response.text == "Paris"
    assert response.model == "deepseek-v4-flash"
    assert response.usage["prompt_cache_hit_tokens"] == 10
    assert response.usage["reasoning_tokens"] == 3
    assert response.raw_response["choices"][0]["message"]["reasoning_content"]
    assert calls == [
        {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Answer this"}],
            "temperature": 0,
            "max_tokens": 123,
            "caching": False,
            "num_retries": 0,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    ]


def test_litellm_model_rejects_missing_text() -> None:
    model = LiteLLMBenchmarkModel(
        model_id="deepseek/deepseek-v4-flash",
        completion=lambda **kwargs: {
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": None}}],
            "usage": {},
        },
    )

    with pytest.raises(ValueError, match="text content"):
        model.complete(_prompt(), attempt=1)


def test_litellm_model_can_disable_thinking_for_short_judges() -> None:
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [{"message": {"content": "yes"}}],
            "usage": {},
        }

    model = LiteLLMBenchmarkModel(completion=completion)
    prompt = ModelPrompt(
        prompt_name="judge",
        messages=({"role": "user", "content": "yes or no"},),
        prompt_digest="judge-digest",
        max_tokens=10,
        thinking_enabled=False,
    )

    assert model.complete(prompt, attempt=1).text == "yes"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
