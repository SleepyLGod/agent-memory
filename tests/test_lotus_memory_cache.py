"""Tests for opt-in LOTUS process-memory caching."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.context import (
    LOTUS_MEMORY_CACHE_MAX_SIZE,
    LotusExecutionConfig,
    LotusExecutionContext,
)


def test_lotus_context_uses_one_bounded_native_memory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    from lotus.cache import InMemoryCache

    monkeypatch.setattr(lotus.settings, "lm", lotus.settings.lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", lotus.settings.enable_cache)
    context = LotusExecutionContext(
        model="test/model",
        config=LotusExecutionConfig(lm_enable_cache=True),
    )
    context.configure()

    assert lotus.settings.enable_cache is True
    assert isinstance(lotus.settings.lm.cache, InMemoryCache)
    assert lotus.settings.lm.cache.max_size == LOTUS_MEMORY_CACHE_MAX_SIZE


def test_lotus_exact_lm_cache_counts_virtual_not_physical_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    from litellm.types.utils import ModelResponse, Usage
    from lotus.cache import InMemoryCache
    from lotus.models import LM

    monkeypatch.setattr(lotus.settings, "lm", lotus.settings.lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", lotus.settings.enable_cache)
    lm = LM(model="test/model", cache=InMemoryCache(max_size=1024))
    physical_calls = 0

    def process(
        uncached_data: list[tuple[list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        show_progress_bar: bool,
        progress_bar_desc: str,
    ) -> list[ModelResponse]:
        nonlocal physical_calls
        del all_kwargs, show_progress_bar, progress_bar_desc
        physical_calls += len(uncached_data)
        return [
            ModelResponse(
                model="test/model",
                choices=[
                    {
                        "message": {"role": "assistant", "content": "true"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                usage=Usage(
                    prompt_tokens=3,
                    completion_tokens=1,
                    total_tokens=4,
                ),
            )
            for _ in uncached_data
        ]

    monkeypatch.setattr(lm, "_process_uncached_messages", process)
    lotus.settings.configure(lm=lm, enable_cache=True)
    messages = [[{"role": "user", "content": "same prompt"}]]

    assert lm(messages, show_progress_bar=False).outputs == ["true"]
    physical_after_first = lm.stats.physical_usage.total_tokens
    virtual_after_first = lm.stats.virtual_usage.total_tokens
    assert lm(messages, show_progress_bar=False).outputs == ["true"]

    assert physical_calls == 1
    assert lm.stats.cache_hits == 1
    assert lm.stats.physical_usage.total_tokens == physical_after_first
    assert lm.stats.virtual_usage.total_tokens > virtual_after_first


def test_lotus_operator_cache_skips_repeated_operator_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    from lotus.cache import InMemoryCache, operator_cache
    from lotus.models import LM
    from lotus.types import LMStats

    monkeypatch.setattr(lotus.settings, "lm", lotus.settings.lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", lotus.settings.enable_cache)
    lm = LM(model="test/model", cache=InMemoryCache(max_size=1024))
    lotus.settings.configure(lm=lm, enable_cache=True)

    class Operator:
        def __init__(self) -> None:
            self._obj = pd.DataFrame({"value": [1, 2]})
            self.calls = 0

        @operator_cache
        def execute(self, increment: int) -> pd.DataFrame:
            self.calls += 1
            lm.stats.virtual_usage += LMStats.TotalUsage(
                prompt_tokens=2,
                completion_tokens=1,
                total_tokens=3,
            )
            return self._obj.assign(value=self._obj["value"] + increment)

    operator = Operator()

    first = operator.execute(1)
    second = operator.execute(1)

    pd.testing.assert_frame_equal(first, second)
    assert operator.calls == 1
    assert lm.stats.operator_cache_hits == 1
