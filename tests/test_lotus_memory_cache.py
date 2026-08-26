"""Tests for opt-in LOTUS process-memory caching."""

from __future__ import annotations

from collections.abc import Callable
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.context import (
    LOTUS_MEMORY_CACHE_ID,
    LOTUS_MEMORY_CACHE_MAX_SIZE,
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.policy.logical import QueryExpr


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
    assert LOTUS_MEMORY_CACHE_ID == "lotus-memory:1024"


class _OriginalExecutorError(RuntimeError):
    pass


class _CacheTraceError(RuntimeError):
    pass


def _failing_executor(
    error: BaseException,
) -> Callable[..., None]:
    def execute(*_args: object) -> None:
        raise error

    return execute


def _cache_enabled_adapter(monkeypatch: pytest.MonkeyPatch) -> LotusAdapter:
    adapter = LotusAdapter(
        config=LotusExecutionConfig(lm_enable_cache=True),
    )
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    return adapter


def test_cache_counter_failure_does_not_mask_executor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _cache_enabled_adapter(monkeypatch)
    original = _OriginalExecutorError("executor failed")

    def fail_counter() -> dict[str, int]:
        raise _CacheTraceError("counter failed")

    monkeypatch.setattr(
        adapter._context,
        "consume_cache_usage_delta",
        fail_counter,
    )

    with pytest.raises(_OriginalExecutorError) as captured:
        adapter._execute_traced_semantic(
            QueryExpr(op="sem_map"),
            {},
            _failing_executor(original),
        )

    assert captured.value is original
    assert captured.value.__notes__ == [
        "LOTUS cache usage trace failed: _CacheTraceError: counter failed"
    ]


def test_cache_trace_failure_does_not_mask_executor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _cache_enabled_adapter(monkeypatch)
    original = _OriginalExecutorError("executor failed")
    monkeypatch.setattr(
        adapter._context,
        "consume_cache_usage_delta",
        lambda: {},
    )

    def fail_trace(*_args: object, **_kwargs: object) -> None:
        raise _CacheTraceError("trace failed")

    monkeypatch.setattr(adapter, "_write_framework_cache_usage", fail_trace)

    with pytest.raises(_OriginalExecutorError) as captured:
        adapter._execute_traced_semantic(
            QueryExpr(op="sem_map"),
            {},
            _failing_executor(original),
        )

    assert captured.value is original
    assert captured.value.__notes__ == [
        "LOTUS cache usage trace failed: _CacheTraceError: trace failed"
    ]


def test_successful_executor_still_fails_when_cache_trace_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _cache_enabled_adapter(monkeypatch)
    monkeypatch.setattr(
        adapter._context,
        "consume_cache_usage_delta",
        lambda: {},
    )

    def fail_trace(*_args: object, **_kwargs: object) -> None:
        raise _CacheTraceError("trace failed")

    monkeypatch.setattr(adapter, "_write_framework_cache_usage", fail_trace)

    with pytest.raises(_CacheTraceError, match="trace failed"):
        adapter._execute_traced_semantic(
            QueryExpr(op="sem_map"),
            {},
            lambda *_args: "result",
        )


def test_framework_cache_trace_uses_shared_cache_identity(tmp_path) -> None:
    adapter = LotusAdapter(
        config=LotusExecutionConfig(semantic_trace_dir=tmp_path),
    )

    adapter._write_framework_cache_usage(
        QueryExpr(op="sem_map"),
        usage={},
        status="success",
        output=pd.DataFrame({"value": [1]}),
    )

    event = json.loads(
        (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    )
    assert event["cache_mode"] == LOTUS_MEMORY_CACHE_ID
    assert event["cache_mode"] == "lotus-memory:1024"


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


def test_cache_trace_deltas_do_not_double_count_nested_operations() -> None:
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)
    stats = SimpleNamespace(
        cache_hits=1,
        operator_cache_hits=0,
        physical_usage=usage,
        virtual_usage=usage,
    )
    context = LotusExecutionContext(model="test/model")
    context._lm = SimpleNamespace(stats=stats)

    inner = context.consume_cache_usage_delta()
    outer = context.consume_cache_usage_delta()
    usage.prompt_tokens = 5
    usage.completion_tokens = 2
    usage.total_tokens = 7
    stats.cache_hits = 2
    later = context.consume_cache_usage_delta()

    assert inner["physical_total_tokens"] == 4
    assert outer["physical_total_tokens"] == 0
    assert later["physical_total_tokens"] == 3
    assert sum(row["physical_total_tokens"] for row in (inner, outer, later)) == 7
    assert sum(row["lm_cache_hits"] for row in (inner, outer, later)) == 2


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
