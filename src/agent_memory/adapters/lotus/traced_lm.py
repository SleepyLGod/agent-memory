"""Trace wrapper for LOTUS LM calls."""

from __future__ import annotations

from collections.abc import Mapping
import time
from pathlib import Path
from typing import Any

from agent_memory.tracing.semantic import write_llm_call_trace


class TracedLM:
    """Proxy a LOTUS LM while recording actual request/response payloads."""

    def __init__(self, base_lm: Any, trace_dir: Path | str | None) -> None:
        self._base_lm = base_lm
        self._trace_dir = trace_dir

    def __call__(
        self,
        messages: list[list[dict[str, Any]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        **kwargs: Any,
    ) -> Any:
        """Call the wrapped LOTUS LM and write side-channel trace artifacts."""

        call_kwargs: dict[str, Any] = {
            "show_progress_bar": show_progress_bar,
            "progress_bar_desc": progress_bar_desc,
            **kwargs,
        }
        before = _usage_snapshot(self._base_lm)
        start = time.perf_counter()
        try:
            output = self._base_lm(
                messages,
                show_progress_bar=show_progress_bar,
                progress_bar_desc=progress_bar_desc,
                **kwargs,
            )
        except Exception as error:
            latency_sec = time.perf_counter() - start
            after = _usage_snapshot(self._base_lm)
            write_llm_call_trace(
                self._trace_dir,
                model=str(getattr(self._base_lm, "model", "")),
                messages=messages,
                kwargs=call_kwargs,
                latency_sec=latency_sec,
                usage_delta=_usage_delta(before, after),
                error=error,
            )
            raise

        latency_sec = time.perf_counter() - start
        after = _usage_snapshot(self._base_lm)
        write_llm_call_trace(
            self._trace_dir,
            model=str(getattr(self._base_lm, "model", "")),
            messages=messages,
            kwargs=call_kwargs,
            outputs=_output_items(output),
            latency_sec=latency_sec,
            usage_delta=_usage_delta(before, after),
        )
        return output

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes and methods to the wrapped LOTUS LM."""

        return getattr(self._base_lm, name)


def _output_items(output: Any) -> list[Mapping[str, Any]]:
    """Return per-message raw LM output records."""

    outputs = list(getattr(output, "outputs", ()))
    logprobs = getattr(output, "logprobs", None)
    if logprobs is None:
        return [{"output": item} for item in outputs]
    logprob_items = list(logprobs)
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(outputs):
        rows.append(
            {
                "output": item,
                "logprobs": logprob_items[index] if index < len(logprob_items) else None,
            }
        )
    return rows


def _usage_snapshot(lm: Any) -> dict[str, float | int]:
    """Return LOTUS usage counters for the wrapped LM."""

    stats = getattr(lm, "stats", None)
    if stats is None:
        return {
            "physical_prompt_tokens": 0,
            "physical_completion_tokens": 0,
            "physical_total_tokens": 0,
            "virtual_prompt_tokens": 0,
            "virtual_completion_tokens": 0,
            "virtual_total_tokens": 0,
            "cache_hits": 0,
        }
    return {
        "physical_prompt_tokens": stats.physical_usage.prompt_tokens,
        "physical_completion_tokens": stats.physical_usage.completion_tokens,
        "physical_total_tokens": stats.physical_usage.total_tokens,
        "virtual_prompt_tokens": stats.virtual_usage.prompt_tokens,
        "virtual_completion_tokens": stats.virtual_usage.completion_tokens,
        "virtual_total_tokens": stats.virtual_usage.total_tokens,
        "cache_hits": stats.cache_hits,
    }


def _usage_delta(
    before: Mapping[str, float | int],
    after: Mapping[str, float | int],
) -> dict[str, float | int]:
    """Return usage counter deltas."""

    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in before
    }
