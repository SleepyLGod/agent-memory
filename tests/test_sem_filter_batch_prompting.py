"""Tests for semantic-filter batch prompting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_filter_batch_prompting import (
    execute_batch_prompted_sem_filter,
)


class FakeBatchLM:
    """Return one configured output batch for each LM invocation."""

    max_tokens = 512
    cache = None

    def __init__(self, output_batches: list[list[str]]) -> None:
        self._output_batches = iter(output_batches)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        return SimpleNamespace(outputs=next(self._output_batches))


def _context(
    *,
    batch_size: int = 2,
    retries: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=LotusExecutionConfig(
            structured_parse_retries=retries,
            prompt_batching=PromptBatching(max_tasks=batch_size),
        ),
    )


def test_batch_prompting_preserves_order_multiplicity_and_uses_stable_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "left": ["same", "other", "same"],
            "right": ["candidate", "candidate", "candidate"],
            "untouched": [1, 2, 3],
        },
        index=[7, 7, 9],
    )
    lm = FakeBatchLM(
        [
            [
                '{"decisions":[{"row_id":"row_1","keep":false},'
                '{"row_id":"row_0","keep":true}]}',
                '{"decisions":[{"row_id":"row_2","keep":true}]}',
            ]
        ]
    )
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    context = _context()

    result = execute_batch_prompted_sem_filter(
        source,
        instruction="{left} matches {right}.",
        prompt_batching=context.config.prompt_batching,
        context=context,
    )

    pd.testing.assert_frame_equal(result.frame, source.iloc[[0, 2]])
    assert result.prompt_count == 2
    assert result.retry_count == 0
    assert result.tuple_count == 3
    messages, kwargs = lm.calls[0]
    assert len(messages) == 2
    prompt_text = str(messages)
    assert all(row_id in prompt_text for row_id in ("row_0", "row_1", "row_2"))
    assert "untouched" not in prompt_text
    assert kwargs["response_format"] == {"type": "json_object"}


def test_batch_prompting_keeps_one_fixed_output_limit_for_many_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    task_count = 128
    source = pd.DataFrame(
        {
            "left": [f"left-{index}" for index in range(task_count)],
            "right": [f"right-{index}" for index in range(task_count)],
        }
    )
    decisions = ",".join(
        f'{{"row_id":"row_{index}","keep":true}}'
        for index in range(task_count)
    )
    lm = FakeBatchLM([[f'{{"decisions":[{decisions}]}}']])
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    context = _context(batch_size=task_count)

    result = execute_batch_prompted_sem_filter(
        source,
        instruction="{left} matches {right}.",
        prompt_batching=context.config.prompt_batching,
        context=context,
    )

    assert len(result.frame) == task_count
    assert lm.calls[0][1]["max_tokens"] == context.config.structured_max_tokens


def test_batch_prompting_retries_only_invalid_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    source = pd.DataFrame(
        {
            "left": ["a", "b", "c", "d"],
            "right": ["A", "B", "C", "D"],
        }
    )
    lm = FakeBatchLM(
        [
            [
                '{"decisions":[{"row_id":"row_0","keep":true},'
                '{"row_id":"row_1","keep":false}]}',
                '{"decisions":[{"row_id":"row_2","keep":true}]}',
            ],
            [
                '{"decisions":[{"row_id":"row_2","keep":false},'
                '{"row_id":"row_3","keep":true}]}',
            ],
        ]
    )
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    context = _context(retries=1)

    result = execute_batch_prompted_sem_filter(
        source,
        instruction="{left} matches {right}.",
        prompt_batching=context.config.prompt_batching,
        context=context,
    )

    pd.testing.assert_frame_equal(result.frame, source.iloc[[0, 3]])
    assert result.prompt_count == 3
    assert result.retry_count == 1
    assert len(lm.calls) == 2
    retry_messages, _kwargs = lm.calls[1]
    assert len(retry_messages) == 1
    retry_prompt = str(retry_messages)
    assert "row_2" in retry_prompt and "row_3" in retry_prompt
    assert "row_0" not in retry_prompt and "row_1" not in retry_prompt


@pytest.mark.parametrize(
    ("raw_output", "message"),
    (
        ('{"decisions":[{"row_id":"row_0","keep":1}]}', "boolean"),
        (
            '{"decisions":[{"row_id":"row_0","keep":true},'
            '{"row_id":"row_0","keep":false}]}',
            "duplicate",
        ),
        ('{"decisions":[{"row_id":"row_9","keep":true}]}', "unknown"),
        ('{"decisions":[]}', "omitted"),
    ),
)
def test_batch_prompting_rejects_invalid_decision_contract(
    monkeypatch: pytest.MonkeyPatch,
    raw_output: str,
    message: str,
) -> None:
    import lotus

    lm = FakeBatchLM([[raw_output]])
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    context = _context(batch_size=1)

    with pytest.raises(ValueError, match=message):
        execute_batch_prompted_sem_filter(
            pd.DataFrame({"left": ["a"], "right": ["A"]}),
            instruction="{left} matches {right}.",
            prompt_batching=context.config.prompt_batching,
            context=context,
        )


def test_batch_prompting_empty_input_skips_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    lm = FakeBatchLM([])
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    source = pd.DataFrame(columns=["left", "right"])
    context = _context()

    result = execute_batch_prompted_sem_filter(
        source,
        instruction="{left} matches {right}.",
        prompt_batching=context.config.prompt_batching,
        context=context,
    )

    pd.testing.assert_frame_equal(result.frame, source)
    assert result.prompt_count == 0
    assert result.tuple_count == 0
    assert lm.calls == []
