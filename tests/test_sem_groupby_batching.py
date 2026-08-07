"""Bounded execution tests for LOTUS semantic grouping pairs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.context import LotusExecutionConfig
import agent_memory.adapters.lotus.sem_groupby as sem_groupby_module
from agent_memory.adapters.lotus.sem_groupby import evaluate_group_matches
from agent_memory.adapters.lotus.sem_groupby import is_retryable_group_match_error


def test_groupby_pair_batch_defaults_preserve_existing_execution() -> None:
    config = LotusExecutionConfig()

    assert config.sem_groupby_pair_batch_size is None
    assert config.sem_groupby_pair_batch_retries == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sem_groupby_pair_batch_size": 0}, "must be positive"),
        ({"sem_groupby_pair_batch_retries": -1}, "cannot be negative"),
    ],
)
def test_groupby_pair_batch_config_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LotusExecutionConfig(**kwargs)


def test_groupby_pair_batch_retries_transient_gateway_errors() -> None:
    from litellm.exceptions import BadGatewayError

    error = BadGatewayError(
        message="temporary gateway failure",
        llm_provider="deepseek",
        model="deepseek-v4-flash",
    )

    assert is_retryable_group_match_error(error) is True


def test_groupby_pair_batches_preserve_pair_order_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    decisions = [True, False, True, False, True, False]
    call_sizes: list[int] = []
    offset = 0

    def sem_filter(docs: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal offset
        call_sizes.append(len(docs))
        batch = decisions[offset : offset + len(docs)]
        offset += len(docs)
        return SimpleNamespace(
            outputs=batch,
            raw_outputs=[str(value) for value in batch],
            explanations=[None] * len(batch),
        )

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame({"name": ["a", "b", "c", "d"]})

    matches = evaluate_group_matches(
        source,
        input_cols=("name",),
        instruction="Rows describe the same topic.",
        pair_batch_size=2,
    )

    assert call_sizes == [2, 2, 2]
    assert matches == [(0, 1), (0, 3), (1, 3)]


def test_groupby_pair_batches_match_unbounded_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    decisions = [True, False, True, False, True, False]

    def evaluate(batch_size: int | None) -> list[tuple[int, int]]:
        offset = 0

        def sem_filter(docs: list[Any], *args: Any, **kwargs: Any) -> Any:
            nonlocal offset
            batch = decisions[offset : offset + len(docs)]
            offset += len(docs)
            return SimpleNamespace(outputs=batch)

        monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
        return evaluate_group_matches(
            pd.DataFrame({"name": ["a", "b", "c", "d"]}),
            input_cols=("name",),
            instruction="Rows describe the same topic.",
            pair_batch_size=batch_size,
        )

    assert evaluate(2) == evaluate(None)


def test_groupby_pair_batch_retries_only_the_failed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    class TransientError(Exception):
        pass

    calls = 0

    def sem_filter(docs: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TransientError("connection reset")
        return SimpleNamespace(outputs=[False] * len(docs))

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    monkeypatch.setattr(
        sem_groupby_module,
        "is_retryable_group_match_error",
        lambda error: isinstance(error, TransientError),
    )
    source = pd.DataFrame({"name": ["a", "b", "c"]})

    matches = evaluate_group_matches(
        source,
        input_cols=("name",),
        instruction="Rows describe the same topic.",
        pair_batch_size=2,
        pair_batch_retries=1,
    )

    assert matches == []
    assert calls == 3


def test_groupby_pair_batch_does_not_retry_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    calls = 0

    def sem_filter(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ValueError("invalid structured output")

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame({"name": ["a", "b"]})

    with pytest.raises(ValueError, match="invalid structured output"):
        evaluate_group_matches(
            source,
            input_cols=("name",),
            instruction="Rows describe the same topic.",
            pair_batch_size=1,
            pair_batch_retries=3,
        )

    assert calls == 1
