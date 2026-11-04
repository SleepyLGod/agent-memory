"""Offline physical-response regressions through the real LOTUS rankers."""

from types import SimpleNamespace
from typing import Any

import lotus
import pandas as pd
import pytest
from litellm import ModelResponse

from agent_memory.adapters.lotus.context import (
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.ranking_validation import (
    checked_ranking_models,
    validate_pairwise_answer,
)
from agent_memory.adapters.lotus.sem_topk import execute_sem_topk
from agent_memory.policy.logical import QueryExpr


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import importlib

    lm_module = importlib.import_module("lotus.models.lm")
    state: dict[str, Any] = {"text": "Document 1", "finish": "stop", "calls": 0}

    def complete(model: str, messages: list[Any], **kwargs: Any) -> list[ModelResponse]:
        state["calls"] += bool(messages)
        count = len(messages) + state.get("count_delta", 0)
        return [
            ModelResponse(
                model="gpt-4o-mini",
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": state["text"]},
                        "finish_reason": state["finish"],
                    }
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            )
            for _ in range(count)
        ]

    monkeypatch.setattr(lm_module, "batch_completion", complete)
    for name in ("lm", "helper_lm", "enable_cache"):
        monkeypatch.setattr(lotus.settings, name, getattr(lotus.settings, name))
    return state


def rank(context: LotusExecutionContext) -> pd.DataFrame:
    """Execute the actual adapter and sorter without replacing their parsers."""
    query = QueryExpr(
        op="sem_topk",
        inputs=(QueryExpr(op="materialized_view", params={"name": "data"}),),
        params={"instruction": "Find relevant notes", "k": 2},
    )
    return execute_sem_topk(
        query, {}, lambda *_: pd.DataFrame({"note": ["a", "b", "c"]}), context
    )


@pytest.mark.parametrize(
    "method", ["pairwise-naive", "pairwise-quick", "pairwise-heap"]
)
@pytest.mark.parametrize(
    "text,finish",
    [
        ("", "stop"),
        ("Document 1", "length"),
        ("Document 1", "content_filter"),
        ("Document 3", "stop"),
        ("1", "stop"),
        ("Document 1 or Document 2", "stop"),
    ],
)
def test_pairwise_rejects_invalid_response_without_retry(
    provider, method, text, finish
):
    provider.update(text=text, finish=finish)
    context = LotusExecutionContext(
        "gpt-4o-mini",
        config=LotusExecutionConfig(
            sem_topk_method=method,
            lm_enable_cache=False,
            lm_num_retries=0,
        ),
    )
    context.configure()
    original = lotus.settings.lm
    with pytest.raises(ValueError, match="incomplete|requires a final"):
        rank(context)
    assert provider["calls"] == 1
    assert lotus.settings.lm is original


def test_pairwise_rejects_wrong_response_count(provider):
    provider["count_delta"] = -1
    context = LotusExecutionContext(
        "gpt-4o-mini", config=LotusExecutionConfig(lm_enable_cache=False)
    )
    with pytest.raises(ValueError, match="expected 3 outputs, got 2"):
        rank(context)
    assert provider["calls"] == 1


@pytest.mark.parametrize(
    "method", ["pairwise-naive", "pairwise-quick", "pairwise-heap"]
)
@pytest.mark.parametrize(
    "text,strategy",
    [
        ("Document 2", None),
        ("Document 2", "COT"),
        ("Answer: Document 1", None),
        ("Compare both documents. Answer: Document 2", "ZS_COT"),
    ],
)
def test_valid_answers_keep_lotus_sorting(provider, method, text, strategy):
    provider["text"] = text
    context = LotusExecutionContext(
        "gpt-4o-mini",
        config=LotusExecutionConfig(
            sem_topk_method=method,
            sem_topk_strategy=strategy,
            lm_enable_cache=False,
        ),
    )
    assert len(rank(context)) == 2


def test_validated_operator_cache_is_reused(provider):
    context = LotusExecutionContext(
        "gpt-4o-mini", config=LotusExecutionConfig(lm_enable_cache=True)
    )
    expected = rank(context)
    calls = provider["calls"]
    pd.testing.assert_frame_equal(rank(context), expected)
    assert provider["calls"] == calls


def test_unchecked_operator_cache_cannot_hide_bad_cached_response(provider):
    provider["text"] = "Document 3"
    context = LotusExecutionContext(
        "gpt-4o-mini", config=LotusExecutionConfig(lm_enable_cache=True)
    )
    context.configure()
    # Seed the old permissive operator cache, including its underlying raw LM responses.
    pd.DataFrame({"note": ["a", "b", "c"]}).sem_topk(
        "{note} is relevant to: Find relevant notes",
        K=2,
        method="naive",
        strategy=None,
        cascade_threshold=None,
        return_stats=False,
        safe_mode=False,
        return_explanations=False,
    )
    calls = provider["calls"]
    with pytest.raises(ValueError, match="requires a final"):
        rank(context)
    assert provider["calls"] == calls


def test_listwise_truncation_is_not_retried_even_if_json_is_valid(provider):
    provider.update(text='{"selected_ids": ["row_0", "row_1"]}', finish="length")
    context = LotusExecutionContext(
        "gpt-4o-mini",
        config=LotusExecutionConfig(
            sem_topk_method="listwise",
            structured_parse_retries=3,
            lm_enable_cache=False,
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        rank(context)
    assert provider["calls"] == 1


def test_helper_model_is_checked_and_restored(monkeypatch):
    class Helper:
        def __call__(self, messages, **kwargs):
            return SimpleNamespace(
                outputs=[""], response_metadata=[{"finish_reason": "stop"}]
            )

    helper = Helper()
    monkeypatch.setattr(lotus.settings, "helper_lm", helper)
    with pytest.raises(ValueError, match="requires a final"):
        with checked_ranking_models(None):
            assert lotus.settings.helper_lm is not None
            lotus.settings.helper_lm([[]])
    assert lotus.settings.helper_lm is helper


@pytest.mark.parametrize(
    "text", ["Document 1 Document 2", "Answer: Document 3", "Reasoning only"]
)
def test_reasoning_requires_explicit_final_choice(text):
    with pytest.raises(ValueError):
        validate_pairwise_answer(text, explanation=True)
