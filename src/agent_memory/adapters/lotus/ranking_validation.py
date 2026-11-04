"""Validate ranking responses before LOTUS can substitute a default choice."""

from collections.abc import Iterator
from contextlib import contextmanager
import re
from typing import Any

from .structured import complete_lm_outputs, normalize_strategy


def validate_pairwise_answer(answer: str, *, explanation: bool) -> None:
    """Accept the requested choice format, never guess from arbitrary digits."""
    if not isinstance(answer, str):
        raise ValueError("sem_topk comparison must return text")
    if explanation:
        valid = re.search(r"\bAnswer:\s*Document\s+[12]\s*\Z", answer, re.IGNORECASE)
    else:
        valid = re.fullmatch(
            r"\s*(?:Answer:\s*)?Document\s+[12]\s*", answer, re.IGNORECASE
        )
    if valid is None:
        raise ValueError(
            "sem_topk comparison requires a final Document 1 or Document 2"
        )


class _RankingCache:
    """Keep unchecked legacy operator results out of the validated namespace."""

    def __init__(self, cache: Any) -> None:
        self._cache = cache

    def get(self, key: str) -> Any:
        return self._cache.get(f"validated-topk-v1:{key}")

    def insert(self, key: str, value: Any) -> None:
        self._cache.insert(f"validated-topk-v1:{key}", value)


class _CheckedRankingLM:
    """Delegate inference unchanged and reject invalid comparisons before parsing."""

    def __init__(self, model: Any, *, explanation: bool) -> None:
        self._model = model
        self._explanation = explanation
        cache = getattr(model, "cache", None)
        self.cache = _RankingCache(cache) if cache is not None else None

    def __call__(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        output = self._model(messages, *args, **kwargs)
        for answer in complete_lm_outputs(output, len(messages), operator="sem_topk"):
            validate_pairwise_answer(answer, explanation=self._explanation)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


@contextmanager
def checked_ranking_models(strategy: Any) -> Iterator[None]:
    """Scope validation to ranking, restoring LOTUS settings even on failure.

    LOTUS settings are already process-global and not thread-safe. This follows
    that execution contract; it does not patch LOTUS functions or prompts.
    """
    import lotus
    from lotus.types import ReasoningStrategy

    originals = {name: getattr(lotus.settings, name) for name in ("lm", "helper_lm")}
    try:
        for name, model in originals.items():
            if model is not None:
                setattr(
                    lotus.settings,
                    name,
                    _CheckedRankingLM(
                        model,
                        explanation=normalize_strategy(strategy)
                        == ReasoningStrategy.ZS_COT,
                    ),
                )
        yield
    finally:
        for name, model in originals.items():
            setattr(lotus.settings, name, model)
