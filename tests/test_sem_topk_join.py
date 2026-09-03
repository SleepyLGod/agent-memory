"""Tests for cardinality-bounded semantic joins."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.adapter import LotusAdapter
from agent_memory.adapters.lotus.sem_join import execute_sem_join
from agent_memory.adapters.lotus.sem_topk_join import (
    LISTWISE_JOIN_SYSTEM_PROMPT,
    _pairwise_topk,
    _parse_listwise_ids,
)
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.policy.logical import QueryExpr


class _StaticJoinContext:
    def __init__(self, config: LotusExecutionConfig) -> None:
        self.config = config
        self.pair_embedding_provider = None

    def configure(self) -> None:
        pass


class _BatchListwiseLM:
    max_ctx_len = 100_000
    max_tokens = 1024
    cache = None

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        return SimpleNamespace(outputs=self.outputs)

    def count_tokens(self, messages: Any) -> int:
        return len(str(messages))


def _query(*, k: int, how: str = "inner", on: tuple[str, ...] = ()) -> QueryExpr:
    return QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={
            "instruction": "The left and right rows describe the same entity.",
            "how": how,
            "k": k,
            **({"on": on} if on else {}),
        },
    )


def test_listwise_topk_join_selects_zero_to_k_with_exact_key_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    left = pd.DataFrame(
        {"tenant": ["a", "b"], "name": ["Melanie", "Poppy"]},
        index=[10, 20],
    )
    right = pd.DataFrame(
        {
            "tenant": ["a", "a", "b"],
            "name": ["Mel", "Melanie", "Max"],
        },
        index=[100, 101, 200],
    )
    lm = _BatchListwiseLM(
        [
            '{"selected_ids": ["candidate_1"]}',
            '{"selected_ids": []}',
        ]
    )
    monkeypatch.setattr(lotus.settings, "lm", lm)

    result = execute_sem_join(
        _query(k=1, how="outer", on=("tenant",)),
        {},
        lambda query, _inputs: left if query.params["name"] == "left" else right,
        _StaticJoinContext(LotusExecutionConfig(sem_join_topk_method="listwise")),
    )

    assert len(lm.calls) == 1
    assert len(lm.calls[0][0]) == 2
    assert lm.calls[0][0][0][0]["content"] == LISTWISE_JOIN_SYSTEM_PROMPT
    assert "one left row" in LISTWISE_JOIN_SYSTEM_PROMPT
    assert "Select at most max_matches candidates" in LISTWISE_JOIN_SYSTEM_PROMPT
    assert "Do not select merely related candidates" in LISTWISE_JOIN_SYSTEM_PROMPT
    assert "empty selected_ids array when none match" in LISTWISE_JOIN_SYSTEM_PROMPT
    assert len(result) == 4
    matched = result[result["name:left"].eq("Melanie") & result["name:right"].notna()]
    assert matched["name:right"].tolist() == ["Melanie"]
    assert result[result["name:left"].eq("Poppy")]["name:right"].isna().all()
    assert set(result.loc[result["name:left"].isna(), "name:right"]) == {"Mel", "Max"}


def test_listwise_topk_join_can_select_multiple_matches_up_to_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    left = pd.DataFrame({"name": ["Melanie"]}, index=[10])
    right = pd.DataFrame(
        {"name": ["Mel", "Melanie", "Max"]},
        index=[100, 101, 102],
    )
    lm = _BatchListwiseLM(
        ['{"selected_ids": ["candidate_0", "candidate_1"]}']
    )
    monkeypatch.setattr(lotus.settings, "lm", lm)

    result = execute_sem_join(
        _query(k=2),
        {},
        lambda query, _inputs: left if query.params["name"] == "left" else right,
        _StaticJoinContext(LotusExecutionConfig(sem_join_topk_method="listwise")),
    )

    assert result["name:right"].tolist() == ["Mel", "Melanie"]


def test_prompt_batching_packs_multiple_listwise_join_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    left = pd.DataFrame({"name": ["Melanie", "Poppy"]}, index=[10, 20])
    right = pd.DataFrame({"name": ["Mel", "Poppy"]}, index=[100, 200])
    lm = _BatchListwiseLM(
        [
            '{"results":['
            '{"task_id":"task_1","selected_ids":["candidate_3"]},'
            '{"task_id":"task_0","selected_ids":["candidate_0"]}]}'
        ]
    )
    monkeypatch.setattr(lotus.settings, "lm", lm)

    result = execute_sem_join(
        _query(k=1),
        {},
        lambda query, _inputs: left if query.params["name"] == "left" else right,
        _StaticJoinContext(
            LotusExecutionConfig(
                sem_join_topk_method="listwise",
                prompt_batching=PromptBatching(),
            )
        ),
    )

    assert list(zip(result["name:left"], result["name:right"], strict=True)) == [
        ("Melanie", "Mel"),
        ("Poppy", "Poppy"),
    ]
    assert len(lm.calls) == 1
    assert len(lm.calls[0][0]) == 1
    assert "task_0" in str(lm.calls[0][0][0])
    assert "task_1" in str(lm.calls[0][0][0])


def test_pairwise_topk_join_reuses_lotus_ranking_after_predicate_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    left = pd.DataFrame({"name": ["Melanie {not_a_column}"]}, index=[10])
    right = pd.DataFrame({"name": ["Mel", "Melanie"]}, index=[100, 101])
    captured: dict[str, Any] = {}

    def verify(candidates: pd.DataFrame, **_kwargs: Any) -> list[tuple[Any, Any, None]]:
        return [
            (row["_left_pair_id"], row["_right_pair_id"], None)
            for _, row in candidates.iterrows()
        ]

    def sem_topk(frame: pd.DataFrame, instruction: str, **kwargs: Any) -> pd.DataFrame:
        captured["frame"] = frame.copy()
        captured["instruction"] = instruction
        captured.update(kwargs)
        return frame.iloc[[1]]

    monkeypatch.setattr(sem_join_module, "verify_semantic_join_candidates", verify)
    monkeypatch.setattr(pd.DataFrame, "sem_topk", sem_topk, raising=False)

    result = execute_sem_join(
        _query(k=1),
        {},
        lambda query, _inputs: left if query.params["name"] == "left" else right,
        _StaticJoinContext(
            LotusExecutionConfig(sem_join_topk_method="pairwise-quick")
        ),
    )

    assert captured["K"] == 1
    assert captured["method"] == "quick"
    assert captured["frame"]["left"].nunique() == 1
    assert "{not_a_column}" in captured["frame"]["left"].iloc[0]
    assert captured["frame"]["right"].tolist() == ["name: Mel", "name: Melanie"]
    assert "{left}" in captured["instruction"]
    assert "{right}" in captured["instruction"]
    assert "{not_a_column}" not in captured["instruction"]
    assert result["name:right"].tolist() == ["Melanie"]


def test_pairwise_topk_join_skips_oracle_when_search_filter_selects_no_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    def unexpected_oracle(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("empty candidate sets must not call the oracle")

    monkeypatch.setattr(
        sem_join_module,
        "verify_semantic_join_candidates",
        unexpected_oracle,
    )
    candidates = pd.DataFrame(
        columns=[
            "_left_pair_id",
            "_right_pair_id",
            "_left_pair_text",
            "_right_pair_text",
        ]
    )

    assert (
        _pairwise_topk(
            candidates,
            instruction="Rows match.",
            left_label="left",
            right_label="right",
            k=1,
            method="pairwise-quick",
            context=_StaticJoinContext(LotusExecutionConfig()),
        )
        == []
    )


def test_exact_key_semantic_join_without_k_restricts_oracle_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    left = pd.DataFrame(
        {"tenant": ["a", "b"], "name": ["Melanie", "Poppy"]},
        index=[10, 20],
    )
    right = pd.DataFrame(
        {"tenant": ["a", "b"], "name": ["Mel", "Max"]},
        index=[100, 200],
    )
    captured: dict[str, Any] = {}

    def verify(candidates: pd.DataFrame, **_kwargs: Any) -> list[tuple[Any, Any, None]]:
        captured["pairs"] = list(
            zip(
                candidates["_left_pair_id"],
                candidates["_right_pair_id"],
                strict=True,
            )
        )
        return []

    monkeypatch.setattr(sem_join_module, "verify_semantic_join_candidates", verify)
    query = _query(k=1, on=("tenant",))
    query = QueryExpr(
        op=query.op,
        inputs=query.inputs,
        params={key: value for key, value in query.params.items() if key != "k"},
    )

    execute_sem_join(
        query,
        {},
        lambda expr, _inputs: left if expr.params["name"] == "left" else right,
        _StaticJoinContext(LotusExecutionConfig()),
    )

    assert captured["pairs"] == [(10, 100), (20, 200)]


def test_pairwise_topk_method_changes_physical_identity_without_changing_default() -> None:
    assert LotusAdapter().maintenance_execution_fingerprint == ""
    assert (
        LotusAdapter(
            config=LotusExecutionConfig(sem_join_topk_method="pairwise-quick")
        ).maintenance_execution_fingerprint
        != ""
    )


def test_listwise_topk_join_parser_allows_no_match_and_rejects_overflow() -> None:
    assert _parse_listwise_ids(
        '{"selected_ids": []}',
        valid_ids={"candidate_0"},
        k=1,
    ) == ()

    with pytest.raises(ValueError, match="more than k=1"):
        _parse_listwise_ids(
            '{"selected_ids": ["candidate_0", "candidate_1"]}',
            valid_ids={"candidate_0", "candidate_1"},
            k=1,
        )


def test_semantic_join_internal_ids_preserve_matched_and_unmatched_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    left = pd.DataFrame({"name": ["Melanie", "Poppy"]}, index=[10, 20])
    right = pd.DataFrame({"name": ["Mel", "Max"]}, index=[100, 200])

    def verify(*_args: Any, **_kwargs: Any) -> list[tuple[int, int, None]]:
        return [(10, 100, None)]

    monkeypatch.setattr(sem_join_module, "evaluate_semantic_join", verify)
    query = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={
            "instruction": "Rows match.",
            "how": "outer",
            "id_columns": ("_left_id", "_right_id"),
        },
    )

    result = execute_sem_join(
        query,
        {},
        lambda expr, _inputs: left if expr.params["name"] == "left" else right,
        _StaticJoinContext(LotusExecutionConfig()),
    )

    ids = result[["_left_id", "_right_id"]].astype(object)
    ids = ids.where(ids.notna(), None)
    assert ids.to_dict(orient="records") == [
        {"_left_id": 10, "_right_id": 100},
        {"_left_id": 20, "_right_id": None},
        {"_left_id": None, "_right_id": 200},
    ]
