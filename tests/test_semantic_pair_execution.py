"""Tests for semantic-pair physical candidate execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
    SemanticPairExecutionProfile,
    select_semantic_pair_candidates,
)
from agent_memory.evaluation.harness import MemorySystemContract
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.sem_filter import execute_sem_filter
from agent_memory.adapters.lotus.sem_groupby import execute_sem_groupby
from agent_memory.adapters.lotus.sem_join import execute_sem_join
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from agent_memory.planner import PolicyDifferentiator
from agent_memory.runtime import MemoryRuntime
from agent_memory.storage import EmbeddingSpec
from agent_memory.tracing.semantic import query_digest


PAIR_EMBEDDING = EmbeddingSpec(
    source_column="text",
    property_name="embedding",
    model="test/embedding",
    revision="revision-1",
    dimensions=2,
    normalize=True,
)


@dataclass
class FakeEmbeddingProvider:
    vectors: dict[str, list[float]]

    def __post_init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(
        self,
        spec: EmbeddingSpec,
        texts: list[str],
    ) -> list[list[float]]:
        assert spec == PAIR_EMBEDDING
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class FakeFilterFrame(pd.DataFrame):
    oracle_batches: list[list[str]] = []

    @property
    def _constructor(self) -> type[FakeFilterFrame]:
        return FakeFilterFrame

    def sem_filter(self, instruction: str, **kwargs: object) -> FakeFilterFrame:
        del instruction, kwargs
        type(self).oracle_batches.append(list(self["memory_later"]))
        return self.copy()


class FakeContext:
    def __init__(
        self,
        config: LotusExecutionConfig,
        embedding_provider: object | None = None,
    ) -> None:
        self.config = config
        self.pair_embedding_provider = embedding_provider

    def configure(self) -> None:
        pass


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "left_id": "old-a",
                "right_id": "new-a",
                "left_text": "old alpha",
                "right_text": "new alpha",
            },
            {
                "left_id": "old-b",
                "right_id": "new-a",
                "left_text": "old beta",
                "right_text": "new alpha",
            },
            {
                "left_id": "old-a",
                "right_id": "new-b",
                "left_text": "old alpha",
                "right_text": "new beta",
            },
        ]
    )


def _profile(
    *,
    direction: str = "right-to-left",
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> SemanticPairExecutionProfile:
    return SemanticPairExecutionProfile(
        mode="search-filter",
        direction=direction,
        left_id_columns=("left_id",),
        right_id_columns=("right_id",),
        left_text_columns=("left_text",),
        right_text_columns=("right_text",),
        embedding=PAIR_EMBEDDING,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def _filter_query() -> QueryExpr:
    return QueryExpr(
        op="sem_filter",
        inputs=(QueryExpr(op="materialized_view", params={"name": "pairs"}),),
        params={
            "instruction": (
                "Existing Memory: {memory:earlier}\n"
                "New Memory: {memory:later}"
            )
        },
    )


def _filter_source() -> FakeFilterFrame:
    return FakeFilterFrame(
        [
            {
                "_row_id:earlier": "old-a",
                "_memory_ordinal:earlier": 0,
                "memory:earlier": "old alpha",
                "_row_id:later": "new-a",
                "_memory_ordinal:later": 0,
                "memory:later": "new alpha",
            },
            {
                "_row_id:earlier": "old-b",
                "_memory_ordinal:earlier": 0,
                "memory:earlier": "old beta",
                "_row_id:later": "new-a",
                "_memory_ordinal:later": 0,
                "memory:later": "new alpha",
            },
        ]
    )


def _filter_profile(min_similarity: float) -> SemanticPairExecutionProfile:
    return SemanticPairExecutionProfile(
        mode="search-filter",
        direction="right-to-left",
        left_id_columns=("_row_id:earlier", "_memory_ordinal:earlier"),
        right_id_columns=("_row_id:later", "_memory_ordinal:later"),
        left_text_columns=("memory:earlier",),
        right_text_columns=("memory:later",),
        embedding=PAIR_EMBEDDING,
        min_similarity=min_similarity,
    )


def _operator_pair_profile(
    *,
    direction: str,
    min_similarity: float,
) -> SemanticPairExecutionProfile:
    return SemanticPairExecutionProfile(
        mode="search-filter",
        direction=direction,
        left_id_columns=(PAIR_LEFT_ID_COLUMN,),
        right_id_columns=(PAIR_RIGHT_ID_COLUMN,),
        left_text_columns=(PAIR_LEFT_TEXT_COLUMN,),
        right_text_columns=(PAIR_RIGHT_TEXT_COLUMN,),
        embedding=PAIR_EMBEDDING,
        min_similarity=min_similarity,
    )


def test_threshold_selects_pairs_and_embeds_unique_endpoint_texts() -> None:
    provider = FakeEmbeddingProvider(
        {
            "left_text: old alpha": [1.0, 0.0],
            "left_text: old beta": [0.0, 1.0],
            "right_text: new alpha": [0.8, 0.6],
            "right_text: new beta": [1.0, 0.0],
        }
    )

    selection = select_semantic_pair_candidates(
        _pairs(),
        profile=_profile(min_similarity=0.75),
        embedding_provider=provider,
    )

    assert selection.selected_positions == (0, 2)
    assert selection.total_pair_count == 3
    assert selection.candidate_pair_count == 2
    assert selection.pair_reduction == pytest.approx(1 / 3)
    assert provider.calls == [
        [
            "left_text: old alpha",
            "right_text: new alpha",
            "left_text: old beta",
            "right_text: new beta",
        ]
    ]


def test_embedding_device_changes_profile_identity_and_must_match_provider() -> None:
    cpu = _profile(min_similarity=0.6)
    cuda = replace(cpu, embedding_device="cuda")

    assert cpu.fingerprint != cuda.fingerprint
    provider = FakeEmbeddingProvider({})
    provider.device = "cpu"
    with pytest.raises(ValueError, match="devices do not match"):
        select_semantic_pair_candidates(
            _pairs(),
            profile=cuda,
            embedding_provider=provider,
        )
@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("left-to-right", (1, 2)),
        ("right-to-left", (0, 2)),
        ("symmetric", (0, 1, 2)),
    ],
)
def test_top_k_respects_pair_direction(
    direction: str,
    expected: tuple[int, ...],
) -> None:
    provider = FakeEmbeddingProvider(
        {
            "left_text: old alpha": [1.0, 0.0],
            "left_text: old beta": [0.0, 1.0],
            "right_text: new alpha": [0.8, 0.6],
            "right_text: new beta": [1.0, 0.0],
        }
    )

    selection = select_semantic_pair_candidates(
        _pairs(),
        profile=_profile(direction=direction, top_k=1),
        embedding_provider=provider,
    )

    assert selection.selected_positions == expected


def test_threshold_is_applied_before_top_k() -> None:
    provider = FakeEmbeddingProvider(
        {
            "left_text: old alpha": [1.0, 0.0],
            "left_text: old beta": [0.0, 1.0],
            "right_text: new alpha": [0.8, 0.6],
            "right_text: new beta": [1.0, 0.0],
        }
    )

    selection = select_semantic_pair_candidates(
        _pairs(),
        profile=_profile(top_k=1, min_similarity=0.9),
        embedding_provider=provider,
    )

    assert selection.selected_positions == (2,)


def test_duplicate_texts_are_embedded_once_and_ties_use_stable_pair_id() -> None:
    source = pd.DataFrame(
        [
            {
                "left_id": "old-a",
                "right_id": "new-a",
                "left_text": "same old fact",
                "right_text": "same new fact",
            },
            {
                "left_id": "old-b",
                "right_id": "new-a",
                "left_text": "same old fact",
                "right_text": "same new fact",
            },
        ]
    )
    provider = FakeEmbeddingProvider(
        {
            "left_text: same old fact": [1.0, 0.0],
            "right_text: same new fact": [1.0, 0.0],
        }
    )

    selection = select_semantic_pair_candidates(
        source,
        profile=_profile(top_k=1),
        embedding_provider=provider,
    )

    assert selection.selected_positions == (0,)
    assert provider.calls == [
        ["left_text: same old fact", "right_text: same new fact"]
    ]

    reversed_selection = select_semantic_pair_candidates(
        source.iloc[::-1].reset_index(drop=True),
        profile=_profile(top_k=1),
        embedding_provider=provider,
    )

    assert reversed_selection.selected_positions == (1,)


def test_symmetric_top_k_uses_one_bucket_per_endpoint() -> None:
    source = pd.DataFrame(
        [
            {"id:left": "a", "id:right": "b", "text:left": "A", "text:right": "B"},
            {"id:left": "b", "id:right": "c", "text:left": "B", "text:right": "C"},
            {"id:left": "c", "id:right": "a", "text:left": "C", "text:right": "A"},
        ]
    )
    provider = FakeEmbeddingProvider(
        {
            "text: A": [1.0, 0.0],
            "text: B": [0.9, 0.435889894],
            "text: C": [0.342020143, 0.939692621],
        }
    )
    profile = SemanticPairExecutionProfile(
        mode="search-filter",
        direction="symmetric",
        left_id_columns=("id:left",),
        right_id_columns=("id:right",),
        left_text_columns=("text:left",),
        right_text_columns=("text:right",),
        embedding=PAIR_EMBEDDING,
        top_k=1,
    )

    selection = select_semantic_pair_candidates(
        source,
        profile=profile,
        embedding_provider=provider,
    )

    assert selection.selected_positions == (0, 1)


def test_top_k_preserves_multiplicity_for_a_selected_endpoint_pair() -> None:
    source = pd.concat([_pairs().iloc[[0]], _pairs().iloc[[0]]], ignore_index=True)
    provider = FakeEmbeddingProvider(
        {
            "left_text: old alpha": [1.0, 0.0],
            "right_text: new alpha": [1.0, 0.0],
        }
    )

    selection = select_semantic_pair_candidates(
        source,
        profile=_profile(top_k=1),
        embedding_provider=provider,
    )

    assert selection.selected_positions == (0, 1)
    assert selection.candidate_pair_count == 2


def test_empty_pair_relation_does_not_call_embedding_provider() -> None:
    provider = FakeEmbeddingProvider({})
    selection = select_semantic_pair_candidates(
        _pairs().iloc[0:0],
        profile=_profile(min_similarity=0.6),
        embedding_provider=provider,
    )

    assert selection.total_pair_count == 0
    assert selection.selected_positions == ()
    assert provider.calls == []


def test_profile_requires_a_search_filter_candidate_bound() -> None:
    with pytest.raises(ValueError, match="top_k or min_similarity"):
        _profile()


def test_embedding_provider_errors_are_not_hidden() -> None:
    class FailingProvider:
        def embed(
            self,
            spec: EmbeddingSpec,
            texts: list[str],
        ) -> list[list[float]]:
            del spec, texts
            raise RuntimeError("embedding unavailable")

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        select_semantic_pair_candidates(
            _pairs(),
            profile=_profile(min_similarity=0.6),
            embedding_provider=FailingProvider(),
        )


def test_sem_filter_search_filter_sends_only_candidates_to_oracle(tmp_path) -> None:
    query = _filter_query()
    source = _filter_source()
    provider = FakeEmbeddingProvider(
        {
            "memory: old alpha": [1.0, 0.0],
            "memory: old beta": [0.0, 1.0],
            "memory: new alpha": [0.8, 0.6],
        }
    )
    profile = _filter_profile(0.75)
    config = LotusExecutionConfig(
        semantic_trace_dir=tmp_path,
        semantic_pair_profiles={query_digest(query): profile},
    )
    FakeFilterFrame.oracle_batches = []

    result = execute_sem_filter(
        query,
        {},
        lambda _query, _inputs: source,
        FakeContext(config, provider),
    )

    assert list(result["memory:earlier"]) == ["old alpha"]
    assert FakeFilterFrame.oracle_batches == [["new alpha"]]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    candidate = next(
        event for event in events if event["event_type"] == "candidate_generation"
    )
    assert candidate["total_pair_count"] == 2
    assert candidate["candidate_pair_count"] == 1
    assert candidate["min_similarity"] == pytest.approx(0.75)
    assert candidate["embedding_device"] == "cpu"
    assert "vectors" not in candidate
    assert "pairs" not in candidate


def test_sem_filter_without_matching_profile_keeps_oracle_path() -> None:
    query = _filter_query()
    source = _filter_source()
    config = LotusExecutionConfig(
        semantic_pair_profiles={"different-query": _filter_profile(0.75)}
    )
    FakeFilterFrame.oracle_batches = []

    result = execute_sem_filter(
        query,
        {},
        lambda _query, _inputs: source,
        FakeContext(config),
    )

    assert len(result) == 2
    assert FakeFilterFrame.oracle_batches == [["new alpha", "new alpha"]]


def test_sem_filter_all_candidates_preserves_the_oracle_result() -> None:
    query = _filter_query()
    source = _filter_source()
    provider = FakeEmbeddingProvider(
        {
            "memory: old alpha": [1.0, 0.0],
            "memory: old beta": [0.0, 1.0],
            "memory: new alpha": [0.8, 0.6],
        }
    )
    config = LotusExecutionConfig(
        semantic_pair_profiles={
            query_digest(query): _filter_profile(-1.0),
        }
    )
    FakeFilterFrame.oracle_batches = []

    result = execute_sem_filter(
        query,
        {},
        lambda _query, _inputs: source,
        FakeContext(config, provider),
    )

    assert result.equals(source)
    assert FakeFilterFrame.oracle_batches == [["new alpha", "new alpha"]]


def test_sem_filter_zero_candidates_skips_oracle() -> None:
    query = _filter_query()
    source = _filter_source()
    provider = FakeEmbeddingProvider(
        {
            "memory: old alpha": [1.0, 0.0],
            "memory: old beta": [0.0, 1.0],
            "memory: new alpha": [-1.0, 0.0],
        }
    )
    config = LotusExecutionConfig(
        semantic_pair_profiles={
            query_digest(query): _filter_profile(0.75),
        }
    )
    FakeFilterFrame.oracle_batches = []

    result = execute_sem_filter(
        query,
        {},
        lambda _query, _inputs: source,
        FakeContext(config, provider),
    )

    assert result.empty
    assert list(result.columns) == list(source.columns)
    assert FakeFilterFrame.oracle_batches == []


def test_sem_join_search_filter_verifies_only_candidates_and_preserves_outer_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    oracle_batch_sizes: list[int] = []
    oracle_contract: dict[str, object] = {}

    class Output:
        outputs = [True]
        raw_outputs = ["True"]
        explanations = ["same topic"]

    def sem_filter(docs: list[object], *args: object, **kwargs: object) -> Output:
        oracle_batch_sizes.append(len(docs))
        oracle_contract["instruction"] = args[1]
        oracle_contract["default"] = kwargs["default"]
        oracle_contract["progress_bar_desc"] = kwargs["progress_bar_desc"]
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    left = pd.DataFrame(
        {"topic": ["alpha project", "tea preference"]},
        index=[10, 20],
    )
    right = pd.DataFrame(
        {"topic": ["alpha initiative", "coffee preference"]},
        index=[100, 200],
    )
    query = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={
            "instruction": "{topic:left} and {topic:right} describe the same topic.",
            "how": "outer",
        },
    )
    provider = FakeEmbeddingProvider(
        {
            "text: alpha project": [1.0, 0.0],
            "text: tea preference": [0.0, 1.0],
            "text: alpha initiative": [0.9, 0.435889894],
            "text: coffee preference": [-1.0, 0.0],
        }
    )
    context = FakeContext(
        LotusExecutionConfig(
            semantic_trace_dir=tmp_path,
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="left-to-right",
                    min_similarity=0.8,
                )
            },
        ),
        provider,
    )

    result = execute_sem_join(
        query,
        {"left": left, "right": right},
        lambda expression, inputs: inputs[str(expression.params["name"])],
        context,
    )

    assert oracle_batch_sizes == [1]
    assert oracle_contract == {
        "instruction": (
            "{topic:left} and {topic:right} describe the same topic."
        ),
        "default": False,
        "progress_bar_desc": "Join comparisons",
    }
    assert result.loc[0].to_dict() == {
        "topic:left": "alpha project",
        "topic:right": "alpha initiative",
    }
    assert result.loc[1, "topic:left"] == "tea preference"
    assert pd.isna(result.loc[1, "topic:right"])
    assert pd.isna(result.loc[2, "topic:left"])
    assert result.loc[2, "topic:right"] == "coffee preference"
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    candidate = next(
        event for event in events if event["event_type"] == "candidate_generation"
    )
    assert candidate["operator"] == "sem_join"
    assert candidate["total_pair_count"] == 4
    assert candidate["candidate_pair_count"] == 1
    assert "vectors" not in candidate
    assert "pairs" not in candidate


def test_sem_groupby_search_filter_verifies_only_candidate_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    oracle_batch_sizes: list[int] = []

    class Output:
        outputs = [True]
        raw_outputs = ["True"]
        explanations = ["same topic"]

    def sem_filter(docs: list[object], *args: object, **kwargs: object) -> Output:
        del args, kwargs
        oracle_batch_sizes.append(len(docs))
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame(
        {"topic": ["alpha project", "alpha initiative", "tea preference"]}
    )
    query = QueryExpr(
        op="sem_groupby",
        inputs=(QueryExpr(op="materialized_view", params={"name": "rows"}),),
        params={
            "input_cols": ("topic",),
            "instruction": "Group rows that describe the same topic: {topic}.",
        },
    )
    provider = FakeEmbeddingProvider(
        {
            "text: topic: alpha project": [1.0, 0.0],
            "text: topic: alpha initiative": [0.9, 0.435889894],
            "text: topic: tea preference": [0.0, 1.0],
        }
    )
    context = FakeContext(
        LotusExecutionConfig(
            semantic_trace_dir=tmp_path,
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="symmetric",
                    min_similarity=0.8,
                )
            },
        ),
        provider,
    )

    result = execute_sem_groupby(
        query,
        {"rows": source},
        lambda expression, inputs: inputs[str(expression.params["name"])],
        context,
    )

    assert oracle_batch_sizes == [1]
    assert list(result["_agent_memory_group_id"]) == [0, 0, 1]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    candidate = next(
        event for event in events if event["event_type"] == "candidate_generation"
    )
    assert candidate["operator"] == "sem_groupby"
    assert candidate["total_pair_count"] == 3
    assert candidate["candidate_pair_count"] == 1
    assert "vectors" not in candidate
    assert "pairs" not in candidate


def test_sem_join_search_filter_zero_candidates_skips_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    def sem_filter(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("zero candidates must not call the oracle")

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    left = pd.DataFrame({"topic": ["alpha"]}, index=[10])
    right = pd.DataFrame({"topic": ["coffee"]}, index=[100])
    query = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={
            "instruction": "{topic:left} and {topic:right} are the same topic.",
            "how": "inner",
        },
    )
    provider = FakeEmbeddingProvider(
        {
            "text: alpha": [1.0, 0.0],
            "text: coffee": [-1.0, 0.0],
        }
    )
    context = FakeContext(
        LotusExecutionConfig(
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="left-to-right",
                    min_similarity=0.8,
                )
            }
        ),
        provider,
    )

    result = execute_sem_join(
        query,
        {"left": left, "right": right},
        lambda expression, inputs: inputs[str(expression.params["name"])],
        context,
    )

    assert result.empty
    assert list(result.columns) == ["topic:left", "topic:right"]


def test_sem_join_search_filter_rejects_lotus_cascade() -> None:
    query = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="materialized_view", params={"name": "left"}),
            QueryExpr(op="materialized_view", params={"name": "right"}),
        ),
        params={"instruction": "Rows match.", "how": "inner"},
    )
    context = FakeContext(
        LotusExecutionConfig(
            sem_join_cascade_args={"sampling_percentage": 0.1},
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="left-to-right",
                    min_similarity=0.8,
                )
            },
        ),
        FakeEmbeddingProvider({}),
    )

    with pytest.raises(ValueError, match="cannot be combined with LOTUS cascade"):
        execute_sem_join(
            query,
            {
                "left": pd.DataFrame({"topic": ["alpha"]}),
                "right": pd.DataFrame({"topic": ["alpha"]}),
            },
            lambda expression, inputs: inputs[str(expression.params["name"])],
            context,
        )


def test_sem_groupby_search_filter_zero_candidates_skips_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    def sem_filter(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("zero candidates must not call the oracle")

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame({"topic": ["alpha", "coffee"]})
    query = QueryExpr(
        op="sem_groupby",
        inputs=(QueryExpr(op="materialized_view", params={"name": "rows"}),),
        params={
            "input_cols": ("topic",),
            "instruction": "Group the same {topic}.",
        },
    )
    provider = FakeEmbeddingProvider(
        {
            "text: topic: alpha": [1.0, 0.0],
            "text: topic: coffee": [-1.0, 0.0],
        }
    )
    context = FakeContext(
        LotusExecutionConfig(
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="symmetric",
                    min_similarity=0.8,
                )
            }
        ),
        provider,
    )

    result = execute_sem_groupby(
        query,
        {"rows": source},
        lambda expression, inputs: inputs[str(expression.params["name"])],
        context,
    )

    assert list(result["_agent_memory_group_id"]) == [0, 1]


def test_partitioned_sem_groupby_search_filter_stays_within_each_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus.sem_ops.sem_filter as sem_filter_module

    oracle_batch_sizes: list[int] = []

    class Output:
        outputs = [True]
        raw_outputs = ["True"]
        explanations = ["same topic"]

    def sem_filter(docs: list[object], *args: object, **kwargs: object) -> Output:
        del args, kwargs
        oracle_batch_sizes.append(len(docs))
        return Output()

    monkeypatch.setattr(sem_filter_module, "sem_filter", sem_filter)
    source = pd.DataFrame(
        {
            "scope": ["a", "a", "b", "b"],
            "topic": ["alpha", "alpha project", "tea", "tea preference"],
        }
    )
    query = QueryExpr(
        op="sem_groupby",
        inputs=(QueryExpr(op="materialized_view", params={"name": "rows"}),),
        params={
            "input_cols": ("topic",),
            "partition_by": ("scope",),
            "instruction": "Group the same {topic} within each scope.",
        },
    )
    provider = FakeEmbeddingProvider(
        {
            "text: topic: alpha": [1.0, 0.0],
            "text: topic: alpha project": [1.0, 0.0],
            "text: topic: tea": [0.0, 1.0],
            "text: topic: tea preference": [0.0, 1.0],
        }
    )
    context = FakeContext(
        LotusExecutionConfig(
            semantic_trace_dir=tmp_path,
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="symmetric",
                    min_similarity=0.8,
                )
            },
        ),
        provider,
    )

    result = execute_sem_groupby(
        query,
        {"rows": source},
        lambda expression, inputs: inputs[str(expression.params["name"])],
        context,
    )

    assert oracle_batch_sizes == [1, 1]
    assert list(result["_agent_memory_group_id"]) == [0, 0, 1, 1]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    candidate_events = [
        event for event in events if event["event_type"] == "candidate_generation"
    ]
    assert len(candidate_events) == 2
    assert [event["total_pair_count"] for event in candidate_events] == [1, 1]


def test_declared_label_sem_groupby_rejects_pair_profile() -> None:
    query = QueryExpr(
        op="sem_groupby",
        inputs=(QueryExpr(op="materialized_view", params={"name": "rows"}),),
        params={
            "input_cols": ("topic",),
            "instruction": "Assign a label.",
            "labels": (ColumnSpec("work", "Work topics."),),
        },
    )
    context = FakeContext(
        LotusExecutionConfig(
            semantic_pair_profiles={
                query_digest(query): _operator_pair_profile(
                    direction="symmetric",
                    min_similarity=0.8,
                )
            }
        ),
        FakeEmbeddingProvider({}),
    )

    with pytest.raises(ValueError, match="only to open-ended pairwise sem_groupby"):
        execute_sem_groupby(
            query,
            {"rows": pd.DataFrame({"topic": ["alpha"]})},
            lambda expression, inputs: inputs[str(expression.params["name"])],
            context,
        )


def test_runtime_snapshot_rejects_a_different_pair_execution_profile() -> None:
    class SnapshotMemory(am.Memory):
        log = am.Log({"message": "Message."})
        rows = log.select(["message"])

    policy = PolicyDifferentiator().differentiate(SnapshotMemory.spec())
    oracle = MemoryRuntime(policy, adapter=LotusAdapter())
    search = MemoryRuntime(
        policy,
        adapter=LotusAdapter(
            config=LotusExecutionConfig(
                semantic_pair_profiles={"query": _filter_profile(0.6)}
            )
        ),
    )

    oracle_snapshot = oracle.snapshot_state()
    search_snapshot = search.snapshot_state()

    assert "adapter_execution_fingerprint" not in oracle_snapshot
    assert search_snapshot["adapter_execution_fingerprint"]
    with pytest.raises(ValueError, match="adapter execution fingerprint"):
        search.restore_state(oracle_snapshot)
    with pytest.raises(ValueError, match="adapter execution fingerprint"):
        oracle.restore_state(search_snapshot)


def test_physical_pair_identity_does_not_change_existing_contracts() -> None:
    oracle = MemorySystemContract(
        system_id="mem0-memory",
        memory_model_id="memory-model",
        memory_provider_model_id="provider-model",
        input_adapter_id="input:v1",
        retrieval_recipe_id="retrieval:v1",
        maintenance_rule="mem0-additive-view:v1",
    )
    search = MemorySystemContract(
        system_id="mem0-memory",
        memory_model_id="memory-model",
        memory_provider_model_id="provider-model",
        input_adapter_id="input:v1",
        retrieval_recipe_id="retrieval:v1",
        maintenance_rule="mem0-additive-view:v1",
        maintenance_execution_id="semantic-pair-search-filter:abc123",
    )

    assert oracle.maintenance_fingerprint != search.maintenance_fingerprint
    assert "execution=semantic-pair-search-filter:abc123" in (
        search.effective_condition_id
    )


def test_physical_profile_does_not_change_policy_or_differential_fingerprint() -> None:
    from agent_memory.memories.mem0 import Mem0Memory
    from agent_memory.memories.mem0.storage import MEM0_QDRANT_STATEMENTS

    spec = Mem0Memory.spec()
    first = PolicyDifferentiator().differentiate(
        spec,
        statements=MEM0_QDRANT_STATEMENTS,
    )
    second = PolicyDifferentiator().differentiate(
        spec,
        statements=MEM0_QDRANT_STATEMENTS,
    )

    assert first.fingerprint == second.fingerprint
