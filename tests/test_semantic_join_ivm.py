"""Incremental maintenance tests for direct semantic inner joins."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
)
from agent_memory.adapters.lotus.sem_join import execute_sem_join
from agent_memory.evaluation.agent_memory_drivers import (
    inventory_operator_semantic_pair_sites,
)
from agent_memory.planner.differential_policy import DifferentialNode
from agent_memory.policy import QueryExpr, Relation


class _GeneralSemanticJoinMemory(am.Memory):
    log = am.Log(
        {
            "side": "Join side.",
            "tenant": "Exact tenant partition.",
            "position": "Event order.",
            "value": "Semantic value.",
        }
    )
    _left = (
        log.filter(log.col("side") == "left")
        .select(["tenant", "position", "value"])
        .alias("left")
    )
    _right = (
        log.filter(log.col("side") == "right")
        .select(["tenant", "position", "value"])
        .alias("right")
    )
    pairs = _left.sem_join(
        _right,
        on=[
            _left.col("tenant") == _right.col("tenant"),
            _left.col("position") < _right.col("position"),
        ],
        instruction="{value:left} and {value:right} mean the same thing.",
        how="inner",
    )


class _RetractingSemanticJoinMemory(am.Memory):
    log = am.Log({"side": "Join side.", "key": "Key.", "value": "Value."})
    _minimum = log.group_by(["side", "key"]).min(
        column="value",
        output_col="value",
    )
    _left = (
        _minimum.filter(_minimum.col("side") == "left")
        .select(["key", "value"])
        .alias("left")
    )
    _right = (
        _minimum.filter(_minimum.col("side") == "right")
        .select(["key", "value"])
        .alias("right")
    )
    pairs = _left.sem_join(
        _right,
        on="key",
        instruction="{value:left} and {value:right} mean the same thing.",
        how="inner",
    )


class _SemanticSelfJoinMemory(am.Memory):
    log = am.Log({"key": "Exact partition.", "value": "Semantic value."})
    _left = log.select(["key", "value"]).alias("left")
    _right = log.select(["key", "value"]).alias("right")
    pairs = _left.sem_join(
        _right,
        on="key",
        instruction="{value:left} and {value:right} mean the same thing.",
        how="inner",
    )


class _TrackingSemanticAdapter(LotusAdapter):
    """Use deterministic semantic decisions while recording candidate counts."""

    def __init__(self) -> None:
        super().__init__()
        self.verified_pair_counts: list[int] = []

    def verify_candidates(self, candidates: pd.DataFrame) -> list[tuple[Any, Any, None]]:
        """Accept candidates whose projected semantic values are equal."""

        if candidates.empty:
            return []
        self.verified_pair_counts.append(len(candidates))
        return [
            (
                row[PAIR_LEFT_ID_COLUMN],
                row[PAIR_RIGHT_ID_COLUMN],
                None,
            )
            for _index, row in candidates.iterrows()
            if row[PAIR_LEFT_TEXT_COLUMN] == row[PAIR_RIGHT_TEXT_COLUMN]
        ]


def _only_semantic_join_node(
    memory_type: type[am.Memory],
) -> DifferentialNode:
    nodes = [
        node
        for node in memory_type.differentiate_policy().nodes.values()
        if node.query.op == "sem_join"
    ]
    assert len(nodes) == 1
    return nodes[0]


def _query_nodes(query: QueryExpr) -> list[QueryExpr]:
    return [query, *(node for item in query.inputs for node in _query_nodes(item))]


def _bag(frame: pd.DataFrame) -> Counter[tuple[object, ...]]:
    return Counter(tuple(row) for row in frame.itertuples(index=False, name=None))


def _full_view(memory: am.Memory, view_name: str) -> pd.DataFrame:
    query = memory.spec().views[view_name].query
    return memory._runtime._engine.adapter.execute(
        query,
        {"log": memory._runtime._state["log"]},
    )


def _patch_semantic_verifier(
    monkeypatch: pytest.MonkeyPatch,
    adapter: _TrackingSemanticAdapter,
) -> None:
    """Replace only the LLM verdict while retaining production join execution."""

    import agent_memory.adapters.lotus.sem_join as sem_join_module

    def verify(
        candidates: pd.DataFrame,
        *,
        left_label: str,
        right_label: str,
        instruction: str,
        config: LotusExecutionConfig,
    ) -> list[tuple[Any, Any, None]]:
        del left_label, right_label, instruction, config
        return adapter.verify_candidates(candidates)

    monkeypatch.setattr(sem_join_module, "verify_semantic_join_candidates", verify)


def test_sem_join_reuses_general_join_on_normalization() -> None:
    left = Relation(
        QueryExpr(
            op="materialized_view",
            params={"name": "left", "columns": ("tenant", "position", "value")},
        )
    ).alias("left")
    right = Relation(
        QueryExpr(
            op="materialized_view",
            params={"name": "right", "columns": ("tenant", "position", "value")},
        )
    ).alias("right")
    on = [
        left.col("tenant") == right.col("tenant"),
        left.col("position") < right.col("position"),
    ]

    semantic = left.sem_join(right, on=on, instruction="Rows match.")
    relational = left.join(right, on=on)

    assert semantic.expr.params["on"] == relational.expr.params["on"]


def test_general_join_on_restricts_candidates_before_semantic_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_join as sem_join_module

    left = Relation(
        QueryExpr(
            op="materialized_view",
            params={"name": "left", "columns": ("tenant", "position", "value")},
        )
    ).alias("left")
    right = Relation(
        QueryExpr(
            op="materialized_view",
            params={"name": "right", "columns": ("tenant", "position", "value")},
        )
    ).alias("right")
    query = left.sem_join(
        right,
        on=[
            left.col("tenant") == right.col("tenant"),
            left.col("position") < right.col("position"),
        ],
        instruction="{value:left} and {value:right} mean the same thing.",
        how="outer",
    ).expr
    captured: list[pd.DataFrame] = []

    def verify(
        candidates: pd.DataFrame,
        **_kwargs: object,
    ) -> list[tuple[Any, Any, None]]:
        captured.append(candidates.copy())
        return [
            (
                row[PAIR_LEFT_ID_COLUMN],
                row[PAIR_RIGHT_ID_COLUMN],
                None,
            )
            for _index, row in candidates.iterrows()
        ]

    monkeypatch.setattr(sem_join_module, "verify_semantic_join_candidates", verify)
    inputs = {
        "left": pd.DataFrame(
            {
                "tenant": ["a", "b"],
                "position": [1, 1],
                "value": ["left-a", "left-b"],
            },
            index=pd.Index([10, 20]),
        ),
        "right": pd.DataFrame(
            {
                "tenant": ["a", "a", "c"],
                "position": [2, 0, 3],
                "value": ["right-a-new", "right-a-old", "right-c"],
            },
            index=pd.Index([100, 200, 300]),
        ),
    }

    class Context:
        config = LotusExecutionConfig()
        pair_embedding_provider = None

        def configure(self) -> None:
            pass

    context: Any = Context()
    result = execute_sem_join(query, inputs, LotusAdapter().execute, context)

    assert len(captured) == 1
    assert captured[0].loc[:, [PAIR_LEFT_ID_COLUMN, PAIR_RIGHT_ID_COLUMN]].to_dict(
        orient="records"
    ) == [{PAIR_LEFT_ID_COLUMN: 10, PAIR_RIGHT_ID_COLUMN: 100}]
    assert len(result) == 4


def test_compiler_builds_bag_preserving_semantic_inner_join_maintenance() -> None:
    node = _only_semantic_join_node(_GeneralSemanticJoinMemory)

    assert node.execution_kind == "semantic_binary_state"
    assert node.maintenance_query is not None
    nodes = _query_nodes(node.maintenance_query)
    joins = [query for query in nodes if query.op == "sem_join"]
    concats = [query for query in nodes if query.op == "concat"]
    materialized_names = {
        str(query.params["name"])
        for query in nodes
        if query.op == "materialized_view"
    }

    assert len(joins) == 3
    assert len(concats) == 3
    assert all(query.params == node.query.params for query in joins)
    assert all(query.inputs[0].params["name"] == "left" for query in joins)
    assert all(query.inputs[1].params["name"] == "right" for query in joins)
    assert node.node_id in materialized_names
    for parent_id in node.input_node_ids:
        assert parent_id in materialized_names
        assert f"{parent_id}__inserted" in materialized_names


def test_semantic_pair_site_inventory_includes_all_maintenance_copies() -> None:
    policy = _GeneralSemanticJoinMemory.differentiate_policy()

    sites = inventory_operator_semantic_pair_sites(
        policy,
        operators=("sem_join",),
    )

    assert len(sites) == 1
    [site] = sites.values()
    assert len(site.query_digests) == 4


def test_semantic_inner_join_maintains_only_new_pairs_and_matches_full_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrackingSemanticAdapter()
    _patch_semantic_verifier(monkeypatch, adapter)
    memory = _GeneralSemanticJoinMemory(adapter=adapter)

    events = (
        {"side": "right", "tenant": "a", "position": 2, "value": "same"},
        {"side": "left", "tenant": "a", "position": 1, "value": "same"},
        {"side": "left", "tenant": "b", "position": 1, "value": "other"},
        {"side": "right", "tenant": "b", "position": 3, "value": "different"},
        {"side": "right", "tenant": "b", "position": 4, "value": "other"},
    )
    incremental_pair_counts: list[int] = []
    for event in events:
        before = len(adapter.verified_pair_counts)
        memory.add(event)
        incremental_pair_counts.extend(adapter.verified_pair_counts[before:])
        assert _bag(memory._runtime._state["pairs"]) == _bag(
            _full_view(memory, "pairs")
        )

    assert _bag(memory._runtime._state["pairs"]) == Counter(
        {
            ("a", 1, "same", "a", 2, "same"): 1,
            ("b", 1, "other", "b", 4, "other"): 1,
        }
    )
    assert incremental_pair_counts == [1, 1, 1]


def test_semantic_inner_join_rejects_retracting_parent_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrackingSemanticAdapter()
    _patch_semantic_verifier(monkeypatch, adapter)
    memory = _RetractingSemanticJoinMemory(adapter=adapter)

    memory.add({"side": "right", "key": "k", "value": 5})
    memory.add({"side": "left", "key": "k", "value": 5})

    with pytest.raises(NotImplementedError, match="append-only"):
        memory.add({"side": "left", "key": "k", "value": 3})


def test_semantic_self_join_counts_simultaneous_delta_pairs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrackingSemanticAdapter()
    _patch_semantic_verifier(monkeypatch, adapter)
    memory = _SemanticSelfJoinMemory(adapter=adapter)

    memory.add({"key": "all", "value": "same"})
    assert len(memory._runtime._state["pairs"]) == 1

    before = len(adapter.verified_pair_counts)
    memory.add({"key": "all", "value": "same"})

    assert adapter.verified_pair_counts[before:] == [1, 1, 1]
    assert _bag(memory._runtime._state["pairs"]) == Counter(
        {("all", "same", "all", "same"): 4}
    )
    assert _bag(memory._runtime._state["pairs"]) == _bag(
        _full_view(memory, "pairs")
    )


def test_semantic_inner_join_snapshot_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrackingSemanticAdapter()
    _patch_semantic_verifier(monkeypatch, adapter)
    original = _SemanticSelfJoinMemory(adapter=adapter)
    original.add({"key": "all", "value": "same"})
    snapshot = original._runtime.snapshot_state()

    restored = _SemanticSelfJoinMemory(adapter=adapter)
    restored._runtime.restore_state(snapshot)
    restored.add({"key": "all", "value": "same"})

    assert _bag(restored._runtime._state["pairs"]) == Counter(
        {("all", "same", "all", "same"): 4}
    )
    assert _bag(restored._runtime._state["pairs"]) == _bag(
        _full_view(restored, "pairs")
    )


@pytest.mark.parametrize(
    ("how", "k", "message"),
    [
        ("outer", None, "how='inner'"),
        ("inner", 1, "k=None"),
    ],
)
def test_semantic_join_differential_rejects_unsupported_contracts(
    how: str,
    k: int | None,
    message: str,
) -> None:
    class UnsupportedSemanticJoinMemory(am.Memory):
        log = am.Log({"value": "Value."})
        pairs = log.sem_join(
            log,
            instruction="{value:left} and {value:right} match.",
            how=how,
            k=k,
        )

    with pytest.raises(NotImplementedError, match=message):
        UnsupportedSemanticJoinMemory.differentiate_policy()
