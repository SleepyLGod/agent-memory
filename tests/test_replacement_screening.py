"""Fixed-oracle regressions for Top-k membership across source replacements."""

from math import cos, sin
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.pair_execution import SemanticPairExecutionProfile, semantic_pair_site_id
from agent_memory.planner.physical import walk
from agent_memory.runtime.executor import NodeOutputUpdate
from agent_memory.storage.embedding import EmbeddingSpec
from agent_memory.tracing.semantic import query_digest


class FixedVectors:
    calls: int = 0

    def embed(self, spec: EmbeddingSpec, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        angles = {"A": 1.2, "B": .1, "C": .2, "D": .3, "E": .4, "F": .5, "G": 0, "H": .05}
        return [[cos(angles[t.rsplit(': ', 1)[-1]]), sin(angles[t.rsplit(': ', 1)[-1]])] for t in texts]


def build(monkeypatch: pytest.MonkeyPatch, direction: str = "right-to-left") -> tuple[Any, ...]:
    source = am.Source({"id": "number", "fact": "text", "episode": "text"})
    state = source.group_by(["id", "fact"]).array_agg(columns=["episode"], output_col="provenance")
    left, right = state.alias("earlier"), state.alias("later")
    view = left.join(right, on=left.col("id") < right.col("id")).sem_filter(
        instruction="Compare {fact:earlier} and {fact:later}.")
    policy = am.SemanticDataflow(source=source, views={"result": view}, adapter=LotusAdapter())._executor.policy
    digests = {query_digest(q) for n in policy.nodes.values()
               for root in (n.query, n.maintenance_query) if root is not None
               for q in walk(root) if q.op == "sem_filter"}
    profile = SemanticPairExecutionProfile(mode="search-filter", direction=direction,
        left_id_columns=("id:earlier",), right_id_columns=("id:later",),
        left_text_columns=("fact:earlier",), right_text_columns=("fact:later",),
        embedding=EmbeddingSpec("fact", "vector", "fake", "v1", 2, True), top_k=5)
    vectors = FixedVectors()
    adapter = LotusAdapter(config=LotusExecutionConfig(
        semantic_pair_profiles={d: profile for d in digests},
        predicate_reuse_sites=(semantic_pair_site_id(view.expr),)), pair_embedding_provider=vectors)
    calls: list[tuple[str, str]] = []

    def oracle(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.extend(zip(frame["fact_earlier"], frame["fact_later"]))
        return frame.copy()

    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    monkeypatch.setattr(pd.DataFrame, "sem_filter", oracle)
    full = LotusAdapter(config=LotusExecutionConfig(
        semantic_pair_profiles={query_digest(view.expr): profile}), pair_embedding_provider=FixedVectors())
    monkeypatch.setattr(full._context, "configure", lambda: None)
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=adapter)
    return source, view, flow, adapter, full, vectors, calls


@pytest.mark.parametrize("direction", ["right-to-left", "left-to-right", "symmetric"])
def test_metadata_only_replacement_keeps_full_topk_and_makes_no_calls(monkeypatch: pytest.MonkeyPatch, direction: str) -> None:
    source, view, flow, adapter, full, vectors, calls = build(monkeypatch, direction)
    initial = pd.DataFrame({"id": range(7), "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    count, embedded = len(calls), vectors.calls
    extra = initial.iloc[[0]].assign(episode="another source")
    flow.apply(extra)
    assert len(calls) == count
    assert vectors.calls == embedded
    expected = full.execute(view.expr, {"log": pd.concat([initial, extra], ignore_index=True)})
    assert NodeOutputUpdate.between(flow.view("result"), expected).is_empty
    restored = am.SemanticDataflow(source=source, views={"result": view}, adapter=adapter)
    restored.restore_state(flow.snapshot_state())
    count = len(calls)
    restored.apply(extra.assign(episode="third source"))
    assert len(calls) == count
    expected = full.execute(view.expr, {"log": pd.concat([initial, extra, extra.assign(episode="third source")], ignore_index=True)})
    assert NodeOutputUpdate.between(restored.view("result"), expected).is_empty


def test_new_stronger_candidate_retracts_displaced_old_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _, view, flow, _, full, _, _ = build(monkeypatch)
    # H arrives later but has an earlier logical ID; it displaces a candidate of G.
    initial = pd.DataFrame({"id": [0, 2, 3, 4, 5, 6, 7], "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    extra = pd.DataFrame({"id": [1], "fact": ["H"], "episode": ["new"]})
    flow.apply(extra)
    expected = full.execute(view.expr, {"log": pd.concat([initial, extra], ignore_index=True)})
    assert NodeOutputUpdate.between(flow.view("result"), expected).is_empty


def test_mixed_update_does_not_rescreen_metadata_only_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_memory.adapters.lotus import sem_filter as filter_module

    _, view, flow, _, full, _, _ = build(monkeypatch)
    initial = pd.DataFrame({"id": range(7), "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    screened: list[tuple[str, str]] = []
    select = filter_module.select_semantic_pair_candidates

    def record(source: pd.DataFrame, **kwargs: Any) -> Any:
        screened.extend(zip(source["fact:earlier"], source["fact:later"]))
        return select(source, **kwargs)

    monkeypatch.setattr(filter_module, "select_semantic_pair_candidates", record)
    extra = pd.DataFrame({"id": [0, 7], "fact": ["A", "H"], "episode": ["new source", "new fact"]})
    flow.apply(extra)
    assert screened == [(fact, "H") for fact in "ABCDEFG"]
    expected = full.execute(view.expr, {"log": pd.concat([initial, extra], ignore_index=True)})
    assert NodeOutputUpdate.between(flow.view("result"), expected).is_empty


@pytest.mark.parametrize("direction", ["right-to-left", "left-to-right", "symmetric"])
def test_mixed_candidate_changes_match_full_after_restore(monkeypatch: pytest.MonkeyPatch, direction: str) -> None:
    source, view, flow, adapter, full, _, _ = build(monkeypatch, direction)
    initial = pd.DataFrame({"id": [0, 2, 3, 4, 5, 6, 7], "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    restored = am.SemanticDataflow(source=source, views={"result": view}, adapter=adapter)
    restored.restore_state(flow.snapshot_state())
    # A's metadata changes while H competes in existing candidate buckets.
    extra = pd.DataFrame({"id": [0, 1], "fact": ["A", "H"], "episode": ["new source", "new fact"]})
    restored.apply(extra)
    expected = full.execute(view.expr, {"log": pd.concat([initial, extra], ignore_index=True)})
    assert NodeOutputUpdate.between(restored.view("result"), expected).is_empty


def test_representative_matching_declares_resolved_time_on_both_paths() -> None:
    from agent_memory.memories.zep.representative import ZepRepresentativeMemory
    from agent_memory.policy.aggregates import ArgMinAggregateSpec

    agg = next(q for q in walk(ZepRepresentativeMemory.facts.expr)
               if any(isinstance(s, ArgMinAggregateSpec) for s in q.params.get("aggregates", ())))
    assert "valid_at" in agg.inputs[0].params["input_cols"]
    assert "{valid_at}" in agg.inputs[0].params["instruction"]
    policy = ZepRepresentativeMemory.differentiate_policy()
    joins = [q for node in policy.nodes.values() if node.maintenance_query is not None
             for q in walk(node.maintenance_query) if q.op == "sem_join"
             and "relation_type:left" in q.params["instruction"]]
    assert joins
    assert all("{valid_at:left}" in q.params["instruction"] and "{valid_at:right}" in q.params["instruction"] for q in joins)


def test_representative_full_grouping_sends_resolved_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import lotus
    from agent_memory.adapters.lotus.prompt_batching import PromptBatching
    from agent_memory.adapters.lotus.sem_groupby import execute_sem_groupby, GROUP_ID_COLUMN
    from agent_memory.memories.zep.representative import ZepRepresentativeMemory
    from agent_memory.policy.aggregates import ArgMinAggregateSpec
    from test_zep_clean_maintenance import Oracle, offline_context

    group = next(q.inputs[0] for q in walk(ZepRepresentativeMemory.facts.expr)
                 if any(isinstance(s, ArgMinAggregateSpec) for s in q.params.get("aggregates", ())))
    rows = pd.DataFrame({"source_entity_id": ["a", "a"], "target_entity_id": ["b", "b"],
                         "relation_type": ["ATTENDED"] * 2,
                         "fact": ["Alice attended a parade last Friday.", "Alice attended a parade last week."],
                         "valid_at": ["2023-06-02", "2023-08-04"]})
    config = LotusExecutionConfig(groupby_prompt_batching={semantic_pair_site_id(group): PromptBatching(max_tasks=16)})
    oracle = Oracle()
    monkeypatch.setattr(lotus.settings, "lm", oracle)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    result = execute_sem_groupby(group, {}, lambda *_: rows, offline_context(config, monkeypatch))
    assert result[GROUP_ID_COLUMN].nunique() == 2  # Fake oracle is False; not a model quality assertion.
    assert len(oracle.requests) == 1
    prompt = json.dumps(oracle.requests[0])
    assert "2023-06-02" in prompt and "2023-08-04" in prompt and "valid_at" in prompt


@pytest.mark.parametrize("direction", ["right-to-left", "left-to-right", "symmetric"])
def test_removal_promotion_text_change_duplicates_and_empty(monkeypatch: pytest.MonkeyPatch, direction: str) -> None:
    _, _, flow, adapter, _, _, _ = build(monkeypatch, direction)
    executor = flow._executor
    node_id, node = next((i, n) for i, n in executor.policy.nodes.items() if n.execution_kind == "semantic_predicate")
    parent = node.input_node_ids[0]
    base = pd.DataFrame({
        "id:earlier": range(6), "fact:earlier": list("ABCDEF"),
        "provenance:earlier": ["original"] * 6,
        "id:later": [6] * 6, "fact:later": ["G"] * 6,
        "provenance:later": ["original"] * 6,
    }, index=pd.Index([f"p{i}" for i in range(6)]))
    duplicate = base.iloc[[0]].copy()
    duplicate.index = pd.Index(["duplicate"])
    frames = [base, base.drop(index="p1"), pd.concat([base.drop(index="p1"), duplicate])]
    modified = frames[-1].copy()
    modified.loc[modified["id:earlier"] == 0, "fact:earlier"] = "H"
    frames.extend([modified, modified.iloc[:0], base])
    cache: dict = {}
    decisions: dict = {}
    previous = base.iloc[:0]
    result = previous
    for current in frames:
        executor._node_state[parent] = previous
        result = executor._execute_semantic_row(node_id, old_state=result,
            parent_update=NodeOutputUpdate.between(previous, current), staged_cache=cache,
            predicate_decisions=decisions)
        expected = adapter.execute(node.query, {parent: current})
        assert NodeOutputUpdate.between(result, expected).is_empty
        previous = current


@pytest.mark.parametrize("mixed", [False, True])
def test_failure_leaves_join_index_membership_and_predicates_uncommitted(monkeypatch: pytest.MonkeyPatch, mixed: bool) -> None:
    from copy import deepcopy

    source, view, flow, adapter, _, _, _ = build(monkeypatch)
    initial = pd.DataFrame({"id": [0, 2, 3, 4, 5, 6, 7], "fact": list("ABCDEFG"), "episode": ["original"] * 7})
    flow.apply(initial)
    snapshot = flow.snapshot_state()
    indexes = deepcopy(flow._executor._join_row_indexes)
    cache = deepcopy(flow._executor._predicate_decisions)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic predicate failure")

    monkeypatch.setattr(pd.DataFrame, "sem_filter", fail)
    with pytest.raises(ValueError, match="synthetic predicate failure"):
        extra = pd.DataFrame({"id": [1], "fact": ["H"], "episode": ["new"]})
        if mixed:
            extra = pd.DataFrame({"id": [0, 8], "fact": ["A", "H"], "episode": ["new source", "new fact"]})
        flow.apply(extra)
    assert flow._executor._join_row_indexes == indexes
    assert flow._executor._predicate_decisions == cache
    restored = am.SemanticDataflow(source=source, views={"result": view}, adapter=adapter)
    restored.restore_state(snapshot)
    pd.testing.assert_frame_equal(flow.view("result"), restored.view("result"))
    assert restored._executor._join_row_indexes == {}
    for node_id, cached in restored._executor._semantic_output_cache.items():
        current = flow._executor._semantic_output_cache[node_id]
        assert current.keys() == cached.keys()
        for occurrence, frame in cached.items():
            pd.testing.assert_frame_equal(current[occurrence], frame)
    snapshot["adapter_execution_fingerprint"] = "old-invocation-local-top5"
    with pytest.raises(ValueError, match="fingerprint"):
        restored.restore_state(snapshot)
