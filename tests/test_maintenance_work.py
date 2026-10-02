"""Regression checks for physical maintenance work, without a provider."""

from types import SimpleNamespace
from typing import Any
import json

import pandas as pd
import pytest

from agent_memory.adapters import LotusAdapter
from agent_memory.policy.logical import QueryExpr
from agent_memory.runtime.executor import NodeOutputUpdate, PolicyExecutor


@pytest.mark.parametrize("same_side", [False, True])
def test_inner_join_replacements_never_recompute_old_pairs(same_side: bool) -> None:
    left = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "a", "b"]})
    right = pd.DataFrame({"k": [1, 2], "w": ["x", "y"]})
    new_left = pd.DataFrame({"k": [1, 2, 3], "v": ["a", "B", "c"]})
    new_right = right if same_side else pd.DataFrame({"k": [1, 1, 3], "w": ["x", "z", "q"]})
    leaves = tuple(QueryExpr(op="materialized_view", params={"name": n, "columns": tuple(f.columns)})
                   for n, f in [("l", left), ("r", right)])
    query = QueryExpr(op="join", inputs=leaves, params={"on": ["k"], "how": "inner"})
    adapter = LotusAdapter()
    old = adapter.execute(query, {"l": left, "r": right})
    expected = adapter.execute(query, {"l": new_left, "r": new_right})
    node = SimpleNamespace(input_node_ids=("l", "r"), execution_kind="relational_state", query=query)
    executor: Any = SimpleNamespace(policy=SimpleNamespace(nodes={"j": node}), adapter=adapter,
                                    _node_state={"l": left, "r": right})
    executor._execute_deterministic = lambda *args: pytest.fail("full join fallback")
    executor._empty_node_frame = lambda name: executor._node_state[name].iloc[:0]
    executor._align_output = lambda name, frame: frame.loc[:, old.columns]
    from itertools import count
    ids = count()
    update = PolicyExecutor._execute_relational_join_update(executor, "j", old,
        (NodeOutputUpdate.between(left, new_left), NodeOutputUpdate.between(right, new_right)),
        allocate=lambda: f"new:{next(ids)}")
    delta = NodeOutputUpdate.between(update.output_rows, expected)
    assert delta.is_empty


def test_site_groups_use_one_provider_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_site_batching import adapter, query, rows
    a, lm = adapter(monkeypatch)
    batches: list[int] = []
    original = type(lm).__call__

    def record(self: Any, prompts: Any, **kwargs: Any) -> Any:
        batches.append(len(prompts))
        return original(self, prompts, **kwargs)

    monkeypatch.setattr(type(lm), "__call__", record)
    result = a.execute(query(), {"pairs": rows()})
    pd.testing.assert_frame_equal(result, rows())
    assert batches == [3]


def test_predicate_reuse_snapshot_and_current_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    import agent_memory as am
    from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
    from test_site_batching import adapter, query, rows

    a, lm = adapter(monkeypatch)
    a.config = replace(a.config, predicate_reuse_sites=(semantic_pair_site_id(query()),))
    a._context.config = a.config
    source = am.Source({c: c for c in rows().columns})
    view = source.sem_filter(instruction=str(query().params["instruction"]))
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=a)
    first = rows().iloc[:1].copy()
    flow.apply(first)
    calls = len(lm.calls)
    snapshot = flow.snapshot_state()
    restored = am.SemanticDataflow(source=source, views={"result": view}, adapter=a)
    restored.restore_state(snapshot)
    changed = first.copy()
    changed["occurrence"] = 99
    restored.apply(changed)
    assert len(lm.calls) == calls
    assert restored.view("result").occurrence.tolist() == [0, 99]
    changed["fact:earlier_added"] = "different text"
    restored.apply(changed)
    assert len(lm.calls) == calls + 1
    disabled, _ = adapter(monkeypatch)
    other = am.SemanticDataflow(source=source, views={"result": view}, adapter=disabled)
    with pytest.raises(ValueError, match="fingerprint"):
        other.restore_state(snapshot)


def test_metadata_replacement_in_real_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    import agent_memory as am
    from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
    from test_site_batching import adapter, query, rows

    a, lm = adapter(monkeypatch)
    a.config = replace(a.config, predicate_reuse_sites=(semantic_pair_site_id(query()),))
    a._context.config = a.config
    source = am.Source({c: c for c in rows().columns})
    grouped = source.group_by(["fact:earlier_added", "fact:later_added", "fact_id:later_added"]).array_agg(
        columns=["occurrence"], output_col="provenance")
    view = grouped.sem_filter(instruction=str(query().params["instruction"]))
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=a)
    flow.apply(rows().iloc[:1])
    calls = len(lm.calls)
    extra = rows().iloc[:1].copy()
    extra["occurrence"] = 99
    flow.apply(extra)
    assert len(lm.calls) == calls
    assert json.loads(flow.view("result").iloc[0].provenance) == [{"occurrence": 0}, {"occurrence": 99}]


def test_inequality_self_join_replacement_scales_with_changes() -> None:
    import agent_memory as am

    class Tracking(LotusAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.join_sizes: list[int] = []

        def execute(self, query: QueryExpr, inputs: Any) -> Any:
            result = super().execute(query, inputs)
            if query.op == "join":
                self.join_sizes.append(len(result))
            return result

    source = am.Source({"id": "id", "value": "number"})
    state = source.group_by("id").min(column="value", output_col="value")
    left, right = state.alias("l"), state.alias("r")
    view = left.join(right, on=left.col("id") < right.col("id"))
    adapter = Tracking()
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=adapter)
    data = pd.DataFrame({"id": list(range(40)), "value": [10] * 40})
    flow.apply(data)
    adapter.join_sizes.clear()
    delta = pd.DataFrame({"id": [0, 0, 39], "value": [5, 5, 3]})
    flow.apply(delta)
    expected = LotusAdapter().execute(view.expr, {"log": pd.concat([data, delta], ignore_index=True)})
    assert NodeOutputUpdate.between(flow.view("result"), expected).is_empty
    assert max(adapter.join_sizes) <= 2 * 40
    snapshot = flow.snapshot_state()
    restored = am.SemanticDataflow(source=source, views={"result": view}, adapter=LotusAdapter())
    restored.restore_state(snapshot)
    for value in [2, 1]:
        event = pd.DataFrame({"id": [20], "value": [value]})
        restored.apply(event)
        flow.apply(event)
        assert NodeOutputUpdate.between(flow.view("result"), restored.view("result")).is_empty


def test_reuse_boolean_null_direction_duplicates_and_failure() -> None:
    from agent_memory.adapters.lotus.predicate_reuse import reuse_predicate_decisions

    cache: dict[str, bool] = {}
    calls: list[int] = []

    def evaluate(frame: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
        calls.append(len(frame))
        return frame.loc[frame["a"] == "yes"].copy()

    data = pd.DataFrame({"a": ["yes", "yes", None, "no"], "b": ["no", "no", "x", "yes"], "meta": range(4)})
    result, _ = reuse_predicate_decisions(data, "Compare {a} to {b}", cache, evaluate)
    assert calls == [3] and result.meta.tolist() == [0, 1]
    data["meta"] += 10
    result, reused = reuse_predicate_decisions(data, "Compare {a} to {b}", cache, evaluate)
    assert calls == [3] and reused == {0, 1, 2, 3} and result.meta.tolist() == [10, 11]
    before = dict(cache)

    def fail(frame: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
        raise ValueError("provider failed")

    with pytest.raises(ValueError, match="provider failed"):
        reuse_predicate_decisions(data, "Changed instruction {a} {b}", cache, fail)
    assert cache == before
    empty, _ = reuse_predicate_decisions(data.iloc[:0], "Compare {a} to {b}", cache, fail)
    assert empty.empty


def test_combined_groupby32_aggregate4_reuse_prepares_without_model() -> None:
    from agent_memory.adapters.lotus.context import LotusExecutionConfig
    from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
    from agent_memory.adapters.lotus.prompt_batching import PromptBatching
    from agent_memory.memories.zep.policy import ZepMemory
    from test_site_batching import setting

    site = semantic_pair_site_id(ZepMemory._contradictory_fact_pairs.expr)
    config = LotusExecutionConfig(physical_fusion="zep-target-state",
        pair_filter_batching={site: setting()}, predicate_reuse_sites=(site,),
        sem_groupby_pair_batch_size=32, sem_agg_prompt_batching=PromptBatching(max_tasks=4))
    plan = LotusAdapter(config=config).prepare_policy(ZepMemory.differentiate_policy())
    assert any(n.execution_kind == "semantic_predicate" for n in plan.nodes.values())
    assert any(n.maintenance_query is not None and n.maintenance_query.op == "fused_target_state"
               for n in plan.nodes.values())


def test_warm_join_retraction_does_not_key_unchanged_pair_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import agent_memory as am
    import agent_memory.runtime.executor as module

    source = am.Source({"id": "number", "value": "number"})
    state = source.group_by("id").min(column="value", output_col="value")
    left, right = state.alias("l"), state.alias("r")
    view = left.join(right, on=left.col("id") < right.col("id"))
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=LotusAdapter())
    initial = pd.DataFrame({"id": range(80), "value": [10] * 80})
    flow.apply(initial)
    original = module._row_key
    scanned = 0

    def count_scan(row: Any) -> Any:
        nonlocal scanned
        if sys._getframe(1).f_code.co_name == "_apply_join_delta":
            scanned += 1
        return original(row)

    monkeypatch.setattr(module, "_row_key", count_scan)
    delta = pd.DataFrame({"id": [0], "value": [5]})
    flow.apply(delta)
    assert scanned == 0
    expected = LotusAdapter().execute(view.expr, {"log": pd.concat([initial, delta], ignore_index=True)})
    assert NodeOutputUpdate.between(flow.view("result"), expected).is_empty
    restored = am.SemanticDataflow(source=source, views={"result": view}, adapter=LotusAdapter())
    restored.restore_state(flow.snapshot_state())
    restored.apply(delta.assign(value=4))
    flow.apply(delta.assign(value=4))
    assert NodeOutputUpdate.between(flow.view("result"), restored.view("result")).is_empty
