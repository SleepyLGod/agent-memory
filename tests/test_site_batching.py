"""Site-only batching preserves screening boundaries and pair occurrences."""
from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.site_batching import PairFilterBatching
from agent_memory.memories.zep.policy import ZepMemory, _CONTRADICTORY_FACT_INSTRUCTION
from agent_memory.policy.logical import QueryExpr


def setting(size: int = 4) -> PairFilterBatching:
    return PairFilterBatching(("fact_id:later_added",), PromptBatching(max_tasks=size))


def query() -> QueryExpr:
    return QueryExpr(op="sem_filter", inputs=(QueryExpr(op="materialized_view", params={
        "name": "pairs", "columns": ("fact:earlier_added", "fact:later_added", "fact_id:later_added", "occurrence")}),),
        params={"instruction": _CONTRADICTORY_FACT_INSTRUCTION})


class Provider:
    max_tokens = 512
    cache = None

    def __init__(self, invalid: bool = False) -> None:
        self.calls: list[Any] = []
        self.invalid = invalid

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        outputs = []
        for prompt in prompts:
            data = json.loads(prompt[1]["content"])
            self.calls.append(data)
            rows = data["rows"]
            assert len(rows) <= 4
            outputs.append(json.dumps({"decisions": [{"row_id": row["row_id"], "keep": True}
                                                     for row in reversed(rows[1:] if self.invalid else rows)]}))
        return SimpleNamespace(outputs=outputs)


def adapter(monkeypatch: pytest.MonkeyPatch, *, invalid: bool = False) -> tuple[LotusAdapter, Provider]:
    import lotus
    lm = Provider(invalid)
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    a = LotusAdapter(config=LotusExecutionConfig(
        pair_filter_batching={semantic_pair_site_id(query()): setting()}, structured_parse_retries=0))
    monkeypatch.setattr(a._context, "configure", lambda: None)
    return a, lm


def rows() -> pd.DataFrame:
    return pd.DataFrame({"fact:earlier_added": ["same old"] * 7,
        "fact:later_added": ["new A", "new B", "new A", "new A", "new A", "new A", "new B"],
        "fact_id:later_added": [(1, 0), (2, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 0)],
        "occurrence": list(range(7))}, index=pd.Index([0] * 7))


def test_group_boundary_split_order_and_duplicate_occurrences(monkeypatch: pytest.MonkeyPatch) -> None:
    a, lm = adapter(monkeypatch)
    data = rows()
    result = a.execute(query(), {"pairs": data})
    pd.testing.assert_frame_equal(result, data)
    assert [len(call["rows"]) for call in lm.calls] == [4, 1, 2]
    assert all(not ("new A" in str(call) and "new B" in str(call)) for call in lm.calls)
    assert len(a.execute(query(), {"pairs": data.iloc[:0]})) == 0
    assert len(lm.calls) == 3


def test_missing_decision_fails_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    a, lm = adapter(monkeypatch, invalid=True)
    with pytest.raises(ValueError):
        a.execute(query(), {"pairs": rows().iloc[:1]})
    assert len(lm.calls) == 1


def test_unconfigured_predicate_remains_native(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus
    a, lm = adapter(monkeypatch)
    assert lotus.settings.lm is lm
    changed = replace(query(), params={"instruction": "Is {fact:earlier_added} true?"})
    calls = []
    monkeypatch.setattr(pd.DataFrame, "sem_filter", lambda frame, *args, **kw: calls.append(args) or frame.copy())
    a.execute(changed, {"pairs": rows()})
    assert len(calls) == 1 and not lm.calls


def test_zep_plan_binding_and_identity() -> None:
    config = LotusExecutionConfig(physical_fusion="zep-target-state",
        pair_filter_batching={semantic_pair_site_id(query()): setting()})
    a = LotusAdapter(config=config)
    plan = a.prepare_policy(ZepMemory.differentiate_policy())
    assert sum(n.maintenance_query is not None and n.maintenance_query.op == "fused_target_state"
               for n in plan.nodes.values()) == 1
    plain = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-target-state"))
    assert a.maintenance_execution_fingerprint != plain.maintenance_execution_fingerprint
    assert a.maintenance_execution_fingerprint != LotusAdapter(config=replace(config,
        pair_filter_batching={semantic_pair_site_id(query()): setting(2)})).maintenance_execution_fingerprint
    with pytest.raises(ValueError, match="not found"):
        LotusAdapter(config=replace(config, pair_filter_batching={"sem_filter:missing": setting()})).prepare_policy(ZepMemory.differentiate_policy())
    with pytest.raises(ValueError):
        replace(config, prompt_batching=PromptBatching(max_tasks=4))


def test_real_runtime_maintenance_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    a, lm = adapter(monkeypatch)
    source = am.Source({c: c for c in rows().columns})
    view = source.sem_filter(instruction=_CONTRADICTORY_FACT_INSTRUCTION)
    flow = am.SemanticDataflow(source=source, views={"result": view}, adapter=a)
    flow.apply(rows().iloc[:2])
    flow.apply(rows().iloc[2:])
    assert flow.view("result").occurrence.tolist() == list(range(7))
    assert len(lm.calls) == 4
    snapshot = flow.snapshot_state()
    plain = am.SemanticDataflow(source=source, views={"result": view}, adapter=LotusAdapter())
    with pytest.raises(ValueError, match="fingerprint"):
        plain.restore_state(snapshot)


@pytest.mark.parametrize("aggregate_batching", [None, PromptBatching(max_tasks=4)])
def test_fusion_and_site_batching_in_one_real_dataflow(monkeypatch: pytest.MonkeyPatch, aggregate_batching: PromptBatching | None) -> None:
    import lotus
    from agent_memory.policy.relation import Relation
    from agent_memory.policy.schema import output_columns
    from agent_memory.policy.aggregates import SemanticAggregateSpec
    from test_physical_fusion import state, fact_input

    calls = {"fusion": 0, "predicate": 0}
    class Model:
        max_tokens = 512
        cache = None
        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            outputs = []
            for prompt in prompts:
                payload = json.loads(prompt[1]["content"])
                if "incoming" in payload:
                    calls["fusion"] += 1
                    outputs.append(json.dumps({"matches": [{"left_id": r["id"], "right_id": None}
                        for r in payload["incoming"]], "states": []}))
                else:
                    calls["predicate"] += 1
                    outputs.append(json.dumps({"decisions": [{"row_id": r["row_id"], "keep": True}
                        for r in payload["rows"]]}))
            return SimpleNamespace(outputs=outputs)

    class OracleAdapter(LotusAdapter):
        def execute(self, query: QueryExpr, inputs: Any) -> Any:
            q = query
            if q.op == "agg" and q.inputs[0].op == "sem_groupby":
                frame = self.execute(q.inputs[0].inputs[0], inputs)
                return pd.DataFrame([state(r["content"], r["add_seq"]) for _, r in frame.iterrows()],
                                    columns=pd.Index(output_columns(q)))
            if q.op == "agg":
                assert not any(isinstance(s, SemanticAggregateSpec) for s in q.params["aggregates"])
            assert q.op != "sem_join", "fused matching was repeated"
            return super().execute(q, inputs)

    source = am.Source({c: c for c in fact_input("old", 0).columns})
    original = ZepMemory._deduplicated_facts.expr
    agg = original.inputs[0]
    facts = Relation(replace(original, inputs=(replace(agg,
        inputs=(replace(agg.inputs[0], inputs=(source.expr,)),)),)))
    earlier, later = facts.alias("earlier_added"), facts.alias("later_added")
    pairs = earlier.join(later, on=earlier.col("fact_id") < later.col("fact_id"))
    view = pairs.sem_filter(instruction=_CONTRADICTORY_FACT_INSTRUCTION)
    a = OracleAdapter(config=LotusExecutionConfig(physical_fusion="zep-target-state", structured_parse_retries=0,
        sem_agg_prompt_batching=aggregate_batching,
        pair_filter_batching={semantic_pair_site_id(view.expr): setting()}))
    monkeypatch.setattr(a._context, "configure", lambda: None)
    monkeypatch.setattr(lotus.settings, "lm", Model())
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    flow = am.SemanticDataflow(source=source, views={"facts": facts, "pairs": view}, adapter=a)
    for i in range(3):
        flow.apply(fact_input(f"fact-{i}", i))
    assert len(flow.view("facts")) == 3 and len(flow.view("pairs")) == 3
    assert calls["fusion"] == 2
    assert calls["predicate"] >= 2
