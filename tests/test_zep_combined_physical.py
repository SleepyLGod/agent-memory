"""Opt-in combined physical lowering, without provider calls."""

import json

import pytest

from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.fusion import parse_resolution
from agent_memory.memories.zep.policy import ZepMemory
from agent_memory.planner.physical import optimize_policy, walk


def test_shared_context_only_once() -> None:
    from agent_memory.adapters.lotus.sem_filter_batch_prompting import (
        _BatchPromptingTask, _build_request,
    )
    tasks = tuple(_BatchPromptingTask(i, f"r{i}", f"old-{i}", "new-fact") for i in range(4))
    request = _build_request(tasks, claim="opposite", max_tokens=100)
    body = request.prompt[1]["content"]
    assert body.count("new-fact") == 1
    payload = json.loads(body)
    assert len(payload["rows"]) == 4
    assert "shared_context" in payload


def test_singleton_shortcut_does_not_touch_multiline_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    from agent_memory.adapters.lotus import relational
    template = ZepMemory._deduplicated_facts.expr.inputs[0]
    from dataclasses import replace
    query = replace(template, params={**template.params, "singleton_identity": True})
    # Use the real deterministic specs, with complete rows and explicit group IDs.
    from test_physical_fusion import state
    rows = pd.DataFrame([state("single", 0), state("first", 1), state("second", 2)])
    rows["episode_id"] = ["0", "1", "2"]
    rows["fact_ordinal"] = 0
    rows["content"] = rows.fact
    from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
    rows[GROUP_ID_COLUMN] = [0, 1, 1]
    rows.attrs["agent_memory_groupby_input_cols"] = ("relation_type", "fact")
    seen = []
    def aggregate(spec, groups, *, context):
        seen.extend(len(g) for g in groups)
        return [{"relation_type": "LIKES", "fact": "merged"} for _ in groups]
    monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many", aggregate)
    from types import SimpleNamespace
    context = SimpleNamespace(config=LotusExecutionConfig())
    result = relational.execute_agg(query, {}, lambda *_: rows, context)
    assert result.fact.tolist() == ["single", "merged"]
    assert seen == [2]


def test_identity_decisions_require_complete_ordered_candidates() -> None:
    from agent_memory.adapters.lotus.identity_reuse import IdentityDecisions
    cache: dict[str, bool] = {}
    decisions = IdentityDecisions(cache)
    key = decisions.key("match", "Alice", ["Alice", "Bob"])
    assert decisions.lookup(key, 2) is None
    decisions.store(key, 2, [0])
    assert decisions.lookup(key, 2) == (0,)
    assert decisions.lookup(decisions.key("match", "Alice", ["Bob", "Alice"]), 2) is None
    assert decisions.lookup(decisions.key("changed", "Alice", ["Alice", "Bob"]), 2) is None
    assert decisions.lookup(decisions.key("match", "Alice", ["Alice"]), 1) is None
    with pytest.raises(ValueError):
        decisions.store(key, 2, [2])


def test_group_identity_reuses_only_same_semantic_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    from types import SimpleNamespace
    from agent_memory.adapters.lotus import sem_groupby
    from agent_memory.adapters.lotus.identity_reuse import identity_scope
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(kwargs["docs"])
        return SimpleNamespace(outputs=[True] * len(kwargs["docs"]))
    monkeypatch.setattr(sem_groupby, "evaluate_group_match_batch", evaluate)
    rows = pd.DataFrame({"name": ["Alice", "Alicia"], "metadata": [1, 2]})
    cache: dict[str, bool] = {}
    with identity_scope(cache):
        assert sem_groupby.evaluate_group_matches(rows, input_cols=("name",), instruction="same {name}") == [(0, 1)]
        rows["metadata"] = [3, 4]
        assert sem_groupby.evaluate_group_matches(rows, input_cols=("name",), instruction="same {name}") == [(0, 1)]
        assert len(calls) == 1
        sem_groupby.evaluate_group_matches(rows.iloc[::-1].reset_index(drop=True), input_cols=("name",), instruction="same {name}")
        assert len(calls) == 2
        rows.loc[1, "name"] = "Bob"
        sem_groupby.evaluate_group_matches(rows, input_cols=("name",), instruction="same {name}")
        assert len(calls) == 3


def test_combined_identity_rejects_changed_model_on_restore() -> None:
    from agent_memory.adapters import LotusAdapter
    from agent_memory.runtime.executor import PolicyExecutor
    first = PolicyExecutor(ZepMemory.differentiate_policy(), adapter=LotusAdapter(model="model-a", config=LotusExecutionConfig(physical_fusion="zep-combined")))
    second = PolicyExecutor(ZepMemory.differentiate_policy(), adapter=LotusAdapter(model="model-b", config=LotusExecutionConfig(physical_fusion="zep-combined")))
    with pytest.raises(ValueError, match="fingerprint"):
        second.restore_state(first.snapshot_state())


def test_singleton_rewrite_preserves_screening_identity() -> None:
    from agent_memory.tracing.semantic import query_digest
    original = optimize_policy(ZepMemory.differentiate_policy(), strategy="zep-target-state")
    combined = optimize_policy(ZepMemory.differentiate_policy(), strategy="zep-combined")
    fact = next(n for n in original.nodes.values() if n.maintenance_query is not None and n.maintenance_query.op == "fused_target_state")
    rewritten = combined.nodes[fact.node_id].maintenance_query
    assert rewritten.params["join_profile_digest"] == query_digest(fact.maintenance_query.inputs[0])


def test_shared_site_preserves_order_duplicates_and_current_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    import pandas as pd
    from test_site_batching import adapter, query, rows
    a, lm = adapter(monkeypatch)
    site, setting = next(iter(a.config.pair_filter_batching.items()))
    a.config = replace(a.config, pair_filter_batching={site: replace(setting, shared_columns=("fact_later_added",))})
    a._context.config = a.config
    result = a.execute(query(), {"pairs": rows()})
    pd.testing.assert_frame_equal(result, rows())
    assert [len(c["rows"]) for c in lm.calls] == [4, 1, 2]
    assert all("shared_context" in c for c in lm.calls)
    assert all("new A" not in str(c["rows"]) and "new B" not in str(c["rows"]) for c in lm.calls)


def test_combined_real_runtime_snapshot_and_current_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    from typing import Any
    import pandas as pd
    import lotus
    import agent_memory as am
    from agent_memory.adapters import LotusAdapter
    from agent_memory.policy.aggregates import SemanticAggregateSpec
    from agent_memory.policy.relation import Relation
    from agent_memory.policy.logical import QueryExpr
    from test_physical_fusion import state

    calls: list[dict[str, Any]] = []
    class Provider:
        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            outputs = []
            for prompt in prompts:
                data = json.loads(prompt[1]["content"])
                calls.append(data)
                matches = [{"left_id": item["id"], "right_id": item["eligible_targets"][0]} for item in data["incoming"]]
                states = []
                for target in data["targets"]:
                    assigned = [item for item, match in zip(data["incoming"], matches) if match["right_id"] == target["id"]]
                    if not assigned:
                        continue
                    if "summary" in data["output_schema"]["states"][0]:
                        states.append({"right_id": target["id"], "name": "Alice", "summary": "current merged summary"})
                    else:
                        states.append({"right_id": target["id"], "relation_type": "LIKES", "fact": "current merged fact"})
                outputs.append(json.dumps({"matches": matches, "states": states}))
            return SimpleNamespace(outputs=outputs)

    class Oracle(LotusAdapter):
        def execute(self, query: QueryExpr, inputs: Any) -> Any:
            if query.op == "agg" and query.inputs[0].op == "sem_groupby":
                specs = [s for s in query.params["aggregates"] if isinstance(s, SemanticAggregateSpec)]
                if specs and tuple(c.name for c in specs[0].output_cols) == ("name", "summary"):
                    rows = super().execute(query.inputs[0], inputs)
                    deterministic = replace(query, params={"aggregates": tuple(s for s in query.params["aggregates"] if not isinstance(s, SemanticAggregateSpec))})
                    result = super().execute(deterministic, inputs)
                    result["name"] = [rows.iloc[0]["name"]]
                    result["summary"] = [rows.iloc[0]["content"]]
                    return result
            return super().execute(query, inputs)

    raw = {**state("fact", 0), "name": "Alice", "entity_type": "Entity", "episode_id": "0", "entity_ordinal": 0,
           "fact_ordinal": 0, "content": "old"}
    source = am.Source({c: c for c in raw})
    entity = ZepMemory.entities.expr
    entity_agg = next(q for q in walk(entity) if q.op == "agg")
    from agent_memory.planner.physical import replace_query
    entity = replace_query(entity, entity_agg.inputs[0].inputs[0], source.expr)
    fact = ZepMemory._deduplicated_facts.expr
    fact_agg = fact.inputs[0]
    fact = replace_query(fact, fact_agg.inputs[0].inputs[0], source.expr)
    adapter = Oracle(config=LotusExecutionConfig(physical_fusion="zep-combined"))
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    monkeypatch.setattr(lotus.settings, "lm", Provider())
    def make_flow():
        return am.SemanticDataflow(source=source, views={"entities": Relation(entity), "facts": Relation(fact)}, adapter=adapter)
    flow = make_flow()
    flow.apply(pd.DataFrame([raw]))
    assert not calls
    flow.apply(pd.DataFrame([{**raw, "add_seq": 1, "episode_id": "1", "content": "new"}]))
    assert len(calls) == 2
    assert flow.view("entities").iloc[0]["entity_id"] == (0, 0)
    assert len(json.loads(flow.view("entities").iloc[0]["mentions"])) == 2
    snapshot = flow.snapshot_state()
    assert snapshot["predicate_decisions"]
    restored = make_flow()
    restored.restore_state(snapshot)
    restored.apply(pd.DataFrame([{**raw, "add_seq": 2, "episode_id": "2", "content": "latest"}]))
    assert len(calls) == 4
    entity_calls = [c for c in calls if "summary" in c["output_schema"]["states"][0]]
    assert entity_calls[-1]["fixed_matches"] == {"l0": "r0"}
    assert "latest" in str(entity_calls[-1]["incoming"])
    assert len(json.loads(restored.view("entities").iloc[0]["mentions"])) == 3


def test_combined_rewrites_entities_and_facts_without_changing_view() -> None:
    original = ZepMemory.differentiate_policy()
    optimized = optimize_policy(original, strategy="zep-combined")
    assert optimize_policy(optimized, strategy="zep-combined") is optimized
    fused = [n for n in optimized.nodes.values()
             if n.maintenance_query is not None
             and n.maintenance_query.op == "fused_target_state"]
    assert len(fused) == 2
    assert original.spec is optimized.spec
    assert all(n.query == original.nodes[k].query for k, n in optimized.nodes.items())
    assert optimized.fingerprint != optimize_policy(original, strategy="zep-target-state").fingerprint
    assert LotusExecutionConfig(physical_fusion="zep-combined").physical_fusion == "zep-combined"
    marked = [q for n in fused for q in walk(n.maintenance_query)
              if q.params.get("singleton_identity")]
    assert len(marked) == 1
    assert marked[0].op == "agg"


def test_entity_resolution_strict_fields_and_many_to_one() -> None:
    raw = json.dumps({"matches": [
        {"left_id": "l0", "right_id": "r0"},
        {"left_id": "l1", "right_id": "r0"}],
        "states": [{"right_id": "r0", "name": "Alice", "summary": "Both facts."}]})
    result = parse_resolution(raw, {"l0": {"r0"}, "l1": {"r0"}},
                              state_columns=("name", "summary"))
    assert result["states"]["r0"] == {"name": "Alice", "summary": "Both facts."}
    with pytest.raises(ValueError):
        parse_resolution(raw.replace('"name": "Alice"', '"name": 3'),
                         {"l0": {"r0"}, "l1": {"r0"}}, state_columns=("name", "summary"))


@pytest.mark.parametrize("unknown_second", [False, True])
def test_cached_targets_prune_only_unneeded_state(
    monkeypatch: pytest.MonkeyPatch, unknown_second: bool,
) -> None:
    from dataclasses import replace
    import pandas as pd
    import test_physical_fusion as fixture
    from agent_memory.adapters.lotus.identity_reuse import identity_scope

    original = fixture.fused_query()
    monkeypatch.setattr(fixture, "fused_query", lambda: replace(
        original, params={**original.params, "identity_reuse": True}))
    old = pd.DataFrame([fixture.state("selected", 0), fixture.state("unselected", 1)])
    incoming = pd.DataFrame([fixture.state("new", 2)])
    cache: dict[str, bool] = {}
    with identity_scope(cache):
        fixture.execute_fused(monkeypatch, incoming, old, [fixture.response(
            [("l0", "r0")], [("r0", "merged")])])
        if unknown_second:
            incoming = pd.concat([incoming, pd.DataFrame([fixture.state("unknown", 3)])], ignore_index=True)
        assignments = [("l0", "r0")] + ([("l1", None)] if unknown_second else [])
        result, model = fixture.execute_fused(monkeypatch, incoming, old, [fixture.response(
            assignments, [("r0", "current merged")])])
    payload = json.loads(model.calls[0][0][0][1]["content"])
    assert payload["incoming"][0]["eligible_targets"] == ["r0"]
    assert payload["fixed_matches"] == {"l0": "r0"}
    assert [t["id"] for t in payload["targets"]] == (["r0", "r1"] if unknown_second else ["r0"])
    assert "unselected" in result.fact.tolist()
    assert "current merged" in result.fact.tolist()
