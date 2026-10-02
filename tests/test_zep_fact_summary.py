"""Offline checks of the distinct fact-derived summary condition."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus import relational
from agent_memory.memories.zep.fact_summary import SUMMARY_SPEC, ZepFactSummaryMemory
from agent_memory.memories.zep.policy import ZepMemory
from agent_memory.planner.physical import optimize_policy, walk
from agent_memory.policy.logical import QueryExpr


def test_query_and_execution_identity_are_distinct() -> None:
    original = ZepMemory.differentiate_policy()
    policy = ZepFactSummaryMemory.differentiate_policy()
    adapter = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-fact-summary"))
    prepared = adapter.prepare_policy(policy)
    assert policy.fingerprint != original.fingerprint
    assert optimize_policy(prepared, strategy="zep-fact-summary").fingerprint == prepared.fingerprint
    assert any(n.query.params.get("fact_summary") for n in prepared.nodes.values())
    from agent_memory.policy.schema import output_columns
    assert "summary" not in output_columns(ZepFactSummaryMemory._identities.expr)
    assert all("summary" not in q.params.get("columns", ()) for q in walk(ZepFactSummaryMemory._identities.expr) if q.op == "select")


@pytest.mark.parametrize("size", [0, 1999, 2000, 2001])
def test_summary_threshold_uses_only_required_model_work(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    query = next(q for q in walk(ZepFactSummaryMemory._summaries.expr) if q.op == "agg")
    query = replace(query, params={**query.params, "fact_summary": True})
    rows = pd.DataFrame({"entity_id": ["a"], "summary": ["x" * size]})
    rows.attrs["agent_memory_groupby_keys"] = ("entity_id",)
    calls = []
    def generate(spec, groups, *, context):
        calls.extend(groups)
        return [{"summary": "compressed"} for _ in groups]
    monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many", generate)
    result = relational.execute_agg(query, {}, lambda *_: rows, SimpleNamespace(config=LotusExecutionConfig()))
    assert len(calls) == int(size > 2000)
    assert result.iloc[0]["summary"] == ("compressed" if size > 2000 else "x" * size)


def test_summary_clips_overlong_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    query = next(q for q in walk(ZepFactSummaryMemory._summaries.expr) if q.op == "agg")
    query = replace(query, params={**query.params, "fact_summary": True})
    rows = pd.DataFrame({"entity_id": ["a"], "summary": ["x" * 2001]})
    rows.attrs["agent_memory_groupby_keys"] = ("entity_id",)
    monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many",
                        lambda *a, **kw: [{"summary": "x" * 1001}])
    result = relational.execute_agg(query, {}, lambda *_: rows, SimpleNamespace(config=LotusExecutionConfig()))
    assert result.iloc[0]["summary"] == "x" * 1000


def test_unregistered_aggregate_cannot_use_shortcut() -> None:
    query = QueryExpr(op="agg", inputs=(QueryExpr(op="group_by", params={"keys": ("entity_id",)}),),
                     params={"aggregates": (replace(SUMMARY_SPEC, instruction="different"),), "fact_summary": True})
    with pytest.raises(ValueError, match="registered"):
        relational.execute_agg(query, {}, lambda *_: pd.DataFrame({"entity_id": ["a"], "summary": ["fact"]}),
                               SimpleNamespace(config=LotusExecutionConfig()))


def test_summary_compresses_once_and_records_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = [pd.DataFrame({"summary": [text]}) for text in ("short", "a" * 2100, "b" * 2100)]
    calls = []
    events = []
    def generate(spec, selected, *, context):
        calls.append((spec, selected))
        return [{"summary": "valid"}, {"summary": "x" * 2034}] if len(calls) == 1 else [{"summary": "repaired"}]
    monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many", generate)
    monkeypatch.setattr("agent_memory.tracing.semantic.write_trace_event", lambda *args, **kwargs: events.append(kwargs))
    result = relational._fact_summary_values(groups, context=SimpleNamespace(config=LotusExecutionConfig()))
    assert result == [{"summary": "short"}, {"summary": "valid"}, {"summary": "x" * 1000}]
    assert len(calls) == 1
    assert calls[0][1][1] is groups[2]
    assert "1000" in calls[0][0].instruction
    assert "2000" not in calls[0][0].instruction
    event = next(e for e in events if e["event_type"] == "fact_summary_truncation")
    assert event["payload"]["raw_summary"] == "x" * 2034
    assert event["payload"]["before_chars"] == 2034
    assert event["payload"]["after_chars"] == 1000


@pytest.mark.parametrize("bad", [None, 42, "", "   "])
def test_summary_invalid_output_fails_without_length_retry(monkeypatch: pytest.MonkeyPatch, bad: object) -> None:
    calls = []
    def generate(*args, **kwargs):
        calls.append(1)
        return [{"summary": bad}]
    monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many", generate)
    with pytest.raises((TypeError, ValueError)):
        relational._fact_summary_values([pd.DataFrame({"summary": ["x" * 2100]})],
                                        context=SimpleNamespace(config=LotusExecutionConfig()))
    assert len(calls) == 1


@pytest.mark.parametrize("raw,expected", [
    ("x" * 1000, "x" * 1000),
    ("x" * 1076, "x" * 1000),
    ("First sentence. " + "x" * 1074, "First sentence."),
    ("x" * 999 + ". tail", "x" * 999 + "."),
    ("中文" * 600, "中文" * 500),
])
def test_native_sentence_boundary_policy(raw: str, expected: str) -> None:
    assert relational._truncate_summary(raw) == expected


def test_summary_non_string_and_wrong_count_fail_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    for invalid in ([{"summary": None}], []):
        calls = []
        def generate(*args, **kwargs):
            calls.append(1)
            return invalid
        monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many", generate)
        with pytest.raises((TypeError, ValueError)):
            relational._fact_summary_values([pd.DataFrame({"summary": ["x" * 2100]})],
                                            context=SimpleNamespace(config=LotusExecutionConfig()))
        assert len(calls) == 1


def test_real_dataflow_fact_summary_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import lotus
    import agent_memory as am
    from agent_memory.planner.physical import replace_query
    from agent_memory.policy.relation import Relation
    from test_physical_fusion import state

    raw = {**state("Alice likes tea.", 0), "name": "Alice", "entity_type": "Entity",
           "episode_id": "0", "entity_ordinal": 0, "fact_ordinal": 0,
           "content": "Not a summary: Bob dislikes rain."}
    source = am.Source({c: c for c in raw})
    identity = ZepFactSummaryMemory._identities.expr
    identity_agg = next(q for q in walk(identity) if q.op == "agg")
    identity = replace_query(identity, identity_agg.inputs[0].inputs[0], source.expr)
    facts = ZepFactSummaryMemory.facts.expr
    fact_agg = next(q for q in walk(facts) if q.op == "agg" and q.inputs[0].op == "sem_groupby")
    facts = replace_query(fact_agg, fact_agg.inputs[0].inputs[0], source.expr)
    entities = replace_query(ZepFactSummaryMemory.entities.expr, ZepFactSummaryMemory.facts.expr, facts)
    entities = replace_query(entities, ZepFactSummaryMemory._identities.expr, identity)
    calls = []
    class Provider:
        def __call__(self, prompts, **kwargs):
            answers = []
            for prompt in prompts:
                payload = json.loads(prompt[1]["content"])
                calls.append(payload)
                is_identity = set(payload["output_schema"]["states"][0]) == {"right_id", "name"}
                matches = [{"left_id": item["id"], "right_id": item["eligible_targets"][0] if is_identity else None}
                           for item in payload["incoming"]]
                states = [{"right_id": target, "name": "Alice"} for target in sorted({m["right_id"] for m in matches if m["right_id"] is not None})]
                answers.append(json.dumps({"matches": matches, "states": states}))
            return SimpleNamespace(outputs=answers)
    adapter = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-fact-summary"))
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    monkeypatch.setattr(lotus.settings, "lm", Provider())
    # This fixture exercises deduplicated facts and summaries, not contradictions.
    def make_flow():
        return am.SemanticDataflow(source=source, views={"entities": Relation(entities), "facts": Relation(facts)}, adapter=adapter)
    flow = make_flow()
    flow.apply(pd.DataFrame([raw]))
    assert flow.view("entities").iloc[0]["summary"] == ""  # entity_id is an occurrence ID, not fixture endpoint "a"
    assert not calls
    snapshot = flow.snapshot_state()
    restored = make_flow()
    restored.restore_state(snapshot)
    # Match the real occurrence-based ID to connect this fact to Alice.
    restored.apply(pd.DataFrame([{**raw, "add_seq": 1, "episode_id": "1", "source_entity_id": (0, 0),
                                  "fact": "Alice likes coffee."}]))
    summary = restored.view("entities").iloc[0]["summary"]
    assert summary == "Alice likes coffee."
    assert "Bob" not in summary
    assert all("summary" not in str(c["output_schema"]) for c in calls)
    compression_groups = []
    def compress(spec, groups, *, context):
        compression_groups.extend(groups)
        return [{"summary": "Alice: compact retained facts."} for _ in groups]
    monkeypatch.setattr(relational, "_execute_grouped_semantic_aggregate_spec_many", compress)
    restored.apply(pd.DataFrame([{**raw, "add_seq": 2, "episode_id": "2", "source_entity_id": (0, 0),
                                  "fact": "x" * 1990}]))
    assert len(compression_groups) == 1
    assert restored.view("entities").iloc[0]["summary"] == "Alice: compact retained facts."
    restarted = make_flow()
    restarted.restore_state(restored.snapshot_state())
    restarted.apply(pd.DataFrame([{**raw, "add_seq": 3, "episode_id": "3", "source_entity_id": (0, 0),
                                  "fact": "Alice likes cocoa."}]))
    assert len(compression_groups) == 1
    assert restarted.view("entities").iloc[0]["summary"] == "Alice: compact retained facts.\nAlice likes cocoa."


def test_restore_rejects_previous_query_and_strategy() -> None:
    from agent_memory.runtime.executor import PolicyExecutor
    previous = PolicyExecutor(ZepMemory.differentiate_policy(), adapter=LotusAdapter(
        config=LotusExecutionConfig(physical_fusion="zep-combined")))
    current = PolicyExecutor(ZepFactSummaryMemory.differentiate_policy(), adapter=LotusAdapter(
        config=LotusExecutionConfig(physical_fusion="zep-fact-summary")))
    with pytest.raises(ValueError, match="fingerprint"):
        current.restore_state(previous.snapshot_state())


def test_direct_summary_execution_preserves_duplicates_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_memory as am
    source = am.Source({"entity_id": "id", "summary": "fact text"})
    query = source.group_by("entity_id").agg(SUMMARY_SPEC)
    adapter = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-fact-summary"))
    monkeypatch.setattr(adapter._context, "configure", lambda: pytest.fail("unexpected model setup"))
    result = adapter.execute(query.expr, {"log": pd.DataFrame({"entity_id": ["a", "a"], "summary": ["same", "same"]})})
    assert result.iloc[0]["summary"] == "same\nsame"
    empty = adapter.execute(query.expr, {"log": pd.DataFrame(columns=["entity_id", "summary"])})
    assert empty.empty
    with pytest.raises(TypeError, match="string evidence"):
        adapter.execute(query.expr, {"log": pd.DataFrame({"entity_id": ["a"], "summary": [None]})})
def test_fact_summary_storage_and_retrieval_plan() -> None:
    from agent_memory.memories.zep.fact_summary import zep_storage_statements
    from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
    from agent_memory.planner import PolicyDifferentiator

    assert zep_storage_statements("none") is GRAPHITI_NEO4J_STATEMENTS
    statements = zep_storage_statements("zep-fact-summary")
    assert len(statements.statements) == 4
    assert statements.statements[1].query == ZepFactSummaryMemory.entities.expr
    assert statements.statements[2].query == ZepFactSummaryMemory.facts.expr
    assert statements.statements[3].query == ZepFactSummaryMemory._episode_entities.expr
    PolicyDifferentiator().differentiate(ZepFactSummaryMemory.spec(), statements=statements)
