"""Offline stable-representative checks, including real incremental execution."""

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.memories.zep.representative import ZepRepresentativeMemory
from agent_memory.memories.zep.fact_summary import ZepFactSummaryMemory


@pytest.mark.parametrize("batch", [None, PromptBatching(max_tasks=16)])
def test_arg_min_direct_and_incremental_restore(batch):
    source = am.Source({"group": "key", "seq": "arrival", "ordinal": "ordinal", "fact": "text"})
    view = source.group_by("group").agg(am.arg_min(order_by=["seq", "ordinal"], columns=["fact"]))
    adapter = LotusAdapter(config=LotusExecutionConfig(prompt_batching=batch))
    rows = pd.DataFrame([
        {"group": "a", "seq": 2, "ordinal": 0, "fact": "later"},
        {"group": "a", "seq": 1, "ordinal": 1, "fact": "first"},
        {"group": "a", "seq": 1, "ordinal": 1, "fact": "first"},
        {"group": "b", "seq": 3, "ordinal": 0, "fact": "independent"},
    ])
    flow = am.SemanticDataflow(source=source, views={"facts": view}, adapter=adapter)
    for i in range(len(rows)):
        flow.apply(rows.iloc[[i]])
        expected = adapter.execute(view.expr, {"log": rows.iloc[:i+1]})
        pd.testing.assert_frame_equal(flow.view("facts").sort_values("group").reset_index(drop=True),
                                      expected.sort_values("group").reset_index(drop=True))
        restored = am.SemanticDataflow(source=source, views={"facts": view}, adapter=adapter)
        restored.restore_state(flow.snapshot_state())
        flow = restored


def test_representative_rebinds_profiles_and_retains_listwise() -> None:
    from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
    from agent_memory.evaluation.agent_memory_drivers import build_site_semantic_pair_profiles
    from agent_memory.evaluation.semantic_pair_config import SemanticPairSiteBinding
    from agent_memory.memories.zep.policy import ZepMemory
    from agent_memory.memories.zep.storage import GRAPHITI_BGE_M3
    from agent_memory.planner.physical import walk
    from agent_memory.tracing.semantic import query_digest

    old = ZepFactSummaryMemory.differentiate_policy()
    new = ZepRepresentativeMemory.differentiate_policy()
    old_joins = [q for n in old.nodes.values() if n.maintenance_query is not None
                for q in walk(n.maintenance_query) if q.op == "sem_join"]
    with pytest.raises(ValueError, match="not found"):
        build_site_semantic_pair_profiles(new, bindings=tuple(
            SemanticPairSiteBinding(semantic_pair_site_id(q), "search-filter", 5, None)
            for q in old_joins), operators=("sem_filter", "sem_join", "sem_groupby"),
            embedding=GRAPHITI_BGE_M3)
    joins = [q for n in new.nodes.values() if n.maintenance_query is not None
             for q in walk(n.maintenance_query) if q.op == "sem_join"]
    bindings = tuple(SemanticPairSiteBinding(semantic_pair_site_id(q), "search-filter", 5, None)
                     for q in joins)
    bindings += (SemanticPairSiteBinding(semantic_pair_site_id(ZepMemory._contradictory_fact_pairs.expr), "search-filter", 5, None),)
    profiles, _ = build_site_semantic_pair_profiles(new, bindings=bindings,
        operators=("sem_filter", "sem_join", "sem_groupby"), embedding=GRAPHITI_BGE_M3)
    adapter = LotusAdapter(config=LotusExecutionConfig(
        physical_fusion="zep-representative", semantic_pair_profiles=profiles,
        sem_join_topk_method="listwise", sem_groupby_pair_batch_size=32,
        sem_agg_prompt_batching=PromptBatching(max_tasks=16)))
    prepared = adapter.prepare_policy(new)
    fused = [n.maintenance_query for n in prepared.nodes.values()
             if n.maintenance_query is not None and n.maintenance_query.op == "fused_target_state"]
    assert len(fused) == 1
    assert profiles[fused[0].params["join_profile_digest"]].top_k == 5
    fact_joins = [q for n in prepared.nodes.values() if n.maintenance_query is not None and n.maintenance_query.op != "fused_target_state"
                  for q in walk(n.maintenance_query) if q.op == "sem_join"]
    assert fact_joins
    for query in fact_joins:
        assert query.params["k"] == 1
        assert profiles[query_digest(query)].top_k == 5
    bad = SemanticPairSiteBinding("sem_filter:missing", "search-filter", 5, None)
    with pytest.raises(ValueError, match="not found"):
        build_site_semantic_pair_profiles(new, bindings=(bad,),
            operators=("sem_filter", "sem_join", "sem_groupby"), embedding=GRAPHITI_BGE_M3)


def test_representative_keeps_distinct_details_and_multiple_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from agent_memory.adapters.lotus import adapter as adapter_module
    from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
    from agent_memory.adapters.lotus.sem_join import assemble_join_frame
    from agent_memory.planner.physical import replace_query, walk
    from agent_memory.policy.aggregates import ArgMinAggregateSpec
    from agent_memory.policy.relation import Relation
    from agent_memory.runtime.executor import NodeOutputUpdate

    equivalent = {"Alice enjoys tea.": "Alice likes tea."}

    def key(text: str) -> str:
        return equivalent.get(text, text)

    def group(query, inputs, execute, context):
        frame = execute(query.inputs[0], inputs).copy()
        frame[GROUP_ID_COLUMN] = pd.factorize(frame.fact.map(key))[0]
        frame.attrs["agent_memory_groupby_input_cols"] = tuple(query.params["input_cols"])
        frame.attrs["agent_memory_sem_groupby_partition_by"] = tuple(query.params.get("partition_by", ()))
        return frame

    def join(query, inputs, execute, context):
        left, right = [execute(q, inputs) for q in query.inputs]
        assert query.params["k"] == 1
        matches = [(i, j, None) for i in left.index for j in right.index
                   if key(left.loc[i, "fact"]) == key(right.loc[j, "fact"])]
        return assemble_join_frame(left, right, matches, how="outer", id_columns=tuple(query.params["id_columns"]))

    monkeypatch.setattr(adapter_module, "execute_sem_groupby", group)
    monkeypatch.setattr(adapter_module, "execute_sem_join", join)
    aggregate = next(q for q in walk(ZepRepresentativeMemory.facts.expr)
                     if any(isinstance(s, ArgMinAggregateSpec) for s in q.params.get("aggregates", ())))
    raw = dict(source_entity_id="a", target_entity_id="b", relation_type="LIKES", fact="Alice likes tea.",
               episode_id="0", fact_ordinal=0, add_seq=0, created_at="2026-01-01",
               valid_at=None, invalid_at=None, expired_at=None, content="original")
    source = am.Source({c: c for c in raw})
    view = Relation(replace_query(aggregate, aggregate.inputs[0].inputs[0], source.expr))
    adapter = LotusAdapter()
    monkeypatch.setattr(adapter._context, "configure", lambda: pytest.fail("unexpected model request"))
    flow = am.SemanticDataflow(source=source, views={"facts": view}, adapter=adapter)
    rows = []
    for i, text in enumerate(["Alice likes tea.", "Alice likes coffee.", "Alice enjoys tea.", "Alice likes green tea.", "Alice likes coffee."]):
        rows.append({**raw, "fact": text, "episode_id": str(i), "add_seq": i, "content": f"source-{i}"})
        flow.apply(pd.DataFrame([rows[-1]]))
        actual = flow.view("facts")
        expected = adapter.execute(view.expr, {"log": pd.DataFrame(rows)})
        for frame in (actual, expected):
            frame["provenance"] = frame.provenance.map(lambda s: json.dumps(
                sorted(json.loads(s), key=lambda r: r["episode_id"]), sort_keys=True))
        assert NodeOutputUpdate.between(actual, expected).is_empty
        assert sum(len(json.loads(s)) for s in actual.provenance) == i + 1
        restored = am.SemanticDataflow(source=source, views={"facts": view}, adapter=adapter)
        restored.restore_state(flow.snapshot_state())
        flow = restored
    assert set(flow.view("facts").fact) == {"Alice likes tea.", "Alice likes coffee.", "Alice likes green tea."}


def test_arg_min_rejects_ambiguous_and_null_keys():
    source = am.Source({"g": "group", "seq": "order", "fact": "text"})
    query = source.group_by("g").agg(am.arg_min(order_by=["seq"], columns=["fact"]))
    for rows, error in [([("a", 1, "x"), ("a", 1, "y")], "conflicting"),
                        ([("a", None, "x")], "null")]:
        with pytest.raises(ValueError, match=error):
            LotusAdapter().execute(query.expr, {"log": pd.DataFrame(rows, columns=pd.Index(["g", "seq", "fact"]))})


def test_representative_view_has_new_identity_and_no_fact_synthesis():
    from agent_memory.planner.physical import walk
    from agent_memory.policy.aggregates import ArgMinAggregateSpec
    query = ZepRepresentativeMemory.facts.expr
    assert any(isinstance(s, ArgMinAggregateSpec) for q in walk(query) for s in q.params.get("aggregates", ()))
    policy = ZepRepresentativeMemory.differentiate_policy()
    assert policy.fingerprint != ZepFactSummaryMemory.differentiate_policy().fingerprint
    node = next(n for n in policy.nodes.values() if any(isinstance(s, ArgMinAggregateSpec) for s in n.query.params.get("aggregates", ())))
    assert node.execution_kind == "semantic_state"
    assert node.maintenance_query is not None
    from agent_memory.memories.zep.fact_summary import zep_memory_type, zep_storage_statements
    assert zep_memory_type('zep-representative') is ZepRepresentativeMemory
    assert zep_storage_statements('zep-representative').statements[2].query == ZepRepresentativeMemory.facts.expr
    adapter = LotusAdapter(config=LotusExecutionConfig(physical_fusion='zep-representative'))
    prepared = adapter.prepare_policy(policy)
    assert adapter.prepare_policy(prepared).fingerprint == prepared.fingerprint
    fused = [n for n in prepared.nodes.values() if n.maintenance_query is not None and n.maintenance_query.op == 'fused_target_state']
    assert len(fused) == 1  # Entity identity only, no model-generated fact state.
    with pytest.raises(ValueError, match='representative logical view'):
        adapter.prepare_policy(ZepFactSummaryMemory.differentiate_policy())


def test_semantic_join_map_preserves_text_and_combines_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from agent_memory.adapters.lotus import adapter as adapter_module
    from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
    from agent_memory.adapters.lotus.sem_join import assemble_join_frame
    from agent_memory.planner.physical import replace_query, walk
    from agent_memory.policy.aggregates import ArgMinAggregateSpec
    from agent_memory.policy.relation import Relation
    from agent_memory.runtime.executor import NodeOutputUpdate

    def group(query, inputs, execute, context):
        frame = execute(query.inputs[0], inputs).copy()
        frame[GROUP_ID_COLUMN] = 0
        frame.attrs['agent_memory_groupby_input_cols'] = tuple(query.params['input_cols'])
        frame.attrs['agent_memory_sem_groupby_partition_by'] = tuple(query.params.get('partition_by', ()))
        return frame

    def join(query, inputs, execute, context):
        left, right = [execute(q, inputs) for q in query.inputs]
        matches = [(i, j, None) for i in left.index for j in right.index]
        return assemble_join_frame(left, right, matches, how='outer', id_columns=tuple(query.params['id_columns']))

    monkeypatch.setattr(adapter_module, 'execute_sem_groupby', group)
    monkeypatch.setattr(adapter_module, 'execute_sem_join', join)
    agg = next(q for q in walk(ZepRepresentativeMemory.facts.expr)
               if any(isinstance(s, ArgMinAggregateSpec) for s in q.params.get('aggregates', ())))
    raw = dict(source_entity_id='a', target_entity_id='b', relation_type='LIKES', fact='Alice likes tea.',
               episode_id='0', fact_ordinal=0, add_seq=0, created_at='2026-01-01',
               valid_at=None, invalid_at=None, expired_at=None, content='original')
    source = am.Source({c: c for c in raw})
    view = Relation(replace_query(agg, agg.inputs[0].inputs[0], source.expr))
    adapter = LotusAdapter()
    monkeypatch.setattr(adapter._context, 'configure', lambda: pytest.fail('unexpected model request'))
    flow = am.SemanticDataflow(source=source, views={'facts': view}, adapter=adapter)
    rows = []
    for i, fact in enumerate(['Alice likes tea.', 'Alice enjoys tea.', 'Alice likes drinking tea.']):
        rows.append({**raw, 'add_seq': i, 'episode_id': str(i), 'fact': fact})
        flow.apply(pd.DataFrame([rows[-1]]))
        current = flow.view('facts')
        assert current.iloc[0].fact == rows[0]['fact']
        assert len(json.loads(current.iloc[0].provenance)) == i + 1
        expected = adapter.execute(view.expr, {'log': pd.DataFrame(rows)})
        # Existing provenance aggregation retains occurrences, not arrival order.
        for frame in (current, expected):
            frame['provenance'] = frame['provenance'].map(lambda value: json.dumps(
                sorted(json.loads(value), key=lambda record: record['episode_id']), sort_keys=True))
        assert NodeOutputUpdate.between(current, expected).is_empty
        restored = am.SemanticDataflow(source=source, views={'facts': view}, adapter=adapter)
        restored.restore_state(flow.snapshot_state())
        flow = restored
