"""Site-scoped listwise packing preserves each anchor's candidate domain."""

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_join import execute_sem_join
from test_sem_topk_join import _query
from test_zep_clean_maintenance import offline_context
from agent_memory.policy.logical import QueryExpr


def site_query() -> QueryExpr:
    query = _query(k=1, on=("tenant",))
    return QueryExpr(query.op, query.inputs, {**query.params, "instruction": "{name:left} and {name:right} describe the same entity."})


class PackedLM:
    max_ctx_len = 100_000
    max_tokens = 1024
    cache = None

    def __init__(self, invalid: str = "") -> None:
        self.requests: list[dict[str, Any]] = []
        self.invalid = invalid

    def count_tokens(self, messages: Any) -> int:
        return len(str(messages))

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        outputs = []
        for prompt in messages:
            payload = json.loads(prompt[1]["content"])
            self.requests.append(payload)
            tasks = payload["tasks"]
            results = [{"task_id": t["task_id"], "selected_ids": [t["right_candidates"][0]["id"]]}
                       for t in reversed(tasks)]
            if self.invalid == "missing":
                results.pop()
            elif self.invalid == "duplicate":
                results.append(results[0])
            elif self.invalid == "cross-task":
                results[0]["selected_ids"] = [tasks[0]["right_candidates"][0]["id"]]
            outputs.append(json.dumps({"results": results}))
        return SimpleNamespace(outputs=outputs)


def test_site_packs_anchors_without_changing_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus

    query = site_query()
    site = semantic_pair_site_id(query)
    config = LotusExecutionConfig(listwise_join_batching={site: PromptBatching(max_tasks=16)})
    left = pd.DataFrame({"tenant": list(range(17)), "name": ["same"] * 17}, index=range(100, 117))
    right = pd.DataFrame({"tenant": list(range(17)), "name": ["same"] * 17}, index=range(200, 217))
    lm = PackedLM()
    monkeypatch.setattr(lotus.settings, "lm", lm)
    output = execute_sem_join(query, {}, lambda q, _: left if q.params["name"] == "left" else right,
                              offline_context(config, monkeypatch))
    assert len(output) == 17  # Identical text never deduplicates occurrences.
    assert [len(p["tasks"]) for p in lm.requests] == [16, 1]
    for request in lm.requests:
        assert request["join_condition"] == query.params["instruction"]
        assert all(len(t["right_candidates"]) == 1 for t in request["tasks"])
        assert all("messages" not in t and "join_condition" not in t for t in request["tasks"])
    assert config.prompt_batching is None
    # A different semantic site still gets the original unbatched path.
    assert site not in LotusExecutionConfig().listwise_join_batching


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "cross-task"])
def test_site_rejects_invalid_or_cross_task_results(monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    import lotus

    query = site_query()
    config = LotusExecutionConfig(listwise_join_batching={semantic_pair_site_id(query): PromptBatching(max_tasks=16)},
                                  structured_parse_retries=0)
    rows = pd.DataFrame({"tenant": [1, 2], "name": ["same", "same"]})
    monkeypatch.setattr(lotus.settings, "lm", PackedLM(invalid))
    with pytest.raises(ValueError):
        execute_sem_join(query, {}, lambda *_: rows, offline_context(config, monkeypatch))


def test_site_execution_identity_and_plan_validation() -> None:
    from agent_memory.memories.zep.representative import ZepRepresentativeMemory
    from agent_memory.planner.physical import walk
    from agent_memory.runtime.executor import PolicyExecutor

    policy = ZepRepresentativeMemory.differentiate_policy()
    fact_join = next(q for n in policy.nodes.values() if n.maintenance_query is not None
                     for q in walk(n.maintenance_query) if q.op == "sem_join" and q.params.get("on"))
    site = semantic_pair_site_id(fact_join)
    old = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-representative"))
    packed = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-representative",
        listwise_join_batching={site: PromptBatching(max_tasks=16)}))
    packed.prepare_policy(policy)
    assert old.maintenance_execution_fingerprint != packed.maintenance_execution_fingerprint
    with pytest.raises(ValueError, match="fingerprint"):
        PolicyExecutor(policy, adapter=packed).restore_state(PolicyExecutor(policy, adapter=old).snapshot_state())
    invalid = LotusAdapter(config=LotusExecutionConfig(listwise_join_batching={"sem_join:missing": PromptBatching(max_tasks=16)}))
    with pytest.raises(ValueError, match="not found"):
        invalid.prepare_policy(policy)


def test_site_rejects_global_or_pairwise_configuration() -> None:
    setting = {"sem_join:test": PromptBatching(max_tasks=16)}
    with pytest.raises(ValueError):
        LotusExecutionConfig(listwise_join_batching=setting, prompt_batching=PromptBatching())
    with pytest.raises(ValueError):
        LotusExecutionConfig(listwise_join_batching=setting, sem_join_topk_method="pairwise-naive")


def test_other_sites_and_empty_candidates_do_not_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus
    from test_sem_topk_join import _BatchListwiseLM

    config = LotusExecutionConfig(listwise_join_batching={"sem_join:other": PromptBatching(max_tasks=16)})
    rows = pd.DataFrame({"name": ["x", "y"]})
    lm = _BatchListwiseLM(['{"selected_ids": []}', '{"selected_ids": []}'])
    monkeypatch.setattr(lotus.settings, "lm", lm)
    result = execute_sem_join(_query(k=1), {}, lambda *_: rows, offline_context(config, monkeypatch))
    assert result.empty
    assert len(lm.calls[0][0]) == 2
    assert all("tasks" not in json.loads(prompt[1]["content"]) for prompt in lm.calls[0][0])
    config = LotusExecutionConfig(listwise_join_batching={semantic_pair_site_id(site_query()): PromptBatching(max_tasks=16)})
    empty_lm = PackedLM()
    monkeypatch.setattr(lotus.settings, "lm", empty_lm)
    for empty in (False, True):
        left = pd.DataFrame({"tenant": [1], "name": ["x"]})
        right = pd.DataFrame({"tenant": [2], "name": ["x"]})
        if empty:
            left = left.iloc[:0]
        result = execute_sem_join(site_query(), {}, lambda q, _: left if q.params["name"] == "left" else right,
                                  offline_context(config, monkeypatch))
        assert result.empty
    assert not empty_lm.requests


def test_real_runtime_uses_packed_join_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus
    import agent_memory as am
    from agent_memory.adapters.lotus import adapter as adapter_module
    from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
    from agent_memory.memories.zep.representative import ZepRepresentativeMemory
    from agent_memory.planner.physical import replace_query, walk
    from agent_memory.policy.aggregates import ArgMinAggregateSpec
    from agent_memory.policy.relation import Relation
    from agent_memory.runtime.executor import NodeOutputUpdate

    def group(query: Any, inputs: Any, execute: Any, context: Any) -> pd.DataFrame:
        frame = execute(query.inputs[0], inputs).copy()
        frame[GROUP_ID_COLUMN] = pd.factorize(frame.fact)[0]
        frame.attrs["agent_memory_groupby_input_cols"] = tuple(query.params["input_cols"])
        frame.attrs["agent_memory_sem_groupby_partition_by"] = tuple(query.params.get("partition_by", ()))
        return frame

    monkeypatch.setattr(adapter_module, "execute_sem_groupby", group)
    aggregate = next(q for q in walk(ZepRepresentativeMemory.facts.expr)
                     if any(isinstance(s, ArgMinAggregateSpec) for s in q.params.get("aggregates", ())))
    raw = dict(source_entity_id="a", target_entity_id="b", relation_type="LIKES", fact="Alice likes tea.",
               episode_id="0", fact_ordinal=0, add_seq=0, created_at="2026-01-01",
               valid_at=None, invalid_at=None, expired_at=None, content="original")
    source = am.Source({c: c for c in raw})
    view = Relation(replace_query(aggregate, aggregate.inputs[0].inputs[0], source.expr))
    # Obtain the actual generated site, rather than guessing a digest.
    full_policy = ZepRepresentativeMemory.differentiate_policy()
    site = next(semantic_pair_site_id(q) for n in full_policy.nodes.values() if n.maintenance_query is not None
                for q in walk(n.maintenance_query) if q.op == "sem_join" and q.params.get("on"))
    adapter = LotusAdapter(config=LotusExecutionConfig(listwise_join_batching={site: PromptBatching(max_tasks=16)}))
    flow = am.SemanticDataflow(source=source, views={"facts": view}, adapter=adapter)
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    lm = PackedLM()
    monkeypatch.setattr(lotus.settings, "lm", lm)
    rows: list[dict[str, Any]] = []
    for seq in range(2):
        batch = [{**raw, "episode_id": str(seq), "add_seq": seq, "valid_at": f"2026-01-0{seq + 1}"},
                 {**raw, "source_entity_id": "c", "target_entity_id": "d", "fact": "Carol likes coffee.",
                  "episode_id": str(seq), "add_seq": seq, "fact_ordinal": 1, "valid_at": f"2026-01-0{seq + 1}"}]
        rows.extend(batch)
        flow.apply(pd.DataFrame(batch))
        actual = flow.view("facts")
        expected = adapter.execute(view.expr, {"log": pd.DataFrame(rows)})
        for frame in (actual, expected):
            frame["provenance"] = frame.provenance.map(lambda s: json.dumps(sorted(json.loads(s), key=lambda r: r["episode_id"]), sort_keys=True))
        assert NodeOutputUpdate.between(actual, expected).is_empty
        restored = am.SemanticDataflow(source=source, views={"facts": view}, adapter=adapter)
        restored.restore_state(flow.snapshot_state())
        flow = restored
    assert [len(request["tasks"]) for request in lm.requests] == [2]
    for task in lm.requests[0]["tasks"]:
        incoming = json.dumps(task["left"])
        previous = json.dumps(task["right_candidates"])
        assert "valid_at" in incoming and "2026-01-02" in incoming
        assert "valid_at" in previous and "2026-01-01" in previous
