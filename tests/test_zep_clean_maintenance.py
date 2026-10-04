"""Offline checks for bounded maintenance shortcuts and real prompt packing."""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig, LotusExecutionContext
from agent_memory.adapters.lotus.fusion import _unchanged_entity_states
from agent_memory.adapters.lotus.pair_execution import semantic_pair_site_id
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN, execute_sem_groupby
from agent_memory.memories.zep.representative import ZepRepresentativeMemory
from agent_memory.planner.physical import replace_query, walk
from agent_memory.policy.aggregates import ArgMinAggregateSpec
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.relation import Relation


class Oracle:
    """Independent false predicates, exact entity assignments, and no fact matches."""

    max_tokens = 1024
    max_ctx_len = 100_000
    cache = None

    def __init__(self, invalid: str = "") -> None:
        self.requests: list[dict[str, Any]] = []
        self.invalid = invalid

    def count_tokens(self, messages: Any) -> int:
        return len(str(messages))

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        outputs = []
        for prompt in prompts:
            data = json.loads(prompt[1]["content"])
            self.requests.append(data)
            if "rows" in data:
                decisions = [{"row_id": row["row_id"], "keep": False} for row in reversed(data["rows"])]
                if self.invalid == "missing":
                    decisions.pop()
                elif self.invalid == "duplicate":
                    decisions.append(decisions[0])
                elif self.invalid == "boolean":
                    decisions[0]["keep"] = "false"
                outputs.append(json.dumps({"decisions": decisions}))
            elif "incoming" in data:
                matches = [{"left_id": row["id"], "right_id": row["eligible_targets"][0]}
                           for row in data["incoming"]]
                targets = {m["right_id"] for m in matches}
                states = [{"right_id": row["id"], "name": row["state"]["name"]}
                          for row in data["targets"] if row["id"] in targets]
                outputs.append(json.dumps({"matches": matches, "states": states}))
            else:
                outputs.append('{"selected_ids": []}')
        return SimpleNamespace(outputs=outputs)


def grouping(partitioned: bool = False) -> QueryExpr:
    return QueryExpr("sem_groupby", (QueryExpr("materialized_view", params={"name": "rows"}),),
                     {"input_cols": ("name",), "instruction": "The same entity: {name}.",
                      "partition_by": ("tenant",) if partitioned else ()})


def offline_context(config: LotusExecutionConfig, monkeypatch: pytest.MonkeyPatch) -> LotusExecutionContext:
    context = LotusExecutionContext(model="fake", config=config)
    monkeypatch.setattr(context, "configure", lambda: None)
    return context


def test_groupby_prompt16_preserves_partitions_duplicates_and_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import lotus
    from agent_memory.adapters.lotus.identity_reuse import identity_scope

    query = grouping(True)
    config = LotusExecutionConfig(groupby_prompt_batching={semantic_pair_site_id(query): PromptBatching(max_tasks=16)},
                                  sem_groupby_pair_batch_size=32, semantic_trace_dir=tmp_path)
    rows = pd.DataFrame({"tenant": [0] * 8 + [1] * 2,
                         "name": [f"first-{i}" for i in range(7)] + ["first-0", "second-a", "second-b"],
                         "metadata": list(range(10))})
    oracle = Oracle()
    monkeypatch.setattr(lotus.settings, "lm", oracle)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    context = offline_context(config, monkeypatch)
    cache: dict[str, bool] = {}
    with identity_scope(cache):
        result = execute_sem_groupby(query, {}, lambda *_: rows, context)
        assert [len(r["rows"]) for r in oracle.requests] == [16, 5, 1]
        assert all(not ("first-" in str(r) and "second-" in str(r)) for r in oracle.requests)
        assert result.metadata.tolist() == list(range(10))
        assert result[GROUP_ID_COLUMN].nunique() == 9
        assert result.iloc[0][GROUP_ID_COLUMN] == result.iloc[7][GROUP_ID_COLUMN]
        assert config.prompt_batching is None and config.sem_groupby_pair_batch_size == 32
        rows.metadata += 100
        reused = execute_sem_groupby(query, {}, lambda *_: rows, context)
        assert len(oracle.requests) == 3
        assert reused.metadata.tolist() == list(range(100, 110))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    batching = [event for event in events if event["event_type"] == "prompt_batching"]
    assert sum(event["prompt_count"] for event in batching) == 3
    assert sum(event["task_count"] for event in batching) == 22
    assert [size for event in batching for size in event["chunk_sizes"]] == [16, 5, 1]


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "boolean"])
def test_groupby_invalid_results_fail(monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    import lotus
    query = grouping()
    config = LotusExecutionConfig(groupby_prompt_batching={semantic_pair_site_id(query): PromptBatching(max_tasks=16)},
                                  structured_parse_retries=0)
    oracle = Oracle(invalid)
    monkeypatch.setattr(lotus.settings, "lm", oracle)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    with pytest.raises(ValueError):
        execute_sem_groupby(query, {}, lambda *_: pd.DataFrame({"name": ["a", "b", "c"]}), offline_context(config, monkeypatch))
    assert len(oracle.requests) == 1


def test_empty_and_unselected_grouping_keep_original_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_memory.adapters.lotus import sem_groupby
    query = grouping()
    seen: list[int] = []
    def evaluate(*args: Any, **kwargs: Any) -> Any:
        seen.append(len(kwargs["docs"]))
        return SimpleNamespace(outputs=[False] * len(kwargs["docs"]))
    monkeypatch.setattr(sem_groupby, "evaluate_group_match_batch", evaluate)
    config = LotusExecutionConfig(groupby_prompt_batching={"sem_groupby:other": PromptBatching(max_tasks=16)},
                                  sem_groupby_pair_batch_size=2)
    rows = pd.DataFrame({"name": ["a", "b", "c"]})
    execute_sem_groupby(query, {}, lambda *_: rows, offline_context(config, monkeypatch))
    assert seen == [2, 1]
    empty_config = replace(config, groupby_prompt_batching={semantic_pair_site_id(query): PromptBatching(max_tasks=16)})
    result = execute_sem_groupby(query, {}, lambda *_: rows.iloc[:0], offline_context(empty_config, monkeypatch))
    assert result.empty and seen == [2, 1]


@pytest.mark.parametrize("names,fixed,expected", [
    (["Alice", "Alice", "new"], {"l0": "r0", "l1": "r0", "l2": None}, {"r0": {"name": "Alice"}}),
    (["Alice", "Alice Smith", "new"], {"l0": "r0", "l1": "r0", "l2": None}, None),
    (["Alice", "Alice", "new"], {"l0": "r0", "l2": None}, None),
    ([None, "Alice", "new"], {"l0": "r0", "l1": "r0", "l2": None}, None),
])
def test_name_shortcut_requires_every_decision_and_identical_names(names: list[Any], fixed: dict[str, Any], expected: Any) -> None:
    left = pd.DataFrame({"name": names}, index=pd.Index([10, 11, 12]))
    right = pd.DataFrame({"name": ["Alice", "unrelated"]}, index=pd.Index([20, 21]))
    assert _unchanged_entity_states({"l0": {"r0"}, "l1": {"r0"}, "l2": {"r1"}}, fixed,
        left, right, {"l0": 10, "l1": 11, "l2": 12}, {"r0": 20, "r1": 21}) == expected


def make_identity_flow(monkeypatch: pytest.MonkeyPatch, oracle: Oracle, enabled: bool,
                       trace_dir: Path, packed: bool = False) -> tuple[Any, dict[str, Any]]:
    import lotus
    from test_physical_fusion import state
    raw = {**state("fact", 0), "name": "Alice", "entity_type": "Entity", "episode_id": "0",
           "entity_ordinal": 0, "fact_ordinal": 0, "content": "original"}
    source = am.Source({c: c for c in raw})
    identities = ZepRepresentativeMemory._identities.expr
    aggregate = next(q for q in walk(identities) if q.op == "agg")
    identities = replace_query(identities, aggregate.inputs[0].inputs[0], source.expr)
    # Keep the actual representative fact aggregate in the registered plan.
    facts = next(q for q in walk(ZepRepresentativeMemory.facts.expr)
                 if any(isinstance(s, ArgMinAggregateSpec) for s in q.params.get("aggregates", ())))
    facts = replace_query(facts, facts.inputs[0].inputs[0], source.expr)
    sites = {semantic_pair_site_id(q): PromptBatching(max_tasks=16)
             for view in (identities, facts) for q in walk(view) if q.op == "sem_groupby"} if packed else {}
    adapter = LotusAdapter(config=LotusExecutionConfig(physical_fusion="zep-representative",
        reuse_unchanged_entity_name=enabled, semantic_trace_dir=trace_dir,
        groupby_prompt_batching=sites, sem_groupby_pair_batch_size=32))
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    monkeypatch.setattr(lotus.settings, "lm", oracle)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    return am.SemanticDataflow(source=source, views={"entities": Relation(identities), "facts": Relation(facts)}, adapter=adapter), raw


@pytest.mark.parametrize("enabled", [False, True])
def test_real_identity_runtime_reuses_after_restore_and_propagates_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: bool) -> None:
    oracle = Oracle()
    flow, raw = make_identity_flow(monkeypatch, oracle, enabled, tmp_path)
    for seq in range(3):
        flow.apply(pd.DataFrame([{**raw, "add_seq": seq, "episode_id": str(seq), "content": f"source-{seq}"}]))
        saved = flow.snapshot_state()
        flow, _ = make_identity_flow(monkeypatch, oracle, enabled, tmp_path)
        flow.restore_state(saved)
    requests = [r for r in oracle.requests if "incoming" in r]
    assert len(requests) == (1 if enabled else 2)
    row = flow.view("entities").iloc[0]
    assert row["name"] == "Alice" and row["entity_id"] == (0, 0)
    assert len(json.loads(row["mentions"])) == 3
    assert "source-2" in row["mentions"]
    if enabled:
        assert "unchanged_entity_name" in (tmp_path / "events.jsonl").read_text()
    # A different name must return to the model even after a match is cached.
    for seq in (3, 4):
        flow.apply(pd.DataFrame([{**raw, "name": "Alice Smith", "add_seq": seq, "episode_id": str(seq)}]))
    assert len([r for r in oracle.requests if "incoming" in r]) == len(requests) + 2


def test_new_options_reject_incompatible_restores_and_unbound_sites() -> None:
    from agent_memory.runtime.executor import PolicyExecutor
    policy = ZepRepresentativeMemory.differentiate_policy()
    sites = {semantic_pair_site_id(q) for n in policy.nodes.values() for q in walk(n.query) if q.op == "sem_groupby"}
    assert len(sites) == 2
    base = LotusExecutionConfig(physical_fusion="zep-representative", sem_groupby_pair_batch_size=32)
    old = PolicyExecutor(policy, adapter=LotusAdapter(config=base))
    for config in (replace(base, reuse_unchanged_entity_name=True),
                   replace(base, groupby_prompt_batching={s: PromptBatching(max_tasks=16) for s in sites})):
        current = PolicyExecutor(policy, adapter=LotusAdapter(config=config))
        with pytest.raises(ValueError, match="fingerprint"):
            current.restore_state(old.snapshot_state())
        restored = PolicyExecutor(policy, adapter=LotusAdapter(config=config))
        restored.restore_state(current.snapshot_state())
    bad = LotusAdapter(config=replace(base, groupby_prompt_batching={"sem_groupby:missing": PromptBatching(max_tasks=16)}))
    with pytest.raises(ValueError, match="not found"):
        bad.prepare_policy(policy)
    invalid_configs: tuple[dict[str, Any], ...] = ({"physical_fusion": "disabled", "reuse_unchanged_entity_name": True},
                   {"reuse_unchanged_entity_name": 1},
                   {"groupby_prompt_batching": {"sem_join:bad": PromptBatching(max_tasks=16)}},
                   {"groupby_prompt_batching": {"sem_groupby:bad": PromptBatching()}},
                   {"groupby_prompt_batching": {"sem_groupby:x": PromptBatching(max_tasks=16)}, "sem_groupby_default": True},
                   {"groupby_prompt_batching": {"sem_groupby:x": PromptBatching(max_tasks=16)}, "prompt_batching": PromptBatching()})
    for kwargs in invalid_configs:
        with pytest.raises((ValueError, TypeError)):
            LotusExecutionConfig(**kwargs)


def test_real_runtime_packs_both_grouping_sites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    oracle = Oracle()
    flow, raw = make_identity_flow(monkeypatch, oracle, True, tmp_path, packed=True)
    rows = pd.DataFrame([{**raw, "name": f"person-{i}", "fact": f"fact-{i}",
                          "entity_ordinal": i, "fact_ordinal": i} for i in range(7)])
    flow.apply(rows)
    assert [len(r["rows"]) for r in oracle.requests] == [16, 5, 16, 5]
    assert len(flow.view("entities")) == len(flow.view("facts")) == 7
    saved = flow.snapshot_state()
    restored, _ = make_identity_flow(monkeypatch, oracle, True, tmp_path, packed=True)
    restored.restore_state(saved)
    pd.testing.assert_frame_equal(restored.view("entities"), flow.view("entities"))
    rows["add_seq"] = 1
    rows["episode_id"] = "1"
    rows["fact"] = [f"new-fact-{i}" for i in range(7)]
    restored.apply(rows)
    # Identity grouping is reused; new fact grouping still uses prompt16.
    grouped = [r for r in oracle.requests if "rows" in r]
    assert [len(r["rows"]) for r in grouped] == [16, 5, 16, 5, 16, 5]
    assert len(restored.view("facts")) == 14


def test_groupby_rejects_unsupported_sites_before_input() -> None:
    from agent_memory.adapters.lotus.sem_groupby import validate_site_groupby_batching
    from agent_memory.adapters.lotus.pair_execution import SemanticPairExecutionProfile
    from agent_memory.tracing.semantic import query_digest
    from test_semantic_pair_execution import PAIR_EMBEDDING
    query = grouping()
    for params in ({"labels": (("x", "label"),)}, {"membership": "overlapping"}):
        with pytest.raises(ValueError, match="exclusive oracle"):
            validate_site_groupby_batching(replace(query, params={**query.params, **params}), LotusExecutionConfig())
    profile = SemanticPairExecutionProfile(mode="proxy-only", embedding=PAIR_EMBEDDING,
        min_similarity=0.5, direction="symmetric", left_id_columns=("l",), right_id_columns=("r",),
        left_text_columns=("left",), right_text_columns=("right",))
    with pytest.raises(ValueError, match="exclusive oracle"):
        validate_site_groupby_batching(query, LotusExecutionConfig(semantic_pair_profiles={query_digest(query): profile}))


@pytest.mark.parametrize("direction", ["symmetric", "left-to-right", "right-to-left"])
def test_memoized_scores_are_bitwise_equal_and_keep_selection(direction: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_memory.adapters.lotus import pair_execution as pairs
    vectors = {"a": [0.1, 0.2, 0.3], "b": [1.0, 2.0, 3.0], "c": [-0.2, 0.4, 0.7], "d": [0.0, 1.0, 0.0]}
    text_pairs = [(a, b) for a in vectors for b in vectors] * 3
    expected = tuple(sum(x * y for x, y in zip(vectors[a], vectors[b], strict=True)) /
                     (math.sqrt(sum(x*x for x in vectors[a])) * math.sqrt(sum(x*x for x in vectors[b])))
                     for a, b in text_pairs)
    sqrt = math.sqrt
    norms: list[float] = []
    def counted(value: float) -> float:
        norms.append(value)
        return sqrt(value)
    monkeypatch.setattr(pairs.math, "sqrt", counted)
    actual = pairs._memoized_cosines(vectors, text_pairs)
    assert [v.hex() for v in actual] == [v.hex() for v in expected]
    assert len(norms) == len(vectors)
    for threshold in (None, expected[1], math.nextafter(expected[1], math.inf)):
        options: dict[str, Any] = dict(left_ids=[a for a, _ in text_pairs], right_ids=[b for _, b in text_pairs],
                       direction=direction, top_k=2, min_similarity=threshold)
        assert pairs._select_positions(actual, **options) == pairs._select_positions(expected, **options)
    assert pairs._memoized_cosines({}, []) == ()


def test_factory_propagates_new_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agent_memory.adapters import lotus
    from agent_memory.evaluation.agent_memory_drivers import ZepMemoryDriverFactory
    class Reached(Exception):
        pass
    config = {"sem_groupby:test": PromptBatching(max_tasks=16)}
    def inspect(**kwargs: Any) -> Any:
        assert kwargs["config"].groupby_prompt_batching == config
        assert kwargs["config"].reuse_unchanged_entity_name is True
        raise Reached
    factory = ZepMemoryDriverFactory(connector=SimpleNamespace(embedding_provider=None),
        base_namespace="test", physical_fusion="zep-representative", groupby_prompt_batching=config,
        reuse_unchanged_entity_name=True, neo4j_image="fake", neo4j_image_digest="fake")
    monkeypatch.setattr(lotus, "LotusAdapter", inspect)
    with pytest.raises(Reached):
        factory("case", tmp_path / "state", tmp_path / "trace")


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("cache_mode", ["disabled", "memory"])
def test_benchmark_records_options_and_changes_resume_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parallel: bool, cache_mode: str) -> None:
    import agent_memory.evaluation.run as run_module
    from test_benchmark_harness import _bundle
    captured: dict[str, Any] = {}

    class Factory:
        @classmethod
        def from_environment(cls, **kwargs: Any) -> Any:
            captured["factory"] = kwargs
            return cls()

        def close(self) -> None:
            pass

        def runtime_provenance(self) -> dict[str, str]:
            return {"connector": "neo4j", "image": "fake", "image_digest": "sha256:fake",
                    "server_version": "fake", "driver_version": "fake"}

    class Runner:
        def __init__(self, **kwargs: Any) -> None:
            captured["runner"] = kwargs

        def run(self, bundle: Any) -> None:
            pass

    monkeypatch.setattr(run_module, "_require_environment", lambda _: None)
    monkeypatch.setattr(run_module, "collect_runtime_provenance", lambda *a, **k: {"source": {}, "runtime": {}})
    monkeypatch.setattr(run_module, "validate_run_provenance", lambda *a, **k: None)
    monkeypatch.setattr(run_module, "ZepMemoryDriverFactory", Factory)
    monkeypatch.setattr(run_module, "BenchmarkRunner", Runner)
    sites = {semantic_pair_site_id(q): PromptBatching(max_tasks=16)
             for n in ZepRepresentativeMemory.differentiate_policy().nodes.values()
             for q in walk(n.query) if q.op == "sem_groupby"}
    identities = []
    for enabled in (False, True):
        run_module.run_agent_memory_bundle(bundle=_bundle(), contracts={}, system_id="zep-memory",
            output_dir=tmp_path / str(enabled), physical_fusion="zep-representative",
            parallel_fact_extraction=parallel and enabled,
            lotus_cache_mode=cache_mode,
            reuse_unchanged_entity_name=enabled, groupby_prompt_batching=sites if enabled else None)
        identities.append(captured["runner"]["system_contract"].maintenance_execution_id)
    assert identities[0] != identities[1]
    assert captured["factory"]["groupby_prompt_batching"] == sites
    assert captured["factory"]["reuse_unchanged_entity_name"] is True
    assert captured["factory"].get("parallel_fact_extraction", False) is parallel
    assert captured["factory"]["lotus_cache_mode"] == cache_mode
    provenance = captured["runner"]["runtime_provenance"]["runtime"]["lotus_execution"]["site_execution"]
    assert provenance["groupby_prompt_batching"] == {s: {"max_tasks": 16} for s in sites}
    assert provenance["reuse_unchanged_entity_name"] is True
    assert provenance.get("parallel_fact_extraction", False) is parallel
    json.dumps(provenance)
