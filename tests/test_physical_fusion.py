"""Offline target-state fusion tests using the real Zep maintenance contract."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.adapter import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.fusion import parse_resolution
from agent_memory.adapters.lotus.pair_execution import (
    SemanticPairExecutionProfile,
    PAIR_LEFT_ID_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
)
from agent_memory.memories.zep.policy import ZepMemory
from agent_memory.planner.physical import optimize_policy
from agent_memory.policy.logical import QueryExpr
from agent_memory.tracing.semantic import query_digest


class FakeLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[Any] = []

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        self.calls.append((prompts, kwargs))
        return SimpleNamespace(outputs=[self.answers.pop(0)])


def fused_query() -> QueryExpr:
    policy = optimize_policy(
        ZepMemory.differentiate_policy(), strategy="zep-target-state"
    )
    return next(
        n.maintenance_query
        for n in policy.nodes.values()
        if n.maintenance_query is not None
        and n.maintenance_query.op == "fused_target_state"
    )


def state(fact: str, seq: int, *, entity: str = "a") -> dict[str, Any]:
    return {
        "fact_id": (seq, 0),
        "source_entity_id": entity,
        "target_entity_id": "b",
        "relation_type": "LIKES",
        "fact": fact,
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
        "provenance": json.dumps(
            [
                {
                    "episode_id": str(seq),
                    "fact_ordinal": 0,
                    "created_at": "2026-01-01",
                    "content": fact,
                }
            ]
        ),
        "created_at": "2026-01-01",
        "add_seq": seq,
    }


def response(
    matches: list[tuple[str, str | None]], states: list[tuple[str, str]]
) -> str:
    return json.dumps(
        {
            "matches": [
                {"left_id": left, "right_id": right} for left, right in matches
            ],
            "states": [
                {"right_id": r, "relation_type": "LIKES", "fact": f} for r, f in states
            ],
        }
    )


def execute_fused(
    monkeypatch: pytest.MonkeyPatch,
    left: pd.DataFrame,
    right: pd.DataFrame,
    answers: list[str],
    *,
    pair_profile: SemanticPairExecutionProfile | None = None,
    provider: Any = None,
    **config: Any,
) -> tuple[pd.DataFrame, FakeLM]:
    import lotus

    query = fused_query()
    model = FakeLM(answers)
    monkeypatch.setattr(lotus.settings, "lm", model)
    joined = query.inputs[0]
    materialized_left = QueryExpr(
        op="materialized_view",
        params={"name": "incoming", "columns": tuple(left.columns)},
    )
    materialized_right = QueryExpr(
        op="materialized_view", params={"name": "old", "columns": tuple(right.columns)}
    )
    query = replace(
        query,
        inputs=(
            replace(joined, inputs=(materialized_left, materialized_right)),
            query.inputs[1],
        ),
    )
    if pair_profile is not None:
        config["semantic_pair_profiles"] = {query_digest(query.inputs[0]): pair_profile}
    adapter = LotusAdapter(
        config=LotusExecutionConfig(physical_fusion="zep-target-state", **config),
        pair_embedding_provider=provider,
    )
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    return adapter.execute(query, {"incoming": left, "old": right}), model


def test_default_and_registered_rewrite() -> None:
    original = ZepMemory.differentiate_policy()
    assert optimize_policy(original) is original
    fused = optimize_policy(original, strategy="zep-target-state")
    assert (
        fused.fingerprint
        == optimize_policy(original, strategy="zep-target-state").fingerprint
    )
    assert fused.fingerprint != original.fingerprint
    assert fused.spec is original.spec
    assert optimize_policy(fused, strategy="zep-target-state") is fused
    changed = [key for key in original.nodes if original.nodes[key] != fused.nodes[key]]
    assert len(changed) == 1
    for key in original.nodes:
        assert original.nodes[key].query == fused.nodes[key].query


def test_matched_new_untouched_and_many_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    left = pd.DataFrame(
        [state("new one", 2), state("new two", 3), state("new only", 4, entity="x")]
    )
    right = pd.DataFrame([state("old", 0), state("untouched", 1, entity="z")])
    result, model = execute_fused(
        monkeypatch,
        left,
        right,
        [response([("l0", "r0"), ("l1", "r0")], [("r0", "merged")])],
    )
    assert result["fact"].tolist() == ["untouched", "merged", "new only"]
    merged = result[result.fact == "merged"].iloc[0]
    assert merged.fact_id == (0, 0)
    provenance = json.loads(merged.provenance)
    assert len(provenance) == 3
    assert {p["episode_id"] for p in provenance} == {"0", "2", "3"}
    assert len(model.calls) == 1
    prompt = json.loads(model.calls[0][0][0][1]["content"])
    assert len(prompt["incoming"]) == 2
    assert len(prompt["targets"]) == 1
    assert "join_instruction" in prompt and "consolidation_instruction" in prompt


def test_no_selected_target_keeps_both(monkeypatch: pytest.MonkeyPatch) -> None:
    result, model = execute_fused(
        monkeypatch,
        pd.DataFrame([state("new", 1)]),
        pd.DataFrame([state("old", 0)]),
        [response([("l0", None)], [])],
    )
    assert result.fact.tolist() == ["old", "new"]
    assert len(model.calls) == 1


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_empty_sides_do_not_call_provider(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    frame = pd.DataFrame([state("old", 0)])
    left = frame.iloc[:0] if side in {"left", "both"} else frame
    right = frame.iloc[:0] if side in {"right", "both"} else frame
    result, model = execute_fused(monkeypatch, left, right, [])
    assert len(result) == len(left) + len(right)
    assert not model.calls


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        response([("l0", "r9")], [("r9", "bad")]),
        response([], []),
        response([("l0", "r0"), ("l0", "r0")], [("r0", "bad")]),
        response([("l0", "r0")], []),
        response([("l0", None)], [("r0", "extra")]),
        '{"matches":[{"left_id":"l0","right_id":"r0"}],"states":[{"right_id":"r0","relation_type":"x","fact":false}]}',
    ],
)
def test_invalid_output_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_resolution(raw, {"l0": {"r0"}})


def test_cross_row_target_assignment_is_rejected() -> None:
    raw = response(
        [("l0", "r1"), ("l1", "r1")],
        [("r1", "merged")],
    )
    with pytest.raises(
        ValueError,
        match=r"target ID 'r1' is not eligible for incoming row 'l0'",
    ):
        parse_resolution(raw, {"l0": {"r0"}, "l1": {"r1"}})


def test_shared_retry_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    result, model = execute_fused(
        monkeypatch,
        pd.DataFrame([state("new", 1)]),
        pd.DataFrame([state("old", 0)]),
        ["{}", response([("l0", "r0")], [("r0", "merged")])],
        structured_parse_retries=1,
    )
    assert result.fact.tolist() == ["merged"]
    assert len(model.calls) == 2
    assert model.calls[0][0] != model.calls[1][0]
    retry_feedback = model.calls[1][0][0][-1]
    assert retry_feedback["role"] == "user"
    assert "fusion output requires only matches and states" in retry_feedback["content"]
    assert "eligible_targets" in retry_feedback["content"]


def test_fusion_keeps_one_safety_retry_when_generic_retries_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model = execute_fused(
        monkeypatch,
        pd.DataFrame([state("new", 1)]),
        pd.DataFrame([state("old", 0)]),
        ["{}", response([("l0", "r0")], [("r0", "merged")])],
        structured_parse_retries=0,
    )
    assert result.fact.tolist() == ["merged"]
    assert len(model.calls) == 2


def test_unsupported_plan_and_config() -> None:
    from agent_memory.memories.mem0.policy import Mem0Memory

    with pytest.raises(ValueError, match="registered"):
        optimize_policy(Mem0Memory.differentiate_policy(), strategy="zep-target-state")
    with pytest.raises(ValueError, match="unknown"):
        optimize_policy(ZepMemory.differentiate_policy(), strategy="invented")
    with pytest.raises(ValueError, match="plain"):
        LotusExecutionConfig(
            physical_fusion="zep-target-state", sem_join_topk_method="pairwise-naive"
        )


def fact_flow(
    monkeypatch: pytest.MonkeyPatch, answers: list[str], *, enabled: bool = True
) -> tuple[Any, FakeLM]:
    """Use the real fact plan; only initial extraction/grouping is an oracle."""
    import agent_memory as am
    import lotus
    from agent_memory.policy.aggregates import SemanticAggregateSpec
    from agent_memory.policy.relation import Relation
    from agent_memory.policy.schema import output_columns

    class FactOracleAdapter(LotusAdapter):
        def execute(self, query: QueryExpr, inputs: Any) -> Any:
            if query.op == "agg" and query.inputs[0].op == "sem_groupby":
                rows = self.execute(query.inputs[0].inputs[0], inputs)
                # These fixtures have one distinct fact per incoming message.
                assert len(rows) <= 1
                return pd.DataFrame(
                    [
                        state(
                            row["content"],
                            row["add_seq"],
                            entity=row["source_entity_id"],
                        )
                        for _, row in rows.iterrows()
                    ],
                    columns=pd.Index(output_columns(query)),
                )
            if query.op == "agg":
                assert not any(
                    isinstance(s, SemanticAggregateSpec)
                    for s in query.params["aggregates"]
                ), "fused merge executed twice"
            if query.op == "sem_join":
                raise AssertionError("fused matching executed twice")
            return super().execute(query, inputs)

    raw = {**state("old", 0), "episode_id": "0", "fact_ordinal": 0, "content": "old"}
    source = am.Source({name: name for name in raw})
    original = ZepMemory._deduplicated_facts.expr
    aggregate = original.inputs[0]
    grouped = replace(aggregate.inputs[0], inputs=(source.expr,))
    view = Relation(replace(original, inputs=(replace(aggregate, inputs=(grouped,)),)))
    model = FakeLM(answers)
    monkeypatch.setattr(lotus.settings, "lm", model)
    adapter = FactOracleAdapter(
        config=LotusExecutionConfig(
            physical_fusion="zep-target-state" if enabled else "disabled",
            structured_parse_retries=0,
        )
    )
    monkeypatch.setattr(adapter._context, "configure", lambda: None)
    return am.SemanticDataflow(
        source=source, views={"facts": view}, adapter=adapter
    ), model


def fact_input(fact: str, seq: int, *, entity: str = "a") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **state(fact, seq, entity=entity),
                "episode_id": str(seq),
                "fact_ordinal": 0,
                "content": fact,
            }
        ]
    )


def test_real_runtime_restore_and_single_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow, model = fact_flow(monkeypatch, [response([("l0", "r0")], [("r0", "merged")])])
    flow.apply(fact_input("old", 0))
    assert not model.calls
    flow.apply(fact_input("new", 1))
    assert flow.view("facts").fact.tolist() == ["merged"]
    assert len(model.calls) == 1
    saved = flow.snapshot_state()
    restored, second_model = fact_flow(monkeypatch, [])
    restored.restore_state(saved)
    restored.apply(fact_input("unrelated", 2, entity="other"))
    assert restored.view("facts").fact.tolist() == ["merged", "unrelated"]
    assert not second_model.calls
    plain, _ = fact_flow(monkeypatch, [], enabled=False)
    with pytest.raises(ValueError, match="fingerprint"):
        plain.restore_state(saved)


def test_failed_fused_update_does_not_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from agent_memory.adapters.lotus import structured

    monkeypatch.setattr(structured, "STRUCTURED_FAILURE_DIR", tmp_path)
    flow, model = fact_flow(monkeypatch, ["{}", "{}"])
    flow.apply(fact_input("old", 0))
    before = flow.view("facts")
    with pytest.raises(ValueError, match="invalid target-state"):
        flow.apply(fact_input("new", 1))
    pd.testing.assert_frame_equal(flow.view("facts"), before)
    assert len(flow.snapshot_state()["state"]["log"]) == 1
    assert len(model.calls) == 2


def test_screening_and_partition_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_memory.storage import EmbeddingSpec

    class Embeddings:
        def embed(self, spec: Any, texts: Any) -> list[list[float]]:
            return [[0.0, 1.0] if "bad" in text else [1.0, 0.0] for text in texts]

    profile = SemanticPairExecutionProfile(
        mode="search-filter",
        direction="left-to-right",
        left_id_columns=(PAIR_LEFT_ID_COLUMN,),
        right_id_columns=(PAIR_RIGHT_ID_COLUMN,),
        left_text_columns=(PAIR_LEFT_TEXT_COLUMN,),
        right_text_columns=(PAIR_RIGHT_TEXT_COLUMN,),
        embedding=EmbeddingSpec(
            source_column="text",
            property_name="embedding",
            model="fake",
            revision="1",
            dimensions=2,
            normalize=True,
        ),
        top_k=1,
        min_similarity=0.5,
    )
    result, model = execute_fused(
        monkeypatch,
        pd.DataFrame([state("new", 3)]),
        pd.DataFrame(
            [state("bad", 0), state("good", 1), state("other", 2, entity="other")]
        ),
        [response([("l0", "r1")], [("r1", "merged")])],
        pair_profile=profile,
        provider=Embeddings(),
    )
    payload = json.loads(model.calls[0][0][0][1]["content"])
    assert payload["incoming"][0]["eligible_targets"] == ["r1"]
    assert result.fact.tolist() == ["bad", "other", "merged"]


def test_legacy_restore_rejected_before_mutation() -> None:
    from agent_memory.runtime.runtime import MemoryRuntime

    runtime = MemoryRuntime(
        ZepMemory.differentiate_policy(),
        adapter=LotusAdapter(
            config=LotusExecutionConfig(physical_fusion="zep-target-state")
        ),
    )
    with pytest.raises(ValueError, match="legacy"):
        runtime.restore_state({"schema_version": 1})


def test_multiple_match_membership_not_silently_narrowed() -> None:
    original = ZepMemory.differentiate_policy()
    nodes = dict(original.nodes)
    key = next(
        key
        for key, node in nodes.items()
        if node.query.op == "agg"
        and node.query.params == ZepMemory._deduplicated_facts.expr.inputs[0].params
    )
    node = nodes[key]
    grouped = replace(
        node.query.inputs[0],
        params={**node.query.inputs[0].params, "membership": "overlapping"},
    )
    nodes[key] = replace(node, query=replace(node.query, inputs=(grouped,)))
    with pytest.raises(ValueError, match="registered"):
        optimize_policy(replace(original, nodes=nodes), strategy="zep-target-state")


def test_trace_preserves_both_logical_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = response([("l0", "r0")], [("r0", "merged")])
    _, model = execute_fused(
        monkeypatch,
        pd.DataFrame([state("new", 1)]),
        pd.DataFrame([state("old", 0)]),
        [raw],
        semantic_trace_dir=tmp_path,
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    (event,) = [item for item in events if item["event_type"] == "fusion_resolution"]
    parsed = json.loads(Path(event["parsed_output_path"]).read_text())
    assert parsed["matches"] == {"l0": "r0"}
    assert parsed["states"]["r0"]["fact"] == "merged"
    assert json.loads(Path(event["raw_output_path"]).read_text()) == [[raw]]
    assert event["eligible_pair_count"] == 1
    assert len(model.calls) == 1


def test_prepared_plan_cannot_run_with_disabled_adapter() -> None:
    from agent_memory.runtime.executor import PolicyExecutor

    fused = optimize_policy(
        ZepMemory.differentiate_policy(), strategy="zep-target-state"
    )
    with pytest.raises(ValueError, match="configuration"):
        PolicyExecutor(fused, adapter=LotusAdapter())


def test_default_execution_identity_unchanged() -> None:
    from agent_memory.runtime.executor import PolicyExecutor

    original = ZepMemory.differentiate_policy()
    adapter = LotusAdapter()
    assert PolicyExecutor(original, adapter=adapter).policy is original
    assert adapter.maintenance_execution_fingerprint == ""
    fused_adapter = LotusAdapter(
        config=LotusExecutionConfig(physical_fusion="zep-target-state")
    )
    assert (
        fused_adapter.maintenance_execution_fingerprint
        != adapter.maintenance_execution_fingerprint
    )


def test_one_provider_call_not_two_logical_charges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent_memory.evaluation.trace_metrics import normalize_provider_calls
    from agent_memory.tracing.semantic import (
        write_llm_call_trace,
        write_provider_usage_trace,
    )

    original_call = FakeLM.__call__

    def traced_call(model: FakeLM, prompts: Any, **kwargs: Any) -> Any:
        result = original_call(model, prompts, **kwargs)
        write_llm_call_trace(
            tmp_path,
            model="fake",
            messages=prompts,
            kwargs=kwargs,
            outputs=result.outputs,
            latency_sec=0.01,
        )
        write_provider_usage_trace(
            tmp_path,
            model="fake",
            responses=[
                SimpleNamespace(
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    }
                )
            ],
        )
        return result

    monkeypatch.setattr(FakeLM, "__call__", traced_call)
    execute_fused(
        monkeypatch,
        pd.DataFrame([state("new", 1)]),
        pd.DataFrame([state("old", 0)]),
        [response([("l0", "r0")], [("r0", "merged")])],
        semantic_trace_dir=tmp_path,
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    calls = normalize_provider_calls(events, output_dir=tmp_path, include_cost=False)
    assert len(calls) == 1
    assert calls[0]["operator"] == "fused_target_state"
    assert calls[0]["total_tokens"] == 120


def test_benchmark_factory_passes_fusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agent_memory.adapters import lotus
    from agent_memory.evaluation.agent_memory_drivers import ZepMemoryDriverFactory

    class AdapterReached(Exception):
        pass

    def inspect_adapter(**kwargs: Any) -> Any:
        assert kwargs["config"].physical_fusion == "zep-target-state"
        raise AdapterReached

    factory = ZepMemoryDriverFactory(connector=SimpleNamespace(embedding_provider=None),
                                    base_namespace="test", physical_fusion="zep-target-state",
                                    neo4j_image="fake", neo4j_image_digest="fake")
    monkeypatch.setattr(lotus, "LotusAdapter", inspect_adapter)
    with pytest.raises(AdapterReached):
        factory("case", tmp_path / "state", tmp_path / "trace")
