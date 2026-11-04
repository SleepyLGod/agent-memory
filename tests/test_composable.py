"""Offline composable-rule contracts, not model-quality measurements."""

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus import composable as impl
from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.adapters.lotus.sem_join import assemble_join_frame
from agent_memory.planner.rules import DifferentialRules
from agent_memory.policy.grouping import GROUP_ID, RECORD_ID
from test_independent_grouping import fake_model as fake_model


@pytest.fixture
def aggregates(monkeypatch):
    calls = []
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)

    def run(query, groups, selected, config, *, documents=None):
        assert documents is not None
        calls.extend(documents)
        return ["summary" for _ in groups]

    monkeypatch.setattr(impl, "execute_native_sem_agg_groups", run)
    return calls


@pytest.fixture
def joins(monkeypatch):
    calls = []

    def run(query, bindings, execute, context):
        left, right = bindings["left"], bindings["right"]
        calls.append(
            (left.copy(), right.copy(), query.params["k"], query.params["instruction"])
        )
        pairs = []
        for lid, row in left.iterrows():
            matches = []
            for rid, other in right.iterrows():
                if any(
                    word in row.message and word in other.message
                    for word in ("shipping", "quality")
                ):
                    matches.append((lid, rid, None))
            pairs.extend(matches[:1] if query.params["k"] == 1 else matches)
        return assemble_join_frame(
            left, right, pairs, how="left", id_columns=query.params["id_columns"]
        )

    monkeypatch.setattr(impl, "execute_sem_join", run)
    return calls


def test_heterogeneous_documents_and_restore(aggregates):
    source = am.Source({"message": "Text"})
    view = source.sem_agg(output_cols=["summary"], instruction="Summarize")

    def create():
        return am.SemanticDataflow(
            source=source,
            views={"out": view},
            adapter=LotusAdapter(),
            rules=DifferentialRules(sem_agg_rule="agg-composable"),
        )

    flow = create()
    flow.apply(pd.DataFrame({"message": ["first", "first"]}))
    assert len(aggregates[0]) == 2
    restored = create()
    restored.restore_state(flow.snapshot_state())
    restored.apply(pd.DataFrame({"message": ["next"]}))
    docs = aggregates[-1]
    assert len(docs) == 2 and "Summary" in docs[0] and "Message" in docs[1]
    assert "first" not in " ".join(docs)


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("descriptor", [False, True])
def test_group_incremental_numeric_and_restore(
    fake_model, joins, aggregates, membership, descriptor
):
    source = am.Source({"message": "Text", "amount": "Cost"})
    groups = source.sem_groupby(instruction="Group issues", membership=membership)
    summary = (
        groups.agg(am.sem_agg(output_cols=["summary"], instruction="Summarize"))
        if descriptor
        else groups.sem_agg(output_cols=["summary"], instruction="Summarize")
    )
    metrics = groups.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )

    def create():
        return am.SemanticDataflow(
            source=source,
            views={"groups": groups, "summary": summary, "metrics": metrics},
            adapter=LotusAdapter(),
            rules=DifferentialRules(
                sem_agg_rule="agg-composable", sem_groupby_rule="group-composable"
            ),
        )

    flow = create()
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"], "amount": [2, 4]}))
    old = flow.view("groups")
    flow.apply(pd.DataFrame({"message": ["shipping quality"], "amount": [6]}))
    new = flow.view("groups")
    assert set(old[GROUP_ID]) == set(new[GROUP_ID])
    assert len(new) == (4 if membership == "multiple" else 3)
    assert not new.duplicated([RECORD_ID, GROUP_ID]).any()
    assert flow.view("metrics").n.sum() == len(new)
    assert len(aggregates) == (4 if membership == "multiple" else 3)
    assert list(joins[0][0].columns) == ["message", "amount"]
    restored = create()
    restored.restore_state(flow.snapshot_state())
    batch = pd.DataFrame({"message": ["quality"], "amount": [8]})
    flow.apply(batch)
    restored.apply(batch)
    for name in ("groups", "summary", "metrics"):
        pd.testing.assert_frame_equal(flow.view(name), restored.view(name))


def test_merge_partial_is_relational():
    old = pd.DataFrame({RECORD_ID: ["old"], GROUP_ID: ["G"], "text": ["old"]})
    local = pd.DataFrame(
        {RECORD_ID: ["r1", "r2", "r3"], GROUP_ID: ["L"] * 3, "text": ["same"] * 3}
    )
    pairs = pd.DataFrame({RECORD_ID: ["r1", "r1"], GROUP_ID: ["G", "G"]})
    result = impl.merge_memberships(old, local, pairs, "batch", "single").set_index(
        RECORD_ID
    )
    assert result.loc["r1", GROUP_ID] == "G"
    assert result.loc["r2", GROUP_ID] == result.loc["r3", GROUP_ID] != "G"
    assert len(result) == 4


def test_C_failure_is_atomic(fake_model, joins):
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", C=1)
    flow = am.SemanticDataflow(
        source=source,
        views={"g": groups},
        adapter=LotusAdapter(),
        rules=DifferentialRules(sem_groupby_rule="group-composable"),
    )
    flow.apply(pd.DataFrame({"message": ["shipping"]}))
    before = flow.snapshot_state()
    with pytest.raises(ValueError, match="target C"):
        flow.apply(pd.DataFrame({"message": ["quality"]}))
    assert flow.snapshot_state()["next_occurrence"] == before["next_occurrence"]
    assert len(flow.view("g")) == 1


def test_fallback_retractions_feed_composable_agg(fake_model, aggregates):
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group")
    summary = groups.sem_agg(output_cols=["summary"], instruction="Summarize")
    flow = am.SemanticDataflow(
        source=source,
        views={"s": summary},
        adapter=LotusAdapter(),
        rules=DifferentialRules(sem_agg_rule="agg-composable"),
    )
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"]}))
    flow.apply(pd.DataFrame({"message": ["quality regroup"]}))
    assert len(flow.view("s")) == 1
    assert len(aggregates[-1]) == 3
    # The surviving quality group only gains members; the emptied shipping
    # group is removed without a model call.
    assert "Summary" in aggregates[-1][0]


def test_aggregate_retraction_reads_current_input_only(aggregates):
    source = am.Source({"message": "Text"})
    query = source.sem_agg(
        input_cols=["message"], output_cols=["summary"], instruction="Summarize"
    ).expr
    impl._aggregate(
        query,
        pd.DataFrame({"summary": ["obsolete"]}),
        pd.DataFrame({"message": ["kept", "new"]}),
        pd.DataFrame({"message": ["new"]}),
        pd.DataFrame({"message": ["removed"]}),
        LotusExecutionContext(model="offline-fake"),
    )
    assert len(aggregates[-1]) == 2
    assert all("Message" in doc for doc in aggregates[-1])
    assert "obsolete" not in " ".join(aggregates[-1])


@pytest.mark.parametrize("structured", [False, True])
def test_real_aggregate_adapter_with_fake_model(monkeypatch, structured):
    import lotus
    from test_sem_agg_batching import _RecordingModel

    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    answer = '{"summary":"done","detail":"ok"}' if structured else "done"
    model = _RecordingModel(outputs=((answer,), (answer,)))
    monkeypatch.setattr(lotus.settings, "lm", model)
    source = am.Source({"message": "Text"})
    outputs = ["summary", "detail"] if structured else ["summary"]
    view = source.sem_agg(output_cols=outputs, instruction="Summarize {message}")
    flow = am.SemanticDataflow(
        source=source,
        views={"out": view},
        adapter=LotusAdapter(),
        rules=DifferentialRules(sem_agg_rule="agg-composable"),
    )
    flow.apply(pd.DataFrame({"message": ["old raw"]}))
    flow.apply(pd.DataFrame({"message": ["new raw", "new raw"]}))
    assert flow.view("out").iloc[0].summary == "done"
    prompt = str(model.calls[-1][0])
    assert "old raw" not in prompt
    assert "new raw" in prompt and "Summary" in prompt


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_real_group_join_adapters_with_fake_model(monkeypatch, membership):
    import lotus
    from test_sem_agg_batching import _RecordingModel

    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    member = '{"rows":[{"record":0,"group":0}]}'

    class Model(_RecordingModel):
        def get_model_name(self):
            return "offline-fake"

        def __call__(self, prompts, **kwargs):
            text = str(prompts)
            if "selected_ids" in text:
                import json

                payload = json.loads(prompts[0][-1]["content"])
                answer = json.dumps(
                    {"selected_ids": [payload["right_candidates"][0]["id"]]}
                )
            elif "Every record" in text:
                answer = member
            else:
                answer = "Answer: True"
            self._outputs = iter([(answer,)])
            return super().__call__(prompts, **kwargs)

    model = Model()
    monkeypatch.setattr(lotus.settings, "lm", model)
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Same issue", membership=membership)
    flow = am.SemanticDataflow(
        source=source,
        views={"g": groups},
        adapter=LotusAdapter(),
        rules=DifferentialRules(sem_groupby_rule="group-composable"),
    )
    flow.apply(pd.DataFrame({"message": ["shipping"]}))
    flow.apply(pd.DataFrame({"message": ["shipping"]}))
    assert len(model.calls) == 3
    assert len(flow.view("g")) == 2
    assert flow.view("g")[GROUP_ID].nunique() == 1


def test_partition_retraction_preserves_other_partition(fake_model, joins):
    source = am.Source({"message": "Text", "tenant": "Tenant"})
    query = source.sem_groupby(instruction="Group", partition_by="tenant").expr
    context = LotusExecutionContext(model="offline-fake")
    data = pd.DataFrame(
        {"message": ["shipping", "quality"], "tenant": ["a", "b"]},
        index=pd.Index(["r1", "r2"]),
    )
    empty = data.iloc[:0]
    old = impl._group(
        query,
        pd.DataFrame(columns=query.params["columns"]),
        data,
        data,
        empty,
        "v1",
        context,
    )
    new = pd.DataFrame(
        {"message": ["quality"], "tenant": ["a"]}, index=pd.Index(["r3"])
    )
    current = pd.concat([data.iloc[1:], new])
    out = impl._group(query, old, current, new, data.iloc[:1], "v2", context)
    pd.testing.assert_frame_equal(
        old[old.tenant == "b"].reset_index(drop=True),
        out[out.tenant == "b"].reset_index(drop=True),
    )
    assert set(out[RECORD_ID]) == {"r2", "r3"}
    assert not set(out[out.tenant == "a"][GROUP_ID]) & set(old[GROUP_ID])


@pytest.mark.parametrize(
    "rule",
    ["rule-re-group", "rule-all-group", "rule-all-group-optimized", "rule-join-map"],
)
def test_new_and_legacy_rules_coexist(rule):
    source = am.Source({"message": "Text"})
    legacy = source.sem_groupby(
        input_cols=["message"], instruction="Group", membership="exclusive"
    ).sem_agg(input_cols=["message"], output_cols=["message"], instruction="Summarize")
    groups = source.sem_groupby(instruction="Group")
    views = {
        "old": legacy,
        "groups": groups,
        "summary": groups.sem_agg(output_cols=["summary"], instruction="Summarize"),
    }
    # Use the public flow builder so policy shape remains consistent with callers.
    flow = am.SemanticDataflow(
        source=source,
        views=views,
        adapter=LotusAdapter(),
        rules=DifferentialRules(
            grouped_agg_rule=rule,
            sem_agg_rule="agg-composable",
            sem_groupby_rule="group-composable",
        ),
    )
    policy = flow._executor.policy
    assert policy.nodes[policy.view_outputs["old"]].execution_kind == "semantic_state"
    assert (
        policy.nodes[policy.view_outputs["groups"]].execution_kind
        == "semantic_composable"
    )
    assert (
        policy.nodes[policy.view_outputs["summary"]].execution_kind
        == "semantic_composable"
    )


@pytest.mark.parametrize("initial,later", [(1, 1.0), (None, float("nan")), (1, True)])
@pytest.mark.parametrize("composite", [False, True])
def test_partition_equality_across_batches(
    fake_model, joins, aggregates, initial, later, composite
):
    source = am.Source({"message": "Text", "tenant": "Tenant", "region": "Region"})
    groups = source.sem_groupby(
        input_cols=["message"],
        instruction="Group",
        C=1,
        partition_by=["tenant", "region"] if composite else "tenant",
    )
    flow = am.SemanticDataflow(
        source=source,
        views={
            "g": groups,
            "summary": groups.sem_agg(
                input_cols=["message"], output_cols=["summary"], instruction="Summarize"
            ),
        },
        adapter=LotusAdapter(),
        rules=DifferentialRules(
            sem_groupby_rule="group-composable", sem_agg_rule="agg-composable"
        ),
    )
    for tenant in (initial, later):
        flow.apply(
            pd.DataFrame(
                {"message": ["shipping"], "tenant": [tenant], "region": ["east"]}
            )
        )
    assert len(joins) == 1
    assert flow.view("g")[GROUP_ID].nunique() == 1
    assert len(flow.view("summary")) == 1
    # Runtime may emit replacement rows when pandas widens a business dtype;
    # that legitimately takes the affected-group fallback rather than append.
    assert len(aggregates[-1]) == 2
    snapshot = flow.snapshot_state()
    with pytest.raises(ValueError, match="target C"):
        flow.apply(
            pd.DataFrame(
                {"message": ["quality"], "tenant": [later], "region": ["east"]}
            )
        )
    assert flow.snapshot_state()["next_occurrence"] == snapshot["next_occurrence"]


@pytest.mark.parametrize("initial,later", [(1, 1.0), (None, float("nan")), (1, True)])
def test_aggregate_partition_equality_preserves_previous_result(
    aggregates, initial, later
):
    source = am.Source({"tenant": "Tenant", "message": "Text"})
    query = (
        source.group_by("tenant")
        .sem_agg(
            input_cols=["message"], output_cols=["summary"], instruction="Summarize"
        )
        .expr
    )
    previous = pd.DataFrame({"tenant": [initial], "summary": ["previous result"]})
    added = pd.DataFrame({"tenant": [later], "message": ["new raw"]})
    result = impl._aggregate(
        query, previous, added, added, added.iloc[:0],
        LotusExecutionContext(model="offline-fake"),
    )
    assert len(result) == 1
    assert len(aggregates) == 1
    assert "previous result" in aggregates[0][0]
    assert "new raw" in aggregates[0][1]


def test_configured_rewriter_is_compiled_and_fingerprinted(fake_model, joins):
    from agent_memory.planner.differential_policy import PolicyDifferentiator
    from agent_memory.planner.rules import DifferentialInstructionRewriter
    from agent_memory.policy.logical import MemorySpec, MemoryView
    from agent_memory.runtime.executor import PolicyExecutor

    class Rewriter(DifferentialInstructionRewriter):
        def __init__(self, condition: str) -> None:
            self.condition = condition
            self.calls = 0

        def groupby_to_join(self, instruction: str, *, input_cols) -> str:
            self.calls += 1
            return self.condition

    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group")
    spec = MemorySpec(
        log=source,
        views={"g": MemoryView(name="g", query=groups.expr)},
        private_relations={},
    )
    rewriter = Rewriter("custom match {message:left} and {message:right}")
    rules = DifferentialRules(sem_groupby_rule="group-composable")
    policy = PolicyDifferentiator(
        rules=rules, instruction_rewriter=rewriter
    ).differentiate(spec)
    assert rewriter.calls == 1
    node = policy.nodes[policy.view_outputs["g"]]
    assert node.query.params["join_instruction"] == rewriter.condition
    runtime = PolicyExecutor(policy, adapter=LotusAdapter())
    for _ in range(2):
        runtime.apply_delta(pd.DataFrame({"message": ["shipping"]}))
    assert rewriter.calls == 1
    assert joins[-1][3] == rewriter.condition
    other = PolicyDifferentiator(
        rules=rules, instruction_rewriter=Rewriter("different")
    ).differentiate(spec)
    assert other.fingerprint != policy.fingerprint
    with pytest.raises(ValueError, match="fingerprint"):
        PolicyExecutor(other, adapter=LotusAdapter()).restore_state(
            runtime.snapshot_state()
        )


def test_claude_experiment_preserves_baseline_and_selects_new_rules():
    from pathlib import Path
    import runpy

    from agent_memory.memories.claude import ClaudeMemory

    original = ClaudeMemory.topics.expr
    experiment = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1] / "examples/claude/composable_memory.py"
        )
    )["ComposableClaudeMemory"]
    rules = DifferentialRules(
        sem_agg_rule="agg-composable", sem_groupby_rule="group-composable"
    )
    baseline = ClaudeMemory.differentiate_policy(rules=rules)
    migrated = experiment.differentiate_policy(rules=rules)
    assert ClaudeMemory.topics.expr == original
    assert not any(
        n.execution_kind == "semantic_composable" for n in baseline.nodes.values()
    )
    nodes = [
        n for n in migrated.nodes.values() if n.execution_kind == "semantic_composable"
    ]
    assert {n.query.op for n in nodes} == {"sem_groupby", "sem_agg"}
    agg = original.inputs[0]
    group = agg.inputs[0]
    for node in nodes:
        reference = group if node.query.op == "sem_groupby" else agg
        assert node.query.params["instruction"] == reference.params["instruction"]
