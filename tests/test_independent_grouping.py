"""Offline contracts for independent grouping and full recomputation."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus import membership as impl
from agent_memory.adapters.lotus.context import LotusExecutionContext
from agent_memory.planner.rules import DifferentialRules
from agent_memory.policy.grouping import GROUP_ID, RECORD_ID
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.schema import output_columns


@pytest.fixture
def fake_model(monkeypatch):
    calls = []
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)

    class Fake:
        def __init__(self, frame):
            self.frame = frame

        def __call__(self, **kwargs):
            assert kwargs.get("system_prompt") is None
            assert kwargs["shape"] == "array"
            assert [column.name for column in kwargs["output_cols"]] == [
                "record",
                "group",
            ]
            assert kwargs["format_example"] == {"record": 0, "group": 0}
            records = json.loads(self.frame.iloc[0]["records"])
            calls.append(records)
            rows = []
            for record in records:
                text = record["content"]["message"]
                if text == "fail":
                    raise ValueError("deliberate failure")
                labels = [label for label in ("shipping", "quality") if label in text]
                if any(
                    item["content"]["message"] == "quality regroup" for item in records
                ):
                    labels = ["quality"]
                if "exactly one" in kwargs["instruction"]:
                    labels = labels[:1]
                for label in labels:
                    rows.append(
                        dict(
                            record=record["record"],
                            group=("shipping", "quality").index(label),
                        )
                    )
            return SimpleNamespace(parsed_outputs=[rows])

    monkeypatch.setattr(impl, "StructuredLMExecutor", Fake)
    return calls


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_memberships_and_snapshot(fake_model, membership):
    source = am.Source({"message": "Complaint", "user": "Tenant"})
    groups = source.sem_groupby(
        instruction="Group issues",
        membership=membership,
        partition_by="user",
    )
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    assert flow.view("groups").empty
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"], "user": ["a", "a"]}))
    before = flow.view("groups")
    flow.apply(
        pd.DataFrame({"message": ["shipping quality", "shipping"], "user": ["a", "b"]})
    )
    result = flow.view("groups")
    assert len(result) == (5 if membership == "multiple" else 4)
    assert not result.duplicated([RECORD_ID, GROUP_ID]).any()
    assert set(before[RECORD_ID]).issubset(result[RECORD_ID])
    assert result.groupby(GROUP_ID)["user"].nunique().max() == 1
    restored = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    restored.restore_state(flow.snapshot_state())
    pd.testing.assert_frame_equal(restored.view("groups"), result)
    with pytest.raises(ValueError, match="deliberate"):
        flow.apply(pd.DataFrame({"message": ["fail"], "user": ["a"]}))
    pd.testing.assert_frame_equal(flow.view("groups"), result)
    count = len(fake_model)
    flow.apply(pd.DataFrame(columns=["message", "user"]))
    assert len(fake_model) == count


def test_duplicate_records_not_collapsed(fake_model):
    source = am.Source({"message": "Complaint"})
    groups = source.sem_groupby(instruction="Group issues")
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    flow.apply(pd.DataFrame({"message": ["shipping", "shipping"]}))
    assert flow.view("groups")[RECORD_ID].nunique() == 2


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [dict(record=9, group=0)],
        [dict(record=0, group=0)] * 2,
        [{"record": 0}],
        [{"group": 0}],
    ],
)
def test_reject_invalid_membership(bad):
    with pytest.raises(ValueError):
        impl.validate_memberships(bad, 1, "multiple")


@pytest.mark.parametrize(
    "mode,rows,size",
    [
        (
            "single",
            [{"record": 0, "group": number} for number in [0, 1]],
            1,
        ),
        (
            "multiple",
            [{"record": 0, "group": 0}],
            2,
        ),
        (
            "multiple",
            [{"record": 0, "group": -1}],
            1,
        ),
    ],
)
def test_reject_semantically_inconsistent_membership(mode, rows, size):
    with pytest.raises(ValueError):
        impl.validate_memberships(rows, size, mode)


def test_api_guards():
    source = am.Source({"message": "text"})
    for kwargs in ({"S": {}}, {"S": False}, {"S": 0}, {"S": "summary"}):
        with pytest.raises(NotImplementedError):
            source.sem_groupby(instruction="group", **kwargs)
    with pytest.raises(NotImplementedError):
        source.sem_agg(instruction="summarize", S={})
    with pytest.raises(NotImplementedError):
        am.sem_agg(instruction="summarize", output_cols=["summary"], S={})
    system_source = am.Log(system_columns=True)
    assert system_source.sem_groupby(instruction="group").expr.params["input_cols"] == (
        "message",
    )


@pytest.mark.parametrize("field", ["record", "group"])
@pytest.mark.parametrize(
    "value", [True, False, 0.0, 1.5, "0", "shipping", None, [], {}, -1]
)
def test_membership_numbers_require_exact_integers(field: str, value: object) -> None:
    row = {"record": 0, "group": 0, field: value}
    with pytest.raises(ValueError):
        impl.validate_memberships([row], 1, "single")


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("C", [None, 2, 3])
def test_validation_returns_ordered_position_and_group_lists(
    membership: str, C: int | None
) -> None:
    rows = [{"record": 1, "group": 9}, {"record": 0, "group": 2}]
    if membership == "multiple":
        rows.append({"record": 0, "group": 9})
    result = impl.validate_memberships(rows, 2, membership, C=C)
    assert type(result) is tuple
    assert all(type(values) is list for values in result)
    assert result == ([row["record"] for row in rows], [row["group"] for row in rows])
    assert impl.validate_memberships([], 0, membership, C=C) == ([], [])
    with pytest.raises(ValueError, match="target C"):
        impl.validate_memberships(rows, 2, membership, C=1)


@pytest.mark.parametrize("C", [0, -1, True, False, 1.0, 2.5, "2", [], {}])
@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_invalid_target_rejected_by_api(C: object, membership: str) -> None:
    source = am.Source({"message": "Text"})
    with pytest.raises(ValueError, match="positive integer"):
        source.sem_groupby(instruction="Group", membership=membership, C=C)


@pytest.mark.parametrize("membership", [None, "exclusive"])
def test_legacy_grouping_rejects_target(membership: str | None) -> None:
    source = am.Source({"message": "Text"})
    source.sem_groupby(instruction="Group", membership=membership, C=None)
    with pytest.raises(NotImplementedError, match="C"):
        source.sem_groupby(instruction="Group", membership=membership, C=2)


@pytest.mark.parametrize(
    "kwargs", [{"labels": {"shipping": "Delivery"}}, {"label_col": "topic"}]
)
def test_removed_label_api_is_rejected(kwargs: dict[str, object]) -> None:
    source = am.Source({"message": "Text"})
    with pytest.raises(TypeError):
        source.sem_groupby(instruction="Group", **kwargs)


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("C", [None, 1, 2, 3])
def test_target_allows_fewer_or_equal_groups_and_rejects_overflow(
    fake_model, record_property, membership: str, C: int | None
) -> None:
    record_property("target_C", C)
    record_property("quality_status", "unverified_offline_fixture")
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", membership=membership, C=C)
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    batch = pd.DataFrame({"message": ["shipping", "quality", "shipping quality"]})
    if C == 1:
        with pytest.raises(ValueError, match="target C"):
            flow.apply(batch)
        assert flow.view("groups").empty
        record_property("acceptance_status", "rejected")
        record_property("publication_status", "not_published")
    else:
        flow.apply(batch)
        result = flow.view("groups")
        assert result[GROUP_ID].nunique() == 2
        assert len(result) == (4 if membership == "multiple" else 3)
        record_property("actual_nonempty_groups", int(result[GROUP_ID].nunique()))
        record_property("acceptance_status", "accepted")
        record_property("publication_status", "published")
    assert len(fake_model) == 1


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("partitioned", [False, True])
def test_small_partition_fails_before_any_configuration_or_model_call(
    monkeypatch, record_property, membership: str, partitioned: bool
) -> None:
    record_property("target_C", 2)
    record_property("quality_status", "unverified_offline_fixture")

    def unexpected(*args, **kwargs):
        pytest.fail(
            "undersized partition must be rejected before model setup or generation"
        )

    monkeypatch.setattr(LotusExecutionContext, "configure", unexpected)
    monkeypatch.setattr(impl, "StructuredLMExecutor", unexpected)
    source = am.Source({"message": "Text", "user": "Tenant"})
    groups = source.sem_groupby(
        instruction="Group",
        membership=membership,
        partition_by="user" if partitioned else None,
        C=2,
    )
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    # The first partition is valid; a later small partition must prevent ALL calls.
    batch = (
        pd.DataFrame(
            {"message": ["shipping", "quality", "shipping"], "user": ["a", "a", "b"]}
        )
        if partitioned
        else pd.DataFrame({"message": ["shipping"], "user": ["a"]})
    )
    flow.apply(batch.iloc[:0])
    with pytest.raises(ValueError, match="fewer records than target C"):
        flow.apply(batch)
    assert flow.view("groups").empty
    record_property("acceptance_status", "rejected")
    record_property("publication_status", "not_published")


def test_target_is_partition_local(fake_model, record_property) -> None:
    record_property("target_C", 2)
    record_property("target_scope", "per_partition")
    record_property("quality_status", "unverified_offline_fixture")
    source = am.Source({"message": "Text", "user": "Tenant"})
    groups = source.sem_groupby(instruction="Group", partition_by="user", C=2)
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    flow.apply(
        pd.DataFrame(
            {"message": ["shipping", "quality"] * 2, "user": ["a", "a", "b", "b"]}
        )
    )
    result = flow.view("groups")
    assert result[GROUP_ID].nunique() == 4
    assert result.groupby("user")[GROUP_ID].nunique().tolist() == [2, 2]
    assert [[row["record"] for row in records] for records in fake_model] == [
        [0, 1],
        [0, 1],
    ]
    record_property("actual_nonempty_groups", int(result[GROUP_ID].nunique()))
    record_property(
        "actual_nonempty_groups_per_partition",
        json.dumps(result.groupby("user")[GROUP_ID].nunique().tolist()),
    )
    record_property("acceptance_status", "accepted")
    record_property("publication_status", "published")


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_target_one_succeeds_with_one_group(
    fake_model, record_property, membership: str
) -> None:
    record_property("target_C", 1)
    record_property("quality_status", "unverified_offline_fixture")
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", membership=membership, C=1)
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    flow.apply(pd.DataFrame({"message": ["shipping"] * 3}))
    result = flow.view("groups")
    assert result[GROUP_ID].tolist() == [json.dumps([0, 0])] * 3
    assert result[RECORD_ID].nunique() == 3
    assert len(fake_model) == 1
    record_property("actual_nonempty_groups", int(result[GROUP_ID].nunique()))
    record_property("acceptance_status", "accepted")
    record_property("publication_status", "published")


@pytest.mark.parametrize("null_size", [1, 2])
def test_target_categorical_partitions_include_nulls_not_unused_categories(
    fake_model,
    record_property,
    null_size: int,
) -> None:
    record_property("target_C", 2)
    record_property("target_scope", "per_partition")
    record_property("quality_status", "unverified_offline_fixture")
    source = am.Source({"message": "Text", "user": "Tenant"})
    groups = source.sem_groupby(instruction="Group", partition_by="user", C=2)
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    batch = pd.DataFrame(
        {
            "message": ["shipping", "quality"] + ["shipping", "quality"][:null_size],
            "user": pd.Categorical(
                ["a", "a"] + [None] * null_size, categories=["a", "unused"]
            ),
        }
    )
    if null_size == 1:
        with pytest.raises(ValueError, match="fewer records than target C"):
            flow.apply(batch)
        assert not fake_model
        assert flow.view("groups").empty
        record_property("acceptance_status", "rejected")
        record_property("publication_status", "not_published")
    else:
        flow.apply(batch)
        result = flow.view("groups")
        assert len(result) == 4
        assert result.user.isna().sum() == 2
        assert result[RECORD_ID].nunique() == 4
        assert result[GROUP_ID].nunique() == 4
        assert len(fake_model) == 2
        assert all(len(records) == 2 for records in fake_model)
        record_property("actual_nonempty_groups", int(result[GROUP_ID].nunique()))
        record_property(
            "actual_nonempty_groups_per_partition",
            json.dumps(
                result.groupby("user", observed=True, dropna=False)[GROUP_ID]
                .nunique()
                .tolist()
            ),
        )
        record_property("acceptance_status", "accepted")
        record_property("publication_status", "published")


def test_target_fingerprint_and_snapshot_mismatch(fake_model) -> None:
    source = am.Source({"message": "Text"})

    def create(C: int | None) -> am.SemanticDataflow:
        groups = source.sem_groupby(instruction="Group", C=C)
        return am.SemanticDataflow(
            source=source, views={"groups": groups}, adapter=LotusAdapter()
        )

    flow = create(2)
    flow.apply(pd.DataFrame({"message": ["shipping", "quality", "shipping"]}))
    snapshot = flow.snapshot_state()
    restored = create(2)
    restored.restore_state(snapshot)
    pd.testing.assert_frame_equal(restored.view("groups"), flow.view("groups"))
    fingerprints = {create(C)._executor.policy.fingerprint for C in (None, 1, 2, 3)}
    assert len(fingerprints) == 4
    for C in (None, 1, 3):
        with pytest.raises(ValueError, match="fingerprint"):
            create(C).restore_state(snapshot)


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_user_label_and_description_columns_are_preserved(
    fake_model, count_aggregates, membership: str
) -> None:
    columns = [
        "message",
        "label",
        "description",
        "_label",
        "_agent_memory_group_description",
        "__am_aggregate_row_count",
    ]
    source = am.Source({name: "Business data" for name in columns})
    groups = source.sem_groupby(instruction="Group", membership=membership)
    summary = groups.sem_agg(output_cols=["summary"], instruction="Count")
    flow = am.SemanticDataflow(
        source=source,
        views={"groups": groups, "summary": summary},
        adapter=LotusAdapter(),
    )
    batch = pd.DataFrame(
        {name: [f"{name} first", f"{name} second"] for name in columns}
    )
    batch["message"] = ["shipping quality", "shipping"]
    flow.apply(batch)
    result = flow.view("groups")
    assert list(result.columns) == [*columns, RECORD_ID, GROUP_ID]
    for message, rows in result.groupby("message"):
        expected = batch[batch.message == message].iloc[0].to_dict()
        assert all(row == expected for row in rows[columns].to_dict("records"))
    assert all(list(frame.columns) == columns for frame in count_aggregates["inputs"])
    assert list(flow.view("summary").columns) == [GROUP_ID, "summary"]
    for name in columns[1:]:
        assert name in output_columns(groups.count(output_col=name).expr)


def test_aggregate_fallback_and_rollback():
    class Adapter:
        def execute(self, query, inputs):
            if query.op == "sem_agg":
                frame = inputs[query.inputs[0].params["name"]]
                if "fail" in frame.message.tolist():
                    raise ValueError("failed")
                return pd.DataFrame({"summary": ["|".join(frame.message)]})
            raise AssertionError(query.op)

    source = am.Source({"message": "text"})
    view = source.sem_agg(
        input_cols=["message"], output_cols=["summary"], instruction="summarize"
    )
    flow = am.SemanticDataflow(
        source=source,
        views={"summary": view},
        adapter=Adapter(),
        rules=DifferentialRules(sem_agg_rule="agg-fallback"),
    )
    for text in ["a", "b"]:
        flow.apply(pd.DataFrame({"message": [text]}))
    assert flow.view("summary").summary.tolist() == ["a|b"]
    with pytest.raises(ValueError):
        flow.apply(pd.DataFrame({"message": ["fail"]}))
    assert flow.view("summary").summary.tolist() == ["a|b"]


@pytest.mark.parametrize("descriptor", [False, True])
def test_group_aggregate_consumes_memberships(fake_model, descriptor):
    class Adapter(LotusAdapter):
        def execute(self, query, inputs):
            if query.op in {"sem_agg", "agg"}:
                grouped = query.inputs[0]
                frame = super().execute(grouped.inputs[0], inputs)
                return (
                    frame.groupby(
                        list(grouped.params["keys"]),
                        dropna=False,
                    )
                    .size()
                    .reset_index(name="summary")
                )
            return super().execute(query, inputs)

    source = am.Source({"message": "text"})
    groups = source.sem_groupby(instruction="group issues", membership="multiple")
    view = (
        groups.agg(am.sem_agg(output_cols=["summary"], instruction="count"))
        if descriptor
        else groups.sem_agg(output_cols=["summary"], instruction="count")
    )
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups, "counts": view}, adapter=Adapter()
    )
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"]}))
    flow.apply(pd.DataFrame({"message": ["shipping quality"]}))
    assert sorted(flow.view("counts").summary) == [2, 2]
    assert len(fake_model) == 2


def test_reassignment_replaces_old_memberships(fake_model):
    source = am.Source({"message": "Complaint"})
    groups = source.sem_groupby(instruction="Group issues")
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups}, adapter=LotusAdapter()
    )
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"]}))
    old_ids = set(flow.view("groups")[RECORD_ID])
    flow.apply(pd.DataFrame({"message": ["quality regroup"]}))
    result = flow.view("groups")
    assert result[GROUP_ID].tolist() == [json.dumps([0, 1])] * 3
    assert old_ids.issubset(result[RECORD_ID])


def test_unsupported_retraction_consumer():
    source = am.Source({"message": "Complaint"})
    groups = source.sem_groupby(instruction="Group issues")
    with pytest.raises(
        NotImplementedError, match="does not preserve semantic grouping"
    ):
        groups.sem_join(source, instruction="match")
    with pytest.raises(
        NotImplementedError, match="does not preserve semantic grouping"
    ):
        groups.count_window(size=2)


def test_memory_rules_do_not_reuse_cached_default_plan():
    class Example(am.Memory):
        log = am.Log()
        summary = log.sem_agg(
            input_cols=["message"], output_cols=["message"], instruction="summarize"
        )

    default = Example.differentiate_policy()
    fallback = Example.differentiate_policy(
        rules=DifferentialRules(sem_agg_rule="agg-fallback")
    )
    assert fallback.fingerprint != default.fingerprint
    assert Example.differentiate_policy() is default


@pytest.mark.parametrize("descriptor", [False, True])
def test_real_aggregation_adapter_receives_only_business_columns(
    fake_model, monkeypatch, descriptor
):
    import agent_memory.adapters.lotus.sem_agg as agg_module

    captured = []

    def aggregate(query, groups, input_cols, output_cols, config):
        captured.extend(group.loc[:, list(input_cols)].copy() for group in groups)
        assert tuple(input_cols) == ("message",)
        return [{"summary": str(len(group)), "details": "ok"} for group in groups]

    monkeypatch.setattr(agg_module, "execute_structured_sem_agg_groups", aggregate)
    import agent_memory.adapters.lotus.relational as relational_module

    def aggregate_one(query, group, input_cols, output_cols, config, **kwargs):
        return aggregate(query, [group], input_cols, output_cols, config)[0]

    monkeypatch.setattr(
        relational_module, "execute_structured_sem_agg_group", aggregate_one
    )
    source = am.Source({"message": "Complaint"})
    groups = source.sem_groupby(instruction="Group issues", membership="multiple")
    outputs = ["summary", "details"]
    summary = (
        groups.agg(am.sem_agg(output_cols=outputs, instruction="count"))
        if descriptor
        else groups.sem_agg(output_cols=outputs, instruction="count")
    )
    flow = am.SemanticDataflow(
        source=source, views={"summary": summary}, adapter=LotusAdapter()
    )
    flow.apply(pd.DataFrame({"message": ["shipping", "quality", "shipping quality"]}))
    assert sorted(flow.view("summary").summary) == ["2", "2"]
    assert sum(len(frame) for frame in captured) == 4


@pytest.fixture
def count_aggregates(monkeypatch):
    """Stub model work only, retaining real grouping/aggregate adapters."""
    import agent_memory.adapters.lotus.sem_agg as agg_impl
    import agent_memory.adapters.lotus.relational as rel_impl

    state = {"inputs": [], "fail": False}

    def native_one(query, group, input_cols, config, **kwargs):
        if state["fail"]:
            raise ValueError("downstream aggregate failed")
        state["inputs"].append(group.loc[:, list(input_cols)].copy())
        return str(len(group))

    def native_many(query, groups, input_cols, config):
        return [native_one(query, group, input_cols, config) for group in groups]

    def structured_one(query, group, input_cols, output_cols, config, **kwargs):
        value = native_one(query, group, input_cols, config)
        return {col.name: value for col in output_cols}

    def structured_many(query, groups, input_cols, output_cols, config):
        return [
            structured_one(query, group, input_cols, output_cols, config)
            for group in groups
        ]

    monkeypatch.setattr(agg_impl, "execute_native_sem_agg_groups", native_many)
    monkeypatch.setattr(agg_impl, "execute_structured_sem_agg_groups", structured_many)
    monkeypatch.setattr(rel_impl, "execute_native_sem_agg_group", native_one)
    monkeypatch.setattr(rel_impl, "execute_structured_sem_agg_group", structured_one)
    return state


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("descriptor", [False, True])
@pytest.mark.parametrize(
    "transform", ["direct", "filter", "select", "alias", "combined"]
)
def test_group_scope_survives_transformations(
    fake_model, count_aggregates, membership, descriptor, transform
):
    source = am.Source({"message": "Complaint", "user": "Tenant", "details": "Context"})
    groups = source.sem_groupby(
        input_cols=["message"],
        instruction="Group issues",
        membership=membership,
        partition_by="user",
    )
    transformed = groups
    if transform in {"filter", "combined"}:
        transformed = transformed.filter(transformed.col("user") == "a")
    if transform in {"select", "combined"}:
        transformed = transformed.select(["user", GROUP_ID, "message", "details"])
    if transform in {"alias", "combined"}:
        transformed = transformed.alias("issues")
    view = (
        transformed.agg(am.sem_agg(output_cols=["summary"], instruction="Count"))
        if descriptor
        else transformed.sem_agg(output_cols=["summary"], instruction="Count")
    )
    assert tuple(view.expr.inputs[0].params["keys"]) == ("user", GROUP_ID)
    assert "passthrough" not in view.expr.inputs[0].params
    assert groups.expr.params["grouping"] == {
        "keys": ("user", GROUP_ID),
        "metadata": (RECORD_ID, GROUP_ID),
    }
    flow = am.SemanticDataflow(
        source=source,
        views={"members": transformed, "summary": view},
        adapter=LotusAdapter(),
    )
    flow.apply(
        pd.DataFrame(
            {
                "message": ["shipping", "quality", "shipping quality", "shipping"],
                "user": ["a", "a", "a", "b"],
                "details": ["one", "two", "three", "four"],
            }
        )
    )
    result = flow.view("summary")
    assert sorted(result[result.user == "a"].summary) == (
        ["1", "2"] if membership == "single" else ["2", "2"]
    )
    assert len(result) == (2 if transform in {"filter", "combined"} else 3)
    assert all(
        tuple(frame.columns) == ("message", "user", "details")
        if transform not in {"select", "combined"}
        else tuple(frame.columns) == ("user", "message", "details")
        for frame in count_aggregates["inputs"]
    )
    members = flow.view("members")
    assert not members.duplicated([GROUP_ID, "message", "user"]).any()
    if transform in {"select", "combined"}:
        assert list(members.columns) == ["user", GROUP_ID, "message", "details"]
    assert list(result.columns) == ["user", GROUP_ID, "summary"]


def test_projection_and_output_guards_before_execution():
    source = am.Source({"message": "Complaint", "user": "Tenant"})
    groups = source.sem_groupby(instruction="Group", partition_by="user")
    for columns in (["message"], [GROUP_ID, "message"], ["user", "message"]):
        with pytest.raises(ValueError, match="retain all semantic group keys"):
            groups.select(columns)
    with pytest.raises(ValueError, match="unknown columns"):
        groups.select([GROUP_ID, "user", "unknown"])
    for column in (GROUP_ID, "user"):
        with pytest.raises(ValueError, match="conflicts"):
            groups.sem_agg(output_cols=[column], instruction="Summary")
        with pytest.raises(ValueError, match="conflicts"):
            groups.agg(am.sem_agg(output_cols=[column], instruction="Summary"))
    # Occurrence metadata is optional in projections, unlike group identity keys.
    selected = groups.select(["user", GROUP_ID, "message"])
    assert output_columns(selected.sem_agg(instruction="Summary").expr) == (
        "user",
        GROUP_ID,
        "message",
    )
    for operation in (
        lambda: groups.assign(x=1),
        lambda: groups.limit(1),
        lambda: groups.over(rows=(-1, 0)),
        lambda: groups.sem_groupby(instruction="Again"),
    ):
        with pytest.raises(
            NotImplementedError, match="does not preserve semantic grouping"
        ):
            operation()
    # An explicit relational regroup is allowed and replaces the scope.
    assert tuple(groups.group_by("user").expr.params["keys"]) == ("user",)


@pytest.mark.parametrize("explicit", [False, True])
def test_standalone_defaults_resolve_before_execution(
    monkeypatch, count_aggregates, explicit
):
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    source = am.Source(
        {"message": "Text", "_agent_memory_business": "Business"}, system_columns=True
    )
    columns = ["message", "_row_id"] if explicit else None
    view = source.sem_agg(input_cols=columns, instruction="Summarize")
    flow = am.SemanticDataflow(
        source=source,
        views={"summary": view},
        adapter=LotusAdapter(),
        rules=DifferentialRules(sem_agg_rule="agg-fallback"),
    )
    expected = columns or ["message", "_agent_memory_business"]
    assert list(flow.view("summary").columns) == expected
    batch = pd.DataFrame(
        {
            "message": ["one", "two"],
            "_agent_memory_business": ["x", "y"],
            "_row_id": ["r1", "r2"],
            "_added_at": [1, 2],
            "_add_seq": [1, 2],
        }
    )
    flow.apply(batch.loc[:, list(output_columns(source.expr))])
    assert list(count_aggregates["inputs"][0].columns) == expected
    assert list(flow.view("summary").columns) == expected
    assert len(flow.view("summary")) == 1


def test_incidental_hidden_column_does_not_group_global_fallback(
    monkeypatch, count_aggregates
):
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    source = am.Source({"message": "Text", GROUP_ID: "Business data"})
    view = source.sem_agg(
        input_cols=["message"], output_cols=["summary"], instruction="Count"
    )
    flow = am.SemanticDataflow(
        source=source,
        views={"summary": view},
        adapter=LotusAdapter(),
        rules=DifferentialRules(sem_agg_rule="agg-fallback"),
    )
    flow.apply(pd.DataFrame({"message": ["a", "b"], GROUP_ID: ["x", "y"]}))
    assert flow.view("summary").summary.tolist() == ["2"]


@pytest.mark.parametrize(
    "rule",
    ["rule-re-group", "rule-all-group", "rule-all-group-optimized", "rule-join-map"],
)
def test_old_and_new_rules_coexist(rule):
    source = am.Source({"message": "Text"})
    legacy = source.sem_groupby(
        input_cols=["message"], instruction="Group", membership=None
    ).sem_agg(input_cols=["message"], output_cols=["message"], instruction="Summarize")
    modern = (
        source.sem_groupby(instruction="Group")
        .filter(source.col("message") != "")
        .sem_agg(output_cols=["summary"], instruction="Summarize")
    )
    global_agg = source.sem_agg(output_cols=["summary"], instruction="Summarize")
    windowed = source.over(rows=(-1, 0)).sem_agg(
        input_cols=["message"], output_cols=["summary"], instruction="Summarize"
    )
    baseline = am.SemanticDataflow(
        source=source,
        views={"legacy": legacy, "windowed": windowed},
        rules=DifferentialRules(grouped_agg_rule=rule),
    )._executor.policy
    mixed = am.SemanticDataflow(
        source=source,
        views={
            "legacy": legacy,
            "modern": modern,
            "global": global_agg,
            "windowed": windowed,
        },
        rules=DifferentialRules(grouped_agg_rule=rule, sem_agg_rule="agg-fallback"),
    )._executor.policy
    for name in ("legacy", "windowed"):
        node_id = baseline.view_outputs[name]
        assert mixed.view_outputs[name] == node_id
        assert mixed.nodes[node_id] == baseline.nodes[node_id]
    assert mixed.grouped_agg_rule == rule
    for name in ("modern", "global"):
        assert (
            mixed.nodes[mixed.view_outputs[name]].execution_kind == "semantic_fallback"
        )


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_filtered_reassignment_rollback_and_restore(
    fake_model, count_aggregates, membership
):
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", membership=membership)
    kept = groups.filter(groups.col(GROUP_ID) == json.dumps([0, 0])).select(
        [GROUP_ID, "message"]
    )
    summary = kept.alias("shipping").sem_agg(
        output_cols=["summary"], instruction="Count"
    )

    def create():
        return am.SemanticDataflow(
            source=source,
            views={"groups": groups, "kept": kept, "summary": summary},
            adapter=LotusAdapter(),
        )

    flow = create()
    flow.apply(pd.DataFrame({"message": ["shipping", "quality", "shipping quality"]}))
    assert flow.view("summary").summary.tolist() == ["2"]
    snapshot = flow.snapshot_state()
    count_aggregates["fail"] = True
    with pytest.raises(ValueError, match="downstream aggregate failed"):
        flow.apply(pd.DataFrame({"message": ["shipping"]}))
    for name in ("groups", "kept", "summary"):
        before = create()
        before.restore_state(snapshot)
        pd.testing.assert_frame_equal(flow.view(name), before.view(name))
    count_aggregates["fail"] = False
    flow = create()
    flow.restore_state(snapshot)
    flow.apply(pd.DataFrame({"message": ["shipping"]}))
    assert flow.view("summary").summary.tolist() == ["3"]
    calls = len(count_aggregates["inputs"])
    flow.apply(pd.DataFrame({"message": ["quality regroup"]}))
    assert flow.view("summary").empty and flow.view("kept").empty
    assert len(count_aggregates["inputs"]) == calls
    assert flow.view("groups")[GROUP_ID].tolist() == [json.dumps([0, 1])] * 5
    assert list(flow.view("summary").columns) == [GROUP_ID, "summary"]
    with pytest.raises(ValueError, match="fingerprint"):
        create().restore_state({**snapshot, "plan_fingerprint": "pre-repair-plan"})


def test_retraction_guard_survives_relational_transforms():
    source = am.Source({"message": "Text"})
    summary = source.sem_groupby(instruction="Group").sem_agg(
        output_cols=["summary"], instruction="Summarize"
    )
    consumer = (
        summary.filter(summary.col("summary") != "")
        .alias("a")
        .sem_join(source, instruction="Match")
    )
    with pytest.raises(NotImplementedError, match="fallback retractions"):
        am.SemanticDataflow(source=source, views={"consumer": consumer})


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_duplicate_content_preserved_after_projection(
    fake_model, count_aggregates, membership
):
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", membership=membership)
    projected = groups.select([GROUP_ID, "message"])
    summary = projected.sem_agg(output_cols=["summary"], instruction="Count")
    counts = projected.count(output_col="n")
    flow = am.SemanticDataflow(
        source=source,
        views={"summary": summary, "counts": counts},
        adapter=LotusAdapter(),
    )
    for _ in range(2):
        flow.apply(pd.DataFrame({"message": ["shipping quality"]}))
    expected_groups = 2 if membership == "multiple" else 1
    assert flow.view("summary").summary.tolist() == ["2"] * expected_groups
    assert flow.view("counts").n.tolist() == [2] * expected_groups
    flow.apply(pd.DataFrame({"message": ["quality regroup"]}))
    assert flow.view("summary").summary.tolist() == ["3"]
    assert flow.view("counts").n.tolist() == [3]


def test_business_labels_and_descriptions_do_not_define_identity(
    monkeypatch, count_aggregates
):
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    source = am.Source(
        {
            "message": "Text",
            "label": "Business label",
            "description": "Business description",
        }
    )
    groups = source.sem_groupby(instruction="Group")
    query = groups.sem_agg(output_cols=["summary"], instruction="Count").expr
    leaf = QueryExpr(op="materialized_view", params={"name": "members"})
    bound = QueryExpr(
        op="sem_agg",
        inputs=(
            QueryExpr(op="group_by", inputs=(leaf,), params=query.inputs[0].params),
        ),
        params=query.params,
    )
    frame = pd.DataFrame(
        {
            "message": ["a", "b"],
            GROUP_ID: ["x", "x"],
            "label": ["one", "two"],
            "description": ["first", "second"],
        }
    )
    result = LotusAdapter().execute(bound, {"members": frame})
    assert result.to_dict("records") == [{GROUP_ID: "x", "summary": "2"}]
    pd.testing.assert_frame_equal(
        count_aggregates["inputs"][0], frame[["message", "label", "description"]]
    )


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("C", [None, 2])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "unknown_record",
        "bool_record",
        "float_record",
        "bool_group",
        "float_group",
        "string_group",
        "negative_group",
    ],
)
def test_low_level_model_response_through_real_parser_and_adapters(
    monkeypatch, membership, C, invalid
):
    import lotus
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    assignments = [(0, 0), (1, 1), (2, 0)]
    if membership == "multiple":
        assignments.append((2, 1))
    if invalid == "unknown_record":
        assignments.append((99, 1))
    response = {
        "rows": [dict(record=index, group=number) for index, number in assignments]
    }
    invalid_values = {
        "bool_record": ("record", True),
        "float_record": ("record", 0.0),
        "bool_group": ("group", False),
        "float_group": ("group", 0.0),
        "string_group": ("group", "0"),
        "negative_group": ("group", -1),
    }
    if invalid in invalid_values:
        field, value = invalid_values[invalid]
        response["rows"][0][field] = value

    class LocalModel:
        max_ctx_len = 100_000
        max_tokens = 2048

        def __init__(self):
            self.calls = []

        def count_tokens(self, messages):
            return 100

        def __call__(self, prompts, **kwargs):
            output = (
                response if not self.calls else {"summary": "count", "details": "ok"}
            )
            self.calls.append(prompts)
            return SimpleNamespace(outputs=[json.dumps(output) for _ in prompts])

    model = LocalModel()
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    source = am.Source({"message": "Complaint"})
    instruction = 'Group complaints using {issue} and "customer intent"'
    groups = source.sem_groupby(instruction=instruction, membership=membership, C=C)
    summary = groups.alias("issues").sem_agg(
        output_cols=["summary", "details"], instruction="Count complaints"
    )
    flow = am.SemanticDataflow(
        source=source,
        views={"groups": groups, "summary": summary},
        adapter=LotusAdapter(config=LotusExecutionConfig(structured_parse_retries=0)),
    )
    batch = pd.DataFrame({"message": ["shipping", "quality", "shipping quality"]})
    if invalid:
        error = "unknown record" if "record" in invalid else "integer group"
        with pytest.raises(ValueError, match=error):
            flow.apply(batch)
        assert flow.view("groups").empty and flow.view("summary").empty
        assert len(model.calls) == 1
    else:
        flow.apply(batch)
        assert len(flow.view("groups")) == len(assignments)
        assert len(flow.view("summary")) == 2
        assert set(flow.view("summary")[GROUP_ID]) == {
            json.dumps([0, 0]),
            json.dumps([0, 1]),
        }
        assert set(zip(flow.view("groups").message, flow.view("groups")[GROUP_ID])) == {
            (batch.message.iloc[index], json.dumps([0, number]))
            for index, number in assignments
        }
        assert len(model.calls) > 1
        rendered = json.dumps(model.calls)
        assert "Group complaints" in rendered and "Count complaints" in rendered
    from lotus.templates import task_instructions

    prompt = model.calls[0][0]
    default_prompt = task_instructions.map_formatter(model, {"text": ""}, "")
    assert prompt[0] == default_prompt[0]
    user_prompt = "\n".join(
        message["content"] for message in prompt if message["role"] == "user"
    )
    assert instruction in user_prompt
    assert ("exactly one" if membership == "single" else "one or more") in user_prompt
    assert '{"rows": [{"record": 0, "group": 0}]}' in user_prompt
    assert '"record": "string"' not in user_prompt
    assert '"group": "string"' not in user_prompt
    field_descriptions = next(
        line.removeprefix("Field descriptions: ").removesuffix(".")
        for line in user_prompt.splitlines()
        if line.startswith("Field descriptions: ")
    )
    assert set(json.loads(field_descriptions)) == {"record", "group"}
    assert (
        'Every object in rows must include these fields: "record", "group".'
        in user_prompt
    )
    assert '"label"' not in user_prompt and '"description"' not in user_prompt
    if C is not None:
        assert f"target number of groups is {C}" in user_prompt
    assert "group" not in prompt[0]["content"].lower()


@pytest.fixture
def local_membership_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Stub only the LM boundary; retain formatting, parsing, and transactions."""
    import lotus
    import agent_memory.adapters.lotus.structured as structured

    class LocalModel:
        max_ctx_len = 100_000
        max_tokens = 256

        def __init__(self) -> None:
            self.outputs = [
                json.dumps(
                    {"rows": [{"record": 0, "group": 0}, {"record": 1, "group": 0}]}
                )
            ]
            self.response_metadata: list[dict[str, Any]] | None = None
            self.calls: list[tuple[Any, dict[str, Any]]] = []
            self.counted_prompts: list[Any] = []

        def count_tokens(self, messages: Any) -> int:
            self.counted_prompts.append(messages)
            return 100

        def __call__(self, prompts: Any, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((prompts, kwargs))
            return SimpleNamespace(
                outputs=self.outputs, response_metadata=self.response_metadata
            )

    model = LocalModel()
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    monkeypatch.setattr(
        structured, "STRUCTURED_FAILURE_DIR", tmp_path / "structured-failures"
    )
    return model


@pytest.mark.parametrize("transport", ["chat-json-object", "responses-json-schema"])
@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("model_tokens,structured_tokens", [(256, 512), (1024, 512)])
def test_low_level_context_overflow_precedes_model_call(
    local_membership_model: Any,
    transport: str,
    membership: str,
    model_tokens: int,
    structured_tokens: int,
) -> None:
    """Count the actual formatted prompt plus output reserve before generation."""
    from agent_memory.adapters.lotus.context import LotusExecutionConfig
    from lotus.templates import task_instructions

    model = local_membership_model
    model.max_tokens = model_tokens
    model.max_ctx_len = 100 + max(model_tokens, structured_tokens) - 1
    source = am.Source({"message": "Complaint"})
    instruction = "Group complaints by {customer intent}"
    groups = source.sem_groupby(instruction=instruction, membership=membership)
    flow = am.SemanticDataflow(
        source=source,
        views={"groups": groups},
        adapter=LotusAdapter(
            config=LotusExecutionConfig(
                structured_max_tokens=structured_tokens,
                structured_parse_retries=0,
                structured_output_transport=transport,
            )
        ),
    )
    batch = pd.DataFrame({"message": ["shipping", "quality"]})
    with pytest.raises(ValueError, match="context limit"):
        flow.apply(batch)
    assert not model.calls
    assert flow.view("groups").empty
    assert len(model.counted_prompts) == 1
    prompt = model.counted_prompts[0]
    assert prompt[0] == task_instructions.map_formatter(model, {"text": ""}, "")[0]
    rendered = "\n".join(message["content"] for message in prompt)
    assert instruction in rendered and "shipping" in rendered and "quality" in rendered
    assert '{"rows": [{"record": 0, "group": 0}]}' in rendered
    # Exactly filling the available context is legal and retries the same batch.
    model.max_ctx_len += 1
    flow.apply(batch)
    assert len(model.calls) == 1
    assert len(flow.view("groups")) == 2


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_low_level_transports_request_only_membership_fields(
    local_membership_model: Any,
    membership: str,
) -> None:
    """The transport must change the wire format, not the LOTUS prompt or result."""
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    model = local_membership_model
    source = am.Source({"message": "Complaint"})
    groups = source.sem_groupby(instruction="Group complaints", membership=membership)
    results = []
    for transport in ("chat-json-object", "responses-json-schema"):
        flow = am.SemanticDataflow(
            source=source,
            views={"groups": groups},
            adapter=LotusAdapter(
                config=LotusExecutionConfig(
                    structured_output_transport=transport,
                    structured_parse_retries=0,
                )
            ),
        )
        flow.apply(pd.DataFrame({"message": ["shipping", "quality"]}))
        results.append(flow.view("groups"))
    assert len(model.calls) == 2
    (chat_prompt, chat_kwargs), (schema_prompt, schema_kwargs) = model.calls
    assert chat_prompt == schema_prompt
    pd.testing.assert_frame_equal(*results)
    assert list(results[0].columns) == ["message", RECORD_ID, GROUP_ID]
    assert chat_kwargs["response_format"] == {"type": "json_object"}
    response_format = schema_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"rows"}
    assert schema["required"] == ["rows"]
    row_schema = schema["properties"]["rows"]["items"]
    assert set(row_schema["properties"]) == {"record", "group"}
    assert set(row_schema["required"]) == {"record", "group"}
    assert row_schema["additionalProperties"] is False
    user_prompt = "\n".join(
        message["content"] for message in chat_prompt[0] if message["role"] == "user"
    )
    assert (
        'Every object in rows must include these fields: "record", "group".'
        in user_prompt
    )
    assert '"label"' not in user_prompt and '"description"' not in user_prompt


@pytest.mark.parametrize("transport", ["chat-json-object", "responses-json-schema"])
@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize(
    "defect",
    [
        "truncated_json",
        "empty_text",
        "empty_rows",
        "missing_record",
        "missing_field",
        "chat_length",
        "responses_incomplete",
    ],
)
def test_low_level_bad_memberships_roll_back_and_recover(
    local_membership_model: Any,
    transport: str,
    membership: str,
    defect: str,
) -> None:
    """Rejected output must preserve prior memberships, aggregate state, and input."""
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    model = local_membership_model
    source = am.Source({"message": "Complaint"})
    groups = source.sem_groupby(instruction="Group complaints", membership=membership)
    counts = groups.count(output_col="n")

    def create() -> am.SemanticDataflow:
        return am.SemanticDataflow(
            source=source,
            views={"groups": groups, "counts": counts},
            adapter=LotusAdapter(
                config=LotusExecutionConfig(
                    structured_output_transport=transport,
                    structured_parse_retries=0,
                )
            ),
        )

    flow = create()
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"]}))
    snapshot = flow.snapshot_state()
    before = {name: flow.view(name).copy() for name in ("groups", "counts")}
    complete = json.dumps(
        {"rows": [{"record": index, "group": 0} for index in range(3)]}
    )
    raw = {
        "truncated_json": '{"rows": [{"record": 0, "group":',
        "empty_text": "",
        "empty_rows": '{"rows": []}',
        "missing_record": json.dumps(
            {"rows": [{"record": 0, "group": 0}, {"record": 1, "group": 0}]}
        ),
        "missing_field": '{"rows": [{"record": 0}]}',
        "chat_length": complete,
        "responses_incomplete": complete,
    }[defect]
    model.outputs = [raw]
    if defect == "chat_length":
        model.response_metadata = [{"finish_reason": "length"}]
    elif defect == "responses_incomplete":
        model.response_metadata = [
            {"status": "incomplete", "incomplete_reason": "max_output_tokens"}
        ]
    batch = pd.DataFrame({"message": ["third"]})
    with pytest.raises(ValueError):
        flow.apply(batch)
    assert len(model.calls) == 2
    for name in before:
        pd.testing.assert_frame_equal(flow.view(name), before[name])
    restored = create()
    restored.restore_state(snapshot)
    model.outputs = [complete]
    model.response_metadata = None
    flow.apply(batch)
    restored.apply(batch)
    for name in before:
        pd.testing.assert_frame_equal(flow.view(name), restored.view(name))
    assert len(flow.view("groups")) == 3
    assert flow.view("counts").n.tolist() == [3]
    assert len(model.calls) == 4


@pytest.mark.parametrize("stage", ["initial", "retry"])
@pytest.mark.parametrize(
    "defect", ["missing_output", "extra_output", "length", "incomplete"]
)
@pytest.mark.parametrize("custom_validator", [False, True])
def test_direct_structured_retry_rejects_incomplete_physical_results(
    stage: str,
    defect: str,
    custom_validator: bool,
) -> None:
    """Response cardinality and completion flags are fatal, even during retries."""
    import agent_memory.adapters.lotus.structured as structured
    from agent_memory.policy.logical import ColumnSpec

    columns = (ColumnSpec("record"), ColumnSpec("group"))
    valid = '{"rows": [{"record": 0, "group": 0}]}'
    invalid = '{"rows": [{"record": 0}]}'
    prompts = [[{"role": "user", "content": text}] for text in ("first", "second")]
    expected = 2 if stage == "initial" else 1
    outputs = [valid] * expected
    metadata = [{} for _ in outputs]
    if defect == "missing_output":
        outputs = outputs[:-1]
    elif defect == "extra_output":
        outputs = [*outputs, valid]
    elif defect == "length":
        metadata[-1] = {"finish_reason": "length"}
    else:
        metadata[-1] = {
            "status": "incomplete",
            "incomplete_reason": "max_output_tokens",
        }
    responses = [SimpleNamespace(outputs=outputs, response_metadata=metadata)]
    if stage == "retry":
        responses.insert(
            0, SimpleNamespace(outputs=[valid, invalid], response_metadata=None)
        )

    class LocalModel:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def __call__(self, messages: Any, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(messages)
            assert len(self.calls) <= len(responses), "fatal response was retried"
            return responses[len(self.calls) - 1]

    def validate(raw: str) -> None:
        structured.parse_structured_array_json(raw, columns)

    model = LocalModel()
    error = (
        f"expected {expected} outputs"
        if defect.endswith("output")
        else "incomplete structured response"
    )
    with pytest.raises(ValueError, match=error):
        structured.execute_structured_lm_retry_result(
            model,
            prompts,
            lm_kwargs={"progress_bar_desc": "Grouping"},
            output_cols=columns,
            shape="array",
            require_explanation=False,
            operator="sem_groupby",
            max_retries=3,
            output_validator=validate if custom_validator else None,
        )
    assert len(model.calls) == (1 if stage == "initial" else 2)
    if stage == "retry":
        assert model.calls[1] == [prompts[1]]


@pytest.mark.parametrize("descriptor", [False, True])
def test_group_aggregate_explicit_metadata_input_is_respected(
    fake_model, count_aggregates, descriptor
):
    source = am.Source({"message": "Text", "details": "Additional context"})
    groups = source.sem_groupby(input_cols=["message"], instruction="Group")
    args = dict(
        input_cols=["details", GROUP_ID, RECORD_ID],
        output_cols=["summary"],
        instruction="Summarize",
    )
    summary = groups.agg(am.sem_agg(**args)) if descriptor else groups.sem_agg(**args)
    flow = am.SemanticDataflow(
        source=source,
        views={"groups": groups, "summary": summary},
        adapter=LotusAdapter(),
    )
    flow.apply(pd.DataFrame({"message": ["shipping"], "details": ["context"]}))
    pd.testing.assert_frame_equal(
        count_aggregates["inputs"][0],
        flow.view("groups")[["details", GROUP_ID, RECORD_ID]],
    )
    with pytest.raises(ValueError, match="existing columns"):
        groups.sem_agg(
            input_cols=["missing"], output_cols=["summary"], instruction="Summarize"
        )


def test_overlap_expansion_retraction_and_restore(monkeypatch, count_aggregates):
    import lotus
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    class LocalModel:
        max_ctx_len = 100_000
        max_tokens = 2048

        def __init__(self):
            self.step = 0

        def count_tokens(self, messages):
            return 100

        def __call__(self, prompts, **kwargs):
            self.step += 1
            if self.step == 1:
                pairs = [(0, 0), (1, 1)]
            elif self.step == 2:
                pairs = [(record, number) for record in range(3) for number in (0, 1)]
            else:
                pairs = [(record, 1) for record in range(4)]
            rows = [dict(record=record, group=number) for record, number in pairs]
            return SimpleNamespace(outputs=[json.dumps({"rows": rows})])

    model = LocalModel()
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", membership="multiple")
    summary = groups.sem_agg(output_cols=["summary"], instruction="Count")

    def create():
        return am.SemanticDataflow(
            source=source,
            views={"groups": groups, "summary": summary},
            adapter=LotusAdapter(
                config=LotusExecutionConfig(structured_parse_retries=0)
            ),
        )

    flow = create()
    flow.apply(pd.DataFrame({"message": ["first", "second"]}))
    old_ids = set(flow.view("groups")[RECORD_ID])
    group_ids = set(flow.view("summary")[GROUP_ID])
    flow.apply(pd.DataFrame({"message": ["third"]}))
    memberships = flow.view("groups")
    assert memberships[memberships[RECORD_ID].isin(old_ids)].groupby(
        RECORD_ID
    ).size().tolist() == [2, 2]
    assert set(flow.view("summary")[GROUP_ID]) == group_ids
    assert flow.view("summary").summary.tolist() == ["3", "3"]
    restored = create()
    restored.restore_state(flow.snapshot_state())
    restored.apply(pd.DataFrame({"message": ["fourth"]}))
    assert restored.view("summary").summary.tolist() == ["4"]
    assert len(restored.view("groups")) == 4
    assert restored.view("summary")[GROUP_ID].tolist() == [json.dumps([0, 1])]


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("keep_record_id", [False, True])
def test_numeric_group_ivm_uses_signed_deltas(fake_model, membership, keep_record_id):
    updates = []

    class RecordingAdapter(LotusAdapter):
        def execute(self, query, inputs):
            assert query.op not in {"sem_agg", "aggregate_state"}, (
                "numeric views must use IVM, not full aggregation"
            )
            if query.op == "aggregate_state_update":
                updates.append(
                    tuple(len(inputs[leaf.params["name"]]) for leaf in query.inputs)
                )
            return super().execute(query, inputs)

    source = am.Source({"message": "Complaint", "amount": "Cost", "user": "Tenant"})
    groups = source.sem_groupby(
        input_cols=["message"],
        instruction="Group",
        membership=membership,
        partition_by="user",
    )
    if not keep_record_id:
        groups = groups.select(["user", GROUP_ID, "message", "amount"])
    groups = groups.filter(groups.col("amount") != -1).alias("costs")
    metrics = groups.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )
    views = {
        "metrics": metrics,
        "count": groups.count(output_col="n"),
        "sum": groups.sum(column="amount", output_col="total"),
        "avg": groups.avg(column="amount", output_col="mean"),
    }

    def create():
        return am.SemanticDataflow(
            source=source, views=views, adapter=RecordingAdapter()
        )

    flow = create()
    states = [
        node
        for node in flow._executor.policy.nodes.values()
        if node.execution_kind == "algebraic_state"
    ]
    assert states
    assert all(node.query.params["group_keys"] == ("user", GROUP_ID) for node in states)
    assert all(not node.query.params.get("passthrough") for node in states)
    assert all(flow.view(name).empty for name in views)
    flow.apply(
        pd.DataFrame(
            {
                "message": ["shipping", "quality", "shipping quality", "shipping"],
                "amount": [10, 20, 30, 100],
                "user": ["a", "a", "a", "b"],
            }
        )
    )
    first = flow.view("metrics")
    a = first[first.user == "a"].set_index(GROUP_ID)
    shipping = a.loc[json.dumps([0, 0])]
    quality = a.loc[json.dumps([0, 1])]
    assert (shipping.n, shipping.total, shipping["mean"]) == (2, 40, 20)
    assert (quality.n, quality.total, quality["mean"]) == (
        (1, 20, 20) if membership == "single" else (2, 50, 25)
    )
    updates.clear()
    flow.apply(pd.DataFrame({"message": ["shipping"], "amount": [10], "user": ["a"]}))
    # Every numeric view reads a compact group-state table and one new membership,
    # not the full membership relation, even though upstream grouping recomputes.
    assert updates and all(
        (inserted, removed) == (1, 0) for _, inserted, removed in updates
    )
    snapshot = flow.snapshot_state()
    restored = create()
    restored.restore_state(snapshot)
    restored.apply(
        pd.DataFrame({"message": ["quality regroup"], "amount": [50], "user": ["a"]})
    )
    result = restored.view("metrics")
    assert len(result) == 2
    row = result[result.user == "a"].iloc[0]
    assert (row.n, row.total, row["mean"]) == (5, 120, 24)
    assert result[result.user == "b"].iloc[0].total == 100
    for name, column in (("count", "n"), ("sum", "total"), ("avg", "mean")):
        pd.testing.assert_series_equal(restored.view(name)[column], result[column])
    assert any(removed > 0 for _, _, removed in updates)
    updates.clear()
    restored.apply(pd.DataFrame(columns=["message", "amount", "user"]))
    assert updates == []


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize(
    "amounts,total",
    [
        ([10, 30, 20], 100),
        ([1e16, -1e16, 1.0], 41),
        ([1e308, -1e308, 1e308], 1e308),
    ],
)
def test_numeric_business_field_changes_and_failure_are_atomic(
    fake_model, membership, amounts, total
):
    updates = []

    class CorruptibleAdapter(LotusAdapter):
        corrupt = False
        version = 1

        def execute(self, query, inputs):
            if query.op == "aggregate_state_update":
                updates.append(
                    tuple(len(inputs[leaf.params["name"]]) for leaf in query.inputs)
                )
            result = super().execute(query, inputs)
            if query.op == "sem_groupby":
                result["label"] = f"business label {self.version}"
                result["description"] = f"business description {self.version}"
            if self.corrupt and query.op == "aggregate_state_update":
                raise ValueError("downstream numeric update failed")
            return result

    source = am.Source(
        {
            "message": "Text",
            "amount": "Cost",
            "keep": "Included",
            "label": "Business label",
            "description": "Business description",
        }
    )
    groups = source.sem_groupby(
        input_cols=["message"], instruction="Group", membership=membership
    )
    kept = groups.filter(groups.col("keep"))
    metrics = kept.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )
    adapter = CorruptibleAdapter()
    flow = am.SemanticDataflow(
        source=source, views={"groups": groups, "metrics": metrics}, adapter=adapter
    )
    flow.apply(
        pd.DataFrame(
            {
                "message": ["shipping quality"] * 3,
                "amount": amounts,
                "keep": [True, True, True],
                "label": ["original"] * 3,
                "description": ["original"] * 3,
            }
        )
    )
    before = flow.view("metrics")
    updates.clear()
    adapter.version = 2
    # Recomputed business fields change; the triggering row is filtered out.
    flow.apply(
        pd.DataFrame(
            {
                "message": ["shipping"],
                "amount": [999],
                "keep": [False],
                "label": ["original"],
                "description": ["original"],
            }
        )
    )
    unchanged = flow.view("metrics")
    assert updates and any(
        inserted > 0 and removed > 0 for _, inserted, removed in updates
    )
    pd.testing.assert_frame_equal(
        unchanged[[GROUP_ID, "n", "total", "mean"]],
        before[[GROUP_ID, "n", "total", "mean"]],
    )
    assert list(unchanged.columns) == [GROUP_ID, "n", "total", "mean"]
    assert set(flow.view("groups").label) == {"business label 2"}
    assert set(flow.view("groups").description) == {"business description 2"}
    before_groups = flow.view("groups").copy()
    snapshot = flow.snapshot_state()
    adapter.corrupt = True
    with pytest.raises(ValueError, match="downstream numeric update failed"):
        flow.apply(
            pd.DataFrame(
                {
                    "message": ["shipping"],
                    "amount": [50],
                    "keep": [True],
                    "label": ["original"],
                    "description": ["original"],
                }
            )
        )
    pd.testing.assert_frame_equal(flow.view("metrics"), unchanged)
    pd.testing.assert_frame_equal(flow.view("groups"), before_groups)
    restored_adapter = CorruptibleAdapter()
    restored_adapter.version = 2
    restored = am.SemanticDataflow(
        source=source,
        views={"groups": groups, "metrics": metrics},
        adapter=restored_adapter,
    )
    restored.restore_state(snapshot)
    adapter.corrupt = False
    batch = pd.DataFrame(
        {
            "message": ["quality regroup"],
            "amount": [40],
            "keep": [True],
            "label": ["original"],
            "description": ["original"],
        }
    )
    flow.apply(batch)
    restored.apply(batch)
    pd.testing.assert_frame_equal(flow.view("groups"), restored.view("groups"))
    pd.testing.assert_frame_equal(flow.view("metrics"), restored.view("metrics"))
    row = flow.view("metrics").iloc[0]
    assert (row.n, row.total, row["mean"]) == (4, total, total / 4)


def test_numeric_group_output_guards_and_nulls(fake_model):
    source = am.Source({"message": "Text", "amount": "Cost"})
    groups = source.sem_groupby(instruction="Group")
    for name in (GROUP_ID,):
        with pytest.raises(ValueError, match="conflicts"):
            groups.count(output_col=name)
    with pytest.raises(NotImplementedError, match="cannot mix"):
        groups.agg(
            am.count(output_col="n"),
            am.sem_agg(output_cols=["summary"], instruction="Summarize"),
        )
    metrics = groups.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )
    filtered = groups.filter(groups.col(GROUP_ID) == json.dumps([0, 0])).count(
        output_col="n"
    )
    flow = am.SemanticDataflow(
        source=source,
        views={"metrics": metrics, "filtered": filtered},
        adapter=LotusAdapter(),
    )
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"], "amount": [None, 4]}))
    row = flow.view("metrics").set_index(GROUP_ID).loc[json.dumps([0, 0])]
    assert row.n == 1 and pd.isna(row.total) and pd.isna(row["mean"])
    flow.apply(pd.DataFrame({"message": ["quality regroup"], "amount": [6]}))
    assert flow.view("filtered").empty
    row = flow.view("metrics").iloc[0]
    assert (row.n, row.total, row["mean"]) == (3, 10, 5)
