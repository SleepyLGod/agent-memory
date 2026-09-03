"""Focused tests for shared policy differentiation and execution."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import agent_memory as am
import agent_memory.planner as planner
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.sem_flat_map import apply_flat_map_outputs
from agent_memory.policy.aggregates import (
    ArrayAggregateSpec,
    CollectListAggregateSpec,
    MinAggregateSpec,
    SemanticAggregateSpec,
)
from agent_memory.planner.legacy import differentiate_legacy_policy
from agent_memory.planner.rules import DifferentialRules
from agent_memory.planner.rules import GROUP_ID_COLUMN
from agent_memory.policy.logical import MemorySpec, MemoryView, QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.runtime import MemoryRuntime
from agent_memory.runtime.executor import NodeOutputUpdate, PolicyExecutor
from agent_memory.runtime.legacy import LegacyViewRuntime


def test_planner_exports_differentiator_components_not_free_functions() -> None:
    assert planner.QueryDifferentiator.__name__ == "QueryDifferentiator"
    assert planner.PolicyDifferentiator.__name__ == "PolicyDifferentiator"
    assert not hasattr(planner, "differentiate_query")
    assert not hasattr(planner, "differentiate_policy")


def test_query_differentiator_can_be_reused_without_state_leakage() -> None:
    log = am.Log({"message": "Message.", "role": "Role."})
    first = MemoryView("first", log.select(["message"]).expr)
    second = MemoryView("second", log.select(["role"]).expr)
    differentiator = planner.QueryDifferentiator()

    first_result = differentiator.differentiate(first)
    second_result = differentiator.differentiate(second)

    assert first_result == planner.QueryDifferentiator().differentiate(first)
    assert second_result == planner.QueryDifferentiator().differentiate(second)


def test_policy_differentiator_can_be_reused_without_builder_state_leakage() -> None:
    class FirstMemory(am.Memory):
        log = am.Log({"message": "Message."})
        rows = log.select(["message"])

    class SecondMemory(am.Memory):
        log = am.Log({"message": "Message."})
        rows = log.assign(kind="second")

    differentiator = planner.PolicyDifferentiator()

    first = differentiator.differentiate(FirstMemory.spec())
    second = differentiator.differentiate(SecondMemory.spec())

    assert tuple(first.view_outputs) == ("rows",)
    assert tuple(second.view_outputs) == ("rows",)
    assert first.fingerprint != second.fingerprint


def test_node_output_update_preserves_bag_multiplicity_and_nested_values() -> None:
    old = pd.DataFrame(
        [
            {"id": 1, "value": {"items": ["a"]}},
            {"id": 1, "value": {"items": ["a"]}},
            {"id": 2, "value": None},
        ]
    )
    new = pd.DataFrame(
        [
            {"id": 1, "value": {"items": ["a"]}},
            {"id": 3, "value": (4, 0)},
        ]
    )

    update = NodeOutputUpdate.between(old, new)

    assert update.retracted_rows.to_dict("records") == [
        {"id": 1, "value": {"items": ["a"]}},
        {"id": 2, "value": None},
    ]
    assert update.inserted_rows.to_dict("records") == [
        {"id": 3, "value": (4, 0)},
    ]


def test_node_output_update_empty_uses_output_schema() -> None:
    frame = pd.DataFrame(columns=["name", "body"])

    update = NodeOutputUpdate.between(frame, frame.copy())

    assert update.is_empty
    assert list(update.inserted_rows.columns) == ["name", "body"]
    assert list(update.retracted_rows.columns) == ["name", "body"]


def test_policy_compiler_shares_structurally_equal_subtrees() -> None:
    class SharedMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        _selected_once = log.select(["message"])
        first = _selected_once.assign(kind="first")
        second = log.select(["message"]).assign(kind="second")

    policy = SharedMemory.differentiate_policy()
    select_nodes = [node for node in policy.nodes.values() if node.query.op == "select"]

    assert len(select_nodes) == 1
    assert policy.view_outputs.keys() == {"first", "second"}
    assert policy.execution_order.index(select_nodes[0].node_id) < policy.execution_order.index(
        policy.view_outputs["first"]
    )


def test_policy_compiler_fuses_semantic_grouped_aggregate_pattern() -> None:
    class GroupedMemory(am.Memory):
        log = am.Log({"name": "Name.", "body": "Body."})
        topics = log.sem_groupby(
            input_cols=["name"],
            instruction="Rows with the same {name} describe one topic.",
        ).sem_agg(
            input_cols=["name", "body"],
            output_cols={"name": "Topic name.", "body": "Merged body."},
            instruction="Merge {name} and {body} into one topic.",
        )

    plan = GroupedMemory.differentiate_policy()
    aggregate_nodes = [
        node for node in plan.nodes.values() if node.execution_kind == "semantic_state"
    ]

    assert len(aggregate_nodes) == 1
    assert aggregate_nodes[0].query.op == "sem_agg"
    assert not any(node.query.op == "sem_groupby" for node in plan.nodes.values())


def test_policy_compiler_supports_append_only_inner_semantic_join() -> None:
    class SemanticJoinMemory(am.Memory):
        log = am.Log({"message": "Message."})
        joined = log.sem_join(log, instruction="{message:left} matches {message:right}.")

    plan = SemanticJoinMemory.differentiate_policy()
    node = next(node for node in plan.nodes.values() if node.query.op == "sem_join")

    assert node.execution_kind == "semantic_binary_state"
    assert node.maintenance_query is not None


def test_policy_compiler_rejects_view_time_semantic_topk() -> None:
    class UnsupportedMemory(am.Memory):
        log = am.Log({"message": "Message."})
        ranked = log.sem_topk("Rank {message}.", 1)

    with pytest.raises(NotImplementedError, match="View-time sem_topk"):
        UnsupportedMemory.differentiate_policy()


def test_policy_compiler_rejects_public_view_dependency_cycle() -> None:
    log = am.Log({"message": "Message."})
    spec = MemorySpec(
        log=log,
        views={
            "first": MemoryView(
                "first",
                QueryExpr(op="materialized_view", params={"name": "second"}),
            ),
            "second": MemoryView(
                "second",
                QueryExpr(op="materialized_view", params={"name": "first"}),
            ),
        },
        private_relations={},
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        planner.PolicyDifferentiator().differentiate(spec)


def test_differentiated_policy_fingerprint_is_stable() -> None:
    class StableMemory(am.Memory):
        log = am.Log({"message": "Message."})
        rows = log.select(["message"])

    first = StableMemory.differentiate_policy()
    second = StableMemory.differentiate_policy()

    assert first.execution_order == second.execution_order
    assert first.fingerprint == second.fingerprint


def test_policy_fingerprint_includes_the_selected_rule() -> None:
    class GroupedMemory(am.Memory):
        log = am.Log({"name": "Name.", "body": "Body."})
        topics = log.sem_groupby(
            input_cols=["name"],
            instruction="Rows with the same {name} describe one topic.",
        ).sem_agg(
            input_cols=["name", "body"],
            output_cols={"name": "Topic name.", "body": "Merged body."},
            instruction="Merge {name} and {body} into one topic.",
        )

    compressed = planner.PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="compressed"),
    ).differentiate(GroupedMemory.spec())
    join_map = planner.PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="join-map"),
    ).differentiate(GroupedMemory.spec())

    assert compressed.fingerprint != join_map.fingerprint
    assert join_map.grouped_agg_rule == "rule-join-map"


def test_policy_compiler_rejects_process_window_external_relation_capture() -> None:
    source_log = am.Log({"message": "Message body."})
    external = source_log.select(["message"])

    class InvalidWindowMemory(am.Memory):
        log = source_log
        _external = external
        blocks = source_log.count_window(size=2).process_window(
            lambda window: window.join(external, on="message")
        )

    with pytest.raises(ValueError, match="process_window callback.*window relation"):
        InvalidWindowMemory.differentiate_policy()


class _CountingAdapter:
    def __init__(self, *, fail_on_op: str | None = None) -> None:
        self._delegate = LotusAdapter()
        self.fail_on_op = fail_on_op
        self.calls: list[str] = []

    def execute(self, query: am.QueryExpr, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.calls.append(query.op)
        if query.op == self.fail_on_op:
            raise RuntimeError(f"injected {query.op} failure")
        return self._delegate.execute(query, inputs)


class _SemanticAggregateAdapter(LotusAdapter):
    """Deterministic semantic stub that exposes root maintenance calls."""

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self.root_queries: list[am.QueryExpr] = []
        self.fail_query: am.QueryExpr | None = None

    def execute(
        self,
        query: am.QueryExpr,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        is_root = self._depth == 0
        self._depth += 1
        try:
            if is_root:
                self.root_queries.append(query)
                if query is self.fail_query:
                    raise RuntimeError("injected semantic recompute failure")
            if query.op == "sem_filter":
                source = self.execute(query.inputs[0], inputs)
                return source.loc[source["value"] >= 5].copy()
            if query.op == "sem_groupby":
                source = self.execute(query.inputs[0], inputs).copy()
                source[GROUP_ID_COLUMN] = pd.factorize(source["key"], sort=False)[0]
                return source
            if query.op == "sem_agg":
                return self._execute_sem_agg(query, inputs)
            return super().execute(query, inputs)
        finally:
            self._depth -= 1

    def _execute_sem_agg(
        self,
        query: am.QueryExpr,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        source = self.execute(query.inputs[0], inputs)
        output_names = [column.name for column in query.params["output_cols"]]
        if source.empty:
            return pd.DataFrame(columns=output_names)
        groups = (
            source.groupby(GROUP_ID_COLUMN, sort=False, dropna=False)
            if GROUP_ID_COLUMN in source.columns
            else [(0, source)]
        )
        rows: list[dict[str, object]] = []
        for _, group in groups:
            row: dict[str, object] = {}
            for output_name in output_names:
                if output_name == "key":
                    row[output_name] = group.iloc[0]["key"]
                    continue
                source_column = "summary" if "summary" in group.columns else "value"
                values = [
                    str(value)
                    for value in group[source_column]
                    if not _missing(value)
                ]
                row[output_name] = " | ".join(dict.fromkeys(values))
            rows.append(row)
        return pd.DataFrame(rows, columns=output_names)


class _SemanticAggregateMemory(am.Memory):
    log = am.Log({"key": "Group key.", "value": "Numeric value."})
    _minimum = log.group_by("key").min(column="value", output_col="value")
    _summary_state = _minimum.sem_groupby(
        input_cols=["key"],
        instruction="Rows with the same {key} belong together.",
    ).sem_agg(
        input_cols=["key", "value"],
        output_cols={"key": "Group key.", "summary": "Combined values."},
        instruction="Summarize {key} and {value}.",
    )
    summaries = _summary_state.assign(propagated=True)


class _RetractOnlySemanticAggregateMemory(am.Memory):
    log = am.Log({"key": "Group key.", "value": "Numeric value."})
    _minimum = log.group_by("key").min(column="value", output_col="value")
    _accepted = _minimum.sem_filter(
        instruction="Keep {value} when it is at least five."
    )
    summaries = _accepted.sem_groupby(
        input_cols=["key"],
        instruction="Rows with the same {key} belong together.",
    ).sem_agg(
        input_cols=["key", "value"],
        output_cols={"key": "Group key.", "summary": "Combined values."},
        instruction="Summarize {key} and {value}.",
    )


def _only_semantic_state_node(
    memory_type: type[am.Memory],
) -> planner.DifferentialNode:
    nodes = [
        node
        for node in memory_type.differentiate_policy().nodes.values()
        if node.execution_kind == "semantic_state"
    ]
    assert len(nodes) == 1
    return nodes[0]


def test_runtime_executes_shared_node_once_and_fans_out() -> None:
    class SharedMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        _selected = log.select(["message"])
        first = _selected.assign(first=True)
        second = _selected.assign(second=True)

    adapter = _CountingAdapter()
    memory = SharedMemory(adapter=adapter)

    memory.add({"message": "hello"})

    assert adapter.calls.count("select") == 1
    assert adapter.calls.count("assign") == 2
    assert memory._runtime._state["first"].to_dict("records") == [
        {"message": "hello", "first": True}
    ]
    assert memory._runtime._state["second"].to_dict("records") == [
        {"message": "hello", "second": True}
    ]


def test_runtime_skips_descendant_when_parent_change_is_empty() -> None:
    class FilteredMemory(am.Memory):
        log = am.Log({"message": "Message body.", "keep": "Keep flag."})
        kept = log.filter(log.col("keep") == True).assign(seen=True)  # noqa: E712

    adapter = _CountingAdapter()
    memory = FilteredMemory(adapter=adapter)

    memory.add({"message": "ignored", "keep": False})

    assert adapter.calls == ["filter"]
    assert memory._runtime._state["kept"].empty


def test_runtime_stages_all_nodes_before_atomic_commit() -> None:
    class AtomicMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        selected = log.select(["message"])
        failed = selected.assign(failed=True)

    memory = AtomicMemory(adapter=_CountingAdapter(fail_on_op="assign"))

    with pytest.raises(RuntimeError, match="injected assign failure"):
        memory.add({"message": "not committed"})

    assert memory._runtime._state == {}
    assert memory._runtime._engine.node_state == {}


def test_semantic_state_uses_maintenance_query_for_insert_only_change() -> None:
    adapter = _SemanticAggregateAdapter()
    memory = _SemanticAggregateMemory(adapter=adapter)
    node = _only_semantic_state_node(_SemanticAggregateMemory)

    memory.add({"key": "a", "value": 5})
    adapter.root_queries.clear()
    memory.add({"key": "b", "value": 7})

    assert any(query is node.maintenance_query for query in adapter.root_queries)
    assert not any(query is node.query for query in adapter.root_queries)


def test_semantic_state_recomputes_node_on_parent_replacement_and_propagates() -> None:
    adapter = _SemanticAggregateAdapter()
    memory = _SemanticAggregateMemory(adapter=adapter)
    node = _only_semantic_state_node(_SemanticAggregateMemory)

    memory.add({"key": "a", "value": 5})
    adapter.root_queries.clear()
    memory.add({"key": "a", "value": 3})

    assert any(query is node.query for query in adapter.root_queries)
    assert not any(query is node.maintenance_query for query in adapter.root_queries)
    assert memory._runtime._state["summaries"].to_dict("records") == [
        {"key": "a", "summary": "3", "propagated": True}
    ]


def test_semantic_state_recomputes_empty_result_for_retract_only_change() -> None:
    adapter = _SemanticAggregateAdapter()
    memory = _RetractOnlySemanticAggregateMemory(adapter=adapter)
    node = _only_semantic_state_node(_RetractOnlySemanticAggregateMemory)

    memory.add({"key": "a", "value": 5})
    adapter.root_queries.clear()
    memory.add({"key": "a", "value": 3})

    assert any(query is node.query for query in adapter.root_queries)
    assert memory._runtime._state["summaries"].empty


def test_semantic_state_recompute_failure_rolls_back_the_whole_step() -> None:
    adapter = _SemanticAggregateAdapter()
    memory = _SemanticAggregateMemory(adapter=adapter)
    node = _only_semantic_state_node(_SemanticAggregateMemory)
    memory.add({"key": "a", "value": 5})
    before = memory._runtime.snapshot_state()
    adapter.fail_query = node.query

    with pytest.raises(RuntimeError, match="injected semantic recompute failure"):
        memory.add({"key": "a", "value": 3})

    after = memory._runtime.snapshot_state()
    assert after["next_occurrence"] == before["next_occurrence"]
    assert after["state"]["log"].to_dict("records") == before["state"]["log"].to_dict(
        "records"
    )
    assert after["state"]["summaries"].to_dict("records") == before["state"][
        "summaries"
    ].to_dict("records")


def test_runtime_v2_snapshot_round_trip_and_plan_match() -> None:
    class SnapshotMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        rows = log.select(["message"])

    original = SnapshotMemory(adapter=_CountingAdapter())
    original.add({"message": "first"})
    snapshot = original._runtime.snapshot_state()

    restored = SnapshotMemory(adapter=_CountingAdapter())
    restored._runtime.restore_state(snapshot)
    restored.add({"message": "second"})

    assert snapshot["schema_version"] == 2
    assert restored._runtime._state["rows"].to_dict("records") == [
        {"message": "first"},
        {"message": "second"},
    ]
    assert isinstance(restored._runtime._engine, PolicyExecutor)


def test_runtime_v2_snapshot_rejects_a_different_policy_plan() -> None:
    class OriginalMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        rows = log.select(["message"])

    class ChangedMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        rows = log.select(["message"]).assign(kind="changed")

    original = OriginalMemory(adapter=_CountingAdapter())
    original.add({"message": "first"})
    snapshot = original._runtime.snapshot_state()

    changed = ChangedMemory(adapter=_CountingAdapter())
    with pytest.raises(ValueError, match="plan fingerprint"):
        changed._runtime.restore_state(snapshot)


def test_runtime_v1_snapshot_switches_to_isolated_legacy_engine() -> None:
    class LegacyMemory(am.Memory):
        log = am.Log({"message": "Message body."})
        rows = log.select(["message"])
        retrieval_query = rows

    memory = LegacyMemory(adapter=_CountingAdapter())
    memory._runtime.restore_state(
        {
            "schema_version": 1,
            "state": {},
            "window_next_start": {},
            "upstream_log_count": {},
        }
    )
    memory.add({"message": "legacy"})

    assert isinstance(memory._runtime._engine, LegacyViewRuntime)
    assert memory._runtime.snapshot_state()["schema_version"] == 1
    assert memory._runtime._state["rows"].to_dict("records") == [
        {"message": "legacy"}
    ]
    assert memory.query("unused").to_dict("records") == [{"message": "legacy"}]


def test_runtime_v1_restore_preserves_the_policy_grouped_rule() -> None:
    class GroupedMemory(am.Memory):
        log = am.Log({"name": "Name.", "body": "Body."})
        topics = log.sem_groupby(
            input_cols=["name"],
            instruction="Rows with the same {name} describe one topic.",
        ).sem_agg(
            input_cols=["name", "body"],
            output_cols={"name": "Topic name.", "body": "Merged body."},
            instruction="Merge {name} and {body} into one topic.",
        )

    policy = planner.PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule="join-map"),
    ).differentiate(GroupedMemory.spec())
    runtime = MemoryRuntime(policy, adapter=_CountingAdapter())
    expected = differentiate_legacy_policy(
        GroupedMemory.spec(),
        grouped_agg_rule="rule-join-map",
    )

    runtime.restore_state(
        {
            "schema_version": 1,
            "state": {},
            "window_next_start": {},
            "upstream_log_count": {},
        }
    )

    assert isinstance(runtime._engine, LegacyViewRuntime)
    assert runtime._engine.policy.view_queries["topics"] == expected.view_queries["topics"]


def test_semantic_output_cache_retracts_replaced_input_without_rerunning_old_row() -> None:
    class LineageMemory(am.Memory):
        log = am.Log({"key": "Group key.", "value": "Numeric value."})
        mapped = (
            log.group_by("key")
            .min(column="value", output_col="value")
            .sem_map(
                input_cols=["key", "value"],
                output_cols={"label": "Mapped value."},
                instruction="Map {key} and {value}.",
            )
        )

    class SemanticAdapter(_CountingAdapter):
        def execute(
            self,
            query: am.QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            self.calls.append(query.op)
            if query.op == "sem_map":
                source = inputs[str(query.inputs[0].params["name"])]
                result = source.copy()
                result["label"] = [f"value-{value}" for value in source["value"]]
                return result
            return self._delegate.execute(query, inputs)

    adapter = SemanticAdapter()
    memory = LineageMemory(adapter=adapter)

    memory.add({"key": "a", "value": 5})
    memory.add({"key": "a", "value": 3})
    memory.add({"key": "a", "value": 7})

    assert adapter.calls.count("sem_map") == 2
    assert memory._runtime._state["mapped"].to_dict("records") == [
        {"key": "a", "value": 3, "label": "value-3"}
    ]


def test_sem_flat_map_output_repeats_private_source_occurrence_index() -> None:
    from agent_memory.policy.logical import ColumnSpec

    source = pd.DataFrame([{"message": "one"}], index=["source:7"])

    result = apply_flat_map_outputs(
        source,
        [[{"topic": "a"}, {"topic": "b"}]],
        (ColumnSpec("topic"),),
    )

    assert list(result.index) == ["source:7", "source:7"]


def test_sem_flat_map_lineage_retracts_multi_and_zero_row_outputs() -> None:
    class LineageMemory(am.Memory):
        log = am.Log({"key": "Group key.", "value": "Numeric value."})
        expanded = (
            log.group_by("key")
            .min(column="value", output_col="value")
            .sem_flat_map(
                input_cols=["key", "value"],
                output_cols={"label": "Expanded label."},
                instruction="Expand {key} and {value}.",
            )
        )

    class FlatMapAdapter(_CountingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.semantic_input_columns: list[tuple[str, ...]] = []

        def execute(
            self,
            query: am.QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            self.calls.append(query.op)
            if query.op != "sem_flat_map":
                return self._delegate.execute(query, inputs)

            source = inputs[str(query.inputs[0].params["name"])]
            self.semantic_input_columns.append(tuple(source.columns))
            parsed = []
            for value in source["value"]:
                if value == 5:
                    parsed.append([{"label": "five-a"}, {"label": "five-b"}])
                elif value == 3:
                    parsed.append([])
                else:
                    parsed.append([{"label": f"value-{value}"}])
            return apply_flat_map_outputs(
                source,
                parsed,
                tuple(query.params["output_cols"]),
            )

    adapter = FlatMapAdapter()
    memory = LineageMemory(adapter=adapter)

    memory.add({"key": "a", "value": 5})
    assert len(memory._runtime._state["expanded"]) == 2

    memory.add({"key": "a", "value": 3})
    assert memory._runtime._state["expanded"].empty

    memory.add({"key": "a", "value": 7})
    assert memory._runtime._state["expanded"].empty
    assert adapter.calls.count("sem_flat_map") == 2
    assert adapter.semantic_input_columns == [("key", "value"), ("key", "value")]


def test_sem_filter_lineage_tracks_rejected_and_replaced_occurrences() -> None:
    class LineageMemory(am.Memory):
        log = am.Log({"key": "Group key.", "value": "Numeric value."})
        accepted = (
            log.group_by("key")
            .min(column="value", output_col="value")
            .sem_filter(instruction="Keep {value} when it is at most three.")
        )

    class FilterAdapter(_CountingAdapter):
        def execute(
            self,
            query: am.QueryExpr,
            inputs: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            self.calls.append(query.op)
            if query.op == "sem_filter":
                source = inputs[str(query.inputs[0].params["name"])]
                return source.loc[source["value"] <= 3].copy()
            return self._delegate.execute(query, inputs)

    adapter = FilterAdapter()
    memory = LineageMemory(adapter=adapter)

    memory.add({"key": "a", "value": 5})
    assert memory._runtime._state["accepted"].empty

    memory.add({"key": "a", "value": 3})
    assert memory._runtime._state["accepted"].to_dict("records") == [
        {"key": "a", "value": 3}
    ]

    memory.add({"key": "a", "value": 2})
    memory.add({"key": "a", "value": 7})
    assert memory._runtime._state["accepted"].to_dict("records") == [
        {"key": "a", "value": 2}
    ]
    assert adapter.calls.count("sem_filter") == 3


class _ClaudeStubAdapter(LotusAdapter):
    def execute(
        self,
        query: am.QueryExpr,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if query.op == "sem_flat_map":
            source = self.execute(query.inputs[0], inputs)
            output_cols = tuple(query.params["output_cols"])
            parsed = [
                [
                    {
                        "name": "documentation_preference",
                        "description": "Documentation preference.",
                        "type": "user",
                        "body": str(row["message"]),
                    }
                ]
                for _, row in source.iterrows()
            ]
            return apply_flat_map_outputs(source, parsed, output_cols)
        if query.op == "sem_groupby":
            source = self.execute(query.inputs[0], inputs).copy()
            source[GROUP_ID_COLUMN] = pd.factorize(source["name"], sort=False)[0]
            return source
        if query.op == "sem_agg":
            source = self.execute(query.inputs[0], inputs)
            output_names = [column.name for column in query.params["output_cols"]]
            groups = (
                source.groupby(GROUP_ID_COLUMN, sort=False, dropna=False)
                if GROUP_ID_COLUMN in source.columns
                else [(0, source)]
            )
            rows: list[dict[str, object]] = []
            for _, group in groups:
                row: dict[str, object] = {}
                for name in output_names:
                    values = [value for value in group.get(name, ()) if pd.notna(value)]
                    row[name] = " | ".join(dict.fromkeys(map(str, values))) if values else None
                rows.append(row)
            return pd.DataFrame(rows, columns=output_names)
        if query.op == "sem_map":
            source = self.execute(query.inputs[0], inputs).copy()
            for column in query.params["output_cols"]:
                if column.name == "hook":
                    source[column.name] = source["description"]
                else:
                    source[column.name] = column.name
            return source
        return super().execute(query, inputs)


def test_claude_runtime_propagates_topic_replacement_without_stale_catalog() -> None:
    memory = am.ClaudeMemory(adapter=_ClaudeStubAdapter())

    memory.add({"message": "I prefer short design docs."})
    memory.add({"message": "Keep architecture notes concise."})

    topics = memory._runtime._state["topics"]
    catalog = memory._runtime._state["catalog"]
    assert len(topics) == 1
    assert len(catalog) == 1
    assert set(topics.loc[0, "body"].split(" | ")) == {
        "I prefer short design docs.",
        "Keep architecture notes concise.",
    }
    assert catalog.loc[0, "name"] == "documentation_preference"
    assert catalog.loc[0, "catalog_title"] == catalog.loc[0, "name"]
    assert isinstance(catalog.loc[0, "hook"], str)

    snapshot = memory._runtime.snapshot_state()
    assert snapshot["schema_version"] == 2
    restored = am.ClaudeMemory(adapter=_ClaudeStubAdapter())
    restored._runtime.restore_state(snapshot)
    restored.add({"message": "Prefer direct wording."})

    restored_topics = restored._runtime._state["topics"]
    restored_catalog = restored._runtime._state["catalog"]
    assert len(restored_catalog) == len(restored_topics) == 1
    assert restored_catalog.loc[0, "catalog_title"] == restored_catalog.loc[0, "name"]


class _ZepStubAdapter(LotusAdapter):
    def __init__(self, *, duplicate_entity_mentions: bool = False) -> None:
        super().__init__()
        self.duplicate_entity_mentions = duplicate_entity_mentions
        self.fact_extraction_episode_ids: list[str] = []

    def execute(
        self,
        query: am.QueryExpr,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if query.op == "sem_flat_map":
            return self._flat_map(query, inputs)
        if query.op == "sem_groupby":
            source = self.execute(query.inputs[0], inputs).copy()
            grouping_columns = (
                *tuple(query.params.get("partition_by", ())),
                *tuple(query.params["input_cols"]),
            )
            keys = (
                source.loc[:, list(grouping_columns)]
                .astype(str)
                .agg("|".join, axis=1)
            )
            source[GROUP_ID_COLUMN] = pd.factorize(keys, sort=False)[0]
            return source
        if query.op == "agg":
            return self._aggregate(query, inputs)
        if query.op == "sem_filter":
            source = self.execute(query.inputs[0], inputs)
            earlier = source.get("fact:earlier_added", pd.Series(index=source.index, dtype=object))
            later = source.get("fact:later_added", pd.Series(index=source.index, dtype=object))
            return source.loc[
                earlier.astype(str).str.contains("likes")
                & later.astype(str).str.contains("dislikes")
            ].copy()
        return super().execute(query, inputs)

    def _flat_map(
        self,
        query: am.QueryExpr,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        source = self.execute(query.inputs[0], inputs)
        columns = tuple(query.params["output_cols"])
        names = {column.name for column in columns}
        parsed: list[list[dict[str, object]]] = []
        for _, row in source.iterrows():
            if names == {"name"}:
                parsed.append(
                    [
                        {"name": "Alice"},
                        {
                            "name": (
                                "Alice" if self.duplicate_entity_mentions else "Tea"
                            )
                        },
                    ]
                )
            else:
                self.fact_extraction_episode_ids.append(str(row["episode_id"]))
                content = str(row["content"])
                relation = "dislikes" if "dislikes" in content else "likes"
                parsed.append(
                    [
                        {
                            "source_entity_ordinal": 0,
                            "target_entity_ordinal": 1,
                            "relation_type": relation,
                            "fact": f"Alice {relation} tea",
                            "valid_at": row["reference_time"],
                            "invalid_at": None,
                        }
                    ]
                )
        return apply_flat_map_outputs(
            source,
            parsed,
            columns,
            ordinal_col=query.params.get("ordinal_col"),
        )

    def _aggregate(
        self,
        query: am.QueryExpr,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        group_query = query.inputs[0]
        source = self.execute(group_query, inputs)
        expected = list(output_columns(query))
        if source.empty:
            return pd.DataFrame(columns=expected)
        if group_query.op == "group_by":
            keys = tuple(group_query.params["keys"])
            grouped = source.groupby(list(keys), sort=False, dropna=False)
            groups = [
                (dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)), group)
                for key, group in grouped
            ]
        else:
            partition_by = tuple(group_query.params.get("partition_by", ()))
            groups = [
                (
                    {
                        column: group.iloc[0][column]
                        for column in partition_by
                    },
                    group,
                )
                for _, group in source.groupby(GROUP_ID_COLUMN, sort=False)
            ]

        rows: list[dict[str, object]] = []
        for key_values, group in groups:
            row: dict[str, object] = dict(key_values)
            for aggregate in query.params["aggregates"]:
                if isinstance(aggregate, SemanticAggregateSpec):
                    for column in aggregate.output_cols:
                        values = [
                            value
                            for value in group.get(column.name, ())
                            if not _missing(value)
                        ]
                        if column.name == "summary":
                            row[column.name] = " | ".join(dict.fromkeys(map(str, values)))
                        else:
                            row[column.name] = values[0] if values else column.name
                elif isinstance(aggregate, ArrayAggregateSpec):
                    records = group.loc[:, list(aggregate.columns)].where(
                        pd.notna(group.loc[:, list(aggregate.columns)]),
                        None,
                    )
                    row[aggregate.output_col] = json.dumps(
                        records.to_dict("records"),
                        default=str,
                    )
                elif isinstance(aggregate, MinAggregateSpec):
                    values = [
                        tuple(item[column] for column in aggregate.columns)
                        if len(aggregate.columns) > 1
                        else item[aggregate.columns[0]]
                        for _, item in group.iterrows()
                    ]
                    values = [value for value in values if not _missing(value)]
                    row[aggregate.output_col] = min(values) if values else None
                elif isinstance(aggregate, CollectListAggregateSpec):
                    row[aggregate.output_col] = json.dumps(
                        list(group[aggregate.column]),
                        default=str,
                    )
                else:
                    raise AssertionError(type(aggregate).__name__)
            rows.append(row)
        return pd.DataFrame(rows, columns=expected)


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, tuple):
        return any(_missing(item) for item in value)
    result = pd.isna(value)
    return isinstance(result, bool) and result


def test_zep_runtime_propagates_temporal_fact_and_community_dependencies() -> None:
    adapter = _ZepStubAdapter()
    memory = am.ZepMemoryExtended(adapter=adapter)

    memory.add(
        {
            "content": "Alice likes tea.",
            "role": "user",
            "speaker": "Alice",
            "reference_time": "2026-01-01T00:00:00Z",
            "source_description": "conversation",
        }
    )
    memory.add(
        {
            "content": "Alice dislikes tea.",
            "role": "user",
            "speaker": "Alice",
            "reference_time": "2026-02-01T00:00:00Z",
            "source_description": "conversation",
        }
    )

    state = memory._runtime._state
    assert set(state) == {"log", "episodes", "entities", "facts", "communities"}
    assert len(state["episodes"]) == 2
    assert not state["entities"].empty
    assert set(state["entities"]["entity_type"]) == {"Entity"}
    assert not state["facts"].empty
    assert not state["communities"].empty
    liked = state["facts"].loc[state["facts"]["relation_type"] == "likes"]
    assert liked["invalid_at"].notna().all()
    assert len(adapter.fact_extraction_episode_ids) == 2
    assert len(set(adapter.fact_extraction_episode_ids)) == 2


def test_zep_runtime_drops_distinct_mentions_resolved_to_the_same_entity() -> None:
    memory = am.ZepMemory(adapter=_ZepStubAdapter(duplicate_entity_mentions=True))

    memory.add(
        {
            "content": "Alice refers to herself.",
            "role": "user",
            "speaker": "Alice",
            "reference_time": "2026-01-01T00:00:00Z",
            "source_description": "conversation",
        }
    )

    state = memory._runtime._state
    assert len(state["entities"]) == 1
    assert state["entities"].iloc[0]["name"] == "Alice"
    assert state["entities"].iloc[0]["entity_id"] == (0, 0)
    assert state["facts"].empty
