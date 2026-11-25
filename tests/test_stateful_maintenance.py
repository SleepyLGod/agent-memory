"""Offline Agent/state contracts; mock success is not semantic sufficiency."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pandas as pd
import pytest
from jsonschema import ValidationError

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import (
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.stateful import expand_assignments
from agent_memory.maintenance import object_schema, validate_json, validate_state_schema
from agent_memory.planner.rules import DifferentialRules
from agent_memory.policy.grouping import GROUP_ID, RECORD_ID
from agent_memory.policy.logical import QueryExpr


@pytest.fixture
def lm(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace only the model boundary, retaining formatter/parser/adapter/runtime."""
    import lotus

    class Model:
        max_ctx_len = 100_000
        max_tokens = 512

        def __init__(self) -> None:
            self.calls = []
            self.handler = lambda payload: {}
            self.metadata = None
            self.raw = None

        def count_tokens(self, messages: Any) -> int:
            return len(json.dumps(messages)) // 4

        def __call__(self, prompts: Any, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(deepcopy((prompts, kwargs)))
            if kwargs.get("response_format", {}).get("type") == "json_object":
                for messages in prompts:
                    if not any(
                        message["role"] in {"system", "user"}
                        and "json" in message["content"].lower()
                        for message in messages
                    ):
                        raise ValueError("JSON mode requires json in the prompt")
            payload = json.JSONDecoder().raw_decode(
                prompts[0][-1]["content"].split("Context:\n", 1)[1]
            )[0]
            result = self.handler(payload)
            return SimpleNamespace(
                outputs=[self.raw if self.raw is not None else json.dumps(result)],
                response_metadata=self.metadata,
            )

    model = Model()
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda self: None)
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    return model


@pytest.mark.parametrize("transport", ["chat-json-object", "responses-json-schema"])
def test_schema_execution_declares_json_without_task_or_example_help(
    lm: Any, transport: str
) -> None:
    from agent_memory.adapters.lotus.structured import execute_schema_lm

    schema = object_schema({"answer": {"type": "string"}})
    context = {"New inputs": [{"message": "Alice"}]}
    instruction = "Return the names."
    lm.handler = lambda payload: {"answer": "Alice"}
    result = execute_schema_lm(
        context=context,
        instruction=instruction,
        requirements="Return the requested answer.",
        schema=schema,
        config=LotusExecutionConfig(structured_output_transport=transport),
        operator="offline-schema-contract",
    )
    assert result == {"answer": "Alice"}
    assert len(lm.calls) == 1
    messages = lm.calls[0][0][0]
    assert len(messages) == 2  # no few-shot supplies the format requirement
    assert any(
        m["role"] in {"system", "user"} and "json" in m["content"].lower()
        for m in messages
    )
    assert instruction in messages[-1]["content"]
    assert request_payload(lm.calls[0]) == context
    rendered_schema = messages[-1]["content"].split("Response schema:\n", 1)[1]
    assert json.loads(rendered_schema) == schema


def state_plan(
    target: am.Relation, *, group: bool = False, index: int = 0
) -> am.MaintenancePlan:
    if group:
        schema = object_schema(
            {
                "items": {
                    "type": "array",
                    "items": object_schema(
                        {
                            "record_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "topic": {"type": "string"},
                        }
                    ),
                },
                "revision": {"type": "integer"},
            }
        )
        binding = {"objects": "items", "record_ids": "record_ids"}
    else:
        schema = object_schema({"count": {"type": "integer", "minimum": 0}})
        binding = None
    return am.MaintenancePlan.bind(
        target,
        {
            "rule": "group-state" if group else "agg-state",
            "state_schema": {"value": schema, "membership": binding},
            "maintenance_instruction": "Retain task-relevant observations and update the original result.",
            "rationale": "Offline fixture, not a proven sufficient state.",
        },
        aggregate_index=index,
    )


def make_flow(
    source: am.Source, views: dict, plans: tuple, **kwargs: Any
) -> am.SemanticDataflow:
    return am.SemanticDataflow(
        source=source,
        views=views,
        adapter=LotusAdapter(**kwargs),
        rules=DifferentialRules(sem_agg_rule="agg-fallback", maintenance_plans=plans),
    )


def counter(payload: dict) -> dict:
    previous = payload["Previous state"]
    count = (0 if previous is None else previous["count"]) + len(payload["New inputs"])
    return {
        "result": {"answer": "repeated" if count >= 2 else ""},
        "state": {"count": count},
    }


def grouper(payload: dict) -> dict:
    previous = payload["Previous state"]
    items = [] if previous is None else deepcopy(previous["items"])
    items.extend(
        {"record_ids": [r["record_id"]], "topic": r["values"]["message"]}
        for r in payload["New inputs"]
    )
    revision = 0 if previous is None else previous["revision"] + 1
    assignments = []
    for i, item in enumerate(items):
        groups = [0] if revision == 0 else ([0] if item["topic"] == "shipping" else [1])
        if (
            item["topic"] == "both"
            and payload["Constraints"]["membership"] == "multiple"
        ):
            groups = [0, 1]
        assignments.extend({"object": i, "group": g} for g in groups)
    return {"state": {"items": items, "revision": revision}, "assignments": assignments}


def test_explicit_agent_no_state_and_frozen_binding(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(
        output_cols=["answer"], instruction="List languages {literally}"
    )
    config = {
        "rule": "agg-composable",
        "state_schema": None,
        "maintenance_instruction": None,
        "rationale": "The answer is the complete set.",
    }
    lm.handler = lambda payload: config
    plan = am.plan_maintenance(
        query, adapter=LotusAdapter(), examples=[{"message": "Python"}]
    )
    assert plan.to_dict() == config
    prompts = lm.calls[0][0][0]
    assert len(prompts) == 2  # one context with designs; requirements appear once
    assert "List languages {literally}" in prompts[-1]["content"]
    assert request_payload(lm.calls[0])["Development inputs"] == [{"message": "Python"}]
    make_flow(source, {"out": query}, (plan,))
    assert len(lm.calls) == 1  # compilation never calls the Agent
    with pytest.raises(ValueError, match="target is not present"):
        make_flow(
            source,
            {"out": source.sem_agg(output_cols=["answer"], instruction="Other")},
            (plan,),
        )


@pytest.mark.parametrize("grouping", [False, True])
@pytest.mark.parametrize("transport", ["chat-json-object", "responses-json-schema"])
@pytest.mark.parametrize("rule_suffix", ["composable", "fallback"])
def test_agent_few_shots_match_rendered_contract(
    lm: Any, grouping: bool, transport: str, rule_suffix: str
) -> None:
    source = am.Source({"message": "Text"})
    query = (
        source.sem_groupby(instruction="Group by the explicitly named language.")
        if grouping
        else source.sem_agg(output_cols=["answer"], instruction="List languages.")
    )
    prefix = "group" if grouping else "agg"
    lm.handler = lambda payload: {
        "rule": f"{prefix}-{rule_suffix}",
        "state_schema": None,
        "maintenance_instruction": None,
        "rationale": "Offline no-state fixture.",
    }
    plan = am.plan_maintenance(
        query,
        adapter=LotusAdapter(
            config=LotusExecutionConfig(structured_output_transport=transport)
        ),
    )
    assert plan.to_dict()["rule"] == f"{prefix}-{rule_suffix}"
    messages = lm.calls[0][0][0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
    )
    payload = request_payload(lm.calls[0])
    assert payload["Original task"] == query.expr.params["instruction"]
    assert payload["Development inputs"] == []
    assert payload["Limits"] == {"max_generated_tokens_per_response": 8192}
    assert payload["Input column descriptions"] == {"message": "Text"}
    assert "Input schema" not in payload and "Output schema" not in payload
    final_prompt = messages[-1]["content"]
    for section in (
        "HOW TO CHOOSE",
        "DESIGNING OPTIONAL STATE",
        "OUTPUT FIELDS",
        "OUTPUT CONTRACT",
        "Response schema:\n",
        f"1. Prefer {prefix}-composable",
        f"2. Choose {prefix}-state only for a concrete maintenance need or benefit",
        f"3. Avoid {prefix}-fallback",
        "Keep S compact and purposeful.",
        "Simply reorganizing historical inputs into fields does not itself",
        "Consider the cost of reading and returning state on updates.",
        "consolidates redundant entries, and makes stored information unnecessary.",
        "Preserve distinctions required by the task.",
    ):
        assert final_prompt.count(section) == 1
    assert "Data arrives in batches" not in final_prompt
    assert "zero-based" not in final_prompt  # operator output protocol, not planning
    assert "Otherwise, try" not in final_prompt
    assert "After validation, the schema stays fixed across updates" in final_prompt
    assert "Declare each field's type and each array's item structure" in final_prompt
    assert "RULE BOUNDARIES" not in final_prompt
    assert "Think in terms of entities and relationships" not in final_prompt
    assert final_prompt.count("knowledge graphs represented") == 1
    assert "Retain information needed for current\nor later updates" in final_prompt
    assert "do not invent their contents or claim measured savings" in final_prompt
    opposite = "agg" if grouping else "group"
    assert f"{opposite}-state" not in final_prompt
    if grouping:
        assert "semantic grouping or clustering" in final_prompt
        assert payload["Constraints"] == {
            "partition_by": [],
            "membership": "single",
            "C": None,
        }
        assert "Aggregation scope" not in payload
        assert payload["Output columns"] == ["message", RECORD_ID, GROUP_ID]
        assert payload["Output column descriptions"] == {"message": "Text"}
        rule = payload["Available rules"]["group-composable"]
        assert "All historical records in that partition are candidates" in rule
        assert "there is no representative-record selection" in rule
        assert "Only unmatched new records" in rule
        assert "not category definitions or group summaries" in final_prompt
        assert (
            "S must be a JSON object containing the grouping-object array"
            in final_prompt
        )
        assert "then the final group count is checked against C" in final_prompt
        state_rule = payload["Available rules"]["group-state"]
        assert "Process the complete S of the affected partition" in state_rule
        assert "there is no selective processing" in state_rule
        assert "when later evidence changes their interpretation" in final_prompt
        assert "why processing it is worthwhile" in final_prompt
        assert "Retain separate representations when they may need different" in final_prompt
        assert "Evidence references are not automatically membership mappings" in final_prompt
    else:
        assert "Constraints" not in payload
        assert payload["Aggregation scope"] == "One result over all input records."
        assert payload["Output columns"] == ["answer"]
        assert "Output column descriptions" not in payload
        assert "GROUPING CONSTRAINTS" not in final_prompt
        assert "do not count them as independent observations" in final_prompt
        assert "A concise result or omitted input details do not by themselves require S" in final_prompt
        assert "the additional state-processing cost" in final_prompt
        assert "Preserve repeated contributions when the task counts" in final_prompt
        assert (
            "New records are not aggregated separately first"
            in payload["Available rules"]["agg-composable"]
        )
    schema = json.loads(final_prompt.split("Response schema:\n", 1)[1])
    assert set(schema["properties"]["rule"]["enum"]) == {
        f"{prefix}-composable",
        f"{prefix}-state",
        f"{prefix}-fallback",
    }
    designs = payload["Design examples"]
    assert len(designs) == (2 if grouping else 3)
    answers = []
    for design in designs:
        assert "Available rules" not in design["Task"]
        if grouping:
            assert design["Task"]["Constraints"]["membership"] in {"single", "multiple"}
        answer = design["Plan"]
        validate_json(answer, schema)
        am.MaintenancePlan.bind(query, answer)
        answers.append(answer)
    assert any(answer["rule"] == f"{prefix}-state" for answer in answers)
    assert any(
        answer["rule"] == f"{prefix}-composable"
        and answer["state_schema"] is None
        and answer["maintenance_instruction"] is None
        for answer in answers
    )
    assert len(lm.calls) == 1


def test_two_operator_plans_preserve_raw_instructions_and_binding(lm: Any) -> None:
    source = am.Source({"message": "Original message", "tenant": "Partition"})
    ig = '按项目归组。\n保留 {message}、"原词" 和 {{literal}}。'
    ia = '概括各组。\n保留 {message}、"原词" 和 {{literal}}。'
    groups = source.sem_groupby(
        input_cols=["message"],
        instruction=ig,
        membership="multiple",
        partition_by=["tenant"],
        C=2,
    )
    aggregate = groups.sem_agg(
        input_cols=["message"],
        output_cols=["summary"],
        instruction=ia,
    )
    adapter = LotusAdapter()

    def choose(payload: dict) -> dict:
        prefix = "group" if payload["Operator"] == "sem_groupby" else "agg"
        return {
            "rule": f"{prefix}-composable",
            "state_schema": None,
            "maintenance_instruction": None,
            "rationale": "Offline fixture.",
        }

    lm.handler = choose
    group_plan = am.plan_maintenance(groups, adapter=adapter)
    agg_plan = am.plan_maintenance(aggregate, adapter=adapter)
    assert len(lm.calls) == 2
    for call, original in zip(lm.calls, (ig, ia), strict=True):
        assert request_payload(call)["Original task"] == original
        assert request_payload(call)["Development inputs"] == []
        content = call[0][0][-1]["content"]
        assert content.count(json.dumps(original, ensure_ascii=False)) == 1
    constraints = request_payload(lm.calls[0])["Constraints"]
    assert constraints == {"partition_by": ["tenant"], "membership": "multiple", "C": 2}
    assert request_payload(lm.calls[1])["Constraints"] == {
        "group_keys": ["tenant", GROUP_ID],
    }
    assert request_payload(lm.calls[1])["Aggregation scope"] == (
        "One result per group; groups are maintained independently."
    )
    assert request_payload(lm.calls[1])["Output columns"] == ["summary"]
    assert groups.expr.params["instruction"] == ig
    assert aggregate.expr.params["instruction"] == ia
    assert group_plan.key != agg_plan.key
    make_flow(source, {"out": aggregate}, (group_plan, agg_plan))
    assert len(lm.calls) == 2  # binding and compilation do not call the Agent


def test_planning_preserves_declared_output_meaning(lm: Any) -> None:
    source = am.Source({"message": "Original message", "tenant": "Partition"})
    query = source.group_by("tenant").sem_agg(
        input_cols=["message"],
        output_cols={"summary": "Concise summary"},
        instruction="Summarize messages.",
    )
    lm.handler = lambda payload: {
        "rule": "agg-fallback",
        "state_schema": None,
        "maintenance_instruction": None,
        "rationale": "Offline fixture.",
    }
    am.plan_maintenance(query, adapter=LotusAdapter())
    payload = request_payload(lm.calls[0])
    assert payload["Input columns"] == ["message"]
    assert payload["Input column descriptions"] == {"message": "Original message"}
    assert "summary" in payload["Output columns"]
    assert payload["Output column descriptions"]["summary"] == "Concise summary"
    assert payload["Constraints"] == {"group_keys": ["tenant"]}


@pytest.mark.parametrize(
    "descriptor,grouped", [(False, False), (False, True), (True, True)]
)
def test_agg_planning_output_fields_match_execution(
    lm: Any, descriptor: bool, grouped: bool
) -> None:
    source = am.Source({"message": "Text", "tenant": "Partition"})
    parent = source.group_by("tenant") if grouped else source
    instruction = "Return the requested {message} identifier."
    outputs = {"group_id": "A business result, not a grouping key"}
    query = (
        source.group_by("tenant").agg(
            am.sem_agg(
                input_cols=["tenant"], output_cols=["other"], instruction="Other task"
            ),
            am.sem_agg(
                input_cols=["message"], output_cols=outputs, instruction=instruction
            ),
        )
        if descriptor
        else parent.sem_agg(
            input_cols=["message"], output_cols=outputs, instruction=instruction
        )
    )
    index = 1 if descriptor else 0
    config = state_plan(query, index=index).to_dict()
    lm.handler = lambda payload: config
    plan = am.plan_maintenance(query, adapter=LotusAdapter(), aggregate_index=index)
    payload = request_payload(lm.calls[0])
    assert payload["Original task"] == instruction
    assert payload["Output columns"] == ["group_id"]
    assert payload["Output column descriptions"] == outputs
    assert payload["Result responsibility"] == (
        "The model generates only Output columns. The program preserves and adds "
        "the aggregation group keys, if any."
    )
    assert payload.get("Constraints") == (
        {"group_keys": ["tenant"]} if grouped else None
    )
    plans = (state_plan(query), plan) if descriptor else (plan,)
    flow = make_flow(source, {"v": query}, plans)
    assert len(lm.calls) == 1

    def response(payload: dict) -> dict:
        field = "group_id" if "message" in payload["New inputs"][0] else "other"
        return {"result": {field: "business"}, "state": {"count": 1}}

    lm.handler = response
    flow.apply(pd.DataFrame({"message": ["one"], "tenant": ["a"]}))
    call = next(
        c for c in lm.calls[1:] if "message" in request_payload(c)["New inputs"][0]
    )
    schema = json.loads(call[0][0][-1]["content"].split("Response schema:\n", 1)[1])
    assert (
        list(schema["properties"]["result"]["properties"]) == payload["Output columns"]
    )
    expected = (
        (["tenant"] if grouped else [])
        + (["other"] if descriptor else [])
        + ["group_id"]
    )
    assert flow.view("v").columns.tolist() == expected


@pytest.mark.parametrize("transport", ["chat-json-object", "responses-json-schema"])
def test_agg_initial_empty_answer_state_restore_and_prompt(
    lm: Any, transport: str
) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(
        output_cols=["answer"], instruction="Return repeated names. {Do not rewrite}"
    )
    plan = state_plan(query)
    config = LotusExecutionConfig(structured_output_transport=transport)
    flow = make_flow(source, {"out": query}, (plan,), config=config)
    lm.handler = counter
    flow.apply(pd.DataFrame({"message": ["Alice"]}))
    assert flow.view("out").answer.tolist() == [""]
    snapshot = flow.snapshot_state()
    restored = make_flow(source, {"out": query}, (plan,), config=config)
    restored.restore_state(snapshot)
    assert len(lm.calls) == 1
    restored.apply(pd.DataFrame({"message": ["Alice", "Alice"]}))
    assert restored.view("out").answer.tolist() == ["repeated"]
    hidden = [
        f
        for f in restored.snapshot_state()["node_state"].values()
        if "_state" in f.columns
    ][0]
    assert hidden.iloc[0]["_state"] == {"count": 3}
    assert [f for f in snapshot["node_state"].values() if "_state" in f.columns][
        0
    ].iloc[0]["_state"] == {"count": 1}
    request = lm.calls[-1][0][0]
    assert len(request) == 2  # no state-design fewshots in execution
    assert "Return repeated names. {Do not rewrite}" in request[-1]["content"]
    assert "Previous result" in request[-1]["content"]
    assert "Do not use nested arrays" not in request[-1]["content"]
    content = request[-1]["content"]
    assert (
        content.count("- result: the user-visible output requested by Instruction.")
        == 1
    )
    assert (
        content.count("- state: the complete auxiliary information for future updates,")
        == 1
    )
    assert "initialized or updated according to State maintenance." in content
    assert "not new observations" in content
    payload = request_payload(lm.calls[-1])
    assert payload["Previous result"] == {"answer": ""}
    assert payload["Previous state"] == {"count": 1}
    assert payload["State maintenance"] == plan.to_dict()["maintenance_instruction"]
    schema = json.loads(content.split("Response schema:\n", 1)[1])
    assert set(schema["properties"]) == {"result", "state"}
    assert set(schema["properties"]["result"]["properties"]) == {"answer"}
    assert set(schema["properties"]["state"]["properties"]) == {"count"}
    calls = len(lm.calls)
    restored.apply(pd.DataFrame(columns=pd.Index(["message"])))
    assert len(lm.calls) == calls


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("transform", ["direct", "filter", "select", "alias"])
def test_group_reassignment_numeric_retractions_and_restore(
    lm: Any, membership: Literal["single", "multiple"], transform: str
) -> None:
    source = am.Source({"message": "Text", "amount": "Cost"})
    groups = source.sem_groupby(
        input_cols=["message"], instruction="Group issues", membership=membership
    )
    transformed = {
        "direct": groups,
        "filter": groups.filter(groups.col("amount") > 0),
        "select": groups.select(["message", "amount", RECORD_ID, GROUP_ID]),
        "alias": groups.alias("g"),
    }[transform]
    metrics = transformed.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )
    plan = state_plan(groups, group=True)
    flow = make_flow(source, {"g": transformed, "m": metrics}, (plan,))
    lm.handler = grouper
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"], "amount": [2, 4]}))
    assert flow.view("m").n.tolist() == [2]
    restored = make_flow(source, {"g": transformed, "m": metrics}, (plan,))
    restored.restore_state(flow.snapshot_state())
    restored.apply(pd.DataFrame({"message": ["both"], "amount": [6]}))
    assert sorted(restored.view("m").n) == (
        [1, 2] if membership == "single" else [2, 2]
    )
    assert sorted(restored.view("m").total) == (
        [2, 10] if membership == "single" else [8, 10]
    )
    assert not restored.view("g").duplicated([RECORD_ID, GROUP_ID]).any()


def test_state_only_change_does_not_change_memberships(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group issues")
    flow = make_flow(source, {"g": groups}, (state_plan(groups, group=True),))
    lm.handler = grouper
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"]}))
    before = flow.view("g")
    # Exercise the actual expansion with only state metadata changed.
    hidden = [
        f for f in flow.snapshot_state()["node_state"].values() if "_state" in f.columns
    ][0].iloc[0]
    result = expand_assignments(
        {**hidden["_state"], "revision": 99},
        [{"object": 0, "group": 8}, {"object": 1, "group": 8}],
        binding={"objects": "items", "record_ids": "record_ids"},
        active_ids=before[RECORD_ID].tolist(),
        previous=hidden["_result"],
        membership="single",
        C=None,
        generation="new",
    )
    assert result == hidden["_result"]


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ({"G": ["a", "b"]}, {7: ["a", "b", "c"]}, {"G": {"a", "b", "c"}}),
        # New records must not outweigh retained old members when splitting.
        (
            {"G": ["a", "b", "c"]},
            {7: ["a", "b"], 8: ["c", "d", "e", "f"]},
            {"G": {"a", "b"}, "next:8": {"c", "d", "e", "f"}},
        ),
        (
            {"G": ["a", "b"], "H": ["c"]},
            {7: ["a", "b", "c"]},
            {"G": {"a", "b", "c"}},
        ),
        # Multiple membership: count the shared record once per pair of groups.
        (
            {"G": ["a", "b"], "H": ["b", "c", "d"]},
            {7: ["a", "b", "c", "d"]},
            {"H": {"a", "b", "c", "d"}},
        ),
        # Reserve unchanged groups before resolving overlaps with other groups.
        (
            {"G": ["a", "b"], "H": ["b"]},
            {7: ["a", "b", "c"], 8: ["b"]},
            {"G": {"a", "b", "c"}, "H": {"b"}},
        ),
        # One old ID cannot be reused by both sides of a split.
        (
            {"G": ["a", "b", "c"], "H": ["d", "e"]},
            {7: ["a", "b", "d"], 8: ["c", "e"]},
            {"G": {"a", "b", "d"}, "H": {"c", "e"}},
        ),
        ({"G": ["a"]}, {7: ["b"]}, {"next:7": {"b"}}),
        (
            {"G": ["a", "b"], "H": ["a", "b"]},
            {7: ["a", "b"], 8: ["a", "b", "c"]},
            {"G": {"a", "b"}, "H": {"a", "b", "c"}},
        ),
    ],
)
def test_group_identity_continuation(
    old: dict[str, list[str]], new: dict[int, list[str]], expected: dict[str, set[str]]
) -> None:
    previous = [
        {"record_id": record, "group_id": group}
        for group, records in old.items()
        for record in records
    ]
    for reverse in (False, True):
        entries = list(new.items())[:: -1 if reverse else 1]
        result = expand_assignments(
            {"items": [{"members": records} for _, records in entries]},
            [
                {"object": index, "group": group}
                for index, (group, _) in enumerate(entries)
            ],
            binding={"objects": "items", "record_ids": "members"},
            active_ids=sorted({r for records in new.values() for r in records}),
            previous=previous[::-1] if reverse else previous,
            membership="multiple",
            C=None,
            generation="next",
        )
        actual: dict[str, set[str]] = {}
        for row in result:
            actual.setdefault(row["group_id"], set()).add(row["record_id"])
        assert actual == expected


def test_group_identity_ties_ignore_temporary_group_numbers() -> None:
    for first, second in ((0, 1), (99, 4)):
        result = expand_assignments(
            {"items": [{"members": ["b", "y"]}, {"members": ["a", "z"]}]},
            [{"object": 0, "group": first}, {"object": 1, "group": second}],
            binding={"objects": "items", "record_ids": "members"},
            active_ids=["a", "b", "y", "z"],
            previous=[{"record_id": r, "group_id": "G"} for r in ("a", "b")],
            membership="single",
            C=None,
            generation="next",
        )
        assert {row["record_id"] for row in result if row["group_id"] == "G"} == {
            "a",
            "z",
        }


@pytest.mark.parametrize("membership", ["single", "multiple"])
@pytest.mark.parametrize("descriptor", [False, True])
@pytest.mark.parametrize("transform", ["direct", "filter", "select", "alias"])
def test_group_growth_reuses_agg_state_after_restore(
    lm: Any,
    monkeypatch: pytest.MonkeyPatch,
    membership: Literal["single", "multiple"],
    descriptor: bool,
    transform: str,
) -> None:
    source = am.Source({"tenant": "Partition", "message": "Text", "amount": "Cost"})
    groups = source.sem_groupby(
        input_cols=["message"],
        instruction="Group issues",
        membership=membership,
        partition_by="tenant",
    )
    transformed = {
        "direct": groups,
        "filter": groups.filter(groups.col("amount") > 0),
        "select": groups.select(["tenant", "message", "amount", RECORD_ID, GROUP_ID]),
        "alias": groups.alias("g"),
    }[transform]
    summary = (
        transformed.agg(am.sem_agg(output_cols=["answer"], instruction="Count"))
        if descriptor
        else transformed.sem_agg(output_cols=["answer"], instruction="Count")
    )
    plans = (state_plan(groups, group=True), state_plan(summary))

    def respond(payload: dict) -> dict:
        if "Constraints" not in payload:
            return counter(payload)
        response = grouper(payload)
        response["assignments"] = [
            {"object": i, "group": group + response["state"]["revision"] * 10}
            for i, item in enumerate(response["state"]["items"])
            for group in (
                [0, 1]
                if item["topic"] == "both"
                else [0]
                if item["topic"] == "shipping"
                else [1]
            )
        ]
        return response

    lm.handler = respond
    flow = make_flow(source, {"g": transformed, "s": summary}, plans)
    flow.apply(
        pd.DataFrame(
            {
                "tenant": ["a", "a", "b"],
                "message": ["shipping", "quality", "shipping"],
                "amount": [1, 1, 1],
            }
        )
    )
    initial = flow.snapshot_state()
    original_pairs = set(
        flow.view("g")[[RECORD_ID, GROUP_ID]].itertuples(index=False, name=None)
    )
    original_ids = set(flow.view("g")[GROUP_ID])
    restored = make_flow(source, {"g": transformed, "s": summary}, plans)
    restored.restore_state(initial)
    original_execute = restored._executor.adapter.execute
    updates: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def capture(query: QueryExpr, inputs: dict) -> Any:
        if (
            query.op == "semantic_stateful_update"
            and query.inputs[0].op != "sem_groupby"
        ):
            parent = query.params["parent"]
            updates.append(
                (
                    inputs[f"{parent}__inserted"].copy(),
                    inputs[f"{parent}__retracted"].copy(),
                )
            )
        return original_execute(query, inputs)

    monkeypatch.setattr(restored._executor.adapter, "execute", capture)
    added = "both" if membership == "multiple" else "shipping"
    for step in (1, 2):
        lm.calls.clear()
        restored.apply(
            pd.DataFrame({"tenant": ["a"], "message": [added], "amount": [1]})
        )
        assert len(updates) == step
        inserted, retracted = updates[-1]
        expected_groups = 2 if membership == "multiple" else 1
        assert retracted.empty and len(inserted) == expected_groups
        assert inserted.message.tolist() == [added] * expected_groups
        assert inserted[RECORD_ID].nunique() == 1
        assert not inserted.duplicated([RECORD_ID, GROUP_ID]).any()
        assert set(restored.view("g")[GROUP_ID]) == original_ids
        assert original_pairs <= set(
            restored.view("g")[[RECORD_ID, GROUP_ID]].itertuples(index=False, name=None)
        )
        requests = [request_payload(call) for call in lm.calls]
        agg_requests = [p for p in requests if "Previous result" in p]
        assert len(agg_requests) == expected_groups
        for payload in agg_requests:
            assert len(payload["New inputs"]) == 1
            assert payload["Previous state"] == {"count": step}
            assert payload["Previous result"] == {
                "answer": "" if step == 1 else "repeated"
            }
        counts = sorted(
            s["count"] for s in hidden_state(restored)._state if "count" in s
        )
        assert counts == (
            [1, step + 1, step + 1] if membership == "multiple" else [1, 1, step + 1]
        )
    # The saved snapshot remains independent of the continued nested states.
    assert all(
        s["count"] == 1
        for frame in initial["node_state"].values()
        if "_state" in frame
        for s in frame._state
        if "count" in s
    )


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_split_then_merge_preserves_identity_and_rebuilds_only_retractions(
    lm: Any, monkeypatch: pytest.MonkeyPatch, membership: Literal["single", "multiple"]
) -> None:
    source = am.Source({"message": "Text", "amount": "Value"})
    groups = source.sem_groupby(
        input_cols=["message"], instruction="Group", membership=membership
    )
    summary = groups.sem_agg(output_cols=["answer"], instruction="Count")
    metrics = groups.agg(
        am.count(output_col="n"), am.sum(column="amount", output_col="total")
    )
    plans = (state_plan(groups, group=True), state_plan(summary))
    views = {"g": groups, "s": summary, "m": metrics}

    def respond(payload: dict) -> dict:
        if "Constraints" not in payload:
            return counter(payload)
        response = grouper(payload)
        revision = response["state"]["revision"]
        definitions = (
            [{"a", "b", "q", "c"}, {"d"}]
            if revision == 0
            else [{"a", "b", "q"}, {"d"}, {"c", "e"}]
            if revision == 1
            else [{"a", "b", "q", "c", "e", "f"}, {"d"}]
        )
        if membership == "multiple":
            definitions[1].add("b")
        response["assignments"] = [
            {"object": i, "group": g + revision * 10}
            for i, item in enumerate(response["state"]["items"])
            for g, topics in enumerate(definitions)
            if item["topic"] in topics
        ]
        return response

    lm.handler = respond
    flow = make_flow(source, views, plans)
    flow.apply(pd.DataFrame({"message": ["a", "b", "q", "c", "d"], "amount": [1] * 5}))
    g = flow.view("g").query("message == 'a'")[GROUP_ID].item()
    h = flow.view("g").query("message == 'd'")[GROUP_ID].item()
    lm.calls.clear()
    flow.apply(pd.DataFrame({"message": ["e"], "amount": [1]}))
    assert flow.view("g").query("message == 'a'")[GROUP_ID].item() == g
    assert flow.view("g").query("message == 'd'")[GROUP_ID].item() == h
    j = flow.view("g").query("message == 'c'")[GROUP_ID].item()
    assert j not in {g, h}
    requests = [request_payload(call) for call in lm.calls]
    agg_requests = [p for p in requests if "Previous result" in p]
    assert len(agg_requests) == 2  # H is untouched.
    assert all(p["Previous state"] is None for p in agg_requests)
    assert {frozenset(r["message"] for r in p["New inputs"]) for p in agg_requests} == {
        frozenset({"a", "b", "q"}),
        frozenset({"c", "e"}),
    }
    restored = make_flow(source, views, plans)
    restored.restore_state(flow.snapshot_state())
    original_execute = restored._executor.adapter.execute
    updates: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def capture(query: QueryExpr, inputs: dict) -> Any:
        if (
            query.op == "semantic_stateful_update"
            and query.inputs[0].op != "sem_groupby"
        ):
            parent = query.params["parent"]
            updates.append(
                (
                    inputs[f"{parent}__inserted"].copy(),
                    inputs[f"{parent}__retracted"].copy(),
                )
            )
        return original_execute(query, inputs)

    monkeypatch.setattr(restored._executor.adapter, "execute", capture)
    lm.calls.clear()
    restored.apply(pd.DataFrame({"message": ["f"], "amount": [1]}))
    assert set(restored.view("g")[GROUP_ID]) == {g, h}
    assert len(updates) == 1
    inserted, retracted = updates[0]
    assert set(inserted[GROUP_ID]) == {g} and set(inserted.message) == {"c", "e", "f"}
    assert set(retracted[GROUP_ID]) == {j} and set(retracted.message) == {"c", "e"}
    requests = [request_payload(call) for call in lm.calls]
    agg_requests = [p for p in requests if "Previous result" in p]
    assert len(agg_requests) == 1  # G updated; J removed without a model call.
    assert agg_requests[0]["Previous state"] == {"count": 3}
    assert agg_requests[0]["Previous result"] == {"answer": "repeated"}
    assert {r["message"] for r in agg_requests[0]["New inputs"]} == {"c", "e", "f"}
    counts = sorted(s["count"] for s in hidden_state(restored)._state if "count" in s)
    assert counts == ([2, 6] if membership == "multiple" else [1, 6])
    metrics_result = restored.view("m")
    assert sorted(metrics_result.n) == counts
    assert sorted(metrics_result.total) == counts


def test_overlapping_merge_does_not_count_shared_members_twice(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", membership="multiple")
    summary = groups.sem_agg(output_cols=["answer"], instruction="Count")
    flow = make_flow(
        source,
        {"g": groups, "s": summary},
        (state_plan(groups, group=True), state_plan(summary)),
    )

    def respond(payload: dict) -> dict:
        if "Constraints" not in payload:
            return counter(payload)
        response = grouper(payload)
        definitions = (
            [{"a", "b"}, {"b", "c", "d"}]
            if response["state"]["revision"] == 0
            else [{"a", "b", "c", "d", "e"}]
        )
        response["assignments"] = [
            {"object": i, "group": g}
            for i, item in enumerate(response["state"]["items"])
            for g, topics in enumerate(definitions)
            if item["topic"] in topics
        ]
        return response

    lm.handler = respond
    flow.apply(pd.DataFrame({"message": ["a", "b", "c", "d"]}))
    retained = flow.view("g").query("message == 'c'")[GROUP_ID].item()
    lm.calls.clear()
    flow.apply(pd.DataFrame({"message": ["e"]}))
    assert len(flow.view("g")) == 5 and set(flow.view("g")[GROUP_ID]) == {retained}
    requests = [request_payload(call) for call in lm.calls]
    aggregates = [p for p in requests if "Previous result" in p]
    assert len(aggregates) == 1
    assert aggregates[0]["Previous state"] == {"count": 3}
    assert {r["message"] for r in aggregates[0]["New inputs"]} == {"a", "e"}
    assert [s["count"] for s in hidden_state(flow)._state if "count" in s] == [5]


@pytest.mark.parametrize(
    "bad", ["unknown", "missing", "duplicate", "float", "overflow"]
)
def test_invalid_group_response_rolls_back(lm: Any, bad: str) -> None:
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", C=1)
    flow = make_flow(source, {"g": groups}, (state_plan(groups, group=True),))
    lm.handler = grouper
    flow.apply(pd.DataFrame({"message": ["shipping"]}))
    snapshot = flow.snapshot_state()

    def invalid(payload: dict) -> dict:
        response = grouper(payload)
        if bad == "unknown":
            response["state"]["items"][0]["record_ids"] = ["not-a-record"]
        elif bad == "missing":
            response["assignments"] = response["assignments"][:1]
        elif bad == "duplicate":
            response["assignments"].append(response["assignments"][0])
        elif bad == "float":
            response["assignments"][0]["object"] = 0.0
        else:
            response["assignments"][1]["group"] = 1
        return response

    lm.handler = invalid
    with pytest.raises((ValueError, ValidationError)):
        flow.apply(pd.DataFrame({"message": ["shipping"]}))
    pd.testing.assert_frame_equal(flow.view("g"), snapshot["state"]["g"])
    assert flow.snapshot_state()["next_occurrence"] == snapshot["next_occurrence"]
    lm.handler = grouper
    flow.apply(pd.DataFrame({"message": ["shipping"]}))
    assert len(flow.view("g")) == 2


@pytest.mark.parametrize(
    "raw",
    [
        '{"result":',
        '{"result":{"answer":"x"},"state":{"count":true}}',
        '{"result":{"answer":"x"},"state":{"count":1},"extra":0}',
        '{"result":{"answer":"x","evidence":null},"state":{"count":1}}',
    ],
)
def test_strict_nested_response_no_retry(lm: Any, raw: str) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    flow = make_flow(source, {"v": query}, (state_plan(query),))
    lm.raw = raw
    with pytest.raises((ValueError, ValidationError)):
        flow.apply(pd.DataFrame({"message": ["one"]}))
    assert len(lm.calls) == 1
    assert flow.view("v").empty


def test_saved_state_schema_stays_fixed_across_updates_and_restore(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Count messages.")
    saved = state_plan(query).to_dict()
    plan = am.MaintenancePlan.bind(query, saved)
    original = plan.to_dict()
    saved["state_schema"]["value"]["properties"]["count"]["type"] = "string"
    exported = plan.to_dict()
    exported["state_schema"]["value"]["properties"]["count"]["type"] = "array"
    assert plan.to_dict() == original

    flow = make_flow(source, {"v": query}, (plan,))
    assert not lm.calls
    lm.handler = counter
    flow.apply(pd.DataFrame({"message": ["first"]}))
    snapshot = flow.snapshot_state()
    before_result, before_state = flow.view("v"), hidden_state(flow)
    lm.handler = lambda payload: {
        "result": {"answer": "changed"},
        "state": {"count": "2"},
    }
    with pytest.raises(ValidationError):
        flow.apply(pd.DataFrame({"message": ["second"]}))
    assert len(lm.calls) == 2
    pd.testing.assert_frame_equal(flow.view("v"), before_result)
    pd.testing.assert_frame_equal(hidden_state(flow), before_state)

    rebound = am.MaintenancePlan.bind(query, original)
    restored = make_flow(source, {"v": query}, (rebound,))
    restored.restore_state(snapshot)
    assert len(lm.calls) == 2
    lm.handler = counter
    restored.apply(pd.DataFrame({"message": ["second"]}))
    assert hidden_state(restored).iloc[0]["_state"] == {"count": 2}
    assert rebound.to_dict() == original
    assert len(lm.calls) == 3
    for call in lm.calls:
        payload = request_payload(call)
        assert "Available rules" not in payload
        assert payload["State maintenance"] == original["maintenance_instruction"]
        schema = json.loads(call[0][0][-1]["content"].split("Response schema:\n", 1)[1])
        assert schema["properties"]["state"] == original["state_schema"]["value"]


def test_schema_and_binding_fail_before_model(lm: Any) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        validate_state_schema({"type": "string", "$ref": "https://example.com/schema"})
    source = am.Source({"message": "Text"})
    groups = source.sem_groupby(instruction="Group", C=2)
    flow = make_flow(source, {"g": groups}, (state_plan(groups, group=True),))
    with pytest.raises(ValueError, match="fewer records"):
        flow.apply(pd.DataFrame({"message": ["one"]}))
    assert not lm.calls


def hidden_state(flow: am.SemanticDataflow) -> pd.DataFrame:
    return pd.concat(
        [
            f
            for f in flow.snapshot_state()["node_state"].values()
            if "_state" in f.columns
        ],
        ignore_index=True,
    )


def request_payload(call: Any) -> dict[str, Any]:
    return json.JSONDecoder().raw_decode(
        call[0][0][-1]["content"].split("Context:\n", 1)[1]
    )[0]


def test_state_only_saved_without_downstream_execution(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Keep answer stable")
    downstream = query.sem_agg(output_cols=["answer"], instruction="Downstream")
    flow = make_flow(
        source,
        {"v": query, "d": downstream},
        (state_plan(query), state_plan(downstream)),
    )

    def answer(payload: dict) -> dict:
        response = counter(payload)
        response["result"]["answer"] = "stable"
        return response

    lm.handler = answer
    flow.apply(pd.DataFrame({"message": ["first"]}))
    snapshot = flow.snapshot_state()
    assert len(lm.calls) == 2
    flow.apply(pd.DataFrame({"message": ["second"]}))
    assert len(lm.calls) == 3  # parent state changes, public result does not
    assert sorted(row["count"] for row in hidden_state(flow)["_state"]) == [1, 2]
    old = [f for f in snapshot["node_state"].values() if "_state" in f.columns]
    assert all(f.iloc[0]["_state"] == {"count": 1} for f in old)


def test_descriptor_and_group_state_isolation(lm: Any) -> None:
    source = am.Source({"tenant": "Partition", "message": "Text", "value": "Number"})
    query = source.group_by("tenant").agg(
        am.sem_agg(
            input_cols=["message"], output_cols=["text_result"], instruction="Text task"
        ),
        am.sem_agg(
            input_cols=["value"],
            output_cols=["number_result"],
            instruction="Number task",
        ),
    )
    flow = make_flow(
        source, {"v": query}, (state_plan(query), state_plan(query, index=1))
    )

    def response(payload: dict) -> dict:
        col = (
            "text_result" if "message" in payload["New inputs"][0] else "number_result"
        )
        if payload["Previous result"] is not None:
            assert set(payload["Previous result"]) == {col}
        result = counter(payload)
        result["result"] = {col: result["state"]["count"]}
        return result

    lm.handler = response
    flow.apply(
        pd.DataFrame({"tenant": ["a", "b"], "message": ["x", "x"], "value": [1, 2]})
    )
    assert len(lm.calls) == 4
    flow.apply(pd.DataFrame({"tenant": ["a"], "message": ["x"], "value": [3]}))
    assert len(lm.calls) == 6
    assert flow.view("v").set_index("tenant").text_result.to_dict() == {"a": 2, "b": 1}
    assert len(hidden_state(flow)) == 4


@pytest.mark.parametrize("descriptor", [False, True])
@pytest.mark.parametrize("growth", [False, True])
def test_group_state_chained_agg_and_downstream_rollback(
    lm: Any, monkeypatch: pytest.MonkeyPatch, descriptor: bool, growth: bool
) -> None:
    source = am.Source({"message": "Text", "amount": "Cost"})
    groups = source.sem_groupby(input_cols=["message"], instruction="Group issues")
    summary = (
        groups.agg(am.sem_agg(output_cols=["answer"], instruction="Count"))
        if descriptor
        else groups.sem_agg(output_cols=["answer"], instruction="Count")
    )
    plans = (state_plan(groups, group=True), state_plan(summary))
    flow = make_flow(source, {"s": summary, "g": groups}, plans)
    lm.handler = lambda payload: (
        grouper(payload) if "Constraints" in payload else counter(payload)
    )
    flow.apply(
        pd.DataFrame(
            {
                "message": ["shipping", "shipping" if growth else "quality"],
                "amount": [2, 4],
            }
        )
    )
    snapshot = flow.snapshot_state()
    adapter = flow._executor.adapter
    original = adapter.execute

    def fail(query: QueryExpr, inputs: dict) -> Any:
        if query.op == "semantic_stateful_finalize" and not query.params["grouping"]:
            raise ValueError("deliberate downstream failure")
        return original(query, inputs)

    monkeypatch.setattr(adapter, "execute", fail)
    batch = pd.DataFrame(
        {"message": ["shipping" if growth else "quality"], "amount": [6]}
    )
    with pytest.raises(ValueError, match="deliberate downstream"):
        flow.apply(batch)
    assert flow.snapshot_state()["next_occurrence"] == snapshot["next_occurrence"]
    for key, frame in snapshot["node_state"].items():
        pd.testing.assert_frame_equal(flow.snapshot_state()["node_state"][key], frame)
    monkeypatch.setattr(adapter, "execute", original)
    flow.apply(batch)
    assert sorted(flow.view("s").answer) == (
        ["repeated"] if growth else ["", "repeated"]
    )
    agg_states = hidden_state(flow)
    counts = [s["count"] for s in agg_states["_state"] if "count" in s]
    assert sorted(counts) == ([3] if growth else [1, 2])
    requests = [
        request_payload(call)
        for call in lm.calls
        if "Previous result" in request_payload(call)
    ]
    if growth:
        assert requests[0]["Previous state"] is None
        assert all(req["Previous state"] == {"count": 2} for req in requests[1:])
        assert all(
            req["Previous result"] == {"answer": "repeated"} for req in requests[1:]
        )
        assert set(flow.view("g")[GROUP_ID]) == set(snapshot["state"]["g"][GROUP_ID])
    else:
        # The surviving side is rebuilt after a real split; it does not reuse S.
        assert all(req["Previous state"] is None for req in requests)


@pytest.mark.parametrize("kind", ["agg", "group"])
def test_retractions_rebuild_state_and_empty_scope_clears(lm: Any, kind: str) -> None:
    from agent_memory.adapters.lotus.stateful import execute_stateful

    source = am.Source({"tenant": "Partition", "message": "Text"})
    query = (
        source.sem_groupby(instruction="Group", partition_by="tenant")
        if kind == "group"
        else source.group_by("tenant").sem_agg(
            output_cols=["answer"], instruction="Count"
        )
    )
    flow = make_flow(source, {"v": query}, (state_plan(query, group=kind == "group"),))
    lm.handler = grouper if kind == "group" else counter
    flow.apply(
        pd.DataFrame(
            {
                "tenant": ["a", "a", "b"],
                "message": ["shipping", "shipping", "quality"],
            }
        )
    )
    node = next(
        n
        for n in flow._executor.policy.nodes.values()
        if n.execution_kind == "semantic_stateful"
    )
    plan = node.maintenance_query
    assert plan is not None
    parent = plan.params["parent"]
    old = flow.snapshot_state()["node_state"][node.node_id]
    data = flow.snapshot_state()["node_state"][parent]
    removed = data[data.tenant == "a"].iloc[:1]
    added = pd.DataFrame(
        {"tenant": ["a"], "message": ["shipping"]}, index=pd.Index(["fresh-record"])
    )
    current = pd.concat([data.drop(index=removed.index), added])
    bindings = {
        parent: current,
        plan.params["previous"]: old,
        f"{parent}__inserted": added,
        f"{parent}__retracted": removed,
        "__group_generation": "next",
    }
    rebuilt = execute_stateful(
        plan, bindings, flow._executor.adapter.execute, flow._executor.adapter._context
    )
    assert request_payload(lm.calls[-1])["Previous state"] is None
    assert len(lm.calls) == 3  # b never reprocessed
    if kind == "group":
        before = old[old._scope.map(lambda k: k == ["a"])].iloc[0]["_result"]
        after = rebuilt[rebuilt._scope.map(lambda k: k == ["a"])].iloc[0]["_result"]
        assert {r["group_id"] for r in after} == {r["group_id"] for r in before}
        assert {r["record_id"] for r in after} == set(
            current[current.tenant == "a"].index
        )
    pd.testing.assert_frame_equal(
        old[old._scope.map(lambda k: k == ["b"])].reset_index(drop=True),
        rebuilt[rebuilt._scope.map(lambda k: k == ["b"])].reset_index(drop=True),
    )
    bindings.update(
        {
            parent: current[current.tenant == "b"],
            plan.params["previous"]: rebuilt,
            f"{parent}__inserted": added.iloc[:0],
            f"{parent}__retracted": current[current.tenant == "a"],
        }
    )
    cleared = execute_stateful(
        plan, bindings, flow._executor.adapter.execute, flow._executor.adapter._context
    )
    assert len(cleared) == 1 and cleared.iloc[0]._scope == ["b"]
    assert len(lm.calls) == 3


def test_member_bindings_ignore_evidence_and_expose_coarse_state_limit() -> None:
    binding = {"objects": "items", "record_ids": "members"}
    state = {"items": [{"members": ["a", "b"], "evidence": ["other-partition"]}]}

    def expand(value: dict, assignments: list[dict[str, int]]) -> list[dict[str, str]]:
        return expand_assignments(
            value,
            assignments,
            binding=binding,
            active_ids=["a", "b"],
            previous=[],
            membership="single",
            C=None,
            generation="batch",
        )

    result = expand(state, [{"object": 0, "group": 0}])
    assert {r["record_id"] for r in result} == {"a", "b"}
    assert len({r["group_id"] for r in result}) == 1
    # One coarse object cannot split its members by giving it two assignments.
    with pytest.raises(ValueError, match="cardinality"):
        expand(state, [{"object": 0, "group": 0}, {"object": 0, "group": 1}])
    split = {"items": [{"members": ["a"]}, {"members": ["b"]}]}
    assert (
        len(
            {
                r["group_id"]
                for r in expand(
                    split, [{"object": 0, "group": 0}, {"object": 1, "group": 1}]
                )
            }
        )
        == 2
    )


def test_changed_plan_cannot_restore_snapshot(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    plan = state_plan(query)
    flow = make_flow(source, {"v": query}, (plan,))
    lm.handler = counter
    flow.apply(pd.DataFrame({"message": ["first"]}))
    config = plan.to_dict()
    config["maintenance_instruction"] += " Changed."
    changed = am.MaintenancePlan.bind(query, config)
    with pytest.raises(ValueError, match="fingerprint"):
        make_flow(source, {"v": query}, (changed,)).restore_state(flow.snapshot_state())
    assert len(lm.calls) == 1


def test_truncated_output_and_capacity_fail_without_retry(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    flow = make_flow(source, {"v": query}, (state_plan(query),))
    lm.handler = counter
    lm.metadata = [{"finish_reason": "length"}]
    with pytest.raises(ValueError, match="incomplete"):
        flow.apply(pd.DataFrame({"message": ["first"]}))
    assert len(lm.calls) == 1 and flow.view("v").empty
    lm.max_ctx_len = 100
    with pytest.raises(ValueError, match="context limit"):
        flow.apply(pd.DataFrame({"message": ["first"]}))
    assert len(lm.calls) == 1


def test_agent_state_plan_and_rendered_prompts(lm: Any, tmp_path: Path) -> None:
    from agent_memory.planning_prompts import planning_examples
    from lotus.templates.task_instructions import map_formatter

    examples = {
        "agg": planning_examples(grouping=False)[1],
        "group": planning_examples(grouping=True)[1],
    }
    source = am.Source({"message": "Original message"})
    agg = source.sem_agg(
        output_cols=["answer"], instruction=examples["agg"][0]["Original task"]
    )
    group = source.sem_groupby(
        instruction=examples["group"][0]["Original task"], membership="multiple"
    )
    for name, target in (("agg", agg), ("group", group)):
        lm.handler = lambda payload, name=name: examples[name][1]
        plan = am.plan_maintenance(target, adapter=LotusAdapter())
        payload = request_payload(lm.calls[-1])
        assert payload["Development inputs"] == []
        assert payload["Input column descriptions"]["message"] == "Original message"
        (tmp_path / f"{name}-agent.json").write_text(
            json.dumps(lm.calls[-1][0][0], ensure_ascii=False, indent=2)
        )
        flow = make_flow(source, {"v": target}, (plan,))

        def response(payload: dict, name: str = name) -> dict:
            if name == "agg":
                count = 1 if payload["Previous state"] is None else 2
                return {
                    "result": {"answer": "Alice" if count == 2 else ""},
                    "state": {"counts": [{"name": "Alice", "record_count": count}]},
                }
            old = payload["Previous state"]
            items = [] if old is None else old["items"]
            items.extend(
                {
                    "record_ids": [r["record_id"]],
                    "project_clues": r["values"]["message"],
                }
                for r in payload["New inputs"]
            )
            return {
                "state": {"items": items},
                "assignments": [{"object": i, "group": 0} for i in range(len(items))],
            }

        lm.handler = response
        flow.apply(
            pd.DataFrame(
                {
                    "message": [
                        "Alice"
                        if name == "agg"
                        else "Project Atlas deadline is Friday."
                    ]
                }
            )
        )
        flow.apply(
            pd.DataFrame(
                {
                    "message": [
                        "Alice"
                        if name == "agg"
                        else "Project Atlas deadline is now Monday."
                    ]
                }
            )
        )
        assert lm.calls[-1][0][0][0] == map_formatter(lm, {"text": "input"}, "task")[0]
        (tmp_path / f"{name}-operator.json").write_text(
            json.dumps(lm.calls[-1][0][0], ensure_ascii=False, indent=2)
        )
    print(f"Rendered offline prompts: {tmp_path}")


@pytest.mark.parametrize("membership", ["single", "multiple"])
def test_group_merge_overlap_removal_and_numeric_state(
    lm: Any, membership: Literal["single", "multiple"]
) -> None:
    source = am.Source({"message": "Text", "amount": "Cost"})
    groups = source.sem_groupby(instruction="Group issues", membership=membership)
    numeric = groups.agg(
        am.count(output_col="n"),
        am.sum(column="amount", output_col="total"),
        am.avg(column="amount", output_col="mean"),
    )
    flow = make_flow(
        source, {"g": groups, "n": numeric}, (state_plan(groups, group=True),)
    )
    lm.handler = grouper
    flow.apply(pd.DataFrame({"message": ["shipping", "quality"], "amount": [2, 4]}))
    flow.apply(pd.DataFrame({"message": ["both"], "amount": [6]}))

    def merge_all(payload: dict) -> dict:
        response = grouper(payload)
        response["assignments"] = [
            {"object": i, "group": 99} for i in range(len(response["state"]["items"]))
        ]
        return response

    lm.handler = merge_all
    flow.apply(pd.DataFrame({"message": ["quality"], "amount": [8]}))
    assert flow.view("n").n.tolist() == [4]
    assert flow.view("n").total.tolist() == [20]
    assert flow.view("n")["mean"].tolist() == [5]
    assert (
        flow.view("g")[RECORD_ID].nunique() == 4
    )  # duplicate text is a distinct record


def test_partitions_never_share_members_and_failure_rolls_back(lm: Any) -> None:
    source = am.Source({"tenant": "Tenant", "message": "Text"})
    groups = source.sem_groupby(
        input_cols=["message"], instruction="Group", partition_by="tenant", C=1
    )
    plan = state_plan(groups, group=True)
    flow = make_flow(source, {"g": groups}, (plan,))
    lm.handler = grouper
    flow.apply(
        pd.DataFrame({"tenant": ["a", "b"], "message": ["shipping", "shipping"]})
    )
    before_b = flow.view("g").query("tenant == 'b'")
    flow.apply(pd.DataFrame({"tenant": ["a"], "message": ["shipping"]}))
    assert len(lm.calls) == 3
    pd.testing.assert_frame_equal(
        flow.view("g").query("tenant == 'b'").reset_index(drop=True),
        before_b.reset_index(drop=True),
    )
    snapshot = flow.snapshot_state()

    def cross_partition(payload: dict) -> dict:
        answer = grouper(payload)
        answer["state"]["items"][0]["record_ids"] = before_b[RECORD_ID].tolist()
        return answer

    lm.handler = cross_partition
    with pytest.raises(ValueError, match="unknown references"):
        flow.apply(pd.DataFrame({"tenant": ["a"], "message": ["shipping"]}))
    assert flow.snapshot_state()["next_occurrence"] == snapshot["next_occurrence"]


@pytest.mark.parametrize(
    "rule",
    ["rule-re-group", "rule-all-group", "rule-all-group-optimized", "rule-join-map"],
)
def test_state_plan_does_not_override_legacy_rule(rule: str) -> None:
    source = am.Source({"message": "Text"})
    legacy = source.sem_groupby(
        input_cols=["message"], instruction="Group", membership="exclusive"
    ).sem_agg(input_cols=["message"], output_cols=["message"], instruction="Summarize")
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    flow = am.SemanticDataflow(
        source=source,
        views={"new": query, "old": legacy},
        adapter=LotusAdapter(),
        rules=DifferentialRules(
            grouped_agg_rule=rule, maintenance_plans=(state_plan(query),)
        ),
    )
    policy = flow._executor.policy
    assert policy.grouped_agg_rule == rule
    assert policy.nodes[policy.view_outputs["old"]].execution_kind == "semantic_state"


def test_source_values_are_not_rounded_or_coerced(lm: Any) -> None:
    source = am.Source({"value": "Number", "time": "Time", "absent": "Optional"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    flow = make_flow(source, {"v": query}, (state_plan(query),))
    lm.handler = counter
    flow.apply(
        pd.DataFrame(
            {
                "value": [9007199254740993],
                "time": [pd.Timestamp("2025-01-01")],
                "absent": [pd.NA],
            }
        )
    )
    row = request_payload(lm.calls[-1])["New inputs"][0]
    assert row == {
        "value": 9007199254740993,
        "time": "2025-01-01T00:00:00",
        "absent": None,
    }
    flow.apply(
        pd.DataFrame(
            {
                "value": [0.1234567890123456],
                "time": [pd.Timestamp("2025-01-02")],
                "absent": [None],
            }
        )
    )
    assert request_payload(lm.calls[-1])["New inputs"][0]["value"] == 0.1234567890123456


def test_bound_plan_and_schema_validation_are_strict(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    good = state_plan(query)
    with pytest.raises(ValueError, match="duplicate"):
        DifferentialRules(maintenance_plans=(good, good))
    with pytest.raises(ValueError, match="nonnegative integer"):
        am.MaintenancePlan.bind(query, good.to_dict(), aggregate_index=True)
    with pytest.raises(ValueError, match="no-state"):
        am.MaintenancePlan.bind(query, {**good.to_dict(), "rule": "agg-composable"})
    lm.handler = lambda payload: {**good.to_dict(), "target": "invented-by-agent"}
    with pytest.raises(ValidationError):
        am.plan_maintenance(query, adapter=LotusAdapter())
    assert len(lm.calls) == 1


@pytest.mark.parametrize(
    "value,schema",
    [
        (None, {"type": "null"}),
        ([], {"type": "array", "items": {"type": "string"}}),
        ({}, object_schema({})),
        (0, {"type": "integer"}),
    ],
)
def test_empty_or_scalar_state_does_not_disable_maintenance(
    lm: Any, value: Any, schema: dict
) -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    config = state_plan(query).to_dict()
    config["state_schema"] = {"value": schema, "membership": None}
    plan = am.MaintenancePlan.bind(query, config)
    flow = make_flow(source, {"v": query}, (plan,))
    lm.handler = lambda payload: {"result": {"answer": "stable"}, "state": value}
    flow.apply(pd.DataFrame({"message": ["first"]}))
    flow.apply(pd.DataFrame({"message": ["second"]}))
    assert len(lm.calls) == 2
    assert request_payload(lm.calls[1])["Previous result"] == {"answer": "stable"}
    assert hidden_state(flow).iloc[0]["_state"] == value


def test_retraction_guard_includes_private_state_projection() -> None:
    source = am.Source({"message": "Text"})
    query = source.sem_agg(output_cols=["answer"], instruction="Task")
    with pytest.raises(NotImplementedError, match="fallback retractions"):
        make_flow(
            source,
            {"joined": query.select(["answer"]).sem_join(source, instruction="Match")},
            (state_plan(query),),
        )


def test_memory_binding_does_not_reuse_default_plan(lm: Any) -> None:
    class Example(am.Memory):
        log = am.Log()
        summary = log.sem_agg(
            input_cols=["message"], output_cols=["message"], instruction="Task"
        )

    default = Example.differentiate_policy()
    plan = state_plan(Example.summary)
    configured = Example.differentiate_policy(
        rules=DifferentialRules(maintenance_plans=(plan,))
    )
    assert configured.fingerprint != default.fingerprint
    assert Example.differentiate_policy() is default
    lm.handler = lambda payload: {
        "result": {"message": "stable"},
        "state": {"count": len(payload["New inputs"])},
    }
    memory = Example(
        adapter=LotusAdapter(), rules=DifferentialRules(maintenance_plans=(plan,))
    )
    memory.add("first")
    memory.flush()
    assert len(lm.calls) == 1


@pytest.mark.parametrize("with_state", [False, True])
def test_legacy_checkpoint_cannot_discard_bound_plan(lm: Any, with_state: bool) -> None:
    class Example(am.Memory):
        log = am.Log()
        summary = log.sem_agg(
            input_cols=["message"], output_cols=["message"], instruction="Task"
        )

    plan = (
        state_plan(Example.summary)
        if with_state
        else am.MaintenancePlan.bind(
            Example.summary,
            {
                "rule": "agg-composable",
                "state_schema": None,
                "maintenance_instruction": None,
                "rationale": "No state needed",
            },
        )
    )
    memory = Example(
        adapter=LotusAdapter(), rules=DifferentialRules(maintenance_plans=(plan,))
    )
    before = memory._runtime.snapshot_state()
    legacy = {
        "schema_version": 1,
        "state": before["state"],
        "window_next_start": {},
        "upstream_log_count": {},
    }
    with pytest.raises(ValueError, match="bound maintenance plans"):
        memory._runtime.restore_state(legacy)
    assert (
        memory._runtime.snapshot_state()["plan_fingerprint"]
        == before["plan_fingerprint"]
    )
    assert not lm.calls


def test_equivalent_normalized_tasks_share_one_state_execution(lm: Any) -> None:
    source = am.Source({"message": "Text"})
    implicit = source.sem_agg(output_cols=["answer"], instruction="Task")
    explicit = source.sem_agg(
        input_cols=["message"], output_cols=["answer"], instruction="Task"
    )
    flow = make_flow(
        source,
        {"a": implicit, "b": explicit},
        (state_plan(implicit), state_plan(explicit)),
    )
    lm.handler = counter
    flow.apply(pd.DataFrame({"message": ["one"]}))
    assert len(lm.calls) == 1
    pd.testing.assert_frame_equal(flow.view("a"), flow.view("b"))
