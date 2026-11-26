"""Value-preserving JSON structural repair and its refusal boundaries."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any

import pytest
from jsonschema import ValidationError

from agent_memory.adapters.lotus import json_output
from agent_memory.adapters.lotus.json_output import (
    load_structured_json_with_syntax_repair,
    parse_schema_json,
    repair_json_structure,
    strict_json_loads,
    validate_json,
)
from agent_memory.maintenance import object_schema


def test_validate_json_accepts_frozen_schema_without_changing_value() -> None:
    schema = MappingProxyType(object_schema({"count": {"type": "integer"}}))
    value = {"count": 2}

    assert validate_json(value, schema) is None
    assert value == {"count": 2}


@pytest.mark.parametrize("value", [{"count": "2"}, {"count": 2, "extra": 0}, {}])
def test_validate_json_preserves_schema_errors(value: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        validate_json(value, object_schema({"count": {"type": "integer"}}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validate_json_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        validate_json(value, {"type": "number"})


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("fenced", [False, True])
def test_schema_parser_records_envelope_without_changing_raw(
    wrapped: bool, fenced: bool,
) -> None:
    value = {"answer": "unchanged } properties text", "count": 2}
    schema = object_schema({"answer": {"type": "string"}, "count": {"type": "integer"}})
    raw = json.dumps({"properties": value} if wrapped else value)
    if fenced:
        raw = f"```json\n{raw}\n```"
    raw = f"  {raw}\n"

    parsed = parse_schema_json(raw, schema)

    assert parsed.value == value
    assert parsed.raw_output == raw
    assert parsed.unwrapped_properties is wrapped


def test_schema_parser_prefers_a_valid_properties_business_field() -> None:
    schema = object_schema({"properties": object_schema({"count": {"type": "integer"}})})
    raw = '{"properties":{"count":2}}'

    parsed = parse_schema_json(raw, schema)

    assert parsed.value == {"properties": {"count": 2}}
    assert not parsed.unwrapped_properties


@pytest.mark.parametrize("raw", [
    '{"properties":{"count":2},"extra":0}',
    '{"properties":{"count":2,"extra":0}}',
    '{"properties":{"count":"2"}}',
    '{"properties":{"count":true}}',
    '{"properties":{}}',
    '{"properties":null}',
    '{"properties":[{"count":2}]}',
    '{"properties":{"properties":{"count":2}}}',
    '{"result":{"count":2}}',
    '{"properties":{"count":2,"count":3}}',
    '{"properties":{"count":2},"properties":{"count":2}}',
    '{"properties":{"count":NaN}}',
    '{"properties":{"count":2}',
    "{'properties':{'count':2}}",
])
def test_schema_parser_refuses_other_repairs(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_schema_json(raw, object_schema({"count": {"type": "integer"}}))


def test_schema_parser_does_not_unwrap_on_business_validation_failure() -> None:
    # Both outer and inner values satisfy this schema, so direct validation wins.
    raw = '{"properties":{"answer":"inner"}}'
    parsed = parse_schema_json(raw, {"type": "object"})

    assert parsed.value == json.loads(raw)
    assert not parsed.unwrapped_properties


def _validate(raw: str) -> Any:
    value = strict_json_loads(raw)
    if value != {"a": ['unchanged } , : " text', 42, True, None]}:
        raise ValueError("contract mismatch")
    return value


@pytest.mark.parametrize(
    "damaged",
    [
        '{"a" ["unchanged } , : \\" text",42,true,null]}',
        '{"a"::["unchanged } , : \\" text",42,true,null]}',
        '{"a":["unchanged } , : \\" text" 42,true,null]}',
        '{"a":["unchanged } , : \\" text",42,true,null]}}',
        '{"a":["unchanged } , : \\" text",42,true,null',
    ],
)
def test_repair_preserves_every_scalar(damaged: str) -> None:
    repaired = repair_json_structure(damaged, validator=_validate)
    assert repaired.value == _validate(json.dumps(repaired.value))
    assert len(repaired.edits) == 1


def test_missing_result_envelope_closure() -> None:
    expected = {
        "results": [
            {"task_id": "task_0", "output": {"rows": []}},
            {"task_id": "task_1", "output": {"rows": []}},
        ]
    }
    damaged = (
        '{"results":[{"task_id":"task_0","output":{"rows":[]},'
        '{"task_id":"task_1","output":{"rows":[]}}]}'
    )

    def validate(raw: str) -> Any:
        value = strict_json_loads(raw)
        if value != expected:
            raise ValueError("task contract")
        return value

    assert repair_json_structure(damaged, validator=validate).value == expected


@pytest.mark.parametrize(
    "damaged",
    [
        '{"a": "cut off',
        '{"a": 1e',
        '{"a": tru',
        '{"a": NaN}',
        '{"a":1,"a":2}',
        '{"a":1,"a":2}}',
        '{"a":1 "b":2 "c":3}',
        '{"a":[1,2]}}}',
        '{"a":1}',  # Valid JSON with a bad operator contract is not syntax damage.
    ],
)
def test_refuses_guessing_values_or_bad_contracts(damaged: str) -> None:
    with pytest.raises(ValueError):
        repair_json_structure(damaged, validator=_validate)


def test_rejects_ambiguous_repairs() -> None:
    # Closing the nested array here or closing it at EOF gives different values.
    with pytest.raises(ValueError, match="one valid candidate"):
        repair_json_structure("[[1,2]", validator=strict_json_loads)


def test_repairs_repeated_batch_envelope_damage() -> None:
    raw = (
        '{"results":[{"task_id":"a","output":{"rows":[{"body":"A"}]},'
        '{"task_id":"b","output":{"rows":[{"body":"B"}]}]}'
    )
    expected = {
        "results": [
            {"task_id": "a", "output": {"rows": [{"body": "A"}]}},
            {"task_id": "b", "output": {"rows": [{"body": "B"}]}},
        ]
    }

    def validate(value: str) -> Any:
        parsed = strict_json_loads(value)
        if parsed != expected:
            raise ValueError("batch contract")
        return parsed

    repaired = repair_json_structure(raw, validator=validate)
    assert repaired.value == expected
    assert len(repaired.edits) == 2


@pytest.mark.parametrize("count", [3, 5, 8])
def test_multiple_closures_preserve_all_items(count: int) -> None:
    expected = {
        "items": [{"id": str(i), "value": {"text": f"item {i}"}} for i in range(count)]
    }
    raw = (
        '{"items":['
        + ",".join(json.dumps(item)[:-1] for item in expected["items"])
        + "]}"
    )

    def validate(text: str) -> Any:
        value = strict_json_loads(text)
        if value != expected:
            raise ValueError("item contract")
        return value

    repaired = repair_json_structure(raw, validator=validate)
    assert repaired.value == expected
    assert len(repaired.edits) == count
    assert all(kind == "insert" and token == "}" for kind, _, token in repaired.edits)


@pytest.mark.parametrize("raw", ['{"a":1 "b":2 "c":3}', '{"a":1}}}'])
def test_multiple_separator_edits(raw: str) -> None:
    expected = {"a": 1, "b": 2, "c": 3} if "b" in raw else {"a": 1}
    result = repair_json_structure(raw, validator=strict_json_loads)
    assert result.value == expected
    assert len(result.edits) == 2


def test_multiple_edits_still_reject_ambiguity() -> None:
    with pytest.raises(ValueError, match="one valid candidate"):
        repair_json_structure("[[1 2]", validator=strict_json_loads)


def test_does_not_accept_first_candidate_before_search_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(json_output, "MAX_REPAIR_CANDIDATES", 1)
    monkeypatch.setattr(json_output, "_local_candidates", lambda *_: {
        "[[1,2]]": ("insert", 6, "]"),
        "[[1],2]": ("insert", 3, "]"),
    })
    checked = []

    def validate(raw: str) -> Any:
        result = strict_json_loads(raw)
        checked.append(result)
        return result

    with pytest.raises(ValueError, match="bound"):
        repair_json_structure("[[1,2]", validator=validate)
    assert checked == [[[1, 2]]]


def test_edit_limit_is_not_silently_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(json_output, "MAX_REPAIR_EDITS", 1)
    with pytest.raises(ValueError):
        repair_json_structure('{"a":1 "b":2 "c":3}', validator=strict_json_loads)


def test_large_output_work_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(json_output, "MAX_REPAIR_CHARACTERS", 1)
    with pytest.raises(ValueError, match="bound"):
        repair_json_structure('{"a":1}}', validator=strict_json_loads)


def test_recorded_edits_reproduce_the_accepted_text() -> None:
    raw = '{"a":1 "b":2 "c":3}}'
    repaired = repair_json_structure(raw, validator=strict_json_loads)
    candidate = raw
    for kind, position, text in repaired.edits:
        if kind == "insert":
            candidate = candidate[:position] + text + candidate[position:]
        else:
            assert candidate[position : position + len(text)] == text
            candidate = candidate[:position] + candidate[position + len(text) :]
    assert strict_json_loads(candidate) == repaired.value == {"a": 1, "b": 2, "c": 3}


@pytest.mark.parametrize(
    ("raw", "method"),
    [
        ('{"a":1}', None),
        ("{a:1,}", "json5"),
        ('```json\n{"a":1}\n```', "json5-code-fence"),
    ],
)
def test_existing_json5_and_fence_contract(raw: str, method: str | None) -> None:
    result = load_structured_json_with_syntax_repair(
        raw, operator="test", expected_shape="object"
    )
    assert result.value == {"a": 1}
    assert result.repair_method == method
