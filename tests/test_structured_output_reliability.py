"""Provider-free integration contracts for structured batch transport and repair."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import lotus
import pandas as pd
import pytest

from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
)
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_agg_batch_prompting import (
    execute_batch_prompted_sem_agg,
)
from agent_memory.adapters.lotus.sem_filter_batch_prompting import (
    execute_batch_prompted_sem_filter,
)
from agent_memory.adapters.lotus.sem_topk_join import _listwise_topk
from agent_memory.adapters.lotus.structured import (
    StructuredLMExecutor,
    _parse_structured_prompt_batch,
)
from agent_memory.policy.logical import ColumnSpec
from agent_memory.runtime.executor import PolicyExecutor


TRANSPORTS = ("chat-json-object", "responses-json-schema")
PROTOCOLS = ("predicate", "aggregate", "object", "array", "listwise")
OUTPUT_COLS = (ColumnSpec("label"),)


class _Model:
    """Capture physical requests and return local output/metadata fixtures only."""

    max_tokens = 256
    max_ctx_len = 100_000
    cache = None

    def __init__(
        self,
        outputs: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.outputs = outputs
        self.metadata = metadata
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Use a deterministic, provider-free context estimate."""
        return sum(len(message["content"]) for message in messages)

    def __call__(self, messages: Any, **kwargs: Any) -> SimpleNamespace:
        """Never retry implicitly or invoke an external model."""
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return SimpleNamespace(outputs=self.outputs, response_metadata=self.metadata)


def _payload(protocol: str, indices: tuple[int, ...] = (0, 1)) -> dict[str, Any]:
    if protocol == "predicate":
        return {
            "decisions": [
                {"row_id": f"row_{index}", "keep": index % 2 == 0}
                for index in indices
            ]
        }
    if protocol == "listwise":
        return {
            "results": [
                {"task_id": f"task_{index}", "selected_ids": [f"candidate_{index * 3}"]}
                for index in indices
            ]
        }
    id_field, prefix = ("group_id", "group") if protocol == "aggregate" else (
        "task_id", "task"
    )
    return {
        "results": [
            {
                id_field: f"{prefix}_{index}",
                "output": (
                    {"rows": [{"label": f"value_{index}"}]}
                    if protocol == "array"
                    else {"label": f"value_{index}"}
                ),
            }
            for index in indices
        ]
    }


def _execute(
    protocol: str,
    model: _Model,
    monkeypatch: pytest.MonkeyPatch,
    *,
    transport: str = "responses-json-schema",
    count: int = 2,
    batch_size: int | None = None,
    trace_dir: Path | None = None,
) -> Any:
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    batching = PromptBatching(max_tasks=batch_size)
    config = LotusExecutionConfig(
        prompt_batching=batching,
        structured_output_transport=transport,
        structured_parse_retries=0,
        structured_max_tokens=256,
        semantic_trace_dir=trace_dir,
    )
    context: Any = SimpleNamespace(config=config)
    source = pd.DataFrame({"text": [f'Input {index}: "quoted"\nnext line' for index in range(count)]})
    if protocol == "predicate":
        return execute_batch_prompted_sem_filter(
            source,
            instruction="Keep relevant {text}.",
            context=context,
            prompt_batching=batching,
        ).decisions
    if protocol == "aggregate":
        return execute_batch_prompted_sem_agg(
            [[text] for text in source["text"]],
            instruction="Summarize each group.",
            output_cols=OUTPUT_COLS,
            model=model,
            prompt_batching=batching,
            structured_output_transport=transport,
            max_retries=0,
            model_kwargs={},
            progress_bar_desc="Testing aggregates",
            trace_dir=trace_dir,
        ).outputs
    if protocol in {"object", "array"}:
        return StructuredLMExecutor(source)(
            input_cols=("text",),
            output_cols=OUTPUT_COLS,
            instruction="Label {text}.",
            shape=protocol,
            progress_bar_desc="Testing structured outputs",
            model_kwargs={},
            structured_max_tokens=256,
            structured_parse_retries=0,
            semantic_trace_dir=trace_dir,
            prompt_batching=batching,
            structured_output_transport=transport,
            operator="sem_map" if protocol == "object" else "sem_flat_map",
        ).parsed_outputs
    candidates = pd.DataFrame({
        PAIR_LEFT_ID_COLUMN: [index for index in range(count) for _ in range(3)],
        PAIR_RIGHT_ID_COLUMN: list(range(count * 3)),
        PAIR_LEFT_TEXT_COLUMN: [text for text in source["text"] for _ in range(3)],
        PAIR_RIGHT_TEXT_COLUMN: [f"candidate text {index}" for index in range(count * 3)],
    })
    matches, retries = _listwise_topk(
        candidates,
        instruction="Match the same entity.",
        left_label="left",
        right_label="right",
        k=2,
        context=context,
    )
    assert retries == 0
    return matches


def _object_schema(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def _expected_schema(protocol: str) -> dict[str, Any]:
    scalar = {"type": ["string", "number", "boolean", "null"]}
    if protocol == "predicate":
        item = _object_schema({"row_id": {"type": "string"}, "keep": {"type": "boolean"}})
        return _object_schema({"decisions": {"type": "array", "items": item}})
    if protocol == "listwise":
        item = _object_schema({
            "task_id": {"type": "string"},
            "selected_ids": {"type": "array", "items": {"type": "string"}},
        })
    else:
        output = _object_schema({"label": scalar})
        if protocol == "array":
            output = _object_schema({"rows": {"type": "array", "items": output}})
        item = _object_schema({
            "group_id" if protocol == "aggregate" else "task_id": {"type": "string"},
            "output": output,
        })
    return _object_schema({"results": {"type": "array", "items": item}})


@pytest.mark.parametrize(
    ("protocol", "expected"),
    (
        ("predicate", (True, False)),
        ("aggregate", ({"label": "value_0"}, {"label": "value_1"})),
        ("object", ({"label": "value_0"}, {"label": "value_1"})),
        ("array", ([{"label": "value_0"}], [{"label": "value_1"}])),
        ("listwise", [(0, 0, None), (1, 3, None)]),
    ),
)
def test_protocol_schema_reaches_model_without_changing_prompt_bytes(
    protocol: str, expected: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only response_format changes across transports, not prompts or task results."""

    raw = json.dumps(_payload(protocol, (1, 0)))
    models = [_Model([raw]), _Model([raw])]
    for transport, model in zip(TRANSPORTS, models, strict=True):
        assert _execute(protocol, model, monkeypatch, transport=transport) == expected
        assert len(model.calls) == 1
    (chat_messages, chat_kwargs), (responses_messages, responses_kwargs) = (
        model.calls[0] for model in models
    )
    assert json.dumps(chat_messages, ensure_ascii=False).encode("utf-8") == (
        json.dumps(responses_messages, ensure_ascii=False).encode("utf-8")
    )
    assert chat_kwargs["response_format"] == {"type": "json_object"}
    assert responses_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "semantic_batch", "schema": _expected_schema(protocol)},
    }
    assert {key: value for key, value in chat_kwargs.items() if key != "response_format"} == {
        key: value for key, value in responses_kwargs.items() if key != "response_format"
    }


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("defect", ("missing", "duplicate", "unknown", "extra"))
@pytest.mark.parametrize("needs_repair", (False, True))
def test_full_task_id_coverage_is_required_even_after_delimiter_repair(
    protocol: str, transport: str, defect: str, needs_repair: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Syntax repair cannot invent, discard, or reassign task identities."""

    indices = {"missing": (0,), "duplicate": (0, 0), "unknown": (0, 9), "extra": (0, 1, 9)}[defect]
    raw = json.dumps(_payload(protocol, indices))
    model = _Model([raw[:-1] if needs_repair else raw])
    with pytest.raises(ValueError, match="invalid structured output after 1 attempt"):
        _execute(protocol, model, monkeypatch, transport=transport)
    assert len(model.calls) == 1


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_task_ids_must_match_each_request_not_just_the_global_batch(
    protocol: str, transport: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Globally valid IDs cannot be swapped across independently chunked prompts."""

    model = _Model([json.dumps(_payload(protocol, (1,))), json.dumps(_payload(protocol, (0,)))])
    with pytest.raises(ValueError, match="unknown task IDs"):
        _execute(protocol, model, monkeypatch, transport=transport, batch_size=1)
    assert len(model.calls[0][0]) == 2


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("needs_repair", (False, True))
@pytest.mark.parametrize(
    ("selected_ids", "message"),
    (
        (["candidate_99"], "unknown IDs"),
        (["candidate_3"], "unknown IDs"),
        (["candidate_0", "candidate_0"], "unique"),
        (["candidate_0", "candidate_1", "candidate_2"], "more than k=2"),
        ([0], "list of strings"),
    ),
)
def test_listwise_candidates_remain_task_local_unique_and_cardinality_bounded(
    transport: str, needs_repair: bool, selected_ids: list[Any], message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repaired selection must still match the complete candidate contract."""

    payload = _payload("listwise")
    payload["results"][0]["selected_ids"] = selected_ids
    raw = json.dumps(payload)
    if not needs_repair:
        with pytest.raises(ValueError, match=message):
            _execute("listwise", _Model([raw]), monkeypatch, transport=transport)
    else:
        with pytest.raises(ValueError, match="invalid structured output after 1 attempt"):
            _execute("listwise", _Model([raw[:-1]]), monkeypatch, transport=transport)


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("shape", ("object", "array"))
def test_bounded_repair_rejects_extra_fields_without_tightening_valid_parser(
    transport: str, shape: Literal["object", "array"], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict repair validator does not replace the established normal parser."""

    payload = _payload(shape)
    output = payload["results"][0]["output"]
    if shape == "object":
        output["extra"] = "must not be silently dropped during repair"
    else:
        output["rows"][0]["extra"] = "must not be silently dropped during repair"
    raw = json.dumps(payload)
    parsed = _parse_structured_prompt_batch(
        raw, output_cols=OUTPUT_COLS, shape=shape,
        require_explanation=False, operator="sem_map" if shape == "object" else "sem_flat_map",
    )
    expected = {"label": "value_0"} if shape == "object" else [{"label": "value_0"}]
    assert parsed.items[0].value == (expected, None)
    assert _execute(shape, _Model([raw]), monkeypatch, transport=transport)[0] == expected
    with pytest.raises(ValueError, match="invalid structured output after 1 attempt"):
        _execute(shape, _Model([raw[:-1]]), monkeypatch, transport=transport)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize(
    "completion",
    ({"finish_reason": "length"}, {"status": "incomplete", "incomplete_reason": "max_output_tokens"}),
)
def test_known_truncation_metadata_blocks_otherwise_valid_structural_repair(
    protocol: str, transport: str, completion: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Complete scalar tokens alone are not permission to repair known truncation."""

    import agent_memory.adapters.lotus.prompt_batching as batching_module

    raw = json.dumps(_payload(protocol))[:-1]
    _execute(protocol, _Model([raw]), monkeypatch, transport=transport)

    def forbidden_repair(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("known truncation reached structural repair")

    monkeypatch.setattr(batching_module, "repair_json_structure", forbidden_repair)
    model = _Model([raw], [completion])
    with pytest.raises(ValueError, match="incomplete structured response"):
        _execute(protocol, model, monkeypatch, transport=transport, trace_dir=tmp_path)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    failures = [event for event in events if event["event_type"] == "prompt_batching_failure"]
    assert len(failures) == 1
    assert failures[0]["task_count"] == 2
    assert failures[0]["repair_result"] == "not-accepted"
    assert not any(event["event_type"] == "prompt_batching" for event in events)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_repair_accounting_counts_requests_separately_from_affected_tasks(
    protocol: str, transport: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Two repaired requests covering four tasks must not be reported as four repairs."""

    model = _Model([
        json.dumps(_payload(protocol, (0, 1)))[:-1],
        json.dumps(_payload(protocol, (2, 3)))[:-1],
        json.dumps(_payload(protocol, (4,))),
    ])
    result = _execute(
        protocol, model, monkeypatch, transport=transport,
        count=5, batch_size=2, trace_dir=tmp_path,
    )
    assert len(result) == 5
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    [event] = [event for event in events if event["event_type"] == "prompt_batching"]
    assert event["task_count"] == 5
    assert event["prompt_count"] == 3
    assert event["chunk_sizes"] == [2, 2, 1]
    assert event["retry_count"] == 0
    assert event["structured_output_repair_count"] == 2
    assert event["structured_output_repaired_task_count"] == 4
    assert event["syntax_repair_count"] == 2
    assert event["structured_output_repair_version"] == "bounded-json-v2"
    repairs = event["structured_output_repairs"]
    assert len(repairs) == 2
    for repair in repairs:
        assert repair["affected_task_count"] == 2
        assert repair["attempt"] == 1
        assert repair["method"] == "bounded-json-syntax"
        assert repair["version"] == "bounded-json-v2"
        assert repair["edits"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_multiple_closures_share_the_existing_protocol_validator(
    protocol: str, transport: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    payload = _payload(protocol)
    field = next(iter(payload))
    raw = "{" + json.dumps(field) + ":[" + ",".join(
        json.dumps(item)[:-1] for item in payload[field]
    ) + "]}"
    expected = _execute(protocol, _Model([json.dumps(payload)]), monkeypatch,
                        transport=transport)
    model = _Model([raw])
    actual = _execute(protocol, model, monkeypatch, transport=transport,
                      trace_dir=tmp_path)
    assert actual == expected
    assert len(model.calls) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    [event] = [event for event in events if event["event_type"] == "prompt_batching"]
    assert event["retry_count"] == 0
    assert event["structured_output_repair_count"] == 1
    assert event["structured_output_repaired_task_count"] == 2
    assert len(event["structured_output_repairs"][0]["edits"]) == 2


def test_provider_usage_trace_preserves_truncated_exception_response(tmp_path: Path) -> None:
    """Persist partial failure output and native usage without fabricating response fields."""

    from agent_memory.tracing.semantic import write_provider_usage_trace

    partial_output = '{"results":[\n{"task_id":"task_0","output":{"label":"unfinished'
    native_usage = {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 18,
    }
    failure = SimpleNamespace(
        response=SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=partial_output)),
        ]),
        provider_response_metadata={
            "transport": "responses-json-schema",
            "status": "incomplete",
            "finish_reason": "length",
            "incomplete_reason": "max_output_tokens",
            "raw_usage": native_usage,
        },
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_cache_hit_tokens=3,
        ),
    )
    rows = write_provider_usage_trace(
        tmp_path,
        model="deepseek/deepseek-v4-flash",
        responses=[failure, SimpleNamespace()],
    )
    persisted = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert persisted == rows
    event, missing = persisted
    assert event["event_type"] == "provider_usage"
    assert event["provider_finish_reason"] == "length"
    assert event["provider_response_status"] == "incomplete"
    assert event["provider_incomplete_reason"] == "max_output_tokens"
    assert event["provider_response_model"] is None
    assert event["provider_usage_available"] is True
    assert event["provider_prompt_tokens"] == 11
    assert event["provider_completion_tokens"] == 7
    assert event["provider_total_tokens"] == 18
    assert event["provider_prompt_cache_hit_tokens"] == 3
    assert json.loads(Path(event["provider_raw_usage_path"]).read_text()) == native_usage
    assert json.loads(Path(event["provider_partial_output_path"]).read_text()) == {
        "output": partial_output,
    }
    assert "raw_output_path" not in event
    for field in (
        "provider_finish_reason", "provider_response_status",
        "provider_incomplete_reason", "provider_response_model",
    ):
        assert missing[field] is None
    assert missing["provider_usage_available"] is False
    assert "provider_raw_usage_path" not in missing
    assert "provider_partial_output_path" not in missing


def _executor(config: LotusExecutionConfig) -> PolicyExecutor:
    import agent_memory as am
    from agent_memory.planner import PolicyDifferentiator

    policy = PolicyDifferentiator().differentiate(am.ClaudeMemory.spec())
    return PolicyExecutor(policy, adapter=LotusAdapter(config=config))


def test_default_nonbatched_checkpoint_identity_remains_compatible() -> None:
    """Explicit default transport retains the old empty physical identity."""

    snapshot = _executor(LotusExecutionConfig()).snapshot_state()
    assert "adapter_execution_fingerprint" not in snapshot
    restored = _executor(LotusExecutionConfig(structured_output_transport="chat-json-object"))
    restored.restore_state(snapshot)
    assert "adapter_execution_fingerprint" not in restored.snapshot_state()


@pytest.mark.parametrize("max_tasks", (None, 2))
@pytest.mark.parametrize("legacy_version", (None, "bounded-json-v1"))
def test_pre_repair_prompt_batching_checkpoint_is_rejected(
    max_tasks: int | None, legacy_version: str | None,
) -> None:
    """Old parsing contracts must not share a checkpoint with multi-edit repair."""

    batching = PromptBatching(max_tasks=max_tasks)
    config = LotusExecutionConfig(prompt_batching=batching)
    executor = _executor(config)
    snapshot = executor.snapshot_state()
    legacy_contract: dict[str, Any] = {"max_tasks": max_tasks}
    if legacy_version is not None:
        legacy_contract["repair_version"] = legacy_version
    legacy_batch_fingerprint = sha256(
        json.dumps(legacy_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    legacy_physical_fingerprint = sha256(
        f"base_execution_fingerprint=\nprompt_batching={legacy_batch_fingerprint}".encode("utf-8")
    ).hexdigest()
    assert batching.fingerprint != legacy_batch_fingerprint
    assert snapshot["adapter_execution_fingerprint"] != legacy_physical_fingerprint
    _executor(config).restore_state(snapshot)
    legacy_snapshot = {**snapshot, "adapter_execution_fingerprint": legacy_physical_fingerprint}
    with pytest.raises(ValueError, match="adapter execution fingerprint does not match"):
        executor.restore_state(legacy_snapshot)


@pytest.mark.parametrize("batching", (None, PromptBatching(max_tasks=2)))
@pytest.mark.parametrize("source_transport", TRANSPORTS)
def test_chat_and_responses_checkpoints_cannot_cross_physical_transports(
    batching: PromptBatching | None, source_transport: str,
) -> None:
    """Transport changes isolate physical state even when the logical policy is identical."""

    target_transport = next(transport for transport in TRANSPORTS if transport != source_transport)
    source_config = LotusExecutionConfig(
        prompt_batching=batching, structured_output_transport=source_transport,
    )
    source = _executor(source_config)
    target = _executor(LotusExecutionConfig(
        prompt_batching=batching, structured_output_transport=target_transport,
    ))
    snapshot = source.snapshot_state()
    assert source.policy.fingerprint == target.policy.fingerprint
    _executor(source_config).restore_state(snapshot)
    with pytest.raises(ValueError, match="adapter execution fingerprint does not match"):
        target.restore_state(snapshot)
