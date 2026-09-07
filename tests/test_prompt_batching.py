"""Tests for the shared semantic prompt batching runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.prompt_batching import (
    ParsedPromptBatch,
    PromptBatchItem,
    PromptBatchRequest,
    PromptBatching,
    run_prompt_batches,
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    text: str


class _Model:
    max_ctx_len = 1_000

    def __init__(self, output_batches: list[list[str]]) -> None:
        self._output_batches = iter(output_batches)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def count_tokens(self, prompt: Any) -> int:
        return len(prompt[1]["content"])

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        self.calls.append((prompts, kwargs))
        return SimpleNamespace(outputs=next(self._output_batches))


def _request(tasks: tuple[_Task, ...]) -> PromptBatchRequest:
    return PromptBatchRequest(
        task_ids=tuple(task.task_id for task in tasks),
        prompt=[
            {"role": "system", "content": "Process each task independently."},
            {"role": "user", "content": "|".join(task.text for task in tasks)},
        ],
        max_tokens=100,
    )


def _parse(raw_output: str) -> ParsedPromptBatch[str]:
    items = tuple(
        PromptBatchItem(task_id, value)
        for task_id, value in (
            part.split("=", 1) for part in raw_output.split(",") if part
        )
    )
    return ParsedPromptBatch(items=items)


def test_prompt_batching_validates_max_tasks() -> None:
    assert PromptBatching().max_tasks is None
    assert PromptBatching(max_tasks=4).max_tasks == 4
    with pytest.raises(TypeError, match="integer"):
        PromptBatching(max_tasks=True)
    with pytest.raises(ValueError, match="at least 1"):
        PromptBatching(max_tasks=0)


def test_execution_config_accepts_one_prompt_batching_contract() -> None:
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    config = LotusExecutionConfig(prompt_batching=PromptBatching(max_tasks=8))

    assert config.prompt_batching == PromptBatching(max_tasks=8)
    with pytest.raises(TypeError, match="PromptBatching"):
        LotusExecutionConfig(prompt_batching="all")  # type: ignore[arg-type]


def test_prompt_batching_changes_only_enabled_maintenance_fingerprint() -> None:
    from agent_memory.adapters.lotus import LotusAdapter
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    baseline = LotusAdapter(config=LotusExecutionConfig())
    explicit_disabled = LotusAdapter(config=LotusExecutionConfig(prompt_batching=None))
    all_ready = LotusAdapter(
        config=LotusExecutionConfig(prompt_batching=PromptBatching())
    )
    bounded = LotusAdapter(
        config=LotusExecutionConfig(prompt_batching=PromptBatching(max_tasks=8))
    )

    assert (
        explicit_disabled.maintenance_execution_fingerprint
        == baseline.maintenance_execution_fingerprint
    )
    assert all_ready.maintenance_execution_fingerprint != baseline.maintenance_execution_fingerprint
    assert bounded.maintenance_execution_fingerprint != all_ready.maintenance_execution_fingerprint


def test_all_ready_tasks_share_one_prompt_and_restore_input_order() -> None:
    model = _Model([["task_2=C,task_0=A,task_1=B"]])
    tasks = tuple(_Task(f"task_{index}", value) for index, value in enumerate("ABC"))

    result = run_prompt_batches(
        tasks,
        task_id=lambda task: task.task_id,
        build_request=_request,
        parse_results=_parse,
        model=model,
        config=PromptBatching(),
        max_retries=0,
        progress_bar_desc="Testing",
        operator="test",
    )

    assert result.outputs == ("A", "B", "C")
    assert result.prompt_count == 1
    assert result.retry_count == 0
    assert result.chunk_sizes == (3,)
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 1


def test_shared_runner_records_compact_execution_mechanics(tmp_path) -> None:
    model = _Model([["task_0=A,task_1=B"]])
    tasks = (_Task("task_0", "A"), _Task("task_1", "B"))

    run_prompt_batches(
        tasks,
        task_id=lambda task: task.task_id,
        build_request=_request,
        parse_results=_parse,
        model=model,
        config=PromptBatching(),
        max_retries=0,
        progress_bar_desc="Testing",
        operator="sem_filter",
        trace_dir=tmp_path,
    )

    [event] = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert event["operator"] == "sem_filter"
    assert event["event_type"] == "prompt_batching"
    assert event["task_count"] == 2
    assert event["prompt_count"] == 1
    assert event["chunk_sizes"] == [2]
    assert event["structured_output_token_limit"] == 100
    assert event["retry_count"] == 0


def test_explicit_limit_chunks_tasks_deterministically() -> None:
    model = _Model([["task_0=A,task_1=B", "task_2=C,task_3=D", "task_4=E"]])
    tasks = tuple(_Task(f"task_{index}", value) for index, value in enumerate("ABCDE"))

    result = run_prompt_batches(
        tasks,
        task_id=lambda task: task.task_id,
        build_request=_request,
        parse_results=_parse,
        model=model,
        config=PromptBatching(max_tasks=2),
        max_retries=0,
        progress_bar_desc="Testing",
        operator="test",
    )

    assert result.outputs == tuple("ABCDE")
    assert result.chunk_sizes == (2, 2, 1)


def test_configured_batch_fails_instead_of_silently_shrinking_for_context() -> None:
    model = _Model([])
    model.max_ctx_len = 109
    tasks = (
        _Task("task_0", "AAAA"),
        _Task("task_1", "BBBB"),
        _Task("task_2", "CCCC"),
    )

    with pytest.raises(ValueError, match="configured prompt batch"):
        run_prompt_batches(
            tasks,
            task_id=lambda task: task.task_id,
            build_request=_request,
            parse_results=_parse,
            model=model,
            config=PromptBatching(),
            max_retries=0,
            progress_bar_desc="Testing",
            operator="test",
        )

    assert model.calls == []


def test_prompt_batches_require_one_fixed_output_token_limit() -> None:
    model = _Model([])
    tasks = (_Task("task_0", "A"), _Task("task_1", "B"))

    def varying_request(batch: tuple[_Task, ...]) -> PromptBatchRequest:
        request = _request(batch)
        return PromptBatchRequest(
            task_ids=request.task_ids,
            prompt=request.prompt,
            max_tokens=100 + int(batch[0].task_id.removeprefix("task_")),
        )

    with pytest.raises(ValueError, match="same output token limit"):
        run_prompt_batches(
            tasks,
            task_id=lambda task: task.task_id,
            build_request=varying_request,
            parse_results=_parse,
            model=model,
            config=PromptBatching(max_tasks=1),
            max_retries=0,
            progress_bar_desc="Testing",
            operator="test",
        )

    assert model.calls == []


def test_only_invalid_prompt_is_retried() -> None:
    model = _Model(
        [
            ["invalid", "task_2=C,task_3=D"],
            ["task_0=A,task_1=B"],
        ]
    )
    tasks = tuple(_Task(f"task_{index}", value) for index, value in enumerate("ABCD"))

    def parse(raw_output: str) -> ParsedPromptBatch[str]:
        if raw_output == "invalid":
            raise ValueError("invalid output")
        return _parse(raw_output)

    result = run_prompt_batches(
        tasks,
        task_id=lambda task: task.task_id,
        build_request=_request,
        parse_results=parse,
        model=model,
        config=PromptBatching(max_tasks=2),
        max_retries=1,
        progress_bar_desc="Testing",
        operator="test",
    )

    assert result.outputs == ("A", "B", "C", "D")
    assert result.prompt_count == 3
    assert result.retry_count == 1
    assert result.raw_output_attempts == (
        ("invalid", "task_0=A,task_1=B"),
        ("invalid", "task_0=A,task_1=B"),
        ("task_2=C,task_3=D",),
        ("task_2=C,task_3=D",),
    )
    assert len(model.calls) == 2
    assert len(model.calls[1][0]) == 1


@pytest.mark.parametrize(
    ("raw_output", "message"),
    (
        ("task_0=A", "missing"),
        ("task_0=A,task_1=B,task_9=X", "unknown"),
        ("task_0=A,task_0=B,task_1=C", "duplicate"),
    ),
)
def test_shared_runner_rejects_invalid_task_id_coverage(
    raw_output: str,
    message: str,
) -> None:
    model = _Model([[raw_output]])
    tasks = (_Task("task_0", "A"), _Task("task_1", "B"))

    with pytest.raises(ValueError, match=message):
        run_prompt_batches(
            tasks,
            task_id=lambda task: task.task_id,
            build_request=_request,
            parse_results=_parse,
            model=model,
            config=PromptBatching(),
            max_retries=0,
            progress_bar_desc="Testing",
            operator="test",
        )


def test_empty_input_skips_model() -> None:
    model = _Model([])

    result = run_prompt_batches(
        (),
        task_id=lambda task: task.task_id,
        build_request=_request,
        parse_results=_parse,
        model=model,
        config=PromptBatching(),
        max_retries=0,
        progress_bar_desc="Testing",
        operator="test",
    )

    assert result.outputs == ()
    assert result.prompt_count == 0
    assert model.calls == []


def test_structured_transform_batches_rows_and_preserves_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    from agent_memory.adapters.lotus.structured import StructuredLMExecutor
    from agent_memory.policy.logical import ColumnSpec

    raw_output = (
        '{"results":['
        '{"task_id":"task_1","output":{"label":"B"}},'
        '{"task_id":"task_0","output":{"label":"A"}}]}'
    )
    model = _Model([[raw_output]])
    model.max_tokens = 128
    model.max_ctx_len = 100_000
    model.cache = None
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    generation = StructuredLMExecutor(
        pd.DataFrame({"text": ["alpha", "beta"]})
    )(
        input_cols=("text",),
        output_cols=(ColumnSpec("label"),),
        instruction="Label {text}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={},
        structured_max_tokens=128,
        prompt_batching=PromptBatching(),
        operator="sem_map",
    )

    assert generation.parsed_outputs == ({"label": "A"}, {"label": "B"})
    assert generation.raw_output_attempts == ((raw_output,), (raw_output,))
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 1
    shared_prompt = str(model.calls[0][0][0])
    assert "task_0" in shared_prompt and "task_1" in shared_prompt


def test_structured_flat_map_keeps_emitted_rows_with_their_source_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    from agent_memory.adapters.lotus.structured import StructuredLMExecutor
    from agent_memory.policy.logical import ColumnSpec

    model = _Model(
        [[
            '{"results":['
            '{"task_id":"task_0","output":{"rows":['
            '{"fact":"A1"},{"fact":"A2"}]}},'
            '{"task_id":"task_1","output":{"rows":[]}}]}'
        ]]
    )
    model.max_tokens = 128
    model.max_ctx_len = 100_000
    model.cache = None
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    generation = StructuredLMExecutor(
        pd.DataFrame({"text": ["alpha", "beta"]})
    )(
        input_cols=("text",),
        output_cols=(ColumnSpec("fact"),),
        instruction="Extract facts from {text}.",
        shape="array",
        progress_bar_desc="Flat mapping",
        model_kwargs={},
        structured_max_tokens=128,
        prompt_batching=PromptBatching(),
        operator="sem_flat_map",
    )

    assert generation.parsed_outputs == (
        [{"fact": "A1"}, {"fact": "A2"}],
        [],
    )


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    (
        ('{"rows":[]}', []),
        ('{"rows":[{"fact":"A1"},{"fact":"A2"}]}', [
            {"fact": "A1"},
            {"fact": "A2"},
        ]),
    ),
)
def test_structured_flat_map_repairs_a_valid_singleton_output_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_output: str,
    expected: list[dict[str, str]],
) -> None:
    import lotus

    from agent_memory.adapters.lotus.structured import StructuredLMExecutor
    from agent_memory.policy.logical import ColumnSpec

    model = _Model([[raw_output]])
    model.max_tokens = 128
    model.max_ctx_len = 100_000
    model.cache = None
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    generation = StructuredLMExecutor(pd.DataFrame({"text": ["alpha"]}))(
        input_cols=("text",),
        output_cols=(ColumnSpec("fact"),),
        instruction="Extract facts from {text}.",
        shape="array",
        progress_bar_desc="Flat mapping",
        model_kwargs={},
        structured_max_tokens=128,
        structured_parse_retries=0,
        semantic_trace_dir=tmp_path,
        prompt_batching=PromptBatching(max_tasks=1),
        operator="sem_flat_map",
    )

    assert generation.parsed_outputs == (expected,)
    assert generation.raw_output_attempts == ((raw_output,),)
    assert len(model.calls) == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    [batch_event] = [
        event for event in events if event["event_type"] == "prompt_batching"
    ]
    assert batch_event["retry_count"] == 0
    assert batch_event["structured_output_repair_count"] == 1
    assert batch_event["structured_output_repair_methods"] == [
        "singleton-envelope"
    ]


def test_structured_map_repairs_a_valid_singleton_output_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    from agent_memory.adapters.lotus.structured import StructuredLMExecutor
    from agent_memory.policy.logical import ColumnSpec

    raw_output = '{"label":"A"}'
    model = _Model([[raw_output]])
    model.max_tokens = 128
    model.max_ctx_len = 100_000
    model.cache = None
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    generation = StructuredLMExecutor(pd.DataFrame({"text": ["alpha"]}))(
        input_cols=("text",),
        output_cols=(ColumnSpec("label"),),
        instruction="Label {text}.",
        shape="object",
        progress_bar_desc="Mapping",
        model_kwargs={},
        structured_max_tokens=128,
        structured_parse_retries=0,
        prompt_batching=PromptBatching(max_tasks=1),
        operator="sem_map",
    )

    assert generation.parsed_outputs == ({"label": "A"},)
    assert generation.raw_output_attempts == ((raw_output,),)
    assert len(model.calls) == 1


def test_structured_prompt_batch_does_not_guess_a_multi_task_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    from agent_memory.adapters.lotus.structured import StructuredLMExecutor
    from agent_memory.policy.logical import ColumnSpec

    model = _Model([['{"rows":[]}']])
    model.max_tokens = 128
    model.max_ctx_len = 100_000
    model.cache = None
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    with pytest.raises(ValueError, match="after 1 attempt"):
        StructuredLMExecutor(pd.DataFrame({"text": ["alpha", "beta"]}))(
            input_cols=("text",),
            output_cols=(ColumnSpec("fact"),),
            instruction="Extract facts from {text}.",
            shape="array",
            progress_bar_desc="Flat mapping",
            model_kwargs={},
            structured_max_tokens=128,
            structured_parse_retries=0,
            prompt_batching=PromptBatching(),
            operator="sem_flat_map",
        )


@pytest.mark.parametrize(
    "raw_output",
    (
        '{"rows":[],"extra":true}',
        '{"rows":[],"rows":[]}',
        '{"rows":[{"wrong":"value"}]}',
        '{"rows":',
    ),
)
def test_structured_prompt_batch_rejects_invalid_singleton_inner_outputs(
    monkeypatch: pytest.MonkeyPatch,
    raw_output: str,
) -> None:
    import lotus

    from agent_memory.adapters.lotus.structured import StructuredLMExecutor
    from agent_memory.policy.logical import ColumnSpec

    model = _Model([[raw_output]])
    model.max_tokens = 128
    model.max_ctx_len = 100_000
    model.cache = None
    monkeypatch.setattr(lotus.settings, "lm", model)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)

    with pytest.raises(ValueError, match="after 1 attempt"):
        StructuredLMExecutor(pd.DataFrame({"text": ["alpha"]}))(
            input_cols=("text",),
            output_cols=(ColumnSpec("fact"),),
            instruction="Extract facts from {text}.",
            shape="array",
            progress_bar_desc="Flat mapping",
            model_kwargs={},
            structured_max_tokens=128,
            structured_parse_retries=0,
            prompt_batching=PromptBatching(max_tasks=1),
            operator="sem_flat_map",
        )
