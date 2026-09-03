"""Physical batching tests for independent semantic aggregate groups."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.adapter import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.relational import execute_agg
from agent_memory.adapters.lotus.sem_agg import (
    execute_native_sem_agg_groups,
    execute_structured_sem_agg_groups,
)
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from agent_memory.policy.aggregates import ArrayAggregateSpec, SemanticAggregateSpec


class _RecordingModel:
    """Return deterministic outputs while retaining every LOTUS batch."""

    max_ctx_len = 32_768
    max_tokens = 512

    def __init__(self, outputs: Sequence[Sequence[str]] | None = None) -> None:
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []
        self._outputs = iter(outputs or ())

    def count_tokens(self, _value: Any) -> int:
        return 1

    def __call__(self, prompts: list[Any], **kwargs: Any) -> Any:
        self.calls.append((prompts, dict(kwargs)))
        try:
            outputs = tuple(next(self._outputs))
        except StopIteration:
            outputs = tuple(_native_output(prompt) for prompt in prompts)
        if len(outputs) != len(prompts):
            raise AssertionError("fake output count must match prompts")
        return SimpleNamespace(outputs=outputs)


def _native_output(prompt: Any) -> str:
    text = str(prompt)
    if "alpha" in text:
        return "alpha summary"
    if "beta" in text:
        return "beta summary"
    return "summary"


def _query(*, structured: bool = False) -> QueryExpr:
    output_cols = (
        (ColumnSpec("name"), ColumnSpec("summary"))
        if structured
        else (ColumnSpec("summary"),)
    )
    return QueryExpr(
        op="sem_agg",
        params={
            "input_cols": ("body",),
            "output_cols": output_cols,
            "instruction": "Summarize {body}.",
        },
    )


def _groups() -> list[pd.DataFrame]:
    return [
        pd.DataFrame({"body": ["alpha"]}),
        pd.DataFrame({"body": ["beta"]}),
    ]


def test_provider_batched_sem_agg_preserves_per_group_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    sequential_model = _RecordingModel()
    monkeypatch.setattr(lotus.settings, "lm", sequential_model)
    sequential = execute_native_sem_agg_groups(
        _query(),
        _groups(),
        ("body",),
        LotusExecutionConfig(),
    )

    batched_model = _RecordingModel()
    monkeypatch.setattr(lotus.settings, "lm", batched_model)
    batched = execute_native_sem_agg_groups(
        _query(),
        _groups(),
        ("body",),
        LotusExecutionConfig(sem_agg_dispatch="provider-batched"),
    )

    sequential_prompts = [
        prompt for prompts, _kwargs in sequential_model.calls for prompt in prompts
    ]
    batched_prompts = [
        prompt for prompts, _kwargs in batched_model.calls for prompt in prompts
    ]
    assert batched == sequential == ["alpha summary", "beta summary"]
    assert batched_prompts == sequential_prompts
    assert [len(prompts) for prompts, _kwargs in sequential_model.calls] == [1, 1]
    assert [len(prompts) for prompts, _kwargs in batched_model.calls] == [2]


def test_provider_batched_sem_agg_preserves_independent_hierarchies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    class HierarchicalModel(_RecordingModel):
        max_ctx_len = 15
        max_tokens = 1

        def count_tokens(self, value: Any) -> int:
            text = str(value)
            if "Your job" in text:
                return 1
            if "Source" in text:
                return 2
            return 8

        def __call__(self, prompts: list[Any], **kwargs: Any) -> Any:
            self.calls.append((prompts, dict(kwargs)))
            outputs = []
            for prompt in prompts:
                text = str(prompt)
                group = "alpha" if "alpha" in text else "beta"
                prefix = "final" if "Source" in text else "partial"
                outputs.append(f"{prefix} {group}")
            return SimpleNamespace(outputs=outputs)

    model = HierarchicalModel()
    monkeypatch.setattr(lotus.settings, "lm", model)
    groups = [
        pd.DataFrame({"body": ["alpha one", "alpha two"]}),
        pd.DataFrame({"body": ["beta one", "beta two"]}),
    ]

    result = execute_native_sem_agg_groups(
        _query(),
        groups,
        ("body",),
        LotusExecutionConfig(sem_agg_dispatch="provider-batched"),
    )

    assert result == ["final alpha", "final beta"]
    assert [len(prompts) for prompts, _kwargs in model.calls] == [4, 2]


def test_provider_batched_structured_sem_agg_retries_only_invalid_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    model = _RecordingModel(
        outputs=(
            ("", '{"name":"beta","summary":"B"}'),
            ('{"name":"alpha","summary":"A"}',),
        )
    )
    monkeypatch.setattr(lotus.settings, "lm", model)

    result = execute_structured_sem_agg_groups(
        _query(structured=True),
        _groups(),
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(
            sem_agg_dispatch="provider-batched",
            structured_parse_retries=1,
        ),
    )

    assert result == [
        {"name": "alpha", "summary": "A"},
        {"name": "beta", "summary": "B"},
    ]
    assert [len(prompts) for prompts, _kwargs in model.calls] == [2, 1]
    assert model.calls[0][1]["response_format"] == {"type": "json_object"}
    assert model.calls[1][1]["response_format"] == {"type": "json_object"}


def test_provider_batched_structured_sem_agg_preserves_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    outputs = (
        '{"name":"alpha","summary":"A"}',
        '{"name":"beta","summary":"B"}',
    )
    sequential_model = _RecordingModel(outputs=((outputs[0],), (outputs[1],)))
    monkeypatch.setattr(lotus.settings, "lm", sequential_model)
    sequential = execute_structured_sem_agg_groups(
        _query(structured=True),
        _groups(),
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(),
    )

    batched_model = _RecordingModel(outputs=(outputs,))
    monkeypatch.setattr(lotus.settings, "lm", batched_model)
    batched = execute_structured_sem_agg_groups(
        _query(structured=True),
        _groups(),
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(sem_agg_dispatch="provider-batched"),
    )

    sequential_prompts = [
        prompt for prompts, _kwargs in sequential_model.calls for prompt in prompts
    ]
    batched_prompts = [
        prompt for prompts, _kwargs in batched_model.calls for prompt in prompts
    ]
    assert batched == sequential
    assert batched_prompts == sequential_prompts
    assert [len(prompts) for prompts, _kwargs in batched_model.calls] == [2]


def test_batch_prompted_sem_agg_keeps_group_outputs_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    audit_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sem_agg_module,
        "write_sem_agg_audit",
        lambda _config, **kwargs: audit_rows.append(kwargs),
    )
    raw_batch_output = (
        '{"results":['
        '{"group_id":"group_0","output":'
        '{"name":"alpha","summary":"A"}},'
        '{"group_id":"group_1","output":'
        '{"name":"beta","summary":"B"}}]}'
    )
    model = _RecordingModel(outputs=((raw_batch_output,),))
    monkeypatch.setattr(lotus.settings, "lm", model)

    result = execute_structured_sem_agg_groups(
        _query(structured=True),
        _groups(),
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=2),
        ),
    )

    assert result == [
        {"name": "alpha", "summary": "A"},
        {"name": "beta", "summary": "B"},
    ]
    assert [len(prompts) for prompts, _kwargs in model.calls] == [1]
    prompt = model.calls[0][0][0]
    assert prompt[0]["role"] == "system"
    assert "Never combine" in prompt[0]["content"]
    assert '"group_id": "group_0"' in prompt[1]["content"]
    assert '"group_id": "group_1"' in prompt[1]["content"]
    assert len(audit_rows) == 2
    assert all(row["raw_output"] == raw_batch_output for row in audit_rows)
    assert all(row["raw_output_attempts"] == (raw_batch_output,) for row in audit_rows)


def test_batch_prompted_sem_agg_accepts_one_extra_closing_brace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    raw_batch_output = (
        '{"results":['
        '{"group_id":"group_0","output":'
        '{"name":"alpha","summary":"A"}},'
        '{"group_id":"group_1","output":'
        '{"name":"beta","summary":"B"}}]}}'
    )
    model = _RecordingModel(outputs=((raw_batch_output,),))
    monkeypatch.setattr(lotus.settings, "lm", model)
    audit_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sem_agg_module,
        "write_sem_agg_audit",
        lambda _config, **kwargs: audit_rows.append(kwargs),
    )

    result = execute_structured_sem_agg_groups(
        _query(structured=True),
        _groups(),
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=2),
        ),
    )

    assert result == [
        {"name": "alpha", "summary": "A"},
        {"name": "beta", "summary": "B"},
    ]
    assert len(model.calls) == 1
    assert len(audit_rows) == 2
    assert all(
        row["syntax_repair_method"] == "json5-extra-closing-brace"
        for row in audit_rows
    )


@pytest.mark.parametrize(
    "raw_batch_output, expected_repair",
    [
        (
            '{"results": [{"group_id": "group_0", "output": '
            '{"name": "alpha", "summary": "A"}},]}',
            "json5",
        ),
        (
            "```json\n"
            '{"results": [{"group_id": "group_0", "output": '
            '{"name": "alpha", "summary": "A"}}]}\n'
            "```",
            "json5-code-fence",
        ),
    ],
)
def test_batch_prompted_sem_agg_accepts_complete_tolerant_json_syntax(
    monkeypatch: pytest.MonkeyPatch,
    raw_batch_output: str,
    expected_repair: str,
) -> None:
    import lotus
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    model = _RecordingModel(outputs=((raw_batch_output,),))
    monkeypatch.setattr(lotus.settings, "lm", model)
    audit_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sem_agg_module,
        "write_sem_agg_audit",
        lambda _config, **kwargs: audit_rows.append(kwargs),
    )

    result = execute_structured_sem_agg_groups(
        _query(structured=True),
        _groups()[:1],
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=1),
        ),
    )

    assert result == [{"name": "alpha", "summary": "A"}]
    assert audit_rows[0]["syntax_repair_method"] == expected_repair


def test_sem_agg_audit_records_syntax_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    traced_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sem_agg_module,
        "write_structured_generation_trace",
        lambda _trace_dir, *, operator, rows, snapshots: traced_rows.extend(rows),
    )

    sem_agg_module.write_sem_agg_audit(
        LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=4),
        ),
        query=_query(structured=True),
        group=_groups()[0],
        input_cols=("body",),
        output_cols=(ColumnSpec("name"), ColumnSpec("summary")),
        group_index=0,
        raw_output='{"results": []}}',
        parsed_output={"name": "alpha", "summary": "A"},
        syntax_repair_method="json5-extra-closing-brace",
    )

    assert traced_rows[0]["structured_output_repaired"] is True
    assert (
        traced_rows[0]["structured_output_repair_method"]
        == "json5-extra-closing-brace"
    )


def test_batch_prompted_sem_agg_rejects_more_than_one_extra_closing_brace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    model = _RecordingModel(
        outputs=(
            (
                '{"results":[{"group_id":"group_0","output":'
                '{"name":"alpha","summary":"A"}}]}}}',
            ),
        )
    )
    monkeypatch.setattr(lotus.settings, "lm", model)

    with pytest.raises(ValueError, match="invalid JSON"):
        execute_structured_sem_agg_groups(
            _query(structured=True),
            _groups()[:1],
            ("body",),
            (ColumnSpec("name"), ColumnSpec("summary")),
            LotusExecutionConfig(
                prompt_batching=PromptBatching(max_tasks=1),
                structured_parse_retries=0,
            ),
        )


@pytest.mark.parametrize(
    "raw_batch_output",
    [
        (
            '{"results":[{"group_id":"group_0","group_id":"group_0",'
            '"output":{"name":"alpha","summary":"A"}}]}'
        ),
        (
            '{"results":[{"group_id":"group_0","output":'
            '{"name":"alpha","summary":"truncated}}]}'
        ),
    ],
)
def test_batch_prompted_sem_agg_rejects_ambiguous_json_repair(
    monkeypatch: pytest.MonkeyPatch,
    raw_batch_output: str,
) -> None:
    import lotus

    model = _RecordingModel(outputs=((raw_batch_output,),))
    monkeypatch.setattr(lotus.settings, "lm", model)

    with pytest.raises(ValueError, match="invalid JSON"):
        execute_structured_sem_agg_groups(
            _query(structured=True),
            _groups()[:1],
            ("body",),
            (ColumnSpec("name"), ColumnSpec("summary")),
            LotusExecutionConfig(
                prompt_batching=PromptBatching(max_tasks=1),
                structured_parse_retries=0,
            ),
        )


def test_batch_prompted_native_sem_agg_audits_actual_shared_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus
    import agent_memory.adapters.lotus.sem_agg as sem_agg_module

    raw_batch_output = (
        '{"results":['
        '{"group_id":"group_0","output":{"summary":"A"}},'
        '{"group_id":"group_1","output":{"summary":"B"}}]}'
    )
    model = _RecordingModel(outputs=((raw_batch_output,),))
    monkeypatch.setattr(lotus.settings, "lm", model)
    audit_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sem_agg_module,
        "write_sem_agg_audit",
        lambda _config, **kwargs: audit_rows.append(kwargs),
    )

    result = execute_native_sem_agg_groups(
        _query(),
        _groups(),
        ("body",),
        LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=2),
        ),
    )

    assert result == ["A", "B"]
    assert len(audit_rows) == 2
    assert all(row["raw_output"] == raw_batch_output for row in audit_rows)


def test_batch_prompted_sem_agg_retries_only_invalid_prompt_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    groups = [
        pd.DataFrame({"body": ["alpha"]}),
        pd.DataFrame({"body": ["beta"]}),
        pd.DataFrame({"body": ["gamma"]}),
    ]
    model = _RecordingModel(
        outputs=(
            (
                "",
                (
                    '{"results":[{"group_id":"group_2","output":'
                    '{"name":"gamma","summary":"C"}}]}'
                ),
            ),
            (
                (
                    '{"results":['
                    '{"group_id":"group_0","output":'
                    '{"name":"alpha","summary":"A"}},'
                    '{"group_id":"group_1","output":'
                    '{"name":"beta","summary":"B"}}]}'
                ),
            ),
        )
    )
    monkeypatch.setattr(lotus.settings, "lm", model)

    result = execute_structured_sem_agg_groups(
        _query(structured=True),
        groups,
        ("body",),
        (ColumnSpec("name"), ColumnSpec("summary")),
        LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=2),
            structured_parse_retries=1,
        ),
    )

    assert [values["name"] for values in result] == ["alpha", "beta", "gamma"]
    assert [len(prompts) for prompts, _kwargs in model.calls] == [2, 1]


def test_batch_prompted_sem_agg_rejects_oversized_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    class TinyContextModel(_RecordingModel):
        max_ctx_len = 20
        max_tokens = 5

        def count_tokens(self, _value: Any) -> int:
            return 100

    monkeypatch.setattr(lotus.settings, "lm", TinyContextModel())

    with pytest.raises(ValueError, match="does not fit the model context"):
        execute_structured_sem_agg_groups(
            _query(structured=True),
            _groups()[:1],
            ("body",),
            (ColumnSpec("name"), ColumnSpec("summary")),
            LotusExecutionConfig(
                structured_max_tokens=5,
                prompt_batching=PromptBatching(max_tasks=1),
            ),
        )


def test_mixed_aggregate_batches_only_semantic_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lotus

    model = _RecordingModel(
        outputs=(
            (
                '{"name":"alpha","summary":"A"}',
                '{"name":"beta","summary":"B"}',
            ),
        )
    )
    monkeypatch.setattr(lotus.settings, "lm", model)
    source = pd.DataFrame(
        {
            "partition": ["a", "a", "b"],
            "body": ["alpha one", "alpha two", "beta"],
            "evidence": [1, 2, 3],
        }
    )
    source.attrs["agent_memory_groupby_keys"] = ("partition",)
    group_query = QueryExpr(op="group_by", params={"keys": ("partition",)})
    query = QueryExpr(
        op="agg",
        inputs=(group_query,),
        params={
            "aggregates": (
                SemanticAggregateSpec(
                    input_cols=("body",),
                    output_cols=(ColumnSpec("name"), ColumnSpec("summary")),
                    instruction="Summarize {body}.",
                ),
                ArrayAggregateSpec(
                    columns=("evidence",),
                    output_col="evidence_rows",
                ),
            )
        },
    )

    result = execute_agg(
        query,
        {},
        lambda _query, _inputs: source,
        SimpleNamespace(
            config=LotusExecutionConfig(sem_agg_dispatch="provider-batched")
        ),
    )

    assert result[["partition", "name", "summary"]].to_dict(orient="records") == [
        {"partition": "a", "name": "alpha", "summary": "A"},
        {"partition": "b", "name": "beta", "summary": "B"},
    ]
    assert [len(prompts) for prompts, _kwargs in model.calls] == [2]
    assert all(result["evidence_rows"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"sem_agg_dispatch": "parallel"}, "sem_agg_dispatch"),
        (
            {
                "sem_agg_dispatch": "provider-batched",
                "prompt_batching": PromptBatching(max_tasks=4),
            },
            "cannot be combined",
        ),
        (
            {
                "sem_groupby_pair_batch_size": 4,
                "prompt_batching": PromptBatching(max_tasks=4),
            },
            "cannot be combined",
        ),
    ),
)
def test_sem_agg_physical_config_rejects_invalid_combinations(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LotusExecutionConfig(**kwargs)


def test_sem_agg_physical_methods_isolate_maintenance_identity() -> None:
    baseline = LotusAdapter().maintenance_execution_fingerprint
    provider_batched = LotusAdapter(
        config=LotusExecutionConfig(sem_agg_dispatch="provider-batched")
    ).maintenance_execution_fingerprint
    batch_prompted_two = LotusAdapter(
        config=LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=2),
        )
    ).maintenance_execution_fingerprint
    batch_prompted_four = LotusAdapter(
        config=LotusExecutionConfig(
            prompt_batching=PromptBatching(max_tasks=4),
        )
    ).maintenance_execution_fingerprint

    assert baseline == ""
    assert (
        len({baseline, provider_batched, batch_prompted_two, batch_prompted_four}) == 4
    )
