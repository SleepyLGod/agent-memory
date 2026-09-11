"""Provider-free tests of genuinely single-context structured predicates."""

from collections import Counter
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import (
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.predicate import (
    execute_schema_predicate,
    single_schema_predicate,
)
from agent_memory.adapters.lotus.prompt_batching import PromptBatching


class FakeLM:
    max_tokens = 128

    def __init__(self, outputs: list[list[str]] | None = None) -> None:
        self.outputs = iter(outputs) if outputs is not None else None
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        self.calls.append((prompts, kwargs))
        if self.outputs is not None:
            return SimpleNamespace(outputs=next(self.outputs))
        results = []
        for prompt in prompts:
            texts = re.findall("«(.*?)»", prompt[1]["content"], re.DOTALL)
            keep = texts[0] != texts[1] if len(texts) == 2 else texts[0] == "positive"
            results.append(json.dumps({"keep": keep}))
        return SimpleNamespace(outputs=results)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LotusExecutionConfig:
    import agent_memory.adapters.lotus.structured as structured

    monkeypatch.setattr(structured, "STRUCTURED_FAILURE_DIR", tmp_path / "failures")
    return LotusExecutionConfig(
        structured_output_transport="responses-json-schema",
        structured_parse_retries=0,
        semantic_trace_dir=tmp_path / "trace",
    )


def test_single_prompt_schema_and_failed_item_retry(
    config: LotusExecutionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lotus
    from dataclasses import replace

    lm = FakeLM([['{"keep":true}', '{"keep":"false"}'], ['{"keep":false}']])
    monkeypatch.setattr(lotus.settings, "lm", lm)
    result = execute_schema_predicate(
        pd.DataFrame({"text": ["positive", "negative"]}),
        instruction="{text} is positive.",
        config=replace(config, structured_parse_retries=1),
        operator="sem_filter",
    )
    assert result.decisions == (True, False)
    assert [len(call[0]) for call in lm.calls] == [2, 1]
    assert lm.calls[1][0] == [lm.calls[0][0][1]]
    assert config.prompt_batching is None
    for prompts, kwargs in lm.calls:
        schema = kwargs["response_format"]["json_schema"]["schema"]
        assert schema["properties"] == {"keep": {"type": "boolean"}}
        assert schema["additionalProperties"] is False
        for prompt in prompts:
            assert len(re.findall("«(.*?)»", prompt[1]["content"])) == 1
            assert all(
                word not in str(prompt)
                for word in ("row_id", "decisions", "independent row")
            )


@pytest.mark.parametrize(
    "raw",
    [
        '{"keep":1}',
        '{"keep":null}',
        '{"keep":true,"extra":0}',
        "[]",
        "garbage",
        '{"keep":"false"}',
    ],
)
def test_invalid_output_is_not_false(
    raw: str,
    config: LotusExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lotus

    monkeypatch.setattr(lotus.settings, "lm", FakeLM([[raw]]))
    with pytest.raises(ValueError, match="failed for rows"):
        execute_schema_predicate(
            pd.DataFrame({"text": ["x"]}),
            instruction="{text} matches.",
            config=config,
            operator="sem_filter",
        )
    artifacts = list((tmp_path / "failures").glob("*.json"))
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_text())["raw_outputs"] == [raw]


@pytest.mark.parametrize("outputs", [[], ['{"keep":true}', '{"keep":false}']])
def test_wrong_output_count_fails(
    outputs: list[str], config: LotusExecutionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lotus

    monkeypatch.setattr(lotus.settings, "lm", FakeLM([outputs]))
    with pytest.raises(ValueError, match="expected 1 outputs"):
        execute_schema_predicate(
            pd.DataFrame({"text": ["x"]}),
            instruction="{text} matches.",
            config=config,
            operator="sem_filter",
        )


def test_empty_and_default_paths(config: LotusExecutionConfig) -> None:
    assert (
        execute_schema_predicate(
            pd.DataFrame({"text": []}),
            instruction="{text} matches.",
            config=config,
            operator="sem_filter",
        ).decisions
        == ()
    )
    assert not single_schema_predicate(LotusExecutionConfig())
    assert not single_schema_predicate(
        LotusExecutionConfig(
            structured_output_transport="responses-json-schema",
            prompt_batching=PromptBatching(max_tasks=1),
        )
    )


@pytest.mark.parametrize(
    "option",
    [
        "sem_join_examples",
        "sem_join_strategy",
        "sem_join_cascade_args",
        "sem_join_safe_mode",
    ],
)
def test_join_rejects_unsupported_before_configure(
    option: str, config: LotusExecutionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from agent_memory.adapters.lotus.sem_join import execute_sem_join

    source = am.Source({"text": "Text."})
    query = source.sem_join(
        source, instruction="{text:left} differs from {text:right}."
    )
    context = LotusExecutionContext(
        model="deepseek/deepseek-v4-flash", config=replace(config, **{option: True})
    )
    monkeypatch.setattr(
        LotusExecutionContext, "configure", lambda _: pytest.fail("configured model")
    )
    with pytest.raises(ValueError, match="does not support"):
        execute_sem_join(
            query.expr, {}, lambda *_: pytest.fail("executed input"), context
        )


def test_real_planner_join_and_filter(
    config: LotusExecutionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lotus

    lm = FakeLM()
    monkeypatch.setattr(lotus.settings, "lm", lm)
    monkeypatch.setattr(lotus.settings, "enable_cache", False)
    monkeypatch.setattr(LotusExecutionContext, "configure", lambda _: None)
    adapter = LotusAdapter(model="deepseek/deepseek-v4-flash", config=config)
    source = am.Source({"id": "ID.", "text": "Text."})
    pairs = source.sem_join(
        source, instruction="{text:left} differs from {text:right}."
    )
    pairs = pairs.filter(pairs.col("id:left") != pairs.col("id:right"))
    pairs = pairs.select(["id:left", "id:right"])
    positive = source.sem_filter(instruction="{text} is positive.")
    flow = am.SemanticDataflow(
        source=source, views={"pairs": pairs, "positive": positive}, adapter=adapter
    )
    rows = pd.DataFrame({"id": [1, 2, 3], "text": ["positive", "negative", "positive"]})
    for i in range(len(rows)):
        flow.apply(rows.iloc[[i]])
    incremental_tasks = sum(len(p) for p, _ in lm.calls)
    full = adapter.execute(pairs.expr, {"log": rows})
    assert Counter(full.itertuples(index=False, name=None)) == Counter(
        flow.view("pairs").itertuples(index=False, name=None)
    )
    assert len(full) == 4
    assert incremental_tasks == 9 + 3
    assert sum(len(p) for p, _ in lm.calls) - incremental_tasks == 9
    assert list(flow.view("positive")["id"]) == [1, 3]
