"""Aggregate-only packing composes with fusion without global batching."""
from dataclasses import replace
from collections.abc import Mapping
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_agg import execute_structured_sem_agg_groups
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from agent_memory.policy.aggregates import ArrayAggregateSpec, SemanticAggregateSpec
from agent_memory.adapters.lotus.relational import execute_agg


class Model:
    max_tokens = 512
    max_ctx_len = 32768

    def __init__(self, invalid: bool = False) -> None:
        self.calls: list[Any] = []
        self.invalid = invalid

    def count_tokens(self, value: Any) -> int:
        return 1

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        outputs = []
        for prompt in prompts:
            data = json.loads(prompt[1]["content"])
            self.calls.append(data)
            outputs.append(json.dumps({"results": [
                {"group_id": g["group_id"], "output": {"summary": str(g["documents"])}}
                for g in reversed(data["groups"][1:] if self.invalid else data["groups"])
            ]}))
        return SimpleNamespace(outputs=outputs)


def config(size: int = 4) -> LotusExecutionConfig:
    return LotusExecutionConfig(physical_fusion="zep-target-state",
        sem_agg_prompt_batching=PromptBatching(max_tasks=size), structured_parse_retries=0)


@pytest.mark.parametrize("invalid", [False, True])
def test_groups_are_packed_and_restored_without_global_batching(monkeypatch: pytest.MonkeyPatch, invalid: bool) -> None:
    import lotus
    model = Model(invalid)
    monkeypatch.setattr(lotus.settings, "lm", model)
    q = QueryExpr(op="sem_agg", params={"instruction": "Summarize {body}.",
        "input_cols": ("body",), "output_cols": (ColumnSpec("summary"),)})
    groups = [pd.DataFrame({"body": [f"value-{i}"], "provenance": ["MUST_NOT_SEND"]}) for i in range(5)]
    def execute(frames: list[pd.DataFrame]) -> list[Mapping[str, Any]]:
        return execute_structured_sem_agg_groups(q, frames, ("body",), (ColumnSpec("summary"),), config())
    if invalid:
        with pytest.raises(ValueError):
            execute(groups)
    else:
        result = execute(groups)
        assert all(f"value-{i}" in row["summary"] for i, row in enumerate(result))
        assert [len(c["groups"]) for c in model.calls] == [4, 1]
        assert "MUST_NOT_SEND" not in str(model.calls)
        assert execute([]) == []
        assert len(model.calls) == 2
    assert config().prompt_batching is None


def test_configuration_conflicts_and_execution_identity() -> None:
    c = config()
    for overrides in ({"prompt_batching": PromptBatching(max_tasks=4)},
                      {"sem_agg_dispatch": "provider-batched"}, {"sem_agg_safe_mode": True}):
        with pytest.raises(ValueError):
            replace(c, **overrides)
    identities = {LotusAdapter(config=x).maintenance_execution_fingerprint
                  for x in (LotusExecutionConfig(), replace(c, sem_agg_prompt_batching=None), c, config(2))}
    assert len(identities) == 4
    assert not LotusAdapter(config=c).supports_legacy_restore


def test_mixed_aggregate_dispatches_packing_and_preserves_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus
    model = Model()
    monkeypatch.setattr(lotus.settings, "lm", model)
    source = pd.DataFrame({"key": ["a", "a", "b"], "body": ["one", "two", "three"], "id": [1, 2, 3]})
    source.attrs["agent_memory_groupby_keys"] = ("key",)
    q = QueryExpr(op="agg", inputs=(QueryExpr(op="group_by", params={"keys": ("key",)}),),
        params={"aggregates": (
            SemanticAggregateSpec(input_cols=("body",), output_cols=(ColumnSpec("summary"),), instruction="Summarize {body}."),
            ArrayAggregateSpec(columns=("id",), output_col="provenance"),
        )})
    result = execute_agg(q, {}, lambda _q, _inputs: source, SimpleNamespace(config=config()))
    assert [len(c["groups"]) for c in model.calls] == [2]
    assert result.key.tolist() == ["a", "b"]
    assert [json.loads(v) for v in result.provenance] == [[{"id": 1}, {"id": 2}], [{"id": 3}]]
    assert "one" in result.summary.iloc[0] and "two" in result.summary.iloc[0]
    assert "three" in result.summary.iloc[1]
