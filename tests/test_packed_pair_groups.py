"""Packed groups retain each pair's complete input and occurrence identity."""
from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.site_batching import PairFilterBatching, _pack_partitions
from test_site_batching import adapter, query, rows


def test_pack_preserves_groups_and_oversized_partition() -> None:
    assert _pack_partitions(((0,), (1, 2), (3, 4), (5,), tuple(range(6, 12))), 4) == (
        (0, 1, 2), (3, 4, 5), tuple(range(6, 12)))
    assert _pack_partitions((), 4) == ()


def test_packed_contexts_preserve_pair_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    import lotus
    a, _ = adapter(monkeypatch)
    site = next(iter(a.config.pair_filter_batching))
    config = PairFilterBatching(("fact_id:later_added",), PromptBatching(max_tasks=4),
                               shared_columns=("fact_later_added",), pack_small_groups=True)
    a._context.config = replace(a.config, pair_filter_batching={site: config})
    seen: list[dict[str, Any]] = []

    class Oracle:
        max_tokens = 512
        cache = None

        def __call__(self, prompts: Any, **kwargs: Any) -> Any:
            outputs = []
            for prompt in prompts:
                payload = json.loads(prompt[1]["content"])
                seen.append(payload)
                assert len(payload["rows"]) <= 4
                decisions = []
                for row in payload["rows"]:
                    common = payload.get("shared_context") or payload["contexts"][row["context_id"]]
                    assert "same old" in row["context"]
                    decisions.append({"row_id": row["row_id"], "keep": "new A" in common})
                outputs.append(json.dumps({"decisions": list(reversed(decisions))}))
            return SimpleNamespace(outputs=outputs)

    monkeypatch.setattr(lotus.settings, "lm", Oracle())
    data = rows().iloc[:3]
    result = a.execute(query(), {"pairs": data})
    pd.testing.assert_frame_equal(result, data.iloc[[0, 2]])
    assert len(seen) == 1 and len(seen[0]["contexts"]) == 2
    assert len(a.execute(query(), {"pairs": data.iloc[:0]})) == 0
    assert len(seen) == 1


def test_packing_changes_execution_identity_and_default_stays_unchanged() -> None:
    from agent_memory.adapters import LotusAdapter
    from agent_memory.adapters.lotus.context import LotusExecutionConfig
    original = PairFilterBatching(("id",), PromptBatching(max_tasks=16), shared_columns=("fact",))
    packed = replace(original, pack_small_groups=True)
    assert original.to_dict()["version"] == "pair-filter-shared-context-v1"
    assert "pack_small_groups" not in original.to_dict()
    assert LotusAdapter(config=LotusExecutionConfig(pair_filter_batching={"sem_filter:test": original})).maintenance_execution_fingerprint != LotusAdapter(config=LotusExecutionConfig(pair_filter_batching={"sem_filter:test": packed})).maintenance_execution_fingerprint
    with pytest.raises(TypeError):
        replace(original, pack_small_groups=1)
