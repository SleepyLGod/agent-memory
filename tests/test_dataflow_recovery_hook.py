"""Offline tests of public recovery and deterministic request identities."""

from typing import Any
import pandas as pd
import pytest
import agent_memory as am
from agent_memory.adapters.lotus.request_hook import next_request_batch, request_scope


def test_dataflow_snapshot_continues_count() -> None:
    source = am.Source({"x": "number"})
    flow = am.SemanticDataflow(source=source, views={"v": source.count(output_col="n")})
    flow.apply(pd.DataFrame({"x": [1, 2]}))
    state = flow.snapshot_state()
    restored = am.SemanticDataflow(
        source=source, views={"v": source.count(output_col="n")}
    )
    restored.restore_state(state)
    restored.apply(pd.DataFrame({"x": [3]}))
    assert restored.view("v").iloc[0]["n"] == 3
    wrong = am.SemanticDataflow(source=source, views={"v": source.limit(1)})
    with pytest.raises(ValueError, match="fingerprint"):
        wrong.restore_state(state)


def test_request_positions_reset_per_unit() -> None:
    class Hook:
        def execute(self, identity: Any, payload: Any, send: Any) -> dict[str, Any]:
            return send()

    hook = Hook()
    for _ in range(2):
        with request_scope(hook, mode="full", event_id="1"):
            assert next_request_batch() == (
                hook,
                {"mode": "full", "event_id": "1", "batch": 0},
            )
            assert next_request_batch() == (
                hook,
                {"mode": "full", "event_id": "1", "batch": 1},
            )
    assert next_request_batch() is None


def test_hook_refuses_unjournaled_text_fallback() -> None:
    from agent_memory.adapters.lotus.deepseek_responses_lm import (
        deepseek_responses_lm_class,
    )

    class Base:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def _process_uncached_messages(self, *args: Any) -> Any:
            pytest.fail("must not fall through to unjournaled provider")

    class Hook:
        def execute(self, *args: Any) -> Any:
            pytest.fail("unsupported request must be rejected before send")

    lm = deepseek_responses_lm_class(Base)("deepseek/deepseek-flash")
    with request_scope(Hook(), mode="full"):
        with pytest.raises(ValueError, match="no fallback"):
            lm._process_uncached_messages([], {}, False, "")


def test_compact_failure_keeps_no_per_row_files() -> None:
    from agent_memory.adapters.lotus.structured import (
        write_structured_failure_artifacts,
    )
    from agent_memory.tracing.semantic import semantic_trace_scope

    with semantic_trace_scope(provider_journal=True):
        assert (
            write_structured_failure_artifacts(
                ["prompt"] * 1000,
                [["invalid"]] * 1000,
                list(range(1000)),
                output_cols=(),
                shape="object",
                require_explanation=False,
                operator="sem_join",
            )
            == []
        )
