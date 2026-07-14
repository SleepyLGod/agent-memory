"""Tests for the real Claude example's runtime-independent helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "claude"
EXAMPLE_PATH = EXAMPLE_DIR / "e2e_demo.py"
_MISSING_MODULE = object()


def _load_example() -> ModuleType:
    """Load the Claude example without leaking temporary import state."""

    original_path = sys.path.copy()
    original_analyzer = sys.modules.pop("analyze_e2e_output", _MISSING_MODULE)
    try:
        sys.path.insert(0, str(EXAMPLE_DIR))
        spec = importlib.util.spec_from_file_location("claude_e2e_demo", EXAMPLE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        if original_analyzer is _MISSING_MODULE:
            sys.modules.pop("analyze_e2e_output", None)
        else:
            sys.modules["analyze_e2e_output"] = original_analyzer


e2e_demo = _load_example()


class _ExampleAdapter:
    """Return fixed full-view results without invoking LOTUS."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def execute(self, query: Any, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.calls.append((query, inputs))
        if len(self.calls) == 1:
            return pd.DataFrame(
                [{"name": "topic", "description": "desc", "type": "project", "body": "body"}]
            )
        if len(self.calls) == 2:
            return pd.DataFrame(
                [{"catalog_title": "Topic", "name": "topic", "hook": "hook"}]
            )
        return pd.DataFrame(
            [{"name": "topic", "description": "desc", "type": "project", "body": "body"}]
        )


class _ExampleMemory:
    """Expose only the public state surface used by the example."""

    class _Runtime:
        _state = {"log": pd.DataFrame([{"message": "hello"}])}

    _runtime = _Runtime()


def test_example_loader_restores_import_state() -> None:
    original_path = sys.path.copy()
    original_analyzer = sys.modules.get("analyze_e2e_output", _MISSING_MODULE)

    loaded = _load_example()

    assert loaded.__name__ == "claude_e2e_demo"
    assert sys.path == original_path
    assert sys.modules.get("analyze_e2e_output", _MISSING_MODULE) is original_analyzer


def test_full_recompute_uses_an_explicit_adapter_not_runtime_private_state() -> None:
    adapter = _ExampleAdapter()

    state, result, metrics = e2e_demo.run_full_recompute(
        _ExampleMemory(),
        adapter,
        "What matters?",
    )

    assert tuple(state) == ("topics", "catalog")
    assert result.iloc[0]["name"] == "topic"
    assert len(metrics) == 3


def test_full_candidates_uses_an_explicit_adapter() -> None:
    adapter = _ExampleAdapter()

    result = e2e_demo.run_full_candidates(_ExampleMemory(), adapter)

    assert result.iloc[0]["name"] == "topic"
