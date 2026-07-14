"""Tests for inspectable artifacts from the semantic window example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "helloworld"
    / "helloworld_window_smoke.py"
)


def _load_example() -> ModuleType:
    """Load the window example without modifying Python's import path."""

    spec = importlib.util.spec_from_file_location("window_e2e_demo", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory(view_name: str, value: str) -> SimpleNamespace:
    """Return a minimal memory carrying one public view frame."""

    return SimpleNamespace(
        _runtime=SimpleNamespace(
            _state={view_name: pd.DataFrame([{"value": value}])},
        )
    )


def test_semantic_window_smoke_writes_all_semantic_views(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    example = _load_example()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(example, "LotusAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(
        example,
        "WindowSemanticInWindowMemory",
        lambda **_kwargs: _memory("window_memories", "in-window"),
    )
    monkeypatch.setattr(
        example,
        "WindowGlobalAfterMemory",
        lambda **_kwargs: _memory("topics", "global"),
    )
    monkeypatch.setattr(
        example,
        "OverSemanticMemory",
        lambda **_kwargs: _memory("contextual_summaries", "over"),
    )
    monkeypatch.setattr(example, "_add_rows", lambda *_args: None)
    monkeypatch.setattr(example, "_print_frame", lambda *_args: None)

    example._run_semantic([], model="test-model", output_dir=tmp_path)

    assert pd.read_csv(tmp_path / "semantic_in_window.csv").iloc[0]["value"] == (
        "in-window"
    )
    assert pd.read_csv(tmp_path / "global_after_window.csv").iloc[0]["value"] == (
        "global"
    )
    assert pd.read_csv(tmp_path / "over_semantic.csv").iloc[0]["value"] == "over"
