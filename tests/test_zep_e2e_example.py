"""Tests for the real Zep LOCOMO example input boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "zep" / "e2e_demo.py"
)


def _load_example() -> ModuleType:
    assert EXAMPLE_PATH.exists(), "examples/zep/e2e_demo.py is required"
    spec = importlib.util.spec_from_file_location("zep_e2e_demo", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locomo_reference_time_is_iso_without_inventing_a_timezone() -> None:
    example = _load_example()

    assert example.locomo_reference_time("1:56 pm on 8 May, 2023") == (
        "2023-05-08T13:56:00"
    )


def test_zep_log_row_preserves_source_identity() -> None:
    example = _load_example()

    result = example.zep_log_row(
        {
            "message": "Caroline is researching adoption agencies.",
            "speaker": "Caroline",
            "session_id": "session_2",
            "turn_id": "D2:8",
            "timestamp": "1:14 pm on 25 May, 2023",
        }
    )

    assert result == {
        "content": "Caroline is researching adoption agencies.",
        "role": "Caroline",
        "speaker": "Caroline",
        "reference_time": "2023-05-25T13:14:00",
        "source_description": "LOCOMO session_2 D2:8",
    }


def test_failure_artifacts_accept_an_atomically_rolled_back_empty_state(
    tmp_path: Path,
) -> None:
    example = _load_example()
    memory = SimpleNamespace(_runtime=SimpleNamespace(_state={}))

    written = example.write_state_artifacts(memory, tmp_path)

    assert set(written) == {
        "state/log",
        "views/episodes",
        "views/entities",
        "views/facts",
        "views/communities",
    }
    assert all(path.exists() for path in written.values())
