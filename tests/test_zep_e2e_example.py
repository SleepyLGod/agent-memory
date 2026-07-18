"""Tests for the real Zep LOCOMO example input boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

from agent_memory.datasets.locomo import flatten_locomo_rows


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
            "sample_index": 0,
            "sample_id": "conv-26",
            "turn_id": "D2:8",
            "timestamp": "1:14 pm on 25 May, 2023",
            "blip_caption": "an adoption agency brochure",
        }
    )

    assert result == {
        "content": (
            "Caroline: Caroline is researching adoption agencies.\n"
            "(description of attached image: an adoption agency brochure)"
        ),
        "role": "Caroline",
        "speaker": "Caroline",
        "reference_time": "2023-05-25T13:14:00",
        "source_description": "LOCOMO sample 0 session 2",
    }
    assert example.policy_input_fingerprint([result]) == (
        "a6e8e8020d4156fac3b7f8828441305e8608e3a9d86f4a6b2d07214df0818e78"
    )


def test_locomo_rows_preserve_sample_identity_and_provenance() -> None:
    example = _load_example()
    rows = flatten_locomo_rows(
        [
            {
                "sample_id": "conv-26",
                "conversation": {
                    "session_1": [{"speaker": "Alice", "text": "First"}],
                    "session_1_date_time": "1:00 pm on 1 January, 2026",
                },
            },
            {
                "sample_id": "conv-27",
                "conversation": {
                    "session_2": [{"speaker": "Bob", "text": "Second"}],
                    "session_2_date_time": "2:00 pm on 2 January, 2026",
                },
            },
        ],
        sample_limit=2,
        turn_limit=2,
    )

    assert [(row["sample_index"], row["sample_id"]) for row in rows] == [
        (0, "conv-26"),
        (1, "conv-27"),
    ]
    assert example.zep_log_row(rows[1])["source_description"] == (
        "LOCOMO sample 1 session 2"
    )


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
    }
    assert all(path.exists() for path in written.values())
