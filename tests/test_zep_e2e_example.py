"""Tests for the real Zep LOCOMO example input boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import agent_memory as am
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


def _main_args(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        start_row=26,
        row_limit=1,
        sample_limit=1,
        model="test-model",
        output_dir=output_dir,
        namespace="zep-test",
        query="Alice",
        trace=False,
    )


def _zep_rows() -> list[dict[str, object]]:
    return [
        {
            "content": "Alice: Hello",
            "role": "Alice",
            "speaker": "Alice",
            "reference_time": "2026-01-01T12:00:00",
            "source_description": "LOCOMO sample 0 session 1",
        }
    ]


def test_locomo_reference_time_is_iso_without_inventing_a_timezone() -> None:
    example = _load_example()

    assert example.locomo_reference_time("1:56 pm on 8 May, 2023") == (
        "2023-05-08T13:56:00"
    )


def test_e2e_always_configures_neo4j_retrieval() -> None:
    example = _load_example()

    parsed = example.parse_args(
        ["--namespace", "zep-test", "--query", "Alice"]
    )

    assert not hasattr(parsed, "with_neo4j")
    assert parsed.model == "deepseek/deepseek-v4-flash"
    assert parsed.namespace == "zep-test"
    assert parsed.query == "Alice"


def test_e2e_requires_neo4j_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _load_example()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_MEMORY_NEO4J_URI", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_NEO4J_PASSWORD", raising=False)

    with pytest.raises(SystemExit, match="AGENT_MEMORY_NEO4J_URI"):
        example.require_environment()


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


def test_storage_deployment_closes_connector_when_binding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _load_example()
    connector = SimpleNamespace(close=Mock())
    monkeypatch.setenv("AGENT_MEMORY_NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("AGENT_MEMORY_NEO4J_PASSWORD", "test-password")
    monkeypatch.setattr(example, "Neo4jConnector", lambda **kwargs: connector)
    monkeypatch.setattr(
        example,
        "SentenceTransformerEmbeddingProvider",
        lambda spec: object(),
    )
    monkeypatch.setattr(
        example,
        "SentenceTransformerCrossEncoderProvider",
        lambda: object(),
    )

    def fail_deployment(**kwargs: object) -> object:
        raise RuntimeError("deployment binding failed")

    monkeypatch.setattr(example, "StorageDeployment", fail_deployment)

    with pytest.raises(RuntimeError, match="deployment binding failed"):
        example.create_storage_deployment("zep-test")

    connector.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("failure_phase", "expected_close_count"),
    [
        ("setup_storage", 0),
        ("setup_adapter", 1),
        ("setup_memory", 1),
    ],
)
def test_main_records_setup_failure_and_closes_owned_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    expected_close_count: int,
) -> None:
    example = _load_example()
    output_dir = tmp_path / failure_phase
    connector = SimpleNamespace(close=Mock())
    storage = SimpleNamespace(connector=connector)
    args = _main_args(output_dir)
    rows = _zep_rows()
    monkeypatch.setattr(example, "parse_args", lambda: args)
    monkeypatch.setattr(example, "require_environment", lambda: None)
    monkeypatch.setattr(example, "reset_structured_retry_stats", lambda: None)
    monkeypatch.setattr(
        example,
        "selected_rows",
        lambda **kwargs: (Path("locomo.json"), rows, rows),
    )

    if failure_phase == "setup_storage":
        monkeypatch.setattr(
            example,
            "create_storage_deployment",
            lambda namespace: (_ for _ in ()).throw(
                RuntimeError("storage setup failed")
            ),
        )
    else:
        monkeypatch.setattr(
            example,
            "create_storage_deployment",
            lambda namespace: storage,
        )

    if failure_phase == "setup_adapter":
        monkeypatch.setattr(
            example,
            "LotusAdapter",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("adapter setup failed")
            ),
        )
    else:
        monkeypatch.setattr(example, "LotusAdapter", lambda **kwargs: object())

    if failure_phase == "setup_memory":
        monkeypatch.setattr(
            example.am,
            "ZepMemory",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("memory setup failed")
            ),
        )

    with pytest.raises(RuntimeError, match="setup failed"):
        example.main()

    failure = (output_dir / "diagnostics" / "failure.json").read_text(
        encoding="utf-8"
    )
    assert f'"phase": "{failure_phase}"' in failure
    assert connector.close.call_count == expected_close_count


def test_main_closes_storage_once_after_retrieval_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _load_example()
    output_dir = tmp_path / "success"
    connector = SimpleNamespace(close=Mock())
    storage = SimpleNamespace(connector=connector)
    args = _main_args(output_dir)
    rows = _zep_rows()
    result = am.RetrievalResult(
        query="Alice",
        channels={
            "entities": pd.DataFrame(
                [{"record_id": "entity-1", "rank": 1, "score": 1.0}]
            ),
            "facts": pd.DataFrame(
                [{"record_id": "fact-1", "rank": 1, "score": 0.9}]
            ),
        },
        metrics={"entities": {}, "facts": {"bfs_origins": ["entity-1"]}},
    )

    memory = SimpleNamespace(
        add=Mock(),
        query=Mock(return_value=result),
        _runtime=SimpleNamespace(
            _state={
                "log": pd.DataFrame(),
                "episodes": pd.DataFrame(),
                "entities": pd.DataFrame(),
                "facts": pd.DataFrame(),
            }
        ),
    )
    zero_usage = {field: 0 for field in example.USAGE_FIELDS}
    monkeypatch.setattr(example, "parse_args", lambda: args)
    monkeypatch.setattr(example, "require_environment", lambda: None)
    monkeypatch.setattr(example, "reset_structured_retry_stats", lambda: None)
    monkeypatch.setattr(
        example,
        "selected_rows",
        lambda **kwargs: (Path("locomo.json"), rows, rows),
    )
    monkeypatch.setattr(
        example,
        "create_storage_deployment",
        lambda namespace: storage,
    )
    monkeypatch.setattr(example, "LotusAdapter", lambda **kwargs: object())
    monkeypatch.setattr(example.am, "ZepMemory", lambda **kwargs: memory)
    monkeypatch.setattr(example, "usage_snapshot", lambda: dict(zero_usage))
    monkeypatch.setattr(
        example,
        "save_and_restore_checkpoint",
        lambda *args, **kwargs: ({}, memory),
    )

    example.main()

    memory.add.assert_called_once_with(rows[0])
    assert [item.args for item in memory.query.call_args_list] == [
        ("Alice",),
        ("Alice",),
    ]
    connector.close.assert_called_once_with()
    assert (output_dir / "retrieval" / "summary.json").exists()


def test_retrieval_artifacts_preserve_channels_metrics_and_restore_order(
    tmp_path: Path,
) -> None:
    example = _load_example()
    channels = {
        "entities": pd.DataFrame(
            [{"record_id": "entity-1", "name": "Alice", "rank": 1, "score": 1.0}]
        ),
        "facts": pd.DataFrame(
            [
                {
                    "record_id": "fact-1",
                    "fact": "Alice likes tea",
                    "rank": 1,
                    "score": 0.9,
                }
            ]
        ),
    }
    result = am.RetrievalResult(
        query="Alice",
        channels=channels,
        metrics={
            "entities": {"methods": [{"kind": "bm25"}]},
            "facts": {"bfs_origins": ["entity-1"]},
        },
    )

    example.validate_retrieval_result(result)
    example.assert_same_retrieval(result, result)
    written = example.write_retrieval_artifacts(
        result,
        label="before_checkpoint",
        output_dir=tmp_path,
    )

    assert written["retrieval/before_checkpoint/entities"].exists()
    assert written["retrieval/before_checkpoint/facts"].exists()
    metrics = written["retrieval/before_checkpoint/metrics"].read_text(
        encoding="utf-8"
    )
    assert '"bfs_origins": [' in metrics
