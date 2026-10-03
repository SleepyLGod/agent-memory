"""Tests for the real A-Mem LOCOMO example input boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import agent_memory as am


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "a_mem" / "e2e_demo.py"
)


def _load_example() -> ModuleType:
    assert EXAMPLE_PATH.exists(), "examples/a_mem/e2e_demo.py is required"
    spec = importlib.util.spec_from_file_location("a_mem_e2e_demo", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main_args(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        row_limit=1,
        model="test-model",
        output_dir=output_dir,
        namespace="amem-test",
    )


def _amem_rows() -> list[dict[str, object]]:
    return [
        {
            "content": "Speaker Carolinesays : Hello",
            "timestamp": "1:14 pm on 25 May, 2023",
            "source_description": "LOCOMO sample 0 session 2 turn D2:8",
        }
    ]


def _retrieval_result() -> am.RetrievalResult:
    return am.RetrievalResult(
        query="Caroline",
        channels={
            "notes": pd.DataFrame(
                [
                    {
                        "record_id": "note-1",
                        "content": "Speaker Carolinesays : Hello",
                        "rank": 1,
                        "score": 0.9,
                    },
                    {
                        "record_id": "note-3",
                        "content": "Speaker Bobsays : Later",
                        "rank": 2,
                        "score": 0.4,
                    },
                ]
            ),
            "neighbors": pd.DataFrame(
                [
                    {
                        "record_id": "note-2",
                        "content": "Speaker Bobsays : Hi",
                        "rank": 1,
                        "score": 0.5,
                    }
                ]
            ),
        },
        metrics={"notes": {}, "neighbors": {"bfs_origins": ["note-1"]}},
    )


def _memory(note_rows: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        add=Mock(),
        query=Mock(return_value=_retrieval_result()),
        _runtime=SimpleNamespace(
            _state={
                "log": pd.DataFrame(),
                "note": pd.DataFrame(note_rows),
            },
            snapshot_state=Mock(
                return_value={
                    "schema_version": 2,
                    "plan_fingerprint": "test-plan",
                    "storage_commit": None,
                }
            ),
            restore_state=Mock(),
        ),
    )


def _patch_common(
    example: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: SimpleNamespace,
    rows: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(example, "parse_args", lambda: args)
    monkeypatch.setattr(example, "require_environment", lambda *a, **k: "test-provider")
    monkeypatch.setattr(
        example,
        "selected_rows",
        lambda **kwargs: (Path("locomo.json"), rows, rows),
    )
    monkeypatch.setattr(
        example,
        "source_state",
        lambda repo: {
            "commit": "test-commit",
            "dirty": False,
            "lockfile": "uv.lock",
            "lockfile_sha256": "0" * 64,
        },
    )


def test_amem_log_row_matches_upstream_speaker_and_caption_format() -> None:
    example = _load_example()

    result = example.amem_log_row(
        {
            "message": "Caroline is researching adoption agencies.",
            "speaker": "Caroline",
            "session_id": "session_2",
            "sample_index": 0,
            "turn_id": "D2:8",
            "timestamp": "1:14 pm on 25 May, 2023",
            "blip_caption": "an adoption agency brochure",
        }
    )

    assert result == {
        "content": (
            "Speaker Carolinesays : [Image: an adoption agency brochure] "
            "Caroline is researching adoption agencies."
        ),
        "timestamp": "1:14 pm on 25 May, 2023",
    }


def test_amem_log_row_omits_an_empty_caption() -> None:
    example = _load_example()

    result = example.amem_log_row(
        {
            "message": "Hello",
            "speaker": "Bob",
            "session_id": "session_1",
            "sample_index": 0,
            "turn_id": "D1:1",
            "timestamp": "1:00 pm on 25 May, 2023",
            "blip_caption": "   ",
        }
    )

    assert result["content"] == "Speaker Bobsays : Hello"


def test_locomo_source_description_keeps_turn_identity() -> None:
    example = _load_example()

    assert (
        example.locomo_source_description(
            {"sample_index": 0, "session_id": "session_2", "turn_id": "D2:8"}
        )
        == "LOCOMO sample 0 session 2 turn D2:8"
    )


def test_policy_input_fingerprint_is_stable_and_content_sensitive() -> None:
    example = _load_example()
    rows = _amem_rows()

    digest = example.policy_input_fingerprint(rows)

    assert len(digest) == 64
    assert digest == example.policy_input_fingerprint(_amem_rows())
    changed = [dict(rows[0], content="Speaker Bobsays : Hello")]
    assert example.policy_input_fingerprint(changed) != digest


def test_e2e_requires_neo4j_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _load_example()
    monkeypatch.setattr(example, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_MEMORY_NEO4J_URI", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_NEO4J_PASSWORD", raising=False)

    with pytest.raises(SystemExit, match="AGENT_MEMORY_NEO4J_URI"):
        example.require_environment()


def test_parse_args_pins_the_exchange_and_generates_a_fresh_namespace() -> None:
    example = _load_example()

    parsed = example.parse_args(["--namespace", "amem-test"])

    assert parsed.row_limit == example.ROW_LIMIT
    assert parsed.model == "deepseek/deepseek-v4-flash"
    assert parsed.namespace == "amem-test"

    generated = example.parse_args([])
    assert generated.namespace.startswith("amem-e2e-")


def test_validate_retrieval_result_allows_an_empty_neighbors_channel() -> None:
    example = _load_example()
    result = am.RetrievalResult(
        query="Caroline",
        channels={
            "notes": pd.DataFrame(
                [
                    {
                        "record_id": "note-1",
                        "content": "Hello",
                        "rank": 1,
                        "score": 0.9,
                    }
                ]
            ),
            "neighbors": pd.DataFrame(),
        },
    )

    example.validate_retrieval_result(result)


def test_validate_retrieval_result_rejects_empty_notes_and_broken_ranks() -> None:
    example = _load_example()
    empty_notes = am.RetrievalResult(
        query="Caroline",
        channels={"notes": pd.DataFrame(), "neighbors": pd.DataFrame()},
    )
    broken_ranks = am.RetrievalResult(
        query="Caroline",
        channels={
            "notes": pd.DataFrame(
                [
                    {
                        "record_id": "note-1",
                        "content": "Hello",
                        "rank": 2,
                        "score": 0.9,
                    }
                ]
            ),
            "neighbors": pd.DataFrame(),
        },
    )
    missing_columns = am.RetrievalResult(
        query="Caroline",
        channels={
            "notes": pd.DataFrame([{"rank": 1}]),
            "neighbors": pd.DataFrame(),
        },
    )

    with pytest.raises(RuntimeError, match="channel 'notes' is empty"):
        example.validate_retrieval_result(empty_notes)
    with pytest.raises(RuntimeError, match="invalid ranks"):
        example.validate_retrieval_result(broken_ranks)
    with pytest.raises(RuntimeError, match="missing columns"):
        example.validate_retrieval_result(missing_columns)


def test_assert_same_retrieval_detects_reordering() -> None:
    example = _load_example()
    before = _retrieval_result()
    reordered = am.RetrievalResult(
        query="Caroline",
        channels={
            "notes": before.channels["notes"].iloc[::-1].reset_index(drop=True),
            "neighbors": before.channels["neighbors"],
        },
    )

    example.assert_same_retrieval(before, before)
    with pytest.raises(RuntimeError, match="retrieval order changed"):
        example.assert_same_retrieval(before, reordered)


def test_evolution_summary_reports_rewrites_without_asserting() -> None:
    example = _load_example()
    memory = _memory(
        [
            {
                "_row_id": "note-1",
                "content": "Speaker Carolinesays : Hello",
                "tags": ["adoption", "research"],
                "context": "Caroline researches adoption agencies",
            }
        ]
    )
    observed = {
        "note-1": {
            "index": 1,
            "content": "Speaker Carolinesays : Hello",
            "tags": ("research",),
            "context": "Caroline research",
        }
    }

    summary = example.evolution_summary(
        observed=observed,
        memory=memory,
        rows=[{}, {}],
        neighbors_rows=2,
    )

    assert summary["asserted"] is False
    assert summary["rewritten_count"] == 1
    assert summary["preserved_count"] == 0
    assert summary["missing_count"] == 0
    assert summary["links_created"] is True
    assert summary["rewritten"][0]["tags_changed"] is True


def test_evolution_summary_skips_notes_first_seen_at_the_last_add() -> None:
    example = _load_example()
    memory = _memory(
        [
            {
                "_row_id": "note-1",
                "content": "Speaker Carolinesays : Hello",
                "tags": ["research"],
                "context": "Caroline research",
            }
        ]
    )
    observed = {
        "note-1": {
            "index": 2,
            "content": "Speaker Carolinesays : Hello",
            "tags": ("research",),
            "context": "Caroline research",
        }
    }

    summary = example.evolution_summary(
        observed=observed,
        memory=memory,
        rows=[{}, {}],
        neighbors_rows=0,
    )

    assert summary["notes_observed"] == 1
    assert summary["rewritten_count"] == 0
    assert summary["preserved_count"] == 0
    assert summary["links_created"] is False


def test_write_state_artifacts_accepts_an_empty_state(tmp_path: Path) -> None:
    example = _load_example()
    memory = SimpleNamespace(_runtime=SimpleNamespace(_state={}))

    written = example.write_state_artifacts(memory, tmp_path)

    assert set(written) == {"state/log", "views/note"}
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

    def fail_deployment(**kwargs: object) -> object:
        raise RuntimeError("deployment binding failed")

    monkeypatch.setattr(example, "StorageDeployment", fail_deployment)

    with pytest.raises(RuntimeError, match="deployment binding failed"):
        example.create_storage_deployment("amem-test")

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
    rows = _amem_rows()
    _patch_common(example, monkeypatch, args=args, rows=rows)

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
            "AMem",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("memory setup failed")),
        )

    with pytest.raises(RuntimeError, match="setup failed"):
        example.main()

    failure = (output_dir / "diagnostics" / "failure.json").read_text(encoding="utf-8")
    assert '"type": "RuntimeError"' in failure
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
    rows = _amem_rows()
    memory = _memory(
        [
            {
                "_row_id": "note-1",
                "content": "Speaker Carolinesays : Hello",
                "tags": ["research"],
                "context": "Caroline research",
            }
        ]
    )
    _patch_common(example, monkeypatch, args=args, rows=rows)
    monkeypatch.setattr(example, "create_storage_deployment", lambda namespace: storage)
    monkeypatch.setattr(example, "LotusAdapter", lambda **kwargs: object())
    monkeypatch.setattr(example.am, "AMem", lambda **kwargs: memory)

    example.main()

    memory.add.assert_called_once_with(
        {"content": rows[0]["content"], "timestamp": rows[0]["timestamp"]}
    )
    assert [item.args for item in memory.query.call_args_list] == [
        (example.QUERY,),
        (example.QUERY,),
    ]
    # The restore reopens the namespace on a second connection, so the run closes
    # both the original and the restored connector.
    assert connector.close.call_count == 2
    assert (output_dir / "retrieval" / "summary.json").exists()
    assert (output_dir / "evolution" / "summary.json").exists()
    assert (output_dir / "manifest.json").exists()
