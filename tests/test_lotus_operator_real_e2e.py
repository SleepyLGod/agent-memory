"""Gated real LOTUS operator audit tests.

These tests intentionally call real LOTUS/DeepSeek execution only when
AGENT_MEMORY_RUN_LOTUS_E2E=1 is set. They write input/output CSV files to
/private/tmp so semantic behavior can be inspected manually after a run.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
from agent_memory.datasets.locomo import ensure_locomo_dataset, load_locomo_rows
from agent_memory.logical import QueryExpr
from agent_memory.relation import Relation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
AUDIT_DIR = Path("/private/tmp/agent-memory-lotus-operator-audit/latest")
_AUDIT_DIR_CLEANED = False


def _require_real_lotus() -> None:
    """Skip unless this run is explicitly allowed to call real LOTUS APIs."""

    load_dotenv(PROJECT_ROOT / ".env")
    if os.getenv("AGENT_MEMORY_RUN_LOTUS_E2E") != "1":
        pytest.skip("set AGENT_MEMORY_RUN_LOTUS_E2E=1 to run real LOTUS audit tests")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for real LOTUS audit tests")
    _clean_output_dir_once()


def _clean_output_dir_once() -> None:
    """Remove stale CSV artifacts before a real audit run."""

    global _AUDIT_DIR_CLEANED
    if _AUDIT_DIR_CLEANED:
        return
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for path in AUDIT_DIR.glob("*.csv"):
        path.unlink()
    _AUDIT_DIR_CLEANED = True


def _view(name: str) -> Relation:
    """Create a materialized-view relation used as an adapter input binding."""

    return Relation(QueryExpr(op="materialized_view", params={"name": name}))


def _audit_name(name: str) -> str:
    """Return a filesystem-safe audit artifact stem."""

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _write_audit(
    operator: str,
    inputs: Mapping[str, pd.DataFrame],
    output: pd.DataFrame,
) -> None:
    """Write CSV audit artifacts and print a concise inspection summary."""

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _audit_name(operator)
    input_paths: dict[str, Path] = {}
    for name, frame in inputs.items():
        suffix = "input" if len(inputs) == 1 else f"input_{_audit_name(name)}"
        path = AUDIT_DIR / f"{stem}__{suffix}.csv"
        frame.to_csv(path, index=False)
        input_paths[name] = path

    output_path = AUDIT_DIR / f"{stem}__output.csv"
    output.to_csv(output_path, index=False)

    print(f"\noperator: {operator}")
    for name, path in input_paths.items():
        print(f"input[{name}]: {path}")
    print(f"output: {output_path}")
    print(f"row_count: {len(output)}")
    print(f"columns: {list(output.columns)}")
    print("assertions: passed")


def _assert_nonempty_text(frame: pd.DataFrame, column: str) -> None:
    """Assert that a semantic output column contains non-empty strings."""

    assert column in frame.columns
    assert frame[column].notna().all()
    assert frame[column].astype(str).str.strip().ne("").all()


class _DifferentialAuditMemory(am.Memory):
    """Small Claude-like view used to audit full query vs Q' execution."""

    log = am.Log({"message": "Raw memory event text."})

    topics = (
        log
        .sem_flat_map(
            output_cols={
                "name": "Short durable memory topic name.",
                "body": "One durable memory fact or summary.",
            },
            instruction="Extract zero or more durable memory facts from {message}.",
        )
        .sem_groupby(
            input_cols=["name", "body"],
            instruction="Rows describe the same durable memory topic.",
        )
        .sem_agg(
            input_cols=["name", "body"],
            output_cols={
                "name": "Canonical durable memory topic name.",
                "body": "Consolidated durable memory body.",
            },
            instruction="Merge topic candidates into canonical durable memory rows.",
        )
        .select(["name", "body"])
    )


def _locomo_audit_source(
    *,
    row_limit: int = 6,
    indices: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Load official LOCOMO rows for semantic audit tests."""

    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH)
    load_limit = max(indices) + 1 if indices else row_limit
    rows = load_locomo_rows(dataset_path, sample_limit=1, turn_limit=load_limit)
    if indices is not None:
        rows = [rows[index] for index in indices if index < len(rows)]
    if not rows:
        raise AssertionError(f"No LOCOMO rows loaded from {dataset_path}")
    source = pd.DataFrame(rows)
    return source[["speaker", "message", "session_id", "turn_id"]]


def test_relational_operators_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    left = pd.DataFrame(
        {
            "message": ["alpha", "alpha", "beta"],
            "kind": ["letter", "letter", "letter"],
        }
    )
    right = pd.DataFrame(
        {
            "message": ["beta", "gamma"],
            "kind": ["letter", "letter"],
        }
    )
    inputs = {"left": left, "right": right}
    left_rel = _view("left")
    right_rel = _view("right")

    select_result = adapter.execute(left_rel.select(["message"]).expr, inputs)
    pd.testing.assert_frame_equal(select_result, left.loc[:, ["message"]])
    _write_audit("select", {"left": left}, select_result)

    concat_result = adapter.execute(left_rel.concat(right_rel).expr, inputs)
    concat_expected = pd.concat([left, right], ignore_index=True)
    pd.testing.assert_frame_equal(concat_result, concat_expected)
    _write_audit("concat", inputs, concat_result)

    union_result = adapter.execute(left_rel.union(right_rel).expr, inputs)
    union_expected = concat_expected.drop_duplicates(ignore_index=True)
    pd.testing.assert_frame_equal(union_result, union_expected)
    _write_audit("union", inputs, union_result)

    subtract_result = adapter.execute(left_rel.subtract(right_rel).expr, inputs)
    subtract_expected = pd.DataFrame(
        {"message": ["alpha", "alpha"], "kind": ["letter", "letter"]}
    )
    pd.testing.assert_frame_equal(subtract_result, subtract_expected)
    _write_audit("subtract", inputs, subtract_result)

    dedupe_result = adapter.execute(left_rel.drop_duplicates().expr, inputs)
    dedupe_expected = left.drop_duplicates(ignore_index=True)
    pd.testing.assert_frame_equal(dedupe_result, dedupe_expected)
    _write_audit("drop_duplicates", {"left": left}, dedupe_result)


def test_sem_filter_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "message": [
                "Alice prefers concise architecture documents.",
                "green sleep quickly because table",
                "Bob plans to attend a cooking class next month.",
            ]
        }
    )
    query = _view("source").sem_filter(
        instruction="{message} is a coherent sentence containing a concrete preference, plan, or personal fact."
    )

    result = adapter.execute(query.expr, {"source": source})

    assert list(result.columns) == ["message"]
    assert 0 < len(result) <= len(source)
    _write_audit("sem_filter", {"source": source}, result)


def test_sem_map_single_output_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "message": [
                "Alice prefers concise architecture documents.",
                "Bob plans to attend a cooking class next month.",
            ]
        }
    )
    query = _view("source").sem_map(
        output_cols={"summary": "One concise memory summary."},
        instruction="Summarize the memory-worthy content in {message}.",
    )

    result = adapter.execute(query.expr, {"source": source})

    assert set(result.columns) == {"message", "summary"}
    assert len(result) == len(source)
    _assert_nonempty_text(result, "summary")
    _write_audit("sem_map_single_output", {"source": source}, result)


def test_sem_map_multi_output_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "message": [
                "Alice prefers concise architecture documents.",
                "Bob plans to attend a cooking class next month.",
            ]
        }
    )
    query = _view("source").sem_map(
        output_cols={
            "memory_type": "Short label such as preference, plan, relationship, or fact.",
            "memory_summary": "One concise memory summary.",
        },
        instruction="Classify the memory type and summarize the memory-worthy content in {message}.",
    )

    result = adapter.execute(query.expr, {"source": source})

    assert set(result.columns) == {"message", "memory_type", "memory_summary"}
    assert len(result) == len(source)
    _assert_nonempty_text(result, "memory_type")
    _assert_nonempty_text(result, "memory_summary")
    _write_audit("sem_map_multi_output", {"source": source}, result)


def test_sem_flat_map_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "message": [
                "Alice prefers concise docs and plans to visit Sarah next weekend.",
                "No durable memory here.",
            ]
        }
    )
    query = _view("source").sem_flat_map(
        output_cols={
            "memory_fact": "One atomic memory fact extracted from the message.",
            "fact_type": "Short type label for the memory fact.",
        },
        instruction="Extract zero or more durable memory facts from {message}.",
    )

    result = adapter.execute(query.expr, {"source": source})

    assert set(result.columns) == {"message", "memory_fact", "fact_type"}
    assert len(result) >= 1
    _assert_nonempty_text(result, "memory_fact")
    _assert_nonempty_text(result, "fact_type")
    _write_audit("sem_flat_map", {"source": source}, result)


def test_sem_topk_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "speaker": ["Alice", "Bob", "Caroline"],
            "memory_summary": [
                "Alice prefers concise architecture documents.",
                "Bob plans to attend a cooking class next month.",
                "Caroline felt supported by an LGBTQ support group.",
            ]
        }
    )
    query = _view("source").sem_topk(
        "Which memories are most relevant to document preferences?",
        2,
    )

    result = adapter.execute(query.expr, {"source": source})

    assert list(result.columns) == ["speaker", "memory_summary"]
    assert 0 < len(result) <= 2
    _assert_nonempty_text(result, "memory_summary")
    _write_audit("sem_topk_plain_multicolumn_naive", {"source": source}, result)

    quick_adapter = LotusAdapter(
        config=LotusExecutionConfig(sem_topk_method="quick")
    )
    quick_query = _view("source").sem_topk(
        "Which memories are most relevant to document preferences?",
        2,
    )

    quick_result = quick_adapter.execute(quick_query.expr, {"source": source})

    assert list(quick_result.columns) == ["speaker", "memory_summary"]
    assert 0 < len(quick_result) <= 2
    _assert_nonempty_text(quick_result, "memory_summary")
    _write_audit("sem_topk_plain_multicolumn_quick", {"source": source}, quick_result)

    explicit_query = _view("source").sem_topk(
        "{memory_summary} is relevant to document preferences.",
        2,
    )

    explicit_result = adapter.execute(explicit_query.expr, {"source": source})

    assert list(explicit_result.columns) == ["speaker", "memory_summary"]
    assert 0 < len(explicit_result) <= 2
    _assert_nonempty_text(explicit_result, "memory_summary")
    _write_audit("sem_topk_explicit_column_naive", {"source": source}, explicit_result)


def test_sem_join_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    left = pd.DataFrame(
        {
            "message": [
                "Alice prefers concise architecture documents.",
                "Bob plans to attend a cooking class next month.",
            ]
        }
    )
    right = pd.DataFrame({"category": ["documentation preference", "sports fandom"]})
    inputs = {"left": left, "right": right}
    left_rel = _view("left")
    right_rel = _view("right")

    inner_query = left_rel.sem_join(
        right_rel,
        instruction="{message:left} belongs to {category:right}.",
        how="inner",
    )
    inner_result = adapter.execute(inner_query.expr, inputs)
    assert set(inner_result.columns) == {"message", "category"}
    assert not inner_result.empty
    _write_audit("sem_join_inner", inputs, inner_result)

    left_query = left_rel.sem_join(
        right_rel,
        instruction="{message:left} belongs to {category:right}.",
        how="left",
    )
    left_result = adapter.execute(left_query.expr, inputs)
    assert set(left_result.columns) == {"message", "category"}
    assert len(left_result) >= len(left)
    assert left_result["message"].notna().all()
    _write_audit("sem_join_left", inputs, left_result)

    right_query = left_rel.sem_join(
        right_rel,
        instruction="{message:left} belongs to {category:right}.",
        how="right",
    )
    right_result = adapter.execute(right_query.expr, inputs)
    assert set(right_result.columns) == {"message", "category"}
    assert len(right_result) >= len(right)
    assert right_result["category"].notna().all()
    _write_audit("sem_join_right", inputs, right_result)

    outer_query = left_rel.sem_join(
        right_rel,
        instruction="{message:left} belongs to {category:right}.",
        how="outer",
    )
    outer_result = adapter.execute(outer_query.expr, inputs)
    assert set(outer_result.columns) == {"message", "category"}
    assert len(outer_result) >= max(len(left), len(right))
    _write_audit("sem_join_outer", inputs, outer_result)

    fallback_query = left_rel.sem_join(
        right_rel,
        instruction="The left row describes a memory that belongs to the right category.",
        how="inner",
    )
    fallback_result = adapter.execute(fallback_query.expr, inputs)
    assert set(fallback_result.columns) == {"message", "category"}
    assert not fallback_result.empty
    _write_audit("sem_join_fallback_format", inputs, fallback_result)


def test_sem_agg_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = _locomo_audit_source(indices=(2, 4, 6, 8, 10))
    relation = _view("source")

    single_query = relation.sem_agg(
        input_cols=["speaker", "message"],
        output_cols={"memory_summary": "Merged durable memory summary."},
        instruction="Merge the LOCOMO dialogue utterances into one concise durable memory summary.",
    )
    single_result = adapter.execute(single_query.expr, {"source": source})

    assert list(single_result.columns) == ["memory_summary"]
    assert len(single_result) == 1
    _assert_nonempty_text(single_result, "memory_summary")
    _write_audit("sem_agg_whole_single_output", {"source": source}, single_result)

    multi_query = relation.sem_agg(
        input_cols=["speaker", "message"],
        output_cols={
            "memory_topic": "Short durable memory topic.",
            "memory_summary": "Merged durable memory summary.",
        },
        instruction="Create a durable memory topic and merge the LOCOMO dialogue utterances into one concise memory summary.",
    )
    multi_result = adapter.execute(multi_query.expr, {"source": source})

    assert set(multi_result.columns) == {"memory_topic", "memory_summary"}
    assert len(multi_result) == 1
    _assert_nonempty_text(multi_result, "memory_topic")
    _assert_nonempty_text(multi_result, "memory_summary")
    _write_audit("sem_agg_whole_multi_output", {"source": source}, multi_result)


def test_sem_agg_grouped_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = _locomo_audit_source(indices=(2, 4, 6, 8, 10))
    source[GROUP_ID_COLUMN] = [0, 0, 0, 1, 1]
    relation = _view("source")
    expected_groups = int(source[GROUP_ID_COLUMN].nunique())

    single_query = relation.sem_agg(
        input_cols=["speaker", "message"],
        output_cols={"memory_summary": "Merged durable memory summary."},
        instruction="Merge each speaker's LOCOMO dialogue utterances into one concise durable memory summary.",
    )
    single_result = adapter.execute(single_query.expr, {"source": source})

    assert list(single_result.columns) == ["memory_summary"]
    assert len(single_result) == expected_groups
    _assert_nonempty_text(single_result, "memory_summary")
    _write_audit("sem_agg_grouped_single_output", {"source": source}, single_result)

    multi_query = relation.sem_agg(
        input_cols=["speaker", "message"],
        output_cols={
            "memory_topic": "Short durable memory topic.",
            "memory_summary": "Merged durable memory summary.",
        },
        instruction="Create a durable memory topic and merge each speaker's LOCOMO dialogue utterances into one concise memory summary.",
    )
    multi_result = adapter.execute(multi_query.expr, {"source": source})

    assert set(multi_result.columns) == {"memory_topic", "memory_summary"}
    assert len(multi_result) == expected_groups
    _assert_nonempty_text(multi_result, "memory_topic")
    _assert_nonempty_text(multi_result, "memory_summary")
    _write_audit("sem_agg_grouped_multi_output", {"source": source}, multi_result)


def test_sem_groupby_sem_agg_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "name": ["docs", "documentation", "cooking"],
            "description": [
                "Preference for concise design docs.",
                "Preference for short architecture documents.",
                "Interest in cooking classes.",
            ],
            "body": [
                "Alice prefers concise design documents.",
                "Alice likes short architecture docs.",
                "Bob wants to attend a cooking class.",
            ],
        }
    )
    query = (
        _view("source")
        .sem_groupby(
            input_cols=["name", "description"],
            instruction="Rows refer to the same durable topic when their {name} and {description} describe the same memory.",
        )
        .sem_agg(
            input_cols=["name", "description", "body"],
            output_cols={"body": "Merged durable memory body."},
            instruction="Merge grouped memory candidates into one concise durable memory row.",
        )
    )

    grouped = adapter.execute(query.expr.inputs[0], {"source": source})
    _write_audit("sem_groupby_only", {"source": source}, grouped)
    result = adapter.execute(query.expr, {"source": source})

    assert list(result.columns) == ["body"]
    assert not result.empty
    _assert_nonempty_text(result, "body")
    _write_audit("sem_groupby_sem_agg", {"source": source}, result)


def test_differential_q_prime_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "message": [
                "Alice prefers concise architecture documents.",
                "Alice likes short design docs and dislikes long specifications.",
            ]
        }
    )
    spec = _DifferentialAuditMemory.spec()

    full_result = adapter.execute(spec.views["topics"].query, {"log": source})
    assert set(full_result.columns) == {"name", "body"}
    assert not full_result.empty
    _assert_nonempty_text(full_result, "body")
    _write_audit("differential_q_prime_full_query", {"log": source}, full_result)

    memory = _DifferentialAuditMemory(adapter=adapter)
    for row in source.to_dict("records"):
        memory.add(row)
    stepwise_result = memory._runtime._state["topics"]

    assert set(stepwise_result.columns) == {"name", "body"}
    assert not stepwise_result.empty
    _assert_nonempty_text(stepwise_result, "body")
    _write_audit(
        "differential_q_prime_stepwise",
        {"log": memory._runtime._state["log"]},
        stepwise_result,
    )


def test_sem_groupby_labels_real_lotus_audit() -> None:
    _require_real_lotus()
    adapter = LotusAdapter()
    source = pd.DataFrame(
        {
            "title": [
                "Distributed cache invalidation",
                "Transformer benchmark suite",
                "Museum visitor interviews",
            ],
            "abstract": [
                "A systems paper about cache consistency and database-backed infrastructure.",
                "A machine learning paper evaluating transformer models on benchmark datasets.",
                "A user study paper about how museum visitors interact with audio guides.",
            ],
        }
    )
    query = _view("source").sem_groupby(
        input_cols=["title", "abstract"],
        instruction="Assign each paper to the best matching research area.",
        labels={
            "systems": "Systems, infrastructure, distributed systems, and databases.",
            "ml": "Machine learning models, training, evaluation, and datasets.",
            "hci": "Human-computer interaction, user studies, and interaction design.",
            "other": "Papers that do not fit the other declared labels.",
        },
    )

    result = adapter.execute(query.expr, {"source": source})

    assert "_label" in result.columns
    assert GROUP_ID_COLUMN in result.columns
    assert len(result) == len(source)
    assert set(result["_label"]).issubset({"systems", "ml", "hci", "other"})
    _write_audit("sem_groupby_labels", {"source": source}, result)
