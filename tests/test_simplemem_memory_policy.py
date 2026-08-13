"""Contract tests for the SimpleMem-style logical policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import agent_memory as am
import pytest

from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.sem_flat_map import apply_flat_map_outputs
from agent_memory.memories.simplemem import (
    SIMPLEMEM_EXTRACTION_PROMPT,
    SimpleMemMemory,
    SimpleMemMemoryEnhanced,
    WINDOW_SIZE,
    WINDOW_SLIDE,
)
from agent_memory.memories.simplemem.storage import (
    SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS,
    SIMPLEMEM_NEO4J_STATEMENTS,
)
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.retrieval import RetrievalQuery
from agent_memory.planner import PolicyDifferentiator


class _FakeSimpleMemAdapter(LotusAdapter):
    """Execute relational nodes with deterministic SimpleMem extraction."""

    def __init__(
        self,
        *,
        outputs_by_window: Mapping[str, list[dict[str, str]]] | None = None,
    ) -> None:
        super().__init__()
        self.outputs_by_window = dict(outputs_by_window or {})
        self.extraction_windows: list[list[dict[str, Any]]] = []

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        if query.op != "sem_flat_map":
            return super().execute(query, inputs)

        source = self.execute(query.inputs[0], inputs)
        parsed_outputs: list[list[dict[str, str]]] = []
        for _, row in source.iterrows():
            raw_dialogues = row["dialogues"]
            if isinstance(raw_dialogues, str):
                raw_dialogues = json.loads(raw_dialogues)
            dialogues = [
                {
                    "content": item["content"],
                    "speaker": item["speaker"],
                    "timestamp": item["timestamp"],
                }
                for item in raw_dialogues
            ]
            self.extraction_windows.append(dialogues)
            first_content = dialogues[0]["content"]
            if first_content in self.outputs_by_window:
                parsed_outputs.append(self.outputs_by_window[first_content])
            else:
                parsed_outputs.append(
                    [
                        {
                            "lossless_restatement": (
                                f"Fact extracted from {first_content}."
                            ),
                            "keywords": "keyword",
                            "timestamp": "2026-08-01T00:00:00",
                            "location": "",
                            "persons": "",
                            "entities": "",
                            "topic": "topic",
                        }
                    ]
                )

        return apply_flat_map_outputs(
            source,
            parsed_outputs,
            tuple(query.params["output_cols"]),
            ordinal_col=query.params.get("ordinal_col"),
        )


def _window_ops(query: QueryExpr) -> tuple[str, ...]:
    return (query.op, *(op for item in query.inputs for op in _window_ops(item)))


def _dialogue(index: int, *, content: str | None = None) -> dict[str, str]:
    return {
        "content": content or f"dialogue-{index}",
        "speaker": "user" if index % 2 == 0 else "assistant",
        "timestamp": f"2026-07-{index + 1:02d}",
    }


def test_simplemem_has_one_public_view_and_retrieval_root() -> None:
    spec = SimpleMemMemory.spec()

    assert tuple(spec.views) == ("facts",)
    assert tuple(spec.retrieval_queries) == ("default",)
    assert am.SimpleMemMemory is SimpleMemMemory


def test_simplemem_window_constants_match_native_pipeline() -> None:
    assert WINDOW_SIZE == 40
    assert WINDOW_SLIDE == 38


def test_simplemem_extraction_prompt_keeps_native_sections() -> None:
    prompt = SIMPLEMEM_EXTRACTION_PROMPT.lower()
    assert "you are a professional information extraction assistant" in prompt
    assert "complete coverage" in prompt
    assert "force disambiguation" in prompt
    assert "lossless information" in prompt
    assert "lossless_restatement" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "keywords" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "timestamp" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "location" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "persons" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "entities" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "topic" in SIMPLEMEM_EXTRACTION_PROMPT
    assert "return only the json object" in prompt


def test_simplemem_facts_view_dedups_on_lossless_restatement_at_view_level() -> None:
    spec = SimpleMemMemory.spec()
    query = spec.views["facts"].query

    assert query.op == "drop_duplicates"
    assert query.params["subset"] == ("lossless_restatement",)
    process_window = query.inputs[0]
    assert process_window.op == "process_window"
    assert process_window.inputs[0].op == "count_window"
    assert process_window.inputs[0].params["size"] == WINDOW_SIZE
    assert process_window.inputs[0].params["slide"] == WINDOW_SLIDE


def test_simplemem_process_window_extracts_native_field_schema() -> None:
    spec = SimpleMemMemory.spec()
    process_window = spec.views["facts"].query.inputs[0]
    process_query = process_window.inputs[1]

    assert process_query.op == "select"
    assert process_query.params["columns"] == (
        "lossless_restatement",
        "keywords",
        "timestamp",
        "location",
        "persons",
        "entities",
        "topic",
    )
    extraction = process_query.inputs[0]
    assert extraction.op == "sem_flat_map"
    assert extraction.params["input_cols"] == ("dialogues",)
    assert [spec.name for spec in extraction.params["output_cols"]] == [
        "lossless_restatement",
        "keywords",
        "timestamp",
        "location",
        "persons",
        "entities",
        "topic",
    ]
    assert extraction.params["instruction"] == SIMPLEMEM_EXTRACTION_PROMPT


def test_simplemem_retrieval_declares_native_cosine_semantics() -> None:
    retrieval = SimpleMemMemory.spec().retrieval_queries["default"]

    assert isinstance(retrieval, RetrievalQuery)
    assert tuple(retrieval.channels) == ("facts",)
    projection = retrieval.channels["facts"]
    assert projection.op == "select"
    assert projection.params["columns"] == (
        "record_id",
        "lossless_restatement",
        "keywords",
        "timestamp",
        "location",
        "persons",
        "entities",
        "topic",
        "rank",
        "score",
    )
    search = projection.inputs[0]
    assert search.op == "search"
    assert search.params["limit"] == 25
    assert search.params["reranker"] is None
    assert len(search.params["methods"]) == 1
    assert search.params["methods"][0].kind == "cosine_similarity"
    assert search.params["methods"][0].params == {
        "candidate_limit": 80,
        "min_score": 0.1,
    }


def test_simplemem_storage_binds_facts_to_neo4j() -> None:
    plan = PolicyDifferentiator().differentiate(
        SimpleMemMemory.spec(),
        statements=SIMPLEMEM_NEO4J_STATEMENTS,
    )

    assert list(plan.retrieval_queries) == ["default"]
    assert len(SIMPLEMEM_NEO4J_STATEMENTS.statements) == 1
    statement = SIMPLEMEM_NEO4J_STATEMENTS.statements[0]
    assert statement.target.connector == "neo4j"
    assert statement.target.mapping.label == "Fact"
    assert statement.target.mapping.identity.columns == ("lossless_restatement",)
    assert statement.target.mapping.embedding.source_column == (
        "lossless_restatement"
    )


def test_simplemem_enhanced_is_self_contained_but_maintenance_identical() -> None:
    base = SimpleMemMemory.spec()
    enhanced = SimpleMemMemoryEnhanced.spec()

    assert am.SimpleMemMemoryEnhanced is SimpleMemMemoryEnhanced
    assert enhanced.log is SimpleMemMemoryEnhanced.log
    assert enhanced.log is not SimpleMemMemory.log
    assert enhanced.log.expr == base.log.expr
    assert enhanced.views == base.views
    assert (
        PolicyDifferentiator().differentiate(enhanced).fingerprint
        == PolicyDifferentiator().differentiate(base).fingerprint
    )


def test_simplemem_enhanced_declares_hybrid_search_retrieval() -> None:
    retrieval = SimpleMemMemoryEnhanced.spec().retrieval_queries["default"]

    assert isinstance(retrieval, RetrievalQuery)
    assert tuple(retrieval.channels) == ("facts",)
    projection = retrieval.channels["facts"]
    assert projection.op == "select"
    assert projection.params["columns"] == (
        "record_id",
        "lossless_restatement",
        "keywords",
        "timestamp",
        "location",
        "persons",
        "entities",
        "topic",
        "rank",
        "score",
    )
    search = projection.inputs[0]
    assert search.op == "search"
    assert search.params["limit"] == 25
    assert search.params["reranker"].kind == "rrf"
    assert [method.kind for method in search.params["methods"]] == [
        "bm25",
        "cosine_similarity",
    ]
    cosine = search.params["methods"][1]
    assert cosine.params == {"candidate_limit": 80, "min_score": 0.1}


def test_simplemem_storage_bindings_preserve_cross_policy_plan_identity() -> None:
    base = PolicyDifferentiator().differentiate(
        SimpleMemMemory.spec(),
        statements=SIMPLEMEM_NEO4J_STATEMENTS,
    )
    enhanced = PolicyDifferentiator().differentiate(
        SimpleMemMemoryEnhanced.spec(),
        statements=SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS,
    )

    assert SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS == SIMPLEMEM_NEO4J_STATEMENTS
    assert enhanced.fingerprint == base.fingerprint
    assert enhanced.sink_outputs == base.sink_outputs


def test_simplemem_runtime_builds_windows_and_extracts_facts() -> None:
    adapter = _FakeSimpleMemAdapter()
    memory = SimpleMemMemory(adapter=adapter)

    for index in range(WINDOW_SIZE + WINDOW_SLIDE):
        memory.add(_dialogue(index))

    assert len(adapter.extraction_windows) == 2
    assert len(adapter.extraction_windows[0]) == WINDOW_SIZE
    assert [item["content"] for item in adapter.extraction_windows[1][:2]] == [
        f"dialogue-{index}" for index in (WINDOW_SIZE - 2, WINDOW_SIZE - 1)
    ]
    facts = memory._runtime._state["facts"]
    assert len(facts) == 2
    assert facts["lossless_restatement"].tolist() == [
        "Fact extracted from dialogue-0.",
        "Fact extracted from dialogue-38.",
    ]


def test_simplemem_view_level_dedup_collapses_overlapping_window_duplicates() -> None:
    adapter = _FakeSimpleMemAdapter(
        outputs_by_window={
            "dialogue-0": [
                {
                    "lossless_restatement": "Shared overlap fact.",
                    "keywords": "k",
                    "timestamp": "2026-08-01",
                    "location": "",
                    "persons": "",
                    "entities": "",
                    "topic": "t",
                }
            ],
            "dialogue-38": [
                {
                    "lossless_restatement": "Shared overlap fact.",
                    "keywords": "k",
                    "timestamp": "2026-08-01",
                    "location": "",
                    "persons": "",
                    "entities": "",
                    "topic": "t",
                }
            ],
        }
    )
    memory = SimpleMemMemory(adapter=adapter)

    for index in range(WINDOW_SIZE + WINDOW_SLIDE):
        memory.add(_dialogue(index))

    facts = memory._runtime._state["facts"]
    assert len(facts) == 1
    assert facts["lossless_restatement"].tolist() == ["Shared overlap fact."]


def test_simplemem_query_requires_a_search_capable_storage_backend() -> None:
    memory = SimpleMemMemory(adapter=_FakeSimpleMemAdapter())

    with pytest.raises(NotImplementedError, match="requires a storage backend"):
        memory.query("Which memories are useful for collaboration?")


def test_simplemem_checkpoint_restore_reproduces_public_state() -> None:
    memory = SimpleMemMemory(adapter=_FakeSimpleMemAdapter())

    for index in range(WINDOW_SIZE + WINDOW_SLIDE):
        memory.add(_dialogue(index))

    snapshot = memory._runtime.snapshot_state()
    assert snapshot["schema_version"] == 2

    recovered = SimpleMemMemory(adapter=_FakeSimpleMemAdapter())
    recovered._runtime.restore_state(snapshot)

    original_log = memory._runtime._state["log"]
    recovered_log = recovered._runtime._state["log"]
    assert recovered_log.reset_index(drop=True).equals(
        original_log.reset_index(drop=True)
    )

    original_facts = memory._runtime._state["facts"]
    recovered_facts = recovered._runtime._state["facts"]
    assert recovered_facts.reset_index(drop=True).equals(
        original_facts.reset_index(drop=True)
    )
    assert recovered_facts["lossless_restatement"].tolist() == [
        "Fact extracted from dialogue-0.",
        "Fact extracted from dialogue-38.",
    ]
    assert len(recovered._runtime._state["log"]) == WINDOW_SIZE + WINDOW_SLIDE
