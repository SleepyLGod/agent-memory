"""Contract tests for the Mem0 OSS v3-style logical policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import agent_memory as am
import pytest

from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.sem_flat_map import apply_flat_map_outputs
from agent_memory.adapters.lotus.structured import (
    escape_structured_formatter_placeholders,
)
from agent_memory.memories.mem0.policy import Mem0Memory
from agent_memory.memories.mem0.policy_enhanced import Mem0MemoryEnhanced
from agent_memory.memories.mem0.prompts import (
    MEM0_ADDITIVE_EXTRACTION_INSTRUCTION,
    MEM0_SEMANTIC_DUPLICATE_INSTRUCTION,
    MEM0_SOURCE_COMMIT,
    MEM0_SOURCE_DEDUPLICATION_EXCERPT,
    MEM0_SOURCE_DEDUPLICATION_EXCERPT_SHA256,
    MEM0_SOURCE_PROMPT_PATH,
    MEM0_SOURCE_PROMPT_SHA256,
)
from agent_memory.memories.mem0.storage import (
    MEM0_ENHANCED_QDRANT_STATEMENTS,
    MEM0_QDRANT_STATEMENTS,
)
from agent_memory.policy import Log
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.retrieval import RetrievalQuery
from agent_memory.planner import PolicyDifferentiator


class _FakeMem0Adapter(LotusAdapter):
    """Execute relational nodes with deterministic Mem0 semantic operations."""

    def __init__(
        self,
        *,
        outputs_by_content: Mapping[str, list[dict[str, str]]] | None = None,
        duplicate_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.outputs_by_content = dict(outputs_by_content or {})
        self.duplicate_pairs = set(duplicate_pairs or ())
        self.extraction_inputs: list[dict[str, Any]] = []
        self.duplicate_pair_batches: list[list[tuple[str, str]]] = []

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Return deterministic rows for Mem0 semantic test inputs."""

        if query.op == "sem_filter":
            source = self.execute(query.inputs[0], inputs)
            pairs = [
                (
                    str(row["memory:earlier"]),
                    str(row["memory:later"]),
                )
                for _, row in source.iterrows()
            ]
            self.duplicate_pair_batches.append(pairs)
            keep = [
                earlier == later or (earlier, later) in self.duplicate_pairs
                for earlier, later in pairs
            ]
            return source.loc[keep].copy()
        if query.op != "sem_flat_map":
            return super().execute(query, inputs)

        source = self.execute(query.inputs[0], inputs)
        parsed_outputs: list[list[dict[str, str]]] = []
        for _, row in source.iterrows():
            previous_messages = json.loads(row["previous_messages"])
            content = str(row["content"])
            self.extraction_inputs.append(
                {
                    "role": row["role"],
                    "content": content,
                    "observation_date": row["observation_date"],
                    "previous_messages": previous_messages,
                }
            )
            if content in self.outputs_by_content:
                parsed_outputs.append(self.outputs_by_content[content])
            elif content.startswith("skip"):
                parsed_outputs.append([])
            elif content.startswith("duplicate"):
                parsed_outputs.append(
                    [
                        {
                            "memory": "The same durable memory.",
                            "attributed_to": str(row["role"]),
                        }
                    ]
                )
            else:
                parsed_outputs.append(
                    [
                        {
                            "memory": f"Memory from {content}.",
                            "attributed_to": str(row["role"]),
                        }
                    ]
                )

        return apply_flat_map_outputs(
            source,
            parsed_outputs,
            tuple(query.params["output_cols"]),
            ordinal_col=query.params.get("ordinal_col"),
        )


def _query_ops(query: QueryExpr) -> tuple[str, ...]:
    return (query.op, *(op for item in query.inputs for op in _query_ops(item)))


def _message(index: int, *, content: str | None = None) -> dict[str, str]:
    return {
        "role": "user" if index % 2 == 0 else "assistant",
        "content": content or f"message-{index}",
        "observation_date": f"2026-07-{index + 1:02d}",
    }


def test_mem0_policy_has_one_public_view_and_named_private_stages() -> None:
    spec = Mem0Memory.spec()

    assert tuple(spec.views) == ("memories",)
    assert tuple(spec.private_relations) == (
        "_windowed_messages",
        "_extracted_memories",
        "_earlier_memories",
        "_later_memories",
        "_candidate_memory_pairs",
        "_duplicate_memory_pairs",
        "_duplicate_memory_keys",
    )
    assert tuple(spec.retrieval_queries) == ("default",)
    assert am.Mem0Memory is Mem0Memory


def test_mem0_enhanced_is_self_contained_but_maintenance_identical() -> None:
    base = Mem0Memory.spec()
    enhanced = Mem0MemoryEnhanced.spec()

    assert am.Mem0MemoryEnhanced is Mem0MemoryEnhanced
    assert enhanced.log is Mem0MemoryEnhanced.log
    assert enhanced.log is not Mem0Memory.log
    assert enhanced.log.expr == base.log.expr
    assert enhanced.private_relations == base.private_relations
    assert enhanced.views == base.views
    assert (
        PolicyDifferentiator().differentiate(enhanced).fingerprint
        == PolicyDifferentiator().differentiate(base).fingerprint
    )


def test_mem0_enhanced_declares_semantic_topk_as_its_only_retrieval() -> None:
    retrieval = Mem0MemoryEnhanced.spec().retrieval_queries["default"]

    assert isinstance(retrieval, QueryExpr)
    assert retrieval.op == "sem_topk"
    assert retrieval.params["instruction"].name == "query"
    assert retrieval.params["k"] == 20
    assert retrieval.inputs == (Mem0MemoryEnhanced.memories.expr,)
    assert not isinstance(retrieval, RetrievalQuery)


def test_mem0_storage_bindings_preserve_cross_policy_plan_identity() -> None:
    base = PolicyDifferentiator().differentiate(
        Mem0Memory.spec(),
        statements=MEM0_QDRANT_STATEMENTS,
    )
    enhanced = PolicyDifferentiator().differentiate(
        Mem0MemoryEnhanced.spec(),
        statements=MEM0_ENHANCED_QDRANT_STATEMENTS,
    )

    assert MEM0_ENHANCED_QDRANT_STATEMENTS == MEM0_QDRANT_STATEMENTS
    assert enhanced.fingerprint == base.fingerprint
    assert enhanced.sink_outputs == base.sink_outputs


def test_mem0_view_uses_native_aligned_semantic_first_earlier_dedup() -> None:
    spec = Mem0Memory.spec()
    query = spec.views["memories"].query

    assert query.op == "drop_duplicates"
    assert query.params["subset"] == ("memory",)
    assert query.inputs[0].op == "select"
    assert query.inputs[0].params["columns"] == ("memory", "attributed_to")
    anti_join = query.inputs[0].inputs[0]
    assert anti_join.op == "join"
    assert anti_join.params["how"] == "left_anti"
    assert anti_join.params["on"] == ("_row_id", "_memory_ordinal")

    extraction = spec.private_relations["_extracted_memories"]
    assert extraction.op == "sem_flat_map"
    assert extraction.params["ordinal_col"] == "_memory_ordinal"
    assert extraction.inputs[0].op == "array_agg"
    assert extraction.inputs[0].inputs[0].op == "over"
    assert extraction.inputs[0].inputs[0].params["rows"] == (-10, -1)
    assert extraction.params["input_cols"] == (
        "role",
        "content",
        "observation_date",
        "previous_messages",
    )
    candidate_pairs = spec.private_relations["_candidate_memory_pairs"]
    assert candidate_pairs.op == "join"
    assert candidate_pairs.params["how"] == "inner"
    assert candidate_pairs.inputs[0].params["name"] == "earlier"
    assert candidate_pairs.inputs[1].params["name"] == "later"

    duplicate_pairs = spec.private_relations["_duplicate_memory_pairs"]
    assert duplicate_pairs.op == "sem_filter"
    assert (
        duplicate_pairs.params["instruction"]
        == MEM0_SEMANTIC_DUPLICATE_INSTRUCTION
    )
    duplicate_keys = spec.private_relations["_duplicate_memory_keys"]
    assert duplicate_keys.op == "drop_duplicates"
    assert duplicate_keys.inputs[0].op == "select"
    assert duplicate_keys.inputs[0].params["columns"] == (
        "_row_id",
        "_memory_ordinal",
    )
    assert set(_query_ops(query)).isdisjoint(
        {"search", "sem_groupby", "sem_agg", "sem_join", "sem_topk"}
    )


def test_mem0_runtime_supplies_only_the_previous_ten_messages_as_context() -> None:
    adapter = _FakeMem0Adapter()
    memory = Mem0Memory(adapter=adapter)

    for index in range(12):
        memory.add(_message(index))

    assert len(adapter.extraction_inputs) == 12
    assert adapter.extraction_inputs[0]["previous_messages"] == []
    assert [
        item["content"]
        for item in adapter.extraction_inputs[10]["previous_messages"]
    ] == [f"message-{index}" for index in range(10)]
    assert [
        item["content"]
        for item in adapter.extraction_inputs[11]["previous_messages"]
    ] == [f"message-{index}" for index in range(1, 11)]


def test_mem0_runtime_keeps_zero_or_more_rows_and_deduplicates_by_memory() -> None:
    memory = Mem0Memory(adapter=_FakeMem0Adapter())

    memory.add(_message(0, content="duplicate first wording"))
    memory.add(_message(1, content="skip greeting"))
    memory.add(_message(2, content="duplicate second wording"))
    memory.add(_message(3, content="a distinct fact"))

    assert memory._runtime._state["memories"].to_dict("records") == [
        {
            "memory": "The same durable memory.",
            "attributed_to": "user",
        },
        {
            "memory": "Memory from a distinct fact.",
            "attributed_to": "assistant",
        },
    ]


def test_mem0_semantic_dedup_keeps_old_and_preserves_related_new_memories() -> None:
    promotion = "Marcus was promoted to Senior Engineer at Shopify."
    promotion_paraphrase = "Marcus got the Senior Engineer promotion at Shopify."
    dog = "User has a dog named Poppy, a golden retriever."
    vet = (
        "User's dog Poppy had a vet checkup and is healthy but needs "
        "to lose weight."
    )
    old_preference = "User prefers almond milk lattes."
    shifted_preference = (
        "User switched from almond milk to oat milk lattes after developing "
        "an almond sensitivity."
    )
    old_location = "User lives in Paris."
    new_location = "User now lives in London."
    outputs = {
        "promotion": [{"memory": promotion, "attributed_to": "user"}],
        "promotion paraphrase": [
            {"memory": promotion_paraphrase, "attributed_to": "assistant"}
        ],
        "dog": [{"memory": dog, "attributed_to": "user"}],
        "vet": [{"memory": vet, "attributed_to": "assistant"}],
        "old preference": [
            {"memory": old_preference, "attributed_to": "user"}
        ],
        "shifted preference": [
            {"memory": shifted_preference, "attributed_to": "user"}
        ],
        "old location": [{"memory": old_location, "attributed_to": "user"}],
        "new location": [{"memory": new_location, "attributed_to": "user"}],
    }
    adapter = _FakeMem0Adapter(
        outputs_by_content=outputs,
        duplicate_pairs={(promotion, promotion_paraphrase)},
    )
    memory = Mem0Memory(adapter=adapter)

    for index, content in enumerate(outputs):
        memory.add(_message(index, content=content))

    assert memory._runtime._state["memories"].to_dict("records") == [
        {"memory": promotion, "attributed_to": "user"},
        {"memory": dog, "attributed_to": "user"},
        {"memory": vet, "attributed_to": "assistant"},
        {"memory": old_preference, "attributed_to": "user"},
        {"memory": shifted_preference, "attributed_to": "user"},
        {"memory": old_location, "attributed_to": "user"},
        {"memory": new_location, "attributed_to": "user"},
    ]


def test_mem0_incremental_semantic_dedup_only_evaluates_new_pairs() -> None:
    adapter = _FakeMem0Adapter()
    memory = Mem0Memory(adapter=adapter)

    for index in range(3):
        memory.add(_message(index))

    non_empty_batches = [
        batch for batch in adapter.duplicate_pair_batches if batch
    ]
    assert [len(batch) for batch in non_empty_batches] == [1, 2]
    assert non_empty_batches[0] == [
        ("Memory from message-0.", "Memory from message-1.")
    ]
    assert non_empty_batches[1] == [
        ("Memory from message-0.", "Memory from message-2."),
        ("Memory from message-1.", "Memory from message-2."),
    ]


def test_mem0_incremental_result_matches_full_view_recompute() -> None:
    duplicate_pairs = {
        ("Memory from message-0.", "Memory from message-2."),
    }
    adapter = _FakeMem0Adapter(duplicate_pairs=duplicate_pairs)
    memory = Mem0Memory(adapter=adapter)
    for index in range(4):
        memory.add(_message(index))

    full_view = adapter.execute(
        Mem0Memory.spec().views["memories"].query,
        {"log": memory._runtime._state["log"]},
    )

    assert full_view.to_dict("records") == memory._runtime._state[
        "memories"
    ].to_dict("records")


def test_mem0_prompt_is_pinned_and_keeps_only_current_message_as_evidence() -> None:
    assert MEM0_SOURCE_COMMIT == "d653b63fac6c8ad0ad84aead0912b366e705d269"
    assert MEM0_SOURCE_PROMPT_PATH.endswith(":ADDITIVE_EXTRACTION_PROMPT")
    assert MEM0_SOURCE_PROMPT_SHA256 == (
        "ad19187a37813ef77ee156e714c0650e6ec749e0264bdc07d499bc9b24115155"
    )
    lowered = MEM0_ADDITIVE_EXTRACTION_INSTRUCTION.lower()
    assert "current message is the only source of new memories" in lowered
    assert "do not extract a memory solely from previous messages" in lowered
    assert "no within-response duplication" in lowered
    assert "no context contamination" in lowered
    assert "exhaustive extraction checklist" in lowered
    assert "{observation_date}" in MEM0_ADDITIVE_EXTRACTION_INSTRUCTION
    assert "linked_memory_ids" not in MEM0_ADDITIVE_EXTRACTION_INSTRUCTION
    assert "existing memories" not in lowered
    assert "recently extracted memories" not in lowered
    assert "{current_date}" not in MEM0_ADDITIVE_EXTRACTION_INSTRUCTION


def test_mem0_semantic_duplicate_prompt_is_pinned_to_native_boundary() -> None:
    assert hashlib.sha256(
        MEM0_SOURCE_DEDUPLICATION_EXCERPT.encode()
    ).hexdigest() == MEM0_SOURCE_DEDUPLICATION_EXCERPT_SHA256
    assert (
        "semantically equivalent to an Existing Memory with no meaningful "
        "new context"
    ) in MEM0_SOURCE_DEDUPLICATION_EXCERPT
    assert (
        "Only skip extraction when the specific fact or event itself is "
        "already captured"
    ) in MEM0_SOURCE_DEDUPLICATION_EXCERPT
    assert "**When in doubt, extract.**" in MEM0_SOURCE_DEDUPLICATION_EXCERPT
    assert "{memory:earlier}" in MEM0_SEMANTIC_DUPLICATE_INSTRUCTION
    assert "{memory:later}" in MEM0_SEMANTIC_DUPLICATE_INSTRUCTION
    assert "Keep this pair only when Native Mem0 would skip" in (
        MEM0_SEMANTIC_DUPLICATE_INSTRUCTION
    )
    assert "Poppy had a vet checkup" in MEM0_SEMANTIC_DUPLICATE_INSTRUCTION
    assert "Max went on a camping trip" in MEM0_SEMANTIC_DUPLICATE_INSTRUCTION


def test_mem0_prompt_formats_only_declared_input_placeholders() -> None:
    from lotus.nl_expression import nle2str

    extraction = Mem0Memory.spec().private_relations["_extracted_memories"]
    output_cols = tuple(extraction.params["output_cols"])
    instruction = escape_structured_formatter_placeholders(
        MEM0_ADDITIVE_EXTRACTION_INSTRUCTION,
        input_cols=tuple(extraction.params["input_cols"]),
        output_cols=output_cols,
    )

    formatted = nle2str(
        instruction,
        list(extraction.params["input_cols"]),
    )

    assert "Role: Role" in formatted
    assert "Content: Content" in formatted
    assert "Previous_messages" in formatted
    assert "Observation_date" in formatted


def test_drop_duplicates_subset_is_general_and_preserves_no_arg_plan_shape() -> None:
    rows = Log({"memory": "Memory text.", "source": "Source label."})

    whole_row = rows.drop_duplicates().expr
    memory_only = rows.drop_duplicates(subset=["memory"]).expr

    assert dict(whole_row.params) == {}
    assert memory_only.params["subset"] == ("memory",)

    with pytest.raises(ValueError, match="cannot be empty"):
        rows.drop_duplicates(subset=[])
    with pytest.raises(ValueError, match="must be unique"):
        rows.drop_duplicates(subset=["memory", "memory"])
    with pytest.raises(ValueError, match="not found"):
        rows.drop_duplicates(subset=["missing"])


def test_mem0_declares_native_base_cosine_retrieval_semantics() -> None:
    retrieval = Mem0Memory.spec().retrieval_queries["default"]

    assert isinstance(retrieval, RetrievalQuery)
    assert tuple(retrieval.channels) == ("memories",)
    projection = retrieval.channels["memories"]
    assert projection.op == "select"
    assert projection.params["columns"] == (
        "record_id",
        "memory",
        "attributed_to",
        "rank",
        "score",
    )
    search = projection.inputs[0]
    assert search.op == "search"
    assert search.params["limit"] == 20
    assert search.params["reranker"] is None
    assert len(search.params["methods"]) == 1
    assert search.params["methods"][0].kind == "cosine_similarity"
    assert search.params["methods"][0].params == {
        "candidate_limit": 80,
        "min_score": 0.1,
    }
    assert "sem_topk" not in _query_ops(projection)


def test_mem0_query_requires_a_search_capable_storage_backend() -> None:
    memory = Mem0Memory(adapter=_FakeMem0Adapter())

    with pytest.raises(NotImplementedError, match="requires a storage backend"):
        memory.query("What does the user remember?")
