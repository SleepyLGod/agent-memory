"""Tests for the A-Mem note-evolution memory policy."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters.lotus import LotusAdapter
from agent_memory.adapters.lotus.relational import _minimum_value
from agent_memory.adapters.lotus.sem_agg import aggregate_groups_with_keys
from agent_memory.memories.a_mem import AMem, AMEM_NEO4J_STATEMENTS
from agent_memory.memories.a_mem.prompts import (
    ANALYSE_CONTENT_PROMPT,
    EMBEDDING_TEXT_PROMPT,
    EVOLUTION_SYSTEM_PROMPT,
    NOTE_CONSOLIDATION_INSTRUCTION,
)
from agent_memory.planner import PolicyDifferentiator, RetrievalPlan
from agent_memory.planner.rules import DifferentialRules
from agent_memory.policy.aggregates import MinAggregateSpec, SemanticAggregateSpec
from agent_memory.policy.schema import output_columns
from agent_memory.runtime import MemoryRuntime


# Helpers and fake adapter

def _missing(value: object) -> bool:
    """True for None and float NaN (pandas NULL representation)."""
    return value is None or (isinstance(value, float) and pd.isna(value))


def _as_list(value: object) -> list[object]:
    """Coerce a scalar/list value into a list, treating NULL as empty."""
    if _missing(value):
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _ordered_union(*lists: list[object]) -> list[object]:
    """Order-preserving union of several lists."""
    result: list[object] = []
    for items in lists:
        for item in items:
            if item not in result:
                result.append(item)
    return result


# The deterministic analyses the fake returns for the coffee/tea corpus.
COFFEE_TEA_ANALYSES: dict[str, dict[str, object]] = {
    "I love coffee": {
        "keywords": ["coffee"],
        "context": "likes coffee",
        "tags": ["coffee"],
    },
    "I switched to tea": {
        "keywords": ["tea"],
        "context": "drinks tea",
        "tags": ["tea"],
    },
    "I only drink coffee on weekends": {
        "keywords": ["coffee", "weekend"],
        "context": "coffee on weekends",
        "tags": ["coffee", "weekend"],
    },
}


class _FakeAMemAdapter(LotusAdapter):
    """Deterministic stand-in for AMem's semantic operators.

    Deterministic operators run on the real adapter; ``sem_map`` and an ``agg``
    holding a semantic aggregate are replaced wholesale, because the engine calls
    the model inside them and exposes no narrower hook. No prompt is read.
    """

    def __init__(
        self,
        *,
        analyses: dict[str, dict[str, object]] | None = None,
        actions_for: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Configure the fake model answers.

        ``actions_for`` maps a later note's content to the actions the model
        returns; repeating ``"strengthen"`` n times emits n identical
        ``suggested_connections`` entries, which the link relation is expected to
        collapse.
        """

        super().__init__()
        self.analyses = dict(analyses or {})
        self.actions_for = dict(actions_for or {})
        self.evolution_inputs: list[dict[str, object]] = []

    def execute(self, query, inputs):
        if query.op == "sem_map":
            return self._sem_map(query, inputs)
        if query.op == "agg" and self._has_semantic_aggregate(query):
            return self._grouped_agg(query, inputs)
        return super().execute(query, inputs)

    @staticmethod
    def _has_semantic_aggregate(query) -> bool:
        return any(
            isinstance(spec, SemanticAggregateSpec)
            for spec in query.params.get("aggregates", ())
        )

    def _sem_map(self, query, inputs):
        source = self.execute(query.inputs[0], inputs)
        raw_outputs = query.params["output_cols"]
        output_names = (
            [str(name) for name in raw_outputs]
            if isinstance(raw_outputs, dict)
            else [column.name for column in raw_outputs]
        )
        result = source.copy()
        if output_names == ["analysis"]:
            # Fixture: the analysis the model would have returned for this content.
            result["analysis"] = [
                json.dumps(self.analyses[str(row["content"])])
                for _, row in source.iterrows()
            ]
        elif output_names == ["embedding_text"]:
            result["embedding_text"] = [
                self._embedding(row) for _, row in source.iterrows()
            ]
        elif output_names == ["evolution"]:
            result["evolution"] = [
                self._latest_lane(row, "evolution") for _, row in source.iterrows()
            ]
        elif set(output_names) <= {"context", "tags"}:
            for name in output_names:
                result[name] = [
                    self._merged_lane(row, name) for _, row in source.iterrows()
                ]
        else:
            raise NotImplementedError(
                f"unsupported sem_map output columns {output_names!r}"
            )
        return result

    @classmethod
    def _latest_lane(cls, row, column: str) -> object:
        """Keep the stored value: the right side of a join-map join is the view."""

        for name in (f"{column}:right", f"{column}:left"):
            if name in row and not _missing(row[name]):
                return row[name]
        return None

    @classmethod
    def _merged_lane(cls, row, column: str) -> object:
        """Combine one column across the joined sides, as the group merge does.

        Must stay idempotent: incremental maintenance re-aggregates the
        already-accumulated row.
        """

        values = [
            value
            for name in (f"{column}:right", f"{column}:left")
            if name in row
            for value in _as_list(row[name])
            if not _missing(value)
        ]
        if column == "tags":
            return _ordered_union(values)
        return "; ".join(str(value) for value in values) if values else None

    @staticmethod
    def _embedding(row) -> str:
        """Concatenate the eq.3 fields in order."""

        parts = [
            str(row.get("content", "")),
            " ".join(str(item) for item in _as_list(row.get("keywords"))),
            str(row.get("context", "")),
            " ".join(str(item) for item in _as_list(row.get("tags"))),
        ]
        return " ".join(part for part in parts if part)

    def _grouped_agg(self, query, inputs):
        """Mirror execute_agg, replacing only the per-group model call."""

        source = self.execute(query.inputs[0], inputs)
        aggregates = tuple(query.params.get("aggregates", ()))
        names = list(output_columns(query))
        if source.empty:
            return pd.DataFrame(columns=names)
        rows: list[dict[str, object]] = []
        for key_values, group in aggregate_groups_with_keys(source):
            row: dict[str, object] = dict(key_values)
            for spec in aggregates:
                if isinstance(spec, SemanticAggregateSpec):
                    for column, value in self._semantic(spec, group).items():
                        row[column] = value
                elif isinstance(spec, MinAggregateSpec):
                    row[spec.output_col] = _minimum_value(group, spec.columns)
                else:
                    raise TypeError(f"unsupported aggregate spec {type(spec).__name__}")
            rows.append(row)
        return pd.DataFrame(rows, columns=names)

    def _semantic(self, spec: SemanticAggregateSpec, group: pd.DataFrame) -> dict[str, object]:
        """Return the per-group answer for one semantic aggregate."""

        names = [column.name for column in spec.output_cols]
        if "evolution" in names:
            return {"evolution": self._evolution(group)}
        if "context" in names and "tags" in names:
            return {"context": self._merge_context(group), "tags": self._merge_tags(group)}
        raise NotImplementedError(f"unsupported semantic aggregate outputs {names!r}")

    def _evolution(self, group: pd.DataFrame) -> str:
        """Build one decision JSON, defaulting to a single neighbour update."""

        # Maintenance remerge: the group already holds computed `evolution` JSON,
        # so pass the latest through instead of recomputing from candidate pairs.
        if "evolution" in group.columns:
            values = [value for value in group["evolution"] if not _missing(value)]
            if values:
                return values[-1]
        if "_add_seq:earlier" in group.columns:
            earlier = group.sort_values("_add_seq:earlier").iloc[0]
        else:
            earlier = group.iloc[0]
        neighbor_id = str(earlier["_row_id:earlier"])
        later_content = str(group["content:later"].iloc[0])
        earliest_earlier_tags = _as_list(earlier.get("tags:earlier"))
        self.evolution_inputs.append(
            {"later": later_content, "earliest_earlier_tags": earliest_earlier_tags}
        )
        new_tags = _ordered_union(
            earliest_earlier_tags,
            _as_list(group["tags:later"].iloc[0]),
        )
        actions = self.actions_for.get(later_content, ("update_neighbor",))
        has_strengthen = "strengthen" in actions
        has_update = "update_neighbor" in actions
        return json.dumps(
            {
                "should_evolve": True,
                "actions": list(actions),
                # Repeating the action repeats the suggestion: the link relation is
                # expected to collapse the duplicates.
                "suggested_connections": [neighbor_id] * actions.count("strengthen"),
                "tags_to_update": new_tags if has_strengthen else [],
                "neighbor_updates": (
                    [{"neighbor_id": neighbor_id, "new_context": "", "new_tags": new_tags}]
                    if has_update
                    else []
                ),
            }
        )

    @staticmethod
    def _merge_tags(group: pd.DataFrame) -> list[object]:
        return _ordered_union(*(_as_list(value) for value in group["tags"]))

    @staticmethod
    def _merge_context(group: pd.DataFrame) -> object:
        parts = [str(value) for value in group["context"] if not _missing(value)]
        return "; ".join(parts) if parts else None


# Spec topology

def test_a_mem_exports_public_api() -> None:
    """AMem is part of the public built-in policy surface."""

    assert am.AMem is AMem


def test_a_mem_spec_exposes_only_note_view() -> None:
    """Only the consolidated note is public; all stages stay private."""

    spec = AMem.spec()

    assert tuple(spec.views) == ("note",)
    assert tuple(spec.private_relations) == (
        "_analysed_notes",
        "_earlier_notes",
        "_later_notes",
        "_candidate_note_pairs",
        "_evolution_output",
        "_evolution_decisions",
        "_actions",
        "_link_generation_writes",
        "_note_links",
        "_link_generation_notes",
        "_unchanged_notes",
        "_base_note_states",
        "_memory_evolution_writes",
        "_rewrites",
        "_memory_evolution_note_states",
    )
    assert tuple(spec.retrieval_queries) == ("default",)
    assert "retrieval_query" not in spec.views
    assert "_retrieved_notes" not in spec.private_relations


# Prompt pinning

def test_a_mem_analysis_and_embedding_prompts_pin_contract() -> None:
    """Analysis and embedding prompts keep their structured contracts."""

    assert "Generate a structured analysis" in ANALYSE_CONTENT_PROMPT
    assert "keywords" in ANALYSE_CONTENT_PROMPT
    assert "context" in ANALYSE_CONTENT_PROMPT
    assert "tags" in ANALYSE_CONTENT_PROMPT

    # Content analysis is not evolution: the decision keys belong to the evolution
    # prompt only (they were copy-pasted here once).
    assert "should_evolve" not in ANALYSE_CONTENT_PROMPT
    assert "actions" not in ANALYSE_CONTENT_PROMPT
    assert "suggested_connections" not in ANALYSE_CONTENT_PROMPT
    assert "tags_to_update" not in ANALYSE_CONTENT_PROMPT
    assert "neighbor_updates" not in ANALYSE_CONTENT_PROMPT

    assert "Concatenate the following fields" in EMBEDDING_TEXT_PROMPT
    assert "word-for-word" in EMBEDDING_TEXT_PROMPT
    assert "content" in EMBEDDING_TEXT_PROMPT
    assert "keywords" in EMBEDDING_TEXT_PROMPT
    assert "context" in EMBEDDING_TEXT_PROMPT
    assert "tags" in EMBEDDING_TEXT_PROMPT


def test_a_mem_evolution_and_consolidation_prompts_pin_contract() -> None:
    """Evolution and consolidation prompts keep their action/merge contracts."""

    assert "strengthen" in EVOLUTION_SYSTEM_PROMPT
    assert "update_neighbor" in EVOLUTION_SYSTEM_PROMPT
    assert "should_evolve" in EVOLUTION_SYSTEM_PROMPT
    assert "neighbor_updates" in EVOLUTION_SYSTEM_PROMPT
    assert "new_context" in EVOLUTION_SYSTEM_PROMPT
    assert "new_tags" in EVOLUTION_SYSTEM_PROMPT

    consolidation = " ".join(NOTE_CONSOLIDATION_INSTRUCTION.split())
    assert "Merge the grouped context and tags" in consolidation
    assert "Preserve every distinct fact" in consolidation
    assert "prefer the newer entry" in consolidation
    assert "larger _add_seq" in consolidation


# Retrieval shape

def test_a_mem_policy_differentiates() -> None:
    """The policy compiles through the differential planner."""

    policy = PolicyDifferentiator().differentiate(
        AMem.spec(),
        statements=AMEM_NEO4J_STATEMENTS,
    )
    assert isinstance(policy.retrieval_queries["default"], RetrievalPlan)


def test_a_mem_retrieval_is_two_channel_storage_backed_dag() -> None:
    """The two retrieval channels compile to two storage search nodes."""

    spec = AMem.spec()
    policy = PolicyDifferentiator().differentiate(
        spec, statements=AMEM_NEO4J_STATEMENTS
    )
    plan = policy.retrieval_queries["default"]
    assert isinstance(plan, RetrievalPlan)
    assert (
        len([node for node in plan.nodes.values() if node.execution_kind == "search"])
        == 2
    )
    assert policy.fingerprint == PolicyDifferentiator().differentiate(
        spec, statements=AMEM_NEO4J_STATEMENTS
    ).fingerprint


def test_a_mem_query_requires_a_search_capable_storage_backend() -> None:
    """The default retrieval needs a storage backend, not an in-memory scan."""

    memory = AMem(adapter=_FakeAMemAdapter())
    with pytest.raises(NotImplementedError, match="requires a storage backend"):
        memory.query("What does the user remember?")


# Behavioural runtime

def _note_by_content(memory: AMem, content: str) -> dict[str, object]:
    """Return the consolidated note with this exact content."""

    for record in memory._runtime._state["note"].to_dict("records"):
        if record["content"] == content:
            return record
    raise AssertionError(f"no consolidated note with content {content!r}")


def _note_link_rows(
    memory: AMem,
    adapter: _FakeAMemAdapter,
) -> list[tuple[object, object]]:
    """Recompute the RELATES_TO edges from the current log.

    ``_note_links`` is a storage sink: it becomes a node only when the policy is
    compiled with the statements, so the edges are recomputed from the log instead of
    read out of a node state.
    """

    links = adapter.execute(
        AMem.spec().private_relations["_note_links"],
        {"log": memory._runtime._state["log"]},
    )
    return [
        (row["source_note_id"], row["target_note_id"])
        for row in links.to_dict("records")
    ]


GROUPED_AGG_RULE_FAMILIES = (
    "rule-all-group",
    "rule-all-group-optimized",
    "rule-join-map",
    "rule-re-group",
)


def _maintained_memory(
    grouped_agg_rule: str,
    adapter: _FakeAMemAdapter,
    contents: tuple[str, ...],
) -> AMem:
    """Return one AMem whose adds ran under the requested grouped-aggregate rule."""

    policy = PolicyDifferentiator(
        rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
    ).differentiate(AMem.spec())
    memory = AMem(adapter=adapter)
    memory._runtime = MemoryRuntime(policy, adapter=adapter)
    for index, content in enumerate(contents, start=1):
        memory.add({"content": content, "timestamp": f"2026010{index}0000"})
    return memory


def _canonical_state_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return order-insensitive, NULL-safe records for state comparison."""

    rows: list[dict[str, object]] = []
    for record in records:
        row: dict[str, object] = {}
        for key, value in record.items():
            if _missing(value):
                row[key] = ""
            elif isinstance(value, (list, tuple)):
                row[key] = sorted(str(item) for item in value)
            else:
                row[key] = str(value)
        rows.append(row)
    return sorted(rows, key=repr)


def test_a_mem_accumulates_neighbor_updates() -> None:
    """The engine accumulates note states across adds."""

    adapter = _FakeAMemAdapter(
        analyses=COFFEE_TEA_ANALYSES
    )
    memory = AMem(adapter=adapter)

    memory.add({"content": "I love coffee", "timestamp": "202601010000"})
    assert _note_by_content(memory, "I love coffee")["tags"] == ["coffee"]

    memory.add({"content": "I switched to tea", "timestamp": "202601020000"})
    assert _note_by_content(memory, "I love coffee")["tags"] == ["coffee", "tea"]

    memory.add(
        {"content": "I only drink coffee on weekends", "timestamp": "202601030000"}
    )
    assert _note_by_content(memory, "I love coffee")["tags"] == ["coffee", "tea", "weekend"]

    assert _note_by_content(memory, "I switched to tea")["tags"] == ["tea"]
    assert _note_by_content(memory, "I only drink coffee on weekends")["tags"] == [
        "coffee",
        "weekend",
    ]

    # Frozen-analysis pin: note3's decision saw note1's *frozen* tags, not the
    # accumulated [coffee, tea] — accumulation happens in consolidation.
    weekend_input = next(
        entry
        for entry in adapter.evolution_inputs
        if entry["later"] == "I only drink coffee on weekends"
    )
    assert weekend_input["earliest_earlier_tags"] == ["coffee"]

    # min-recovery: the identity columns are identical on every state row of a note,
    # so min() carries them through unchanged.
    coffee = _note_by_content(memory, "I love coffee")
    assert coffee["keywords"] == ["coffee"]
    assert coffee["timestamp"] == "202601010000"


@pytest.mark.parametrize("grouped_agg_rule", GROUPED_AGG_RULE_FAMILIES)
def test_a_mem_incremental_note_matches_full_recompute(grouped_agg_rule: str) -> None:
    """Incremental maintenance reproduces a from-scratch recompute of note."""

    adapter = _FakeAMemAdapter(
        analyses=COFFEE_TEA_ANALYSES
    )
    try:
        memory = _maintained_memory(
            grouped_agg_rule,
            adapter,
            ("I love coffee", "I switched to tea", "I only drink coffee on weekends"),
        )
    except NotImplementedError as error:
        pytest.skip(f"{grouped_agg_rule} is not expressible here: {error}")

    def canonical(records: list[dict[str, object]]) -> list[dict[str, object]]:
        rows = []
        for record in records:
            record = dict(record)
            record["tags"] = sorted(record["tags"])
            record.pop("embedding_text", None)
            rows.append(record)
        return sorted(rows, key=lambda row: str(row["content"]))

    full = adapter.execute(
        AMem.spec().views["note"].query,
        {"log": memory._runtime._state["log"]},
    )
    incremental = memory._runtime._state["note"]
    assert canonical(full.to_dict("records")) == canonical(
        incremental.to_dict("records")
    )


@pytest.mark.parametrize("grouped_agg_rule", GROUPED_AGG_RULE_FAMILIES)
def test_a_mem_maintenance_query_ignores_an_empty_change(
    grouped_agg_rule: str,
) -> None:
    """Every generated maintenance query is a fixed point with no new rows."""

    adapter = _FakeAMemAdapter(
        analyses=COFFEE_TEA_ANALYSES,
        actions_for={"I switched to tea": ("strengthen", "update_neighbor")},
    )
    try:
        memory = _maintained_memory(
            grouped_agg_rule, adapter, ("I love coffee", "I switched to tea")
        )
    except NotImplementedError as error:
        pytest.skip(f"{grouped_agg_rule} is not expressible here: {error}")

    policy = memory._runtime.policy
    node_state = memory._runtime._engine.node_state
    checked = 0
    for node_id in policy.execution_order:
        node = policy.nodes[node_id]
        current = node_state[node_id]
        if node.maintenance_query is None or current.empty:
            continue
        inputs = dict(node_state)
        inputs[node_id] = current
        for parent_id in node.input_node_ids:
            inputs[f"{parent_id}__inserted"] = node_state[parent_id].iloc[0:0]
        maintained = adapter.execute(node.maintenance_query, inputs)
        assert _canonical_state_records(
            maintained.to_dict("records")
        ) == _canonical_state_records(current.to_dict("records")), (
            f"{node_id} ({node.execution_kind}) is not a fixed point"
        )
        checked += 1
    # Only populated nodes can be checked; an empty node has nothing to maintain.
    expected = sum(
        1
        for node_id, node in policy.nodes.items()
        if node.maintenance_query is not None and not node_state[node_id].empty
    )
    assert checked == expected


def test_a_mem_single_add_handles_strengthen_and_update_neighbor() -> None:
    """One decision fans out to both strengthen and update_neighbor."""

    adapter = _FakeAMemAdapter(
        analyses=COFFEE_TEA_ANALYSES,
        actions_for={"I switched to tea": ("strengthen", "update_neighbor")},
    )
    memory = AMem(adapter=adapter)

    memory.add({"content": "I love coffee", "timestamp": "202601010000"})
    memory.add({"content": "I switched to tea", "timestamp": "202601020000"})

    # update_neighbor: note1 was rewritten by note2.
    assert _note_by_content(memory, "I love coffee")["tags"] == ["coffee", "tea"]
    # strengthen: note2's own tags were overridden (otherwise they'd be ["tea"]).
    assert _note_by_content(memory, "I switched to tea")["tags"] == ["coffee", "tea"]


def test_a_mem_strengthen_alone_overrides_tags_and_leaves_neighbor() -> None:
    """strengthen (LINK) alone overrides the new note's tags, not the neighbor's."""

    adapter = _FakeAMemAdapter(
        analyses=COFFEE_TEA_ANALYSES,
        actions_for={"I switched to tea": ("strengthen",)},
    )
    memory = AMem(adapter=adapter)

    memory.add({"content": "I love coffee", "timestamp": "202601010000"})
    memory.add({"content": "I switched to tea", "timestamp": "202601020000"})

    # strengthen overrides note2's tags (would otherwise be ["tea"]).
    assert _note_by_content(memory, "I switched to tea")["tags"] == ["coffee", "tea"]
    # no update_neighbor: note1 is untouched.
    assert _note_by_content(memory, "I love coffee")["tags"] == ["coffee"]


def test_a_mem_strengthen_emits_one_deduplicated_note_link() -> None:
    """Repeated strengthen suggestions collapse to one RELATES_TO edge."""

    adapter = _FakeAMemAdapter(
        analyses=COFFEE_TEA_ANALYSES,
        actions_for={"I switched to tea": ("strengthen", "strengthen")},
    )
    memory = AMem(adapter=adapter)

    memory.add({"content": "I love coffee", "timestamp": "202601010000"})
    memory.add({"content": "I switched to tea", "timestamp": "202601020000"})

    coffee_id = _note_by_content(memory, "I love coffee")["_row_id"]
    tea_id = _note_by_content(memory, "I switched to tea")["_row_id"]
    # The pair is the link's identity, so a duplicate row would be rejected by the
    # storage sink's inserted-key validation.
    assert _note_link_rows(memory, adapter) == [(tea_id, coffee_id)]


def test_a_mem_evolution_state_carries_identity_and_the_writer_order() -> None:
    """A rewrite's state row is the rewritten note, stamped with the writer's _add_seq."""

    adapter = _FakeAMemAdapter(analyses=COFFEE_TEA_ANALYSES)
    memory = AMem(adapter=adapter)

    memory.add({"content": "I love coffee", "timestamp": "202601010000"})
    memory.add({"content": "I switched to tea", "timestamp": "202601020000"})

    states = adapter.execute(
        AMem.spec().private_relations["_memory_evolution_note_states"],
        {"log": memory._runtime._state["log"]},
    ).to_dict("records")

    # note2 rewrote note1: identity comes from the rewritten note, _add_seq from the
    # rewriting one, and no identity column is left NULL.
    assert [row["content"] for row in states] == ["I love coffee"]
    assert [row["_add_seq"] for row in states] == [1]
    assert not any(
        _missing(row[column])
        for row in states
        for column in ("content", "keywords", "timestamp")
    )
