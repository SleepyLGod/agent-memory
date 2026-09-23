"""A-Mem style note evolution memory policy."""

from __future__ import annotations

from agent_memory.policy.aggregates import min, sem_agg
from agent_memory.api import Log, Memory
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.retrieval import (
    BFS,
    CosineSimilarity,
    RetrievalQuery,
)
from .prompts import (
    ANALYSE_CONTENT_PROMPT,
    EMBEDDING_TEXT_PROMPT,
    EVOLUTION_SYSTEM_PROMPT,
    NOTE_CONSOLIDATION_INSTRUCTION,
)


_NOTE_COLUMNS = [
    "_row_id",
    "content",
    "keywords",
    "context",
    "tags",
    "timestamp",
    "embedding_text",
]


_NOTE_STATE_COLUMNS = [
    "_row_id",
    "_add_seq",
    "content",
    "keywords",
    "context",
    "tags",
    "timestamp",
]


_NOTE_LINK_COLUMNS = ["source_note_id", "target_note_id"]

# Shared by the two retrieval channels (base similarity + BFS neighbour expansion):
# record_id/rank/score are engine-supplied, content carries the note payload.
_RETRIEVAL_COLUMNS = ["record_id", "content", "rank", "score"]


class AMem(Memory):
    """
    A-Mem style note evolution memory policy.
    """

    # Input log

    log = Log(
        {
            "content": "The main text content of the memory.",
            "timestamp": "Creation time in format YYYYMMDDHHMM.",
        },
        system_columns=True,
    )

    # Analysis

    # Each log entry enriched with keywords, context, and tags through llm (§3.1 eq. 2).
    _analysed_notes = log.sem_map(
        input_cols=["content"],
        output_cols={
            "analysis": (
                "The structured content analysis as one JSON object with keys "
                "keywords, context, and tags."
            ),
        },
        instruction=ANALYSE_CONTENT_PROMPT,
    ).unnest(
        column="analysis",
        fields={
            "keywords": "keywords",
            "context": "context",
            "tags": "tags",
        },
    )

    # Note pairs

    # Instead of retrieving the top-k most relevant memories for decision-making, (§3.2 eq. 4 and 5 originally)
    # we consider all earlier notes as potential neighbors for evolution.
    _earlier_notes = _analysed_notes.alias("earlier")
    _later_notes = _analysed_notes.alias("later")
    # Each row is one ordered (earlier, later) note pair, suffixed :earlier / :later.
    _candidate_note_pairs = _earlier_notes.join(
        _later_notes,
        on=(
            _earlier_notes.col("_add_seq")
            < _later_notes.col("_add_seq")
        ),
    )

    # Evolution decision (§3.1 eq. 6 and §3.2 eq. 7)

    # Each row is one evolution decision for a later note against its earlier neighbors.
    # Two possible operations (can be none or either one or both):
    #   1. "strengthen"      -> LINK GENERATION:    emit a RELATES_TO edge to each suggested neighbor.
    #   2. "update_neighbor" -> MEMORY EVOLUTION:   rewrite the neighbor's tags/context.
    _evolution_output = _candidate_note_pairs.group_by(
        ["_row_id:later", "_add_seq:later"]
    ).agg(
        sem_agg(
            input_cols=[
                "_row_id:earlier",
                "content:earlier",
                "context:earlier",
                "keywords:earlier",
                "tags:earlier",
                "content:later",
                "context:later",
                "keywords:later",
                "tags:later",
            ],
            output_cols={
                "evolution": (
                    "The complete evolution decision as one JSON object with keys "
                    "should_evolve, actions, suggested_connections, tags_to_update, "
                    "neighbor_updates."
                ),
            },
            instruction=EVOLUTION_SYSTEM_PROMPT,
        ),
    )

    # unnest: one column per decision key (note the list-valued payload keys below).
    # should_evolve (bool)
    # actions (list of "strengthen"/"update_neighbor")
    # suggested_connections (list of note ids)
    # tags_to_update (list)
    # neighbor_updates (list of {neighbor_id, new_context, new_tags}).
    _evolution_decisions = _evolution_output.unnest(
        column="evolution",
        fields={
            "should_evolve": "should_evolve",
            "actions": "actions",
            "suggested_connections": "suggested_connections",
            "tags_to_update": "tags_to_update",
            "neighbor_updates": "neighbor_updates",
        },
    )

    # One row per action; the full decision payload repeats on every row.
    _actions = _evolution_decisions.explode(column="actions", output_col="action")

    # Link generation (§3.2)

    # Each row is one note that gets link generation, plus its tags_to_update and suggested_connections.
    _link_generation_writes = (
        _actions.filter(
            _actions.col("should_evolve") & (_actions.col("action") == "strengthen")
        )
        .assign(_row_id=_actions.col("_row_id:later"))
        .select(["_row_id", "suggested_connections", "tags_to_update"])
    )

    # Each row is one directed RELATES_TO edge (source -> target note).
    # This will be stored in graphDb, and BFS neighbour expansion traverses them.
    _note_links = (
        _link_generation_writes
        .explode(column="suggested_connections", output_col="target_note_id")
        .assign(source_note_id=_link_generation_writes.col("_row_id"))         
        .select(_NOTE_LINK_COLUMNS)
        .drop_duplicates(subset=_NOTE_LINK_COLUMNS)
    )

    # Note states


    # A note's state is assembled from one state row per state it passes through:
    #   - base state: before memory evolution
    #   - evolution state: one per later memory evolution
    # Memory evolution rewriting an earlier note introduces a circular dependency on note.
    # Note states mitigate this by turning mutation into append 
    # LLM then merges the states into one final state per note.

    # Notes with link-generation: update tags, keep keywords/context unchanged.
    _link_generation_notes = (
        _analysed_notes.join(_link_generation_writes, on="_row_id", how="inner")
        .assign(tags=_link_generation_writes.col("tags_to_update"))
        .select(_NOTE_STATE_COLUMNS)
    )

    # Notes with no link-generation: keep keywords/context/tags unchanged.
    _unchanged_notes = (
        _analysed_notes.join(_link_generation_writes, on="_row_id", how="left_anti")
        .select(_NOTE_STATE_COLUMNS)
    )

    # This form the base state before memory evolution
    _base_note_states = _link_generation_notes.concat(_unchanged_notes)

    # Memory evolution (§3.2)

    # Each row is one note that gets memory evolution, plus its new_context and new_tags.
    # _add_seq is added to guide consolidation to prefers the later rewrite.
    _memory_evolution_writes = (
        _actions.filter(
            _actions.col("should_evolve") & (_actions.col("action") == "update_neighbor")
        )
        .select(["_row_id:later", "_add_seq:later", "neighbor_updates"])
        .explode(column="neighbor_updates", output_col="neighbor_update")
        .unnest(
            column="neighbor_update",
            fields={
                "neighbor_id": "_neighbor_note_id",
                "new_context": "context",
                "new_tags": "tags",
            },
        )
        .assign(
            _row_id=_actions.col("_neighbor_note_id"),
            _add_seq=_actions.col("_add_seq:later"),
        )
        .select(["_row_id", "context", "tags", "_add_seq"])
    )

    
    _rewrites = _memory_evolution_writes.alias("rewrite")

    # Applies memory evolution's change as a state of the rewritten note.
    _memory_evolution_note_states = (
        _analysed_notes.join(_rewrites, on="_row_id")
        .assign(
            context=_rewrites.col("context"),
            tags=_rewrites.col("tags"),
            _add_seq=_rewrites.col("_add_seq"),
        )
        .select(_NOTE_STATE_COLUMNS)
    )

    # Consolidation

    # An LLM consolidates a note's successive states (each representing a change) into one note.
    note = (
        _base_note_states.concat(_memory_evolution_note_states)
        .group_by("_row_id")
        .agg(
            # These columns require llm to consolidate the successive states into one final state.
            sem_agg(
                input_cols=["context", "tags", "_add_seq"],
                output_cols={
                    "context": "Accumulated note context.",
                    "tags": "Accumulated note tags.",
                },
                instruction=NOTE_CONSOLIDATION_INSTRUCTION,
            ),
            # These columns keep constant across states.
            min(column="content", output_col="content"),
            min(column="keywords", output_col="keywords"),
            min(column="timestamp", output_col="timestamp"),
        )
        .sem_map(
            # Produce embedding text by merging content, keywords, context, tags (§3.1 eq. 3)
            input_cols=["content", "keywords", "context", "tags"],
            output_cols={
                "embedding_text": (
                    "The verbatim concatenation of content, keywords, context, and "
                    "tags, in that order, with no rephrasing. Output only the "
                    "concatenated text."
                ),
            },
            instruction=EMBEDDING_TEXT_PROMPT,
        )
        .select(_NOTE_COLUMNS)
    )

    # Retrieval (§3.4)

    # Retrieve the top-5 notes from the note table. (§3.4 eq.8 & 9 & eq.10)
    _retrieved_notes = note.search(
        UserQuery(),
        methods=[CosineSimilarity()],
        reranker=None,
        limit=5,
    ).select(_RETRIEVAL_COLUMNS)

    # Neighbor expansion via BFS (Figure 2).
    retrieval_query = RetrievalQuery(
        notes=_retrieved_notes,
        neighbors=note.search(
            UserQuery(),                                  # query text ignored by BFS
            methods=[BFS(origins=_retrieved_notes, max_depth=1)],
            reranker=None,
            limit=10,                                     # neighbor budget, independent of base k
        ).select(_RETRIEVAL_COLUMNS)
    )
