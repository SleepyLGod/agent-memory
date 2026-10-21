"""The `sem_join` lowering: pair generation, `how`, `k`, `id_columns`, and the assembled frame.

No model: the loopback endpoint decides each pair from a marker in the request, so DocETL's `filter`
validation and rendering still run while the deterministic half is asserted on the published frame.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
import json
from pathlib import Path

import pandas as pd
import pytest

from agent_memory.adapters import DocetlAdapter
from agent_memory.adapters.docetl.context import DocetlExecutionConfig
from agent_memory.adapters.docetl.prompt import (
    build_join_prompt,
    build_topk_prompt,
    join_comparison_columns,
)
from agent_memory.adapters.docetl.sem_join import SELECTED_COLUMN
from agent_memory.api import Log, Memory
from agent_memory.policy.logical import QueryExpr

from adapter_support import (
    CannedEndpoint,
    build_adapters,
    build_docetl_adapter,
    canned_model,
)

LEFT = pd.DataFrame(
    {"note": ["alpha left", "beta left"], "dept": ["d1", "d2"], "tag": ["x", "y"]}
)
RIGHT = pd.DataFrame(
    {"note": ["alpha right", "beta right"], "dept": ["d1", "d1"], "kind": ["p", "q"]}
)


@pytest.fixture
def endpoint() -> Iterator[CannedEndpoint]:
    """Keep every pair, so the frame's shape is decided only by the deterministic half."""

    with canned_model("") as running:
        yield running


def _source(name: str, frame: pd.DataFrame) -> QueryExpr:
    return QueryExpr(
        op="materialized_view",
        params={"name": name, "columns": tuple(frame.columns)},
    )


def _join_query(**params: object) -> QueryExpr:
    return QueryExpr(
        op="sem_join",
        inputs=(_source("left", LEFT), _source("right", RIGHT)),
        params={
            "instruction": "the subjects match: {note:left} and {note:right}",
            **params,
        },
    )


def _adapter(running: CannedEndpoint) -> DocetlAdapter:
    return build_docetl_adapter(running.base_url)


def test_the_prompt_labels_each_field_and_states_the_condition() -> None:
    """The whole verdict template, in order: context lines, a blank line, condition, question."""

    prompt = build_join_prompt(
        "the subjects match: {note:left} and {note:right}",
        left_columns=("note",),
        right_columns=("note",),
    )

    assert prompt == (
        "Context:\n"
        "[left] note: «{{ input.left_0 }}»\n"
        "[right] note: «{{ input.right_0 }}»\n"
        "\n"
        "Condition: the subjects match: note and note\n"
        "Does the context satisfy this condition?"
    )
    # a placeholder must not survive into the Jinja the renderer sees
    assert "{note" not in prompt


def test_a_bare_placeholder_is_resolved_by_the_side_that_owns_it() -> None:
    left, right = join_comparison_columns(
        "{title} against {body}",
        left_columns=("title", "shared"),
        right_columns=("body", "shared"),
    )

    assert left == ("title",)
    assert right == ("body",)


def test_an_ambiguous_bare_placeholder_falls_back_to_whole_rows() -> None:
    """One name on both sides cannot say which value it means, so neither side is narrowed."""

    left, right = join_comparison_columns(
        "the {note} values match",
        left_columns=("note", "tag"),
        right_columns=("note", "kind"),
    )

    assert left == ("note", "tag")
    assert right == ("note", "kind")


def test_a_side_qualified_placeholder_must_exist_on_that_side() -> None:
    with pytest.raises(ValueError, match="does not have"):
        join_comparison_columns(
            "{note:right} matches",
            left_columns=("note",),
            right_columns=("other",),
        )


def test_an_unknown_placeholder_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown sem_join column placeholder"):
        join_comparison_columns(
            "{missing} matches",
            left_columns=("note",),
            right_columns=("note",),
        )


def test_an_unsupported_how_is_refused_before_any_call(
    endpoint: CannedEndpoint,
) -> None:
    with pytest.raises(ValueError, match="Unsupported sem_join how"):
        _adapter(endpoint).execute(_join_query(how="sideways"), {})

    assert endpoint.requests == []


def test_a_literal_parameter_is_refused_before_the_data_is_read(
    endpoint: CannedEndpoint,
) -> None:
    """`k` is a shape, not a data fact, so an empty side does not excuse it."""

    empty = pd.DataFrame({"note": [], "dept": [], "tag": []})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", empty), _source("right", RIGHT)),
        params={
            "instruction": "the subjects match: {note:left} and {note:right}",
            "k": 0,
        },
    )

    with pytest.raises(ValueError, match="k must be a positive integer"):
        _adapter(endpoint).execute(query, {"left": empty, "right": RIGHT})

    assert endpoint.requests == []


def test_inner_join_emits_one_row_per_matched_pair(
    endpoint: CannedEndpoint,
) -> None:
    """A bag, not a keyed join: two left rows times two right rows is four matched pairs."""

    result = _adapter(endpoint).execute(
        _join_query(), {"left": LEFT, "right": RIGHT}
    )

    assert list(result.columns) == ["note:left", "dept:left", "tag", "note:right", "dept:right", "kind"]
    assert len(result) == 4


def test_keyed_on_restricts_the_pairs_a_model_is_asked_about(
    endpoint: CannedEndpoint,
) -> None:
    """`on` filters candidates before the call, so two same-dept pairs cost two requests, not four."""

    result = _adapter(endpoint).execute(
        _join_query(on=("dept",)), {"left": LEFT, "right": RIGHT}
    )

    assert len(endpoint.requests) == 2
    assert result["dept:left"].tolist() == ["d1", "d1"]
    assert result["dept:right"].tolist() == ["d1", "d1"]


def test_a_predicate_on_restricts_before_the_call(endpoint: CannedEndpoint) -> None:
    """A predicate is evaluated on the frames, so only the pairs it admits reach the model."""

    from agent_memory.api import Log
    from agent_memory.policy.relation import _normalize_join_on

    left_rel = Log({"dept": "d", "note": "n"}).alias("left")
    right_rel = Log({"dept": "d", "note": "n"}).alias("right")
    on = _normalize_join_on([left_rel.col("dept") == right_rel.col("dept")])

    result = _adapter(endpoint).execute(
        _join_query(on=on), {"left": LEFT, "right": RIGHT}
    )

    assert len(endpoint.requests) == 2
    assert result["dept:left"].tolist() == ["d1", "d1"]


def test_how_left_adds_a_null_row_for_an_unmatched_left(
    endpoint: CannedEndpoint,
) -> None:
    """The d2 left row has no same-dept candidate, so it survives with the right side empty."""

    result = _adapter(endpoint).execute(
        _join_query(on=("dept",), how="left"), {"left": LEFT, "right": RIGHT}
    )

    assert len(result) == 3
    unmatched = result.loc[result["note:right"].isna()]
    assert unmatched["note:left"].tolist() == ["beta left"]
    assert unmatched["tag"].tolist() == ["y"]


def test_an_empty_side_is_answered_without_a_model_call() -> None:
    """Nothing can match, so `outer` emits both sides' rows and no request goes out."""

    with canned_model() as running:
        result = _adapter(running).execute(
            _join_query(how="outer"),
            {"left": LEFT, "right": RIGHT.iloc[0:0]},
        )

    assert running.requests == []
    assert result["note:left"].tolist() == ["alpha left", "beta left"]
    assert result["note:right"].isna().all()


def test_the_ranking_prompt_leaves_candidates_to_the_renderer() -> None:
    """`candidates` is DocETL's Jinja slot, so it must survive Python's `format` untouched."""

    prompt = build_topk_prompt(
        "the subjects match: {note:left} and {note:right}",
        k=2,
        ids_column=SELECTED_COLUMN,
    )

    assert prompt.startswith("{{ input.candidates }}\n")
    assert "Condition" not in prompt
    assert f"put their ids in {SELECTED_COLUMN}" in prompt
    assert "at most 2 of them" in prompt


def test_k_keeps_at_most_k_matches_per_left_row() -> None:
    """One row overflows `k`, so exactly one ranking call is made and only its pick survives."""

    wide = pd.DataFrame({"note": ["alpha one"], "dept": ["d1"]})
    rights = pd.DataFrame(
        {"note": ["alpha two", "beta two", "gamma two"], "dept": ["d1", "d1", "d1"]}
    )
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", wide), _source("right", rights)),
        params={
            "instruction": "the subjects match: {note:left} and {note:right}",
            "k": 1,
        },
    )

    with canned_model("", map_answers={SELECTED_COLUMN: ["2"]}) as running:
        result = _adapter(running).execute(query, {"left": wide, "right": rights})

    assert len(result) == 1
    assert result["note:right"].tolist() == ["gamma two"]
    # three verdicts, then one ranking call
    assert len(running.requests) == 4


def test_a_ranking_answer_outside_the_candidates_is_refused() -> None:
    wide = pd.DataFrame({"note": ["alpha one"]})
    rights = pd.DataFrame({"note": ["alpha two", "beta two"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", wide), _source("right", rights)),
        params={
            "instruction": "the subjects match: {note:left} and {note:right}",
            "k": 1,
        },
    )

    with canned_model("", map_answers={SELECTED_COLUMN: ["9"]}) as running:
        with pytest.raises(ValueError, match="not a candidate"):
            _adapter(running).execute(query, {"left": wide, "right": rights})


def test_a_bad_ranking_answer_is_asked_again() -> None:
    """The first answer names a non-candidate, the second is valid, so the run completes.

    Based on: `adapters/lotus/sem_topk_listwise.py:93`, which retries a malformed listwise answer;
    DocETL's `map` has no such loop, so the retry lives in `_cap_matches`.
    """

    left = pd.DataFrame({"note": ["alpha"]})
    rights = pd.DataFrame({"note": ["alpha one", "alpha two"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", rights)),
        params={
            "instruction": "the subjects match: {note:left} and {note:right}",
            "k": 1,
            "how": "outer",
        },
    )
    answers = iter([["9"], ["0"]])

    with canned_model(
        "", map_answers={SELECTED_COLUMN: lambda raw: next(answers)}
    ) as running:
        result = _adapter(running).execute(query, {"left": left, "right": rights})

    assert result.loc[result["note:left"].notna(), "note:right"].tolist() == [
        "alpha one"
    ]
    assert result.loc[result["note:left"].isna(), "note:right"].tolist() == [
        "alpha two"
    ]


def test_every_asked_pair_is_traced(tmp_path: Path) -> None:
    """One row per asked pair, carrying the text that was compared, not only the matches."""

    left = pd.DataFrame({"note": ["alpha"]})
    rights = pd.DataFrame({"note": ["alpha one", "alpha two"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", rights)),
        params={"instruction": "the subjects match: {note:left} and {note:right}"},
    )

    with canned_model("") as running:
        build_docetl_adapter(
            running.base_url,
            config=DocetlExecutionConfig(
                lm_api_base=running.base_url,
                semantic_trace_dir=tmp_path,
            ),
        ).execute(query, {"left": left, "right": rights})

    events = [
        json.loads(line)
        for path in tmp_path.rglob("events.jsonl")
        for line in path.read_text().splitlines()
    ]
    pairs = [event for event in events if event["event_type"] == "pair_decision"]

    assert len(pairs) == 2
    assert {event["left"] for event in pairs} == {"alpha"}
    # compact traces keep the verdict as `decision`; `parsed_output` is an artifact-backed key
    assert all(event["decision"] is True for event in pairs)


def test_an_empty_side_is_traced_as_a_skipped_pairing(tmp_path: Path) -> None:
    """`skipped_pairwise` separates "no pairs existed" from "every pair was judged false"."""

    with canned_model() as running:
        build_docetl_adapter(
            running.base_url,
            config=DocetlExecutionConfig(
                lm_api_base=running.base_url,
                semantic_trace_dir=tmp_path,
            ),
        ).execute(_join_query(how="outer"), {"left": LEFT, "right": RIGHT.iloc[0:0]})

    events = [
        json.loads(line)
        for path in tmp_path.rglob("events.jsonl")
        for line in path.read_text().splitlines()
    ]
    skipped = [event for event in events if event.get("skipped_pairwise") is True]

    assert len(skipped) == 1
    assert skipped[0]["left_rows"] == len(LEFT)
    assert skipped[0]["right_rows"] == 0
    assert skipped[0]["how"] == "outer"


def test_an_invalid_k_on_an_empty_side_is_refused_before_any_call(
    endpoint: CannedEndpoint,
) -> None:
    """A non-positive `k` is a literal error, so it is rejected at entry even when a side is empty."""

    with pytest.raises(ValueError, match="sem_join k must be a positive integer"):
        _adapter(endpoint).execute(
            _join_query(k=0),
            {"left": LEFT, "right": RIGHT.iloc[0:0]},
        )

    assert endpoint.requests == []


def test_a_mixed_on_shape_on_an_empty_side_is_refused_before_any_call(
    endpoint: CannedEndpoint,
) -> None:
    """A mix of an exact key and a predicate is rejected at entry, not silently swallowed on an empty side."""

    from agent_memory.api import Log
    from agent_memory.policy.relation import _normalize_join_on

    left_rel = Log({"dept": "d", "note": "n"}).alias("left")
    right_rel = Log({"dept": "d", "note": "n"}).alias("right")
    predicate = _normalize_join_on([left_rel.col("dept") == right_rel.col("dept")])

    with pytest.raises(ValueError, match="exact keys or predicate expressions"):
        _adapter(endpoint).execute(
            _join_query(on=("dept", predicate)),
            {"left": LEFT, "right": RIGHT.iloc[0:0]},
        )

    assert endpoint.requests == []


def test_id_columns_are_written_on_matched_and_unmatched_rows(
    endpoint: CannedEndpoint,
) -> None:
    """The planner's join-map splits on these columns' nullness, so their placement is a contract."""

    result = _adapter(endpoint).execute(
        _join_query(on=("dept",), how="left", id_columns=("l_id", "r_id")),
        {"left": LEFT, "right": RIGHT},
    )

    assert list(result.columns)[-2:] == ["l_id", "r_id"]
    assert result["l_id"].tolist() == [0, 0, 1]
    assert result["r_id"].iloc[:2].tolist() == [0, 1]
    assert pd.isna(result["r_id"].iloc[2])


def test_id_columns_that_shadow_an_output_column_are_refused(
    endpoint: CannedEndpoint,
) -> None:
    with pytest.raises(ValueError, match="conflict with output columns"):
        _adapter(endpoint).execute(
            _join_query(id_columns=("note:left", "r_id")),
            {"left": LEFT, "right": RIGHT},
        )


def test_a_bare_string_on_is_one_key(endpoint: CannedEndpoint) -> None:
    """The DSL normalizes a single key name; a hand-built query must not be iterated per character."""

    result = _adapter(endpoint).execute(
        _join_query(on="dept"), {"left": LEFT, "right": RIGHT}
    )

    assert len(endpoint.requests) == 2
    assert result["dept:left"].tolist() == ["d1", "d1"]


def test_an_unsupported_resolver_is_refused_before_any_run() -> None:
    with pytest.raises(ValueError, match="sem_join_topk_method must be one of"):
        DocetlExecutionConfig(
            lm_api_base="http://127.0.0.1:9/v1",
            sem_join_topk_method="bogus",
        )


def test_a_match_capped_out_becomes_an_unmatched_row() -> None:
    """Documented consequence: `k` caps before `how`, so a right row can surface as right-only."""

    left = pd.DataFrame({"note": ["alpha"]})
    rights = pd.DataFrame({"note": ["alpha one", "alpha two"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", rights)),
        params={
            "instruction": "the subjects match: {note:left} and {note:right}",
            "k": 1,
            "how": "outer",
        },
    )

    with canned_model("", map_answers={SELECTED_COLUMN: ["0"]}) as running:
        result = _adapter(running).execute(query, {"left": left, "right": rights})

    assert len(result) == 2
    assert result.loc[result["note:left"].notna(), "note:right"].tolist() == [
        "alpha one"
    ]
    assert result.loc[result["note:left"].isna(), "note:right"].tolist() == [
        "alpha two"
    ]


class _JoinMemory(Memory):
    """A policy whose whole view is one semantic inner join of two branches of the log."""

    log = Log({"side": "Join side.", "key": "Exact partition.", "value": "Value."})

    left = log.filter(log.col("side") == "left").select(["key", "value"]).alias("left")
    right = log.filter(log.col("side") == "right").select(["key", "value"]).alias("right")
    pairs = left.sem_join(
        right,
        on="key",
        instruction="{value:left} and {value:right} mean the same thing.",
        how="inner",
    )


class _RetractingJoinMemory(Memory):
    """A join whose left input can lose a row, which the runtime must refuse to maintain."""

    log = Log({"side": "Join side.", "key": "Key.", "value": "Value."})

    minimum = log.group_by(["side", "key"]).min(column="value", output_col="value")
    left = minimum.filter(minimum.col("side") == "left").select(["key", "value"]).alias(
        "left"
    )
    right = minimum.filter(minimum.col("side") == "right").select(
        ["key", "value"]
    ).alias("right")
    pairs = left.sem_join(
        right,
        on="key",
        instruction="{value:left} and {value:right} mean the same thing.",
        how="inner",
    )


def _bag(frame: pd.DataFrame) -> Counter[tuple[object, ...]]:
    return Counter(tuple(row) for row in frame.itertuples(index=False, name=None))


def _full_view(memory: Memory, adapter: DocetlAdapter) -> pd.DataFrame:
    query = memory.spec().views["pairs"].query
    return adapter.execute(query, {"log": memory._runtime._state["log"]})


def test_an_appended_join_equals_a_full_recompute_on_both_engines() -> None:
    """`semantic_binary_state` maintains the join by three delta joins; the bag must not drift."""

    with canned_model("") as endpoint:
        lotus_adapter, docetl_adapter = build_adapters(endpoint.base_url)
        memories = [
            _JoinMemory(adapter=lotus_adapter),
            _JoinMemory(adapter=docetl_adapter),
        ]
        for event in (
            {"side": "left", "key": "k", "value": "same"},
            {"side": "right", "key": "k", "value": "same"},
            {"side": "left", "key": "k", "value": "same"},
        ):
            for memory in memories:
                memory.add(event)
        # inside the canned context: a full recompute is another model-backed run
        full = _full_view(memories[1], docetl_adapter)

    lotus_state, docetl_state = (memory._runtime._state["pairs"] for memory in memories)
    assert _bag(docetl_state) == _bag(lotus_state)
    assert _bag(docetl_state) == _bag(full)
    assert len(docetl_state) == 2


def test_a_semantic_binary_state_join_survives_snapshot_and_restore() -> None:
    """The join node's own state is what a checkpoint carries; it must resume without re-deriving."""

    with canned_model("") as endpoint:
        _lotus, docetl = build_adapters(endpoint.base_url)
        original = _JoinMemory(adapter=docetl)
        for event in (
            {"side": "left", "key": "k", "value": "same"},
            {"side": "right", "key": "k", "value": "same"},
        ):
            original.add(event)
        snapshot = original._runtime.snapshot_state()

        restored = _JoinMemory(adapter=docetl)
        restored._runtime.restore_state(snapshot)
        for memory in (original, restored):
            memory.add({"side": "left", "key": "k", "value": "same"})

    assert _bag(restored._runtime._state["pairs"]) == _bag(
        original._runtime._state["pairs"]
    )
    assert len(restored._runtime._state["pairs"]) == 2


def test_a_join_that_would_lose_a_row_is_refused() -> None:
    """The maintenance is append-only; a retraction must fail rather than silently drop a match."""

    with canned_model("") as endpoint:
        adapter = build_docetl_adapter(endpoint.base_url)
        memory = _RetractingJoinMemory(adapter=adapter)
        memory.add({"side": "right", "key": "k", "value": 5})
        memory.add({"side": "left", "key": "k", "value": 5})

        with pytest.raises(NotImplementedError, match="append-only"):
            memory.add({"side": "left", "key": "k", "value": 3})
