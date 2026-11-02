"""The join lowerings: `sem_join` pair generation, `how`, `k`, `id_columns` and the assembled frame, plus the
name the deterministic join publishes.

No model: the loopback endpoint decides each pair from a marker in the request, so DocETL's `filter`
validation and rendering still run while the deterministic half is asserted on the published frame.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters import DocetlAdapter
from agent_memory.adapters.docetl.context import (
    DocetlExecutionConfig,
    DocetlExecutionContext,
)
from agent_memory.adapters.docetl.prompt import (
    build_join_prompt,
    build_topk_prompt,
    compared_columns,
)
from agent_memory.adapters.docetl.relational import execute_join as execute_key_join
from agent_memory.adapters.docetl.sem_join import (
    ANCHOR_COLUMN,
    LEFT_ID_COLUMN,
    RIGHT_ID_COLUMN,
    SELECTED_COLUMN,
    build_verdict_config,
)
from agent_memory.adapters.docetl.strict import strict_equijoin
from agent_memory.api import Log, Memory
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.placeholders import JoinScope

from adapter_support import (
    CANNED_MODEL,
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
        scope=JoinScope.of(("note",), ("note",)),
    )

    # each line reads its own side's record, because `equijoin` renders the prompt against
    #   `left` and `right` rather than against one record holding both
    assert prompt == (
        "Context:\n"
        "[left] note: «{{ left['left_0'] }}»\n"
        "[right] note: «{{ right['right_0'] }}»\n"
        "\n"
        "Condition: the subjects match: note and note\n"
        "Does the context satisfy this condition?"
    )
    # a placeholder must not survive into the Jinja the renderer sees
    assert "{note" not in prompt


def test_a_bare_placeholder_is_resolved_by_the_side_that_owns_it() -> None:
    scope = compared_columns(
        "{title} against {body}",
        JoinScope.of(("title", "shared"), ("body", "shared")),
    )

    assert scope.left == ("title",)
    assert scope.right == ("body",)


def _aliased(source: QueryExpr, name: str) -> QueryExpr:
    return QueryExpr(op="alias", inputs=(source,), params={"name": name})


def test_an_alias_on_this_joins_input_names_the_side_it_carries(
    endpoint: CannedEndpoint,
) -> None:
    """§3 ①: `earlier` is this input's alias, so `{fact:earlier}` is the left `fact`.

    Only this join's own inputs are consulted, and the alias is one of their names:
    without reading it, `earlier` would name a suffix neither side has. The alias names a
    side for references only: a semantic join still publishes `left`/`right`.
    """

    left = pd.DataFrame({"fact": ["a"]})
    right = pd.DataFrame({"fact": ["b"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_aliased(_source("left", left), "earlier"), _source("right", right)),
        params={"instruction": "Is {fact:earlier} older than {fact:right}?"},
    )

    result = _adapter(endpoint).execute(query, {"left": left, "right": right})

    assert len(result) == 1
    assert list(result.columns) == ["fact:left", "fact:right"]
    sent = json.dumps(endpoint.requests, ensure_ascii=False)
    assert "[left] fact: \u00aba\u00bb" in sent
    assert "[right] fact: \u00abb\u00bb" in sent


def test_a_name_a_previous_join_published_is_read_whole(
    endpoint: CannedEndpoint,
) -> None:
    """§3 ②: `earlier` is part of the column name here, not a side of this join.

    The left frame is what `alias("earlier").join(...)` publishes, and this join carries
    no alias to inherit, so the reference names that column and the left side owns it.
    """

    left = pd.DataFrame({"fact:earlier": ["a"]})
    right = pd.DataFrame({"id": ["1"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", right)),
        params={"instruction": "Is {fact:earlier} the same as {id:right}?"},
    )

    _adapter(endpoint).execute(query, {"left": left, "right": right})

    sent = json.dumps(endpoint.requests, ensure_ascii=False)
    assert "[left] fact:earlier: \u00aba\u00bb" in sent
    assert "[right] id: \u00ab1\u00bb" in sent


def test_a_suffixed_name_that_collides_with_a_published_name_is_refused(
    endpoint: CannedEndpoint,
) -> None:
    """§6: suffixing may not overwrite a column the other side already publishes.

    `note` is on both sides, so the left copy becomes `note:left`; the right side
    already has a column of that name, and neither may be dropped.
    """

    left = pd.DataFrame({"note": ["a"]})
    right = pd.DataFrame({"note": ["b"], "note:left": ["c"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", right)),
        params={"instruction": "the notes match: {note:left} and {note:right}"},
    )

    with pytest.raises(ValueError, match="two columns"):
        _adapter(endpoint).execute(query, {"left": left, "right": right})


def test_a_deterministic_join_that_would_publish_one_name_twice_is_refused(
    endpoint: CannedEndpoint,
) -> None:
    """§6 on the other join path: pandas suffixes `note` without looking at the other frame.

    The right frame is what an earlier join published, so it already carries `note:left`;
    suffixing the left side's `note` lands on that label, and the merge returns it twice.
    A clash inside one frame is refused by pandas itself, so only this half is silent.
    """

    left = pd.DataFrame({"id": ["1"], "note": ["a"]})
    right = pd.DataFrame({"id": ["1"], "note": ["c"], "note:left": ["b"]})
    query = QueryExpr(
        op="join",
        inputs=(_source("left", left), _source("right", right)),
        params={"on": ("id",), "how": "inner"},
    )

    with pytest.raises(ValueError, match="two columns"):
        _adapter(endpoint).execute(query, {"left": left, "right": right})

    assert endpoint.requests == []


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


def test_a_verdict_that_is_not_a_boolean_is_refused(endpoint: CannedEndpoint) -> None:
    """A string where a verdict belongs is a defect, not a match.

    The engine returns the field unchecked (`equijoin.py:185`), so `"maybe"` would read as true and
    publish a joined row; the pair is asked about and no verdict about it was given.
    """

    endpoint.refuse()

    with pytest.raises(RuntimeError, match="not a boolean"):
        _adapter(endpoint).execute(_join_query(), {"left": LEFT, "right": RIGHT})


def test_identical_rows_are_matched_one_by_one(endpoint: CannedEndpoint) -> None:
    """A bag, not a set: two equal left rows are two rows, each with its own match.

    The engine keys its blocking bookkeeping by a hash of the record (`equijoin.py:49-50`), so
    identical records are only distinguished by the id they carry.
    """

    left = pd.DataFrame({"note": ["alpha", "alpha"]})
    right = pd.DataFrame({"note": ["alpha"]})
    query = QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", right)),
        params={"instruction": "the subjects match: {note:left} and {note:right}"},
    )

    result = _adapter(endpoint).execute(query, {"left": left, "right": right})

    assert sorted(result["note:left"].tolist()) == ["alpha", "alpha"]
    assert len(endpoint.requests) == 2


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


def test_the_ranking_prompt_carries_both_sides_and_the_limits() -> None:
    """Building it states both slots and the limits; sending it puts the anchor on the wire.

    DocETL's own system message asks for high recall, so the user message has to say that a smaller
    answer is legal (`sem_topk_join.py::LISTWISE_JOIN_SYSTEM_PROMPT` says the same three things).
    """

    prompt = build_topk_prompt(
        "the subjects match: {note:left} and {note:right}",
        scope=JoinScope.of(("note",), ("note",)),
        k=2,
        ids_column=SELECTED_COLUMN,
        anchor_column=ANCHOR_COLUMN,
    )

    assert "{{ input.left_row }}" in prompt
    assert "{{ input.candidates }}" in prompt
    assert "Condition" not in prompt
    assert f"put their ids in {SELECTED_COLUMN}" in prompt
    assert "at most 2 of them" in prompt
    assert "do not fill the limit" in prompt.lower()
    assert "empty list when none match" in prompt.lower()
    # the condition precedes the candidates, so the reader knows what the list is for
    assert prompt.index("join condition") < prompt.index("{{ input.candidates }}")

    # the wire half: the anchor is rendered per record, so the model sees the row it matches against
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

    with canned_model("", map_answers={SELECTED_COLUMN: ["0"]}) as running:
        result = _adapter(running).execute(query, {"left": wide, "right": rights})

    ranking = [body for body in running.requests if SELECTED_COLUMN in json.dumps(body)]
    assert len(ranking) == 1
    assert "alpha one" in json.dumps(ranking[0])
    assert result["note:right"].tolist() == ["alpha two"]


def test_an_empty_ranking_answer_leaves_the_left_row_unmatched() -> None:
    """The prompt allows "none of these"; taking that answer must not be treated as a malformed one."""

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

    with canned_model("", map_answers={SELECTED_COLUMN: []}) as running:
        result = _adapter(running).execute(query, {"left": wide, "right": rights})

    assert list(result.columns) == ["note:left", "note:right"]
    assert result.empty


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
    DocETL's `map` has no such loop, so the retry lives in `_cap_matches`. Only `_cap_matches`'s
    contribution is asserted here: the capped-out frame itself is the case above.
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

    ranking = [body for body in running.requests if SELECTED_COLUMN in json.dumps(body)]
    assert len(ranking) == 2  # refused once, answered once
    assert result.loc[result["note:left"].notna(), "note:right"].tolist() == [
        "alpha one"
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
    assert str(result["l_id"].dtype) == "int64"
    assert str(result["r_id"].dtype) == "Int64"


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


def test_a_row_local_op_after_a_self_join_can_name_a_suffixed_column() -> None:
    """A self-join suffixes every column, and that suffix is the only way to tell the sides apart.

    Downstream the column is literally `note:left`, so a semantic instruction has to be able to name
    it. Today the placeholder grammar reads `:left` as a side qualifier and refuses it.
    """

    frame = pd.DataFrame({"note": ["alpha asks for a refund", "beta ships"]})
    log = _source("log", frame)
    joined = QueryExpr(
        op="sem_join",
        inputs=(
            QueryExpr(op="alias", inputs=(log,), params={"name": "l"}),
            QueryExpr(op="alias", inputs=(log,), params={"name": "r"}),
        ),
        params={"instruction": "{note:left} and {note:right} are different notes"},
    )
    query = QueryExpr(
        op="sem_filter",
        inputs=(joined,),
        params={"instruction": "the left note {note:left} asks for a refund"},
    )

    with canned_model("refund") as running:
        result = _adapter(running).execute(query, {"log": frame})

    assert "note:left" in result.columns
    assert "note:right" in result.columns


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


BIG = 9007199254740993


def _join_of(
    left: pd.DataFrame, right: pd.DataFrame, **params: object
) -> QueryExpr:
    """Build a key-scoped semantic join over frames chosen by the case."""

    return QueryExpr(
        op="sem_join",
        inputs=(_source("left", left), _source("right", right)),
        params={"instruction": "same dept: {dept:left} and {dept:right}", **params},
    )


def _materialized(query: QueryExpr, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resolve a source with no storage: the case hands the frame in by name."""

    return inputs[str(query.params["name"])]


def test_a_sem_join_keeps_an_unmatched_integer_exact(
    endpoint: CannedEndpoint,
) -> None:
    """The right side loses a row, so its integer column gains a null. The nullable carrier keeps both
    the null and the value, where the float carrier pandas would have chosen rounds 2**53 + 1.
    """

    left = pd.DataFrame({"dept": ["d1", "d2"], "tag": ["x", "y"]})
    right = pd.DataFrame({"dept": ["d1"], "amount": [BIG]})

    result = _adapter(endpoint).execute(
        _join_of(left, right, on=("dept",), how="left"),
        {"left": left, "right": right},
    )

    assert str(result["amount"].dtype) == "Int64"
    assert result["amount"].iloc[0] == BIG
    assert pd.isna(result["amount"].iloc[1])


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_a_key_join_keeps_an_integer_exact(how: str) -> None:
    """Pandas converts a carried integer to float the moment `how` can leave it unmatched, and the
    conversion happens inside the merge. The side that can lose a row is converted first, and a column
    that lost none comes back under the dtype it entered with.
    """

    left = pd.DataFrame({"dept": ["d1", "d2"], "rank": [1, 2]})
    right = pd.DataFrame({"dept": ["d1", "d3"], "amount": [BIG, 7]})
    query = QueryExpr(
        op="join",
        inputs=(_source("left", left), _source("right", right)),
        params={"on": ("dept",), "how": how},
    )

    result = execute_key_join(query, {"left": left, "right": right}, _materialized)

    if how in {"left", "outer"}:
        assert str(result["amount"].dtype) == "Int64"
        assert result["amount"].iloc[0] == BIG
        assert result["amount"].isna().any()
    else:
        assert str(result["amount"].dtype) == "int64"
        assert result["amount"].iloc[0] == BIG
        assert not result["amount"].isna().any()
    if how in {"right", "outer"}:
        assert str(result["rank"].dtype) == "Int64"
        assert result["rank"].isna().any()
    else:
        assert str(result["rank"].dtype) == "int64"
        assert not result["rank"].isna().any()


# --------------------------------------------------- the override, against the engine's method


def _verdict_operations(running: CannedEndpoint) -> tuple[Any, Any]:
    """Return the engine's `equijoin` and the overridden one: one handle, one spec, two classes."""

    from docetl.operations.equijoin import EquijoinOperation

    context = DocetlExecutionContext(
        model=CANNED_MODEL,
        config=DocetlExecutionConfig(lm_api_base=running.base_url),
    )
    spec = build_verdict_config(
        "the notes match: {note:left} and {note:right}",
        scope=JoinScope.of(("note",), ("note",)),
    )
    override = strict_equijoin(
        left_id_key=LEFT_ID_COLUMN, right_id_key=RIGHT_ID_COLUMN
    )
    return (
        EquijoinOperation(context.handle, dict(spec), CANNED_MODEL, 1),
        override(context.handle, dict(spec), CANNED_MODEL, 1),
    )


def _pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the two records the verdict prompt reads: one line per side."""

    return {"left_id": 0, "left_0": "same note"}, {"right_id": 3, "right_0": "same note"}


def test_the_override_asks_the_question_the_engine_asks(
    endpoint: CannedEndpoint,
) -> None:
    """Same spec, same pair: the request body and the verdict are identical, path by path.

    The override rests on one claim — that only the failure paths changed — and this is where the
    claim can be seen: the bytes sent to the model, and what came back.
    """

    engine, ours = _verdict_operations(endpoint)
    left, right = _pair()
    prompt = engine.config["comparison_prompt"]

    expected = engine.compare_pair(prompt, CANNED_MODEL, left, right)
    actual = ours.compare_pair(prompt, CANNED_MODEL, left, right)

    assert len(endpoint.requests) == 2
    assert endpoint.requests[0] == endpoint.requests[1]
    assert actual == expected


def test_a_prompt_that_will_not_render_is_a_failure_not_a_no_match(
    endpoint: CannedEndpoint,
) -> None:
    """An undefined field raises where the engine logs the render error and answers `False`."""

    from jinja2.exceptions import UndefinedError

    engine, ours = _verdict_operations(endpoint)
    left, right = _pair()
    broken = "notes: {{ left['absent'] }} against {{ right['absent'] }}"

    assert engine.compare_pair(broken, CANNED_MODEL, left, right) == (False, 0)
    with pytest.raises(UndefinedError):
        ours.compare_pair(broken, CANNED_MODEL, left, right)
    # the failure is the renderer's, so no pair was ever sent
    assert endpoint.requests == []


def test_a_call_that_never_returned_is_a_failure_not_a_no_match(
    endpoint: CannedEndpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausted retries arrive as `response=None`, which the engine parses into `False`."""

    engine, ours = _verdict_operations(endpoint)
    left, right = _pair()
    prompt = engine.config["comparison_prompt"]
    unanswered = SimpleNamespace(response=None, total_cost=0.0)
    # one handle, so one patch reaches both operations
    monkeypatch.setattr(engine.runner.api, "call_llm", lambda *args, **kwargs: unanswered)

    assert engine.compare_pair(prompt, CANNED_MODEL, left, right) == (False, 0.0)
    with pytest.raises(RuntimeError, match="got no answer"):
        ours.compare_pair(prompt, CANNED_MODEL, left, right)


def test_a_pair_operation_cost_reaches_the_handle() -> None:
    """`execute(left, right)` returns cost too, and this path adds it the way the row path does."""

    from docetl.operations.equijoin import EquijoinOperation

    class PricedEquijoin(EquijoinOperation):
        def execute(self, *args: Any, **kwargs: Any) -> Any:
            return [], 3.5

    context = DocetlExecutionContext()
    spec = build_verdict_config(
        "the notes match: {note:left} and {note:right}",
        scope=JoinScope.of(("note",), ("note",)),
    )
    context.run_pair_operation("equijoin", spec, [], [], operation_cls=PricedEquijoin)

    assert context.handle.total_cost == 3.5
