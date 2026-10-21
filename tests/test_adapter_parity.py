"""Cross-adapter parity: the same `QueryExpr` through both backends must produce the same frame.

`exact` compares rows, values, dtypes and index (no model involved); `identity` compares only the surviving
rows and columns (a model decides); `raises` requires both to refuse with the same exception class. The
semantic cases share one loopback endpoint, so both engines' own validation runs — this proves the plumbing
agrees, not that a real model answers well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterator, Mapping

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.policy.aggregates import ArrayAggregateSpec
from agent_memory.policy.expressions import ColumnExpr
from agent_memory.policy.logical import ColumnSpec, QueryExpr

from adapter_support import (
    CannedEndpoint,
    build_adapters,
    canned_model,
    default_adapters,
    require_real_provider,
)

# ---------------------------------------------------------------- the one question

INSTRUCTION = (
    "Decide whether this note asks for a refund: {note}\n\n"
    "Respond with 'true' if the note asks for a refund, otherwise respond with 'false'."
)

# the markers appear only in the notes: one in the instruction would decide every row
KEEP_MARKERS = ("arrived broken", "duplicate charge")
assert not any(marker in INSTRUCTION for marker in KEEP_MARKERS)

NOTES = [
    "I would like a refund for the order that arrived broken.",
    "Please return the money for the duplicate charge on my card.",
    "The weather here has been lovely all week.",
    "I finished the book you lent me, thank you.",
]
NOTE_INDEX = [10, 11, 12, 13]
EXPECTED_KEPT = [10, 11]

@pytest.fixture
def canned_endpoint() -> Iterator[CannedEndpoint]:
    """Serve both protocols on loopback; a note carrying a keep marker is the one kept."""

    with canned_model(*KEEP_MARKERS) as endpoint:
        yield endpoint


# ---------------------------------------------------------------- the cases


@dataclass(frozen=True)
class ParityCase:
    """One logical query, its inputs, and what the two adapters must agree on."""

    name: str
    query: QueryExpr
    inputs: Mapping[str, pd.DataFrame]
    comparison: str
    needs_model: bool = False
    excuses: dict[str, str] = field(default_factory=dict)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"note": list(NOTES)}, index=NOTE_INDEX)


def _notes_frame() -> pd.DataFrame:
    return _frame()


def _source(name: str, frame: pd.DataFrame) -> QueryExpr:
    """A materialized view over `frame` that declares its columns, so schema checks can run."""

    return QueryExpr(
        op="materialized_view",
        params={"name": name, "columns": tuple(str(c) for c in frame.columns)},
    )


def _relational_cases() -> list[ParityCase]:
    """The deterministic operators: no model decides, so the frames must match exactly."""

    rows = pd.DataFrame(
        {"id": [1, 1, 2], "tag": ["a", "b", "a"], "score": [10, 20, 30]}
    )
    named = pd.DataFrame({"id": [1, 2], "name": ["x", "y"]})
    other = pd.DataFrame({"id": [2, 3], "other": ["p", "q"]})
    booleans = pd.DataFrame({"keep": [True, False], "tag": ["a", "b"]})
    arrays = pd.DataFrame(
        {
            "id": [1, 2],
            "items": [json.dumps(["a", "b"]), json.dumps(["c"])],
            "nested": [json.dumps([["a"], ["b"]]), json.dumps([["c"]])],
            "object": [json.dumps({"x": 1, "y": 2}), json.dumps({"x": 3, "y": 4})],
        }
    )
    single_array = pd.DataFrame({"items": [json.dumps(["z"])]})

    def case(
        name: str,
        query: QueryExpr,
        inputs: Mapping[str, pd.DataFrame],
        comparison: str = "exact",
    ) -> ParityCase:
        return ParityCase(name=name, query=query, inputs=inputs, comparison=comparison)

    return [
        case(
            "select",
            QueryExpr(
                op="select",
                inputs=(_source("rows", rows),),
                params={"columns": ("id", "tag")},
            ),
            {"rows": rows},
        ),
        case(
            "limit",
            QueryExpr(op="limit", inputs=(_source("rows", rows),), params={"n": 2}),
            {"rows": rows},
        ),
        case(
            "alias",
            QueryExpr(op="alias", inputs=(_source("rows", rows),), params={"name": "r"}),
            {"rows": rows},
        ),
        case(
            "assign scalar",
            QueryExpr(
                op="assign",
                inputs=(_source("rows", rows),),
                params={"assignments": {"label": "x"}},
            ),
            {"rows": rows},
        ),
        case(
            "filter column",
            QueryExpr(
                op="filter",
                inputs=(_source("rows", booleans),),
                params={"predicate": ColumnExpr("keep").to_param()},
            ),
            {"rows": booleans},
        ),
        case(
            "group_by",
            QueryExpr(
                op="group_by", inputs=(_source("rows", rows),), params={"keys": ("id",)}
            ),
            {"rows": rows},
        ),
        case(
            "let",
            QueryExpr(
                op="let",
                inputs=(_source("rows", rows), _source("bound", named)),
                params={"name": "bound"},
            ),
            # the binding is injected by `let`, so it must not already be an input
            {"rows": rows},
        ),
        case(
            "concat",
            QueryExpr(
                op="concat", inputs=(_source("left", named), _source("right", named))
            ),
            {"left": named, "right": named},
        ),
        case(
            "union",
            QueryExpr(
                op="union", inputs=(_source("left", named), _source("right", named))
            ),
            {"left": named, "right": named},
        ),
        case(
            "union_by_name",
            QueryExpr(
                op="union_by_name",
                inputs=(_source("left", named), _source("right", other)),
            ),
            {"left": named, "right": other},
        ),
        case(
            "subtract",
            QueryExpr(
                op="subtract", inputs=(_source("left", named), _source("right", named))
            ),
            {"left": named, "right": named},
        ),
        case(
            "join hash",
            QueryExpr(
                op="join",
                inputs=(_source("left", named), _source("right", other)),
                params={"on": ("id",), "how": "inner"},
            ),
            {"left": named, "right": other},
        ),
        case(
            "drop_duplicates",
            QueryExpr(
                op="drop_duplicates",
                inputs=(_source("rows", rows),),
                params={"subset": ("id",)},
            ),
            {"rows": rows},
        ),
        case(
            "array_agg",
            QueryExpr(
                op="array_agg",
                inputs=(_source("rows", rows),),
                params={"columns": ("id",), "output_col": "payload"},
            ),
            {"rows": rows},
        ),
        case(
            "min",
            QueryExpr(
                op="min",
                inputs=(_source("rows", rows),),
                params={"columns": ("score",), "output_col": "lowest"},
            ),
            {"rows": rows},
        ),
        case(
            "flatten",
            QueryExpr(
                op="flatten", inputs=(_source("rows", arrays),), params={"column": "nested"}
            ),
            {"rows": arrays},
        ),
        case(
            "explode",
            QueryExpr(
                op="explode", inputs=(_source("rows", arrays),), params={"column": "items"}
            ),
            {"rows": arrays},
        ),
        case(
            "unnest",
            QueryExpr(
                op="unnest",
                inputs=(_source("rows", arrays),),
                params={"column": "object", "fields": (("x", "x"), ("y", "y"))},
            ),
            {"rows": arrays},
        ),
        case(
            "array_cat",
            QueryExpr(
                op="array_cat",
                inputs=(
                    _source("left", single_array),
                    _source("right", single_array),
                ),
                params={"column": "items"},
            ),
            {"left": single_array, "right": single_array},
        ),
        case(
            "agg",
            QueryExpr(
                op="agg",
                inputs=(
                    QueryExpr(
                        op="group_by",
                        inputs=(_source("rows", rows),),
                        params={"keys": ("id",)},
                    ),
                ),
                params={
                    "aggregates": (
                        ArrayAggregateSpec(columns=("tag",), output_col="tags"),
                    )
                },
            ),
            {"rows": rows},
        ),
    ]


def _engine_free_cases() -> list[ParityCase]:
    """Engine-free operators: both adapters must produce identical frames."""

    log = QueryExpr(op="log")
    count_window = QueryExpr(
        op="count_window",
        inputs=(log,),
        params={"size": 2, "slide": 2},
    )
    return [
        ParityCase(
            name="log",
            query=log,
            inputs={"log": _frame()},
            comparison="exact",
        ),
        ParityCase(
            name="materialized_view",
            query=QueryExpr(op="materialized_view", params={"name": "view"}),
            inputs={"view": _frame()},
            comparison="exact",
        ),
        ParityCase(
            name="process_window over count_window",
            query=QueryExpr(
                op="process_window",
                inputs=(count_window, QueryExpr(op="window_source")),
            ),
            inputs={"log": _frame()},
            comparison="exact",
        ),
        ParityCase(
            name="window_source outside a window",
            query=QueryExpr(op="window_source"),
            inputs={"log": _frame()},
            comparison="raises",
        ),
        ParityCase(
            name="count_window outside process_window",
            query=count_window,
            inputs={"log": _frame()},
            comparison="raises",
        ),
        *_relational_cases(),
    ]


def _semantic_query() -> QueryExpr:
    return QueryExpr(
        op="sem_filter",
        inputs=(QueryExpr(op="log"),),
        params={"instruction": INSTRUCTION},
    )


# ---------------------------------------------------------------- the runner


def _run(both: tuple[LotusAdapter, DocetlAdapter], case: ParityCase) -> list[Any]:
    """Return each adapter's answer, or the exception class it raised."""

    answers: list[Any] = []
    for adapter in both:
        try:
            answers.append(adapter.execute(case.query, case.inputs))
        except Exception as error:  # noqa: BLE001 - the exception itself is the answer here
            answers.append(type(error))
    return answers


def _assert_agreement(case: ParityCase, answers: list[Any]) -> None:
    left, right = answers
    if case.comparison == "raises":
        assert isinstance(left, type) and isinstance(right, type), (
            f"{case.name}: expected both adapters to refuse, got {left!r} and {right!r}"
        )
        assert left is right, f"{case.name}: raised {left.__name__} against {right.__name__}"
        return

    assert isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame), (
        f"{case.name}: both adapters must return a frame — a raised error is a failed run, not a "
        f"parity finding — got "
        f"{getattr(left, '__name__', type(left).__name__)} and "
        f"{getattr(right, '__name__', type(right).__name__)}"
    )
    if case.comparison == "exact":
        assert_frame_equal(left, right)
        return

    # identity: the same rows survived, in the same order, with the same columns
    assert list(left.columns) == list(right.columns), (
        f"{case.name}: columns {list(left.columns)} against {list(right.columns)}"
    )
    assert list(left.index) == list(right.index), (
        f"{case.name}: kept {list(left.index)} against {list(right.index)}"
    )


# ------------------------------------------------- engine-free cases, no model


@pytest.mark.parametrize("case", _engine_free_cases(), ids=lambda case: case.name)
def test_engine_free_ops_agree(case: ParityCase) -> None:
    """No model is involved, so the two adapters must answer identically."""

    assert not case.needs_model
    _assert_agreement(case, _run(default_adapters(), case))


# --------------------------------------------- a model decides, canned endpoint


def test_sem_filter_agrees_through_a_canned_endpoint(
    canned_endpoint: CannedEndpoint,
) -> None:
    """Both engines, real operation classes, one canned verdict per row, no model called."""

    case = ParityCase(
        name="sem_filter",
        query=_semantic_query(),
        inputs={"log": _notes_frame()},
        comparison="identity",
        needs_model=True,
    )
    answers = _run(build_adapters(canned_endpoint.base_url), case)
    _assert_agreement(case, answers)

    # the kept rows are the ones the canned endpoint approved: the claim is not vacuous
    assert list(answers[0].index) == EXPECTED_KEPT, list(answers[0].index)


def test_an_unusable_answer_diverges_by_design(
    canned_endpoint: CannedEndpoint,
) -> None:
    """Pins the recorded divergence: on an unusable answer LOTUS keeps the row, DocETL drops it.

    LOTUS's parse failure falls back to keeping (``lotus.sem_ops.sem_filter::sem_filter``,
    ``default=True``); DocETL drops a row whose answer fails the declared type.
    """

    case = ParityCase(
        name="sem_filter",
        query=_semantic_query(),
        inputs={"log": _notes_frame()},
        comparison="identity",
        needs_model=True,
    )
    canned_endpoint.refuse()
    lotus_answer, docetl_answer = _run(build_adapters(canned_endpoint.base_url), case)

    assert list(lotus_answer.index) == NOTE_INDEX, list(lotus_answer.index)
    assert docetl_answer.empty, list(docetl_answer.index)


def test_sem_map_agrees_through_a_canned_endpoint() -> None:
    """Both engines add the same column, from the same canned answer, on the same rows."""

    case = ParityCase(
        name="sem_map",
        query=QueryExpr(
            op="sem_map",
            inputs=(QueryExpr(op="log"),),
            params={
                "instruction": "Summarise {note}",
                "output_cols": (ColumnSpec(name="summary", description="one line"),),
            },
        ),
        inputs={"log": _notes_frame()},
        comparison="exact",
        needs_model=True,
    )

    with canned_model(map_answers={"summary": "canned summary"}) as endpoint:
        answers = _run(build_adapters(endpoint.base_url), case)

    _assert_agreement(case, answers)
    assert answers[0]["summary"].tolist() == ["canned summary"] * len(NOTES)


def test_sem_flat_map_agrees_through_a_canned_endpoint() -> None:
    """Both engines expand one input row into the same emitted rows, with the same ordinals."""

    case = ParityCase(
        name="sem_flat_map",
        query=QueryExpr(
            op="sem_flat_map",
            inputs=(QueryExpr(op="log"),),
            params={
                "instruction": "Split {note} into entries",
                "output_cols": (ColumnSpec(name="entry", description="one entry"),),
                "ordinal_col": "ord",
            },
        ),
        inputs={"log": _notes_frame()},
        comparison="exact",
        needs_model=True,
    )

    with canned_model(map_rows=({"entry": "a"}, {"entry": "b"})) as endpoint:
        answers = _run(build_adapters(endpoint.base_url), case)

    _assert_agreement(case, answers)
    assert answers[0]["entry"].tolist()[:4] == ["a", "b", "a", "b"]


def test_sem_join_agrees_through_a_canned_endpoint() -> None:
    """Both engines emit the same bag for the same keyed join, with the same suffixed columns."""

    left = pd.DataFrame({"note": ["alpha left", "beta left"], "dept": ["d1", "d2"]})
    right = pd.DataFrame({"note": ["alpha right", "beta right"], "dept": ["d1", "d1"]})
    case = ParityCase(
        name="sem_join",
        query=QueryExpr(
            op="sem_join",
            inputs=(
                QueryExpr(
                    op="materialized_view",
                    params={"name": "left", "columns": tuple(left.columns)},
                ),
                QueryExpr(
                    op="materialized_view",
                    params={"name": "right", "columns": tuple(right.columns)},
                ),
            ),
            params={
                "instruction": "the subjects match: {note:left} and {note:right}",
                "how": "inner",
                "on": ("dept",),
            },
        ),
        inputs={"left": left, "right": right},
        comparison="exact",
        needs_model=True,
    )

    with canned_model(
        "the subjects match",
        map_answers={"join_ok": True},
        # LOTUS reads a verdict as an `Answer:` line; DocETL reads the same marker as a kept row
        answer_markers={"the subjects match": "join_ok"},
    ) as endpoint:
        answers = _run(build_adapters(endpoint.base_url), case)

    _assert_agreement(case, answers)
    assert len(answers[0]) == 2
    assert answers[0]["dept:left"].tolist() == ["d1", "d1"]


def test_sem_filter_agrees_against_the_real_model() -> None:
    """The semantic claim: both backends keep the same rows when a real model answers."""

    model, base_url = require_real_provider("the real-model parity case")
    case = ParityCase(
        name="sem_filter",
        query=_semantic_query(),
        inputs={"log": _notes_frame()},
        comparison="identity",
        needs_model=True,
    )
    # pinned: the engines ask the same question in different words, so a free temperature makes a
    # single divergent row indistinguishable from sampling noise
    answers = _run(
        build_adapters(
            base_url,
            model=model,
            api_key=None,
            completion_kwargs={"temperature": 0},
        ),
        case,
    )
    _assert_agreement(case, answers)


# ---------------------------------------------------------------- coverage


def test_every_supported_op_has_a_parity_case() -> None:
    """No operator can be implemented and never compared.

    The excuse list is the only way to leave one out, and naming it here is what makes the omission a
    decision rather than an oversight.
    """

    covered = {case.query.op for case in _engine_free_cases()} | {
        # each of these has a canned case below, and the parity file is where the engine claim lives
        "sem_filter",
        "sem_map",
        "sem_flat_map",
        "sem_join",
    }
    excused = {
        # the window pair is only legal inside a process_window body, and is covered there
        "window_source",
        "count_window",
    }
    assert covered | excused >= set(DocetlAdapter().supported_ops())
