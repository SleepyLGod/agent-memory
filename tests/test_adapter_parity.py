"""Cross-adapter parity: the same logical query through both execution backends.

The project defines correctness as the two adapters agreeing, so this is the only place a second,
independently written backend checks the first. Three comparison modes: ``exact`` (no model: frames must
match in rows, values, dtypes and index), ``identity`` (a model decides, so only the surviving rows and
columns must agree) and ``raises`` (the operator is illegal, and both must refuse it the same way).

The semantic cases point both adapters at one loopback endpoint, so their real operation classes,
validation and record handling all run with nothing leaving the machine — which proves the plumbing
agrees, not that a real model answers correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.policy.logical import QueryExpr

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

    covered = {case.query.op for case in _engine_free_cases()} | {"sem_filter"}
    excused = {
        # the window pair is only legal inside a process_window body, and is covered there
        "window_source",
        "count_window",
    }
    assert covered | excused >= set(DocetlAdapter().supported_ops())
