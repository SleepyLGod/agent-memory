"""The semantic lowerings: instruction translation, the config handshake, and each operator offline.

One group per operator — `sem_filter`, `sem_map`, `sem_flat_map` — and this is the home for the next one
rather than a new file. No model is called: the endpoint is a loopback server reached through
`lm_api_base`, so DocETL's own validation, parsing and record handling all run.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent_memory.adapters import DocetlAdapter
from agent_memory.adapters.docetl.context import (
    DEFAULT_DOCETL_MODEL,
    DocetlExecutionConfig,
    DocetlExecutionContext,
)
from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.reserved import KEEP_KEY, RESERVED_NAMES, ROWS_KEY
from agent_memory.adapters.docetl.sem_filter import build_filter_config
from agent_memory.adapters.docetl.sem_flat_map import build_flat_map_config
from agent_memory.adapters.docetl.sem_map import build_map_config
from agent_memory.policy.logical import ColumnSpec, QueryExpr

from adapter_support import CANNED_MODEL, CannedEndpoint, build_docetl_adapter, canned_model

# the discard port: a base URL that is valid to configure and impossible to call
_UNREACHABLE_BASE = "http://127.0.0.1:9/v1"
# only the row text carries these, so the instruction's own wording can never match the rule below
_KEEP_MARKER = "__keep__"
# a row carrying this gets an answer that parses as JSON and fails the declared type
_MALFORMED_MARKER = "__malformed__"


@pytest.fixture
def canned_endpoint() -> Iterator[CannedEndpoint]:
    """A row carrying the keep marker is kept; the malformed marker fails the declared type."""

    with canned_model(_KEEP_MARKER, malformed_markers=(_MALFORMED_MARKER,)) as endpoint:
        yield endpoint


def _context() -> DocetlExecutionContext:
    return DocetlExecutionContext(
        model=DEFAULT_DOCETL_MODEL,
        config=DocetlExecutionConfig(lm_api_base=_UNREACHABLE_BASE),
    )


def _filter_query(instruction: str = "the note reads {note}") -> QueryExpr:
    return QueryExpr(
        op="sem_filter",
        inputs=(QueryExpr(op="log"),),
        params={"instruction": instruction},
    )


def _adapter(endpoint: CannedEndpoint) -> DocetlAdapter:
    return build_docetl_adapter(endpoint.base_url)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (
            "{message} asks about {topic}",
            "{{ input.message }} asks about {{ input.topic }}",
        ),
        # a doubled brace is the escape for a literal one
        ("{{note}} is literal, {note} is a column", "{note} is literal, {{ input.note }} is a column"),
        # Jinja opens a construct only on a doubled brace
        ("{note} opens with a bare {", "{{ input.note }} opens with a bare {"),
    ],
)
def test_the_instruction_becomes_the_jinja_docetl_renders(
    instruction: str, expected: str
) -> None:
    assert render_prompt(instruction) == expected


@pytest.mark.parametrize(
    ("instruction", "reason"),
    [
        # nothing to interpolate, so DocETL could never fill it
        ("the message asks a question", "never interpolate"),
        # a qualified placeholder names another row: that is a join, not this operator
        ("{message:earlier} precedes {message:later}", "qualified"),
    ],
)
def test_an_instruction_that_cannot_become_a_prompt_is_refused(
    instruction: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        render_prompt(instruction)


def test_a_qualified_placeholder_is_a_context_root_for_an_operator_that_binds_it() -> None:
    """`equijoin` renders against `left`/`right`, so there the qualifier names the row."""

    assert render_prompt("{note:left} vs {note:right}", allow_tags=True) == (
        "{{ left.note }} vs {{ right.note }}"
    )


def test_completion_kwargs_are_forwarded_under_docetls_own_key() -> None:
    """Temperature and the reasoning switches reach litellm only through this passthrough."""

    assert "litellm_completion_kwargs" not in build_filter_config(_filter_query())

    pinned = build_filter_config(
        _filter_query(), litellm_completion_kwargs={"temperature": 0}
    )

    assert pinned["litellm_completion_kwargs"] == {"temperature": 0}


def test_docetl_accepts_the_generated_config() -> None:
    """The handshake, with DocETL's own validation running inside its constructor."""

    context = _context()
    config = build_filter_config(_filter_query("the note reads {note}"))
    operation = context.build_operation("filter")(
        context.handle, config, DEFAULT_DOCETL_MODEL, 1
    )

    assert type(operation).__name__ == "FilterOperation"
    # the operation read our key out of the config, so the config reached the engine
    assert operation._filter_key == KEEP_KEY


def test_docetl_rejects_a_malformed_config() -> None:
    """The control for the test above: construction does validate, so it cannot pass vacuously."""

    context = _context()
    config = build_filter_config(_filter_query("the note reads {note}"))
    config["output"] = {"schema": {"keep": "bool", "also_keep": "bool"}}
    factory = context.build_operation("filter")

    with pytest.raises(ValueError):
        factory(context.handle, config, DEFAULT_DOCETL_MODEL, 1)


def test_sem_filter_keeps_only_the_rows_the_model_accepts(canned_endpoint: CannedEndpoint) -> None:
    """End to end: our lowering, DocETL's operation, a real request, no model."""

    frame = pd.DataFrame({"note": [_KEEP_MARKER, "other"]}, index=[7, 8])

    result = _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})

    assert list(result.columns) == ["note"]
    assert list(result.index) == [7]


def test_sem_filter_preserves_the_index_and_hides_the_decision(
    canned_endpoint: CannedEndpoint,
) -> None:
    frame = pd.DataFrame({"note": [_KEEP_MARKER, _KEEP_MARKER]}, index=[7, 8])

    result = _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})

    assert list(result.index) == [7, 8]
    assert list(result.columns) == ["note"]
    assert result["note"].tolist() == [_KEEP_MARKER, _KEEP_MARKER]


def test_a_row_whose_answer_fails_validation_is_dropped(canned_endpoint: CannedEndpoint) -> None:
    """Pins the recorded divergence: a row DocETL cannot validate is dropped, not raised.

    Pinned here rather than in the real-API audit, because a real model cannot be made to answer
    wrongly on demand.
    """

    frame = pd.DataFrame({"note": [_MALFORMED_MARKER]}, index=[7])

    result = _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})

    assert list(result.columns) == ["note"]
    assert result.empty


def test_a_trace_directory_receives_one_event_per_operator(tmp_path: Path) -> None:
    """The adapter traces every operator it runs; nothing else checks that it writes."""

    with canned_model(_KEEP_MARKER) as endpoint:
        adapter = DocetlAdapter(
            model=CANNED_MODEL,
            config=DocetlExecutionConfig(
                lm_api_base=endpoint.base_url,
                semantic_trace_dir=tmp_path,
            ),
        )
        adapter.execute(_filter_query(), {"log": pd.DataFrame({"note": [_KEEP_MARKER, "other"]}, index=[7, 8])})

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    # the child query is executed first, so the log node traces before the filter
    assert [event["operator"] for event in events] == ["log", "sem_filter"]
    assert events[-1]["event_type"] == "operator_result"
    assert events[-1]["output_rows"] == 1
    assert events[-1]["output_columns"] == ["note"]


@pytest.fixture
def mapping_endpoint() -> Iterator[CannedEndpoint]:
    """Answer a column-filling operator with one fixed value for the declared column."""

    with canned_model(map_answers={"summary": "canned summary"}) as endpoint:
        yield endpoint


@pytest.fixture
def rows_endpoint() -> Iterator[CannedEndpoint]:
    """Answer a flat map with two emitted rows."""

    with canned_model(map_rows=({"entry": "a"}, {"entry": "b"})) as endpoint:
        yield endpoint


def _map_query() -> QueryExpr:
    return QueryExpr(
        op="sem_map",
        inputs=(QueryExpr(op="log"),),
        params={
            "instruction": "Summarise {message}",
            "output_cols": (ColumnSpec(name="summary", description="one line"),),
        },
    )


def _flat_map_query() -> QueryExpr:
    return QueryExpr(
        op="sem_flat_map",
        inputs=(QueryExpr(op="log"),),
        params={
            "instruction": "Split {message} into entries",
            "output_cols": (ColumnSpec(name="entry", description="one entry"),),
            "ordinal_col": "ord",
        },
    )


def test_docetl_accepts_the_generated_map_and_flat_map_configs() -> None:
    """A scalar schema and a `list[{...}]` schema both survive DocETL's own validation."""

    context = _context()
    factory = context.build_operation("map")

    for config in (
        build_map_config(_map_query()),
        build_flat_map_config(_flat_map_query(), rows_column=ROWS_KEY),
    ):
        operation = factory(context.handle, config, DEFAULT_DOCETL_MODEL, 1)
        assert type(operation).__name__ == "MapOperation"


def _two_column_map_query() -> QueryExpr:
    return QueryExpr(
        op="sem_map",
        inputs=(QueryExpr(op="log"),),
        params={
            "instruction": "Describe {message}",
            "output_cols": (
                ColumnSpec(name="summary", description="One line."),
                ColumnSpec(name="topic", description="One or two words."),
            ),
        },
    )


def test_a_single_column_map_sends_no_descriptions_like_the_native_path() -> None:
    """Mirror the sibling adapter's rule so both engines ask the same question about one policy.

    LOTUS lowers a one-column `sem_map` through LOTUS's native op, which takes only the instruction
    (`lotus/sem_map.py:34-39`), and sends descriptions only on its structured paths — `sem_map` with two or
    more columns, and `sem_flat_map` always. The DocETL side must match, or the same policy is described to
    one model and not the other.
    """

    assert "Fill these fields" not in build_map_config(_map_query())["prompt"]

    multiple = build_map_config(_two_column_map_query())
    assert "- summary: One line." in multiple["prompt"]
    assert "- topic: One or two words." in multiple["prompt"]

    flat = build_flat_map_config(_flat_map_query(), rows_column=ROWS_KEY)
    assert "- entry: one entry" in flat["prompt"]


def test_sem_map_adds_the_declared_column(mapping_endpoint: CannedEndpoint) -> None:
    """One row in, one row out, with the input columns and the index intact."""

    frame = pd.DataFrame({"message": ["alpha", "beta"]}, index=[7, 8])

    result = _adapter(mapping_endpoint).execute(_map_query(), {"log": frame})

    assert list(result.columns) == ["message", "summary"]
    assert list(result.index) == [7, 8]
    assert result["summary"].tolist() == ["canned summary", "canned summary"]


def test_sem_map_refuses_to_lose_a_row() -> None:
    """A 1:1 operator must not turn a dropped answer into a vanished memory."""

    with canned_model(malformed_markers=("alpha",)) as endpoint:
        frame = pd.DataFrame({"message": ["alpha", "beta"]}, index=[7, 8])

        with pytest.raises(RuntimeError, match="sem_map received 1 records"):
            _adapter(endpoint).execute(_map_query(), {"log": frame})


def test_a_flat_map_emits_one_row_per_entry_with_an_ordinal(
    rows_endpoint: CannedEndpoint,
) -> None:
    """Each emitted entry becomes a row that keeps the input row's index and its own ordinal."""

    frame = pd.DataFrame({"message": ["alpha", "beta"]}, index=[7, 8])

    result = _adapter(rows_endpoint).execute(_flat_map_query(), {"log": frame})

    assert list(result.columns) == ["message", "entry", "ord"]
    assert list(result.index) == [7, 7, 8, 8]
    assert result["entry"].tolist() == ["a", "b", "a", "b"]
    assert result["ord"].tolist() == [0, 1, 0, 1]
    assert result["message"].tolist() == ["alpha", "alpha", "beta", "beta"]


def test_a_flat_map_that_emits_nothing_adds_no_rows() -> None:
    """Zero entries is a legal answer: no rows, but the declared columns are still there."""

    with canned_model(map_rows=()) as endpoint:
        frame = pd.DataFrame({"message": ["alpha"]}, index=[7])

        result = _adapter(endpoint).execute(_flat_map_query(), {"log": frame})

    assert result.empty
    assert list(result.columns) == ["message", "entry", "ord"]


def test_a_flat_map_row_whose_list_fails_validation_is_refused() -> None:
    """DocETL drops a record whose answer fails the schema, and zero emitted rows is a legal result, so
    the record count is the only thing that can tell "the engine dropped it" from "the model emitted
    nothing".
    """

    frame = pd.DataFrame({"message": [_MALFORMED_MARKER, "beta"]}, index=[7, 8])

    with canned_model(
        map_rows=({"entry": "a"},), malformed_markers=(_MALFORMED_MARKER,)
    ) as endpoint:
        with pytest.raises(RuntimeError, match="sem_flat_map"):
            _adapter(endpoint).execute(_flat_map_query(), {"log": frame})


def _invariant_cases() -> list:
    return [
        pytest.param(
            _filter_query(),
            pd.DataFrame({"note": [_KEEP_MARKER, "other"]}, index=[7, 8]),
            ("note",),
            (_KEEP_MARKER,),
            {},
            id="sem_filter",
        ),
        pytest.param(
            _map_query(),
            pd.DataFrame({"message": ["alpha"]}, index=[7]),
            ("message", "summary"),
            (),
            {"map_answers": {"summary": "canned summary"}},
            id="sem_map",
        ),
        pytest.param(
            _flat_map_query(),
            pd.DataFrame({"message": ["alpha"]}, index=[7]),
            ("message", "entry", "ord"),
            (),
            {"map_rows": ({"entry": "a"},)},
            id="sem_flat_map",
        ),
    ]


@pytest.mark.parametrize(
    ("query", "frame", "expected", "markers", "answers"),
    _invariant_cases(),
)
def test_no_injected_column_escapes(
    query: QueryExpr,
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    markers: tuple[str, ...],
    answers: dict[str, Any],
) -> None:
    """Every semantic operator returns through `strip`, so no reserved key may reach a caller.

    The column list is asserted too: an equality catches a *missing* column, which a disjointness check
    would happily pass.
    """

    with canned_model(*markers, **answers) as endpoint:
        output = _adapter(endpoint).execute(query, {"log": frame})

    assert RESERVED_NAMES.isdisjoint(output.columns), list(output.columns)
    assert list(output.columns) == list(expected), list(output.columns)


def test_the_flat_map_asks_for_a_list_of_objects() -> None:
    """The schema string and the expansion are one mechanism, so pin the shape that makes them agree."""

    config = build_flat_map_config(_flat_map_query(), rows_column=ROWS_KEY)

    assert config["output"]["schema"][ROWS_KEY] == "list[{entry: string}]"
