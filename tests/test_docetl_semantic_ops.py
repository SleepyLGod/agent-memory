"""Row-local lowerings: the config handshake, and `sem_filter` / `sem_map` / `sem_flat_map` end to end.

The instruction grammar is a pure function and is tested in `test_docetl_placeholders.py`. No model here: a
loopback endpoint answers through `lm_api_base`, so DocETL's own validation, parsing and record handling
run. An operator carrying a cross-row contract belongs in its own file, as `sem_join` has.
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
from agent_memory.adapters.docetl.reserved import INDEX_KEY, KEEP_KEY, RESERVED_NAMES, ROWS_KEY
from agent_memory.adapters.docetl.sem_filter import build_filter_config
from agent_memory.adapters.docetl.sem_flat_map import build_flat_map_config
from agent_memory.adapters.docetl.sem_map import build_map_config
from agent_memory.adapters.docetl.strict import strict_equijoin, strict_filter
from agent_memory.policy.logical import ColumnSpec, QueryExpr

from adapter_support import CANNED_MODEL, CannedEndpoint, build_docetl_adapter, canned_model

# the discard port: a base URL that is valid to configure and impossible to call
_UNREACHABLE_BASE = "http://127.0.0.1:9/v1"
# only the row text carries these, so the instruction's own wording can never match the rule below
_KEEP_MARKER = "__keep__"
# a row carrying this gets an answer that parses as JSON and fails the declared type
_MALFORMED_MARKER = "__malformed__"
# 2**53 + 1, the first integer float64 cannot hold; a carried column must not round it
_BIG = 9007199254740993


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


def test_completion_kwargs_are_forwarded_under_docetls_own_key() -> None:
    """Temperature and the reasoning switches reach litellm only through this passthrough."""

    assert "litellm_completion_kwargs" not in build_filter_config(_filter_query(), columns=("note",))

    context = DocetlExecutionContext(
        model=DEFAULT_DOCETL_MODEL,
        config=DocetlExecutionConfig(litellm_completion_kwargs={"temperature": 0}),
    )
    merged = context.with_op_defaults(build_filter_config(_filter_query(), columns=("note",)))

    assert merged["litellm_completion_kwargs"] == {"temperature": 0}
    assert merged["skip_on_error"] is False
    # the answer count means "one answer per row" only because neither knob reaches the spec:
    # a cascade never calls `_handle_result` (`filter.py:130-131`), and a filter's `limit` stops the
    # run after the rows it kept (`filter.py:99-100`)
    assert {"cascade", "limit"}.isdisjoint(merged)


def test_docetl_accepts_the_generated_config() -> None:
    """The handshake, with DocETL's own validation running inside its constructor."""

    context = _context()
    config = build_filter_config(_filter_query("the note reads {note}"), columns=("note",))
    operation = strict_filter(index_key=INDEX_KEY)(
        context.handle, config, DEFAULT_DOCETL_MODEL, 1
    )

    # a subclass, but the operation `sem_filter` corresponds to: `filter`'s own class is constructed
    assert isinstance(operation, context.build_operation("filter"))
    # the declared field is the one the model fills, so the answer lands under the adapter's key
    assert operation.config["output"]["schema"] == {KEEP_KEY: "bool"}


def test_docetl_rejects_a_malformed_config() -> None:
    """The control for the test above: construction does validate, so it cannot pass vacuously."""

    context = _context()
    config = build_filter_config(_filter_query("the note reads {note}"), columns=("note",))
    # a filter takes exactly one bool key; the two-key answer a map allows is not a filter
    config["output"] = {"schema": {KEEP_KEY: "bool", "reason": "str"}}
    factory = strict_filter(index_key=INDEX_KEY)

    with pytest.raises(ValueError):
        factory(context.handle, config, DEFAULT_DOCETL_MODEL, 1)


def test_a_class_for_another_operation_is_refused() -> None:
    """`operation_cls` substitutes a class for the operation named, not a class of any kind."""

    context = _context()
    config = build_filter_config(_filter_query(), columns=("note",))
    equijoin = strict_equijoin(left_id_key="left_id", right_id_key="right_id")

    with pytest.raises(TypeError, match="takes a subclass of FilterOperation"):
        context.run_operation("filter", config, [], operation_cls=equijoin)


def test_an_operation_cost_reaches_the_handle() -> None:
    """`execute()` returns cost and nothing outside adds it, so the context does, as `runner.py:617` does."""

    from docetl.operations.filter import FilterOperation

    class PricedFilter(FilterOperation):
        def execute(self, *args: Any, **kwargs: Any) -> Any:
            return [], 2.5

    context = _context()
    config = build_filter_config(_filter_query(), columns=("note",))
    context.run_operation("filter", config, [], operation_cls=PricedFilter)
    context.run_operation("filter", config, [], operation_cls=PricedFilter)

    assert context.handle.total_cost == 5.0


def test_sem_filter_keeps_only_the_rows_the_model_accepts(canned_endpoint: CannedEndpoint) -> None:
    """End to end: our lowering, DocETL's operation, a real request, no model."""

    frame = pd.DataFrame({"note": [_KEEP_MARKER, "other"]}, index=[7, 8])

    result = _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})

    assert list(result.columns) == ["note"]
    assert list(result.index) == [7]


def test_a_column_whose_name_carries_a_colon_reaches_the_prompt(
    canned_endpoint: CannedEndpoint,
) -> None:
    """A join publishes `note:left` / `note:right`; a later filter names both and nothing renames them.

    The marker rides on `note:right`, so a placeholder rendered empty would drop the row: the assertion
    is that the whole name became a key, not that part of it did.
    """

    frame = pd.DataFrame(
        {"note:left": ["alpha", "beta"], "note:right": [_KEEP_MARKER, "gamma"]},
        index=[7, 8],
    )
    query = _filter_query("Does {note:right} contradict {note:left}?")

    result = _adapter(canned_endpoint).execute(query, {"log": frame})

    assert list(result.columns) == ["note:left", "note:right"]
    assert list(result.index) == [7]


def test_sem_filter_preserves_the_index_and_hides_the_decision(
    canned_endpoint: CannedEndpoint,
) -> None:
    frame = pd.DataFrame({"note": [_KEEP_MARKER, _KEEP_MARKER]}, index=[7, 8])

    result = _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})

    assert list(result.index) == [7, 8]
    assert list(result.columns) == ["note"]
    assert result["note"].tolist() == [_KEEP_MARKER, _KEEP_MARKER]


def test_a_row_whose_answer_fails_validation_is_refused(canned_endpoint: CannedEndpoint) -> None:
    """A filter has two outcomes, so it may not answer with the third: no answer at all.

    DocETL drops a record whose answer fails the declared type and reports nothing, and `filter`
    publishes only the rows it kept, so the count is not in the output. `strict_filter` counts the
    answers the engine actually got, which is what turns a vanished row into a failure.
    """

    frame = pd.DataFrame({"note": [_MALFORMED_MARKER]}, index=[7])

    with pytest.raises(RuntimeError, match="sem_filter answered 0 of 1 rows"):
        _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})


def test_a_rejected_row_is_not_mistaken_for_a_lost_answer(
    canned_endpoint: CannedEndpoint,
) -> None:
    """The count decides, not the verdict: a row the model turned down is a legal result.

    The two rows differ only in why they are absent from the answer, so the raise has to name the one
    whose answer never arrived.
    """

    frame = pd.DataFrame({"note": [_MALFORMED_MARKER, "other"]}, index=[7, 8])

    with pytest.raises(RuntimeError, match=r"sem_filter answered 1 of 2 rows.*missing \[7\]"):
        _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})


def test_an_answer_without_the_decision_field_is_refused() -> None:
    """`filter.py:103` reads the decision with `bool(result.get(...))`, so its shape is ours to check.

    The engine's own type gate drops such a row before `_handle_result`, which is what the count sees;
    this asserts the second half directly, because that gate sits on a path `gleaning` and `agent` skip.
    """

    context = _context()
    config = build_filter_config(_filter_query(), columns=("note",))
    operation = strict_filter(index_key=INDEX_KEY)(
        context.handle, config, DEFAULT_DOCETL_MODEL, 1
    )

    with pytest.raises(RuntimeError, match=f"without {KEEP_KEY!r}"):
        operation._handle_result({INDEX_KEY: 7, "note": "x"})


@pytest.mark.parametrize("answered", ["maybe", 1, None])
def test_an_answer_that_is_not_a_boolean_is_refused(answered: Any) -> None:
    """A present but wrong decision is not a refusal either: `bool("maybe")` reads as kept."""

    context = _context()
    config = build_filter_config(_filter_query(), columns=("note",))
    operation = strict_filter(index_key=INDEX_KEY)(
        context.handle, config, DEFAULT_DOCETL_MODEL, 1
    )

    with pytest.raises(RuntimeError, match="which is not a boolean"):
        operation._handle_result({INDEX_KEY: 7, "note": "x", KEEP_KEY: answered})


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
        build_map_config(_map_query(), columns=("message",)),
        build_flat_map_config(_flat_map_query(), columns=("message",), rows_column=ROWS_KEY),
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
    """Mirror LOTUS's rule so both engines ask the same question about one policy.

    LOTUS lowers a one-column `sem_map` through LOTUS's native op, which takes only the instruction
    (`lotus/sem_map.py:34-39`), and sends descriptions only on its structured paths — `sem_map` with two or
    more columns, and `sem_flat_map` always. The DocETL side must match, or the same policy is described to
    one model and not the other.
    """

    assert "Fill these fields" not in build_map_config(_map_query(), columns=("message",))["prompt"]

    multiple = build_map_config(_two_column_map_query(), columns=("message",))
    assert "- summary: One line." in multiple["prompt"]
    assert "- topic: One or two words." in multiple["prompt"]

    flat = build_flat_map_config(_flat_map_query(), columns=("message",), rows_column=ROWS_KEY)
    assert "- entry: one entry" in flat["prompt"]


def test_sem_filter_carries_a_large_integer_column_exact(
    canned_endpoint: CannedEndpoint,
) -> None:
    """A filter that keeps every row must not round a carried integer via the record echo."""

    frame = pd.DataFrame(
        {
            "note": [_KEEP_MARKER, _KEEP_MARKER],
            "fact_id": pd.array([_BIG, pd.NA], dtype="Int64"),
        },
        index=[7, 8],
    )

    result = _adapter(canned_endpoint).execute(_filter_query(), {"log": frame})

    assert str(result["fact_id"].dtype) == "Int64"
    assert result["fact_id"].iloc[0] == _BIG
    assert pd.isna(result["fact_id"].iloc[1])


def test_sem_map_carries_a_large_integer_column_exact(
    mapping_endpoint: CannedEndpoint,
) -> None:
    """Only the declared output comes from the answer; a carried column comes from the source."""

    frame = pd.DataFrame(
        {"message": ["alpha"], "fact_id": pd.array([_BIG], dtype="Int64")},
        index=[7],
    )

    result = _adapter(mapping_endpoint).execute(_map_query(), {"log": frame})

    assert list(result.columns) == ["message", "fact_id", "summary"]
    assert str(result["fact_id"].dtype) == "Int64"
    assert result["fact_id"].iloc[0] == _BIG
    assert result["summary"].iloc[0] == "canned summary"


def test_sem_flat_map_carries_a_large_integer_column_exact(
    rows_endpoint: CannedEndpoint,
) -> None:
    """One input row emits two, and both copies are the source row's: exact, and still the source's
    dtype rather than one a rebuilt row would have inferred.
    """

    frame = pd.DataFrame(
        {"message": ["alpha"], "fact_id": pd.array([_BIG], dtype="Int64")},
        index=[7],
    )

    result = _adapter(rows_endpoint).execute(_flat_map_query(), {"log": frame})

    assert str(result["fact_id"].dtype) == "Int64"
    assert result["fact_id"].tolist() == [_BIG, _BIG]
    assert list(result.index) == [7, 7]


# §2: `I` and `O` are the operator's vocabulary. `sem_filter`'s only output is the adapter's own
# `keep_row`, which no policy may name, so a name it has no value for is refused rather than
# accepted as something to fill.
def test_a_filter_declares_no_output_the_policy_can_name() -> None:
    with pytest.raises(ValueError, match="input column"):
        build_filter_config(_filter_query("Keep rows about {topic}"), columns=("note",))


# §2: `sem_map` declares its outputs, so the same kind of name is a field the model
# must fill and the instruction keeps it as written.
def test_a_map_declares_the_fields_it_fills() -> None:
    query = QueryExpr(
        op="sem_map",
        inputs=(QueryExpr(op="log"),),
        params={
            "instruction": "Produce {summary} about {message}",
            "output_cols": (ColumnSpec(name="summary", description="One line."),),
        },
    )

    assert build_map_config(query, columns=("message",))["prompt"] == (
        "Produce {summary} about {{ input['message'] }}"
    )


# §2: `I` is `input_cols` when the query gives one. Only the builder can make that
# choice: supplying every source column instead would resolve this reference.
def test_a_builder_supplies_the_selected_inputs_not_every_column(
    canned_endpoint: CannedEndpoint,
) -> None:
    frame = pd.DataFrame({"note": ["a"], "secret": ["b"]})
    query = QueryExpr(
        op="sem_map",
        inputs=(QueryExpr(op="log"),),
        params={
            "instruction": "Summarise {secret}",
            "input_cols": ("note",),
            "output_cols": (ColumnSpec(name="summary", description="One line."),),
        },
    )

    with pytest.raises(ValueError, match="input column"):
        _adapter(canned_endpoint).execute(query, {"log": frame})

    assert canned_endpoint.requests == []


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
    """No operator publishes a reserved key: the keys are consumed where the records are read, so the
    frame a caller receives holds policy columns only.

    The column list is asserted too: an equality catches a *missing* column, which a disjointness check
    would happily pass.
    """

    with canned_model(*markers, **answers) as endpoint:
        output = _adapter(endpoint).execute(query, {"log": frame})

    assert RESERVED_NAMES.isdisjoint(output.columns), list(output.columns)
    assert list(output.columns) == list(expected), list(output.columns)


def test_the_flat_map_asks_for_a_list_of_objects() -> None:
    """The schema string and the expansion are one mechanism, so pin the shape that makes them agree."""

    config = build_flat_map_config(_flat_map_query(), columns=("message",), rows_column=ROWS_KEY)

    assert config["output"]["schema"][ROWS_KEY] == "list[{entry: string}]"
