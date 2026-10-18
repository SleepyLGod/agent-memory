"""The `sem_filter` lowering: instruction translation, the config handshake, and the operator offline.

Also the home for the other semantic operators' tests — a second one adds cases here rather than a file.
No model is called: the endpoint is a loopback server reached through `lm_api_base`, so DocETL's own
validation, parsing and record handling all run.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

import pandas as pd
import pytest

from agent_memory.adapters import DocetlAdapter
from agent_memory.adapters.docetl.context import (
    DEFAULT_DOCETL_MODEL,
    DocetlExecutionConfig,
    DocetlExecutionContext,
)
from agent_memory.adapters.docetl.prompt import render_prompt
from agent_memory.adapters.docetl.sem_filter import (
    FILTER_DECISION_KEY,
    build_filter_config,
)
from agent_memory.policy.logical import QueryExpr

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
    assert operation._filter_key == FILTER_DECISION_KEY


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
