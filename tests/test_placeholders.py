"""The brace grammar on its own: `policy/placeholders.py::split` against §1.

No adapter, no operator, no column set. This file reads instructions into tokens; what
a token means is decided by a resolver, and is tested where the operator supplies the
two sets (`test_docetl_placeholders.py` for the DocETL adapter).
"""

from __future__ import annotations

import pytest

from agent_memory.policy.placeholders import (
    PlaceholderError,
    Reference,
    Text,
    Token,
    compose,
    split,
)


# §1: one row per line of the splitting table. The scan is sequential, so a run of
# three braces is an escape followed by a reference, not a special case.
@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        # §1 `{{X}}` is a literal brace pair
        ("{{X}}", (Text("{X}"),)),
        # §1 `{X}` is a reference
        ("{X}", (Reference("X"),)),
        # §1 `{{{X}}}` is a literal brace, the reference, and a literal brace
        ("{{{X}}}", (Text("{"), Reference("X"), Text("}"))),
        # §1 the inner pair is the reference; the outer braces arrive before a `}`
        ('{"a": {X}}', (Text('{"a": '), Reference("X"), Text("}"))),
        # §1 a `{` that opens no reference is text
        ("{ and {X}", (Text("{ and "), Reference("X"))),
        # §1 a `}` that closes nothing is text
        ("X} and Y", (Text("X} and Y"),)),
    ],
)
def test_the_splitting_table(
    instruction: str, expected: tuple[Token, ...]
) -> None:
    assert split(instruction) == expected


# §1: the whole content between the braces is the name. Splitting admits the `:`
# segments a join generates; which one is a side is not read here.
@pytest.mark.parametrize(
    "name", ["fact", "fact_left", "_x9", "FACT", "fact:left", "fact:left:left"]
)
def test_a_name_is_the_whole_text_between_the_braces(name: str) -> None:
    assert split("{" + name + "}") == (Reference(name),)


# §1: writing tokens back is the inverse of reading them. Text is escaped again, so
# a brace that was read as text is read as text the next time round.
@pytest.mark.parametrize(
    "instruction",
    [
        "plain",
        "{note}",
        "{{note}}",
        "{{{note}}}",
        '{{"a": }} {note}',
        "a } b",
        "{{{{note}}}}",
    ],
)
def test_composing_keeps_the_meaning(instruction: str) -> None:
    assert split(compose(split(instruction))) == split(instruction)


# §1: a value and a field written back look like the reference that named them.
def test_composing_reassembles_a_reference() -> None:
    assert compose((Text("filling "), Reference("name"))) == "filling {name}"


# §1: a pair holding no name is a typo, refused before any set is consulted.
@pytest.mark.parametrize(
    "instruction",
    ["{}", '{"k": 1}', "{% if x %}", "{# note #}", "{a.b}", "{my col}", "{0}"],
)
def test_a_brace_pair_that_holds_no_name_is_refused(instruction: str) -> None:
    with pytest.raises(PlaceholderError, match="not a column"):
        split(instruction)


# §1: doubling is the only escape, and it is consumed here rather than left for the
# renderer, which would compile it instead.
def test_a_doubled_brace_becomes_one_literal_brace() -> None:
    assert split("{{note}}") == (Text("{note}"),)
    assert split("}} {note}") == (Text("} "), Reference("note"))


# §1: escaping twice keeps two braces, and the splitter has no opinion about them.
# Carrying a literal `{{` is the renderer's problem, so it is refused or wrapped where
# the prompt is written, not here.
@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("{{{{note}}}}", (Text("{{note}}"),)),
        ("{{%endraw%}}", (Text("{%endraw%}"),)),
        ("keep 50{% of them", (Text("keep 50{% of them"),)),
    ],
)
def test_an_opener_is_text_like_any_other(
    instruction: str, expected: tuple[Token, ...]
) -> None:
    assert split(instruction) == expected


# §1: a LaTeX fraction splits into two references, because `a` and `b` are names. It is
# refused later, where the columns are known, not here, where they are not.
def test_a_fraction_splits_into_references() -> None:
    assert split(r"\frac{a}{b}") == (
        Text("\\frac"),
        Reference("a"),
        Reference("b"),
    )


def test_an_empty_instruction_holds_no_token() -> None:
    assert split("") == ()


# The adapter contract keeps `ValueError`: every raiser is caught there by that type.
def test_the_error_is_a_value_error() -> None:
    assert issubclass(PlaceholderError, ValueError)
