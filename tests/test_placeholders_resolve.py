"""What a reference means: `policy/placeholders.py` resolve against §2 and §3.

No adapter, no prompt, no frame. The operator supplies the sets: `inputs` and
`outputs` for one input, `left` and `right` for a join. The resolver says which
token each reference becomes. Layout is tested where a prompt is written.
"""

from __future__ import annotations

import pytest

from agent_memory.policy.placeholders import (
    Field,
    PlaceholderError,
    Text,
    Value,
    join_suffixes,
    resolve_join,
    resolve_single,
    split,
)


# §2 ①: an input column names a value the operator supplies.
def test_a_reference_to_an_input_column_names_its_value() -> None:
    assert resolve_single(split("{note}"), inputs=("note",)) == (Value("note"),)


# §2: a suffix means nothing where one input is in scope, so the whole text is the
# column name and it is looked up as written.
def test_a_suffix_is_part_of_the_name_in_a_single_input() -> None:
    assert resolve_single(split("Is {fact:left} true?"), inputs=("fact:left",)) == (
        Text("Is "),
        Value("fact:left"),
        Text(" true?"),
    )


# §2 ②: a field the model must fill has no value, so the reference keeps naming the
# field rather than a column to read.
def test_a_reference_to_an_output_field_is_a_field() -> None:
    assert resolve_single(
        split("Extract the person, filling {name} from {text}"),
        inputs=("text",),
        outputs=("name",),
    ) == (
        Text("Extract the person, filling "),
        Field("name"),
        Text(" from "),
        Value("text"),
    )


# §2 ①: a name in both sets is an input, because the value already exists.
def test_an_input_column_wins_over_an_output_field() -> None:
    assert resolve_single(
        split("Rewrite {text}"), inputs=("text",), outputs=("text",)
    ) == (Text("Rewrite "), Value("text"))


# §2 ③: a name in neither set is a mistake, and the message has to name it.
@pytest.mark.parametrize("instruction", ["{nott}", "Is {fact:left} true?"])
def test_a_reference_to_neither_set_is_refused(instruction: str) -> None:
    with pytest.raises(PlaceholderError, match="input column"):
        resolve_single(split(instruction), inputs=("note",), outputs=("name",))


# §2: text between references is carried through untouched, in order, so layout keeps
# the instruction as written.
def test_text_passes_through_between_values() -> None:
    assert resolve_single(
        split("a {note} b {body} c"), inputs=("note", "body")
    ) == (
        Text("a "),
        Value("note"),
        Text(" b "),
        Value("body"),
        Text(" c"),
    )


# §2: the two sets are consulted only when a reference appears, so an instruction that
# names nothing resolves against nothing.
def test_an_instruction_with_no_reference_resolves_to_its_text() -> None:
    assert resolve_single(split("keep rows that look like a complaint"), inputs=()) == (
        Text("keep rows that look like a complaint"),
    )


# §3 ①: `left` and `right` name their own side, and the token keeps the column name
# without the suffix.
def test_a_side_reference_names_its_column_and_side() -> None:
    assert resolve_join(
        split("Is {fact:left} the same as {fact:right}?"),
        left=("fact",),
        right=("fact",),
    ) == (
        Text("Is "),
        Value("fact", side="left"),
        Text(" the same as "),
        Value("fact", side="right"),
        Text("?"),
    )


# §3 ①: an alias is a second name for the side that carries it, and the two sides are
# named independently.
def test_an_alias_names_its_side() -> None:
    assert resolve_join(
        split("Is {fact:earlier} the same as {fact:later}?"),
        left=("fact",),
        right=("fact",),
        suffixes=join_suffixes(left_alias="earlier", right_alias="later"),
    ) == (
        Text("Is "),
        Value("fact", side="left"),
        Text(" the same as "),
        Value("fact", side="right"),
        Text("?"),
    )


# §3 ①: the side is the text after the last colon, so stacked suffixes keep the
# segments they grew on the way.
def test_the_side_is_read_from_the_last_colon() -> None:
    assert resolve_join(
        split("Is {fact:left:left} still true?"),
        left=("fact", "fact:left"),
        right=("fact",),
    ) == (Text("Is "), Value("fact:left", side="left"), Text(" still true?"))


# §3 ①: the side is checked first, so a column that happens to end in a side word does
# not shadow the side.
def test_a_side_wins_over_a_column_of_the_same_name() -> None:
    assert resolve_join(
        split("Is {fact:left} the same as {fact:right}?"),
        left=("fact", "fact:left"),
        right=("fact",),
    ) == (
        Text("Is "),
        Value("fact", side="left"),
        Text(" the same as "),
        Value("fact", side="right"),
        Text("?"),
    )


# §3 ②: a bare name belongs to the one side that has it.
def test_a_whole_name_belongs_to_the_side_that_has_it() -> None:
    assert resolve_join(
        split("Does {valid_at} rule out {fact:right}?"),
        left=("id", "fact", "valid_at"),
        right=("id", "fact"),
    ) == (
        Text("Does "),
        Value("valid_at", side="left"),
        Text(" rule out "),
        Value("fact", side="right"),
        Text("?"),
    )


# §3: a suffix the join publishes is not a side here. `earlier` is part of a column
# name, left over from an earlier join, and this join carries no such alias.
def test_a_published_suffix_is_part_of_the_name() -> None:
    assert resolve_join(
        split("Is {fact:earlier} still true?"),
        left=("fact:earlier",),
        right=("fact",),
    ) == (Text("Is "), Value("fact:earlier", side="left"), Text(" still true?"))


# §3 ③: one reference cannot mean two different values, so it is refused until it is
# qualified.
def test_a_name_both_sides_own_is_refused() -> None:
    with pytest.raises(PlaceholderError, match="both sides"):
        resolve_join(
            split("Is {note} the same on both rows?"),
            left=("note", "left_only"),
            right=("note", "right_only"),
        )


# §3 ④: a name no side has is a mistake, not a whole-row comparison.
@pytest.mark.parametrize("instruction", ["{missing}", "{missing:left}"])
def test_an_unknown_join_reference_is_refused(instruction: str) -> None:
    with pytest.raises(PlaceholderError, match="neither side"):
        resolve_join(split(instruction), left=("fact",), right=("fact",))


# §3: a join instruction may name nothing; whether that compares whole rows is a
# layout decision, so the resolver only reports the text.
def test_a_join_instruction_with_no_reference_resolves_to_its_text() -> None:
    assert resolve_join(
        split("Are these two rows about the same thing?"),
        left=("fact",),
        right=("fact",),
    ) == (Text("Are these two rows about the same thing?"),)


# §3: `left` and `right` are always in the map, and an alias adds to them.
def test_the_suffix_map_starts_with_the_two_side_names() -> None:
    assert join_suffixes() == {"left": "left", "right": "right"}
    assert join_suffixes(left_alias="earlier", right_alias="later") == {
        "left": "left",
        "right": "right",
        "earlier": "left",
        "later": "right",
    }


# §3: an alias repeating the built-in name of its own side is redundant, not
# conflicting.
def test_an_alias_may_repeat_the_name_of_its_own_side() -> None:
    assert join_suffixes(left_alias="left") == {"left": "left", "right": "right"}
    assert join_suffixes(right_alias="right") == {"left": "left", "right": "right"}


# §3: a suffix that would name both inputs cannot say which one is meant, and the
# check happens when the join is built, before any reference is read.
def test_a_suffix_may_not_name_two_inputs() -> None:
    with pytest.raises(PlaceholderError, match="two inputs"):
        join_suffixes(left_alias="right")

    with pytest.raises(PlaceholderError, match="two inputs"):
        join_suffixes(right_alias="left")

    with pytest.raises(PlaceholderError, match="two inputs"):
        join_suffixes(left_alias="side", right_alias="side")


# §3: an alias is a variable name, so a suffix can always be written in a reference.
@pytest.mark.parametrize("alias", ["fact:left", "2x", "my alias", ""])
def test_an_alias_that_cannot_be_written_is_refused(alias: str) -> None:
    with pytest.raises(PlaceholderError, match="not a variable name"):
        join_suffixes(left_alias=alias)


# §3: an alias renames how a side is named, never the column it resolves to.
def test_an_alias_does_not_rename_the_column() -> None:
    assert resolve_join(
        split("Is {fact:earlier} the same as {fact:right}?"),
        left=("fact",),
        right=("fact",),
        suffixes=join_suffixes(left_alias="earlier"),
    ) == (
        Text("Is "),
        Value("fact", side="left"),
        Text(" the same as "),
        Value("fact", side="right"),
        Text("?"),
    )
