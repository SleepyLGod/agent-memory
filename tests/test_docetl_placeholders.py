"""Executable form of `docs/design/instruction_placeholders.md` for the DocETL adapter.

A `{...}` written in an instruction is a placeholder. Splitting the text turns it
into either a literal brace or a *reference*, and a reference is then resolved
against the two sets the operator supplies: the inputs that have a value, and the
output fields the model must fill. This file pins both stages; the operator, not
the text, decides which rule set applies.

| section | rule | test |
| --- | --- | --- |
| §0 | the operator decides, not the text | `test_the_same_text_names_a_column_in_a_single_input_and_a_side_in_a_join` |
| §1 | braces are split by a greedy scan | `test_braces_are_split_by_a_greedy_scan` |
| §1 | a brace pair that names nothing is refused | `test_a_brace_pair_that_is_not_a_name_is_refused` |
| §1 | a literal run holding an opener is wrapped | `test_a_doubled_brace_shows_two_braces` |
| §7 | a raw terminator cannot be written | `test_a_raw_terminator_cannot_be_written` |
| §2 ① | a reference names one whole column name | `test_a_reference_names_the_column_with_that_whole_name` |
| §2 ① | an input column wins over an output field | `test_an_input_column_wins_over_an_output_field_of_the_same_name` |
| §2 ② | a field the model must fill stays as text | `test_a_field_the_model_must_fill_stays_as_text` |
| §2 ③ | a name in neither set, or outside `input_cols`, is refused | `test_a_reference_that_names_neither_set_is_refused` |
| §2 | an instruction may name no column | `test_an_instruction_may_name_no_column` |
| §2 | a column the instruction does not name is still supplied | `test_a_column_the_instruction_does_not_name_is_still_supplied` |
| §3 ① | a side is named by `left`, `right` or an input alias | `test_a_side_reference_names_left_right_or_an_alias` |
| §3 ② | a whole name must belong to exactly one side | `test_a_whole_name_reference_needs_exactly_one_side` |
| §3 ③ | a name both sides own is refused | `test_a_name_on_both_sides_is_refused` |
| §3 ④ | a name neither side owns is refused | `test_an_unknown_join_reference_is_refused` |
| §3 | the split takes the last colon | `test_the_split_takes_the_last_colon` |
| §3 ① | a side is checked before a whole name | `test_a_side_wins_over_a_same_named_column` |
| §3 ① | an alias may repeat its own side's name | `test_an_alias_may_repeat_the_name_of_its_own_side` |
| §3 ① | an alias may not point at the other side | `test_an_alias_may_not_point_at_the_other_side` |
| §3 ① | two inputs may not share an alias | `test_two_inputs_may_not_share_an_alias` |
| §3 ① | an alias does not rename the column | `test_an_alias_does_not_rename_the_resolved_column` |
| §3 | a join instruction may name nothing | `test_a_join_instruction_may_compare_whole_rows` |
| §3 | references on one side make the join compare whole rows | `test_a_join_whose_references_land_on_one_side_compares_whole_rows` |
| §3 | an alias from an earlier join is not inherited | `test_an_alias_from_an_earlier_join_is_not_inherited` |
| §7 | a name containing a brace cannot be written | `test_a_name_containing_a_brace_cannot_be_written` |
| §3 | a rewritten instruction is read by the same splitter | `test_a_rewritten_instruction_is_resolved_by_the_same_splitter` |

Every case runs unmasked. The functions called here are the ones the adapter calls, so a
failure names a rule the implementation does not follow; a mark belongs to the
individual case rather than the test, because a parametrised test is often green in most
rows and unimplemented in one.

`§6` is deliberately absent: it is about the columns a join publishes, not about
reading a reference, and belongs with the join's execution tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_memory.adapters.docetl.prompt import (
    compared_columns,
    render_join_condition,
    render_prompt,
)
from agent_memory.planner.rules import DifferentialInstructionRewriter
from agent_memory.policy.logical import ColumnSpec
from agent_memory.policy.placeholders import JoinScope


def _scope(
    *,
    left_columns: tuple[str, ...] = (),
    right_columns: tuple[str, ...] = (),
    left_alias: str | None = None,
    right_alias: str | None = None,
) -> JoinScope:
    """Build a join scope from the columns-and-alias keywords the cases are written in."""

    return JoinScope.of(
        left_columns, right_columns, left_alias=left_alias, right_alias=right_alias
    )


def join_columns(instruction: str, **kwargs: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve a join instruction, including the alias keywords §3 ① adds."""

    scope = compared_columns(instruction, _scope(**kwargs))
    return scope.left, scope.right


def render_input(instruction: str, **kwargs: Any) -> str:
    """Render a single-input prompt, including the output set §2 ② needs."""

    return render_prompt(instruction, **kwargs)


# §1: the text is read left to right. `{{` and `}}` are literal braces, a `{`
# whose `}` comes before any other `{` opens a reference, and a brace that does
# neither is text.
@pytest.mark.parametrize(
    ("instruction", "columns", "expected"),
    [
        # §1 a plain reference
        ("{note}", ("note",), "{{ input['note'] }}"),
        # §1 a doubled brace is literal, even where it holds a name
        ("{{note}} {note}", ("note",), "{note} {{ input['note'] }}"),
        # §1 a literal brace on each side of a reference; the leading one is wrapped,
        # because bare it would join the reference's `{{` into an expression
        (
            "{{{note}}}",
            ("note",),
            "{% raw %}{{% endraw %}{{ input['note'] }}}",
        ),
        # §1 outer braces are literal because another `{` arrives first
        ('{"a": {note}}', ("note",), '{"a": {{ input[\'note\'] }}}'),
        # §1 a `{` with no closer is text
        ("{note} and {note", ("note",), "{{ input['note'] }} and {note"),
        # §1 the same, with a reference after it
        ("{a{b}", ("b",), "{a{{ input['b'] }}"),
        # §1 a doubled brace that is not opened by anything
        ("}} {note}", ("note",), "} {{ input['note'] }}"),
        # §1 an unclosed `{{` is one literal brace
        ("{{ {note}", ("note",), "{ {{ input['note'] }}"),
        # §1 a single brace with no closer before another `{`
        ("{ {note}", ("note",), "{ {{ input['note'] }}"),
    ],
)
def test_braces_are_split_by_a_greedy_scan(
    instruction: str, columns: tuple[str, ...], expected: str
) -> None:
    assert render_input(instruction, columns=columns) == expected


# §2 ①: the whole content between the braces is the column name. §1 decides which
# contents are names; what survives is read whole, including its `:` segments.
@pytest.mark.parametrize(
    ("instruction", "columns", "expected"),
    [
        # §2 ① an ordinary name
        ("{note}", ("note",), "{{ input['note'] }}"),
        # §2 ① dotted access would resolve this one to `dict.items`
        ("{items}", ("items",), "{{ input['items'] }}"),
        # §2 ① a name a join published earlier
        ("{fact:later}", ("fact:later",), "{{ input['fact:later'] }}"),
        # §2 ① three segments, still one name
        ("{fact:left:left}", ("fact:left:left",), "{{ input['fact:left:left'] }}"),
    ],
)
def test_a_reference_names_the_column_with_that_whole_name(
    instruction: str, columns: tuple[str, ...], expected: str
) -> None:
    assert render_input(instruction, columns=columns) == expected


# §1: a brace pair holds a name only when it has the variable shape. Anything else
# is a typo, not a column, and is refused before any set is looked at.
@pytest.mark.parametrize(
    ("instruction", "columns"),
    [
        # §1 an empty pair names nothing
        ("{}", ("note",)),
        # §1 an object literal is not a name
        ('{"k": 1}', ("note",)),
        # §1 a Jinja statement would run instead of being shown
        ("{% if x %}", ("note",)),
        # §1 a Jinja comment is deleted without a word
        ("{# note #}", ("note",)),
        # §1 a dot would read as attribute access in Jinja
        ("{a.b}", ("a.b",)),
        # §1 a space cannot be written as a variable
        ("{my col}", ("my col",)),
    ],
)
def test_a_brace_pair_that_is_not_a_name_is_refused(
    instruction: str, columns: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="not a column"):
        render_input(instruction, columns=columns)


# §1: escaping twice shows two braces, and no brace a user writes ever reaches the
# template: the literal run is wrapped, so the renderer shows it instead of compiling
# it. The empty column set keeps the data block out, leaving only the wrapper.
def test_a_doubled_brace_shows_two_braces() -> None:
    assert render_input("{{{{note}}}}", columns=()) == "{% raw %}{{note}}{% endraw %}"


# §7: a raw block ends at the first terminator Jinja recognises
# (`jinja2/lexer.py:571-579`), so text that spells one cannot be carried at all.
# Refusing it keeps the rest of the instruction out of the template.
def test_a_raw_terminator_cannot_be_written() -> None:
    with pytest.raises(ValueError, match="endraw"):
        render_input("{{%endraw%}}", columns=())


# §2 ②: a field the model must fill has no value yet, so it stays as text while
# the values around it are substituted.
def test_a_field_the_model_must_fill_stays_as_text() -> None:
    assert render_input(
        "Extract the person, filling {name} from {text}",
        columns=("text",),
        outputs=("name",),
    ) == "Extract the person, filling {name} from {{ input['text'] }}"


# §2 ①: an input column wins over an output field of the same name, because the
# value already exists.
def test_an_input_column_wins_over_an_output_field_of_the_same_name() -> None:
    assert render_input(
        "Rewrite {text}",
        columns=("text",),
        outputs=("text",),
    ) == "Rewrite {{ input['text'] }}"


# §2 ③: a name in neither set is refused, and so is a name the frame carries but
# the operator did not select: the renderer cannot see the frame, so it names both.
@pytest.mark.parametrize(
    ("instruction", "columns"),
    [
        # §2 ③ a typo
        ("{nott}", ("note",)),
        # §2 ③ a real column the operator left out of `input_cols`
        ("{note}", ("other",)),
    ],
)
def test_a_reference_that_names_neither_set_is_refused(
    instruction: str, columns: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="input column"):
        render_input(instruction, columns=columns)


# §2: a reference is one way to name a column, not a requirement. A prompt that
# names nothing still reaches the model with the operator's input attached.
def test_an_instruction_may_name_no_column() -> None:
    assert render_input("the message asks a question", columns=("note",)) == (
        "the message asks a question\n\nnote: {{ input['note'] }}"
    )


# §2: a reference is not the only path a column takes. One the instruction leaves
# out is supplied after it, exactly once.
def test_a_column_the_instruction_does_not_name_is_still_supplied() -> None:
    assert render_input("Summarize {title}", columns=("title", "body")) == (
        "Summarize {{ input['title'] }}\n\nbody: {{ input['body'] }}"
    )


# §3 ①: a side is `left`, `right`, or the alias that side carries. The two sides
# are named independently, so one may use the built-in name and the other its
# alias.
@pytest.mark.parametrize(
    ("instruction", "left_alias", "right_alias", "expected"),
    [
        # §3 ① both sides by their built-in names
        (
            "Is {fact:left} the same as {fact:right}?",
            None,
            None,
            (("fact",), ("fact",)),
        ),
        # §3 ① both sides by their aliases
        (
            "Is {fact:earlier} the same as {fact:later}?",
            "earlier",
            "later",
            (("fact",), ("fact",)),
        ),
        # §3 ① the left by its built-in name, the right by its alias
        (
            "Is {fact:left} the same as {fact:later}?",
            None,
            "later",
            (("fact",), ("fact",)),
        ),
    ],
)
def test_a_side_reference_names_left_right_or_an_alias(
    instruction: str,
    left_alias: str | None,
    right_alias: str | None,
    expected: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    assert (
        join_columns(
            instruction,
            left_columns=("fact",),
            right_columns=("fact",),
            left_alias=left_alias,
            right_alias=right_alias,
        )
        == expected
    )


# §3 ②: with no side to say, a bare name belongs to the one side that has it.
@pytest.mark.parametrize(
    ("instruction", "left_columns", "right_columns", "expected"),
    [
        # §3 ② each name owned by exactly one side
        (
            "Does {fact} rule out {other}?",
            ("fact",),
            ("other",),
            (("fact",), ("other",)),
        ),
        # §3 ② a name that only the left side has, and that contains a colon
        pytest.param(
            "Is {fact:later} the same as {fact:right}?",
            ("fact:later",),
            ("fact",),
            (("fact:later",), ("fact",)),
        ),
    ],
)
def test_a_whole_name_reference_needs_exactly_one_side(
    instruction: str,
    left_columns: tuple[str, ...],
    right_columns: tuple[str, ...],
    expected: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    assert (
        join_columns(
            instruction,
            left_columns=left_columns,
            right_columns=right_columns,
        )
        == expected
    )


# §3 ③: one reference cannot mean two different columns.
def test_a_name_on_both_sides_is_refused() -> None:
    with pytest.raises(ValueError, match="both sides"):
        join_columns(
            "Is {note} the same on both rows?",
            left_columns=("note", "left_only"),
            right_columns=("note", "right_only"),
        )


# §3 ④: a name no side owns is a mistake, not a whole-row comparison.
@pytest.mark.parametrize(
    "instruction",
    [
        # §3 ④ a bare name
        "{missing}",
        # §3 ④ a sided name whose base is not on that side
        "{missing:left}",
    ],
)
def test_an_unknown_join_reference_is_refused(instruction: str) -> None:
    with pytest.raises(ValueError, match="neither side"):
        join_columns(
            instruction,
            left_columns=("fact",),
            right_columns=("fact",),
        )


# §3: the last colon is the side, everything before it is the name.
def test_the_split_takes_the_last_colon() -> None:
    assert join_columns(
        "Is {fact:left:left} the same as {fact:right}?",
        left_columns=("fact", "fact:left"),
        right_columns=("fact",),
    ) == (("fact:left",), ("fact",))

    # the rendered condition keeps the name it named, without the side
    assert render_join_condition(
        "{fact:left:left} vs {fact:right}",
        _scope(left_columns=("fact", "fact:left"), right_columns=("fact",)),
    ) == "fact:left vs fact"


# §3 ①: an alias and a column name can both fit; the side is checked first, so
# the reference is the side's column and `{fact:left:left}` is the column.
def test_a_side_wins_over_a_same_named_column() -> None:
    assert join_columns(
        "Is {fact:left} the same as {fact:right}?",
        left_columns=("fact", "fact:left"),
        right_columns=("fact",),
        left_alias="left",
    ) == (("fact",), ("fact",))


# §3: repeating the built-in name of the same side is redundant, not conflicting.
def test_an_alias_may_repeat_the_name_of_its_own_side() -> None:
    assert join_columns(
        "Is {fact:left} the same as {fact:right}?",
        left_columns=("fact",),
        right_columns=("fact",),
        left_alias="left",
    ) == (("fact",), ("fact",))


# §3: a name that maps to both inputs cannot say which one is meant.
def test_an_alias_may_not_point_at_the_other_side() -> None:
    with pytest.raises(ValueError, match="two inputs"):
        join_columns(
            "Is {fact:right} the same as {fact:right}?",
            left_columns=("fact",),
            right_columns=("fact",),
            left_alias="right",
        )


# §3: the same conflict, with both inputs claiming one alias.
def test_two_inputs_may_not_share_an_alias() -> None:
    with pytest.raises(ValueError, match="two inputs"):
        join_columns(
            "Is {fact:side} the same as {fact:right}?",
            left_columns=("fact",),
            right_columns=("fact",),
            left_alias="side",
            right_alias="side",
        )


# §3: an alias names a side; it never rewrites the column it resolves to.
def test_an_alias_does_not_rename_the_resolved_column() -> None:
    assert join_columns(
        "Is {fact:earlier} the same as {fact:right}?",
        left_columns=("fact",),
        right_columns=("fact",),
        left_alias="earlier",
    ) == (("fact",), ("fact",))


# §3: references that land on one side leave the other side with no compared
# column, so the join falls back to whole rows instead of refusing. `valid_at` is
# owned by the left side alone; nothing names the right side at all.
def test_a_join_whose_references_land_on_one_side_compares_whole_rows() -> None:
    assert join_columns(
        "Is {valid_at} in the past?",
        left_columns=("id", "fact", "valid_at"),
        right_columns=("id", "fact"),
    ) == (("id", "fact", "valid_at"), ("id", "fact"))


# §3: the suffix map is built from this join's own inputs. The left frame here was
# published by `alias("earlier").join(...)`, so `earlier` is already part of its
# column name; this join passes no alias, and nothing is carried over.
def test_an_alias_from_an_earlier_join_is_not_inherited() -> None:
    assert join_columns(
        "Is {fact:earlier} the same as {fact:right}?",
        left_columns=("fact:earlier",),
        right_columns=("fact",),
        left_alias=None,
    ) == (("fact:earlier",), ("fact",))


# §3: a join instruction is allowed to name nothing at all.
def test_a_join_instruction_may_compare_whole_rows() -> None:
    assert join_columns(
        "Are these two rows about the same thing?",
        left_columns=("fact", "left_only"),
        right_columns=("fact", "right_only"),
    ) == (("fact", "left_only"), ("fact", "right_only"))


# §0: the same five characters, resolved twice.
def test_the_same_text_names_a_column_in_a_single_input_and_a_side_in_a_join() -> None:
    # a single-input instruction reads the whole token as a name
    assert render_input("{fact:left}", columns=("fact:left",)) == (
        "{{ input['fact:left'] }}"
    )
    # a join instruction reads the same token as a side plus a name
    assert render_join_condition(
        "{fact:left}", _scope(left_columns=("fact",), right_columns=("fact",))
    ) == "fact"

    # an alias is part of the name in a single-input instruction...
    assert render_input("{fact:earlier}", columns=("fact:earlier",)) == (
        "{{ input['fact:earlier'] }}"
    )
    # ...and the name of a side in a join instruction
    assert join_columns(
        "Is {fact:earlier} the same as {fact:right}?",
        left_columns=("fact",),
        right_columns=("fact",),
        left_alias="earlier",
    ) == (("fact",), ("fact",))


# §7: the brace ends the name, so a column whose name holds one has no reference.
def test_a_name_containing_a_brace_cannot_be_written() -> None:
    # the odd name reaches the model through the data block, never through a reference
    assert render_input("{a{b}", columns=("a{b", "b")) == (
        "{a{{ input['b'] }}\n\na{b: {{ input['a{b'] }}"
    )


# §3 through a rewrite: both consumers read one grammar, so every reference that
# resolved before a rewrite resolves after it, against the frame it produced.
@pytest.mark.parametrize(
    ("instruction", "input_cols", "output_cols"),
    [
        # a bare input name becomes the two sided names a join publishes
        (
            "Rows share a topic when {name} and {description} match.",
            ("name", "description"),
            (),
        ),
        # an output field survives the rewrite as text
        ("Produce {summary} from {name}.", ("name",), ("summary",)),
    ],
)
def test_a_rewritten_instruction_is_resolved_by_the_same_splitter(
    instruction: str,
    input_cols: tuple[str, ...],
    output_cols: tuple[str, ...],
) -> None:
    rewriter = DifferentialInstructionRewriter()
    if output_cols:
        rewritten = rewriter.agg_to_map(
            instruction,
            input_cols=input_cols,
            output_cols=tuple(ColumnSpec(name=name) for name in output_cols),
        )
        # the rewrite targets a map over a join of the two frames, so the names it
        # writes are the ones that join publishes; the declared field stays text
        joined = tuple(
            f"{name}:{side}" for name in input_cols for side in ("left", "right")
        )
        render_input(rewritten, columns=joined, outputs=output_cols)
    else:
        rewritten = rewriter.groupby_to_join(instruction, input_cols=input_cols)
        # the rewrite targets a join, so its `:left`/`:right` names are sides and
        # the join, not the single-input rule, has to read them
        assert join_columns(
            rewritten, left_columns=input_cols, right_columns=input_cols
        ) == (input_cols, input_cols)
