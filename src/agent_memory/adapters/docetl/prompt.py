"""Translate our `{column}` DSL into the Jinja DocETL renders.

An instruction becomes a prompt in four steps, and the first two are not here:

1. `split` (`policy/placeholders.py`) reads text and `{name}` references, and de-escapes
   `{{` / `}}`;
2. `resolve_single` or `resolve_join` (same module) says what each name means for this
   operator -- a value it supplies (`Value`), a field the model fills (`Field`) -- and, for
   a join, which side owns it;
3. layout -- `_render_row` or `build_join_prompt` writes each token as Jinja: a value as
   `input['name']`, a field as `{name}`, text as written but wrapped in `{% raw %}` where
   Jinja would otherwise compile it (`_verbatim`);
4. supply -- `_data_block` appends every `I` column no reference named, so each column
   reaches the model exactly once.

Two pipelines live here:

* single input -- all four steps, in `render_prompt`;
* join -- the same steps 1 and 2 through `JoinScope.resolve`, then a projection onto the two
  sides (`compared_columns`) and the pair layout (`build_join_prompt`,
  `render_join_condition`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

from agent_memory.policy.placeholders import (
    Field,
    JoinScope,
    Resolved,
    Side,
    Text,
    Value,
    resolve_single,
    split,
)


# ---------------------------------------------------------------------------
# Row prompts: one row in, one prompt out.
# ---------------------------------------------------------------------------


# DocETL renders the prompt with Jinja, which reads any of these as syntax wherever it occurs.
_OPENERS = ("{{", "{%", "{#")

# The one string a `{% raw %}` block cannot hold: its own terminator (`jinja2/lexer.py:571-579`).
_RAW_END = re.compile(r"\{%[-+]?\s*endraw\s*[-+]?%\}")


def render_prompt(
    instruction: str,
    *,
    columns: Iterable[str],
    outputs: Iterable[str] = (),
) -> str:
    """Return the DocETL prompt for one row: the instruction, then the columns it leaves out.

    `columns` is the operator's `I` and `outputs` its `O`, read by `resolve_single`. Every
    `I` column the instruction does not name is appended as a `name: value` line, so a
    column reaches the model once, and the prompt always holds an interpolation.
    """

    names = _unique(columns)
    resolved = resolve_single(
        split(instruction),
        inputs=names,
        outputs=_unique(outputs),
    )
    prompt = _render_row(resolved)
    named = {token.name for token in resolved if isinstance(token, Value)}
    unreferenced = [name for name in names if name not in named]
    if not unreferenced:
        return prompt
    return f"{prompt}\n\n{_data_block(unreferenced)}"


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return the names as strings, in operator order and without repeats."""

    return tuple(dict.fromkeys(str(value) for value in values))


def _render_row(resolved: Sequence[Resolved]) -> str:
    """Write a resolved stream as the template DocETL renders for one row."""

    pieces: list[str] = []
    for index, token in enumerate(resolved):
        if isinstance(token, Text):
            pieces.append(_verbatim(token.text, before_a_brace=index + 1 < len(resolved)))
        elif isinstance(token, Value):
            pieces.append(_input_expression(token.name))
        elif isinstance(token, Field):
            pieces.append("{" + token.name + "}")
        else:
            # Reason: the three types are the whole of `Resolved`, so reaching here means a new
            #   token type arrived without a writer; rendering it as a field would be a guess.
            raise TypeError(f"unexpected resolved token {type(token).__name__}")
    return "".join(pieces)


def _verbatim(text: str, *, before_a_brace: bool) -> str:
    """Return `text` so DocETL shows it as written, wrapping it only where it has to be.

    Two runs need a literal block: one holding an opener, which Jinja would compile, and
    one ending in `{` just before a reference, whose `{{` the two runs would otherwise
    join into one.
    """

    if _RAW_END.search(text):
        # Reason: a raw block ends at the first terminator Jinja recognises
        #   (`jinja2/lexer.py:571-579`), so a run spelling one cannot be carried at all.
        raise ValueError(
            f"instruction contains {text!r}, which spells the end of a literal block. The "
            "words can be written; the `{%` before `endraw` cannot."
        )
    if any(opener in text for opener in _OPENERS) or (before_a_brace and text.endswith("{")):
        return "{% raw %}" + text + "{% endraw %}"
    return text


def _input_expression(name: str) -> str:
    """Return the Jinja expression that reads one column's value from the row."""

    # Reason: `{{ input.name }}` would be attribute access, and DocETL renders the row as a dict
    #   (`docetl.operations.map::MapOperation`), so a column called `items` resolves to `dict.items`,
    #   and one carrying the colon a join added (`relational.py::_rename_join_side`) would not
    #   compile at all. Subscript reads the name as a literal key, whatever characters it holds.
    return "{{ input[" + _string_literal(name) + "] }}"


def _data_block(names: Sequence[str]) -> str:
    """Return the trailing lines that supply every column the instruction left out."""

    return "\n".join(f"{name}: {_input_expression(name)}" for name in names)


def _string_literal(name: str) -> str:
    """Return `name` as a Jinja string literal."""

    return "'" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"


# ---------------------------------------------------------------------------
# Join prompts: one prompt about a pair of rows.
# ---------------------------------------------------------------------------


def compared_columns(instruction: str, scope: JoinScope) -> JoinScope:
    """Return `scope` narrowed to the columns whose values this join compares (§3).

    A `{column:side}` names its side and must exist there; a bare `{column}` names the one
    side that owns it, and a name both sides own is refused rather than guessed.
    """

    named = _named_sides(scope.resolve(instruction))
    if not named["left"] or not named["right"]:
        # Reason: one side has no named column, so comparing a single value per side would guess;
        #   whole rows are the only reading that cannot drop the value the instruction meant.
        return scope
    return scope.with_columns(named["left"], named["right"])


def _named_sides(resolved: Sequence[Resolved]) -> dict[Side, list[str]]:
    """Return the columns each side contributes, in the order the instruction names them."""

    named: dict[Side, list[str]] = {"left": [], "right": []}
    for token in resolved:
        if not isinstance(token, Value) or token.side is None:
            continue
        if token.name not in named[token.side]:
            named[token.side].append(token.name)
    return named


# The verdict prompt, verbatim. `{context}` / `{condition}` are filled by Python below; whatever
# `{{ ... }}` survives that is DocETL's Jinja, read from each pair's record.
_VERDICT_TEMPLATE = (
    "Context:\n"
    "{context}\n"
    "\n"
    "Condition: {condition}\n"
    "Does the context satisfy this condition?"
)

# One compared column: `[left] note: «{{ left['left_0'] }}»`. The quadruple braces are Python's
# escape for a literal `{{` / `}}`, which DocETL then renders. The root is the side's own name,
# because `EquijoinOperation` renders one prompt against `left` and `right` (`equijoin.py:161`).
_CONTEXT_LINE = "{label}: «{{{{ {root}[{slot!r}] }}}}»"

# The listwise ranking prompt for one left row and its candidate right rows.
# Based on: `adapters/lotus/sem_topk_join.py::LISTWISE_JOIN_SYSTEM_PROMPT`.
_TOPK_TEMPLATE = (
    "The left row to match:\n"
    "{{{{ input.{anchor} }}}}\n"
    "\n"
    "These right rows already satisfy this join condition: {condition}\n"
    "{{{{ input.candidates }}}}\n"
    "\n"
    "Select at most {k} of them for the left row, strongest first, and put their ids in "
    "{ids_column}. Do not select merely related candidates, and do not fill the limit "
    "when fewer candidates match. Return an empty list when none match."
)


def join_slot(side: str, index: int) -> str:
    """Return the record key one compared column is read from: `left_0`, `right_1`, ..."""

    return f"{side}_{index}"


def build_join_prompt(instruction: str, *, scope: JoinScope) -> str:
    """Return the verdict prompt: one labelled line per compared field, then the condition.

    `sem_join` fills the record keys with `join_slot`, so the template and the records cannot
    disagree about a key's name. The scope is the one `compared_columns` narrowed, so every
    reference the condition names is among the columns it lists. Each line reads its own side's
    record: the engine compares two records, not one record holding both sides.
    """

    # Based on: LOTUS's `join_series` + `df2text` layout (`[<label>]: «<value>»`, condition stated
    #   separately). Slots are positional so a column name cannot become a Jinja expression, and
    #   each root is the side it reads, which is what `equijoin` renders the prompt with.
    context = "\n".join(
        _CONTEXT_LINE.format(
            label=f"[{side}] {column}", root=side, slot=join_slot(side, index)
        )
        for side, columns in (("left", scope.left), ("right", scope.right))
        for index, column in enumerate(columns)
    )
    return _VERDICT_TEMPLATE.format(
        context=context, condition=render_join_condition(instruction, scope)
    )


def build_topk_prompt(
    instruction: str,
    *,
    k: int,
    ids_column: str,
    anchor_column: str,
    scope: JoinScope,
) -> str:
    """Return the ranking prompt for one left row whose matches overflow `k`."""

    return _TOPK_TEMPLATE.format(
        anchor=anchor_column,
        condition=render_join_condition(instruction, scope),
        k=k,
        ids_column=ids_column,
    )


def render_join_condition(instruction: str, scope: JoinScope) -> str:
    """Return the instruction with join placeholders replaced by the column names they name.

    A literal run is carried the same way a row prompt carries it, because this text is
    inserted into the verdict template and then rendered by Jinja.
    """

    # Based on: `src/agent_memory/adapters/lotus/sem_join.py::join_series`, which compares one
    #   rendered value per side and therefore cannot leave a placeholder for a renderer to fill.
    pieces: list[str] = []
    for token in scope.resolve(instruction):
        if isinstance(token, Text):
            pieces.append(_verbatim(token.text, before_a_brace=False))
        else:
            pieces.append(token.name)
    return "".join(pieces)
