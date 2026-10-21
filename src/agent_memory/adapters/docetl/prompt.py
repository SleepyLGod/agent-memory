"""Translate our `{column}` DSL into the Jinja DocETL renders; the context root differs per operation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

# Reason: Jinja will not unescape the pair, and a separate pass would either damage the Jinja we emit
#   or make escaped text live.
_ESCAPED_PAIR = r"\{\{([^{}]*)\}\}"

# Based on: the placeholder shape `lotus/structured.py` parses
#   (`src/agent_memory/adapters/lotus/structured.py::PLACEHOLDER_PATTERN`, the identifier-shaped
#   reading of `lotus.nl_expression::parse_cols`), copied with the escaped pair added.
_PLACEHOLDER_BODY = r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(?::([A-Za-z_][A-Za-z0-9_]*))?\}(?!\})"

_PLACEHOLDER = re.compile(_ESCAPED_PAIR + "|" + _PLACEHOLDER_BODY)


def render_prompt(
    instruction: str,
    *,
    root: str = "input",
    allow_tags: bool = False,
) -> str:
    """Return the DocETL prompt for an instruction that names columns as `{column}`.

    A doubled brace is the language's escape for a literal brace, so it is carried through as text
    rather than translated. `allow_tags` is for an operator whose context binds the qualified name, as
    `equijoin` does with `left`/`right`; an operator that renders one row must refuse it.
    """

    found = False

    def substitute(match: re.Match[str]) -> str:
        nonlocal found
        escaped, base, qualifier = match.group(1), match.group(2), match.group(3)
        if escaped is not None:
            return "{" + escaped + "}"
        found = True
        if qualifier is not None:
            if not allow_tags:
                raise ValueError(
                    f"instruction placeholder {match.group(0)} is qualified: it names another "
                    "row, so it belongs to a join"
                )
            # `{base:qualifier}`: the qualifier names the row, so it becomes the context root
            return "{{ " + qualifier + "." + base + " }}"
        return "{{ " + root + "." + base + " }}"

    prompt = _PLACEHOLDER.sub(substitute, instruction)
    if not found:
        # Reason: DocETL asks for confirmation when a prompt is not Jinja and answers itself when
        #   stdin is not a terminal (`docetl.operations.map::MapOperation.__init__`), so a
        #   placeholder-free prompt would pass for Jinja and then never interpolate.
        raise ValueError(
            "instruction names no {column} placeholder, so the prompt would never interpolate"
        )
    return prompt


# The verdict prompt, verbatim. `{context}` / `{condition}` are filled by Python below; whatever
# `{{ ... }}` survives that is DocETL's Jinja, read from each pair's record.
_VERDICT_TEMPLATE = (
    "Context:\n"
    "{context}\n"
    "\n"
    "Condition: {condition}\n"
    "Does the context satisfy this condition?"
)

# One compared column: `[left] note: «{{ input.left_0 }}»`. The quadruple braces are Python's escape
# for a literal `{{` / `}}`, which DocETL then renders.
_CONTEXT_LINE = "{label}: «{{{{ input.{slot} }}}}»"

# The ranking prompt, verbatim. `candidates` is filled by DocETL, not by Python.
_TOPK_TEMPLATE = (
    "{{{{ input.candidates }}}}\n"
    "\n"
    "These are the candidates that already satisfy this semantic join condition: {condition}\n"
    "Select at most {k} of them, best first, and put their ids in {ids_column}."
)


def join_slot(side: str, index: int) -> str:
    """Return the record key one compared column is read from: `left_0`, `right_1`, ..."""

    return f"{side}_{index}"


def build_join_prompt(
    instruction: str,
    *,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> str:
    """Return the verdict prompt: one labelled line per compared field, then the condition.

    Based on: LOTUS's `join_series` + `df2text` layout (`[<label>]: «<value>»`, condition stated
    separately). Slots are positional so a column name cannot become a Jinja expression, and the root is
    `input` because the verdict runs as a `filter` over a one-pair frame. `sem_join` fills those keys with
    `join_slot`, so the template and the records cannot disagree about a key's name.
    """

    context = "\n".join(
        _CONTEXT_LINE.format(label=f"[{side}] {column}", slot=join_slot(side, index))
        for side, columns in (("left", left_columns), ("right", right_columns))
        for index, column in enumerate(columns)
    )
    return _VERDICT_TEMPLATE.format(
        context=context,
        condition=lower_join_instruction(instruction),
    )


def build_topk_prompt(instruction: str, *, k: int, ids_column: str) -> str:
    """Return the ranking prompt for one left row whose matches overflow `k`."""

    return _TOPK_TEMPLATE.format(
        condition=lower_join_instruction(instruction),
        k=k,
        ids_column=ids_column,
    )


def lower_join_instruction(instruction: str) -> str:
    """Return the instruction with join placeholders replaced by the column names they name.

    Based on: `src/agent_memory/adapters/lotus/sem_join.py::join_series`, which compares one rendered
    value per side and therefore cannot leave a placeholder for a renderer to fill.
    """

    def substitute(match: re.Match[str]) -> str:
        escaped, base, _qualifier = match.group(1), match.group(2), match.group(3)
        if escaped is not None:
            return "{" + escaped + "}"
        return base

    return _PLACEHOLDER.sub(substitute, instruction)


def join_comparison_columns(
    instruction: str,
    *,
    left_columns: Iterable[str],
    right_columns: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the left and right columns whose values the join compares.

    A `{column:side}` placeholder names its side and must exist there. A bare `{column}` names the side
    that owns it; when both sides own it the instruction cannot say which value it means, so both sides
    fall back to whole-row text — the same choice `join_series` makes.
    """

    left_set = {str(column) for column in left_columns}
    right_set = {str(column) for column in right_columns}
    left: list[str] = []
    right: list[str] = []
    ambiguous = False

    for match in _PLACEHOLDER.finditer(instruction):
        escaped, base, qualifier = match.group(1), match.group(2), match.group(3)
        if escaped is not None:
            continue
        if qualifier is not None:
            if qualifier not in {"left", "right"}:
                raise ValueError(
                    f"sem_join placeholder {match.group(0)} names {qualifier!r}; a join has only "
                    "left and right"
                )
            available = left_set if qualifier == "left" else right_set
            if base not in available:
                raise ValueError(
                    f"sem_join placeholder {match.group(0)} names a column the {qualifier} side "
                    "does not have"
                )
            target = left if qualifier == "left" else right
            if base not in target:
                target.append(base)
            continue

        in_left = base in left_set
        in_right = base in right_set
        if in_left and in_right:
            ambiguous = True
        elif in_left:
            if base not in left:
                left.append(base)
        elif in_right:
            if base not in right:
                right.append(base)
        else:
            raise ValueError(f"unknown sem_join column placeholder: {base!r}")

    if ambiguous or not left or not right:
        # Reason: one side has no named column, so comparing a single value per side would guess; whole
        #   rows are the only reading that cannot drop the value the instruction meant.
        return (
            tuple(str(column) for column in left_columns),
            tuple(str(column) for column in right_columns),
        )
    return tuple(left), tuple(right)
