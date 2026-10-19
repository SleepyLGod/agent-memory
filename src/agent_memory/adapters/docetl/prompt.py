"""Translate our `{column}` DSL into the Jinja DocETL renders; the context root differs per operation."""

from __future__ import annotations

import re

# Reason: Jinja will not unescape the pair, and a separate pass would either damage the Jinja we emit
#   or make escaped text live.
_ESCAPED_PAIR = r"\{\{([^{}]*)\}\}"

# Based on: the placeholder shape the sibling adapter parses
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
