"""The brace grammar of an instruction: text in, tokens out.

`split` gives tokens, `resolve_single` or `resolve_join` says what each reference means
(`JoinScope` pairs a join's two sides with the aliases they carry), and `compose` writes
tokens back as an instruction for a rewriter. Drawing a prompt is the adapter's next step;
nothing here knows which engine renders it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Literal

# Based on: `docs/design/instruction_placeholders.md` §1, which states the scan as three
#   rules and the name shape as `[A-Za-z_][A-Za-z0-9_]*` with `:`-separated segments.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)*")

# §3: an alias is a variable name, so a suffix can always be written in a reference.
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PlaceholderError(ValueError):
    """An instruction cannot be read as text plus references."""


Side = Literal["left", "right"]


@dataclass(frozen=True)
class Text:
    """A run of instruction text, already de-escaped."""

    text: str


@dataclass(frozen=True)
class Reference:
    """A `{...}` pair holding a name.

    The whole content is the name, `:` segments included. Which set the name belongs
    to, and whether its last segment is a join side, is decided after this stage.
    """

    name: str


Token = Text | Reference


@dataclass(frozen=True)
class Value:
    """A reference naming a column whose value reaches the model.

    `side` is None where one input is in scope, and `left` or `right` where a join has
    two. `name` is the input column, never the suffix that named the side.
    """

    name: str
    side: Side | None = None


@dataclass(frozen=True)
class Field:
    """A reference naming a column the model must fill, so it has no value yet."""

    name: str


Resolved = Text | Value | Field


def split(instruction: str) -> tuple[Token, ...]:
    """Read an instruction into text and references, left to right.

    `{{` and `}}` are literal braces; a `{` whose `}` comes before any other `{`
    opens a reference; a brace that does neither is text. A pair holding no name is
    refused here, before any column set is known.

    Splitting is syntactic, so text that a renderer would have to carry specially
    (such as an escaped run that still holds `{{`) is returned as text. Whether a
    prompt can carry it is decided where the prompt is written.
    """

    tokens: list[Token] = []
    text: list[str] = []
    cursor = 0
    while cursor < len(instruction):
        character = instruction[cursor]
        if character == "{":
            if instruction.startswith("{{", cursor):
                text.append("{")
                cursor += 2
                continue
            close = _closing_brace(instruction, cursor)
            if close is None:
                text.append("{")
                cursor += 1
                continue
            _flush_text(tokens, text)
            tokens.append(Reference(_require_name(instruction[cursor + 1 : close])))
            cursor = close + 1
            continue
        if character == "}" and instruction.startswith("}}", cursor):
            text.append("}")
            cursor += 2
            continue
        text.append(character)
        cursor += 1
    _flush_text(tokens, text)
    return tuple(tokens)


def resolve_single(
    tokens: Sequence[Token],
    *,
    inputs: Iterable[str],
    outputs: Iterable[str] = (),
) -> tuple[Resolved, ...]:
    """Read every reference against one operator's vocabulary (§2).

    An input column names a value, an output field names what the model fills, and a
    name in both is an input: the value already exists. A suffix means nothing here,
    so `fact:left` is looked up as the whole column name it is.
    """

    values = {str(name) for name in inputs}
    fields = {str(name) for name in outputs}
    resolved: list[Resolved] = []
    for token in tokens:
        if isinstance(token, Text):
            resolved.append(token)
        elif token.name in values:
            resolved.append(Value(token.name))
        elif token.name in fields:
            resolved.append(Field(token.name))
        else:
            raise PlaceholderError(
                f"instruction references {{{token.name}}}, which is neither an input "
                "column nor an output field"
            )
    return tuple(resolved)


def join_suffixes(
    *, left_alias: str | None = None, right_alias: str | None = None
) -> dict[str, Side]:
    """Return the suffix → side map for one join (§3).

    `left` and `right` always name their own side; an alias adds a second name for the
    side carrying it. Only what this join's inputs are called counts, so an alias from
    an earlier join is not inherited.
    """

    suffixes: dict[str, Side] = {"left": "left", "right": "right"}
    for alias, side in ((left_alias, "left"), (right_alias, "right")):
        if alias is None:
            continue
        if not _VARIABLE.fullmatch(alias):
            raise PlaceholderError(
                f"join alias {alias!r} is not a variable name: use letters, digits and "
                "`_`, and do not start with a digit"
            )
        owner = suffixes.get(alias)
        if owner is not None and owner != side:
            raise PlaceholderError(
                f"join suffix {alias!r} names two inputs; a suffix points at one side"
            )
        suffixes[alias] = side
    return suffixes


def resolve_join(
    tokens: Sequence[Token],
    *,
    left: Iterable[str],
    right: Iterable[str],
    suffixes: Mapping[str, Side] | None = None,
) -> tuple[Resolved, ...]:
    """Read every reference against the two inputs of one join (§3).

    Checks in order, first match wins: a sided reference split at its last colon; a
    whole name exactly one side owns; a whole name both sides own, which cannot say
    which value is meant; anything else. A suffix the join publishes is not a side
    here, so `fact:earlier` is a whole column name until `earlier` is a suffix.
    """

    if suffixes is None:
        suffixes = join_suffixes()
    columns: dict[Side, set[str]] = {
        "left": {str(name) for name in left},
        "right": {str(name) for name in right},
    }
    resolved: list[Resolved] = []
    for token in tokens:
        if isinstance(token, Text):
            resolved.append(token)
            continue
        base, separator, suffix = token.name.rpartition(":")
        if separator and suffix in suffixes:
            side = suffixes[suffix]
            if base in columns[side]:
                resolved.append(Value(base, side=side))
                continue
        owned_by_left = token.name in columns["left"]
        owned_by_right = token.name in columns["right"]
        if owned_by_left and owned_by_right:
            raise PlaceholderError(
                f"instruction references {{{token.name}}}, which both sides have; add a "
                f"suffix, as in {{{token.name}:left}} or {{{token.name}:right}}"
            )
        if owned_by_left or owned_by_right:
            resolved.append(Value(token.name, side="left" if owned_by_left else "right"))
            continue
        raise PlaceholderError(
            f"instruction references {{{token.name}}}, which neither side has"
        )
    return tuple(resolved)


@dataclass(frozen=True)
class JoinScope:
    """The two inputs of one join: the columns each side has and the alias it carries.

    A scope is built for one join, from that join's own inputs: an alias an earlier join
    carried is not inherited, and the names this join publishes are not sides here.
    """

    left: tuple[str, ...]
    right: tuple[str, ...]
    left_alias: str | None = None
    right_alias: str | None = None

    @classmethod
    def of(
        cls,
        left: Iterable[str],
        right: Iterable[str],
        *,
        left_alias: str | None = None,
        right_alias: str | None = None,
    ) -> "JoinScope":
        """Return a scope whose column names are strings, in the order each side lists them."""

        return cls(
            tuple(str(column) for column in left),
            tuple(str(column) for column in right),
            left_alias,
            right_alias,
        )

    def with_columns(self, left: Iterable[str], right: Iterable[str]) -> "JoinScope":
        """Return these two sides narrowed to other columns, keeping this join's aliases."""

        return JoinScope.of(
            left, right, left_alias=self.left_alias, right_alias=self.right_alias
        )

    def suffixes(self) -> dict[str, Side]:
        """Return this join's suffix map, built before any reference is read."""

        return join_suffixes(left_alias=self.left_alias, right_alias=self.right_alias)

    def resolve(self, instruction: str) -> tuple[Resolved, ...]:
        """Read an instruction against both sides (§3)."""

        return resolve_join(
            split(instruction), left=self.left, right=self.right, suffixes=self.suffixes()
        )


def compose(tokens: Iterable[Token]) -> str:
    """Write tokens back as an instruction: the inverse of `split`.

    Text is escaped again, because a brace that was read as text has to be read as
    text next time too. `compose(split(text))` therefore keeps the meaning of `text`
    even where it changes its bytes: a lone `}` comes back doubled.
    """

    pieces: list[str] = []
    for token in tokens:
        if isinstance(token, Text):
            pieces.append(token.text.replace("{", "{{").replace("}", "}}"))
        else:
            pieces.append("{" + token.name + "}")
    return "".join(pieces)


def _closing_brace(instruction: str, cursor: int) -> int | None:
    """Return the index closing the reference opened at `cursor`, if it opens one.

    Another `{` before the first `}` means this one is literal: the inner pair is the
    reference and the outer braces are text.
    """

    index = cursor + 1
    while index < len(instruction):
        if instruction[index] == "{":
            return None
        if instruction[index] == "}":
            return index
        index += 1
    return None


def _require_name(content: str) -> str:
    """Return `content` as a name, or refuse the pair that holds it."""

    if not _NAME.fullmatch(content):
        raise PlaceholderError(
            f"instruction contains {{{content}}}, which is not a column name"
        )
    return content


def _flush_text(tokens: list[Token], text: list[str]) -> None:
    """Move the pending text into a token, so adjacent runs stay one token."""

    if text:
        tokens.append(Text("".join(text)))
        text.clear()
