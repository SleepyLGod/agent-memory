"""JSON loading and bounded, value-preserving structural repair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import heapq
from itertools import count
import json
import re
from typing import Any, Generic, TypeVar

import json5


JSON_REPAIR_VERSION = "bounded-json-v2"
MAX_REPAIR_CANDIDATES = 4096
MAX_REPAIR_EDITS = 8
MAX_REPAIR_CHARACTERS = 8 * 1024 * 1024
_DELIMITERS = "}],:"
_TOKEN = re.compile(
    r'\s+|"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"|true|false|null|[{}\[\],:]"
)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class StructuredJsonResult:
    """A decoded value and any existing JSON5 normalization provenance."""

    value: Any
    repair_method: str | None = None


@dataclass(frozen=True)
class RepairedJson(Generic[ResultT]):
    """A unique minimum-edit result; edit offsets apply sequentially."""

    value: ResultT
    edits: tuple[tuple[str, int, str], ...]


def strict_json_loads(value: str) -> Any:
    """Decode JSON without duplicate object keys or non-finite constants."""

    return json.loads(
        value, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )


def load_structured_json_with_syntax_repair(
    raw_output: str, *, operator: str, expected_shape: str
) -> StructuredJsonResult:
    """Preserve the established strict JSON, fence and complete JSON5 path."""

    text = raw_output.strip()
    try:
        return StructuredJsonResult(strict_json_loads(text))
    except ValueError:
        pass
    candidate, fenced = _strip_fence(text)
    try:
        value = json5.loads(
            candidate,
            allow_duplicate_keys=False,
            consume_trailing=True,
            parse_constant=_reject_constant,
        )
    except ValueError as error:
        preview = raw_output[:240] + ("..." if len(raw_output) > 240 else "")
        raise ValueError(
            f"{operator} returned invalid JSON; expected {expected_shape}; "
            f"raw_output={preview!r}"
        ) from error
    return StructuredJsonResult(value, "json5-code-fence" if fenced else "json5")


def repair_json_structure(
    raw_output: str, *, validator: Callable[[str], ResultT]
) -> RepairedJson[ResultT]:
    """Accept a unique minimum-edit result within a bounded, forward-only search."""

    # Valid JSON/JSON5 with a bad schema is not a syntax-repair problem.
    try:
        load_structured_json_with_syntax_repair(
            raw_output, operator="repair", expected_shape="JSON"
        )
    except ValueError:
        pass
    else:
        raise ValueError("valid JSON contract errors cannot be structurally repaired")

    text, _ = _strip_fence(raw_output.strip())
    tokens = _tokens(text)
    scalars = _scalars(tokens)
    try:
        strict_json_loads(text)
    except json.JSONDecodeError as error:
        position = error.pos
    except ValueError as error:
        raise ValueError(
            "duplicate keys or non-finite values cannot be repaired"
        ) from error
    else:
        raise ValueError("output does not require syntax repair")

    serial = count()
    pending: list[tuple[int, int, str, tuple[tuple[str, int, str], ...], int]] = [
        (0, next(serial), text, (), position)
    ]
    seen = {tuple(token for token, _, _ in tokens): 0}
    accepted: dict[tuple[str, ...], RepairedJson[ResultT]] = {}
    best_cost = MAX_REPAIR_EDITS + 1
    examined = 0
    characters = 0
    while pending:
        cost, _, current, edits, position = heapq.heappop(pending)
        if cost >= min(best_cost, MAX_REPAIR_EDITS):
            continue
        for candidate, edit in _local_candidates(current, position).items():
            candidate_cost = cost + len(edit[2])
            if candidate_cost > min(best_cost, MAX_REPAIR_EDITS):
                continue
            characters += len(candidate)
            if characters > MAX_REPAIR_CHARACTERS:
                raise ValueError("JSON repair character-work bound exceeded")
            candidate_tokens = _tokens(candidate)
            # The same delimiter moved across whitespace is one candidate.
            key = tuple(token for token, _, _ in candidate_tokens)
            if seen.get(key, MAX_REPAIR_EDITS + 1) <= candidate_cost:
                continue
            seen[key] = candidate_cost
            examined += 1
            if examined > MAX_REPAIR_CANDIDATES:
                raise ValueError("JSON repair candidate bound exceeded")
            if _scalars(candidate_tokens) != scalars:
                continue
            candidate_edits = (*edits, edit)
            try:
                strict_json_loads(candidate)
            except json.JSONDecodeError as error:
                # Resolve the current error before attempting the next one.
                if error.pos > position or (
                    error.pos == position and len(candidate) < len(current)
                ):
                    heapq.heappush(
                        pending,
                        (
                            candidate_cost,
                            next(serial),
                            candidate,
                            candidate_edits,
                            error.pos,
                        ),
                    )
                continue
            except ValueError:
                continue
            try:
                value = validator(candidate)
            except ValueError:
                # Syntactically valid contract errors must never be edited further.
                continue
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                accepted.clear()
            accepted[key] = RepairedJson(value, candidate_edits)
    if len(accepted) != 1:
        raise ValueError(
            f"JSON repair requires one valid candidate at minimum edits; got {len(accepted)}"
        )
    return next(iter(accepted.values()))


def _local_candidates(text: str, position: int) -> dict[str, tuple[str, int, str]]:
    tokens = _tokens(text)
    nearby = [i for i, (_, start, end) in enumerate(tokens) if start <= position <= end]
    anchor = nearby[-1] if nearby else len(tokens) - 1
    boundaries = {position}
    for _, start, end in tokens[max(0, anchor - 2) : anchor + 2]:
        boundaries.update((start, end))
    # Never insert inside a string, number, boolean, or null token.
    boundaries = {
        p for p in boundaries if not any(start < p < end for _, start, end in tokens)
    }
    candidates: dict[str, tuple[str, int, str]] = {}
    for p in sorted(boundaries):
        if p < len(text) and text[p] in _DELIMITERS:
            candidates[text[:p] + text[p + 1 :]] = ("delete", p, text[p])
        for delimiter in _DELIMITERS:
            candidates[text[:p] + delimiter + text[p:]] = ("insert", p, delimiter)
    if position == len(text):
        suffix = _closing_suffix(tokens)
        if suffix:
            candidates[text + suffix] = ("insert", len(text), suffix)
    return candidates


def _tokens(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise ValueError("incomplete string or scalar cannot be repaired")
        token = match.group()
        if not token.isspace():
            tokens.append((token, position, match.end()))
        position = match.end()
    return tokens


def _scalars(tokens: Sequence[tuple[str, int, int]]) -> tuple[str, ...]:
    return tuple(token for token, _, _ in tokens if token not in "{}[],:")


def _closing_suffix(tokens: Sequence[tuple[str, int, int]]) -> str:
    stack: list[str] = []
    for token, _, _ in tokens:
        if token in ("{", "["):
            stack.append("}" if token == "{" else "]")
        elif token in ("}", "]"):
            if not stack or stack.pop() != token:
                return ""
    return "".join(reversed(stack))


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value is not supported: {value}")


def _strip_fence(value: str) -> tuple[str, bool]:
    match = re.fullmatch(r"```(?:json)?[ \t]*\n(.*)\n```", value, flags=re.DOTALL)
    return (value, False) if match is None else (match.group(1).strip(), True)
