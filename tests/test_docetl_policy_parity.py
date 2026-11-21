"""Policy-level parity: one `Memory` per operator shape must publish the same view on both backends.

Between the adapter and that view sit the declared-column check, the index check, the per-row cache and
publication (`PolicyExecutor._execute_semantic_row`), so this asks what a *policy* makes of a frame rather
than whether the lowering is right. Adding an operator means adding a `PolicyCase`, never a file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from agent_memory.api import Log, Memory

from adapter_support import build_adapters, canned_model, require_real_provider

AUDIT_DIR = Path("/tmp/agent-memory-docetl-policy-parity") / datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")

# the lowering states no polarity, so the instruction must (`sem_filter.py::build_filter_config`)
FILTER_INSTRUCTION = (
    "Keep this note only when it records something durable about the user, such as a preference or a "
    "lasting fact: {message}\n\n"
    "Respond with 'true' when the note records something durable about the user, and 'false' when it "
    "does not."
)


class _SemFilterNotes(Memory):
    """A memory whose entire policy is one semantic keep/drop decision."""

    log = Log({"message": "One short note the user typed."})

    # the filter's own output: the registry has no relational operator to chain on
    kept = log.sem_filter(instruction=FILTER_INSTRUCTION)


class _SummarisedNotes(Memory):
    """A memory whose entire policy is one semantic column fill."""

    log = Log({"message": "One short note the user typed."})

    summarised = log.sem_map(
        output_cols={"summary": "One line, no longer than a sentence."},
        instruction="Summarise the note in one line: {message}",
    )


class _SplitNotes(Memory):
    """A memory whose policy splits each note into numbered parts."""

    log = Log({"message": "One short note the user typed."})

    split = log.sem_flat_map(
        output_cols={"part": "One clause of the note."},
        instruction="Split the note into its clauses: {message}",
        ordinal_col="part_number",
    )


@dataclass(frozen=True)
class PolicyCase:
    """One policy, the notes fed to it, and what its view must publish."""

    name: str
    policy: type[Memory]
    view: str
    notes: tuple[str, ...]
    # a note appears once per row it must produce, so a flat map lists its note twice
    expected: tuple[str, ...]
    keep_markers: tuple[str, ...]
    expected_columns: tuple[str, ...] = ("message",)
    map_answers: Mapping[str, Any] = field(default_factory=dict)
    map_rows: tuple[Mapping[str, Any], ...] | None = None
    # a real-model case is worth its cost only when a model's answer determines the exact rows
    real: bool = True


# the two durable notes come first, so the expected set reads straight off this tuple
NOTES = (
    "I always take the aisle seat on long flights.",
    "My sister's name is Priya.",
    "The weather here has been lovely all week.",
    "Thanks, that worked!",
)
EXPECTED_KEPT = (
    "I always take the aisle seat on long flights.",
    "My sister's name is Priya.",
)

# small talk only, so a correct run publishes nothing
SMALL_TALK = (
    "The weather here has been lovely all week.",
    "Thanks, that worked!",
    "The bus was late again this morning.",
)

# markers appear only in the notes: a marker the instruction contained would decide every row
DURABLE_MARKERS = ("aisle seat", "Priya")


def _cases() -> list[PolicyCase]:
    return [
        PolicyCase(
            name="sem_filter/mixed",
            policy=_SemFilterNotes,
            view="kept",
            notes=NOTES,
            expected=EXPECTED_KEPT,
            keep_markers=DURABLE_MARKERS,
        ),
        PolicyCase(
            name="sem_filter/nothing qualifies",
            policy=_SemFilterNotes,
            view="kept",
            notes=SMALL_TALK,
            expected=(),
            keep_markers=DURABLE_MARKERS,
            real=False,
        ),
        PolicyCase(
            name="sem_map",
            policy=_SummarisedNotes,
            view="summarised",
            notes=NOTES[:2],
            expected=NOTES[:2],
            keep_markers=(),
            expected_columns=("message", "summary"),
            map_answers={"summary": "canned summary"},
            # a model's sentence is free text, so the real check for this op is its audit case
            real=False,
        ),
        PolicyCase(
            name="sem_flat_map",
            policy=_SplitNotes,
            view="split",
            notes=NOTES[:2],
            # one row per emitted entry, so each note appears twice
            expected=(NOTES[0], NOTES[0], NOTES[1], NOTES[1]),
            keep_markers=(),
            expected_columns=("message", "part", "part_number"),
            map_rows=({"part": "a"}, {"part": "b"}),
            real=False,
        ),
    ]


def _real_cases() -> list[PolicyCase]:
    """The cases worth a real call: the verdicts decide exact rows, so a real answer is comparable."""

    return [case for case in _cases() if case.real and case.expected]


def _expected_prefix(case: PolicyCase, notes: Sequence[str]) -> list[str]:
    """Return the messages the view should hold after these notes, honouring repeated rows."""

    counts: dict[str, int] = {}
    for note in case.expected:
        counts[note] = counts.get(note, 0) + 1
    messages = [note for note in notes for _ in range(counts.get(note, 0))]
    return sorted(messages)


def _case_id(case: PolicyCase) -> str:
    return case.name


def _build(case: PolicyCase, endpoint: str, *, api_key: str | None) -> tuple[Memory, Memory]:
    """Return the case's policy instantiated twice, once per backend."""

    lotus, docetl = build_adapters(endpoint, api_key=api_key)
    return case.policy(adapter=lotus), case.policy(adapter=docetl)


def _kept(memory: Memory, view: str) -> pd.DataFrame:
    state = memory._runtime._state
    assert view in state, sorted(state)
    return state[view]


def _messages(memory: Memory, view: str) -> list[str]:
    return sorted(_kept(memory, view)["message"].tolist())


def _feed(both: Sequence[Memory], notes: Sequence[str]) -> None:
    """Add each note to every memory, so both backends see identical input in identical order."""

    for note in notes:
        for memory in both:
            memory.add(note)


def _write_audit(case: PolicyCase, kept: Mapping[str, Sequence[str]]) -> None:
    """Write one row per note and print the verdicts, before any assertion can fail."""

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "note": list(case.notes),
            "expected": [note in case.expected for note in case.notes],
            **{
                backend: [note in messages for note in case.notes]
                for backend, messages in kept.items()
            },
        }
    )
    path = AUDIT_DIR / f"{case.name.replace('/', '-')}.csv"
    frame.to_csv(path, index=False)

    print(f"\n{case.name}: {path}")
    for note in case.notes:
        verdicts = " ".join(
            f"{backend}={'keep' if note in messages else 'drop'}"
            for backend, messages in kept.items()
        )
        print(f"  [{'keep' if note in case.expected else 'drop'}] {verdicts} :: {note}")


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_the_canned_endpoint_publishes_the_expected_memories(case: PolicyCase) -> None:
    """One policy, two engines, the declared view published with the rows it should hold."""

    with canned_model(
        *case.keep_markers,
        map_answers=case.map_answers,
        map_rows=case.map_rows,
    ) as endpoint:
        both = _build(case, endpoint.base_url, api_key=None)
        _feed(both, case.notes)

    for name, memory in zip(("lotus", "docetl"), both, strict=True):
        view = _kept(memory, case.view)
        assert list(view.columns) == list(case.expected_columns), (name, list(view.columns))
        assert _messages(memory, case.view) == _expected_prefix(case, case.notes), (
            name,
            _messages(memory, case.view),
        )
    # frames, not names: a rebuilt frame can change a dtype, and `_align_output` checks names only
    assert_frame_equal(_kept(both[0], case.view), _kept(both[1], case.view))


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_the_canned_endpoint_publishes_each_note_as_it_arrives(case: PolicyCase) -> None:
    """Incremental ingestion: the view must converge one note at a time, on both engines."""

    with canned_model(
        *case.keep_markers,
        map_answers=case.map_answers,
        map_rows=case.map_rows,
    ) as endpoint:
        both = _build(case, endpoint.base_url, api_key=None)
        for position, note in enumerate(case.notes):
            _feed(both, (note,))
            expected = _expected_prefix(case, case.notes[: position + 1])
            for name, memory in zip(("lotus", "docetl"), both, strict=True):
                assert _messages(memory, case.view) == expected, (
                    name,
                    f"after adding {note!r}",
                    _messages(memory, case.view),
                )


@pytest.mark.parametrize("case", _real_cases(), ids=_case_id)
def test_the_real_model_publishes_the_expected_memories(case: PolicyCase) -> None:
    """A real model decides, and both backends publish the same memories."""

    model, base_url = require_real_provider(f"the real-model policy case ({case.name})")
    lotus_adapter, docetl_adapter = build_adapters(base_url, model=model, api_key=None)
    both = (
        case.policy(adapter=lotus_adapter),
        case.policy(adapter=docetl_adapter),
    )

    _feed(both, case.notes)

    # written before any assertion, so a failure still leaves the answers to read
    _write_audit(
        case,
        {
            "lotus": _messages(both[0], case.view),
            "docetl": _messages(both[1], case.view),
        },
    )

    for name, memory in zip(("lotus", "docetl"), both, strict=True):
        assert _messages(memory, case.view) == sorted(case.expected), (
            name,
            _messages(memory, case.view),
        )
    assert_frame_equal(_kept(both[0], case.view), _kept(both[1], case.view))
    # a canned answer cannot move the counters, so a non-zero total proves a real request went out
    usage = docetl_adapter._context.usage_snapshot()
    assert sum(usage.values()) > 0, usage
