"""Policy-level parity: one `Memory` policy per operator shape, run through the runtime on both backends.

Between the adapter and a policy's published view sit the declared-column check, the index check, the
per-row cache and publication (`PolicyExecutor._execute_semantic_row`), so this asks what a *policy* makes
of the frame — not whether the operator lowers (`test_docetl_semantic_ops.py`) or whether the engines agree at
the operator level (`test_adapter_parity.py`).

This is the home for every operator's policy case: adding one means adding a `PolicyCase`, never a file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class PolicyCase:
    """One policy, the notes fed to it, and the notes its view must publish."""

    name: str
    policy: type[Memory]
    view: str
    notes: tuple[str, ...]
    expected: tuple[str, ...]
    keep_markers: tuple[str, ...]


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
        ),
    ]


def _real_cases() -> list[PolicyCase]:
    """The cases worth a real call: empty expectation is a publication test, covered canned."""

    return [case for case in _cases() if case.expected]


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
    """One policy, two engines, the kept notes published as the policy's own view."""

    with canned_model(*case.keep_markers) as endpoint:
        both = _build(case, endpoint.base_url, api_key=None)
        _feed(both, case.notes)

    for name, memory in zip(("lotus", "docetl"), both, strict=True):
        view = _kept(memory, case.view)
        assert list(view.columns) == ["message"], (name, list(view.columns))
        assert _messages(memory, case.view) == sorted(case.expected), (
            name,
            _messages(memory, case.view),
        )
    # frames, not names: a rebuilt frame can change a dtype, and `_align_output` checks names only
    assert_frame_equal(_kept(both[0], case.view), _kept(both[1], case.view))


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_the_canned_endpoint_publishes_each_note_as_it_arrives(case: PolicyCase) -> None:
    """Incremental ingestion: the view must converge one note at a time, on both engines."""

    with canned_model(*case.keep_markers) as endpoint:
        both = _build(case, endpoint.base_url, api_key=None)
        for position, note in enumerate(case.notes):
            _feed(both, (note,))
            expected = sorted(
                candidate
                for candidate in case.notes[: position + 1]
                if candidate in case.expected
            )
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
