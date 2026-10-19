"""Gated real-model operator audit: are the answers right?

The only suite that asserts a real model's *decisions*. One `AuditCase` per operator, each with the check
that operator can actually promise — a verdict set for a filter, a row count for a map, an expansion for a
flat map — and the CSVs are written before the assertions so a failure still leaves the answers to read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from agent_memory.policy.logical import ColumnSpec, QueryExpr

from adapter_support import build_docetl_adapter, require_real_provider

AUDIT_DIR = Path("/tmp/agent-memory-docetl-operator-audit") / datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")

# the lowering states no polarity, so the instruction must (`sem_filter.py::build_filter_config`)
FILTER_INSTRUCTION = (
    "Decide whether this note asks for a refund: {note}\n\n"
    "Respond with 'true' if the note asks for a refund, otherwise respond with 'false'."
)

MAP_INSTRUCTION = "Summarise this note in one short sentence: {note}"

FLAT_MAP_INSTRUCTION = (
    "Split this note into its clauses, one entry per clause: {note}\n\n"
    "Return exactly one entry when the note has a single clause."
)

# plain in the text: the first two ask for money back, the last two do not
ROWS = (
    "I would like a refund for the order that arrived broken.",
    "Please return the money for the duplicate charge on my card.",
    "The weather here has been lovely all week.",
    "I finished the book you lent me, thank you.",
)

INDEX = [10, 11, 12, 13]


@dataclass(frozen=True)
class AuditCase:
    """One operator, the rows it runs on, and the check only that operator can promise."""

    name: str
    query: QueryExpr
    frame: pd.DataFrame
    expected_columns: tuple[str, ...]
    check: Callable[[pd.DataFrame], None]


def _kept_only_the_refunds(output: pd.DataFrame) -> None:
    assert set(output.index) == {10, 11}, sorted(output.index)


def _one_summary_per_note(output: pd.DataFrame) -> None:
    assert len(output) == len(INDEX), len(output)
    assert output["summary"].map(lambda value: bool(str(value).strip())).all()


def _each_note_splits_into_numbered_clauses(output: pd.DataFrame) -> None:
    for index in INDEX:
        rows = output.loc[output.index == index]
        assert not rows.empty, f"note {index} produced no clause"
        assert rows["clause_number"].tolist() == list(range(len(rows))), (
            index,
            rows["clause_number"].tolist(),
        )
        assert rows["clause"].map(lambda value: bool(str(value).strip())).all()


def _cases() -> list[AuditCase]:
    return [
        AuditCase(
            name="sem_filter",
            query=QueryExpr(
                op="sem_filter",
                inputs=(QueryExpr(op="log"),),
                params={"instruction": FILTER_INSTRUCTION},
            ),
            frame=pd.DataFrame({"note": list(ROWS)}, index=INDEX),
            expected_columns=("note",),
            check=_kept_only_the_refunds,
        ),
        AuditCase(
            name="sem_map",
            query=QueryExpr(
                op="sem_map",
                inputs=(QueryExpr(op="log"),),
                params={
                    "instruction": MAP_INSTRUCTION,
                    "output_cols": (
                        ColumnSpec(name="summary", description="One short sentence."),
                    ),
                },
            ),
            frame=pd.DataFrame({"note": list(ROWS)}, index=INDEX),
            expected_columns=("note", "summary"),
            check=_one_summary_per_note,
        ),
        AuditCase(
            name="sem_flat_map",
            query=QueryExpr(
                op="sem_flat_map",
                inputs=(QueryExpr(op="log"),),
                params={
                    "instruction": FLAT_MAP_INSTRUCTION,
                    "output_cols": (
                        ColumnSpec(name="clause", description="One clause of the note."),
                    ),
                    "ordinal_col": "clause_number",
                },
            ),
            frame=pd.DataFrame({"note": list(ROWS)}, index=INDEX),
            expected_columns=("note", "clause", "clause_number"),
            check=_each_note_splits_into_numbered_clauses,
        ),
    ]


def _write_audit(case: AuditCase, output: pd.DataFrame) -> None:
    """Write the CSVs and print each input row with the rows it produced."""

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = AUDIT_DIR / f"{case.name}__input.csv"
    output_path = AUDIT_DIR / f"{case.name}__output.csv"
    case.frame.to_csv(input_path, index=False)
    output.to_csv(output_path, index=False)

    present = set(output.index)
    print(f"\n{case.name}")
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"{len(output)} rows from {len(case.frame)} input rows")
    for index, note in zip(case.frame.index, case.frame["note"], strict=True):
        print(f"  [{'present' if index in present else 'missing'}] {note}")


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.name)
def test_the_real_model_answers_correctly(case: AuditCase) -> None:
    """Run one operator on the real API and hold its output to what the rows actually say."""

    model, base_url = require_real_provider(f"the real-model audit ({case.name})")
    adapter = build_docetl_adapter(base_url, model=model)

    output = adapter.execute(case.query, {"log": case.frame})

    # written before any assertion, so a failure still leaves the answers to read
    _write_audit(case, output)

    assert list(output.columns) == list(case.expected_columns), list(output.columns)
    assert set(output.index).issubset(set(case.frame.index)), sorted(output.index)
    case.check(output)
    # a canned answer cannot move the counters, so a non-zero total proves a request really went out
    usage = adapter._context.usage_snapshot()
    assert sum(usage.values()) > 0, usage
