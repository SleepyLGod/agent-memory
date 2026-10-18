"""Gated real-model operator audit: are the answers right?

The only suite that asserts a real model's *decisions*. One `AuditCase` per operator; the CSVs are written
before the assertions so a failure still leaves the answers to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from agent_memory.policy.logical import QueryExpr

from adapter_support import build_docetl_adapter, require_real_provider

AUDIT_DIR = Path("/tmp/agent-memory-docetl-operator-audit") / datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")

# the lowering states no polarity, so the instruction must (`sem_filter.py::build_filter_config`)
INSTRUCTION = (
    "Decide whether this note asks for a refund: {note}\n\n"
    "Respond with 'true' if the note asks for a refund, otherwise respond with 'false'."
)

# plain in the text: the first two ask for money back, the last two do not
ROWS = (
    "I would like a refund for the order that arrived broken.",
    "Please return the money for the duplicate charge on my card.",
    "The weather here has been lovely all week.",
    "I finished the book you lent me, thank you.",
)


@dataclass(frozen=True)
class AuditCase:
    """One operator, the rows it runs on, and the rows a correct run keeps."""

    name: str
    query: QueryExpr
    frame: pd.DataFrame
    expected_index: tuple[int, ...]


def _cases() -> list[AuditCase]:
    return [
        AuditCase(
            name="sem_filter",
            query=QueryExpr(
                op="sem_filter",
                inputs=(QueryExpr(op="log"),),
                params={"instruction": INSTRUCTION},
            ),
            frame=pd.DataFrame({"note": list(ROWS)}, index=[10, 11, 12, 13]),
            expected_index=(10, 11),
        )
    ]


def _write_audit(case: AuditCase, output: pd.DataFrame) -> None:
    """Write the CSVs and print which rows the model kept, which is what a human reads."""

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = AUDIT_DIR / f"{case.name}__input.csv"
    output_path = AUDIT_DIR / f"{case.name}__output.csv"
    case.frame.to_csv(input_path, index=False)
    output.to_csv(output_path, index=False)

    kept = set(output.index)
    print(f"\n{case.name}")
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"kept {len(kept)} of {len(case.frame)} rows")
    for index, note in zip(case.frame.index, case.frame["note"], strict=True):
        print(f"  [{'keep' if index in kept else 'drop'}] {note}")


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.name)
def test_the_real_model_answers_correctly(case: AuditCase) -> None:
    """Run one operator on the real API and hold its verdicts to what the rows actually say."""

    model, base_url = require_real_provider(f"the real-model audit ({case.name})")
    adapter = build_docetl_adapter(base_url, model=model)

    output = adapter.execute(case.query, {"log": case.frame})

    # written before any assertion, so a failure still leaves the answers to read
    _write_audit(case, output)

    assert list(output.columns) == ["note"]
    assert set(output.index).issubset(set(case.frame.index))
    assert set(output.index) == set(case.expected_index), sorted(output.index)
    # a canned answer cannot move the counters, so a non-zero total proves a request really went out
    usage = adapter._context.usage_snapshot()
    assert sum(usage.values()) > 0, usage
