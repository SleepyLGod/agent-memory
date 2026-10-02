"""Workload scope checks stop before source validation or any provider setup."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import zep_combined_smoke as smoke


@pytest.mark.parametrize("counts, error", [
    ((419, 198), "wrong AM source imported"),
    ((128, 198), "expected frozen bundle"),
    ((419, 69), "expected frozen bundle"),
])
def test_full_combined_scope_keeps_exact_sample_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, counts: tuple[int, int], error: str,
) -> None:
    case = SimpleNamespace(
        events=tuple(SimpleNamespace(event_id=f"e{i}") for i in range(counts[0])),
        questions=tuple(SimpleNamespace(evidence_event_ids=()) for _ in range(counts[1])),
    )
    monkeypatch.setattr(smoke, "read_bundle", lambda _: SimpleNamespace(cases=(case,)))
    with pytest.raises(ValueError, match=error):
        smoke.execute(tmp_path, "preflight", full_sample0=True, maintenance_work=True,
            combined_physical=True, fact_summary=True, representative=True,
            node_batch_size=16, pack_small_groups=True, parallel_fact_extraction=True,
            lotus_cache_mode="memory")


def test_prefix_scope_still_requires_complete_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    case = SimpleNamespace(events=(SimpleNamespace(event_id="e1"),),
                           questions=(SimpleNamespace(evidence_event_ids=()),))
    monkeypatch.setattr(smoke, "read_bundle", lambda _: SimpleNamespace(cases=(case,)))
    with pytest.raises(ValueError, match="complete evidence"):
        smoke.execute(tmp_path, "preflight", prefix128=True, maintenance_work=True,
                      combined_physical=True, fact_summary=True, representative=True)


def test_full_and_prefix_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="choose full Sample 0"):
        smoke.execute(tmp_path, "preflight", full_sample0=True, prefix128=True)
