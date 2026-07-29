from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from agent_memory.evaluation.provenance import (
    collect_runtime_provenance,
    validate_run_provenance,
)


def test_runtime_provenance_records_git_and_lockfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    outputs = iter(("commit", ""))
    monkeypatch.setattr(
        "agent_memory.evaluation.provenance._git",
        lambda *_args: next(outputs),
    )

    provenance = collect_runtime_provenance(
        tmp_path,
        lockfile="uv.lock",
        dependencies=(),
    )

    assert provenance["source"] == {"commit": "commit", "dirty": False}
    assert len(provenance["runtime"]["lockfile_sha256"]) == 64


def test_formal_run_rejects_dirty_source_but_smoke_records_it() -> None:
    provenance = {
        "source": {"commit": "commit", "dirty": True},
        "runtime": {"lockfile_sha256": "digest"},
    }

    validate_run_provenance(provenance, run_mode="integration-smoke")
    with pytest.raises(RuntimeError, match="clean source"):
        validate_run_provenance(provenance, run_mode="full")


def test_runtime_provenance_rejects_a_missing_required_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    outputs = iter(("commit", ""))
    monkeypatch.setattr(
        "agent_memory.evaluation.provenance._git",
        lambda *_args: next(outputs),
    )
    monkeypatch.setattr(
        "agent_memory.evaluation.provenance.version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError),
    )

    with pytest.raises(RuntimeError, match="is not installed"):
        collect_runtime_provenance(
            tmp_path,
            lockfile="uv.lock",
            dependencies=("missing",),
        )
