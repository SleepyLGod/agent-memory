from __future__ import annotations

from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
import subprocess

import pytest

from agent_memory.evaluation.provenance import (
    SOURCE_EVIDENCE_ENV,
    build_source_evidence,
    collect_runtime_provenance,
    validate_run_provenance,
)


def test_runtime_provenance_records_git_and_lockfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOURCE_EVIDENCE_ENV, raising=False)
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


def test_formal_run_requires_validated_source_evidence() -> None:
    provenance = {
        "source": {"commit": "commit", "dirty": True},
        "runtime": {"lockfile_sha256": "digest"},
    }

    validate_run_provenance(provenance, run_mode="integration-smoke")
    with pytest.raises(RuntimeError, match="source inventory evidence"):
        validate_run_provenance(provenance, run_mode="full")


def test_source_evidence_binds_the_complete_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "config", "user.email", "benchmark@example.com"),
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Benchmark"),
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "uv.lock", "source.py"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "fixture"),
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "source.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("evidence\n", encoding="utf-8")
    evidence_path = tmp_path.parent / f"{tmp_path.name}-source-evidence.json"
    evidence_path.write_text(
        json.dumps(build_source_evidence(tmp_path)),
        encoding="utf-8",
    )
    monkeypatch.setenv(SOURCE_EVIDENCE_ENV, str(evidence_path))

    provenance = collect_runtime_provenance(
        tmp_path,
        lockfile="uv.lock",
        dependencies=(),
    )
    validate_run_provenance(provenance, run_mode="full")

    assert provenance["source"]["dirty"] is True
    assert provenance["source"]["source_file_count"] == 3
    assert len(provenance["source"]["source_snapshot_sha256"]) == 64

    (tmp_path / "source.py").write_text("value = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory"):
        collect_runtime_provenance(
            tmp_path,
            lockfile="uv.lock",
            dependencies=(),
        )


def test_runtime_provenance_rejects_a_missing_required_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOURCE_EVIDENCE_ENV, raising=False)
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
