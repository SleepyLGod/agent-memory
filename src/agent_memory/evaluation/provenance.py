"""Reproducible source and runtime evidence for benchmark manifests."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping

SOURCE_EVIDENCE_ENV = "BENCHMARK_SOURCE_EVIDENCE"


def build_source_evidence(repository: Path) -> dict[str, Any]:
    """Build the controller-owned evidence for one exact worktree."""

    root = repository.resolve()
    files = _source_inventory(root)
    return {
        "schema_version": 1,
        "repository": str(root),
        "head": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(
            _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
        ),
        "dirty_patch_sha256": sha256(
            _git_bytes(root, "diff", "--binary", "HEAD")
        ).hexdigest(),
        "untracked_inventory_sha256": _digest_json(
            [row for row in files if row["untracked"]]
        ),
        "source_snapshot_sha256": _digest_json(files),
        "files": files,
    }


def collect_runtime_provenance(
    repository: Path,
    *,
    lockfile: str,
    dependencies: Iterable[str],
) -> dict[str, Any]:
    """Collect source, lockfile, interpreter, and dependency versions."""

    root = repository.resolve()
    lock_path = root / lockfile
    if not lock_path.is_file():
        raise FileNotFoundError(f"benchmark lockfile does not exist: {lock_path}")
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(
        _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    )
    source: dict[str, Any] = {
        "commit": commit,
        "dirty": dirty,
    }
    evidence_path = os.environ.get(SOURCE_EVIDENCE_ENV)
    if evidence_path:
        evidence = _validate_source_evidence(
            root,
            Path(evidence_path).expanduser().resolve(),
            commit=commit,
            dirty=dirty,
        )
        source.update(
            {
                "source_snapshot_sha256": evidence[
                    "source_snapshot_sha256"
                ],
                "dirty_patch_sha256": evidence["dirty_patch_sha256"],
                "untracked_inventory_sha256": evidence[
                    "untracked_inventory_sha256"
                ],
                "source_file_count": len(evidence["files"]),
                "evidence_path": str(
                    Path(evidence_path).expanduser().resolve()
                ),
                "evidence_sha256": _sha256_file(
                    Path(evidence_path).expanduser().resolve()
                ),
            }
        )
    return {
        "source": source,
        "runtime": {
            "language": "python",
            "language_version": platform.python_version(),
            "lockfile": lockfile,
            "lockfile_sha256": _sha256_file(lock_path),
            "dependencies": {
                dependency: _package_version(dependency)
                for dependency in sorted(set(dependencies))
            },
        },
    }


def validate_run_provenance(
    provenance: Mapping[str, Any],
    *,
    run_mode: str,
) -> None:
    """Require formal runs to identify their exact source and runtime."""

    source = provenance.get("source")
    runtime = provenance.get("runtime")
    if not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("benchmark provenance requires source and runtime objects")
    if not isinstance(source.get("commit"), str) or not source["commit"]:
        raise ValueError("benchmark provenance requires a source commit")
    if not isinstance(source.get("dirty"), bool):
        raise ValueError("benchmark provenance requires a boolean dirty state")
    source_snapshot = source.get("source_snapshot_sha256")
    if not isinstance(runtime.get("lockfile_sha256"), str) or not runtime[
        "lockfile_sha256"
    ]:
        raise ValueError("benchmark provenance requires a lockfile digest")
    if (
        not run_mode.startswith("integration-smoke")
        and (
            not isinstance(source_snapshot, str)
            or len(source_snapshot) != 64
            or not isinstance(source.get("evidence_sha256"), str)
        )
    ):
        raise RuntimeError(
            "formal benchmark runs require validated source inventory evidence"
        )


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_evidence(
    repository: Path,
    evidence_path: Path,
    *,
    commit: str,
    dirty: bool,
) -> Mapping[str, Any]:
    if not evidence_path.is_file():
        raise FileNotFoundError(
            f"benchmark source evidence does not exist: {evidence_path}"
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != 1:
        raise ValueError("benchmark source evidence must use schema version 1")
    if evidence.get("head") != commit or evidence.get("dirty") is not dirty:
        raise ValueError("benchmark source evidence does not match Git state")

    actual_files = _source_inventory(repository)
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or evidence_files != actual_files:
        raise ValueError(
            "benchmark source evidence file inventory does not match worktree"
        )
    snapshot_sha256 = _digest_json(actual_files)
    patch_sha256 = sha256(
        _git_bytes(repository, "diff", "--binary", "HEAD")
    ).hexdigest()
    untracked_files = [
        row for row in actual_files if bool(row.get("untracked"))
    ]
    untracked_sha256 = _digest_json(untracked_files)
    expected = {
        "source_snapshot_sha256": snapshot_sha256,
        "dirty_patch_sha256": patch_sha256,
        "untracked_inventory_sha256": untracked_sha256,
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("benchmark source evidence digests do not match worktree")
    return evidence


def _source_inventory(repository: Path) -> list[dict[str, Any]]:
    tracked = set(
        _git_paths(repository, "ls-files", "--cached")
    )
    untracked = set(
        _git_paths(repository, "ls-files", "--others", "--exclude-standard")
    )
    rows = []
    for relative in sorted(tracked | untracked):
        path = repository / relative
        if not path.is_file() and not path.is_symlink():
            raise ValueError(f"benchmark source file is missing: {relative}")
        payload = (
            os.readlink(path).encode("utf-8")
            if path.is_symlink()
            else path.read_bytes()
        )
        rows.append(
            {
                "path": relative,
                "sha256": sha256(payload).hexdigest(),
                "untracked": relative in untracked,
            }
        )
    return rows


def _git_paths(repository: Path, *arguments: str) -> list[str]:
    output = _git_bytes(repository, *arguments, "-z")
    return [
        value.decode("utf-8")
        for value in output.split(b"\0")
        if value
    ]


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    ).stdout


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"benchmark dependency {package!r} is not installed"
        ) from error


__all__ = [
    "SOURCE_EVIDENCE_ENV",
    "build_source_evidence",
    "collect_runtime_provenance",
    "validate_run_provenance",
]
