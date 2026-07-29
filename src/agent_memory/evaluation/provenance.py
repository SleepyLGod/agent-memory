"""Reproducible source and runtime evidence for benchmark manifests."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping


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
    return {
        "source": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "dirty": bool(_git(root, "status", "--porcelain=v1", "--untracked-files=normal")),
        },
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
    """Reject dirty source for formal runs while allowing explicit smoke runs."""

    source = provenance.get("source")
    runtime = provenance.get("runtime")
    if not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("benchmark provenance requires source and runtime objects")
    if not isinstance(source.get("commit"), str) or not source["commit"]:
        raise ValueError("benchmark provenance requires a source commit")
    if not isinstance(source.get("dirty"), bool):
        raise ValueError("benchmark provenance requires a boolean dirty state")
    if not isinstance(runtime.get("lockfile_sha256"), str) or not runtime[
        "lockfile_sha256"
    ]:
        raise ValueError("benchmark provenance requires a lockfile digest")
    if not run_mode.startswith("integration-smoke") and source["dirty"]:
        raise RuntimeError("formal benchmark runs require a clean source checkout")


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


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"benchmark dependency {package!r} is not installed"
        ) from error


__all__ = ["collect_runtime_provenance", "validate_run_provenance"]
