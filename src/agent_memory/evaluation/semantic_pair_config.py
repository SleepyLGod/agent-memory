"""Load benchmark physical profiles bound to semantic predicate sites."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from agent_memory.adapters.lotus.pair_execution import (
    SEMANTIC_PAIR_EXECUTION_MODES,
)


@dataclass(frozen=True)
class SemanticPairSiteBinding:
    """One user-selected physical profile for a semantic predicate site."""

    site_id: str
    mode: str
    top_k: int | None
    min_similarity: float | None

    def __post_init__(self) -> None:
        """Validate the profile independently of a compiled policy."""

        if not self.site_id:
            raise ValueError("semantic pair site_id must be non-empty")
        if self.mode not in SEMANTIC_PAIR_EXECUTION_MODES:
            raise ValueError(
                "semantic pair site mode must be one of: "
                + ", ".join(SEMANTIC_PAIR_EXECUTION_MODES)
            )
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValueError("semantic pair site top_k must be a positive integer")
        if self.min_similarity is not None and (
            not isinstance(self.min_similarity, (int, float))
            or isinstance(self.min_similarity, bool)
            or not math.isfinite(float(self.min_similarity))
        ):
            raise ValueError("semantic pair site min_similarity must be finite")
        if self.mode == "oracle-only" and (
            self.top_k is not None or self.min_similarity is not None
        ):
            raise ValueError("oracle-only site bindings cannot select candidates")
        if self.mode == "search-filter" and (
            self.top_k is None and self.min_similarity is None
        ):
            raise ValueError("search-filter site bindings require a candidate bound")
        if self.mode == "proxy-only" and self.min_similarity is None:
            raise ValueError("proxy-only site bindings require min_similarity")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical manifest representation."""

        return {
            "site_id": self.site_id,
            "mode": self.mode,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
        }


@dataclass(frozen=True)
class SemanticPairProfileConfig:
    """A validated site binding file and its immutable source digest."""

    bindings: tuple[SemanticPairSiteBinding, ...]
    source_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return normalized provenance without the machine-local path."""

        return {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }


def load_semantic_pair_profile_config(path: Path) -> SemanticPairProfileConfig:
    """Load one strict JSON profile without initializing external resources."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read semantic pair profile config {path}: {error}") from error
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid semantic pair profile config JSON {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise TypeError("semantic pair profile config must be a JSON object")
    _require_exact_keys(payload, {"schema_version", "bindings"}, "config")
    if payload["schema_version"] != 1:
        raise ValueError("semantic pair profile config schema_version must be 1")
    raw_bindings = payload["bindings"]
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("semantic pair profile config bindings must be non-empty")

    bindings: list[SemanticPairSiteBinding] = []
    seen_sites: set[str] = set()
    for index, item in enumerate(raw_bindings):
        if not isinstance(item, dict):
            raise TypeError(f"semantic pair binding {index} must be an object")
        _require_exact_keys(
            item,
            {"site_id", "mode", "top_k", "min_similarity"},
            f"binding {index}",
        )
        site_id = item["site_id"]
        mode = item["mode"]
        if not isinstance(site_id, str) or not isinstance(mode, str):
            raise TypeError(f"semantic pair binding {index} IDs must be strings")
        if site_id in seen_sites:
            raise ValueError(f"duplicate semantic pair site binding {site_id!r}")
        seen_sites.add(site_id)
        bindings.append(
            SemanticPairSiteBinding(
                site_id=site_id,
                mode=mode,
                top_k=item["top_k"],
                min_similarity=item["min_similarity"],
            )
        )
    return SemanticPairProfileConfig(
        bindings=tuple(bindings),
        source_sha256=sha256(content).hexdigest(),
    )


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"semantic pair {label} keys must be {sorted(expected)}, got {sorted(actual)}"
        )


__all__ = [
    "SemanticPairProfileConfig",
    "SemanticPairSiteBinding",
    "load_semantic_pair_profile_config",
]
