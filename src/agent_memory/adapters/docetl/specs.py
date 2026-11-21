"""The semantic half of the operation-scoped channel: the spec frame DocETL operations receive.

A spec is the `dict` handed to a DocETL operation class's constructor
(`context.py::DocetlExecutionContext.run_operation`); the keys here are DocETL's, not ours. The
other channels live in `context.py`: runner-scoped settings (`runner_config`), constructor-only
arguments (`run_operation`), and the uniform operation-scoped defaults (`with_op_defaults`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def op_config(
    name: str,
    op_type: str,
    prompt: str,
    schema: Mapping[str, str],
) -> dict[str, Any]:
    """Return one semantic DocETL op spec: the frame every build_*_config fills in."""

    return {
        "name": name,
        "type": op_type,
        "prompt": prompt,
        "output": {"schema": dict(schema)},
    }
