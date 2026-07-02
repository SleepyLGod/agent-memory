"""ClaudeMemory-specific evaluation bindings."""

from __future__ import annotations

from typing import Any

from agent_memory.evaluation.types import BenchmarkEvent


def event_to_claude_log_row(event: BenchmarkEvent) -> dict[str, Any]:
    """Render one LOCOMO event into the current ClaudeMemory log schema."""

    return {
        "message": event.text,
        "role": event.speaker,
        "timestamp": event.timestamp,
        "session_id": event.session_id,
    }
