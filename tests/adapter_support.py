"""Shared test machinery: one loopback model, one provider gate.

The endpoint is configuration (`lm_api_base`), not a patch, so both engines run their real operation
classes against a model that answers deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from typing import Any

from dotenv import load_dotenv
import pytest

from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.adapters.docetl.context import (
    DOCETL_API_BASE_ENV,
    DOCETL_MODEL_ENV,
    DocetlExecutionConfig,
)
from agent_memory.adapters.lotus.context import LotusExecutionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANNED_KEY = "canned-endpoint"
# litellm routes on the model prefix, and the configured value already carries one (`openai/...`)
CANNED_MODEL = "openai/canned-model"
REAL_RUN_ENV = "AGENT_MEMORY_RUN_DOCETL_E2E"


def _canned_value(
    schema: Mapping[str, Any],
    key: str,
    *,
    keeps: bool,
    malformed: bool,
    answers: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Any:
    """Return one value of the type the caller declared, so a map can be answered like a filter."""

    kind = schema.get("type")
    if malformed:
        # parses as JSON and fails the declared type: a string where a boolean was asked for, a number
        # where text was
        return "maybe" if kind == "boolean" else 123
    if kind == "boolean":
        return keeps
    if kind == "array":
        return [dict(row) for row in (rows or ())]
    return answers.get(key, "canned")


class _CannedHandler(BaseHTTPRequestHandler):
    """Answer both protocols: LOTUS reads an `Answer:` line, DocETL reads a tool call."""

    def do_POST(self) -> None:  # noqa: N802 - http.server defines the name
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        body = json.loads(raw)
        server: Any = self.server
        server.canned_requests.append(body)

        keeps = any(marker in raw for marker in server.canned_keep_markers)
        malformed = server.canned_refuse or any(
            marker in raw for marker in server.canned_malformed_markers
        )

        tools = body.get("tools") or []
        properties = tools[0]["function"]["parameters"]["properties"] if tools else {}
        if properties:
            arguments = {
                key: _canned_value(
                    schema,
                    key,
                    keeps=keeps,
                    malformed=malformed,
                    answers=server.canned_map_answers,
                    rows=server.canned_map_rows,
                )
                for key, schema in properties.items()
            }
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "canned",
                        "type": "function",
                        "function": {
                            "name": tools[0]["function"]["name"],
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        elif server.canned_map_rows is not None:
            # LOTUS asks for emitted rows under a `rows` key (`lotus/structured.py::FLAT_MAP_ROWS_FIELD`)
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {"rows": [dict(row) for row in server.canned_map_rows]}
                ),
            }
            finish = "stop"
        elif server.canned_map_answers:
            values = list(server.canned_map_answers.values())
            message = {
                "role": "assistant",
                "content": (
                    str(values[0])
                    if len(values) == 1
                    else json.dumps(dict(server.canned_map_answers))
                ),
            }
            finish = "stop"
        else:
            message = {
                "role": "assistant",
                "content": "not an answer" if malformed else f"Answer: {keeps}",
            }
            finish = "stop"

        payload = json.dumps(
            {
                "id": "canned",
                "object": "chat.completion",
                "created": 0,
                "model": "canned",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        """Stay quiet: a suite's output is its assertions, not an access log."""


@dataclass
class CannedEndpoint:
    """A running loopback model, the requests it saw, and a switch to make it unusable."""

    base_url: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    _server: Any = None

    def refuse(self, refusing: bool = True) -> None:
        self._server.canned_refuse = refusing


@contextmanager
def canned_model(
    *keep_markers: str,
    malformed_markers: Sequence[str] = (),
    map_answers: Mapping[str, Any] | None = None,
    map_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[CannedEndpoint]:
    """Serve canned completions; a request carrying a keep marker is kept.

    `map_answers` and `map_rows` answer a column-filling operator — a value per declared column, or the
    rows a flat map should emit — because such a request cannot be answered with a verdict. Passing an
    empty `map_rows` is how the "emits nothing" case is served.

    `OPENAI_API_KEY` is set for the duration and restored afterwards, so a free run cannot leave a
    placeholder key behind for a billed one.
    """

    previous_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = CANNED_KEY

    server = ThreadingHTTPServer(("127.0.0.1", 0), _CannedHandler)
    server.canned_keep_markers = tuple(keep_markers)  # type: ignore[attr-defined]
    server.canned_malformed_markers = tuple(malformed_markers)  # type: ignore[attr-defined]
    server.canned_map_answers = dict(map_answers or {})  # type: ignore[attr-defined]
    server.canned_map_rows = (  # type: ignore[attr-defined]
        None if map_rows is None else tuple(dict(row) for row in map_rows)
    )
    server.canned_refuse = False  # type: ignore[attr-defined]
    server.canned_requests = []  # type: ignore[attr-defined]
    Thread(target=server.serve_forever, daemon=True).start()

    endpoint = CannedEndpoint(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        requests=server.canned_requests,  # type: ignore[attr-defined]
        _server=server,
    )
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        if previous_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_key


def build_adapters(
    endpoint: str,
    *,
    model: str = CANNED_MODEL,
    api_key: str | None = CANNED_KEY,
    completion_kwargs: Mapping[str, Any] | None = None,
) -> tuple[LotusAdapter, DocetlAdapter]:
    """Point both backends at one endpoint, spelled the way each of them takes it.

    `api_key=None` sends both sides to the environment, which is how a real run is configured.
    `completion_kwargs` pins provider knobs — `temperature` above all, because two engines that ask
    differently cannot be compared while sampling is free to differ.
    """

    lotus_kwargs: dict[str, Any] = {
        "api_base": endpoint,
        **(completion_kwargs or {}),
    }
    if api_key is not None:
        lotus_kwargs["api_key"] = api_key
    return (
        LotusAdapter(
            model=model,
            config=LotusExecutionConfig(lm_model_kwargs=lotus_kwargs),
        ),
        DocetlAdapter(
            model=model,
            config=DocetlExecutionConfig(
                lm_api_base=endpoint,
                litellm_completion_kwargs=dict(completion_kwargs or {}),
            ),
        ),
    )


def default_adapters() -> tuple[LotusAdapter, DocetlAdapter]:
    """Both backends with their default configuration, for a case that calls no model."""

    return LotusAdapter(), DocetlAdapter()


def build_docetl_adapter(
    endpoint: str,
    *,
    model: str = CANNED_MODEL,
) -> DocetlAdapter:
    """The DocETL backend alone, for a suite that is not comparing engines."""

    return build_adapters(endpoint, model=model)[1]


def require_real_provider(purpose: str) -> tuple[str, str]:
    """Return (model, base_url) for a gated run, or skip when that run was not asked for.

    A run that was asked for and cannot reach a provider fails rather than skips.
    """

    load_dotenv(PROJECT_ROOT / ".env")
    if os.getenv(REAL_RUN_ENV) != "1":
        pytest.skip(f"set {REAL_RUN_ENV}=1 to run {purpose}")

    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.fail(f"DEEPSEEK_API_KEY is required for {purpose}")
    model = os.getenv(DOCETL_MODEL_ENV)
    base_url = os.getenv(DOCETL_API_BASE_ENV)
    if not (model and base_url):
        pytest.fail(f"set {DOCETL_MODEL_ENV} and {DOCETL_API_BASE_ENV} for {purpose}")
    # the `openai/` route reads this variable, and setting it beats inheriting an unrelated real key
    os.environ["OPENAI_API_KEY"] = key
    return model, base_url
