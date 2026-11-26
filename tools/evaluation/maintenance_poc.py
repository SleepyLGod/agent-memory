"""One original case, one condition, one process; dry compilation by default."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import html
import importlib.util
import ipaddress
import json
import os
import pickle
from pathlib import Path
import socket
import sys
import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import pandas as pd
    import agent_memory as am
    from agent_memory.planner.rules import DifferentialRules

MODEL = "deepseek/deepseek-flash"


def model_settings(
    backend: str,
    model: str | None,
    base_url: str | None,
    output_tokens: int,
    *,
    thinking: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve experiment settings without reading credentials or contacting a model."""
    if backend == "deepseek":
        if thinking not in (None, "enabled", "disabled"):
            raise ValueError("DeepSeek thinking must be enabled or disabled")
        thinking = thinking or "disabled"
        if base_url is not None:
            raise ValueError("--base-url is only supported with --backend chatgpt-plan")
        name = model or MODEL
        if not name.startswith("deepseek/") or not name.removeprefix("deepseek/"):
            raise ValueError("DeepSeek model must use deepseek/<model>")
        options = {
            "max_tokens": output_tokens,
            "temperature": 0,
            "extra_body": {"thinking": {"type": thinking}},
        }
        return (
            name,
            options,
            {
                "backend": backend,
                "model": name,
                "requested_parameters": options,
                "known_omissions": [],
                "thinking": thinking,
                "output_tokens_per_request": output_tokens,
            },
        )
    if backend != "chatgpt-plan":
        raise ValueError("Unsupported experiment backend")
    if thinking is not None:
        raise ValueError("--thinking is only supported with --backend deepseek")
    if (
        not model
        or model.strip() != model
        or "/" in model
        or any(c.isspace() for c in model)
    ):
        raise ValueError("Gateway requires --model with an unprefixed model slug")
    url = base_url or "http://127.0.0.1:8787/v1"
    parsed = urlsplit(url)
    try:
        local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        port = parsed.port
    except ValueError:
        local, port = False, None
    if (
        not local
        or parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
        or port == 0
    ):
        raise ValueError(
            "Gateway URL must be http://<numeric-loopback>[:port]/v1 without credentials or query"
        )
    name = f"openai/{model}"
    options = {"api_base": url.rstrip("/")}
    return (
        name,
        options,
        {
            "backend": backend,
            "model": name,
            "base_url": options["api_base"],
            "requested_parameters": {
                "structured_calls": {"max_tokens": output_tokens},
                "other_parameters": "LOTUS defaults; see gateway history for each request",
            },
            "known_omissions": ["max_tokens", "temperature"],
            "parameter_note": "LOTUS supplies these defaults; gateway profile=lotus omits them. Gateway history records actual transformations.",
            "thinking": "provider-default; not controlled",
            "output_tokens_per_request": None,
        },
    )


def write_json(path: Path, value: Any) -> None:
    """Write experiment artifacts, not source data."""
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class Progress:
    """Publish one small observational record, independent of runtime state."""

    def __init__(
        self, directory: Path, case: str, mode: str, total: int, *, model: str = MODEL
    ) -> None:
        self.directory = directory
        self.started = time.time()
        self.data: dict[str, Any] = {
            "case": case,
            "condition": mode,
            "model": model,
            "status": "running",
            "stage": "initializing",
            "completed_steps": 0,
            "total_steps": total,
            "requests": 0,
            "estimated_reserved_tokens": 0,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": self.started,
        }
        self.update()

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.data["updated_at"] = time.time()
        self.data["elapsed_seconds"] = time.time() - self.started
        write_json(self.directory / "progress.json", self.data)


def read_status(directory: Path) -> dict:
    """Read a run status; a dead process is not reported as still running."""
    data = json.loads((directory / "progress.json").read_text())
    if data["status"] == "running" and data["host"] == socket.gethostname():
        try:
            os.kill(data["pid"], 0)
        except ProcessLookupError:
            data["status"] = "process-exited-without-final-status"
        except PermissionError:
            data["status"] = "process-status-unavailable"
        if data["status"] == "running":
            data["elapsed_seconds"] = time.time() - data["started_at"]
    return data


def show_status(directory: Path, *, watch: bool = False) -> None:
    """Watch experiment records only; never import or call the model stack."""
    try:
        while True:
            paths = sorted(directory.glob("*/progress.json"))
            if (directory / "progress.json").exists():
                paths = [directory / "progress.json"]
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(
                time.strftime("%H:%M:%S"),
                "case / condition / model | stage | steps | requests | seconds | status",
                flush=True,
            )
            for path in paths:
                data = read_status(path.parent)
                print(
                    f"{data['case']} / {data['condition']} / {data['model']} | {data['stage']} | "
                    f"{data['completed_steps']}/{data['total_steps']} | {data['requests']} | "
                    f"{data['elapsed_seconds']:.0f}s | {data['status']}",
                    flush=True,
                )
                if data.get("error"):
                    print(f"  {data['error']}", flush=True)
            if not paths:
                print("No runs have started in this directory.", flush=True)
            if not watch:
                return
            time.sleep(3)
    except KeyboardInterrupt:
        return


def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Use pandas' JSON conversion for nullable and NumPy values."""
    return json.loads(cast(str, frame.to_json(orient="records", force_ascii=False)))


def digest(path: Path) -> str:
    """Fingerprint bytes without changing them."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoExecution:
    """Fail if a dry compilation attempts execution."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"Dry run attempted execution: {name}")


class BudgetExceeded(RuntimeError):
    """A request was stopped before model execution."""


class BudgetLM:
    """PoC-only request accounting; token reservations are tokenizer estimates."""

    def __init__(
        self,
        lm: Any,
        directory: Path,
        requests: int,
        tokens: int,
        output: int,
        progress: Progress | None = None,
        *,
        initial: dict | None = None,
        min_interval: float = 0,
        completion_limit: int | None = None,
        ledger: Path | None = None,
    ) -> None:
        if min(requests, tokens, output) <= 0:
            raise ValueError("All budget limits must be positive")
        self.lm = lm
        self.directory = directory
        self.request_limit, self.token_limit, self.output_limit = (
            requests,
            tokens,
            output,
        )
        self.requests = 0
        self.reserved_tokens = 0
        self.calls = 0
        self.progress = progress
        self.min_interval = min_interval
        self.completion_limit = completion_limit
        self.completion_tokens = 0
        self.unknown_usage_requests = 0
        self.ledger = ledger
        self.last_started = 0.0
        if ledger is not None:
            if initial is not None:
                raise ValueError("Shared budget must not import a prior experiment's usage")
            initial = json.loads(ledger.read_text()) if ledger.exists() else None
        if initial is not None:
            self.requests = initial["requests"]
            self.reserved_tokens = initial["estimated_reserved_tokens"]
            self.calls = initial["calls"]
            self.completion_tokens = initial["completion_tokens"]
            self.last_started = initial["last_started"]
            self.unknown_usage_requests = initial.get("unknown_usage_requests", 0)

    def accounting(self) -> dict:
        """Persist cumulative attempts separately from committed runtime state."""
        return {"requests": self.requests, "calls": self.calls,
                "estimated_reserved_tokens": self.reserved_tokens,
                "completion_tokens": self.completion_tokens,
                "unknown_usage_requests": self.unknown_usage_requests,
                "last_started": self.last_started}

    def persist(self) -> None:
        """Record attempted calls before sending, independently of state commits."""
        write_json(self.directory.parent / "budget-live.json", self.accounting())
        if self.ledger is not None:
            write_json(self.ledger, self.accounting())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.lm, name)

    def __call__(self, prompts: Any, **kwargs: Any) -> Any:
        if self.min_interval and len(prompts) > 1:
            from lotus.types import LMOutput

            parts = [self([prompt], **kwargs) for prompt in prompts]
            combined: Any = LMOutput(
                outputs=[text for part in parts for text in part.outputs],
                logprobs=[probs for part in parts for probs in part.logprobs]
                if all(getattr(part, "logprobs", None) is not None for part in parts)
                else None,
            )
            metadata = [getattr(part, "response_metadata", None) for part in parts]
            combined.response_metadata = [item for part in metadata if part is not None for item in part] if all(
                part is not None for part in metadata
            ) else None
            return combined
        output = kwargs.get("max_tokens", self.output_limit)
        if type(output) is not int or not 0 < output <= self.output_limit:
            raise ValueError("Request output limit exceeds this experiment's limit")
        reservation = sum(self.lm.count_tokens(p) + output for p in prompts)
        if (
            self.requests + len(prompts) > self.request_limit
            or self.reserved_tokens + reservation > self.token_limit
            or (self.completion_limit is not None and self.completion_tokens >= self.completion_limit)
        ):
            raise BudgetExceeded(
                "Request/token reservation limit reached; no retry or fallback"
            )
        self.requests += len(prompts)
        self.reserved_tokens += reservation
        self.calls += 1
        self.unknown_usage_requests += len(prompts)
        self.persist()
        if self.progress is not None:
            self.progress.update(
                requests=self.requests,
                estimated_reserved_tokens=self.reserved_tokens,
                last_request_started_at=time.time(),
            )
        prefix = self.directory / f"call-{self.calls:05d}"
        write_json(
            prefix.with_suffix(".request.json"),
            {
                "prompts": prompts,
                "kwargs": kwargs,
                "reserved_tokens": reservation,
                "requests_total": self.requests,
                "reserved_tokens_total": self.reserved_tokens,
            },
        )
        # Artifact writes must not shorten the interval between LM calls.
        time.sleep(max(0, self.min_interval - (time.time() - self.last_started)))
        self.last_started = time.time()
        started = time.perf_counter()
        try:
            result = self.lm(prompts, **kwargs)
        except Exception as exc:
            write_json(
                prefix.with_suffix(".error.json"),
                {"type": type(exc).__name__, "error": str(exc)},
            )
            raise
        finally:
            usage = getattr(self.lm, "last_call_usage", None)
            if usage is not None:
                known_tokens = [value for value in usage.completion_tokens if value is not None]
                self.completion_tokens += sum(known_tokens)
                self.unknown_usage_requests -= len(known_tokens) + usage.cache_hits
            self.persist()
        write_json(
            prefix.with_suffix(".response.json"),
            {
                "outputs": result.outputs,
                "response_metadata": getattr(result, "response_metadata", None),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return result


def claude_type() -> type:
    """Load the existing experimental policy without rewriting its query."""
    import agent_memory as am

    repo = Path(am.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "poc_claude", repo / "examples/claude/composable_memory.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Existing Claude experimental policy is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComposableClaudeMemory


def build(case: dict, payload: dict, mode: str) -> tuple[Any, list[am.Relation], Any]:
    """Expose actual targets and their unchanged source query."""
    import agent_memory as am
    import pandas as pd
    from agent_memory.memories.claude import ClaudeMemory

    if case["kind"] == "policy":
        policy = ClaudeMemory if mode == "baseline" else claude_type()
        if mode == "baseline":
            return policy, [], None
        aggregate = policy.topics.expr.inputs[0]
        scope = aggregate.inputs[0]
        group = scope.inputs[0]
        if (
            aggregate.op != "sem_agg"
            or scope.op != "group_by"
            or group.op != "sem_groupby"
        ):
            raise ValueError(
                "Claude experimental query changed; inspect targets before running"
            )
        return policy, [am.Relation(group), am.Relation(aggregate)], None
    if mode == "baseline":
        raise ValueError("baseline is only the original Claude policy")
    # Serialize nested source values without dropping content. Original dates remain data.
    if any(
        set(record) & set(payload["context"] or {}) for record in payload["records"]
    ):
        raise ValueError("Dataset context would overwrite an original record field")
    records = [
        {**record, **(payload["context"] or {})} for record in payload["records"]
    ]
    if case.get("id", "").startswith("lme-"):
        def without_annotation(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: without_annotation(v) for k, v in value.items() if k != "has_answer"}
            if isinstance(value, list):
                return [without_annotation(v) for v in value]
            return value
        records = without_annotation(records)
    frame = pd.DataFrame(
        [
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in record.items()
            }
            for record in records
        ]
    )
    source = am.Source({key: "Original dataset field" for key in frame.columns})
    query = (
        source.sem_groupby(
            input_cols=["Description"], instruction=payload["instruction"]
        )
        if case["kind"] == "groupby"
        else source.sem_agg(output_cols=["answer"], instruction=payload["instruction"])
    )
    return source, [query], frame


def planning_targets(
    targets: list, *, plan_only: bool, target_index: int | None
) -> list:
    """Select a planning target without changing the query compiled afterward."""
    if target_index is None:
        return targets
    if not plan_only or not 0 <= target_index < len(targets):
        raise ValueError("--plan-target requires --plan-only and a valid target index")
    return [targets[target_index]]


def rules_for(
    mode: str,
    targets: list,
    adapter: Any,
    directory: Path | None,
    *,
    saved_plans: list[dict] | None = None,
) -> DifferentialRules:
    """Plan exactly once per target, or select an explicitly named control."""
    import agent_memory as am
    from agent_memory.planner.rules import DifferentialRules

    if saved_plans is not None:
        if mode != "agent" or not targets or len(saved_plans) != len(targets):
            raise ValueError(
                "--saved-plan requires --mode agent and one plan per target, in target order"
            )
        plans = tuple(
            am.MaintenancePlan.bind(target, config)
            for target, config in zip(targets, saved_plans, strict=True)
        )
        if directory is not None:
            for index, plan in enumerate(plans):
                write_json(directory / f"plan-{index}.json", plan.to_dict())
        return DifferentialRules(maintenance_plans=plans)
    if mode == "agent":
        if directory is None:
            raise ValueError("Planning needs a new run directory")
        plans = []
        for index, target in enumerate(targets):
            plan = am.plan_maintenance(target, adapter=adapter, examples=[])
            write_json(directory / f"plan-{index}.json", plan.to_dict())
            plans.append(plan)
        return DifferentialRules(maintenance_plans=tuple(plans))
    if mode == "baseline":
        return DifferentialRules()
    return DifferentialRules(
        sem_agg_rule=f"agg-{mode}", sem_groupby_rule=f"group-{mode}"
    )


def make_engine(
    case: dict, owner: Any, targets: list, adapter: Any, rules: DifferentialRules
) -> Any:
    """Compile with explicit plans; compilation cannot invoke the planning Agent."""
    import agent_memory as am

    if case["kind"] == "policy":
        return owner(adapter=adapter, rules=rules)
    return am.SemanticDataflow(
        source=owner, views={"answer": targets[0]}, adapter=adapter, rules=rules
    )


def executor_of(engine: Any) -> Any:
    """Inspect the existing executor for diagnostics, not an additional state store."""
    import agent_memory as am

    return (
        engine._executor
        if isinstance(engine, am.SemanticDataflow)
        else engine._runtime._engine
    )


def record_step(
    engine: Any, directory: Path, index: int, previous: dict, elapsed: float
) -> dict:
    """Record public output differences and S, without duplicating all internal tables."""
    import agent_memory as am
    from agent_memory.runtime.executor import NodeOutputUpdate

    executor = executor_of(engine)
    snapshot = executor.snapshot_state()
    names = (
        ["answer"] if isinstance(engine, am.SemanticDataflow) else ["topics", "catalog"]
    )
    outputs = {name: snapshot["state"][name] for name in names}
    differences = {}
    for name, new in outputs.items():
        old = previous.get(name, new.iloc[:0])
        change = NodeOutputUpdate.between(old, new)
        differences[name] = {
            "inserted": rows(change.inserted_rows),
            "retracted": rows(change.retracted_rows),
        }
    states = {
        key: rows(snapshot["node_state"][key])
        for key, node in executor.policy.nodes.items()
        if node.execution_kind == "semantic_stateful" and key in snapshot["node_state"]
    }
    write_json(
        directory / f"step-{index:04d}.json",
        {
            "step": index,
            "elapsed_seconds": elapsed,
            "outputs": {k: rows(v) for k, v in outputs.items()},
            "changes": differences,
            "state_nodes": states,
        },
    )
    return outputs


def render_report(directory: Path, summary: dict) -> None:
    """Show runs as completed, failed or budget-stopped, never as missing successes."""
    esc = html.escape
    sections = []
    table_rows = []
    contract = directory / "run-contract.json"
    if contract.exists():
        sections.append(
            f"<details><summary>Run settings: requested parameters and known omissions</summary><pre>{esc(contract.read_text())}</pre></details>"
        )
    for path in sorted(directory.glob("plan-*.json")):
        plan = json.loads(path.read_text())
        sections.append(
            f"<h3>{esc(path.stem)} · {esc(plan['rule'])}</h3><p>{esc(plan['rationale'])}</p>"
        )
        sections.append(
            f"<details><summary>完整方案：S schema 与维护说明</summary><pre>{esc(path.read_text())}</pre></details>"
        )
    for path in sorted(directory.glob("step-*.json")):
        step = json.loads(path.read_text())
        inserted = sum(len(value["inserted"]) for value in step["changes"].values())
        retracted = sum(len(value["retracted"]) for value in step["changes"].values())
        count = sum(len(value) for value in step["outputs"].values())
        state_bytes = len(json.dumps(step["state_nodes"], ensure_ascii=False).encode())
        table_rows.append(
            f'<tr><td><a href="#{path.stem}">{step["step"]}</a></td><td>{count}</td><td>+{inserted} / -{retracted}</td><td>{state_bytes if step["state_nodes"] else "无 S"}</td><td>{step["elapsed_seconds"]:.3f}</td></tr>'
        )
        sections.append(
            f'<details id="{path.stem}"><summary>第 {step["step"]} 批：结果、S 与增删明细</summary><pre>{esc(path.read_text())}</pre></details>'
        )
    document = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Agent State PoC / 单次记录</title><style>body{max-width:1050px;margin:32px auto;padding:0 20px;background:#f5f1e8;color:#203443;font:16px/1.6 Georgia,"Songti SC",serif}'
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:20px}details{border-top:1px solid #bbb;padding:12px}h1{border-top:5px solid #a94725;padding-top:16px}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}a{color:#a94725}</style>"
        "<h1>规则选择 → 每批结果与 S</h1><p>工程完成不等于答案正确；token 预留量不是实际账单。原始调用见 calls/，provider 用量见 trace/。</p>"
        '<p><a href="budget.json">请求计数与预算记录</a> · <a href="summary.json">机器可读总览</a></p>'
        f"<pre>{esc(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>"
        + "<h2>逐批变化</h2><table><tr><th>批次</th><th>当前结果行数</th><th>插入 / 撤回</th><th>S 节点 JSON bytes</th><th>执行秒数</th></tr>"
        + "".join(table_rows)
        + "</table><h2>方案与明细</h2>"
        + "".join(sections)
        + "</html>"
    )
    (directory / "index.html").write_text(document)


def evaluate_grouping(output: pd.DataFrame, gold_path: Path) -> dict:
    """Compare a final single-membership partition, independent of group names."""
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from agent_memory.policy.grouping import GROUP_ID

    gold = pd.DataFrame(json.loads(gold_path.read_text()))
    if output["id"].duplicated().any() or gold["id"].duplicated().any():
        raise ValueError("This evaluator requires one membership per original record")
    paired = gold.merge(
        output[["id", GROUP_ID]],
        on="id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not bool(paired["_merge"].eq("both").all()):
        raise ValueError("Final prediction and gold cover different records")
    return {
        "adjusted_rand_index": adjusted_rand_score(paired["Class"], paired[GROUP_ID]),
        "normalized_mutual_information": normalized_mutual_info_score(
            paired["Class"], paired[GROUP_ID]
        ),
        "records": len(paired),
        "predicted_groups": paired[GROUP_ID].nunique(),
        "scope": "Original static labels, selected records only; not intermediate reassignment truth",
    }


def identity_hash(value: dict) -> str:
    """Fingerprint execution inputs without changing their original representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stable_identity(value: dict) -> dict:
    """Retries affect request attempts, not the state being restored."""
    result = json.loads(json.dumps(value))
    if result.get("execution") is not None:
        result["execution"].pop("validation_retries", None)
    return result


def validate_resume(
    checkpoint: dict, current: dict, directory: Path, source_changes: dict | None,
) -> dict:
    """Verify a saved identity before allowing explicitly reviewed runner changes."""
    previous = checkpoint.get("identity_inputs")
    if previous is None:
        previous = json.loads(json.dumps(current))
        if current.get("execution") is not None:
            contract = json.loads((directory / "run-contract.json").read_text())
            previous["execution"] = {
                "source": contract["source"]["files"],
                "model": {key: contract[key] for key in current["execution"]["model"]},
                "validation_retries": contract["retries"],
            }
        if identity_hash(previous) != checkpoint["identity"]:
            raise ValueError("Legacy checkpoint identity mismatch")
    elif identity_hash(stable_identity(previous)) != checkpoint["identity"]:
        raise ValueError("Checkpoint identity metadata mismatch")
    before, after = stable_identity(previous), stable_identity(current)
    changes = source_changes or {}
    old_source = (before.get("execution") or {}).get("source", {})
    new_source = (after.get("execution") or {}).get("source", {})
    actual = {name: {"before": old_source.get(name), "after": new_source.get(name)}
              for name in old_source.keys() | new_source.keys()
              if old_source.get(name) != new_source.get(name)}
    permitted = {"tools/evaluation/maintenance_poc.py", "tests/test_maintenance_poc.py"}
    if changes != actual or not changes.keys() <= permitted:
        raise ValueError("Resume requires exact reviewed runner/test source changes")
    for name in changes:
        old_source[name] = new_source[name]
    if before != after:
        raise ValueError("Checkpoint data, plan, model or execution identity mismatch")
    return {"source_changes": changes,
            "previous_retries": (previous.get("execution") or {}).get("validation_retries"),
            "current_retries": (current.get("execution") or {}).get("validation_retries")}


def execute_case(
    case: dict,
    payload: dict,
    mode: str,
    adapter: Any,
    directory: Path,
    *,
    data_dir: Path,
    progress: Progress | None = None,
    saved_plans: list[dict] | None = None,
    plan_only: bool = False,
    target_index: int | None = None,
    model: str = MODEL,
    stop_after: int | None = None,
    resume_from: Path | None = None,
    execution_identity: dict | None = None,
    resume_source_changes: dict | None = None,
    maintenance_only: bool = False,
) -> dict:
    """Run one condition. Exceptions propagate after retaining committed steps."""
    if plan_only and (mode != "agent" or saved_plans is not None):
        raise ValueError("--plan-only requires --mode agent without --saved-plan")
    if (resume_from is not None or stop_after is not None) and plan_only:
        raise ValueError("Checkpoints apply to execution, not planning")
    if resume_from is not None and mode == "agent" and saved_plans is None:
        raise ValueError("Resume requires the saved plans; never replan during restore")
    total = len(payload["records"]) if case["kind"] == "policy" else len(case["batch_ranges"])
    limit = total if stop_after is None else stop_after
    if not 0 < limit <= total:
        raise ValueError("--stop-after must be within the original input steps")
    identity_inputs = {"case": case, "payload": payload, "mode": mode, "model": model,
                       "plans": saved_plans, "execution": execution_identity}
    identity = identity_hash(stable_identity(identity_inputs))
    owner, targets, frame = build(case, payload, mode)
    selected = planning_targets(targets, plan_only=plan_only, target_index=target_index)
    summary = {
        "case": case["id"],
        "condition": mode,
        "model": model,
        "planning": "saved-plan" if saved_plans is not None else mode,
        "plan_only": plan_only,
        "plan_target": target_index,
        "status": "running",
        "stage": "planning",
        "completed_steps": 0,
    }
    started = time.perf_counter()
    previous: dict = {}
    metrics: list = []

    def publish() -> None:
        if progress is not None:
            progress.update(**summary)

    try:
        publish()
        rules = rules_for(mode, selected, adapter, directory, saved_plans=saved_plans)
        summary["stage"] = "compile"
        publish()
        engine = make_engine(case, owner, targets, adapter, rules)
        executor = executor_of(engine)
        summary["plan_fingerprint"] = executor.policy.fingerprint
        summary["execution_kinds"] = [
            n.execution_kind for n in executor.policy.nodes.values()
        ]
        if plan_only:
            summary.update(status="completed", stage="planned")
            return summary
        state_owner = engine._runtime if case["kind"] == "policy" else engine
        if resume_from is not None:
            checkpoint = json.loads(resume_from.read_text())
            state_path = resume_from.parent / checkpoint["state_file"]
            if digest(state_path) != checkpoint["sha256"]:
                raise ValueError("Checkpoint file checksum mismatch")
            transition = validate_resume(checkpoint, identity_inputs, resume_from.parent,
                                         resume_source_changes)
            write_json(directory / "resume-validation.json", transition)
            # Only locally produced experiment snapshots are supported, like LOCOMO's state.pkl.
            with state_path.open("rb") as handle:
                restored = pickle.load(handle)
            state_owner.restore_state(restored["runtime"])
            previous, metrics = restored["previous"], restored["metrics"]
            summary["completed_steps"] = checkpoint["completed_steps"]
            if summary["completed_steps"] > limit:
                raise ValueError("Stop boundary precedes the saved checkpoint")

        def checkpoint_committed() -> None:
            index = summary["completed_steps"]
            path = directory / f"checkpoint-{index:04d}.pkl"
            temporary = path.with_suffix(".pending")
            with temporary.open("wb") as handle:
                pickle.dump({"runtime": state_owner.snapshot_state(),
                             "previous": previous, "metrics": metrics}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            write_json(directory / "checkpoint.json", {
                "identity": identity, "completed_steps": index,
                "identity_inputs": identity_inputs,
                "state_file": path.name, "sha256": digest(path),
            })
        summary["stage"] = "maintenance"
        publish()
        if case["kind"] == "policy":
            from agent_memory.evaluation.claude_memory.locomo import (
                run_memory_ingest,
                run_questions,
            )
            from agent_memory.evaluation.types import BenchmarkEvent, BenchmarkQuestion

            events = [BenchmarkEvent(**record) for record in payload["records"]]

            def committed() -> None:
                nonlocal previous
                summary["completed_steps"] += 1
                previous = record_step(
                    engine,
                    directory,
                    summary["completed_steps"],
                    previous,
                    metrics[-1]["latency_sec"],
                )
                checkpoint_committed()
                write_json(directory / "ingest-metrics.json", metrics)
                publish()

            run_memory_ingest(
                engine, events[:limit], start_index=summary["completed_steps"],
                step_metrics=metrics, checkpoint_callback=committed
            )
            write_json(directory / "ingest-metrics.json", metrics)
            if limit < total:
                summary.update(status="paused", stage="maintenance-checkpoint",
                               evaluation="Not evaluated: only a prefix of the conversation was ingested")
                return summary
            if maintenance_only:
                summary.update(status="completed", stage="maintenance-completed",
                               evaluation="Not requested: maintenance only; no QA or scoring")
                return summary
            # Gold is only opened after all maintenance and planning have finished.
            summary["stage"] = "retrieval-and-answer"
            publish()
            questions = [
                BenchmarkQuestion(**r)
                for r in json.loads((data_dir / case["gold"]).read_text())
            ]
            results: list = []

            def answered() -> None:
                write_json(directory / "qa-results.json", results)
                summary["completed_questions"] = len(results)
                publish()

            run_questions(
                engine,
                questions,
                answer=True,
                result_rows=results,
                checkpoint_callback=answered,
            )
            summary["evaluation"] = (
                "Existing LOCOMO scorer; QA is not a gold label for S"
            )
        else:
            for index, (start, end) in enumerate(case["batch_ranges"], 1):
                if index <= summary["completed_steps"] or index > limit:
                    continue
                step_start = time.perf_counter()
                engine.apply(frame.iloc[start:end])
                previous = record_step(
                    engine, directory, index, previous, time.perf_counter() - step_start
                )
                summary["completed_steps"] = index
                checkpoint_committed()
                publish()
            if limit < total:
                summary.update(status="paused", stage="maintenance-checkpoint")
                return summary
            summary["evaluation"] = (
                "Original gold retained separately; semantic correctness not yet judged"
            )
            if case["kind"] == "groupby" and "gold" in case:
                summary["stage"] = "evaluation"
                summary["evaluation"] = evaluate_grouping(
                    previous["answer"], data_dir / case["gold"]
                )
        summary["status"] = "completed"
        return summary
    except Exception as exc:
        summary.update(
            status="budget-stopped" if isinstance(exc, BudgetExceeded) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        publish()
        write_json(directory / "summary.json", summary)
        render_report(directory, summary)


def verify_source(receipt: Path) -> dict:
    """Reject a changed source tree rather than mislabel the executed version."""
    import agent_memory as am

    manifest = json.loads(receipt.read_text())
    repo = Path(am.__file__).resolve().parents[2]
    for name, expected in manifest["files"].items():
        if digest(repo / name) != expected:
            raise ValueError(f"Source receipt is stale: {name}")
    return manifest


def main() -> None:
    """Require explicit execution and limits; normal invocation is model-free."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--backend", choices=["deepseek", "chatgpt-plan"], default="deepseek"
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--saved-plan", type=Path, action="append")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-target", type=int)
    parser.add_argument("--thinking", choices=["enabled", "disabled"])
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-from", type=Path, help="Trusted local checkpoint.json")
    parser.add_argument("--resume-source-changes", type=Path,
                        help="Reviewed runner/test before/after hashes; no operator changes")
    parser.add_argument("--maintenance-only", action="store_true")
    parser.add_argument("--budget-ledger", type=Path,
                        help="Shared cumulative budget for sequential conditions")
    parser.add_argument("--validation-retries", type=int, default=0)
    parser.add_argument("--min-interval", type=float, default=0)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument(
        "--mode",
        choices=["agent", "fallback", "composable", "baseline"],
        default="agent",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-reserved-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args()
    if not 0 <= args.validation_retries <= 3 or args.min_interval < 0:
        parser.error("Validation retries must be 0..3; interval must be nonnegative")
    if args.plan_only and args.validation_retries:
        parser.error("Planner retries are not part of the operator experiment")
    if args.resume_from is not None and (
        args.plan_only or (args.mode == "agent" and not args.saved_plan)
    ):
        parser.error("Resume requires execution mode and, for agent mode, saved plans")
    if args.resume_source_changes is not None and args.resume_from is None:
        parser.error("Source transition requires --resume-from")
    if args.case == "status":
        if args.runs_dir is None:
            parser.error("status requires --runs-dir")
        show_status(args.runs_dir, watch=args.watch)
        return
    if args.data_dir is None:
        parser.error("Case preparation directory is required: --data-dir")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    try:
        model, model_options, model_contract = model_settings(
            args.backend,
            args.model,
            args.base_url,
            args.max_output_tokens,
            thinking=args.thinking,
        )
    except ValueError as exc:
        parser.error(str(exc))
    data_dir = args.data_dir
    manifest = json.loads((data_dir / "manifest.json").read_text())
    case = next((c for c in manifest["cases"] if c["id"] == args.case), None)
    if case is None or case["status"] != "READY_DATA_ONLY":
        parser.error("Unknown case or unresolved original task mapping")
    payload = json.loads((data_dir / case["inputs"]).read_text())
    owner, targets, _ = build(case, payload, args.mode)
    if args.plan_only and (args.mode != "agent" or args.saved_plan is not None):
        parser.error("--plan-only requires --mode agent without --saved-plan")
    try:
        selected = planning_targets(
            targets, plan_only=args.plan_only, target_index=args.plan_target
        )
    except ValueError as exc:
        parser.error(str(exc))
    saved_plans = None
    saved_rules = None
    if args.saved_plan is not None:
        saved_plans = [json.loads(path.read_text()) for path in args.saved_plan]
        if not all(isinstance(plan, dict) for plan in saved_plans):
            parser.error("Each saved plan must be a JSON object")
        saved_rules = rules_for(
            args.mode, targets, NoExecution(), None, saved_plans=saved_plans
        )
    if not args.run:
        checks = []
        for control in (
            ["saved-plan"]
            if saved_rules is not None
            else ["baseline"]
            if args.mode == "baseline"
            else ["fallback", "composable"]
        ):
            engine = make_engine(
                case,
                owner,
                targets,
                NoExecution(),
                saved_rules
                if saved_rules is not None
                else rules_for(control, targets, None, None),
            )
            checks.append(
                {
                    "control": control,
                    "fingerprint": executor_of(engine).policy.fingerprint,
                }
            )
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "model_settings": model_contract,
                    "model_calls": 0,
                    "agent_targets": len(selected),
                    "plan_only": args.plan_only,
                    "plan_target": args.plan_target,
                    "maintenance_steps": len(case["batch_ranges"]),
                    "grouping_binding": {
                        "C": None,
                        "membership": "single",
                        "category_catalog": case.get("task_binding", {}).get(
                            "categories"
                        ),
                    }
                    if case["kind"] == "groupby"
                    else None,
                    "compile_checks": checks,
                    "agent_execution": "skipped; saved plan bound and compiled"
                    if saved_rules is not None
                    else "not attempted; Agent's selected plan must compile after planning",
                    "original_instructions": [
                        t.expr.params["instruction"] for t in selected
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if (
        args.output is None
        or args.source_receipt is None
        or args.max_requests is None
        or args.max_reserved_tokens is None
    ):
        parser.error(
            "--run requires new --output, --source-receipt and both budget limits"
        )
    if min(args.max_requests, args.max_reserved_tokens, args.max_output_tokens) <= 0:
        parser.error("Budget limits must be positive")
    source = verify_source(args.source_receipt)
    if (
        digest(data_dir / "data" / case["source"])
        != manifest["sources"][case["source"]]["sha256"]
    ):
        raise ValueError("Raw source bytes changed")
    args.output.mkdir(parents=True, exist_ok=False)
    calls = args.output / "calls"
    calls.mkdir()
    write_json(
        args.output / "run-contract.json",
        {
            "case": case,
            "mode": args.mode,
            "source": source,
            "input_sha256": digest(data_dir / case["inputs"]),
            "gold_sha256": digest(data_dir / case["gold"]),
            "runner_sha256": digest(Path(__file__)),
            "manifest_sha256": digest(data_dir / "manifest.json"),
            **model_contract,
            "plan_only": args.plan_only,
            "plan_target": args.plan_target,
            "saved_plan_sha256": [digest(path) for path in args.saved_plan]
            if args.saved_plan is not None
            else None,
            "retries": args.validation_retries,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "stop_after": args.stop_after,
            "maintenance_only": args.maintenance_only,
            "budget_ledger": str(args.budget_ledger) if args.budget_ledger else None,
            "resume_source_changes": json.loads(args.resume_source_changes.read_text())
            if args.resume_source_changes else None,
            "limits": {
                "requests": args.max_requests,
                "completion_tokens_stop_threshold": args.max_completion_tokens,
                "estimated_reserved_tokens": args.max_reserved_tokens,
                "output_tokens_per_request": model_contract[
                    "output_tokens_per_request"
                ],
                "output_tokens_for_local_estimation": args.max_output_tokens,
            },
        },
    )
    from agent_memory.adapters.lotus import LotusAdapter
    from agent_memory.adapters.lotus.context import LotusExecutionConfig

    progress = Progress(
        args.output, case["id"], args.mode, len(case["batch_ranges"]), model=model
    )
    config = LotusExecutionConfig(
        lm_num_retries=0,
        lm_timeout=240 if args.backend == "chatgpt-plan" else 120,
        lm_max_batch_size=1,
        lm_enable_cache=False,
        lm_model_kwargs=model_options,
        structured_max_tokens=args.max_output_tokens,
        structured_parse_retries=args.validation_retries,
        maintenance_parse_retries=args.validation_retries,
        semantic_trace_dir=args.output / "trace",
        semantic_trace_snapshot_mode="full",
    )
    budget = None
    clients = ExitStack()
    try:
        if args.budget_ledger is not None:
            lock = clients.enter_context(args.budget_ledger.with_suffix(".lock").open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.backend == "chatgpt-plan":
            key = os.environ.get("OPENAI_API_KEY")
            if not key or not key.strip():
                raise ValueError(
                    "Set OPENAI_API_KEY to the local gateway key for this process"
                )
            import httpx
            from openai import OpenAI

            # Reuse the gateway's documented client setup: no proxy or redirects.
            client = clients.enter_context(
                OpenAI(
                    api_key=key,
                    base_url=model_options["api_base"],
                    max_retries=0,
                    timeout=240,
                    http_client=httpx.Client(trust_env=False, follow_redirects=False),
                )
            )
            model_options["client"] = client
        adapter = LotusAdapter(model=model, config=config)
        adapter._context.configure()
        import lotus

        budget = BudgetLM(
            lotus.settings.lm,
            calls,
            args.max_requests,
            args.max_reserved_tokens,
            args.max_output_tokens,
            progress,
            initial=json.loads((args.resume_from.parent / "budget-live.json").read_text())
            if args.resume_from is not None and args.budget_ledger is None else None,
            min_interval=args.min_interval,
            completion_limit=args.max_completion_tokens,
            ledger=args.budget_ledger,
        )
        # One adapter per process; the wrapper only accounts for this run's requests.
        adapter._context._lm = budget
        lotus.settings.configure(lm=budget)
        execute_case(
            case,
            payload,
            args.mode,
            adapter,
            args.output,
            data_dir=data_dir,
            progress=progress,
            saved_plans=saved_plans,
            plan_only=args.plan_only,
            target_index=args.plan_target,
            model=model,
            stop_after=args.stop_after,
            resume_from=args.resume_from,
            execution_identity={"source": source["files"], "model": model_contract,
                                "validation_retries": args.validation_retries},
            resume_source_changes=json.loads(args.resume_source_changes.read_text())
            if args.resume_source_changes else None,
            maintenance_only=args.maintenance_only,
        )
    except BaseException as exc:
        progress.update(
            status="budget-stopped" if isinstance(exc, BudgetExceeded) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        clients.close()
        write_json(
            args.output / "budget.json",
            {
                "requests": budget.requests if budget is not None else 0,
                "estimated_reserved_tokens": budget.reserved_tokens
                if budget is not None
                else 0,
                "invoice_cost_usd": None,
                "note": "Not billable token totals; actual provider usage is in trace/",
            },
        )


if __name__ == "__main__":
    main()
