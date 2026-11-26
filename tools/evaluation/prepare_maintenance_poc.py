"""Inventory original PoC samples and render a static, model-free preflight."""

from __future__ import annotations

import ast
import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import importlib.util
import json
from pathlib import Path
from typing import Any

import agent_memory as am
from agent_memory.memories.claude import ClaudeMemory
from agent_memory.planner.rules import DifferentialRules
from agent_memory.evaluation.locomo import (
    LOCOMO_SHA256,
    eligible_questions,
    load_locomo_sample,
)


ROOT: Path
DATA: Path
SOURCES = {
    "qmsum-test.jsonl": "https://raw.githubusercontent.com/Yale-LILY/QMSum/83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e/data/ALL/jsonl/test.jsonl",
    "longmemeval_oracle.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "ag-news.csv": "https://raw.githubusercontent.com/YouranSun/HOLDUP/a400a6c19072725bfd2a00ff9baf1d5f0c45c2e8/examples/ag-news.csv",
    "holdup-original-config.txt": "https://raw.githubusercontent.com/YouranSun/HOLDUP/a400a6c19072725bfd2a00ff9baf1d5f0c45c2e8/configs/example_datasets.py",
    "banking77-test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/test.csv",
    "nlupp-fold0.json": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/nlupp/data/banking/fold0.json",
    "locomo10.json": "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json",
}


def dump(path: Path, value: Any) -> None:
    """Write a generated artifact, never a raw source file."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def batches(count: int, size: int) -> list[list[int]]:
    """Return half-open index ranges in original selected-record order."""
    return [[start, min(start + size, count)] for start in range(0, count, size)]


def make_case(
    case_id: str,
    kind: str,
    source: str,
    locator: Any,
    instruction: str | None,
    records: list[Any],
    gold: Any,
    size: int,
    note: str,
    *,
    context: Any = None,
    ready: bool = True,
) -> dict[str, Any]:
    """Separate model-visible inputs from evaluator-only answers."""
    inputs = {"instruction": instruction, "records": records, "context": context}
    dump(ROOT / "cases" / f"{case_id}.inputs.json", inputs)
    dump(ROOT / "cases" / f"{case_id}.gold.json", gold)
    serialized = json.dumps(inputs, ensure_ascii=False)
    return {
        "id": case_id,
        "kind": kind,
        "source": source,
        "locator": locator,
        "instruction": instruction,
        "records": len(records),
        "batch_ranges": batches(len(records), size),
        "input_utf8_bytes": len(serialized.encode()),
        "input_characters": len(serialized),
        "inputs": f"cases/{case_id}.inputs.json",
        "gold": f"cases/{case_id}.gold.json",
        "note": note,
        "status": "READY_DATA_ONLY" if ready else "TASK_MAPPING_PENDING",
        "agent_rule": None,
        "model_result": None,
        "cost_usd": None,
    }


def render(cases: list[dict[str, Any]], files: dict[str, Any]) -> str:
    """Build a self-contained report; missing results stay visibly missing."""
    esc = html.escape
    rows = []
    details = []
    for case in cases:
        key = case["id"]
        instruction = case["instruction"] or (
            "维护指令从原 Claude policy 读取；不以检索问题替代。"
            if case["kind"] == "policy"
            else "没有逐字原 query；不自动编写"
        )
        ranges = case["batch_ranges"]
        range_preview = (
            str(ranges) if len(ranges) <= 5 else f"{ranges[:3]} ... {ranges[-1]}"
        )
        rows.append(
            f'<tr><td><a href="#{key}">{esc(key)}</a></td><td>{case["kind"]}</td>'
            f"<td>{case['records']}</td><td>{len(case['batch_ranges'])}</td>"
            f"<td>{case['input_utf8_bytes']:,}</td><td>{case['status']}</td><td>未运行</td></tr>"
        )
        payload = json.loads((ROOT / case["inputs"]).read_text())
        gold = json.loads((ROOT / case["gold"]).read_text())
        details.append(
            f'<article id="{key}"><h3>{esc(key)}</h3><p class="instruction">{esc(instruction)}</p>'
            f"<p>{esc(case['note'])}</p><p>原始定位：<code>{esc(json.dumps(case['locator'], ensure_ascii=False))}</code></p>"
            f"<p>批次 [start, end)：<code>{range_preview}</code>；顺序不改。</p>"
            f"<p>无模型编译：{esc(json.dumps(case.get('compile_checks', [])))}</p>"
            f'<p><a href="{case["inputs"]}">完整执行输入</a> · <a href="{case["gold"]}">评价标注（不发给 Agent）</a> · <a href="data/{case["source"]}">完整源文件</a></p>'
            f"<details><summary>第一条原记录预览</summary><pre>{esc(json.dumps(payload['records'][0], ensure_ascii=False, indent=2))}</pre></details>"
            f"<details><summary>原答案／标签预览，仅用于人工检查</summary><pre>{esc(json.dumps(gold, ensure_ascii=False, indent=2))}</pre></details>"
            '<p class="pending">Agent 方案 → 每批结果与 S → 成员变化 → 评分／成本：均未运行，不填假数据。</p></article>'
        )
    source_rows = "".join(
        f'<tr><td><a href="{esc(info["url"], quote=True)}">{esc(name)}</a></td><td>{info["bytes"]:,}</td><td><code>{info["sha256"]}</code></td></tr>'
        for name, info in files.items()
    )
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent State PoC · 数据预检</title><style>
:root{--paper:#f5f1e8;--ink:#203443;--muted:#53626a;--accent:#a94725;--line:#d5d0c5}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(at top right,#e4ebe5,transparent 55%),var(--paper);color:var(--ink);font:16px/1.65 Georgia,'Songti SC',serif}
main{max-width:1200px;margin:auto;padding:40px 24px}header{border-top:5px solid var(--accent);padding-top:20px}h1{font-size:clamp(28px,5vw,48px);margin:8px 0}
h2{margin-top:40px}a{color:var(--accent)}.muted{color:var(--muted)}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{border:1px solid var(--line);padding:16px 24px;background:#fff9;flex:1;min-width:160px}
.card strong{display:block;font-size:28px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{white-space:nowrap;background:#e7e8df}
article{margin-top:24px;padding:24px;border:1px solid var(--line);background:#fffc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf0e9;padding:16px;max-height:400px;overflow:auto}
code{overflow-wrap:anywhere;font-size:12px}.instruction{font-size:20px}.pending{border-left:3px solid var(--accent);padding-left:12px;color:var(--muted)}summary{cursor:pointer} @media(max-width:650px){main{padding:20px 12px}article{padding:16px}}
</style><main><header><span class="muted">AGENT STATE / PROOF OF CONCEPT</span><h1>先看原题，再看 Agent 留下什么</h1>
<p>本页是无模型预检，不是实验结果。原数据、原问题、原答案不改；分批与子集选择单独记录。</p></header>
<div class="cards"><div class="card"><strong>7</strong>候选执行案例（含 Claude policy）</div><div class="card"><strong>2</strong>待确认任务映射</div><div class="card"><strong>0</strong>模型调用</div><div class="card"><strong>未批准</strong>付费调用预算</div></div>
<h2>01 / 样本与执行范围</h2><p>输入字节数不是 provider token 数，更不是费用预算。Agent、operator、批次重读和生成的 S 另有开销。</p>
<div class="scroll"><table><tr><th>案例</th><th>类型</th><th>记录</th><th>批次</th><th>输入 UTF-8 bytes</th><th>数据状态</th><th>模型状态</th></tr>"""
        + "".join(rows)
        + """</table></div>
<h2>02 / 如何读后续结果</h2><p>分别记录：① Agent 选的规则及理由；② 完整 S schema 与维护说明；③ 每批用户结果／S／归属增删；④ 与原标注的比较；⑤ 规划与执行各自的调用量、token、费用、耗时。失败显示发生层次及原始响应，不自动补救。</p>
<p>六条规则当前均为“未运行”。Agent 自由选择；没选到的规则记为未覆盖。fallback/composable 对照单独标明，不能冒充 Agent 决策。</p>
<p>不强迫历史 reassignment 发生；静态类别标签不等于每阶段改组金标。Claude 的最终 QA 不是 S 的金标。</p>
<p>执行入口已接通并通过离线检查，仍未调用真实模型。<a href="README.md">运行方式与限制</a> · <a href="BUDGET.md">待批准的范围和预算</a>。实际运行时，每次独立生成可读页面，按“方案 → 每批结果与 S → 增删变化 → 失败／评分”展示。</p>
<h2>03 / 原题与证据</h2>"""
        + "".join(details)
        + """<h2>04 / 原文件与版本</h2><p>链接来源可能变化，以本地原文件 SHA-256 固定此次内容；后续同步不重新下载或改写原文件。</p><div class="scroll"><table><tr><th>来源</th><th>bytes</th><th>SHA-256</th></tr>"""
        + source_rows
        + """</table></div></main></html>"""
    )


def compile_cases(cases: list[dict[str, Any]]) -> None:
    """Compile no-state controls without constructing or calling a model."""

    class NoExecution:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"Preflight attempted operator execution: {name}")

    for case in cases:
        case["compile_checks"] = []
        if case["status"] != "READY_DATA_ONLY" or case["kind"] == "policy":
            continue
        payload = json.loads((ROOT / case["inputs"]).read_text())
        source = am.Source(
            {key: "Original source field" for key in payload["records"][0]}
        )
        query = (
            source.sem_groupby(
                input_cols=["Description"], instruction=case["instruction"]
            )
            if case["kind"] == "groupby"
            else source.sem_agg(output_cols=["answer"], instruction=case["instruction"])
        )
        for suffix in ["fallback", "composable"]:
            am.SemanticDataflow(
                source=source,
                views={"answer": query},
                adapter=NoExecution(),
                rules=DifferentialRules(
                    sem_agg_rule="agg-" + suffix, sem_groupby_rule="group-" + suffix
                ),
            )
            case["compile_checks"].append(suffix + ": PASS (compile only)")
    repo = Path(am.__file__).resolve().parents[2]
    module_spec = importlib.util.spec_from_file_location(
        "poc_claude", repo / "examples/claude/composable_memory.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("Missing existing Claude experiment policy")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    policy_case = next(case for case in cases if case["kind"] == "policy")
    ClaudeMemory(adapter=NoExecution())
    policy_case["compile_checks"].append("legacy baseline: PASS (compile only)")
    for suffix in ["fallback", "composable"]:
        module.ComposableClaudeMemory(
            adapter=NoExecution(),
            rules=DifferentialRules(
                sem_agg_rule="agg-" + suffix, sem_groupby_rule="group-" + suffix
            ),
        )
        policy_case["compile_checks"].append(suffix + ": PASS (compile only)")


def main() -> None:
    """Prepare original cases without loading a model or provider credentials."""
    global ROOT, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing immutable data/; derived cases are written here",
    )
    args = parser.parse_args()
    ROOT = args.data_dir
    DATA = ROOT / "data"
    (ROOT / "cases").mkdir(exist_ok=True)
    files = {
        name: {
            "url": url,
            "bytes": (DATA / name).stat().st_size,
            "sha256": hashlib.sha256((DATA / name).read_bytes()).hexdigest(),
        }
        for name, url in SOURCES.items()
    }
    if files["locomo10.json"]["sha256"] != LOCOMO_SHA256:
        raise ValueError("LOCOMO does not match the repository's pinned dataset")
    cases = []
    meetings = [
        json.loads(line)
        for line in (DATA / "qmsum-test.jsonl").read_text().splitlines()
    ]
    meeting = meetings[0]
    for field in ["general_query_list", "specific_query_list"]:
        question = meeting[field][0]
        cases.append(
            make_case(
                "qmsum-" + field.split("_")[0],
                "agg",
                "qmsum-test.jsonl",
                {"jsonl_line": 1, "query_field": field, "query_index": 0},
                question["query"],
                meeting["meeting_transcripts"],
                question,
                45,
                "同一场完整会议，两种原任务；保留全部发言，不用参考答案中的相关区间筛输入。人工摘要不是唯一正确答案。",
            )
        )
    questions = json.loads((DATA / "longmemeval_oracle.json").read_text())
    for question_id in ["gpt4_2655b836", "6a1eabeb"]:
        question = next(q for q in questions if q["question_id"] == question_id)
        records = [
            {"session_id": sid, "date": date, "messages": messages}
            for sid, date, messages in zip(
                question["haystack_session_ids"],
                question["haystack_dates"],
                question["haystack_sessions"],
                strict=True,
            )
        ]
        dates = question["haystack_dates"]
        note = "官方 oracle 版本，已预选证据会话，不代表全历史检索难度；每批一会话，记录内部消息完整保留。"
        if dates != sorted(dates):
            note += " 原文件会话非时间顺序，本轮不排序；模型可读取原时间字段。"
        cases.append(
            make_case(
                "lme-" + question_id,
                "agg",
                "longmemeval_oracle.json",
                {"question_id": question_id},
                question["question"],
                records,
                {
                    k: question[k]
                    for k in ["answer", "question_type", "answer_session_ids"]
                },
                1,
                note,
                context={"question_date": question["question_date"]},
            )
        )
    tree = ast.parse((DATA / "holdup-original-config.txt").read_text())
    declaration = tree.body[0]
    if not isinstance(declaration, ast.Assign):
        raise ValueError("Expected original HOLDUP DATASETS assignment")
    config = ast.literal_eval(declaration.value)
    with (DATA / "ag-news.csv").open() as stream:
        news = list(csv.DictReader(stream))
    indices = list(range(0, len(news), 10))
    selected = [news[i] for i in indices]
    for mode in ["classification", "clustering"]:
        spec = config["Ag_News_" + mode]
        cases.append(
            make_case(
                "ag-news-" + mode,
                "groupby",
                "ag-news.csv",
                {"zero_based_indices": indices, "config": "Ag_News_" + mode},
                spec["hint"]
                + (
                    "\nCategories: " + "; ".join(spec["gt_names"]) + "."
                    if mode == "classification"
                    else ""
                ),
                [
                    {"id": row["id"], "Description": row["Description"]}
                    for row in selected
                ],
                [{"id": row["id"], "Class": row["Class"]} for row in selected],
                10,
                "原新闻内容不变，按文件位置每十条选一条。分类任务把原 hint 与预设类别合为 Ig；聚类任务使用原 hint。静态标签仅评价最终分组，不充当逐批重判金标。single；C=None；不是完整复现原系统执行方案。",
            )
        )
        cases[-1]["task_binding"] = {
            "original_hint": spec["hint"],
            "categories": spec["gt_names"] if mode == "classification" else None,
            "C": None,
            "membership": "single",
        }
    sample = load_locomo_sample(DATA / "locomo10.json", sample_index=0)
    events = sample.events[:32]
    qa = eligible_questions(
        [q for q in sample.questions if q.evidence_event_ids],
        ingested_event_ids=[e.event_id for e in events],
        question_limit=5,
    )
    if len(qa) != 5:
        raise ValueError("Expected five original, evidence-covered LOCOMO questions")
    cases.append(
        make_case(
            "claude-locomo-s0-32",
            "policy",
            "locomo10.json",
            {
                "sample_index": 0,
                "sample_id": sample.sample_id,
                "event_slice": [0, 32],
                "question_ids": [q.question_id for q in qa],
            },
            None,
            [asdict(e) for e in events],
            [asdict(q) for q in qa],
            1,
            "仅使用原对话前32条消息，不改文本；选前五个标注证据已覆盖的原问题。这是前缀小测，不是完整benchmark成绩。每条消息一次维护；baseline 与实验版顺序运行，Agent 不读取未来检索问题。",
            context={"policy": "agent_memory.memories.claude.ClaudeMemory"},
        )
    )
    with (DATA / "banking77-test.csv").open() as stream:
        bank = list(csv.DictReader(stream))[:20]
    cases.append(
        make_case(
            "banking77-reserve",
            "groupby",
            "banking77-test.csv",
            {"slice": [0, 20]},
            None,
            [{"text": r["text"]} for r in bank],
            [r["category"] for r in bank],
            5,
            "原分类数据；没有逐字原 query。本轮仅预检，不为凑单归属覆盖自行写 instruction。",
            ready=False,
        )
    )
    nlu = json.loads((DATA / "nlupp-fold0.json").read_text())[:20]
    cases.append(
        make_case(
            "nlupp-reserve",
            "groupby",
            "nlupp-fold0.json",
            {"slice": [0, 20]},
            None,
            [{"text": r["text"]} for r in nlu],
            [r["intents"] for r in nlu],
            5,
            "原多意图数据；没有逐字原 query。本轮仅预检，标签不进入输入。",
            ready=False,
        )
    )
    compile_cases(cases)
    manifest = {
        "stage": "data-preflight-only",
        "model_calls": 0,
        "proposed_model": "deepseek/flash",
        "thinking": False,
        "budget_approved": False,
        "sources": files,
        "cases": cases,
    }
    dump(ROOT / "manifest.json", manifest)
    (ROOT / "index.html").write_text(render(cases, files))
    for case in cases:
        print(case["id"], case["records"], len(case["batch_ranges"]), case["status"])
    print("No models called. Report:", ROOT / "index.html")


if __name__ == "__main__":
    main()
