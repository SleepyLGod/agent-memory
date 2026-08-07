"""LongMemEval-specific evaluator artifact export."""

from __future__ import annotations

import json
from pathlib import Path

from agent_memory.evaluation.artifacts import BenchmarkArtifactStore
from agent_memory.evaluation.bundle import BenchmarkBundle

from .scoring import hypothesis_record


def write_official_hypotheses(
    bundle: BenchmarkBundle,
    output_dir: Path,
) -> Path:
    """Write completed answers in the official evaluator JSONL shape."""

    records: list[dict[str, object]] = []
    store = BenchmarkArtifactStore(output_dir)
    for case in bundle.cases:
        answers_path = store.case_dir(case.case_id) / "answers.jsonl"
        if not answers_path.is_file():
            raise ValueError(f"completed LongMemEval case has no answers: {case.case_id}")
        answers: dict[str, str] = {}
        for line_number, line in enumerate(
            answers_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{answers_path}:{line_number} must be an object")
            question_id = row.get("question_id")
            answer = row.get("answer")
            if not isinstance(question_id, str) or not isinstance(answer, str):
                raise ValueError(
                    f"{answers_path}:{line_number} requires string question_id and answer"
                )
            if question_id in answers:
                raise ValueError(f"duplicate LongMemEval answer {question_id!r}")
            answers[question_id] = answer

        expected_ids = {question.question_id for question in case.questions}
        if set(answers) != expected_ids:
            raise ValueError(f"LongMemEval answer set is incomplete for {case.case_id}")
        for question in case.questions:
            records.append(
                dict(hypothesis_record(question, answers[question.question_id]))
            )

    path = output_dir / "official_hypotheses.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


__all__ = ["write_official_hypotheses"]
