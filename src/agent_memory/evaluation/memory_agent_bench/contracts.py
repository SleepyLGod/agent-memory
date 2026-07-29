"""Executable task contracts for every pinned MemoryAgentBench source."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import re
from typing import Any, cast

from agent_memory.evaluation.harness import (
    GradeResult,
    JudgeStep,
    ModelPrompt,
    TaskContract,
)
from agent_memory.evaluation.longmemeval.scoring import (
    LONGMEMEVAL_JUDGE_PROMPT_DIGEST,
    build_judge_prompt,
    parse_judge_response,
)
from agent_memory.evaluation.types import BenchmarkQuestion

from .infbench_prompts import (
    INF_BENCH_FLUENCY_PROMPT,
    INF_BENCH_PRECISION_PROMPT,
    INF_BENCH_RECALL_PROMPT,
)
from .scoring import score_deterministic, score_movie_recommendations
from .tasks import MEMORY_AGENT_TASKS, MemoryAgentTask

_OFFICIAL_COMMIT = "455306dcabc3842526eb83cd4e225e5d486c5c5d"
_ANSWER_WRAPPER = "Retrieved Memory:\n{context}\n\n{query}"


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _parse_nonempty(response: str) -> str:
    answer = response.strip()
    if not answer:
        raise ValueError("MemoryAgentBench answer must be non-empty")
    return answer


def _retrieval_query(question: BenchmarkQuestion) -> str:
    question_date = question.metadata.get("question_date")
    if isinstance(question_date, str) and question_date:
        return f"Current Date: {question_date}\nQuestion: {question.question}"
    return question.question


def _answer_prompt(task: MemoryAgentTask, prompt_digest: str):
    def build(question: BenchmarkQuestion, context: str) -> ModelPrompt:
        return ModelPrompt(
            prompt_name=f"memory_agent_bench.{task.source}.answer",
            messages=(
                {"role": "system", "content": task.system_prompt},
                {
                    "role": "user",
                    "content": task.format_answer_prompt(question.question, context),
                },
            ),
            prompt_digest=prompt_digest,
            temperature=0,
            max_tokens=8192,
            thinking_enabled=False,
        )

    return build


def _deterministic_scorer(
    task: MemoryAgentTask,
    movie_entity_mapping: Mapping[str, int] | None,
):
    def score(question: BenchmarkQuestion, answer: str) -> GradeResult:
        details: dict[str, Any] = {"source": task.source}
        if task.scorer == "recall_at_5":
            if movie_entity_mapping is None:
                raise ValueError(
                    "ReDial Recall@5 requires the pinned movie entity mapping"
                )
            value, predicted, ground_truth = score_movie_recommendations(
                answer,
                question.gold_answer,
                movie_entity_mapping,
            )
            details.update(
                predicted_movies=predicted,
                ground_truth_movies=ground_truth,
            )
        else:
            value = score_deterministic(task, answer, question.gold_answer)
        return GradeResult(
            scorer_id=task.scorer,
            score=value,
            label="correct" if value == 1 else "incorrect",
            details=details,
        )

    return score


def _longmemeval_plan(
    question: BenchmarkQuestion, answer: str
) -> tuple[JudgeStep, ...]:
    return (
        JudgeStep(
            prompt=ModelPrompt(
                prompt_name="memory_agent_bench.longmemeval.v4_flash_judge",
                messages=(
                    {
                        "role": "user",
                        "content": build_judge_prompt(question, answer),
                    },
                ),
                prompt_digest=LONGMEMEVAL_JUDGE_PROMPT_DIGEST,
                temperature=0,
                max_tokens=10,
                thinking_enabled=False,
            ),
            parse=parse_judge_response,
        ),
    )


def _longmemeval_reduce(
    question: BenchmarkQuestion,
    answer: str,
    results: tuple[object, ...],
) -> GradeResult:
    del question, answer
    if len(results) != 1 or not isinstance(results[0], bool):
        raise ValueError("MemoryAgentBench LongMemEval judge requires one boolean")
    correct = results[0]
    return GradeResult(
        scorer_id="longmemeval_v4_flash_judge",
        score=float(correct),
        label="yes" if correct else "no",
        details={"official_metric_model": False},
    )


def _parse_json_object(response: str) -> Mapping[str, Any]:
    candidates = re.findall(r"\{.*?\}", response, re.DOTALL)
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    for candidate in reversed((*candidates, *fenced)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("InfBench judge response must contain one JSON object")


def _parse_infbench_metric(name: str, *, total_name: str | None = None):
    def parse(response: str) -> Mapping[str, int]:
        value = _parse_json_object(response)
        raw_score = value.get(name)
        if isinstance(raw_score, bool) or not isinstance(raw_score, int):
            raise ValueError(f"InfBench {name} must be an integer")
        if raw_score < 0:
            raise ValueError(f"InfBench {name} must be non-negative")
        result = {name: raw_score}
        if total_name is not None:
            raw_total = value.get(total_name)
            if isinstance(raw_total, bool) or not isinstance(raw_total, int):
                raise ValueError(f"InfBench {total_name} must be an integer")
            if raw_total < raw_score:
                raise ValueError(f"InfBench {total_name} cannot be less than {name}")
            result[total_name] = raw_total
        return result

    return parse


_parse_fluency = _parse_infbench_metric("fluency")
_parse_recall = _parse_infbench_metric("recall")
_parse_precision = _parse_infbench_metric(
    "precision", total_name="sentence_count"
)


def _infbench_plan(
    question: BenchmarkQuestion, answer: str
) -> tuple[JudgeStep, ...]:
    keypoints = question.metadata.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints or not all(
        isinstance(item, str) and item for item in keypoints
    ):
        raise ValueError("InfBench questions require non-empty keypoints")
    expert_summary = question.gold_answer
    if isinstance(expert_summary, list) and len(expert_summary) == 1:
        expert_summary = expert_summary[0]
    if not isinstance(expert_summary, str) or not expert_summary:
        raise ValueError("InfBench questions require one expert summary")

    prompts = (
        (
            "memory_agent_bench.infbench.fluency",
            INF_BENCH_FLUENCY_PROMPT.format(text=answer.strip()),
            sha256(INF_BENCH_FLUENCY_PROMPT.encode("utf-8")).hexdigest(),
            _parse_fluency,
        ),
        (
            "memory_agent_bench.infbench.recall",
            INF_BENCH_RECALL_PROMPT.format(
                keypoints="\n".join(
                    f"{index}. {keypoint}"
                    for index, keypoint in enumerate(keypoints, start=1)
                ),
                summary=answer.strip(),
            ),
            sha256(INF_BENCH_RECALL_PROMPT.encode("utf-8")).hexdigest(),
            _parse_recall,
        ),
        (
            "memory_agent_bench.infbench.precision",
            INF_BENCH_PRECISION_PROMPT.format(
                expert_summary=expert_summary,
                summary=answer.strip(),
            ),
            sha256(INF_BENCH_PRECISION_PROMPT.encode("utf-8")).hexdigest(),
            _parse_precision,
        ),
    )
    return tuple(
        JudgeStep(
            prompt=ModelPrompt(
                prompt_name=prompt_name,
                messages=({"role": "user", "content": content},),
                prompt_digest=prompt_digest,
                temperature=0,
                max_tokens=8192,
                thinking_enabled=False,
            ),
            parse=parse,
        )
        for prompt_name, content, prompt_digest, parse in prompts
    )


def _infbench_reduce(
    question: BenchmarkQuestion,
    answer: str,
    results: tuple[object, ...],
) -> GradeResult:
    del answer
    if len(results) != 3 or not all(isinstance(item, Mapping) for item in results):
        raise ValueError("InfBench judge requires fluency, recall, and precision")
    fluency = cast(Mapping[str, Any], results[0])
    recall_result = cast(Mapping[str, Any], results[1])
    precision_result = cast(Mapping[str, Any], results[2])
    keypoints = question.metadata.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints:
        raise ValueError("InfBench questions require keypoints")

    fluency_score = int(fluency["fluency"])
    recall_found = int(recall_result["recall"])
    precision_found = int(precision_result["precision"])
    precision_total = int(precision_result["sentence_count"])
    recall_total = len(keypoints)
    if recall_found > recall_total:
        raise ValueError("InfBench recall cannot exceed keypoint count")
    recall = recall_found / recall_total
    precision = precision_found / precision_total if precision_total else 0
    score = (
        fluency_score * 2 * recall * precision / (recall + precision)
        if recall + precision > 0
        else 0
    )
    return GradeResult(
        scorer_id="infbench_v4_flash_judge",
        score=score,
        details={
            "official_metric_model": False,
            "fluency": fluency_score,
            "recall_found": recall_found,
            "recall_total": recall_total,
            "recall": recall,
            "precision_found": precision_found,
            "precision_total": precision_total,
            "precision": precision,
        },
    )


def _task_contract(
    task: MemoryAgentTask,
    movie_entity_mapping: Mapping[str, int] | None,
) -> TaskContract:
    answer_digest = _digest(
        {
            "system": task.system_prompt,
            "wrapper": _ANSWER_WRAPPER,
            "query": task.query_template,
        }
    )
    common = {
        "task_id": task.contract_id,
        "answer_prompt": _answer_prompt(task, answer_digest),
        "answer_parser": _parse_nonempty,
        "retrieval_query": _retrieval_query,
        "answer_prompt_digest": answer_digest,
        "checkpoint_boundary": "event",
        "memory_system_error_score": 0.0,
    }
    if task.scorer == "longmemeval_v4_flash_judge":
        return TaskContract(
            **common,
            scorer_id=task.scorer,
            scorer_digest=LONGMEMEVAL_JUDGE_PROMPT_DIGEST,
            judge_plan=_longmemeval_plan,
            judge_reducer=_longmemeval_reduce,
        )
    if task.scorer == "infbench_v4_flash_judge":
        return TaskContract(
            **common,
            scorer_id=task.scorer,
            scorer_digest=_digest(
                {
                    "official_commit": _OFFICIAL_COMMIT,
                    "prompts": [
                        sha256(INF_BENCH_FLUENCY_PROMPT.encode()).hexdigest(),
                        sha256(INF_BENCH_RECALL_PROMPT.encode()).hexdigest(),
                        sha256(INF_BENCH_PRECISION_PROMPT.encode()).hexdigest(),
                    ],
                }
            ),
            judge_plan=_infbench_plan,
            judge_reducer=_infbench_reduce,
        )
    return TaskContract(
        **common,
        scorer_id=task.scorer,
        scorer_digest=_digest(
            {
                "official_commit": _OFFICIAL_COMMIT,
                "scorer": task.scorer,
                "movie_mapping": (
                    _digest(movie_entity_mapping)
                    if task.scorer == "recall_at_5"
                    and movie_entity_mapping is not None
                    else None
                ),
            }
        ),
        deterministic_scorer=_deterministic_scorer(task, movie_entity_mapping),
    )


def memory_agent_bench_task_contracts(
    *,
    movie_entity_mapping: Mapping[str, int] | None = None,
) -> Mapping[str, TaskContract]:
    """Return one source-specific contract for every pinned official task."""

    return {
        task.contract_id: _task_contract(task, movie_entity_mapping)
        for task in MEMORY_AGENT_TASKS.values()
    }


__all__ = ["memory_agent_bench_task_contracts"]
