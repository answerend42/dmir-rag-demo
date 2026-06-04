#!/usr/bin/env python3
"""! @file run_eval.py
@brief 运行 Issue #7 阶段 1 的课程 QA 离线评测主流程。
@details 本阶段只读取 RAG 可见的 course_qa_public 数据，重新执行 fake RAG，
并输出每个问题、每个模式对应的 RagRequest 与 RagAnswer。评测专用 labels
参数仅作为后续阶段预留，本阶段不会读取 answer_quality。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag_core.contracts.enums import RagMode
from rag_core.contracts.models import RagAnswer, RagRequest
from rag_core.pipeline import FakeRagPipeline
from rag_core.testing import (
    CourseQaCandidate,
    build_course_qa_document,
    load_course_qa_candidates,
    summarize_course_qa,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "sample_data/course_qa_public.json"
DEFAULT_LABELS = "eval/labels/course_qa_quality_labels.json"
DEFAULT_OUTPUT_DIR = "eval/results"


def parse_args() -> argparse.Namespace:
    """! @brief 解析阶段 1 离线评测 CLI 参数。
    @return argparse 命名空间。
    """
    parser = argparse.ArgumentParser(description="运行课程 QA 离线 fake RAG 评测主流程。")
    parser.add_argument(
        "--dataset-type",
        choices=["course_qa", "paper"],
        default="course_qa",
        help="数据集类型；阶段 1 仅实现 course_qa，paper 为后续阶段预留。",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="课程 QA 公开输入路径。")
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="评测专用 labels 路径；阶段 1 不读取，仅保留参数兼容性。",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help="要运行的 RAG 模式。可填 all，或用逗号分隔：llm_only,basic_rag,optimized_rag。",
    )
    parser.add_argument("--provider", default="mock", help="阶段 1 只支持 mock。")
    parser.add_argument("--model", default="mock-generator", help="阶段 1 只支持 mock-generator。")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回的 top-k 数量。")
    parser.add_argument("--limit", type=int, default=None, help="最多评测的问题数；不影响全量候选索引。")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="报告输出目录；阶段 1 不写文件，阶段 3 起使用。",
    )
    parser.add_argument("--pretty", action="store_true", help="格式化输出 JSON。")
    args = parser.parse_args()

    if args.dataset_type != "course_qa":
        parser.error("阶段 1 仅支持 --dataset-type course_qa；paper 将在后续阶段实现。")
    if args.provider != "mock" or args.model != "mock-generator":
        parser.error("阶段 1 只运行 fake pipeline，请使用 --provider mock --model mock-generator。")
    if args.top_k <= 0:
        parser.error("--top-k 必须为正整数。")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数。")

    return args


def parse_modes(raw_modes: str) -> list[RagMode]:
    """! @brief 将 CLI 模式字符串转换为 RagMode 列表。
    @param raw_modes all 或逗号分隔的模式值。
    @return 按用户指定顺序排列的 RagMode 列表。
    @throws ValueError 当模式名称不合法时抛出。
    """
    if raw_modes.strip() == "all":
        return [RagMode.LLM_ONLY, RagMode.BASIC_RAG, RagMode.OPTIMIZED_RAG]

    modes: list[RagMode] = []
    for value in (part.strip() for part in raw_modes.split(",")):
        if not value:
            continue
        try:
            modes.append(RagMode(value))
        except ValueError as exc:
            valid = ", ".join(mode.value for mode in RagMode)
            raise ValueError(f"不支持的 RAG 模式：{value}。可选：all, {valid}") from exc
    if not modes:
        raise ValueError("--modes 不能为空。")
    return modes


def collect_questions(candidates: Iterable[CourseQaCandidate], limit: int | None = None) -> list[dict[str, Any]]:
    """! @brief 按公开数据顺序收集唯一课程 QA 问题。
    @param candidates 公开候选答案列表。
    @param limit 可选最大问题数。
    @return 仅包含 category、qa_id、question 的题目列表。
    """
    questions: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for candidate in candidates:
        key = (candidate.category, candidate.qa_id)
        if key in seen:
            continue
        seen.add(key)
        questions.append(
            {
                "category": candidate.category,
                "qa_id": candidate.qa_id,
                "question": candidate.question,
            }
        )
        if limit is not None and len(questions) >= limit:
            break
    return questions


def run_course_qa(args: argparse.Namespace, modes: list[RagMode]) -> dict[str, Any]:
    """! @brief 使用全量课程 QA 候选答案运行阶段 1 fake RAG。
    @param args CLI 参数。
    @param modes 要运行的 RAG 模式列表。
    @return 可序列化的原始回答记录。
    """
    all_candidates = load_course_qa_candidates(dataset_path=args.dataset)
    questions = collect_questions(all_candidates, limit=args.limit)
    document = build_course_qa_document(all_candidates, source=args.dataset)
    dataset_summary = summarize_course_qa(all_candidates)

    records: list[dict[str, Any]] = []
    for question_index, question in enumerate(questions, start=1):
        for mode in modes:
            request = RagRequest(
                query=question["question"],
                rag_mode=mode,
                top_k=args.top_k,
                collection_id="course-qa-eval",
                model=args.model,
                provider=args.provider,
                require_citations=True,
                metadata={
                    "dataset_type": "course_qa",
                    "category": question["category"],
                    "qa_id": question["qa_id"],
                    "dataset_summary": dataset_summary,
                },
            )
            answer = FakeRagPipeline().answer_document(document, request)
            _assert_rag_answer(answer)
            answer.metadata["dataset_type"] = "course_qa"
            answer.metadata["dataset_path"] = args.dataset
            answer.metadata["question_key"] = {
                "category": question["category"],
                "qa_id": question["qa_id"],
            }

            records.append(
                {
                    "question_index": question_index,
                    "qa_id": question["qa_id"],
                    "category": question["category"],
                    "question": question["question"],
                    "mode": mode.value,
                    "provider": args.provider,
                    "model": args.model,
                    "request": request.model_dump(mode="json"),
                    "answer": answer.model_dump(mode="json"),
                }
            )

    return {
        "stage": "issue_07_phase_1_course_qa_raw_answers",
        "dataset_type": "course_qa",
        "dataset_path": args.dataset,
        "labels_path_configured_but_not_read": args.labels,
        "labels_read": False,
        "output_dir_reserved": args.output_dir,
        "modes": [mode.value for mode in modes],
        "provider": args.provider,
        "model": args.model,
        "top_k": args.top_k,
        "question_count": len(questions),
        "record_count": len(records),
        "dataset_summary": dataset_summary,
        "records": records,
    }


def _assert_rag_answer(answer: RagAnswer) -> None:
    """! @brief 保护阶段 1 输出必须保持 RagAnswer 契约对象。"""
    if not isinstance(answer, RagAnswer):
        raise TypeError("FakeRagPipeline must return RagAnswer")


def sanitize_for_report(value: Any) -> Any:
    """! @brief 递归脱敏输出中的绝对路径。
    @param value 任意 JSON 兼容结构。
    @return 绝对路径已替换为仓库相对路径或 basename 的结构。
    """
    if isinstance(value, dict):
        return {key: sanitize_for_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_report(item) for item in value]
    if isinstance(value, str):
        return sanitize_string(value)
    return value


def sanitize_string(text: str) -> str:
    """! @brief 将字符串中的本仓库绝对路径替换为相对路径。"""
    repo_root = str(REPO_ROOT)
    if repo_root in text:
        return text.replace(f"{repo_root}/", "").replace(repo_root, ".")

    path = Path(text)
    if path.is_absolute():
        return path.name or "<absolute-path-redacted>"
    return text


def main() -> int:
    """! @brief CLI 入口。"""
    args = parse_args()
    try:
        modes = parse_modes(args.modes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    result = run_course_qa(args, modes)
    sanitized = sanitize_for_report(result)
    print(json.dumps(sanitized, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
