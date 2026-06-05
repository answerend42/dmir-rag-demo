#!/usr/bin/env python3
"""! @file run_eval.py
@brief 运行 Issue #7 课程 QA 离线评测主流程。
@details 本阶段只读取 RAG 可见的 course_qa_public 数据，重新执行 fake RAG，
并输出每个问题、每个模式对应的 RagRequest、RagAnswer 与评测指标。评测
专用 labels 只会在所有 RagAnswer 生成完成后读取，不会进入请求、检索命中
或 trace。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
COURSE_QA_CSV_COLUMNS = [
    "qa_id",
    "category",
    "question",
    "mode",
    "provider",
    "model",
    "latency_ms",
    "citation_hit",
    "groundedness",
    "label_distribution",
    "retrieved_hit_count",
    "same_question_hit_count",
    "cross_question_hit_count",
    "citation_count",
    "top_hit_answer_id",
    "top_hit_quality",
    "avg_hit_quality",
    "warning_count",
]


def parse_args() -> argparse.Namespace:
    """! @brief 解析课程 QA 离线评测 CLI 参数。
    @return argparse 命名空间。
    """
    parser = argparse.ArgumentParser(description="运行课程 QA 离线 fake RAG 评测主流程。")
    parser.add_argument(
        "--dataset-type",
        choices=["course_qa", "paper"],
        default="course_qa",
        help="数据集类型；当前仅实现 course_qa，paper 为后续阶段预留。",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="课程 QA 公开输入路径。")
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="评测专用 labels 路径；只在所有回答生成完成后读取。",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help="要运行的 RAG 模式。可填 all，或用逗号分隔：llm_only,basic_rag,optimized_rag。",
    )
    parser.add_argument("--provider", default="mock", help="当前只支持 mock。")
    parser.add_argument("--model", default="mock-generator", help="当前只支持 mock-generator。")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回的 top-k 数量。")
    parser.add_argument("--limit", type=int, default=None, help="最多评测的问题数；不影响全量候选索引。")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="JSON、CSV 与 Markdown 报告输出目录。",
    )
    parser.add_argument("--pretty", action="store_true", help="格式化输出 JSON。")
    args = parser.parse_args()

    if args.dataset_type != "course_qa":
        parser.error("当前仅支持 --dataset-type course_qa；paper 将在后续阶段实现。")
    if args.provider != "mock" or args.model != "mock-generator":
        parser.error("当前只运行 fake pipeline，请使用 --provider mock --model mock-generator。")
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
    """! @brief 使用全量课程 QA 候选答案运行 fake RAG 并计算指标。
    @param args CLI 参数。
    @param modes 要运行的 RAG 模式列表。
    @return 可序列化的原始回答记录和评测指标。
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

    quality_labels = load_course_qa_quality_label_map(args.labels)
    metrics_records = [compute_course_qa_metrics(record, quality_labels) for record in records]

    return {
        "stage": "issue_07_phase_3_course_qa_saved_reports",
        "dataset_type": "course_qa",
        "dataset_path": args.dataset,
        "labels_path": args.labels,
        "labels_read_after_generation": True,
        "quality_label_count": len(quality_labels),
        "output_dir_reserved": args.output_dir,
        "modes": [mode.value for mode in modes],
        "provider": args.provider,
        "model": args.model,
        "top_k": args.top_k,
        "question_count": len(questions),
        "record_count": len(records),
        "dataset_summary": dataset_summary,
        "records": records,
        "metrics_records": metrics_records,
        "summary_by_mode": summarize_metrics_by_mode(metrics_records, modes),
    }


def _assert_rag_answer(answer: RagAnswer) -> None:
    """! @brief 保护输出必须保持 RagAnswer 契约对象。"""
    if not isinstance(answer, RagAnswer):
        raise TypeError("FakeRagPipeline must return RagAnswer")


def load_course_qa_quality_label_map(labels_path: str | Path) -> dict[tuple[str, int, str], int]:
    """! @brief 读取评测专用质量标签并构造复合键映射。
    @param labels_path 隐藏 labels JSON 文件路径。
    @return (category, qa_id, answer_id) 到质量档次的映射。
    @throws ValueError 当标签重复、档次越界或 answer_id 非唯一时抛出。
    @details 该函数只能在 RAG 生成完成后调用；返回值仅供指标计算使用。
    """
    raw_data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    labels: dict[tuple[str, int, str], int] = {}
    seen_answer_ids: set[str] = set()

    for row in raw_data.get("labels", []):
        category = str(row["category"])
        qa_id = int(row["qa_id"])
        answer_id = str(row["answer_id"])
        quality = int(row["answer_quality"])
        key = (category, qa_id, answer_id)

        if key in labels:
            raise ValueError(f"重复的课程 QA 标签复合键：{key}")
        if answer_id in seen_answer_ids:
            raise ValueError(f"answer_id 不是全局唯一：{answer_id}")
        if quality < 0 or quality > 9:
            raise ValueError(f"质量档次越界：{quality}")

        labels[key] = quality
        seen_answer_ids.add(answer_id)

    return labels


def compute_course_qa_metrics(
    record: dict[str, Any],
    quality_labels: dict[tuple[str, int, str], int],
) -> dict[str, Any]:
    """! @brief 从单条 RagAnswer 与生成后 labels 计算课程 QA 指标。
    @param record 单个问题和模式的原始评测记录。
    @param quality_labels 复合键质量标签映射。
    @return 单条扁平指标记录。
    """
    answer = record["answer"]
    hits = answer.get("retrieved_hits", [])
    citations = answer.get("citations", [])
    current_key = (str(record["category"]), int(record["qa_id"]))
    hit_by_chunk_id = {str(hit.get("chunk_id")): hit for hit in hits}
    same_question_hits = [hit for hit in hits if hit_belongs_to_question(hit, current_key)]
    cross_question_hits = [hit for hit in hits if not hit_belongs_to_question(hit, current_key)]

    same_hit_qualities = [lookup_hit_quality(hit, quality_labels) for hit in same_question_hits]
    quality_distribution = Counter(str(quality) for quality in same_hit_qualities)
    top_hit = min(same_question_hits, key=lambda hit: int(hit.get("rank", 10**9)), default=None)
    top_hit_answer_id = get_hit_answer_id(top_hit) if top_hit else None
    top_hit_quality = lookup_hit_quality(top_hit, quality_labels) if top_hit else None

    return {
        "qa_id": record["qa_id"],
        "category": record["category"],
        "question": record["question"],
        "mode": record["mode"],
        "provider": record["provider"],
        "model": record["model"],
        "latency_ms": sum(float(stage.get("latency_ms", 0.0)) for stage in answer.get("trace", [])),
        "citation_hit": compute_citation_hit(citations, hit_by_chunk_id, current_key),
        "groundedness": compute_groundedness(citations, hits, hit_by_chunk_id, current_key),
        "label_distribution": dict(sorted(quality_distribution.items(), key=lambda item: int(item[0]))),
        "retrieved_hit_count": len(hits),
        "same_question_hit_count": len(same_question_hits),
        "cross_question_hit_count": len(cross_question_hits),
        "citation_count": len(citations),
        "top_hit_answer_id": top_hit_answer_id,
        "top_hit_quality": top_hit_quality,
        "avg_hit_quality": average(same_hit_qualities),
        "warning_count": len(answer.get("warnings", [])),
    }


def hit_belongs_to_question(hit: dict[str, Any], question_key: tuple[str, int]) -> bool:
    """! @brief 判断检索命中是否属于当前问题。"""
    metadata = hit.get("metadata", {})
    return (str(metadata.get("category")), int(metadata.get("qa_id", -1))) == question_key


def compute_citation_hit(
    citations: list[dict[str, Any]],
    hit_by_chunk_id: dict[str, dict[str, Any]],
    question_key: tuple[str, int],
) -> int:
    """! @brief 判断引用是否命中当前问题候选答案。"""
    for citation in citations:
        hit = hit_by_chunk_id.get(str(citation.get("chunk_id")))
        if hit and hit_belongs_to_question(hit, question_key):
            return 1
    return 0


def compute_groundedness(
    citations: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    hit_by_chunk_id: dict[str, dict[str, Any]],
    question_key: tuple[str, int],
) -> float:
    """! @brief 计算答案引用对当前问题证据的接地程度。"""
    if not hits and not citations:
        return 0.0
    if hits and not citations:
        return 0.3

    on_question_citations = 0
    for citation in citations:
        quote = str(citation.get("quote", ""))
        is_traceable = bool(quote) and any(quote in str(hit.get("text", "")) for hit in hits)
        hit = hit_by_chunk_id.get(str(citation.get("chunk_id")))
        if is_traceable and hit and hit_belongs_to_question(hit, question_key):
            on_question_citations += 1

    return on_question_citations / len(citations) if citations else 0.0


def lookup_hit_quality(hit: dict[str, Any], quality_labels: dict[tuple[str, int, str], int]) -> int:
    """! @brief 用复合键查找同题命中的隐藏质量档次。"""
    metadata = hit.get("metadata", {})
    key = (
        str(metadata.get("category")),
        int(metadata.get("qa_id", -1)),
        get_hit_answer_id(hit),
    )
    if key not in quality_labels:
        raise ValueError(f"检索命中缺少评测标签：{key}")
    return quality_labels[key]


def get_hit_answer_id(hit: dict[str, Any] | None) -> str | None:
    """! @brief 从检索命中 metadata 中取 answer_id。"""
    if not hit:
        return None
    answer_id = hit.get("metadata", {}).get("answer_id")
    return str(answer_id) if answer_id is not None else None


def average(values: list[int]) -> float | None:
    """! @brief 计算整数列表均值；空列表返回 None。"""
    if not values:
        return None
    return sum(values) / len(values)


def summarize_metrics_by_mode(metrics_records: list[dict[str, Any]], modes: list[RagMode]) -> dict[str, dict[str, Any]]:
    """! @brief 按 RAG 模式聚合阶段 3 所需汇总指标。
    @param metrics_records 单条指标记录列表。
    @param modes 输出顺序使用的模式列表。
    @return mode 到汇总指标的映射。
    """
    summary: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_records = [record for record in metrics_records if record["mode"] == mode.value]
        label_distribution: Counter[str] = Counter()
        for record in mode_records:
            label_distribution.update(record.get("label_distribution", {}))

        summary[mode.value] = {
            "num_questions": len({(record["category"], record["qa_id"]) for record in mode_records}),
            "avg_latency_ms": average_float(mode_records, "latency_ms"),
            "avg_citation_hit": average_float(mode_records, "citation_hit"),
            "avg_groundedness": average_float(mode_records, "groundedness"),
            "avg_same_question_hit_count": average_float(mode_records, "same_question_hit_count"),
            "avg_cross_question_hit_count": average_float(mode_records, "cross_question_hit_count"),
            "avg_top_hit_quality": average_float(mode_records, "top_hit_quality"),
            "avg_hit_quality": average_float(mode_records, "avg_hit_quality"),
            "label_distribution": dict(sorted(label_distribution.items(), key=lambda item: int(item[0]))),
        }
    return summary


def average_float(records: list[dict[str, Any]], field: str) -> float | None:
    """! @brief 对记录中的数值字段求平均，自动跳过 None。"""
    values = [float(record[field]) for record in records if record.get(field) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def save_course_qa_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """! @brief 保存阶段 3 的 JSON、CSV 和 Markdown 报告文件。
    @param result 已经脱敏的完整评测结果。
    @param output_dir 输出目录。
    @return 输出文件名到相对路径的映射。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_answers = build_raw_answers_output(result)
    metrics_output = build_metrics_output(result)
    raw_path = output_dir / "course_qa_raw_answers.json"
    metrics_path = output_dir / "course_qa_metrics.json"
    csv_path = output_dir / "course_qa_eval.csv"
    markdown_path = output_dir / "course_qa_eval.md"

    raw_path.write_text(json.dumps(raw_answers, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics_csv(result["metrics_records"], csv_path)
    markdown_path.write_text(build_markdown_report(result), encoding="utf-8")

    return {
        "raw_answers": sanitize_string(str(raw_path)),
        "metrics_json": sanitize_string(str(metrics_path)),
        "metrics_csv": sanitize_string(str(csv_path)),
        "markdown_report": sanitize_string(str(markdown_path)),
    }


def build_raw_answers_output(result: dict[str, Any]) -> dict[str, Any]:
    """! @brief 构造不含派生质量指标的 raw answers JSON。"""
    return {
        "dataset_type": result["dataset_type"],
        "dataset_path": result["dataset_path"],
        "modes": result["modes"],
        "provider": result["provider"],
        "model": result["model"],
        "top_k": result["top_k"],
        "question_count": result["question_count"],
        "record_count": result["record_count"],
        "dataset_summary": result["dataset_summary"],
        "records": result["records"],
    }


def build_metrics_output(result: dict[str, Any]) -> dict[str, Any]:
    """! @brief 构造逐条指标和按模式汇总指标 JSON。"""
    return {
        "dataset_type": result["dataset_type"],
        "dataset_path": result["dataset_path"],
        "labels_path": result["labels_path"],
        "labels_read_after_generation": result["labels_read_after_generation"],
        "quality_label_count": result["quality_label_count"],
        "modes": result["modes"],
        "provider": result["provider"],
        "model": result["model"],
        "top_k": result["top_k"],
        "question_count": result["question_count"],
        "metric_record_count": len(result["metrics_records"]),
        "summary_by_mode": result["summary_by_mode"],
        "metrics_records": result["metrics_records"],
    }


def write_metrics_csv(metrics_records: list[dict[str, Any]], csv_path: Path) -> None:
    """! @brief 用固定列顺序保存逐条扁平指标 CSV。"""
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COURSE_QA_CSV_COLUMNS)
        writer.writeheader()
        for record in metrics_records:
            writer.writerow({column: format_csv_value(record.get(column)) for column in COURSE_QA_CSV_COLUMNS})


def format_csv_value(value: Any) -> Any:
    """! @brief 将 dict 和空值转换为 CSV 中的稳定表示。"""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "null"
    return value


def build_markdown_report(result: dict[str, Any]) -> str:
    """! @brief 构造课程 QA 人类可读 Markdown 评测报告。"""
    lines = [
        "# Course QA 三模式评测报告",
        "",
        "## 数据与运行配置",
        "",
        f"- 数据集：`{result['dataset_path']}`",
        f"- 模式：{', '.join(result['modes'])}",
        f"- Provider / Model：`{result['provider']}` / `{result['model']}`",
        f"- Top-k：{result['top_k']}",
        f"- 评测问题数：{result['question_count']}",
        f"- 原始回答记录数：{result['record_count']}",
        "",
        "## 指标定义",
        "",
        "- `latency_ms`：该条回答所有 trace 阶段耗时之和。",
        "- `citation_hit`：引用是否命中当前问题的候选答案。",
        "- `groundedness`：引用是否可追溯且属于当前问题证据。",
        "- `same_question_hit_count`：检索命中中属于当前问题的数量。",
        "- `cross_question_hit_count`：检索命中中属于其他问题的数量。",
        "- `label_distribution`：同题命中的隐藏质量档次分布，仅用于评测报告。",
        "- `top_hit_quality` / `avg_hit_quality`：同题命中的最高排名答案质量与平均质量。",
        "",
        "## 按模式汇总",
        "",
        "| mode | num_questions | avg_latency_ms | avg_citation_hit | avg_groundedness | avg_same_hits | avg_cross_hits | avg_top_hit_quality | avg_hit_quality | label_distribution |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for mode, summary in result["summary_by_mode"].items():
        lines.append(
            "| {mode} | {num_questions} | {avg_latency_ms} | {avg_citation_hit} | {avg_groundedness} | "
            "{avg_same_question_hit_count} | {avg_cross_question_hit_count} | {avg_top_hit_quality} | "
            "{avg_hit_quality} | `{label_distribution}` |".format(
                mode=mode,
                num_questions=summary["num_questions"],
                avg_latency_ms=format_markdown_number(summary["avg_latency_ms"]),
                avg_citation_hit=format_markdown_number(summary["avg_citation_hit"]),
                avg_groundedness=format_markdown_number(summary["avg_groundedness"]),
                avg_same_question_hit_count=format_markdown_number(summary["avg_same_question_hit_count"]),
                avg_cross_question_hit_count=format_markdown_number(summary["avg_cross_question_hit_count"]),
                avg_top_hit_quality=format_markdown_number(summary["avg_top_hit_quality"]),
                avg_hit_quality=format_markdown_number(summary["avg_hit_quality"]),
                label_distribution=json.dumps(summary["label_distribution"], ensure_ascii=False, sort_keys=True),
            )
        )

    lines.extend(
        [
            "",
            "## 现场展示问题",
            "",
            *[f"{index}. {question}" for index, question in enumerate(select_demo_questions(result["metrics_records"]), start=1)],
            "",
            "## 输出文件",
            "",
            "- `course_qa_raw_answers.json`：每条问题、每种模式的完整 `RagAnswer`。",
            "- `course_qa_metrics.json`：逐条指标与按模式汇总指标。",
            "- `course_qa_eval.csv`：固定列顺序的逐条扁平指标。",
            "- `course_qa_eval.md`：当前 Markdown 报告。",
            "",
        ]
    )
    return "\n".join(lines)


def format_markdown_number(value: float | None) -> str:
    """! @brief Markdown 表格中使用的数字格式。"""
    if value is None:
        return "null"
    return f"{value:.4f}"


def select_demo_questions(metrics_records: list[dict[str, Any]], max_questions: int = 5) -> list[str]:
    """! @brief 从当前评测记录中选取前五个不重复展示问题。"""
    questions: list[str] = []
    seen: set[tuple[str, int]] = set()
    for record in metrics_records:
        key = (record["category"], int(record["qa_id"]))
        if key in seen:
            continue
        seen.add(key)
        questions.append(str(record["question"]))
        if len(questions) >= max_questions:
            break
    return questions


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
    saved_files = save_course_qa_outputs(sanitized, Path(args.output_dir))
    sanitized["saved_files"] = saved_files
    print(json.dumps(sanitized, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
