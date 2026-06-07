#!/usr/bin/env python3
"""! @file run_eval.py
@brief 课程 QA 与论文 fixture 的离线三模式评测脚本。
@details 本脚本只在生成完成后读取隐藏标签文件；输出给前端的 JSON 不包含
answer_quality 字段，避免评测档次进入 RAG 系统或展示层。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag_core.chunkers import ResearchPaperChunker
from rag_core.contracts.enums import RagMode
from rag_core.contracts.models import ParsedDocument, RagAnswer, RagRequest
from rag_core.parsers import MarkdownPaperParser
from rag_core.pipeline import FakeRagPipeline
from rag_core.testing import build_course_qa_document, load_course_qa_candidates, load_course_qa_quality_labels

DEFAULT_OUTPUT_DIR = Path("eval/results")
DEFAULT_COURSE_QA_INPUT = Path("sample_data/course_qa_public.json")
DEFAULT_PAPER_FIXTURE = Path("sample_data/papers/paper_eval_fixture.json")
FORBIDDEN_OUTPUT_KEY = "answer_quality"


@dataclass(frozen=True)
class EvalQuestion:
    """! @brief 单条评测问题。"""

    question_id: str
    question: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvalDataset:
    """! @brief 评测数据集，包含文档、问题和可选隐藏标签。"""

    dataset_type: str
    document: ParsedDocument
    questions: list[EvalQuestion]
    labels: dict[str, int]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """! @brief 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行课程 QA / 论文 fixture 三模式离线评测。")
    parser.add_argument("--dataset-type", choices=["course_qa", "paper"], default="course_qa")
    parser.add_argument("--dataset", default=str(DEFAULT_COURSE_QA_INPUT), help="课程 QA public 数据路径。")
    parser.add_argument("--labels", default="eval/labels/course_qa_quality_labels.json", help="课程 QA 隐藏标签路径。")
    parser.add_argument("--paper-fixture", default=str(DEFAULT_PAPER_FIXTURE), help="论文阶段 fixture JSON 路径。")
    parser.add_argument("--modes", default="all", help="all 或逗号分隔的模式列表。")
    parser.add_argument("--limit", type=int, default=5, help="最多评测的问题数。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="评测结果输出目录。")
    parser.add_argument("--pretty", action="store_true", help="格式化 JSON 输出。")
    return parser.parse_args()


def main() -> int:
    """! @brief 运行评测并写出 JSON/CSV/Markdown。"""
    args = parse_args()
    modes = _resolve_modes(args.modes)
    dataset = _load_dataset(args)
    results = _run_modes(dataset, modes)
    payload = _build_payload(dataset, results)
    _assert_no_forbidden_output(payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "course_qa_eval" if dataset.dataset_type == "course_qa" else "paper_eval"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    _write_csv(csv_path, payload["summary"])
    _write_markdown(md_path, payload)

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


def _resolve_modes(raw_modes: str) -> list[RagMode]:
    """! @brief 解析 all 或逗号分隔的 RAG 模式。"""
    if raw_modes == "all":
        return [RagMode.LLM_ONLY, RagMode.BASIC_RAG, RagMode.OPTIMIZED_RAG]
    modes: list[RagMode] = []
    for item in raw_modes.split(","):
        name = item.strip()
        if name:
            modes.append(RagMode(name))
    if not modes:
        raise ValueError("--modes 至少需要一个模式")
    return modes


def _load_dataset(args: argparse.Namespace) -> EvalDataset:
    """! @brief 根据 dataset-type 加载评测数据集。"""
    if args.dataset_type == "paper":
        return _load_paper_dataset(Path(args.paper_fixture), args.limit)
    return _load_course_qa_dataset(Path(args.dataset), Path(args.labels), args.limit)


def _load_course_qa_dataset(dataset_path: Path, labels_path: Path, limit: int) -> EvalDataset:
    """! @brief 加载课程 QA public 输入和生成后评测标签。"""
    candidates = load_course_qa_candidates(dataset_path=dataset_path, max_questions=limit)
    document = build_course_qa_document(candidates, source=_safe_display_path(dataset_path))
    questions: list[EvalQuestion] = []
    seen_questions: set[tuple[str, int]] = set()
    for candidate in candidates:
        key = (candidate.category, candidate.qa_id)
        if key in seen_questions:
            continue
        seen_questions.add(key)
        questions.append(
            EvalQuestion(
                question_id=f"course-{candidate.qa_id}",
                question=candidate.question,
                metadata={"category": candidate.category, "qa_id": candidate.qa_id},
            )
        )
    labels = load_course_qa_quality_labels(labels_path)
    return EvalDataset(
        dataset_type="course_qa",
        document=document,
        questions=questions[:limit],
        labels=labels,
        metadata={"dataset_path": _safe_display_path(dataset_path), "labels_path": _safe_display_path(labels_path)},
    )


def _load_paper_dataset(fixture_path: Path, limit: int) -> EvalDataset:
    """! @brief 加载论文阶段 fixture，并解析目标论文 Markdown。"""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    target = fixture["target_paper"]
    source_path = Path(target["source_path"])
    document = MarkdownPaperParser().parse(str(source_path))
    questions = [
        EvalQuestion(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            metadata={"expected_evidence": row.get("expected_evidence", [])},
        )
        for row in fixture.get("questions", [])
    ]
    return EvalDataset(
        dataset_type="paper",
        document=document,
        questions=questions[:limit],
        labels={},
        metadata={
            "fixture_path": _safe_display_path(fixture_path),
            "target_paper": target,
            "distractor_papers": fixture.get("distractor_papers", []),
        },
    )


def _run_modes(dataset: EvalDataset, modes: list[RagMode]) -> dict[str, list[dict[str, Any]]]:
    """! @brief 对每个模式运行全部评测问题。"""
    pipeline = _build_pipeline(dataset.dataset_type)
    results: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        mode_rows = []
        for question in dataset.questions:
            request = RagRequest(
                query=question.question,
                rag_mode=mode,
                top_k=3,
                collection_id=f"{dataset.dataset_type}-eval",
                provider="mock",
                model="mock-generator",
                require_citations=True,
                metadata={"dataset_type": dataset.dataset_type, **question.metadata},
            )
            answer = pipeline.answer_document(dataset.document, request)
            _assert_no_forbidden_output(answer.model_dump(mode="json"))
            mode_rows.append(_summarize_answer(question, answer, dataset.labels))
        results[mode.value] = mode_rows
    return results


def _build_pipeline(dataset_type: str) -> FakeRagPipeline:
    """! @brief 构造评测流水线，论文 fixture 使用论文分块器。"""
    if dataset_type == "paper":
        return FakeRagPipeline(chunker=ResearchPaperChunker())
    return FakeRagPipeline()


def _summarize_answer(
    question: EvalQuestion,
    answer: RagAnswer,
    labels: dict[str, int],
) -> dict[str, Any]:
    """! @brief 将 RagAnswer 汇总为安全评测行。"""
    refused = "无法生成有证据支撑的回答" in answer.answer_markdown
    label_counter: Counter[str] = Counter()
    for hit in answer.retrieved_hits:
        answer_id = str(hit.metadata.get("answer_id", ""))
        if answer_id in labels:
            label_counter[str(labels[answer_id])] += 1
    return {
        "question_id": question.question_id,
        "question": question.question,
        "answerable": not refused,
        "cited": bool(answer.citations),
        "refused": refused,
        "latency_ms": round(sum(stage.latency_ms for stage in answer.trace), 3),
        "retrieved_count": len(answer.retrieved_hits),
        "citation_count": len(answer.citations),
        "label_distribution": dict(sorted(label_counter.items())),
        "answer_preview": answer.answer_markdown[:240],
    }


def _build_payload(dataset: EvalDataset, results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """! @brief 构造 JSON/CSV/Markdown 共用评测 payload。"""
    summary = {mode: _summarize_mode(mode, rows) for mode, rows in results.items()}
    return {
        "dataset_type": dataset.dataset_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(dataset.questions),
        "metadata": dataset.metadata,
        "summary": summary,
        "demo_questions": [
            {"question_id": question.question_id, "question": question.question, "metadata": question.metadata}
            for question in dataset.questions
        ],
        "result_samples": {mode: rows[:5] for mode, rows in results.items()},
    }


def _summarize_mode(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """! @brief 汇总单个 RAG 模式的指标。"""
    total = len(rows)
    answerable = sum(1 for row in rows if row["answerable"])
    cited = sum(1 for row in rows if row["cited"])
    refused = sum(1 for row in rows if row["refused"])
    latency_values = [float(row["latency_ms"]) for row in rows]
    label_counter: Counter[str] = Counter()
    for row in rows:
        label_counter.update(row["label_distribution"])
    return {
        "total": total,
        "answerable": answerable,
        "cited": cited,
        "refused": refused,
        "citation_hit": round(cited / total, 3) if total else 0.0,
        "groundedness": round(sum(1 for row in rows if row["answerable"] and row["cited"]) / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(latency_values) / total, 3) if total else 0.0,
        "label_distribution": dict(sorted(label_counter.items())),
        "note": _mode_note(mode),
    }


def _mode_note(mode: str) -> str:
    """! @brief 返回前端评测表使用的中文模式说明。"""
    notes = {
        RagMode.LLM_ONLY.value: "不使用检索证据，用于展示纯模型盲区。",
        RagMode.BASIC_RAG.value: "dense top-k 检索后直接生成。",
        RagMode.OPTIMIZED_RAG.value: "预留 query rewrite / rerank / grounded prompt，对证据不足场景更谨慎。",
    }
    return notes.get(mode, "")


def _write_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    """! @brief 写出三模式评测 CSV 摘要。"""
    fieldnames = [
        "mode",
        "total",
        "answerable",
        "cited",
        "refused",
        "citation_hit",
        "groundedness",
        "avg_latency_ms",
        "label_distribution",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for mode, row in summary.items():
            writer.writerow(
                {
                    **row,
                    "mode": mode,
                    "label_distribution": json.dumps(row["label_distribution"], ensure_ascii=False, sort_keys=True),
                }
            )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """! @brief 写出中文 Markdown 评测报告摘要。"""
    lines = [
        "# 三模式离线评测摘要",
        "",
        f"- 数据集：`{payload['dataset_type']}`",
        f"- 问题数：{payload['question_count']}",
        f"- 生成时间：{payload['generated_at']}",
        "",
        "| 模式 | 可回答 | 有引用 | 拒答 | citation_hit | groundedness | 平均耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, row in payload["summary"].items():
        lines.append(
            f"| `{mode}` | {row['answerable']} | {row['cited']} | {row['refused']} | "
            f"{row['citation_hit']:.3f} | {row['groundedness']:.3f} | {row['avg_latency_ms']:.1f} ms |"
        )
    lines.extend(["", "## 现场问题", ""])
    for question in payload["demo_questions"]:
        lines.append(f"- `{question['question_id']}` {question['question']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_display_path(path: Path) -> str:
    """! @brief 将路径转换为评测报告可公开展示的相对路径。"""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _assert_no_forbidden_output(value: Any) -> None:
    """! @brief 确保输出中没有隐藏评测字段名。"""
    serialized = json.dumps(value, ensure_ascii=False)
    if FORBIDDEN_OUTPUT_KEY in serialized:
        raise ValueError("评测输出禁止包含 answer_quality 字段")


if __name__ == "__main__":
    raise SystemExit(main())
