"""! @file test_course_qa_report_stage4.py
@brief 验证 Issue #7 阶段 4 的报告隔离说明和固定展示问题。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_eval import (  # noqa: E402
    COURSE_QA_DEMO_QUESTIONS,
    CourseQaCandidate,
    build_markdown_report,
    load_course_qa_candidates,
    select_demo_questions,
)


def test_select_demo_questions_uses_fixed_public_dataset_questions():
    """! @brief 固定现场展示问题必须来自课程 QA public 数据。"""
    candidates = load_course_qa_candidates()

    assert select_demo_questions(candidates) == COURSE_QA_DEMO_QUESTIONS


def test_select_demo_questions_handles_single_pass_iterable_fallback():
    """! @brief 固定问题缺失时，一次性迭代器也能按公开候选顺序补足。"""
    candidates = (
        candidate
        for candidate in [
            CourseQaCandidate(
                category="测试分类",
                qa_id=999,
                question="备用展示问题？",
                answer_id="ans-stage4-fallback",
                answer="备用展示答案。",
            )
        ]
    )

    assert select_demo_questions(candidates, max_questions=1) == ["备用展示问题？"]


def test_markdown_report_documents_label_and_frontend_boundaries():
    """! @brief Markdown 报告必须写清隐藏标签隔离和前端展示边界。"""
    report = build_markdown_report(
        {
            "dataset_path": "sample_data/course_qa_public.json",
            "labels_path": "eval/labels/course_qa_quality_labels.json",
            "modes": ["llm_only", "basic_rag", "optimized_rag"],
            "provider": "mock",
            "model": "mock-generator",
            "top_k": 3,
            "question_count": 5,
            "record_count": 15,
            "demo_questions": COURSE_QA_DEMO_QUESTIONS,
            "summary_by_mode": {
                "llm_only": _summary_row(),
                "basic_rag": _summary_row(),
                "optimized_rag": _summary_row(),
            },
        }
    )

    assert "## 数据隔离说明" in report
    assert "`sample_data/course_qa_public.json`" in report
    assert "`eval/labels/course_qa_quality_labels.json`" in report
    assert "`answer_quality` 不进入 RAG 索引、LLM prompt、trace、retrieved hits 或前端展示" in report
    assert "## 前端展示边界" in report
    assert "前端只消费 `RagAnswer` schema" in report
    assert "`label_distribution`、`top_hit_quality`、`avg_hit_quality` 属于评测派生字段" in report
    for question in COURSE_QA_DEMO_QUESTIONS:
        assert question in report


def _summary_row() -> dict[str, object]:
    """! @brief 构造报告表格使用的最小汇总指标。"""
    return {
        "num_questions": 5,
        "avg_latency_ms": 1.0,
        "avg_citation_hit": 1.0,
        "avg_groundedness": 1.0,
        "avg_same_question_hit_count": 3.0,
        "avg_cross_question_hit_count": 0.0,
        "avg_top_hit_quality": None,
        "avg_hit_quality": None,
        "label_distribution": {},
    }
