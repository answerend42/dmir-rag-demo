"""! @file test_label_isolation.py
@brief 用小规模 fake 评测验证隐藏质量标签不会进入 RAG 原始输出。
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_eval import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_LABELS,
    build_raw_answers_output,
    parse_modes,
    run_course_qa,
    sanitize_for_report,
    save_course_qa_outputs,
)


FORBIDDEN_LABEL_KEY = "answer_quality"
PATH_FIELD_MARKERS = ("path", "file", "filename", "filepath")


def test_parse_modes_rejects_duplicate_modes():
    """! @brief 重复 RAG 模式必须报错，避免逐条记录与模式汇总数量不一致。"""
    with pytest.raises(ValueError, match="重复的 RAG 模式：llm_only"):
        parse_modes("llm_only,llm_only")


def test_small_fake_eval_keeps_answer_quality_out_of_raw_outputs(tmp_path):
    """! @brief 小规模 fake pipeline 评测后，确认隐藏标签不进入请求、回答和 raw answers。"""
    result = sanitize_for_report(run_course_qa(_small_eval_args(tmp_path), parse_modes("all")))
    saved_files = save_course_qa_outputs(result, tmp_path)
    raw_output = build_raw_answers_output(result)

    for record in raw_output["records"]:
        request_text = json.dumps(record["request"], ensure_ascii=False)
        retrieved_hits_text = json.dumps(record["answer"]["retrieved_hits"], ensure_ascii=False)
        trace_text = json.dumps(record["answer"]["trace"], ensure_ascii=False)

        assert FORBIDDEN_LABEL_KEY not in request_text
        assert FORBIDDEN_LABEL_KEY not in retrieved_hits_text
        assert FORBIDDEN_LABEL_KEY not in trace_text

    raw_answers_text = (tmp_path / saved_files["raw_answers"]).read_text(encoding="utf-8")
    raw_answers_json = json.loads(raw_answers_text)

    assert FORBIDDEN_LABEL_KEY not in raw_answers_text
    assert not _contains_absolute_path(raw_answers_json)


def test_absolute_path_checker_only_flags_repo_file_fields():
    """! @brief 绝对路径检查只关注项目文件字段，避免误伤普通文本路径。"""
    payload = {
        "answer_markdown": "接口路径 /rag/answer 不应被当作文件路径。",
        "route": "/api/rag/answer",
        "source": "/paper/section/1",
        "dataset_path": str(REPO_ROOT / "sample_data" / "course_qa_public.json"),
    }

    assert _contains_absolute_path(payload)


def test_absolute_path_checker_ignores_non_file_absolute_strings():
    """! @brief 非路径字段中的斜杠字符串不应触发绝对文件路径告警。"""
    payload = {
        "answer_markdown": "模型文本里可能包含 /tmp/example 或 /api/rag/answer。",
        "route": "/api/rag/answer",
        "source": "/paper/section/1",
    }

    assert not _contains_absolute_path(payload)


def _small_eval_args(output_dir: Path) -> Namespace:
    """! @brief 构造 limit=5 的课程 QA fake 评测参数。"""
    return Namespace(
        dataset=DEFAULT_DATASET,
        labels=DEFAULT_LABELS,
        provider="mock",
        model="mock-generator",
        top_k=3,
        limit=5,
        output_dir=str(output_dir),
    )


def _contains_absolute_path(value: Any, key_context: str = "") -> bool:
    """! @brief 递归判断疑似文件字段中是否包含本项目绝对路径。"""
    if isinstance(value, dict):
        return any(_contains_absolute_path(item, str(key)) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_absolute_path(item, key_context) for item in value)
    if isinstance(value, str) and _is_path_field(key_context):
        path = Path(value)
        return path.is_absolute() and _is_under_repo_root(path)
    return False


def _is_path_field(key_context: str) -> bool:
    """! @brief 判断字段名是否像文件路径字段。"""
    normalized = key_context.lower()
    return any(marker in normalized for marker in PATH_FIELD_MARKERS)


def _is_under_repo_root(path: Path) -> bool:
    """! @brief 判断绝对路径是否指向当前项目目录。"""
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True
