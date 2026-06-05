"""! @file test_label_isolation.py
@brief 用小规模 fake 评测验证隐藏质量标签不会进入 RAG 原始输出。
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


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


def _contains_absolute_path(value: Any) -> bool:
    """! @brief 递归判断 JSON 结构中是否包含绝对路径字符串。"""
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False
