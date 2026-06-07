"""! @file test_run_eval.py
@brief run_eval.py 离线评测脚本测试。
"""

from __future__ import annotations

import json
import subprocess
import sys


def test_course_qa_eval_generates_frontend_json_without_hidden_label(tmp_path):
    """! @brief 课程 QA eval 输出前端可读 JSON，且不包含隐藏标签字段。"""
    subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--dataset-type",
            "course_qa",
            "--modes",
            "all",
            "--limit",
            "2",
            "--output-dir",
            str(tmp_path),
            "--pretty",
        ],
        check=True,
    )
    payload = json.loads((tmp_path / "course_qa_eval.json").read_text(encoding="utf-8"))

    assert set(payload["summary"]) == {"llm_only", "basic_rag", "optimized_rag"}
    assert payload["summary"]["basic_rag"]["cited"] >= 1
    assert "answer_quality" not in json.dumps(payload, ensure_ascii=False)


def test_paper_eval_fixture_runs(tmp_path):
    """! @brief 论文 fixture eval 应能离线运行。"""
    subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--dataset-type",
            "paper",
            "--modes",
            "basic_rag",
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    payload = json.loads((tmp_path / "paper_eval.json").read_text(encoding="utf-8"))

    assert payload["dataset_type"] == "paper"
    assert payload["summary"]["basic_rag"]["total"] == 1
