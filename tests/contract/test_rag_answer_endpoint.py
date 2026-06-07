"""! @file test_rag_answer_endpoint.py
@brief #8 阶段 A `/rag/answer` 集成主链路契约测试。
"""

from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient

from rag_core.contracts import RagAnswer


def _client() -> TestClient:
    """! @brief 重新导入 FastAPI app，避免其他测试污染全局状态。"""
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    return TestClient(module.app)


def test_rag_answer_returns_course_qa_rag_answer():
    """! @brief 验证 `/rag/answer` 返回可供前端和评测消费的 RagAnswer。"""
    response = _client().post(
        "/rag/answer",
        json={
            "query": "什么是自然语言处理？",
            "rag_mode": "basic_rag",
            "top_k": 3,
            "collection_id": "course-qa-default",
            "provider": "mock",
            "model": "mock-generator",
            "require_citations": True,
            "metadata": {"max_questions": 5},
        },
    )

    assert response.status_code == 200
    answer = RagAnswer.model_validate(response.json())
    assert answer.retrieved_hits
    assert answer.citations
    assert answer.metadata["integration_spine"] == "course-qa-fake-spine"
    assert answer.metadata["dataset_path"] == "sample_data/course_qa_public.json"
    assert "answer_quality" not in answer.model_dump_json()


def test_rag_answer_llm_only_uses_same_contract_without_search():
    """! @brief 验证 llm_only 走同一接口但不返回检索命中。"""
    response = _client().post(
        "/rag/answer",
        json={
            "query": "什么是自然语言处理？",
            "rag_mode": "llm_only",
            "top_k": 3,
            "collection_id": "course-qa-default",
            "provider": "mock",
            "model": "mock-generator",
            "require_citations": True,
            "metadata": {"max_questions": 5},
        },
    )

    assert response.status_code == 200
    answer = RagAnswer.model_validate(response.json())
    assert not answer.retrieved_hits
    assert not answer.citations
    assert "纯模型模式" in answer.warnings[0]
    assert "search" not in [stage.stage_name for stage in answer.trace]


def test_rag_answer_rejects_hidden_quality_label_in_request():
    """! @brief 保护隐藏评测标签不能进入 `/rag/answer` 请求。"""
    response = _client().post(
        "/rag/answer",
        json={
            "query": "什么是自然语言处理？",
            "provider": "mock",
            "model": "mock-generator",
            "metadata": {"answer_quality": 9},
        },
    )

    assert response.status_code == 400
    assert "answer_quality" in response.json()["detail"]


def test_rag_answer_rejects_real_provider_until_adapters_land():
    """! @brief 阶段 A 不伪装真实 provider 已接通。"""
    response = _client().post(
        "/rag/answer",
        json={
            "query": "什么是自然语言处理？",
            "provider": "qwen_api",
            "model": "qwen-plus",
            "metadata": {},
        },
    )

    assert response.status_code == 400
    assert "provider=mock" in response.json()["detail"]


def test_eval_result_endpoint_serves_frontend_summary():
    """! @brief 前端 dashboard 可读取 run_eval.py 生成的课程 QA 评测摘要。"""
    response = _client().get("/eval/results/course_qa_eval.json")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["summary"]) == {"llm_only", "basic_rag", "optimized_rag"}
    assert "answer_quality" not in response.text


def test_eval_result_endpoint_rejects_path_traversal():
    """! @brief 评测结果端点必须拒绝路径穿越文件名。"""
    response = _client().get("/eval/results/%2E%2E%2Flabels%2Fcourse_qa_quality_labels.json")

    assert response.status_code in {400, 404}
