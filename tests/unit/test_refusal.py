"""! @file test_refusal.py
@brief 缺证据拒答策略单元测试。
"""

from rag_core.contracts.enums import RagMode
from rag_core.contracts.models import SearchHit
from rag_core.generation.refusal import build_refusal_answer, evaluate_evidence


def test_evaluate_evidence_refuses_empty_contexts_in_basic_rag():
    """! @brief basic_rag 在无检索上下文时应拒答。"""
    decision = evaluate_evidence([], RagMode.BASIC_RAG)
    assert decision.should_refuse is True
    assert "没有检索" in decision.reason


def test_evaluate_evidence_refuses_low_score_in_optimized_rag():
    """! @brief optimized_rag 在相关性过低时应拒答。"""
    contexts = [
        SearchHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            text="弱相关证据",
            score=0.01,
            rank=1,
            source="source",
            metadata={"section_path": ["S"], "block_type": "text"},
        )
    ]
    decision = evaluate_evidence(contexts, RagMode.OPTIMIZED_RAG, min_score=0.05)
    assert decision.should_refuse is True


def test_build_refusal_answer_returns_markdown_and_warning():
    """! @brief 拒答响应应包含 Markdown 标题与 warning。"""
    markdown, warnings = build_refusal_answer("没有检索到相关证据")
    assert "无法生成有证据支撑的回答" in markdown
    assert warnings == ["没有检索到相关证据"]
