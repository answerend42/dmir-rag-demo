"""! @file test_query_rewrite.py
@brief query rewrite 单元测试。
"""

from rag_core.contracts.enums import RagMode
from rag_core.retrieval.query_rewrite import rewrite_query


def test_rewrite_query_passthrough_for_basic_rag():
    """! @brief basic_rag 模式不应改写 query。"""
    result = rewrite_query("请问 什么是自然语言处理？", RagMode.BASIC_RAG)
    assert result.was_rewritten is False
    assert result.rewritten_query == "请问 什么是自然语言处理？"


def test_rewrite_query_expands_definition_question_in_optimized_mode():
    """! @brief optimized 模式应扩展定义型问题。"""
    result = rewrite_query("请问什么是自然语言处理？", RagMode.OPTIMIZED_RAG)
    assert result.was_rewritten is True
    assert "自然语言处理" in result.rewritten_query
    assert "定义" in result.rewritten_query
