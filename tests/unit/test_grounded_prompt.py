"""! @file test_grounded_prompt.py
@brief grounded prompt 单元测试。
"""

from rag_core.contracts.enums import RagMode
from rag_core.generation.grounded_prompt import build_grounded_prompt, build_llm_only_prompt
from rag_core.retrieval.context_packing import PackedContext


def test_build_grounded_prompt_uses_chinese_rules_and_shared_template():
    """! @brief grounded prompt 应包含中文约束与共用证据块。"""
    packed = [
        PackedContext(
            index=1,
            chunk_id="chunk-1",
            doc_id="doc-1",
            formatted_text="[证据1] 来源：paper.pdf | 页码：4 | 章节：Method | 类型：text\n正文",
            source="paper.pdf",
            metadata={"block_type": "text"},
        )
    ]
    prompt = build_grounded_prompt("什么是 UCOSA？", packed, RagMode.OPTIMIZED_RAG)

    assert "只能基于“检索证据”回答" in prompt
    assert "中文 Markdown" in prompt
    assert "什么是 UCOSA？" in prompt
    assert "[证据1]" in prompt


def test_build_llm_only_prompt_mentions_no_retrieval():
    """! @brief llm_only prompt 应说明未使用检索证据。"""
    prompt = build_llm_only_prompt("什么是 NLP？")
    assert "未使用检索证据" in prompt
    assert "什么是 NLP？" in prompt
