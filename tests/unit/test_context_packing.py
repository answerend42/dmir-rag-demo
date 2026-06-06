"""! @file test_context_packing.py
@brief context packing 单元测试。
"""

from rag_core.contracts.models import SearchHit
from rag_core.retrieval.context_packing import pack_contexts


def test_pack_contexts_includes_course_qa_and_paper_metadata():
    """! @brief 打包结果应同时携带课程 QA 与论文 metadata。"""
    contexts = [
        SearchHit(
            chunk_id="chunk-qa",
            doc_id="doc-qa",
            text="课程主题：NLP\n问题：什么是 NLP？\n候选答案：NLP 是语言计算。",
            score=1.2,
            rank=1,
            source="sample_data/course_qa_public.json",
            metadata={"page_numbers": [], "section_path": ["课程 QA"], "block_type": "text"},
        ),
        SearchHit(
            chunk_id="chunk-paper",
            doc_id="doc-paper",
            text="UCOSA improves retrieval quality.",
            score=0.9,
            rank=2,
            source="paper.pdf",
            metadata={
                "page_numbers": [4],
                "section_path": ["Method", "UCOSA"],
                "block_type": "table",
            },
        ),
    ]

    packed = pack_contexts(contexts)

    assert len(packed) == 2
    assert packed[0].index == 1
    assert "课程 QA" in packed[0].formatted_text
    assert "页码：4" in packed[1].formatted_text
    assert "Method > UCOSA" in packed[1].formatted_text
    assert "类型：table" in packed[1].formatted_text


def test_pack_contexts_respects_max_chars_budget():
    """! @brief 超过字符预算时应截断后续证据。"""
    contexts = [
        SearchHit(
            chunk_id=f"chunk-{index}",
            doc_id="doc",
            text="长证据" * 200,
            score=1.0 - index * 0.01,
            rank=index,
            source="source",
            metadata={"section_path": ["S"], "block_type": "text"},
        )
        for index in range(1, 4)
    ]

    packed = pack_contexts(contexts, max_chars=500)

    assert len(packed) >= 1
    assert len(packed) < len(contexts)
