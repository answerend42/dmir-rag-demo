"""! @file test_citation_format.py
@brief citation 格式化单元测试。
"""

from rag_core.contracts.models import SearchHit
from rag_core.generation.citation_format import format_citations


def test_format_citations_maps_search_hits_to_contract_fields():
    """! @brief Citation 字段应与 SearchHit metadata 对齐。"""
    contexts = [
        SearchHit(
            chunk_id="chunk-paper",
            doc_id="paper-ucosa",
            text="UCOSA improves retrieval quality on long documents.",
            score=0.88,
            rank=1,
            source="paper.pdf",
            metadata={
                "page_numbers": [4],
                "section_path": ["Method", "UCOSA"],
                "block_type": "caption",
            },
        )
    ]

    citations = format_citations(contexts, quote_max_len=30)

    assert len(citations) == 1
    citation = citations[0]
    assert citation.doc_id == "paper-ucosa"
    assert citation.chunk_id == "chunk-paper"
    assert citation.page_number == 4
    assert citation.section_path == ["Method", "UCOSA"]
    assert citation.source == "paper.pdf"
    assert citation.metadata["block_type"] == "caption"
    assert len(citation.quote) <= 30
