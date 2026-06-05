"""! @file citation_format.py
@brief 将检索命中转换为契约层 Citation 列表。
"""

from __future__ import annotations

from rag_core.contracts.models import Citation, SearchHit


def format_citations(contexts: list[SearchHit], quote_max_len: int = 240) -> list[Citation]:
    """! @brief 从 SearchHit 构造标准 Citation，字段与 metadata 对齐。
    @param contexts 检索命中列表。
    @param quote_max_len quote 字段的最大长度。
    @return 契约层 Citation 列表。
    """
    citations: list[Citation] = []
    for hit in sorted(contexts, key=lambda item: item.rank):
        metadata = hit.metadata or {}
        page_numbers = metadata.get("page_numbers") or []
        page_number = page_numbers[0] if page_numbers else None
        section_path = metadata.get("section_path") or []
        block_type = metadata.get("block_type")
        citation_metadata = {}
        if block_type:
            citation_metadata["block_type"] = block_type
        citations.append(
            Citation(
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                page_number=page_number,
                section_path=list(section_path),
                quote=hit.text[:quote_max_len],
                source=hit.source,
                metadata=citation_metadata,
            )
        )
    return citations
