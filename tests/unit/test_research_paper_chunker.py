"""! @file test_research_paper_chunker.py
@brief 论文分块器单元测试。
"""

from rag_core.chunkers import ResearchPaperChunker
from rag_core.parsers import parse_markdown_paper_text


def test_research_paper_chunker_splits_long_sections():
    """! @brief max_chars 较小时，分块器应切分长章节。"""
    document = parse_markdown_paper_text(
        text="# Title\n\n## Method\n\n第一段很长。" * 20,
        title="unit-paper",
        source="unit",
    )

    chunks = ResearchPaperChunker(max_chars=80).chunk(document)

    assert len(chunks) >= 2
    assert all(chunk.metadata["section_path"] for chunk in chunks)
