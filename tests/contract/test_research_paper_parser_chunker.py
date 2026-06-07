"""! @file test_research_paper_parser_chunker.py
@brief 研究论文 parser/chunker 契约测试。
"""

from pathlib import Path

from rag_core.chunkers import ResearchPaperChunker
from rag_core.contracts.enums import BlockType
from rag_core.contracts.models import Chunk, ParsedDocument
from rag_core.parsers import MarkdownPaperParser


def test_markdown_paper_parser_outputs_contract_document():
    """! @brief Markdown 论文 fixture 应解析为 ParsedDocument。"""
    document = MarkdownPaperParser().parse("sample_data/papers/demo_research_paper.md")

    assert isinstance(document, ParsedDocument)
    assert document.blocks
    assert document.parser_name == "markdown-paper-parser"
    assert any(block.block_type == BlockType.TITLE for block in document.blocks)
    assert any(block.block_type == BlockType.TABLE for block in document.blocks)
    assert any(block.block_type == BlockType.CAPTION for block in document.blocks)
    assert "answer_quality" not in document.model_dump_json()


def test_research_paper_chunker_preserves_metadata():
    """! @brief 论文 chunk 必须保留页码、章节和 block_type metadata。"""
    document = MarkdownPaperParser().parse("sample_data/papers/demo_research_paper.md")
    chunks = ResearchPaperChunker(max_chars=600).chunk(document)

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks
    assert any(chunk.metadata["page_numbers"] for chunk in chunks)
    assert any(chunk.metadata["section_path"] for chunk in chunks)
    assert any(chunk.metadata["block_type"] == "table" for chunk in chunks)
    assert "answer_quality" not in "\n".join(chunk.model_dump_json() for chunk in chunks)


def test_markdown_paper_parser_rejects_missing_file(tmp_path):
    """! @brief 缺失论文文件应抛出 EmptyCorpus 类错误。"""
    missing_path = tmp_path / "missing.md"

    try:
        MarkdownPaperParser().parse(str(missing_path))
    except Exception as exc:
        assert "论文文件不存在" in str(exc)
    else:
        raise AssertionError("缺失文件不应被解析成功")
