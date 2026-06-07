"""! @file test_research_paper_parser_chunker.py
@brief 研究论文 parser/chunker 契约测试。
"""

import json
from pathlib import Path

import pytest

from rag_core.contracts.errors import EmptyCorpus
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


def test_llm_wiki_paper_fixture_is_final_stage_b_input():
    """! @brief 默认论文 fixture 应指向真实目标论文并包含 20-30 条 QA。"""
    fixture_path = Path("sample_data/papers/paper_eval_fixture.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    target = fixture["target_paper"]
    questions = fixture["questions"]
    document = MarkdownPaperParser().parse(target["source_path"])
    chunks = ResearchPaperChunker(max_chars=1200).chunk(document)

    assert target["paper_id"] == "llm-wiki-2605.25480"
    assert target["source_path"] == "sample_data/papers/llm_wiki_retrieval_as_reasoning.md"
    assert 20 <= len(questions) <= 30
    assert len(fixture["distractor_papers"]) >= 2
    assert any(15 in chunk.metadata["page_numbers"] for chunk in chunks)
    assert any(chunk.metadata["block_type"] == "table" for chunk in chunks)
    assert all(row["expected_evidence"] for row in questions)
    assert "answer_quality" not in fixture_path.read_text(encoding="utf-8")


def test_markdown_paper_parser_rejects_missing_file(tmp_path):
    """! @brief 缺失论文文件应抛出 EmptyCorpus 类错误。"""
    missing_path = tmp_path / "missing.md"

    with pytest.raises(EmptyCorpus) as exc_info:
        MarkdownPaperParser().parse(str(missing_path))

    assert "论文文件不存在" in str(exc_info.value)
