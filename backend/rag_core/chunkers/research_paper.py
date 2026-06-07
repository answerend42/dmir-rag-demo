"""! @file research_paper.py
@brief 保留论文页码、章节、表格和图注信息的分块器。
"""

from __future__ import annotations

import re

from rag_core.contracts.enums import BlockType
from rag_core.contracts.errors import EmptyCorpus
from rag_core.contracts.models import Chunk, ContentBlock, ParsedDocument
from rag_core.testing.fakes import stable_id
from rag_core.vector_indexes.metadata import strip_forbidden_metadata

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class ResearchPaperChunker:
    """! @brief 将 ParsedDocument 转换为论文检索 chunk。"""

    name = "research-paper-chunker"

    def __init__(self, max_chars: int = 1200):
        """! @brief 初始化论文分块器。
        @param max_chars 单个 chunk 的近似最大字符数。
        @throws ValueError max_chars 不是正整数时抛出。
        """
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正整数")
        self.max_chars = max_chars

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """! @brief 按论文结构生成检索 chunk。
        @param document 解析器输出的 ParsedDocument。
        @return 保留 page_numbers、section_path、block_type 的 Chunk 列表。
        @throws EmptyCorpus 文档没有可分块文本时抛出。
        """
        chunks: list[Chunk] = []
        text_buffer: list[ContentBlock] = []

        for block in document.blocks:
            if not block.text.strip():
                continue
            if block.block_type in {BlockType.TABLE, BlockType.CAPTION, BlockType.FIGURE, BlockType.FORMULA}:
                self._flush_text_buffer(chunks, document, text_buffer)
                text_buffer = []
                chunks.append(self._make_chunk(document, [block]))
                continue
            if block.block_type == BlockType.TITLE:
                self._flush_text_buffer(chunks, document, text_buffer)
                text_buffer = [block]
                continue

            candidate_text = "\n".join(item.text for item in [*text_buffer, block])
            if text_buffer and len(candidate_text) > self.max_chars:
                self._flush_text_buffer(chunks, document, text_buffer)
                text_buffer = [block]
            else:
                text_buffer.append(block)

        self._flush_text_buffer(chunks, document, text_buffer)
        if not chunks:
            raise EmptyCorpus("论文分块器没有生成任何 chunk")
        return chunks

    def _flush_text_buffer(
        self,
        chunks: list[Chunk],
        document: ParsedDocument,
        text_buffer: list[ContentBlock],
    ) -> None:
        """! @brief 将普通文本缓存写入 chunks。"""
        payload = [block for block in text_buffer if block.text.strip()]
        if payload:
            chunks.append(self._make_chunk(document, payload))

    def _make_chunk(self, document: ParsedDocument, blocks: list[ContentBlock]) -> Chunk:
        """! @brief 根据一个或多个内容块构造契约 Chunk。"""
        text = "\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()
        block_ids = [block.block_id for block in blocks]
        block_types = [block.block_type for block in blocks]
        page_numbers = sorted({block.page_number for block in blocks if block.page_number is not None})
        section_path = _choose_section_path(blocks)
        metadata = strip_forbidden_metadata(
            {
                "page_numbers": page_numbers,
                "section_path": section_path,
                "block_type": block_types[-1].value if block_types else BlockType.TEXT.value,
                "parser_name": document.parser_name,
                "source": document.metadata.get("source", document.title or document.doc_id),
                "line_start": min(_line_value(block, "line_start") for block in blocks),
                "line_end": max(_line_value(block, "line_end") for block in blocks),
            }
        )
        return Chunk(
            chunk_id=stable_id("paper-chunk", document.doc_id, *block_ids),
            doc_id=document.doc_id,
            text=text,
            source=str(document.metadata.get("source", document.title or document.doc_id)),
            block_ids=block_ids,
            block_types=block_types,
            token_count=max(1, len(_TOKEN_PATTERN.findall(text))),
            metadata=metadata,
        )


def _choose_section_path(blocks: list[ContentBlock]) -> list[str]:
    """! @brief 选取最后一个非空 section_path 作为 chunk 章节路径。"""
    for block in reversed(blocks):
        section_path = block.metadata.get("section_path")
        if isinstance(section_path, list) and section_path:
            return [str(part) for part in section_path]
    return []


def _line_value(block: ContentBlock, key: str) -> int:
    """! @brief 从 block metadata 读取行号，缺失时返回 0。"""
    value = block.metadata.get(key)
    return int(value) if isinstance(value, int) else 0
