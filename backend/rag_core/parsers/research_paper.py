"""! @file research_paper.py
@brief 研究论文 Markdown/PDF 解析器。
@details 阶段 A 使用 Markdown fixture 离线验证；PDF 解析通过 PyMuPDF 懒加载，
缺少依赖时抛出 ProviderUnavailable，不阻塞课程 QA 默认链路。
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_core.contracts.enums import BlockType
from rag_core.contracts.errors import EmptyCorpus, ProviderUnavailable
from rag_core.contracts.models import ContentBlock, ParsedDocument
from rag_core.testing.fakes import stable_id

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE_PATTERN = re.compile(r"^\[page\s+(\d+)\]\s*$", re.IGNORECASE)
_TABLE_PATTERN = re.compile(r"^\s*\|.+\|\s*$")
_FIGURE_CAPTION_PATTERN = re.compile(r"^(图|Figure|Fig\.)\s*[\w\d.-]*[:：].+", re.IGNORECASE)
_TABLE_CAPTION_PATTERN = re.compile(r"^(表|Table)\s*[\w\d.-]*[:：].+", re.IGNORECASE)


class MarkdownPaperParser:
    """! @brief 将论文 Markdown 或轻量 PDF 转为 ParsedDocument。"""

    name = "markdown-paper-parser"

    def parse(self, file_path: str) -> ParsedDocument:
        """! @brief 根据文件扩展名解析论文文件。
        @param file_path Markdown、文本或 PDF 文件路径。
        @return 契约层 ParsedDocument。
        @throws EmptyCorpus 文件不存在或内容为空时抛出。
        @throws ProviderUnavailable PDF 解析依赖不可用时抛出。
        """
        path = Path(file_path)
        if not path.exists():
            raise EmptyCorpus(f"论文文件不存在: {file_path}")
        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path)
        text = path.read_text(encoding="utf-8")
        return parse_markdown_paper_text(text=text, title=path.stem, source=str(path))

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        """! @brief 使用 PyMuPDF 将 PDF 页面文本转成 Markdown 风格内容。"""
        try:
            import fitz
        except ImportError as exc:
            raise ProviderUnavailable("PDF 解析需要安装 PyMuPDF；Markdown fixture 可直接运行") from exc

        page_texts: list[str] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    page_texts.append(f"[page {page_index}]\n{page_text}")
        return parse_markdown_paper_text(text="\n\n".join(page_texts), title=path.stem, source=str(path))


def parse_markdown_paper_text(text: str, title: str = "research-paper", source: str = "memory") -> ParsedDocument:
    """! @brief 解析论文 Markdown 文本。
    @param text Markdown 或纯文本。
    @param title 文档标题。
    @param source 数据来源。
    @return 契约层 ParsedDocument。
    @throws EmptyCorpus 文本为空时抛出。
    """
    markdown = text.strip()
    if not markdown:
        raise EmptyCorpus("论文 Markdown 内容为空")

    doc_id = stable_id("paper", title, markdown)
    blocks = _extract_blocks(markdown=markdown, doc_id=doc_id, source=source)
    if not blocks:
        raise EmptyCorpus("论文解析后没有可用内容块")

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        markdown=markdown,
        blocks=blocks,
        parser_name=MarkdownPaperParser.name,
        metadata={
            "source": source,
            "parser": MarkdownPaperParser.name,
            "block_count": len(blocks),
        },
    )


def _extract_blocks(markdown: str, doc_id: str, source: str) -> list[ContentBlock]:
    """! @brief 将 Markdown 拆成带 section/page metadata 的内容块。"""
    blocks: list[ContentBlock] = []
    section_path: list[str] = []
    current_page: int | None = None
    buffer: list[str] = []
    buffer_start_line = 1

    for line_number, raw_line in enumerate(markdown.splitlines(), start=1):
        line = raw_line.rstrip()
        page_match = _PAGE_PATTERN.match(line.strip())
        heading_match = _HEADING_PATTERN.match(line)

        if page_match:
            _flush_text_block(blocks, buffer, doc_id, source, section_path, current_page, buffer_start_line, line_number - 1)
            buffer = []
            current_page = int(page_match.group(1))
            buffer_start_line = line_number + 1
            continue

        if heading_match:
            _flush_text_block(blocks, buffer, doc_id, source, section_path, current_page, buffer_start_line, line_number - 1)
            buffer = []
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            section_path = section_path[: max(0, level - 1)] + [heading]
            blocks.append(
                _make_block(
                    doc_id=doc_id,
                    index=len(blocks) + 1,
                    block_type=BlockType.TITLE,
                    text=heading,
                    source=source,
                    page_number=current_page,
                    section_path=list(section_path),
                    line_start=line_number,
                    line_end=line_number,
                )
            )
            buffer_start_line = line_number + 1
            continue

        block_type = _special_block_type(line)
        if block_type is not None:
            _flush_text_block(blocks, buffer, doc_id, source, section_path, current_page, buffer_start_line, line_number - 1)
            buffer = []
            blocks.append(
                _make_block(
                    doc_id=doc_id,
                    index=len(blocks) + 1,
                    block_type=block_type,
                    text=line.strip(),
                    source=source,
                    page_number=current_page,
                    section_path=list(section_path),
                    line_start=line_number,
                    line_end=line_number,
                )
            )
            buffer_start_line = line_number + 1
            continue

        if line.strip():
            if not buffer:
                buffer_start_line = line_number
            buffer.append(line.strip())
        else:
            _flush_text_block(blocks, buffer, doc_id, source, section_path, current_page, buffer_start_line, line_number - 1)
            buffer = []
            buffer_start_line = line_number + 1

    _flush_text_block(blocks, buffer, doc_id, source, section_path, current_page, buffer_start_line, len(markdown.splitlines()))
    return blocks


def _flush_text_block(
    blocks: list[ContentBlock],
    buffer: list[str],
    doc_id: str,
    source: str,
    section_path: list[str],
    page_number: int | None,
    line_start: int,
    line_end: int,
) -> None:
    """! @brief 将段落缓存写入普通文本块。"""
    text = " ".join(part.strip() for part in buffer if part.strip()).strip()
    if not text:
        return
    blocks.append(
        _make_block(
            doc_id=doc_id,
            index=len(blocks) + 1,
            block_type=BlockType.TEXT,
            text=text,
            source=source,
            page_number=page_number,
            section_path=list(section_path),
            line_start=line_start,
            line_end=max(line_start, line_end),
        )
    )


def _special_block_type(line: str) -> BlockType | None:
    """! @brief 识别表格行、图注和表注。"""
    stripped = line.strip()
    if not stripped:
        return None
    if _FIGURE_CAPTION_PATTERN.match(stripped) or _TABLE_CAPTION_PATTERN.match(stripped):
        return BlockType.CAPTION
    if _TABLE_PATTERN.match(stripped):
        return BlockType.TABLE
    return None


def _make_block(
    *,
    doc_id: str,
    index: int,
    block_type: BlockType,
    text: str,
    source: str,
    page_number: int | None,
    section_path: list[str],
    line_start: int,
    line_end: int,
) -> ContentBlock:
    """! @brief 构造带稳定 ID 和论文元数据的内容块。"""
    return ContentBlock(
        block_id=stable_id("paper-block", doc_id, index, block_type.value, text),
        block_type=block_type,
        text=text,
        page_number=page_number,
        metadata={
            "source": source,
            "section_path": section_path,
            "line_start": line_start,
            "line_end": line_end,
        },
    )
