#!/usr/bin/env python3
"""! @file parse_paper_sample.py
@brief 用真实 parser/chunker 解析论文 corpus 并输出 chunk 摘要。
@details 面向 Issue #8 集成同学的最小调用入口：不依赖真实 embedding/LLM，
仅展示 [`MarkdownPaperParser`](../backend/rag_core/parsers/research_paper.py) →
[`ResearchPaperChunker`](../backend/rag_core/chunkers/research_paper.py)
的端到端调用方式与契约字段，便于上层据此对接前端展示与评测。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag_core.chunkers import ResearchPaperChunker
from rag_core.contracts.errors import EmptyCorpus, ProviderUnavailable
from rag_core.parsers import MarkdownPaperParser


DEFAULT_PAPER = "sample_data/papers/llm_wiki_retrieval_as_reasoning.md"


def parse_args() -> argparse.Namespace:
    """! @brief 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="解析论文 corpus 并输出 chunk 摘要。")
    parser.add_argument("--paper", default=DEFAULT_PAPER, help="论文 Markdown 路径，默认 LLM-Wiki digest。")
    parser.add_argument("--max-chars", type=int, default=1200, help="ResearchPaperChunker 的近似字符上限。")
    parser.add_argument("--limit", type=int, default=5, help="预览 chunks 的条数。")
    parser.add_argument("--pretty", action="store_true", help="格式化输出 JSON。")
    return parser.parse_args()


def _summarise_chunk(chunk, head_chars: int = 80) -> dict:
    """! @brief 抽取 chunk 摘要展示字段，避免在 stdout 打印全文。"""
    text = chunk.text or ""
    return {
        "chunk_id": chunk.chunk_id,
        "section_path": chunk.metadata.get("section_path", []),
        "block_type": chunk.metadata.get("block_type", ""),
        "page_numbers": chunk.metadata.get("page_numbers", []),
        "token_count": chunk.token_count,
        "text_head": text[:head_chars] + ("..." if len(text) > head_chars else ""),
    }


def main() -> int:
    """! @brief 解析论文、分块、输出摘要 JSON。
    @return 解析与分块成功返回 0；遇 EmptyCorpus / ProviderUnavailable 返回 1。
    """
    args = parse_args()

    try:
        document = MarkdownPaperParser().parse(args.paper)
        chunks = ResearchPaperChunker(max_chars=args.max_chars).chunk(document)
    except (EmptyCorpus, ProviderUnavailable) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    block_type_counts = Counter(block.block_type.value for block in document.blocks)
    chunk_block_type_counts = Counter(chunk.metadata.get("block_type", "") for chunk in chunks)

    payload = {
        "paper": args.paper,
        "doc_id": document.doc_id,
        "title": document.title,
        "parser_name": document.parser_name,
        "block_count": len(document.blocks),
        "chunk_count": len(chunks),
        "block_type_counts": dict(block_type_counts),
        "chunk_block_type_counts": dict(chunk_block_type_counts),
        "chunks_preview": [_summarise_chunk(chunk) for chunk in chunks[: max(0, args.limit)]],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
