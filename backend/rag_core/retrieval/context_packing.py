"""! @file context_packing.py
@brief 将 SearchHit 列表打包为 prompt 可用的证据块。
@details 同时兼容课程 QA 与论文 chunk 的 metadata 字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_core.contracts.models import SearchHit


@dataclass(frozen=True)
class PackedContext:
    """! @brief 单条已打包证据，供 grounded prompt 引用。"""

    index: int
    chunk_id: str
    doc_id: str
    formatted_text: str
    source: str
    metadata: dict[str, Any]


def pack_contexts(contexts: list[SearchHit], max_chars: int = 8000) -> list[PackedContext]:
    """! @brief 按 rank 顺序将检索命中打包为带元数据的证据块。
    @param contexts 检索命中列表。
    @param max_chars 全部证据文本的最大字符预算。
    @return 供 prompt 使用的 PackedContext 列表。
    """
    packed: list[PackedContext] = []
    used_chars = 0

    for hit in sorted(contexts, key=lambda item: item.rank):
        header = _format_header(hit)
        body = hit.text.strip()
        block = f"{header}\n{body}"
        if used_chars + len(block) > max_chars and packed:
            break
        packed.append(
            PackedContext(
                index=len(packed) + 1,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                formatted_text=block,
                source=hit.source,
                metadata=dict(hit.metadata),
            )
        )
        used_chars += len(block)

    return packed


def _format_header(hit: SearchHit) -> str:
    """! @brief 构造单条证据的元数据头部。"""
    metadata = hit.metadata or {}
    page_numbers = metadata.get("page_numbers") or []
    page_label = str(page_numbers[0]) if page_numbers else "未知"
    section_path = metadata.get("section_path") or []
    section_label = " > ".join(str(part) for part in section_path) if section_path else "未知"
    block_type = metadata.get("block_type") or "text"
    score_label = f"{hit.score:.3f}"
    return (
        f"[证据{hit.rank}] 来源：{hit.source} | 相关性：{score_label} | "
        f"页码：{page_label} | 章节：{section_label} | 类型：{block_type}"
    )
