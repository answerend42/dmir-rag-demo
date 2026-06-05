"""! @file query_rewrite.py
@brief optimized RAG 模式下的确定性 query rewrite 纯函数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_core.contracts.enums import RagMode

_FILLER_PATTERN = re.compile(r"^(请问|请告诉我|能否|可以|麻烦)?\s*", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class QueryRewriteResult:
    """! @brief query rewrite 的输出摘要。"""

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    rewrite_notes: list[str]


def rewrite_query(query: str, rag_mode: RagMode) -> QueryRewriteResult:
    """! @brief 在 optimized 模式下对查询做轻量改写，其它模式原样返回。
    @param query 用户原始问题。
    @param rag_mode 当前 RAG 模式。
    @return rewrite 结果，包含是否发生改写及说明。
    """
    normalized = _WHITESPACE_PATTERN.sub(" ", query.strip())
    if normalized.endswith("?"):
        normalized = normalized[:-1] + "？"
    if rag_mode != RagMode.OPTIMIZED_RAG:
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=normalized or query,
            was_rewritten=False,
            rewrite_notes=[],
        )

    rewritten = _FILLER_PATTERN.sub("", normalized).strip() or normalized
    notes: list[str] = []

    if rewritten != normalized:
        notes.append("移除礼貌性前缀")

    if rewritten.endswith("？"):
        core = rewritten[:-1].strip()
        if core.startswith("什么是"):
            topic = core.removeprefix("什么是").strip()
            if topic:
                rewritten = f"请解释“{topic}”的定义、核心目标与典型应用"
                notes.append("将定义型问题扩展为结构化检索查询")
        elif core.startswith("如何") or core.startswith("怎么"):
            rewritten = f"{core}？请结合定义、步骤与关键注意事项回答"
            notes.append("将操作型问题扩展为步骤检索查询")

    was_rewritten = rewritten != normalized or bool(notes)
    return QueryRewriteResult(
        original_query=query,
        rewritten_query=rewritten or normalized or query,
        was_rewritten=was_rewritten,
        rewrite_notes=notes,
    )
