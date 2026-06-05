"""! @file refusal.py
@brief optimized/basic 模式下的缺证据拒答策略。
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_core.contracts.enums import RagMode
from rag_core.contracts.models import SearchHit

DEFAULT_MIN_SCORE = 0.05


@dataclass(frozen=True)
class EvidenceDecision:
    """! @brief 对当前检索证据是否足以生成回答的判断结果。"""

    should_refuse: bool
    reason: str


def evaluate_evidence(
    contexts: list[SearchHit],
    rag_mode: RagMode,
    min_score: float = DEFAULT_MIN_SCORE,
) -> EvidenceDecision:
    """! @brief 判断当前证据是否允许进入 grounded 生成。
    @param contexts 检索命中列表。
    @param rag_mode 当前 RAG 模式。
    @param min_score optimized 模式下允许回答的最低相关性阈值。
    @return 是否拒答及原因。
    """
    if rag_mode == RagMode.LLM_ONLY:
        return EvidenceDecision(should_refuse=False, reason="")

    if not contexts:
        return EvidenceDecision(should_refuse=True, reason="没有检索到相关证据")

    best_score = max(hit.score for hit in contexts)
    if rag_mode == RagMode.OPTIMIZED_RAG and best_score < min_score:
        return EvidenceDecision(
            should_refuse=True,
            reason=f"最高相关性 {best_score:.3f} 低于阈值 {min_score:.3f}",
        )

    return EvidenceDecision(should_refuse=False, reason="")


def build_refusal_answer(reason: str) -> tuple[str, list[str]]:
    """! @brief 构造缺证据拒答 Markdown 与 warnings。
    @param reason 拒答原因。
    @return answer_markdown 与 warnings 列表。
    """
    markdown = (
        "## 无法生成有证据支撑的回答\n\n"
        "现有检索证据不足以支持可靠回答，因此系统拒绝作答。\n\n"
        f"原因：{reason}"
    )
    return markdown, [reason or "证据不足，已拒答"]
