"""! @file __init__.py
@brief grounded prompt、citation 格式化与拒答策略。
"""

from rag_core.generation.citation_format import format_citations
from rag_core.generation.grounded_prompt import build_grounded_prompt, build_llm_only_prompt
from rag_core.generation.refusal import EvidenceDecision, build_refusal_answer, evaluate_evidence

__all__ = [
    "EvidenceDecision",
    "build_grounded_prompt",
    "build_llm_only_prompt",
    "build_refusal_answer",
    "evaluate_evidence",
    "format_citations",
]
