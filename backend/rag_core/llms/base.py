"""! @file base.py
@brief Qwen generator 共享生成逻辑与 trace 组装。
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from rag_core.contracts.enums import RagMode
from rag_core.contracts.models import RagAnswer, RagRequest, SearchHit, StageTrace
from rag_core.generation.citation_format import format_citations
from rag_core.generation.grounded_prompt import build_grounded_prompt, build_llm_only_prompt
from rag_core.generation.refusal import DEFAULT_MIN_SCORE, build_refusal_answer, evaluate_evidence
from rag_core.retrieval.context_packing import pack_contexts
from rag_core.retrieval.query_rewrite import rewrite_query

CompleteFn = Callable[[str], str]


class BaseQwenGenerator:
    """! @brief 基于可注入 complete 函数的 Qwen 生成器基类。"""

    name: str

    def __init__(
        self,
        name: str,
        complete_fn: CompleteFn,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        provider: str,
        model: str,
    ):
        self.name = name
        self._complete = complete_fn
        self.min_score = min_score
        self.provider = provider
        self.model = model

    def generate(self, request: RagRequest, contexts: list[SearchHit]) -> RagAnswer:
        """! @brief 根据 RagRequest 与检索上下文生成 RagAnswer。
        @param request 统一问答请求。
        @param contexts 检索命中列表；llm_only 模式下会被忽略。
        @return 契约层 RagAnswer。
        """
        traces: list[StageTrace] = []
        warnings: list[str] = []
        effective_query = request.query

        if request.rag_mode == RagMode.LLM_ONLY:
            start = perf_counter()
            prompt = build_llm_only_prompt(request.query)
            answer_markdown = self._complete(prompt)
            warnings.append("纯模型模式没有检索证据")
            traces.append(
                self._trace(
                    "generate",
                    start,
                    {"query": request.query, "context_count": 0, "mode": request.rag_mode.value},
                    {"provider": self.provider, "model": self.model},
                )
            )
            return self._build_answer(
                request=request,
                contexts=[],
                answer_markdown=answer_markdown,
                citations=[],
                warnings=warnings,
                traces=traces,
                generator_name=self.name,
            )

        decision = evaluate_evidence(contexts, request.rag_mode, self.min_score)
        if decision.should_refuse:
            start = perf_counter()
            answer_markdown, refusal_warnings = build_refusal_answer(decision.reason)
            warnings.extend(refusal_warnings)
            traces.append(
                self._trace(
                    "generate",
                    start,
                    {"query": request.query, "context_count": len(contexts), "refused": True},
                    {"reason": decision.reason},
                )
            )
            return self._build_answer(
                request=request,
                contexts=contexts,
                answer_markdown=answer_markdown,
                citations=[],
                warnings=warnings,
                traces=traces,
                generator_name=self.name,
            )

        if request.rag_mode == RagMode.OPTIMIZED_RAG:
            start = perf_counter()
            rewrite_result = rewrite_query(request.query, request.rag_mode)
            effective_query = rewrite_result.rewritten_query
            traces.append(
                self._trace(
                    "query_rewrite",
                    start,
                    {"query": request.query},
                    {
                        "rewritten_query": rewrite_result.rewritten_query,
                        "was_rewritten": rewrite_result.was_rewritten,
                        "notes": rewrite_result.rewrite_notes,
                    },
                )
            )

        start = perf_counter()
        packed_contexts = pack_contexts(contexts)
        traces.append(
            self._trace(
                "context_pack",
                start,
                {"context_count": len(contexts)},
                {"packed_count": len(packed_contexts)},
            )
        )

        start = perf_counter()
        prompt = build_grounded_prompt(effective_query, packed_contexts, request.rag_mode)
        answer_markdown = self._complete(prompt)
        citations = format_citations(contexts) if request.require_citations else []
        traces.append(
            self._trace(
                "generate",
                start,
                {
                    "query": effective_query,
                    "context_count": len(contexts),
                    "mode": request.rag_mode.value,
                },
                {
                    "provider": self.provider,
                    "model": self.model,
                    "citation_count": len(citations),
                },
            )
        )

        return self._build_answer(
            request=request,
            contexts=contexts,
            answer_markdown=answer_markdown,
            citations=citations,
            warnings=warnings,
            traces=traces,
            generator_name=self.name,
        )

    @staticmethod
    def _build_answer(
        *,
        request: RagRequest,
        contexts: list[SearchHit],
        answer_markdown: str,
        citations: list,
        warnings: list[str],
        traces: list[StageTrace],
        generator_name: str,
    ) -> RagAnswer:
        """! @brief 组装最终 RagAnswer。"""
        return RagAnswer(
            answer_markdown=answer_markdown,
            citations=citations,
            retrieved_hits=contexts,
            warnings=warnings,
            trace=traces,
            metadata={
                "generator": generator_name,
                "provider": request.provider,
                "model": request.model,
                "rag_mode": request.rag_mode.value,
            },
        )

    @staticmethod
    def _trace(
        stage_name: str,
        started_at: float,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
    ) -> StageTrace:
        """! @brief 构造带毫秒耗时的 StageTrace。"""
        return StageTrace(
            stage_name=stage_name,
            latency_ms=max(0.0, (perf_counter() - started_at) * 1000),
            input_summary=input_summary,
            output_summary=output_summary,
        )
