"""! @file test_qwen_generator.py
@brief Qwen generator 契约测试（mock provider，离线）。
"""

from rag_core.contracts import CONTRACT_VERSION, RagMode, RagRequest
from rag_core.contracts.models import SearchHit
from rag_core.llms.qwen_api import QwenApiGenerator
from rag_core.llms.qwen_local import QwenLocalGenerator


def test_qwen_generators_return_contract_compliant_rag_answer():
    """! @brief API/local generator 在 mock 下都应返回契约合规 RagAnswer。"""
    hit = SearchHit(
        chunk_id="chunk-qa",
        doc_id="doc-qa",
        text="自然语言处理是面向人类语言的计算建模与应用技术。",
        score=1.1,
        rank=1,
        source="sample_data/course_qa_public.json",
        metadata={"page_numbers": [], "section_path": ["课程 QA"], "block_type": "text"},
    )
    request = RagRequest(
        query="什么是自然语言处理？",
        rag_mode=RagMode.OPTIMIZED_RAG,
        top_k=3,
        provider="qwen_api",
        model="qwen-turbo",
        require_citations=True,
    )

    api_answer = QwenApiGenerator(
        model="qwen-turbo",
        client=_MockChatClient("## 优化回答\n\nNLP 是语言计算。[证据1]"),
    ).generate(request, [hit])
    local_answer = QwenLocalGenerator(
        model="Qwen/Qwen3-1.7B",
        generate_text=lambda _model, _prompt: "## 优化回答\n\nNLP 是语言计算。[证据1]",
    ).generate(
        request.model_copy(update={"provider": "qwen_local", "model": "Qwen/Qwen3-1.7B"}),
        [hit],
    )

    for answer in (api_answer, local_answer):
        assert answer.contract_version == CONTRACT_VERSION
        assert answer.answer_markdown
        assert answer.citations
        assert answer.retrieved_hits == [hit]
        assert any(stage.stage_name == "query_rewrite" for stage in answer.trace)
        assert any(stage.stage_name == "context_pack" for stage in answer.trace)
        assert any(stage.stage_name == "generate" for stage in answer.trace)


class _MockChatClient:
    """! @brief 契约测试用 mock OpenAI client。"""

    def __init__(self, content: str):
        self.chat = self
        self.completions = self
        self._content = content

    def create(self, **_kwargs):
        message = type("Message", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()
