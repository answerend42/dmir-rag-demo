"""! @file test_qwen_generators.py
@brief Qwen generator adapter 单元测试（完全离线）。
"""

import pytest

from rag_core.contracts.enums import RagMode
from rag_core.contracts.errors import ProviderUnavailable
from rag_core.contracts.models import RagRequest, SearchHit
from rag_core.llms.qwen_api import QwenApiGenerator, build_qwen_api_complete_fn
from rag_core.llms.qwen_local import QwenLocalGenerator


def _sample_hit(text: str = "自然语言处理是语言计算技术。", score: float = 1.2) -> SearchHit:
    return SearchHit(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text=text,
        score=score,
        rank=1,
        source="sample_data/course_qa_public.json",
        metadata={"page_numbers": [], "section_path": ["课程 QA"], "block_type": "text"},
    )


def test_qwen_api_generator_uses_mock_client_for_basic_rag():
    """! @brief mock client 下 basic_rag 应返回 citations 与 Markdown。"""
    generator = QwenApiGenerator(
        model="qwen-turbo",
        client=_MockChatClient("## 回答\n\n自然语言处理是语言计算。[证据1]"),
    )
    request = RagRequest(
        query="什么是自然语言处理？",
        rag_mode=RagMode.BASIC_RAG,
        provider="qwen_api",
        model="qwen-turbo",
        require_citations=True,
    )

    answer = generator.generate(request, [_sample_hit()])

    assert "自然语言处理" in answer.answer_markdown
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chunk-1"
    assert answer.retrieved_hits
    assert answer.metadata["generator"] == "qwen-api-generator"
    assert any(stage.stage_name == "generate" for stage in answer.trace)


def test_qwen_api_generator_refuses_when_no_context_in_optimized_mode():
    """! @brief optimized 模式在无证据时必须拒答。"""
    generator = QwenApiGenerator(
        model="qwen-turbo",
        client=_MockChatClient("不应被调用"),
    )
    request = RagRequest(
        query="什么是自然语言处理？",
        rag_mode=RagMode.OPTIMIZED_RAG,
        provider="qwen_api",
        model="qwen-turbo",
    )

    answer = generator.generate(request, [])

    assert "无法生成有证据支撑的回答" in answer.answer_markdown
    assert not answer.citations
    assert answer.warnings


def test_qwen_api_generator_llm_only_warns_without_citations():
    """! @brief llm_only 模式应警告且无 citations。"""
    generator = QwenApiGenerator(
        model="qwen-turbo",
        client=_MockChatClient("## 纯模型回答\n\nNLP 是语言计算。"),
    )
    request = RagRequest(
        query="什么是自然语言处理？",
        rag_mode=RagMode.LLM_ONLY,
        provider="qwen_api",
        model="qwen-turbo",
    )

    answer = generator.generate(request, [_sample_hit()])

    assert "纯模型模式没有检索证据" in answer.warnings[0]
    assert not answer.citations
    assert not answer.retrieved_hits


def test_qwen_local_generator_uses_injected_generate_text():
    """! @brief 本地 generator 可通过 generate_text 注入 mock。"""
    generator = QwenLocalGenerator(
        model="Qwen/Qwen3-1.7B",
        generate_text=lambda _model, _prompt: "## 本地回答\n\n基于证据作答。",
    )
    request = RagRequest(
        query="什么是自然语言处理？",
        rag_mode=RagMode.BASIC_RAG,
        provider="qwen_local",
        model="Qwen/Qwen3-1.7B",
    )

    answer = generator.generate(request, [_sample_hit()])

    assert "本地回答" in answer.answer_markdown
    assert answer.metadata["provider"] == "qwen_local"
    assert answer.metadata["generator"] == "qwen-local-generator"


def test_build_qwen_api_complete_fn_requires_api_key_without_client():
    """! @brief 未注入 client 且缺少 API Key 时应抛出 ProviderUnavailable。"""
    with pytest.raises(ProviderUnavailable):
        build_qwen_api_complete_fn(api_key=None, client=None)


class _MockChatClient:
    """! @brief 用于离线测试的最小 OpenAI 兼容 client。"""

    def __init__(self, content: str):
        self._content = content
        self.chat = self
        self.completions = self

    def create(self, **_kwargs):
        return _MockCompletion(self._content)


class _MockCompletion:
    """! @brief mock chat completion 响应。"""

    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


class _MockChoice:
    """! @brief mock completion choice。"""

    def __init__(self, content: str):
        self.message = _MockMessage(content)


class _MockMessage:
    """! @brief mock chat message。"""

    def __init__(self, content: str):
        self.content = content
