"""! @file test_embedding_contract.py
@brief 契约测试：验证 mock embedder 满足 EmbeddingVector 契约。
"""

import pytest

from rag_core.contracts.enums import BlockType
from rag_core.contracts.models import Chunk
from rag_core.embeddings import MockEmbedder, QwenApiEmbedder, QwenLocalEmbedder


@pytest.fixture
def mock_embedder():
    """! @brief 构造离线 mock embedder。"""
    return MockEmbedder(dim=128)


@pytest.fixture
def qwen_api_embedder(monkeypatch):
    """! @brief 构造 mock HTTP 响应下的 Qwen API embedder。"""
    import requests_mock
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    embedder = QwenApiEmbedder()
    with requests_mock.Mocker() as m:
        def callback(request, context):
            texts = request.json()["input"]["texts"]
            embeddings = [{"text_index": i, "embedding": [0.1] * 1536} for i in range(len(texts))]
            return {"output": {"embeddings": embeddings}}
        m.post(embedder.api_base, json=callback)
        yield embedder


@pytest.fixture
def qwen_local_embedder(monkeypatch):
    """! @brief 构造注入 DummyModel 的本地 Qwen embedder。"""
    class DummyModel:
        """! @brief 测试用本地模型替身。"""

        def encode(self, texts, **kwargs):
            """! @brief 返回固定维度向量。"""
            return [[1.0] * 256 for _ in texts]
    embedder = QwenLocalEmbedder()
    monkeypatch.setattr(embedder, "_model", DummyModel())
    return embedder


@pytest.mark.parametrize("fixture_name", [
    "mock_embedder",
])
def test_contract_batch_input_output(fixture_name, request):
    """! @brief embed_batch 输出必须满足 EmbeddingVector 契约。"""
    embedder = request.getfixturevalue(fixture_name)
    texts = ["chunk A", "chunk B", "chunk C"]
    results = embedder.embed_batch(texts)
    assert len(results) == len(texts)
    for r in results:
        assert len(r.vector) == r.dim > 0
        assert r.model != ""
        assert r.provider != ""
        assert isinstance(r.metadata, dict)
        assert len(r.metadata) > 0, "metadata should not be empty"
        assert isinstance(r.item_id, str) and len(r.item_id) > 0, "item_id must be a non-empty string"


def test_mock_embedder_supports_chunk_contract_and_query_embedding(mock_embedder):
    """! @brief MockEmbedder 同时支持 Chunk 列表和查询向量。"""
    chunks = [
        Chunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            text="自然语言处理",
            source="contract-test",
            block_ids=[],
            block_types=[BlockType.TEXT],
            token_count=1,
        )
    ]

    vectors = mock_embedder.embed(chunks)
    query_vector = mock_embedder.embed_query("自然语言处理是什么？")

    assert vectors[0].item_id == "chunk-a"
    assert query_vector.item_id.startswith("query_")
    assert query_vector.dim == mock_embedder.dim
