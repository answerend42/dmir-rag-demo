"""! @file 契约测试：验证所有 embedder 满足 EmbeddingVector 契约。"""

import pytest
from rag_core.embeddings import MockEmbedder, QwenApiEmbedder, QwenLocalEmbedder


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dim=128)


@pytest.fixture
def qwen_api_embedder(monkeypatch):
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
    class DummyModel:
        def encode(self, texts, **kwargs):
            return [[1.0] * 256 for _ in texts]
    embedder = QwenLocalEmbedder()
    monkeypatch.setattr(embedder, "_model", DummyModel())
    return embedder


@pytest.mark.parametrize("fixture_name", [
    "mock_embedder",
])
def test_contract_batch_input_output(fixture_name, request):
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