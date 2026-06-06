"""! @file 单元测试：不依赖网络或真实模型。"""

import pytest
from rag_core.embeddings import MockEmbedder, QwenApiEmbedder, QwenLocalEmbedder


def test_mock_embedder_batch():
    embedder = MockEmbedder(dim=32)
    texts = ["hello", "world"]
    results = embedder.embed_batch(texts)
    assert len(results) == 2
    assert results[0].dim == 32
    assert results[0].provider == "mock"
    assert "text_hash" in results[0].metadata
    assert results[0].item_id.startswith("mock_")


def test_mock_embedder_single():
    embedder = MockEmbedder(dim=10)
    vec = embedder.embed("test")
    assert vec.dim == 10
    assert len(vec.vector) == 10


def test_qwen_api_requires_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        QwenApiEmbedder()


def test_qwen_api_mocked(monkeypatch):
    import requests
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    embedder = QwenApiEmbedder()

    fake_response = {
        "output": {
            "embeddings": [
                {"text_index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"text_index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
    }

    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return fake_response

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)
    results = embedder.embed_batch(["a", "b"])
    assert len(results) == 2
    assert results[0].dim == 3
    assert results[0].provider == "qwen_api"
    assert results[0].item_id == "qwen_api_0"


def test_qwen_local_mocked(monkeypatch):
    class DummyModel:
        def encode(self, texts, **kwargs):
            return [[1.0, 2.0, 3.0] for _ in texts]

    embedder = QwenLocalEmbedder()
    monkeypatch.setattr(embedder, "_model", DummyModel())
    results = embedder.embed_batch(["x"])
    assert results[0].dim == 3
    assert results[0].provider == "qwen_local"
    assert results[0].item_id == "qwen_local_0"