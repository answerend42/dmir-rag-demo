"""! @file test_embeddings.py
@brief embedding adapter 单元测试：不依赖网络或真实模型。
"""

import pytest
from rag_core.embeddings import MockEmbedder, QwenApiEmbedder, QwenLocalEmbedder


def test_mock_embedder_batch():
    """! @brief MockEmbedder 批量输出稳定 metadata。"""
    embedder = MockEmbedder(dim=32)
    texts = ["hello", "world"]
    results = embedder.embed_batch(texts)
    assert len(results) == 2
    assert results[0].dim == 32
    assert results[0].provider == "mock"
    assert "text_hash" in results[0].metadata
    assert results[0].item_id.startswith("mock_")


def test_mock_embedder_single():
    """! @brief MockEmbedder 保留旧版单文本调用兼容性。"""
    embedder = MockEmbedder(dim=10)
    vec = embedder.embed("test")
    assert vec.dim == 10
    assert len(vec.vector) == 10


def test_qwen_api_requires_key(monkeypatch):
    """! @brief Qwen API embedder 缺少环境变量时拒绝初始化。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        QwenApiEmbedder()


def test_qwen_api_mocked(monkeypatch):
    """! @brief Qwen API embedder 在 mock HTTP 下返回契约向量。"""
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
        """! @brief requests 响应替身。"""

        def raise_for_status(self):
            """! @brief mock 响应不抛出 HTTP 错误。"""
            pass

        def json(self):
            """! @brief 返回固定 DashScope 风格响应。"""
            return fake_response

    def mock_post(*args, **kwargs):
        """! @brief 替代 requests.post。"""
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)
    results = embedder.embed_batch(["a", "b"])
    assert len(results) == 2
    assert results[0].dim == 3
    assert results[0].provider == "qwen_api"
    assert results[0].item_id == "qwen_api_0"


def test_qwen_local_mocked(monkeypatch):
    """! @brief Qwen 本地 embedder 在注入模型时不下载真实模型。"""
    class DummyModel:
        """! @brief 本地模型替身。"""

        def encode(self, texts, **kwargs):
            """! @brief 返回固定向量。"""
            return [[1.0, 2.0, 3.0] for _ in texts]

    embedder = QwenLocalEmbedder()
    monkeypatch.setattr(embedder, "_model", DummyModel())
    results = embedder.embed_batch(["x"])
    assert results[0].dim == 3
    assert results[0].provider == "qwen_local"
    assert results[0].item_id == "qwen_local_0"
