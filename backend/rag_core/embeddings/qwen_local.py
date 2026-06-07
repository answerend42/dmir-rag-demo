"""! @file qwen_local.py
@brief 本地加载 sentence-transformers 模型实现 Qwen embedding。
"""

from __future__ import annotations

import time

from rag_core.contracts import EmbeddingVector, Embedder
from rag_core.contracts.models import Chunk
from rag_core.contracts.errors import ProviderUnavailable


class QwenLocalEmbedder(Embedder):
    """! @brief 本地通义千问嵌入模型，基于 sentence-transformers 懒加载。"""

    name = "qwen-local-embedder"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Embedding",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """! @brief 初始化本地 Qwen embedding 适配器。
        @param model_name sentence-transformers 模型名或本地路径。
        @param device 可选运行设备。
        @param batch_size 每批编码文本数量。
        @throws ValueError batch_size 非法时抛出。
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        self.model_name = model_name
        self.batch_size = batch_size
        self.provider = "qwen_local"
        self._device = device
        self._model = None

    @property
    def model(self):
        """! @brief 懒加载 sentence-transformers 模型实例。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ProviderUnavailable(
                    "QwenLocalEmbedder 需要安装 sentence-transformers。请运行: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def embed_batch(self, texts: list[str], item_ids: list[str] | None = None) -> list[EmbeddingVector]:
        """! @brief 批量编码文本。
        @param texts 待嵌入文本列表。
        @param item_ids 可选外部稳定 ID；为空时按 batch 位置生成。
        @return 与输入文本一一对应的 EmbeddingVector 列表。
        @throws ValueError item_ids 数量与 texts 不一致时抛出。
        """
        if not texts:
            return []
        if item_ids is not None and len(item_ids) != len(texts):
            raise ValueError("item_ids 数量必须与 texts 一致")

        start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results: list[EmbeddingVector] = []
        for i, vec in enumerate(embeddings):
            if hasattr(vec, 'shape'):
                dim = vec.shape[0]
                vector = vec.tolist()
            else:
                dim = len(vec)
                vector = vec
            results.append(
                EmbeddingVector(
                    item_id=item_ids[i] if item_ids else f"{self.provider}_{i}",
                    vector=vector,
                    dim=dim,
                    model=self.model_name,
                    provider=self.provider,
                    metadata={
                        "inference_time_ms": round(elapsed_ms / len(texts), 3),
                        "batch_index": i,
                    },
                )
            )
        return results

    def embed(self, chunks_or_text: list[Chunk] | str) -> list[EmbeddingVector] | EmbeddingVector:
        """! @brief 兼容 Chunk 列表契约和旧版单文本调用。"""
        if isinstance(chunks_or_text, str):
            return self.embed_batch([chunks_or_text])[0]
        return self.embed_batch(
            [chunk.text for chunk in chunks_or_text],
            item_ids=[chunk.chunk_id for chunk in chunks_or_text],
        )

    def embed_query(self, query: str) -> EmbeddingVector:
        """! @brief 为查询文本生成可用于检索的向量。"""
        return self.embed_batch([query], item_ids=["query_qwen_local_0"])[0]
