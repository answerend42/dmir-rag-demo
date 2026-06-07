"""! @file 本地加载 sentence-transformers 模型实现 Qwen Embedding。"""

import time
from typing import List, Optional
from rag_core.contracts import EmbeddingVector, Embedder
from rag_core.contracts.errors import ProviderUnavailable


class QwenLocalEmbedder(Embedder):
    """本地通义千问嵌入模型（基于 sentence-transformers）。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Embedding",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.provider = "qwen_local"
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ProviderUnavailable(
                    "QwenLocalEmbedder 需要安装 sentence-transformers。请运行: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """批量编码文本。"""
        if not texts:
            return []
        start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results = []
        for i, vec in enumerate(embeddings):
            if hasattr(vec, 'shape'):
                dim = vec.shape[0]
                vector = vec.tolist()
            else:
                dim = len(vec)
                vector = vec
            results.append(EmbeddingVector(
                item_id=f"{self.provider}_{i}",
                vector=vector,
                dim=dim,
                model=self.model_name,
                provider=self.provider,
                metadata={
                    "inference_time_ms": round(elapsed_ms / len(texts), 3),
                    "batch_index": i,
                }
            ))
        return results

    def embed(self, text: str) -> EmbeddingVector:
        """单条文本嵌入。"""
        return self.embed_batch([text])[0]