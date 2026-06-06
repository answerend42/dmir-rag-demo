"""! @file MockEmbedder 用于测试和冒烟流水线。"""

import random
import time
from typing import List
from rag_core.contracts import EmbeddingVector, Embedder


class MockEmbedder(Embedder):
    """确定性随机向量生成器，固定种子可复现。"""

    def __init__(self, dim: int = 384, fixed_seed: int = 42, model_name: str = "mock-embedder"):
        self.dim = dim
        self.model_name = model_name
        self.provider = "mock"
        self._fixed_seed = fixed_seed   # 保存种子，每次调用时重建局部 RNG

    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """批量生成模拟向量。"""
        results = []
        start = time.perf_counter()
        # 每次调用重建局部 RNG，保证调用间独立性且可复现
        rng = random.Random(self._fixed_seed)
        for idx, text in enumerate(texts):
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append(EmbeddingVector(
                item_id=f"{self.provider}_{idx}",
                vector=vec,
                dim=self.dim,
                model=self.model_name,
                provider=self.provider,
                metadata={
                    "generation_mode": "deterministic_random",
                    "text_hash": hash(text) & 0xffffffff,
                    "batch_time_ms": round(elapsed_ms, 2),
                }
            ))
        return results

    def embed(self, text: str) -> EmbeddingVector:
        """单条文本嵌入。"""
        return self.embed_batch([text])[0]