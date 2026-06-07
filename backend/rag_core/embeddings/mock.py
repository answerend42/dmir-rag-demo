"""! @file mock.py
@brief 用于测试、评测和冒烟流水线的 Mock embedding 适配器。
"""

from __future__ import annotations

import random
import zlib
from time import perf_counter
from typing import Any

from rag_core.contracts import Chunk, EmbeddingVector, Embedder


class MockEmbedder(Embedder):
    """! @brief 生成确定性随机向量的本地 embedding 兜底实现。"""

    name = "mock-embedder"

    def __init__(self, dim: int = 384, fixed_seed: int = 42, model_name: str = "mock-embedder"):
        """! @brief 初始化 Mock embedding 适配器。
        @param dim 向量维度。
        @param fixed_seed 固定随机种子。
        @param model_name 输出到 EmbeddingVector.model 的模型名。
        @throws ValueError dim 不是正整数时抛出。
        """
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim
        self.model_name = model_name
        self.provider = "mock"
        self._fixed_seed = fixed_seed

    def embed_batch(self, texts: list[str], item_ids: list[str] | None = None) -> list[EmbeddingVector]:
        """! @brief 批量生成模拟向量。
        @param texts 待嵌入文本列表。
        @param item_ids 可选外部稳定 ID；为空时按 batch 位置生成。
        @return 与输入文本一一对应的 EmbeddingVector 列表。
        @throws ValueError item_ids 数量与 texts 不一致时抛出。
        """
        if item_ids is not None and len(item_ids) != len(texts):
            raise ValueError("item_ids 数量必须与 texts 一致")

        results: list[EmbeddingVector] = []
        start = perf_counter()
        for idx, text in enumerate(texts):
            rng = random.Random(self._seed_for_text(text))
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            elapsed_ms = (perf_counter() - start) * 1000.0
            results.append(
                EmbeddingVector(
                    item_id=item_ids[idx] if item_ids else f"{self.provider}_{idx}",
                    vector=vec,
                    dim=self.dim,
                    model=self.model_name,
                    provider=self.provider,
                    metadata={
                        "generation_mode": "deterministic_random",
                        "text_hash": zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF,
                        "batch_time_ms": round(elapsed_ms, 2),
                    },
                )
            )
        return results

    def embed(self, chunks_or_text: list[Chunk] | str) -> list[EmbeddingVector] | EmbeddingVector:
        """! @brief 兼容 Chunk 列表契约和旧版单文本调用。
        @param chunks_or_text Chunk 列表或单条文本。
        @return Chunk 输入返回向量列表；字符串输入返回单条向量。
        """
        if isinstance(chunks_or_text, str):
            return self.embed_batch([chunks_or_text])[0]
        texts = [chunk.text for chunk in chunks_or_text]
        item_ids = [chunk.chunk_id for chunk in chunks_or_text]
        return self.embed_batch(texts, item_ids=item_ids)

    def embed_query(self, query: str) -> EmbeddingVector:
        """! @brief 为查询文本生成可用于检索的向量。"""
        return self.embed_batch([query], item_ids=[f"query_{self._text_hash(query):08x}"])[0]

    def _seed_for_text(self, text: str) -> int:
        """! @brief 基于固定种子和文本内容生成稳定随机种子。"""
        return self._fixed_seed ^ self._text_hash(text)

    @staticmethod
    def _text_hash(text: str) -> int:
        """! @brief 生成跨进程稳定的文本哈希值。"""
        return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
