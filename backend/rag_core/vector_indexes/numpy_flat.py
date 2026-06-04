"""! @file numpy_flat.py
@brief NumpyFlat 精确向量检索基线。
@details 该实现按余弦相似度对所有向量做全量扫描，作为 Chroma HNSW
profiles 的 recall 上界。运行环境未安装 numpy 时会退回标准库精确计算，
但接口和排序语义保持不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag_core.contracts.errors import EmptyCorpus, VectorDimensionMismatch
from rag_core.contracts.models import Chunk, EmbeddingVector, SearchHit
from rag_core.vector_indexes.metadata import strip_forbidden_metadata

try:
    import numpy as _np
except ModuleNotFoundError:
    _np = None


@dataclass(frozen=True)
class _StoredVector:
    """! @brief 索引内部保存的向量、文本块和插入顺序。"""

    chunk: Chunk
    vector: tuple[float, ...]
    order: int


class NumpyFlatIndex:
    """! @brief 精确余弦检索基线，满足 VectorIndex Protocol。
    @details score 直接使用余弦相似度，因此越大越相关。相同分数时按首次
    插入顺序稳定排序，便于 benchmark 复现。
    """

    name = "numpy_flat"

    def __init__(self, name: str = "numpy_flat"):
        """! @brief 初始化空的精确向量索引。
        @param name 暴露给 benchmark 和 trace 的索引名称。
        """
        self.name = name
        self._vectors: dict[str, _StoredVector] = {}
        self._dim: int | None = None
        self._next_order = 0

    def upsert(self, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> None:
        """! @brief 新增或更新文本块向量。
        @param chunks 与 embeddings 一一对应的文本块。
        @param embeddings 与文本块 chunk_id 对齐的向量。
        @throws EmptyCorpus chunks 为空时抛出。
        @throws VectorDimensionMismatch 向量维度不一致时抛出。
        """
        if not chunks:
            raise EmptyCorpus("NumpyFlatIndex received no chunks")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk, embedding in zip(chunks, embeddings):
            self._validate_pair(chunk, embedding)
            vector = _coerce_vector(embedding.vector, embedding.dim)
            existing = self._vectors.get(chunk.chunk_id)
            order = existing.order if existing else self._next_order
            if existing is None:
                self._next_order += 1
            self._vectors[chunk.chunk_id] = _StoredVector(chunk=chunk, vector=vector, order=order)

    def search(self, query_embedding: EmbeddingVector, top_k: int) -> list[SearchHit]:
        """! @brief 按精确余弦相似度返回 top-k 命中。
        @param query_embedding 查询向量。
        @param top_k 返回命中数量上限。
        @return 按 score 降序排列的 SearchHit 列表。
        @throws EmptyCorpus 尚未写入任何向量时抛出。
        @throws VectorDimensionMismatch 查询维度与索引维度不一致时抛出。
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._vectors:
            raise EmptyCorpus("NumpyFlatIndex has no embeddings")
        if self._dim != query_embedding.dim:
            raise VectorDimensionMismatch("Query dimension does not match index dimension")

        query_vector = _coerce_vector(query_embedding.vector, query_embedding.dim)
        scored = [
            (_cosine_similarity(query_vector, stored.vector), stored.order, stored.chunk)
            for stored in self._vectors.values()
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        hits: list[SearchHit] = []
        for rank, (score, _order, chunk) in enumerate(scored[:top_k], start=1):
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                    rank=rank,
                    source=chunk.source,
                    metadata=strip_forbidden_metadata(chunk.metadata),
                )
            )
        return hits

    def _validate_pair(self, chunk: Chunk, embedding: EmbeddingVector) -> None:
        """! @brief 校验文本块和向量的一一对应关系。"""
        if chunk.chunk_id != embedding.item_id:
            raise ValueError("Embedding item_id must match Chunk chunk_id")
        if self._dim is None:
            self._dim = embedding.dim
        elif self._dim != embedding.dim:
            raise VectorDimensionMismatch("All indexed embeddings must share one dimension")


def _coerce_vector(vector: list[float], dim: int) -> tuple[float, ...]:
    """! @brief 将契约向量转换为稳定的 float tuple。"""
    if len(vector) != dim:
        raise VectorDimensionMismatch("EmbeddingVector.dim must match len(vector)")
    return tuple(float(value) for value in vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """! @brief 精确计算余弦相似度，优先使用 numpy。"""
    if len(left) != len(right):
        raise VectorDimensionMismatch("Vectors must have the same dimension")
    if _np is not None:
        left_array = _np.asarray(left, dtype=float)
        right_array = _np.asarray(right, dtype=float)
        denominator = float(_np.linalg.norm(left_array) * _np.linalg.norm(right_array))
        if denominator == 0.0:
            return 0.0
        return float(_np.dot(left_array, right_array) / denominator)

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
