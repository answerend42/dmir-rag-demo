"""! @file test_vector_indexes.py
@brief Issue #2 向量索引单元测试。
"""

import json

import pytest

from rag_core.contracts.enums import BlockType
from rag_core.contracts.errors import VectorDimensionMismatch
from rag_core.contracts.models import Chunk, EmbeddingVector
from rag_core.vector_indexes import (
    CHROMA_HNSW_PROFILES,
    ChromaHnswIndex,
    NumpyFlatIndex,
    chroma_distance_to_score,
    resolve_chroma_hnsw_profile,
)


def test_numpy_flat_returns_top_k_by_descending_score():
    """! @brief NumpyFlat 按精确余弦相似度降序返回 top-k。"""
    chunks = [
        make_chunk("c1", "完全相关"),
        make_chunk("c2", "部分相关"),
        make_chunk("c3", "反向相关"),
    ]
    embeddings = [
        make_embedding("c1", [1.0, 0.0]),
        make_embedding("c2", [0.2, 0.98]),
        make_embedding("c3", [-1.0, 0.0]),
    ]
    index = NumpyFlatIndex()
    index.upsert(chunks, embeddings)

    hits = index.search(make_embedding("query", [1.0, 0.0]), top_k=2)

    assert [hit.chunk_id for hit in hits] == ["c1", "c2"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score > hits[1].score
    assert hits == sorted(hits, key=lambda hit: hit.score, reverse=True)


def test_numpy_flat_rejects_dimension_mismatch():
    """! @brief NumpyFlat 拒绝维度不一致的向量。"""
    index = NumpyFlatIndex()
    index.upsert([make_chunk("c1", "文本")], [make_embedding("c1", [1.0, 0.0])])

    with pytest.raises(VectorDimensionMismatch):
        index.search(make_embedding("query", [1.0, 0.0, 0.0]), top_k=1)


def test_chroma_hnsw_profiles_are_named_profiles_not_algorithms():
    """! @brief Chroma HNSW profile 名称和参数稳定可复现。"""
    assert set(CHROMA_HNSW_PROFILES) == {
        "chroma_hnsw_fast",
        "chroma_hnsw_balanced",
        "chroma_hnsw_high_recall",
    }
    fast = resolve_chroma_hnsw_profile("chroma_hnsw_fast")
    high_recall = resolve_chroma_hnsw_profile("chroma_hnsw_high_recall")

    assert fast.space == "cosine"
    assert high_recall.search_ef > fast.search_ef
    assert fast.configuration["hnsw"]["space"] == "cosine"
    assert fast.legacy_metadata["hnsw:search_ef"] == fast.search_ef


def test_chroma_distance_to_score_keeps_larger_more_relevant():
    """! @brief Chroma distance 越小，转换后的 score 越大。"""
    assert chroma_distance_to_score(0.0) == 1.0
    assert chroma_distance_to_score(0.25) > chroma_distance_to_score(0.75)
    assert chroma_distance_to_score(2.0, space="l2") > chroma_distance_to_score(4.0, space="l2")


def test_chroma_adapter_converts_scores_and_reranks_results():
    """! @brief Chroma adapter 将 distance 转为 score 后重新按相关性排序。"""
    collection = FakeChromaCollection()
    index = ChromaHnswIndex(profile="chroma_hnsw_balanced", collection=collection)
    chunks = [make_chunk("c1", "更相关"), make_chunk("c2", "较弱相关")]
    embeddings = [make_embedding("c1", [1.0, 0.0]), make_embedding("c2", [0.0, 1.0])]
    index.upsert(chunks, embeddings)
    collection.query_order = ["c2", "c1"]
    collection.query_distances = [0.7, 0.1]

    hits = index.search(make_embedding("query", [1.0, 0.0]), top_k=2)

    assert [hit.chunk_id for hit in hits] == ["c1", "c2"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score == pytest.approx(0.9)
    assert hits[0].metadata["section_path"] == ["测试"]
    assert collection.last_include == ["documents", "metadatas", "distances"]


def test_vector_indexes_do_not_surface_answer_quality_metadata():
    """! @brief 向量索引不会把评测标签带入 SearchHit metadata。"""
    chunk = make_chunk("c1", "带隐藏标签")
    chunk.metadata["answer_quality"] = 9
    chunk.metadata["nested"] = {"answer_quality": 1, "keep": "ok"}
    embedding = make_embedding("c1", [1.0, 0.0])

    flat = NumpyFlatIndex()
    flat.upsert([chunk], [embedding])
    flat_hit = flat.search(make_embedding("query", [1.0, 0.0]), top_k=1)[0]
    assert "answer_quality" not in flat_hit.model_dump_json()

    collection = FakeChromaCollection()
    chroma = ChromaHnswIndex(profile="chroma_hnsw_fast", collection=collection)
    chroma.upsert([chunk], [embedding])
    collection.query_order = ["c1"]
    collection.query_distances = [0.0]
    chroma_hit = chroma.search(make_embedding("query", [1.0, 0.0]), top_k=1)[0]
    assert "answer_quality" not in chroma_hit.model_dump_json()
    assert json.loads(collection.items["c1"]["metadata"]["_rag_chunk_metadata_json"])["nested"] == {"keep": "ok"}


class FakeChromaCollection:
    """! @brief 无需 chromadb 依赖的 collection fake。"""

    def __init__(self):
        self.items = {}
        self.query_order: list[str] = []
        self.query_distances: list[float] = []
        self.last_include: list[str] | None = None

    def upsert(self, ids, embeddings, documents, metadatas):
        """! @brief 记录写入参数，供 query 返回。"""
        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.items[item_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(self, query_embeddings, n_results, include):
        """! @brief 按测试指定顺序返回 Chroma 风格结果。"""
        self.last_include = include
        ids = self.query_order[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.items[item_id]["document"] for item_id in ids]],
            "metadatas": [[self.items[item_id]["metadata"] for item_id in ids]],
            "distances": [self.query_distances[:n_results]],
        }


def make_chunk(chunk_id: str, text: str) -> Chunk:
    """! @brief 构造测试 Chunk。"""
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        source="unit-test",
        block_ids=[],
        block_types=[BlockType.TEXT],
        token_count=1,
        metadata={"section_path": ["测试"], "page_numbers": []},
    )


def make_embedding(item_id: str, vector: list[float]) -> EmbeddingVector:
    """! @brief 构造测试 EmbeddingVector。"""
    return EmbeddingVector(
        item_id=item_id,
        vector=vector,
        dim=len(vector),
        model="unit",
        provider="unit",
    )
