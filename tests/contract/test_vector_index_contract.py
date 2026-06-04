"""! @file test_vector_index_contract.py
@brief Issue #2 向量索引契约测试。
"""

from rag_core.contracts.enums import BlockType
from rag_core.contracts.models import Chunk, EmbeddingVector, SearchHit
from rag_core.contracts.protocols import VectorIndex
from rag_core.vector_indexes import NumpyFlatIndex


def test_numpy_flat_satisfies_vector_index_contract():
    """! @brief NumpyFlatIndex 满足 VectorIndex Protocol 的核心行为。"""
    index: VectorIndex = NumpyFlatIndex()
    chunks = [
        Chunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            text="向量检索基线",
            source="contract-test",
            block_ids=[],
            block_types=[BlockType.TEXT],
            token_count=1,
            metadata={"section_path": ["契约"], "page_numbers": []},
        ),
        Chunk(
            chunk_id="chunk-b",
            doc_id="doc-a",
            text="无关文本",
            source="contract-test",
            block_ids=[],
            block_types=[BlockType.TEXT],
            token_count=1,
            metadata={"section_path": ["契约"], "page_numbers": []},
        ),
    ]
    embeddings = [
        EmbeddingVector(item_id="chunk-a", vector=[1.0, 0.0], dim=2, model="unit", provider="unit"),
        EmbeddingVector(item_id="chunk-b", vector=[0.0, 1.0], dim=2, model="unit", provider="unit"),
    ]

    index.upsert(chunks, embeddings)
    hits = index.search(
        EmbeddingVector(item_id="query", vector=[1.0, 0.0], dim=2, model="unit", provider="unit"),
        top_k=2,
    )

    assert all(isinstance(hit, SearchHit) for hit in hits)
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score >= hits[1].score
    assert hits[0].chunk_id == "chunk-a"
