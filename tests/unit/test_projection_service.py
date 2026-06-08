"""! @file test_projection_service.py
@brief 向量投影服务单元测试。
"""

import pytest

from services.projection_service import VectorProjectionService


def test_tsne_projection_keeps_query_overlay_without_retrieval_rerank():
    """! @brief t-SNE 三维投影只计算坐标，不改变分块列表顺序。"""
    embeddings = [
        make_embedding("near", [1.0, 0.0, 0.0], 1),
        make_embedding("side", [0.0, 1.0, 0.0], 2),
        make_embedding("opposite", [-1.0, 0.0, 0.0], 3),
        make_embedding("diag", [0.5, 0.5, 0.0], 4),
        make_embedding("z", [0.0, 0.0, 1.0], 5),
    ]

    projection = VectorProjectionService.project_embeddings(
        embeddings,
        method="tsne",
        overlays=[{"id": "query", "role": "query", "label": "查询", "vector": [1.0, 0.0, 0.0]}],
    )

    chunk_ids = [
        point["embedding"]["metadata"]["chunk_id"]
        for point in projection["points"]
    ]

    assert projection["method"] == "tsne"
    assert projection["method_label"] == "t-SNE"
    assert projection["target_dimensions"] == 3
    assert projection["axes"]["x"]["label"] == "t-SNE-1"
    assert projection["axes"]["z"]["label"] == "t-SNE-3"
    assert projection["available_methods"][0]["id"] == "tsne"
    assert projection["available_dimensions"] == [3, 2]
    assert "raw_z" in projection["points"][0]
    assert "z" in projection["points"][0]
    assert chunk_ids == ["near", "side", "opposite", "diag", "z"]
    assert projection["overlays"][0]["role"] == "query"
    assert "z" in projection["overlays"][0]


def test_pca_projection_keeps_input_order_with_query_overlay():
    """! @brief PCA 投影不使用查询余弦重排分块顺序。"""
    embeddings = [
        make_embedding("near", [1.0, 0.0], 1),
        make_embedding("side", [0.0, 1.0], 2),
        make_embedding("opposite", [-1.0, 0.0], 3),
    ]

    projection = VectorProjectionService.project_embeddings(
        embeddings,
        method="pca",
        overlays=[{"id": "query", "role": "query", "label": "查询", "vector": [1.0, 0.0]}],
        target_dimensions=2,
    )

    chunk_ids = [
        point["embedding"]["metadata"]["chunk_id"]
        for point in projection["points"]
    ]

    assert projection["method"] == "pca"
    assert projection["target_dimensions"] == 2
    assert projection["available_methods"] == [
        {"id": "tsne", "label": "t-SNE", "requires_query": False},
        {"id": "pca", "label": "PCA", "requires_query": False},
    ]
    assert projection["axes"]["x"]["label"] == "PC1"
    assert "z" not in projection["points"][0]
    assert chunk_ids == ["near", "side", "opposite"]
    assert projection["overlays"][0]["role"] == "query"


def test_projection_service_rejects_unsupported_dimensions():
    """! @brief 投影服务只接受二维或三维展示。"""
    embeddings = [make_embedding("a", [1.0, 0.0], 1)]

    with pytest.raises(ValueError, match="2 或 3"):
        VectorProjectionService.project_embeddings(embeddings, target_dimensions=4)


def test_projection_service_rejects_unsupported_methods():
    """! @brief 投影服务不接受未登记的方法。"""
    embeddings = [make_embedding("a", [1.0, 0.0], 1)]

    with pytest.raises(ValueError, match="不支持"):
        VectorProjectionService.project_embeddings(embeddings, method="cosine_mds")


def test_projection_service_rejects_query_cosine_method():
    """! @brief 投影服务不再接受查询余弦排序视图。"""
    embeddings = [make_embedding("a", [1.0, 0.0], 1)]

    with pytest.raises(ValueError, match="不支持"):
        VectorProjectionService.project_embeddings(
            embeddings,
            method="query_cosine",
            overlays=[{"id": "query", "role": "query", "label": "查询", "vector": [1.0, 0.0]}],
        )


def make_embedding(chunk_id: str, vector: list[float], page: int) -> dict:
    """! @brief 构造投影服务测试条目。"""
    return {
        "embedding": vector,
        "metadata": {
            "chunk_id": chunk_id,
            "page_number": page,
            "content": f"分块 {chunk_id}",
            "vector_dimension": len(vector),
        },
    }
