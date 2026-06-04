"""! @file chroma_hnsw.py
@brief Chroma HNSW profile 适配器。
@details 本文件只封装 Chroma 的 HNSW 参数 profile，不声称支持 Milvus 式
多算法切换。Chroma 返回的 distance 会在 adapter 内转换为 SearchHit.score，
确保 score 始终越大越相关。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_core.contracts.errors import EmptyCorpus, ProviderUnavailable, VectorDimensionMismatch
from rag_core.contracts.models import Chunk, EmbeddingVector, SearchHit
from rag_core.vector_indexes.metadata import strip_forbidden_metadata

_CHUNK_METADATA_KEY = "_rag_chunk_metadata_json"


@dataclass(frozen=True)
class ChromaHnswProfile:
    """! @brief Chroma HNSW 参数 profile。
    @details Chroma 的核心索引仍是 HNSW；不同 profile 只调整构建与查询参数。
    """

    name: str
    m: int
    construction_ef: int
    search_ef: int
    space: str = "cosine"
    description: str = ""

    @property
    def configuration(self) -> dict[str, Any]:
        """! @brief 返回新版 Chroma collection configuration 参数。"""
        return {
            "hnsw": {
                "space": self.space,
                "max_neighbors": self.m,
                "ef_construction": self.construction_ef,
                "ef_search": self.search_ef,
            }
        }

    @property
    def legacy_metadata(self) -> dict[str, Any]:
        """! @brief 返回旧版 Chroma collection metadata HNSW 参数。"""
        return {
            "hnsw:space": self.space,
            "hnsw:M": self.m,
            "hnsw:construction_ef": self.construction_ef,
            "hnsw:search_ef": self.search_ef,
            "profile": self.name,
        }


CHROMA_HNSW_PROFILES: dict[str, ChromaHnswProfile] = {
    "chroma_hnsw_fast": ChromaHnswProfile(
        name="chroma_hnsw_fast",
        m=16,
        construction_ef=64,
        search_ef=24,
        description="快速构建与低延迟查询，适合小型演示和冒烟 benchmark。",
    ),
    "chroma_hnsw_balanced": ChromaHnswProfile(
        name="chroma_hnsw_balanced",
        m=32,
        construction_ef=128,
        search_ef=64,
        description="构建成本、查询延迟和召回率之间的默认均衡 profile。",
    ),
    "chroma_hnsw_high_recall": ChromaHnswProfile(
        name="chroma_hnsw_high_recall",
        m=48,
        construction_ef=256,
        search_ef=128,
        description="提高搜索 ef 和邻居数以优先追求 recall。",
    ),
}


def resolve_chroma_hnsw_profile(profile: str | ChromaHnswProfile) -> ChromaHnswProfile:
    """! @brief 将 profile 名称解析为 ChromaHnswProfile。
    @param profile profile 名称或已构造的 profile 对象。
    @return 对应的 HNSW profile。
    @throws ValueError profile 名称不存在时抛出。
    """
    if isinstance(profile, ChromaHnswProfile):
        return profile
    try:
        return CHROMA_HNSW_PROFILES[profile]
    except KeyError as exc:
        names = ", ".join(sorted(CHROMA_HNSW_PROFILES))
        raise ValueError(f"Unknown Chroma HNSW profile {profile!r}; expected one of: {names}") from exc


def chroma_distance_to_score(distance: float, space: str = "cosine") -> float:
    """! @brief 将 Chroma distance 转成越大越相关的 score。
    @param distance Chroma query 返回的 distance，数值越小越相近。
    @param space Chroma HNSW 空间；阶段 A profiles 默认使用 cosine。
    @return 满足 SearchHit.score 语义的相关性分数。
    """
    value = float(distance)
    if space == "cosine":
        return 1.0 - value
    return -value


class ChromaHnswIndex:
    """! @brief 使用 Chroma HNSW profile 的 VectorIndex 适配器。"""

    def __init__(
        self,
        profile: str | ChromaHnswProfile = "chroma_hnsw_balanced",
        collection_name: str = "rag-demo-chroma-hnsw",
        persist_directory: str | Path | None = None,
        client: Any | None = None,
        collection: Any | None = None,
    ):
        """! @brief 初始化 Chroma collection。
        @param profile HNSW profile 名称或对象。
        @param collection_name Chroma collection 名称。
        @param persist_directory 可选持久化目录；为空时使用内存 client。
        @param client 可注入的 Chroma client，便于测试。
        @param collection 可注入的 collection，便于无 Chroma 依赖的单元测试。
        @throws ProviderUnavailable 本地没有安装 chromadb 时抛出。
        """
        self.profile = resolve_chroma_hnsw_profile(profile)
        self.name = self.profile.name
        self._dim: int | None = None
        self._chunks: dict[str, Chunk] = {}
        self._collection = collection or self._get_or_create_collection(
            collection_name=collection_name,
            persist_directory=persist_directory,
            client=client,
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> None:
        """! @brief 将文本块和向量写入 Chroma collection。
        @param chunks 与 embeddings 一一对应的文本块。
        @param embeddings 与 chunk_id 对齐的向量。
        @throws EmptyCorpus chunks 为空时抛出。
        @throws VectorDimensionMismatch 向量维度不一致时抛出。
        """
        if not chunks:
            raise EmptyCorpus("ChromaHnswIndex received no chunks")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids: list[str] = []
        vectors: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings):
            self._validate_pair(chunk, embedding)
            ids.append(chunk.chunk_id)
            vectors.append([float(value) for value in embedding.vector])
            documents.append(chunk.text)
            metadatas.append(_metadata_for_chroma(chunk))
            self._chunks[chunk.chunk_id] = chunk

        self._collection.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)

    def search(self, query_embedding: EmbeddingVector, top_k: int) -> list[SearchHit]:
        """! @brief 查询 Chroma 并返回按 score 降序排列的 SearchHit。
        @param query_embedding 查询向量。
        @param top_k 返回命中数量上限。
        @return 满足契约 score 语义的命中列表。
        @throws VectorDimensionMismatch 查询维度与索引维度不一致时抛出。
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._dim is not None and self._dim != query_embedding.dim:
            raise VectorDimensionMismatch("Query dimension does not match index dimension")

        result = self._collection.query(
            query_embeddings=[[float(value) for value in query_embedding.vector]],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_result_list(result, "ids")
        documents = _first_result_list(result, "documents")
        metadatas = _first_result_list(result, "metadatas")
        distances = _first_result_list(result, "distances")

        hits: list[SearchHit] = []
        for index, chunk_id_value in enumerate(ids):
            chunk_id = str(chunk_id_value)
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document = str(documents[index]) if index < len(documents) and documents[index] else ""
            distance = float(distances[index]) if index < len(distances) else 0.0
            chunk = self._chunks.get(chunk_id)
            restored_metadata = _restore_chunk_metadata(metadata)

            doc_id = _string_or_none(metadata.get("doc_id")) or (chunk.doc_id if chunk else chunk_id)
            source = _string_or_none(metadata.get("source")) or (chunk.source if chunk else "chroma")
            text = document or (chunk.text if chunk else chunk_id)
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    score=chroma_distance_to_score(distance, self.profile.space),
                    rank=index + 1,
                    source=source,
                    metadata=restored_metadata,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [
            SearchHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=hit.text,
                score=hit.score,
                rank=rank,
                source=hit.source,
                metadata=hit.metadata,
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    def _get_or_create_collection(
        self,
        collection_name: str,
        persist_directory: str | Path | None,
        client: Any | None,
    ) -> Any:
        """! @brief 使用新版 configuration 创建 collection，并兼容旧版 metadata。"""
        chroma_client = client or _build_chroma_client(persist_directory)
        metadata = {"profile": self.profile.name, **self.profile.legacy_metadata}
        try:
            return chroma_client.get_or_create_collection(
                name=collection_name,
                configuration=self.profile.configuration,
                metadata=metadata,
            )
        except TypeError:
            return chroma_client.get_or_create_collection(name=collection_name, metadata=metadata)

    def _validate_pair(self, chunk: Chunk, embedding: EmbeddingVector) -> None:
        """! @brief 校验文本块和向量的 ID 与维度。"""
        if chunk.chunk_id != embedding.item_id:
            raise ValueError("Embedding item_id must match Chunk chunk_id")
        if self._dim is None:
            self._dim = embedding.dim
        elif self._dim != embedding.dim:
            raise VectorDimensionMismatch("All indexed embeddings must share one dimension")


def _build_chroma_client(persist_directory: str | Path | None) -> Any:
    """! @brief 延迟导入 chromadb 并创建本地 client。"""
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise ProviderUnavailable("chromadb is required for ChromaHnswIndex") from exc

    if persist_directory is None:
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=str(persist_directory))


def _metadata_for_chroma(chunk: Chunk) -> dict[str, Any]:
    """! @brief 将 Chunk metadata 序列化为 Chroma 可保存的标量字段。"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        _CHUNK_METADATA_KEY: json.dumps(strip_forbidden_metadata(chunk.metadata), ensure_ascii=False, sort_keys=True),
    }


def _restore_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """! @brief 从 Chroma metadata 恢复原始 Chunk metadata。"""
    raw_value = metadata.get(_CHUNK_METADATA_KEY)
    if not isinstance(raw_value, str) or not raw_value:
        return {}
    try:
        restored = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return strip_forbidden_metadata(restored) if isinstance(restored, dict) else {}


def _first_result_list(result: dict[str, Any], key: str) -> list[Any]:
    """! @brief 读取 Chroma query 结果中的第一组命中。"""
    values = result.get(key) or []
    if not values:
        return []
    first = values[0]
    return first if isinstance(first, list) else []


def _string_or_none(value: Any) -> str | None:
    """! @brief 将非空值转换成字符串。"""
    if value is None:
        return None
    text = str(value)
    return text if text else None
