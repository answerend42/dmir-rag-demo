"""! @file faiss_index_service.py
@brief 使用 FAISS 构建和查询 Flat、IVF、LSH 向量索引。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import faiss
except Exception:  # pragma: no cover - 由运行环境决定是否安装。
    faiss = None


FAISS_INDEX_DIR = Path("03-vector-store/faiss-indexes")
FAISS_INDEX_MODES = {"flat", "ivf", "lsh"}
_EPSILON = 1e-12


class FaissVectorIndexService:
    """! @brief 管理 FAISS 索引文件、元数据和检索逻辑。"""

    def __init__(self, index_dir: Path | str = FAISS_INDEX_DIR):
        """! @brief 初始化 FAISS 索引目录。"""
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, embeddings_data: Dict[str, Any], collection_name: str, index_mode: str) -> Dict[str, Any]:
        """! @brief 根据嵌入文件构建 FAISS 索引。
        @param embeddings_data 嵌入文件完整 JSON。
        @param collection_name 前端后续检索使用的集合名。
        @param index_mode 索引模式，支持 flat、ivf、lsh。
        @return 索引摘要。
        """
        self._ensure_faiss()
        normalized_mode = str(index_mode or "flat").strip().lower()
        if normalized_mode not in FAISS_INDEX_MODES:
            raise ValueError(f"FAISS 不支持索引模式: {index_mode}")

        entries, matrix, vector_dimension = self._entries_from_embeddings(embeddings_data)
        if not entries:
            raise ValueError("嵌入文件没有可索引的向量。")

        normalized_matrix = self._normalize_matrix(matrix).astype("float32")
        faiss_index, parameters = self._build_faiss_index(normalized_matrix, normalized_mode)

        sample_metadata = entries[0].get("metadata", {})
        payload = {
            "name": collection_name,
            "database": "faiss",
            "index_mode": normalized_mode,
            "index_family": "faiss",
            "created_at": datetime.now().isoformat(),
            "filename": embeddings_data.get("filename", ""),
            "document_name": embeddings_data.get("document_name") or embeddings_data.get("filename", ""),
            "embedding_provider": embeddings_data.get("embedding_provider", ""),
            "embedding_model": embeddings_data.get("embedding_model", ""),
            "vector_dimension": vector_dimension,
            "dataset_type": sample_metadata.get("dataset_type"),
            "source_role": sample_metadata.get("source_role"),
            "source_format": sample_metadata.get("source_format"),
            "count": len(entries),
            "entries": entries,
            "index": {
                "type": normalized_mode,
                "parameters": parameters,
            },
        }

        faiss.write_index(faiss_index, str(self._binary_path(collection_name)))
        self._metadata_path(collection_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "database": "faiss",
            "index_size": len(entries),
            "collection_name": collection_name,
            "index_mode": normalized_mode,
            "index_family": "faiss",
            "index_parameters": parameters,
        }

    def list_indexes(self) -> List[Dict[str, Any]]:
        """! @brief 列出 FAISS 索引摘要。"""
        indexes = []
        for path in sorted(self.index_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            indexes.append(self._summary_from_payload(payload))
        return indexes

    def exists(self, collection_id: str) -> bool:
        """! @brief 判断集合名是否对应 FAISS 索引。"""
        return self._metadata_path(collection_id).exists() and self._binary_path(collection_id).exists()

    def delete_index(self, collection_id: str) -> bool:
        """! @brief 删除 FAISS 索引文件和元数据。"""
        deleted = False
        for path in [self._metadata_path(collection_id), self._binary_path(collection_id)]:
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    def get_info(self, collection_id: str) -> Dict[str, Any]:
        """! @brief 读取 FAISS 索引详情。"""
        payload = self.load_metadata(collection_id)
        return {
            "name": payload.get("name", collection_id),
            "num_entities": payload.get("count", len(payload.get("entries", []))),
            "schema": {
                "database": "faiss",
                "index_mode": payload.get("index_mode"),
                "index_family": payload.get("index_family", "faiss"),
                "vector_dimension": payload.get("vector_dimension"),
                "embedding_provider": payload.get("embedding_provider"),
                "embedding_model": payload.get("embedding_model"),
                "index_parameters": (payload.get("index") or {}).get("parameters", {}),
            },
        }

    def load_metadata(self, collection_id: str) -> Dict[str, Any]:
        """! @brief 读取 FAISS 索引元数据。"""
        path = self._metadata_path(collection_id)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index metadata not found: {collection_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_embeddings(self, collection_id: str) -> Dict[str, Any]:
        """! @brief 返回 FAISS 索引中的原始向量条目。"""
        payload = self.load_metadata(collection_id)
        entries = payload.get("entries", [])
        return {
            "collection_id": collection_id,
            "count": len(entries),
            "embeddings": [
                {
                    "embedding": entry.get("embedding", []),
                    "metadata": {
                        **(entry.get("metadata") or {}),
                        "document_name": payload.get("document_name") or payload.get("filename") or collection_id,
                        "chunk_id": str((entry.get("metadata") or {}).get("chunk_id") or entry.get("id")),
                        "total_chunks": len(entries),
                        "content": entry.get("document", ""),
                        "embedding_model": payload.get("embedding_model", ""),
                        "embedding_provider": payload.get("embedding_provider", ""),
                        "vector_dimension": payload.get("vector_dimension", 0),
                        "index_mode": payload.get("index_mode"),
                        "index_family": payload.get("index_family", "faiss"),
                    },
                }
                for entry in entries
            ],
        }

    def sample_metadata(self, collection_id: str) -> Dict[str, Any]:
        """! @brief 返回 FAISS 索引用于生成查询向量的配置。"""
        payload = self.load_metadata(collection_id)
        return {
            "embedding_provider": payload.get("embedding_provider"),
            "embedding_model": payload.get("embedding_model"),
            "dataset_type": payload.get("dataset_type"),
            "source_role": payload.get("source_role"),
            "document_name": payload.get("document_name"),
            "index_mode": payload.get("index_mode"),
            "index_family": payload.get("index_family", "faiss"),
            "database": "faiss",
        }

    def search(
        self,
        collection_id: str,
        query_embedding: List[float],
        top_k: int,
        threshold: float,
        word_count_threshold: int,
    ) -> Dict[str, Any]:
        """! @brief 在 FAISS 索引上执行向量检索。
        @param collection_id FAISS 索引集合名。
        @param query_embedding 查询向量。
        @param top_k 返回数量。
        @param threshold 最低相似度分数。
        @param word_count_threshold 最低词数。
        @return 检索命中和算法说明。
        """
        self._ensure_faiss()
        payload = self.load_metadata(collection_id)
        entries = payload.get("entries", [])
        if not entries:
            return {
                "results": [],
                "score_algorithm": self._score_algorithm(payload),
                "index_mode": payload.get("index_mode"),
                "index_family": payload.get("index_family", "faiss"),
            }

        dimension = int(payload.get("vector_dimension") or len(entries[0].get("embedding", [])))
        query_values = [float(value) for value in query_embedding[:dimension]]
        if len(query_values) < dimension:
            query_values.extend([0.0] * (dimension - len(query_values)))
        normalized_query = self._normalize_vector(np.array(query_values, dtype="float32")).astype("float32")

        index = faiss.read_index(str(self._binary_path(collection_id)))
        if hasattr(index, "nprobe"):
            index.nprobe = int(((payload.get("index") or {}).get("parameters") or {}).get("nprobe") or 1)

        mode = payload.get("index_mode", "flat")
        candidate_count = self._candidate_count(mode, len(entries), top_k)
        distances, indexes = index.search(normalized_query.reshape(1, -1), candidate_count)
        candidate_indexes = [
            int(index_value)
            for index_value in indexes[0].tolist()
            if int(index_value) >= 0 and int(index_value) < len(entries)
        ]

        if mode == "lsh":
            matrix = np.array([entry.get("embedding", [])[:dimension] for entry in entries], dtype="float32")
            normalized_matrix = self._normalize_matrix(matrix).astype("float32")
            scored = [
                (entry_index, float(normalized_matrix[entry_index] @ normalized_query))
                for entry_index in dict.fromkeys(candidate_indexes)
            ]
        else:
            scored = [
                (int(index_value), float(score))
                for index_value, score in zip(indexes[0].tolist(), distances[0].tolist())
                if int(index_value) >= 0 and int(index_value) < len(entries)
            ]

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        results = []
        for entry_index, score in ranked:
            entry = entries[entry_index]
            metadata = entry.get("metadata") or {}
            word_count = int(metadata.get("word_count") or 0)
            if float(score) < threshold or word_count < word_count_threshold:
                continue
            results.append({
                "text": entry.get("document", ""),
                "score": float(score),
                "metadata": {
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key not in {"answer_quality", "content"} and value is not None
                    },
                    "source": metadata.get("document_name") or payload.get("document_name"),
                    "page": metadata.get("page_number"),
                    "chunk": entry.get("id"),
                    "total_chunks": metadata.get("total_chunks") or len(entries),
                    "page_range": metadata.get("page_range"),
                    "embedding_provider": payload.get("embedding_provider"),
                    "embedding_model": payload.get("embedding_model"),
                    "index_mode": payload.get("index_mode"),
                    "index_family": payload.get("index_family", "faiss"),
                    "database": "faiss",
                },
            })
            if len(results) >= top_k:
                break

        return {
            "results": results,
            "score_algorithm": self._score_algorithm(payload),
            "index_mode": payload.get("index_mode"),
            "index_family": payload.get("index_family", "faiss"),
        }

    def _build_faiss_index(self, normalized_matrix: np.ndarray, index_mode: str):
        """! @brief 构建指定类型的 FAISS index 对象。"""
        sample_count, dimension = normalized_matrix.shape
        contiguous_matrix = np.ascontiguousarray(normalized_matrix.astype("float32"))
        if index_mode == "ivf":
            if sample_count < 2:
                index = faiss.IndexFlatIP(dimension)
                index.add(contiguous_matrix)
                return index, {
                    "nlist": 1,
                    "nprobe": 1,
                    "metric": "inner_product_on_normalized_vectors",
                    "note": "样本数少于 2 时使用 FAISS Flat 承载 IVF 演示集合，避免不可训练的倒排簇。",
                }
            nlist = min(sample_count, max(1, round(sample_count ** 0.5)))
            nprobe = min(2, nlist)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(contiguous_matrix)
            index.nprobe = nprobe
            index.add(contiguous_matrix)
            return index, {
                "nlist": nlist,
                "nprobe": nprobe,
                "metric": "inner_product_on_normalized_vectors",
            }
        if index_mode == "lsh":
            nbits = min(64, max(8, int(np.ceil(np.log2(max(sample_count, 2)))) * 4))
            index = faiss.IndexLSH(dimension, nbits)
            index.add(contiguous_matrix)
            return index, {
                "nbits": nbits,
                "candidate_factor": 8,
                "metric": "lsh_hamming_candidates_cosine_rerank",
            }

        index = faiss.IndexFlatIP(dimension)
        index.add(contiguous_matrix)
        return index, {"metric": "inner_product_on_normalized_vectors"}

    def _summary_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """! @brief 将完整索引转换为列表摘要。"""
        return {
            "id": payload.get("name"),
            "name": payload.get("name"),
            "count": payload.get("count", len(payload.get("entries", []))),
            "database": "faiss",
            "index_mode": payload.get("index_mode"),
            "index_family": payload.get("index_family", "faiss"),
            "dataset_type": payload.get("dataset_type"),
            "source_role": payload.get("source_role"),
            "document_name": payload.get("document_name"),
            "embedding_provider": payload.get("embedding_provider"),
            "embedding_model": payload.get("embedding_model"),
        }

    def _metadata_path(self, collection_id: str) -> Path:
        """! @brief 将集合名映射到 FAISS metadata 路径。"""
        return self.index_dir / f"{self._safe_collection_stem(collection_id)}.json"

    def _binary_path(self, collection_id: str) -> Path:
        """! @brief 将集合名映射到 FAISS 二进制索引路径。"""
        return self.index_dir / f"{self._safe_collection_stem(collection_id)}.faiss"

    @staticmethod
    def _safe_collection_stem(collection_id: str) -> str:
        """! @brief 清理集合名，避免路径穿越。"""
        safe_name = Path(str(collection_id or "")).name
        if safe_name.endswith(".json"):
            safe_name = safe_name[:-5]
        if safe_name.endswith(".faiss"):
            safe_name = safe_name[:-6]
        return safe_name

    @staticmethod
    def _entries_from_embeddings(embeddings_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
        """! @brief 从嵌入文件提取索引条目和矩阵。"""
        entries = []
        vectors = []
        for index, item in enumerate(embeddings_data.get("embeddings", []), 1):
            vector = [float(value) for value in item.get("embedding", [])]
            if not vector:
                continue
            metadata = item.get("metadata") or {}
            entry_id = str(metadata.get("chunk_id") or index)
            entries.append({
                "id": entry_id,
                "document": str(metadata.get("content", "")),
                "metadata": {
                    **metadata,
                    "document_name": embeddings_data.get("filename", ""),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                },
                "embedding": vector,
            })
            vectors.append(vector)

        if not vectors:
            return [], np.empty((0, 0), dtype="float32"), 0
        dimension = min(len(vector) for vector in vectors)
        matrix = np.array([vector[:dimension] for vector in vectors], dtype="float32")
        for entry in entries:
            entry["embedding"] = entry["embedding"][:dimension]
            entry["metadata"]["vector_dimension"] = dimension
        return entries, matrix, dimension

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """! @brief 对矩阵按行做 L2 归一化。"""
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms > _EPSILON, norms, 1.0)
        return matrix / safe_norms

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """! @brief 对单个向量做 L2 归一化。"""
        norm = float(np.linalg.norm(vector))
        if norm <= _EPSILON:
            return np.zeros_like(vector, dtype="float32")
        return vector / norm

    @staticmethod
    def _candidate_count(index_mode: str, sample_count: int, top_k: int) -> int:
        """! @brief 根据索引类型决定 FAISS 返回候选数量。"""
        if sample_count <= 0:
            return 0
        if index_mode == "flat":
            return sample_count
        return min(sample_count, max(top_k * 8, top_k, 1))

    @staticmethod
    def _score_algorithm(payload: Dict[str, Any]) -> Dict[str, str]:
        """! @brief 返回前端展示用分数算法说明。"""
        index_mode = payload.get("index_mode", "flat")
        labels = {
            "flat": "FAISS Flat cosine",
            "ivf": "FAISS IVF cosine",
            "lsh": "FAISS LSH candidates + cosine rerank",
        }
        notes = {
            "flat": "FAISS IndexFlatIP 在归一化向量上等价于 cosine，精确遍历全部向量。",
            "ivf": "FAISS IndexIVFFlat 先用倒排簇筛候选，再返回归一化向量内积，score 越大越相关。",
            "lsh": "FAISS IndexLSH 先按二进制签名召回候选，再由后端按 cosine 对候选重排，score 越大越相关。",
        }
        return {
            "name": labels.get(index_mode, f"FAISS {index_mode}"),
            "formula": "score = cosine(query, chunk)",
            "note": notes.get(index_mode, "FAISS 检索结果统一展示为越大越相关的 score。"),
        }

    @staticmethod
    def _ensure_faiss() -> None:
        """! @brief 确保当前环境已安装 FAISS。"""
        if faiss is None:
            raise RuntimeError("当前环境没有安装 faiss-cpu，请先安装 backend/requirements.txt 中的 faiss-cpu。")

    @staticmethod
    def _configure_faiss_runtime() -> None:
        """! @brief 限制 FAISS 本地线程，避免 macOS OpenMP 冲突影响演示。"""
        return None
