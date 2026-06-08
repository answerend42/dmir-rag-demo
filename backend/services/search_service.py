"""! @file search_service.py
@brief 向量集合发现、语义检索和搜索结果持久化。
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility
from services.embedding_service import EmbeddingService
from services.projection_service import VectorProjectionService
from utils.config import VectorDBProvider, MILVUS_CONFIG
import os
import json
from pymilvus import MilvusClient, exceptions
import chromadb

chromadb_path = "./03-vector-store/chromadb"

logger = logging.getLogger(__name__)


class SearchService:
    """! @brief 查询向量集合并持久化检索结果。

    搜索服务类，负责向量数据库的连接和向量搜索功能
    提供集合列表查询、向量相似度搜索和搜索结果保存等功能
    """

    def __init__(self):
        """
        初始化搜索服务
        创建嵌入服务实例，设置Milvus连接URI，初始化搜索结果保存目录
        """
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.search_results_dir = "04-search-results"
        os.makedirs(self.search_results_dir, exist_ok=True)
        self.client=chromadb.PersistentClient(chromadb_path)

    def get_providers(self) -> List[Dict[str, str]]:
        """
        获取支持的向量数据库列表

        返回:
            List[Dict[str, str]]: 支持的向量数据库提供商列表
        """
        return [
            #     {"id": VectorDBProvider.MILVUS.value, "name": "Milvus"}
            {"id": VectorDBProvider.CHROMA.value, "name": "chroma"}
        ]

    def list_collections(self, provider: str = VectorDBProvider.CHROMA.value) -> List[Dict[str, Any]]:
        """! @brief 获取指定向量数据库中的所有集合。
        @param provider 向量数据库提供方，支持 chroma 或 milvus。
        @return 集合信息列表，包含 id、name 和 count。
        """
        try:
            provider_value = str(provider).strip().lower()
            logger.info(f"List collections for provider: {provider_value}")

            if provider_value == VectorDBProvider.MILVUS.value:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                try:
                    return [
                        {"id": name, "name": name, "count": 0}
                        for name in utility.list_collections()
                    ]
                finally:
                    connections.disconnect("default")

            if provider_value != VectorDBProvider.CHROMA.value:
                return []

            collections = []
            collection_names = self.client.list_collections()

            for sample in collection_names:
                name = sample.name if hasattr(sample, "name") else str(sample)
                try:
                    collection = self.client.get_or_create_collection(name)
                    sample_metadata = {}
                    try:
                        sample_metadata = self._get_sample_metadata(collection)
                    except Exception:
                        sample_metadata = {}
                    collections.append({
                        "id": name,
                        "name": name,
                        "count": collection.count() if hasattr(collection, "count") else 0,
                        "dataset_type": sample_metadata.get("dataset_type"),
                        "source_role": sample_metadata.get("source_role"),
                        "document_name": sample_metadata.get("document_name") or sample_metadata.get("source"),
                        "embedding_provider": sample_metadata.get("embedding_provider"),
                        "embedding_model": sample_metadata.get("embedding_model"),
                    })
                except Exception as e:
                    logger.error(f"Error getting info for collection {name}: {str(e)}")

            return collections

        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise

    def save_search_results(self, query: str, collection_id: str, results: List[Dict[str, Any]]) -> str:
        """
        保存搜索结果到JSON文件

        参数:
            query (str): 搜索查询文本
            collection_id (str): 集合ID
            results (List[Dict[str, Any]]): 搜索结果列表

        返回:
            str: 保存文件的路径

        异常:
            Exception: 保存文件时发生错误
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 使用集合ID的基础名称（去掉路径相关字符）
            collection_base = os.path.basename(collection_id)
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = os.path.join(self.search_results_dir, filename)

            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }

            logger.info(f"Saving search results to: {filepath}")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved search results to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise

    @staticmethod
    def _coerce_vector(vector: Any) -> List[float]:
        """! @brief 将 Chroma 或 numpy 风格向量转换为普通浮点数组。
        @param vector 原始向量对象。
        @return 可 JSON 序列化的浮点数组。
        """
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector or []]

    @staticmethod
    def _get_sample_metadata(collection) -> Dict[str, Any]:
        """! @brief 从 Chroma 集合读取一条已入库 metadata，用于恢复 embedding 配置。
        @param collection Chroma collection 对象。
        @return 第一条非空 metadata。
        @throws ValueError 集合为空或没有 embedding 配置时抛出。
        """
        sample = collection.get(limit=1, include=["metadatas"])
        metadatas = sample.get("metadatas") or []
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if not metadatas:
            raise ValueError("集合为空或缺少 metadata。")

        metadata = metadatas[0]
        if not metadata.get("embedding_provider") or not metadata.get("embedding_model"):
            raise ValueError("集合 metadata 缺少 embedding_provider 或 embedding_model。")
        return metadata

    def get_collection_embeddings(self, collection_id: str) -> Dict[str, Any]:
        """! @brief 读取 Chroma collection 中的全部向量，用于数值查看。
        @param collection_id Chroma collection 名称。
        @return 包含 collection 元信息和 embedding 条目的响应。
        """
        collection = self.client.get_collection(collection_id)
        raw_data = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = raw_data.get("ids") or []
        documents = raw_data.get("documents") or []
        metadatas = raw_data.get("metadatas") or []
        raw_embeddings = raw_data.get("embeddings")
        if raw_embeddings is None:
            raw_embeddings = []
        if hasattr(raw_embeddings, "tolist"):
            raw_embeddings = raw_embeddings.tolist()

        embeddings = []
        for index, vector in enumerate(raw_embeddings):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document_text = documents[index] if index < len(documents) else ""
            chunk_id = metadata.get("chunk_id") or metadata.get("chunk") or (ids[index] if index < len(ids) else index + 1)
            page_number = metadata.get("page_number") or metadata.get("page") or ""
            embeddings.append({
                "embedding": self._coerce_vector(vector),
                "metadata": {
                    **metadata,
                    "document_name": metadata.get("document_name") or metadata.get("source") or collection_id,
                    "chunk_id": str(chunk_id),
                    "total_chunks": len(raw_embeddings),
                    "content": document_text,
                    "page_number": page_number,
                    "page_range": metadata.get("page_range", page_number),
                    "embedding_model": metadata.get("embedding_model", ""),
                    "embedding_provider": metadata.get("embedding_provider", ""),
                    "embedding_timestamp": metadata.get("embedding_timestamp", ""),
                    "vector_dimension": len(vector),
                },
            })

        return {
            "collection_id": collection_id,
            "count": len(embeddings),
            "embeddings": embeddings,
        }

    def get_collection_projection(
        self,
        collection_id: str,
        method: str = "tsne",
        overlays: Optional[List[Dict[str, Any]]] = None,
        target_dimensions: int = 3,
    ) -> Dict[str, Any]:
        """! @brief 读取 Chroma collection 并在后端计算二维投影。
        @param collection_id Chroma collection 名称。
        @param method 投影方法。
        @param overlays 附加向量，例如查询向量。
        @param target_dimensions 输出维度，只支持 2 或 3。
        @return 投影响应。
        """
        payload = self.get_collection_embeddings(collection_id)
        return VectorProjectionService.project_embeddings(
            payload.get("embeddings", []),
            method=method,
            overlays=overlays or [],
            source_id=collection_id,
            target_dimensions=target_dimensions,
        )

    async def search(self,
                     query: str,
                     collection_id: str,
                     top_k: int = 3,
                     threshold: float = 0.3,
                     word_count_threshold: int = 0,
                     save_results: bool = False,
                     include_query_embedding: bool = False) -> Dict[str, Any]:
        """! @brief 执行向量搜索，并可选择持久化结果。
        @param query 用户查询文本。
        @param collection_id 向量库中的集合标识。
        @param top_k 返回的最大命中数量。
        @param threshold 保留结果的最低相似度分数。
        @param word_count_threshold 为保持 API 兼容而保留的最小字数过滤阈值。
        @param save_results 是否持久化处理后的搜索命中。
        @param include_query_embedding 是否在响应中回传查询向量，供前端可视化。
        @return 包含处理后结果和可选 saved_filepath 的搜索响应。

        执行向量搜索

        参数:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            top_k (int): 返回的最大结果数量，默认为3
            threshold (float): 相似度阈值，低于此值的结果将被过滤，默认为0.3
            word_count_threshold (int): 文本字数阈值，低于此值的结果将被过滤，默认为0
            save_results (bool): 是否保存搜索结果，默认为False

        返回:
            Dict[str, Any]: 包含搜索结果的字典，如果保存结果则包含保存路径

        异常:
            Exception: 搜索过程中发生错误
        """
        try:
            # 添加参数日志
            logger.info(f"Search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results} (type: {type(save_results)})")

            logger.info(
                f"Starting search with parameters - Collection: {collection_id}, Query: {query}, Top K: {top_k}")

            # 连接到 Chroma
            # 获取collection
            logger.info(f"Loading collection: {collection_id}")

            collection = self.client.get_collection(collection_id)
            # 记录collection的基本信息
            num_entities=collection.count()
            logger.info(f"Collection info - Entities: {num_entities}")

            logger.info(f"query: {query}")

            sample_metadata = self._get_sample_metadata(collection)

            # 使用collection中存储的配置创建查询向量
            logger.info("Creating query embedding")
            query_embedding = self.embedding_service.create_single_embedding(
                query,
                provider=sample_metadata.get('embedding_provider'),
                model=sample_metadata.get('embedding_model')
            )
            logger.info(f"Query embedding created with dimension: {len(query_embedding)}")

            results =collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            logger.info(f"Sample query results: {results.get('documents')[0][0]}")

            # 处理结果
            processed_results = []
            results_count=len(results['ids'][0])
            logger.info(f"Raw search results count: {results_count}")

            for hit in range(results_count):
                hit_score=1-results['distances'][0][hit]
                raw_metadata = results['metadatas'][0][hit] or {}
                word_count = int(raw_metadata.get('word_count') or 0)
                logger.info(f"Processing hit - Score: {hit_score}, Word Count: {word_count}")
                if hit_score >= threshold and word_count >= word_count_threshold:
                    passthrough_metadata = {
                        key: value
                        for key, value in raw_metadata.items()
                        if key not in {"answer_quality", "content"} and value is not None
                    }
                    processed_results.append({
                        "text": results.get('documents')[0][hit],
                        "score": float(hit_score),
                        "metadata": {
                            **passthrough_metadata,
                            "source": raw_metadata.get('document_name'),
                            "page": raw_metadata.get('page_number'),
                            "chunk": results.get('ids')[0][hit],
                            "total_chunks": raw_metadata.get('total_chunks'),
                            "page_range": raw_metadata.get('page_range'),
                            "embedding_provider": raw_metadata.get('embedding_provider'),
                            "embedding_model": raw_metadata.get('embedding_model'),
                            "embedding_timestamp": raw_metadata.get('embedding_timestamp')
                        }
                    })



            # 连接到 Milvus
            #logger.info(f"Connecting to Milvus at {self.milvus_uri}")
            #connections.connect(
            #    alias="default",
            #    uri=self.milvus_uri
            #)



            # 获取collection
            # logger.info(f"Loading collection: {collection_id}")
            #collection = Collection(collection_id)
            #collection.load()

            # 记录collection的基本信息
            # logger.info(f"Collection info - Entities: {collection.num_entities}")

            # 执行搜索
            # logger.info("Querying sample entity")
            # sample_entity = collection.query(
            #    expr="id >= 0",
            #    output_fields=["embedding_provider", "embedding_model"],
            #    limit=1
            # )

            #
            # if not sample_entity:
            #     logger.error(f"Collection {collection_id} is empty")
            #     raise ValueError(f"Collection {collection_id} is empty")
            #
            # logger.info(f"Sample entity configuration: {sample_entity[0]}")
            #
            # # 使用collection中存储的配置创建查询向量
            # logger.info("Creating query embedding")
            # query_embedding = self.embedding_service.create_single_embedding(
            #     query,
            #     provider=sample_entity[0]["embedding_provider"],
            #     model=sample_entity[0]["embedding_model"]
            # )
            # logger.info(f"Query embedding created with dimension: {len(query_embedding)}")
            #
            # # 执行搜索
            # search_params = {
            #     "metric_type": "COSINE",
            #     "params": {"nprobe": 10}
            # }
            # logger.info(f"Executing search with params: {search_params}")
            # logger.info(f"Word count threshold filter: word_count >= {word_count_threshold}")
            #
            # results = collection.search(
            #     data=[query_embedding],
            #     anns_field="vector",
            #     param=search_params,
            #     limit=top_k,
            #     expr=f"word_count >= {word_count_threshold}",
            #     output_fields=[
            #         "content",
            #         "document_name",
            #         "chunk_id",
            #         "total_chunks",
            #         "word_count",
            #         "page_number",
            #         "page_range",
            #         "embedding_provider",
            #         "embedding_model",
            #         "embedding_timestamp"
            #     ]
            # )

            # 处理结果
            # processed_results = []
            # logger.info(f"Raw search results count: {len(results[0])}")
            #
            # for hits in results:
            #     for hit in hits:
            #         logger.info(f"Processing hit - Score: {hit.score}, Word Count: {hit.entity.get('word_count')}")
            #         if hit.score >= threshold:
            #             processed_results.append({
            #                 "text": hit.entity.content,
            #                 "score": float(hit.score),
            #                 "metadata": {
            #                     "source": hit.entity.document_name,
            #                     "page": hit.entity.page_number,
            #                     "chunk": hit.entity.chunk_id,
            #                     "total_chunks": hit.entity.total_chunks,
            #                     "page_range": hit.entity.page_range,
            #                     "embedding_provider": hit.entity.embedding_provider,
            #                     "embedding_model": hit.entity.embedding_model,
            #                     "embedding_timestamp": hit.entity.embedding_timestamp
            #                 }
            #             })

            response_data = {"results": processed_results}
            if include_query_embedding:
                response_data["query_embedding"] = self._coerce_vector(query_embedding)
                response_data["query_embedding_metadata"] = {
                    "query": query,
                    "collection_id": collection_id,
                    "embedding_provider": sample_metadata.get("embedding_provider"),
                    "embedding_model": sample_metadata.get("embedding_model"),
                    "vector_dimension": len(query_embedding),
                }
                response_data["score_algorithm"] = {
                    "name": "Chroma HNSW cosine",
                    "formula": "score = 1 - Chroma distance",
                    "note": "Chroma distance 越小越相近；前端展示的 score 越大越相关。",
                }

            # 添加详细的保存逻辑日志
            logger.info(f"Preparing to handle save_results (flag: {save_results})")
            if save_results:
                logger.info("Save results is True, attempting to save...")
                if processed_results:
                    try:
                        filepath = self.save_search_results(query, collection_id, processed_results)
                        logger.info(f"Successfully saved results to: {filepath}")
                        response_data["saved_filepath"] = filepath
                    except Exception as e:
                        logger.error(f"Error saving results: {str(e)}")
                        response_data["save_error"] = str(e)
                        raise  # 添加这行来查看完整的错误堆栈
                else:
                    logger.info("No results to save")
            else:
                logger.info("Save results is False, skipping save")

            return response_data

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
        finally:
            connections.disconnect("default")
