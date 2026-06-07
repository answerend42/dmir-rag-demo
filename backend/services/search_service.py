"""! @file search_service.py
@brief 向量集合发现、语义检索和搜索结果持久化。
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility
from services.embedding_service import EmbeddingService
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
                    collections.append({
                        "id": name,
                        "name": name,
                        "count": collection.count() if hasattr(collection, "count") else 0,
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

    async def search(self,
                     query: str,
                     collection_id: str,
                     top_k: int = 3,
                     threshold: float = 0.3,
                     word_count_threshold: int = 0,
                     save_results: bool = False) -> Dict[str, Any]:
        """! @brief 执行向量搜索，并可选择持久化结果。
        @param query 用户查询文本。
        @param collection_id 向量库中的集合标识。
        @param top_k 返回的最大命中数量。
        @param threshold 保留结果的最低相似度分数。
        @param word_count_threshold 为保持 API 兼容而保留的最小字数过滤阈值。
        @param save_results 是否持久化处理后的搜索命中。
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
                word_count = int(results['metadatas'][0][hit].get('word_count') or 0)
                logger.info(f"Processing hit - Score: {hit_score}, Word Count: {word_count}")
                if hit_score >= threshold and word_count >= word_count_threshold:
                    processed_results.append({
                        "text": results.get('documents')[0][hit],
                        "score": float(hit_score),
                        "metadata": {
                            "source": results['metadatas'][0][hit].get('document_name'),
                            "page": results['metadatas'][0][hit].get('page_number'),
                            "chunk": results.get('ids')[0][hit],
                            "total_chunks": results['metadatas'][0][hit].get('total_chunks'),
                            "page_range": results['metadatas'][0][hit].get('page_range'),
                            "embedding_provider": results['metadatas'][0][hit].get('embedding_provider'),
                            "embedding_model": results['metadatas'][0][hit].get('embedding_model'),
                            "embedding_timestamp": results['metadatas'][0][hit].get('embedding_timestamp')
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
