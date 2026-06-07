"""! @file qwen_api.py
@brief 基于阿里云 DashScope API 的 Qwen embedding 适配器。
"""

from __future__ import annotations

import os
import time

import requests

from rag_core.contracts import Chunk, EmbeddingVector, Embedder


class QwenApiEmbedder(Embedder):
    """! @brief 通义千问 Embedding API 调用器。"""

    name = "qwen-api-embedder"

    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_base: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        batch_size: int = 8,
        timeout: int = 30,
    ):
        """! @brief 初始化 Qwen API embedding 适配器。
        @param model DashScope embedding 模型名。
        @param api_base DashScope embedding API 地址。
        @param batch_size 每批请求文本数量。
        @param timeout HTTP 超时时间，单位秒。
        @throws ValueError API key 缺失或数值参数非法时抛出。
        """
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("QwenApiEmbedder 需要 API key。请设置环境变量 DASHSCOPE_API_KEY 或 QWEN_API_KEY。")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if timeout <= 0:
            raise ValueError("timeout 必须为正整数")
        self.api_base = api_base
        self.batch_size = batch_size
        self.timeout = timeout
        self.provider = "qwen_api"

    def embed_batch(self, texts: list[str], item_ids: list[str] | None = None) -> list[EmbeddingVector]:
        """! @brief 批量调用 API 生成向量。
        @param texts 待嵌入文本列表。
        @param item_ids 可选外部稳定 ID；为空时按全局位置生成。
        @return 与输入文本一一对应的 EmbeddingVector 列表。
        @throws ValueError item_ids 数量与 texts 不一致时抛出。
        """
        if not texts:
            return []
        if item_ids is not None and len(item_ids) != len(texts):
            raise ValueError("item_ids 数量必须与 texts 一致")

        results: list[EmbeddingVector] = []
        global_offset = 0
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_item_ids = item_ids[i:i + self.batch_size] if item_ids else None
            batch_results = self._call_api(batch, start_index=global_offset, item_ids=batch_item_ids)
            results.extend(batch_results)
            global_offset += len(batch)
        return results

    def embed(self, chunks_or_text: list[Chunk] | str) -> list[EmbeddingVector] | EmbeddingVector:
        """! @brief 兼容 Chunk 列表契约和旧版单文本调用。"""
        if isinstance(chunks_or_text, str):
            return self.embed_batch([chunks_or_text])[0]
        return self.embed_batch(
            [chunk.text for chunk in chunks_or_text],
            item_ids=[chunk.chunk_id for chunk in chunks_or_text],
        )

    def embed_query(self, query: str) -> EmbeddingVector:
        """! @brief 为查询文本生成可用于检索的向量。"""
        return self._call_api([query], item_ids=["query_qwen_api_0"], text_type="query")[0]

    def _call_api(
        self,
        texts: list[str],
        start_index: int = 0,
        item_ids: list[str] | None = None,
        text_type: str = "document",
    ) -> list[EmbeddingVector]:
        """! @brief 调用 DashScope API 并转换为 EmbeddingVector。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"text_type": text_type},
        }
        start = time.perf_counter()
        resp = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        data = resp.json()
        emb_map: dict[int, list[float]] = {}
        for emb_item in data.get("output", {}).get("embeddings", []):
            idx = emb_item.get("text_index")
            vec = emb_item.get("embedding")
            if idx is not None and vec is not None:
                emb_map[idx] = vec

        expected_indices = set(range(len(texts)))
        returned_indices = set(emb_map.keys())
        if expected_indices != returned_indices:
            missing = expected_indices - returned_indices
            raise RuntimeError(f"API 返回的 embedding 缺少以下索引: {missing}")

        results: list[EmbeddingVector] = []
        for i, text in enumerate(texts):
            vec = emb_map[i]
            dim = len(vec)
            results.append(
                EmbeddingVector(
                    item_id=item_ids[i] if item_ids else f"{self.provider}_{start_index + i}",
                    vector=vec,
                    dim=dim,
                    model=self.model,
                    provider=self.provider,
                    metadata={
                        "api_latency_ms": round(elapsed_ms, 2),
                        "batch_size": len(texts),
                        "position_in_batch": i,
                        "text_length": len(text),
                        "text_type": text_type,
                    },
                )
            )
        return results
