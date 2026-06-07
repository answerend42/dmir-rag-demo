"""! @file 基于阿里云 DashScope API 的 Qwen Embedding 适配器。"""

import os
import time
from typing import List, Optional
import requests
from rag_core.contracts import EmbeddingVector, Embedder


class QwenApiEmbedder(Embedder):
    """通义千问 Embedding API 调用器。"""

    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_base: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        batch_size: int = 8,
        timeout: int = 30,
    ):
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("QwenApiEmbedder 需要 API key。请设置环境变量 DASHSCOPE_API_KEY 或 QWEN_API_KEY。")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.api_base = api_base
        self.batch_size = batch_size
        self.timeout = timeout
        self.provider = "qwen_api"

    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """批量调用 API 生成向量。"""
        if not texts:
            return []
        results = []
        global_offset = 0
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._call_api(batch, start_index=global_offset)
            results.extend(batch_results)
            global_offset += len(batch)
        return results

    def embed(self, text: str) -> EmbeddingVector:
        """单条文本嵌入。"""
        return self.embed_batch([text])[0]

    def _call_api(self, texts: List[str], start_index: int = 0) -> List[EmbeddingVector]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"text_type": "document"},
        }
        start = time.perf_counter()
        resp = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        data = resp.json()
        emb_map = {}
        for emb_item in data.get("output", {}).get("embeddings", []):
            idx = emb_item.get("text_index")
            vec = emb_item.get("embedding")
            if idx is not None and vec is not None:
                emb_map[idx] = vec

        # 校验返回完整性
        expected_indices = set(range(len(texts)))
        returned_indices = set(emb_map.keys())
        if expected_indices != returned_indices:
            missing = expected_indices - returned_indices
            raise RuntimeError(f"API 返回的 embedding 缺少以下索引: {missing}")

        results = []
        for i, text in enumerate(texts):
            vec = emb_map[i]
            dim = len(vec)
            results.append(EmbeddingVector(
                item_id=f"{self.provider}_{start_index + i}",
                vector=vec,
                dim=dim,
                model=self.model,
                provider=self.provider,
                metadata={
                    "api_latency_ms": round(elapsed_ms, 2),
                    "batch_size": len(texts),
                    "position_in_batch": i,
                }
            ))
        return results