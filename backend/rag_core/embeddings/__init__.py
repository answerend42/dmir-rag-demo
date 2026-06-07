"""! @file __init__.py
@brief embeddings 模块公开导出。
"""

from rag_core.contracts import EmbeddingVector, Embedder

from .mock import MockEmbedder
from .qwen_api import QwenApiEmbedder
from .qwen_local import QwenLocalEmbedder

__all__ = [
    "MockEmbedder",
    "QwenApiEmbedder",
    "QwenLocalEmbedder",
    "EmbeddingVector",
    "Embedder",
]
