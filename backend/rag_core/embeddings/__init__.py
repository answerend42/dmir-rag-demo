"""! @file embeddings 模块，提供多种 Embedder 实现。"""

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