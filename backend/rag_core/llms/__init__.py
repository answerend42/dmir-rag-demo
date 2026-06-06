"""! @file __init__.py
@brief Qwen LLM generator adapter 导出。
"""

from rag_core.llms.qwen_api import QwenApiGenerator
from rag_core.llms.qwen_local import QwenLocalGenerator

__all__ = ["QwenApiGenerator", "QwenLocalGenerator"]
