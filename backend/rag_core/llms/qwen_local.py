"""! @file qwen_local.py
@brief 基于本地 HuggingFace 模型的 Qwen generator skeleton。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from rag_core.contracts.errors import ProviderUnavailable
from rag_core.llms.base import BaseQwenGenerator, CompleteFn

DEFAULT_QWEN_LOCAL_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_QWEN_LOCAL_MODEL_ENV = "QWEN_LOCAL_MODEL"


def build_qwen_local_complete_fn(
    model_name: str = DEFAULT_QWEN_LOCAL_MODEL,
    *,
    generate_text: Callable[[str, str], str] | None = None,
) -> CompleteFn:
    """! @brief 构造本地 Qwen 文本补全函数，便于测试注入 mock generate_text。
    @param model_name HuggingFace 模型名称或本地路径。
    @param generate_text 可选注入 `(model_name, prompt) -> text` 函数。
    @return 接收 prompt 并返回模型文本的函数。
    @throws ProviderUnavailable 本地模型不可加载时抛出。
    """
    if generate_text is not None:
        return lambda prompt: generate_text(model_name, prompt).strip()

    def complete(prompt: str) -> str:
        raise ProviderUnavailable(
            "Qwen 本地模型不可用：请在运行时注入 generate_text，或设置本地模型加载逻辑"
        )

    return complete


class QwenLocalGenerator(BaseQwenGenerator):
    """! @brief 通过本地 Qwen 模型生成 grounded 回答。"""

    name = "qwen-local-generator"

    def __init__(
        self,
        model: str | None = None,
        *,
        complete_fn: CompleteFn | None = None,
        generate_text: Callable[[str, str], str] | None = None,
        min_score: float | None = None,
    ):
        resolved_model = model or os.getenv(DEFAULT_QWEN_LOCAL_MODEL_ENV) or DEFAULT_QWEN_LOCAL_MODEL
        resolved_complete = complete_fn or build_qwen_local_complete_fn(
            model_name=resolved_model,
            generate_text=generate_text,
        )
        kwargs: dict[str, Any] = {
            "name": self.name,
            "complete_fn": resolved_complete,
            "provider": "qwen_local",
            "model": resolved_model,
        }
        if min_score is not None:
            kwargs["min_score"] = min_score
        super().__init__(**kwargs)
