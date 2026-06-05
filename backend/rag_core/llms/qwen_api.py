"""! @file qwen_api.py
@brief 基于阿里云百炼兼容 OpenAI 接口的 Qwen API generator。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from rag_core.contracts.errors import ProviderUnavailable
from rag_core.llms.base import BaseQwenGenerator, CompleteFn

DEFAULT_QWEN_API_MODEL = "qwen-turbo"
DEFAULT_QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"


def build_qwen_api_complete_fn(
    model: str = DEFAULT_QWEN_API_MODEL,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_QWEN_API_BASE_URL,
    client: Any | None = None,
) -> CompleteFn:
    """! @brief 构造 Qwen API 文本补全函数，便于测试注入 mock client。
    @param model 百炼模型名称。
    @param api_key API Key；为空时从环境变量读取。
    @param base_url OpenAI 兼容接口地址。
    @param client 可选注入客户端；为空时按 api_key 创建。
    @return 接收 prompt 并返回模型文本的函数。
    @throws ProviderUnavailable API Key 缺失或 client 创建失败时抛出。
    """
    resolved_key = api_key or os.getenv(DEFAULT_QWEN_API_KEY_ENV)
    if client is None and not resolved_key:
        raise ProviderUnavailable(
            f"Qwen API 不可用：请设置环境变量 {DEFAULT_QWEN_API_KEY_ENV}"
        )

    def complete(prompt: str) -> str:
        active_client = client or _create_openai_client(resolved_key, base_url)
        response = active_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是 RAG 演示助手，请严格遵循用户提示。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if not str(content).strip():
            raise ProviderUnavailable("Qwen API 返回空内容")
        return str(content).strip()

    return complete


def _create_openai_client(api_key: str, base_url: str) -> Any:
    """! @brief 延迟导入 OpenAI 客户端，避免测试环境强依赖。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProviderUnavailable("Qwen API 不可用：缺少 openai 依赖") from exc
    return OpenAI(api_key=api_key, base_url=base_url)


class QwenApiGenerator(BaseQwenGenerator):
    """! @brief 通过 Qwen API 生成 grounded 回答。"""

    name = "qwen-api-generator"

    def __init__(
        self,
        model: str = DEFAULT_QWEN_API_MODEL,
        *,
        complete_fn: CompleteFn | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_QWEN_API_BASE_URL,
        client: Any | None = None,
        min_score: float | None = None,
    ):
        resolved_complete = complete_fn or build_qwen_api_complete_fn(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
        )
        kwargs: dict[str, Any] = {
            "name": self.name,
            "complete_fn": resolved_complete,
            "provider": "qwen_api",
            "model": model,
        }
        if min_score is not None:
            kwargs["min_score"] = min_score
        super().__init__(**kwargs)
