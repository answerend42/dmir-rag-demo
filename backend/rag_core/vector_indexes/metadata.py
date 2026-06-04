"""! @file metadata.py
@brief 向量索引 metadata 安全处理工具。
"""

from __future__ import annotations

from typing import Any

FORBIDDEN_METADATA_KEYS = {"answer_quality"}


def strip_forbidden_metadata(value: Any) -> Any:
    """! @brief 递归移除禁止进入 RAG 索引和命中的 metadata 字段。"""
    if isinstance(value, dict):
        return {
            key: strip_forbidden_metadata(item)
            for key, item in value.items()
            if key not in FORBIDDEN_METADATA_KEYS
        }
    if isinstance(value, list):
        return [strip_forbidden_metadata(item) for item in value]
    return value
