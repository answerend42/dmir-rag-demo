"""! @file config.py
@brief 后端共享配置常量和枚举。
"""

from enum import Enum
from typing import Dict, Any

class VectorDBProvider(str, Enum):
    """! @brief 支持的向量数据库提供方标识。"""
    MILVUS = "milvus",
    CHROMA = "chroma"
    FAISS = "faiss"
    # 后续可继续添加更多提供方

# 可以在这里添加其他配置相关的内容
MILVUS_CONFIG = {
    "uri": "myrag",
    "index_types": {
        "flat": "FLAT",
        "ivf_flat": "IVF_FLAT",
        "ivf_sq8": "IVF_SQ8",
        "hnsw": "HNSW"
    },
    "index_params": {
        "flat": {},
        "ivf_flat": {"nlist": 1024},
        "ivf_sq8": {"nlist": 1024},
        "hnsw": {
            "M": 16,
            "efConstruction": 500
        }
    }
}
