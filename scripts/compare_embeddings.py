#!/usr/bin/env python3
"""! @file compare_embeddings.py
@brief 对比不同 embedding 模式的输出摘要。
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from rag_core.embeddings import MockEmbedder, QwenApiEmbedder, QwenLocalEmbedder
from rag_core.contracts.errors import ProviderUnavailable


def cosine_similarity(v1, v2):
    """! @brief 计算余弦相似度，若维度不一致则抛出 ValueError。"""
    if len(v1) != len(v2):
        raise ValueError(f"向量维度不一致: {len(v1)} vs {len(v2)}")
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sqrt(sum(a * a for a in v1))
    n2 = sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def main():
    """! @brief 解析命令行参数并输出 embedding 摘要。"""
    parser = argparse.ArgumentParser(description="对比 Embedder 输出")
    parser.add_argument("--mode", choices=["mock", "api", "local"], default="mock",
                        help="选择 embedder 模式")
    parser.add_argument("--texts", nargs="+", default=[
        "机器学习是人工智能的一个分支。",
        "深度学习使用多层神经网络。",
    ], help="要嵌入的文本列表")
    parser.add_argument("--dim", type=int, default=384, help="Mock 模式下的向量维度")
    parser.add_argument("--input", type=str, help="阶段 B 预留：JSON 文件路径，包含 'chunks' 数组")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            texts = data.get("chunks", args.texts)
    else:
        texts = args.texts

    if args.mode == "mock":
        embedder = MockEmbedder(dim=args.dim)
    elif args.mode == "api":
        try:
            embedder = QwenApiEmbedder()
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
    else:
        embedder = QwenLocalEmbedder()

    print(f"使用 {args.mode} 嵌入器 (provider={embedder.provider}, model={getattr(embedder, 'model_name', getattr(embedder, 'model', 'unknown'))})")
    print(f"嵌入 {len(texts)} 条文本...")

    try:
        results = embedder.embed_batch(texts)
    except ProviderUnavailable as e:
        print(f"本地模型不可用: {e}")
        sys.exit(1)

    print("\n--- 对比摘要 ---")
    for i, r in enumerate(results):
        print(f"{i}: item_id={r.item_id}, dim={r.dim}, provider={r.provider}, model={r.model}")
        print(f"   向量前3个值: {r.vector[:3]} ...")
        print(f"   metadata = {r.metadata}")
        print()

    if len(results) >= 2:
        try:
            sim = cosine_similarity(results[0].vector, results[1].vector)
            print(f"前两条文本的余弦相似度: {sim:.4f}")
        except ValueError as e:
            print(f"无法计算相似度: {e}")


if __name__ == "__main__":
    main()
