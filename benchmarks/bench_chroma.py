#!/usr/bin/env python3
"""! @file bench_chroma.py
@brief Chroma HNSW profiles 与 NumpyFlat 精确基线 benchmark。
@details 阶段 A 默认读取 sample_data/course_qa_public.json 的公开候选答案，
不会读取 answer_quality。阶段 B 可通过 --corpus paper_chunks --data-path
切换到论文 chunk fixture，并复用同一套 recall/latency 统计逻辑。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag_core.contracts.enums import BlockType
from rag_core.contracts.models import Chunk, EmbeddingVector
from rag_core.vector_indexes import CHROMA_HNSW_PROFILES, ChromaHnswIndex, NumpyFlatIndex

DEFAULT_COURSE_QA_PATH = REPO_ROOT / "sample_data" / "course_qa_public.json"
RECALL_KS = (3, 5, 10)
TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class BenchmarkFixture:
    """! @brief benchmark 使用的 chunks 与查询集合。"""

    corpus_name: str
    chunks: list[Chunk]
    queries: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QueryRun:
    """! @brief 单次查询的延迟和命中 ID。"""

    latency_ms: float
    hit_ids: list[str]


def main() -> int:
    """! @brief CLI 入口。"""
    args = parse_args()
    fixture = load_fixture(args)
    embeddings = embed_chunks(fixture.chunks, dim=args.dim)
    baseline_index = NumpyFlatIndex()
    build_index(baseline_index, fixture.chunks, embeddings)
    baseline_runs = run_queries(baseline_index, fixture.queries, dim=args.dim, top_k=max(RECALL_KS))
    selected_profiles = resolve_requested_profiles(args.profile)

    results = []
    for profile_name in selected_profiles:
        result = benchmark_profile(profile_name, fixture, embeddings, baseline_runs, args)
        results.append(result)

    payload = {
        "benchmark": "chroma_hnsw_profiles",
        "corpus": fixture.corpus_name,
        "corpus_metadata": fixture.metadata,
        "embedding": {"provider": "local_hashing", "dim": args.dim},
        "recall_baseline": "numpy_flat",
        "recall_ks": list(RECALL_KS),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    """! @brief 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Benchmark Chroma HNSW profiles against NumpyFlat exact baseline.")
    parser.add_argument(
        "--profile",
        default="all",
        help="all、numpy_flat 或具体 Chroma HNSW profile 名称。",
    )
    parser.add_argument(
        "--corpus",
        choices=["course_qa", "paper_chunks"],
        default="course_qa",
        help="阶段 A 默认 course_qa；阶段 B 可切换到 paper_chunks fixture。",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_COURSE_QA_PATH, help="corpus JSON 路径。")
    parser.add_argument("--persist-dir", type=Path, default=None, help="可选 Chroma 持久化目录。")
    parser.add_argument("--dim", type=int, default=64, help="本地 hashing embedding 维度。")
    parser.add_argument("--max-questions", type=int, default=None, help="仅 course_qa 使用的最大问题数。")
    parser.add_argument("--max-chunks", type=int, default=None, help="可选最大 chunk 数，用于快速手动 benchmark。")
    return parser.parse_args()


def load_fixture(args: argparse.Namespace) -> BenchmarkFixture:
    """! @brief 按 corpus 类型加载 benchmark fixture。"""
    if args.corpus == "course_qa":
        return load_course_qa_fixture(args.data_path, max_questions=args.max_questions, max_chunks=args.max_chunks)
    return load_paper_chunks_fixture(args.data_path, max_chunks=args.max_chunks)


def load_course_qa_fixture(
    dataset_path: Path,
    max_questions: int | None,
    max_chunks: int | None,
) -> BenchmarkFixture:
    """! @brief 从课程 QA public 数据构造阶段 A 小型 benchmark fixture。"""
    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if raw_data.get("dataset") != "course_qa_public":
        raise ValueError("阶段 A benchmark 只能默认读取 sample_data/course_qa_public.json")

    chunks: list[Chunk] = []
    queries: list[str] = []
    item_count = 0
    for item in raw_data.get("items", []):
        if max_questions is not None and item_count >= max_questions:
            break
        category = str(item.get("category", "")).strip()
        question = str(item.get("question", "")).strip()
        if not category or not question:
            continue
        item_count += 1
        queries.append(question)
        qa_id = int(item.get("qa_id", item_count))
        for answer in item.get("answers", []):
            text = str(answer.get("answer", "")).strip()
            if not text:
                continue
            answer_id = str(answer.get("answer_id") or stable_id("ans", category, qa_id, text))
            chunk_text = f"课程主题：{category}\n问题：{question}\n候选答案：{text}"
            chunks.append(
                Chunk(
                    chunk_id=stable_id("course-qa-chunk", category, qa_id, answer_id),
                    doc_id=stable_id("course-qa-doc", category, qa_id),
                    text=chunk_text,
                    source=display_path(dataset_path),
                    block_ids=[],
                    block_types=[BlockType.TEXT],
                    token_count=max(1, len(tokenize(chunk_text))),
                    metadata={
                        "category": category,
                        "qa_id": qa_id,
                        "question": question,
                        "answer_id": answer_id,
                        "section_path": [category],
                        "page_numbers": [],
                    },
                )
            )
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    if not chunks or not queries:
        raise ValueError("课程 QA public 数据没有可 benchmark 的候选答案或查询")
    return BenchmarkFixture(
        corpus_name="course_qa_public",
        chunks=chunks,
        queries=queries,
        metadata={"question_count": len(queries), "chunk_count": len(chunks), "data_path": display_path(dataset_path)},
    )


def load_paper_chunks_fixture(dataset_path: Path, max_chunks: int | None) -> BenchmarkFixture:
    """! @brief 加载阶段 B 论文 chunks fixture 的预留入口。"""
    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = raw_data.get("chunks", raw_data if isinstance(raw_data, list) else [])
    queries = raw_data.get("queries", []) if isinstance(raw_data, dict) else []
    chunks: list[Chunk] = []
    for index, row in enumerate(rows, start=1):
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunk_id = str(row.get("chunk_id") or stable_id("paper-chunk", index, text))
        doc_id = str(row.get("doc_id") or "paper-fixture")
        source = str(row.get("source") or dataset_path)
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("section_path", row.get("section_path", []))
        metadata.setdefault("page_numbers", row.get("page_numbers", []))
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                source=source,
                block_ids=list(row.get("block_ids", [])),
                block_types=[BlockType.TEXT],
                token_count=max(1, len(tokenize(text))),
                metadata=metadata,
            )
        )

    query_texts = [str(item.get("query", item)).strip() if isinstance(item, dict) else str(item).strip() for item in queries]
    query_texts = [query for query in query_texts if query]
    if not query_texts:
        query_texts = [chunk.text[:120] for chunk in chunks[: min(10, len(chunks))]]
    if not chunks or not query_texts:
        raise ValueError("论文 chunks fixture 没有可 benchmark 的 chunks 或 queries")
    return BenchmarkFixture(
        corpus_name="paper_chunks",
        chunks=chunks,
        queries=query_texts,
        metadata={"query_count": len(query_texts), "chunk_count": len(chunks), "data_path": display_path(dataset_path)},
    )


def benchmark_profile(
    profile_name: str,
    fixture: BenchmarkFixture,
    embeddings: list[EmbeddingVector],
    baseline_runs: list[QueryRun],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """! @brief 对单个 profile 运行构建、查询和 recall 统计。"""
    if profile_name == "numpy_flat":
        index = NumpyFlatIndex()
        profile_params: dict[str, Any] = {"type": "exact_cosine_baseline"}
    else:
        collection_name = f"rag-demo-{profile_name.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
        index = ChromaHnswIndex(profile=profile_name, collection_name=collection_name, persist_directory=args.persist_dir)
        profile = CHROMA_HNSW_PROFILES[profile_name]
        profile_params = {
            "space": profile.space,
            "M": profile.m,
            "construction_ef": profile.construction_ef,
            "search_ef": profile.search_ef,
        }

    build_time_ms = build_index(index, fixture.chunks, embeddings)
    runs = run_queries(index, fixture.queries, dim=args.dim, top_k=max(RECALL_KS))
    metrics = summarize_runs(runs, baseline_runs)
    return {
        "profile": profile_name,
        "profile_params": profile_params,
        "build_time_ms": round(build_time_ms, 3),
        "p50_latency_ms": round(metrics["p50_latency_ms"], 3),
        "p95_latency_ms": round(metrics["p95_latency_ms"], 3),
        "recall@3": round(metrics["recall@3"], 4),
        "recall@5": round(metrics["recall@5"], 4),
        "recall@10": round(metrics["recall@10"], 4),
        "query_count": len(runs),
    }


def build_index(index: Any, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> float:
    """! @brief 构建索引并返回耗时毫秒。"""
    started_at = time.perf_counter()
    index.upsert(chunks, embeddings)
    return (time.perf_counter() - started_at) * 1000.0


def run_queries(index: Any, queries: list[str], dim: int, top_k: int) -> list[QueryRun]:
    """! @brief 运行全部查询并记录延迟与命中 ID。"""
    runs: list[QueryRun] = []
    for query in queries:
        query_embedding = embed_query(query, dim=dim)
        started_at = time.perf_counter()
        hits = index.search(query_embedding, top_k=top_k)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        runs.append(QueryRun(latency_ms=latency_ms, hit_ids=[hit.chunk_id for hit in hits]))
    return runs


def summarize_runs(runs: list[QueryRun], baseline_runs: list[QueryRun]) -> dict[str, float]:
    """! @brief 汇总延迟分位数和相对 NumpyFlat 的 recall。"""
    latencies = [run.latency_ms for run in runs]
    summary: dict[str, float] = {
        "p50_latency_ms": percentile(latencies, 50.0),
        "p95_latency_ms": percentile(latencies, 95.0),
    }
    for k in RECALL_KS:
        recalls = [recall_at_k(run.hit_ids, baseline.hit_ids, k) for run, baseline in zip(runs, baseline_runs)]
        summary[f"recall@{k}"] = statistics.fmean(recalls) if recalls else 0.0
    return summary


def recall_at_k(candidate_ids: list[str], baseline_ids: list[str], k: int) -> float:
    """! @brief 计算单个查询相对精确基线的 recall@k。"""
    expected = baseline_ids[:k]
    if not expected:
        return 1.0
    actual = set(candidate_ids[:k])
    return len(actual.intersection(expected)) / len(expected)


def percentile(values: list[float], percent: float) -> float:
    """! @brief 不依赖 numpy 的线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def resolve_requested_profiles(profile_arg: str) -> list[str]:
    """! @brief 将 --profile 参数解析为待运行列表。"""
    if profile_arg == "all":
        return ["numpy_flat", *CHROMA_HNSW_PROFILES.keys()]
    if profile_arg == "numpy_flat":
        return ["numpy_flat"]
    if profile_arg in CHROMA_HNSW_PROFILES:
        return [profile_arg]
    names = ", ".join(["all", "numpy_flat", *CHROMA_HNSW_PROFILES.keys()])
    raise ValueError(f"Unknown --profile {profile_arg!r}; expected one of: {names}")


def embed_chunks(chunks: list[Chunk], dim: int) -> list[EmbeddingVector]:
    """! @brief 为 chunks 生成本地确定性 hashing embedding。"""
    return [
        EmbeddingVector(
            item_id=chunk.chunk_id,
            vector=embed_text(chunk.text, dim=dim),
            dim=dim,
            model="local-hashing",
            provider="benchmark",
            metadata={"doc_id": chunk.doc_id, "source": chunk.source},
        )
        for chunk in chunks
    ]


def embed_query(query: str, dim: int) -> EmbeddingVector:
    """! @brief 为查询生成与 chunks 同空间的 hashing embedding。"""
    return EmbeddingVector(
        item_id=stable_id("query", query),
        vector=embed_text(query, dim=dim),
        dim=dim,
        model="local-hashing",
        provider="benchmark",
        metadata={"query": query},
    )


def embed_text(text: str, dim: int) -> list[float]:
    """! @brief 通过 token hashing 生成单位长度向量。"""
    if dim <= 0:
        raise ValueError("dim must be positive")
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], byteorder="big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def tokenize(text: str) -> list[str]:
    """! @brief 切分英文、数字和中文片段。"""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    """! @brief 基于内容生成稳定短 ID。"""
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def display_path(path: Path) -> str:
    """! @brief 优先以仓库相对路径展示数据来源。"""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
