# RAG Demo Contract 与接口说明

v0.1.0 | 2026-06-03

本文件定义一周冲刺期间各模块必须遵守的边界。contracts 不是业务逻辑，而是多人并行开发的协议。

## 目标流水线

```text
upload/load -> parse -> chunk -> embed -> index -> search -> generate -> evaluate/display
```

所有模块通过 Pydantic models 或 `model_dump()` 后的 schema 交互。禁止裸 `dict` 成为跨模块接口。

## P0 参考实现入口

第一版契约与 fake pipeline 已经落在以下位置，后续真实 adapter 应以这些文件为边界：

- `backend/rag_core/contracts/models.py`：Pydantic 契约模型。
- `backend/rag_core/contracts/protocols.py`：parser、chunker、embedder、vector index、generator Protocol。
- `backend/rag_core/testing/fakes.py`：离线 mock parser/chunker/embedder/index/generator。
- `backend/rag_core/pipeline/orchestrator.py`：本地 fake RAG 流水线。
- `scripts/run_smoke_pipeline.py`：P0 冒烟脚本。
- `tests/contract/`：契约测试。

P0 验收命令：

```shell
python -m compileall backend/rag_core scripts/run_smoke_pipeline.py
pytest tests/contract
python scripts/run_smoke_pipeline.py --mode fake --pretty
```

## 默认测试数据

第一阶段默认输入不是论文语料，而是老师要求的课程 QA 数据：

- RAG 可见输入：`sample_data/course_qa_public.json`
- 评测专用标签：`eval/labels/course_qa_quality_labels.json`

`course_qa_public.json` 只包含课程主题、问题、候选答案和不含语义的 `answer_id`。候选答案顺序已稳定打散，不包含 0-9 质量档次。

`course_qa_quality_labels.json` 只允许评测脚本在模型生成完成后读取。禁止把 `answer_quality` 放进：

- RAG 索引
- LLM prompt
- `RagAnswer.retrieved_hits`
- `RagAnswer.trace`
- 前端展示字段

## 核心模型

| 模型 | 作用 | 必须字段/约束 |
| --- | --- | --- |
| `ParsedDocument` | Parser 输出 | `contract_version`, `doc_id`, `title`, `markdown`, `blocks`, `parser_name`, `metadata` |
| `ContentBlock` | 半结构化内容单位 | `block_id`, `block_type`, `text`, `page_number`, `bbox`, `metadata` |
| `Chunk` | 检索最小单位 | `chunk_id`, `doc_id`, `text`, `source`, `block_ids`, `block_types`, `token_count`, `metadata` |
| `EmbeddingVector` | 向量输出 | `item_id`, `vector`, `dim`, `model`, `provider`, `metadata` |
| `SearchHit` | 检索命中 | `chunk_id`, `doc_id`, `text`, `score`, `rank`, `source`, `metadata` |
| `RagRequest` | 统一问答输入 | `query`, `rag_mode`, `top_k`, `collection_id`, `model`, `provider`, `require_citations` |
| `RagAnswer` | 统一问答输出 | `contract_version`, `answer_markdown`, `citations`, `retrieved_hits`, `trace`, `warnings` |
| `StageTrace` | 每层过程记录 | `stage_name`, `latency_ms`, `input_summary`, `output_summary`, `artifacts` |

## 语义硬约束

- `SearchHit.score` 必须越大越相关；Chroma distance、loss、rank score 等只能在 adapter 内转换。
- `doc_id`、`chunk_id` 应用 hash 或稳定规则生成，避免每次运行随机变化。
- 所有 chunk 和 hit 必须携带 `doc_id`、`page_numbers`、`section_path`，否则不能进入 optimized RAG。
- 每个阶段都要记录 trace，前端和评测只从 contract 字段读取数据。
- 缺模型、空文档、空检索、维度不匹配、API 超时等异常要统一包装为项目错误类型。

## 建议目录

```text
backend/rag_core/contracts/
  __init__.py
  models.py       # ParsedDocument, Chunk, SearchHit, RagRequest, RagAnswer...
  protocols.py    # DocumentParser, Embedder, VectorIndex, Generator...
  enums.py        # ParserType, EmbeddingProvider, IndexBackend, RagMode...
  errors.py       # ContractViolation, ProviderUnavailable, EmptyCorpus...
```

## Protocol 草案

```python
class DocumentParser(Protocol):
    name: str

    def parse(self, file_path: str) -> ParsedDocument: ...


class Chunker(Protocol):
    name: str

    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...


class Embedder(Protocol):
    name: str

    def embed(self, chunks: list[Chunk]) -> list[EmbeddingVector]: ...


class VectorIndex(Protocol):
    name: str

    def upsert(self, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> None: ...
    def search(self, query_embedding: EmbeddingVector, top_k: int) -> list[SearchHit]: ...


class Generator(Protocol):
    name: str

    def generate(self, request: RagRequest, contexts: list[SearchHit]) -> RagAnswer: ...
```

## `/rag/answer` 阶段 A 主链路

#8 阶段 A 已固定最小集成入口：

```http
POST /rag/answer
```

请求体使用 `RagRequest` 兼容 JSON。阶段 A 默认读取 `sample_data/course_qa_public.json`，
只支持 `provider=mock` 与 `model=mock-generator`，用于让前端和评测先接统一接口。
真实 Qwen embedding / LLM / Chroma adapter 仍由 #2/#3/#4 独立接入，不阻塞这个主链路。

最小请求示例：

```json
{
  "query": "什么是自然语言处理？",
  "rag_mode": "basic_rag",
  "top_k": 3,
  "collection_id": "course-qa-default",
  "provider": "mock",
  "model": "mock-generator",
  "require_citations": true,
  "metadata": {}
}
```

本地不启动 uvicorn 的 smoke 命令：

```shell
python scripts/run_rag_answer_smoke.py --pretty
```

隐藏标签规则不变：`answer_quality` 禁止进入 `/rag/answer` 请求、RAG 索引、prompt、
retrieved hits、trace 和前端展示；只允许 #7 评测脚本在生成完成后读取。

## 论文 parser/chunker 阶段 B 入口

#5 已提供轻量论文解析和分块入口：

- `backend/rag_core/parsers/research_paper.py`
- `backend/rag_core/chunkers/research_paper.py`
- `sample_data/papers/demo_research_paper.md`
- `sample_data/papers/paper_eval_fixture.json`

当前 parser 优先支持 Markdown fixture；PDF 路径通过 PyMuPDF 懒加载，缺依赖时不会阻塞
课程 QA 默认链路。`ResearchPaperChunker` 输出仍使用 `Chunk` contract，并在 metadata 中保留：

- `page_numbers`
- `section_path`
- `block_type`
- `parser_name`
- `line_start` / `line_end`

真实目标论文确定后，应替换 `sample_data/papers/` 下的 metadata、corpus 和 QA/evidence，
而不是重新设计接口。

## 离线评测结果端点

#7 评测脚本入口：

```shell
python scripts/run_eval.py --dataset-type course_qa --modes all --limit 5
python scripts/run_eval.py --dataset-type paper --modes all --limit 3
```

脚本输出：

- `eval/results/course_qa_eval.json`
- `eval/results/course_qa_eval.csv`
- `eval/results/course_qa_eval.md`
- `eval/results/paper_eval.json`
- `eval/results/paper_eval.csv`
- `eval/results/paper_eval.md`

后端提供只读结果端点：

```http
GET /eval/results/course_qa_eval.json
```

前端 dashboard 只读取 JSON 摘要。`answer_quality` 字段不得进入这些前端可读结果；
评测脚本只允许在生成完成后读取 labels，并输出不含隐藏字段名的汇总分布。

## `/rag/answer` 输出目标

前端、评测脚本和展示都应最终读取同一个 `RagAnswer`：

```json
{
  "contract_version": "0.1.0",
  "answer_markdown": "带引用的 Markdown 答案",
  "citations": [
    {
      "doc_id": "paper-ucosa",
      "chunk_id": "paper-ucosa:12",
      "page_number": 4,
      "section_path": ["Method", "UCOSA"],
      "quote": "短证据片段"
    }
  ],
  "retrieved_hits": [],
  "trace": [],
  "warnings": []
}
```

## 三模式对比

| 模式 | 作用 | 展示重点 |
| --- | --- | --- |
| `llm_only` | 不使用检索，直接问模型 | 展示新论文专有概念容易答错或编造 |
| `basic_rag` | dense top-k 检索 + 直接生成 | 展示能回答基本定义 |
| `optimized_rag` | query rewrite/rerank/context packing/grounded prompt | 展示答案更完整、引用更稳、可拒答 |
