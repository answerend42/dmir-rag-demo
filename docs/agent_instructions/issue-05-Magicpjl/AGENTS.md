# Issue #5 Agent 指示：Research paper parser and chunker

你正在为 RAG Demo 项目完成 GitHub Issue #5。目标是增强研究论文解析和分块，让 parser 输出 `ParsedDocument`，chunker 保留 page/section/table/caption 元数据。课程 QA 是阶段 A 的默认链路；目标新论文和干扰论文是本 Issue 的阶段 B，不另开新 Issue。

## Owner

当前集中收口 Owner：`answerend42`。

原模块负责人：`Magicpjl`。

## 必读文档

- `docs/agent_rules.md`
- `docs/interfaces.md`
- `docs/agent_instructions/issue-01-answerend42/AGENTS.md`

## ALLOWED_PATHS

- `backend/rag_core/parsers/`
- `backend/rag_core/chunkers/`
- `tests/contract/`
- `tests/unit/`
- `sample_data/`
- `scripts/parse_paper_sample.py`
- `docs/agent_instructions/issue-05-Magicpjl/AGENTS.md`

## 硬性限制

1. 不得修改 embedding、index、generation 内部实现。
2. OCR/Docling 只能作为可选增强，不得阻塞 Basic RAG。
3. 输出必须对齐 `ParsedDocument`、`ContentBlock`、`Chunk` contract。
4. chunk metadata 必须尽量保留 `page_numbers`、`section_path`、`block_type`。
5. 代码注释必须使用中文 Doxygen 风格。
6. 大体积论文 PDF 不要直接提交到仓库；优先提交 metadata、小样例或下载说明。

## 实施顺序

1. 读取 #1 的 `DocumentParser`、`Chunker` Protocol。
2. 第一阶段默认数据是 `sample_data/course_qa_public.json`，PDF/论文解析不得阻塞默认 QA 流水线。
3. 当前 main 收口目标是 Markdown/PDF parser skeleton 与 `ResearchPaperChunker`；默认论文 corpus 已固定为 `sample_data/papers/llm_wiki_retrieval_as_reasoning.md`。
4. parser/chunker 输出必须保留 `page_numbers`、`section_path`、`block_type`。
5. 用小样例 Markdown fixture 写 contract tests。
6. 对复杂版式/OCR 标记可选路线，不作为普通 PR 的硬依赖。
7. 真实目标论文和干扰论文清单已写入 `sample_data/papers/paper_eval_fixture.json`；继续保持 parser/chunker 能支持该 corpus 和最小 `demo_research_paper.md` fixture。

## 验收命令

```shell
python -m compileall backend
pytest tests/contract tests/unit -m "not integration and not benchmark"
```

## 当前 PDF 能力边界

阶段 A 主要支持 Markdown / OCR 后文本，PDF 路径仅作最小可用兜底：

- [`backend/rag_core/parsers/research_paper.py`](../../../backend/rag_core/parsers/research_paper.py) 的 `_parse_pdf` 用 PyMuPDF 抽取每页纯文本，再用 `[page N]` 包裹后走 Markdown parser。
- 仅覆盖纯文字、单列、可选简单表格的 PDF；多列布局、扫描件、复杂版式、嵌入图表的语义全部不保证。
- 未安装 PyMuPDF 时抛 `ProviderUnavailable`，不阻塞 Markdown / fake pipeline。
- 阶段 B 真实论文一律先转 digest Markdown 或 OCR 后 Markdown 再入仓库；大体积 / 复杂 PDF 不进仓库。

## PR 输出

PR 中必须写明：

- parser 方法与 fallback。
- 输出的 block 类型。
- chunk metadata 示例。
- 目标论文 digest 与干扰论文 metadata 的接入格式。
- 复杂 PDF/OCR 尚未覆盖的风险。
- `scripts/parse_paper_sample.py` 在默认 corpus 上的输出摘要。
