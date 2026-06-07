# 论文阶段样例数据

默认论文输入已经锁定为：

- 目标论文：Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki
- arXiv：2605.25480
- 版本日期：2026-05-26
- 本地 corpus：`llm_wiki_retrieval_as_reasoning.md`
- 评测 fixture：`paper_eval_fixture.json`

`llm_wiki_retrieval_as_reasoning.md` 是人工整理的中文结构化 digest，不是论文全文或 PDF。它保留页码、章节、关键表格数字和模型易混点，用于离线 parser、chunker、retrieval 和三模式评测。

`paper_eval_fixture.json` 包含：

- 目标论文 metadata、arXiv 链接和 PDF SHA-256。
- 4 篇相关干扰/背景论文 metadata。
- 26 个论文 QA 与 expected evidence 标注。

`demo_research_paper.md` 只保留给 parser/chunker 的最小 contract test 使用，不再作为默认论文评测目标。

运行完整论文评测：

```shell
python scripts/run_eval.py --dataset-type paper --modes all --limit 26 --pretty
```

大体积 PDF 不要直接提交到仓库；需要复核原文时从 `paper_eval_fixture.json` 中的 arXiv 链接重新下载。

## 阶段 B 论文接入格式

新增目标论文或干扰论文时，按下面三种模式择一接入；优先级从上到下，能用 digest Markdown 解决就不要走 OCR 与下载脚本路径。

### 1. Digest Markdown（默认）

- 用法：人工或 LLM 整理的中文结构化全文，保留 `[page N]` 页码、章节、表格关键数字、模型易混点。
- 范例：[`llm_wiki_retrieval_as_reasoning.md`](llm_wiki_retrieval_as_reasoning.md)。
- 命名：`{paper_slug}.md`，slug 用 ASCII 蛇形（如 `llm_wiki_retrieval_as_reasoning`）。
- 体积上限：≤ 50 KB。超过则改走 OCR 模式或下载脚本模式。
- 对应 fixture：在 [`paper_eval_fixture.json`](paper_eval_fixture.json) 的 `target_paper` / `distractors` 中挂 metadata + arXiv link + PDF SHA-256。

### 2. OCR 后 Markdown

- 用法：原文是非数字 PDF / 图表多的扫描件，先离线 OCR（Docling / unstructured / 任意 OCR 工具），再把识别结果整理成与 digest 同结构的 Markdown 提交。
- 命名：`{paper_slug}.ocr.md`；同篇论文允许 `{paper_slug}.md`（digest）和 `{paper_slug}.ocr.md`（OCR）共存。
- 体积上限：≤ 200 KB。超过则改走下载脚本模式。
- 风险：OCR 错误必须人工抽样核对，禁止直接拿生成结果当 ground truth。

### 3. metadata + 下载脚本

- 用法：原 PDF 体积大、license 受限或语义结构难以人工压缩到 50 KB digest。
- 仓库内只放：`paper_eval_fixture.json` 中的 metadata（含 arXiv 链接、`pdf_sha256`） + 下载脚本（如有，置于 `scripts/` 并在 PR 中标注）。
- 评测时：脚本根据 `pdf_sha256` 校验后写入本地缓存目录（不进 git），由 [`backend/rag_core/parsers/research_paper.py`](../../backend/rag_core/parsers/research_paper.py) 的 `_parse_pdf` 用 PyMuPDF 抽取。

### 不变量

- 仓库内不出现 `*.pdf` 文件。
- `demo_research_paper.md` 仅作为 parser/chunker 的 contract test fixture，禁止用于评测目标。
- `answer_quality` 等评测专用标签**禁止**进入这里的任何 markdown / JSON；它们只能放在 [`eval/labels/`](../../eval/labels/) 由评测脚本生成后读取。

## 当前 PDF 能力边界

[`backend/rag_core/parsers/research_paper.py`](../../backend/rag_core/parsers/research_paper.py) 的 `_parse_pdf` 路径仅作最小可用兜底：

- 用 PyMuPDF (`fitz`) 抽取每页纯文本，`[page N]` 包裹后送 Markdown parser。
- 仅覆盖纯文字、单列、可选简单表格的 PDF。
- 多列布局、扫描件、复杂版式、嵌入图表的语义都不保证；此类论文应走"OCR 后 Markdown"或"metadata + 下载脚本"模式。
- 未安装 PyMuPDF 时抛 `ProviderUnavailable`，不阻塞 Markdown / fake pipeline。

## 集成入口脚本

```shell
python scripts/parse_paper_sample.py --pretty
python scripts/parse_paper_sample.py --paper sample_data/papers/demo_research_paper.md --limit 3 --pretty
```

输出 `doc_id` / `block_count` / `chunk_count` / 各 BlockType 计数 / 前 N 个 chunk 的 `section_path` / `block_type` / `page_numbers` / `token_count`，方便 #8 集成同学不读测试文件就能调通 parser/chunker。
