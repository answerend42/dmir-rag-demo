# RAG Demo 一周冲刺任务看板

v1.1 | 2026-06-03

## 本周目标

本周目标不是把所有想法都做完，而是确保结课展示可运行、可解释、可量化。

| 优先级 | 定义 | 本周必须做到什么 |
| --- | --- | --- |
| P0 | 系统可运行 | fake pipeline、Basic RAG、CI、固定 demo 数据必须可用 |
| P1 | 满足老师要求 | 课程 QA 默认数据、新论文资料准备、API/local 模型、React 功能、PDF 解析、Chroma 索引、检索优化 |
| P2 | 展示质量 | trace、citation、benchmark、课程 QA 与论文评测报告、对比表 |
| P3 | 扩展优化 | 更多模型、更多 OCR 方案、UI 美化 |

## 每日节奏

| 时间 | 目标 | 负责人 | 验收/关卡 |
| --- | --- | --- | --- |
| D0：今天 | 冻结标准与任务 | `answerend42` | 合并 docs、Issue/PR 模板；开 8 个任务 issue |
| D1 | contracts + fake pipeline | `answerend42` | Pydantic models、Protocol、MockParser/MockEmbedder/NumpyFlat/MockGenerator；CI 跑 contract tests |
| D2 | 各模块 adapter 雏形 | 全员 | 每人只在自己目录提交 adapter skeleton + contract test + fixture |
| D3 | Basic RAG 端到端 | `answerend42` + 各模块负责人 | 上传 PDF -> 解析 -> chunk -> embedding -> Chroma -> search -> generate -> markdown 展示 |
| D4 | 优化与基准 | `KeeperHihi`, `cheng1608`, `Ryan-137` | Chroma HNSW profile benchmark；query rewrite/rerank/context packing；三模式评测脚本 |
| D5 | 数据与前端展示 | `Magicpjl`, `yourskenny`, `Ryan-137` | 课程 QA public/labels、trace panel、eval dashboard，固定 5 个课程 QA 演示问题；同步确定目标新论文、干扰论文和论文 QA/evidence 计划 |
| D6 | 锁定 demo | 全员 | 只修 bug 不扩功能；生成最终报告表；每人准备自己负责模块的 1 分钟说明 |
| D7：展示前 | 排练与兜底 | `answerend42` + 全员 | 录屏/截图/本地缓存模型与索引；准备 fallback 问题和离线结果 |

## 第一批 GitHub Issues

每个 Issue 都有对应的 Agent 指示文档，可直接复制给 Codex、Claude Code 或 Cursor Agent：

- #1: [answerend42/AGENTS.md](agent_instructions/issue-01-answerend42/AGENTS.md)
- #2: [KeeperHihi/AGENTS.md](agent_instructions/issue-02-KeeperHihi/AGENTS.md)
- #3: [irishibi/AGENTS.md](agent_instructions/issue-03-irishibi/AGENTS.md)
- #4: [cheng1608/AGENTS.md](agent_instructions/issue-04-cheng1608/AGENTS.md)
- #5: [Magicpjl/AGENTS.md](agent_instructions/issue-05-Magicpjl/AGENTS.md)
- #6: [yourskenny/AGENTS.md](agent_instructions/issue-06-yourskenny/AGENTS.md)
- #7: [Ryan-137/AGENTS.md](agent_instructions/issue-07-Ryan-137/AGENTS.md)
- #8: [answerend42/AGENTS.md](agent_instructions/issue-08-answerend42/AGENTS.md)

### 1. Freeze contracts and fake RAG pipeline

- Owner: `answerend42`
- ALLOWED_PATHS: `backend/rag_core/contracts/`, `backend/rag_core/testing/`, `backend/rag_core/pipeline/`, `tests/contract/`, `scripts/run_smoke_pipeline.py`, `docs/interfaces.md`
- 工作内容：新增 Pydantic models、Protocol、MockParser、MockEmbedder、NumpyFlat、MockGenerator 和 smoke pipeline。
- Definition of Done：`pytest tests/contract` 通过；`python scripts/run_smoke_pipeline.py --mode fake` 输出 `RagAnswer`；`docs/interfaces.md` 更新。
- 风险控制：不要依赖真实模型、真实 Chroma 或网络。

### 2. Chroma HNSW profiles and vector benchmark

- Owner: `KeeperHihi`
- ALLOWED_PATHS: `backend/rag_core/vector_indexes/`, `benchmarks/`, `tests/contract/`, `tests/unit/`
- 工作内容：实现 `chroma_hnsw_fast`、`chroma_hnsw_balanced`、`chroma_hnsw_high_recall`、`NumpyFlat` exact；统一 `SearchHit.score`。
- Definition of Done：`bench_chroma.py` 输出 `build_time`、p50/p95 latency、recall@3/5/10；与 NumpyFlat 对齐。
- 风险控制：不要声称 Chroma 支持 Milvus 式多算法；用 HNSW profile 表述。

### 3. Qwen embedding API/local adapters

- Owner: `irishibi`
- ALLOWED_PATHS: `backend/rag_core/embeddings/`, `tests/contract/`, `tests/unit/`, `scripts/compare_embeddings.py`
- 工作内容：实现 `QwenApiEmbedder`、`QwenLocalEmbedder`、`MockEmbedder`；支持批处理和维度记录。
- Definition of Done：contract test 验证向量数量、维度、metadata；`compare_embeddings.py` 可跑小数据。
- 风险控制：unit test 不调用真实 API；API key 只读环境变量。

### 4. Qwen LLM API/local and optimized generation

- Owner: `cheng1608`
- ALLOWED_PATHS: `backend/rag_core/llms/`, `backend/rag_core/retrieval/`, `backend/rag_core/generation/`, `tests/contract/`, `tests/unit/`
- 工作内容：实现 `QwenApiGenerator`、`QwenLocalGenerator`、query rewrite、grounded prompt、citation formatting。
- Definition of Done：同一 `RagRequest` 可切换 provider；输出 `answer_markdown` + `citations` + `warnings`。
- 风险控制：禁止把缺证据问题直接自由发挥；optimized mode 必须可拒答。

### 5. Research paper parser and chunker

- Owner: 当前集中收口 `answerend42`；原模块负责人 `Magicpjl`
- ALLOWED_PATHS: `backend/rag_core/parsers/`, `backend/rag_core/chunkers/`, `tests/contract/`, `tests/unit/`, `sample_data/`
- 工作内容：Parser 输出 `ParsedDocument`；新增 PDF->Markdown 路线、`research_paper_chunker`。
- Definition of Done：样例 PDF 能输出 markdown、blocks、chunks；chunk 保留 page/section/block_type。
- 风险控制：OCR/Docling 作为可选增强，不阻塞 Basic RAG。

### 6. Frontend trace/config/eval dashboard

- Owner: `yourskenny`
- ALLOWED_PATHS: `frontend/src/components/rag/`, `frontend/src/pages/`, `frontend/src/config/`, `frontend/src/**/*.test.*`
- 工作内容：新增 `MarkdownAnswer`、`RetrievalTracePanel`、`PipelineConfigPanel`、`EvaluationDashboard`。
- Definition of Done：`npm run build` 通过；可显示 retrieved_hits、citations、trace、三模式评测表。
- 风险控制：前端不得依赖后端私有文件结构，只读 `RagAnswer` schema。

### 7. Course QA and paper evaluation report

- Owner: 当前集中收口 `answerend42`；原模块负责人 `Ryan-137`
- ALLOWED_PATHS: `eval/`, `scripts/run_eval.py`, `sample_data/`, `docs/`
- 工作内容：阶段 A 基于 `sample_data/course_qa_public.json` 和 `eval/labels/course_qa_quality_labels.json` 实现课程 QA 评测报告；阶段 B 确定目标新论文、相关干扰论文、论文 QA、evidence 标注和三模式评测报告。
- Definition of Done：`run_eval.py` 输出 JSON/CSV/Markdown；至少 5 个课程 QA 现场展示问题稳定；论文阶段有目标论文 metadata、干扰论文清单、20-30 个论文问题与 evidence 标注方案。
- 风险控制：`answer_quality` 只允许评测脚本在生成后读取，禁止进入 RAG 索引、prompt、trace 或前端展示。

### 8. Integration and demo lock

- Owner: `answerend42`
- ALLOWED_PATHS: `backend/`, `frontend/`, `scripts/`, `eval/`, `docs/`
- 工作内容：统一 `/rag/answer`；固定模型、数据、索引、问题；生成最终报告。
- Definition of Done：展示前 24 小时只修 bug；准备录屏、截图、离线结果 fallback。
- 风险控制：任何 P3 扩展不得影响主线 demo。

## 固定演示问题建议

第一阶段演示问题先从课程 QA 数据中选；论文阶段已经固定为 LLM-Wiki 论文，完整 26 个问题见 `sample_data/papers/paper_eval_fixture.json` 和 `eval/results/paper_eval.md`。

| 问题类型 | 示例问题 | 期望现象 |
| --- | --- | --- |
| 课程定义 | 什么是自然语言处理？ | RAG 应命中自然语言处理课程 QA 候选答案 |
| 方法对比 | 监督学习和无监督学习有什么区别？ | RAG 应给出带标签/无标签、任务目标差异 |
| 模型机制 | 激活函数为什么对神经网络重要？ | RAG 应检索神经网络核心问答 |
| 数据结构 | 哈希表解决冲突的常用方法有哪些？ | RAG 应检索数据结构课程问答 |
| 程序设计 | C++ 中虚函数有什么作用？ | RAG 应检索 C++ 高级程序设计问答 |

## 固定论文演示问题

| 问题类型 | 示例问题 | 期望现象 |
| --- | --- | --- |
| 元信息易混 | 这篇论文的作者和单位是什么？为什么 Feifei Li 容易被模型混淆？ | LLM-only 容易编作者或混淆同名作者；RAG 应命中 WeChat, Tencent Inc., Beijing, China |
| 工具接口 | wiki_search(query) 和 wiki_read(paths) 分别返回什么、用于什么？ | RAG 应命中工具接口章节 |
| Error Book | Error Book 的五阶段生命周期按顺序是什么？ | RAG 应给出 Discover、Attribute、Constrain、Inject、Verify & Close |
| 反例纠偏 | AuthTrace 上 LLM-Wiki 是否四列都超过 HippoRAG 2？ | RAG 应指出 Single-doc 上 HippoRAG 2 更高，其余多文档/总体 LLM-Wiki 更高 |
| 消融结论 | 消融实验里去掉哪个组件影响最大？为什么？ | RAG 应指出 w/o Progressive Traversal 降幅最大 |
