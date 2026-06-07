# DMIR RAG Demo

信息检索课程结课实验项目。当前目标不是泛泛做一个 RAG 模板，而是按老师要求完成两个可展示任务：

1. **课程 QA 任务**：使用老师提供的上一项目 QA 数据，先跑通检索、生成、trace 和评测。
2. **新论文 RAG 任务**：同一批 Issue 内继续补充一篇大模型未充分掌握的新论文及相关干扰论文，展示 LLM-only 不足、Basic RAG 改善、Optimized RAG 进一步提升。

本仓库基于课程 RAG 框架模板整理而来，但现在已经进入 contract-first 的多人协作阶段：先冻结接口和默认数据，再由各成员并行实现 parser、embedding、index、generation、frontend 和 evaluation。

![RAG Frontend](images/RAG-fontend.png)

## 当前状态

- P0 已完成：`backend/rag_core/contracts/`、fake pipeline、contract tests 和 smoke pipeline 已进入 CI。
- #8 阶段 A 已固定最小 `/rag/answer` integration spine：先用课程 QA public 数据和 fake/mock pipeline 返回 `RagAnswer`。
- #5 已有离线论文 Markdown/PDF parser skeleton 与 `ResearchPaperChunker`，可先用小型 Markdown fixture 验证 page/section/table/caption metadata。
- #7 已有 `scripts/run_eval.py`，可输出课程 QA 与论文 fixture 的 JSON/CSV/Markdown 三模式评测摘要。
- 前端评测 dashboard 可通过后端只读端点 `/eval/results/course_qa_eval.json` 读取自动评测结果；文件缺失时仍保留 fallback。
- 第一阶段默认输入是课程 QA 数据：`sample_data/course_qa_public.json`。
- 课程 QA 的 0-9 档质量标签只保存在 `eval/labels/course_qa_quality_labels.json`，禁止进入 RAG 索引、prompt、trace 或前端展示。
- 新论文任务没有取消，已经并入 #5、#7、#8 和各模块的阶段 B；当前仓库先提供离线 fixture，真实目标论文、干扰论文和 20-30 个论文 QA 仍需替换进同一套格式。

## 快速验证

```shell
python -m compileall backend/rag_core scripts/run_smoke_pipeline.py
pytest tests/contract tests/unit eval/tests -m "not integration and not benchmark"
python scripts/run_smoke_pipeline.py --mode fake --pretty
python scripts/run_rag_answer_smoke.py --pretty
python scripts/run_eval.py --dataset-type course_qa --modes all --limit 5 --pretty
python scripts/run_eval.py --dataset-type paper --modes all --limit 3 --pretty
```

`run_smoke_pipeline.py` 默认读取 `sample_data/course_qa_public.json`，输出 `RagAnswer`，包含：

- `answer_markdown`
- `citations`
- `retrieved_hits`
- `trace`
- `warnings`

## 数据说明

| 路径 | 用途 | 是否允许进入 RAG |
| --- | --- | --- |
| `sample_data/course_qa_public.json` | 课程 QA 默认测试输入，只含主题、问题、候选答案、`answer_id` | 是 |
| `eval/labels/course_qa_quality_labels.json` | 课程 QA 质量档次标签，只供评测脚本生成报告 | 否 |
| `sample_data/papers/demo_research_paper.md` | 论文阶段 parser/chunker/eval 格式 fixture | 是 |
| `sample_data/papers/paper_eval_fixture.json` | 论文阶段 metadata、干扰论文和 QA/evidence 格式 fixture | 是 |
| `eval/results/*.json|csv|md` | `run_eval.py` 生成的离线评测摘要，前端只读 JSON | 否，属于生成后报告 |

质量档次隔离规则：

- `answer_quality` 只能由评测脚本在模型生成完成后读取。
- 不得把 `answer_quality` 放进检索索引、LLM prompt、`retrieved_hits`、`trace` 或前端展示。
- 候选答案在 public 数据中已稳定打散，不能用顺序推断质量。

## 项目结构

```text
backend/
  main.py                         # FastAPI 入口
  services/                       # 旧服务层，保留已有前后端流程
  rag_core/
    contracts/                    # Pydantic contracts 与 Protocol
    parsers/                      # 论文 Markdown/PDF parser skeleton
    chunkers/                     # 论文结构化 chunker
    embeddings/                   # Qwen API/local 与 mock embedding adapter
    testing/                      # fake adapters 与课程 QA loader
    pipeline/                     # fake RAG pipeline

frontend/
  src/                            # React/Vite 前端

sample_data/
  course_qa_public.json           # 第一阶段 RAG 默认输入
  papers/                         # 论文阶段小型 fixture

eval/
  labels/course_qa_quality_labels.json
  results/                        # run_eval.py 输出的前端可读摘要
  tests/                          # eval 脚本测试

scripts/
  run_smoke_pipeline.py           # P0 冒烟流水线
  run_rag_answer_smoke.py         # /rag/answer 接口冒烟脚本
  run_eval.py                     # 三模式离线评测脚本

tests/
  contract/                       # contract tests

docs/
  agent_rules.md                  # 工程协作与 AI Agent 约束
  interfaces.md                   # RAG contracts 与默认数据说明
  sprint_board.md                 # 一周冲刺任务看板
  agent_instructions/             # 每个 Issue 对应的 Agent 指示
```

## 协作入口

- [一周冲刺任务看板](docs/sprint_board.md)
- [工程协作与 AI Agent 约束规范](docs/agent_rules.md)
- [贡献与 PR 规则](docs/contribution.md)
- [Contract 与接口说明](docs/interfaces.md)
- [Agent 全局指示](AGENTS.md)
- [Claude Code 指示](CLAUDE.md)

所有成员开工前必须阅读对应 Issue 的 `docs/agent_instructions/.../AGENTS.md`。代码注释必须使用中文 Doxygen 风格；面向 LLM 的 prompt、mock 输出和展示文案尽量使用中文。

## GitHub Issues

当前冲刺按 GitHub Issue 分工：

- #2：Chroma HNSW profiles and vector benchmark
- #3：Qwen embedding API/local adapters
- #4：Qwen LLM API/local and optimized generation
- #5：Research paper parser and chunker
- #6：Frontend trace/config/eval dashboard
- #7：Course QA and paper evaluation report
- #8：Integration and demo lock
- #17：修复 #16 合并后的 embedding 规范问题

注意：论文任务不是新增一轮分工，而是现有 Issue 的阶段 B。当前收口 PR 由 `answerend42` 统一处理未完成项；课程 QA 默认链路先跑通，真实论文资料后续替换 `sample_data/papers/` fixture 即可复用同一套 parser、chunker、eval 和前端展示。

## 后端环境

推荐使用 `uv` 管理 Python 环境。当前 CI 使用 Python `3.14.4`、`backend/requirements.txt` 和 `backend/requirements-dev.txt`。

```shell
uv venv --python 3.14.4
source .venv/bin/activate
uv pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

国内环境可设置镜像：

```shell
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

启动后端：

```shell
cd backend
../.venv/bin/uvicorn main:app --reload --port 8001 --host 0.0.0.0
```

涉及真实模型或 API 时，从环境变量读取密钥：

```shell
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

不得把 API key 写入源码、测试 fixture、日志或评测结果。

## 前端环境

当前前端是 React + Vite，已在 CI 中执行 `npm ci` 和 `npm run build`。

```shell
cd frontend
npm ci
npm run dev
```

如果后端地址不是默认值，修改 `frontend/src/config/config.js` 中的 `apiBaseUrl`。

## CI 门禁

GitHub Actions 当前会运行：

- 前端：`npm ci` + `npm run build`
- 后端：安装依赖、`compileall backend`、导入服务模块
- P0：`pytest tests/contract`
- P0：`python scripts/run_smoke_pipeline.py --mode fake`
- #8 阶段 A：`python scripts/run_rag_answer_smoke.py`
- #7：`python scripts/run_eval.py --dataset-type course_qa --modes all --limit 5`

普通 PR 不应依赖真实网络、真实 API、真实模型下载或大型 benchmark。

## 后续重点

第一阶段：

- 课程 QA 默认数据端到端跑通。
- 质量标签隔离，完成课程 QA 评测报告。
- 前端展示答案、引用、检索命中和 trace。

第二阶段，已并入现有 Issue：

- #7 找到老师要求的新论文，准备目标论文、相关干扰论文、论文 QA、evidence 标注和评测报告。
- #5 增加 PDF/Markdown/OCR 解析与论文分块。
- #2/#3/#4/#6 复用索引、embedding、生成和前端能力支持论文 corpus。
- #8 做 LLM-only / Basic RAG / Optimized RAG 三模式最终对比与 demo lock。
