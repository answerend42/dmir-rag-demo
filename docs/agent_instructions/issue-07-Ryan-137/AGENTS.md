# Issue #7 Agent 指示：Course QA and paper evaluation report

你正在为 RAG Demo 项目完成 GitHub Issue #7。这个 Issue 分成两个阶段：阶段 A 基于老师要求的课程 QA 数据实现三模式评测报告；阶段 B 继续确定目标新论文、相关干扰论文、论文 QA、evidence 标注和论文评测报告。论文阶段已经包含在本 Issue 内，不另开新 Issue。

## Owner

当前集中收口 Owner：`answerend42`。

原模块负责人：`Ryan-137`。

## 必读文档

- `docs/agent_rules.md`
- `docs/interfaces.md`
- `docs/agent_instructions/issue-01-answerend42/AGENTS.md`

## ALLOWED_PATHS

- `eval/`
- `scripts/run_eval.py`
- `sample_data/`
- `docs/`
- `docs/agent_instructions/issue-07-Ryan-137/AGENTS.md`

## 硬性限制

1. 不得把临场手工结果伪装成自动评测结果。
2. RAG 默认输入只能使用 `sample_data/course_qa_public.json`。
3. `answer_quality` 只能从 `eval/labels/course_qa_quality_labels.json` 在生成后读取，禁止进入索引、prompt、trace 或前端展示。
4. 输出报告不得包含 API key、绝对路径、个人隐私。
5. 文档、评测说明和展示问题尽量使用中文。
6. 阶段 A 不要因为论文未确定而阻塞课程 QA；阶段 B 必须在同一 Issue 内完成论文资料与评测设计。
7. 大体积 PDF 不要直接提交到仓库；优先提交论文 metadata、下载链接、hash、摘要和小型样例。

## 实施顺序

阶段 A：课程 QA

1. 读取 `sample_data/course_qa_public.json` 作为系统可见输入。
2. 读取 `eval/labels/course_qa_quality_labels.json` 作为评测专用隐藏标签。
3. 设计至少 5 个课程 QA 现场稳定展示问题。
4. 实现 `run_eval.py` 的 mock/small 模式，输出 JSON/CSV/Markdown；前端读取 `eval/results/course_qa_eval.json`。
5. 三模式指标至少包含 citation_hit、label_distribution、groundedness、latency。
6. 写测试确认 `answer_quality` 不会出现在 RAG 请求、检索命中、trace 或前端展示数据中。

阶段 B：新论文

7. `sample_data/papers/paper_eval_fixture.json` 已固定为 LLM-Wiki 论文阶段 B 输入。
8. 目标论文 metadata 已锁定为 arXiv:2605.25480 v2，source digest 为 `sample_data/papers/llm_wiki_retrieval_as_reasoning.md`。
9. 已选择 4 篇相关干扰/背景论文，记录 metadata 和用途。
10. 已设计 26 个论文 QA，覆盖方法、实验数字、消融、图表/表格结论和与干扰论文的差异。
11. 每个论文问题都应保留 evidence：paper_id、page、section、paragraph/table/caption 线索。
12. `run_eval.py` 必须同时支持课程 QA 数据集和论文数据集，输出同一套三模式指标。

## 验收命令

```shell
python scripts/run_eval.py --modes all --limit 5
```

## PR 输出

PR 中必须写明：

- 课程 QA public 输入与隐藏 labels 的隔离方式。
- 评测 JSON/CSV/Markdown 格式。
- 5 个现场问题。
- 目标新论文和干扰论文 metadata。
- 论文 QA/evidence 标注格式。
- 指标定义和小样例结果。
