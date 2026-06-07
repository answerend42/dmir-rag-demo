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
