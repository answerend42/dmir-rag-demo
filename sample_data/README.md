# 默认测试数据

`course_qa_public.json` 是第一阶段 RAG 系统默认输入，来自老师要求的课程 QA 数据。

该文件只包含：

- 课程主题
- 问题
- 候选答案
- 不含语义的 `answer_id`

该文件不包含 0-9 质量档次，允许进入 RAG 索引、prompt、trace 和前端展示。

质量标签在 `eval/labels/course_qa_quality_labels.json` 中，只允许评测脚本在模型生成完成后读取。

论文阶段默认输入位于 `sample_data/papers/llm_wiki_retrieval_as_reasoning.md`。它是
LLM-Wiki 论文的中文结构化 digest，配套 `sample_data/papers/paper_eval_fixture.json`
提供 26 个论文 QA 与 evidence 标注。论文 digest 允许进入 RAG；arXiv PDF 本身不提交仓库。
