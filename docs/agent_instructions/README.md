# Issue Agent 指示文档

这些目录为每个 GitHub Issue 准备了可直接交给 Codex/Claude Code 的中文任务说明。使用方式：

1. 找到自己负责的 Issue 目录。
2. 把其中 `AGENTS.md` 的内容贴给 Codex、Claude Code、Cursor Agent 或其他代码 Agent。
3. 严格按 `ALLOWED_PATHS` 修改文件。
4. PR 中复制测试命令和风险说明。

统一要求：

- 代码注释必须使用中文 Doxygen 风格。
- 面向 LLM 的 prompt、mock 输出、展示文案尽量使用中文。
- 单元测试不得依赖真实网络、真实 API key、真实模型下载。
- 不得修改其他成员负责目录。
- 不得删除 fake/mock fallback。
- 第一阶段默认测试数据是 `sample_data/course_qa_public.json`。
- `answer_quality` 档次只能由评测脚本读取，禁止进入 RAG 索引、prompt、trace 或前端展示。
- 新论文 RAG 任务已经并入现有 Issue 的阶段 B；默认论文 corpus 已固定为 LLM-Wiki digest，论文 QA/evidence 和最终评测结果位于 `sample_data/papers/` 与 `eval/results/paper_eval.*`。
