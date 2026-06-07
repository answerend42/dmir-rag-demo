# Issue #8 Agent 指示：Integration and demo lock

你正在为 RAG Demo 项目完成 GitHub Issue #8。目标是集成 `/rag/answer`，固定展示数据、模型、索引和问题，并在展示前 24 小时进入 demo lock。

## Owner

`answerend42` + 全员

## 必读文档

- `docs/agent_rules.md`
- `docs/interfaces.md`
- `docs/sprint_board.md`
- `docs/agent_instructions/issue-01-answerend42/AGENTS.md`

## ALLOWED_PATHS

- `backend/`
- `frontend/`
- `scripts/`
- `eval/`
- `docs/`
- `docs/agent_instructions/issue-08-answerend42/AGENTS.md`

## 硬性限制

1. 任何 P3 扩展不得影响主线 demo。
2. 真实模型不可用时必须可以切换 fake/mock fallback。
3. 展示前 24 小时只修 bug，不扩功能。
4. `/rag/answer` 输出必须是 `RagAnswer` 或其序列化形式。
5. 代码注释必须使用中文 Doxygen 风格。
6. 面向 LLM 的 prompt 和现场展示文案尽量使用中文。
7. 第一阶段固定 demo 数据为 `sample_data/course_qa_public.json`；`answer_quality` 只能在评测脚本生成报告时读取。
8. 论文任务已并入 #5/#7/#8 的阶段 B；集成时不要要求另开新 Issue 才能接论文 corpus。

## 实施顺序

1. 确认 #1 fake pipeline 可运行。
2. 汇总 parser、embedding、index、generation、frontend、eval 各模块 PR。
3. 统一 `/rag/answer` 输入输出。
4. 固定课程 QA demo 数据、索引、模型 provider、top_k 和 5 个现场问题。
5. 用 `scripts/run_eval.py` 生成 `eval/results/course_qa_eval.json`，供前端 dashboard 读取。
6. LLM-Wiki 已作为真实目标论文固定到 `sample_data/papers/`，继续用同一 corpus 验证论文 parser/chunker/eval 格式。
7. 使用 `paper_eval_fixture.json` 的 26 个论文演示问题和 `eval/results/paper_eval.*` 三模式评测结果进入 demo lock。
8. 生成最终评测表、截图、录屏和离线 fallback。
9. 展示前锁定 main，只合并 P0/P1 bugfix。

## 验收命令

```shell
python -m compileall backend
pytest tests/contract tests/unit -m "not integration and not benchmark"
python scripts/run_smoke_pipeline.py --mode fake
cd frontend && npm run build
```

## PR 输出

PR 中必须写明：

- 三模式是否都可演示。
- 真实路径和 fake fallback 的切换方式。
- 课程 QA 与论文两类最终演示问题。
- 录屏、截图或离线结果位置。
