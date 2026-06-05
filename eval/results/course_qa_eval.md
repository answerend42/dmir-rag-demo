# Course QA 三模式评测报告

## 数据与运行配置

- 数据集：`sample_data/course_qa_public.json`
- 模式：llm_only, basic_rag, optimized_rag
- Provider / Model：`mock` / `mock-generator`
- Top-k：3
- 评测问题数：120
- 原始回答记录数：360

## 指标定义

- `latency_ms`：该条回答所有 trace 阶段耗时之和。
- `citation_hit`：引用是否命中当前问题的候选答案。
- `groundedness`：引用是否可追溯且属于当前问题证据。
- `same_question_hit_count`：检索命中中属于当前问题的数量。
- `cross_question_hit_count`：检索命中中属于其他问题的数量。
- `label_distribution`：同题命中的隐藏质量档次分布，仅用于评测报告。
- `top_hit_quality` / `avg_hit_quality`：同题命中的最高排名答案质量与平均质量。

## 按模式汇总

| mode | num_questions | avg_latency_ms | avg_citation_hit | avg_groundedness | avg_same_hits | avg_cross_hits | avg_top_hit_quality | avg_hit_quality | label_distribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| llm_only | 120 | 13.3203 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | null | null | `{}` |
| basic_rag | 120 | 15.1795 | 1.0000 | 1.0000 | 3.0000 | 0.0000 | 4.6250 | 3.9889 | `{"0": 40, "1": 51, "2": 46, "3": 43, "4": 29, "5": 30, "6": 34, "7": 28, "8": 33, "9": 26}` |
| optimized_rag | 120 | 15.1793 | 1.0000 | 1.0000 | 3.0000 | 0.0000 | 4.6250 | 3.9889 | `{"0": 40, "1": 51, "2": 46, "3": 43, "4": 29, "5": 30, "6": 34, "7": 28, "8": 33, "9": 26}` |

## 现场展示问题

1. 什么是自然语言处理？
2. 中文分词为什么重要？
3. 词袋模型有什么特点？
4. TF-IDF 为什么常用于关键词提取？
5. 停用词一定要删除吗？

## 输出文件

- `course_qa_raw_answers.json`：每条问题、每种模式的完整 `RagAnswer`。
- `course_qa_metrics.json`：逐条指标与按模式汇总指标。
- `course_qa_eval.csv`：固定列顺序的逐条扁平指标。
- `course_qa_eval.md`：当前 Markdown 报告。
