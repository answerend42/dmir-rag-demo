# 三模式离线评测摘要

- 数据集：`paper`
- 问题数：3
- 生成时间：2026-06-07T03:17:37.263670+00:00

| 模式 | 可回答 | 有引用 | 拒答 | citation_hit | groundedness | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_only` | 3 | 0 | 0 | 0.000 | 0.000 | 0.2 ms |
| `basic_rag` | 3 | 3 | 0 | 1.000 | 1.000 | 0.2 ms |
| `optimized_rag` | 3 | 3 | 0 | 1.000 | 1.000 | 0.3 ms |

## 现场问题

- `paper-q1` DemoRAG 的三阶段流程包括哪些步骤？
- `paper-q2` DemoRAG 在没有证据时应该怎么处理？
- `paper-q3` 在 fixture 的实验数字中 Optimized RAG 回答了几个问题？
