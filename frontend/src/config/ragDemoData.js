/**
 * @file ragDemoData.js
 * @brief 前端展示用评测摘要。
 * @details 只包含前端可见字段，不包含隐藏评测标签。
 */

export const courseQaEvaluationSummary = {
  llm_only: {
    answerable: 0,
    cited: 0,
    refused: 0,
    avg_latency_ms: 0,
    note: '直接评估候选答案，不检索外部知识。',
  },
  basic_rag: {
    answerable: 0,
    cited: 0,
    refused: 0,
    avg_latency_ms: 0,
    note: '用课程问题检索外部知识，再评估候选答案。',
  },
};

export const paperDemoEvaluationSummary = {
  llm_only: {
    answerable: 26,
    cited: 0,
    refused: 0,
    avg_latency_ms: 2.2,
    note: '不使用检索证据，用于展示纯模型盲区。',
  },
  basic_rag: {
    answerable: 26,
    cited: 26,
    refused: 0,
    avg_latency_ms: 2.4,
    note: '检索 top-k 证据后直接生成。',
  },
};
