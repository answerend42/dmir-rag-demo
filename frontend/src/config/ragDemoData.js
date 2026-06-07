/**
 * @file ragDemoData.js
 * @brief 前端展示用 RagAnswer 示例和三模式评测摘要。
 * @details 示例只包含 RagAnswer contract 可见字段，不包含隐藏评测标签。
 */

export const courseQaMockRagAnswer = {
  contract_version: '0.1.0',
  answer_markdown: [
    '## 有证据支撑的模拟回答',
    '',
    '- 自然语言处理是面向人类语言的计算建模与应用技术，目标是让机器理解、表示和生成文本或语音。',
    '- 在课程 QA 默认数据中，检索命中展示了候选答案、来源、score 和 trace，可用于说明 Basic RAG 如何提供证据。',
  ].join('\n'),
  warnings: [],
  citations: [
    {
      doc_id: 'course-qa-demo',
      chunk_id: 'chunk-nlp-definition',
      page_number: null,
      section_path: ['课程 QA 默认测试数据'],
      quote: '自然语言处理是面向人类语言的计算建模与应用技术，核心目标是让机器能够对文本或语音中的语言信息进行有效表示、理解和生成。',
      source: 'sample_data/course_qa_public.json',
      metadata: {},
    },
  ],
  retrieved_hits: [
    {
      chunk_id: 'chunk-nlp-definition',
      doc_id: 'course-qa-demo',
      text: '课程主题：自然语言处理课程知识问答\n问题：什么是自然语言处理？\n候选答案：自然语言处理是面向人类语言的计算建模与应用技术，核心目标是让机器能够对文本或语音中的语言信息进行有效表示、理解和生成，并服务于检索、翻译、对话和摘要等实际任务。',
      score: 1.447,
      rank: 1,
      source: 'sample_data/course_qa_public.json',
      metadata: {
        category: '自然语言处理课程知识问答',
        qa_id: 1,
        question: '什么是自然语言处理？',
        answer_id: 'ans-53fcdf050c42',
        page_numbers: [],
        section_path: ['课程 QA 默认测试数据'],
        block_type: 'text',
        parser_name: 'course-qa-loader',
      },
    },
    {
      chunk_id: 'chunk-nlp-short',
      doc_id: 'course-qa-demo',
      text: '课程主题：自然语言处理课程知识问答\n问题：什么是自然语言处理？\n候选答案：自然语言处理是处理文字的一种技术。',
      score: 1.408,
      rank: 2,
      source: 'sample_data/course_qa_public.json',
      metadata: {
        category: '自然语言处理课程知识问答',
        qa_id: 1,
        question: '什么是自然语言处理？',
        answer_id: 'ans-ae3886612812',
        page_numbers: [],
        section_path: ['课程 QA 默认测试数据'],
        block_type: 'text',
        parser_name: 'course-qa-loader',
      },
    },
  ],
  trace: [
    {
      stage_name: 'chunk',
      latency_ms: 1.9,
      input_summary: { doc_id: 'course-qa-demo', blocks: 200 },
      output_summary: { chunks: 200 },
      artifacts: {},
    },
    {
      stage_name: 'embed',
      latency_ms: 2.8,
      input_summary: { chunks: 200 },
      output_summary: { embeddings: 200, dim: 16 },
      artifacts: {},
    },
    {
      stage_name: 'index',
      latency_ms: 0.1,
      input_summary: { embeddings: 200 },
      output_summary: { backend: 'mock-numpy-flat' },
      artifacts: {},
    },
    {
      stage_name: 'search',
      latency_ms: 0.5,
      input_summary: { query: '什么是自然语言处理？', top_k: 3 },
      output_summary: { hits: 3, best_score: 1.447 },
      artifacts: {},
    },
    {
      stage_name: 'generate',
      latency_ms: 0.1,
      input_summary: { query: '什么是自然语言处理？', contexts: 3 },
      output_summary: { citations: 1, warnings: 0 },
      artifacts: {},
    },
  ],
  metadata: {
    generator: 'mock-generator',
    rag_mode: 'basic_rag',
    dataset_path: 'sample_data/course_qa_public.json',
    dataset_summary: {
      question_count: 20,
      candidate_count: 200,
      categories: {
        自然语言处理课程知识问答: 200,
      },
    },
  },
};

export const demoEvaluationSummary = {
  llm_only: {
    answerable: 2,
    cited: 0,
    refused: 0,
    avg_latency_ms: 820,
    note: '无检索证据，适合展示新论文盲区。',
  },
  basic_rag: {
    answerable: 4,
    cited: 4,
    refused: 1,
    avg_latency_ms: 1250,
    note: 'dense top-k 检索后直接生成。',
  },
  optimized_rag: {
    answerable: 5,
    cited: 5,
    refused: 1,
    avg_latency_ms: 1680,
    note: '预留 query rewrite / rerank / grounded prompt。',
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
    note: 'dense top-k 检索后直接生成。',
  },
  optimized_rag: {
    answerable: 26,
    cited: 26,
    refused: 0,
    avg_latency_ms: 2.4,
    note: '预留 query rewrite / rerank / grounded prompt。',
  },
};
