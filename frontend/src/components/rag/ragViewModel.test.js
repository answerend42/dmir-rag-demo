import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildEvaluationRows,
  createSafeRagAnswerViewModel,
  normalizeEvaluationSummary,
  sanitizeLlmOnlyAnswerMarkdown,
} from './ragViewModel.js';

test('createSafeRagAnswerViewModel removes answer_quality from nested display data', () => {
  const viewModel = createSafeRagAnswerViewModel({
    answer_markdown: '## 回答',
    warnings: ['提示'],
    citations: [
      {
        doc_id: 'doc-1',
        chunk_id: 'chunk-1',
        quote: '证据',
        metadata: { answer_quality: 9, keep: 'visible' },
      },
    ],
    retrieved_hits: [
      {
        rank: 1,
        score: 0.98,
        source: 'sample_data/papers/llm_wiki_retrieval_as_reasoning.md',
        text: '命中文本',
        metadata: { answer_quality: 8, question: 'LLM-Wiki 的核心贡献是什么？' },
      },
    ],
    trace: [
      {
        stage_name: 'search',
        latency_ms: 1.5,
        input_summary: { answer_quality: 7, top_k: 3 },
        output_summary: { hits: 1 },
      },
    ],
    metadata: { answer_quality: 6, rag_mode: 'basic_rag' },
  });

  assert.equal(JSON.stringify(viewModel).includes('answer_quality'), false);
  assert.equal(viewModel.retrievedHits[0].metadata.question, 'LLM-Wiki 的核心贡献是什么？');
  assert.equal(viewModel.citations[0].metadata.keep, 'visible');
});

test('buildEvaluationRows keeps stable llm/basic/optimized ordering', () => {
  const rows = buildEvaluationRows({
    llm_only: { answerable: 2, cited: 0, avg_latency_ms: 1200 },
    basic_rag: { answerable: 4, cited: 3, avg_latency_ms: 1800 },
    optimized_rag: { answerable: 5, cited: 5, avg_latency_ms: 2200 },
  });

  assert.deepEqual(
    rows.map((row) => row.mode),
    ['llm_only', 'basic_rag', 'optimized_rag'],
  );
  assert.equal(rows[2].label, 'Optimized RAG');
  assert.equal(rows[1].cited, 3);
});

test('createSafeRagAnswerViewModel suppresses hallucinated LLM-only evidence', () => {
  const viewModel = createSafeRagAnswerViewModel({
    answer_markdown: '## 已引用证据\nA6 更合适 [证据1]\n证据1：多个候选答案都提到充电。',
    citations: [
      {
        doc_id: 'hallucinated-doc',
        chunk_id: 'hallucinated-chunk',
        quote: '这不是检索证据',
      },
    ],
    retrieved_hits: [
      {
        rank: 1,
        score: 0.9,
        text: '这也不应在 LLM-only 展示',
      },
    ],
    metadata: { rag_mode: 'llm_only' },
  });

  assert.equal(viewModel.citations.length, 0);
  assert.equal(viewModel.retrievedHits.length, 0);
  assert.equal(viewModel.answerMarkdown.includes('[证据1]'), false);
  assert.equal(viewModel.answerMarkdown.includes('## 模型判断依据'), true);
  assert.equal(viewModel.answerMarkdown.includes('判断依据1：多个候选答案都提到充电。'), true);
});

test('sanitizeLlmOnlyAnswerMarkdown keeps RAG citations outside LLM-only', () => {
  const markdown = '回答 [证据1]';

  assert.equal(sanitizeLlmOnlyAnswerMarkdown(markdown, 'basic_rag'), markdown);
  assert.equal(sanitizeLlmOnlyAnswerMarkdown(markdown, 'llm_only'), '回答');
});

test('normalizeEvaluationSummary accepts remote wrapper payload and removes hidden labels', () => {
  const summary = normalizeEvaluationSummary({
    summary: {
      basic_rag: {
        answerable: 4,
        cited: 3,
        answer_quality: 8,
      },
    },
  });

  assert.equal(summary.basic_rag.answerable, 4);
  assert.equal(summary.basic_rag.cited, 3);
  assert.equal(JSON.stringify(summary).includes('answer_quality'), false);
});
