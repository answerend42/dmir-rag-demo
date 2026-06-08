/**
 * @file ragViewModel.js
 * @brief 将 RagAnswer contract 转换为前端安全展示模型。
 */

const FORBIDDEN_KEYS = new Set(['answer_quality']);

export const RAG_MODE_LABELS = {
  llm_only: 'LLM-only',
  basic_rag: 'Basic RAG',
  optimized_rag: 'Optimized RAG',
};

const RAG_MODE_ORDER = ['llm_only', 'basic_rag', 'optimized_rag'];
const LLM_ONLY_EVIDENCE_MARKER_PATTERN = /\s*\[证据\d+\]/g;
const LLM_ONLY_EVIDENCE_HEADING_PATTERN = /(^|\n)(#{1,6}\s*)(已引用证据|检索证据|引用证据|证据)(\s*$)/gm;
const LLM_ONLY_EVIDENCE_LABEL_PATTERN = /(^|\n)(\s*(?:[-*]\s*)?)证据(\s*\d*\s*[:：])/g;

/**
 * @brief 深度移除前端禁止展示的隐藏评测字段。
 * @param {unknown} value 任意 contract 或扩展数据。
 * @returns {unknown} 不包含隐藏评测字段的安全数据。
 */
export const removeForbiddenFields = (value) => {
  if (Array.isArray(value)) {
    return value.map((item) => removeForbiddenFields(item));
  }

  if (!value || typeof value !== 'object') {
    return value;
  }

  return Object.entries(value).reduce((safeObject, [key, nestedValue]) => {
    if (FORBIDDEN_KEYS.has(key)) {
      return safeObject;
    }
    return {
      ...safeObject,
      [key]: removeForbiddenFields(nestedValue),
    };
  }, {});
};

/**
 * @brief 清理纯模型回答中由模型幻觉产生的 RAG 证据标记。
 * @param {string} markdown 原始 Markdown 回答。
 * @param {string} ragMode 当前 RAG 模式。
 * @returns {string} 适合前端展示的回答文本。
 */
export const sanitizeLlmOnlyAnswerMarkdown = (markdown = '', ragMode = '') => {
  const safeMarkdown = String(markdown || '');
  if (ragMode !== 'llm_only') {
    return safeMarkdown;
  }

  return safeMarkdown
    .replace(LLM_ONLY_EVIDENCE_MARKER_PATTERN, '')
    .replace(LLM_ONLY_EVIDENCE_HEADING_PATTERN, '$1$2模型判断依据$4')
    .replace(LLM_ONLY_EVIDENCE_LABEL_PATTERN, '$1$2判断依据$3');
};

/**
 * @brief 把 RagAnswer 标准字段整理为组件可直接读取的结构。
 * @param {object | null | undefined} answer RagAnswer 序列化对象。
 * @returns {object} 安全展示模型。
 */
export const createSafeRagAnswerViewModel = (answer) => {
  const safeAnswer = removeForbiddenFields(answer || {});
  const metadata = safeAnswer.metadata || {};
  const ragMode = metadata.rag_mode || safeAnswer.rag_mode || '';
  const isLlmOnly = ragMode === 'llm_only';

  return {
    contractVersion: safeAnswer.contract_version || 'unknown',
    answerMarkdown: sanitizeLlmOnlyAnswerMarkdown(safeAnswer.answer_markdown || '', ragMode),
    warnings: Array.isArray(safeAnswer.warnings) ? safeAnswer.warnings : [],
    citations: !isLlmOnly && Array.isArray(safeAnswer.citations) ? safeAnswer.citations : [],
    retrievedHits: !isLlmOnly && Array.isArray(safeAnswer.retrieved_hits) ? safeAnswer.retrieved_hits : [],
    trace: Array.isArray(safeAnswer.trace) ? safeAnswer.trace : [],
    metadata,
  };
};

/**
 * @brief 构造三模式评测表行，并固定展示顺序。
 * @param {object} summary 三模式评测摘要。
 * @returns {Array<object>} 表格行。
 */
export const buildEvaluationRows = (summary = {}) =>
  RAG_MODE_ORDER.map((mode) => ({
    mode,
    label: RAG_MODE_LABELS[mode],
    ...(summary[mode] || {}),
  }));

/**
 * @brief 规范化评测摘要响应，兼容直接摘要和 {summary: ...} 包装。
 * @param {object | null | undefined} payload 评测摘要响应。
 * @returns {object} 可交给 EvaluationDashboard 的摘要。
 */
export const normalizeEvaluationSummary = (payload) => {
  const safePayload = removeForbiddenFields(payload || {});
  if (safePayload.summary && typeof safePayload.summary === 'object') {
    return safePayload.summary;
  }
  return safePayload;
};

/**
 * @brief 将数值耗时格式化为毫秒。
 * @param {number | string | null | undefined} value 原始耗时。
 * @returns {string} 前端展示文本。
 */
export const formatLatency = (value) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return '-';
  }
  return `${numericValue.toFixed(numericValue >= 10 ? 0 : 1)} ms`;
};

/**
 * @brief 将检索相关性分数格式化为定长小数。
 * @param {number | string | null | undefined} value 原始分数。
 * @returns {string} 前端展示文本。
 */
export const formatScore = (value) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return '-';
  }
  return numericValue.toFixed(3);
};
