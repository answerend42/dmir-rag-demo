/**
 * @file Generation.jsx
 * @brief RAG 演示回答页面。
 */
import { useMemo, useState, useEffect } from 'react';
import { apiBaseUrl } from '../config/config';
import EvaluationDashboard from '../components/rag/EvaluationDashboard';
import MarkdownAnswer from '../components/rag/MarkdownAnswer';
import RetrievalTracePanel from '../components/rag/RetrievalTracePanel';
import {
  createRagAnswerRequestPayload,
  createSafeRagAnswerViewModel,
  normalizeEvaluationSummary,
  removeForbiddenFields,
} from '../components/rag/ragViewModel';
import { courseQaMockRagAnswer, demoEvaluationSummary, paperDemoEvaluationSummary } from '../config/ragDemoData';

const DEFAULT_COURSE_QA_QUERY = '什么是自然语言处理？';
const DEFAULT_PAPER_QUERY = 'LLM-Wiki 在 AuthTrace 的 Single-doc 和 Overall 上与 HippoRAG 2 谁更强？具体数字是多少？';
const DEFAULT_PAPER_COLLECTION_ID = 'llm_qwen_api_20260607125456';
const DEFAULT_PAPER_PROVIDER = 'aliyun';
const DEFAULT_PAPER_MODEL = 'qwen-turbo';
const DEFAULT_DOCUMENT_TOP_K = 10;
const DEFAULT_SEARCH_THRESHOLD = 0.3;
const EVALUATION_DATASETS = [
  { key: 'course_qa', label: '课程 QA', filename: 'course_qa_eval.json' },
  { key: 'paper', label: 'LLM-Wiki 论文', filename: 'paper_eval.json' },
];
const DEMO_DATASETS = [
  { key: 'course_qa', label: '课程 QA' },
  { key: 'paper', label: '论文 RAG' },
];
const EVALUATION_FALLBACKS = {
  course_qa: demoEvaluationSummary,
  paper: paperDemoEvaluationSummary,
};
const DEMO_RAG_MODES = [
  { value: 'basic_rag', label: 'Basic RAG' },
  { value: 'llm_only', label: 'LLM-only' },
  { value: 'optimized_rag', label: 'Optimized RAG' },
];
const NUMERIC_QUERY_PATTERN = /具体数字|数值|多少|表格|表|对比|排名|分数|score|accuracy|\bAC\b|\bF1\b|\bEM\b|%/i;
const NUMBER_PATTERN = /\d+(?:\.\d+)?%?/g;

const parseHttpError = async (response) => {
  try {
    const errorBody = await response.json();
    return errorBody.detail || `HTTP ${response.status}`;
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
};

const getHitPageNumber = (metadata = {}) => {
  const pageNumber = metadata.page ?? metadata.page_number;
  const numericPage = Number(pageNumber);
  return Number.isFinite(numericPage) && numericPage > 0 ? numericPage : null;
};

const tokenizeEvidenceQuery = (query) => {
  const latinTokens = query.match(/[A-Za-z][A-Za-z0-9.-]*/g) || [];
  const splitTokens = latinTokens.flatMap((token) => token.split(/[.-]+/));
  const normalizedTokens = splitTokens.map((token) => token.toLowerCase()).filter((token) => token.length >= 2);
  const aliases = [];
  if (normalizedTokens.includes('overall')) {
    aliases.push('all');
  }
  return [...new Set([...normalizedTokens, ...aliases])];
};

const buildDocumentSearchQuery = (query, ragMode) => {
  if (ragMode !== 'optimized_rag' || !NUMERIC_QUERY_PATTERN.test(query)) {
    return query;
  }
  return [
    query,
    'table raw values exact scores metric accuracy judged accuracy original numbers all columns',
  ].join(' ');
};

const countMatches = (text, pattern) => (text.match(pattern) || []).length;

const hasEvidenceToken = (text, token) => {
  const escapedToken = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(^|[^a-z0-9])${escapedToken}([^a-z0-9]|$)`, 'i').test(text);
};

const prioritizeDocumentHitsForGeneration = (hits = [], query, ragMode) => {
  if (ragMode !== 'optimized_rag' || !NUMERIC_QUERY_PATTERN.test(query)) {
    return hits;
  }

  const queryTokens = tokenizeEvidenceQuery(query);
  return [...hits]
    .map((hit, index) => {
      const text = String(hit?.text || '');
      const lowerText = text.toLowerCase();
      const numberCount = countMatches(text, NUMBER_PATTERN);
      const tokenOverlap = queryTokens.filter((token) => hasEvidenceToken(lowerText, token)).length;
      const tableBonus = /\btable\b|表格|表\s*\d+/i.test(text) ? 4 : 0;
      const score = Number(hit?.score || 0);

      return {
        hit,
        index,
        priority: tokenOverlap * 8 + Math.min(numberCount, 24) + tableBonus + score,
      };
    })
    .sort((left, right) => right.priority - left.priority || left.index - right.index)
    .map((item) => item.hit);
};

const normalizeDocumentHits = ({ hits = [], query, collectionId }) => hits
  .filter((hit) => typeof hit?.text === 'string' && hit.text.trim())
  .map((hit, index) => {
    const metadata = hit.metadata || {};
    const pageNumber = getHitPageNumber(metadata);
    const source = metadata.source || metadata.document_name || collectionId || 'unknown-source';
    const chunkId = String(metadata.chunk || metadata.chunk_id || `${collectionId || 'document'}-${index + 1}`);

    return removeForbiddenFields({
      chunk_id: chunkId,
      doc_id: String(metadata.doc_id || metadata.document_id || source),
      text: hit.text,
      score: Number(hit.score || 0),
      rank: index + 1,
      source: String(source),
      metadata: {
        ...metadata,
        question: query,
        collection_id: collectionId,
        page_numbers: pageNumber ? [pageNumber] : [],
        section_path: Array.isArray(metadata.section_path) ? metadata.section_path : [],
        block_type: metadata.block_type || metadata.block_types || 'text',
      },
    });
  });

const buildDocumentCitations = (hits = []) => hits.slice(0, 3).map((hit) => ({
  doc_id: hit.doc_id,
  chunk_id: hit.chunk_id,
  page_number: hit.metadata?.page_numbers?.[0] || null,
  section_path: hit.metadata?.section_path || [],
  quote: hit.text,
  source: hit.source,
  metadata: {},
}));

const buildDocumentRagAnswer = ({
  query,
  collectionId,
  provider,
  model,
  ragMode = 'basic_rag',
  answerMarkdown,
  hits,
  trace,
  warnings = [],
}) => {
  const normalizedHits = Array.isArray(hits) ? hits : [];
  return removeForbiddenFields({
    contract_version: 'document-rag-ui-0.1.0',
    answer_markdown: answerMarkdown,
    citations: buildDocumentCitations(normalizedHits),
    retrieved_hits: normalizedHits,
    trace,
    warnings,
    metadata: {
      dataset_type: 'document',
      query,
      collection_id: collectionId,
      provider,
      model,
      generator: 'search-generate-pipeline',
      rag_mode: ragMode,
    },
  });
};

const buildNoEvidenceAnswer = ({ query, collectionId, provider, model, ragMode, trace }) => buildDocumentRagAnswer({
  query,
  collectionId,
  provider,
  model,
  ragMode,
  answerMarkdown: [
    '## 证据不足，无法回答',
    '',
    `当前知识库 \`${collectionId || '未选择'}\` 没有检索到可以支撑该问题的证据。`,
    '请确认已经完成文档导入、分块、嵌入、向量库索引，并且选择了正确的 collection。',
  ].join('\n'),
  hits: [],
  trace,
  warnings: ['检索结果为空，因此没有调用生成模型，也没有使用任何内置答案。'],
});

/**
 * @brief 渲染一键 RAG 演示控件、回答、证据和评测摘要。
 * @returns {JSX.Element} 生成工作流页面。
 */
const Generation = () => {
  const [query, setQuery] = useState(DEFAULT_COURSE_QA_QUERY);
  const [demoDataset, setDemoDataset] = useState('course_qa');
  const [pipelineConfig, setPipelineConfig] = useState({
    ragMode: 'basic_rag',
    topK: 3,
    threshold: DEFAULT_SEARCH_THRESHOLD,
    provider: 'mock',
    model: 'mock-generator',
    collectionId: 'course-qa-default',
  });
  const [availableCollections, setAvailableCollections] = useState([]);
  const [hasAutoRun, setHasAutoRun] = useState(false);
  const [ragAnswer, setRagAnswer] = useState(null);
  const [isRagAnswerRunning, setIsRagAnswerRunning] = useState(false);
  const [ragRequestStatus, setRagRequestStatus] = useState({
    type: 'info',
    message: '主路径为 POST /rag/answer；后端不可用时可使用课程 QA Mock fallback。',
  });
  const [selectedEvaluationDataset, setSelectedEvaluationDataset] = useState('course_qa');
  const [evaluationSummaries, setEvaluationSummaries] = useState(EVALUATION_FALLBACKS);
  const [evaluationStatus, setEvaluationStatus] = useState({
    type: 'info',
    message: '正在尝试加载评测摘要，失败时使用 fallback 摘要。',
  });

  const safeRagAnswer = useMemo(() => createSafeRagAnswerViewModel(ragAnswer), [ragAnswer]);
  const selectedEvaluationSummary = evaluationSummaries[selectedEvaluationDataset] || EVALUATION_FALLBACKS[selectedEvaluationDataset];

  /** @brief 加载当前可用向量集合，供论文文档 RAG 选择数据源。 */
  useEffect(() => {
    let isMounted = true;

    const fetchCollections = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/collections?provider=chroma`);
        if (!response.ok) {
          throw new Error(await parseHttpError(response));
        }
        const payload = await response.json();
        if (isMounted) {
          setAvailableCollections(Array.isArray(payload.collections) ? payload.collections : []);
        }
      } catch (error) {
        console.info('Collection list unavailable:', error);
      }
    };

    fetchCollections();
    return () => {
      isMounted = false;
    };
  }, []);

  /** @brief 加载课程 QA 与论文评测摘要；后端或静态文件不可用时保留 fallback。 */
  useEffect(() => {
    let isMounted = true;
    const controllers = new Set();

    const fetchWithTimeout = async (url, timeoutMs = 8000) => {
      const controller = new AbortController();
      controllers.add(controller);
      const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

      try {
        return await fetch(url, { signal: controller.signal });
      } finally {
        window.clearTimeout(timeoutId);
        controllers.delete(controller);
      }
    };

    const fetchEvaluationSummaries = async () => {
      const loadedSummaries = {};
      const failedLabels = [];

      await Promise.all(EVALUATION_DATASETS.map(async (dataset) => {
        try {
          const response = await fetchWithTimeout(`${apiBaseUrl}/eval/results/${dataset.filename}`);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          const payload = await response.json();
          loadedSummaries[dataset.key] = normalizeEvaluationSummary(payload);
        } catch (error) {
          console.info(`Evaluation summary fallback: ${dataset.filename}`, error);
          failedLabels.push(dataset.label);
        }
      }));

      if (!isMounted) {
        return;
      }
      setEvaluationSummaries({ ...EVALUATION_FALLBACKS, ...loadedSummaries });
      if (Object.keys(loadedSummaries).length === EVALUATION_DATASETS.length) {
        setEvaluationStatus({ type: 'info', message: '已加载课程 QA 与 LLM-Wiki 论文评测摘要。' });
      } else if (Object.keys(loadedSummaries).length > 0) {
        setEvaluationStatus({
          type: 'error',
          message: `部分评测摘要暂不可用：${failedLabels.join('、')}。缺失项显示 fallback。`,
        });
      } else {
        setEvaluationStatus({
          type: 'error',
          message: '评测摘要暂不可用，当前显示 fallback 示例摘要。',
        });
      }
    };

    fetchEvaluationSummaries();
    return () => {
      isMounted = false;
      controllers.forEach((controller) => controller.abort());
      controllers.clear();
    };
  }, []);

  const handleRunDocumentRagAnswer = async (trimmedQuery) => {
    const collectionId = pipelineConfig.collectionId.trim();
    const provider = pipelineConfig.provider.trim();
    const model = pipelineConfig.model.trim();
    const ragMode = pipelineConfig.ragMode;

    if (ragMode !== 'llm_only' && !collectionId) {
      setRagRequestStatus({
        type: 'error',
        message: '请选择或输入要检索的论文文档 collection。',
      });
      return;
    }

    if (ragMode === 'llm_only') {
      const generationStartedAt = performance.now();
      setIsRagAnswerRunning(true);
      setRagRequestStatus({
        type: 'info',
        message: '正在以 LLM-only 模式调用百炼，不使用检索证据...',
      });

      try {
        const generationResponse = await fetch(`${apiBaseUrl}/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: trimmedQuery,
            provider,
            model_name: model,
            search_results: [],
            load_model: false,
            rag_mode: ragMode,
          }),
        });
        const trace = [
          {
            stage_name: 'generate',
            latency_ms: performance.now() - generationStartedAt,
            input_summary: { query: trimmedQuery, provider, model, rag_mode: ragMode, contexts: 0 },
            output_summary: { status: generationResponse.ok ? 'ok' : 'failed' },
            artifacts: {},
          },
        ];

        if (!generationResponse.ok) {
          throw new Error(await parseHttpError(generationResponse));
        }

        const generationPayload = await generationResponse.json();
        setRagAnswer(buildDocumentRagAnswer({
          query: trimmedQuery,
          collectionId,
          provider,
          model,
          ragMode,
          answerMarkdown: generationPayload.response || '## 生成结果为空\nLLM-only 模式没有返回可展示的回答。',
          hits: [],
          trace,
          warnings: ['当前为 LLM-only 模式：未执行检索，也没有引用证据。'],
        }));
        setRagRequestStatus({
          type: 'info',
          message: `LLM-only 已完成：由 ${provider}/${model} 直接生成，未使用检索证据。`,
        });
      } catch (error) {
        console.error('Document LLM-only demo error:', error);
        setRagAnswer(buildDocumentRagAnswer({
          query: trimmedQuery,
          collectionId,
          provider,
          model,
          ragMode,
          answerMarkdown: '## LLM-only 生成失败\n当前没有可展示回答。',
          hits: [],
          trace: [],
          warnings: [`LLM-only 请求失败：${error.message}`],
        }));
        setRagRequestStatus({
          type: 'error',
          message: `LLM-only 暂不可用：${error.message}`,
        });
      } finally {
        setIsRagAnswerRunning(false);
      }
      return;
    }

    const searchQuery = buildDocumentSearchQuery(trimmedQuery, ragMode);
    const searchStartedAt = performance.now();
    setIsRagAnswerRunning(true);
    setRagRequestStatus({
      type: 'info',
      message: ragMode === 'optimized_rag'
        ? '正在以 Optimized RAG 模式检索并重排论文证据...'
        : '正在以 Basic RAG 模式检索论文文档向量库...',
    });

    try {
      const searchResponse = await fetch(`${apiBaseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          collection_id: collectionId,
          top_k: pipelineConfig.topK,
          threshold: pipelineConfig.threshold,
          word_count_threshold: 0,
          save_results: false,
        }),
      });

      if (!searchResponse.ok) {
        throw new Error(await parseHttpError(searchResponse));
      }

      const searchPayload = await searchResponse.json();
      const rawHits = searchPayload.results?.results || [];
      const prioritizedRawHits = prioritizeDocumentHitsForGeneration(rawHits, trimmedQuery, ragMode);
      const searchLatencyMs = performance.now() - searchStartedAt;
      const normalizedHits = normalizeDocumentHits({
        query: trimmedQuery,
        collectionId,
        hits: prioritizedRawHits,
      });
      const trace = [
        {
          stage_name: 'search',
          latency_ms: searchLatencyMs,
          input_summary: {
            query: trimmedQuery,
            search_query: searchQuery,
            collection_id: collectionId,
            top_k: pipelineConfig.topK,
            threshold: pipelineConfig.threshold,
          },
          output_summary: {
            hits: normalizedHits.length,
            best_score: normalizedHits.length > 0
              ? Math.max(...normalizedHits.map((hit) => Number(hit.score || 0)))
              : null,
            evidence_order: ragMode === 'optimized_rag' ? 'numeric-evidence-priority' : 'score',
          },
          artifacts: {},
        },
      ];

      if (normalizedHits.length === 0) {
        setRagAnswer(buildNoEvidenceAnswer({
          query: trimmedQuery,
          collectionId,
          provider,
          model,
          ragMode,
          trace,
        }));
        setRagRequestStatus({
          type: 'error',
          message: '没有检索到可用证据，已停止生成，避免无依据回答。',
        });
        return;
      }

      setRagRequestStatus({
        type: 'info',
        message: `已命中 ${normalizedHits.length} 条证据，正在调用百炼生成回答...`,
      });

      const generationStartedAt = performance.now();
      const generationResponse = await fetch(`${apiBaseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: trimmedQuery,
          provider,
          model_name: model,
          search_results: prioritizedRawHits,
          load_model: false,
          rag_mode: ragMode,
        }),
      });
      const generationLatencyMs = performance.now() - generationStartedAt;
      trace.push({
        stage_name: 'generate',
        latency_ms: generationLatencyMs,
        input_summary: { query: trimmedQuery, provider, model, contexts: normalizedHits.length },
        output_summary: { status: generationResponse.ok ? 'ok' : 'failed' },
        artifacts: {},
      });

      if (!generationResponse.ok) {
        const detail = await parseHttpError(generationResponse);
        setRagAnswer(buildDocumentRagAnswer({
          query: trimmedQuery,
          collectionId,
          provider,
          model,
          ragMode,
          answerMarkdown: [
            '## 已检索到证据，但生成失败',
            '',
            `当前已经从 \`${collectionId}\` 检索到 ${normalizedHits.length} 条证据，但生成模型调用失败。`,
            '右侧仍保留真实检索命中，方便检查问题是否出在检索还是生成阶段。',
          ].join('\n'),
          hits: normalizedHits,
          trace,
          warnings: [`生成模型调用失败：${detail}`],
        }));
        setRagRequestStatus({
          type: 'error',
          message: `已完成检索，但生成失败：${detail}`,
        });
        return;
      }

      const generationPayload = await generationResponse.json();
      setRagAnswer(buildDocumentRagAnswer({
        query: trimmedQuery,
        collectionId,
        provider,
        model,
        ragMode,
        answerMarkdown: generationPayload.response || '## 生成结果为空\n已完成检索，但模型没有返回可展示的回答。',
        hits: normalizedHits,
        trace,
      }));
      setRagRequestStatus({
        type: 'info',
        message: `文档 RAG 已完成：检索 ${normalizedHits.length} 条证据，并由 ${provider}/${model} 生成回答。`,
      });
    } catch (error) {
      console.error('Document RAG demo error:', error);
      setRagAnswer(buildDocumentRagAnswer({
        query: trimmedQuery,
        collectionId,
        provider,
        model,
        ragMode,
        answerMarkdown: [
          '## 检索失败，无法回答',
          '',
          '当前请求没有拿到真实检索证据，因此不会继续生成回答。',
        ].join('\n'),
        hits: [],
        trace: [],
        warnings: [`文档 RAG 请求失败：${error.message}`],
      }));
      setRagRequestStatus({
        type: 'error',
        message: `文档 RAG 暂不可用：${error.message}。没有使用内置答案或检索 fallback。`,
      });
    } finally {
      setIsRagAnswerRunning(false);
    }
  };

  const handleRunRagAnswer = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setRagRequestStatus({
        type: 'error',
        message: '请输入问题后再运行 RAG 演示。',
      });
      return;
    }

    if (demoDataset === 'paper') {
      setSelectedEvaluationDataset('paper');
      await handleRunDocumentRagAnswer(trimmedQuery);
      return;
    }

    const payload = createRagAnswerRequestPayload({
      query: trimmedQuery,
      ragMode: pipelineConfig.ragMode,
      topK: pipelineConfig.topK,
      provider: pipelineConfig.provider,
      model: pipelineConfig.model,
      collectionId: pipelineConfig.collectionId,
      metadata: {},
    });

    setIsRagAnswerRunning(true);
    setRagRequestStatus({
      type: 'info',
      message: '正在调用 POST /rag/answer...',
    });

    try {
      const response = await fetch(`${apiBaseUrl}/rag/answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(await parseHttpError(response));
      }

      const data = await response.json();
      setRagAnswer(removeForbiddenFields(data));
      setRagRequestStatus({
        type: 'info',
        message: '已从 /rag/answer 获取 RagAnswer，当前展示真实主链路响应。',
      });
    } catch (error) {
      console.error('RAG answer request error:', error);
      setRagAnswer(null);
      setRagRequestStatus({
        type: 'error',
        message: `/rag/answer 暂不可用：${error.message}。可使用课程 QA Mock fallback 继续展示。`,
      });
    } finally {
      setIsRagAnswerRunning(false);
    }
  };

  const handleSelectDemoDataset = (nextDataset) => {
    setDemoDataset(nextDataset);
    setSelectedEvaluationDataset(nextDataset);
    setQuery(nextDataset === 'paper' ? DEFAULT_PAPER_QUERY : DEFAULT_COURSE_QA_QUERY);
    if (nextDataset === 'course_qa') {
      setPipelineConfig((currentConfig) => ({
        ...currentConfig,
        ragMode: 'basic_rag',
        topK: 3,
        threshold: DEFAULT_SEARCH_THRESHOLD,
        provider: 'mock',
        model: 'mock-generator',
        collectionId: 'course-qa-default',
      }));
    } else {
      setPipelineConfig((currentConfig) => ({
        ...currentConfig,
        ragMode: 'basic_rag',
        topK: DEFAULT_DOCUMENT_TOP_K,
        threshold: DEFAULT_SEARCH_THRESHOLD,
        provider: DEFAULT_PAPER_PROVIDER,
        model: DEFAULT_PAPER_MODEL,
        collectionId: currentConfig.collectionId === 'course-qa-default'
          ? DEFAULT_PAPER_COLLECTION_ID
          : currentConfig.collectionId,
      }));
    }
  };

  const handleUseMockAnswer = () => {
    setRagAnswer(courseQaMockRagAnswer);
    setPipelineConfig((currentConfig) => ({
      ...currentConfig,
      ragMode: 'basic_rag',
      topK: 3,
      threshold: DEFAULT_SEARCH_THRESHOLD,
      provider: 'mock',
      model: 'mock-generator',
      collectionId: 'course-qa-default',
    }));
    setRagRequestStatus({
      type: 'info',
      message: '当前展示课程 QA Mock fallback；默认主路径仍是 POST /rag/answer。',
    });
  };

  /** @brief 页面首次打开时自动跑一次 mock RAG，保证演示区不是空白。 */
  useEffect(() => {
    if (!hasAutoRun) {
      setHasAutoRun(true);
      handleRunRagAnswer();
    }
  });

  return (
    <div className="bg-slate-50 p-6">
      <h1 className="sr-only">RAG 响应生成演示</h1>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-4">
          <section className="rounded-lg border bg-white p-4 shadow-sm">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-slate-900">一键演示</h2>
              <p className="text-xs text-slate-500">
                选择数据，输入问题，运行。
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700">演示数据</label>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  {DEMO_DATASETS.map((dataset) => {
                    const isActive = demoDataset === dataset.key;
                    return (
                      <button
                        key={dataset.key}
                        type="button"
                        onClick={() => handleSelectDemoDataset(dataset.key)}
                        className={`rounded border px-3 py-2 text-center text-sm font-semibold ${
                          isActive
                            ? 'border-green-500 bg-green-50 text-green-800'
                            : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                        }`}
                      >
                        {dataset.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700">问题</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="输入要演示的问题"
                  className="mt-1 block h-28 w-full resize-none rounded border border-slate-300 p-2 text-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-100"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-slate-700">RAG 模式</label>
                  <select
                    value={pipelineConfig.ragMode}
                    onChange={(event) => setPipelineConfig({ ...pipelineConfig, ragMode: event.target.value })}
                    className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm"
                  >
                    {DEMO_RAG_MODES.map((mode) => (
                      <option key={mode.value} value={mode.value}>{mode.label}</option>
                    ))}
                  </select>
                  {demoDataset === 'paper' && (
                    <p className="mt-1 text-xs text-slate-500">
                      Optimized 适合数字/表格题。
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700">Top K：{pipelineConfig.topK}</label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={pipelineConfig.topK}
                    onChange={(event) => setPipelineConfig({ ...pipelineConfig, topK: Number(event.target.value) })}
                    className="mt-3 block w-full"
                  />
                </div>
              </div>

              <button
                type="button"
                onClick={handleRunRagAnswer}
                disabled={isRagAnswerRunning}
                className="w-full rounded bg-green-600 px-4 py-3 text-base font-semibold text-white shadow-sm hover:bg-green-700 disabled:bg-green-300"
              >
                {isRagAnswerRunning ? '运行中...' : '运行'}
              </button>

              <button
                type="button"
                onClick={handleUseMockAnswer}
                className="w-full rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                本地 Mock
              </button>

              {ragRequestStatus && (
                <div className={`rounded border px-3 py-2 text-sm ${
                  ragRequestStatus.type === 'error'
                    ? 'border-amber-200 bg-amber-50 text-amber-800'
                    : 'border-blue-200 bg-blue-50 text-blue-800'
                }`}>
                  {ragRequestStatus.message}
                </div>
              )}
            </div>
          </section>

          <details className="rounded-lg border bg-white p-4 shadow-sm">
            <summary className="cursor-pointer text-sm font-semibold text-slate-800">高级 RAG 请求参数</summary>
            <div className="mt-4 space-y-3">
              <div>
                <label className="block text-sm font-medium text-slate-700">Provider</label>
                <input
                  value={pipelineConfig.provider}
                  onChange={(event) => setPipelineConfig({ ...pipelineConfig, provider: event.target.value })}
                  className="mt-1 block w-full rounded border border-slate-300 px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700">Model</label>
                <input
                  value={pipelineConfig.model}
                  onChange={(event) => setPipelineConfig({ ...pipelineConfig, model: event.target.value })}
                  className="mt-1 block w-full rounded border border-slate-300 px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700">
                  检索阈值：{pipelineConfig.threshold}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={pipelineConfig.threshold}
                  onChange={(event) => setPipelineConfig({ ...pipelineConfig, threshold: Number(event.target.value) })}
                  className="mt-3 block w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700">Collection / 向量集合</label>
                {availableCollections.length > 0 ? (
                  <select
                    value={pipelineConfig.collectionId}
                    onChange={(event) => setPipelineConfig({ ...pipelineConfig, collectionId: event.target.value })}
                    className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm"
                  >
                    <option value="">请选择 collection...</option>
                    {pipelineConfig.collectionId
                      && !availableCollections.some((collection) => collection.id === pipelineConfig.collectionId) && (
                      <option value={pipelineConfig.collectionId}>{pipelineConfig.collectionId}</option>
                    )}
                    {availableCollections.map((collection) => (
                      <option key={collection.id} value={collection.id}>
                        {collection.name} ({collection.count})
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    value={pipelineConfig.collectionId}
                    onChange={(event) => setPipelineConfig({ ...pipelineConfig, collectionId: event.target.value })}
                    className="mt-1 block w-full rounded border border-slate-300 px-3 py-2 text-sm"
                  />
                )}
              </div>
            </div>
          </details>
        </div>

        <div className="min-w-0 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-y border-slate-200 bg-white/70 px-4 py-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-950">当前 RAG 输出</h3>
              <p className="mt-0.5 text-xs text-slate-500">回答、来源和调试 trace 来自同一次请求。</p>
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-slate-600">
              <div className="rounded-full bg-slate-100 px-3 py-1 ring-1 ring-slate-200">
                Hits {safeRagAnswer.retrievedHits.length}
              </div>
              <div className="rounded-full bg-slate-100 px-3 py-1 ring-1 ring-slate-200">
                Citations {safeRagAnswer.citations.length}
              </div>
              <div className="rounded-full bg-slate-100 px-3 py-1 ring-1 ring-slate-200">
                Trace {safeRagAnswer.trace.length}
              </div>
            </div>
          </div>

          <MarkdownAnswer
            answerMarkdown={safeRagAnswer.answerMarkdown}
            warnings={safeRagAnswer.warnings}
            contractVersion={safeRagAnswer.contractVersion}
            citations={safeRagAnswer.citations}
            retrievedHits={safeRagAnswer.retrievedHits}
          />

          <RetrievalTracePanel
            retrievedHits={safeRagAnswer.retrievedHits}
            citations={safeRagAnswer.citations}
            trace={safeRagAnswer.trace}
          />

          <EvaluationDashboard
            summary={selectedEvaluationSummary}
            status={evaluationStatus}
            datasets={EVALUATION_DATASETS}
            selectedDataset={selectedEvaluationDataset}
            onSelectDataset={setSelectedEvaluationDataset}
          />
        </div>
      </div>
    </div>
  );
};

export default Generation;
