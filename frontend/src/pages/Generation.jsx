/**
 * @file Generation.jsx
 * @brief RAG 演示回答页面。
 */
import { useCallback, useMemo, useState, useEffect } from 'react';
import { apiBaseUrl } from '../config/config';
import EvaluationDashboard from '../components/rag/EvaluationDashboard';
import MarkdownAnswer from '../components/rag/MarkdownAnswer';
import RetrievalTracePanel from '../components/rag/RetrievalTracePanel';
import VectorProjectionView from '../components/rag/VectorProjectionView';
import {
  createSafeRagAnswerViewModel,
  removeForbiddenFields,
} from '../components/rag/ragViewModel';
import { courseQaEvaluationSummary, paperDemoEvaluationSummary } from '../config/ragDemoData';

const DEFAULT_COURSE_QA_QUERY = '什么是自然语言处理？';
const DEFAULT_PAPER_QUERY = 'LLM-Wiki 在 AuthTrace 的 Single-doc 和 Overall 上与 HippoRAG 2 谁更强？具体数字是多少？';
const DEFAULT_PAPER_PROVIDER = 'aliyun';
const DEFAULT_PAPER_MODEL = 'qwen-turbo';
const DEFAULT_DOCUMENT_TOP_K = 3;
const DEFAULT_SEARCH_THRESHOLD = 0.3;
const DEMO_DATASETS = {
  course_qa: {
    key: 'course_qa',
    label: '课程 QA',
    chainLabel: '课程 QA 评估 · 外部知识检索 · 百炼排序',
    collectionLabel: '外部知识索引库',
    defaultQuery: DEFAULT_COURSE_QA_QUERY,
    defaultTopK: 3,
    datasetType: 'course_qa',
  },
  paper: {
    key: 'paper',
    label: '论文 RAG',
    chainLabel: '论文 RAG · 向量检索 · 百炼生成',
    collectionLabel: '论文索引库',
    defaultQuery: DEFAULT_PAPER_QUERY,
    defaultTopK: DEFAULT_DOCUMENT_TOP_K,
    datasetType: 'paper',
  },
};
const EVALUATION_SUMMARIES = {
  course_qa: courseQaEvaluationSummary,
  paper: paperDemoEvaluationSummary,
};
const DEMO_DATASET_LIST = Object.values(DEMO_DATASETS);
const DEMO_RAG_MODES = [
  { value: 'basic_rag', label: 'RAG' },
  { value: 'llm_only', label: 'LLM-only' },
];

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

const buildCourseQaEvaluationPrompt = (qaItem = {}, ragMode = 'basic_rag') => {
  const answers = Array.isArray(qaItem.answers) ? qaItem.answers : [];
  const useRetrievedEvidence = ragMode !== 'llm_only';
  const answerLines = answers.map((answer, index) => {
    const answerId = answer.answer_id || `A${index + 1}`;
    return `${answerId}. ${answer.answer || answer.text || ''}`.trim();
  });

  return [
    '课程 QA 答案评估任务',
    '',
    `课程主题：${qaItem.topic || '未标注'}`,
    `原问题：${qaItem.question || ''}`,
    '',
    '候选答案：',
    ...answerLines,
    '',
    useRetrievedEvidence
      ? '请结合检索证据完成评估：'
      : '请在 LLM-only 模式下完成评估：当前没有检索证据，只能基于模型知识和候选答案文本判断。',
    '1. 选出最佳答案编号，并用 2-3 句话说明理由。',
    '2. 给出所有候选答案的质量排序，格式为 A? > A? > ...。',
    useRetrievedEvidence
      ? '3. 指出哪些判断由检索证据支持，并使用 [证据N] 标注。'
      : '3. 说明模型判断依据或候选答案文本线索；不要使用“检索证据”“引用证据”或 [证据N] 标注。',
    useRetrievedEvidence
      ? '4. 如果证据不足以区分部分候选答案，请明确说明不确定部分。'
      : '4. 如果仅凭模型知识和候选答案文本无法区分部分候选答案，请明确说明不确定部分。',
  ].join('\n');
};

const buildCourseQaTaskMetadata = (qaItem = {}, sourceId = '') => removeForbiddenFields({
  task_type: 'course_qa_answer_ranking',
  source_id: sourceId,
  item_id: qaItem.item_id,
  topic: qaItem.topic,
  qa_id: qaItem.qa_id,
  question: qaItem.question,
  answers: Array.isArray(qaItem.answers) ? qaItem.answers : [],
});

const getCourseQaQuestionLabel = (item = {}, index = 0) => {
  const topic = item.topic ? `${item.topic} · ` : '';
  const question = item.question || `题目 ${index + 1}`;
  return `${topic}${question}`;
};

const getPreferredCourseQaItemId = (items = []) => {
  const dailyLifeItem = items.find((item) => item.topic === '日常生活');
  return dailyLifeItem?.item_id || items[0]?.item_id || '';
};

const getCourseQaSourceLabel = (source = {}) => {
  const importedAt = String(source.timestamp || '').replace('T', ' ').slice(0, 16);
  const suffixParts = [
    importedAt,
    source.question_count,
  ].filter((item) => item !== undefined && item !== null && item !== '');
  return `${source.name || source.id}${suffixParts.length ? ` (${suffixParts.join(' · ')})` : ''}`;
};

const getCollectionKindLabel = (collection = {}) => {
  if (collection.dataset_type === 'course_knowledge' || collection.source_role === 'external_knowledge') {
    return '课程知识';
  }
  if (collection.dataset_type === 'course_qa') {
    return '课程 QA';
  }
  if (collection.dataset_type === 'paper') {
    return '论文';
  }
  return collection.document_name || '';
};

const getCollectionOptionLabel = (collection = {}) => {
  const kindLabel = getCollectionKindLabel(collection);
  const modelLabel = [collection.embedding_provider, collection.embedding_model].filter(Boolean).join('/');
  const suffixParts = [
    collection.count,
    collection.database,
    collection.index_mode,
    kindLabel,
    modelLabel,
  ].filter((item) => item !== undefined && item !== null && item !== '');
  return `${collection.name}${suffixParts.length ? ` (${suffixParts.join(' · ')})` : ''}`;
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
  datasetType = 'paper',
  query,
  collectionId,
  collectionProvider = '',
  provider,
  model,
  ragMode = 'basic_rag',
  answerMarkdown,
  hits,
  trace,
  warnings = [],
  queryEmbedding = null,
  scoreAlgorithm = null,
  taskMetadata = null,
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
      dataset_type: datasetType,
      query,
      collection_id: collectionId,
      collection_provider: collectionProvider,
      provider,
      model,
      generator: 'search-generate-pipeline',
      rag_mode: ragMode,
      query_embedding: Array.isArray(queryEmbedding) ? queryEmbedding : null,
      score_algorithm: scoreAlgorithm,
      task: taskMetadata,
    },
  });
};

const buildNoEvidenceAnswer = ({
  datasetType,
  query,
  collectionId,
  collectionProvider = '',
  provider,
  model,
  ragMode,
  trace,
  queryEmbedding = null,
  scoreAlgorithm = null,
  taskMetadata = null,
}) => buildDocumentRagAnswer({
  datasetType,
  query,
  collectionId,
  collectionProvider,
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
  queryEmbedding,
  scoreAlgorithm,
  taskMetadata,
  warnings: ['检索结果为空，因此没有调用生成模型。'],
});

/**
 * @brief 渲染一键 RAG 演示控件、回答、证据和评测摘要。
 * @returns {JSX.Element} 生成工作流页面。
 */
const Generation = () => {
  const [demoDataset, setDemoDataset] = useState('course_qa');
  const [query, setQuery] = useState(DEFAULT_COURSE_QA_QUERY);
  const [pipelineConfig, setPipelineConfig] = useState({
    ragMode: 'basic_rag',
    topK: DEMO_DATASETS.course_qa.defaultTopK,
    threshold: DEFAULT_SEARCH_THRESHOLD,
    provider: DEFAULT_PAPER_PROVIDER,
    model: DEFAULT_PAPER_MODEL,
    collectionId: '',
  });
  const [availableCollections, setAvailableCollections] = useState([]);
  const [courseQaSources, setCourseQaSources] = useState([]);
  const [courseQaSourceId, setCourseQaSourceId] = useState('');
  const [courseQaItems, setCourseQaItems] = useState([]);
  const [courseQaItemId, setCourseQaItemId] = useState('');
  const [isCourseQaLoading, setIsCourseQaLoading] = useState(false);
  const [ragAnswer, setRagAnswer] = useState(null);
  const [isRagAnswerRunning, setIsRagAnswerRunning] = useState(false);
  const [ragRequestStatus, setRagRequestStatus] = useState({
    type: 'info',
    message: '请选择已建立的课程 QA 或论文索引库，运行后会展示真实检索、生成和向量视图。',
  });
  const selectedDatasetConfig = DEMO_DATASETS[demoDataset] || DEMO_DATASETS.course_qa;
  const selectedEvaluationSummary = EVALUATION_SUMMARIES[demoDataset] || EVALUATION_SUMMARIES.course_qa;
  const selectedEvidenceLabel = demoDataset === 'course_qa' ? '外部知识' : selectedDatasetConfig.label;
  const isLlmOnlyMode = pipelineConfig.ragMode === 'llm_only';
  const selectedCourseQaItem = useMemo(
    () => courseQaItems.find((item) => String(item.item_id) === String(courseQaItemId)) || null,
    [courseQaItems, courseQaItemId]
  );
  const evaluationStatus = {
    type: 'info',
    message: demoDataset === 'paper'
      ? '当前展示 LLM-Wiki 论文评测摘要。'
      : '课程 QA 的运行结果来自题目候选答案与外部知识 collection。',
  };

  const safeRagAnswer = useMemo(() => createSafeRagAnswerViewModel(ragAnswer), [ragAnswer]);
  const documentVectorCollectionId = ['course_qa', 'paper', 'document'].includes(safeRagAnswer.metadata.dataset_type)
    ? safeRagAnswer.metadata.collection_id
    : pipelineConfig.collectionId;
  const selectedVectorCollection = availableCollections.find(
    (collection) => collection.id === documentVectorCollectionId
  );
  const documentVectorProvider = safeRagAnswer.metadata.collection_provider
    || selectedVectorCollection?.database
    || 'chroma';
  const showDocumentVectorView = !isLlmOnlyMode
    && Boolean(documentVectorCollectionId);

  /** @brief 加载当前可用向量集合，供当前 RAG 入口选择数据源。 */
  const fetchAvailableCollections = useCallback(async () => {
    try {
      const providerIds = ['chroma', 'faiss'];
      const payloads = await Promise.all(providerIds.map(async (providerId) => {
        const response = await fetch(`${apiBaseUrl}/collections?provider=${providerId}`);
        if (!response.ok) {
          throw new Error(await parseHttpError(response));
        }
        const payload = await response.json();
        return {
          providerId,
          collections: Array.isArray(payload.collections) ? payload.collections : [],
        };
      }));
      const collections = payloads.flatMap(({ providerId, collections: providerCollections }) => (
        providerCollections.map((collection) => ({
          ...collection,
          database: collection.database || providerId,
        }))
      ));
      setAvailableCollections(collections);
      setPipelineConfig((currentConfig) => {
        if (collections.some((collection) => collection.id === currentConfig.collectionId)) {
          return currentConfig;
        }
        return { ...currentConfig, collectionId: '' };
      });
      return collections;
    } catch (error) {
      console.info('Collection list unavailable:', error);
      setAvailableCollections([]);
      return [];
    }
  }, []);

  /** @brief 加载已从 01 前端导入的课程 QA 任务文件。 */
  const fetchCourseQaSources = useCallback(async () => {
    setIsCourseQaLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/course-qa/sources`);
      if (!response.ok) {
        throw new Error(await parseHttpError(response));
      }
      const payload = await response.json();
      const sources = Array.isArray(payload.sources) ? payload.sources : [];
      setCourseQaSources(sources);
      setCourseQaSourceId((currentSourceId) => {
        if (sources.some((source) => source.id === currentSourceId)) {
          return currentSourceId;
        }
        return sources[0]?.id || '';
      });
      return sources;
    } catch (error) {
      console.info('Course QA source list unavailable:', error);
      setCourseQaSources([]);
      setCourseQaSourceId('');
      setCourseQaItems([]);
      setCourseQaItemId('');
      setRagRequestStatus({
        type: 'error',
        message: `课程 QA 题目文件读取失败：${error.message}`,
      });
      return [];
    } finally {
      setIsCourseQaLoading(false);
    }
  }, []);

  const handleDatasetChange = (nextDataset) => {
    const nextConfig = DEMO_DATASETS[nextDataset] || DEMO_DATASETS.course_qa;
    setDemoDataset(nextConfig.key);
    setQuery(nextConfig.defaultQuery);
    setRagAnswer(null);
    setPipelineConfig((currentConfig) => ({
      ...currentConfig,
      topK: nextConfig.defaultTopK,
      collectionId: '',
    }));
    setRagRequestStatus({
      type: 'info',
      message: `已切换到${nextConfig.label}，请选择${nextConfig.collectionLabel}后运行。`,
    });
  };

  useEffect(() => {
    fetchAvailableCollections();
  }, [fetchAvailableCollections]);

  useEffect(() => {
    if (demoDataset === 'course_qa') {
      fetchCourseQaSources();
    }
  }, [demoDataset, fetchCourseQaSources]);

  useEffect(() => {
    if (demoDataset !== 'course_qa') {
      return;
    }
    if (!courseQaSourceId) {
      setCourseQaItems([]);
      setCourseQaItemId('');
      return;
    }

    let isCancelled = false;
    const fetchItems = async () => {
      setIsCourseQaLoading(true);
      try {
        const response = await fetch(`${apiBaseUrl}/course-qa/sources/${encodeURIComponent(courseQaSourceId)}/items`);
        if (!response.ok) {
          throw new Error(await parseHttpError(response));
        }
        const payload = await response.json();
        const items = Array.isArray(payload.items) ? payload.items : [];
        if (isCancelled) {
          return;
        }
        setCourseQaItems(items);
        setCourseQaItemId((currentItemId) => {
          if (items.some((item) => String(item.item_id) === String(currentItemId))) {
            return currentItemId;
          }
          return getPreferredCourseQaItemId(items);
        });
      } catch (error) {
        if (!isCancelled) {
          console.info('Course QA items unavailable:', error);
          setCourseQaItems([]);
          setCourseQaItemId('');
          setRagRequestStatus({
            type: 'error',
            message: `课程 QA 题目读取失败：${error.message}`,
          });
        }
      } finally {
        if (!isCancelled) {
          setIsCourseQaLoading(false);
        }
      }
    };

    fetchItems();
    return () => {
      isCancelled = true;
    };
  }, [courseQaSourceId, demoDataset]);

  useEffect(() => {
    if (demoDataset === 'course_qa' && selectedCourseQaItem?.question) {
      setQuery(selectedCourseQaItem.question);
    }
  }, [demoDataset, selectedCourseQaItem]);

  /** @brief 如果索引库已被删除，则清空选择，避免请求不存在的 collection。 */
  useEffect(() => {
    if (!pipelineConfig.collectionId) {
      return;
    }
    const selectedCollectionExists = availableCollections.some(
      (collection) => collection.id === pipelineConfig.collectionId
    );
    if (!selectedCollectionExists) {
      setPipelineConfig((currentConfig) => ({ ...currentConfig, collectionId: '' }));
    }
  }, [availableCollections, pipelineConfig.collectionId]);

  const handleRunDocumentRagAnswer = async (trimmedQuery, taskOptions = {}) => {
    const collectionId = pipelineConfig.collectionId.trim();
    const provider = pipelineConfig.provider.trim();
    const model = pipelineConfig.model.trim();
    const ragMode = pipelineConfig.ragMode;
    const retrievalQuery = taskOptions.searchQuery || trimmedQuery;
    const generationQuery = taskOptions.generationQuery || trimmedQuery;
    const taskMetadata = taskOptions.taskMetadata || null;

    if (ragMode !== 'llm_only' && !collectionId) {
      setRagRequestStatus({
        type: 'error',
        message: `请先在“${selectedDatasetConfig.collectionLabel}”中选择已经建立的索引库。`,
      });
      return;
    }
    if (
      ragMode !== 'llm_only'
      && availableCollections.length > 0
      && !availableCollections.some((collection) => collection.id === collectionId)
    ) {
      setRagRequestStatus({
        type: 'error',
        message: `当前选择的${selectedDatasetConfig.collectionLabel}已经不在后端集合列表中，请刷新后重新选择。`,
      });
      setPipelineConfig((currentConfig) => ({ ...currentConfig, collectionId: '' }));
      return;
    }
    const selectedCollection = availableCollections.find((collection) => collection.id === collectionId);
    const collectionProvider = selectedCollection?.database || 'chroma';
    if (demoDataset === 'course_qa' && ragMode !== 'llm_only' && selectedCollection?.dataset_type === 'course_qa') {
      setRagRequestStatus({
        type: 'error',
        message: '课程 QA 模式需要选择外部知识索引库，不能选择由课程 QA JSON 本身建立的 collection。',
      });
      return;
    }

    if (ragMode === 'llm_only') {
      const generationStartedAt = performance.now();
      setIsRagAnswerRunning(true);
      setRagRequestStatus({
        type: 'info',
        message: `正在以 LLM-only 模式调用百炼，不使用${selectedEvidenceLabel}检索证据...`,
      });

      try {
        const generationResponse = await fetch(`${apiBaseUrl}/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: generationQuery,
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
            input_summary: { query: trimmedQuery, provider, model, rag_mode: ragMode, contexts: 0, task_type: taskMetadata?.task_type },
            output_summary: { status: generationResponse.ok ? 'ok' : 'failed' },
            artifacts: {},
          },
        ];

        if (!generationResponse.ok) {
          throw new Error(await parseHttpError(generationResponse));
        }

        const generationPayload = await generationResponse.json();
        setRagAnswer(buildDocumentRagAnswer({
          datasetType: selectedDatasetConfig.datasetType,
          query: trimmedQuery,
          collectionId,
          collectionProvider,
          provider,
          model,
          ragMode,
          answerMarkdown: generationPayload.response || '## 生成结果为空\nLLM-only 模式没有返回可展示的回答。',
          hits: [],
          trace,
          taskMetadata,
          warnings: ['当前为 LLM-only 模式：未执行检索，也没有引用证据。'],
        }));
        setRagRequestStatus({
          type: 'info',
          message: `LLM-only 已完成：由 ${provider}/${model} 直接生成，未使用${selectedEvidenceLabel}检索证据。`,
        });
      } catch (error) {
        console.error('Document LLM-only demo error:', error);
        setRagAnswer(buildDocumentRagAnswer({
          datasetType: selectedDatasetConfig.datasetType,
          query: trimmedQuery,
          collectionId,
          collectionProvider,
          provider,
          model,
          ragMode,
          answerMarkdown: '## LLM-only 生成失败\n当前没有可展示回答。',
          hits: [],
          trace: [],
          taskMetadata,
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

    const searchQuery = retrievalQuery;
    const searchStartedAt = performance.now();
    setIsRagAnswerRunning(true);
    setRagRequestStatus({
      type: 'info',
        message: `正在以 RAG 模式检索${selectedEvidenceLabel}向量库...`,
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
          include_query_embedding: true,
        }),
      });

      if (!searchResponse.ok) {
        throw new Error(await parseHttpError(searchResponse));
      }

      const searchPayload = await searchResponse.json();
      const rawHits = searchPayload.results?.results || [];
      const queryEmbedding = searchPayload.results?.query_embedding || null;
      const scoreAlgorithm = searchPayload.results?.score_algorithm || {
        name: '向量检索相似度',
        formula: 'score = normalized relevance score',
        note: '不同索引后端统一展示为越大越相关；具体算法以本次检索返回为准。',
      };
      const prioritizedRawHits = rawHits;
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
            evidence_order: 'score',
          },
          artifacts: {},
        },
      ];

      if (normalizedHits.length === 0) {
        setRagAnswer(buildNoEvidenceAnswer({
          datasetType: selectedDatasetConfig.datasetType,
          query: trimmedQuery,
          collectionId,
          collectionProvider,
          provider,
          model,
          ragMode,
          trace,
          queryEmbedding,
          scoreAlgorithm,
          taskMetadata,
        }));
        setRagRequestStatus({
          type: 'error',
          message: '没有检索到可用证据，已停止生成，避免无依据回答。',
        });
        return;
      }

      setRagRequestStatus({
        type: 'info',
        message: `已命中 ${normalizedHits.length} 条${selectedEvidenceLabel}证据，正在调用百炼生成回答...`,
      });

      const generationStartedAt = performance.now();
      const generationResponse = await fetch(`${apiBaseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: generationQuery,
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
          datasetType: selectedDatasetConfig.datasetType,
          query: trimmedQuery,
          collectionId,
          collectionProvider,
          provider,
          model,
          ragMode,
          answerMarkdown: [
            '## 已检索到证据，但生成失败',
            '',
            `当前已经从 \`${collectionId}\` 检索到 ${normalizedHits.length} 条${selectedEvidenceLabel}证据，但生成模型调用失败。`,
            '右侧仍保留真实检索命中，方便检查问题是否出在检索还是生成阶段。',
          ].join('\n'),
          hits: normalizedHits,
          trace,
          queryEmbedding,
          scoreAlgorithm,
          taskMetadata,
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
        datasetType: selectedDatasetConfig.datasetType,
        query: trimmedQuery,
        collectionId,
        collectionProvider,
        provider,
        model,
        ragMode,
        answerMarkdown: generationPayload.response || '## 生成结果为空\n已完成检索，但模型没有返回可展示的回答。',
        hits: normalizedHits,
        trace,
        queryEmbedding,
        scoreAlgorithm,
        taskMetadata,
      }));
      setRagRequestStatus({
        type: 'info',
        message: `${selectedDatasetConfig.label} 已完成：检索 ${normalizedHits.length} 条证据，并由 ${provider}/${model} 生成回答。`,
      });
    } catch (error) {
      console.error('Document RAG demo error:', error);
      const collectionWasDeleted = /Collection .* does not exist/i.test(error.message);
      if (collectionWasDeleted) {
        await fetchAvailableCollections();
        setPipelineConfig((currentConfig) => ({ ...currentConfig, collectionId: '' }));
      }
      setRagAnswer(buildDocumentRagAnswer({
        datasetType: selectedDatasetConfig.datasetType,
        query: trimmedQuery,
        collectionId,
        collectionProvider,
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
        taskMetadata,
        warnings: [`文档 RAG 请求失败：${error.message}`],
      }));
      setRagRequestStatus({
        type: 'error',
        message: collectionWasDeleted
          ? `所选${selectedDatasetConfig.collectionLabel}已被删除或不存在，请重新选择后再运行。`
          : `${selectedDatasetConfig.label} 暂不可用：${error.message}。当前没有真实检索证据，已停止生成。`,
      });
    } finally {
      setIsRagAnswerRunning(false);
    }
  };

  const handleRunRagAnswer = async () => {
    if (demoDataset === 'course_qa') {
      if (!selectedCourseQaItem) {
        setRagRequestStatus({
          type: 'error',
          message: '请先选择课程 QA 题目；如果列表为空，请先在 01 导入课程 QA JSON。',
        });
        return;
      }
      if (!Array.isArray(selectedCourseQaItem.answers) || selectedCourseQaItem.answers.length === 0) {
        setRagRequestStatus({
          type: 'error',
          message: '当前课程 QA 题目没有候选答案，无法进行答案排序。',
        });
        return;
      }

      const courseQaQuestion = String(selectedCourseQaItem.question || '').trim();
      await handleRunDocumentRagAnswer(courseQaQuestion, {
        searchQuery: courseQaQuestion,
        generationQuery: buildCourseQaEvaluationPrompt(selectedCourseQaItem, pipelineConfig.ragMode),
        taskMetadata: buildCourseQaTaskMetadata(selectedCourseQaItem, courseQaSourceId),
      });
      return;
    }

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setRagRequestStatus({
        type: 'error',
        message: '请输入问题后再运行 RAG 演示。',
      });
      return;
    }

    await handleRunDocumentRagAnswer(trimmedQuery);
  };

  return (
    <div className="bg-slate-50 p-6">
      <h1 className="sr-only">RAG 响应生成演示</h1>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-4">
          <section className="rounded-lg border bg-white p-4 shadow-sm">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-slate-900">一键演示</h2>
              <p className="text-xs text-slate-500">
                {demoDataset === 'course_qa'
                  ? '选择 QA 题目和外部知识索引库，检索证据后调用真实生成模型排序。'
                  : '选择论文索引库，检索证据后调用真实生成模型回答。'}
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700">演示数据</label>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  {DEMO_DATASET_LIST.map((dataset) => {
                    const isActive = demoDataset === dataset.key;
                    return (
                      <button
                        key={dataset.key}
                        type="button"
                        onClick={() => handleDatasetChange(dataset.key)}
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
                <label className="block text-sm font-medium text-slate-700">演示链路</label>
                <div className="mt-2 rounded border border-green-500 bg-green-50 px-3 py-2 text-sm font-semibold text-green-800">
                  {selectedDatasetConfig.chainLabel}
                </div>
              </div>

              {demoDataset === 'course_qa' ? (
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between gap-3">
                      <label className="block text-sm font-medium text-slate-700">课程 QA 文件</label>
                      <button
                        type="button"
                        onClick={fetchCourseQaSources}
                        disabled={isCourseQaLoading}
                        className="rounded border border-slate-300 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:bg-slate-100 disabled:text-slate-400"
                      >
                        刷新
                      </button>
                    </div>
                    <select
                      value={courseQaSourceId}
                      onChange={(event) => setCourseQaSourceId(event.target.value)}
                      disabled={courseQaSources.length === 0 || isCourseQaLoading}
                      className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-500"
                    >
                      <option value="">
                        {courseQaSources.length > 0 ? '请选择课程 QA 文件...' : '暂无课程 QA 文件'}
                      </option>
                      {courseQaSources.map((source) => (
                        <option key={source.id} value={source.id}>
                          {getCourseQaSourceLabel(source)}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700">题目</label>
                    <select
                      value={courseQaItemId}
                      onChange={(event) => setCourseQaItemId(event.target.value)}
                      disabled={courseQaItems.length === 0 || isCourseQaLoading}
                      className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-500"
                    >
                      <option value="">
                        {courseQaItems.length > 0 ? '请选择题目...' : '暂无题目'}
                      </option>
                      {courseQaItems.map((item, index) => (
                        <option key={item.item_id || index} value={item.item_id}>
                          {getCourseQaQuestionLabel(item, index)}
                        </option>
                      ))}
                    </select>
                  </div>

                  {selectedCourseQaItem && (
                    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3">
                      <div className="text-xs font-medium text-slate-500">
                        {selectedCourseQaItem.topic || '课程 QA'}
                      </div>
                      <p className="mt-1 text-sm font-semibold leading-6 text-slate-900">
                        {selectedCourseQaItem.question}
                      </p>
                      <div className="mt-3 max-h-72 space-y-2 overflow-auto pr-1">
                        {(selectedCourseQaItem.answers || []).map((answer, index) => (
                          <div
                            key={answer.answer_id || index}
                            className="rounded border border-slate-200 bg-white px-3 py-2 text-sm leading-6 text-slate-700"
                          >
                            <span className="mr-2 font-semibold text-slate-950">
                              {answer.answer_id || `A${index + 1}`}
                            </span>
                            {answer.answer || answer.text}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium text-slate-700">问题</label>
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="输入要演示的问题"
                    className="mt-1 block h-28 w-full resize-none rounded border border-slate-300 p-2 text-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-100"
                  />
                </div>
              )}

              <div>
                <div className="flex items-center justify-between gap-3">
                  <label className="block text-sm font-medium text-slate-700">
                    {selectedDatasetConfig.collectionLabel}
                  </label>
                  <button
                    type="button"
                    onClick={fetchAvailableCollections}
                    className="rounded border border-slate-300 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                  >
                    刷新
                  </button>
                </div>
                <select
                  value={pipelineConfig.collectionId}
                  onChange={(event) => setPipelineConfig({ ...pipelineConfig, collectionId: event.target.value })}
                  disabled={availableCollections.length === 0}
                  className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-500"
                >
                  <option value="">
                    {availableCollections.length > 0 ? `请选择${selectedDatasetConfig.collectionLabel}...` : '暂无可用索引库'}
                  </option>
                  {availableCollections.map((collection) => (
                    <option key={collection.id} value={collection.id}>
                      {getCollectionOptionLabel(collection)}
                    </option>
                  ))}
                </select>
              </div>

              <div className={`grid gap-3 ${isLlmOnlyMode ? 'grid-cols-1' : 'grid-cols-2'}`}>
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
                  <p className="mt-1 text-xs text-slate-500">
                    {pipelineConfig.ragMode === 'llm_only'
                      ? 'LLM-only 不检索外部知识。'
                      : 'RAG 使用检索到的证据回答或排序。'}
                  </p>
                </div>
                {!isLlmOnlyMode && (
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
                )}
              </div>

              <button
                type="button"
                onClick={handleRunRagAnswer}
                disabled={isRagAnswerRunning}
                className="w-full rounded bg-green-600 px-4 py-3 text-base font-semibold text-white shadow-sm hover:bg-green-700 disabled:bg-green-300"
              >
                {isRagAnswerRunning ? '运行中...' : demoDataset === 'course_qa' ? '运行答案评估' : '运行'}
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
            scoreAlgorithm={safeRagAnswer.metadata.score_algorithm}
            ragMode={safeRagAnswer.metadata.rag_mode || pipelineConfig.ragMode}
          />

          {showDocumentVectorView && (
            <VectorProjectionView
              source="collection"
              collectionId={documentVectorCollectionId}
              collectionProvider={documentVectorProvider}
              queryVector={safeRagAnswer.metadata.query_embedding}
              retrievedHits={safeRagAnswer.retrievedHits}
              title="07 检索向量视图"
              compact
            />
          )}

          <EvaluationDashboard
            summary={selectedEvaluationSummary}
            status={evaluationStatus}
            title={demoDataset === 'course_qa' ? '课程 QA 答案评估' : `${selectedDatasetConfig.label} 对照评测`}
            description={demoDataset === 'paper' ? 'LLM-Wiki 论文评测摘要。' : '题目候选答案与外部知识检索链路摘要。'}
          />
        </div>
      </div>
    </div>
  );
};

export default Generation;
