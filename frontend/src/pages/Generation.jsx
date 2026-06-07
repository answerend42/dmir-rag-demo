/**
 * @file Generation.jsx
 * @brief 响应生成工作流页面。
 */
import { useMemo, useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { apiBaseUrl } from '../config/config';
import EvaluationDashboard from '../components/rag/EvaluationDashboard';
import MarkdownAnswer from '../components/rag/MarkdownAnswer';
import PipelineConfigPanel from '../components/rag/PipelineConfigPanel';
import RetrievalTracePanel from '../components/rag/RetrievalTracePanel';
import {
  createRagAnswerRequestPayload,
  createSafeRagAnswerViewModel,
  normalizeEvaluationSummary,
  removeForbiddenFields,
} from '../components/rag/ragViewModel';
import { courseQaMockRagAnswer, demoEvaluationSummary, paperDemoEvaluationSummary } from '../config/ragDemoData';

const DEFAULT_COURSE_QA_QUERY = '什么是自然语言处理？';
const EVALUATION_DATASETS = [
  { key: 'course_qa', label: '课程 QA', filename: 'course_qa_eval.json' },
  { key: 'paper', label: 'LLM-Wiki 论文', filename: 'paper_eval.json' },
];
const EVALUATION_FALLBACKS = {
  course_qa: demoEvaluationSummary,
  paper: paperDemoEvaluationSummary,
};

/**
 * @brief 渲染回答生成控件和检索上下文预览。
 * @returns {JSX.Element} 生成工作流页面。
 */
const Generation = () => {
  const location = useLocation();
  const [provider, setProvider] = useState('');
  const [modelName, setModelName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState('');
  const [query, setQuery] = useState(DEFAULT_COURSE_QA_QUERY);
  const [searchResults, setSearchResults] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [searchFiles, setSearchFiles] = useState([]);
  const [showReasoning, setShowReasoning] = useState(true);
  const [loadModel, setLoadModel] = useState(false);
  const [pipelineConfig, setPipelineConfig] = useState({
    ragMode: 'basic_rag',
    topK: 3,
    provider: 'mock',
    model: 'mock-generator',
    collectionId: 'course-qa-default',
  });
  const [ragAnswer, setRagAnswer] = useState(null);
  const [isRagAnswerRunning, setIsRagAnswerRunning] = useState(false);
  const [ragRequestStatus, setRagRequestStatus] = useState({
    type: 'info',
    message: '主路径为 POST /rag/answer；后端不可用时可使用课程 QA Mock fallback。',
  });
  const [legacyStatus, setLegacyStatus] = useState(null);
  const [selectedEvaluationDataset, setSelectedEvaluationDataset] = useState('course_qa');
  const [evaluationSummaries, setEvaluationSummaries] = useState(EVALUATION_FALLBACKS);
  const [evaluationStatus, setEvaluationStatus] = useState({
    type: 'info',
    message: '正在尝试加载评测摘要，失败时使用 fallback 摘要。',
  });

  const safeRagAnswer = useMemo(() => createSafeRagAnswerViewModel(ragAnswer), [ragAnswer]);
  const selectedEvaluationSummary = evaluationSummaries[selectedEvaluationDataset] || EVALUATION_FALLBACKS[selectedEvaluationDataset];

  /** @brief 加载旧生成流程的可选数据；失败不影响 /rag/answer dashboard。 */
  useEffect(() => {
    const fetchData = async () => {
      try {
        const modelsResponse = await fetch(`${apiBaseUrl}/generation/models`);
        if (modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          setModels(modelsData.models || {});
        } else {
          setModels({});
        }

        const filesResponse = await fetch(`${apiBaseUrl}/search-results`);
        if (filesResponse.ok) {
          const filesData = await filesResponse.json();
          setSearchFiles(filesData.files || []);
        } else {
          setSearchFiles([]);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        setModels({});
        setSearchFiles([]);
        setLegacyStatus('旧生成流程数据不可用；不影响 /rag/answer 展示。');
      }
    };

    fetchData();
  }, []);

  /** @brief 加载课程 QA 与论文评测摘要；后端或静态文件不可用时保留 fallback。 */
  useEffect(() => {
    const fetchEvaluationSummaries = async () => {
      const loadedSummaries = {};
      const failedLabels = [];

      await Promise.all(EVALUATION_DATASETS.map(async (dataset) => {
        try {
          const response = await fetch(`${apiBaseUrl}/eval/results/${dataset.filename}`);
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
  }, []);

  /** @brief 加载选中的搜索结果文件内容。 */
  useEffect(() => {
    const loadSearchResults = async () => {
      if (!selectedFile) {
        setQuery(DEFAULT_COURSE_QA_QUERY);
        setSearchResults([]);
        return;
      }

      try {
        const response = await fetch(`${apiBaseUrl}/search-results/${selectedFile}`);
        const data = await response.json();
        setQuery(data.query);
        setSearchResults(data.results);
      } catch (error) {
        console.error('Error loading search results:', error);
        setStatus('加载搜索结果失败');
      }
    };

    loadSearchResults();
  }, [selectedFile]);

  /** @brief 如果从搜索页面跳转过来，获取搜索结果。 */
  useEffect(() => {
    if (location.state) {
      const { query: searchQuery, results } = location.state;
      if (searchQuery) setQuery(searchQuery);
      if (results) setSearchResults(results);
    }
  }, [location]);

  const handleGenerate = async () => {
    if (!provider || !modelName) {
      setStatus('请选择生成模型');
      return;
    }

    if (!query /*|| searchResults.length === 0 */) {
      setStatus('请输入问题并确保有搜索结果');
      return;
    }

    setIsGenerating(true);
    setStatus('');
    try {
      const response = await fetch(`${apiBaseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          provider,
          model_name: modelName,
          search_results: searchResults,
          load_model: loadModel,
          api_key: apiKey || null,
          show_reasoning: showReasoning,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setLoadModel(false);
      setStatus(`生成完成！modelStatus: ${loadModel} 结果已保存至: ${data.saved_filepath}`);
    } catch (error) {
      console.error('Generation error:', error);
      setStatus(`生成失败: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setLoadModel(false);
    }
  };

  const handleRunRagAnswer = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setRagRequestStatus({
        type: 'error',
        message: '请输入问题后再调用 /rag/answer。',
      });
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
        let detail = `HTTP ${response.status}`;
        try {
          const errorBody = await response.json();
          detail = errorBody.detail || detail;
        } catch {
          detail = response.statusText || detail;
        }
        throw new Error(detail);
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

  const handleUseMockAnswer = () => {
    setRagAnswer(courseQaMockRagAnswer);
    setPipelineConfig((currentConfig) => ({
      ...currentConfig,
      ragMode: 'basic_rag',
      topK: 3,
      provider: 'mock',
      model: 'mock-generator',
      collectionId: 'course-qa-default',
    }));
    setRagRequestStatus({
      type: 'info',
      message: '当前展示课程 QA Mock fallback；默认主路径仍是 POST /rag/answer。',
    });
  };

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">响应生成</h2>
      
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-4 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              <div>
                    <label className="block text-sm font-medium mb-1">提问</label>
                    <textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Enter your question..."
                      className="block w-full p-2 border rounded h-32 resize-none"
                    />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">检索文档（可选）</label>
                <select
                  value={selectedFile}
                  onChange={(e) => setSelectedFile(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Select search results file...</option>
                  {searchFiles.map(file => (
                    <option key={file.id} value={file.id}>
                      {file.name}
                    </option>
                  ))}
                </select>
                {legacyStatus && (
                  <div className="mt-2 rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                    {legacyStatus}
                  </div>
                )}
              </div>

              <>
                <div>
                  <label className="block text-sm font-medium mb-1">生成模型提供方</label>
                  <select
                    value={provider}
                    onChange={(e) => setProvider(e.target.value)}
                    className="block w-full p-2 border rounded"
                  >
                    <option value="">Select provider...</option>
                    {Object.keys(models).map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>

                {provider && (
                  <div>
                    <label className="block text-sm font-medium mb-1">生成模型</label>
                    <select
                      value={modelName}
                      onChange={(e) => {setModelName(e.target.value); setLoadModel(true)}}
                      className="block w-full p-2 border rounded"
                    >
                      <option value="">Select model...</option>
                      {Object.entries(models[provider] || {}).map(([id, name]) => (
                        <option key={id} value={id}>
                          {id === 'deepseek-v3' ? 'DeepSeek V3' :
                           id === 'deepseek-r1' ? 'DeepSeek R1' :
                           name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {(provider === 'openai' || provider === 'deepseek') && (
                  <div>
                    <label className="block text-sm font-medium mb-1">API Key</label>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="Enter your API key..."
                      className="block w-full p-2 border rounded"
                    />
                  </div>
                )}

                {provider === 'deepseek' && modelName === 'deepseek-r1' && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="showReasoning"
                      checked={showReasoning}
                      onChange={(e) => setShowReasoning(e.target.checked)}
                      className="rounded border-gray-300 text-green-500 focus:ring-green-500"
                    />
                    <label htmlFor="showReasoning" className="text-sm font-medium">
                      显示思维链过程
                    </label>
                  </div>
                )}

                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-green-300"
                >
                  {isGenerating ? '生成回答中...' : '生成回答'}
                </button>

                {status && (
                  <div className={`p-4 rounded-lg ${
                    status.includes('失败') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                  }`}>
                    {status}
                  </div>
                )}
              </>
            </div>
          </div>

          <PipelineConfigPanel
            config={pipelineConfig}
            onConfigChange={setPipelineConfig}
            onRunRagAnswer={handleRunRagAnswer}
            onUseMockAnswer={handleUseMockAnswer}
            isRunning={isRagAnswerRunning}
            requestStatus={ragRequestStatus}
          />
        </div>

        <div className="col-span-8 space-y-6">
          <div className="rounded-lg border bg-slate-900 p-4 text-white shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-xl font-semibold">Contract RAG 展示</h3>
                <p className="text-sm text-slate-300">
                  前端只读取 RagAnswer 字段，可复用在课程 QA 和论文 RAG。
                </p>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded bg-white/10 px-3 py-2">
                  <div className="text-slate-300">Hits</div>
                  <div className="text-lg font-semibold">{safeRagAnswer.retrievedHits.length}</div>
                </div>
                <div className="rounded bg-white/10 px-3 py-2">
                  <div className="text-slate-300">Citations</div>
                  <div className="text-lg font-semibold">{safeRagAnswer.citations.length}</div>
                </div>
                <div className="rounded bg-white/10 px-3 py-2">
                  <div className="text-slate-300">Trace</div>
                  <div className="text-lg font-semibold">{safeRagAnswer.trace.length}</div>
                </div>
              </div>
            </div>
          </div>

          <MarkdownAnswer
            answerMarkdown={safeRagAnswer.answerMarkdown}
            warnings={safeRagAnswer.warnings}
            contractVersion={safeRagAnswer.contractVersion}
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

          <div className="rounded-lg border bg-white p-4 shadow-sm">
            <h3 className="mb-4 text-xl font-semibold">旧流程检索上下文</h3>
            {selectedFile ? (
              <div className="max-h-[300px] space-y-4 overflow-y-auto">
                {searchResults.map((result, idx) => (
                  <div key={`${result.text}-${idx}`} className="rounded border bg-gray-50 p-4">
                    <div className="mb-2 flex items-start justify-between gap-3">
                      <span className="text-sm font-medium text-gray-500">
                        Match Score: {(Number(result.score || 0) * 100).toFixed(1)}%
                      </span>
                      <div className="text-sm text-gray-500">
                        <div>Source: {result.metadata?.source || '-'}</div>
                        <div>Page: {result.metadata?.page || result.metadata?.page_number || '-'}</div>
                      </div>
                    </div>
                    <p className="whitespace-pre-wrap text-sm">{result.text}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded border border-dashed p-4 text-sm text-gray-500">无检索上下文。</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Generation;
