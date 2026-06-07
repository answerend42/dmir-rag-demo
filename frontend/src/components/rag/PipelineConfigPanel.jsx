/**
 * @file PipelineConfigPanel.jsx
 * @brief 配置 RAG 展示请求的模式、检索数量和模型信息。
 */
/* eslint-disable react/prop-types */
const RAG_MODES = [
  { value: 'llm_only', label: 'LLM-only' },
  { value: 'basic_rag', label: 'Basic RAG' },
  { value: 'optimized_rag', label: 'Optimized RAG' },
];

const CONTROL_CLASS =
  'mt-1 block w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100';

const CONTROL_IDS = {
  ragMode: 'rag-config-mode',
  topK: 'rag-config-top-k',
  provider: 'rag-config-provider',
  model: 'rag-config-model',
  collectionId: 'rag-config-collection',
};

/**
 * @brief 渲染 RAG pipeline 配置控件。
 * @param {object} props 组件属性。
 * @returns {JSX.Element} 配置面板。
 */
const PipelineConfigPanel = ({
  config,
  onConfigChange,
  onRunRagAnswer,
  onUseMockAnswer,
  isRunning,
  requestStatus,
}) => {
  const updateConfig = (key, value) => {
    onConfigChange({ ...config, [key]: value });
  };

  return (
    <section className="rounded-lg border bg-white p-4 shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Pipeline 配置</h3>
        <p className="text-xs text-gray-500">只生成前端请求配置，不读取后端私有文件。</p>
      </div>

      <div className="space-y-4">
        <div>
          <label htmlFor={CONTROL_IDS.ragMode} className="block text-sm font-medium text-gray-700">RAG 模式</label>
          <select
            id={CONTROL_IDS.ragMode}
            value={config.ragMode}
            onChange={(event) => updateConfig('ragMode', event.target.value)}
            className={CONTROL_CLASS}
          >
            {RAG_MODES.map((mode) => (
              <option key={mode.value} value={mode.value}>
                {mode.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor={CONTROL_IDS.topK} className="block text-sm font-medium text-gray-700">Top K：{config.topK}</label>
          <input
            id={CONTROL_IDS.topK}
            type="range"
            min="1"
            max="10"
            value={config.topK}
            onChange={(event) => updateConfig('topK', Number(event.target.value))}
            className="mt-2 block w-full"
          />
        </div>

        <div className="grid grid-cols-1 gap-3">
          <div>
            <label htmlFor={CONTROL_IDS.provider} className="block text-sm font-medium text-gray-700">Provider</label>
            <input
              id={CONTROL_IDS.provider}
              value={config.provider}
              onChange={(event) => updateConfig('provider', event.target.value)}
              className={CONTROL_CLASS}
            />
          </div>
          <div>
            <label htmlFor={CONTROL_IDS.model} className="block text-sm font-medium text-gray-700">Model</label>
            <input
              id={CONTROL_IDS.model}
              value={config.model}
              onChange={(event) => updateConfig('model', event.target.value)}
              className={CONTROL_CLASS}
            />
          </div>
        </div>

        <div>
          <label htmlFor={CONTROL_IDS.collectionId} className="block text-sm font-medium text-gray-700">Collection</label>
          <input
            id={CONTROL_IDS.collectionId}
            value={config.collectionId}
            onChange={(event) => updateConfig('collectionId', event.target.value)}
            className={CONTROL_CLASS}
          />
        </div>

        <div className="grid grid-cols-1 gap-2">
          <button
            type="button"
            onClick={onRunRagAnswer}
            disabled={isRunning}
            className="rounded bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:bg-green-300"
          >
            {isRunning ? '调用 /rag/answer 中...' : '运行 /rag/answer'}
          </button>
          <button
            type="button"
            onClick={onUseMockAnswer}
            className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300"
          >
            使用课程 QA Mock fallback
          </button>
        </div>

        {requestStatus && (
          <div className={`rounded border px-3 py-2 text-sm ${
            requestStatus.type === 'error'
              ? 'border-amber-200 bg-amber-50 text-amber-800'
              : 'border-blue-200 bg-blue-50 text-blue-800'
          }`}>
            {requestStatus.message}
          </div>
        )}
      </div>
    </section>
  );
};

export default PipelineConfigPanel;
