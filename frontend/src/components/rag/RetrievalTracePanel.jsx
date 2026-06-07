/**
 * @file RetrievalTracePanel.jsx
 * @brief 展示 RagAnswer 的检索命中、引用和流水线 trace。
 */
/* eslint-disable react/prop-types */
import { formatLatency, formatScore } from './ragViewModel';

const renderListValue = (value) => {
  if (Array.isArray(value)) {
    return value.filter(Boolean).join(' / ') || '-';
  }
  return value || '-';
};

const renderJsonSummary = (value) => {
  if (!value || Object.keys(value).length === 0) {
    return '-';
  }
  return JSON.stringify(value);
};

const renderSourceLabel = (item = {}) => {
  const page = item.page_number || item.metadata?.page_numbers?.[0] || item.metadata?.page;
  return [item.source || item.doc_id || '未知来源', page ? `p.${page}` : null].filter(Boolean).join(' · ');
};

const renderEvidenceMeta = (item = {}) => {
  const section = renderListValue(item.section_path || item.metadata?.section_path);
  const type = item.metadata?.block_type || item.metadata?.block_types;
  return [renderSourceLabel(item), section !== '-' ? section : null, type].filter(Boolean).join(' · ');
};

/**
 * @brief 渲染检索命中、引用和阶段追踪。
 * @param {{retrievedHits: Array<object>, citations: Array<object>, trace: Array<object>}} props 组件属性。
 * @returns {JSX.Element} 检索证据与 trace 面板。
 */
const RetrievalTracePanel = ({ retrievedHits = [], citations = [], trace = [] }) => {
  const visibleCitations = citations.length > 0 ? citations : retrievedHits.slice(0, 3);

  return (
    <section className="rounded-md border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-6 py-4">
        <div>
          <h3 className="text-base font-semibold text-slate-950">证据</h3>
          <p className="mt-1 text-xs text-slate-500">
            {retrievedHits.length} 个检索命中 · {citations.length} 条引用 · {trace.length} 个阶段
          </p>
        </div>
        <a
          href="#trace-details"
          className="rounded-full bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-700 ring-1 ring-slate-200 transition hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-200"
        >
          查看 trace
        </a>
      </div>

      <div className="px-6 py-5">
        <div className="mb-3 flex items-center justify-between">
          <h4 className="text-sm font-semibold text-slate-900">已引用证据</h4>
          <span className="text-xs text-slate-500">点击正文中的证据编号可定位到这里</span>
        </div>

        {visibleCitations.length > 0 ? (
          <div className="divide-y divide-slate-100 rounded-md border border-slate-200">
            {visibleCitations.map((item, index) => (
              <article
                id={`evidence-${index + 1}`}
                key={`${item.doc_id || item.source}-${item.chunk_id || index}`}
                className="scroll-mt-24 px-4 py-4 transition target:bg-emerald-50"
              >
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-emerald-700 px-2 text-xs font-semibold text-white">
                      {index + 1}
                    </span>
                    <span className="truncate text-sm font-semibold text-slate-900">
                      {renderSourceLabel(item)}
                    </span>
                  </div>
                  {formatScore(item.score) !== '-' && (
                    <span className="rounded-full bg-slate-50 px-2 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                      score {formatScore(item.score)}
                    </span>
                  )}
                </div>
                <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">
                  {item.quote || item.text || '无引用片段'}
                </p>
                <div className="mt-2 text-xs text-slate-500">
                  {renderEvidenceMeta(item)}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="rounded-md border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-500">
            暂无引用证据。
          </div>
        )}

        <details className="mt-5 rounded-md border border-slate-200">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-800">
            全部检索命中
          </summary>
          <div className="max-h-[420px] divide-y divide-slate-100 overflow-y-auto border-t border-slate-100">
            {retrievedHits.length > 0 ? (
              retrievedHits.map((hit, index) => (
                <article key={`${hit.chunk_id}-${hit.rank}`} className="px-4 py-4">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-emerald-800">
                      #{hit.rank || index + 1} · score {formatScore(hit.score)}
                    </span>
                    <span className="max-w-[60%] truncate text-xs text-slate-500">{renderSourceLabel(hit)}</span>
                  </div>
                  <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{hit.text}</p>
                </article>
              ))
            ) : (
              <div className="px-4 py-6 text-sm text-slate-500">暂无检索命中。</div>
            )}
          </div>
        </details>

        <details id="trace-details" className="mt-3 rounded-md border border-slate-200">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-800">
            流水线 Trace
          </summary>
          <div className="divide-y divide-slate-100 border-t border-slate-100">
            {trace.length > 0 ? (
              trace.map((stage, index) => (
                <div key={`${stage.stage_name}-${index}`} className="px-4 py-4">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <span className="text-sm font-semibold text-slate-900">{index + 1}. {stage.stage_name}</span>
                    <span className="text-xs text-slate-500">{formatLatency(stage.latency_ms)}</span>
                  </div>
                  <div className="space-y-1 break-words text-xs leading-5 text-slate-600">
                    <div>Input: {renderJsonSummary(stage.input_summary)}</div>
                    <div>Output: {renderJsonSummary(stage.output_summary)}</div>
                  </div>
                </div>
              ))
            ) : (
              <div className="px-4 py-6 text-sm text-slate-500">暂无 trace。</div>
            )}
          </div>
        </details>
      </div>
    </section>
  );
};

export default RetrievalTracePanel;
