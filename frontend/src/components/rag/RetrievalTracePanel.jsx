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
 * @brief 判断 citation 与 retrieved hit 是否指向同一条证据。
 * @param {object} citation 引用条目。
 * @param {object} hit 检索命中条目。
 * @returns {boolean} 两者可以合并展示时返回 true。
 */
const sameEvidence = (citation = {}, hit = {}) => {
  if (citation.chunk_id && hit.chunk_id && citation.chunk_id === hit.chunk_id) {
    return true;
  }
  return Boolean(
    citation.doc_id
      && hit.doc_id
      && citation.doc_id === hit.doc_id
      && citation.quote
      && hit.text === citation.quote
  );
};

/**
 * @brief 为 citation 补齐对应检索命中的相似度分数。
 * @param {Array<object>} citations 引用条目列表。
 * @param {Array<object>} retrievedHits 检索命中列表。
 * @returns {Array<object>} 带 score/rank 的引用条目。
 */
const attachCitationScores = (citations = [], retrievedHits = []) =>
  citations.map((citation, index) => {
    const matchedHit = retrievedHits.find((hit) => sameEvidence(citation, hit)) || retrievedHits[index];
    return {
      ...citation,
      score: citation.score ?? matchedHit?.score,
      rank: citation.rank ?? matchedHit?.rank,
    };
  });

const defaultScoreAlgorithm = {
  name: 'Chroma HNSW cosine',
  formula: 'score = 1 - Chroma distance',
  note: 'Chroma distance 越小越相近；前端展示的 score 越大越相关。',
};

/**
 * @brief 渲染检索命中、引用和阶段追踪。
 * @param {{retrievedHits: Array<object>, citations: Array<object>, trace: Array<object>, scoreAlgorithm: object, ragMode: string}} props 组件属性。
 * @returns {JSX.Element} 检索证据与 trace 面板。
 */
const RetrievalTracePanel = ({ retrievedHits = [], citations = [], trace = [], scoreAlgorithm = null, ragMode = 'basic_rag' }) => {
  const isLlmOnly = ragMode === 'llm_only';
  const visibleCitations = !isLlmOnly && citations.length > 0
    ? attachCitationScores(citations, retrievedHits)
    : !isLlmOnly ? retrievedHits.slice(0, 3) : [];
  const algorithm = scoreAlgorithm || defaultScoreAlgorithm;

  return (
    <section className="rounded-md border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-6 py-4">
        <div>
          <h3 className="text-base font-semibold text-slate-950">证据</h3>
          <p className="mt-1 text-xs text-slate-500">
            {isLlmOnly
              ? `LLM-only 未执行检索 · ${trace.length} 个阶段`
              : `${retrievedHits.length} 个检索命中 · ${citations.length} 条引用 · ${trace.length} 个阶段`}
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
        {isLlmOnly ? (
          <div className="rounded-md border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm leading-6 text-slate-600">
            LLM-only 模式未执行相似性检索，因此没有 RAG 证据。模型回答中的判断依据不作为“已引用证据”展示。
          </div>
        ) : (
          <>
            <div className="mb-3 flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-900">已引用证据</h4>
              <span className="text-xs text-slate-500">点击正文中的证据编号可定位到这里</span>
            </div>
            <div className="mb-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-600">
              相似性分数算法：{algorithm.name || defaultScoreAlgorithm.name}；{algorithm.formula || defaultScoreAlgorithm.formula}。
              {algorithm.note || defaultScoreAlgorithm.note}
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
                          Top {item.rank || index + 1} · 相似性分数 {formatScore(item.score)}
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
                          #{hit.rank || index + 1} · 相似性分数 {formatScore(hit.score)}
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
          </>
        )}

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
