/**
 * @file MarkdownAnswer.jsx
 * @brief 展示 RagAnswer 中的 Markdown 回答和警告信息。
 */
/* eslint-disable react/prop-types */
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const citationPattern = /\[证据(\d+)\]/g;

const withCitationLinks = (markdown = '') =>
  markdown.replace(citationPattern, '[证据$1](#evidence-$1)');

const getCitationTitle = (item = {}, index) => {
  const source = item.source || item.doc_id || '未知来源';
  const page = item.page_number || item.metadata?.page_numbers?.[0];
  return page ? `${source} · p.${page}` : source;
};

const buildSourceCards = (citations = [], retrievedHits = []) => {
  const citationCards = citations.map((citation, index) => ({
    id: citation.chunk_id || `citation-${index + 1}`,
    number: index + 1,
    title: getCitationTitle(citation, index),
    quote: citation.quote || '',
    href: `#evidence-${index + 1}`,
  }));

  if (citationCards.length > 0) {
    return citationCards;
  }

  return retrievedHits.slice(0, 3).map((hit, index) => ({
    id: hit.chunk_id || `hit-${index + 1}`,
    number: index + 1,
    title: getCitationTitle(hit, index),
    quote: hit.text || '',
    href: `#evidence-${index + 1}`,
  }));
};

/**
 * @brief 渲染有引用支撑的 Markdown 回答。
 * @param {{answerMarkdown: string, warnings?: string[], contractVersion?: string, citations?: Array<object>, retrievedHits?: Array<object>}} props 组件属性。
 * @returns {JSX.Element} Markdown 回答面板。
 */
const MarkdownAnswer = ({
  answerMarkdown,
  warnings = [],
  contractVersion = 'unknown',
  citations = [],
  retrievedHits = [],
}) => {
  const sourceCards = buildSourceCards(citations, retrievedHits);

  return (
    <section className="overflow-hidden rounded-md border border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-6 py-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold text-slate-950">RAG 回答</h3>
            <p className="mt-1 text-xs text-slate-500">Contract v{contractVersion}</p>
          </div>
          {warnings.length > 0 && (
            <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
              {warnings.length} 条警告
            </span>
          )}
        </div>

        {sourceCards.length > 0 && (
          <div className="mt-5">
            <div className="mb-2 text-xs font-medium text-slate-500">引用来源</div>
            <div className="grid gap-2 md:grid-cols-3">
              {sourceCards.map((card) => (
                <a
                  key={card.id}
                  href={card.href}
                  className="min-w-0 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-left transition hover:border-emerald-300 hover:bg-emerald-50 focus:outline-none focus:ring-2 focus:ring-emerald-200"
                >
                  <div className="mb-1 flex items-center gap-2">
                    <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-emerald-700 px-1.5 text-[11px] font-semibold text-white">
                      {card.number}
                    </span>
                    <span className="truncate text-xs font-semibold text-slate-800">{card.title}</span>
                  </div>
                  <p className="line-clamp-2 text-xs leading-5 text-slate-600">
                    {card.quote || '无引用片段'}
                  </p>
                </a>
              ))}
            </div>
          </div>
        )}
      </div>

      {warnings.length > 0 && (
        <div className="border-b border-amber-100 bg-amber-50/70 px-6 py-3">
          <div className="space-y-2">
            {warnings.map((warning, index) => (
              <div key={`${warning}-${index}`} className="text-sm text-amber-900">
                {warning}
              </div>
            ))}
          </div>
        </div>
      )}

      <article className="max-w-[78ch] px-6 py-6 text-slate-800">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => <h1 className="mb-4 text-2xl font-semibold leading-tight text-slate-950">{children}</h1>,
            h2: ({ children }) => <h2 className="mb-3 mt-6 text-xl font-semibold leading-tight text-slate-950 first:mt-0">{children}</h2>,
            h3: ({ children }) => <h3 className="mb-2 mt-5 text-base font-semibold text-slate-900">{children}</h3>,
            p: ({ children }) => <p className="mb-4 text-[15px] leading-7 text-slate-800 last:mb-0">{children}</p>,
            ul: ({ children }) => <ul className="mb-4 list-disc space-y-2 pl-5 text-[15px] leading-7 text-slate-800">{children}</ul>,
            ol: ({ children }) => <ol className="mb-4 list-decimal space-y-2 pl-5 text-[15px] leading-7 text-slate-800">{children}</ol>,
            li: ({ children }) => <li className="pl-1">{children}</li>,
            table: ({ children }) => <div className="my-5 overflow-x-auto rounded-md border border-slate-200"><table className="min-w-full divide-y divide-slate-200 text-sm">{children}</table></div>,
            th: ({ children }) => <th className="bg-slate-50 px-3 py-2 text-left text-xs font-semibold text-slate-600">{children}</th>,
            td: ({ children }) => <td className="border-t border-slate-100 px-3 py-2 text-slate-700">{children}</td>,
            strong: ({ children }) => <strong className="font-semibold text-slate-950">{children}</strong>,
            a: ({ href, children }) => {
              const citationMatch = href?.match(/^#evidence-(\d+)$/);
              if (citationMatch) {
                return (
                  <a
                    href={href}
                    className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-full bg-emerald-50 px-1.5 py-0.5 text-[11px] font-semibold text-emerald-800 ring-1 ring-emerald-200 transition hover:bg-emerald-100 focus:outline-none focus:ring-2 focus:ring-emerald-300"
                    aria-label={`跳转到${children}`}
                  >
                    {children}
                  </a>
                );
              }
              return <a href={href} className="text-emerald-700 underline decoration-emerald-300 underline-offset-4">{children}</a>;
            },
          }}
        >
          {withCitationLinks(answerMarkdown || '暂无回答。')}
        </ReactMarkdown>
      </article>
    </section>
  );
};

export default MarkdownAnswer;
