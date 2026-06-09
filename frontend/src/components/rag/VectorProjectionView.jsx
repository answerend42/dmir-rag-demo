/**
 * @file VectorProjectionView.jsx
 * @brief 可复用的嵌入向量投影视图。
 */
/* eslint-disable react/prop-types */
import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import RandomImage from '../RandomImage';
import { apiBaseUrl } from '../../config/config';

const VectorProjectionScene3D = lazy(() => import('./VectorProjectionScene3D'));

const VIEWBOX_WIDTH = 1000;
const VIEWBOX_HEIGHT = 560;
const VIEWBOX_PADDING = 56;

const METHOD_OPTIONS = [
  { id: 'tsne', label: 't-SNE', requiresQuery: false },
  { id: 'pca', label: 'PCA', requiresQuery: false },
];

const EMPTY_PROJECTION = {
  method: 'tsne',
  method_label: 't-SNE',
  target_dimensions: 3,
  available_dimensions: [3, 2],
  dimension: 0,
  points: [],
  overlays: [],
  axes: {
    x: { label: 'x', explained_variance: null },
    y: { label: 'y', explained_variance: null },
    z: { label: 'z', explained_variance: null },
  },
};

const PAGE_COLORS = [
  'oklch(0.49 0.105 178)',
  'oklch(0.58 0.13 56)',
  'oklch(0.54 0.12 235)',
  'oklch(0.56 0.14 28)',
  'oklch(0.48 0.10 290)',
  'oklch(0.50 0.11 138)',
];

const formatNumber = (value, digits = 3) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return '-';
  }
  return numericValue.toFixed(digits);
};

const clampUnit = (value) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 0.5;
  }
  return Math.min(Math.max(numericValue, 0), 1);
};

const toSvgPoint = (point) => {
  const width = VIEWBOX_WIDTH - VIEWBOX_PADDING * 2;
  const height = VIEWBOX_HEIGHT - VIEWBOX_PADDING * 2;
  return {
    ...point,
    x: VIEWBOX_PADDING + clampUnit(point.x) * width,
    y: VIEWBOX_HEIGHT - VIEWBOX_PADDING - clampUnit(point.y) * height,
  };
};

const axisSummary = (axis) => {
  const label = axis?.label || '-';
  const explained = Number(axis?.explained_variance);
  if (!Number.isFinite(explained)) {
    return label;
  }
  return `${label} ${formatNumber(explained * 100, 1)}%`;
};

const colorForPage = (pageNumber) => {
  const pageIndex = Math.max(Number(pageNumber) || 1, 1) - 1;
  return PAGE_COLORS[pageIndex % PAGE_COLORS.length];
};

const normalizeChunkId = (value) => String(value ?? '').trim();

const getPointChunkId = (point) => {
  const metadata = point.embedding?.metadata || {};
  return normalizeChunkId(metadata.chunk || metadata.chunk_id || point.index + 1);
};

const getPointLabel = (point) => {
  const metadata = point.embedding?.metadata || {};
  return `分块 ${metadata.chunk_id || metadata.chunk || point.index + 1} · p.${metadata.page_number || metadata.page || '-'}`;
};

const pointMatchesFilter = (point, filterText) => {
  const query = filterText.trim().toLowerCase();
  if (!query) {
    return true;
  }
  const metadata = point.embedding?.metadata || {};
  return [
    metadata.chunk,
    metadata.chunk_id,
    metadata.page,
    metadata.page_number,
    metadata.page_range,
    metadata.content,
    metadata.document_name,
  ].some((value) => String(value || '').toLowerCase().includes(query));
};

const buildHitMap = (retrievedHits = []) => {
  const hitMap = new Map();
  retrievedHits.forEach((hit, index) => {
    const metadata = hit.metadata || {};
    const chunkId = normalizeChunkId(hit.chunk_id || metadata.chunk || metadata.chunk_id);
    if (!chunkId) {
      return;
    }
    hitMap.set(chunkId, {
      rank: hit.rank || index + 1,
      score: hit.score,
    });
  });
  return hitMap;
};

/**
 * @brief 展示当前选中投影点对应的分块详情。
 * @param {{point: object | null, hitInfo: object | null}} props 组件属性。
 * @returns {JSX.Element} 分块详情面板。
 */
const SelectedPointPanel = ({ point, hitInfo }) => {
  if (!point) {
    return (
      <div className="rounded-md border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
        选择一个点后，这里会显示对应分块、页码和文本片段。
      </div>
    );
  }

  const metadata = point.embedding?.metadata || {};
  return (
    <article className="rounded-md border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">
          分块 {metadata.chunk_id || metadata.chunk || point.index + 1}
        </h3>
        <div className="flex flex-wrap gap-2">
          {hitInfo && (
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800 ring-1 ring-emerald-200">
              Top {hitInfo.rank} · 相似性 {formatNumber(hitInfo.score)}
            </span>
          )}
          <span className="rounded-full bg-slate-50 px-3 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
            p.{metadata.page_number || metadata.page || '-'}
          </span>
        </div>
      </div>
      <div className="mt-2 grid gap-2 text-xs text-slate-500 sm:grid-cols-2">
        <div>模型：{metadata.embedding_model || '-'}</div>
        <div>提供方：{metadata.embedding_provider || '-'}</div>
        <div>维度：{metadata.vector_dimension || point.embedding?.embedding?.length || '-'}</div>
        <div>页码范围：{metadata.page_range || '-'}</div>
      </div>
      <p className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap text-sm leading-6 text-slate-700">
        {metadata.content || '无分块文本'}
      </p>
    </article>
  );
};

const QueryMarker = ({ point }) => (
  <g>
    <path
      d={`M ${point.x} ${point.y - 15} L ${point.x + 15} ${point.y} L ${point.x} ${point.y + 15} L ${point.x - 15} ${point.y} Z`}
      fill="oklch(0.23 0.026 218)"
      stroke="white"
      strokeWidth="3"
    />
    <circle cx={point.x} cy={point.y} r="4" fill="white" />
    <text x={point.x + 20} y={point.y - 16} className="fill-slate-900 text-[18px] font-bold">
      Query
    </text>
    <title>{point.label || '用户查询向量'}</title>
  </g>
);

/**
 * @brief 渲染可内嵌到 04/07 的后端向量投影视图。
 * @param {object} props 组件属性。
 * @returns {JSX.Element} 向量投影视图。
 */
const VectorProjectionView = ({
  source = 'embedded-file',
  collectionId = '',
  collectionProvider = 'chroma',
  queryVector = null,
  retrievedHits = [],
  title = '向量投影视图',
  compact = false,
}) => {
  const [embeddedDocs, setEmbeddedDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('');
  const [projection, setProjection] = useState(EMPTY_PROJECTION);
  const [projectionMethod, setProjectionMethod] = useState('tsne');
  const [projectionDimensions, setProjectionDimensions] = useState(3);
  const [status, setStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [filterText, setFilterText] = useState('');
  const [zoom, setZoom] = useState(1);

  const isCollectionMode = source === 'collection';
  const hitMap = useMemo(() => buildHitMap(retrievedHits), [retrievedHits]);
  const overlays = useMemo(() => (
    Array.isArray(queryVector) && queryVector.length > 0
      ? [{ id: 'query', role: 'query', label: '用户查询向量', vector: queryVector }]
      : []
  ), [queryVector]);
  const hasQueryOverlay = overlays.length > 0;
  const methodOptions = useMemo(() => (
    projection.available_methods?.length ? projection.available_methods : METHOD_OPTIONS
  ), [projection.available_methods]);
  const allowedProjectionMethods = useMemo(() => methodOptions.map((option) => option.id), [methodOptions]);
  const defaultProjectionMethod = methodOptions[0]?.id || 'tsne';
  const requestedProjectionMethod = allowedProjectionMethods.includes(projectionMethod)
    ? projectionMethod
    : defaultProjectionMethod;
  const availableProjectionDimensions = useMemo(() => (
    projection.available_dimensions?.length ? projection.available_dimensions : [3, 2]
  ), [projection.available_dimensions]);
  const requestedProjectionDimensions = availableProjectionDimensions.includes(projectionDimensions)
    ? projectionDimensions
    : availableProjectionDimensions[0] || 3;

  useEffect(() => {
    if (!allowedProjectionMethods.includes(projectionMethod)) {
      setProjectionMethod(defaultProjectionMethod);
    }
  }, [allowedProjectionMethods, defaultProjectionMethod, projectionMethod]);

  useEffect(() => {
    if (!availableProjectionDimensions.includes(projectionDimensions)) {
      setProjectionDimensions(availableProjectionDimensions[0] || 3);
    }
  }, [availableProjectionDimensions, projectionDimensions]);

  useEffect(() => {
    if (isCollectionMode) {
      return;
    }

    const fetchEmbeddedDocs = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/list-embedded`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        const documents = Array.isArray(payload.documents) ? payload.documents : [];
        setEmbeddedDocs(documents);
        setSelectedDoc((currentDoc) => (
          documents.some((doc) => doc.name === currentDoc) ? currentDoc : documents[0]?.name || ''
        ));
      } catch (error) {
        setStatus(`获取向量文件失败: ${error.message}`);
      }
    };

    fetchEmbeddedDocs();
  }, [isCollectionMode]);

  useEffect(() => {
    const activeSource = isCollectionMode ? collectionId : selectedDoc;
    if (!activeSource) {
      setProjection(EMPTY_PROJECTION);
      setStatus(isCollectionMode ? '请选择索引库后查看向量视图。' : '');
      return;
    }

    const fetchProjection = async () => {
      setIsLoading(true);
      setStatus(isCollectionMode ? '正在计算 collection 向量投影...' : '正在计算向量文件投影...');
      setSelectedIndex(null);
      setHoveredIndex(null);
      try {
        const endpoint = isCollectionMode
          ? `${apiBaseUrl}/collections/${encodeURIComponent(collectionProvider)}/${encodeURIComponent(activeSource)}/projection`
          : `${apiBaseUrl}/embedded-docs/${encodeURIComponent(activeSource)}/projection`;
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            method: requestedProjectionMethod,
            target_dimensions: requestedProjectionDimensions,
            overlays,
          }),
        });
        if (!response.ok) {
          const errorPayload = await response.json().catch(() => ({}));
          throw new Error(errorPayload.detail || `HTTP ${response.status}`);
        }
        const payload = await response.json();
        setProjection({
          ...EMPTY_PROJECTION,
          ...payload,
          axes: payload.axes || EMPTY_PROJECTION.axes,
          points: Array.isArray(payload.points) ? payload.points : [],
          overlays: Array.isArray(payload.overlays) ? payload.overlays : [],
        });
        setStatus('');
      } catch (error) {
        setProjection(EMPTY_PROJECTION);
        setStatus(`计算向量投影失败: ${error.message}`);
      } finally {
        setIsLoading(false);
      }
    };

    fetchProjection();
  }, [
    collectionId,
    collectionProvider,
    isCollectionMode,
    overlays,
    requestedProjectionDimensions,
    requestedProjectionMethod,
    selectedDoc,
  ]);

  const svgProjection = useMemo(() => ({
    ...projection,
    points: projection.points.map(toSvgPoint),
    overlays: projection.overlays.map(toSvgPoint),
  }), [projection]);
  const isThreeDimensional = Number(projection.target_dimensions) === 3
    && projection.points.some((point) => Number.isFinite(Number(point.z)));
  const chartProjection = isThreeDimensional ? projection : svgProjection;
  const visiblePoints = useMemo(
    () => chartProjection.points.filter((point) => pointMatchesFilter(point, filterText)),
    [chartProjection.points, filterText]
  );
  const selectedPoint = visiblePoints.find((point) => point.index === selectedIndex) || visiblePoints[0] || null;
  const hoveredPoint = chartProjection.points.find((point) => point.index === hoveredIndex) || null;
  const selectedHitInfo = selectedPoint ? hitMap.get(getPointChunkId(selectedPoint)) || null : null;
  const queryPoint = chartProjection.overlays.find((point) => point.role === 'query');
  const axisX = projection.axes?.x || EMPTY_PROJECTION.axes.x;
  const axisY = projection.axes?.y || EMPTY_PROJECTION.axes.y;
  const axisZ = projection.axes?.z || EMPTY_PROJECTION.axes.z;
  const centerX = VIEWBOX_WIDTH / 2;
  const centerY = VIEWBOX_HEIGHT / 2;
  const chartHeightClass = compact ? 'h-[460px]' : 'h-[560px]';

  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900">{title}</h3>
            <p className="mt-1 text-xs text-slate-500">
              {projection.method_label || '后端投影'} · {visiblePoints.length} / {projection.points.length} 个分块 · 原始维度 {projection.dimension || '-'}
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-slate-600">
            <span className="rounded-full bg-slate-50 px-3 py-1 ring-1 ring-slate-200">
              {axisSummary(axisX)}
            </span>
            <span className="rounded-full bg-slate-50 px-3 py-1 ring-1 ring-slate-200">
              {axisSummary(axisY)}
            </span>
            {isThreeDimensional && (
              <span className="rounded-full bg-slate-50 px-3 py-1 ring-1 ring-slate-200">
                {axisSummary(axisZ)}
              </span>
            )}
            {queryPoint && (
              <span className="rounded-full bg-slate-900 px-3 py-1 font-semibold text-white">
                Query 已标注
              </span>
            )}
            {retrievedHits.length > 0 && (
              <span className="rounded-full bg-emerald-50 px-3 py-1 font-semibold text-emerald-800 ring-1 ring-emerald-200">
                TopK {retrievedHits.length}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className={`grid gap-4 p-4 ${compact ? 'xl:grid-cols-[260px_minmax(0,1fr)]' : 'xl:grid-cols-[300px_minmax(0,1fr)]'}`}>
        <div className="space-y-4">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <div className="space-y-3">
              {!isCollectionMode && (
                <div>
                  <label htmlFor="vector-projection-doc" className="block text-sm font-medium text-slate-700">向量文件</label>
                  <select
                    id="vector-projection-doc"
                    name="vector_projection_doc"
                    value={selectedDoc}
                    onChange={(event) => setSelectedDoc(event.target.value)}
                    className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm"
                  >
                    <option value="">请选择向量文件...</option>
                    {embeddedDocs.map((doc) => (
                      <option key={doc.name} value={doc.name}>{doc.name}</option>
                    ))}
                  </select>
                </div>
              )}

              <div>
                <label htmlFor="vector-projection-method" className="block text-sm font-medium text-slate-700">投影方法</label>
                <select
                  id="vector-projection-method"
                  name="vector_projection_method"
                  value={projectionMethod}
                  onChange={(event) => setProjectionMethod(event.target.value)}
                  className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm"
                >
                  {methodOptions.map((option) => (
                    <option
                      key={option.id}
                      value={option.id}
                      disabled={Boolean(option.requires_query || option.requiresQuery) && !hasQueryOverlay}
                    >
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700">视图维度</label>
                <div className="mt-1 grid grid-cols-2 gap-2">
                  {availableProjectionDimensions.map((dimension) => (
                    <button
                      key={dimension}
                      type="button"
                      onClick={() => setProjectionDimensions(dimension)}
                      className={`rounded border px-3 py-2 text-sm font-semibold ${
                        projectionDimensions === dimension
                          ? 'border-slate-900 bg-slate-900 text-white'
                          : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      {dimension}D
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label htmlFor="vector-projection-filter" className="block text-sm font-medium text-slate-700">过滤分块</label>
                <input
                  id="vector-projection-filter"
                  name="vector_projection_filter"
                  value={filterText}
                  onChange={(event) => setFilterText(event.target.value)}
                  placeholder="页码、分块号或关键词"
                  className="mt-1 block w-full rounded border border-slate-300 bg-white p-2 text-sm"
                />
              </div>

              <div>
                <label htmlFor="vector-projection-zoom" className="block text-sm font-medium text-slate-700">
                  缩放：{zoom.toFixed(1)}x
                </label>
                <input
                  id="vector-projection-zoom"
                  name="vector_projection_zoom"
                  type="range"
                  min="0.7"
                  max="2.4"
                  step="0.1"
                  value={zoom}
                  onChange={(event) => setZoom(Number(event.target.value))}
                  className="mt-2 block w-full"
                />
              </div>

              <button
                type="button"
                onClick={() => {
                  setFilterText('');
                  setZoom(1);
                  setSelectedIndex(null);
                  setHoveredIndex(null);
                }}
                className="w-full rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
              >
                重置视图
              </button>
            </div>
          </div>

          {status && (
            <div className={`rounded-md px-3 py-2 text-sm ${
              status.includes('失败') ? 'bg-red-100 text-red-700' : 'bg-blue-50 text-blue-800'
            }`}>
              {status}
            </div>
          )}

          <SelectedPointPanel point={selectedPoint} hitInfo={selectedHitInfo} />
        </div>

        <div className="min-w-0">
          {projection.points.length > 0 ? (
            <div className={`overflow-hidden rounded-md border border-slate-200 bg-slate-50 ${chartHeightClass}`}>
              {isThreeDimensional ? (
                <Suspense
                  fallback={(
                    <div className="flex h-full items-center justify-center text-sm text-slate-500">
                      正在加载三维视图...
                    </div>
                  )}
                >
                  <VectorProjectionScene3D
                    points={visiblePoints}
                    queryPoint={queryPoint}
                    hitMap={hitMap}
                    zoom={zoom}
                    selectedIndex={selectedIndex}
                    hoveredIndex={hoveredIndex}
                    onSelectIndex={setSelectedIndex}
                    onHoverIndex={setHoveredIndex}
                    getPointChunkId={getPointChunkId}
                    getPointLabel={getPointLabel}
                  />
                </Suspense>
              ) : (
                <svg
                  viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
                  role="img"
                  aria-label="嵌入向量二维投影散点图"
                  className="block h-full w-full"
                >
                  <rect x="0" y="0" width={VIEWBOX_WIDTH} height={VIEWBOX_HEIGHT} fill="transparent" />
                  <line x1={VIEWBOX_PADDING} y1={centerY} x2={VIEWBOX_WIDTH - VIEWBOX_PADDING} y2={centerY} stroke="oklch(0.84 0.014 208)" strokeWidth="1" />
                  <line x1={centerX} y1={VIEWBOX_PADDING} x2={centerX} y2={VIEWBOX_HEIGHT - VIEWBOX_PADDING} stroke="oklch(0.84 0.014 208)" strokeWidth="1" />
                  <text x={VIEWBOX_WIDTH - VIEWBOX_PADDING} y={centerY - 10} textAnchor="end" className="fill-slate-500 text-[20px]">{axisX.label || 'x'}</text>
                  <text x={centerX + 12} y={VIEWBOX_PADDING + 20} className="fill-slate-500 text-[20px]">{axisY.label || 'y'}</text>
                  <g transform={`translate(${centerX} ${centerY}) scale(${zoom}) translate(${-centerX} ${-centerY})`}>
                    {visiblePoints.map((point) => {
                      const metadata = point.embedding?.metadata || {};
                      const hitInfo = hitMap.get(getPointChunkId(point));
                      const isSelected = selectedIndex === point.index;
                      const isHovered = hoveredIndex === point.index;
                      const isHit = Boolean(hitInfo);
                      const muted = hitMap.size > 0 && !isHit;
                      return (
                        <g key={`${getPointChunkId(point)}-${point.index}`}>
                          <circle
                            cx={point.x}
                            cy={point.y}
                            r={isSelected ? 8 : isHovered ? 7 : isHit ? 6.5 : 5}
                            fill={colorForPage(metadata.page_number || metadata.page)}
                            fillOpacity={muted ? 0.28 : isSelected || isHovered || isHit ? 0.95 : 0.72}
                            stroke={isHit ? 'oklch(0.49 0.105 178)' : isSelected ? 'oklch(0.22 0.026 218)' : 'white'}
                            strokeWidth={isHit || isSelected ? 2.5 : 1.5}
                            className="cursor-pointer transition"
                            tabIndex="0"
                            role="button"
                            aria-label={getPointLabel(point)}
                            onMouseEnter={() => setHoveredIndex(point.index)}
                            onMouseLeave={() => setHoveredIndex(null)}
                            onFocus={() => setHoveredIndex(point.index)}
                            onBlur={() => setHoveredIndex(null)}
                            onClick={() => setSelectedIndex(point.index)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault();
                                setSelectedIndex(point.index);
                              }
                            }}
                          >
                            <title>
                              {hitInfo
                                ? `${getPointLabel(point)} · Top ${hitInfo.rank} · 相似性 ${formatNumber(hitInfo.score)}`
                                : getPointLabel(point)}
                            </title>
                          </circle>
                          {hitInfo && (
                            <text
                              x={point.x + 9}
                              y={point.y - 9}
                              className="fill-emerald-800 text-[16px] font-bold"
                            >
                              #{hitInfo.rank}
                            </text>
                          )}
                        </g>
                      );
                    })}
                    {queryPoint && <QueryMarker point={queryPoint} />}
                  </g>
                </svg>
              )}
            </div>
          ) : (
            <div className="p-4">
              {isLoading ? (
                <div className="rounded-md border border-slate-200 bg-slate-50 px-4 py-10 text-center text-sm text-slate-500">
                  正在计算向量投影...
                </div>
              ) : (
                <RandomImage message={isCollectionMode ? '运行 RAG 后，这里会标出 Query 和 TopK 命中点。' : '选择向量文件后，这里会显示后端投影。'} />
              )}
            </div>
          )}

          {hoveredPoint && (
            <div className="mt-2 text-xs text-slate-500">{getPointLabel(hoveredPoint)}</div>
          )}
        </div>
      </div>
    </section>
  );
};

export default VectorProjectionView;
