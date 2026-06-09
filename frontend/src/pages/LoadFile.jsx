// 文件路径：src/pages/LoadFile.jsx
/**
 * @file LoadFile.jsx
 * @brief PDF 读入和已读入文档管理页面。
 */
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const IMPORT_MODE_CONFIG = {
  pdf: {
    label: '文档 / PDF',
    inputLabel: '导入 PDF 文件',
    accept: '.pdf',
    emptyMessage: '请选择 PDF 文件',
    loadingMessage: '正在读入文档...',
    successMessage: '文档读入完成',
    buttonLabel: '文档读入',
  },
  course_qa_json: {
    label: '课程 QA JSON',
    inputLabel: '导入课程 QA JSON',
    accept: '.json,application/json',
    emptyMessage: '请选择课程 QA JSON 文件',
    loadingMessage: '正在导入课程 QA JSON...',
    successMessage: '课程 QA JSON 导入完成，请到 02 使用“课程 QA 条目分块”。',
    buttonLabel: '导入课程 QA JSON',
  },
  course_knowledge: {
    label: '课程知识文档',
    inputLabel: '导入课程知识文档',
    accept: '.md,.markdown,.txt,text/markdown,text/plain',
    emptyMessage: '请选择课程知识 Markdown/TXT 文件',
    loadingMessage: '正在导入课程知识文档...',
    successMessage: '课程知识文档导入完成，请到 02 分块，再到 04/05 建立外部知识索引库。',
    buttonLabel: '导入课程知识文档',
  },
};

const MINERU_ACCEPT_TYPES = '.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.png,.jpg,.jpeg,.jp2,.webp,.gif,.bmp';

/**
 * @brief 渲染 PDF 上传、读入和已读入文档管理控件。
 * @returns {JSX.Element} 文档读入工作流页面。
 */
const LoadFile = () => {
  const [importMode, setImportMode] = useState('pdf');
  const [file, setFile] = useState(null);
  const [loadingMethod, setLoadingMethod] = useState('pymupdf');
  const [unstructuredStrategy, setUnstructuredStrategy] = useState('fast');
  const [chunkingStrategy, setChunkingStrategy] = useState('basic');
  const [chunkingOptions, setChunkingOptions] = useState({
    maxCharacters: 4000,
    newAfterNChars: 3000,
    combineTextUnderNChars: 500,
    overlap: 200,
    overlapAll: false,
    multiPageSections: false
  });
  const [loadedContent, setLoadedContent] = useState(null);
  const [status, setStatus] = useState('');
  const [documents, setDocuments] = useState([]);
  const [activeTab, setActiveTab] = useState('preview'); // 'preview' 或 'documents'
  const [selectedDoc, setSelectedDoc] = useState(null);
  const isMinerULoading = importMode === 'pdf' && loadingMethod === 'mineru_vlm';

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents?type=loaded`);
      const data = await response.json();
      setDocuments(data.documents);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleProcess = async () => {
    const modeConfig = IMPORT_MODE_CONFIG[importMode] || IMPORT_MODE_CONFIG.pdf;
    if (!file) {
      setStatus(modeConfig.emptyMessage);
      return;
    }
    if (importMode === 'pdf' && !loadingMethod) {
      setStatus('请选择文件和读入工具');
      return;
    }

    setStatus(isMinerULoading ? '正在调用 MinerU VLM 精准解析并导入文档...' : modeConfig.loadingMessage);
    setLoadedContent(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      let endpoint = `${apiBaseUrl}/load-course-qa-json`;
      if (importMode === 'course_knowledge') {
        endpoint = `${apiBaseUrl}/load-course-knowledge-doc`;
      } else if (importMode === 'pdf') {
        endpoint = `${apiBaseUrl}/load`;
        formData.append('loading_method', loadingMethod);
        if (loadingMethod === 'unstructured') {
          formData.append('strategy', unstructuredStrategy);
          formData.append('chunking_strategy', chunkingStrategy);
          formData.append('chunking_options', JSON.stringify(chunkingOptions));
        }
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setLoadedContent(data.loaded_content);
      setStatus(isMinerULoading ? 'MinerU VLM 精准解析导入完成，请到 02 分块或直接到 04 生成向量。' : modeConfig.successMessage);
      fetchDocuments();
      setActiveTab('preview');

    } catch (error) {
      console.error('Error:', error);
      setStatus(`读入失败: ${error.message}`);
    }
  };

  const handleDeleteDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setStatus('文档已删除');
      fetchDocuments();
      if (selectedDoc?.name === docName) {
        setSelectedDoc(null);
        setLoadedContent(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setStatus(`删除文档失败: ${error.message}`);
    }
  };

  const handleViewDocument = async (doc) => {
    try {
      setStatus('正在加载文档...');
      const response = await fetch(`${apiBaseUrl}/documents/${doc.name}.json`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setSelectedDoc(doc);
      setLoadedContent(data);
      setActiveTab('preview');
      setStatus('');
    } catch (error) {
      console.error('Error loading document:', error);
      setStatus(`加载文档失败: ${error.message}`);
    }
  };

  const renderRightPanel = () => {
    const isCourseQaPreview = loadedContent?.dataset_type === 'course_qa';
    const isCourseKnowledgePreview = loadedContent?.dataset_type === 'course_knowledge';
    return (
      <div className="p-4">
        {/* 标签页切换 */}
        <div className="flex mb-4 border-b">
          <button
            className={`px-4 py-2 ${
              activeTab === 'preview'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('preview')}
          >
            文档预览
          </button>
          <button
            className={`px-4 py-2 ml-4 ${
              activeTab === 'documents'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('documents')}
          >
            文档管理
          </button>
        </div>

        {/* 内容区域 */}
        {activeTab === 'preview' ? (
          loadedContent ? (
            <div>
              <h3 className="text-xl font-semibold mb-4">文档内容</h3>
              <div className="mb-4 p-3 border rounded bg-gray-100">
                <h4 className="font-medium mb-2">文档信息</h4>
                <div className="text-sm text-gray-600">
                  <p>页数: {loadedContent.total_pages || 'N/A'}</p>
                  {isCourseQaPreview && (
                    <p>数据类型: 课程 QA JSON</p>
                  )}
                  {isCourseKnowledgePreview && (
                    <p>数据类型: 课程知识文档</p>
                  )}
                  <p>分块数: {loadedContent.total_chunks || 'N/A'}</p>
                  <p>读入方法: {loadedContent.loading_method || 'N/A'}</p>
                  {loadedContent.source_format && (
                    <p>来源格式: {loadedContent.source_format}</p>
                  )}
                  {loadedContent.mineru_batch_id && (
                    <p>MinerU 批次: {loadedContent.mineru_batch_id}</p>
                  )}
                  {loadedContent.mineru_model_version && (
                    <p>MinerU 模型: {loadedContent.mineru_model_version}</p>
                  )}
                  {loadedContent.mineru_full_zip_url && (
                    <p>
                      MinerU 结果:
                      {' '}
                      <a
                        href={loadedContent.mineru_full_zip_url}
                        className="text-blue-600 underline"
                        target="_blank"
                        rel="noreferrer"
                      >
                        查看 zip
                      </a>
                    </p>
                  )}
                  <p>分块方法: {loadedContent.chunking_method || 'N/A'}</p>
                  <p>处理时间: {loadedContent.timestamp ?
                    new Date(loadedContent.timestamp).toLocaleString() : 'N/A'}</p>
                </div>
              </div>
              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {loadedContent.chunks.map((chunk) => (
                  <div key={chunk.metadata.chunk_id} className="p-3 border rounded bg-gray-50">
                    <div className="font-medium text-sm text-gray-500 mb-1">
                      分块 {chunk.metadata.chunk_id}
                      {isCourseQaPreview
                        ? `（${chunk.metadata.topic || chunk.metadata.page_range || '课程 QA'}）`
                        : isCourseKnowledgePreview
                          ? `（${chunk.metadata.section_title || chunk.metadata.page_range || '课程知识'}）`
                        : `（第 ${chunk.metadata.page_number} 页）`}
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                      词数: {chunk.metadata.word_count} | 范围: {chunk.metadata.page_range}
                    </div>
                    <div className="text-sm mt-2">
                      <div className="text-gray-600">{chunk.content}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="上传并读入文件，或从文档管理中选择已有文档后，这里会显示预览结果。" />
          )
        ) : (
          // 文档管理页面
          <div>
            <h3 className="text-xl font-semibold mb-4">导入文档集：</h3>
            <div className="space-y-4">
              {documents.map((doc) => (
                <div key={doc.name} className="p-4 border rounded-lg bg-gray-50">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium text-lg">{doc.name}</h4>
                      <div className="text-sm text-gray-600 mt-1">
                        <p>页数: {doc.metadata?.total_pages || 'N/A'}</p>
                        {doc.metadata?.dataset_type === 'course_qa' && (
                          <p>数据类型: 课程 QA JSON</p>
                        )}
                        {doc.metadata?.dataset_type === 'course_knowledge' && (
                          <p>数据类型: 课程知识文档</p>
                        )}
	                        <p>分块数: {doc.metadata?.total_chunks || 'N/A'}</p>
	                        <p>读入方法: {doc.metadata?.loading_method || 'N/A'}</p>
	                        {doc.metadata?.source_format && (
	                          <p>来源格式: {doc.metadata.source_format}</p>
	                        )}
	                        {doc.metadata?.mineru_model_version && (
	                          <p>MinerU 模型: {doc.metadata.mineru_model_version}</p>
	                        )}
	                        <p>分块方法: {doc.metadata?.chunking_method || 'N/A'}</p>
                        <p>创建时间: {doc.metadata?.timestamp ?
                          new Date(doc.metadata.timestamp).toLocaleString() : 'N/A'}</p>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => handleViewDocument(doc)}
                        className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                      >
                        浏览
                      </button>
                      <button
                        onClick={() => handleDeleteDocument(doc.name)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                </div>
              ))}
              {documents.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  暂无已导入文档
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-6">
	<h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
	<hr />
      <h2 className="text-2xl font-bold mb-6">文档导入</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板 */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div>
              <label className="block text-sm font-medium mb-2">导入类型</label>
              <div className="grid grid-cols-1 gap-2">
                {Object.entries(IMPORT_MODE_CONFIG).map(([mode, config]) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => {
                      setImportMode(mode);
                      setFile(null);
                      setStatus('');
                    }}
                    className={`rounded border px-3 py-2 text-sm font-semibold ${
                      importMode === mode
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {config.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
	              <label htmlFor="load-file-input" className="mt-4 block text-sm font-medium mb-1">
	                {isMinerULoading
	                  ? '导入 MinerU 支持的文档'
	                  : (IMPORT_MODE_CONFIG[importMode] || IMPORT_MODE_CONFIG.pdf).inputLabel}
	              </label>
	              <input
	                id="load-file-input"
	                name="load-file"
	                aria-label={isMinerULoading ? '导入 MinerU 支持的文档' : (IMPORT_MODE_CONFIG[importMode] || IMPORT_MODE_CONFIG.pdf).inputLabel}
	                key={`${importMode}-${loadingMethod}`}
	                type="file"
	                accept={isMinerULoading
	                  ? MINERU_ACCEPT_TYPES
	                  : (IMPORT_MODE_CONFIG[importMode] || IMPORT_MODE_CONFIG.pdf).accept}
	                onChange={(e) => setFile(e.target.files[0])}
                className="block w-full border rounded px-3 py-2"
              />
            </div>

            {importMode === 'pdf' && (
              <div className="mt-4">
	                <label htmlFor="load-method-select" className="block text-sm font-medium mb-1">读入工具选择</label>
	                <select
	                  id="load-method-select"
	                  name="loading_method"
	                  aria-label="读入工具选择"
	                  value={loadingMethod}
	                  onChange={(e) => {
	                    setLoadingMethod(e.target.value);
	                    setFile(null);
	                    setStatus('');
	                  }}
	                  className="block w-full p-2 border rounded"
	                >
	                  <option value="pymupdf">PyMuPDF</option>
	                  <option value="pypdf">PyPDF</option>
	                  <option value="unstructured">Unstructured</option>
	                  <option value="pdfplumber">PDF Plumber</option>
	                  <option value="mineru_vlm">MinerU VLM 精准解析</option>
	                </select>
	              </div>
            )}

            {importMode === 'pdf' && loadingMethod === 'unstructured' && (
              <>
                <div className="mt-4">
	                  <label htmlFor="unstructured-strategy-select" className="block text-sm font-medium mb-1">Unstructured Strategy</label>
	                  <select
	                    id="unstructured-strategy-select"
	                    name="unstructured_strategy"
                    value={unstructuredStrategy}
                    onChange={(e) => setUnstructuredStrategy(e.target.value)}
                    className="block w-full p-2 border rounded"
                  >
                    <option value="fast">Fast</option>
                    <option value="hi_res">High Resolution</option>
                    <option value="ocr_only">OCR Only</option>
                  </select>
                </div>

                <div className="mt-4">
	                  <label htmlFor="unstructured-chunking-select" className="block text-sm font-medium mb-1">Chunking Strategy</label>
	                  <select
	                    id="unstructured-chunking-select"
	                    name="chunking_strategy"
                    value={chunkingStrategy}
                    onChange={(e) => setChunkingStrategy(e.target.value)}
                    className="block w-full p-2 border rounded"
                  >
                    <option value="basic">Basic</option>
                    <option value="by_title">By Title</option>
                  </select>
                </div>

                {chunkingStrategy === 'basic' && (
                  <div className="mt-4 space-y-3">
                    <div>
	                      <label htmlFor="max-characters-input" className="block text-sm font-medium mb-1">Max Characters</label>
	                      <input
	                        id="max-characters-input"
	                        name="maxCharacters"
                        type="number"
                        value={chunkingOptions.maxCharacters}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          maxCharacters: parseInt(e.target.value)
                        }))}
                        className="block w-full p-2 border rounded"
                      />
                    </div>
                    <div>
	                      <label htmlFor="new-after-n-chars-input" className="block text-sm font-medium mb-1">New After N Chars</label>
	                      <input
	                        id="new-after-n-chars-input"
	                        name="newAfterNChars"
                        type="number"
                        value={chunkingOptions.newAfterNChars}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          newAfterNChars: parseInt(e.target.value)
                        }))}
                        className="block w-full p-2 border rounded"
                      />
                    </div>
                    <div>
	                      <label htmlFor="combine-text-under-input" className="block text-sm font-medium mb-1">Combine Text Under N Chars</label>
	                      <input
	                        id="combine-text-under-input"
	                        name="combineTextUnderNChars"
                        type="number"
                        value={chunkingOptions.combineTextUnderNChars}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          combineTextUnderNChars: parseInt(e.target.value)
                        }))}
                        className="block w-full p-2 border rounded"
                      />
                    </div>
                    <div>
	                      <label htmlFor="overlap-input" className="block text-sm font-medium mb-1">Overlap</label>
	                      <input
	                        id="overlap-input"
	                        name="overlap"
                        type="number"
                        value={chunkingOptions.overlap}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          overlap: parseInt(e.target.value)
                        }))}
                        className="block w-full p-2 border rounded"
                      />
                    </div>
                    <div className="flex items-center">
	                      <input
	                        id="overlap-all-input"
	                        name="overlapAll"
                        type="checkbox"
                        checked={chunkingOptions.overlapAll}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          overlapAll: e.target.checked
                        }))}
                        className="mr-2"
                      />
	                      <label htmlFor="overlap-all-input" className="text-sm font-medium">Overlap All</label>
                    </div>
                  </div>
                )}

                {chunkingStrategy === 'by_title' && (
                  <div className="mt-4 space-y-3">
                    <div>
	                      <label htmlFor="by-title-combine-text-under-input" className="block text-sm font-medium mb-1">Combine Text Under N Chars</label>
	                      <input
	                        id="by-title-combine-text-under-input"
	                        name="byTitleCombineTextUnderNChars"
                        type="number"
                        value={chunkingOptions.combineTextUnderNChars}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          combineTextUnderNChars: parseInt(e.target.value)
                        }))}
                        className="block w-full p-2 border rounded"
                      />
                    </div>
                    <div className="flex items-center">
	                      <input
	                        id="multi-page-sections-input"
	                        name="multiPageSections"
                        type="checkbox"
                        checked={chunkingOptions.multiPageSections}
                        onChange={(e) => setChunkingOptions(prev => ({
                          ...prev,
                          multiPageSections: e.target.checked
                        }))}
                        className="mr-2"
                      />
	                      <label htmlFor="multi-page-sections-input" className="text-sm font-medium">Multi-page Sections</label>
                    </div>
                  </div>
                )}
              </>
            )}

            <button 
              onClick={handleProcess}
              className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              disabled={!file}
            >
              {(IMPORT_MODE_CONFIG[importMode] || IMPORT_MODE_CONFIG.pdf).buttonLabel}
            </button>
          </div>

          {status && (
            <div className={`p-4 rounded-lg ${
              status.includes('失败') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
            }`}>
              {status}
            </div>
          )}
        </div>

        {/* 右侧面板 */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {renderRightPanel()}
        </div>
      </div>
    </div>
  );
};

export default LoadFile;
