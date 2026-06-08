// 文件路径：src/pages/EmbeddingFile.jsx
/**
 * @file EmbeddingFile.jsx
 * @brief 嵌入生成和已嵌入文档管理页面。
 */
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import VectorProjectionView from '../components/rag/VectorProjectionView';
import { apiBaseUrl } from '../config/config';

const VECTOR_PREVIEW_SIZE = 16;
const EMBEDDING_MODEL_OPTIONS = {
  qwen_api: [
    { value: 'text-embedding-v4', label: 'text-embedding-v4（百炼）' },
  ],
  huggingface: [
    { value: 'BAAI/bge-small-zh-v1.5', label: 'bge-small-zh-v1.5（中文）' },
    { value: 'intfloat/multilingual-e5-small', label: 'multilingual-e5-small（多语言轻量）' },
  ],
};

/**
 * @brief 将向量数值格式化为便于投屏查看的短数组。
 * @param {number[]} vector 嵌入向量。
 * @param {number} limit 展示的最大维度数量。
 * @returns {string} 格式化后的向量片段。
 */
const formatVectorValues = (vector = [], limit = vector.length) => {
  const safeVector = Array.isArray(vector) ? vector : [];
  const values = safeVector.slice(0, limit).map((value) => {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue.toFixed(6) : String(value);
  });
  const suffix = safeVector.length > limit ? ', ...' : '';
  return `[${values.join(', ')}${suffix}]`;
};

/**
 * @brief 渲染从已读入或已分块文档创建嵌入的控件。
 * @returns {JSX.Element} 嵌入工作流页面。
 */
const EmbeddingFile = () => {
  const [selectedDoc, setSelectedDoc] = useState('');
  const [embeddingProvider, setEmbeddingProvider] = useState('qwen_api');
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-v4');
  const [status, setStatus] = useState('');
  const [availableDocs, setAvailableDocs] = useState([]);
  const [embeddedDocs, setEmbeddedDocs] = useState([]);
  const [embeddings, setEmbeddings] = useState(null);
  const [activeTab, setActiveTab] = useState('preview'); // 'preview' 或 'documents'
  const [expandedVectorKey, setExpandedVectorKey] = useState(null);

  useEffect(() => {
    fetchAvailableDocs();
    fetchEmbeddedDocs();
  }, []);

  useEffect(() => {
    const providerOptions = EMBEDDING_MODEL_OPTIONS[embeddingProvider] || EMBEDDING_MODEL_OPTIONS.qwen_api;
    setEmbeddingModel(providerOptions[0].value);
  }, [embeddingProvider]);

  const fetchAvailableDocs = async () => {
    try {
      console.log('开始获取文档列表...');
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5秒超时
      
      const response = await fetch(`${apiBaseUrl}/documents?type=all`, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      console.log('API响应状态:', response.status);
      const data = await response.json();
      console.log('API响应数据:', data);
      console.log('文档列表:', data.documents);
      if (!Array.isArray(data.documents)) {
        console.error('文档数据不是数组格式:', data.documents);
        return;
      }
      setAvailableDocs(data.documents);
    } catch (error) {
      console.error('获取文档列表出错:', error);
      if (error.name === 'AbortError') {
        setStatus('获取文档列表超时，请检查后端服务是否正常运行');
      } else {
        setStatus('获取文档列表失败: ' + error.message);
      }
    }
  };

  const fetchEmbeddedDocs = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/list-embedded`);
      const data = await response.json();
      setEmbeddedDocs(data.documents);
    } catch (error) {
      console.error('Error fetching embedded documents:', error);
    }
  };

  const handleEmbed = async () => {
    if (!selectedDoc) {
      setStatus('请选择文档');
      return;
    }
    
    setStatus('正在生成向量...');
    try {
      const response = await fetch(`${apiBaseUrl}/embed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          documentId: selectedDoc,  // 使用完整的文件名
          provider: embeddingProvider,
          model: embeddingModel
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '生成向量失败');
      }
      
      const data = await response.json();
      setEmbeddings(data.embeddings);
      setExpandedVectorKey(null);
      setStatus(`向量生成完成，已保存至: ${data.filepath}`);
      fetchEmbeddedDocs(); // 刷新嵌入文档列表
    } catch (error) {
      console.error('Error:', error);
      setStatus('生成向量失败: ' + error.message);
    }
  };

  const handleDeleteEmbedding = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/embedded-docs/${docName}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setStatus('向量文件已删除');
      fetchEmbeddedDocs();
      if (embeddings && selectedDoc === docName) {
        setEmbeddings(null);
      }
    } catch (error) {
      console.error('Error deleting embedding:', error);
      setStatus(`删除向量文件失败: ${error.message}`);
    }
  };

  const handleViewEmbedding = async (docName) => {
    try {
      setStatus('正在加载向量文件...');
      const response = await fetch(`${apiBaseUrl}/embedded-docs/${docName}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setEmbeddings(data.embeddings);
      setExpandedVectorKey(null);
      setActiveTab('preview');
      setStatus('');
    } catch (error) {
      console.error('Error loading embedding:', error);
      setStatus(`加载向量文件失败: ${error.message}`);
    }
  };

  const renderRightPanel = () => {
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
            嵌入文档预览
          </button>
          <button
            className={`px-4 py-2 ml-4 ${
              activeTab === 'documents'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('documents')}
          >
            嵌入文档管理
          </button>
          <button
            className={`px-4 py-2 ml-4 ${
              activeTab === 'projection'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('projection')}
          >
            向量投影视图
          </button>
        </div>

        {activeTab === 'preview' ? (
          embeddings ? (
            <div>
              <h3 className="text-xl font-semibold mb-4">向量结果</h3>
              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {embeddings.map((embedding, idx) => (
                  <div key={idx} className="p-3 border rounded bg-gray-50">
                    <div className="font-medium text-sm text-gray-500 mb-1">
                      分块 {embedding.metadata.chunk_id} / {embedding.metadata.total_chunks}
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                      文档: {embedding.metadata.filename || embedding.metadata.document_name || 'N/A'} |
                      页码: {embedding.metadata.page_number || 'N/A'} |
                      页码范围: {embedding.metadata.page_range || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                      模型: {embedding.metadata.embedding_model || 'N/A'} |
                      提供方: {embedding.metadata.embedding_provider || 'N/A'} |
                      维度: {embedding.metadata.vector_dimension || 'N/A'} |
                      时间: {new Date(embedding.metadata.embedding_timestamp).toLocaleString()}
                    </div>
                    <div className="text-sm mt-2">
                      <div className="font-medium text-gray-600">内容:</div>
                      <div className="text-gray-600">{embedding.metadata.content || 'N/A'}</div>
                    </div>
                    <div className="mt-3 rounded border border-gray-200 bg-white p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <div className="text-sm font-medium text-gray-700">嵌入向量数值</div>
                          <div className="text-xs text-gray-500">
                            维度 {Array.isArray(embedding.embedding) ? embedding.embedding.length : 'N/A'}，默认展示前 {VECTOR_PREVIEW_SIZE} 维
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            const vectorKey = `${embedding.metadata.chunk_id}-${idx}`;
                            setExpandedVectorKey(expandedVectorKey === vectorKey ? null : vectorKey);
                          }}
                          className="rounded border border-gray-300 px-3 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50"
                        >
                          {expandedVectorKey === `${embedding.metadata.chunk_id}-${idx}` ? '收起完整向量' : '展开完整向量'}
                        </button>
                      </div>
                      <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap break-words rounded border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">
                        {formatVectorValues(embedding.embedding, VECTOR_PREVIEW_SIZE)}
                      </pre>
                      {expandedVectorKey === `${embedding.metadata.chunk_id}-${idx}` && (
                        <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded border border-slate-200 bg-white p-3 text-xs leading-5 text-slate-700">
                          {formatVectorValues(embedding.embedding)}
                        </pre>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="选择文档并生成向量，或查看已有向量文件后，这里会显示每个分块的嵌入元信息。" />
          )
        ) : activeTab === 'documents' ? (
          // 嵌入文档管理页面
          <div>
            <h3 className="text-xl font-semibold mb-4">向量文件管理</h3>
            <div className="space-y-4">
              {embeddedDocs.map((doc) => (
                <div key={doc.name} className="p-4 border rounded-lg bg-gray-50">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium text-lg">{doc.name}</h4>
                      <div className="text-sm text-gray-600 mt-1">
                        <p>模型: {doc.metadata?.embedding_model}</p>
                        <p>提供方: {doc.metadata?.embedding_provider}</p>
                        <p>创建时间: {new Date(doc.metadata?.embedding_timestamp).toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => handleViewEmbedding(doc.name)}
                        className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                      >
                        查看
                      </button>
                      <button
                        onClick={() => handleDeleteEmbedding(doc.name)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                </div>
              ))}
              {embeddedDocs.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  暂无向量文件
                </div>
              )}
            </div>
          </div>
        ) : (
          <VectorProjectionView title="04 向量投影视图" />
        )}
      </div>
    );
  };

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="mb-6 text-2xl font-bold">向量存储</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板 */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div>
              <label className="block text-sm font-medium mb-1">选择文档</label>
              <div className="text-sm text-gray-500 mb-2">
                可用文档数量: {availableDocs.length}
              </div>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="">请选择文档...</option>
                {availableDocs.map(doc => (
                  <option key={doc.id} value={doc.name}>
                    {doc.name} ({doc.type})
                  </option>
                ))}
              </select>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">嵌入模型提供方</label>
              <select
                value={embeddingProvider}
                onChange={(e) => setEmbeddingProvider(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="qwen_api">Qwen（百炼）</option>
                <option value="huggingface">HuggingFace</option>
              </select>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">嵌入模型</label>
              <select
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                {(EMBEDDING_MODEL_OPTIONS[embeddingProvider] || EMBEDDING_MODEL_OPTIONS.qwen_api).map(model => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </div>

            <button 
              onClick={handleEmbed}
              className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              disabled={!selectedDoc}
            >
              产生向量
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

export default EmbeddingFile;
