/**
 * @file ChunkFile.jsx
 * @brief 已读入文档分块工作流页面。
 */
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

/**
 * @brief 渲染已读入文档分块和分块集合查看控件。
 * @returns {JSX.Element} 分块工作流页面。
 */
const ChunkFile = () => {
  const [loadedDocuments, setLoadedDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('');
  const [chunkingOption, setChunkingOption] = useState('by_pages');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [chunks, setChunks] = useState(null);
  const [status, setStatus] = useState('');
  const [activeTab, setActiveTab] = useState('chunks');
  const [processingStatus, setProcessingStatus] = useState('');
  const [chunkedDocuments, setChunkedDocuments] = useState([]);

  useEffect(() => {
    fetchLoadedDocuments();
  }, []);

  const fetchLoadedDocuments = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents?type=loaded`);
      const data = await response.json();
      setLoadedDocuments(data.documents);

      const chunkedResponse = await fetch(`${apiBaseUrl}/documents?type=chunked`);
      if (!chunkedResponse.ok) {
        throw new Error(`HTTP error! status: ${chunkedResponse.status}`);
      }
      const chunkedData = await chunkedResponse.json();
      console.log('Chunked documents response:', chunkedData);
      
      if (!chunkedData.documents || !Array.isArray(chunkedData.documents)) {
        console.error('Invalid chunked documents data:', chunkedData);
        return;
      }

      const chunkedDocsWithDetails = await Promise.all(
        chunkedData.documents.map(async (doc) => {
          try {
            const detailResponse = await fetch(`${apiBaseUrl}/documents/${doc.name}?type=chunked`);
            if (!detailResponse.ok) {
              console.error(`Error fetching details for ${doc.name}:`, detailResponse.status);
              return doc;
            }
            const detailData = await detailResponse.json();
            console.log(`Details for ${doc.name}:`, detailData);
            
            return {
              ...doc,
              total_pages: detailData.total_pages,
              total_chunks: detailData.total_chunks,
              chunking_method: detailData.chunking_method,
              chunk_size: detailData.chunk_size,
              chunk_overlap: detailData.chunk_overlap,
              timestamp: detailData.timestamp
            };
          } catch (error) {
            console.error(`Error processing document ${doc.name}:`, error);
            return doc;
          }
        })
      );
      
      console.log('Final chunked documents:', chunkedDocsWithDetails);
      setChunkedDocuments(chunkedDocsWithDetails);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setProcessingStatus(`获取文档失败: ${error.message}`);
    }
  };

  const handleChunk = async () => {
    if (!selectedDoc || !chunkingOption) {
      setStatus('请选择文档和分块方法');
      return;
    }

    setStatus('分块处理中...');
    setChunks(null);

    try {
      const docId = selectedDoc.endsWith('.json') ? selectedDoc : `${selectedDoc}.json`;
      
      const response = await fetch(`${apiBaseUrl}/chunk`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doc_id: docId,
          chunking_option: chunkingOption,
          chunk_size: chunkSize,
          chunk_overlap: chunkingOption === 'fixed_size' ? chunkOverlap : 0,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Chunk response:', data);

      setChunks({
        filename: data.filename,
        total_pages: data.total_pages,
        total_chunks: data.total_chunks,
        loading_method: data.loading_method,
        chunking_method: data.chunking_method,
        chunk_size: data.chunk_size,
        chunk_overlap: data.chunk_overlap,
        timestamp: data.timestamp,
        chunks: data.chunks
      });

      setStatus('分块完成');
      fetchLoadedDocuments();

    } catch (error) {
      console.error('Error:', error);
      setStatus(`分块失败: ${error.message}`);
    }
  };

  const handleDeleteDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}?type=chunked`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setProcessingStatus('分块文档已删除');
      fetchLoadedDocuments();
      if (selectedDoc === docName) {
        setSelectedDoc('');
        setChunks(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setProcessingStatus(`删除文档失败: ${error.message}`);
    }
  };

  const handleViewDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}?type=chunked`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setChunks(data);
      setActiveTab('chunks');
    } catch (error) {
      console.error('Error viewing document:', error);
      setProcessingStatus(`查看文档失败: ${error.message}`);
    }
  };

  const renderRightPanel = () => {
    return (
      <div className="p-4 w-full h-full flex flex-col">
        <div className="flex mb-4 border-b">
          <button
            className={`px-4 py-2 ${
              activeTab === 'chunks'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('chunks')}
          >
            分块预览
          </button>
          <button
            className={`px-4 py-2 ml-4 ${
              activeTab === 'documents'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('documents')}
          >
            分块管理
          </button>
        </div>

        {activeTab === 'chunks' ? (
          chunks ? (
            <div className="w-full">
              <div className="mb-4 p-3 border rounded bg-gray-100">
                <h4 className="font-medium mb-2">文档信息</h4>
                <div className="text-sm text-gray-600">
                  <p>文件名: {chunks.filename}</p>
                  <p>总页数: {chunks.total_pages}</p>
                  <p>分块数: {chunks.total_chunks}</p>
                  <p>读入方法: {chunks.loading_method}</p>
                  <p>分块方法: {chunks.chunking_method}</p>
                  {chunks.chunking_method === 'fixed_size' && (
                    <p>块大小/重叠: {chunks.chunk_size} / {chunks.chunk_overlap}</p>
                  )}
                  <p>时间: {chunks.timestamp ? new Date(chunks.timestamp).toLocaleString() : 'N/A'}</p>
                </div>
              </div>
              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {Array.isArray(chunks.chunks) && chunks.chunks.map((chunk) => (
                  <div key={chunk.metadata.chunk_id} className="p-3 border rounded bg-gray-50">
                    <div className="font-medium text-sm text-gray-500 mb-1">
                      Chunk {chunk.metadata.chunk_id}
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                      页码: {chunk.metadata.page_range} |
                      词数: {chunk.metadata.word_count}
                    </div>
                    <div className="text-sm mt-2">
                      <div className="text-gray-600">{chunk.content}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="选择文档并产生分块后在这里查看结果" />
          )
        ) : (
          <div className="flex flex-col w-full h-full">
            <h3 className="text-xl font-semibold mb-4">分块文档管理</h3>
            <div className="space-y-4 w-full">
              {chunkedDocuments.length > 0 ? (
                chunkedDocuments.map((doc) => (
                  <div key={doc.name} className="p-4 border rounded-lg bg-gray-50 w-full">
                    <div className="flex justify-between items-start w-full">
                      <div className="flex-grow">
                        <h4 className="font-medium text-lg">{doc.name}</h4>
                        <div className="text-sm text-gray-600 mt-1">
                          <p>页数: {doc.total_pages || 'N/A'}</p>
                          <p>分块数: {doc.total_chunks || 'N/A'}</p>
                          <p>分块方法: {doc.chunking_method || 'N/A'}</p>
                          {doc.chunking_method === 'fixed_size' && (
                            <p>块大小/重叠: {doc.chunk_size || 'N/A'} / {doc.chunk_overlap ?? 'N/A'}</p>
                          )}
                          <p>处理时间: {doc.timestamp ? new Date(doc.timestamp).toLocaleString() : 'N/A'}</p>
                        </div>
                      </div>
                      <div className="flex space-x-2 ml-4">
                        <button
                          onClick={() => handleViewDocument(doc.name)}
                          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                          查看
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
                ))
              ) : (
                <div className="text-center text-gray-500 py-8 w-full">
                  暂无分块文档
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
      <h2 className="text-2xl font-bold mb-6">知识分块</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板 */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">选择文档</label>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="">请选择文档...</option>
                {loadedDocuments.map((doc) => (
                  <option key={doc.name} value={doc.name}>
                    {doc.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">分块方法</label>
              <select
                value={chunkingOption}
                onChange={(e) => setChunkingOption(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="by_pages">按页分块</option>
                <option value="fixed_size">固定大小</option>
                <option value="by_paragraphs">按段落</option>
                <option value="by_sentences">按句子</option>
              </select>
            </div>

            {chunkingOption === 'fixed_size' && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">块大小（字符）</label>
                <input
                  type="number"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="block w-full p-2 border rounded"
                  min="100"
                  max="5000"
                />
                <label className="block text-sm font-medium mb-1 mt-3">重叠字符数</label>
                <input
                  type="number"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(Number(e.target.value))}
                  className="block w-full p-2 border rounded"
                  min="0"
                  max={Math.max(0, chunkSize - 1)}
                />
              </div>
            )}

            <button 
              onClick={handleChunk}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              disabled={!selectedDoc}
            >
              产生分块
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

export default ChunkFile;
