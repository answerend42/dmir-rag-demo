/**
 * @file ParseFile.jsx
 * @brief 单次 PDF 解析工作流页面。
 */
import React, { useState } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

/**
 * @brief 渲染上传 PDF 的临时解析控件，不持久化解析结果。
 * @returns {JSX.Element} 解析工作流页面。
 */
const ParseFile = () => {
  const [file, setFile] = useState(null);
  const [loadingMethod, setLoadingMethod] = useState('pymupdf');
  const [parsingOption, setParsingOption] = useState('all_text');
  const [parsedContent, setParsedContent] = useState(null);
  const [status, setStatus] = useState('');
  const [docName, setDocName] = useState('');
  const [isProcessed, setIsProcessed] = useState(false);
  const isMinerU = loadingMethod === 'mineru_vlm' || loadingMethod === 'mineru_agent';

  const handleProcess = async () => {
    if (!file || !loadingMethod || !parsingOption) {
      setStatus('请选择文件、装载工具和解析选项');
      return;
    }

    setStatus(isMinerU ? '正在调用 MinerU VLM 精准解析...' : '正在解析文件...');
    setParsedContent(null);
    setIsProcessed(false);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('loading_method', loadingMethod);
      formData.append('parsing_option', parsingOption);

      const response = await fetch(`${apiBaseUrl}/parse`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => null);
        throw new Error(errorPayload?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setParsedContent(data.parsed_content);
      setStatus('文件解析完成');
      setIsProcessed(true);
    } catch (error) {
      console.error('Error:', error);
      setStatus(`解析失败: ${error.message}`);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file);
      const baseName = file.name.replace(/\.[^.]+$/, '');
      setDocName(baseName);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">文件解析</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板（3/12） */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div>
              <label htmlFor="parse-file-input" className="block text-sm font-medium mb-1">
                {isMinerU ? '选择文档文件' : '选择PDF文件'}
              </label>
              <input
                id="parse-file-input"
                name="parse-file"
                type="file"
                accept={isMinerU ? '.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.png,.jpg,.jpeg,.jp2,.webp,.gif,.bmp' : '.pdf'}
                onChange={handleFileSelect}
                className="block w-full border rounded px-3 py-2"
                required
              />
            </div>

            <div className="mt-4">
              <label htmlFor="parse-loading-method" className="block text-sm font-medium mb-1">装载工具</label>
              <select
                id="parse-loading-method"
                name="loading_method"
                value={loadingMethod}
                onChange={(e) => setLoadingMethod(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="pymupdf">PyMuPDF</option>
                <option value="pypdf">PyPDF</option>
                <option value="unstructured">Unstructured</option>
                <option value="pdfplumber">PDF Plumber</option>
                <option value="mineru_vlm">MinerU VLM 精准解析</option>
              </select>
            </div>

            <div className="mt-4">
              <label htmlFor="parse-option" className="block text-sm font-medium mb-1">解析选项</label>
              <select
                id="parse-option"
                name="parsing_option"
                value={parsingOption}
                onChange={(e) => setParsingOption(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="all_text">全部文本</option>
                <option value="by_pages">按页解析</option>
                <option value="by_titles">按标题解析</option>
                <option value="text_and_tables">文本和表格</option>
              </select>
            </div>

            <button 
              onClick={handleProcess}
              className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              disabled={!file}
            >
              解析文件
            </button>
          </div>
        </div>

        {/* 右侧面板（9/12） */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {parsedContent ? (
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-4">解析结果</h3>
              <div className="mb-4 p-3 border rounded bg-gray-100">
                <h4 className="font-medium mb-2">文档信息</h4>
                <div className="text-sm text-gray-600">
                  <p>总页数: {parsedContent.metadata?.total_pages ?? '未返回'}</p>
                  <p>解析方法: {parsedContent.metadata?.parsing_method}</p>
                  <p>解析来源: {parsedContent.metadata?.source ?? '本地解析'}</p>
                  {parsedContent.metadata?.mineru_task_id && (
                    <p>MinerU 任务: {parsedContent.metadata.mineru_task_id}</p>
                  )}
                  {parsedContent.metadata?.mineru_batch_id && (
                    <p>MinerU 批次: {parsedContent.metadata.mineru_batch_id}</p>
                  )}
                  {parsedContent.metadata?.mineru_model_version && (
                    <p>MinerU 模型: {parsedContent.metadata.mineru_model_version}</p>
                  )}
                  {parsedContent.metadata?.mineru_markdown_url && (
                    <p>
                      Markdown:
                      {' '}
                      <a
                        href={parsedContent.metadata.mineru_markdown_url}
                        className="text-blue-600 underline"
                        target="_blank"
                        rel="noreferrer"
                      >
                        查看结果
                      </a>
                    </p>
                  )}
                  {parsedContent.metadata?.mineru_full_zip_url && (
                    <p>
                      结果压缩包:
                      {' '}
                      <a
                        href={parsedContent.metadata.mineru_full_zip_url}
                        className="text-blue-600 underline"
                        target="_blank"
                        rel="noreferrer"
                      >
                        查看结果
                      </a>
                    </p>
                  )}
                  <p>时间: {parsedContent.metadata?.timestamp && new Date(parsedContent.metadata.timestamp).toLocaleString()}</p>
                </div>
              </div>
              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {parsedContent.content.map((item, idx) => (
	                  <div key={idx} className="p-3 border rounded bg-gray-50">
	                    <div className="font-medium text-sm text-gray-500 mb-1">
	                      {item.type}{item.page ? ` - 第 ${item.page} 页` : ''}
	                    </div>
	                    {item.title && (
	                      <div className="font-bold text-gray-700 mb-2">
                        {item.title}
                      </div>
                    )}
	                    <div className="text-sm text-gray-600 whitespace-pre-wrap">
	                      {item.content}
	                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="上传并解析 PDF 后，这里会显示按选项抽取出的文本、标题或表格内容。" />
          )}
        </div>
      </div>
    </div>
  );
};

export default ParseFile;
