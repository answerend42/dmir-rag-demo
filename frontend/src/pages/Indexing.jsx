// 文件路径：src/pages/Indexing.jsx
/**
 * @file Indexing.jsx
 * @brief 向量数据库索引和集合管理页面。
 */
import { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const INDEX_MODE_CONFIG = {
  chroma: {
    hnsw: {
      label: 'HNSW（Chroma）',
      summary: 'Chroma 本地 collection 使用 HNSW 图结构近似最近邻索引，度量设置为 cosine。',
    },
  },
  faiss: {
    flat: {
      label: 'Flat 精确检索',
      summary: 'FAISS IndexFlatIP 在归一化向量上精确计算全部向量的 cosine，相当于暴力检索基线。',
    },
    ivf: {
      label: 'IVF 倒排文件',
      summary: 'FAISS IndexIVFFlat 先用倒排簇缩小候选集合，再返回归一化向量内积。',
    },
    lsh: {
      label: 'LSH 随机超平面',
      summary: 'FAISS IndexLSH 用二进制签名召回候选，再由后端按 cosine 重排。',
    },
  },
};

const getProviderIndexConfig = (provider) => (
  INDEX_MODE_CONFIG[provider] || INDEX_MODE_CONFIG.chroma
);

const formatProviderName = (provider) => {
  const labels = {
    chroma: 'Chroma',
    faiss: 'FAISS',
  };
  return labels[String(provider || '').toLowerCase()] || provider || '-';
};

const formatIndexModeName = (mode) => {
  const labels = {
    hnsw: 'HNSW',
    flat: 'Flat',
    ivf: 'IVF',
    lsh: 'LSH',
  };
  return labels[String(mode || '').toLowerCase()] || mode || '未标注索引';
};

const getCollectionOptionLabel = (collection = {}, selectedProvider = '') => {
  const provider = collection.database || selectedProvider;
  const parts = [
    `${collection.count ?? 0} 个文档`,
    formatProviderName(provider),
    formatIndexModeName(collection.index_mode),
  ].filter(Boolean);
  return `${collection.name}${parts.length ? ` (${parts.join(' · ')})` : ''}`;
};

/**
 * @brief 渲染嵌入索引和集合查看控件。
 * @returns {JSX.Element} 向量索引工作流页面。
 */
const Indexing = () => {
  const [embeddingFile, setEmbeddingFile] = useState('');
  const [indexMode, setIndexMode] = useState('hnsw');
  const [status, setStatus] = useState('');
  const [embeddedFiles, setEmbeddedFiles] = useState([]);
  const [indexingResult, setIndexingResult] = useState(null);
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('chroma');
  const activeIndexConfig = getProviderIndexConfig(selectedProvider);
  const activeIndexModes = Object.keys(activeIndexConfig);
  const activeIndexModeConfig = activeIndexConfig[indexMode] || activeIndexConfig[activeIndexModes[0]];

  useEffect(() => {
    fetchEmbeddedFiles();
    fetchCollections();
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取providers列表
        const providersResponse = await fetch(`${apiBaseUrl}/providers`);
        const providersData = await providersResponse.json();
        const nextProviders = providersData.providers || [];
        setProviders(nextProviders);
        if (nextProviders.length > 0 && !nextProviders.some(provider => provider.id === selectedProvider)) {
          setSelectedProvider(nextProviders[0].id);
          return;
        }

        // 获取collections列表
        const collectionsResponse = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
        const collectionsData = await collectionsResponse.json();
        const nextCollections = collectionsData.collections || [];
        setCollections(nextCollections);
        setSelectedCollection((currentCollection) => (
          nextCollections.some((collection) => collection.id === currentCollection)
            ? currentCollection
            : ''
        ));
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedProvider]);

  useEffect(() => {
    const modes = Object.keys(getProviderIndexConfig(selectedProvider));
    if (!modes.includes(indexMode)) {
      setIndexMode(modes[0]);
    }
  }, [selectedProvider, indexMode]);

  const fetchEmbeddedFiles = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/list-embedded`);
      const data = await response.json();
      if (data.documents) {
        setEmbeddedFiles(data.documents.map(doc => ({
          ...doc,
          id: doc.name,
          displayName: doc.name
        })));
      }
    } catch (error) {
      console.error('Error fetching embedded files:', error);
      setStatus('加载向量文件失败');
    }
  };

  const fetchCollections = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
      const data = await response.json();
      const nextCollections = data.collections || [];
      setCollections(nextCollections);
      setSelectedCollection((currentCollection) => (
        nextCollections.some((collection) => collection.id === currentCollection)
          ? currentCollection
          : ''
      ));
    } catch (error) {
      console.error('Error fetching collections:', error);
    }
  };

  const handleIndex = async () => {
    if (!embeddingFile) {
      setStatus('请选择需要索引的向量文件');
      return;
    }

    setStatus('正在建立索引...');
    try {
      const selectedIndexMode = activeIndexModes.includes(indexMode) ? indexMode : activeIndexModes[0];
      const response = await fetch(`${apiBaseUrl}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileId: embeddingFile,
          vectorDb: selectedProvider,
          indexMode: selectedIndexMode
        }),
      });
      
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }
      setIndexingResult(data);
      setSelectedCollection(data.collection_name || '');
      setStatus('索引建立完成');
      const collectionsResponse = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
      const collectionsData = await collectionsResponse.json();
      setCollections(collectionsData.collections || []);
    } catch (error) {
      console.error('Error indexing:', error);
      setStatus('索引失败: ' + error.message);
    }
  };

  const handleDisplay = async (collectionName) => {
    if (!collectionName) return;
    
    try {
      const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`);
      const data = await response.json();
      console.log("after await")
      
      // 只包含有实际值的属性
      const result = {
        database: selectedProvider,
        collection_name: data.name,
        total_vectors: data.num_entities,
        index_size: data.num_entities,
        index_mode: data.schema?.index_mode,
        index_family: data.schema?.index_family,
        index_parameters: data.schema?.index_parameters,
      };

      // 只在有实际值时添加可选属性
      //const indexType="hnsw"

     // const indexType = data.schema?.fields?.find(f => f.name === 'vector')?.index_params?.index_type;
     // if (indexType) {
     //   result.index_mode = indexType;
     // }

      if (data.processing_time) {
        result.processing_time = data.processing_time;
      }

      setIndexingResult(result);
    } catch (error) {
      console.error('Error displaying collection:', error);
    }
  };

  const handleDelete = async (collectionName) => {
    if (!collectionName) return;
    
    if (window.confirm(`确定要删除集合 "${collectionName}" 吗？`)) {
      try {
        await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`, {
          method: 'DELETE',
        });
        setSelectedCollection('');
        // 重新获取collections列表
        const response = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
        const data = await response.json();
        setCollections(data.collections);
      } catch (error) {
        console.error('Error deleting collection:', error);
      }
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">向量库索引</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板：控制区 */}
        <div className="col-span-3">
          <div className="p-4 border rounded-lg bg-white shadow-sm space-y-4">
            {/* 嵌入文件选择 */}
            <div>
              <label className="block text-sm font-medium mb-1">需要索引的文件</label>
              <select
                value={embeddingFile}
                onChange={(e) => setEmbeddingFile(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                  <option value="">请选择文件...</option>
                {embeddedFiles.map(file => (
                  <option key={file.name} value={file.name}>
                    {file.displayName}
                  </option>
                ))}
              </select>
            </div>

            {/* 向量数据库选择 */}
            <div>
              <label className="block text-sm font-medium mb-1">向量库</label>
              <select
                value={selectedProvider}
                onChange={(e) => {
                  setSelectedProvider(e.target.value);
                  setSelectedCollection('');
                  setIndexingResult(null);
                  setCollections([]);
                }}
                className="block w-full p-2 border rounded"
              >
                {providers.map(provider => (
                  <option key={provider.id} value={provider.id}>
                    {provider.name}
                  </option>
                ))}
              </select>
            </div>

            {/* 索引模式选择 */}
            <div>
              <label className="block text-sm font-medium mb-1">索引方式</label>
              <select
                value={indexMode}
                onChange={(e) => setIndexMode(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                {activeIndexModes.map(mode => (
                  <option key={mode} value={mode}>
                    {activeIndexConfig[mode].label}
                  </option>
                ))}
              </select>
              <p className="mt-2 rounded border border-slate-200 bg-slate-50 p-2 text-xs leading-5 text-slate-600">
                {activeIndexModeConfig.summary}
              </p>
            </div>

            {/* 操作按钮和集合管理 */}
            <div className="space-y-2">
              {/* 数据索引按钮 */}
              <button 
                onClick={handleIndex}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
                disabled={!embeddingFile}
              >
                索引数据
              </button>

              {/* 集合选择 */}
              <div>
                <label className="block text-sm font-medium mb-1">索引集合</label>
                <select
                  value={selectedCollection}
                  onChange={(e) => setSelectedCollection(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">请选择集合...</option>
                  {collections.map(coll => (
                    <option key={coll.id} value={coll.id}>
                    {getCollectionOptionLabel(coll, selectedProvider)}
                  </option>
                ))}
              </select>
              </div>

              {/* 集合显示按钮 */}
              <button
                onClick={() => handleDisplay(selectedCollection)}
                disabled={!selectedCollection}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                显示集合
              </button>

              {/* 集合删除按钮 */}
              <button
                onClick={() => handleDelete(selectedCollection)}
                disabled={!selectedCollection}
                className="w-full px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-red-300"
              >
                删除集合
              </button>
            </div>

            {status && (
              <div className="mt-4 p-3 rounded border bg-gray-50">
                <p className="text-sm">{status}</p>
              </div>
            )}
          </div>
        </div>

        {/* 右侧面板：结果 */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {indexingResult ? (
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-4">索引结果</h3>
              <div className="space-y-3">
                <div className="p-3 border rounded bg-gray-50">
                  <div className="text-sm text-gray-600">
                    <p>数据库: {indexingResult.database}</p>
                    {indexingResult.index_mode && (
                      <p>索引模式: {indexingResult.index_mode}</p>
                    )}
                    {indexingResult.index_family && (
                      <p>索引族: {indexingResult.index_family}</p>
                    )}
                    <p>向量总数: {indexingResult.total_vectors}</p>
                    <p>索引大小: {indexingResult.index_size}</p>
                    {indexingResult.processing_time && (
                      <p>处理耗时: {indexingResult.processing_time}s</p>
                    )}
                    <p>集合名称: {indexingResult.collection_name}</p>
                    {indexingResult.index_parameters && Object.keys(indexingResult.index_parameters).length > 0 && (
                      <pre className="mt-2 overflow-auto rounded border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700">
                        {JSON.stringify(indexingResult.index_parameters, null, 2)}
                      </pre>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <RandomImage message="选择向量文件并建立索引后，这里会显示向量库集合和索引统计。" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Indexing;
