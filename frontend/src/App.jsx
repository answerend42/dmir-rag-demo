// 文件路径：src/App.jsx
/**
 * @file App.jsx
 * @brief RAG 演示前端的顶层 React 路由。
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import LoadFile from './pages/LoadFile';
import ChunkFile from './pages/ChunkFile';
import EmbeddingFile from './pages/EmbeddingFile';
import Indexing from './pages/Indexing';
import Search from './pages/Search';
import ParseFile from './pages/ParseFile';
import Generation from './pages/Generation';

/**
 * @brief 渲染带固定侧边栏导航的应用外壳。
 * @returns {JSX.Element} 单页应用路由布局。
 */
const App = () => {
  return (
    <Router>
      <div className="flex">
        <Sidebar />
        <main className="app-main ml-64 flex-1 min-h-screen">
          <Routes>
            <Route path="/load-file" element={<LoadFile />} />  
            <Route path="/chunk-file" element={<ChunkFile />} />  
            <Route path="/parse-file" element={<ParseFile />} />
            <Route path="/embedding" element={<EmbeddingFile />} />
            <Route path="/indexing" element={<Indexing />} />
            <Route path="/search" element={<Search />} />
            <Route path="/generation" element={<Generation />} />
            <Route path="/" element={<LoadFile />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
