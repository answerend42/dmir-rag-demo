// 文件路径：src/components/Sidebar.jsx
/**
 * @file Sidebar.jsx
 * @brief RAG 工作流的固定导航侧边栏。
 */
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import ragLogo from '../assets/raglogo.png';

/**
 * @brief 渲染路由链接并高亮当前工作流步骤。
 * @returns {JSX.Element} 侧边栏导航。
 */
const Sidebar = () => {
  const location = useLocation();
  const links = [
    { to: "/load-file", text: "文档导入" },
    { to: "/chunk-file", text: "知识分块" },
    { to: "/parse-file", text: "文件解析" },
    { to: "/embedding", text: "向量存储" },
    { to: "/indexing", text: "向量库索引" },
    { to: "/search", text: "相似性检索" },
    { to: "/generation", text: "响应生成" }
  ];

  return (
    <aside className="app-sidebar fixed left-0 top-0 h-screen w-64">
      <div className="p-4">
        <img 
          src={ragLogo} 
          alt="RAG Demo"
          className="mb-4 w-full rounded"
        />
        <div className="rounded border px-3 py-2 text-xs">
          <div className="font-semibold">RAG Demo</div>
          <div className="mt-1 opacity-75">从文档到证据回答</div>
        </div>
      </div>
      <nav className="px-3">
        {links.map((link, index) => (
          <Link
            key={link.to}
            to={link.to}
            className={`sidebar-link ${location.pathname === link.to ? 'sidebar-link-active' : ''}`}
          >
            <span className="sidebar-step">{String(index + 1).padStart(2, '0')}</span>
            <span>{link.text}</span>
          </Link>
        ))}
      </nav>
      <div className="absolute bottom-0 left-0 right-0 p-4">
        <div className="rounded border px-3 py-3 text-xs leading-6 opacity-80">
          基于课程参考 RAG 框架改造，用于展示完整检索增强生成流程。
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
