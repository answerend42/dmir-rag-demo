"""! @file main.py
@brief RAG 演示后端的 FastAPI 入口。
@details 本模块串联文档读入、解析、分块、嵌入、向量索引、检索、评估与回答生成服务。
端点处理函数保持轻量，领域逻辑委托给 service 层。
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## @brief 由 Uvicorn 提供服务的后端应用对象。
app = FastAPI()

# 确保必要的目录存在
os.makedirs("temp", exist_ok=True)
os.makedirs("01-chunked-docs", exist_ok=True)
os.makedirs("02-embedded-docs", exist_ok=True)

# 配置跨域访问
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _safe_storage_stem(filename: str, fallback: str = "course_qa") -> str:
    """! @brief 从上传文件名生成安全的相对存储名称。"""
    original_name = Path(filename or fallback).name
    stem = Path(original_name).stem or fallback
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def _answer_texts_without_quality(answers: Any) -> List[str]:
    """! @brief 提取候选答案文本，丢弃 answer_quality 等评测标签。"""
    if not isinstance(answers, list):
        return []
    answer_texts = []
    for answer_item in answers:
        text = ""
        if isinstance(answer_item, dict):
            text = str(answer_item.get("answer") or answer_item.get("text") or "").strip()
        elif isinstance(answer_item, str):
            text = answer_item.strip()
        if text:
            answer_texts.append(text)
    return answer_texts


def _build_course_qa_loaded_document(payload: Any, filename: str) -> Dict[str, Any]:
    """! @brief 将课程 QA JSON 规范化为可在 02 继续分块的已导入文档。"""
    if not isinstance(payload, dict):
        raise ValueError("课程 QA JSON 顶层必须是对象，键为课程主题。")

    chunks = []
    qa_items = []
    topic_count = 0
    for topic_index, (topic, questions) in enumerate(payload.items(), 1):
        if not isinstance(questions, list):
            continue
        topic_count += 1
        for question_index, item in enumerate(questions, 1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            answer_texts = _answer_texts_without_quality(item.get("answers"))
            item_id = f"{topic_index}-{question_index}"
            answer_lines = [
                f"{index}. {answer_text}"
                for index, answer_text in enumerate(answer_texts, 1)
            ]
            content_parts = [
                f"课程主题：{topic}",
                f"问题：{question}",
            ]
            if answer_lines:
                content_parts.extend(["候选答案：", *answer_lines])
            content = "\n".join(content_parts)
            chunks.append({
                "content": content,
                "metadata": {
                    "chunk_id": len(chunks) + 1,
                    "page_number": topic_index,
                    "page_range": str(topic),
                    "word_count": len(content.split()),
                    "dataset_type": "course_qa",
                    "topic": str(topic),
                    "qa_id": str(item.get("id") or question_index),
                    "source_file": Path(filename or "course_qa.json").name,
                    "answer_count": len(answer_texts),
                },
            })
            qa_items.append({
                "item_id": item_id,
                "topic": str(topic),
                "qa_id": str(item.get("id") or question_index),
                "question": question,
                "answers": [
                    {
                        "answer_id": f"A{answer_index}",
                        "answer": answer_text,
                    }
                    for answer_index, answer_text in enumerate(answer_texts, 1)
                ],
                "answer_count": len(answer_texts),
            })

    if not chunks:
        raise ValueError("课程 QA JSON 中没有可导入的问题。")

    timestamp = datetime.now().isoformat()
    return {
        "filename": Path(filename or "course_qa.json").name,
        "document_name": Path(filename or "course_qa.json").name,
        "dataset_type": "course_qa",
        "source_format": "json",
        "total_chunks": len(chunks),
        "total_pages": topic_count,
        "loading_method": "course_qa_json",
        "chunking_method": "structured_json_load",
        "timestamp": timestamp,
        "qa_items": qa_items,
        "chunks": chunks,
    }


def _chunk_course_qa_document(doc_data: Dict[str, Any]) -> Dict[str, Any]:
    """! @brief 将已导入课程 QA JSON 转成一题一块的标准 chunk 文档。"""
    source_chunks = doc_data.get("chunks", [])
    chunks = []
    for source_chunk in source_chunks:
        if not isinstance(source_chunk, dict) or not source_chunk.get("content"):
            continue
        source_metadata = source_chunk.get("metadata") or {}
        content = str(source_chunk.get("content", "")).strip()
        metadata = {
            **{
                key: value
                for key, value in source_metadata.items()
                if key != "answer_quality"
            },
            "chunk_id": len(chunks) + 1,
            "dataset_type": "course_qa",
            "page_number": source_metadata.get("page_number", 1),
            "page_range": source_metadata.get("page_range", source_metadata.get("topic", "course_qa")),
            "word_count": len(content.split()),
        }
        chunks.append({
            "content": content,
            "metadata": metadata,
        })

    if not chunks:
        raise ValueError("课程 QA 文档没有可分块内容。")

    return {
        "filename": doc_data.get("filename", "course_qa.json"),
        "document_name": doc_data.get("document_name", doc_data.get("filename", "course_qa.json")),
        "dataset_type": "course_qa",
        "source_format": "json",
        "total_chunks": len(chunks),
        "total_pages": doc_data.get("total_pages", 1),
        "loading_method": doc_data.get("loading_method", "course_qa_json"),
        "chunking_method": "course_qa_items",
        "chunk_size": None,
        "chunk_overlap": 0,
        "timestamp": datetime.now().isoformat(),
        "chunks": chunks,
    }


def _parse_course_qa_chunk_content(content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """! @brief 从旧版课程 QA chunk 文本恢复前端任务条目。"""
    question = ""
    answers = []
    reading_answers = False
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if line.startswith("问题："):
            question = line.replace("问题：", "", 1).strip()
            reading_answers = False
        elif line == "候选答案：":
            reading_answers = True
        elif reading_answers and "." in line:
            prefix, answer_text = line.split(".", 1)
            if prefix.strip().isdigit() and answer_text.strip():
                answers.append({
                    "answer_id": f"A{len(answers) + 1}",
                    "answer": answer_text.strip(),
                })

    return {
        "item_id": str(metadata.get("item_id") or metadata.get("chunk_id") or metadata.get("qa_id") or len(answers)),
        "topic": str(metadata.get("topic") or metadata.get("page_range") or "课程 QA"),
        "qa_id": str(metadata.get("qa_id") or metadata.get("chunk_id") or ""),
        "question": question,
        "answers": answers,
        "answer_count": len(answers),
    }


def _extract_course_qa_items(doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """! @brief 从课程 QA 已导入文档提取无 answer_quality 的题目与候选答案。"""
    items = []
    if isinstance(doc_data.get("qa_items"), list):
        for index, item in enumerate(doc_data.get("qa_items", []), 1):
            if not isinstance(item, dict):
                continue
            answers = []
            for answer_index, answer_item in enumerate(item.get("answers") or [], 1):
                answer_text = ""
                if isinstance(answer_item, dict):
                    answer_text = str(answer_item.get("answer") or answer_item.get("text") or "").strip()
                elif isinstance(answer_item, str):
                    answer_text = answer_item.strip()
                if answer_text:
                    answers.append({
                        "answer_id": f"A{answer_index}",
                        "answer": answer_text,
                    })
            question = str(item.get("question") or "").strip()
            if question:
                items.append({
                    "item_id": str(item.get("item_id") or index),
                    "topic": str(item.get("topic") or "课程 QA"),
                    "qa_id": str(item.get("qa_id") or index),
                    "question": question,
                    "answers": answers,
                    "answer_count": len(answers),
                })
        return items

    for index, chunk in enumerate(doc_data.get("chunks") or [], 1):
        if not isinstance(chunk, dict):
            continue
        item = _parse_course_qa_chunk_content(chunk.get("content", ""), chunk.get("metadata") or {})
        if item.get("question"):
            item["item_id"] = str(item.get("item_id") or index)
            items.append(item)
    return items


def _split_text_document_sections(text: str) -> List[Dict[str, str]]:
    """! @brief 将 Markdown/TXT 课程知识文档按标题切成章节。"""
    sections = []
    current_title = "课程知识文档"
    current_lines = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append({"title": current_title, "content": content})
            current_title = stripped.lstrip("#").strip() or current_title
            current_lines = [stripped]
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"title": current_title, "content": content})

    if not sections and text.strip():
        sections.append({"title": "课程知识文档", "content": text.strip()})
    return sections


def _build_course_knowledge_loaded_document(text: str, filename: str) -> Dict[str, Any]:
    """! @brief 将课程知识 Markdown/TXT 规范化为可继续分块的外部知识文档。"""
    sections = _split_text_document_sections(text)
    if not sections:
        raise ValueError("课程知识文档没有可导入的文本内容。")

    source_name = Path(filename or "course_knowledge.md").name
    suffix = Path(source_name).suffix.lower()
    source_format = "markdown" if suffix in {".md", ".markdown"} else "text"
    chunks = []
    for section_index, section in enumerate(sections, 1):
        content = section["content"].strip()
        if not content:
            continue
        chunks.append({
            "content": content,
            "metadata": {
                "chunk_id": len(chunks) + 1,
                "page_number": section_index,
                "page_range": section.get("title") or str(section_index),
                "word_count": len(content.split()),
                "dataset_type": "course_knowledge",
                "source_role": "external_knowledge",
                "source_format": source_format,
                "section_title": section.get("title") or f"章节 {section_index}",
                "source_file": source_name,
            },
        })

    if not chunks:
        raise ValueError("课程知识文档没有可导入的有效章节。")

    return {
        "filename": source_name,
        "document_name": source_name,
        "dataset_type": "course_knowledge",
        "source_role": "external_knowledge",
        "source_format": source_format,
        "total_chunks": len(chunks),
        "total_pages": len(chunks),
        "loading_method": "course_knowledge_text",
        "chunking_method": "section_load",
        "timestamp": datetime.now().isoformat(),
        "chunks": chunks,
    }

@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    loading_method: str = Form(...),
    chunking_option: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(0),
):
    """! @brief 在一次请求中读入并分块上传的 PDF。
    @param file 上传的 PDF 文件。
    @param loading_method 读入后端，例如 pymupdf、pypdf 或 unstructured。
    @param chunking_option 分块策略名称。
    @param chunk_size 固定大小分块的最大块长度。
    @param chunk_overlap 固定大小分块时相邻块重叠字符数。
    @return 包含分块文档数据的 JSON 对象。
    """
    try:
        # 保存上传的文件
        temp_path = os.path.join("temp", file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 准备元数据
        metadata = {
            "filename": file.filename,
            "loading_method": loading_method,
            "original_file_size": len(content),
            "processing_date": datetime.now().isoformat(),
            "chunking_method": chunking_option,
        }
        
        from services.loading_service import LoadingService
        from services.chunking_service import ChunkingService

        loading_service = LoadingService()
        raw_text = loading_service.load_pdf(temp_path, loading_method)
        metadata["total_pages"] = loading_service.get_total_pages()
        
        page_map = loading_service.get_page_map()
        
        chunking_service = ChunkingService()
        chunks = chunking_service.chunk_text(
            raw_text, 
            chunking_option, 
            metadata,
            page_map=page_map,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # 清理临时文件
        os.remove(temp_path)
        
        return {"chunks": chunks}
    except ValueError as e:
        logger.warning(f"分块参数非法: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

@app.post("/save")
async def save_chunks(data: dict):
    """! @brief 持久化手动提交的分块文档。
    @param data 请求体，包含 docName、chunks 和可选 metadata。
    @return 保存状态和相对文件路径。
    """
    try:
        doc_name = data.get("docName")
        chunks = data.get("chunks")
        metadata = data.get("metadata", {})
        
        if not doc_name or not chunks:
            raise ValueError("Missing required fields")
        
        # 构建文件名
        filename = f"{doc_name}.json"
        filepath = os.path.join("01-chunked-docs", filename)
        
        # 保存数据
        document_data = {
            "document_name": doc_name,
            "metadata": metadata,
            "chunks": chunks
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "message": "Document saved successfully",
            "filepath": filepath
        }
    except Exception as e:
        logger.error(f"Error saving document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/list-docs")
async def list_documents():
    """! @brief 列出已保存的分块文档。
    @return 01-chunked-docs 中的文档标识和显示名称。
    """
    try:
        docs = []
        docs_dir = "01-chunked-docs"
        for filename in os.listdir(docs_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    docs.append({
                        "id": filename,
                        "name": doc_data["document_name"]
                    })
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise

@app.post("/embed")
async def embed_document(data: dict = Body(...)):
    """! @brief 为已读入或已分块的文档创建嵌入。
    @param data 请求体，包含 documentId、provider 和 model。
    @return 嵌入数据和持久化 JSON 路径。
    """
    try:
        doc_id = data.get("documentId")
        provider = data.get("provider")
        model = data.get("model")
        
        if not all([doc_id, provider, model]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
            
        # 直接使用完整文件名查找
        loaded_path = os.path.join("01-loaded-docs", doc_id)
        chunked_path = os.path.join("01-chunked-docs", doc_id)
        
        doc_path = None
        if os.path.exists(loaded_path):
            doc_path = loaded_path
        elif os.path.exists(chunked_path):
            doc_path = chunked_path
            
        if not doc_path:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
            
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        from services.embedding_service import EmbeddingService, EmbeddingConfig

        # 创建 EmbeddingConfig 和 EmbeddingService
        config = EmbeddingConfig(provider=provider, model_name=model)
        embedding_service = EmbeddingService()
        
        # 准备输入数据
        input_data = {
            "chunks": doc_data["chunks"],
            "metadata": {
                "filename": doc_data["filename"],
                "document_name": doc_data.get("document_name", doc_data["filename"]),
                "total_chunks": doc_data["total_chunks"],
                "total_pages": doc_data["total_pages"],
                "loading_method": doc_data["loading_method"],
                "chunking_method": doc_data["chunking_method"],
                "dataset_type": doc_data.get("dataset_type"),
                "source_format": doc_data.get("source_format"),
                "source_role": doc_data.get("source_role"),
            }
        }
        
        # 创建嵌入 - 只接收两个返回值
        embeddings, _ = embedding_service.create_embeddings(input_data, config)
        
        # 保存嵌入结果
        output_path = embedding_service.save_embeddings(doc_id, embeddings)
        
        return {
            "status": "success",
            "message": "Embeddings created successfully",
            "filepath": output_path,
            "embeddings": embeddings  # 添加embeddings到响应中
        }
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-embedded")
async def list_embedded_docs():
    """! @brief 列出所有已嵌入的文档。
    @return 02-embedded-docs 中的嵌入文档元数据。
    """
    try:
        documents = []
        embedded_dir = "02-embedded-docs"
        logger.info(f"Scanning directory: {embedded_dir}")
        
        if not os.path.exists(embedded_dir):
            logger.warning(f"Directory {embedded_dir} does not exist")
            return {"documents": []}
            
        for filename in os.listdir(embedded_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(embedded_dir, filename)
                logger.info(f"Reading file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 使用实际的文件名，而不是文档名
                        doc_info = {
                            "name": filename,  # 保持原始文件名
                            "metadata": {
                                "document_name": data.get("document_name", filename),
                                "embedding_model": data.get("embedding_model", ""),
                                "embedding_provider": data.get("embedding_provider", ""),
                                "embedding_timestamp": data.get("created_at", ""),
                                "vector_dimension": data.get("vector_dimension", 0)
                            }
                        }
                        logger.info(f"Added document info: {doc_info}")
                        documents.append(doc_info)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    
        logger.info(f"Total documents found: {len(documents)}")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing embedded documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_embeddings(data: dict):
    """! @brief 将嵌入文件写入配置的向量库索引。
    @param data 请求体，包含 fileId、vectorDb 和 indexMode。
    @return 向量库索引结果摘要。
    """
    try:
        file_id = data.get("fileId")
        vector_db = data.get("vectorDb")
        index_mode = data.get("indexMode")
        
        if not all([file_id, vector_db, index_mode]):
            raise ValueError("Missing required fields")
            
        embedding_file = os.path.join("02-embedded-docs", file_id)
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {file_id}")
            
        from services.vector_store_service import VectorStoreService, VectorDBConfig

        config = VectorDBConfig(provider=vector_db, index_mode=index_mode)
        vector_store_service = VectorStoreService()
        result = vector_store_service.index_embeddings(embedding_file, config)
        
        return result
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/providers")
async def get_providers():
    """! @brief 获取支持的向量数据库列表."""
    try:
        from services.search_service import SearchService

        search_service = SearchService()
        providers = search_service.get_providers()
        return {"providers": providers}
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/collections")
async def get_collections(
    provider: str = Query(default="chroma")
):
    """! @brief 获取指定向量数据库中的集合."""
    try:
        from services.search_service import SearchService

        search_service = SearchService()
        collections = search_service.list_collections(provider)
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/search")
async def search(
    query: str = Body(...),
    collection_id: str = Body(...),
    top_k: int = Body(3),
    threshold: float = Body(0.3),
    word_count_threshold: int = Body(0),
    save_results: bool = Body(False),
    include_query_embedding: bool = Body(False),
):
    """! @brief 执行向量搜索.
    @param query 用户查询文本。
    @param collection_id 要检索的向量集合。
    @param top_k 返回的最大命中数量。
    @param threshold 最低相似度分数。
    @param word_count_threshold 最小词数过滤阈值。
    @param save_results 是否保存检索结果。
    @param include_query_embedding 是否回传查询向量用于前端可视化。
    @return 包装在 results 对象中的搜索结果。
    """
    try:
        # 记录传入的搜索请求详情
        logger.info(f"Search request - Query: {query}, Collection: {collection_id}, Top K: {top_k}, Threshold: {threshold}, Word Count Threshold: {word_count_threshold}")
        
        from services.search_service import SearchService

        search_service = SearchService()
        
        # 调用搜索函数前记录日志
        logger.info("Calling search service...")
        
        results = await search_service.search(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            threshold=threshold,
            word_count_threshold=word_count_threshold,
            save_results=save_results,
            include_query_embedding=include_query_embedding,
        )
        
        # 记录搜索结果
        logger.info(f"Search response: {results}")
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/collections/{provider}/{collection_name}/embeddings")
async def get_collection_embeddings(provider: str, collection_name: str):
    """! @brief 获取指定 collection 的全部向量，用于数值查看。
    @param provider 向量数据库提供方，支持 chroma 或 faiss。
    @param collection_name 要读取的 collection 名称。
    @return collection 内的向量、文本和元数据。
    """
    try:
        provider_value = provider.strip().lower()
        if provider_value not in {"chroma", "faiss"}:
            raise HTTPException(status_code=400, detail="当前向量可视化只支持 Chroma 或 FAISS collection")

        from services.search_service import SearchService

        search_service = SearchService()
        return search_service.get_collection_embeddings(collection_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{provider}/{collection_name}/projection")
async def get_collection_projection(provider: str, collection_name: str, payload: dict = Body(None)):
    """! @brief 获取指定 collection 的后端二维投影。
    @param provider 向量数据库提供方，支持 chroma 或 faiss。
    @param collection_name 要读取的 collection 名称。
    @param payload 投影方法和附加向量。
    @return collection 向量二维投影。
    """
    try:
        provider_value = provider.strip().lower()
        if provider_value not in {"chroma", "faiss"}:
            raise HTTPException(status_code=400, detail="当前向量投影只支持 Chroma 或 FAISS collection")

        from services.search_service import SearchService

        request_data = payload or {}
        search_service = SearchService()
        return search_service.get_collection_projection(
            collection_name,
            method=request_data.get("method", "tsne"),
            overlays=request_data.get("overlays", []),
            target_dimensions=request_data.get(
                "target_dimensions",
                request_data.get("dimensions", request_data.get("target_dimension", 3)),
            ),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting collection projection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{provider}")
async def get_provider_collections(provider: str):
    """! @brief 获取指定向量数据库提供方的集合列表。"""
    try:
        from services.search_service import SearchService

        search_service = SearchService()
        collections = search_service.list_collections(provider)
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error getting collections for provider {provider}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/collections/{provider}/{collection_name}")
async def get_collection_info(provider: str, collection_name: str):
    """! @brief 获取指定集合的详细信息。"""
    try:
        from services.vector_store_service import VectorStoreService

        vector_store_service = VectorStoreService()
        info = vector_store_service.get_collection_info(provider, collection_name)
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.delete("/collections/{provider}/{collection_name}")
async def delete_collection(provider: str, collection_name: str):
    """! @brief 删除指定集合。"""
    try:
        from services.vector_store_service import VectorStoreService

        vector_store_service = VectorStoreService()
        success = vector_store_service.delete_collection(provider, collection_name)
        if success:
            return {"message": f"Collection {collection_name} deleted successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to delete collection {collection_name}"
            )
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/documents")
async def get_documents(type: str = Query("all")):
    """! @brief 列出已读入和/或已分块的文档。
    @param type 可选 all、loaded 或 chunked。
    @return 按持久化存储类型归类的文档摘要。
    """
    try:
        documents = []
        
        # 读取loaded文档
        if type in ["all", "loaded"]:
            loaded_dir = "01-loaded-docs"
            if os.path.exists(loaded_dir):
                for filename in os.listdir(loaded_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(loaded_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            documents.append({
                                "id": filename,
                                "name": filename,
                                "type": "loaded",
                                "metadata": {
                                    "total_pages": doc_data.get("total_pages"),
                                    "total_chunks": doc_data.get("total_chunks"),
                                    "loading_method": doc_data.get("loading_method"),
                                    "chunking_method": doc_data.get("chunking_method"),
                                    "dataset_type": doc_data.get("dataset_type"),
                                    "source_format": doc_data.get("source_format"),
                                    "timestamp": doc_data.get("timestamp")
                                }
                            })

        # 读取chunked文档
        if type in ["all", "chunked"]:
            chunked_dir = "01-chunked-docs"
            if os.path.exists(chunked_dir):
                for filename in os.listdir(chunked_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(chunked_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            documents.append({
                                "id": filename,
                                "name": filename,  # 保持原始文件名
                                "type": "chunked",
                                "metadata": {
                                    "total_pages": doc_data.get("total_pages"),
                                    "total_chunks": doc_data.get("total_chunks"),
                                    "chunking_method": doc_data.get("chunking_method"),
                                    "dataset_type": doc_data.get("dataset_type"),
                                    "source_format": doc_data.get("source_format"),
                                    "timestamp": doc_data.get("timestamp")
                                }
                            })
        
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_name}")
async def get_document(doc_name: str, type: str = Query("loaded")):
    """! @brief 读取已持久化的读入文档或分块文档。
    @param doc_name JSON 文件名或基础文档名。
    @param type 存储分组，可选 loaded 或 chunked。
    @return 完整文档 JSON 数据。
    """
    try:

        base_name = doc_name.replace('.json', '')
        file_name = f"{base_name}.json"
        
        # 根据类型选择不同的目录
        directory = "01-loaded-docs" if type == "loaded" else "01-chunked-docs"
        file_path = os.path.join(directory, file_name)
        
        logger.info(f"Attempting to read document from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Document not found at path: {file_path}")
            raise HTTPException(status_code=404, detail="Document not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        return doc_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str, type: str = Query("loaded")):
    """! @brief 删除已持久化的读入文档或分块文档。
    @param doc_name JSON 文件名或基础文档名。
    @param type 存储分组，可选 loaded 或 chunked。
    @return 删除状态。
    """
    try:
        # 移除已有的 .json 扩展名（如果有）然后添加一个
        base_name = doc_name.replace('.json', '')
        file_name = f"{base_name}.json"
        
        # 根据类型选择不同的目录
        directory = "01-loaded-docs" if type == "loaded" else "01-chunked-docs"
        file_path = os.path.join(directory, file_name)
        
        logger.info(f"Attempting to delete document: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Document not found at path: {file_path}")
            raise HTTPException(status_code=404, detail="Document not found")
            
        # 删除文件
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"Document {doc_name} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _load_embedded_doc_embeddings(doc_name: str) -> Dict[str, Any]:
    """! @brief 读取 02-embedded-docs 中的向量条目。"""
    logger.info(f"Attempting to read document: {doc_name}")
    file_path = os.path.join("02-embedded-docs", doc_name)

    if not os.path.exists(file_path):
        logger.error(f"Document not found: {file_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_name} not found"
        )

    with open(file_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
        logger.info(f"Successfully read document: {doc_name}")
        doc_embeddings = doc_data.get("embeddings", [])

        return {
            "embeddings": [
                {
                    "embedding": embedding["embedding"],
                    "metadata": {
                        "document_name": doc_data.get("document_name", doc_name),
                        "chunk_id": idx + 1,
                        "total_chunks": len(doc_embeddings),
                        "content": embedding["metadata"].get("content", ""),
                        "page_number": embedding["metadata"].get("page_number", ""),
                        "page_range": embedding["metadata"].get("page_range", ""),
                        "embedding_model": doc_data.get("embedding_model", ""),
                        "embedding_provider": doc_data.get("embedding_provider", ""),
                        "embedding_timestamp": doc_data.get("created_at", ""),
                        "vector_dimension": doc_data.get("vector_dimension", 0)
                    }
                }
                for idx, embedding in enumerate(doc_embeddings)
            ]
        }

@app.get("/embedded-docs/{doc_name}")
async def get_embedded_doc(doc_name: str):
    """! @brief 获取指定的嵌入文档。"""
    try:
        return _load_embedded_doc_embeddings(doc_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embedded document {doc_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embedded-docs/{doc_name}/projection")
async def get_embedded_doc_projection(doc_name: str, payload: dict = Body(None)):
    """! @brief 获取指定嵌入文档的后端二维投影。"""
    try:
        from services.projection_service import VectorProjectionService

        request_data = payload or {}
        doc_payload = _load_embedded_doc_embeddings(doc_name)
        return VectorProjectionService.project_embeddings(
            doc_payload.get("embeddings", []),
            method=request_data.get("method", "tsne"),
            overlays=request_data.get("overlays", []),
            source_id=doc_name,
            target_dimensions=request_data.get(
                "target_dimensions",
                request_data.get("dimensions", request_data.get("target_dimension", 3)),
            ),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting embedded document projection {doc_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embedded-docs/{doc_name}")
async def delete_embedded_doc(doc_name: str):
    """! @brief 删除指定的嵌入文档。"""
    try:
        file_path = os.path.join("02-embedded-docs", doc_name)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_name} not found"
            )
            
        os.remove(file_path)
        return {"message": f"Document {doc_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting embedded document {doc_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    loading_method: str = Form(...),
    parsing_option: str = Form(...)
):
    """! @brief 解析上传的 PDF，但不持久化结果。
    @param file 上传的 PDF 文件。
    @param loading_method 读入后端。
    @param parsing_option 解析策略。
    @return 解析后的内容结构。
    """
    try:
        # 保存上传文件
        temp_path = os.path.join("temp", file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 准备元数据
        metadata = {
            "filename": file.filename,
            "loading_method": loading_method,
            "original_file_size": len(content),
            "processing_date": datetime.now().isoformat(),
            "parsing_method": parsing_option,
        }
        
        from services.loading_service import LoadingService
        from services.parsing_service import ParsingService

        loading_service = LoadingService()
        raw_text = loading_service.load_pdf(temp_path, loading_method)
        metadata["total_pages"] = loading_service.get_total_pages()
        
        page_map = loading_service.get_page_map()
        
        parsing_service = ParsingService()
        parsed_content = parsing_service.parse_pdf(
            raw_text, 
            parsing_option, 
            metadata,
            page_map=page_map
        )
        
        # 清理临时文件
        os.remove(temp_path)
        
        return {"parsed_content": parsed_content}
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise

@app.post("/load")
async def load_file(
    file: UploadFile = File(...),
    loading_method: str = Form(...),
    strategy: str = Form(None),
    chunking_strategy: str = Form(None),
    chunking_options: str = Form(None)
):
    """! @brief 读入上传的 PDF，并持久化页级块。
    @param file 上传的 PDF 文件。
    @param loading_method 读入后端。
    @param strategy 可选的 unstructured 读入策略。
    @param chunking_strategy 可选的 unstructured 分块策略。
    @param chunking_options JSON 编码的 unstructured 分块选项。
    @return 读入后的文档数据和持久化 JSON 路径。
    """
    try:
        # 保存上传的文件
        temp_path = os.path.join("temp", file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 准备元数据
        metadata = {
            "filename": file.filename,
            "total_chunks": 0,  # 将在后面更新
            "total_pages": 0,   # 将在后面更新
            "loading_method": loading_method,
            "loading_strategy": strategy,  
            "chunking_strategy": chunking_strategy, 
            "timestamp": datetime.now().isoformat()
        }
        
        # 如有提供则解析分块选项
        chunking_options_dict = None
        if chunking_options:
            chunking_options_dict = json.loads(chunking_options)
        
        from services.loading_service import LoadingService

        # 使用 LoadingService 加载文档
        loading_service = LoadingService()
        raw_text = loading_service.load_pdf(
            temp_path, 
            loading_method, 
            strategy=strategy,
            chunking_strategy=chunking_strategy,
            chunking_options=chunking_options_dict
        )
        
        metadata["total_pages"] = loading_service.get_total_pages()
        
        page_map = loading_service.get_page_map()
        
        # 转换成标准化的chunks格式
        chunks = []
        for idx, page in enumerate(page_map, 1):
            chunk_metadata = {
                "chunk_id": idx,
                "page_number": page["page"],
                "page_range": str(page["page"]),
                "word_count": len(page["text"].split())
            }
            if "metadata" in page:
                chunk_metadata.update(page["metadata"])
            
            chunks.append({
                "content": page["text"],
                "metadata": chunk_metadata
            })
        
        # 使用 LoadingService 保存文档，传递strategy参数
        filepath = loading_service.save_document(
            filename=file.filename,
            chunks=chunks,
            metadata=metadata,
            loading_method=loading_method,
            strategy=strategy,
            chunking_strategy=chunking_strategy,
        )
        
        # 读取保存的文档以返回
        with open(filepath, "r", encoding="utf-8") as f:
            document_data = json.load(f)
        
        # 清理临时文件
        os.remove(temp_path)
        
        return {"loaded_content": document_data, "filepath": filepath}
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise

@app.post("/load-course-qa-json")
async def load_course_qa_json(file: UploadFile = File(...)):
    """! @brief 导入课程 QA JSON，并持久化为可在 02 分块的结构化文档。
    @param file 上传的课程 QA JSON 文件。
    @return 已导入的结构化文档和相对文件路径。
    """
    try:
        raw_bytes = await file.read()
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            payload = json.loads(raw_bytes.decode("utf-8-sig"))

        document_data = _build_course_qa_loaded_document(payload, file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        storage_name = f"{_safe_storage_stem(file.filename, 'course_qa')}_course_qa_json_{timestamp}.json"
        os.makedirs("01-loaded-docs", exist_ok=True)
        filepath = os.path.join("01-loaded-docs", storage_name)

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(document_data, handle, ensure_ascii=False, indent=2)

        return {
            "loaded_content": document_data,
            "filepath": filepath,
        }
    except ValueError as e:
        logger.warning(f"课程 QA JSON 导入失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error loading course QA JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-course-knowledge-doc")
async def load_course_knowledge_doc(file: UploadFile = File(...)):
    """! @brief 导入课程外部知识 Markdown/TXT，并持久化为可分块文档。
    @param file 上传的课程知识文档，支持 .md、.markdown 和 .txt。
    @return 已导入的外部知识文档和相对文件路径。
    """
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".md", ".markdown", ".txt"}:
            raise HTTPException(status_code=400, detail="课程知识文档仅支持 .md、.markdown 或 .txt。")

        raw_bytes = await file.read()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8-sig")

        document_data = _build_course_knowledge_loaded_document(text, file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        storage_name = f"{_safe_storage_stem(file.filename, 'course_knowledge')}_course_knowledge_{timestamp}.json"
        os.makedirs("01-loaded-docs", exist_ok=True)
        filepath = os.path.join("01-loaded-docs", storage_name)

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(document_data, handle, ensure_ascii=False, indent=2)

        return {
            "loaded_content": document_data,
            "filepath": filepath,
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"课程知识文档导入失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error loading course knowledge document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/course-qa/sources")
async def list_course_qa_sources():
    """! @brief 列出前端可复现导入的课程 QA 任务文件。"""
    try:
        sources = []
        loaded_dir = "01-loaded-docs"
        if not os.path.exists(loaded_dir):
            return {"sources": []}

        for filename in os.listdir(loaded_dir):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(loaded_dir, filename)
            with open(file_path, "r", encoding="utf-8") as handle:
                doc_data = json.load(handle)
            if doc_data.get("dataset_type") != "course_qa":
                continue
            qa_items = _extract_course_qa_items(doc_data)
            sources.append({
                "id": filename,
                "name": doc_data.get("filename") or filename,
                "storage_name": filename,
                "topic_count": doc_data.get("total_pages", 0),
                "question_count": len(qa_items),
                "timestamp": doc_data.get("timestamp"),
            })

        sources.sort(key=lambda source: source.get("timestamp") or "", reverse=True)
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error listing course QA sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/course-qa/sources/{doc_name}/items")
async def get_course_qa_items(doc_name: str):
    """! @brief 读取指定课程 QA 任务文件中的题目与候选答案。
    @param doc_name 01-loaded-docs 中的课程 QA JSON 存储文件名。
    @return 无 answer_quality 的题目列表。
    """
    try:
        safe_name = Path(doc_name).name
        if not safe_name.endswith(".json"):
            safe_name = f"{safe_name}.json"
        file_path = os.path.join("01-loaded-docs", safe_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="课程 QA 任务文件不存在。")

        with open(file_path, "r", encoding="utf-8") as handle:
            doc_data = json.load(handle)
        if doc_data.get("dataset_type") != "course_qa":
            raise HTTPException(status_code=400, detail="所选文件不是课程 QA 任务文件。")

        items = _extract_course_qa_items(doc_data)
        return {
            "source": {
                "id": safe_name,
                "name": doc_data.get("filename") or safe_name,
                "topic_count": doc_data.get("total_pages", 0),
                "question_count": len(items),
                "timestamp": doc_data.get("timestamp"),
            },
            "items": items,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading course QA items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk")
async def chunk_document(data: dict = Body(...)):
    """! @brief 对已读入文档重新分块。
    @param data 请求体，包含 doc_id、chunking_option 和可选 chunk_size/chunk_overlap。
    @return 分块后的文档数据。
    """
    try:
        doc_id = data.get("doc_id")
        chunking_option = data.get("chunking_option")
        chunk_size = data.get("chunk_size", 1000)
        chunk_overlap = data.get("chunk_overlap", 0)
        
        if not doc_id or not chunking_option:
            raise HTTPException(
                status_code=400, 
                detail="Missing required parameters: doc_id and chunking_option"
            )
        
        # 读取已加载的文档
        file_path = os.path.join("01-loaded-docs", doc_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        if doc_data.get("dataset_type") == "course_qa" or doc_data.get("loading_method") == "course_qa_json":
            if chunking_option != "course_qa_items":
                raise HTTPException(status_code=400, detail="课程 QA JSON 请选择“课程 QA 条目分块”。")
            result = _chunk_course_qa_document(doc_data)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            base_name = _safe_storage_stem(doc_data.get("filename", "course_qa"), "course_qa")
            output_filename = f"{base_name}_course_qa_items_{timestamp}.json"
            output_path = os.path.join("01-chunked-docs", output_filename)
            os.makedirs("01-chunked-docs", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result
            
        # 构建页面映射
        page_map = [
            {
                'page': chunk['metadata']['page_number'],
                'text': chunk['content'],
                'metadata': {
                    key: value
                    for key, value in (chunk.get('metadata') or {}).items()
                    if key != "answer_quality"
                },
            }
            for chunk in doc_data['chunks']
        ]
            
        # 准备元数据
        metadata = {
            "filename": doc_data['filename'],
            "loading_method": doc_data['loading_method'],
            "total_pages": doc_data['total_pages'],
            "dataset_type": doc_data.get("dataset_type"),
            "source_format": doc_data.get("source_format"),
            "source_role": doc_data.get("source_role"),
        }
            
        from services.chunking_service import ChunkingService

        chunking_service = ChunkingService()
        result = chunking_service.chunk_text(
            text="",  # 不需要传递文本，因为我们使用 page_map
            method=chunking_option,
            metadata=metadata,
            page_map=page_map,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        base_name = _safe_storage_stem(doc_data.get('filename', 'document'), 'document')
        output_filename = f"{base_name}_{chunking_option}_{timestamp}.json"
        
        output_path = os.path.join("01-chunked-docs", output_filename)
        os.makedirs("01-chunked-docs", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"分块参数非法: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_search(
    file: UploadFile = File(...),
    collection_id: str = Form(...),
    top_k: int = Form(10),
    threshold: float = Form(0.7)
):
    """! @brief 使用带标签 CSV 文件评估检索质量。
    @param file CSV 文件，其中 LABEL 列包含期望命中的页码。
    @param collection_id 要搜索的向量集合。
    @param top_k 参与评分的搜索命中数量。
    @param threshold 相似度阈值。
    @return 单条查询得分和聚合平均值。
    """
    try:
        import pandas as pd
        from services.search_service import SearchService

        # 读取CSV文件
        df = pd.read_csv(file.file)
        
        # 只合并前四列的文本内容
        df['combined_text'] = df.apply(
            lambda row: ' '.join(
                str(val) for i, val in enumerate(row) 
                if i < 4 and pd.notna(val) and val != '[]'
            ), 
            axis=1
        )
        
        # 初始化SearchService
        search_service = SearchService()
        
        results = []
        total_score_hit = 0
        total_score_find = 0
        valid_queries = 0
        
        # 处理每个查询
        for _, row in df.iterrows():
            # 跳过没有标签的行
            if pd.isna(row['LABEL']) or row['LABEL'] == '[]':
                continue
                
            try:
                # 解析标签页码列表
                label_str = str(row['LABEL']).strip('[]').replace(' ', '')
                if label_str:
                    expected_pages = [int(x.strip()) for x in label_str.split(',') if x.strip()]
                else:
                    continue
                
                # 执行搜索
                search_payload = await search_service.search(
                    query=row['combined_text'],
                    collection_id=collection_id,
                    top_k=top_k,
                    threshold=threshold
                )
                search_results = (
                    search_payload.get("results", []) or []
                    if isinstance(search_payload, dict)
                    else search_payload
                )
                
                # 提取找到的页码
                found_pages = []
                for result in search_results:
                    metadata = result.get('metadata', {})
                    page = metadata.get('page', metadata.get('page_number'))
                    if page is not None:
                        found_pages.append(int(page))
                
                # 计算分数
                hits = sum(1 for page in found_pages if page in expected_pages)
                score_hit = hits / len(found_pages) if found_pages else 0
                score_find = len(set(found_pages) & set(expected_pages)) / len(expected_pages)
                
                # 添加到结果列表，包括所有top_k结果的文本
                result_entry = {
                    "query": row['combined_text'],
                    "expected_pages": expected_pages,
                    "found_pages": found_pages,
                    "score_hit": score_hit,
                    "score_find": score_find
                }
                
                # 添加每个top_k结果的文本作为单独的字段
                for i, result in enumerate(search_results, 1):
                    metadata = result.get('metadata', {})
                    result_entry[f"text_{i}"] = result.get('text', '')
                    result_entry[f"page_{i}"] = metadata.get('page', metadata.get('page_number'))
                    result_entry[f"score_{i}"] = result.get('score', 0)
                
                results.append(result_entry)
                
                total_score_hit += score_hit
                total_score_find += score_find
                valid_queries += 1
                
            except Exception as e:
                logger.warning(f"Error processing row: {str(e)}")
                continue
        
        if valid_queries == 0:
            raise ValueError("CSV 中没有可评估的有效查询")
        
        # 计算平均分数
        average_scores = {
            "score_hit": total_score_hit / valid_queries,
            "score_find": total_score_find / valid_queries
        }
        
        # 保存结果
        output_dir = Path("06-evaluation-result")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细的JSON结果
        output_path = output_dir / f"evaluation_results_{timestamp}.json"
        evaluation_results = {
            "results": results,
            "average_scores": average_scores,
            "total_queries": valid_queries,
            "parameters": {
                "collection_id": collection_id,
                "top_k": top_k,
                "threshold": threshold
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2)
            
        # 保存CSV格式的结果，每个top_k结果单独一列
        results_df = pd.DataFrame(results)
        
        # 重新排列列的顺序，使其更有逻辑性
        column_order = ['query', 'expected_pages', 'found_pages', 'score_hit', 'score_find']
        for i in range(1, top_k + 1):
            column_order.extend([f'page_{i}', f'score_{i}', f'text_{i}'])
        
        # 只选择存在的列
        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/save-search")
async def save_search_results(request: Request):
    """! @brief 持久化搜索结果，供后续生成使用。
    @param request JSON 请求，包含 query、collection_id 和 results。
    @return 持久化后的搜索结果路径。
    """
    try:
        data = await request.json()
        query = data.get("query")
        collection_id = data.get("collection_id")
        results = data.get("results")
        
        if not all([query, collection_id, results]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from services.search_service import SearchService

        # 直接创建 SearchService 实例
        search_service = SearchService()
        filepath = search_service.save_search_results(query, collection_id, results)
        return {"saved_filepath": filepath}
        
    except Exception as e:
        logger.error(f"Error saving search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generation/models")
async def get_generation_models():
    """! @brief 获取可用的生成模型列表."""
    try:
        from services.generation_service import GenerationService

        generation_service = GenerationService()
        models = generation_service.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting generation models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_response(
    query: str = Body(...),
    provider: str = Body(...),
    model_name: str = Body(...),
    search_results: List[Dict] = Body(...),
    load_model: bool = Body(...),
    api_key: Optional[str] = Body(None),
    rag_mode: str = Body("basic_rag"),
):
    """! @brief 基于查询和检索上下文生成最终回答。
    @return 生成回答和持久化结果路径。
    """
    try:
        from services.generation_service import GenerationService

        generation_service = GenerationService()
        result = generation_service.generate(
            provider=provider,
            model_name=model_name,
            query=query,
            search_results=search_results,
            load_model=load_model,
            api_key=api_key,
            rag_mode=rag_mode,
        )
        return result
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-results")
async def list_search_results():
    """! @brief 获取所有搜索结果文件列表."""
    try:
        search_results_dir = "04-search-results"
        if not os.path.exists(search_results_dir):
            return {"files": []}
            
        files = []
        for filename in os.listdir(search_results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(search_results_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    files.append({
                        "id": filename,
                        "name": f"Search: {data.get('query', 'Unknown')} ({filename})",
                        "timestamp": data.get('timestamp', '')
                    })
                    
        # 按时间戳排序，最新的在前面
        files.sort(key=lambda x: x['timestamp'], reverse=True)
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Error listing search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-results/{file_id}")
async def get_search_result(file_id: str):
    """! @brief 获取特定搜索结果文件的内容."""
    try:
        file_path = os.path.join("04-search-results", file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Search result file not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        logger.error(f"Error reading search result file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
