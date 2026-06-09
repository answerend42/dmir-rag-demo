import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def main_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    return module


@pytest.fixture()
def client(main_module):
    return TestClient(main_module.app)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def loaded_doc(filename="doc.pdf"):
    return {
        "filename": filename,
        "total_chunks": 1,
        "total_pages": 1,
        "loading_method": "pymupdf",
        "chunking_method": "loaded",
        "timestamp": "2026-01-01T00:00:00",
        "chunks": [
            {
                "content": "first page content",
                "metadata": {"chunk_id": 1, "page_number": 1, "page_range": "1", "word_count": 3},
            }
        ],
    }


def chunked_doc(filename="doc.pdf"):
    data = loaded_doc(filename)
    data["chunking_method"] = "by_pages"
    return data


def embedded_doc():
    return {
        "filename": "doc.pdf",
        "document_name": "doc",
        "embedding_model": "unit-model",
        "embedding_provider": "huggingface",
        "created_at": "2026-01-01T00:00:00",
        "vector_dimension": 3,
        "embeddings": [
            {
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "content": "first page content",
                    "page_number": 1,
                    "page_range": "1",
                    "embedding_timestamp": "2026-01-01T00:00:00",
                },
            }
        ],
    }


class StubLoadingService:
    def __init__(self):
        self.page_map = [{"page": 1, "text": "loaded text", "metadata": {"source": "unit"}}]

    def load_pdf(self, *args, **kwargs):
        return "loaded text"

    def get_total_pages(self):
        return 1

    def get_page_map(self):
        return self.page_map

    def save_document(self, filename, chunks, metadata, loading_method, strategy=None, chunking_strategy=None):
        data = {
            "filename": filename,
            "total_chunks": len(chunks),
            "total_pages": metadata.get("total_pages", 1),
            "loading_method": loading_method,
            "loading_strategy": strategy,
            "chunking_strategy": chunking_strategy,
            "chunking_method": "loaded",
            "timestamp": "2026-01-01T00:00:00",
            "chunks": chunks,
        }
        path = Path("01-loaded-docs") / "loaded_from_endpoint.json"
        write_json(path, data)
        return str(path)


class StubChunkingService:
    def chunk_text(self, text, method, metadata, page_map=None, chunk_size=1000, chunk_overlap=0):
        return {
            "filename": metadata.get("filename", "doc.pdf"),
            "total_chunks": len(page_map or [1]),
            "total_pages": len(page_map or [1]),
            "loading_method": metadata.get("loading_method", "pymupdf"),
            "chunking_method": method,
            "chunk_size": chunk_size if method == "fixed_size" else None,
            "chunk_overlap": chunk_overlap if method == "fixed_size" else 0,
            "timestamp": "2026-01-01T00:00:00",
            "chunks": [
                {
                    "content": (page_map or [{"text": "chunk"}])[0].get("text", "chunk"),
                    "metadata": {"chunk_id": 1, "page_number": 1, "page_range": "1", "word_count": 1},
                }
            ],
        }


class StubParsingService:
    def parse_pdf(self, text, method, metadata, page_map=None):
        return {
            "metadata": {"filename": metadata["filename"], "total_pages": 1, "parsing_method": method},
            "content": [{"type": "Text", "content": text, "page": 1}],
        }


class StubMinerUPrecisionParser:
    def parse_file(self, file_path, file_name):
        return {
            "metadata": {
                "filename": file_name,
                "total_pages": 2,
                "parsing_method": "mineru_vlm",
                "source": "MinerU VLM 精准解析 API",
                "mineru_batch_id": "batch-1",
                "mineru_model_version": "vlm",
                "mineru_full_zip_url": "https://example.test/full.zip",
            },
            "content": [
                {
                    "type": "Markdown",
                    "title": "章节一",
                    "content": "精准解析内容",
                    "page": None,
                }
            ],
        }


class StubEmbeddingService:
    def create_embeddings(self, input_data, config):
        return [
            {
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "chunk_id": 1,
                    "total_chunks": 1,
                    "content": "content",
                    "page_number": 1,
                    "page_range": "1",
                    "embedding_provider": config.provider,
                    "embedding_model": config.model_name,
                    "embedding_timestamp": "2026-01-01T00:00:00",
                    "vector_dimension": 3,
                },
            }
        ], {}

    def save_embeddings(self, doc_id, embeddings):
        path = Path("02-embedded-docs") / "embedded_from_endpoint.json"
        write_json(path, {"embeddings": embeddings, "embedding_provider": "huggingface", "embedding_model": "unit", "created_at": "now", "vector_dimension": 3})
        return str(path)


class StubSearchService:
    last_search_kwargs = {}

    def get_providers(self):
        return [{"id": "chroma", "name": "chroma"}]

    def list_collections(self, provider):
        return [{"id": "collection", "name": "collection", "count": 1}]

    async def search(self, **kwargs):
        StubSearchService.last_search_kwargs = kwargs
        return {
            "results": [
                {
                    "text": "hit text",
                    "score": 0.9,
                    "metadata": {"source": "doc.pdf", "page": 1, "chunk": 1},
                }
            ]
        }

    def save_search_results(self, query, collection_id, results):
        path = Path("04-search-results") / "saved_search.json"
        write_json(path, {"query": query, "collection_id": collection_id, "timestamp": "2026-01-01T00:00:00", "results": results})
        return str(path)


class StubVectorStoreService:
    def index_embeddings(self, embedding_file, config):
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": 1,
            "index_size": 1,
            "collection_name": "collection",
        }

    def list_collections(self, provider):
        return ["collection"]

    def get_collection_info(self, provider, collection_name):
        return {"name": collection_name, "num_entities": 1, "schema": {"field": "value"}}

    def delete_collection(self, provider, collection_name):
        return collection_name != "fail"


def test_process_parse_load_save_and_list_routes(client, main_module, monkeypatch):
    import services.chunking_service as chunking_module
    import services.loading_service as loading_module
    import services.mineru_service as mineru_module
    import services.parsing_service as parsing_module

    monkeypatch.setattr(loading_module, "LoadingService", StubLoadingService)
    monkeypatch.setattr(chunking_module, "ChunkingService", StubChunkingService)
    monkeypatch.setattr(parsing_module, "ParsingService", StubParsingService)
    monkeypatch.setattr(mineru_module, "MinerUPrecisionParser", StubMinerUPrecisionParser)

    upload = {"file": ("doc.pdf", b"%PDF unit", "application/pdf")}
    response = client.post("/process", files=upload, data={"loading_method": "pymupdf", "chunking_option": "fixed_size", "chunk_size": "100", "chunk_overlap": "20"})
    assert response.status_code == 200
    assert response.json()["chunks"]["chunking_method"] == "fixed_size"
    assert response.json()["chunks"]["chunk_overlap"] == 20

    response = client.post("/parse", files=upload, data={"loading_method": "pymupdf", "parsing_option": "all_text"})
    assert response.status_code == 200
    assert response.json()["parsed_content"]["content"][0]["type"] == "Text"

    response = client.post("/load", files=upload, data={"loading_method": "unstructured", "strategy": "fast", "chunking_strategy": "basic", "chunking_options": "{}"})
    assert response.status_code == 200
    assert response.json()["loaded_content"]["filename"] == "doc.pdf"

    response = client.post(
        "/load",
        files={"file": ("mineru.pdf", b"%PDF mineru", "application/pdf")},
        data={"loading_method": "mineru_vlm"},
    )
    assert response.status_code == 200
    mineru_doc = response.json()["loaded_content"]
    assert mineru_doc["loading_method"] == "mineru_vlm"
    assert mineru_doc["chunking_method"] == "mineru_markdown_sections"
    assert mineru_doc["mineru_batch_id"] == "batch-1"
    assert mineru_doc["chunks"][0]["metadata"]["section_title"] == "章节一"

    response = client.post(
        "/load-course-knowledge-doc",
        files={
            "file": (
                "course_knowledge.md",
                "# NLP\n\n自然语言处理研究计算机如何处理自然语言。\n\n# 数据结构\n\n哈希表用于快速查找。".encode("utf-8"),
                "text/markdown",
            )
        },
    )
    assert response.status_code == 200
    knowledge_doc = response.json()["loaded_content"]
    assert knowledge_doc["dataset_type"] == "course_knowledge"
    assert knowledge_doc["source_role"] == "external_knowledge"
    assert knowledge_doc["chunks"][0]["metadata"]["section_title"] == "NLP"
    assert "answer_quality" not in json.dumps(response.json(), ensure_ascii=False)

    course_qa_payload = {
        "自然语言处理课程知识问答": [
            {
                "id": 1,
                "question": "什么是自然语言处理？",
                "answers": [
                    {"answer_quality": 0, "answer": "处理文字。"},
                    {"answer_quality": 9, "answer": "自然语言处理研究如何让计算机理解和生成自然语言。"},
                ],
            }
        ]
    }
    response = client.post(
        "/load-course-qa-json",
        files={"file": ("course_qa.json", json.dumps(course_qa_payload).encode("utf-8"), "application/json")},
    )
    assert response.status_code == 200
    loaded = response.json()["loaded_content"]
    assert loaded["dataset_type"] == "course_qa"
    assert loaded["total_chunks"] == 1
    assert loaded["qa_items"][0]["answers"][0]["answer"] == "处理文字。"
    assert "answer_quality" not in json.dumps(response.json(), ensure_ascii=False)

    loaded_name = Path(response.json()["filepath"]).name
    response = client.get("/course-qa/sources")
    assert response.status_code == 200
    assert response.json()["sources"][0]["id"] == loaded_name
    assert response.json()["sources"][0]["question_count"] == 1

    response = client.get(f"/course-qa/sources/{loaded_name}/items")
    assert response.status_code == 200
    items_payload = response.json()
    assert items_payload["items"][0]["question"] == "什么是自然语言处理？"
    assert items_payload["items"][0]["answers"][1]["answer"].startswith("自然语言处理研究")
    assert "answer_quality" not in json.dumps(items_payload, ensure_ascii=False)

    response = client.post("/chunk", json={"doc_id": loaded_name, "chunking_option": "course_qa_items"})
    assert response.status_code == 200
    assert response.json()["chunking_method"] == "course_qa_items"
    assert response.json()["chunks"][0]["metadata"]["topic"] == "自然语言处理课程知识问答"
    assert "answer_quality" not in json.dumps(response.json(), ensure_ascii=False)

    response = client.post("/save", json={"docName": "saved_doc", "chunks": [{"content": "x"}], "metadata": {"m": 1}})
    assert response.status_code == 200
    response = client.get("/list-docs")
    assert response.status_code == 200
    assert any(document["name"] == "saved_doc" for document in response.json()["documents"])

    assert client.post("/save", json={}).status_code == 500


def test_empty_directory_routes(client):
    embedded_dir = Path("02-embedded-docs")
    if embedded_dir.exists():
        for item in embedded_dir.iterdir():
            item.unlink()
        embedded_dir.rmdir()

    assert client.get("/list-embedded").json() == {"documents": []}
    assert client.get("/search-results").json() == {"files": []}


def test_document_chunk_embedding_and_index_routes(client, main_module, monkeypatch):
    import services.chunking_service as chunking_module
    import services.embedding_service as embedding_module
    import services.vector_store_service as vector_module

    monkeypatch.setattr(chunking_module, "ChunkingService", StubChunkingService)
    monkeypatch.setattr(embedding_module, "EmbeddingService", StubEmbeddingService)
    monkeypatch.setattr(vector_module, "VectorStoreService", StubVectorStoreService)

    write_json(Path("01-loaded-docs/loaded.json"), loaded_doc())
    write_json(Path("01-chunked-docs/chunked.json"), chunked_doc())
    write_json(Path("02-embedded-docs/embedded.json"), embedded_doc())

    all_docs = client.get("/documents?type=all")
    assert all_docs.status_code == 200
    assert len(all_docs.json()["documents"]) == 2
    assert client.get("/documents/loaded.json?type=loaded").json()["filename"] == "doc.pdf"
    assert client.get("/documents/missing.json?type=loaded").status_code == 404

    chunk_response = client.post("/chunk", json={"doc_id": "loaded.json", "chunking_option": "fixed_size", "chunk_size": 1000, "chunk_overlap": 100})
    assert chunk_response.status_code == 200
    assert chunk_response.json()["chunking_method"] == "fixed_size"
    assert chunk_response.json()["chunk_overlap"] == 100
    assert client.post("/chunk", json={}).status_code == 400

    embed_response = client.post("/embed", json={"documentId": "chunked.json", "provider": "huggingface", "model": "unit"})
    assert embed_response.status_code == 200
    assert embed_response.json()["status"] == "success"
    assert client.post("/embed", json={}).status_code == 500

    embedded = client.get("/list-embedded")
    assert embedded.status_code == 200
    assert embedded.json()["documents"]

    detail = client.get("/embedded-docs/embedded.json")
    assert detail.status_code == 200
    assert detail.json()["embeddings"][0]["metadata"]["document_name"] == "doc"
    assert client.get("/embedded-docs/missing.json").status_code == 404

    index_response = client.post("/index", json={"fileId": "embedded.json", "vectorDb": "chroma", "indexMode": "hnsw"})
    assert index_response.status_code == 200
    assert index_response.json()["collection_name"] == "collection"
    assert client.post("/index", json={}).status_code == 500

    assert client.delete("/documents/chunked.json?type=chunked").status_code == 200
    assert client.delete("/documents/missing.json?type=chunked").status_code == 404
    assert client.delete("/embedded-docs/embedded.json").status_code == 200
    assert client.delete("/embedded-docs/missing.json").status_code == 500


def test_search_collection_evaluation_generation_and_result_routes(client, main_module, monkeypatch):
    import services.generation_service as generation_module
    import services.search_service as search_module
    import services.vector_store_service as vector_module

    class StubGenerationService:
        def get_available_models(self):
            return {"deepseek": {"deepseek-v3": "deepseek-v3"}}

        def generate(self, **kwargs):
            return {"response": "answer", "saved_filepath": "05-generation-results/out.json"}

    monkeypatch.setattr(search_module, "SearchService", StubSearchService)
    monkeypatch.setattr(vector_module, "VectorStoreService", StubVectorStoreService)
    monkeypatch.setattr(generation_module, "GenerationService", StubGenerationService)

    assert client.get("/providers").json()["providers"][0]["id"] == "chroma"
    assert client.get("/collections?provider=chroma").json()["collections"][0]["id"] == "collection"
    assert client.get("/collections/chroma").json()["collections"][0]["id"] == "collection"
    assert client.get("/collections/chroma/collection").json()["num_entities"] == 1
    assert client.delete("/collections/chroma/collection").status_code == 200
    assert client.delete("/collections/chroma/fail").status_code == 500

    search = client.post("/search", json={"query": "q", "collection_id": "collection", "top_k": 1, "threshold": 0.1, "word_count_threshold": 0, "save_results": True})
    assert search.status_code == 200
    assert search.json()["results"]["results"][0]["text"] == "hit text"
    assert StubSearchService.last_search_kwargs["save_results"] is True

    saved = client.post("/save-search", json={"query": "q", "collection_id": "collection", "results": [{"text": "hit"}]})
    assert saved.status_code == 200
    assert client.post("/save-search", json={}).status_code == 500

    csv = "A,B,C,D,LABEL\nfoo,bar,baz,qux,[1]\nempty,,,,[]\n"
    evaluation = client.post("/evaluate", files={"file": ("eval.csv", csv.encode(), "text/csv")}, data={"collection_id": "collection", "top_k": "1", "threshold": "0.1"})
    assert evaluation.status_code == 200
    assert evaluation.json()["average_scores"]["score_find"] == 1

    assert client.get("/generation/models").status_code == 200
    generated = client.post("/generate", json={"query": "q", "provider": "deepseek", "model_name": "deepseek-v3", "search_results": [], "load_model": False})
    assert generated.status_code == 200
    assert generated.json()["response"] == "answer"

    write_json(Path("04-search-results/result.json"), {"query": "q", "timestamp": "2026-01-01T00:00:00", "results": [{"text": "hit"}]})
    listed = client.get("/search-results")
    assert listed.status_code == 200
    assert listed.json()["files"][0]["id"] == "result.json"
    assert client.get("/search-results/result.json").json()["query"] == "q"
    assert client.get("/search-results/missing.json").status_code == 500
