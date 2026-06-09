import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest


def sample_page_map():
    return [
        {"page": 1, "text": "TITLE\nFirst page text.\nA|B"},
        {"page": 2, "text": "SECOND\nSecond page text.\nOne two three four"},
    ]


def sample_embeddings():
    return {
        "filename": "Doc Name.pdf",
        "embedding_provider": "huggingface",
        "embedding_model": "unit-model",
        "vector_dimension": 3,
        "embeddings": [
            {
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "content": "alpha",
                    "chunk_id": 1,
                    "total_chunks": 1,
                    "word_count": 5,
                    "page_number": 1,
                    "page_range": "1",
                    "embedding_timestamp": "now",
                },
            }
        ],
    }


def test_chunking_service_methods_and_errors():
    from services.chunking_service import ChunkingService

    service = ChunkingService()
    metadata = {"filename": "a.pdf", "loading_method": "pymupdf"}

    by_pages = service.chunk_text("", "by_pages", metadata, sample_page_map())
    assert by_pages["total_pages"] == 2
    assert by_pages["total_chunks"] == 2
    assert by_pages["chunks"][0]["metadata"]["page_range"] == "1"

    fixed = service.chunk_text("", "fixed_size", metadata, sample_page_map(), chunk_size=8)
    assert fixed["chunking_method"] == "fixed_size"
    assert fixed["total_chunks"] >= 4
    overlapped = service.chunk_text(
        "",
        "fixed_size",
        metadata,
        [{"page": 1, "text": "alpha beta gamma delta epsilon"}],
        chunk_size=11,
        chunk_overlap=10,
    )
    assert overlapped["chunk_overlap"] == 10
    assert overlapped["chunks"][0]["content"].endswith("beta")
    assert overlapped["chunks"][1]["content"].startswith("beta")

    paragraphs = service.chunk_text("", "by_paragraphs", metadata, [{"page": 1, "text": "A\n\nB"}])
    assert [c["content"] for c in paragraphs["chunks"]] == ["A", "B"]

    sentences = service.chunk_text("", "by_sentences", metadata, sample_page_map())
    assert sentences["chunks"]

    with pytest.raises(ValueError):
        service.chunk_text("", "unknown", metadata, sample_page_map())
    with pytest.raises(ValueError):
        service.chunk_text("", "by_pages", metadata, None)
    with pytest.raises(ValueError):
        service.chunk_text("", "fixed_size", metadata, sample_page_map(), chunk_size=100, chunk_overlap=100)


def test_parsing_service_methods_and_errors():
    from services.parsing_service import ParsingService

    service = ParsingService()
    metadata = {"filename": "a.pdf"}
    page_map = sample_page_map()

    assert service.parse_pdf("", "all_text", metadata, page_map)["content"][0]["type"] == "Text"
    assert service.parse_pdf("", "by_pages", metadata, page_map)["content"][1]["type"] == "Page"
    titles = service.parse_pdf("", "by_titles", metadata, page_map)
    assert titles["content"][0]["title"] == "TITLE"
    tables = service.parse_pdf("", "text_and_tables", metadata, page_map)
    assert tables["content"][0]["type"] == "table"

    with pytest.raises(ValueError):
        service.parse_pdf("", "bad", metadata, page_map)
    with pytest.raises(ValueError):
        service.parse_pdf("", "all_text", metadata, None)


def make_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF text")
    doc.save(path)
    doc.close()


def test_loading_service_pdf_methods_save_and_unstructured(tmp_path, monkeypatch):
    from services.loading_service import LoadingService
    import services.loading_service as loading_module

    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf)

    service = LoadingService()
    assert "Hello PDF text" in service.load_pdf(str(pdf), "pymupdf")
    assert service.get_total_pages() == 1
    assert service.get_page_map()[0]["page"] == 1

    service = LoadingService()
    assert "Hello PDF text" in service.load_pdf(str(pdf), "pypdf")

    service = LoadingService()
    assert "Hello PDF text" in service.load_pdf(str(pdf), "pdfplumber")

    class StubMetadata:
        def __init__(self):
            self.page_number = 3
            self.extra = {"nested": "value"}
            self._known_field_names = ["skip-me"]
            self.not_json = object()

    class StubElement:
        category = "NarrativeText"
        id = "element-1"
        metadata = StubMetadata()

        def __str__(self):
            return "Unstructured text"

    monkeypatch.setattr(loading_module, "partition_pdf", lambda *args, **kwargs: [StubElement()])
    service = LoadingService()
    assert service.load_pdf(str(pdf), "unstructured", strategy="fast", chunking_strategy="basic", chunking_options={}) == "Unstructured text"
    assert service.get_total_pages() == 3
    assert service.get_page_map()[0]["metadata"]["element_type"] == "StubElement"
    assert "not_json" in service.get_page_map()[0]["metadata"]

    service = LoadingService()
    assert service.load_pdf(
        str(pdf),
        "unstructured",
        strategy="unknown",
        chunking_strategy="by_title",
        chunking_options={"combineTextUnderNChars": 10, "multiPageSections": True},
    ) == "Unstructured text"

    monkeypatch.chdir(tmp_path)
    saved = service.save_document(
        filename="doc.pdf",
        chunks=[{"content": "x", "metadata": {"chunk_id": 1}}],
        metadata={"total_pages": 1},
        loading_method="unstructured",
        strategy="fast",
        chunking_strategy="basic",
    )
    saved_data = json.loads(Path(saved).read_text(encoding="utf-8"))
    assert saved_data["loading_strategy"] == "fast"
    saved_regular = service.save_document(
        filename="regular.pdf",
        chunks=[{"content": "x", "metadata": {"chunk_id": 1}}],
        metadata={"total_pages": 1},
        loading_method="pymupdf",
    )
    assert json.loads(Path(saved_regular).read_text(encoding="utf-8"))["loading_strategy"] is None

    with pytest.raises(ValueError):
        service.load_pdf(str(pdf), "bad")


class StubEmbeddingFunction:
    def embed_documents(self, texts):
        return [[float(i), float(i + 1)] for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def test_embedding_service_create_save_factory_and_config(tmp_path, monkeypatch):
    from services.embedding_service import EmbeddingConfig, EmbeddingFactory, EmbeddingService
    import services.embedding_service as embedding_module

    original_create_embedding_function = EmbeddingFactory.create_embedding_function
    monkeypatch.setattr(EmbeddingFactory, "create_embedding_function", staticmethod(lambda config: StubEmbeddingFunction()))
    service = EmbeddingService()
    input_data = {
        "chunks": [
            {"content": "one", "metadata": {"chunk_id": 1, "page_number": 1, "page_range": "1", "word_count": 1}},
            {"content": "two", "metadata": {"chunk_id": 2, "page_number": 2, "page_range": "2", "word_count": 1}},
        ],
        "metadata": {"filename": "doc.pdf"},
    }
    openai_embeddings, _ = service.create_embeddings(input_data, EmbeddingConfig("openai", "text-embedding-3-small"))
    assert len(openai_embeddings) == 2
    assert openai_embeddings[0]["metadata"]["filename"] == "doc.pdf"

    hf_embeddings, _ = service.create_embeddings(input_data, EmbeddingConfig("huggingface", "unit"))
    assert hf_embeddings[0]["embedding"] == [0.1, 0.2, 0.3]
    assert service.create_single_embedding("hello", "huggingface", "unit") == [0.1, 0.2, 0.3]

    qwen_embeddings, _ = service.create_embeddings(input_data, EmbeddingConfig("qwen_api", "text-embedding-v2"))
    assert qwen_embeddings[1]["embedding"] == [1.0, 2.0]
    assert qwen_embeddings[0]["metadata"]["embedding_provider"] == "qwen_api"

    monkeypatch.chdir(tmp_path)
    saved = service.save_embeddings("doc_by_pages_1.json", hf_embeddings)
    saved_data = json.loads(Path(saved).read_text(encoding="utf-8"))
    assert saved_data["embedding_provider"] == "huggingface"
    write_config = Path("02-embedded-docs") / "manual_config.json"
    write_config.write_text(
        json.dumps({"filename": "doc", "embedding_provider": "huggingface", "embedding_model": "unit"}),
        encoding="utf-8",
    )

    config = service.get_document_embedding_config("doc_collection")
    assert config.provider == "huggingface"
    with pytest.raises(ValueError):
        service.get_document_embedding_config("missing_collection")

    class StubOpenAIEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubBedrockEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubHFEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubQwenApiEmbedder:
        def __init__(self, model):
            self.model = model

        def embed_batch(self, texts):
            return [SimpleNamespace(vector=[float(i), float(i + 1)]) for i, _ in enumerate(texts)]

        def embed_query(self, text):
            return SimpleNamespace(vector=[0.4, 0.5, 0.6])

    monkeypatch.setattr(embedding_module, "OpenAIEmbeddings", StubOpenAIEmbeddings)
    monkeypatch.setattr(embedding_module, "BedrockEmbeddings", StubBedrockEmbeddings)
    monkeypatch.setattr(embedding_module, "HuggingFaceEmbeddings", StubHFEmbeddings)
    monkeypatch.setattr(embedding_module, "QwenApiEmbedder", StubQwenApiEmbedder)
    monkeypatch.setattr(embedding_module.boto3, "client", lambda **kwargs: "bedrock-client")
    monkeypatch.setattr(embedding_module, "get_huggingface_model_path", lambda name: f"local/{name}")
    monkeypatch.setattr(EmbeddingFactory, "create_embedding_function", staticmethod(original_create_embedding_function))

    assert isinstance(EmbeddingFactory.create_embedding_function(EmbeddingConfig("openai", "m")), StubOpenAIEmbeddings)
    assert isinstance(EmbeddingFactory.create_embedding_function(EmbeddingConfig("bedrock", "m")), StubBedrockEmbeddings)
    assert isinstance(EmbeddingFactory.create_embedding_function(EmbeddingConfig("huggingface", "m")), StubHFEmbeddings)
    qwen_function = EmbeddingFactory.create_embedding_function(EmbeddingConfig("qwen_api", "m"))
    assert qwen_function.embed_documents(["a", "b"]) == [[0.0, 1.0], [1.0, 2.0]]
    assert qwen_function.embed_query("query") == [0.4, 0.5, 0.6]
    with pytest.raises(ValueError):
        EmbeddingFactory.create_embedding_function(EmbeddingConfig("bad", "m"))


def test_model_utils_local_and_remote(tmp_path, monkeypatch):
    from utils.model_utils import get_huggingface_model_path

    monkeypatch.delenv("HF_MODEL_PATH", raising=False)
    assert get_huggingface_model_path("org/model") == "org/model"

    monkeypatch.setenv("HF_MODEL_PATH", str(tmp_path))
    assert get_huggingface_model_path("org/model") == "org/model"
    local = tmp_path / "org" / "model"
    local.mkdir(parents=True)
    assert get_huggingface_model_path("org/model") == str(local)


class StubChromaCollection:
    def __init__(self, name="collection"):
        self.name = name
        self.added = []
        self.created_kwargs = {}

    def add(self, **kwargs):
        self.added.append(kwargs)

    def count(self):
        return 1

    def query(self, **kwargs):
        assert "query_texts" not in kwargs
        return {
            "ids": [["1", "2"]],
            "distances": [[0.1, 0.8]],
            "documents": [["good hit", "bad hit"]],
            "metadatas": [[
                {
                    "document_name": "doc.pdf",
                    "page_number": "1",
                    "total_chunks": 2,
                    "page_range": "1",
                    "embedding_provider": "huggingface",
                    "embedding_model": "unit",
                    "embedding_timestamp": "now",
                    "word_count": 10,
                },
                {"word_count": 1},
            ]],
        }

    def get(self, **kwargs):
        return {"metadatas": [{"embedding_provider": "huggingface", "embedding_model": "unit"}]}


class StubChromaClient:
    def __init__(self):
        self.collection = StubChromaCollection("collection")
        self.deleted = []

    def list_collections(self):
        return [SimpleNamespace(name="collection")]

    def get_or_create_collection(self, name, **kwargs):
        self.collection.name = name
        self.collection.created_kwargs = kwargs
        return self.collection

    def get_collection(self, name):
        self.collection.name = name
        return self.collection

    def delete_collection(self, name=None, **kwargs):
        self.deleted.append(name or kwargs.get("name"))


def test_vector_store_chroma_helpers_and_milvus_paths(tmp_path, monkeypatch):
    import services.vector_store_service as vector_module
    from services.vector_store_service import VectorDBConfig, VectorStoreService, clean_filename
    from utils.config import VectorDBProvider

    assert clean_filename("") == "default_filename"
    assert clean_filename(" 9 bad name! ") == "file_9_bad_name_file"
    config = VectorDBConfig("chroma", "hnsw")
    assert config._get_chroma_index_type() == "hnsw"
    assert config._get_milvus_index_type("flat") == "FLAT"
    assert config._get_milvus_index_params("hnsw")["M"] == 16

    unit_client = StubChromaClient()
    monkeypatch.setattr(vector_module.chromadb, "PersistentClient", lambda path: unit_client)
    monkeypatch.setattr(
        vector_module.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: f"embedding:{model_name}",
    )
    monkeypatch.chdir(tmp_path)
    service = VectorStoreService()

    data_path = tmp_path / "embeddings.json"
    data_path.write_text(json.dumps(sample_embeddings()), encoding="utf-8")
    assert service._load_embeddings(str(data_path))["filename"] == "Doc Name.pdf"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        service._load_embeddings(str(bad_path))

    chroma_result = service.index_embeddings(str(data_path), VectorDBConfig("chroma", "hnsw"))
    assert chroma_result["index_size"] == 1
    assert chroma_result["index_family"] == "hnsw"
    assert unit_client.collection.added
    assert "embedding_function" not in unit_client.collection.created_kwargs
    assert "vector" not in unit_client.collection.added[0]["metadatas"][0]
    assert unit_client.collection.added[0]["metadatas"][0]["vector_dimension"] == 3

    for mode in ["flat", "ivf", "lsh"]:
        faiss_result = service.index_embeddings(str(data_path), VectorDBConfig("faiss", mode))
        assert faiss_result["database"] == "faiss"
        assert faiss_result["index_family"] == "faiss"
        assert faiss_result["index_mode"] == mode
        assert faiss_result["index_size"] == 1
        faiss_info = service.get_collection_info(VectorDBProvider.FAISS, faiss_result["collection_name"])
        assert faiss_info["schema"]["index_mode"] == mode
        assert service.delete_collection(VectorDBProvider.FAISS, faiss_result["collection_name"]) is True

    assert service.list_collections(VectorDBProvider.CHROMA)[0].name == "collection"
    assert service.delete_collection(VectorDBProvider.CHROMA, "collection") is True
    assert service.get_collection_info(VectorDBProvider.CHROMA, "collection")["num_entities"] == 1
    assert service.list_collections("unknown") == []
    assert service.delete_collection("unknown", "collection") is False
    assert service.get_collection_info("unknown", "collection") == {}

    class StubMilvusClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_collection(self, **kwargs):
            return "created"

        def insert(self, **kwargs):
            return {"ids": [1]}

        def prepare_index_params(self):
            return SimpleNamespace(add_index=lambda **kwargs: None)

        def create_index(self, **kwargs):
            return None

        def load_collection(self, **kwargs):
            return None

    class StubFieldSchema:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubCollectionSchema:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    unit_connections = SimpleNamespace(connect=lambda **kwargs: None, disconnect=lambda *args, **kwargs: None)
    unit_utility = SimpleNamespace(list_collections=lambda: ["milvus_collection"], drop_collection=lambda name: None)
    unit_collection = lambda name: SimpleNamespace(num_entities=7, schema=SimpleNamespace(to_dict=lambda: {"name": name}))
    unit_datatype = SimpleNamespace(INT64="INT64", VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR")
    monkeypatch.setattr(vector_module, "MilvusClient", StubMilvusClient)
    monkeypatch.setattr(vector_module, "FieldSchema", StubFieldSchema)
    monkeypatch.setattr(vector_module, "CollectionSchema", StubCollectionSchema)
    monkeypatch.setattr(vector_module, "DataType", unit_datatype)
    monkeypatch.setattr(vector_module, "connections", unit_connections)
    monkeypatch.setattr(vector_module, "utility", unit_utility)
    monkeypatch.setattr(vector_module, "Collection", unit_collection)

    milvus_result = service.index_embeddings(str(data_path), VectorDBConfig("milvus", "flat"))
    assert milvus_result["index_size"] == 1
    assert service.list_collections(VectorDBProvider.MILVUS) == ["milvus_collection"]
    assert service.delete_collection(VectorDBProvider.MILVUS, "milvus_collection") is True
    assert service.get_collection_info(VectorDBProvider.MILVUS, "milvus_collection")["num_entities"] == 7


@pytest.mark.asyncio
async def test_search_service_collections_search_and_save(tmp_path, monkeypatch):
    import services.search_service as search_module
    from services.search_service import SearchService

    unit_client = StubChromaClient()

    class StubEmbeddingService:
        def create_single_embedding(self, text, provider, model):
            return [0.1, 0.2, 0.3]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(search_module.chromadb, "PersistentClient", lambda path: unit_client)
    monkeypatch.setattr(search_module, "EmbeddingService", StubEmbeddingService)
    monkeypatch.setattr(search_module.connections, "disconnect", lambda *args, **kwargs: None)

    service = SearchService()
    assert service.get_providers() == [
        {"id": "chroma", "name": "Chroma"},
        {"id": "faiss", "name": "FAISS"},
    ]
    assert service.list_collections() == [
        {
            "id": "collection",
            "name": "collection",
            "count": 1,
            "database": "chroma",
            "index_mode": "hnsw",
            "index_family": "hnsw",
            "dataset_type": None,
            "source_role": None,
            "document_name": None,
            "embedding_provider": "huggingface",
            "embedding_model": "unit",
        }
    ]

    saved_path = service.save_search_results("query", "collection", [{"text": "x"}])
    assert Path(saved_path).exists()

    result = await service.search("query", "collection", top_k=2, threshold=0.5, save_results=True)
    assert len(result["results"]) == 1
    assert result["saved_filepath"].endswith(".json")
    assert result["score_algorithm"]["name"] == "Chroma HNSW cosine"

    filtered_by_words = await service.search("query", "collection", top_k=2, threshold=0.5, word_count_threshold=20)
    assert filtered_by_words["results"] == []
    assert filtered_by_words["index_family"] == "hnsw"

    no_hits = await service.search("query", "collection", top_k=2, threshold=0.95, save_results=True)
    assert no_hits["results"] == []

    service.faiss_index_service.build_index(sample_embeddings(), "faiss_flat", "flat")
    assert service.list_collections("faiss")[0]["name"] == "faiss_flat"
    faiss_result = await service.search("query", "faiss_flat", top_k=1, threshold=-1, include_query_embedding=True)
    assert len(faiss_result["results"]) == 1
    assert faiss_result["index_family"] == "faiss"
    assert faiss_result["score_algorithm"]["name"] == "FAISS Flat cosine"
    assert faiss_result["query_embedding_metadata"]["collection_id"] == "faiss_flat"
    assert service.faiss_index_service.delete_index("faiss_flat") is True

    service.client = SimpleNamespace(
        list_collections=lambda: [SimpleNamespace(name="bad")],
        get_or_create_collection=lambda name: (_ for _ in ()).throw(RuntimeError("collection info failed")),
    )
    assert service.list_collections() == []
    service.client = SimpleNamespace(list_collections=lambda: (_ for _ in ()).throw(RuntimeError("list failed")))
    with pytest.raises(RuntimeError):
        service.list_collections()

    service.client = unit_client
    monkeypatch.setattr(service, "save_search_results", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("save failed")))
    with pytest.raises(RuntimeError):
        await service.search("query", "collection", top_k=2, threshold=0.5, save_results=True)

    service.client = SimpleNamespace(get_collection=lambda collection_id: (_ for _ in ()).throw(RuntimeError("search failed")))
    with pytest.raises(RuntimeError):
        await service.search("query", "collection")


def test_generation_service_all_providers_and_save(tmp_path, monkeypatch):
    import services.generation_service as generation_module
    from services.generation_service import GenerationService

    monkeypatch.chdir(tmp_path)
    service = GenerationService()
    assert "deepseek" in service.get_available_models()

    monkeypatch.setattr(service, "_generate_with_deepseek", lambda *args, **kwargs: "deep answer")
    generated = service.generate("deepseek", "deepseek-v3", "q", [{"text": "context"}], False, api_key="key")
    assert generated["response"] == "deep answer"
    assert Path(generated["saved_filepath"]).exists()

    with pytest.raises(ValueError):
        service.generate("bad", "m", "q", [], False)

    with pytest.raises(ValueError):
        GenerationService()._generate_with_deepseek("deepseek-v3", "q", "ctx", api_key=None)

    class StubOpenAI:
        def __init__(self, **kwargs):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    message = SimpleNamespace(content="final", reasoning_content="reason")
                    return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(generation_module, "OpenAI", StubOpenAI)
    assert GenerationService()._generate_with_deepseek("deepseek-v3", "q", "ctx", api_key="key") == "final"
    assert "【思维过程】" in GenerationService()._generate_with_deepseek("deepseek-r1", "q", "ctx", api_key="key")
    assert GenerationService()._generate_with_deepseek("deepseek-r1", "q", "ctx", api_key="key", show_reasoning=False) == "final"

    class StubDelta:
        def __init__(self, reasoning_content=None, content=None):
            self.reasoning_content = reasoning_content
            self.content = content

    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=StubDelta(reasoning_content="think"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=StubDelta(content="answer"))]),
        SimpleNamespace(choices=[], usage={"tokens": 1}),
    ]

    class StubAliyunOpenAI:
        def __init__(self, **kwargs):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return chunks

    monkeypatch.setattr(generation_module, "OpenAI", StubAliyunOpenAI)
    assert GenerationService()._generate_with_aliyun("qwen-turbo", "q", "ctx") == "answer"

    class StubChatModel:
        def invoke(self, prompt):
            return SimpleNamespace(content="<think>plan</think>reply")

    hf = GenerationService()
    hf.model = StubChatModel()
    response = hf._generate_with_huggingface("Qwen-Qwen3-1.7B", "q", "ctx", load_model=False)
    assert "AI思考过程：plan" in response
    assert "AI回复：reply" in response
    monkeypatch.setattr(hf, "_load_huggingface_model", lambda model_name: (StubChatModel(), "tokenizer"))
    assert "AI回复：reply" in hf._generate_with_huggingface("Qwen-Qwen3-1.7B", "q2", "ctx", load_model=True)

    class StubCausalLM:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return {"model_name": model_name, **kwargs}

    class StubTokenizer:
        eos_token_id = 0

        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return StubTokenizer()

    monkeypatch.setattr(generation_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generation_module, "get_huggingface_model_path", lambda name: f"local/{name}")
    monkeypatch.setattr(generation_module, "AutoModelForCausalLM", StubCausalLM)
    monkeypatch.setattr(generation_module, "AutoTokenizer", StubTokenizer)
    monkeypatch.setattr(generation_module, "pipeline", lambda *args, **kwargs: {"args": args, "kwargs": kwargs})
    monkeypatch.setattr(generation_module, "HuggingFacePipeline", lambda pipeline: SimpleNamespace(pipeline=pipeline))
    monkeypatch.setattr(generation_module, "ChatHuggingFace", lambda llm: SimpleNamespace(llm=llm))
    chat_model, tokenizer = GenerationService()._load_huggingface_model("Qwen-Qwen3-1.7B")
    assert chat_model.llm.pipeline["args"][0] == "text-generation"
    assert tokenizer.eos_token_id == 0
