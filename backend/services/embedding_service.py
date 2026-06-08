"""! @file embedding_service.py
@brief 嵌入模型提供方抽象和持久化辅助逻辑。
"""

import os
import dotenv
dotenv.load_dotenv()
import json
from datetime import datetime
from enum import Enum
import boto3
from langchain_community.embeddings import BedrockEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from rag_core.embeddings import QwenApiEmbedder
from utils.model_utils import get_huggingface_model_path

class EmbeddingProvider(str, Enum):
    """! @brief 支持的嵌入提供方标识。

    嵌入提供商枚举类，定义支持的嵌入模型提供商
    """
    OPENAI = "openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    QWEN_API = "qwen_api"

class EmbeddingConfig:
    """! @brief 单个嵌入模型的运行时配置。

    嵌入配置类，用于存储嵌入模型的配置信息
    """
    def __init__(self, provider: str, model_name: str):
        """
        初始化嵌入配置
        
        参数:
            provider: 嵌入提供商名称
            model_name: 嵌入模型名称
        """
        self.provider = provider
        self.model_name = model_name
        self.aws_region = "ap-southeast-1"  # 可配置

class EmbeddingService:
    """! @brief 创建、保存和查看文档嵌入。

    嵌入服务类，提供创建和管理文本嵌入的功能
    """
    def __init__(self):
        """初始化嵌入服务，创建嵌入工厂实例"""
        self.embedding_factory = EmbeddingFactory()

    def create_embeddings(self, input_data: dict, config: EmbeddingConfig) -> tuple:
        """! @brief 创建文本块的嵌入向量并返回必要的信息.
        @param input_data 包含 chunks 和 metadata 的输入数据字典。
        @param config 嵌入配置对象。
        @return (embeddings, metadata) 元组；当前 metadata 为空字典。

        创建文本块的嵌入向量并返回必要的信息
        
        参数:
            input_data: 包含文本块和元数据的输入数据字典
            config: 嵌入配置对象
            
        返回:
            包含嵌入结果和元数据的元组
        """
        embedding_function = self.embedding_factory.create_embedding_function(config)
        provider_value = self._provider_value(config.provider)
        
        chunks = input_data.get('chunks', [])
        filename = input_data.get('metadata', {}).get('filename', '')  # 获取文件名
        
        # 批处理大小
        BATCH_SIZE = 20
        results = []
        
        batch_providers = {EmbeddingProvider.OPENAI.value, EmbeddingProvider.QWEN_API.value}
        if provider_value in batch_providers:
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                # 提取当前批次的文本内容
                texts = [chunk.get("content", "") for chunk in batch]
                
                # 批量获取embeddings
                embedding_vectors = embedding_function.embed_documents(texts)
                
                # 将结果与原始chunk数据组合
                for chunk, embedding_vector in zip(batch, embedding_vectors):
                    chunk_metadata = {
                        key: value
                        for key, value in (chunk.get("metadata") or {}).items()
                        if key != "answer_quality"
                    }
                    metadata = {
                        **chunk_metadata,
                        "chunk_id": chunk_metadata.get("chunk_id"),
                        "page_number": chunk_metadata.get("page_number", ""),
                        "page_range": chunk_metadata.get("page_range", ""),
                        "content": chunk["content"],
                        "word_count": chunk_metadata.get("word_count", 0),
                        # "chunking_method": input_data.get("chunking_method", "loaded"),
                        "total_chunks": len(chunks),
                        "embedding_provider": provider_value,
                        "embedding_model": config.model_name,
                        "embedding_timestamp": datetime.now().isoformat(),
                        "vector_dimension": len(embedding_vector),
                        "filename": filename  # 添加文件名到metadata
                    }
                    
                    embedding_result = {
                        "embedding": embedding_vector,
                        "metadata": metadata
                    }
                    results.append(embedding_result)
        else:
            # 对其他提供商保持原有的逐个处理逻辑
            for chunk in chunks:
                embedding_vector = embedding_function.embed_query(chunk["content"])
                chunk_metadata = {
                    key: value
                    for key, value in (chunk.get("metadata") or {}).items()
                    if key != "answer_quality"
                }
                metadata = {
                    **chunk_metadata,
                    "chunk_id": chunk_metadata.get("chunk_id"),
                    "page_number": chunk_metadata.get("page_number", ""),
                    "page_range": chunk_metadata.get("page_range", ""),
                    "content": chunk["content"],
                    "word_count": chunk_metadata.get("word_count", 0),
                    # "chunking_method": input_data.get("chunking_method", "loaded"),
                    "total_chunks": len(chunks),
                    "embedding_provider": provider_value,
                    "embedding_model": config.model_name,
                    "embedding_timestamp": datetime.now().isoformat(),
                    "vector_dimension": len(embedding_vector),
                    "filename": filename  # 添加文件名到metadata
                }
                
                embedding_result = {
                    "embedding": embedding_vector,
                    "metadata": metadata
                }
                results.append(embedding_result)
        
        # 返回结果和空的metadata（因为metadata已经包含在每个embedding中）
        return results, {}

    @staticmethod
    def _provider_value(provider: str) -> str:
        """! @brief 将枚举或字符串形式的 provider 统一为可序列化字符串。"""
        return provider.value if isinstance(provider, EmbeddingProvider) else str(provider)

    def save_embeddings(self, doc_name: str, embeddings: list) -> str:
        """
        保存嵌入向量到JSON文件
        
        参数:
            doc_name: 文档名称
            embeddings: 嵌入向量列表
            
        返回:
            保存的文件路径
        """
        os.makedirs("02-embedded-docs", exist_ok=True)
        
        # 获取第一个embedding的元数据
        first_embedding = embeddings[0]
        provider = first_embedding["metadata"]["embedding_provider"]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 优先使用文档元数据中的原始文件名，避免把 JSON 数据集误标成 PDF。
        source_filename = first_embedding.get("metadata", {}).get("filename") or doc_name
        base_name = os.path.basename(str(source_filename))
        base_root, base_ext = os.path.splitext(base_name)
        if not base_root:
            base_root = os.path.splitext(os.path.basename(str(doc_name)))[0] or "document"
        if not base_ext:
            base_ext = ".json" if "json" in str(doc_name).lower() else ".pdf"
        source_display_name = f"{base_root}{base_ext}"
        
        # 构建新的文件名：基础名称_provider_时间戳。
        filename = f"{base_root}_{provider}_{timestamp}.json"
        filepath = os.path.join("02-embedded-docs", filename)
        
        # 从第一个embedding中获取配置信息
        config_info = {
            "filename": source_display_name,
            "chunked_doc_name": doc_name,  # 添加 chunked_doc_name
            "created_at": datetime.now().isoformat(),
            "embedding_provider": provider,
            "embedding_model": first_embedding["metadata"]["embedding_model"],
            "vector_dimension": first_embedding["metadata"]["vector_dimension"]
        }
        
        class CompactJSONEncoder(json.JSONEncoder):
            """自定义JSON编码器，用于优化嵌入向量的存储格式"""
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
            
            def encode(self, obj):
                # 将 embedding 数组转换为单行，其他保持格式化
                def format_list(lst):
                    if isinstance(lst, list):
                        # 检查是否为 embedding 数组（通过检查第一个元素是否为数字）
                        if lst and isinstance(lst[0], (int, float)):
                            return '[' + ','.join(map(str, lst)) + ']'
                        return [format_list(item) for item in lst]
                    elif isinstance(lst, dict):
                        return {k: format_list(v) for k, v in lst.items()}
                    return lst
                
                return super().encode(format_list(obj))
        
        # 保存数据，配置信息放在顶层
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                **config_info,  # 配置信息放在顶层
                "embeddings": embeddings
            }, f, ensure_ascii=False, indent=2, cls=CompactJSONEncoder)
            
        return filepath

    def create_single_embedding(self, text: str, provider: str, model: str) -> list:
        """
        创建单个文本的嵌入向量
        
        参数:
            text: 需要嵌入的文本
            provider: 嵌入提供商
            model: 嵌入模型名称
            
        返回:
            嵌入向量列表
        """
        config = EmbeddingConfig(provider=provider, model_name=model)
        embedding_function = self.embedding_factory.create_embedding_function(config)
        return embedding_function.embed_query(text)

    def get_document_embedding_config(self, collection_name: str) -> EmbeddingConfig:
        """
        从已存在的文档中获取嵌入配置
        
        参数:
            collection_name: 集合名称
            
        返回:
            嵌入配置对象
            
        异常:
            ValueError: 当找不到匹配的嵌入配置时抛出
        """
        try:
            # 只取第一个下划线之前的部分
            doc_name = collection_name.split('_')[0]
            
            # 查找对应的embedding文件
            embedded_docs_dir = "02-embedded-docs"
            for filename in os.listdir(embedded_docs_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(embedded_docs_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 使用 filename 而不是 document_name
                        if data.get("filename") == doc_name:
                            return EmbeddingConfig(
                                provider=data.get("embedding_provider"),
                                model_name=data.get("embedding_model")
                            )
                            
            raise ValueError(f"No matching embedding configuration found for collection: {collection_name}")
        except Exception as e:
            raise ValueError(f"Error getting embedding config: {str(e)}")

class EmbeddingFactory:
    """
    嵌入工厂类，负责创建不同提供商的嵌入函数
    """
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        """
        根据配置创建嵌入函数
        
        参数:
            config: 嵌入配置对象
            
        返回:
            嵌入函数对象
            
        异常:
            ValueError: 当提供商不支持时抛出
        """
        if config.provider == EmbeddingProvider.BEDROCK:
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.aws_region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            return BedrockEmbeddings(
                client=bedrock_client,
                model_id=config.model_name
            )
            
        elif config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
        elif config.provider == EmbeddingProvider.HUGGINGFACE:
            model_name = get_huggingface_model_path(config.model_name)
            return HuggingFaceEmbeddings(
                model_name=model_name
            )

        elif config.provider == EmbeddingProvider.QWEN_API:
            return QwenApiEmbeddingFunction(model_name=config.model_name)
            
        raise ValueError(f"Unsupported embedding provider: {config.provider}")


class QwenApiEmbeddingFunction:
    """! @brief 将契约层 QwenApiEmbedder 适配为旧 service 的 list[float] 接口。"""

    def __init__(self, model_name: str):
        """! @brief 初始化百炼 embedding 适配器。
        @param model_name DashScope embedding 模型名。
        """
        self.embedder = QwenApiEmbedder(model=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """! @brief 批量生成文档向量。
        @param texts 待向量化的文档分块文本。
        @return 与 texts 顺序一致的向量列表。
        """
        return [embedding.vector for embedding in self.embedder.embed_batch(texts)]

    def embed_query(self, text: str) -> list[float]:
        """! @brief 生成查询向量。
        @param text 用户查询文本。
        @return 查询向量。
        """
        return self.embedder.embed_query(text).vector
