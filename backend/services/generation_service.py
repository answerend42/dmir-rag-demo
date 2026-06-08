"""! @file generation_service.py
@brief 面向本地和托管大模型提供方的回答生成服务。
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import dotenv
from openai import OpenAI
import requests
from utils.model_utils import get_huggingface_model_path
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import time

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

GROUNDED_SYSTEM_PROMPT = (
    "你是 RAG 演示系统的中文回答生成器。必须只基于提供的检索证据回答；"
    "如果证据不足，请明确说明“证据不足，无法回答”。如果问题要求具体数字、"
    "表格对比或排名，必须列出证据中的原始数值，不能只给差值或概括。"
    "回答使用中文 Markdown，并尽量用 [证据N] 标注依据。"
)
LLM_ONLY_SYSTEM_PROMPT = (
    "你是 RAG 演示系统的纯模型回答器。当前模式不使用检索证据，"
    "请直接基于模型知识用中文 Markdown 回答。若问题依赖特定课程、论文或私有资料细节，"
    "必须说明无法核验外部证据，不要伪装成已检索。严禁输出 [证据N]、证据N、"
    "已引用证据、检索证据、引用来源等 RAG 证据格式；如需解释理由，请使用"
    "“模型判断依据”或“候选答案文本线索”。"
)


def _build_chat_messages(query: str, context: str, rag_mode: str) -> List[Dict[str, str]]:
    """! @brief 根据 RAG 模式构造中文聊天消息。
    @param query 用户问题。
    @param context 已编号的检索证据文本。
    @param rag_mode 当前生成模式。
    @return 可传给 OpenAI 兼容接口的 messages。
    """
    if rag_mode == "llm_only":
        return [
            {"role": "system", "content": LLM_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{query}\n\n请直接回答："},
        ]
    return [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{query}\n\n检索证据：\n{context}\n\n请基于以上证据回答："},
    ]


class GenerationService:
    """! @brief 基于查询和检索上下文生成最终回答。

    生成服务类：负责调用不同的模型提供商（HuggingFace、OpenAI、DeepSeek）生成回答
    支持本地模型和API调用，并将生成结果保存到文件
    """

    def __init__(self):
        """
        初始化生成服务，配置支持的模型列表和创建输出目录
        """
        self.model = ""
        self.tokenizer = ""
        self.history = ""
        self.models = {
            "huggingface": {
                "Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
                "DeepSeek-7b": "deepseek-ai/deepseek-llm-7b-chat",
                "DeepSeek-R1-Distill-Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "Qwen-Qwen3-1.7B": "Qwen/Qwen3-1.7B",
            },
            "aliyun": {
                "qwen-turbo": "qwen-turbo",
                "qwen3.6-plus": "qwen3.6-plus",
            },
            "deepseek": {
                "deepseek-v3": "deepseek-chat",
                "deepseek-r1": "deepseek-reasoner",
            }
        }

        # 确保输出目录存在
        os.makedirs("05-generation-results", exist_ok=True)

    def _load_huggingface_model(self, model_name: str):
        """
        加载HuggingFace模型

        参数:
            model_name: 模型名称，对应self.models["huggingface"]中的键

        返回:
            model: 加载的模型
            tokenizer: 对应的分词器
        """
        try:
            tensor_device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.models["huggingface"][model_name]
            model_name = get_huggingface_model_path(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=tensor_device,  # 自动分配GPU/CPU
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,  # 控制生成文本的最大长度
                temperature=0.7,  # 控制生成随机性
                top_k=50,  # 限制候选词数量
                top_p=0.9,  # 核采样参数
                num_return_sequences=1,  # 生成单个序列
                truncation=True,  # 启用输入截断
                pad_token_id=tokenizer.eos_token_id,  # 填充token设置
                clean_up_tokenization_spaces=False  # 保留原始分词空格
            )

            hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
            chat_model = ChatHuggingFace(llm=hf_pipeline)

            return chat_model, tokenizer
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise

    def _generate_with_huggingface(
            self,
            model_name: str,
            query: str,
            context: str,
            load_model: bool,
            rag_mode: str = "basic_rag",
            max_length: int = 1024,
    ) -> str:
        """
        使用HuggingFace模型生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            max_length: 生成文本的最大长度

        返回:
            生成的回答文本
        """
        try:
            if bool(load_model) == True:
                self.model, self.tokenizer = self._load_huggingface_model(model_name)

            # 构建提示
            # prompt = f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。
            template = (
                """请直接基于模型知识回答问题。如果问题依赖特定课程、论文或私有资料细节，请明确说明无法核验外部证据。严禁输出 [证据N]、证据N、已引用证据、检索证据、引用来源等 RAG 证据格式；如需解释理由，请使用“模型判断依据”或“候选答案文本线索”。
                        回话记录：{history}
                        问题：{query}

                        回答："""
                if rag_mode == "llm_only"
                else """请基于上下文与回话记录回答问题。如果上下文和回话记录中没有相关信息，请明确说明“证据不足，无法回答”，不要自由发挥。如果问题要求具体数字、表格对比或排名，必须列出证据中的原始数值，不能只给差值或概括。
                        回话记录：{history}
                        问题：{query}

                        上下文：
                        {context}

                        回答："""
            )
            prompt = ChatPromptTemplate.from_template(template)

            # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # outputs = model.generate(
            #     **inputs,
            #     max_length=max_length,
            #     num_return_sequences=1,
            #     temperature=0.7,
            #     do_sample=True
            # )

            # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ts = time.time()
            answer = self.model.invoke(prompt.format(query=query, history=self.history, context=context))
            spent_sec = int(time.time() - ts)
            text = answer.content
            text_parts = text.split(r"<think>")
            parts = text_parts[1].split(r"</think>")
            thinkingInfo = parts[0]
            responseInfo = parts[1]

            self.history += f'''
用户提问：{query}
AI回复：{responseInfo}

'''
            answer_content = f'''s:{spent_sec}
用户提问：{query}
AI思考过程：{thinkingInfo}
AI回复：{responseInfo}

'''
            # return answer.split("回答：")[-1].strip()
            return answer_content

        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            raise

    def _generate_with_aliyun(
            self,
            model_name: str,
            query: str,
            context: str,
            api_key: Optional[str] = None,
            rag_mode: str = "basic_rag",
    ) -> str:
        """
        使用OpenAI API生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: OpenAI API密钥，如不提供则从环境变量获取

        返回:
            生成的回答文本
        """
        try:
            client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            # client = OpenAI(api_key=api_key)

            messages = _build_chat_messages(query, context, rag_mode)

            completion = client.chat.completions.create(
                model=self.models["aliyun"][model_name],
                messages=messages,
                # 通过 extra_body 设置 enable_thinking 开启思考模式
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={
                    "include_usage": True
                },
            )
            reasoning_content = ""  # 完整思考过程
            answer_content = ""  # 完整回复
            is_answering = False  # 是否进入回复阶段
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

            for chunk in completion:
                if not chunk.choices:
                    print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
                    print(chunk.usage)
                    continue

                delta = chunk.choices[0].delta

                # 只收集思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if not is_answering:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                # 收到content，开始进行回复
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end="", flush=True)
                    answer_content += delta.content
            return answer_content.strip()

        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise

    def _generate_with_deepseek(
            self,
            model_name: str,
            query: str,
            context: str,
            api_key: Optional[str] = None,
            show_reasoning: bool = True,
            rag_mode: str = "basic_rag",
    ) -> str:
        """
        使用DeepSeek API生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: DeepSeek API密钥，如不提供则从环境变量获取
            show_reasoning: 是否显示推理过程（仅对推理模型有效）

        返回:
            生成的回答文本，对于推理模型可能包含思维过程
        """
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            messages = _build_chat_messages(query, context, rag_mode)

            response = client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                max_tokens=512,
                stream=False
            )

            # 如果是推理模型，处理思维链输出
            if model_name == "deepseek-r1":
                message = response.choices[0].message
                reasoning = message.reasoning_content
                answer = message.content

                if show_reasoning and reasoning:
                    return f"【思维过程】\n{reasoning}\n\n【最终答案】\n{answer}"
                return answer

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {str(e)}")
            raise

    def generate(
            self,
            provider: str,
            model_name: str,
            query: str,
            search_results: List[Dict],
            load_model: bool,
            api_key: Optional[str] = None,
            show_reasoning: bool = True,
            rag_mode: str = "basic_rag",
    ) -> Dict:
        """! @brief 生成回答并持久化生成结果。
        @param provider 模型提供方，例如 huggingface、aliyun 或 deepseek。
        @param model_name 提供方内部的模型键。
        @param query 用户问题。
        @param search_results 检索到的上下文列表。
        @param load_model 调用前是否需要加载本地模型。
        @param api_key 托管提供方的可选 API 密钥。
        @param show_reasoning 支持时是否包含推理输出。
        @param rag_mode 生成模式，支持 basic_rag、optimized_rag 和 llm_only。
        @return 回答文本和保存文件路径。

        生成回答并保存结果

        参数:
            provider: 模型提供商，可选值为"huggingface"、"openai"、"deepseek"
            model_name: 模型名称
            query: 用户查询
            search_results: 搜索结果列表，用于构建上下文
            api_key: API密钥（对于API调用）
            show_reasoning: 是否显示推理过程（仅对DeepSeek推理模型有效）
            load_model: 是否装载模型
            rag_mode: 当前 RAG 模式

        返回:
            包含生成回答和保存路径的字典
        """
        try:
            normalized_rag_mode = rag_mode or "basic_rag"
            effective_search_results = [] if normalized_rag_mode == "llm_only" else search_results
            # 准备上下文
            context = "\n\n".join([
                f"[证据{i + 1}]: {result['text']}"
                for i, result in enumerate(effective_search_results)
            ])

            ts = time.time()
            # 根据不同提供商生成回答
            if provider == "huggingface":
                response = self._generate_with_huggingface(model_name, query, context, load_model, normalized_rag_mode)
            elif provider == "aliyun":
                response = self._generate_with_aliyun(model_name, query, context, api_key, normalized_rag_mode)
            elif provider == "deepseek":
                response = self._generate_with_deepseek(
                    model_name,
                    query,
                    context,
                    api_key,
                    show_reasoning,
                    normalized_rag_mode,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # 准备保存的结果
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "rag_mode": normalized_rag_mode,
                "response": response,
                "context": effective_search_results
            }

            # 生成文件名并保存
            spend_sec = int(time.time() - ts)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"generation_{provider}_{model_name}_{spend_sec}s_{timestamp}_{load_model}.json"
            filepath = os.path.join("05-generation-results", filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return {
                "response": response,
                "saved_filepath": filepath
            }

        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise

    def get_available_models(self) -> Dict:
        """
        获取可用的模型列表

        返回:
            包含所有支持模型的字典
        """
        return self.models
