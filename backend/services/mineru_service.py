"""! @file mineru_service.py
@brief MinerU 文档解析服务。
"""

import io
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
import requests

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


class MinerUAgentError(RuntimeError):
    """! @brief MinerU Agent API 调用失败时抛出的业务异常。"""


class MinerUPrecisionError(RuntimeError):
    """! @brief MinerU 精准解析 API 调用失败时抛出的业务异常。"""


class MinerUAgentParser:
    """! @brief 调用 MinerU Agent API 并归一化为前端解析结果结构。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        poll_timeout_seconds: int = 180,
        poll_interval_seconds: int = 3,
        request_timeout_seconds: int = 30,
    ):
        """! @brief 初始化 MinerU Agent API 客户端配置。"""
        self.base_url = (
            base_url
            or os.getenv("MINERU_AGENT_BASE_URL")
            or "https://mineru.net/api/v1/agent"
        ).rstrip("/")
        self.poll_timeout_seconds = poll_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds

    def parse_file(self, file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """! @brief 上传文件到 MinerU 并返回结构化 Markdown 解析结果。"""
        display_name = Path(file_name or file_path).name
        task_id, upload_url = self._create_upload_task(display_name)
        self._upload_file(upload_url, file_path)
        result_data = self._poll_task(task_id)
        markdown_url = result_data.get("markdown_url")
        if not markdown_url:
            raise MinerUAgentError("MinerU 解析完成但未返回 Markdown 下载链接。")
        markdown = self._download_markdown(markdown_url)
        return self._build_parsed_content(display_name, task_id, markdown_url, markdown)

    def _create_upload_task(self, file_name: str) -> tuple[str, str]:
        """! @brief 创建 MinerU 文件上传解析任务并获取签名上传地址。"""
        language = os.getenv("MINERU_AGENT_LANGUAGE", "ch")
        payload = {
            "file_name": file_name,
            "language": language,
            "enable_table": self._env_bool("MINERU_AGENT_ENABLE_TABLE", True),
            "is_ocr": self._env_bool("MINERU_AGENT_IS_OCR", False),
            "enable_formula": self._env_bool("MINERU_AGENT_ENABLE_FORMULA", True),
        }
        page_range = os.getenv("MINERU_AGENT_PAGE_RANGE")
        if page_range:
            payload["page_range"] = page_range

        response = requests.post(
            f"{self.base_url}/parse/file",
            json=payload,
            timeout=self.request_timeout_seconds,
        )
        result = self._json_response(response, "创建 MinerU 解析任务失败")
        data = result.get("data") or {}
        task_id = data.get("task_id")
        upload_url = data.get("file_url")
        if not task_id or not upload_url:
            raise MinerUAgentError("MinerU 解析任务响应缺少 task_id 或 file_url。")
        return task_id, upload_url

    def _upload_file(self, upload_url: str, file_path: str) -> None:
        """! @brief 使用 MinerU 返回的签名 URL 上传本地文件。"""
        with open(file_path, "rb") as file_obj:
            response = requests.put(
                upload_url,
                data=file_obj,
                timeout=self.request_timeout_seconds,
            )
        if response.status_code not in (200, 201):
            raise MinerUAgentError(f"上传文件到 MinerU 失败，HTTP {response.status_code}。")

    def _poll_task(self, task_id: str) -> Dict[str, Any]:
        """! @brief 轮询 MinerU 解析任务直到完成或超时。"""
        deadline = time.time() + self.poll_timeout_seconds
        while time.time() < deadline:
            response = requests.get(
                f"{self.base_url}/parse/{task_id}",
                timeout=self.request_timeout_seconds,
            )
            result = self._json_response(response, "查询 MinerU 解析任务失败")
            data = result.get("data") or {}
            state = data.get("state")
            if state == "done":
                return data
            if state == "failed":
                err_msg = data.get("err_msg") or "未知错误"
                err_code = data.get("err_code")
                raise MinerUAgentError(f"MinerU 解析失败：{err_msg}（{err_code}）。")
            time.sleep(self.poll_interval_seconds)
        raise MinerUAgentError(f"MinerU 解析超时，请稍后重试或缩小文件范围：{task_id}")

    def _download_markdown(self, markdown_url: str) -> str:
        """! @brief 下载 MinerU 返回的 Markdown 结果。"""
        response = requests.get(markdown_url, timeout=self.request_timeout_seconds)
        if response.status_code != 200:
            raise MinerUAgentError(f"下载 MinerU Markdown 失败，HTTP {response.status_code}。")
        return response.text

    def _json_response(self, response: requests.Response, failure_message: str) -> Dict[str, Any]:
        """! @brief 校验 MinerU JSON 响应。"""
        if response.status_code != 200:
            raise MinerUAgentError(f"{failure_message}，HTTP {response.status_code}。")
        try:
            result = response.json()
        except ValueError as exc:
            raise MinerUAgentError(f"{failure_message}，响应不是合法 JSON。") from exc
        if result.get("code") != 0:
            raise MinerUAgentError(f"{failure_message}：{result.get('msg') or '未知错误'}。")
        return result

    def _build_parsed_content(
        self,
        file_name: str,
        task_id: str,
        markdown_url: str,
        markdown: str,
    ) -> Dict[str, Any]:
        """! @brief 将 MinerU Markdown 转换为现有 parsed_content 结构。"""
        return {
            "metadata": {
                "filename": file_name,
                "total_pages": None,
                "parsing_method": "mineru_agent",
                "source": "MinerU Agent API",
                "timestamp": datetime.now().isoformat(),
                "mineru_task_id": task_id,
                "mineru_markdown_url": markdown_url,
            },
            "content": self._markdown_sections(markdown),
        }

    def _markdown_sections(self, markdown: str) -> List[Dict[str, Any]]:
        """! @brief 按 Markdown 标题切分 MinerU 解析结果。"""
        lines = markdown.splitlines()
        sections: List[Dict[str, Any]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []

        def flush_section() -> None:
            content = "\n".join(current_lines).strip()
            if current_title or content:
                sections.append(
                    {
                        "type": "Markdown",
                        "title": current_title,
                        "content": content,
                        "page": None,
                    }
                )

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                flush_section()
                current_title = heading_match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        flush_section()
        if sections:
            return sections
        return [{"type": "Markdown", "title": None, "content": markdown.strip(), "page": None}]

    def _env_bool(self, name: str, default: bool) -> bool:
        """! @brief 从环境变量读取布尔开关。"""
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}


class MinerUPrecisionParser:
    """! @brief 调用 MinerU VLM 精准解析 API 并归一化为前端解析结果结构。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        poll_timeout_seconds: int = 300,
        poll_interval_seconds: int = 5,
        request_timeout_seconds: int = 60,
    ):
        """! @brief 初始化 MinerU 精准解析 API 客户端配置。"""
        self.base_url = (
            base_url
            or os.getenv("MINERU_API_BASE_URL")
            or "https://mineru.net/api/v4"
        ).rstrip("/")
        self.token = os.getenv("MINERU_API_TOKEN")
        if not self.token:
            raise MinerUPrecisionError("缺少 MINERU_API_TOKEN，请先在 .env 或环境变量中配置 MinerU Token。")
        self.model_version = os.getenv("MINERU_MODEL_VERSION", "vlm")
        self.poll_timeout_seconds = poll_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds

    def parse_file(self, file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """! @brief 使用 MinerU VLM 精准解析上传文件并返回 Markdown 结构。"""
        display_name = Path(file_name or file_path).name
        batch_id, upload_url = self._create_upload_task(display_name)
        self._upload_file(upload_url, file_path)
        extract_result = self._poll_batch_result(batch_id)
        full_zip_url = extract_result.get("full_zip_url")
        if not full_zip_url:
            raise MinerUPrecisionError("MinerU 精准解析完成但未返回 full_zip_url。")
        markdown = self._download_markdown_from_zip(full_zip_url)
        return self._build_parsed_content(display_name, batch_id, full_zip_url, markdown, extract_result)

    def _headers(self) -> Dict[str, str]:
        """! @brief 构造 MinerU 精准解析 API 请求头。"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def _create_upload_task(self, file_name: str) -> tuple[str, str]:
        """! @brief 申请 MinerU 精准解析签名上传 URL。"""
        data_id = _safe_data_id(file_name)
        file_payload: Dict[str, Any] = {
            "name": file_name,
            "data_id": data_id,
            "is_ocr": _env_bool("MINERU_IS_OCR", False),
        }
        page_ranges = os.getenv("MINERU_PAGE_RANGES")
        if page_ranges:
            file_payload["page_ranges"] = page_ranges

        payload = {
            "files": [file_payload],
            "model_version": self.model_version,
            "language": os.getenv("MINERU_LANGUAGE", "ch"),
            "enable_table": _env_bool("MINERU_ENABLE_TABLE", True),
            "enable_formula": _env_bool("MINERU_ENABLE_FORMULA", True),
        }
        extra_formats = _env_list("MINERU_EXTRA_FORMATS")
        if extra_formats:
            payload["extra_formats"] = extra_formats

        response = requests.post(
            f"{self.base_url}/file-urls/batch",
            headers=self._headers(),
            json=payload,
            timeout=self.request_timeout_seconds,
        )
        result = _json_response(response, "创建 MinerU VLM 精准解析上传任务失败", MinerUPrecisionError)
        data = result.get("data") or {}
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls") or []
        if not batch_id or not file_urls:
            raise MinerUPrecisionError("MinerU 精准解析响应缺少 batch_id 或 file_urls。")
        return batch_id, file_urls[0]

    def _upload_file(self, upload_url: str, file_path: str) -> None:
        """! @brief 使用 MinerU 返回的签名 URL 上传本地文件。"""
        with open(file_path, "rb") as file_obj:
            response = requests.put(
                upload_url,
                data=file_obj,
                timeout=self.request_timeout_seconds,
            )
        if response.status_code not in (200, 201):
            raise MinerUPrecisionError(f"上传文件到 MinerU 精准解析失败，HTTP {response.status_code}。")

    def _poll_batch_result(self, batch_id: str) -> Dict[str, Any]:
        """! @brief 轮询 MinerU 批量解析结果直到单文件任务完成。"""
        deadline = time.time() + self.poll_timeout_seconds
        while time.time() < deadline:
            response = requests.get(
                f"{self.base_url}/extract-results/batch/{batch_id}",
                headers=self._headers(),
                timeout=self.request_timeout_seconds,
            )
            result = _json_response(response, "查询 MinerU VLM 精准解析结果失败", MinerUPrecisionError)
            data = result.get("data") or {}
            extract_results = data.get("extract_result") or []
            if not extract_results:
                time.sleep(self.poll_interval_seconds)
                continue

            extract_result = extract_results[0]
            state = extract_result.get("state")
            if state == "done":
                return extract_result
            if state == "failed":
                err_msg = extract_result.get("err_msg") or "未知错误"
                raise MinerUPrecisionError(f"MinerU VLM 精准解析失败：{err_msg}。")
            time.sleep(self.poll_interval_seconds)
        raise MinerUPrecisionError(f"MinerU VLM 精准解析超时，请稍后重试或缩小文件范围：{batch_id}")

    def _download_markdown_from_zip(self, full_zip_url: str) -> str:
        """! @brief 下载精准解析 zip，并从中读取 full.md。"""
        response = requests.get(full_zip_url, timeout=self.request_timeout_seconds)
        if response.status_code != 200:
            raise MinerUPrecisionError(f"下载 MinerU 精准解析结果压缩包失败，HTTP {response.status_code}。")

        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = archive.namelist()
                markdown_names = [
                    name for name in names
                    if name.endswith("/full.md") or name == "full.md"
                ]
                if not markdown_names:
                    markdown_names = [name for name in names if name.lower().endswith(".md")]
                if not markdown_names:
                    raise MinerUPrecisionError("MinerU 精准解析结果压缩包中没有 Markdown 文件。")
                raw_markdown = archive.read(markdown_names[0])
        except zipfile.BadZipFile as exc:
            raise MinerUPrecisionError("MinerU 精准解析结果不是合法 zip 文件。") from exc

        return raw_markdown.decode("utf-8-sig", errors="replace")

    def _build_parsed_content(
        self,
        file_name: str,
        batch_id: str,
        full_zip_url: str,
        markdown: str,
        extract_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """! @brief 将 MinerU VLM Markdown 转换为现有 parsed_content 结构。"""
        progress = extract_result.get("extract_progress") or {}
        return {
            "metadata": {
                "filename": file_name,
                "total_pages": progress.get("total_pages"),
                "parsing_method": "mineru_vlm",
                "source": "MinerU VLM 精准解析 API",
                "timestamp": datetime.now().isoformat(),
                "mineru_batch_id": batch_id,
                "mineru_full_zip_url": full_zip_url,
                "mineru_model_version": self.model_version,
            },
            "content": _markdown_sections(markdown),
        }


def _safe_data_id(file_name: str) -> str:
    """! @brief 为 MinerU data_id 生成稳定且合法的短标识。"""
    stem = Path(file_name).stem or "mineru_file"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return (cleaned or "mineru_file")[:128]


def _env_bool(name: str, default: bool) -> bool:
    """! @brief 从环境变量读取布尔开关。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_list(name: str) -> List[str]:
    """! @brief 从逗号分隔环境变量读取字符串列表。"""
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_response(
    response: requests.Response,
    failure_message: str,
    error_type: type[RuntimeError],
) -> Dict[str, Any]:
    """! @brief 校验 MinerU JSON 响应。"""
    if response.status_code != 200:
        raise error_type(f"{failure_message}，HTTP {response.status_code}。")
    try:
        result = response.json()
    except ValueError as exc:
        raise error_type(f"{failure_message}，响应不是合法 JSON。") from exc
    if result.get("code") != 0:
        raise error_type(f"{failure_message}：{result.get('msg') or '未知错误'}。")
    return result


def _markdown_sections(markdown: str) -> List[Dict[str, Any]]:
    """! @brief 按 Markdown 标题切分解析结果。"""
    lines = markdown.splitlines()
    sections: List[Dict[str, Any]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush_section() -> None:
        content = "\n".join(current_lines).strip()
        if current_title or content:
            sections.append(
                {
                    "type": "Markdown",
                    "title": current_title,
                    "content": content,
                    "page": None,
                }
            )

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            flush_section()
            current_title = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()
    if sections:
        return sections
    return [{"type": "Markdown", "title": None, "content": markdown.strip(), "page": None}]
