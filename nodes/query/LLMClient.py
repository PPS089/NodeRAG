from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.FindProjectRoot import find_project_root as fr  # noqa: E402


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"
DEFAULT_TIMEOUT = 60


def validate_ascii_secret(name: str, value: str) -> None:
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} 必须是真实 API Key，不能包含中文或其他非 ASCII 字符") from exc


class OpenAICompatibleChatClient:
    """
    百炼 OpenAI 兼容 Chat Completions 客户端。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        project_root = fr()
        load_dotenv(project_root / ".env")

        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.getenv("LLM_MODEL_ID") or DEFAULT_MODEL
        self.timeout = int(timeout or os.getenv("LLM_TIMEOUT", DEFAULT_TIMEOUT))

        if not self.api_key:
            raise ValueError("请在 .env 中配置 LLM_API_KEY")
        validate_ascii_secret("LLM_API_KEY", self.api_key)

    @property
    def chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            self.chat_completions_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return parse_json_object(content)


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"模型未返回 JSON 对象: {text}")

    parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError(f"模型返回 JSON 不是对象: {text}")
    return parsed


def find_available_documents() -> List[str]:
    result_dir = fr() / "MinerUResult"
    if not result_dir.exists():
        return []

    names = []
    for path in sorted(result_dir.iterdir()):
        if not path.is_dir():
            continue
        name = re.sub(r"_[0-9a-f]{12}$", "", path.name)
        names.append(name)
    return names


def build_question_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("question_arg", nargs="*", help="用户问题，未传 --question 时使用。")
    parser.add_argument("--question", "-q", help="用户问题。")
    parser.add_argument("--temperature", type=float, default=0.1, help="模型 temperature。")
    return parser


def get_question(args: argparse.Namespace) -> str:
    question = args.question or " ".join(args.question_arg)
    question = question.strip()
    if not question:
        raise ValueError("请通过 --question 或位置参数传入用户问题")
    return question


def print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def documents_prompt_text(documents: Sequence[str]) -> str:
    if not documents:
        return "当前未发现本地知识库文档。"
    return "\n".join(f"- {name}" for name in documents)
