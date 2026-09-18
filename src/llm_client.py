"""DeepSeek 大模型调用客户端。

与项目 A（PCB 预投校验 Agent）的 ``src/agent/llm/client.py`` 保持同一套调用约定：
直接 requests 调 OpenAI 兼容接口 ``/chat/completions``，Bearer 鉴权，带重试。

与本项目 A 的差异（仅两处，均为问答场景所需）：
- 不强制 ``response_format={"type": "json_object"}``：项目 A 要的是归因 JSON，本项目的
  答案需要普通文本，强制 JSON 会让 DeepSeek 把自然语言答案塞进 JSON 字符串里，反而别扭；
- 增加 ``temperature``：默认 0.3，事实型问答需要稳定、少发挥。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

# 支持 `python src/llm_client.py` 直接运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402


class DeepSeekClient:
    """DeepSeek Chat 的轻量封装，接口与项目 A 的 OpenAICompatClient 对齐。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = None,
        timeout: int = 60,
        max_retries: int = 2,
        retry_backoff: int = 2,
        temperature: float = 0.3,
    ):
        llm = config.get_llm_config()
        base_url = base_url or llm["base_url"]
        model = model or llm["model"]
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.temperature = temperature

    def chat(self, messages: list[dict]) -> str:
        """输入 messages 列表，返回模型回复文本。

        Args:
            messages: OpenAI 消息格式 ``[{"role": "system"|"user", "content": ...}]``。

        Returns:
            模型回复的纯文本。

        Raises:
            requests.exceptions.HTTPError: 鉴权失败（401/403）立即抛出，不重试。
            requests.exceptions.RequestException: 重试耗尽后的网络/服务端错误。
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as exc:
                # 401/403 鉴权错误不重试；其余（429 限流 / 5xx / 网络）重试
                response = getattr(exc, "response", None)
                if response is not None and response.status_code in (401, 403):
                    raise
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff * (attempt + 1))

        raise RuntimeError("unreachable")  # 循环必然 return 或 raise，仅作静态兜底
