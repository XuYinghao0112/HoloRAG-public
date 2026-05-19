import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from .config import NaiveHoloRAGConfig
from .utils import safe_parse_json

logger = logging.getLogger(__name__)


class LocalLLMClient:
    def __init__(self, config: NaiveHoloRAGConfig) -> None:
        self.config = config
        timeout = httpx.Timeout(
            connect=float(os.getenv("HOLORAG_LLM_CONNECT_TIMEOUT", "20")),
            read=float(os.getenv("HOLORAG_LLM_READ_TIMEOUT", "180")),
            write=float(os.getenv("HOLORAG_LLM_WRITE_TIMEOUT", "180")),
            pool=float(os.getenv("HOLORAG_LLM_POOL_TIMEOUT", "180")),
        )
        http_client = httpx.Client(trust_env=False, timeout=timeout)
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=os.getenv("OPENAI_API_KEY", "sk-"),
            max_retries=0,
            http_client=http_client,
        )
        self.stats = {"completion_calls": 0, "json_calls": 0, "text_calls": 0}

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def _completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.stats["completion_calls"] += 1
        params: Dict[str, Any] = {
            "model": self.config.llm_model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_new_tokens,
        }
        if response_format:
            params["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**params)
        except TypeError:
            params.pop("response_format", None)
            response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(str(getattr(item, "text", item)) for item in content).strip()
        return str(content).strip()

    def infer_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        self.stats["json_calls"] += 1
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw_text = self._completion(messages, response_format={"type": "json_object"}, max_tokens=max_tokens)
            parsed = safe_parse_json(raw_text, fallback)
            return parsed if isinstance(parsed, dict) else fallback, raw_text
        except Exception as exc:
            logger.warning("LLM JSON call failed; using fallback. Error: %s", exc)
            return fallback, ""

    def infer_messages_text(
        self,
        messages: List[Dict[str, str]],
        fallback: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        self.stats["text_calls"] += 1
        try:
            return self._completion(messages, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("LLM text call failed; using fallback. Error: %s", exc)
            return fallback
