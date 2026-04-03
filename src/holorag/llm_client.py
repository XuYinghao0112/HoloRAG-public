import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from .config import HoloRAGConfig
from .utils import safe_parse_json

logger = logging.getLogger(__name__)


class LocalLLMClient:
    def __init__(self, config: HoloRAGConfig) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "sk-")
        self.config = config
        connect_timeout = float(os.getenv("HOLORAG_LLM_CONNECT_TIMEOUT", "20"))
        read_timeout = float(os.getenv("HOLORAG_LLM_READ_TIMEOUT", "180"))
        write_timeout = float(os.getenv("HOLORAG_LLM_WRITE_TIMEOUT", "180"))
        pool_timeout = float(os.getenv("HOLORAG_LLM_POOL_TIMEOUT", "180"))
        max_connections = int(os.getenv("HOLORAG_LLM_MAX_CONNECTIONS", "128"))
        max_keepalive = int(os.getenv("HOLORAG_LLM_MAX_KEEPALIVE", "32"))

        http_client = httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
            ),
        )
        self.client = OpenAI(base_url=config.llm_base_url, api_key=api_key, max_retries=0, http_client=http_client)
        self.stats = {
            "completion_calls": 0,
            "json_calls": 0,
            "text_calls": 0,
        }
        self._stats_lock = threading.Lock()
        self._max_attempts = int(os.getenv("HOLORAG_LLM_MAX_ATTEMPTS", "3"))

    def _inc_stat(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] = int(self.stats.get(key, 0)) + 1

    def _create_with_retries(self, params: Dict[str, Any]):
        delay = 0.6
        for attempt in range(1, max(1, self._max_attempts) + 1):
            self._inc_stat("completion_calls")
            try:
                return self.client.chat.completions.create(**params)
            except Exception as exc:
                if attempt >= self._max_attempts:
                    raise
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self._max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 5.0)

    def _create_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ):
        params: Dict[str, Any] = {
            "model": self.config.llm_model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_new_tokens,
        }
        if response_format is not None:
            params["response_format"] = response_format
        try:
            return self._create_with_retries(params)
        except TypeError:
            params.pop("response_format", None)
            return self._create_with_retries(params)

    def _extract_content(self, response: Any) -> str:
        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, dict):
                    pieces.append(item.get("text", ""))
                else:
                    pieces.append(getattr(item, "text", str(item)))
            return "".join(pieces)
        return str(content)

    def infer_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        self._inc_stat("json_calls")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._create_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        raw_text = self._extract_content(response).strip()
        parsed = safe_parse_json(raw_text, fallback=fallback)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from LLM, got: {raw_text}")
        return parsed, raw_text

    def infer_text(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.infer_messages_text(messages=messages, fallback=fallback, max_tokens=max_tokens)

    def infer_messages_text(
        self,
        messages: List[Dict[str, str]],
        fallback: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        self._inc_stat("text_calls")
        try:
            response = self._create_completion(messages=messages, max_tokens=max_tokens)
            return self._extract_content(response).strip()
        except Exception as exc:
            logger.warning("LLM text call failed, using fallback. Error: %s", exc)
            return fallback

    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self.stats)

    def reset_stats(self) -> None:
        with self._stats_lock:
            for key in self.stats:
                self.stats[key] = 0
