import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import HoloRAGConfig
from .utils import safe_parse_json

logger = logging.getLogger(__name__)


class LocalLLMClient:
    def __init__(self, config: HoloRAGConfig) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "sk-")
        self.config = config
        self.client = OpenAI(base_url=config.llm_base_url, api_key=api_key, max_retries=3)
        self.stats = {
            "completion_calls": 0,
            "json_calls": 0,
            "text_calls": 0,
        }

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
            self.stats["completion_calls"] += 1
            return self.client.chat.completions.create(**params)
        except TypeError:
            params.pop("response_format", None)
            self.stats["completion_calls"] += 1
            return self.client.chat.completions.create(**params)

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
        self.stats["json_calls"] += 1
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
        self.stats["text_calls"] += 1
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._create_completion(messages=messages, max_tokens=max_tokens)
            return self._extract_content(response).strip()
        except Exception as exc:
            logger.warning("LLM text call failed, using fallback. Error: %s", exc)
            return fallback

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0
