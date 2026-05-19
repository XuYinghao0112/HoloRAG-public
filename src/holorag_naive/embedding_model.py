import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel

from .config import NaiveHoloRAGConfig

logger = logging.getLogger(__name__)


class NVEmbedV2Encoder:
    def __init__(self, config: NaiveHoloRAGConfig) -> None:
        self.config = config
        self.embedding_device = self._normalize_device(config.embedding_device)
        self.torch_dtype = self._resolve_dtype(config.embedding_dtype, self.embedding_device)
        logger.info("Loading embedding model: %s on %s", config.embedding_model_name, self.embedding_device)
        kwargs = {
            "pretrained_model_name_or_path": config.embedding_model_name,
            "trust_remote_code": True,
        }
        if self.embedding_device == "auto":
            kwargs["device_map"] = "auto"
        try:
            self.model = AutoModel.from_pretrained(**kwargs, dtype=self.torch_dtype)
        except TypeError:
            self.model = AutoModel.from_pretrained(**kwargs, torch_dtype=self.torch_dtype)
        self.model.eval()
        if self.embedding_device != "auto":
            self.model.to(self.embedding_device)

    def encode(
        self,
        texts: List[str],
        instruction: str = "",
        text_type: str = "query",
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        resolved_max_length = max_length or self._resolve_max_length(text_type)
        prompt_instruction = f"Instruct: {instruction}\nQuery: " if instruction else ""
        embeddings = self._encode_with_backoff(texts, prompt_instruction, self.config.embedding_batch_size, resolved_max_length)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        array = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-8
        return array / norms

    def _encode_with_backoff(self, texts: List[str], instruction: str, batch_size: int, max_length: int):
        try:
            return self._encode_batches(texts, instruction, max(1, batch_size), max_length)
        except torch.OutOfMemoryError:
            if batch_size <= 1:
                raise
            return self._encode_with_backoff(texts, instruction, batch_size // 2, max_length)

    def _encode_batches(self, texts: List[str], instruction: str, batch_size: int, max_length: int) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            with torch.inference_mode():
                encoded = self.model.encode(
                    prompts=batch,
                    instruction=instruction,
                    max_length=max_length,
                    batch_size=len(batch),
                    num_workers=0,
                )
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.detach().cpu().numpy()
            arrays.append(np.asarray(encoded, dtype=np.float32))
        return np.vstack(arrays)

    def _resolve_max_length(self, text_type: str) -> int:
        normalized = (text_type or "query").lower()
        if normalized == "entity":
            return min(self.config.entity_max_length, self.config.embedding_max_seq_len)
        if normalized == "fact":
            return min(self.config.fact_max_length, self.config.embedding_max_seq_len)
        if normalized == "sentence":
            return min(self.config.sentence_max_length, self.config.embedding_max_seq_len)
        if normalized == "chunk":
            return min(self.config.chunk_max_length, self.config.embedding_max_seq_len)
        return min(self.config.query_max_length, self.config.embedding_max_seq_len)

    def _resolve_dtype(self, raw_dtype: str, device: str):
        dtype = (raw_dtype or "").lower()
        if device == "cpu":
            return torch.float32
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def _normalize_device(self, raw_device: str) -> str:
        device = (raw_device or "auto").strip().lower()
        if device.startswith("gpu") and device[3:].isdigit():
            device = f"cuda:{device[3:]}"
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda:") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to cpu.")
            return "cpu"
        return device
