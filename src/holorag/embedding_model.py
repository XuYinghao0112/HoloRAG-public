import logging
import gc
from typing import List

import numpy as np
import torch
from transformers import AutoModel

from .config import HoloRAGConfig

logger = logging.getLogger(__name__)


class NVEmbedV2Encoder:
    def __init__(self, config: HoloRAGConfig) -> None:
        self.config = config
        logger.info("Loading embedding model: %s", config.embedding_model_name)
        normalized_device = self._normalize_device(config.embedding_device)
        self.embedding_device = normalized_device
        self.torch_dtype = self._resolve_dtype(config.embedding_dtype, normalized_device)
        model_kwargs = {
            "pretrained_model_name_or_path": config.embedding_model_name,
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
        }
        if normalized_device == "auto":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(**model_kwargs)
        self.model.eval()
        if normalized_device != "auto":
            self.model.to(normalized_device)

    def encode(self, texts: List[str], instruction: str = "", text_type: str = "query", max_length: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        prompt_instruction = f"Instruct: {instruction}\nQuery: " if instruction else ""
        resolved_max_length = max_length or self._resolve_max_length(text_type)
        embeddings = self._encode_with_backoff(
            texts=texts,
            instruction=prompt_instruction,
            batch_size=self.config.embedding_batch_size,
            max_length=resolved_max_length,
        )
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / norms

    def _encode_with_backoff(self, texts: List[str], instruction: str, batch_size: int, max_length: int):
        try:
            return self._encode_in_batches(
                texts=texts,
                instruction=instruction,
                batch_size=max(1, batch_size),
                max_length=max_length,
            )
        except torch.OutOfMemoryError:
            self._cleanup_cuda()
            if batch_size <= 1:
                raise
            next_batch_size = max(1, batch_size // 2)
            logger.warning(
                "Embedding OOM at batch_size=%s. Retrying with batch_size=%s.",
                batch_size,
                next_batch_size,
            )
            return self._encode_with_backoff(
                texts=texts,
                instruction=instruction,
                batch_size=next_batch_size,
                max_length=max_length,
            )

    def _encode_in_batches(self, texts: List[str], instruction: str, batch_size: int, max_length: int) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            with torch.inference_mode():
                embeddings = self.model.encode(
                    prompts=batch_texts,
                    instruction=instruction,
                    max_length=max_length,
                    batch_size=len(batch_texts),
                    num_workers=0,
                )
            if isinstance(embeddings, torch.Tensor):
                batch_array = embeddings.detach().cpu().numpy()
            else:
                batch_array = np.asarray(embeddings)
            all_embeddings.append(np.asarray(batch_array, dtype=np.float32))
            self._cleanup_cuda()
        return np.vstack(all_embeddings)

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_max_length(self, text_type: str) -> int:
        normalized = (text_type or "query").strip().lower()
        if normalized == "entity":
            return min(self.config.entity_max_length, self.config.embedding_max_seq_len)
        if normalized == "sentence":
            return min(self.config.sentence_max_length, self.config.embedding_max_seq_len)
        if normalized == "chunk":
            return min(self.config.chunk_max_length, self.config.embedding_max_seq_len)
        if normalized in {"query", "sub_question"}:
            return min(self.config.query_max_length, self.config.embedding_max_seq_len)
        return self.config.embedding_max_seq_len

    def _resolve_dtype(self, raw_dtype: str, device: str):
        dtype = (raw_dtype or "").strip().lower()
        if device == "cpu":
            return torch.float32
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _normalize_device(self, raw_device: str) -> str:
        device = (raw_device or "auto").strip().lower()
        if device.startswith("gpu"):
            suffix = device[3:]
            if suffix.isdigit():
                device = f"cuda:{suffix}"
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda:"):
            try:
                requested_index = int(device.split(":", 1)[1])
            except ValueError:
                return "auto"
            if not torch.cuda.is_available():
                logger.warning("Requested embedding device %s but CUDA is unavailable. Falling back to cpu.", raw_device)
                return "cpu"
            if requested_index >= torch.cuda.device_count():
                logger.warning(
                    "Requested embedding device %s but only %s CUDA device(s) are visible. Falling back to cuda:0.",
                    raw_device,
                    torch.cuda.device_count(),
                )
                return "cuda:0"
        return device
