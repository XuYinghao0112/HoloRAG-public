import json
import math
import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

QUESTION_WORDS = {
    "which", "what", "who", "whom", "whose", "when", "where", "why", "how",
    "tell", "list", "show", "name", "give", "find", "identify",
}
LIGHT_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "do", "does", "did", "and", "or", "of", "to", "for", "in", "on", "at", "with",
}
GENERIC_ENTITY_TERMS = {
    "he", "she", "they", "them", "him", "her", "his", "hers", "their", "theirs",
    "it", "its", "state", "states", "city", "country", "governor", "president",
    "american", "union", "confederate", "civil", "civil war", "war", "u s", "us",
    "united states", "currently", "introduction",
}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stable_node_id(prefix: str, *parts: str) -> str:
    normalized = "||".join(str(part).strip() for part in parts)
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return f"{prefix}:{clean[:120] or 'node'}"


def cosine_similarity_matrix(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    if len(docs) == 0:
        return np.array([])
    query = np.asarray(query, dtype=np.float32)
    docs = np.asarray(docs, dtype=np.float32)
    query_norm = np.linalg.norm(query) + 1e-8
    doc_norms = np.linalg.norm(docs, axis=1) + 1e-8
    return (docs @ query) / (doc_norms * query_norm)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    positives = {key: max(float(value), 0.0) for key, value in scores.items()}
    total = sum(positives.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in positives.items() if value > 0}


def top_k(items: Sequence[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    return sorted(items, key=lambda item: item[1], reverse=True)[:k]


def safe_parse_json(text: str, fallback: Any) -> Any:
    stripped = str(text or "").strip()
    candidates = [stripped]
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE))
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if 0 <= start < end:
            candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return fallback


def split_words(text: str) -> List[str]:
    return re.findall(r"\S+", str(text or ""))


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    words = split_words(text)
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [" ".join(words)]
    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size_words]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def dump_pickle(path: str, payload: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def clean_entity_text(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", str(text or ""))
    while tokens and tokens[0].lower() in QUESTION_WORDS | LIGHT_STOPWORDS:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in LIGHT_STOPWORDS:
        tokens.pop()
    cleaned = re.sub(r"\s+", " ", " ".join(tokens)).strip(" ,.;:!?")
    return cleaned if len(cleaned) >= 2 else ""


def normalize_entity_key(text: str) -> str:
    normalized = clean_entity_text(text)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    normalized = re.sub(r"\b(?:a|an|the)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def is_generic_entity(text: str) -> bool:
    normalized = re.sub(r"[\.\-_/]+", " ", str(text or "").lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized or normalized in GENERIC_ENTITY_TERMS:
        return True
    parts = normalized.split()
    return bool(parts and all(len(part) == 1 for part in parts))


def looks_like_named_entity(text: str) -> bool:
    if not text or is_generic_entity(text):
        return False
    tokens = str(text).split()
    lowered = [token.lower() for token in tokens]
    if any(token in QUESTION_WORDS for token in lowered):
        return False
    if len(tokens) == 1:
        return tokens[0][:1].isupper() or tokens[0].isupper()
    return any(token[:1].isupper() for token in tokens)


def lexical_overlap_score(query: str, text: str) -> float:
    q_terms = {token for token in re.findall(r"\w+", str(query).lower()) if len(token) > 2}
    t_terms = {token for token in re.findall(r"\w+", str(text).lower()) if len(token) > 2}
    if not q_terms or not t_terms:
        return 0.0
    return len(q_terms & t_terms) / max(1, len(q_terms))


def merge_weighted_scores(target: Dict[str, float], scores: Dict[str, float], weight: float) -> None:
    for key, value in scores.items():
        target[key] = target.get(key, 0.0) + weight * float(value)


def normalize_alpha(alpha: Dict[str, float]) -> Dict[str, float]:
    """Normalize the public granularity profile to fact/sentence/chunk.

    Legacy four-way configs may still contain entity/alpha_E. Entity remains a
    graph anchor, so its old mass is folded into fact rather than kept as a
    final evidence granularity.
    """
    keys = ["fact", "sentence", "chunk"]
    values = {key: max(float(alpha.get(key, alpha.get(f"alpha_{key[0].upper()}", 0.0))), 0.0) for key in keys}
    legacy_entity = max(float(alpha.get("entity", alpha.get("alpha_E", 0.0))), 0.0)
    values["fact"] += legacy_entity
    total = sum(values.values()) or 1.0
    return {key: values[key] / total for key in keys}


def entropy_confidence(alpha: Dict[str, float]) -> float:
    values = [value for value in normalize_alpha(alpha).values() if value > 0]
    if not values:
        return 0.0
    entropy = -sum(value * math.log(value) for value in values)
    return max(0.0, 1.0 - entropy / math.log(3))
