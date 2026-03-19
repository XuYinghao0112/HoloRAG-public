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
    "american", "union", "confederate", "civil", "civil war", "war", "u s", "us", "u.s", "u.s.",
    "united states", "north", "south", "east", "west", "currently", "introduction",
}
DISAMBIGUATION_PARENTHESES = {
    "film", "movie", "song", "album", "novel", "book", "poem", "play", "band", "tv", "tv series",
    "television series", "episode", "character", "documentary", "series", "magazine", "journal",
    "newspaper", "video game", "game", "software", "company", "organization", "missile", "weapon",
    "rocket", "vehicle", "ship", "aircraft", "station", "radio station", "university",
}


def _is_alias_like_parenthetical(text: str) -> bool:
    cleaned = clean_entity_text(text)
    if not cleaned:
        return False
    normalized = normalize_entity_key(cleaned)
    if normalized in DISAMBIGUATION_PARENTHESES:
        return False
    if re.fullmatch(r"[A-Z][A-Za-z0-9&.\-]{1,9}", cleaned):
        return True
    if 2 <= len(cleaned) <= 10 and any(char.isupper() for char in cleaned):
        return True
    return looks_like_named_entity(cleaned) and not is_generic_entity(cleaned)


def strip_parenthetical_text(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", " ", str(text or ""))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stable_node_id(prefix: str, *parts: str) -> str:
    normalized = "||".join(part.strip() for part in parts)
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


def top_k(items: Sequence[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    return sorted(items, key=lambda item: item[1], reverse=True)[:k]


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {key: value / total for key, value in exp_scores.items()}


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def lexical_overlap_score(query: str, text: str) -> float:
    q_terms = {token for token in re.findall(r"\w+", query.lower()) if len(token) > 2}
    t_terms = {token for token in re.findall(r"\w+", text.lower()) if len(token) > 2}
    if not q_terms or not t_terms:
        return 0.5
    return 0.2 + 0.8 * (len(q_terms & t_terms) / len(q_terms))


def extract_json_candidates(text: str) -> List[str]:
    stripped = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates = [match.strip() for match in fenced if match.strip()]
    candidates.append(stripped)

    stack = []
    start = None
    for idx, char in enumerate(stripped):
        if char in "[{":
            if not stack:
                start = idx
            stack.append(char)
        elif char in "]}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(stripped[start:idx + 1])
                    start = None
    return candidates


def safe_parse_json(text: str, fallback: Any) -> Any:
    for candidate in extract_json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return fallback


def split_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


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
        if not window:
            continue
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


def clean_entity_text(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    while tokens and tokens[0].lower() in QUESTION_WORDS | LIGHT_STOPWORDS:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in LIGHT_STOPWORDS:
        tokens.pop()
    if not tokens:
        return ""
    cleaned = " ".join(tokens)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:!?")
    if len(cleaned) < 2:
        return ""
    return cleaned


def normalize_entity_key(text: str) -> str:
    normalized = clean_entity_text(text)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    normalized = re.sub(r"\b(?:a|an|the)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def generate_entity_aliases(text: str) -> List[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    parenthetical_matches = re.findall(r"\(([^)]{1,40})\)", raw_text)
    stripped_text = strip_parenthetical_text(raw_text)
    candidates: List[str] = [
        stripped_text,
        clean_entity_text(stripped_text),
    ]

    if not parenthetical_matches or all(_is_alias_like_parenthetical(match) for match in parenthetical_matches):
        candidates.extend([raw_text, clean_entity_text(raw_text)])

    for match in parenthetical_matches:
        if _is_alias_like_parenthetical(match):
            candidates.append(clean_entity_text(match))

    acronym_source = clean_entity_text(strip_parenthetical_text(raw_text))
    acronym_tokens = [token for token in acronym_source.split() if token and token[0].isalnum()]
    acronym = "".join(token[0] for token in acronym_tokens if token[0].isalnum())
    if 2 <= len(acronym) <= 8:
        candidates.append(acronym.upper())

    cleaned_source = clean_entity_text(raw_text)
    cleaned_tokens = cleaned_source.split()
    if cleaned_tokens:
        connector_tokens = {"now", "formerly", "later", "current", "currently"}
        for connector in connector_tokens:
            if connector in {token.lower() for token in cleaned_tokens}:
                prefix_tokens = []
                for token in cleaned_tokens:
                    if token.lower() == connector:
                        break
                    prefix_tokens.append(token)
                if len(prefix_tokens) >= 2:
                    candidates.append(" ".join(prefix_tokens))
        abbreviation_filtered = [
            token for token in cleaned_tokens
            if not (len(token) <= 5 and any(char.isupper() for char in token[1:]))
        ]
        if len(abbreviation_filtered) >= 2 and len(abbreviation_filtered) < len(cleaned_tokens):
            candidates.append(" ".join(abbreviation_filtered))

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        cleaned = clean_entity_text(candidate)
        key = normalize_entity_key(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def entity_matches_title(entity_text: str, title: str) -> bool:
    entity_keys = {normalize_entity_key(alias) for alias in generate_entity_aliases(entity_text)}
    title_keys = {normalize_entity_key(alias) for alias in generate_entity_aliases(title)}
    entity_keys.discard("")
    title_keys.discard("")
    return bool(entity_keys & title_keys)


def entity_match_score(left: str, right: str) -> float:
    left_keys = {normalize_entity_key(alias) for alias in generate_entity_aliases(left)}
    right_keys = {normalize_entity_key(alias) for alias in generate_entity_aliases(right)}
    left_keys.discard("")
    right_keys.discard("")
    if not left_keys or not right_keys:
        return 0.0
    if left_keys & right_keys:
        return 1.0
    token_overlap = 0.0
    for left_key in left_keys:
        left_tokens = set(left_key.split())
        if not left_tokens:
            continue
        for right_key in right_keys:
            right_tokens = set(right_key.split())
            if not right_tokens:
                continue
            token_overlap = max(token_overlap, len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens)))
    return token_overlap


def text_contains_entity(text: str, entity_text: str) -> bool:
    normalized_texts = {
        normalize_entity_key(text),
        normalize_entity_key(strip_parenthetical_text(text)),
    }
    entity_keys = {normalize_entity_key(alias) for alias in generate_entity_aliases(entity_text)}
    normalized_texts.discard("")
    entity_keys.discard("")
    for entity_key in entity_keys:
        padded_entity = f" {entity_key} "
        for normalized_text in normalized_texts:
            padded_text = f" {normalized_text} "
            if padded_entity in padded_text:
                return True
    return False


def is_generic_entity(text: str) -> bool:
    normalized = re.sub(r"[\.\-_/]+", " ", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return True
    if normalized in GENERIC_ENTITY_TERMS:
        return True
    parts = normalized.split()
    if parts and all(len(part) == 1 for part in parts):
        return True
    if len(parts) == 1 and parts[0] in QUESTION_WORDS | LIGHT_STOPWORDS:
        return True
    return False


def looks_like_named_entity(text: str) -> bool:
    if not text:
        return False
    if is_generic_entity(text):
        return False
    tokens = text.split()
    lowered = [token.lower() for token in tokens]
    if any(token in QUESTION_WORDS for token in lowered):
        return False
    if len(tokens) > 0 and all(len(token) == 1 for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        return token[:1].isupper() or token.isupper()
    return any(token[:1].isupper() for token in tokens)
