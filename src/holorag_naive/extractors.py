import re
from typing import Dict, List

from .llm_client import LocalLLMClient
from .utils import clean_entity_text, is_generic_entity, looks_like_named_entity


class TripleExtractor:
    def __init__(self, llm_client: LocalLLMClient, index_extraction_mode: str = "heuristic") -> None:
        self.llm_client = llm_client
        self.index_extraction_mode = index_extraction_mode
        self._sentence_cache: Dict[str, Dict[str, List]] = {}
        self._query_cache: Dict[str, Dict[str, List]] = {}

    def extract_sentence(self, sentence: str) -> Dict[str, List]:
        cache_key = " ".join(str(sentence or "").split())
        if cache_key in self._sentence_cache:
            return {
                "triples": [dict(item) for item in self._sentence_cache[cache_key].get("triples", [])],
                "entities": list(self._sentence_cache[cache_key].get("entities", [])),
            }
        fallback = self._heuristic_extract(sentence)
        if self.index_extraction_mode == "heuristic":
            self._sentence_cache[cache_key] = fallback
            return {"triples": [dict(item) for item in fallback["triples"]], "entities": list(fallback["entities"])}
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Extract concise factual triples and named entities from the sentence. "
                "Return JSON with keys triples and entities. "
                "triples is a list of objects with head, relation, tail."
            ),
            user_prompt=f"Sentence:\n{sentence}",
            fallback=fallback,
            max_tokens=256,
        )
        triples = self._clean_triples(payload.get("triples", []), fallback["triples"])
        entities = self._clean_entities(payload.get("entities", []), fallback["entities"])
        result = {"triples": triples, "entities": entities}
        self._sentence_cache[cache_key] = result
        return {"triples": [dict(item) for item in triples], "entities": list(entities)}

    def extract_query(self, query: str) -> Dict[str, List]:
        cache_key = " ".join(str(query or "").split())
        if cache_key in self._query_cache:
            return {
                "triples": [dict(item) for item in self._query_cache[cache_key].get("triples", [])],
                "entities": list(self._query_cache[cache_key].get("entities", [])),
            }
        fallback = self._heuristic_extract(query)
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Extract explicit entity mentions and relation-like factual constraints from the question. "
                "Return JSON with keys triples and entities. "
                "For unknown answer slots, omit the answer entity rather than inventing it."
            ),
            user_prompt=f"Question:\n{query}",
            fallback=fallback,
            max_tokens=192,
        )
        triples = self._clean_triples(payload.get("triples", []), fallback["triples"])
        entities = self._clean_entities(payload.get("entities", []), fallback["entities"])
        result = {"triples": triples, "entities": entities}
        self._query_cache[cache_key] = result
        return {"triples": [dict(item) for item in triples], "entities": list(entities)}

    def _heuristic_extract(self, text: str) -> Dict[str, List]:
        entities = []
        pattern = r"[A-Z][A-Za-z0-9&\.\-']*(?:\s+[A-Z][A-Za-z0-9&\.\-']*){0,5}"
        for match in re.findall(pattern, str(text or "")):
            cleaned = clean_entity_text(match)
            if cleaned and looks_like_named_entity(cleaned) and not is_generic_entity(cleaned):
                entities.append(cleaned)
        entities = list(dict.fromkeys(entities))
        triples = []
        if len(entities) >= 2:
            for left, right in zip(entities, entities[1:]):
                triples.append({"head": left, "relation": "related_to", "tail": right})
        return {"triples": triples[:3], "entities": entities}

    def _clean_entities(self, entities: List[str], fallback: List[str]) -> List[str]:
        cleaned = []
        for entity in entities:
            normalized = clean_entity_text(str(entity))
            if normalized and looks_like_named_entity(normalized) and not is_generic_entity(normalized):
                cleaned.append(normalized)
        return list(dict.fromkeys(cleaned)) or list(fallback)

    def _clean_triples(self, triples: List[Dict], fallback: List[Dict]) -> List[Dict]:
        cleaned = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            head = clean_entity_text(str(triple.get("head", "")))
            relation = str(triple.get("relation", "")).strip() or "related_to"
            tail = clean_entity_text(str(triple.get("tail", "")))
            if head and tail and not is_generic_entity(head) and not is_generic_entity(tail):
                cleaned.append({"head": head, "relation": relation, "tail": tail})
        return cleaned or list(fallback)


class QueryDecomposer:
    def __init__(self, llm_client: LocalLLMClient) -> None:
        self.llm_client = llm_client
        self._cache: Dict[str, List[str]] = {}

    def decompose(self, query: str) -> List[str]:
        cache_key = " ".join(str(query or "").split())
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        fallback = {"sub_questions": self._heuristic_decompose(query)}
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Break the multi-hop query into 1 to 4 atomic retrieval sub-questions. "
                "Preserve dependency order and keep each sub-question close to the original wording. "
                "Return JSON with key sub_questions."
            ),
            user_prompt=f"Query:\n{query}",
            fallback=fallback,
            max_tokens=160,
        )
        raw = payload.get("sub_questions", [])
        sub_questions = []
        seen = set()
        for item in raw:
            question = " ".join(str(item).strip().split())
            if len(question.split()) < 3:
                continue
            if not question.endswith("?"):
                question += "?"
            key = question.lower().rstrip("?")
            if key not in seen:
                seen.add(key)
                sub_questions.append(question)
        resolved = sub_questions[:4] or fallback["sub_questions"]
        self._cache[cache_key] = list(resolved)
        return resolved

    def _heuristic_decompose(self, query: str) -> List[str]:
        normalized = " ".join(str(query or "").strip().split())
        if not normalized:
            return []
        parts = re.split(r"\s+(?:and then|then|after|before|while)\s+", normalized, flags=re.IGNORECASE)
        cleaned = [part.strip(" ?") + "?" for part in parts if len(part.strip().split()) >= 3]
        if len(cleaned) > 1:
            return cleaned[:4]
        if " which " in normalized.lower() and "," in normalized:
            prefix, suffix = normalized.rsplit(",", 1)
            return [prefix.strip(" ?") + "?", suffix.strip(" ?") + "?"]
        return [normalized.rstrip("?") + "?"]
