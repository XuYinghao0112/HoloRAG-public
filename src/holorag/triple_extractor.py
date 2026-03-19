import re
from typing import Dict, List

from .llm_client import LocalLLMClient
from .utils import QUESTION_WORDS, clean_entity_text, is_generic_entity, looks_like_named_entity


class TripleExtractor:
    def __init__(self, llm_client: LocalLLMClient) -> None:
        self.llm_client = llm_client

    def extract(self, sentence: str) -> Dict[str, List]:
        fallback = self._heuristic_extract(sentence)
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Extract concise knowledge triples from the sentence. "
                "Return JSON with keys triples and entities. "
                "triples must be a list of objects with keys head, relation, tail. "
                "entities must contain clean named entities only."
            ),
            user_prompt=f"Sentence:\n{sentence}",
            fallback=fallback,
            max_tokens=256,
        )
        triples = self._clean_triples(payload.get("triples", []), fallback["triples"])
        entities = self._clean_entities(payload.get("entities", []), fallback["entities"])
        pattern_result = self._pattern_extract(sentence)
        triples = self._merge_triples(triples, pattern_result["triples"])
        entities = self._merge_entities(entities, pattern_result["entities"])
        return {"triples": triples, "entities": entities}

    def extract_query_entities(self, query: str) -> List[str]:
        fallback = {"entities": self._heuristic_query_entities(query)}
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Extract only explicit entity mentions from the question. "
                "Do not include question words, partial phrases, or generic roles. "
                "Return JSON with key entities."
            ),
            user_prompt=f"Question:\n{query}",
            fallback=fallback,
            max_tokens=96,
        )
        return self._clean_entities(payload.get("entities", []), fallback["entities"])

    def _clean_triples(self, triples: List[Dict], fallback: List[Dict]) -> List[Dict]:
        cleaned = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            head = clean_entity_text(str(triple.get("head", "")))
            relation = str(triple.get("relation", "")).strip() or "related_to"
            tail = clean_entity_text(str(triple.get("tail", "")))
            if (
                head and tail
                and looks_like_named_entity(head)
                and looks_like_named_entity(tail)
                and not is_generic_entity(head)
                and not is_generic_entity(tail)
            ):
                cleaned.append({"head": head, "relation": relation, "tail": tail})
        return cleaned or fallback

    def _clean_entities(self, entities: List[str], fallback: List[str]) -> List[str]:
        cleaned: List[str] = []
        for entity in entities:
            normalized = clean_entity_text(str(entity))
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in QUESTION_WORDS:
                continue
            if looks_like_named_entity(normalized) and not is_generic_entity(normalized):
                cleaned.append(normalized)
        deduped = list(dict.fromkeys(cleaned))
        return deduped or fallback

    def _heuristic_extract(self, sentence: str) -> Dict[str, List]:
        entities = self._heuristic_sentence_entities(sentence)
        triples = []
        if len(entities) >= 2:
            for left, right in zip(entities, entities[1:]):
                triples.append({"head": left, "relation": "related_to", "tail": right})
        return {"triples": triples[:3], "entities": entities}

    def _pattern_extract(self, sentence: str) -> Dict[str, List]:
        triples: List[Dict[str, str]] = []
        entities: List[str] = []
        normalized = " ".join(sentence.split())
        entity_pattern = (
            r"[A-Z][A-Za-z0-9&\.\-']*"
            r"(?:\s+\([^)]+\))?"
            r"(?:\s+[A-Z][A-Za-z0-9&\.\-']*(?:\s+\([^)]+\))?){0,5}"
        )

        governor_match = re.match(
            r"^\s*([A-Z][A-Za-z\.\-']+(?:\s+[A-Z][A-Za-z\.\-']+){1,4})\s*\([^)]*\)\s+was\b.*?\bGovernor of ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})",
            normalized,
        )
        if governor_match:
            person = clean_entity_text(governor_match.group(1))
            state = clean_entity_text(governor_match.group(2))
            if person and state:
                triples.append({"head": person, "relation": "governor_of", "tail": state})
                entities.extend([person, state])

        statehood_match = re.search(
            r"\bOn ([A-Z][a-z]+ \d{1,2}, \d{4}), ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}) became\b",
            normalized,
        )
        if statehood_match:
            date = clean_entity_text(statehood_match.group(1))
            state = clean_entity_text(statehood_match.group(2))
            if state and date:
                triples.append({"head": state, "relation": "statehood_date", "tail": date})
                entities.extend([state, date])

        part_of_us_match = re.search(
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b.*?\bbecom(?:e|ing)\b.*?\bpart of the United States\b(?:\s+in\s+([A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}))?",
            normalized,
        )
        if part_of_us_match:
            place = clean_entity_text(part_of_us_match.group(1))
            date = clean_entity_text(part_of_us_match.group(2) or "")
            if place:
                if date:
                    triples.append({"head": place, "relation": "became_part_of_united_states_on", "tail": date})
                    entities.extend([place, date])
                else:
                    entities.append(place)

        succession_patterns = [
            rf"\b({entity_pattern})\b,\s+later to become\b\s+({entity_pattern})",
            rf"\b({entity_pattern})\b\s+later became\s+({entity_pattern})",
            rf"\b({entity_pattern})\b\s+became\s+({entity_pattern})",
        ]
        for pattern in succession_patterns:
            for match in re.finditer(pattern, normalized):
                earlier = clean_entity_text(match.group(1))
                later = clean_entity_text(match.group(2))
                if earlier and later and earlier.lower() != later.lower():
                    triples.append({"head": earlier, "relation": "became", "tail": later})
                    triples.append({"head": later, "relation": "formerly", "tail": earlier})
                    entities.extend([earlier, later])

        for expanded in self._expand_enumeration_triples(normalized):
            triples.append(expanded)
            entities.extend([expanded["head"], expanded["tail"]])

        return {"triples": self._clean_triples(triples, []), "entities": self._clean_entities(entities, [])}

    def _expand_enumeration_triples(self, sentence: str) -> List[Dict[str, str]]:
        entity_pattern = (
            r"[A-Z][A-Za-z0-9&\.\-']*"
            r"(?:\s+\([^)]+\))?"
            r"(?:\s+[A-Z][A-Za-z0-9&\.\-']*(?:\s+\([^)]+\))?){0,5}"
        )
        relation_patterns = [
            "consisting of",
            "consisted of",
            "consists of",
            "composed of",
            "comprises",
            "comprised",
            "includes",
            "included",
            "featuring",
            "featured",
            "has members",
            "had members",
            "with members",
            "member of",
            "members include",
            "lineup consisted of",
            "lineup includes",
        ]
        triples: List[Dict[str, str]] = []
        lowered = sentence.lower()
        for relation_phrase in relation_patterns:
            marker = f" {relation_phrase} "
            start = lowered.find(marker)
            if start < 0:
                continue
            head_text = sentence[:start].strip(" ,;:-")
            tail_text = sentence[start + len(marker):].strip(" .")
            subject_prefix = re.split(r"\b(?:is|was|were|are)\b", head_text, maxsplit=1)[0].strip(" ,;:-")
            head_matches = re.findall(entity_pattern, subject_prefix) or re.findall(entity_pattern, head_text)
            member_matches = re.findall(entity_pattern, tail_text)
            if not head_matches or len(member_matches) < 2:
                continue
            head = clean_entity_text(head_matches[0])
            members = []
            merged_member_matches = self._merge_member_suffixes(member_matches)
            for match in merged_member_matches:
                cleaned = clean_entity_text(match)
                if not cleaned or cleaned == head:
                    continue
                members.append(cleaned)
            seen = set()
            deduped_members = []
            for member in members:
                key = member.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped_members.append(member)
            canonical_relation = self._canonicalize_relation_phrase(relation_phrase)
            for member in deduped_members:
                triples.append({"head": head, "relation": canonical_relation, "tail": member})
                if canonical_relation == "has_member":
                    triples.append({"head": member, "relation": "member_of", "tail": head})
            if triples:
                break
        return triples

    def _merge_member_suffixes(self, members: List[str]) -> List[str]:
        suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
        merged: List[str] = []
        for member in members:
            cleaned = clean_entity_text(member)
            if not cleaned:
                continue
            if cleaned.lower() in suffixes and merged:
                merged[-1] = f"{merged[-1]} {cleaned}"
            else:
                merged.append(cleaned)
        return merged

    def _canonicalize_relation_phrase(self, relation_phrase: str) -> str:
        lowered = relation_phrase.lower()
        if any(token in lowered for token in ["member", "lineup", "consist", "compose", "comprise"]):
            return "has_member"
        if any(token in lowered for token in ["include", "feature"]):
            return "includes"
        return lowered.replace(" ", "_")

    def _merge_triples(self, left: List[Dict], right: List[Dict]) -> List[Dict]:
        merged: List[Dict] = []
        seen = set()
        for triple in left + right:
            key = (triple["head"], triple["relation"], triple["tail"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(triple)
        return merged

    def _merge_entities(self, left: List[str], right: List[str]) -> List[str]:
        merged = []
        for entity in left + right:
            if entity not in merged:
                merged.append(entity)
        return merged

    def _heuristic_sentence_entities(self, text: str) -> List[str]:
        candidates = re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b", text)
        cleaned = [clean_entity_text(candidate) for candidate in candidates]
        return [item for item in dict.fromkeys(cleaned) if looks_like_named_entity(item)]

    def _heuristic_query_entities(self, query: str) -> List[str]:
        title_case = re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b", query)
        cleaned = []
        for candidate in title_case:
            normalized = clean_entity_text(candidate)
            if looks_like_named_entity(normalized):
                cleaned.append(normalized)
        if cleaned:
            return list(dict.fromkeys(cleaned))

        organization_like = re.findall(r"\b(?:[A-Z][a-z0-9]+)\b", query)
        fallback = []
        for token in organization_like:
            normalized = clean_entity_text(token)
            if looks_like_named_entity(normalized):
                fallback.append(normalized)
        return list(dict.fromkeys(fallback))
