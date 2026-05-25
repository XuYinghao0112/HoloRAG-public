import logging
import re
from typing import Dict, List, Optional, Tuple

from .llm_client import LocalLLMClient
from .utils import clean_entity_text, is_generic_entity, looks_like_named_entity

logger = logging.getLogger(__name__)


class TripleExtractor:
    SPACY_ENTITY_LABELS = {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
        "QUANTITY", "ORDINAL", "CARDINAL",
    }

    def __init__(self, llm_client: LocalLLMClient, index_extraction_mode: str = "heuristic") -> None:
        self.llm_client = llm_client
        self.index_extraction_mode = index_extraction_mode
        self._sentence_cache: Dict[str, Dict[str, List]] = {}
        self._query_cache: Dict[str, Dict[str, List]] = {}
        self._spacy_nlp = self._load_spacy_model()

    def extract_sentence(self, sentence: str) -> Dict[str, List]:
        cache_key = " ".join(str(sentence or "").split())
        if cache_key in self._sentence_cache:
            return {
                "triples": [dict(item) for item in self._sentence_cache[cache_key].get("triples", [])],
                "entities": list(self._sentence_cache[cache_key].get("entities", [])),
            }
        fallback = self._spacy_extract(sentence) or self._heuristic_extract(sentence)
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
        fallback = self._spacy_extract(query) or self._heuristic_extract(query)
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

    def _load_spacy_model(self):
        try:
            import spacy
        except Exception as exc:
            logger.warning("spaCy is not available; using regex entity/fact fallback. Error: %s", exc)
            return None
        model_name = getattr(self.llm_client.config, "spacy_model_name", "en_core_web_sm")
        try:
            return spacy.load(model_name)
        except Exception as exc:
            logger.warning(
                "Could not load spaCy model %s; using regex entity/fact fallback. Error: %s",
                model_name,
                exc,
            )
            return None

    def _spacy_extract(self, text: str) -> Optional[Dict[str, List]]:
        if self._spacy_nlp is None or not str(text or "").strip():
            return None
        doc = self._spacy_nlp(str(text))
        mentions = self._spacy_entity_mentions(doc)
        entities = list(dict.fromkeys(item["text"] for item in mentions))
        triples = self._dependency_triples(doc, mentions)
        if not entities:
            return None
        return {"triples": triples[:8], "entities": entities}

    def _spacy_entity_mentions(self, doc) -> List[Dict]:
        mentions: List[Dict] = []
        seen = set()

        def add_span(span, label: str = "") -> None:
            text = clean_entity_text(span.text)
            if not text or is_generic_entity(text):
                return
            if not looks_like_named_entity(text) and label not in {"DATE", "CARDINAL", "ORDINAL"}:
                return
            key = (span.start, span.end, text.lower())
            if key in seen:
                return
            seen.add(key)
            mentions.append({"text": text, "start": span.start, "end": span.end, "root": span.root, "label": label})

        for ent in doc.ents:
            if ent.label_ in self.SPACY_ENTITY_LABELS:
                add_span(ent, ent.label_)

        try:
            noun_chunks = list(doc.noun_chunks)
        except Exception:
            noun_chunks = []
        for chunk in noun_chunks:
            raw = chunk.text.strip()
            if 1 <= len(raw.split()) <= 6 and looks_like_named_entity(clean_entity_text(raw)):
                add_span(chunk, "NOUN_CHUNK")
        return mentions

    def _dependency_triples(self, doc, mentions: List[Dict]) -> List[Dict]:
        triples: List[Dict] = []
        seen = set()

        def add(head: str, relation: str, tail: str, confidence: float, extractor: str) -> None:
            head = clean_entity_text(head)
            tail = clean_entity_text(tail)
            relation = self._clean_relation(relation)
            if not head or not tail or head.lower() == tail.lower() or is_generic_entity(head) or is_generic_entity(tail):
                return
            key = (head.lower(), relation, tail.lower())
            if key in seen:
                return
            seen.add(key)
            triples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
                "confidence": confidence,
                "extractor": extractor,
            })

        for token in doc:
            if token.pos_ in {"VERB", "AUX"}:
                subjects = self._mentions_for_deps(token, mentions, {"nsubj", "nsubjpass", "csubj"})
                objects = self._mentions_for_deps(token, mentions, {"dobj", "obj", "attr", "oprd", "dative"})
                prep_objects: List[Tuple[Dict, str]] = []
                for child in token.children:
                    if child.dep_ == "prep":
                        for pobj in child.children:
                            if pobj.dep_ in {"pobj", "pcomp"}:
                                for mention in self._mentions_in_subtree(pobj, mentions):
                                    prep_objects.append((mention, child.lemma_.lower()))
                    if child.dep_ == "agent":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                for mention in self._mentions_in_subtree(pobj, mentions):
                                    prep_objects.append((mention, "by"))

                for subj in subjects:
                    for obj in objects:
                        add(subj["text"], token.lemma_, obj["text"], 0.86, "spacy_svo")
                    for obj, prep in prep_objects:
                        relation = f"{token.lemma_}_{prep}"
                        if token.dep_ == "ROOT" and any(child.dep_ == "nsubjpass" for child in token.children) and prep == "by":
                            add(obj["text"], token.lemma_, subj["text"], 0.88, "spacy_passive")
                        else:
                            add(subj["text"], relation, obj["text"], 0.84, "spacy_prep")

        for mention in mentions:
            root = mention["root"]
            for child in root.children:
                if child.dep_ in {"appos", "attr"}:
                    for other in self._mentions_in_subtree(child, mentions):
                        add(mention["text"], child.dep_, other["text"], 0.72, "spacy_appos")
                if child.dep_ == "prep" and child.lemma_.lower() == "of":
                    for other in self._mentions_in_subtree(child, mentions):
                        add(mention["text"], "of", other["text"], 0.70, "spacy_of")

        if len(triples) < 3:
            self._path_fallback_triples(doc, mentions, add)
        return triples

    def _mentions_for_deps(self, token, mentions: List[Dict], deps: set) -> List[Dict]:
        found: List[Dict] = []
        for child in token.children:
            if child.dep_ in deps:
                found.extend(self._mentions_in_subtree(child, mentions))
        return self._dedupe_mentions(found)

    def _mentions_in_subtree(self, token, mentions: List[Dict]) -> List[Dict]:
        token_ids = {item.i for item in token.subtree}
        return [mention for mention in mentions if mention["root"].i in token_ids or mention["start"] <= token.i < mention["end"]]

    def _path_fallback_triples(self, doc, mentions: List[Dict], add) -> None:
        for left_index, left in enumerate(mentions):
            for right in mentions[left_index + 1:left_index + 5]:
                relation_tokens = []
                start = min(left["end"], right["end"])
                end = max(left["start"], right["start"])
                for token in doc[start:end]:
                    if token.pos_ in {"VERB", "AUX", "ADP", "NOUN", "PROPN", "ADJ"} and token.is_alpha:
                        relation_tokens.append(token.lemma_.lower())
                relation = "_".join(relation_tokens[:4])
                if relation:
                    add(left["text"], relation, right["text"], 0.55, "spacy_path")

    def _dedupe_mentions(self, mentions: List[Dict]) -> List[Dict]:
        deduped = []
        seen = set()
        for mention in mentions:
            key = mention["text"].lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(mention)
        return deduped

    def _clean_relation(self, relation: str) -> str:
        relation = re.sub(r"[^A-Za-z0-9]+", "_", str(relation or "").lower()).strip("_")
        return relation or "related_to"

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
                triples.append({"head": left, "relation": "related_to", "tail": right, "confidence": 0.35, "extractor": "regex_fallback"})
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
                try:
                    confidence = float(triple.get("confidence", 1.0) or 1.0)
                except (TypeError, ValueError):
                    confidence = 1.0
                cleaned.append({
                    "head": head,
                    "relation": self._clean_relation(relation),
                    "tail": tail,
                    "confidence": confidence,
                    "extractor": str(triple.get("extractor", "llm_or_heuristic")),
                })
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
