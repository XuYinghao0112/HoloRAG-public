from collections import defaultdict
import re
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from .config import NaiveHoloRAGConfig
from .embedding_model import NVEmbedV2Encoder
from .utils import cosine_similarity_matrix, lexical_overlap_score, merge_weighted_scores, normalize_alpha, normalize_scores, top_k


class NaiveRetriever:
    def __init__(self, config: NaiveHoloRAGConfig, embedder: NVEmbedV2Encoder) -> None:
        self.config = config
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        query_entities: Sequence[str],
        query_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
        alpha: Dict[str, float],
    ) -> Dict:
        alpha = normalize_alpha(alpha)
        entity_scores = self._retrieve_entities(query_entities, state)
        fact_scores_pre = self._retrieve_facts(query, query_facts, state)
        fact_scores, rerank_meta = self._rerank_facts(query, fact_scores_pre, state)
        sentence_scores = self._retrieve_sentences(sub_questions, state)
        chunk_scores = self._retrieve_chunks(query, state)

        seed_scores: Dict[str, float] = defaultdict(float)
        merge_weighted_scores(seed_scores, entity_scores, alpha["entity"])
        merge_weighted_scores(seed_scores, sentence_scores, alpha["sentence"])
        merge_weighted_scores(seed_scores, chunk_scores, alpha["chunk"])
        for fact_id, score in fact_scores.items():
            fact = self._fact_by_id(state, fact_id)
            if not fact:
                continue
            weighted = alpha["fact"] * float(score)
            for node_id, scale in [
                (fact.get("head_id", ""), 0.35),
                (fact.get("tail_id", ""), 0.35),
                (fact.get("sentence_id", ""), 0.20),
                (fact.get("chunk_id", ""), 0.10),
            ]:
                if node_id in graph:
                    seed_scores[node_id] += weighted * scale

        # Reduce hub entities before normalization to avoid over-seeding generic nodes.
        seed_scores = self._apply_entity_hub_suppression(graph, seed_scores)

        fallback_used = False
        if self.config.enable_no_fact_fallback and len(fact_scores) == 0:
            fallback_used = True
            for node_id, score in chunk_scores.items():
                if node_id in graph:
                    seed_scores[node_id] += max(float(score), 0.0)
            for sentence_id, score in sentence_scores.items():
                if sentence_id in graph:
                    seed_scores[sentence_id] += 0.5 * max(float(score), 0.0)

        seed_scores = normalize_scores(dict(seed_scores))
        return {
            "seed_scores": seed_scores,
            "channel_scores": {
                "entity": entity_scores,
                "fact": fact_scores,
                "sentence": sentence_scores,
                "chunk": chunk_scores,
            },
            "ranked_facts": self._rank_facts(fact_scores, state),
            "rerank_meta": rerank_meta,
            "fallback_used": fallback_used,
        }

    def rank_passages(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        channel_scores: Dict[str, Dict[str, float]],
        ranked_facts: Sequence[Dict],
    ) -> List[Dict]:
        chunk_scores: Dict[str, float] = defaultdict(float)
        for node_id, score in pagerank_scores.items():
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            node_type = attrs.get("node_type")
            if node_type == "chunk":
                chunk_scores[node_id] += 0.70 * float(score)
            elif node_type == "sentence":
                chunk_id = attrs.get("metadata", {}).get("chunk_id")
                if chunk_id:
                    chunk_scores[chunk_id] += 0.20 * float(score)
            elif node_type == "entity":
                for neighbor in list(graph.successors(node_id)) + list(graph.predecessors(node_id)):
                    if neighbor in graph and graph.nodes[neighbor].get("node_type") == "sentence":
                        chunk_id = graph.nodes[neighbor].get("metadata", {}).get("chunk_id")
                        if chunk_id:
                            chunk_scores[chunk_id] += 0.05 * float(score)
        for chunk_id, score in channel_scores.get("chunk", {}).items():
            chunk_scores[chunk_id] += 0.30 * float(score)
        for fact in ranked_facts[: self.config.fact_top_k]:
            chunk_id = fact.get("chunk_id")
            if chunk_id:
                chunk_scores[chunk_id] += 0.15 * float(fact.get("score", 0.0))

        passages = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True):
            if chunk_id not in graph:
                continue
            attrs = graph.nodes[chunk_id]
            metadata = attrs.get("metadata", {})
            passages.append({
                "chunk_id": chunk_id,
                "node_id": chunk_id,
                "score": float(score),
                "title": metadata.get("title", ""),
                "text": attrs.get("text", ""),
                "metadata": metadata,
            })
            if len(passages) >= self.config.passage_output_top_k:
                break
        return passages

    def rank_evidence(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        channel_scores: Dict[str, Dict[str, float]],
        ranked_facts: Sequence[Dict],
        ranked_passages: Sequence[Dict],
        profile: str,
        sub_questions: Sequence[str] = (),
    ) -> Dict:
        sentence_scores = self._combined_sentence_scores(graph, pagerank_scores, channel_scores, ranked_facts)
        ranked_sentences = [
            self._sentence_record(graph, sentence_id, score)
            for sentence_id, score in sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
            if sentence_id in graph
        ]
        ranked_sentences = [item for item in ranked_sentences if item]

        if profile == "single_hop":
            facts = list(ranked_facts[: min(self.config.fact_top_k, 10)])
            source_sentences = self._source_sentences_for_facts(graph, facts)
            sentences = self._dedupe_records(source_sentences + ranked_sentences[:4], "node_id")[:6]
            chunks = list(ranked_passages[:1]) if len(sentences) < 2 else []
            evidence_groups = [
                {"label": "Fact source sentences", "items": sentences},
            ]
        elif profile == "long_context":
            facts = list(ranked_facts[: min(self.config.fact_top_k, 5)])
            sentences = ranked_sentences[:5]
            chunks = list(ranked_passages[: self.config.qa_passage_top_k])
            evidence_groups = [
                {"label": "High-signal sentences", "items": sentences},
            ]
        else:
            facts = list(ranked_facts[: min(self.config.fact_top_k, 12)])
            source_sentences = self._source_sentences_for_facts(graph, facts)
            evidence_groups, sentences = self._multi_hop_sentence_groups(
                graph=graph,
                ranked_sentences=ranked_sentences,
                source_sentences=source_sentences,
                sub_questions=sub_questions,
            )
            # Multi-hop QA needs a small amount of passage context because the
            # bridge or answer span often lives outside the extracted sentence set.
            # The reader enforces the global evidence budget, so this remains
            # generic and bounded across datasets.
            chunks = list(ranked_passages[: min(2, self.config.qa_passage_top_k)])

        return {
            "profile": profile,
            "facts": facts,
            "sentences": sentences,
            "chunks": chunks,
            "evidence_groups": evidence_groups,
            "fallback_passages": list(ranked_passages[: self.config.qa_passage_top_k]),
        }

    def _retrieve_entities(self, query_entities: Sequence[str], state: Dict) -> Dict[str, float]:
        embeddings = state.get("embeddings", {}).get("entity", {})
        if not query_entities or not embeddings:
            return {}
        scores: Dict[str, float] = defaultdict(float)
        for entity in query_entities:
            scores.update(
                self._dense_layer(
                    str(entity),
                    embeddings,
                    self.config.query_instruction_text,
                    "query",
                    self.config.entity_top_k,
                )
            )
        return normalize_scores(dict(scores))

    def _retrieve_facts(self, query: str, query_facts: Sequence[Dict], state: Dict) -> Dict[str, float]:
        embeddings = state.get("embeddings", {}).get("fact", {})
        if not embeddings:
            return {}
        fact_queries = [f"{item.get('head', '')} {item.get('relation', '')} {item.get('tail', '')}".strip() for item in query_facts]
        fact_queries = [item for item in fact_queries if item] or [query]
        scores: Dict[str, float] = defaultdict(float)
        for fact_query in fact_queries:
            dense = self._dense_layer(
                fact_query,
                embeddings,
                self.config.query_instruction_fact,
                "query",
                max(self.config.fact_top_k, self.config.fact_rerank_top_k),
            )
            for fact_id, score in dense.items():
                fact = self._fact_by_id(state, fact_id)
                lexical = lexical_overlap_score(fact_query, fact.get("text", "") if fact else "")
                scores[fact_id] = max(scores.get(fact_id, 0.0), 0.85 * score + 0.15 * lexical)
        return normalize_scores(dict(scores))

    def _retrieve_sentences(self, sub_questions: Sequence[str], state: Dict) -> Dict[str, float]:
        embeddings = state.get("embeddings", {}).get("sentence", {})
        if not sub_questions or not embeddings:
            return {}
        scores: Dict[str, float] = defaultdict(float)
        for question in sub_questions:
            dense = self._dense_layer(
                question,
                embeddings,
                self.config.query_instruction_text,
                "query",
                self.config.sentence_top_k,
            )
            for node_id, score in dense.items():
                scores[node_id] = max(scores.get(node_id, 0.0), score)
        return normalize_scores(dict(scores))

    def _retrieve_chunks(self, query: str, state: Dict) -> Dict[str, float]:
        embeddings = state.get("embeddings", {}).get("chunk", {})
        if not embeddings:
            return {}
        return normalize_scores(
            self._dense_layer(
                query,
                embeddings,
                self.config.query_instruction_text,
                "query",
                self.config.chunk_top_k,
            )
        )

    def _dense_layer(self, query: str, node_embeddings: Dict[str, np.ndarray], instruction: str, text_type: str, top_k_value: int) -> Dict[str, float]:
        node_ids = list(node_embeddings.keys())
        if not node_ids:
            return {}
        docs = np.asarray([node_embeddings[node_id] for node_id in node_ids], dtype=np.float32)
        query_vec = self.embedder.encode([query], instruction=instruction, text_type=text_type)[0]
        similarities = cosine_similarity_matrix(query_vec, docs)
        ranked = top_k(list(zip(node_ids, similarities.tolist())), top_k_value)
        return {node_id: max(float(score), 0.0) for node_id, score in ranked}

    def _rank_facts(self, fact_scores: Dict[str, float], state: Dict) -> List[Dict]:
        facts = []
        for fact_id, score in sorted(fact_scores.items(), key=lambda item: item[1], reverse=True):
            fact = self._fact_by_id(state, fact_id)
            if fact:
                record = dict(fact)
                record["score"] = float(score)
                facts.append(record)
            if len(facts) >= self.config.fact_top_k:
                break
        return facts

    def _rerank_facts(self, query: str, fact_scores: Dict[str, float], state: Dict) -> Tuple[Dict[str, float], Dict]:
        if not fact_scores:
            return {}, {"candidate_count": 0, "kept_count": 0, "mode": "none"}
        ranked = sorted(fact_scores.items(), key=lambda item: item[1], reverse=True)
        candidates = ranked[: max(1, self.config.fact_rerank_top_k)]
        kept_count = min(len(candidates), max(1, self.config.fact_rerank_keep_k))

        # Lightweight lexical rerank by query overlap with fact text.
        scored = []
        for fact_id, dense_score in candidates:
            fact = self._fact_by_id(state, fact_id)
            text = fact.get("text", "") if fact else ""
            lex = lexical_overlap_score(query, text)
            score = 0.7 * float(dense_score) + 0.3 * float(lex)
            scored.append((fact_id, score))
        selected = sorted(scored, key=lambda item: item[1], reverse=True)[:kept_count]
        reranked = normalize_scores({fact_id: score for fact_id, score in selected})
        return reranked, {
            "candidate_count": len(candidates),
            "kept_count": len(selected),
            "mode": "lexical_rerank",
        }

    def _combined_sentence_scores(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        channel_scores: Dict[str, Dict[str, float]],
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for node_id, score in pagerank_scores.items():
            if node_id in graph and graph.nodes[node_id].get("node_type") == "sentence":
                scores[node_id] += 0.65 * float(score)
        for node_id, score in channel_scores.get("sentence", {}).items():
            if node_id in graph:
                scores[node_id] += 0.35 * float(score)
        for fact in ranked_facts[: self.config.fact_top_k]:
            sentence_id = fact.get("sentence_id")
            if sentence_id in graph:
                scores[sentence_id] += 0.45 * float(fact.get("score", 0.0))
        return normalize_scores(dict(scores))

    def _sentence_record(self, graph: nx.DiGraph, sentence_id: str, score: float) -> Dict:
        attrs = graph.nodes[sentence_id]
        metadata = attrs.get("metadata", {})
        return {
            "node_id": sentence_id,
            "sentence_id": sentence_id,
            "chunk_id": metadata.get("chunk_id", ""),
            "score": float(score),
            "title": metadata.get("title", ""),
            "text": attrs.get("text", ""),
            "metadata": metadata,
        }

    def _source_sentences_for_facts(self, graph: nx.DiGraph, ranked_facts: Sequence[Dict]) -> List[Dict]:
        records = []
        for fact in ranked_facts:
            sentence_id = fact.get("sentence_id")
            if sentence_id not in graph:
                continue
            record = self._sentence_record(graph, sentence_id, float(fact.get("score", 0.0)))
            record["source_fact"] = fact.get("text", "")
            records.append(record)
        return records

    def _multi_hop_sentence_groups(
        self,
        graph: nx.DiGraph,
        ranked_sentences: Sequence[Dict],
        source_sentences: Sequence[Dict],
        sub_questions: Sequence[str],
    ) -> Tuple[List[Dict], List[Dict]]:
        selected: List[Dict] = []
        title_counts: Dict[str, int] = defaultdict(int)

        def add(record: Dict, title_limit: int = 3) -> bool:
            node_id = record.get("node_id")
            if not node_id or any(item.get("node_id") == node_id for item in selected):
                return False
            title_key = self._title_key(record)
            if title_key and title_counts[title_key] >= title_limit:
                return False
            selected.append(record)
            if title_key:
                title_counts[title_key] += 1
            return True

        fact_sources = []
        for record in source_sentences:
            if add(record, title_limit=3):
                fact_sources.append(record)
            if len(fact_sources) >= 6:
                break

        groups = []
        if fact_sources:
            groups.append({"label": "Fact source sentences", "items": fact_sources})

        for index, sub_question in enumerate(sub_questions, start=1):
            sub_question = str(sub_question or "").strip()
            if not sub_question:
                continue
            candidates = sorted(
                ranked_sentences,
                key=lambda item: (
                    self._coverage_score(sub_question, item),
                    float(item.get("score", 0.0)),
                ),
                reverse=True,
            )
            group_items = []
            for candidate in candidates:
                if add(candidate, title_limit=3):
                    group_items.append(candidate)
                if len(group_items) >= 3:
                    break
            if group_items:
                groups.append({"label": f"Evidence for sub-question {index}", "question": sub_question, "items": group_items})
            if len(selected) >= 14:
                break

        if len(selected) < 10:
            for record in ranked_sentences:
                add(record, title_limit=3)
                if len(selected) >= 12:
                    break

        if not groups:
            groups.append({"label": "Retrieved sentences", "items": selected[:12]})
        return groups, selected[:14]

    def _dedupe_records(self, records: Sequence[Dict], key: str) -> List[Dict]:
        deduped = []
        seen = set()
        for record in records:
            value = record.get(key)
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(record)
        return deduped

    def _title_key(self, record: Dict) -> str:
        return str(record.get("title", "")).strip().lower()

    def _coverage_score(self, question: str, record: Dict) -> float:
        q_terms = self._content_terms(question)
        text = str(record.get("title", "")) + " " + str(record.get("text", ""))
        text_terms = self._content_terms(text)
        if not q_terms or not text_terms:
            return 0.0
        overlap = len(q_terms & text_terms)
        return overlap / max(1, len(q_terms))

    def _content_terms(self, text: str) -> set:
        stopwords = {
            "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "who", "what", "when", "where", "which", "how", "was", "were", "is", "are", "did", "does",
            "do", "this", "that", "its", "his", "her", "their", "has", "have", "had", "film", "song",
        }
        return {term for term in re.findall(r"[A-Za-z0-9']+", str(text or "").lower()) if len(term) > 2 and term not in stopwords}

    def _apply_entity_hub_suppression(self, graph: nx.DiGraph, seed_scores: Dict[str, float]) -> Dict[str, float]:
        gamma = max(float(self.config.entity_hub_suppression), 0.0)
        if gamma <= 0:
            return seed_scores
        adjusted: Dict[str, float] = {}
        for node_id, score in seed_scores.items():
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            if attrs.get("node_type") == "entity":
                degree = graph.degree(node_id)
                adjusted[node_id] = float(score) / (1.0 + gamma * np.log1p(max(0, degree)))
            else:
                adjusted[node_id] = float(score)
        return adjusted

    def _fact_by_id(self, state: Dict, fact_id: str) -> Dict:
        cache = state.setdefault("_fact_by_id", {fact["fact_id"]: fact for fact in state.get("facts", [])})
        return cache.get(fact_id, {})
