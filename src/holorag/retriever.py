from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .config import HoloRAGConfig
from .embedding_model import NVEmbedV2Encoder
from .llm_client import LocalLLMClient
from .utils import cosine_similarity_matrix, lexical_overlap_score, normalize_alpha, normalize_scores, top_k

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config: HoloRAGConfig, embedder: NVEmbedV2Encoder, llm_client: LocalLLMClient = None) -> None:
        self.config = config
        self.embedder = embedder
        self.llm_client = llm_client
        self._device_retrievers: Dict[str, "Retriever"] = {}

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
        mode = self._execution_mode()
        if mode == "sequential":
            entity_scores = self._retrieve_entities(query_entities, state)
            fact_scores_pre = self._retrieve_facts(query, query_facts, state)
            fact_scores, rerank_meta = self._rerank_facts(query, fact_scores_pre, state)
            sentence_scores = self._retrieve_sentences(sub_questions, state)
            chunk_scores = self._retrieve_chunks(query, state)
            candidate_meta = {"execution_mode": "sequential", "num_workers": 1}
        else:
            candidates = self.retrieve_candidates_multi_worker(
                query=query,
                query_entities=query_entities,
                query_facts=query_facts,
                sub_questions=sub_questions,
                state=state,
            )
            entity_scores = candidates["entity"]
            fact_scores_pre = candidates["fact_pre"]
            fact_scores, rerank_meta = self._rerank_facts(query, fact_scores_pre, state)
            sentence_scores = candidates["sentence"]
            chunk_scores = candidates["chunk"]
            candidate_meta = candidates.get("meta", {})

        seed_scores: Dict[str, float] = defaultdict(float)
        # v1 reset distribution: reranked fact endpoints plus a light dense-passage prior.
        for fact_id, score in fact_scores.items():
            fact = self._fact_by_id(state, fact_id)
            if not fact:
                continue
            for node_id in [fact.get("head_id", ""), fact.get("tail_id", "")]:
                if node_id not in graph:
                    continue
                weighted_score = (
                    float(score)
                    * max(alpha.get("fact", 0.0), 0.05)
                    / max(1.0, float(self._entity_chunk_count(graph, node_id)))
                )
                seed_scores[node_id] += weighted_score
        for node_id, score in chunk_scores.items():
            if node_id in graph:
                seed_scores[node_id] += max(float(score), 0.0) * max(alpha.get("chunk", 0.0), 0.05)

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
            "rerank_meta": {
                **rerank_meta,
                "candidate_retrieval": candidate_meta,
            },
            "fallback_used": fallback_used,
        }

    def retrieve_candidates(
        self,
        query: str,
        query_entities: Sequence[str],
        query_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        state: Dict,
    ) -> Dict:
        mode = self._execution_mode()
        if mode == "multi_worker":
            return self.retrieve_candidates_multi_worker(query, query_entities, query_facts, sub_questions, state)
        return self.retrieve_candidates_sequential(query, query_entities, query_facts, sub_questions, state)

    def _execution_mode(self) -> str:
        mode = str(getattr(self.config, "execution_mode", "sequential") or "sequential").strip().lower()
        if mode not in {"sequential", "multi_worker"}:
            raise ValueError(f"Unsupported HoloRAG execution_mode={mode!r}; expected 'sequential' or 'multi_worker'.")
        return mode

    def retrieve_candidates_sequential(
        self,
        query: str,
        query_entities: Sequence[str],
        query_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        state: Dict,
    ) -> Dict:
        return {
            "entity": self._retrieve_entities(query_entities, state),
            "fact_pre": self._retrieve_facts(query, query_facts, state),
            "sentence": self._retrieve_sentences(sub_questions, state),
            "chunk": self._retrieve_chunks(query, state),
            "meta": {"execution_mode": "sequential", "num_workers": 1},
        }

    def retrieve_candidates_multi_worker(
        self,
        query: str,
        query_entities: Sequence[str],
        query_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        state: Dict,
    ) -> Dict:
        tasks = [
            ("fact_entity", (query, query_entities, query_facts, state)),
            ("sentence", (sub_questions, state)),
            ("chunk", (query, state)),
        ]
        task_specs = [
            (name, self._multi_worker_task_fn(name, index), args)
            for index, (name, args) in enumerate(tasks)
        ]
        max_workers = max(1, int(getattr(self.config, "num_workers", 3) or 3))
        max_workers = min(max_workers, len(task_specs))
        results: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="holorag-retrieval") as executor:
            future_to_name = {
                executor.submit(fn, *args): name
                for name, fn, args in task_specs
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    if name == "fact_entity":
                        entity_scores, fact_scores = result
                        results["entity"] = entity_scores
                        results["fact_pre"] = fact_scores
                    else:
                        results[name] = result
                except Exception as exc:
                    raise RuntimeError(f"HoloRAG multi_worker retrieval branch failed: {name}") from exc

        return {
            "entity": results.get("entity", {}),
            "fact_pre": results.get("fact_pre", {}),
            "sentence": results.get("sentence", {}),
            "chunk": results.get("chunk", {}),
            "meta": {
                "execution_mode": "multi_worker",
                "num_workers": max_workers,
                "embedding_devices": self._multi_worker_task_devices(),
            },
        }

    def _multi_worker_task_fn(self, task_name: str, task_index: int):
        retriever = self._retriever_for_multi_worker_task(task_index)
        if task_name == "fact_entity":
            return retriever._retrieve_fact_entity_candidates
        if task_name == "sentence":
            return retriever._retrieve_sentences
        if task_name == "chunk":
            return retriever._retrieve_chunks
        raise ValueError(f"Unknown multi_worker retrieval task: {task_name}")

    def _multi_worker_task_devices(self) -> Dict[str, str]:
        task_names = ["fact_entity", "sentence", "chunk"]
        devices = self._configured_multi_worker_devices()
        if not devices:
            current = getattr(self.embedder, "embedding_device", str(getattr(self.config, "embedding_device", "")))
            return {name: current for name in task_names}
        return {name: devices[index % len(devices)] for index, name in enumerate(task_names)}

    def _retriever_for_multi_worker_task(self, task_index: int) -> "Retriever":
        devices = self._configured_multi_worker_devices()
        if not devices:
            return self
        return self._retriever_for_device(devices[task_index % len(devices)])

    def _configured_multi_worker_devices(self) -> List[str]:
        raw = str(getattr(self.config, "multi_worker_embedding_devices", "") or "").strip()
        if not raw:
            return []
        devices = [item.strip() for item in raw.split(",") if item.strip()]
        return devices

    def _retriever_for_device(self, raw_device: str) -> "Retriever":
        normalized = self.embedder._normalize_device(raw_device)
        current = getattr(self.embedder, "embedding_device", "")
        if normalized == current:
            return self
        if normalized not in self._device_retrievers:
            branch_config = replace(
                self.config,
                embedding_device=normalized,
                multi_worker_embedding_devices="",
            )
            logger.info("Loading multi_worker retrieval encoder on %s", normalized)
            branch_embedder = NVEmbedV2Encoder(branch_config)
            self._device_retrievers[normalized] = Retriever(branch_config, branch_embedder, self.llm_client)
        return self._device_retrievers[normalized]

    def _retrieve_fact_entity_candidates(
        self,
        query: str,
        query_entities: Sequence[str],
        query_facts: Sequence[Dict],
        state: Dict,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        entity_scores = self._retrieve_entities(query_entities, state)
        fact_scores = self._retrieve_facts(query, query_facts, state)
        return entity_scores, fact_scores

    def rank_passages(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        channel_scores: Dict[str, Dict[str, float]],
        ranked_facts: Sequence[Dict],
        alpha: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        alpha_weights = self._alpha_passage_weights(alpha) if self._use_llm_alpha_evidence(alpha) else {}
        chunk_scores: Dict[str, float] = defaultdict(float)
        for node_id, score in pagerank_scores.items():
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            node_type = attrs.get("node_type")
            if node_type == "chunk":
                chunk_scores[node_id] += alpha_weights.get("chunk_pagerank", 0.70) * float(score)
            elif node_type == "sentence":
                chunk_id = attrs.get("metadata", {}).get("chunk_id")
                if chunk_id:
                    chunk_scores[chunk_id] += alpha_weights.get("sentence_to_chunk", 0.20) * float(score)
            elif node_type == "entity":
                for neighbor in list(graph.successors(node_id)) + list(graph.predecessors(node_id)):
                    if neighbor in graph and graph.nodes[neighbor].get("node_type") == "sentence":
                        chunk_id = graph.nodes[neighbor].get("metadata", {}).get("chunk_id")
                        if chunk_id:
                            chunk_scores[chunk_id] += alpha_weights.get("entity_to_chunk", 0.05) * float(score)
        for chunk_id, score in channel_scores.get("chunk", {}).items():
            chunk_scores[chunk_id] += alpha_weights.get("chunk_dense", 0.30) * float(score)
        for fact in ranked_facts[: self.config.fact_top_k]:
            chunk_id = fact.get("chunk_id")
            if chunk_id:
                chunk_scores[chunk_id] += alpha_weights.get("fact_to_chunk", 0.15) * float(fact.get("score", 0.0))
                if getattr(self.config, "enable_fact_chunk_boost", False):
                    chunk_scores[chunk_id] += float(getattr(self.config, "fact_chunk_boost", 0.35)) * float(fact.get("score", 0.0))

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
        query: str = "",
        sub_questions: Sequence[str] = (),
        token_budget: int = 620,
        alpha: Optional[Dict[str, float]] = None,
    ) -> Dict:
        use_alpha_evidence = self._use_llm_alpha_evidence(alpha)
        sentence_scores = self._combined_sentence_scores(graph, pagerank_scores, channel_scores, ranked_facts)
        ranked_sentences = [
            self._sentence_record(graph, sentence_id, score)
            for sentence_id, score in sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
            if sentence_id in graph
        ]
        ranked_sentences = [item for item in ranked_sentences if item]

        if not getattr(self.config, "enable_granularity_awareness", True):
            facts = list(ranked_facts[: self.config.fact_top_k])
            source_sentences = self._source_sentences_for_facts(graph, facts)
            sentences = self._dedupe_records(source_sentences + ranked_sentences[: self.config.sentence_top_k], "node_id")
            chunks = list(ranked_passages[: self.config.passage_output_top_k])
            evidence_groups = [
                {"label": "Uniform fact source evidence", "items": source_sentences},
                {"label": "Uniform sentence evidence", "items": ranked_sentences[: self.config.sentence_top_k]},
            ]
        elif use_alpha_evidence:
            facts, sentences, chunks, evidence_groups = self._alpha_guided_evidence(
                graph=graph,
                ranked_facts=ranked_facts,
                ranked_sentences=ranked_sentences,
                ranked_passages=ranked_passages,
                alpha=alpha or {},
                query=query,
                sub_questions=sub_questions,
            )
        elif profile == "single_hop":
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
            facts = list(ranked_facts[: min(self.config.fact_top_k, 8)])
            source_sentences = self._source_sentences_for_facts(graph, facts)
            if getattr(self.config, "enable_fact_source_first_evidence", False):
                evidence_groups, sentences = self._fact_source_first_sentence_groups(
                    graph=graph,
                    ranked_sentences=ranked_sentences,
                    source_sentences=source_sentences,
                    ranked_facts=facts,
                    sub_questions=sub_questions,
                )
            else:
                evidence_groups, sentences = self._multi_hop_sentence_groups(
                    graph=graph,
                    ranked_sentences=ranked_sentences,
                    source_sentences=source_sentences,
                    sub_questions=sub_questions,
                )
            if getattr(self.config, "enable_fair_sentence_context", False):
                evidence_groups, sentences = self._add_fair_ranked_sentence_context(
                    evidence_groups=evidence_groups,
                    selected_sentences=sentences,
                    ranked_sentences=ranked_sentences,
                    query=query,
                    sub_questions=sub_questions,
                )
            # Multi-hop QA needs a little passage context, but long chunks tend
            # to crowd out the second-hop answer sentence on 2Wiki-style tasks.
            use_expanded_passage_context = int(getattr(self.config, "evidence_passage_context_k", 1)) > 1
            if getattr(self.config, "enable_fair_sentence_context", False) or use_expanded_passage_context:
                chunks = self._diverse_passage_context(ranked_passages)
            else:
                chunks = list(ranked_passages[: min(1, self.config.qa_passage_top_k)])

        result = {
            "profile": profile,
            "facts": facts,
            "sentences": sentences,
            "chunks": chunks,
            "evidence_groups": evidence_groups,
            "fallback_passages": list(ranked_passages[: self.config.qa_passage_top_k]),
        }
        packed = self._pack_profile_evidence(
            profile=profile,
            query=query,
            facts=facts,
            sentences=sentences,
            chunks=chunks,
            evidence_groups=evidence_groups,
            fallback_passages=result["fallback_passages"],
            sub_questions=sub_questions,
            token_budget=token_budget,
            alpha=alpha if use_alpha_evidence else None,
        )
        result.update(packed)
        return result

    def _use_llm_alpha_evidence(self, alpha: Optional[Dict[str, float]]) -> bool:
        return bool(
            alpha
            and getattr(self.config, "enable_granularity_awareness", True)
            and getattr(self.config, "evidence_use_alpha_weights", True)
        )

    def _alpha_guided_evidence(
        self,
        graph: nx.DiGraph,
        ranked_facts: Sequence[Dict],
        ranked_sentences: Sequence[Dict],
        ranked_passages: Sequence[Dict],
        alpha: Dict[str, float],
        query: str,
        sub_questions: Sequence[str],
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        alpha = normalize_alpha(alpha)
        facts = list(ranked_facts[: self.config.fact_top_k])
        source_sentences = self._source_sentences_for_facts(graph, facts)
        sentences = self._dedupe_records(source_sentences + list(ranked_sentences[: self.config.sentence_top_k]), "node_id")
        chunks = list(ranked_passages[: self.config.passage_output_top_k])
        evidence_groups = [
            {
                "label": "Fact source evidence",
                "items": source_sentences,
            },
            {
                "label": "Sentence evidence",
                "items": ranked_sentences[: self.config.sentence_top_k],
            },
        ]
        return facts, sentences, chunks, evidence_groups

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
        if getattr(self.config, "fact_rerank_use_llm", False):
            llm_scores, meta = self._llm_filter_facts(query, candidates, state)
            if llm_scores:
                return llm_scores, meta

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

    def _llm_filter_facts(self, query: str, candidates: Sequence[Tuple[str, float]], state: Dict) -> Tuple[Dict[str, float], Dict]:
        candidate_limit = max(1, int(getattr(self.config, "fact_rerank_llm_candidate_k", 12)))
        keep_k = max(1, int(getattr(self.config, "fact_rerank_llm_keep_k", 5)))
        rows = []
        for index, (fact_id, dense_score) in enumerate(list(candidates)[:candidate_limit], start=1):
            fact = self._fact_by_id(state, fact_id)
            if not fact:
                continue
            rows.append({
                "id": index,
                "fact_id": fact_id,
                "fact": str(fact.get("text", "")),
                "title": str(fact.get("title", "")),
                "score": float(dense_score),
            })
        if not rows:
            return {}, {"candidate_count": 0, "kept_count": 0, "mode": "llm_fact_filter_empty"}
        if self.llm_client is None:
            return {}, {"candidate_count": len(rows), "kept_count": 0, "mode": "llm_fact_filter_no_client"}
        compact_rows = [
            {
                "id": row["id"],
                "fact": row["fact"],
                "title": row["title"],
            }
            for row in rows
        ]
        fallback = {"selected_ids": [row["id"] for row in rows[:keep_k]]}
        system_prompt = (
            "You filter factual triples for multi-hop question answering. "
            "Select only facts that are directly useful for answering the question or for linking to the next hop. "
            "Prefer facts that identify bridge entities and final-answer attributes. "
            "Return JSON only with key selected_ids, a list of integer ids. "
            f"Select at most {keep_k} facts."
        )
        payload, raw = self.llm_client.infer_json(
            system_prompt=system_prompt,
            user_prompt=(
                f"Question:\n{query}\n\n"
                f"Candidate facts:\n{compact_rows}\n\n"
                f"Return JSON: {{\"selected_ids\": [ids...]}}"
            ),
            fallback=fallback,
            max_tokens=128,
        )
        raw_ids = payload.get("selected_ids", []) if isinstance(payload, dict) else []
        selected_ids = []
        for value in raw_ids:
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                continue
            if int_value not in selected_ids:
                selected_ids.append(int_value)
            if len(selected_ids) >= keep_k:
                break
        if not selected_ids:
            selected_ids = fallback["selected_ids"]
        row_by_id = {row["id"]: row for row in rows}
        selected_scores = {}
        for rank, selected_id in enumerate(selected_ids):
            row = row_by_id.get(selected_id)
            if not row:
                continue
            dense_score = float(row.get("score", 0.0))
            selected_scores[row["fact_id"]] = dense_score * (1.0 + 0.05 * (len(selected_ids) - rank))
        if not selected_scores:
            return {}, {
                "candidate_count": len(rows),
                "kept_count": 0,
                "mode": "llm_fact_filter_empty_selection",
                "raw_response": raw,
            }
        return normalize_scores(selected_scores), {
            "candidate_count": len(rows),
            "kept_count": len(selected_scores),
            "mode": "llm_fact_filter",
            "selected_ids": selected_ids,
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

    def _fact_source_first_sentence_groups(
        self,
        graph: nx.DiGraph,
        ranked_sentences: Sequence[Dict],
        source_sentences: Sequence[Dict],
        ranked_facts: Sequence[Dict],
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
            if len(fact_sources) >= 5:
                break

        chunk_context = []
        for fact in ranked_facts[:5]:
            for record in self._chunk_sentences_around_fact(graph, fact):
                if add(record, title_limit=3):
                    chunk_context.append(record)
                if len(chunk_context) >= 5:
                    break
            if len(chunk_context) >= 5:
                break

        subquestion_sentences = []
        for sub_question in sub_questions:
            candidates = sorted(
                ranked_sentences,
                key=lambda item: (
                    self._coverage_score(str(sub_question), item),
                    float(item.get("score", 0.0)),
                ),
                reverse=True,
            )
            for candidate in candidates:
                if add(candidate, title_limit=3):
                    subquestion_sentences.append(candidate)
                    break
            if len(subquestion_sentences) >= 4:
                break

        if len(selected) < 10:
            for record in ranked_sentences:
                add(record, title_limit=3)
                if len(selected) >= 12:
                    break

        groups = []
        if fact_sources:
            groups.append({"label": "Filtered fact source sentences", "items": fact_sources})
        if chunk_context:
            groups.append({"label": "Local sentence context", "items": chunk_context})
        if subquestion_sentences:
            groups.append({"label": "Sub-question evidence", "items": subquestion_sentences})
        if not groups:
            groups.append({"label": "Retrieved sentences", "items": selected[:12]})
        return groups, selected[:14]

    def _add_fair_ranked_sentence_context(
        self,
        evidence_groups: Sequence[Dict],
        selected_sentences: Sequence[Dict],
        ranked_sentences: Sequence[Dict],
        query: str,
        sub_questions: Sequence[str],
    ) -> Tuple[List[Dict], List[Dict]]:
        groups = list(evidence_groups)
        selected = list(selected_sentences)
        selected_ids = {item.get("node_id") for item in selected if item.get("node_id")}
        title_limit = max(1, int(getattr(self.config, "evidence_title_limit", 3)))
        max_sentences = max(len(selected), int(getattr(self.config, "evidence_max_sentences", 18)))
        extra_k = max(0, int(getattr(self.config, "evidence_extra_ranked_sentence_k", 6)))
        title_counts: Dict[str, int] = defaultdict(int)
        for record in selected:
            title_key = self._title_key(record)
            if title_key:
                title_counts[title_key] += 1

        def fair_score(record: Dict) -> float:
            coverage = max(
                [self._coverage_score(str(item), record) for item in sub_questions if str(item or "").strip()]
                or [self._coverage_score(query, record)]
            )
            return float(record.get("score", 0.0)) + 0.35 * coverage

        additions: List[Dict] = []
        for record in sorted(ranked_sentences, key=fair_score, reverse=True):
            if len(additions) >= extra_k or len(selected) >= max_sentences:
                break
            node_id = record.get("node_id")
            if not node_id or node_id in selected_ids:
                continue
            title_key = self._title_key(record)
            if title_key and title_counts[title_key] >= title_limit:
                continue
            additions.append(record)
            selected.append(record)
            selected_ids.add(node_id)
            if title_key:
                title_counts[title_key] += 1

        if additions:
            groups.append({"label": "Additional ranked sentences", "items": additions})
        return groups, selected[:max_sentences]

    def _diverse_passage_context(self, ranked_passages: Sequence[Dict]) -> List[Dict]:
        limit = max(1, int(getattr(self.config, "evidence_passage_context_k", 2)))
        limit = min(limit, max(1, self.config.qa_passage_top_k))
        selected: List[Dict] = []
        seen_titles = set()
        for passage in ranked_passages:
            title = str(passage.get("title", "")).strip().lower()
            if title and title in seen_titles:
                continue
            selected.append(passage)
            if title:
                seen_titles.add(title)
            if len(selected) >= limit:
                break
        if not selected:
            return list(ranked_passages[:limit])
        return selected

    def _chunk_sentences_around_fact(self, graph: nx.DiGraph, fact: Dict) -> List[Dict]:
        sentence_id = fact.get("sentence_id")
        if sentence_id not in graph:
            return []
        attrs = graph.nodes[sentence_id]
        metadata = attrs.get("metadata", {})
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            return []
        sentence_index = metadata.get("sentence_index")
        rows = []
        for candidate_id in graph.successors(chunk_id):
            if candidate_id not in graph or graph.nodes[candidate_id].get("node_type") != "sentence":
                continue
            candidate_meta = graph.nodes[candidate_id].get("metadata", {})
            candidate_index = candidate_meta.get("sentence_index")
            if isinstance(sentence_index, int) and isinstance(candidate_index, int) and abs(candidate_index - sentence_index) > 1:
                continue
            record = self._sentence_record(graph, candidate_id, float(fact.get("score", 0.0)) * 0.82)
            rows.append(record)
        return sorted(rows, key=lambda row: row.get("metadata", {}).get("sentence_index", 0))

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

    def _pack_profile_evidence(
        self,
        profile: str,
        query: str,
        facts: Sequence[Dict],
        sentences: Sequence[Dict],
        chunks: Sequence[Dict],
        evidence_groups: Sequence[Dict],
        fallback_passages: Sequence[Dict],
        sub_questions: Sequence[str],
        token_budget: int,
        alpha: Optional[Dict[str, float]] = None,
    ) -> Dict:
        packing_mode = str(getattr(self.config, "evidence_packing_mode", "alpha_count")).strip().lower()
        use_alpha_count = packing_mode in {"alpha_count", "count", "count_first"}
        soft_budget = int(getattr(self.config, "evidence_soft_token_budget", 0) or 0)
        budget = 0
        if soft_budget > 0:
            budget = max(128, soft_budget)
        elif not use_alpha_count:
            budget = max(128, int(token_budget or self.config.qa_max_input_tokens))
        candidates: List[Dict] = []
        seen_sentence_ids = set()
        for group in evidence_groups:
            label = str(group.get("label", "Evidence")).strip() or "Evidence"
            group_question = str(group.get("question", "")).strip()
            for row in list(group.get("items", []) or []):
                node_id = row.get("node_id")
                if node_id in seen_sentence_ids:
                    continue
                seen_sentence_ids.add(node_id)
                candidates.append(self._packed_candidate(
                    kind="sentence",
                    text=str(row.get("text", "")),
                    title=str(row.get("title", "")),
                    score=float(row.get("score", 0.0)),
                    label=label,
                    node_id=node_id,
                    question=group_question,
                    query=query,
                    sub_questions=sub_questions,
                ))

        for row in sentences:
            node_id = row.get("node_id")
            if node_id in seen_sentence_ids:
                continue
            seen_sentence_ids.add(node_id)
            candidates.append(self._packed_candidate(
                kind="sentence",
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                score=float(row.get("score", 0.0)),
                label="Evidence",
                node_id=node_id,
                query=query,
                sub_questions=sub_questions,
            ))

        for fact in facts:
            candidates.append(self._packed_candidate(
                kind="fact",
                text=str(fact.get("text", "")),
                title=str(fact.get("title", "")),
                score=float(fact.get("score", 0.0)),
                label="Fact",
                node_id=str(fact.get("fact_id", "")),
                query=query,
                sub_questions=sub_questions,
            ))

        use_alpha_evidence = self._use_llm_alpha_evidence(alpha)
        if use_alpha_evidence:
            pack_alpha = self._packing_alpha(alpha or {})
        elif getattr(self.config, "enable_granularity_awareness", True):
            pack_alpha = normalize_alpha(self.config.profile_alpha_priors.get(profile, {}))
        else:
            pack_alpha = normalize_alpha({"fact": 1.0, "sentence": 1.0, "chunk": 1.0})

        for passage in list(chunks) or list(fallback_passages):
            passage_limit = self._dynamic_passage_limit(
                passage=passage,
                query=query,
                alpha=pack_alpha,
                budget=budget,
            )
            text = self._passage_excerpt(str(passage.get("text", "")), query, passage_limit)
            candidates.append(self._packed_candidate(
                kind="chunk",
                text=text,
                title=str(passage.get("title", "")),
                score=float(passage.get("score", 0.0)),
                label="Passage",
                node_id=str(passage.get("chunk_id", passage.get("node_id", ""))),
                query=query,
                sub_questions=sub_questions,
            ))

        if use_alpha_evidence:
            weights = self._alpha_pack_weights(pack_alpha, ("fact", "sentence", "chunk"))
        elif getattr(self.config, "enable_granularity_awareness", True):
            weights = self._alpha_pack_weights(self.config.profile_alpha_priors.get(profile, {}), ("fact", "sentence", "chunk"))
        else:
            weights = {"sentence": 1.0, "fact": 1.0, "chunk": 1.0}
        candidates = [item for item in candidates if str(item.get("line", "")).strip()]
        for item in candidates:
            label = str(item.get("label", "")).lower()
            source_bonus = 0.0
            if getattr(self.config, "enable_granularity_awareness", True) and getattr(self.config, "enable_fact_source_first_evidence", False):
                if "filtered fact source" in label:
                    source_bonus += 0.22
                elif "local sentence context" in label:
                    source_bonus += 0.12
            item["pack_score"] = (
                weights.get(item["kind"], 1.0) * float(item.get("score", 0.0))
                + 0.35 * float(item.get("coverage", 0.0))
                + source_bonus
            )

        selected = []
        selected_ids = set()
        selected_texts: List[str] = []
        title_counts: Dict[str, int] = defaultdict(int)
        title_limit = max(1, int(getattr(self.config, "evidence_title_limit", 3)))
        min_score = float(getattr(self.config, "evidence_min_score", 0.0))
        redundancy_threshold = float(getattr(self.config, "evidence_redundancy_threshold", 0.85))
        count_limits = self._alpha_count_limits(pack_alpha) if use_alpha_count else {}
        if use_alpha_count:
            for kind in ("fact", "sentence", "chunk"):
                self._select_count_limited_kind(
                    kind=kind,
                    candidates=candidates,
                    limit=int(count_limits.get(kind, 0)),
                    selected=selected,
                    selected_ids=selected_ids,
                    selected_texts=selected_texts,
                    title_counts=title_counts,
                    title_limit=title_limit,
                    min_score=min_score,
                    redundancy_threshold=redundancy_threshold,
                )
        else:
            used = 0
            for item in sorted(candidates, key=lambda row: row.get("pack_score", 0.0), reverse=True):
                if float(item.get("pack_score", 0.0)) < min_score:
                    continue
                remaining = budget - used
                if remaining <= 0:
                    break
                prepared = self._prepare_selected_item(
                    item=item,
                    selected_ids=selected_ids,
                    selected_texts=selected_texts,
                    title_counts=title_counts,
                    title_limit=title_limit,
                    redundancy_threshold=redundancy_threshold,
                    remaining=remaining,
                )
                if not prepared:
                    continue
                selected.append(prepared)
                selected_texts.append(prepared["line"])
                used += int(prepared.get("tokens", 0) or 0)

        packed_text = "\n".join(item["line"] for item in selected).strip()
        used_by_kind: Dict[str, int] = defaultdict(int)
        count_by_kind: Dict[str, int] = defaultdict(int)
        for item in selected:
            kind = str(item.get("kind", ""))
            used_by_kind[kind] += int(item.get("tokens", 0) or 0)
            count_by_kind[kind] += 1
        return {
            "packed_text": packed_text,
            "packed_records": selected,
            "packed_token_budget": budget,
            "packed_token_count": self._token_count(packed_text),
            "evidence_count_limits_by_granularity": {
                "fact": int(count_limits.get("fact", 0)),
                "sentence": int(count_limits.get("sentence", 0)),
                "chunk": int(count_limits.get("chunk", 0)),
            },
            "used_tokens_by_granularity": {
                "fact": int(used_by_kind.get("fact", 0)),
                "sentence": int(used_by_kind.get("sentence", 0)),
                "chunk": int(used_by_kind.get("chunk", 0)),
            },
            "evidence_counts_by_granularity": {
                "fact": int(count_by_kind.get("fact", 0)),
                "sentence": int(count_by_kind.get("sentence", 0)),
                "chunk": int(count_by_kind.get("chunk", 0)),
            },
        }

    def _alpha_pack_weights(self, alpha: Dict[str, float], kinds: Sequence[str]) -> Dict[str, float]:
        alpha = normalize_alpha(alpha)
        return {kind: 0.05 + 3.0 * float(alpha.get(kind, 0.0)) for kind in kinds}

    def _packing_alpha(self, alpha: Dict[str, float]) -> Dict[str, float]:
        alpha = normalize_alpha(alpha)
        mix = max(0.0, min(1.0, float(getattr(self.config, "evidence_alpha_uniform_mix", 0.0) or 0.0)))
        if mix <= 0.0:
            return alpha
        uniform = normalize_alpha({"fact": 1.0, "sentence": 1.0, "chunk": 1.0})
        return normalize_alpha({
            "fact": (1.0 - mix) * float(alpha.get("fact", 0.0)) + mix * uniform["fact"],
            "sentence": (1.0 - mix) * float(alpha.get("sentence", 0.0)) + mix * uniform["sentence"],
            "chunk": (1.0 - mix) * float(alpha.get("chunk", 0.0)) + mix * uniform["chunk"],
        })

    def _alpha_passage_weights(self, alpha: Optional[Dict[str, float]]) -> Dict[str, float]:
        alpha = normalize_alpha(alpha or {})
        return {
            "chunk_pagerank": 0.35 + 0.90 * alpha.get("chunk", 0.0),
            "sentence_to_chunk": 0.08 + 0.55 * alpha.get("sentence", 0.0),
            "entity_to_chunk": 0.02 + 0.25 * alpha.get("fact", 0.0),
            "chunk_dense": 0.12 + 0.70 * alpha.get("chunk", 0.0),
            "fact_to_chunk": 0.06 + 0.45 * alpha.get("fact", 0.0),
        }

    def _alpha_passage_limit(self, alpha: Dict[str, float], budget: int) -> int:
        alpha = normalize_alpha(alpha)
        chunk_weight = float(alpha.get("chunk", 0.0))
        return max(80, min(max(80, budget // 2), int(80 + chunk_weight * max(80, budget // 2))))

    def _dynamic_passage_limit(self, passage: Dict, query: str, alpha: Dict[str, float], budget: int) -> int:
        legacy_limit = int(getattr(self.config, "evidence_passage_excerpt_tokens", 0) or 0)
        configured_max = int(getattr(self.config, "evidence_chunk_max_tokens", 0) or 0)
        max_tokens = max(80, configured_max or legacy_limit or 256)
        min_tokens = min(80, max_tokens)
        span = max(0, max_tokens - min_tokens)
        title = str(passage.get("title", ""))
        text = str(passage.get("text", ""))
        score_signal = max(0.0, min(1.0, float(passage.get("score", 0.0))))
        coverage_signal = self._coverage_text_score(query, f"{title} {text}")
        evidence_signal = max(score_signal, coverage_signal)
        chunk_alpha = max(0.0, min(1.0, float(normalize_alpha(alpha).get("chunk", 0.0))))
        dynamic = min_tokens + span * (0.85 * chunk_alpha + 0.15 * evidence_signal)
        budget_cap = max_tokens if budget <= 0 else max(1, budget)
        return max(min_tokens, min(max_tokens, budget_cap, int(round(dynamic))))

    def _alpha_count_limits(self, alpha: Dict[str, float]) -> Dict[str, int]:
        alpha = normalize_alpha(alpha)
        kinds = ("fact", "sentence", "chunk")
        total = max(1, int(getattr(self.config, "evidence_alpha_total_units", 20)))
        caps = {
            "fact": max(0, int(getattr(self.config, "fact_top_k", 0))),
            "sentence": max(0, int(getattr(self.config, "sentence_top_k", 0))),
            "chunk": max(0, int(getattr(self.config, "passage_output_top_k", 0))),
        }
        raw = {kind: total * float(alpha.get(kind, 0.0)) for kind in kinds}
        limits = {kind: min(caps[kind], int(raw[kind])) for kind in kinds}
        remaining = total - sum(limits.values())
        remainders = sorted(
            kinds,
            key=lambda kind: (raw[kind] - int(raw[kind]), float(alpha.get(kind, 0.0))),
            reverse=True,
        )
        while remaining > 0:
            changed = False
            for kind in remainders:
                if limits[kind] >= caps[kind]:
                    continue
                limits[kind] += 1
                remaining -= 1
                changed = True
                if remaining <= 0:
                    break
            if not changed:
                break
        return {kind: int(limits.get(kind, 0)) for kind in kinds}

    def _select_count_limited_kind(
        self,
        kind: str,
        candidates: Sequence[Dict],
        limit: int,
        selected: List[Dict],
        selected_ids: set,
        selected_texts: List[str],
        title_counts: Dict[str, int],
        title_limit: int,
        min_score: float,
        redundancy_threshold: float,
    ) -> None:
        if limit <= 0:
            return
        ranked = sorted(
            [row for row in candidates if row.get("kind") == kind],
            key=lambda row: row.get("pack_score", 0.0),
            reverse=True,
        )
        kind_selected = 0
        title_pass_limits = [1]
        if title_limit > 1:
            title_pass_limits.append(title_limit)
        for pass_title_limit in title_pass_limits:
            for item in ranked:
                if kind_selected >= limit:
                    return
                if float(item.get("pack_score", 0.0)) < min_score:
                    continue
                prepared = self._prepare_selected_item(
                    item=item,
                    selected_ids=selected_ids,
                    selected_texts=selected_texts,
                    title_counts=title_counts,
                    title_limit=pass_title_limit,
                    redundancy_threshold=redundancy_threshold,
                    remaining=0,
                )
                if not prepared:
                    continue
                selected.append(prepared)
                selected_texts.append(prepared["line"])
                kind_selected += 1

    def _prepare_selected_item(
        self,
        item: Dict,
        selected_ids: set,
        selected_texts: Sequence[str],
        title_counts: Dict[str, int],
        title_limit: int,
        redundancy_threshold: float,
        remaining: int = 0,
    ) -> Optional[Dict]:
        node_id = item.get("node_id")
        if node_id and node_id in selected_ids:
            return None
        title_key = self._title_count_key(item)
        if title_key and title_counts[title_key] >= title_limit:
            return None
        line = str(item.get("line", "")).strip()
        if not line or self._is_redundant(line, selected_texts, redundancy_threshold):
            return None
        cost = self._token_count(line)
        if remaining > 0 and cost > remaining:
            if item["kind"] != "chunk" and remaining < 24:
                return None
            line = self._truncate_words(line, remaining)
            cost = self._token_count(line)
        if cost <= 0:
            return None
        if node_id:
            selected_ids.add(node_id)
        if title_key:
            title_counts[title_key] += 1
        return {**item, "line": line, "tokens": cost}

    def _title_count_key(self, item: Dict) -> str:
        title = str(item.get("title", "")).strip().lower()
        if not title:
            return ""
        kind = str(item.get("kind", "")).strip().lower() or "unknown"
        return f"{kind}\t{title}"

    def _packed_candidate(
        self,
        kind: str,
        text: str,
        title: str,
        score: float,
        label: str,
        node_id: str = "",
        question: str = "",
        query: str = "",
        sub_questions: Sequence[str] = (),
    ) -> Dict:
        text = " ".join(str(text or "").split())
        title = str(title or "").strip()
        if not text:
            return {"kind": kind, "line": "", "score": 0.0}
        prefix = label
        if title:
            prefix += f": [{title}] "
        else:
            prefix += ": "
        coverage = max([self._coverage_text_score(item, f"{title} {text}") for item in sub_questions] or [0.0])
        if not coverage:
            coverage = self._coverage_text_score(query, f"{title} {text}")
        return {
            "kind": kind,
            "line": prefix + text,
            "score": float(score),
            "coverage": float(coverage),
            "title": title,
            "node_id": node_id,
            "label": label,
        }

    def _passage_excerpt(self, text: str, query: str, max_tokens: int) -> str:
        text = " ".join(str(text or "").split())
        if self._token_count(text) <= max_tokens:
            return text
        terms = self._content_terms(query)
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
        scored = []
        for index, sentence in enumerate(sentences or [text]):
            words = self._content_terms(sentence)
            scored.append((len(terms & words), -index, sentence))
        selected = []
        used = 0
        for _, _, sentence in sorted(scored, reverse=True):
            cost = self._token_count(sentence)
            if cost <= 0:
                continue
            if selected and used + cost > max_tokens:
                continue
            if cost > max_tokens:
                sentence = self._truncate_words(sentence, max_tokens - used)
                cost = self._token_count(sentence)
            selected.append(sentence)
            used += cost
            if used >= max_tokens:
                break
        return " ".join(selected) if selected else self._truncate_words(text, max_tokens)

    def _coverage_text_score(self, question: str, text: str) -> float:
        q_terms = self._content_terms(question)
        t_terms = self._content_terms(text)
        if not q_terms or not t_terms:
            return 0.0
        return len(q_terms & t_terms) / max(1, len(q_terms))

    def _is_redundant(self, line: str, selected_lines: Sequence[str], threshold: float) -> bool:
        if threshold <= 0 or threshold >= 1.0:
            return False
        terms = self._content_terms(line)
        if not terms:
            return False
        for selected in selected_lines:
            other_terms = self._content_terms(selected)
            if not other_terms:
                continue
            overlap = len(terms & other_terms) / max(1, len(terms | other_terms))
            if overlap >= threshold:
                return True
        return False

    def _token_count(self, text: str) -> int:
        return len(re.findall(r"\S+", str(text or "")))

    def _truncate_words(self, text: str, max_tokens: int) -> str:
        tokens = re.findall(r"\S+", str(text or ""))
        if len(tokens) <= max_tokens:
            return str(text or "").strip()
        return " ".join(tokens[:max(0, max_tokens)]).strip()

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

    def _entity_chunk_count(self, graph: nx.DiGraph, entity_id: str) -> int:
        chunks = set()
        for neighbor in list(graph.successors(entity_id)) + list(graph.predecessors(entity_id)):
            if neighbor not in graph:
                continue
            attrs = graph.nodes[neighbor]
            if attrs.get("node_type") == "chunk":
                chunks.add(neighbor)
            elif attrs.get("node_type") == "sentence":
                chunk_id = attrs.get("metadata", {}).get("chunk_id")
                if chunk_id:
                    chunks.add(chunk_id)
        return max(1, len(chunks))

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
