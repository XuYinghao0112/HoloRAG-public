import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .biased_pagerank import GranularityBiasedPageRank
from .config import HoloRAGConfig
from .embedding_model import NVEmbedV2Encoder
from .evidence_extractor import EvidenceExtractor
from .graph_builder import HierarchicalGraphBuilder
from .intent_parser import IntentParser
from .llm_client import LocalLLMClient
from .qa_reader import (
    build_answer_focus_messages,
    build_hipporag_hop_messages,
    build_hipporag_qa_messages,
    build_short_answer_normalization_messages,
    parse_hipporag_qa_response,
)
from .query_decomposer import QueryDecomposer
from .recognition_filter import RecognitionFilter
from .seed_selector import SeedSelector
from .sentence_segmenter import SentenceSegmenter
from .triple_extractor import TripleExtractor
from .utils import (
    clean_entity_text,
    cosine_similarity_matrix,
    dump_pickle,
    ensure_dir,
    entity_match_score,
    lexical_overlap_score,
    load_pickle,
    normalize_scores,
    normalize_entity_key,
    text_contains_entity,
    looks_like_named_entity,
)

logger = logging.getLogger(__name__)


class HoloRAG:
    def __init__(self, config: Optional[HoloRAGConfig] = None) -> None:
        self.config = config or HoloRAGConfig()
        self.artifact_dir = ensure_dir(self.config.save_dir)
        self.index_path = os.path.join(self.artifact_dir, "holorag_index.pkl")
        self.embedder = NVEmbedV2Encoder(self.config)
        self.llm_client = LocalLLMClient(self.config)
        self.sentence_segmenter = SentenceSegmenter()
        self.triple_extractor = TripleExtractor(self.llm_client)
        self.intent_parser = IntentParser(self.llm_client)
        self.query_decomposer = QueryDecomposer(self.llm_client)

        cache_dir = os.getenv("HOLORAG_CACHE_DIR", os.path.join(self.artifact_dir, "cache"))
        if hasattr(self.triple_extractor, "set_cache_dir"):
            self.triple_extractor.set_cache_dir(cache_dir)
        if hasattr(self.query_decomposer, "set_cache_dir"):
            self.query_decomposer.set_cache_dir(cache_dir)

        self.recognition_filter = RecognitionFilter(self.config, self.llm_client)
        self.seed_selector = SeedSelector()
        self.page_rank = GranularityBiasedPageRank(self.config)
        self.evidence_extractor = EvidenceExtractor()
        self.graph_builder = HierarchicalGraphBuilder(
            config=self.config,
            sentence_segmenter=self.sentence_segmenter,
            triple_extractor=self.triple_extractor,
            embedder=self.embedder,
        )
        self.state: Optional[Dict] = None

    def index(self, documents: List[Dict[str, str]]) -> Dict:
        self.llm_client.reset_stats()
        logger.info("Building HoloRAG hierarchical graph for %d documents", len(documents))
        self.state = self.graph_builder.build(documents)
        dump_pickle(self.index_path, self.state)
        logger.info("Saved HoloRAG index to %s", self.index_path)
        return {"index_path": self.index_path, "stats": self.describe_index()}

    def load(self) -> Dict:
        if self.state is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(
                    f"HoloRAG index not found at {self.index_path}. Run the index command first."
                )
            self.state = load_pickle(self.index_path)
        return self.state

    def describe_index(self) -> Dict:
        state = self.state or {}
        graph: nx.DiGraph = state.get("graph", nx.DiGraph())
        counts = {"entity": 0, "sentence": 0, "chunk": 0}
        for _, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type")
            if node_type in counts:
                counts[node_type] += 1
        return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "layer_counts": counts}

    def query(self, query: str, query_hints: Optional[Dict] = None) -> Dict:
        state = self.load()
        graph: nx.DiGraph = state["graph"]
        self._ensure_fact_index(state, graph)
        self._prepare_retrieval_objects(state, graph)

        alpha = self.intent_parser.predict(query) if self.config.enable_intent_routing else self._fallback_alpha()
        raw_query_entities = self.triple_extractor.extract_query_entities(query)
        initial_sub_questions = self.query_decomposer.decompose(query)
        if not initial_sub_questions:
            initial_sub_questions = [query.strip()]
        query_entity_resolutions = self._resolve_query_entities(query, initial_sub_questions, graph, raw_query_entities)
        confident_entity_resolutions = [item for item in query_entity_resolutions if item.get("confident")]
        if confident_entity_resolutions:
            sub_questions = self.query_decomposer.decompose(query, resolved_entities=confident_entity_resolutions)
            if not sub_questions:
                sub_questions = initial_sub_questions
        else:
            sub_questions = initial_sub_questions
        query_entities = [item["resolved_text"] for item in confident_entity_resolutions] or raw_query_entities
        query_entity_node_ids = [item["node_id"] for item in confident_entity_resolutions]

        ranked_facts, dense_chunk_scores, graph_chunk_scores = self._hipporag_backbone(
            query=query,
            sub_questions=sub_questions,
            query_entities=query_entities,
            query_entity_node_ids=query_entity_node_ids,
            graph=graph,
            state=state,
        )
        layer_scores, layer_buckets, reasoning_chain, bridge_entities = self._build_multigranular_candidates(
            query=query,
            query_entities=query_entities,
            query_entity_node_ids=query_entity_node_ids,
            sub_questions=sub_questions,
            graph=graph,
            state=state,
            ranked_facts=ranked_facts,
            dense_chunk_scores=dense_chunk_scores,
            graph_chunk_scores=graph_chunk_scores,
        )
        filtered_scores = self._apply_recognition_filter(query, graph, layer_scores)
        reset_scores = self._build_reset_scores(
            graph=graph,
            filtered_scores=filtered_scores,
            dense_chunk_scores=dense_chunk_scores,
            graph_chunk_scores=graph_chunk_scores,
            ranked_facts=ranked_facts,
            query_entity_node_ids=query_entity_node_ids,
        )
        seeds = self._build_seed_view(graph, reset_scores)
        ranked = self.page_rank.run(graph, alpha=alpha, seed_scores={}, prior_scores=reset_scores)

        ranked_nodes = [
            {"node_id": node_id, "score": float(score), "node_type": graph.nodes[node_id].get("node_type")}
            for node_id, score in sorted(ranked.items(), key=lambda item: item[1], reverse=True)
        ]
        ranked_passages = self._extract_ranked_passages(graph, ranked, dense_chunk_scores, graph_chunk_scores, ranked_facts)
        evidence = self.evidence_extractor.extract(
            graph=graph,
            ranked_nodes=ranked_nodes,
            alpha=alpha,
            ranked_passages=ranked_passages,
        )
        evidence["retrieved_passages"] = ranked_passages[: self.config.passage_output_top_k]
        evidence["reasoning_chain"] = reasoning_chain
        qa_result = self._generate_final_qa_answer(query, ranked_passages, reasoning_chain, ranked_facts)
        evidence["qa_messages"] = qa_result["messages"]
        qa_context_passages = list(qa_result.get("focused_passages") or qa_result.get("selected_passages") or [])
        evidence["qa_context"] = self._build_passage_context(
            qa_context_passages,
            min(len(qa_context_passages), self.config.qa_passage_top_k),
        )

        result = {
            "query": query,
            "alpha": alpha,
            "query_entities": query_entities,
            "raw_query_entities": raw_query_entities,
            "query_entity_resolutions": query_entity_resolutions,
            "sub_questions": sub_questions,
            "sub_question_source": "model_or_heuristic",
            "seeds": seeds,
            "ranked_facts": ranked_facts[: self.config.fact_output_top_k],
            "bridge_entities": bridge_entities,
            "reasoning_chain": reasoning_chain,
            "ranked_nodes": ranked_nodes[:20],
            "ranked_passages": ranked_passages[: self.config.passage_output_top_k],
            "evidence": evidence,
            "qa_thought": qa_result["thought"],
            "predicted_answer": qa_result["answer"],
            "qa_raw_response": qa_result["raw_response"],
        }
        result_path = os.path.join(self.artifact_dir, "last_query_result.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    def _prepare_retrieval_objects(self, state: Dict, graph: nx.DiGraph) -> None:
        cache = state.setdefault("retrieval_cache", {})
        if "passage_node_ids" not in cache:
            cache["passage_node_ids"] = [node_id for node_id, attrs in graph.nodes(data=True) if attrs.get("node_type") == "chunk"]
            cache["passage_embeddings"] = np.asarray(
                [state["embeddings"]["chunk"][node_id] for node_id in cache["passage_node_ids"]],
                dtype=np.float32,
            ) if cache["passage_node_ids"] else np.zeros((0, 1), dtype=np.float32)
        if "fact_node_ids" not in cache:
            cache["fact_node_ids"] = list(state.get("embeddings", {}).get("fact", {}).keys())
            cache["fact_embeddings"] = np.asarray(
                [state["embeddings"]["fact"][node_id] for node_id in cache["fact_node_ids"]],
                dtype=np.float32,
            ) if cache["fact_node_ids"] else np.zeros((0, 1), dtype=np.float32)
        state.setdefault("query_to_embedding", {"fact": {}, "passage": {}, "entity": {}, "sentence": {}})

    def _hipporag_backbone(
        self,
        query: str,
        sub_questions: Sequence[str],
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
    ) -> Tuple[List[Dict], Dict[str, float], Dict[str, float]]:
        retrieval_queries = [query] + [item for item in sub_questions if item]
        self._get_query_embeddings(retrieval_queries, state)
        # Keep the retrieval backbone query-first like HippoRAG, but let decomposed hops
        # contribute to fact linking and passage association instead of treating them as
        # reader-only hints. This helps bridge completion without per-question rules.
        fact_scores = self._get_multi_query_fact_scores(query, sub_questions, state)
        ranked_facts = self._rerank_facts(query, sub_questions, fact_scores, graph, state)
        dense_chunk_scores = self._dense_passage_retrieval_multi(query, sub_questions, state)
        if ranked_facts:
            graph_chunk_scores = self._graph_search_with_fact_entities(
                query,
                sub_questions,
                ranked_facts,
                query_entities,
                query_entity_node_ids,
                graph,
                state,
            )
        else:
            graph_chunk_scores = dense_chunk_scores
        return ranked_facts, dense_chunk_scores, graph_chunk_scores

    def _build_multigranular_candidates(
        self,
        query: str,
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
        sub_questions: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
        ranked_facts: Sequence[Dict],
        dense_chunk_scores: Dict[str, float],
        graph_chunk_scores: Dict[str, float],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[List[Dict]]], List[Dict], List[str]]:
        entity_scores = self._entity_candidates(query_entities, query_entity_node_ids, graph, state, ranked_facts)
        sentence_scores, reasoning_chain, bridge_entities = self._sentence_candidates(
            query=query,
            sub_questions=sub_questions,
            graph=graph,
            state=state,
            ranked_facts=ranked_facts,
            query_entities=query_entities,
            query_entity_node_ids=query_entity_node_ids,
        )
        chunk_scores = self._chunk_candidates(graph, dense_chunk_scores, graph_chunk_scores, entity_scores, sentence_scores, ranked_facts)
        layer_scores = {
            "entity": entity_scores,
            "sentence": sentence_scores,
            "chunk": chunk_scores,
        }
        layer_buckets = {
            "entity": [[{"node_id": node_id, "score": score} for node_id, score in sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.entity_top_k]]],
            "sentence": [[{"node_id": node_id, "score": score} for node_id, score in sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.sentence_top_k]]],
            "chunk": [[{"node_id": node_id, "score": score} for node_id, score in sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.chunk_top_k]]],
        }
        return layer_scores, layer_buckets, reasoning_chain, bridge_entities

    def _entity_candidates(
        self,
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, float]:
        if not query_entities:
            query_entities = []
        entity_scores = self._dense_layer_retrieval(
            queries=list(query_entities) or [""],
            node_embeddings=state["embeddings"].get("entity", {}),
            graph=graph,
            instruction="Retrieve the matching entity node.",
            top_k_value=self.config.entity_top_k,
            query_text_type="query",
        ) if query_entities else {}
        for node_id in query_entity_node_ids:
            if node_id in graph:
                entity_scores[node_id] = max(entity_scores.get(node_id, 0.0), 1.0)
        for fact in ranked_facts[: self.config.linking_top_k]:
            entity_scores[fact["head_id"]] = max(entity_scores.get(fact["head_id"], 0.0), float(fact["score"]))
            entity_scores[fact["tail_id"]] = max(entity_scores.get(fact["tail_id"], 0.0), float(fact["score"]))
        return normalize_scores(entity_scores)

    def _sentence_candidates(
        self,
        query: str,
        sub_questions: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
        ranked_facts: Sequence[Dict],
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
    ) -> Tuple[Dict[str, float], List[Dict], List[str]]:
        sentence_scores: Dict[str, float] = defaultdict(float)
        reasoning_chain: List[Dict] = []
        global_scores = self._dense_layer_retrieval(
            queries=[query],
            node_embeddings=state["embeddings"].get("sentence", {}),
            graph=graph,
            instruction="Retrieve the most relevant sentence evidence.",
            top_k_value=self.config.sentence_top_k,
            query_text_type="query",
        )
        for sentence_id, score in global_scores.items():
            sentence_scores[sentence_id] = max(sentence_scores.get(sentence_id, 0.0), score)
        first_hop_sentence_ids: List[str] = []
        for fact in ranked_facts[: self.config.linking_top_k]:
            sentence_id = fact.get("sentence_id")
            if sentence_id:
                sentence_scores[sentence_id] = max(sentence_scores.get(sentence_id, 0.0), 1.25 * float(fact["score"]))
                if sentence_id not in first_hop_sentence_ids:
                    first_hop_sentence_ids.append(sentence_id)
            for entity_id in (fact.get("head_id"), fact.get("tail_id")):
                if entity_id not in graph:
                    continue
                for neighbor_sentence_id in self._sentences_for_entity(graph, entity_id):
                    if neighbor_sentence_id not in graph:
                        continue
                    lexical = lexical_overlap_score(query, graph.nodes[neighbor_sentence_id].get("text", ""))
                    induced = 0.35 * float(fact["score"]) + 0.25 * lexical
                    sentence_scores[neighbor_sentence_id] = max(sentence_scores.get(neighbor_sentence_id, 0.0), induced)

        resolved_entity_boosts = self._resolved_entity_sentence_boost(
            graph=graph,
            query=query,
            sub_question=query,
            query_entity_node_ids=query_entity_node_ids,
        )
        for sentence_id, score in resolved_entity_boosts.items():
            sentence_scores[sentence_id] = max(sentence_scores.get(sentence_id, 0.0), score)

        candidate_sentence_ids = first_hop_sentence_ids or [fact.get("sentence_id") for fact in ranked_facts[: self.config.linking_top_k] if fact.get("sentence_id")]
        bridge_entities = self._extract_bridge_entities(
            graph=graph,
            candidate_sentence_ids=candidate_sentence_ids,
            sub_questions=sub_questions,
            query=query,
            query_entities=list(query_entities),
            ranked_facts=ranked_facts,
        )
        rolling_bridge_entities = list(bridge_entities)
        ranked_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
        hop_history: List[Dict] = []

        for hop_index, sub_question in enumerate(sub_questions or [query]):
            resolved_sub_question = self._expand_sub_question_with_hop_context(sub_question, hop_history)
            future_sub_questions = [item for item in sub_questions[hop_index + 1:] if item]
            hop_candidate_scores: Dict[str, float] = dict(sentence_scores)
            subquestion_scores = self._subquestion_sentence_retrieval(
                graph=graph,
                state=state,
                sub_question=resolved_sub_question,
                previous_hops=hop_history,
            )
            for sentence_id, score in subquestion_scores.items():
                boosted = 0.92 * float(score)
                hop_candidate_scores[sentence_id] = max(hop_candidate_scores.get(sentence_id, 0.0), boosted)
                sentence_scores[sentence_id] = max(sentence_scores.get(sentence_id, 0.0), 0.75 * boosted)
            if hop_index > 0 and rolling_bridge_entities:
                bridge_scores = self._bridge_entity_neighborhood_scores(
                    graph=graph,
                    state=state,
                    bridge_entities=rolling_bridge_entities,
                    hop_queries=[resolved_sub_question],
                )
                for sentence_id, score in bridge_scores.items():
                    hop_candidate_scores[sentence_id] = max(hop_candidate_scores.get(sentence_id, 0.0), score)
                    sentence_scores[sentence_id] = max(sentence_scores.get(sentence_id, 0.0), score)

            ranked_hop_sentences = sorted(hop_candidate_scores.items(), key=lambda item: item[1], reverse=True)
            top_hop_candidates: List[Tuple[str, float]] = []
            for sentence_id, score in ranked_hop_sentences:
                if sentence_id not in graph:
                    continue
                sentence_text = graph.nodes[sentence_id].get("text", "")
                hop_overlap = lexical_overlap_score(resolved_sub_question, sentence_text)
                bridge_hit = any(text_contains_entity(sentence_text, entity) for entity in rolling_bridge_entities)
                answer_hit = any(
                    hop.get("hop_answer") and str(hop.get("hop_answer")).lower() != "unknown" and text_contains_entity(sentence_text, str(hop.get("hop_answer")))
                    for hop in hop_history
                )
                continuation_score = self._sentence_continuation_score(
                    graph=graph,
                    sentence_id=sentence_id,
                    future_sub_questions=future_sub_questions,
                    previous_hops=hop_history,
                )
                adjusted = (
                    0.44 * float(score)
                    + 0.28 * hop_overlap
                    + 0.18 * continuation_score
                    + (0.07 if bridge_hit else 0.0)
                    + (0.03 if answer_hit else 0.0)
                )
                top_hop_candidates.append((sentence_id, adjusted))
                if len(top_hop_candidates) >= self.config.sentence_step_top_k * 2:
                    break
            top_hop_candidates.sort(key=lambda item: item[1], reverse=True)
            top_hop_candidates = top_hop_candidates[: self.config.sentence_step_top_k]
            if not top_hop_candidates:
                continue

            hop_answer_result = self._extract_hop_answer(
                sub_question=resolved_sub_question,
                candidate_sentence_ids=[sentence_id for sentence_id, _ in top_hop_candidates],
                graph=graph,
                previous_hops=hop_history,
            )
            best_sentence_id, best_score = top_hop_candidates[0]
            best_sentence = graph.nodes[best_sentence_id].get("text", "")
            source_title = graph.nodes[best_sentence_id].get("metadata", {}).get("title", "")
            hop_record = {
                "sub_question": sub_question,
                "resolved_sub_question": resolved_sub_question,
                "sentence_id": best_sentence_id,
                "sentence": best_sentence,
                "score": best_score,
                "source_title": source_title,
                "hop_answer": hop_answer_result["answer"],
                "hop_thought": hop_answer_result["thought"],
                "hop_raw_response": hop_answer_result["raw_response"],
                "evidence_passages": hop_answer_result["evidence_passages"],
            }
            reasoning_chain.append(hop_record)
            hop_history.append(hop_record)
            cleaned_hop_answer = clean_entity_text(hop_answer_result["answer"])
            if cleaned_hop_answer and cleaned_hop_answer.lower() != "unknown":
                rolling_bridge_entities.append(cleaned_hop_answer)
                if looks_like_named_entity(cleaned_hop_answer):
                    bridge_entities.append(cleaned_hop_answer)

        deduped_bridge_entities: List[str] = []
        seen_bridge_keys = set()
        for entity in bridge_entities:
            key = normalize_entity_key(entity)
            if not key or key in seen_bridge_keys:
                continue
            seen_bridge_keys.add(key)
            deduped_bridge_entities.append(entity)
        return normalize_scores(sentence_scores), reasoning_chain, deduped_bridge_entities

    def _chunk_candidates(
        self,
        graph: nx.DiGraph,
        dense_chunk_scores: Dict[str, float],
        graph_chunk_scores: Dict[str, float],
        entity_scores: Dict[str, float],
        sentence_scores: Dict[str, float],
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, float]:
        chunk_scores: Dict[str, float] = dict(graph_chunk_scores)
        for chunk_id, score in dense_chunk_scores.items():
            chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)
        for sentence_id, score in sentence_scores.items():
            if sentence_id in graph:
                chunk_id = graph.nodes[sentence_id].get("metadata", {}).get("chunk_id")
                if chunk_id:
                    chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), 0.85 * score)
        for entity_id, score in entity_scores.items():
            for chunk_id, induced in self._induce_chunk_scores_from_entity(graph, entity_id, score).items():
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), induced)
        for fact in ranked_facts[: self.config.fact_rerank_top_k]:
            chunk_id = fact.get("chunk_id")
            if chunk_id:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), float(fact["score"]))
        return normalize_scores(chunk_scores)

    def _apply_recognition_filter(
        self,
        query: str,
        graph: nx.DiGraph,
        layer_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        filtered = {}
        for layer, scores in layer_scores.items():
            node_texts = {node_id: graph.nodes[node_id].get("text", "") for node_id in scores if node_id in graph}
            filtered[layer] = normalize_scores(self.recognition_filter.rerank(query, scores, node_texts))
        return filtered

    def _build_prior_scores(
        self,
        working_graph: nx.DiGraph,
        filtered_scores: Dict[str, Dict[str, float]],
        dense_chunk_scores: Dict[str, float],
        graph_chunk_scores: Dict[str, float],
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, float]:
        priors: Dict[str, float] = defaultdict(float)
        layer_weights = {"entity": 1.0, "sentence": 0.55, "chunk": 1.0}
        for layer in ("entity", "sentence", "chunk"):
            for node_id, score in filtered_scores.get(layer, {}).items():
                if node_id in working_graph:
                    priors[node_id] += layer_weights.get(layer, 1.0) * score
        for chunk_scores in (dense_chunk_scores, graph_chunk_scores):
            for chunk_id, score in chunk_scores.items():
                if chunk_id in working_graph:
                    priors[chunk_id] += self.config.passage_node_weight * score
        for fact in ranked_facts[: self.config.fact_rerank_top_k]:
            sentence_id = fact.get("sentence_id")
            if sentence_id in working_graph:
                priors[sentence_id] += float(fact["score"])
        return dict(priors)

    def _build_reset_scores(
        self,
        graph: nx.DiGraph,
        filtered_scores: Dict[str, Dict[str, float]],
        dense_chunk_scores: Dict[str, float],
        graph_chunk_scores: Dict[str, float],
        ranked_facts: Sequence[Dict],
        query_entity_node_ids: Sequence[str],
    ) -> Dict[str, float]:
        reset_scores: Dict[str, float] = defaultdict(float)

        fact_entity_occurs: Dict[str, int] = defaultdict(int)
        for fact in ranked_facts[: self.config.fact_rerank_top_k]:
            for entity_id in (fact.get("head_id"), fact.get("tail_id")):
                if entity_id in graph:
                    fact_entity_occurs[entity_id] += 1

        for fact in ranked_facts[: self.config.fact_rerank_top_k]:
            fact_score = float(fact["score"])
            sentence_id = fact.get("sentence_id")
            chunk_id = fact.get("chunk_id")
            if sentence_id in graph:
                reset_scores[sentence_id] += 0.40 * fact_score
            if chunk_id in graph:
                reset_scores[chunk_id] += 0.20 * fact_score
            for entity_id in (fact.get("head_id"), fact.get("tail_id")):
                if entity_id not in graph:
                    continue
                weighted_fact_score = fact_score / max(1, fact_entity_occurs.get(entity_id, 1))
                reset_scores[entity_id] += weighted_fact_score
                for neighbor in list(graph.successors(entity_id)) + list(graph.predecessors(entity_id)):
                    if neighbor not in graph or graph.nodes[neighbor].get("node_type") != "entity":
                        continue
                    edge_data = graph.get_edge_data(entity_id, neighbor) or graph.get_edge_data(neighbor, entity_id) or {}
                    if "entity_alias" not in edge_data.get("edge_kinds", []):
                        continue
                    reset_scores[neighbor] += 0.60 * weighted_fact_score

        for entity_id in query_entity_node_ids:
            if entity_id in graph:
                reset_scores[entity_id] += 0.80

        for chunk_scores, scale in ((dense_chunk_scores, self.config.passage_node_weight), (graph_chunk_scores, 0.25)):
            for chunk_id, score in chunk_scores.items():
                if chunk_id in graph:
                    reset_scores[chunk_id] += scale * float(score)

        # Keep the sentence layer as an auxiliary HoloRAG signal rather than a reset driver.
        for sentence_id, score in filtered_scores.get("sentence", {}).items():
            if sentence_id in graph:
                reset_scores[sentence_id] += 0.05 * float(score)
        for entity_id, score in filtered_scores.get("entity", {}).items():
            if entity_id in graph:
                reset_scores[entity_id] += 0.04 * float(score)
        for chunk_id, score in filtered_scores.get("chunk", {}).items():
            if chunk_id in graph:
                reset_scores[chunk_id] += 0.04 * float(score)

        return dict(reset_scores)

    def _build_seed_view(self, graph: nx.DiGraph, reset_scores: Dict[str, float]) -> List[Dict]:
        ranked = sorted(reset_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.seed_budget]
        return [
            {
                "node_id": node_id,
                "score": float(score),
                "layer": graph.nodes[node_id].get("node_type", "unknown"),
            }
            for node_id, score in ranked
            if node_id in graph
        ]

    def _extract_ranked_passages(
        self,
        working_graph: nx.DiGraph,
        ranked_scores: Dict[str, float],
        dense_chunk_scores: Dict[str, float],
        graph_chunk_scores: Dict[str, float],
        ranked_facts: Sequence[Dict],
    ) -> List[Dict]:
        fact_chunk_scores = {}
        for fact in ranked_facts:
            chunk_id = fact.get("chunk_id")
            if chunk_id:
                fact_chunk_scores[chunk_id] = max(fact_chunk_scores.get(chunk_id, 0.0), float(fact["score"]))
        dense_norm = normalize_scores(dense_chunk_scores)
        graph_norm = normalize_scores(graph_chunk_scores)
        fact_norm = normalize_scores(fact_chunk_scores)
        ppr_chunk_scores: Dict[str, float] = defaultdict(float)
        for node_id, score in ranked_scores.items():
            if node_id not in working_graph:
                continue
            node_type = working_graph.nodes[node_id].get("node_type")
            if node_type == "chunk":
                ppr_chunk_scores[node_id] += float(score)
            elif node_type == "sentence":
                chunk_id = working_graph.nodes[node_id].get("metadata", {}).get("chunk_id")
                if chunk_id:
                    ppr_chunk_scores[chunk_id] += 0.7 * float(score)
            elif node_type == "entity":
                for chunk_id, induced in self._induce_chunk_scores_from_entity(working_graph, node_id, float(score)).items():
                    ppr_chunk_scores[chunk_id] += induced
        ppr_norm = normalize_scores(ppr_chunk_scores)

        chunk_ids = [node_id for node_id, attrs in working_graph.nodes(data=True) if attrs.get("node_type") == "chunk"]
        ranked_passages = []
        for chunk_id in chunk_ids:
            attrs = working_graph.nodes[chunk_id]
            dense_score = float(dense_norm.get(chunk_id, 0.0))
            graph_score = float(ppr_norm.get(chunk_id, graph_norm.get(chunk_id, 0.0)))
            fact_score = float(fact_norm.get(chunk_id, 0.0))
            final_score = (
                self.config.dense_passage_weight * dense_score
                + self.config.graph_passage_weight * graph_score
                + self.config.fact_passage_weight * fact_score
            )
            ranked_passages.append({
                "node_id": chunk_id,
                "passage_index": attrs.get("metadata", {}).get("document_index"),
                "title": attrs.get("metadata", {}).get("title", ""),
                "score": final_score,
                "text": attrs.get("text", ""),
                "score_breakdown": {"dense": dense_score, "graph": graph_score, "fact": fact_score},
            })
        ranked_passages.sort(key=lambda item: item["score"], reverse=True)
        return ranked_passages

    def _get_query_embeddings(self, queries: Sequence[str], state: Dict) -> None:
        cache = state["query_to_embedding"]
        missing = [query for query in queries if query and (query not in cache["fact"] or query not in cache["passage"])]
        if not missing:
            return
        fact_embeddings = self.embedder.encode(
            missing,
            instruction="Given a question, retrieve relevant triplet facts that matches this question.",
            text_type="query",
        )
        passage_embeddings = self.embedder.encode(
            missing,
            instruction="Given a question, retrieve relevant documents that best answer the question.",
            text_type="query",
        )
        for query, embedding in zip(missing, fact_embeddings):
            cache["fact"][query] = embedding
        for query, embedding in zip(missing, passage_embeddings):
            cache["passage"][query] = embedding

    def _get_fact_scores(self, query: str, state: Dict) -> Dict[str, float]:
        self._get_query_embeddings([query], state)
        fact_node_ids = state["retrieval_cache"]["fact_node_ids"]
        if not fact_node_ids:
            return {}
        query_embedding = state["query_to_embedding"]["fact"][query]
        scores = cosine_similarity_matrix(query_embedding, state["retrieval_cache"]["fact_embeddings"])
        return normalize_scores({fact_id: float(score) for fact_id, score in zip(fact_node_ids, scores.tolist())})

    def _get_multi_query_fact_scores(
        self,
        query: str,
        sub_questions: Sequence[str],
        state: Dict,
    ) -> Dict[str, float]:
        all_queries: List[str] = []
        for item in [query] + list(sub_questions):
            query_text = str(item or "").strip()
            if query_text and query_text not in all_queries:
                all_queries.append(query_text)
        aggregated_scores: Dict[str, float] = defaultdict(float)
        for index, query_text in enumerate(all_queries):
            query_scores = self._get_fact_scores(query_text, state)
            weight = 1.0 if index == 0 else 0.85
            for fact_id, score in query_scores.items():
                aggregated_scores[fact_id] = max(aggregated_scores.get(fact_id, 0.0), weight * float(score))
        return normalize_scores(aggregated_scores)

    def _dense_passage_retrieval(self, query: str, state: Dict) -> Dict[str, float]:
        self._get_query_embeddings([query], state)
        passage_node_ids = state["retrieval_cache"]["passage_node_ids"]
        if not passage_node_ids:
            return {}
        query_embedding = state["query_to_embedding"]["passage"][query]
        scores = cosine_similarity_matrix(query_embedding, state["retrieval_cache"]["passage_embeddings"])
        normalized = normalize_scores({node_id: float(score) for node_id, score in zip(passage_node_ids, scores.tolist())})
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True)[: self.config.retrieval_top_k])

    def _dense_passage_retrieval_multi(
        self,
        query: str,
        sub_questions: Sequence[str],
        state: Dict,
    ) -> Dict[str, float]:
        all_queries: List[str] = []
        for item in [query] + list(sub_questions):
            query_text = str(item or "").strip()
            if query_text and query_text not in all_queries:
                all_queries.append(query_text)
        aggregated_scores: Dict[str, float] = defaultdict(float)
        for index, query_text in enumerate(all_queries):
            query_scores = self._dense_passage_retrieval(query_text, state)
            weight = 1.0 if index == 0 else 0.85
            for chunk_id, score in query_scores.items():
                aggregated_scores[chunk_id] = max(aggregated_scores.get(chunk_id, 0.0), weight * float(score))
        ranked = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.retrieval_top_k]
        return dict(ranked)

    def _title_anchor_chunk_scores(
        self,
        graph: nx.DiGraph,
        mentions: Sequence[str],
        context_queries: Sequence[str],
    ) -> Dict[str, float]:
        candidate_scores: Dict[str, float] = {}
        normalized_mentions = [str(mention).strip() for mention in mentions if str(mention).strip()]
        if not normalized_mentions:
            return {}
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "chunk":
                continue
            title = str(attrs.get("metadata", {}).get("title", "")).strip()
            if not title:
                continue
            alias_score = max((entity_match_score(mention, title) for mention in normalized_mentions), default=0.0)
            if alias_score < 0.55:
                continue
            title_context = max((lexical_overlap_score(context_query, title) for context_query in context_queries if context_query), default=0.0)
            chunk_context = max((lexical_overlap_score(context_query, attrs.get("text", "")) for context_query in context_queries if context_query), default=0.0)
            candidate_scores[node_id] = max(
                candidate_scores.get(node_id, 0.0),
                0.70 * alias_score + 0.20 * title_context + 0.10 * chunk_context,
            )
        return normalize_scores(candidate_scores)

    def _rerank_facts(
        self,
        query: str,
        sub_questions: Sequence[str],
        fact_scores: Dict[str, float],
        graph: nx.DiGraph,
        state: Dict,
    ) -> List[Dict]:
        if not fact_scores:
            return []
        fact_lookup = {record["fact_id"]: record for record in state.get("facts", [])}
        semantic_candidate_ids = [
            fact_id
            for fact_id, _ in sorted(fact_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.fact_candidate_top_k]
            if fact_id in fact_lookup
        ]
        symbolic_ranked = sorted(
            state.get("facts", []),
            key=lambda record: (
                self._fact_symbolic_score(query, sub_questions, record, graph),
                float(fact_scores.get(record["fact_id"], 0.0)),
            ),
            reverse=True,
        )
        symbolic_limit = max(4, self.config.fact_candidate_top_k // 2)
        symbolic_candidate_ids = [
            record["fact_id"]
            for record in symbolic_ranked[:symbolic_limit]
            if self._fact_symbolic_score(query, sub_questions, record, graph) > 0.0
        ]
        candidate_ids = []
        for fact_id in semantic_candidate_ids + symbolic_candidate_ids:
            if fact_id in fact_lookup and fact_id not in candidate_ids:
                candidate_ids.append(fact_id)
        if not candidate_ids:
            return []

        candidate_records = [fact_lookup[fact_id] for fact_id in candidate_ids]
        fallback_ranked = self._rank_fact_candidates(query, sub_questions, candidate_records, fact_scores, graph)
        selected_records = self._llm_filter_facts(query, sub_questions, candidate_records, fallback_ranked, fact_scores, graph)
        rescored = self._rescore_and_expand_facts(
            query=query,
            sub_questions=sub_questions,
            selected_records=selected_records,
            fact_scores=fact_scores,
            graph=graph,
            state=state,
        )
        rescored.sort(key=lambda item: item["score"], reverse=True)
        return rescored

    def _fact_candidate_score(
        self,
        query: str,
        sub_questions: Sequence[str],
        record: Dict,
        fact_scores: Dict[str, float],
        graph: nx.DiGraph,
    ) -> float:
        hop_overlap = max(
            [lexical_overlap_score(query, record["text"])]
            + [lexical_overlap_score(sub_question, record["text"]) for sub_question in sub_questions or []]
        )
        relation_score = self._fact_relation_compatibility(query, sub_questions, record, graph)
        structural_score = self._fact_structure_score(record, graph)
        terminal_penalty = self._fact_terminal_penalty(query, sub_questions, record, graph)
        concrete_bonus = 0.0
        for key in ("head_id", "tail_id"):
            node_id = str(record.get(key, ""))
            if node_id.startswith("entity:title_anchor_"):
                concrete_bonus += 0.05
        score = (
            0.45 * float(fact_scores.get(record["fact_id"], 0.0))
            + 0.15 * hop_overlap
            + 0.20 * relation_score
            + 0.15 * structural_score
            + concrete_bonus
            - terminal_penalty
        )
        return max(0.0, score)

    def _rank_fact_candidates(
        self,
        query: str,
        sub_questions: Sequence[str],
        candidate_records: Sequence[Dict],
        fact_scores: Dict[str, float],
        graph: nx.DiGraph,
    ) -> List[Dict]:
        ranked = sorted(
            candidate_records,
            key=lambda record: self._fact_candidate_score(query, sub_questions, record, fact_scores, graph),
            reverse=True,
        )
        diversified: List[Dict] = []
        seen_pairs = set()
        for record in ranked:
            pair = tuple(sorted((str(record.get("head_id", "")), str(record.get("tail_id", "")))))
            if pair in seen_pairs and len(diversified) >= self.config.linking_top_k:
                continue
            diversified.append(record)
            seen_pairs.add(pair)
        return diversified

    def _select_subquestion_coverage_facts(
        self,
        sub_questions: Sequence[str],
        candidate_records: Sequence[Dict],
        fact_scores: Dict[str, float],
        graph: nx.DiGraph,
    ) -> List[Dict]:
        coverage: List[Dict] = []
        used_fact_ids = set()
        used_sentences = set()
        for sub_question in sub_questions:
            ranked = sorted(
                candidate_records,
                key=lambda record: (
                    lexical_overlap_score(sub_question, record["text"]),
                    self._fact_candidate_score(sub_question, [sub_question], record, fact_scores, graph),
                ),
                reverse=True,
            )
            for record in ranked:
                overlap = lexical_overlap_score(sub_question, record["text"])
                if overlap <= 0.0:
                    continue
                if record["fact_id"] in used_fact_ids:
                    continue
                sentence_id = record.get("sentence_id")
                if sentence_id and sentence_id in used_sentences and len(ranked) > 1:
                    continue
                coverage.append(record)
                used_fact_ids.add(record["fact_id"])
                if sentence_id:
                    used_sentences.add(sentence_id)
                break
        return coverage

    def _llm_filter_facts(
        self,
        query: str,
        sub_questions: Sequence[str],
        candidate_records: Sequence[Dict],
        fallback_ranked: Sequence[Dict],
        fact_scores: Dict[str, float],
        graph: nx.DiGraph,
    ) -> List[Dict]:
        if not candidate_records:
            return []
        selection_budget = max(1, min(4, self.config.linking_top_k))
        candidate_lines = []
        for index, record in enumerate(candidate_records):
            candidate_lines.append(f"{index}: ({record['text']})")
        fallback_indices = list(range(min(len(fallback_ranked), selection_budget)))
        fallback = {"fact_indices": fallback_indices}
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Select the candidate facts that are most useful for answering the question. "
                "Prefer a small set of facts that directly support multi-hop reasoning, cover different hops, and introduce concrete bridge entities. "
                "Prefer bridge relations over terminal attributes unless the attribute is the final target being asked. "
                "Avoid generic background facts that overlap only on broad topics like wars, eras, or reconstruction unless they connect two concrete facts. "
                "Return JSON with key fact_indices, a list of integer indices from the candidate list."
            ),
            user_prompt=(
                f"Question:\n{query}\n\n"
                + (
                    "Sub-questions:\n"
                    + "\n".join(f"- {item}" for item in sub_questions if item)
                    + "\n\n"
                    if sub_questions
                    else ""
                )
                +
                "Candidate facts:\n"
                + "\n".join(candidate_lines)
                + f"\n\nReturn at most {selection_budget} indices."
            ),
            fallback=fallback,
            max_tokens=256,
        )
        selected_indices = payload.get("fact_indices", fallback_indices)
        cleaned_indices: List[int] = []
        for item in selected_indices:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidate_records) and index not in cleaned_indices:
                cleaned_indices.append(index)
        if not cleaned_indices:
            cleaned_indices = fallback_indices
        selected_records = [candidate_records[index] for index in cleaned_indices[:selection_budget]]
        deterministic_records = list(fallback_ranked[:selection_budget])
        if not selected_records:
            selected_records = list(deterministic_records)

        fallback_score_lookup = {
            record["fact_id"]: float(len(fallback_ranked) - index) / max(1, len(fallback_ranked))
            for index, record in enumerate(fallback_ranked)
        }
        candidate_fact_scores = {
            record["fact_id"]: fallback_score_lookup.get(record["fact_id"], 0.0)
            for record in candidate_records
        }
        coverage_records = self._select_subquestion_coverage_facts(sub_questions, candidate_records, candidate_fact_scores, graph)
        merged: List[Dict] = []
        seen_fact_ids = set()
        # Use the LLM as a filter, but keep deterministic high-utility facts in the loop.
        # This is closer to HippoRAG's fact-first retrieval while being more stable with weaker local models.
        deterministic_limit = max(1, min(2, selection_budget))
        deterministic_records = deterministic_records[:deterministic_limit]
        # Stay query-first like HippoRAG: use sub-question coverage only as a light supplement.
        coverage_limit = max(0, min(1, selection_budget // 2))
        coverage_records = coverage_records[:coverage_limit]
        for record in selected_records + deterministic_records + coverage_records:
            if record["fact_id"] in seen_fact_ids:
                continue
            merged.append(record)
            seen_fact_ids.add(record["fact_id"])
            if len(merged) >= selection_budget + deterministic_limit + coverage_limit:
                break
        if not merged:
            merged = list(fallback_ranked[:selection_budget])
        return merged

    def _rescore_and_expand_facts(
        self,
        query: str,
        sub_questions: Sequence[str],
        selected_records: Sequence[Dict],
        fact_scores: Dict[str, float],
        graph: nx.DiGraph,
        state: Dict,
    ) -> List[Dict]:
        rescored: List[Dict] = []
        selected_by_id = {}
        for rank, record in enumerate(selected_records):
            base_score = self._fact_candidate_score(query, sub_questions, record, fact_scores, graph)
            score = base_score + 0.02 * max(0, len(selected_records) - rank)
            enriched = {**record, "score": float(score)}
            rescored.append(enriched)
            selected_by_id[record["fact_id"]] = enriched

        facts_by_sentence: Dict[str, List[Dict]] = defaultdict(list)
        for record in state.get("facts", []):
            sentence_id = record.get("sentence_id")
            if sentence_id:
                facts_by_sentence[sentence_id].append(record)

        companion_records: List[Dict] = []
        for parent in rescored:
            parent_head = parent.get("head_id")
            parent_tail = parent.get("tail_id")
            for companion in facts_by_sentence.get(parent.get("sentence_id"), []):
                if companion["fact_id"] in selected_by_id:
                    continue
                if companion.get("head_id") not in {parent_head, parent_tail} and companion.get("tail_id") not in {parent_head, parent_tail}:
                    continue
                companion_base = self._fact_candidate_score(query, sub_questions, companion, fact_scores, graph)
                if companion_base <= 0.0:
                    continue
                if self._fact_terminal_penalty(query, sub_questions, companion, graph) >= 0.12 and self._fact_structure_score(companion, graph) < 0.2:
                    continue
                companion_score = min(0.85 * float(parent["score"]), companion_base)
                if companion_score < 0.55 * float(parent["score"]):
                    continue
                companion_records.append({**companion, "score": companion_score})
                selected_by_id[companion["fact_id"]] = companion_records[-1]

        merged = rescored + companion_records
        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[: self.config.fact_rerank_top_k]

    def _fact_relation_compatibility(
        self,
        query: str,
        sub_questions: Sequence[str],
        record: Dict,
        graph: nx.DiGraph,
    ) -> float:
        relation = self._fact_relation_text(record, graph)
        if not relation:
            return 0.0
        all_text = " ".join([query] + list(sub_questions)).lower()
        relation_score = 0.0
        relation_cues = self._extract_relation_cues(" ".join([query] + list(sub_questions)))
        if self._relation_matches_cues(relation, relation_cues):
            relation_score = max(relation_score, 0.35)

        tail_text = graph.nodes.get(record.get("tail_id"), {}).get("text", "") if record.get("tail_id") in graph else ""
        temporal_query = any(token in all_text for token in ["when", "year", "date", "time"])
        if temporal_query:
            if any(token in relation.lower() for token in ["date", "year", "time", "born", "death", "died", "statehood", "became", "ratif", "admitt"]):
                relation_score = max(relation_score, 0.55)
            if re.search(r"\b\d{4}\b", tail_text) or re.search(r"[A-Z][a-z]+ \d{1,2} \d{4}", tail_text):
                relation_score = max(relation_score, 0.45)

        if any(token in all_text for token in ["part of the u", "united states", "statehood", "became"]) and any(
            token in relation.lower() for token in ["statehood", "became", "ratif", "admitt"]
        ):
            relation_score = max(relation_score, 0.65)
        if any(token in all_text for token in ["part of the u", "united states", "statehood", "became"]) and any(
            token in tail_text.lower() for token in ["constitution", "united states"]
        ):
            relation_score = max(relation_score, 0.55)
        if "governor" in all_text and any(token in relation.lower() for token in ["governor", "position", "office", "term"]):
            relation_score = max(relation_score, 0.35)
        if any(token in all_text for token in ["died", "death"]) and any(token in relation.lower() for token in ["death", "died"]):
            relation_score = max(relation_score, 0.45)
        return relation_score

    def _fact_structure_score(
        self,
        record: Dict,
        graph: nx.DiGraph,
    ) -> float:
        relation = self._fact_relation_text(record, graph).lower()
        head_text = graph.nodes.get(record.get("head_id"), {}).get("text", "") if record.get("head_id") in graph else ""
        tail_text = graph.nodes.get(record.get("tail_id"), {}).get("text", "") if record.get("tail_id") in graph else ""
        tail_lower = str(tail_text).lower()
        score = 0.0
        bridge_relations = [
            "governor_of", "located in", "located_in", "directed by", "directed_by",
            "written by", "written_by", "born in", "born_in", "capital of", "capital_of",
            "is in", "is_in", "positionheld", "position held"
        ]
        if any(token in relation for token in bridge_relations):
            score = max(score, 0.75)
        if any(token in relation for token in ["statehood", "ratify", "admitt", "became"]):
            score = max(score, 0.70)
        if re.search(r"\b\d{4}\b", tail_lower):
            score = max(score, 0.15)
        if any(month in tail_lower for month in [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]):
            score = max(score, 0.10)
        if head_text and tail_text and not re.search(r"\b\d{4}\b", head_text.lower()) and not re.search(r"\b\d{4}\b", tail_text.lower()):
            score = max(score, 0.25)
        return score

    def _fact_terminal_penalty(
        self,
        query: str,
        sub_questions: Sequence[str],
        record: Dict,
        graph: nx.DiGraph,
    ) -> float:
        relation = self._fact_relation_text(record, graph).lower()
        all_text = " ".join([query] + list(sub_questions)).lower()
        penalty = 0.0
        if any(token in relation for token in ["dateofbirth", "date of birth", "dateofdeath", "date of death", "born on", "died on"]):
            penalty = max(penalty, 0.18)
        if any(token in relation for token in ["termtime", "timeinoffice", "time in office"]):
            penalty = max(penalty, 0.12)
        if any(token in relation for token in ["position", "positionheld", "position held"]) and "governor" in all_text:
            penalty = max(penalty, 0.08)
        if any(token in all_text for token in ["death date", "date of death", "when did", "when was"]) and any(
            token in relation for token in ["dateofdeath", "date of death", "died on"]
        ):
            penalty = max(0.0, penalty - 0.12)
        return penalty

    def _fact_symbolic_score(
        self,
        query: str,
        sub_questions: Sequence[str],
        record: Dict,
        graph: nx.DiGraph,
    ) -> float:
        relation_score = self._fact_relation_compatibility(query, sub_questions, record, graph)
        structure_score = self._fact_structure_score(record, graph)
        lexical_score = max(
            [lexical_overlap_score(query, record["text"])]
            + [lexical_overlap_score(sub_question, record["text"]) for sub_question in sub_questions or []]
        )
        penalty = self._fact_terminal_penalty(query, sub_questions, record, graph)
        return max(0.0, 0.45 * relation_score + 0.35 * structure_score + 0.20 * lexical_score - penalty)

    def _fact_relation_text(self, record: Dict, graph: nx.DiGraph) -> str:
        sentence_id = record.get("sentence_id")
        if sentence_id not in graph:
            return ""
        sentence_metadata = graph.nodes[sentence_id].get("metadata", {})
        head_text = graph.nodes.get(record.get("head_id"), {}).get("text", "") if record.get("head_id") in graph else ""
        tail_text = graph.nodes.get(record.get("tail_id"), {}).get("text", "") if record.get("tail_id") in graph else ""
        normalized_head = clean_entity_text(head_text).lower()
        normalized_tail = clean_entity_text(tail_text).lower()
        for triple in sentence_metadata.get("triples", []):
            triple_head = clean_entity_text(str(triple.get("head", ""))).lower()
            triple_tail = clean_entity_text(str(triple.get("tail", ""))).lower()
            if triple_head == normalized_head and triple_tail == normalized_tail:
                return str(triple.get("relation", ""))
        return ""

    def _graph_search_with_fact_entities(
        self,
        query: str,
        sub_questions: Sequence[str],
        ranked_facts: Sequence[Dict],
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
    ) -> Dict[str, float]:
        dense_passages = self._dense_passage_retrieval_multi(query, sub_questions, state)
        title_anchor_chunk_scores = self._title_anchor_chunk_scores(
            graph=graph,
            mentions=list(query_entities),
            context_queries=[query] + [item for item in sub_questions if item],
        )
        personalization: Dict[str, float] = defaultdict(float)
        entity_chunk_count: Dict[str, int] = defaultdict(int)
        linked_entities = set()
        for fact in ranked_facts[: self.config.fact_rerank_top_k]:
            for entity_id in (fact["head_id"], fact["tail_id"]):
                linked_entities.add(entity_id)
                entity_chunk_count[entity_id] = max(entity_chunk_count.get(entity_id, 0), len(self._sentences_for_entity(graph, entity_id)))
        for fact in ranked_facts[: self.config.fact_rerank_top_k]:
            score = float(fact["score"])
            for entity_id in (fact["head_id"], fact["tail_id"]):
                personalization[entity_id] += score / max(1, entity_chunk_count.get(entity_id, 1))
        for entity_id in query_entity_node_ids:
            if entity_id in graph:
                personalization[entity_id] += 0.8
        for entity_text in query_entities:
            if any(entity_match_score(entity_text, graph.nodes[node_id].get("text", "")) >= 1.0 for node_id in query_entity_node_ids if node_id in graph):
                continue
            entity_id = self._find_entity_node_by_text(graph, entity_text, context_text=" ".join([query] + [item for item in sub_questions if item]))
            if entity_id is not None:
                personalization[entity_id] += 0.8
                linked_entities.add(entity_id)

        for entity_id in list(linked_entities):
            if entity_id not in graph:
                continue
            for neighbor in list(graph.successors(entity_id)) + list(graph.predecessors(entity_id)):
                if neighbor not in graph or graph.nodes[neighbor].get("node_type") != "entity":
                    continue
                edge_data = graph.get_edge_data(entity_id, neighbor) or graph.get_edge_data(neighbor, entity_id) or {}
                if "entity_alias" not in edge_data.get("edge_kinds", []):
                    continue
                personalization[neighbor] += 0.35 * personalization.get(entity_id, 0.0)
        for chunk_id, score in dense_passages.items():
            personalization[chunk_id] += self.config.passage_node_weight * score
        for chunk_id, score in title_anchor_chunk_scores.items():
            personalization[chunk_id] += 0.45 * score
            for entity_id in graph.nodes[chunk_id].get("metadata", {}).get("entity_ids", []):
                if entity_id in graph:
                    personalization[entity_id] += 0.12 * score
        total = sum(personalization.values()) or 1.0
        personalization = {node_id: score / total for node_id, score in personalization.items() if node_id in graph}
        if not personalization:
            return dense_passages
        pagerank_scores = self.page_rank.run(
            graph,
            alpha=self._fallback_alpha(),
            seed_scores={},
            prior_scores=personalization,
        )
        chunk_scores: Dict[str, float] = defaultdict(float)
        for node_id, score in pagerank_scores.items():
            if node_id not in graph:
                continue
            node_type = graph.nodes[node_id].get("node_type")
            if node_type == "chunk":
                chunk_scores[node_id] += float(score)
            elif node_type == "sentence":
                chunk_id = graph.nodes[node_id].get("metadata", {}).get("chunk_id")
                if chunk_id:
                    chunk_scores[chunk_id] += 0.6 * float(score)
            elif node_type == "entity":
                for chunk_id, induced in self._induce_chunk_scores_from_entity(graph, node_id, float(score)).items():
                    chunk_scores[chunk_id] += induced
        return normalize_scores(chunk_scores)

    def _resolve_query_entities(
        self,
        query: str,
        sub_questions: Sequence[str],
        graph: nx.DiGraph,
        query_entities: Sequence[str],
    ) -> List[Dict]:
        resolutions: List[Dict] = []
        used_node_ids = set()
        relation_cues = self._extract_relation_cues(query)
        context_queries = [query] + [item for item in sub_questions if item]

        for entity_text in query_entities:
            candidates = []
            for node_id, attrs in graph.nodes(data=True):
                if attrs.get("node_type") != "entity":
                    continue
                metadata = attrs.get("metadata", {})
                aliases = [attrs.get("text", "")]
                aliases.extend(metadata.get("aliases", []))
                aliases.extend(metadata.get("surface_forms", []))
                alias_score = max((entity_match_score(entity_text, alias) for alias in aliases), default=0.0)
                if alias_score <= 0.0:
                    continue

                sentence_ids = self._sentences_for_entity(graph, node_id)
                sentence_texts = [graph.nodes[sentence_id].get("text", "") for sentence_id in sentence_ids if sentence_id in graph]
                context_score = max(
                    [lexical_overlap_score(context_query, attrs.get("text", "")) for context_query in context_queries]
                    + [lexical_overlap_score(context_query, sentence_text) for context_query in context_queries for sentence_text in sentence_texts]
                ) if context_queries else 0.0

                relation_score = 0.0
                for sentence_id in sentence_ids:
                    if sentence_id not in graph:
                        continue
                    sentence_text = graph.nodes[sentence_id].get("text", "")
                    if any(cue in sentence_text.lower() for cue in relation_cues):
                        relation_score = max(relation_score, 0.7)
                    for triple in graph.nodes[sentence_id].get("metadata", {}).get("triples", []):
                        relation = str(triple.get("relation", "")).lower()
                        if self._relation_matches_cues(relation, relation_cues):
                            relation_score = max(relation_score, 1.0)

                title_anchor = str(metadata.get("title_anchor", "")).strip()
                title_score = lexical_overlap_score(query, title_anchor) if title_anchor else 0.0
                compatibility_score = self._query_entity_compatibility_score(
                    query=query,
                    sub_questions=sub_questions,
                    mention=entity_text,
                    node_attrs=attrs,
                    sentence_ids=sentence_ids,
                    graph=graph,
                )
                final_score = (
                    0.38 * alias_score
                    + 0.20 * relation_score
                    + 0.14 * context_score
                    + 0.04 * title_score
                    + 0.24 * compatibility_score
                )
                candidates.append({
                    "mention": entity_text,
                    "node_id": node_id,
                    "resolved_text": attrs.get("text", ""),
                    "score": float(final_score),
                    "alias_score": float(alias_score),
                    "relation_score": float(relation_score),
                    "context_score": float(context_score),
                    "compatibility_score": float(compatibility_score),
                })

            if not candidates:
                continue
            ranked_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
            best = ranked_candidates[0]
            second_score = ranked_candidates[1]["score"] if len(ranked_candidates) > 1 else 0.0
            best["score_margin"] = float(best["score"] - second_score)
            best["confident"] = bool(
                best["score"] >= self.config.entity_resolution_score_threshold
                and (
                    best["score_margin"] >= self.config.entity_resolution_margin_threshold
                    or best["compatibility_score"] >= 0.88
                )
            )
            if best["node_id"] in used_node_ids:
                continue
            used_node_ids.add(best["node_id"])
            resolutions.append(best)
        return resolutions

    def _resolved_entity_sentence_boost(
        self,
        graph: nx.DiGraph,
        query: str,
        sub_question: str,
        query_entity_node_ids: Sequence[str],
    ) -> Dict[str, float]:
        boosts: Dict[str, float] = {}
        relation_cues = self._extract_relation_cues(sub_question or query)
        for entity_id in query_entity_node_ids:
            if entity_id not in graph:
                continue
            for sentence_id in self._sentences_for_entity(graph, entity_id):
                if sentence_id not in graph:
                    continue
                sentence_text = graph.nodes[sentence_id].get("text", "")
                lexical = max(
                    lexical_overlap_score(sub_question, sentence_text),
                    lexical_overlap_score(query, sentence_text),
                )
                relation_score = 0.0
                if any(cue in sentence_text.lower() for cue in relation_cues):
                    relation_score = max(relation_score, 0.35)
                for triple in graph.nodes[sentence_id].get("metadata", {}).get("triples", []):
                    relation = str(triple.get("relation", "")).lower()
                    if self._relation_matches_cues(relation, relation_cues):
                        relation_score = max(relation_score, 0.55)
                boosts[sentence_id] = max(boosts.get(sentence_id, 0.0), 0.45 * lexical + relation_score)
        return boosts

    def _query_entity_compatibility_score(
        self,
        query: str,
        sub_questions: Sequence[str],
        mention: str,
        node_attrs: Dict,
        sentence_ids: Sequence[str],
        graph: nx.DiGraph,
    ) -> float:
        query_text = " ".join([query] + [item for item in sub_questions if item]).lower()
        mention_text = str(mention or "").lower()
        metadata = node_attrs.get("metadata", {})
        title_anchor = str(metadata.get("title_anchor", "")).strip().lower()
        sentence_texts = [graph.nodes[sentence_id].get("text", "").lower() for sentence_id in sentence_ids if sentence_id in graph]
        chunk_texts = []
        chunk_titles = []
        for sentence_id in sentence_ids:
            if sentence_id not in graph:
                continue
            chunk_id = graph.nodes[sentence_id].get("metadata", {}).get("chunk_id")
            if chunk_id in graph:
                chunk_texts.append(str(graph.nodes[chunk_id].get("text", "")).lower())
                chunk_titles.append(str(graph.nodes[chunk_id].get("metadata", {}).get("title", "")).lower())
        neighborhood_text = " ".join(sentence_texts + chunk_texts + chunk_titles)

        expected_types = set()
        if any(token in query_text for token in ["who", "mother", "father", "author", "singer", "governor", "physicist", "member", "tsarevich", "person"]):
            expected_types.add("person")
        if any(token in query_text for token in ["where", "state", "city", "country", "county", "continent", "capital", "location", "born in", "born"]):
            expected_types.add("place")
        if any(token in query_text for token in ["company", "organization", "university", "agency", "band", "group"]):
            expected_types.add("organization")
        if any(token in query_text for token in ["movie", "film", "song", "album", "book", "novel", "work"]):
            expected_types.add("work")
        if any(token in mention_text for token in ["war", "bombing", "battle", "attack", "treaty"]):
            expected_types.add("event")

        observed_types = set()
        media_cues = ["film", "movie", "song", "album", "novel", "book", "band", "documentary", "episode", "tv", "television"]
        organization_cues = ["company", "organization", "university", "agency", "ministry", "band", "group"]
        place_cues = ["city", "county", "state", "country", "continent", "station", "province", "located", "capital", "border", "shares border", "adjacent", "township"]
        person_cues = ["was born", "born in", "birth", "died", "served", "professor", "politician", "singer", "author", "physicist", "governor", "son of", "daughter of", "mother", "father"]
        event_cues = ["war", "bombing", "battle", "attack", "treaty"]

        if any(token in title_anchor for token in media_cues) or any(any(token in sentence for token in media_cues) for sentence in sentence_texts[:3]):
            observed_types.add("work")
        if any(token in neighborhood_text for token in organization_cues):
            observed_types.add("organization")
        if any(token in neighborhood_text for token in place_cues) or any(token in title_anchor for token in place_cues):
            observed_types.add("place")
        if any(token in neighborhood_text for token in person_cues):
            observed_types.add("person")
        if any(token in neighborhood_text for token in event_cues) or any(token in title_anchor for token in event_cues):
            observed_types.add("event")

        type_score = 0.5
        if expected_types:
            if expected_types & observed_types:
                type_score = 0.65 + 0.35 * (len(expected_types & observed_types) / max(1, len(expected_types)))
            elif observed_types:
                type_score = 0.05
            else:
                type_score = 0.30

        media_query = any(token in query_text for token in media_cues)
        production_query = any(token in query_text for token in ["company", "manufacturer", "missile", "system", "weapon", "device", "product", "industrial", "contractor"])
        media_entity = "work" in observed_types
        production_entity = any(token in neighborhood_text for token in ["offered", "built", "manufactured", "designed", "missile", "system", "contractor", "company"])
        domain_score = 0.5
        if production_query and media_entity and not media_query:
            domain_score = 0.0
        elif media_query and production_entity and not media_entity:
            domain_score = 0.15
        elif production_query and production_entity:
            domain_score = 1.0
        elif media_query and media_entity:
            domain_score = 1.0
        elif media_entity and not media_query:
            domain_score = 0.35
        elif production_entity:
            domain_score = 0.75

        return 0.55 * type_score + 0.45 * domain_score

    def _dense_layer_retrieval(
        self,
        queries: Sequence[str],
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
        instruction: str,
        top_k_value: int,
        query_text_type: str,
    ) -> Dict[str, float]:
        valid_queries = [query for query in queries if query]
        if top_k_value <= 0 or not node_embeddings or not valid_queries:
            return {}
        node_ids = list(node_embeddings.keys())
        matrix = np.asarray([node_embeddings[node_id] for node_id in node_ids], dtype=np.float32)
        query_embeddings = self.embedder.encode(valid_queries, instruction=instruction, text_type=query_text_type)
        merged_scores: Dict[str, float] = {}
        for query_text, query_embedding in zip(valid_queries, query_embeddings):
            scores = cosine_similarity_matrix(query_embedding, matrix)
            for node_id, score in zip(node_ids, scores.tolist()):
                lexical = lexical_overlap_score(query_text, graph.nodes[node_id].get("text", "")) if node_id in graph else 0.0
                mixed = (1.0 - self.config.lexical_mix_weight) * float(score) + self.config.lexical_mix_weight * lexical
                merged_scores[node_id] = max(merged_scores.get(node_id, 0.0), mixed)
        ranked = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)[:top_k_value]
        return dict(ranked)

    def _subquestion_sentence_retrieval(
        self,
        graph: nx.DiGraph,
        state: Dict,
        sub_question: str,
        previous_hops: Sequence[Dict],
    ) -> Dict[str, float]:
        sentence_embeddings = state.get("embeddings", {}).get("sentence", {})
        if not sentence_embeddings:
            return {}
        retrieval_queries: List[str] = []
        cleaned_sub_question = str(sub_question or "").strip()
        if cleaned_sub_question:
            retrieval_queries.append(cleaned_sub_question)
        prior_answers = [
            str(hop.get("hop_answer", "")).strip()
            for hop in previous_hops[-2:]
            if str(hop.get("hop_answer", "")).strip()
            and str(hop.get("hop_answer", "")).strip().lower() != "unknown"
        ]
        for answer in prior_answers:
            retrieval_queries.append(f"{answer} {cleaned_sub_question}".strip())
        return self._dense_layer_retrieval(
            queries=retrieval_queries,
            node_embeddings=sentence_embeddings,
            graph=graph,
            instruction="Retrieve sentence evidence for the current multi-hop sub-question.",
            top_k_value=self.config.sentence_top_k,
            query_text_type="sub_question",
        )

    def _bridge_sentence_candidates(
        self,
        query: str,
        sub_questions: Sequence[str],
        bridge_entities: Sequence[str],
        graph: nx.DiGraph,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        later_hop_queries = list(sub_questions[1:]) if len(sub_questions) > 1 else list(sub_questions)
        hop_queries = later_hop_queries or [query]
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "sentence":
                continue
            score = self._bridge_relation_match_score(
                text=attrs.get("text", ""),
                metadata=attrs.get("metadata", {}),
                bridge_entities=list(bridge_entities),
                hop_queries=hop_queries,
            )
            if score > 0:
                scores[node_id] = score
        return normalize_scores(scores)

    def _extract_bridge_entities(
        self,
        graph: nx.DiGraph,
        candidate_sentence_ids: Sequence[str],
        sub_questions: Sequence[str],
        query: str,
        query_entities: List[str],
        ranked_facts: Sequence[Dict],
    ) -> List[str]:
        later_hops = [item for item in sub_questions[1:] if item] or ([query] if query else [])
        if not candidate_sentence_ids and not ranked_facts:
            return []
        connected_query_entities = set(query_entities)
        candidate_scores: Dict[str, float] = defaultdict(float)
        candidate_support: Dict[str, int] = defaultdict(int)
        for sentence_id in candidate_sentence_ids:
            if sentence_id not in graph:
                continue
            metadata = graph.nodes[sentence_id].get("metadata", {})
            sentence_text = graph.nodes[sentence_id].get("text", "")
            for triple in metadata.get("triples", []):
                head = str(triple.get("head", "")).strip()
                tail = str(triple.get("tail", "")).strip()
                relation = str(triple.get("relation", "")).lower()
                if not head or not tail:
                    continue
                head_matches_query = any(entity_match_score(head, query_entity) >= 1.0 for query_entity in connected_query_entities)
                tail_matches_query = any(entity_match_score(tail, query_entity) >= 1.0 for query_entity in connected_query_entities)
                if head_matches_query == tail_matches_query:
                    continue
                candidate = tail if head_matches_query else head
                if any(entity_match_score(candidate, query_entity) >= 1.0 for query_entity in query_entities):
                    continue
                score = self._score_bridge_candidate(
                    candidate=candidate,
                    relation=relation,
                    sentence_text=sentence_text,
                    later_hops=later_hops,
                    graph=graph,
                )
                candidate_scores[candidate] += score
                candidate_support[candidate] += 1

            if metadata.get("entities"):
                query_matched = any(text_contains_entity(sentence_text, query_entity) for query_entity in connected_query_entities)
                if query_matched:
                    for entity in metadata.get("entities", []):
                        candidate = str(entity).strip()
                        if not candidate:
                            continue
                        if any(entity_match_score(candidate, query_entity) >= 1.0 for query_entity in query_entities):
                            continue
                        score = self._score_bridge_candidate(
                            candidate=candidate,
                            relation="",
                            sentence_text=sentence_text,
                            later_hops=later_hops,
                            graph=graph,
                        )
                        candidate_scores[candidate] += 0.6 * score
                        candidate_support[candidate] += 1

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (item[1], candidate_support.get(item[0], 0)),
            reverse=True,
        )
        bridge_entities = []
        seen_keys = set()
        for candidate, _ in ranked:
            key = normalize_entity_key(candidate)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            bridge_entities.append(candidate)
            if len(bridge_entities) >= self.config.bridge_entity_top_k:
                break
        return bridge_entities

    def _bridge_entity_neighborhood_scores(
        self,
        graph: nx.DiGraph,
        state: Dict,
        bridge_entities: Sequence[str],
        hop_queries: Sequence[str],
    ) -> Dict[str, float]:
        candidate_sentence_ids = []
        title_anchor_chunk_scores = self._title_anchor_chunk_scores(
            graph=graph,
            mentions=list(bridge_entities),
            context_queries=list(hop_queries),
        )
        for bridge_entity in bridge_entities:
            entity_id = self._find_entity_node_by_text(graph, bridge_entity, context_text=" ".join(hop_queries))
            if entity_id is None:
                continue
            candidate_sentence_ids.extend(self._sentences_for_entity(graph, entity_id))
        for chunk_id in title_anchor_chunk_scores:
            candidate_sentence_ids.extend(self._sentences_for_chunk(graph, chunk_id))
        candidate_sentence_ids = list(dict.fromkeys(sentence_id for sentence_id in candidate_sentence_ids if sentence_id in graph))
        if not candidate_sentence_ids:
            return {}
        sentence_embeddings = state.get("embeddings", {}).get("sentence", {})
        subset_embeddings = {
            sentence_id: sentence_embeddings[sentence_id]
            for sentence_id in candidate_sentence_ids
            if sentence_id in sentence_embeddings
        }
        expanded_queries = []
        for hop_query in hop_queries:
            if hop_query:
                expanded_queries.append(hop_query)
                for bridge_entity in bridge_entities[: self.config.bridge_entity_top_k]:
                    expanded_queries.append(f"{bridge_entity} {hop_query}")
        dense_scores = self._dense_layer_retrieval(
            queries=expanded_queries,
            node_embeddings=subset_embeddings,
            graph=graph,
            instruction="Retrieve sentences that connect the entity to the requested property or relation.",
            top_k_value=max(len(subset_embeddings), self.config.sentence_step_top_k),
            query_text_type="sub_question",
        ) if subset_embeddings else {}
        boosted_scores: Dict[str, float] = dict(dense_scores)
        for sentence_id in candidate_sentence_ids:
            sentence_text = graph.nodes[sentence_id].get("text", "")
            entity_bonus = 0.0
            if any(text_contains_entity(sentence_text, bridge_entity) for bridge_entity in bridge_entities):
                entity_bonus = 0.2
            chunk_id = graph.nodes[sentence_id].get("metadata", {}).get("chunk_id")
            title_bonus = 0.18 * title_anchor_chunk_scores.get(chunk_id, 0.0) if chunk_id else 0.0
            lexical = max((lexical_overlap_score(hop_query, sentence_text) for hop_query in hop_queries), default=0.0)
            boosted_scores[sentence_id] = max(
                boosted_scores.get(sentence_id, 0.0),
                entity_bonus + title_bonus + 0.35 * lexical,
            )
        return normalize_scores(boosted_scores)

    def _bridge_relation_match_score(
        self,
        text: str,
        metadata: Dict,
        bridge_entities: List[str],
        hop_queries: List[str],
    ) -> float:
        if not any(text_contains_entity(text, entity) for entity in bridge_entities):
            return 0.0
        relation_match = 0.0
        for triple in metadata.get("triples", []):
            head = str(triple.get("head", "")).strip()
            tail = str(triple.get("tail", "")).strip()
            bridge_match = any(
                entity_match_score(entity, head) >= 1.0 or entity_match_score(entity, tail) >= 1.0
                for entity in bridge_entities
            )
            if bridge_match:
                relation_match = max(relation_match, 0.35)
        lexical = max(lexical_overlap_score(question, text) for question in hop_queries) if hop_queries else 0.0
        return relation_match + 0.45 * lexical

    def _score_bridge_candidate(
        self,
        candidate: str,
        relation: str,
        sentence_text: str,
        later_hops: Sequence[str],
        graph: nx.DiGraph,
    ) -> float:
        lexical = max((lexical_overlap_score(hop, sentence_text) for hop in later_hops), default=0.0)
        relation_support = max((lexical_overlap_score(hop, relation) for hop in later_hops), default=0.0) if relation else 0.0
        neighborhood_support = 0.0
        candidate_node_id = self._find_entity_node_by_text(graph, candidate, context_text=" ".join(later_hops))
        if candidate_node_id is not None:
            sentence_ids = self._sentences_for_entity(graph, candidate_node_id)
            neighborhood_texts = [graph.nodes[sentence_id].get("text", "") for sentence_id in sentence_ids if sentence_id in graph]
            neighborhood_support = max(
                (lexical_overlap_score(hop, text) for hop in later_hops for text in neighborhood_texts),
                default=0.0,
            )
        return 0.25 + 0.20 * lexical + 0.20 * relation_support + 0.35 * neighborhood_support

    def _extract_relation_cues(self, text: str) -> List[str]:
        lowered = text.lower()
        cues = [token for token in re.findall(r"[a-z]+", lowered) if len(token) > 3]
        return list(dict.fromkeys(cues[:8]))

    def _relation_matches_cues(self, relation: str, cues: List[str]) -> bool:
        if not relation or not cues:
            return False
        relation_tokens = {token for token in re.findall(r"[a-z]+", relation.lower()) if len(token) > 2}
        if not relation_tokens:
            return False
        for cue in cues:
            if cue in relation or relation in cue:
                return True
            cue_tokens = {token for token in re.findall(r"[a-z]+", cue.lower()) if len(token) > 2}
            if relation_tokens & cue_tokens:
                return True
        return False

    def _sentences_for_entity(self, graph: nx.DiGraph, entity_id: str) -> List[str]:
        sentence_ids = []
        for neighbor in list(graph.successors(entity_id)) + list(graph.predecessors(entity_id)):
            if graph.nodes[neighbor].get("node_type") == "sentence":
                sentence_ids.append(neighbor)
        return list(dict.fromkeys(sentence_ids))

    def _induce_chunk_scores_from_entity(self, graph: nx.DiGraph, entity_id: str, score: float) -> Dict[str, float]:
        chunk_scores: Dict[str, float] = {}
        sentence_ids = self._sentences_for_entity(graph, entity_id)
        if not sentence_ids:
            return chunk_scores
        spread = self.config.fact_entity_spread_weight * score / max(1, len(sentence_ids))
        for sentence_id in sentence_ids:
            chunk_id = graph.nodes[sentence_id].get("metadata", {}).get("chunk_id")
            if chunk_id:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), spread)
        return chunk_scores

    def _sentences_for_chunk(self, graph: nx.DiGraph, chunk_id: str) -> List[str]:
        sentence_ids = []
        for neighbor in list(graph.successors(chunk_id)) + list(graph.predecessors(chunk_id)):
            if graph.nodes[neighbor].get("node_type") == "sentence":
                sentence_ids.append(neighbor)
        return list(dict.fromkeys(sentence_id for sentence_id in sentence_ids if sentence_id in graph))

    def _expand_sub_question_with_hop_context(self, sub_question: str, previous_hops: Sequence[Dict]) -> str:
        cleaned = str(sub_question or "").strip()
        prior_answers = [
            str(hop.get("hop_answer", "")).strip()
            for hop in previous_hops
            if str(hop.get("hop_answer", "")).strip() and str(hop.get("hop_answer", "")).strip().lower() != "unknown"
        ]
        if not cleaned or not prior_answers:
            return cleaned
        context_tail = "; ".join(prior_answers[-2:])
        return f"{cleaned} Context: {context_tail}."

    def _sentence_candidate_entities(self, metadata: Dict) -> List[str]:
        candidates: List[str] = []
        for triple in metadata.get("triples", []):
            for key in ("head", "tail"):
                value = clean_entity_text(str(triple.get(key, "")).strip())
                if value and value.lower() != "unknown" and value not in candidates:
                    candidates.append(value)
        for entity in metadata.get("entities", []):
            value = clean_entity_text(str(entity).strip())
            if value and value.lower() != "unknown" and value not in candidates:
                candidates.append(value)
        return candidates

    def _candidate_future_support_score(
        self,
        graph: nx.DiGraph,
        candidate: str,
        future_sub_questions: Sequence[str],
    ) -> float:
        candidate_text = clean_entity_text(candidate)
        if not candidate_text or not future_sub_questions:
            return 0.0
        lexical = max((lexical_overlap_score(question, candidate_text) for question in future_sub_questions), default=0.0)
        relation_cues = self._extract_relation_cues(" ".join(future_sub_questions))
        entity_id = self._find_entity_node_by_text(graph, candidate_text, context_text=" ".join(future_sub_questions))
        if entity_id is None:
            return 0.35 * lexical
        sentence_ids = self._sentences_for_entity(graph, entity_id)
        neighborhood_texts = [graph.nodes[sentence_id].get("text", "") for sentence_id in sentence_ids if sentence_id in graph]
        chunk_texts = []
        chunk_titles = []
        for sentence_id in sentence_ids:
            if sentence_id not in graph:
                continue
            chunk_id = graph.nodes[sentence_id].get("metadata", {}).get("chunk_id")
            if chunk_id in graph:
                chunk_texts.append(str(graph.nodes[chunk_id].get("text", "")))
                chunk_titles.append(str(graph.nodes[chunk_id].get("metadata", {}).get("title", "")))
        neighborhood_overlap = max(
            (lexical_overlap_score(question, text) for question in future_sub_questions for text in neighborhood_texts + chunk_texts + chunk_titles),
            default=0.0,
        )
        relation_overlap = 0.0
        cue_overlap = 0.0
        for sentence_id in sentence_ids:
            if sentence_id not in graph:
                continue
            for triple in graph.nodes[sentence_id].get("metadata", {}).get("triples", []):
                relation_text = str(triple.get("relation", "")).strip()
                if not relation_text:
                    continue
                relation_overlap = max(
                    relation_overlap,
                    max((lexical_overlap_score(question, relation_text) for question in future_sub_questions), default=0.0),
                )
                if self._relation_matches_cues(relation_text.lower(), relation_cues):
                    cue_overlap = max(cue_overlap, 1.0)
        return 0.16 * lexical + 0.42 * neighborhood_overlap + 0.24 * relation_overlap + 0.18 * cue_overlap

    def _sentence_continuation_score(
        self,
        graph: nx.DiGraph,
        sentence_id: str,
        future_sub_questions: Sequence[str],
        previous_hops: Sequence[Dict],
    ) -> float:
        if sentence_id not in graph or not future_sub_questions:
            return 0.0
        sentence_text = graph.nodes[sentence_id].get("text", "")
        metadata = graph.nodes[sentence_id].get("metadata", {})
        future_overlap = max((lexical_overlap_score(question, sentence_text) for question in future_sub_questions), default=0.0)
        relation_cues = self._extract_relation_cues(" ".join(future_sub_questions))
        relation_overlap = 0.0
        cue_overlap = 0.0
        for triple in metadata.get("triples", []):
            relation_text = str(triple.get("relation", "")).strip()
            if not relation_text:
                continue
            relation_overlap = max(
                relation_overlap,
                max((lexical_overlap_score(question, relation_text) for question in future_sub_questions), default=0.0),
            )
            if self._relation_matches_cues(relation_text.lower(), relation_cues):
                cue_overlap = max(cue_overlap, 1.0)
        blocked_answers = {
            clean_entity_text(str(hop.get("hop_answer", "")).strip())
            for hop in previous_hops
            if str(hop.get("hop_answer", "")).strip() and str(hop.get("hop_answer", "")).strip().lower() != "unknown"
        }
        entity_support = 0.0
        for candidate in self._sentence_candidate_entities(metadata):
            if candidate in blocked_answers:
                continue
            entity_support = max(
                entity_support,
                self._candidate_future_support_score(
                    graph=graph,
                    candidate=candidate,
                    future_sub_questions=future_sub_questions,
                ),
            )
        return 0.24 * future_overlap + 0.16 * relation_overlap + 0.15 * cue_overlap + 0.45 * entity_support

    def _build_hop_evidence_passages(
        self,
        graph: nx.DiGraph,
        candidate_sentence_ids: Sequence[str],
        previous_hops: Sequence[Dict],
        sub_question: str,
    ) -> List[Dict]:
        passages: List[Dict] = []
        seen = set()
        expanded_sentence_ids: List[str] = list(candidate_sentence_ids)
        for hop in previous_hops[-2:]:
            hop_answer = str(hop.get("hop_answer", "")).strip()
            if not hop_answer or hop_answer.lower() == "unknown":
                continue
            entity_id = self._find_entity_node_by_text(graph, hop_answer, context_text=sub_question)
            if entity_id is None:
                continue
            for sentence_id in self._sentences_for_entity(graph, entity_id):
                if sentence_id not in expanded_sentence_ids:
                    expanded_sentence_ids.append(sentence_id)

        for sentence_id in expanded_sentence_ids:
            if sentence_id not in graph:
                continue
            sentence_attrs = graph.nodes[sentence_id]
            metadata = sentence_attrs.get("metadata", {})
            title = str(metadata.get("title", "")).strip()
            text = str(sentence_attrs.get("text", "")).strip()
            key = (title, text)
            if text and key not in seen:
                passages.append({"title": title, "text": text})
                seen.add(key)
            chunk_id = metadata.get("chunk_id")
            if chunk_id in graph:
                chunk_text = str(graph.nodes[chunk_id].get("text", "")).strip()
                chunk_title = str(graph.nodes[chunk_id].get("metadata", {}).get("title", title)).strip()
                chunk_key = (chunk_title, chunk_text)
                if chunk_text and chunk_key not in seen:
                    passages.append({"title": chunk_title, "text": chunk_text})
                    seen.add(chunk_key)
            if len(passages) >= max(self.config.hop_answer_passage_top_k + 2, self.config.hop_answer_passage_top_k):
                break
        return passages

    def _extract_hop_answer(
        self,
        sub_question: str,
        candidate_sentence_ids: Sequence[str],
        graph: nx.DiGraph,
        previous_hops: Sequence[Dict],
    ) -> Dict[str, str]:
        evidence_passages = self._build_hop_evidence_passages(
            graph=graph,
            candidate_sentence_ids=candidate_sentence_ids,
            previous_hops=previous_hops,
            sub_question=sub_question,
        )
        messages = build_hipporag_hop_messages(
            sub_question=sub_question,
            evidence_passages=evidence_passages,
            previous_hops=previous_hops,
            top_k=self.config.hop_answer_passage_top_k,
        )
        raw_response = self.llm_client.infer_messages_text(
            messages=messages,
            fallback="Answer: Unknown",
            max_tokens=min(self.config.max_new_tokens, 192),
        )
        thought, answer = parse_hipporag_qa_response(raw_response)
        normalized_answer = self._normalize_short_answer(
            question=sub_question,
            answer=answer,
            evidence_passages=evidence_passages,
        )
        return {
            "thought": thought,
            "answer": normalized_answer,
            "raw_response": raw_response,
            "evidence_passages": evidence_passages[: self.config.hop_answer_passage_top_k],
        }

    def _is_atomic_answer(self, answer: str) -> bool:
        normalized = str(answer or "").strip()
        if not normalized or normalized.lower() == "unknown":
            return True
        if "\n" in normalized or len(normalized) > 72:
            return False
        tokens = normalized.split()
        if len(tokens) > 8:
            return False
        lowered = normalized.lower()
        if re.search(r"\b(and|or)\b", lowered):
            return False
        if len(tokens) > 4 and re.search(r"\b(is|are|was|were|be|being|been|served|located|founded|born|died|became|includes|include|consists|consisting)\b", lowered):
            return False
        return True

    def _normalize_short_answer(self, question: str, answer: str, evidence_passages: Sequence[Dict]) -> str:
        normalized = str(answer or "").strip()
        if not normalized:
            return "Unknown"
        if normalized.lower() == "unknown":
            return "Unknown"
        if self._is_atomic_answer(normalized):
            return normalized
        messages = build_short_answer_normalization_messages(
            question=question,
            candidate_answer=normalized,
            evidence_passages=evidence_passages,
            top_k=min(len(evidence_passages), self.config.hop_answer_passage_top_k + 1),
        )
        raw_response = self.llm_client.infer_messages_text(
            messages=messages,
            fallback="Answer: Unknown",
            max_tokens=min(self.config.max_new_tokens, 96),
        )
        _, reduced_answer = parse_hipporag_qa_response(raw_response)
        if not self._is_atomic_answer(reduced_answer):
            return "Unknown"
        return reduced_answer

    def _answer_supported_by_evidence(self, answer: str, passages: Sequence[Dict]) -> bool:
        normalized = clean_entity_text(str(answer or "").strip())
        if not normalized or normalized.lower() == "unknown":
            return True
        answer_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized.lower()) if token}
        for passage in passages:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            combined = f"{title}\n{text}" if title else text
            if text_contains_entity(combined, normalized):
                return True
            passage_tokens = {token for token in re.findall(r"[a-z0-9]+", combined.lower()) if token}
            if answer_tokens and answer_tokens.issubset(passage_tokens):
                return True
        return False

    def _support_span_in_passages(self, support: str, passages: Sequence[Dict]) -> bool:
        snippet = str(support or "").strip()
        if not snippet:
            return False
        for passage in passages:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            combined = f"{title}\n{text}" if title else text
            if snippet in combined:
                return True
        return False

    def _answer_supported_by_span(self, answer: str, support: str) -> bool:
        normalized_answer = clean_entity_text(str(answer or "").strip())
        normalized_support = str(support or "").strip()
        if not normalized_answer or normalized_answer.lower() == "unknown":
            return True
        if not normalized_support:
            return False
        if text_contains_entity(normalized_support, normalized_answer):
            return True
        answer_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized_answer.lower()) if token}
        support_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized_support.lower()) if token}
        return bool(answer_tokens) and answer_tokens.issubset(support_tokens)

    def _collect_fact_anchor_texts(self, reasoning_chain: Sequence[Dict], candidate_answer: str) -> List[str]:
        anchors: List[str] = []
        latest_answer = ""
        for hop in reasoning_chain:
            hop_answer = clean_entity_text(str(hop.get("hop_answer", "")).strip())
            if hop_answer and hop_answer.lower() != "unknown":
                anchors.append(hop_answer)
                latest_answer = hop_answer
            source_title = clean_entity_text(str(hop.get("source_title", "")).strip())
            if source_title:
                anchors.append(source_title)
        normalized_candidate = clean_entity_text(candidate_answer)
        if normalized_candidate and normalized_candidate.lower() != "unknown":
            anchors.append(normalized_candidate)
        if latest_answer:
            anchors.append(latest_answer)
        deduped: List[str] = []
        seen = set()
        for anchor in reversed(anchors):
            key = normalize_entity_key(anchor)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(anchor)
        deduped.reverse()
        return deduped

    def _fact_matches_anchor(self, fact_text: str, anchor: str) -> bool:
        cleaned_anchor = clean_entity_text(anchor)
        if not cleaned_anchor:
            return False
        if text_contains_entity(fact_text, cleaned_anchor):
            return True
        return entity_match_score(fact_text, cleaned_anchor) >= 0.55

    def _score_fact_for_answer_support(
        self,
        query: str,
        fact: Dict,
        reasoning_chain: Sequence[Dict],
        candidate_answer: str,
    ) -> float:
        fact_text = str(fact.get("text", "")).strip()
        if not fact_text:
            return 0.0
        score = 0.10 * lexical_overlap_score(query, fact_text)
        score += 0.35 * float(fact.get("score", 0.0) or 0.0)
        normalized_candidate = clean_entity_text(candidate_answer)
        if normalized_candidate and normalized_candidate.lower() != "unknown":
            if self._fact_matches_anchor(fact_text, normalized_candidate):
                score += 0.55
            else:
                score -= 0.10

        anchor_hits = 0.0
        anchor_texts = self._collect_fact_anchor_texts(reasoning_chain, candidate_answer)
        if anchor_texts:
            total_anchors = len(anchor_texts)
            for index, anchor in enumerate(anchor_texts, start=1):
                weight = 0.18 + 0.28 * (index / total_anchors)
                if self._fact_matches_anchor(fact_text, anchor):
                    anchor_hits += weight
                else:
                    alias_overlap = entity_match_score(fact_text, anchor)
                    if alias_overlap >= 0.35:
                        anchor_hits += 0.45 * weight * alias_overlap
            if anchor_hits == 0.0:
                score -= 0.22
        score += min(anchor_hits, 1.05)

        sentence_id = str(fact.get("sentence_id", "") or "")
        if sentence_id:
            source_hits = 0.0
            for hop in reasoning_chain:
                hop_sentence_id = str(hop.get("sentence_id", "") or "")
                if hop_sentence_id and hop_sentence_id == sentence_id:
                    source_hits += 0.18
            score += min(source_hits, 0.36)

        if anchor_texts and not normalized_candidate:
            latest_anchor = anchor_texts[-1]
            if latest_anchor and text_contains_entity(fact_text, latest_anchor):
                score += 0.20

        return score

    def _select_fact_hints(
        self,
        query: str,
        ranked_facts: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        candidate_answer: str,
        limit: int = 5,
    ) -> List[str]:
        if not ranked_facts:
            return []
        scored_facts = [
            (self._score_fact_for_answer_support(query, fact, reasoning_chain, candidate_answer), fact)
            for fact in ranked_facts
        ]
        scored_facts.sort(key=lambda item: item[0], reverse=True)

        hints: List[str] = []
        seen = set()
        anchors = self._collect_fact_anchor_texts(reasoning_chain, candidate_answer)

        def maybe_add(fact_text: str) -> None:
            if fact_text and fact_text not in seen and len(hints) < limit:
                seen.add(fact_text)
                hints.append(fact_text)

        anchored_pool = []
        fallback_pool = []
        for score, fact in scored_facts:
            fact_text = str(fact.get("text", "")).strip()
            if not fact_text or score <= 0.0:
                continue
            if anchors and any(self._fact_matches_anchor(fact_text, anchor) for anchor in anchors):
                anchored_pool.append((score, fact_text))
            else:
                fallback_pool.append((score, fact_text))

        for _, fact_text in anchored_pool:
            maybe_add(fact_text)

        if not hints:
            for _, fact_text in fallback_pool:
                maybe_add(fact_text)
        return hints

    def _build_structured_answer_verification_prompt(
        self,
        query: str,
        passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        draft_answer: str,
        fact_hints: Sequence[str],
    ) -> Tuple[str, str]:
        reasoning_lines: List[str] = []
        for index, hop in enumerate(reasoning_chain, start=1):
            sub_question = str(hop.get("sub_question", "")).strip()
            hop_answer = str(hop.get("hop_answer", "")).strip()
            if not sub_question:
                continue
            line = f"{index}. {sub_question}"
            if hop_answer and hop_answer.lower() != "unknown":
                line += f" => {hop_answer}"
            reasoning_lines.append(line)

        passage_blocks: List[str] = []
        for passage in passages:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text:
                continue
            if title:
                passage_blocks.append(f"Wikipedia Title: {title}\n{text}")
            else:
                passage_blocks.append(text)

        system_prompt = (
            "Answer the question using only the provided passages. "
            "Follow the intermediate reasoning hints to stay anchored to the target entities they establish. "
            "Use the fact hints only as compact graph evidence summaries grounded in the same retrieved context. "
            "Prefer evidence that explicitly links those target entities to the final answer, and ignore competitors, alternatives, or nearby entities unless the passages directly connect them to the target. "
            "Return JSON with keys thought, answer, and support. "
            "The support value must be a short exact span copied from one provided passage that gives the decisive named evidence for the answer. "
            "Do not invent support. If no single supported answer exists, set answer to 'Unknown' and support to ''."
        )
        fact_block = "\n".join(f"- {hint}" for hint in fact_hints if str(hint).strip())
        user_prompt = "\n\n".join(
            part
            for part in [
                ("Intermediate reasoning hints:\n" + "\n".join(reasoning_lines)) if reasoning_lines else "",
                ("Fact hints:\n" + fact_block) if fact_block else "",
                "\n\n".join(passage_blocks),
                f"Question: {query}\nDraft Answer: {draft_answer}",
            ]
            if part
        )
        return system_prompt, user_prompt

    def _verify_answer_with_support(
        self,
        query: str,
        passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        draft_answer: str,
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, str]:
        if not passages:
            return {"thought": "", "answer": "Unknown", "support": "", "raw_response": ""}
        fact_hints = self._select_fact_hints(
            query=query,
            ranked_facts=ranked_facts,
            reasoning_chain=reasoning_chain,
            candidate_answer=draft_answer,
            limit=min(5, self.config.fact_rerank_top_k),
        )
        support_passages = self._select_support_sentence_passages(
            query=query,
            passages=passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
            candidate_answer=draft_answer,
            max_sentences=6,
        )
        verification_passages = support_passages or list(passages)
        system_prompt, user_prompt = self._build_structured_answer_verification_prompt(
            query=query,
            passages=verification_passages,
            reasoning_chain=reasoning_chain,
            draft_answer=draft_answer,
            fact_hints=fact_hints,
        )
        payload, raw_response = self.llm_client.infer_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback={"thought": "", "answer": "Unknown", "support": ""},
            max_tokens=min(self.config.max_new_tokens, 192),
        )
        answer = self._normalize_short_answer(
            question=query,
            answer=str(payload.get("answer", "Unknown")),
            evidence_passages=passages,
        )
        support = str(payload.get("support", "") or "").strip()
        thought = str(payload.get("thought", "") or "").strip()
        if not self._support_span_in_passages(support, verification_passages) and not self._support_span_in_passages(support, passages):
            answer = "Unknown"
            support = ""
        if not self._answer_supported_by_span(answer, support):
            answer = "Unknown"
            support = ""
        if not self._answer_supported_by_evidence(answer, verification_passages) and not self._answer_supported_by_evidence(answer, passages):
            answer = "Unknown"
            support = ""
        return {
            "thought": thought,
            "answer": answer,
            "support": support,
            "raw_response": raw_response,
        }

    def _score_passage_for_answer_focus(
        self,
        query: str,
        passage: Dict,
        reasoning_chain: Sequence[Dict],
        candidate_answer: str,
        fact_hints: Sequence[str] | None = None,
    ) -> float:
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        combined = f"{title}\n{text}" if title else text
        lexical = lexical_overlap_score(query, combined)
        candidate_hit = 0.0
        normalized_candidate = clean_entity_text(candidate_answer)
        if normalized_candidate and normalized_candidate.lower() != "unknown" and text_contains_entity(combined, normalized_candidate):
            candidate_hit = 0.30
        hop_hit = 0.0
        for hop in reasoning_chain:
            hop_answer = clean_entity_text(str(hop.get("hop_answer", "")).strip())
            if hop_answer and hop_answer.lower() != "unknown" and text_contains_entity(combined, hop_answer):
                hop_hit += 0.10
        fact_hit = 0.0
        for fact_hint in fact_hints or []:
            if lexical_overlap_score(fact_hint, combined) >= 0.45:
                fact_hit += 0.08
        length_penalty = min(len(combined.split()) / 220.0, 0.18)
        return lexical + candidate_hit + min(hop_hit, 0.30) + min(fact_hit, 0.24) - length_penalty

    def _sentence_noise_penalty(self, sentence: str, title: str = "") -> float:
        token_count = len(sentence.split())
        char_count = len(sentence)
        digit_count = len(re.findall(r"\d", sentence))
        uppercase_token_count = len(re.findall(r"\b[A-Z]{2,}\b", sentence))
        punctuation_count = sum(sentence.count(mark) for mark in [":", ";", "|", "/"])
        comma_count = sentence.count(",")
        bracket_count = sum(sentence.count(mark) for mark in ["(", ")", "[", "]"])
        alpha_words = re.findall(r"\b[A-Za-z][A-Za-z'.-]*\b", sentence)
        leading_titlecase_count = 0
        for word in alpha_words[:12]:
            if word and word[0].isupper():
                leading_titlecase_count += 1
            else:
                break

        penalty = 0.0
        if token_count > 38:
            penalty += min((token_count - 38) / 90.0, 0.14)
        if char_count > 260:
            penalty += min((char_count - 260) / 900.0, 0.10)
        if digit_count >= 8:
            penalty += min((digit_count - 8) / 40.0, 0.10)
        if punctuation_count >= 3:
            penalty += min((punctuation_count - 3) / 10.0, 0.08)
        if comma_count >= 8:
            penalty += min((comma_count - 8) / 20.0, 0.08)
        if bracket_count >= 4:
            penalty += min((bracket_count - 4) / 12.0, 0.05)
        if uppercase_token_count >= 4:
            penalty += min((uppercase_token_count - 4) / 12.0, 0.04)
        if alpha_words and len(alpha_words) <= 12 and leading_titlecase_count >= min(len(alpha_words), 5):
            penalty += 0.10
        normalized_title = clean_entity_text(title).lower()
        normalized_sentence = clean_entity_text(sentence).lower()
        if normalized_title and normalized_sentence.startswith(normalized_title):
            remainder = normalized_sentence[len(normalized_title):].strip(" -:,.\n")
            remainder_words = re.findall(r"\b[a-z]+\b", remainder)
            if len(remainder_words) <= 6:
                penalty += 0.10
        return min(penalty, 0.34)

    def _score_evidence_sentence(
        self,
        query: str,
        sentence: str,
        title: str,
        reasoning_chain: Sequence[Dict],
        fact_hints: Sequence[str],
        candidate_answer: str,
    ) -> float:
        combined = f"{title}\n{sentence}" if title else sentence
        score = 0.55 * lexical_overlap_score(query, combined)
        normalized_candidate = clean_entity_text(candidate_answer)
        if normalized_candidate and normalized_candidate.lower() != "unknown":
            if text_contains_entity(combined, normalized_candidate):
                score += 0.28
            elif entity_match_score(combined, normalized_candidate) >= 0.45:
                score += 0.18

        anchor_texts = self._collect_fact_anchor_texts(reasoning_chain, candidate_answer)
        anchor_hits = 0.0
        anchor_match_count = 0
        for index, anchor in enumerate(anchor_texts, start=1):
            weight = 0.10 + 0.10 * (index / max(1, len(anchor_texts)))
            if text_contains_entity(combined, anchor):
                anchor_hits += weight
                anchor_match_count += 1
            elif entity_match_score(combined, anchor) >= 0.45:
                anchor_hits += 0.5 * weight
                anchor_match_count += 1
        score += min(anchor_hits, 0.45)
        if anchor_match_count >= 2:
            score += min(0.06 * (anchor_match_count - 1), 0.12)

        fact_hits = 0.0
        for fact_hint in fact_hints:
            overlap = lexical_overlap_score(fact_hint, combined)
            if overlap >= 0.45:
                fact_hits += 0.12 * overlap
        score += min(fact_hits, 0.36)
        score -= min(len(sentence.split()) / 120.0, 0.12)
        score -= self._sentence_noise_penalty(sentence, title=title)
        return score

    def _count_passage_alignment_hits(
        self,
        passage: Dict,
        reasoning_chain: Sequence[Dict],
        candidate_answer: str,
        fact_hints: Sequence[str],
    ) -> Tuple[int, int]:
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        combined = f"{title}\n{text}" if title else text
        anchor_hits = 0
        for anchor in self._collect_fact_anchor_texts(reasoning_chain, candidate_answer):
            if text_contains_entity(combined, anchor) or entity_match_score(combined, anchor) >= 0.45:
                anchor_hits += 1
        fact_hits = 0
        for fact_hint in fact_hints:
            if lexical_overlap_score(fact_hint, combined) >= 0.45:
                fact_hits += 1
        return anchor_hits, fact_hits

    def _select_support_sentence_passages(
        self,
        query: str,
        passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
        candidate_answer: str,
        max_sentences: int = 6,
    ) -> List[Dict]:
        if not passages:
            return []
        fact_hints = self._select_fact_hints(
            query=query,
            ranked_facts=ranked_facts,
            reasoning_chain=reasoning_chain,
            candidate_answer=candidate_answer,
            limit=min(5, self.config.fact_rerank_top_k),
        )
        candidates_by_title: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        title_order: List[Tuple[float, str]] = []
        seen = set()
        for passage in passages:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text:
                continue
            best_title_score = None
            for sentence in self.sentence_segmenter.split(text) or [text]:
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                key = (title, cleaned)
                if key in seen:
                    continue
                seen.add(key)
                score = self._score_evidence_sentence(
                    query=query,
                    sentence=cleaned,
                    title=title,
                    reasoning_chain=reasoning_chain,
                    fact_hints=fact_hints,
                    candidate_answer=candidate_answer,
                )
                candidates_by_title[title].append((score, cleaned))
                if best_title_score is None or score > best_title_score:
                    best_title_score = score
            if best_title_score is not None:
                title_order.append((best_title_score, title))

        for title, items in candidates_by_title.items():
            items.sort(key=lambda item: item[0], reverse=True)
        title_order.sort(key=lambda item: item[0], reverse=True)
        if title_order:
            top_title_score = title_order[0][0]
            min_title_score = max(0.18, 0.55 * top_title_score)
            title_order = [item for item in title_order if item[0] >= min_title_score]

        support_passages: List[Dict] = []
        used_counts: Dict[str, int] = defaultdict(int)
        ordered_titles = [title for _, title in title_order]
        while len(support_passages) < max_sentences and ordered_titles:
            progressed = False
            next_titles: List[str] = []
            for title in ordered_titles:
                items = candidates_by_title.get(title, [])
                if not items:
                    continue
                if used_counts[title] >= 2:
                    continue
                _, sentence = items.pop(0)
                support_passages.append({"title": title, "text": sentence})
                used_counts[title] += 1
                progressed = True
                if items and used_counts[title] < 2 and len(support_passages) < max_sentences:
                    next_titles.append(title)
                if len(support_passages) >= max_sentences:
                    break
            if not progressed:
                break
            ordered_titles = next_titles
        return support_passages

    def _build_aligned_passage_views(
        self,
        query: str,
        passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
        candidate_answer: str,
    ) -> List[Dict]:
        if not passages:
            return []
        fact_hints = self._select_fact_hints(
            query=query,
            ranked_facts=ranked_facts,
            reasoning_chain=reasoning_chain,
            candidate_answer=candidate_answer,
            limit=min(5, self.config.fact_rerank_top_k),
        )
        aligned: List[Dict] = []
        seen = set()
        for passage in passages:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text:
                continue
            sentences = self.sentence_segmenter.split(text) or [text]
            scored_sentences = sorted(
                sentences,
                key=lambda sent: self._score_evidence_sentence(
                    query=query,
                    sentence=sent,
                    title=title,
                    reasoning_chain=reasoning_chain,
                    fact_hints=fact_hints,
                    candidate_answer=candidate_answer,
                ),
                reverse=True,
            )
            kept: List[str] = []
            for sent in scored_sentences:
                cleaned = sent.strip()
                if not cleaned or cleaned in kept:
                    continue
                kept.append(cleaned)
                if len(kept) >= 2:
                    break
            compressed_text = " ".join(kept) if kept else text
            aligned_passage = dict(passage)
            aligned_passage["text"] = compressed_text
            key = (title, compressed_text)
            if key in seen:
                continue
            seen.add(key)
            aligned.append(aligned_passage)
        aligned.sort(
            key=lambda passage: (
                self._count_passage_alignment_hits(
                    passage,
                    reasoning_chain,
                    candidate_answer,
                    fact_hints,
                ),
                self._score_passage_for_answer_focus(
                    query=query,
                    passage=passage,
                    reasoning_chain=reasoning_chain,
                    candidate_answer=candidate_answer,
                    fact_hints=fact_hints,
                ),
            ),
            reverse=True,
        )
        return aligned

    def _expand_reasoning_support_passages(
        self,
        query: str,
        base_passages: Sequence[Dict],
        ranked_passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
        candidate_answer: str,
        budget: int,
    ) -> List[Dict]:
        merged: List[Dict] = []
        seen = set()

        def add_passage(passage: Dict) -> None:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text:
                return
            key = (title, text)
            if key in seen:
                return
            seen.add(key)
            merged.append(passage)

        for passage in base_passages:
            add_passage(passage)
        for hop in reasoning_chain:
            for passage in hop.get("evidence_passages", []) or []:
                add_passage(passage)

        fact_hints = self._select_fact_hints(
            query=query,
            ranked_facts=ranked_facts,
            reasoning_chain=reasoning_chain,
            candidate_answer=candidate_answer,
            limit=min(5, self.config.fact_rerank_top_k),
        )
        candidate_entries = []
        for rank, passage in enumerate(ranked_passages):
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text or (title, text) in seen:
                continue
            alignment_hits = self._count_passage_alignment_hits(
                passage,
                reasoning_chain,
                candidate_answer,
                fact_hints,
            )
            focus_score = self._score_passage_for_answer_focus(
                query=query,
                passage=passage,
                reasoning_chain=reasoning_chain,
                candidate_answer=candidate_answer,
                fact_hints=fact_hints,
            )
            retrieval_score = float(passage.get("score", 0.0) or 0.0)
            candidate_entries.append((alignment_hits, focus_score, retrieval_score - 0.01 * rank, passage))
        candidate_entries.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        if candidate_entries:
            best_focus = candidate_entries[0][1]
            min_focus = max(0.12, 0.60 * best_focus)
            for alignment_hits, focus_score, _, passage in candidate_entries:
                if len(merged) >= budget:
                    break
                if alignment_hits == (0, 0) and focus_score < min_focus:
                    continue
                add_passage(passage)
        return merged[:budget]

    def _select_answer_focus_passages(
        self,
        query: str,
        selected_passages: Sequence[Dict],
        ranked_passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        candidate_answer: str,
        ranked_facts: Sequence[Dict],
    ) -> List[Dict]:
        if not selected_passages:
            return []
        support_pool = self._expand_reasoning_support_passages(
            query=query,
            base_passages=selected_passages,
            ranked_passages=ranked_passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
            candidate_answer=candidate_answer,
            budget=max(len(selected_passages), self.config.qa_passage_top_k + self.config.hop_answer_passage_top_k + 3),
        )
        aligned_passages = self._build_aligned_passage_views(
            query=query,
            passages=support_pool,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
            candidate_answer=candidate_answer,
        )
        ranked = aligned_passages or list(support_pool)
        focus_budget = max(3, min(len(ranked), self.config.qa_passage_top_k))
        deduped_ranked: List[Dict] = []
        seen = set()
        for passage in ranked:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text:
                continue
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            deduped_ranked.append(passage)

        focused: List[Dict] = []
        used_titles = set()
        deferred: List[Dict] = []
        for passage in deduped_ranked:
            title = str(passage.get("title", "")).strip()
            if title and title in used_titles:
                deferred.append(passage)
                continue
            focused.append(passage)
            if title:
                used_titles.add(title)
            if len(focused) >= focus_budget:
                return focused

        for passage in deferred:
            focused.append(passage)
            if len(focused) >= focus_budget:
                break
        return focused

    def _get_terminal_hop_answer(
        self,
        query: str,
        selected_passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, str]:
        if not reasoning_chain:
            return {"thought": "", "answer": "Unknown", "support": "", "raw_response": ""}

        last_hop = reasoning_chain[-1]
        hop_evidence = list(last_hop.get("evidence_passages", []) or [])
        evidence_pool: List[Dict] = []
        seen = set()
        for passage in [*hop_evidence, *selected_passages]:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not text:
                continue
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            evidence_pool.append(passage)

        terminal_answer = self._normalize_short_answer(
            question=query,
            answer=str(last_hop.get("hop_answer", "Unknown")),
            evidence_passages=evidence_pool or selected_passages,
        )
        if terminal_answer.lower() == "unknown":
            return {"thought": "", "answer": "Unknown", "support": "", "raw_response": ""}

        if not self._answer_supported_by_evidence(terminal_answer, evidence_pool or selected_passages):
            return {"thought": "", "answer": "Unknown", "support": "", "raw_response": ""}

        verified = self._verify_answer_with_support(
            query=query,
            passages=evidence_pool or selected_passages,
            reasoning_chain=reasoning_chain,
            draft_answer=terminal_answer,
            ranked_facts=ranked_facts,
        )
        verified_answer = str(verified.get("answer", "") or "").strip()
        if verified_answer and verified_answer.lower() != "unknown":
            return {
                "thought": str(verified.get("thought", "") or ""),
                "answer": verified_answer,
                "support": str(verified.get("support", "") or ""),
                "raw_response": str(verified.get("raw_response", "") or ""),
            }

        return {
            "thought": str(last_hop.get("hop_thought", "") or ""),
            "answer": terminal_answer,
            "support": "",
            "raw_response": str(last_hop.get("hop_raw_response", "") or ""),
        }

    def _refine_final_answer(
        self,
        query: str,
        draft_answer: str,
        selected_passages: Sequence[Dict],
        ranked_passages: Sequence[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, object]:
        focused_passages = self._select_answer_focus_passages(
            query,
            selected_passages,
            ranked_passages,
            reasoning_chain,
            draft_answer,
            ranked_facts,
        )
        if not focused_passages:
            return {
                "thought": "",
                "answer": self._normalize_short_answer(query, draft_answer, selected_passages),
                "raw_response": "",
                "messages": [],
                "focused_passages": [],
            }
        focus_fact_hints = self._select_fact_hints(
            query=query,
            ranked_facts=ranked_facts,
            reasoning_chain=reasoning_chain,
            candidate_answer=draft_answer,
            limit=min(5, self.config.fact_rerank_top_k),
        )
        messages = build_answer_focus_messages(
            question=query,
            candidate_answer=draft_answer,
            ranked_passages=focused_passages,
            top_k=len(focused_passages),
            reasoning_chain=reasoning_chain,
            fact_hints=focus_fact_hints,
        )
        raw_response = self.llm_client.infer_messages_text(
            messages=messages,
            fallback="Answer: Unknown",
            max_tokens=min(self.config.max_new_tokens, 192),
        )
        thought, answer = parse_hipporag_qa_response(raw_response)
        normalized_answer = self._normalize_short_answer(
            question=query,
            answer=answer,
            evidence_passages=focused_passages,
        )
        if not self._answer_supported_by_evidence(normalized_answer, focused_passages):
            normalized_answer = "Unknown"
        verified = self._verify_answer_with_support(
            query=query,
            passages=focused_passages,
            reasoning_chain=reasoning_chain,
            draft_answer=normalized_answer or draft_answer,
            ranked_facts=ranked_facts,
        )
        final_answer = normalized_answer
        final_thought = thought
        final_raw_response = raw_response
        if str(verified.get("answer", "")).strip().lower() != "unknown":
            final_answer = str(verified["answer"])
            final_thought = str(verified.get("thought", "") or thought)
            final_raw_response = str(verified.get("raw_response", "") or raw_response)
        return {
            "thought": final_thought,
            "answer": final_answer,
            "raw_response": final_raw_response,
            "messages": messages,
            "focused_passages": focused_passages,
            "support": verified.get("support", ""),
        }

    def _select_qa_passages_for_reader(
        self,
        query: str,
        ranked_passages: List[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
    ) -> List[Dict]:
        base_passages = list(ranked_passages[: self.config.qa_passage_top_k])
        return self._expand_reasoning_support_passages(
            query=query,
            base_passages=base_passages,
            ranked_passages=ranked_passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
            candidate_answer="",
            budget=self.config.qa_passage_top_k + self.config.hop_answer_passage_top_k + 1,
        )

    def _build_passage_context(self, ranked_passages: List[Dict], top_k: int) -> str:
        parts = []
        for passage in ranked_passages[:top_k]:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if text:
                parts.append(f"{title}\n{text}" if title else text)
        return "\n\n".join(parts)

    def _generate_final_qa_answer(
        self,
        query: str,
        ranked_passages: List[Dict],
        reasoning_chain: Sequence[Dict],
        ranked_facts: Sequence[Dict],
    ) -> Dict[str, str]:
        selected_passages = self._select_qa_passages_for_reader(
            query=query,
            ranked_passages=ranked_passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
        )
        aligned_passages = self._build_aligned_passage_views(
            query=query,
            passages=selected_passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
            candidate_answer="",
        )
        qa_reader_passages = aligned_passages or selected_passages
        qa_fact_hints = self._select_fact_hints(
            query=query,
            ranked_facts=ranked_facts,
            reasoning_chain=reasoning_chain,
            candidate_answer="",
            limit=min(5, self.config.fact_rerank_top_k),
        )
        messages = build_hipporag_qa_messages(
            question=query,
            ranked_passages=qa_reader_passages,
            top_k=len(qa_reader_passages),
            reasoning_chain=reasoning_chain,
            fact_hints=qa_fact_hints,
        )
        raw_response = self.llm_client.infer_messages_text(
            messages=messages,
            fallback="Answer: Unknown",
            max_tokens=min(self.config.max_new_tokens, 256),
        )
        thought, answer = parse_hipporag_qa_response(raw_response)
        normalized_answer = self._normalize_short_answer(
            question=query,
            answer=answer,
            evidence_passages=selected_passages,
        )
        if not self._answer_supported_by_evidence(normalized_answer, selected_passages):
            normalized_answer = "Unknown"

        verified_result = self._verify_answer_with_support(
            query=query,
            passages=selected_passages,
            reasoning_chain=reasoning_chain,
            draft_answer=normalized_answer,
            ranked_facts=ranked_facts,
        )
        verified_answer = str(verified_result.get("answer", "") or "").strip()
        final_answer = normalized_answer
        final_thought = thought
        final_raw_response = raw_response
        final_messages = messages
        final_support = ""

        if verified_answer and verified_answer.lower() != "unknown":
            final_answer = verified_answer
            final_thought = str(verified_result.get("thought", "") or thought)
            final_raw_response = str(verified_result.get("raw_response", "") or raw_response)
            final_support = str(verified_result.get("support", "") or "")

        refined_result = self._refine_final_answer(
            query=query,
            draft_answer=final_answer,
            selected_passages=selected_passages,
            ranked_passages=ranked_passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
        )
        refined_answer = str(refined_result.get("answer", "") or "").strip()
        if refined_answer and refined_answer.lower() != "unknown":
            final_answer = refined_answer
            final_thought = str(refined_result.get("thought", "") or final_thought)
            final_raw_response = str(refined_result.get("raw_response", "") or final_raw_response)
            final_messages = refined_result.get("messages", messages)
            final_support = str(refined_result.get("support", "") or final_support)

        terminal_hop_result = self._get_terminal_hop_answer(
            query=query,
            selected_passages=selected_passages,
            reasoning_chain=reasoning_chain,
            ranked_facts=ranked_facts,
        )
        terminal_hop_answer = str(terminal_hop_result.get("answer", "") or "").strip()
        if terminal_hop_answer and terminal_hop_answer.lower() != "unknown":
            final_answer = terminal_hop_answer
            final_thought = str(terminal_hop_result.get("thought", "") or final_thought)
            final_raw_response = str(terminal_hop_result.get("raw_response", "") or final_raw_response)
            terminal_support = str(terminal_hop_result.get("support", "") or "")
            if terminal_support:
                final_support = terminal_support

        return {
            "thought": final_thought,
            "answer": final_answer,
            "raw_response": final_raw_response,
            "messages": final_messages,
            "selected_passages": selected_passages,
            "focused_passages": refined_result.get("focused_passages", []),
            "support": final_support,
        }

    def _ensure_fact_index(self, state: Dict, graph: nx.DiGraph) -> None:
        facts = state.get("facts")
        fact_embeddings = state.get("embeddings", {}).get("fact")
        if facts and fact_embeddings:
            return
        fact_records = []
        for sentence_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "sentence":
                continue
            metadata = attrs.get("metadata", {})
            for triple in metadata.get("triples", []):
                head = str(triple.get("head", "")).strip()
                relation = str(triple.get("relation", "")).strip()
                tail = str(triple.get("tail", "")).strip()
                if not head or not relation or not tail:
                    continue
                head_id = self._find_entity_node_by_text(graph, head)
                tail_id = self._find_entity_node_by_text(graph, tail)
                if head_id is None or tail_id is None:
                    continue
                fact_records.append({
                    "fact_id": f"fact:{sentence_id}:{len(fact_records)}",
                    "text": f"{head} {relation} {tail}",
                    "head_id": head_id,
                    "tail_id": tail_id,
                    "sentence_id": sentence_id,
                    "chunk_id": metadata.get("chunk_id"),
                    "document_index": metadata.get("document_index"),
                })
        state["facts"] = fact_records
        state.setdefault("embeddings", {})
        if fact_records:
            fact_vectors = self.embedder.encode(
                [record["text"] for record in fact_records],
                instruction="Encode the triplet fact for retrieval.",
                text_type="sentence",
            )
            state["embeddings"]["fact"] = {record["fact_id"]: fact_vectors[idx] for idx, record in enumerate(fact_records)}
        else:
            state["embeddings"]["fact"] = {}

    def _find_entity_node_by_text(self, graph: nx.DiGraph, text: str, context_text: str = "") -> Optional[str]:
        best_node_id: Optional[str] = None
        best_score = 0.0
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "entity":
                continue
            metadata = attrs.get("metadata", {})
            aliases = [attrs.get("text", "")]
            aliases.extend(metadata.get("aliases", []))
            aliases.extend(metadata.get("surface_forms", []))
            alias_score = max((entity_match_score(text, alias) for alias in aliases), default=0.0)
            if alias_score <= 0.0:
                continue
            context_score = 0.0
            if context_text:
                context_score = max(
                    [lexical_overlap_score(context_text, attrs.get("text", ""))]
                    + [lexical_overlap_score(context_text, graph.nodes[sentence_id].get("text", "")) for sentence_id in self._sentences_for_entity(graph, node_id)]
                )
            final_score = alias_score + 0.35 * context_score
            if final_score > best_score:
                best_score = final_score
                best_node_id = node_id
        return best_node_id

    def _fallback_alpha(self) -> Dict[str, float]:
        return {"entity": 0.33, "sentence": 0.34, "chunk": 0.33}
