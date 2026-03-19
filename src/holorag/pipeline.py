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
from .query_decomposer import QueryDecomposer
from .recognition_filter import RecognitionFilter
from .seed_selector import SeedSelector
from .sentence_segmenter import SentenceSegmenter
from .triple_extractor import TripleExtractor
from .utils import (
    cosine_similarity_matrix,
    dump_pickle,
    ensure_dir,
    entity_match_score,
    lexical_overlap_score,
    load_pickle,
    normalize_scores,
    normalize_entity_key,
    text_contains_entity,
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
        sub_questions = self.query_decomposer.decompose(query, resolved_entities=confident_entity_resolutions)
        if not sub_questions:
            sub_questions = initial_sub_questions
        query_entities = [item["resolved_text"] for item in confident_entity_resolutions] or raw_query_entities
        query_entity_node_ids = [item["node_id"] for item in confident_entity_resolutions]

        ranked_facts, dense_chunk_scores, graph_chunk_scores = self._hipporag_backbone(
            query=query,
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
        evidence["qa_context"] = self._build_passage_context(ranked_passages, self.config.qa_passage_top_k)
        evidence["reasoning_chain"] = reasoning_chain

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
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
    ) -> Tuple[List[Dict], Dict[str, float], Dict[str, float]]:
        self._get_query_embeddings([query], state)
        fact_scores = self._get_fact_scores(query, state)
        ranked_facts = self._rerank_facts(query, fact_scores, state)
        dense_chunk_scores = self._dense_passage_retrieval(query, state)
        if ranked_facts:
            graph_chunk_scores = self._graph_search_with_fact_entities(query, ranked_facts, query_entities, query_entity_node_ids, graph, state)
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
        bridge_entities: List[str] = []
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

        ranked_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
        for hop_index, sub_question in enumerate(sub_questions or [query]):
            hop_best = None
            for sentence_id, score in ranked_sentences:
                sentence_text = graph.nodes[sentence_id].get("text", "") if sentence_id in graph else ""
                hop_overlap = lexical_overlap_score(sub_question, sentence_text)
                if hop_index == 0:
                    adjusted = 0.7 * float(score) + 0.3 * hop_overlap
                else:
                    bridge_hit = any(text_contains_entity(sentence_text, entity) for entity in bridge_entities)
                    adjusted = 0.55 * float(score) + 0.25 * hop_overlap + (0.20 if bridge_hit else 0.0)
                if hop_best is None or adjusted > hop_best[1]:
                    hop_best = (sentence_id, adjusted)
            if hop_best is not None:
                reasoning_chain.append({
                    "sub_question": sub_question,
                    "sentence_id": hop_best[0],
                    "sentence": graph.nodes[hop_best[0]].get("text", ""),
                    "score": hop_best[1],
                })
        return normalize_scores(sentence_scores), reasoning_chain, bridge_entities

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
        for fact in ranked_facts[: self.config.linking_top_k]:
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
        for fact in ranked_facts[: self.config.linking_top_k]:
            for entity_id in (fact.get("head_id"), fact.get("tail_id")):
                if entity_id in graph:
                    fact_entity_occurs[entity_id] += 1

        for fact in ranked_facts[: self.config.linking_top_k]:
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

        # Keep the sentence layer as an auxiliary HoloRAG signal rather than the main driver.
        for sentence_id, score in filtered_scores.get("sentence", {}).items():
            if sentence_id in graph:
                reset_scores[sentence_id] += 0.18 * float(score)
        for entity_id, score in filtered_scores.get("entity", {}).items():
            if entity_id in graph:
                reset_scores[entity_id] += 0.10 * float(score)
        for chunk_id, score in filtered_scores.get("chunk", {}).items():
            if chunk_id in graph:
                reset_scores[chunk_id] += 0.10 * float(score)

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

    def _dense_passage_retrieval(self, query: str, state: Dict) -> Dict[str, float]:
        self._get_query_embeddings([query], state)
        passage_node_ids = state["retrieval_cache"]["passage_node_ids"]
        if not passage_node_ids:
            return {}
        query_embedding = state["query_to_embedding"]["passage"][query]
        scores = cosine_similarity_matrix(query_embedding, state["retrieval_cache"]["passage_embeddings"])
        normalized = normalize_scores({node_id: float(score) for node_id, score in zip(passage_node_ids, scores.tolist())})
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True)[: self.config.retrieval_top_k])

    def _rerank_facts(self, query: str, fact_scores: Dict[str, float], state: Dict) -> List[Dict]:
        if not fact_scores:
            return []
        fact_lookup = {record["fact_id"]: record for record in state.get("facts", [])}
        candidate_ids = [fact_id for fact_id, _ in sorted(fact_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.linking_top_k] if fact_id in fact_lookup]
        if not candidate_ids:
            return []

        candidate_records = [fact_lookup[fact_id] for fact_id in candidate_ids]
        fallback_ranked = sorted(
            candidate_records,
            key=lambda record: 0.7 * float(fact_scores[record["fact_id"]]) + 0.3 * lexical_overlap_score(query, record["text"]),
            reverse=True,
        )
        selected_records = self._llm_filter_facts(query, candidate_records, fallback_ranked)

        rescored = []
        for rank, record in enumerate(selected_records):
            base_score = 0.7 * float(fact_scores.get(record["fact_id"], 0.0)) + 0.3 * lexical_overlap_score(query, record["text"])
            score = base_score + 0.02 * max(0, len(selected_records) - rank)
            rescored.append({**record, "score": float(score)})
        rescored.sort(key=lambda item: item["score"], reverse=True)
        return rescored

    def _llm_filter_facts(
        self,
        query: str,
        candidate_records: Sequence[Dict],
        fallback_ranked: Sequence[Dict],
    ) -> List[Dict]:
        if not candidate_records:
            return []
        candidate_lines = []
        for index, record in enumerate(candidate_records):
            candidate_lines.append(f"{index}: ({record['text']})")
        fallback_indices = list(range(min(len(fallback_ranked), self.config.linking_top_k)))
        fallback = {"fact_indices": fallback_indices}
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Select the candidate facts that are most useful for answering the question. "
                "Prefer facts that directly support multi-hop reasoning and avoid distractors with only surface-word overlap. "
                "Return JSON with key fact_indices, a list of integer indices from the candidate list."
            ),
            user_prompt=(
                f"Question:\n{query}\n\n"
                "Candidate facts:\n"
                + "\n".join(candidate_lines)
                + f"\n\nReturn at most {self.config.linking_top_k} indices."
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
        selected_records = [candidate_records[index] for index in cleaned_indices[: self.config.linking_top_k]]
        if not selected_records:
            selected_records = list(fallback_ranked[: self.config.linking_top_k])
        return selected_records

    def _graph_search_with_fact_entities(
        self,
        query: str,
        ranked_facts: Sequence[Dict],
        query_entities: Sequence[str],
        query_entity_node_ids: Sequence[str],
        graph: nx.DiGraph,
        state: Dict,
    ) -> Dict[str, float]:
        dense_passages = self._dense_passage_retrieval(query, state)
        personalization: Dict[str, float] = defaultdict(float)
        entity_chunk_count: Dict[str, int] = defaultdict(int)
        linked_entities = set()
        for fact in ranked_facts[: self.config.linking_top_k]:
            for entity_id in (fact["head_id"], fact["tail_id"]):
                linked_entities.add(entity_id)
                entity_chunk_count[entity_id] = max(entity_chunk_count.get(entity_id, 0), len(self._sentences_for_entity(graph, entity_id)))
        for fact in ranked_facts[: self.config.linking_top_k]:
            score = float(fact["score"])
            for entity_id in (fact["head_id"], fact["tail_id"]):
                personalization[entity_id] += score / max(1, entity_chunk_count.get(entity_id, 1))
        for entity_id in query_entity_node_ids:
            if entity_id in graph:
                personalization[entity_id] += 0.8
        for entity_text in query_entities:
            if any(entity_match_score(entity_text, graph.nodes[node_id].get("text", "")) >= 1.0 for node_id in query_entity_node_ids if node_id in graph):
                continue
            entity_id = self._find_entity_node_by_text(graph, entity_text, context_text=query)
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
                    node_attrs=attrs,
                    sentence_ids=sentence_ids,
                    graph=graph,
                )
                final_score = (
                    0.40 * alias_score
                    + 0.22 * relation_score
                    + 0.13 * context_score
                    + 0.05 * title_score
                    + 0.20 * compatibility_score
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
                best["score"] >= 0.62
                and (best["score_margin"] >= 0.08 or best["compatibility_score"] >= 0.75)
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
        node_attrs: Dict,
        sentence_ids: Sequence[str],
        graph: nx.DiGraph,
    ) -> float:
        query_text = " ".join([query] + [item for item in sub_questions if item]).lower()
        metadata = node_attrs.get("metadata", {})
        title_anchor = str(metadata.get("title_anchor", "")).strip().lower()
        sentence_texts = [graph.nodes[sentence_id].get("text", "").lower() for sentence_id in sentence_ids if sentence_id in graph]
        neighborhood_text = " ".join(sentence_texts)

        media_cues = ["film", "movie", "song", "album", "novel", "book", "band", "documentary", "episode", "tv", "television"]
        production_cues = ["company", "manufacturer", "missile", "system", "weapon", "device", "product", "industrial", "contractor"]
        media_query = any(token in query_text for token in media_cues)
        production_query = any(token in query_text for token in production_cues)
        media_entity = any(token in title_anchor for token in media_cues) or any(
            any(token in sentence for token in ["directed by", "starring", "released", "album", "song", "film"])
            for sentence in sentence_texts[:3]
        )
        production_entity = any(token in neighborhood_text for token in ["offered", "built", "manufactured", "designed", "missile", "system", "contractor", "company"])

        if production_query and media_entity and not media_query:
            return 0.0
        if media_query and production_entity and not media_entity:
            return 0.15
        if production_query and production_entity:
            return 1.0
        if media_query and media_entity:
            return 1.0
        if media_entity and not media_query:
            return 0.35
        if production_entity:
            return 0.75
        return 0.5

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
        for bridge_entity in bridge_entities:
            entity_id = self._find_entity_node_by_text(graph, bridge_entity, context_text=" ".join(hop_queries))
            if entity_id is None:
                continue
            candidate_sentence_ids.extend(self._sentences_for_entity(graph, entity_id))
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
            lexical = max((lexical_overlap_score(hop_query, sentence_text) for hop_query in hop_queries), default=0.0)
            boosted_scores[sentence_id] = max(boosted_scores.get(sentence_id, 0.0), entity_bonus + 0.35 * lexical)
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

    def _build_passage_context(self, ranked_passages: List[Dict], top_k: int) -> str:
        parts = []
        for passage in ranked_passages[:top_k]:
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()
            if text:
                parts.append(f"{title}\n{text}" if title else text)
        return "\n\n".join(parts)

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
