import logging
import os
import time
from typing import Dict, List, Optional

import networkx as nx

from .config import NaiveHoloRAGConfig
from .embedding_model import NVEmbedV2Encoder
from .extractors import QueryDecomposer, TripleExtractor
from .graph_builder import NaiveGraphBuilder
from .intent import IntentRouter
from .llm_client import LocalLLMClient
from .pagerank import GranularityPageRank
from .reader import NaiveQAReader
from .retriever import NaiveRetriever
from .sentence_segmenter import SentenceSegmenter
from .utils import dump_json, dump_pickle, ensure_dir, load_pickle

logger = logging.getLogger(__name__)


class NaiveHoloRAG:
    def __init__(self, config: Optional[NaiveHoloRAGConfig] = None) -> None:
        self.config = config or NaiveHoloRAGConfig()
        self.artifact_dir = ensure_dir(self.config.save_dir)
        self.index_path = os.path.join(self.artifact_dir, "holorag_naive_index.pkl")
        self.llm_client = LocalLLMClient(self.config)
        self.embedder = NVEmbedV2Encoder(self.config)
        self.sentence_segmenter = SentenceSegmenter()
        self.triple_extractor = TripleExtractor(self.llm_client, index_extraction_mode=self.config.index_extraction_mode)
        self.query_decomposer = QueryDecomposer(self.llm_client)
        self.intent_router = IntentRouter(self.config, self.llm_client)
        self.graph_builder = NaiveGraphBuilder(
            config=self.config,
            sentence_segmenter=self.sentence_segmenter,
            triple_extractor=self.triple_extractor,
            embedder=self.embedder,
        )
        self.retriever = NaiveRetriever(self.config, self.embedder, self.llm_client)
        self.page_rank = GranularityPageRank(self.config)
        self.reader = NaiveQAReader(self.config, self.llm_client)
        self.state: Optional[Dict] = None

    def index(self, documents: List[Dict[str, str]]) -> Dict:
        self.llm_client.reset_stats()
        logger.info("Building naive HoloRAG index for %d documents", len(documents))
        self.state = self.graph_builder.build(documents)
        dump_pickle(self.index_path, self.state)
        result = {
            "index_path": self.index_path,
            "stats": self.describe_index(),
            "llm_stats": self.llm_client.get_stats(),
        }
        dump_json(os.path.join(self.artifact_dir, "index_summary.json"), result)
        return result

    def load(self) -> Dict:
        if self.state is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Index not found at {self.index_path}. Run index first.")
            self.state = load_pickle(self.index_path)
        return self.state

    def describe_index(self) -> Dict:
        state = self.state or {}
        graph: nx.DiGraph = state.get("graph", nx.DiGraph())
        counts = {"entity": 0, "sentence": 0, "chunk": 0, "fact": len(state.get("facts", []))}
        for _, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type")
            if node_type in counts:
                counts[node_type] += 1
        return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "layer_counts": counts}

    def query(self, query: str, query_hints: Optional[Dict] = None) -> Dict:
        start = time.perf_counter()
        normalized_query = " ".join(str(query or "").split())
        state = self.load()
        graph: nx.DiGraph = state["graph"]
        self.llm_client.reset_stats()
        after_load = time.perf_counter()

        route = self.intent_router.route(normalized_query, forced_profile=(query_hints or {}).get("task_profile", self.config.task_profile))
        profile = route["profile"]
        alpha = route["alpha"]
        after_route = time.perf_counter()
        query_parse = self.triple_extractor.extract_query(normalized_query)
        after_query_parse = time.perf_counter()
        if self._should_decompose(profile):
            sub_questions = self.query_decomposer.decompose(normalized_query)
        else:
            sub_questions = [normalized_query]
        after_decomposition = time.perf_counter()

        retrieval = self.retriever.retrieve(
            query=normalized_query,
            query_entities=query_parse["entities"],
            query_facts=query_parse["triples"],
            sub_questions=sub_questions,
            graph=graph,
            state=state,
            alpha=alpha,
        )
        after_retrieval = time.perf_counter()
        pagerank_scores = self.page_rank.run(graph, alpha=alpha, seed_scores=retrieval["seed_scores"])
        after_pagerank = time.perf_counter()
        ranked_nodes = self._rank_nodes(graph, pagerank_scores)
        ranked_passages = self.retriever.rank_passages(
            graph=graph,
            pagerank_scores=pagerank_scores,
            channel_scores=retrieval["channel_scores"],
            ranked_facts=retrieval["ranked_facts"],
        )
        ranked_evidence = self.retriever.rank_evidence(
            graph=graph,
            pagerank_scores=pagerank_scores,
            channel_scores=retrieval["channel_scores"],
            ranked_facts=retrieval["ranked_facts"],
            ranked_passages=ranked_passages,
            profile=profile,
            query=normalized_query,
            sub_questions=sub_questions,
            token_budget=self.config.qa_evidence_token_budget,
        )
        after_ranking = time.perf_counter()
        qa_result = self.reader.answer(
            normalized_query,
            ranked_passages,
            retrieval["ranked_facts"],
            sub_questions,
            evidence=ranked_evidence,
        )
        after_qa = time.perf_counter()
        result = {
            "query": normalized_query,
            "task_profile": profile,
            "alpha": alpha,
            "intent_confidence": route["confidence"],
            "query_entities": query_parse["entities"],
            "query_facts": query_parse["triples"],
            "sub_questions": sub_questions,
            "channel_scores": self._trim_channel_scores(retrieval["channel_scores"]),
            "retrieval_meta": {
                "fallback_used": bool(retrieval.get("fallback_used", False)),
                "fact_rerank": retrieval.get("rerank_meta", {}),
            },
            "seeds": self._top_seed_view(graph, retrieval["seed_scores"]),
            "ranked_facts": retrieval["ranked_facts"][: self.config.fact_top_k],
            "ranked_evidence": ranked_evidence,
            "ranked_nodes": ranked_nodes[:30],
            "ranked_passages": ranked_passages[: self.config.passage_output_top_k],
            "predicted_answer": qa_result["answer"],
            "qa_thought": qa_result["thought"],
            "qa_raw_response": qa_result["raw_response"],
            "qa_messages": qa_result["messages"],
            "qa_answer_mode": qa_result.get("answer_mode", ""),
            "llm_stats": self.llm_client.get_stats(),
            "query_timing": {
                "query_total_latency": float(after_qa - start),
                "load_latency": float(after_load - start),
                "intent_latency": float(after_route - after_load),
                "query_parse_latency": float(after_query_parse - after_route),
                "decomposition_latency": float(after_decomposition - after_query_parse),
                "retrieval_latency": float(after_retrieval - after_decomposition),
                "pagerank_latency": float(after_pagerank - after_retrieval),
                "evidence_ranking_latency": float(after_ranking - after_pagerank),
                "retrieval_pipeline_latency": float(after_ranking - after_load),
                "qa_latency": float(after_qa - after_ranking),
            },
        }
        dump_json(os.path.join(self.artifact_dir, "last_query_result.json"), result)
        return result

    def _should_decompose(self, profile: str) -> bool:
        return bool(self.config.enable_query_decomposition and profile == "multi_hop")

    def _rank_nodes(self, graph: nx.DiGraph, scores: Dict[str, float]) -> List[Dict]:
        ranked = []
        for node_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            ranked.append({
                "node_id": node_id,
                "score": float(score),
                "node_type": attrs.get("node_type"),
                "text": attrs.get("text", ""),
                "metadata": attrs.get("metadata", {}),
            })
        return ranked

    def _top_seed_view(self, graph: nx.DiGraph, seed_scores: Dict[str, float]) -> List[Dict]:
        items = []
        for node_id, score in sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)[:30]:
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            items.append({
                "node_id": node_id,
                "score": float(score),
                "node_type": attrs.get("node_type"),
                "text": attrs.get("text", ""),
            })
        return items

    def _trim_channel_scores(self, channel_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[Dict]]:
        graph: nx.DiGraph = self.load()["graph"]
        result: Dict[str, List[Dict]] = {}
        for channel, scores in channel_scores.items():
            rows = []
            for node_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:20]:
                if channel == "fact":
                    fact = self.state.get("_fact_by_id", {}).get(node_id) if self.state else None
                    rows.append({"id": node_id, "score": float(score), "text": (fact or {}).get("text", "")})
                else:
                    rows.append({
                        "id": node_id,
                        "score": float(score),
                        "text": graph.nodes[node_id].get("text", "") if node_id in graph else "",
                    })
            result[channel] = rows
        return result
