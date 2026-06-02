#!/usr/bin/env python3
"""Evaluate the wo_graph_propagation ablation.

This entrypoint reuses scripts/eval.py for sampling, indexing, logging,
prediction serialization, metrics, and QA. The ablation-specific behavior is
limited to query-time retrieval: graph construction and all granularity layers
are kept, but PageRank/random-walk propagation is skipped and final evidence is
formed from direct fact/sentence/chunk retrieval scores only; entities remain
graph anchors but are not packed as a separate final evidence granularity.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval as eval_base  # noqa: E402

try:
    import networkx as nx  # noqa: E402
    import holorag  # noqa: E402
    from holorag.pipeline import HoloRAG as BaseHoloRAG  # noqa: E402
    from holorag.retriever import Retriever  # noqa: E402
    from holorag.utils import dump_json  # noqa: E402

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    # Keep --help usable without the full runtime environment.
    nx = None
    holorag = None
    BaseHoloRAG = object
    Retriever = object
    _IMPORT_ERROR = exc


class DirectOnlyRetriever(Retriever):
    """Retriever variant that avoids graph-neighbor propagation in ranking."""

    def rank_passages(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        channel_scores: Dict[str, Dict[str, float]],
        ranked_facts: Sequence[Dict],
        alpha: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        passages = []
        direct_limit = max(1, int(getattr(self.config, "qa_passage_top_k", 4)))
        for chunk_id, score in sorted(channel_scores.get("chunk", {}).items(), key=lambda item: item[1], reverse=True):
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
            if len(passages) >= direct_limit:
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
        ranked_sentences = [
            self._sentence_record(graph, sentence_id, float(score))
            for sentence_id, score in sorted(channel_scores.get("sentence", {}).items(), key=lambda item: item[1], reverse=True)
            if sentence_id in graph
        ]
        ranked_sentences = [item for item in ranked_sentences if item]

        if use_alpha_evidence:
            facts = list(ranked_facts[: self.config.fact_top_k])
            sentences = ranked_sentences[: self.config.sentence_top_k]
            chunks = list(ranked_passages[: self.config.passage_output_top_k])
            evidence_groups = [{"label": "Direct LLM-alpha sentence evidence", "items": sentences}]
        elif profile == "single_hop":
            facts = list(ranked_facts[: min(self.config.fact_top_k, 10)])
            sentences = ranked_sentences[:6]
            chunks = list(ranked_passages[:1]) if len(sentences) < 2 else []
            evidence_groups = [{"label": "Direct sentence evidence", "items": sentences}]
        elif profile == "long_context":
            facts = list(ranked_facts[: min(self.config.fact_top_k, 5)])
            sentences = ranked_sentences[:5]
            chunks = list(ranked_passages[: self.config.qa_passage_top_k])
            evidence_groups = [{"label": "Direct sentence evidence", "items": sentences}]
        else:
            facts = list(ranked_facts[: min(self.config.fact_top_k, 8)])
            evidence_groups, sentences = self._multi_hop_sentence_groups(
                graph=graph,
                ranked_sentences=ranked_sentences,
                source_sentences=[],
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
            use_expanded_passage_context = int(getattr(self.config, "evidence_passage_context_k", 1)) > 1
            if getattr(self.config, "enable_fair_sentence_context", False) or use_expanded_passage_context:
                chunks = self._diverse_passage_context(ranked_passages)
            else:
                chunks = list(ranked_passages[: min(1, self.config.qa_passage_top_k)])

        result = {
            "profile": profile,
            "allocation_mode": "strict_direct_retrieval_without_graph_propagation",
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


class WoGraphPropagationHoloRAG(BaseHoloRAG):
    """HoloRAG wrapper that skips PageRank and uses only direct channel scores."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.retriever = DirectOnlyRetriever(self.config, self.embedder, self.llm_client)

    def query(self, query: str, query_hints: Optional[Dict] = None) -> Dict:
        start = time.perf_counter()
        normalized_query = " ".join(str(query or "").split())
        state = self.load()
        graph: nx.DiGraph = state["graph"]
        self.llm_client.reset_stats()
        after_load = time.perf_counter()

        route = self.intent_router.route(
            normalized_query,
            forced_profile=(query_hints or {}).get("task_profile", self.config.task_profile),
        )
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
        direct_scores = self._direct_node_scores(graph, retrieval["channel_scores"])
        after_direct_scoring = time.perf_counter()
        ranked_nodes = self._rank_nodes(graph, direct_scores)
        ranked_passages = self.retriever.rank_passages(
            graph=graph,
            pagerank_scores=direct_scores,
            channel_scores=retrieval["channel_scores"],
            ranked_facts=retrieval["ranked_facts"],
            alpha=alpha,
        )
        ranked_evidence = self.retriever.rank_evidence(
            graph=graph,
            pagerank_scores=direct_scores,
            channel_scores=retrieval["channel_scores"],
            ranked_facts=retrieval["ranked_facts"],
            ranked_passages=ranked_passages,
            profile=profile,
            query=normalized_query,
            sub_questions=sub_questions,
            token_budget=0,
            alpha=alpha,
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
                "graph_propagation": "disabled",
                "scoring": "direct_multi_granularity_retrieval",
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
                "pagerank_latency": 0.0,
                "direct_scoring_latency": float(after_direct_scoring - after_retrieval),
                "evidence_ranking_latency": float(after_ranking - after_direct_scoring),
                "retrieval_pipeline_latency": float(after_ranking - after_load),
                "qa_latency": float(after_qa - after_ranking),
            },
        }
        dump_json(os.path.join(self.artifact_dir, "last_query_result.json"), result)
        return result

    def _direct_node_scores(self, graph: nx.DiGraph, channel_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for channel in ("entity", "sentence", "chunk"):
            for node_id, score in channel_scores.get(channel, {}).items():
                if node_id in graph:
                    scores[node_id] = max(scores.get(node_id, 0.0), float(score))
        return scores


_ORIGINAL_PARSE_ARGS = eval_base.parse_args
_ORIGINAL_BUILD_CONFIG = eval_base.build_config


def parse_args_with_ablation_name():
    args = _ORIGINAL_PARSE_ARGS()
    if not args.ablation_name:
        args.ablation_name = "wo_graph_propagation"
    return args


def build_wo_graph_propagation_config(args, save_dir: str):
    config = _ORIGINAL_BUILD_CONFIG(args, save_dir)
    config.enable_granularity_awareness = True
    config.enable_sentence_layer = True
    config.enable_granularity_pagerank_bias = True
    config.graph_propagation_enabled = False
    config.ppr_enabled = False
    config.use_ppr = False
    return config


def apply_wo_graph_propagation_defaults(args, ablation_name: str, argv: Sequence[str]) -> None:
    if not eval_base._arg_provided(argv, "--disable_granularity_awareness"):
        args.disable_granularity_awareness = False
    if not eval_base._arg_provided(argv, "--disable_sentence_layer"):
        args.disable_sentence_layer = False
    if not eval_base._arg_provided(argv, "--disable_granularity_pagerank_bias"):
        args.disable_granularity_pagerank_bias = False


def main() -> None:
    if _IMPORT_ERROR is not None:
        if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
            eval_base.parse_args()
            return
        raise _IMPORT_ERROR
    eval_base.parse_args = parse_args_with_ablation_name
    eval_base.build_config = build_wo_graph_propagation_config
    eval_base.apply_ablation_defaults = apply_wo_graph_propagation_defaults
    holorag.HoloRAG = WoGraphPropagationHoloRAG
    eval_base.main()


if __name__ == "__main__":
    main()
