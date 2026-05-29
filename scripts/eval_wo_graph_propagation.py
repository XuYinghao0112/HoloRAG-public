#!/usr/bin/env python3
"""Evaluate the wo_graph_propagation ablation.

This entrypoint reuses scripts/eval.py for sampling, indexing, logging,
prediction serialization, metrics, and QA. The ablation-specific behavior is
limited to query-time retrieval: graph construction and all granularity layers
are kept, but PageRank/random-walk propagation is skipped and final evidence is
formed from direct entity/fact/sentence/chunk retrieval scores only.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
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
        entities = self._direct_entity_records(graph, channel_scores)
        ranked_sentences = [
            self._sentence_record(graph, sentence_id, float(score))
            for sentence_id, score in sorted(channel_scores.get("sentence", {}).items(), key=lambda item: item[1], reverse=True)
            if sentence_id in graph
        ]
        ranked_sentences = [item for item in ranked_sentences if item]

        if use_alpha_evidence:
            alpha_norm = self._normalize_alpha_for_direct(alpha or {})
            facts = list(ranked_facts[: max(3, min(self.config.fact_top_k, int(round(3 + alpha_norm.get("fact", 0.0) * self.config.fact_top_k))))])
            sentences = ranked_sentences[: max(3, min(self.config.sentence_top_k, int(round(3 + alpha_norm.get("sentence", 0.0) * 14))))]
            chunks = list(ranked_passages[: max(1, min(self.config.qa_passage_top_k, int(round(1 + alpha_norm.get("chunk", 0.0) * self.config.qa_passage_top_k))))])
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
            "allocation_mode": "direct_retrieval_without_graph_propagation",
            "entities": entities,
            "facts": facts,
            "sentences": sentences,
            "chunks": chunks,
            "evidence_groups": evidence_groups,
            "fallback_passages": list(ranked_passages[: self.config.qa_passage_top_k]),
        }
        packed = self._pack_direct_profile_evidence(
            profile=profile,
            query=query,
            entities=entities,
            facts=facts,
            sentences=sentences,
            chunks=chunks,
            fallback_passages=result["fallback_passages"],
            sub_questions=sub_questions,
            token_budget=token_budget,
            alpha=alpha if use_alpha_evidence else None,
        )
        result.update(packed)
        return result

    def _direct_entity_records(self, graph: nx.DiGraph, channel_scores: Dict[str, Dict[str, float]]) -> List[Dict]:
        records = []
        for node_id, score in sorted(channel_scores.get("entity", {}).items(), key=lambda item: item[1], reverse=True):
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            records.append({
                "node_id": node_id,
                "score": float(score),
                "title": "",
                "text": attrs.get("text", ""),
                "metadata": attrs.get("metadata", {}),
            })
            if len(records) >= self.config.entity_top_k:
                break
        return records

    def _pack_direct_profile_evidence(
        self,
        profile: str,
        query: str,
        entities: Sequence[Dict],
        facts: Sequence[Dict],
        sentences: Sequence[Dict],
        chunks: Sequence[Dict],
        fallback_passages: Sequence[Dict],
        sub_questions: Sequence[str],
        token_budget: int,
        alpha: Optional[Dict[str, float]] = None,
    ) -> Dict:
        budget = max(128, int(token_budget or self.config.qa_evidence_token_budget))
        passage_limit = self._direct_alpha_passage_limit(alpha or {}, budget) if self._use_llm_alpha_evidence(alpha) else self._direct_passage_limit(profile, budget)
        candidates: List[Dict] = []

        for row in entities:
            candidates.append(self._packed_candidate(
                kind="entity",
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                score=float(row.get("score", 0.0)),
                label="Entity",
                node_id=str(row.get("node_id", "")),
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
        for row in sentences:
            candidates.append(self._packed_candidate(
                kind="sentence",
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                score=float(row.get("score", 0.0)),
                label="Evidence",
                node_id=str(row.get("node_id", "")),
                query=query,
                sub_questions=sub_questions,
            ))
        for passage in list(chunks) or list(fallback_passages):
            candidates.append(self._packed_candidate(
                kind="chunk",
                text=self._passage_excerpt(str(passage.get("text", "")), query, passage_limit),
                title=str(passage.get("title", "")),
                score=float(passage.get("score", 0.0)),
                label="Passage",
                node_id=str(passage.get("chunk_id", passage.get("node_id", ""))),
                query=query,
                sub_questions=sub_questions,
            ))

        weights = self._direct_alpha_weights(alpha or {}) if self._use_llm_alpha_evidence(alpha) else self._direct_profile_weights(profile)
        for item in candidates:
            item["pack_score"] = weights.get(item.get("kind"), 1.0) * float(item.get("score", 0.0)) + 0.35 * float(item.get("coverage", 0.0))

        selected = []
        selected_ids = set()
        title_counts: Dict[str, int] = defaultdict(int)
        title_limit = max(1, int(getattr(self.config, "evidence_title_limit", 3)))
        used = 0
        for item in sorted(candidates, key=lambda row: row.get("pack_score", 0.0), reverse=True):
            if used >= budget:
                break
            node_id = item.get("node_id")
            if node_id and node_id in selected_ids:
                continue
            title_key = str(item.get("title", "")).strip().lower()
            if title_key and title_counts[title_key] >= title_limit:
                continue
            line = str(item.get("line", "")).strip()
            cost = self._token_count(line)
            if cost > budget - used:
                if budget - used < 24:
                    continue
                line = self._truncate_words(line, budget - used)
                cost = self._token_count(line)
            if cost <= 0:
                continue
            selected.append({**item, "line": line, "tokens": cost})
            used += cost
            if node_id:
                selected_ids.add(node_id)
            if title_key:
                title_counts[title_key] += 1

        packed_text = "\n".join(item["line"] for item in selected).strip()
        return {
            "packed_text": packed_text,
            "packed_records": selected,
            "packed_token_budget": budget,
            "packed_token_count": self._token_count(packed_text),
        }

    def _direct_profile_weights(self, profile: str) -> Dict[str, float]:
        if not getattr(self.config, "enable_granularity_awareness", True):
            return {"entity": 1.0, "fact": 1.0, "sentence": 1.0, "chunk": 1.0}
        return {
            "single_hop": {"entity": 1.35, "fact": 1.45, "sentence": 1.20, "chunk": 0.65},
            "multi_hop": {"entity": 1.05, "fact": 1.20, "sentence": 1.45, "chunk": 0.80},
            "long_context": {"entity": 0.70, "fact": 0.75, "sentence": 1.00, "chunk": 1.45},
        }.get(profile, {"entity": 1.0, "fact": 1.0, "sentence": 1.2, "chunk": 0.9})

    def _direct_passage_limit(self, profile: str, budget: int) -> int:
        if profile == "multi_hop" and int(getattr(self.config, "evidence_passage_context_k", 1)) > 1:
            return max(80, int(getattr(self.config, "evidence_passage_excerpt_tokens", 150)))
        if profile == "multi_hop":
            return max(80, min(120, budget // 5))
        return max(80, budget // 3)

    def _normalize_alpha_for_direct(self, alpha: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(float(alpha.get(kind, 0.0)), 0.0) for kind in ("entity", "fact", "sentence", "chunk"))
        if total <= 0:
            return {"entity": 0.25, "fact": 0.25, "sentence": 0.25, "chunk": 0.25}
        return {kind: max(float(alpha.get(kind, 0.0)), 0.0) / total for kind in ("entity", "fact", "sentence", "chunk")}

    def _direct_alpha_weights(self, alpha: Dict[str, float]) -> Dict[str, float]:
        alpha = self._normalize_alpha_for_direct(alpha)
        return {kind: 0.75 + 2.0 * float(alpha.get(kind, 0.0)) for kind in ("entity", "fact", "sentence", "chunk")}

    def _direct_alpha_passage_limit(self, alpha: Dict[str, float], budget: int) -> int:
        alpha = self._normalize_alpha_for_direct(alpha)
        return max(80, min(max(80, budget // 2), int(80 + float(alpha.get("chunk", 0.0)) * max(80, budget // 2))))


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
            token_budget=self.config.qa_evidence_token_budget,
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
