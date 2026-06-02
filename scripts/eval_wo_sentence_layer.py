#!/usr/bin/env python3
"""Evaluate the clean wo_sentence_layer ablation.

This entrypoint intentionally reuses scripts/eval.py for sampling, indexing,
metrics, logging, and QA. The only behavioral change is the ablation itself:
sentence nodes are removed from the runtime graph, and final evidence is packed
from fact/chunk candidates while entities remain graph anchors.
"""

from __future__ import annotations

import sys
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
    from holorag.utils import normalize_scores  # noqa: E402
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    # Keep --help usable in a bare shell; real evaluation still requires the
    # same project dependencies as scripts/eval.py.
    nx = None
    holorag = None
    BaseHoloRAG = object
    Retriever = object
    _IMPORT_ERROR = exc

    def normalize_scores(scores):  # type: ignore[no-redef]
        return scores


NON_SENTENCE_TYPES = ("fact", "chunk")


class WoSentenceProfileRetriever(Retriever):
    """Retriever variant that keeps full retrieval but removes sentence evidence."""

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
            elif node_type == "entity":
                for neighbor in list(graph.successors(node_id)) + list(graph.predecessors(node_id)):
                    if neighbor in graph and graph.nodes[neighbor].get("node_type") == "chunk":
                        chunk_scores[neighbor] += alpha_weights.get("entity_to_chunk", 0.08) * float(score)

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
        allocation = self._renormalized_non_sentence_allocation(profile, alpha=alpha)
        facts = self._rank_fact_evidence(graph, pagerank_scores, ranked_facts)
        chunks = list(ranked_passages[: self.config.passage_output_top_k])

        packed = self._pack_non_sentence_evidence(
            allocation=allocation,
            query=query,
            facts=facts,
            chunks=chunks,
            fallback_passages=ranked_passages,
            sub_questions=sub_questions,
        )
        result = {
            "profile": profile,
            "allocation_mode": "profile_renormalized_without_sentence",
            "allocation": allocation,
            "facts": facts,
            "sentences": [],
            "chunks": chunks,
            "evidence_groups": [
                {"label": "Profile-renormalized fact evidence", "items": facts},
                {"label": "Profile-renormalized chunk evidence", "items": chunks},
            ],
            "fallback_passages": list(ranked_passages[: self.config.qa_passage_top_k]),
        }
        result.update(packed)
        return result

    def _renormalized_non_sentence_allocation(self, profile: str, alpha: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if self._use_llm_alpha_evidence(alpha):
            priors = dict(alpha or {})
        else:
            priors = dict(self.config.profile_alpha_priors.get(profile) or self.config.profile_alpha_priors.get("multi_hop", {}))
        kept = {
            "fact": max(float(priors.get("fact", 0.0)), 0.0),
            "chunk": max(float(priors.get("chunk", 0.0)), 0.0) + max(float(priors.get("sentence", 0.0)), 0.0),
        }
        total = sum(kept.values())
        if total <= 0:
            kept = {"fact": 0.40, "chunk": 0.60}
            total = 1.0
        return {kind: value / total for kind, value in kept.items()}

    def _rank_entity_evidence(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        channel_scores: Dict[str, Dict[str, float]],
        facts: Sequence[Dict],
    ) -> List[Dict]:
        fact_texts_by_entity: Dict[str, List[str]] = defaultdict(list)
        for fact in facts:
            for entity_id in (fact.get("head_id", ""), fact.get("tail_id", "")):
                if entity_id:
                    fact_texts_by_entity[entity_id].append(str(fact.get("text", "")))

        scores: Dict[str, float] = defaultdict(float)
        for node_id, score in pagerank_scores.items():
            if node_id in graph and graph.nodes[node_id].get("node_type") == "entity":
                scores[node_id] += 0.80 * float(score)
        for node_id, score in channel_scores.get("entity", {}).items():
            if node_id in graph and graph.nodes[node_id].get("node_type") == "entity":
                scores[node_id] += 0.20 * float(score)

        records = []
        for node_id, score in sorted(normalize_scores(dict(scores)).items(), key=lambda item: item[1], reverse=True):
            attrs = graph.nodes[node_id]
            name = str(attrs.get("text", "")).strip()
            if not name:
                continue
            hints = [item for item in fact_texts_by_entity.get(node_id, []) if item][:2]
            text = name if not hints else f"{name}. Linked facts: {'; '.join(hints)}"
            records.append({
                "node_id": node_id,
                "score": float(score),
                "title": "",
                "text": text,
                "metadata": attrs.get("metadata", {}),
            })
            if len(records) >= 18:
                break
        return records

    def _rank_fact_evidence(
        self,
        graph: nx.DiGraph,
        pagerank_scores: Dict[str, float],
        ranked_facts: Sequence[Dict],
    ) -> List[Dict]:
        scored = []
        for fact in ranked_facts[: self.config.fact_top_k]:
            head_id = fact.get("head_id", "")
            tail_id = fact.get("tail_id", "")
            chunk_id = fact.get("chunk_id", "")
            endpoint_score = float(pagerank_scores.get(head_id, 0.0)) + float(pagerank_scores.get(tail_id, 0.0))
            chunk_score = float(pagerank_scores.get(chunk_id, 0.0)) if chunk_id in graph else 0.0
            score = float(fact.get("score", 0.0)) + 0.35 * endpoint_score + 0.20 * chunk_score
            record = dict(fact)
            record["score"] = score
            record["sentence_id"] = ""
            scored.append(record)
        return sorted(scored, key=lambda item: item.get("score", 0.0), reverse=True)

    def _pack_non_sentence_evidence(
        self,
        allocation: Dict[str, float],
        query: str,
        facts: Sequence[Dict],
        chunks: Sequence[Dict],
        fallback_passages: Sequence[Dict],
        sub_questions: Sequence[str],
    ) -> Dict:
        count_limits = self._non_sentence_count_limits(allocation)
        candidates_by_kind = {
            "fact": self._fact_candidates(facts, query, sub_questions),
            "chunk": self._chunk_candidates(list(chunks) or list(fallback_passages), query, sub_questions, allocation),
        }

        selected: List[Dict] = []
        used_ids = set()
        selected_texts: List[str] = []
        title_counts: Dict[str, int] = defaultdict(int)
        candidates = [item for items in candidates_by_kind.values() for item in items]
        title_limit = max(1, int(getattr(self.config, "evidence_title_limit", 3)))
        min_score = float(getattr(self.config, "evidence_min_score", 0.0))
        redundancy_threshold = float(getattr(self.config, "evidence_redundancy_threshold", 0.85))

        for kind in NON_SENTENCE_TYPES:
            self._select_count_limited_kind(
                kind=kind,
                candidates=candidates,
                limit=int(count_limits.get(kind, 0)),
                selected=selected,
                selected_ids=used_ids,
                selected_texts=selected_texts,
                title_counts=title_counts,
                title_limit=title_limit,
                min_score=min_score,
                redundancy_threshold=redundancy_threshold,
            )
        self._fill_non_sentence_underflow_with_chunks(
            candidates=candidates,
            count_limits=count_limits,
            selected=selected,
            used_ids=used_ids,
            selected_texts=selected_texts,
            title_counts=title_counts,
            title_limit=title_limit,
            min_score=min_score,
            redundancy_threshold=redundancy_threshold,
        )

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
            "packed_token_budget": 0,
            "packed_token_count": self._token_count(packed_text),
            "evidence_count_limits_by_granularity": {
                "fact": int(count_limits.get("fact", 0)),
                "sentence": 0,
                "chunk": int(count_limits.get("chunk", 0)),
            },
            "used_tokens_by_granularity": {
                "fact": int(used_by_kind.get("fact", 0)),
                "sentence": 0,
                "chunk": int(used_by_kind.get("chunk", 0)),
            },
            "evidence_counts_by_granularity": {
                "fact": int(count_by_kind.get("fact", 0)),
                "sentence": 0,
                "chunk": int(count_by_kind.get("chunk", 0)),
            },
        }

    def _fill_non_sentence_underflow_with_chunks(
        self,
        candidates: Sequence[Dict],
        count_limits: Dict[str, int],
        selected: List[Dict],
        used_ids: set,
        selected_texts: List[str],
        title_counts: Dict[str, int],
        title_limit: int,
        min_score: float,
        redundancy_threshold: float,
    ) -> None:
        target_total = min(
            max(1, int(getattr(self.config, "evidence_alpha_total_units", 20))),
            int(count_limits.get("fact", 0)) + int(getattr(self.config, "passage_output_top_k", 0)),
        )
        remaining = target_total - len(selected)
        if remaining <= 0:
            return
        self._select_count_limited_kind(
            kind="chunk",
            candidates=candidates,
            limit=remaining,
            selected=selected,
            selected_ids=used_ids,
            selected_texts=selected_texts,
            title_counts=title_counts,
            title_limit=title_limit,
            min_score=min_score,
            redundancy_threshold=redundancy_threshold,
        )

    def _non_sentence_count_limits(self, allocation: Dict[str, float]) -> Dict[str, int]:
        total = max(1, int(getattr(self.config, "evidence_alpha_total_units", 20)))
        raw = {kind: total * max(float(allocation.get(kind, 0.0)), 0.0) for kind in NON_SENTENCE_TYPES}
        caps = {
            "fact": max(0, int(getattr(self.config, "fact_top_k", 0))),
            "chunk": max(0, int(getattr(self.config, "passage_output_top_k", 0))),
        }
        limits = {kind: min(caps[kind], int(raw[kind])) for kind in NON_SENTENCE_TYPES}
        remaining = total - sum(limits.values())
        remainders = sorted(
            NON_SENTENCE_TYPES,
            key=lambda kind: (raw[kind] - int(raw[kind]), float(allocation.get(kind, 0.0))),
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
        return {kind: int(limits.get(kind, 0)) for kind in NON_SENTENCE_TYPES}

    def _fact_candidates(self, facts: Sequence[Dict], query: str, sub_questions: Sequence[str]) -> List[Dict]:
        return [
            self._candidate(
                kind="fact",
                label="Fact",
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                score=float(row.get("score", 0.0)),
                node_id=str(row.get("fact_id", "")),
                query=query,
                sub_questions=sub_questions,
                weight=1.15,
            )
            for row in facts
        ]

    def _chunk_candidates(
        self,
        chunks: Sequence[Dict],
        query: str,
        sub_questions: Sequence[str],
        allocation: Dict[str, float],
    ) -> List[Dict]:
        alpha = {"fact": allocation.get("fact", 0.0), "sentence": 0.0, "chunk": allocation.get("chunk", 0.0)}
        return [
            self._candidate(
                kind="chunk",
                label="Passage",
                text=self._passage_excerpt(
                    str(row.get("text", "")),
                    query,
                    self._dynamic_passage_limit(row, query, alpha, 0),
                ),
                title=str(row.get("title", "")),
                score=float(row.get("score", 0.0)),
                node_id=str(row.get("chunk_id", row.get("node_id", ""))),
                query=query,
                sub_questions=sub_questions,
                weight=0.95,
            )
            for row in chunks
        ]

    def _candidate(
        self,
        kind: str,
        label: str,
        text: str,
        title: str,
        score: float,
        node_id: str,
        query: str,
        sub_questions: Sequence[str],
        weight: float,
    ) -> Dict:
        item = self._packed_candidate(
            kind=kind,
            text=text,
            title=title,
            score=score,
            label=label,
            node_id=node_id,
            query=query,
            sub_questions=sub_questions,
        )
        item["pack_score"] = weight * float(item.get("score", 0.0)) + 0.35 * float(item.get("coverage", 0.0))
        return item

    def _try_add_packed_item(
        self,
        item: Dict,
        selected: List[Dict],
        used_ids: set,
        title_counts: Dict[str, int],
        budget: int,
    ) -> tuple[bool, int]:
        node_id = item.get("node_id")
        if node_id and node_id in used_ids:
            return False, 0
        title_key = str(item.get("title", "")).strip().lower()
        title_limit = max(1, int(getattr(self.config, "evidence_title_limit", 3)))
        if title_key and title_counts[title_key] >= title_limit:
            return False, 0
        line = str(item.get("line", "")).strip()
        cost = self._token_count(line)
        if budget > 0 and cost > budget:
            if item.get("kind") != "chunk" and budget < 24:
                return False, 0
            line = self._truncate_words(line, budget)
            cost = self._token_count(line)
        if cost <= 0:
            return False, 0
        selected.append({**item, "line": line, "tokens": cost})
        if node_id:
            used_ids.add(node_id)
        if title_key:
            title_counts[title_key] += 1
        return True, cost


class WoSentenceLayerHoloRAG(BaseHoloRAG):
    """HoloRAG wrapper that removes sentence nodes even when reusing old indexes."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.retriever = WoSentenceProfileRetriever(self.config, self.embedder, self.llm_client)

    def load(self) -> Dict:
        state = super().load()
        return self._strip_sentence_state(state)

    def _strip_sentence_state(self, state: Dict) -> Dict:
        if state.get("_wo_sentence_layer_stripped"):
            return state
        graph = state.get("graph")
        if not isinstance(graph, nx.DiGraph):
            return state
        sentence_nodes = [node_id for node_id, attrs in graph.nodes(data=True) if attrs.get("node_type") == "sentence"]
        new_state = dict(state)
        if sentence_nodes:
            graph = graph.copy()
            graph.remove_nodes_from(sentence_nodes)
            new_state["graph"] = graph
        facts = []
        for fact in state.get("facts", []):
            row = dict(fact)
            row["sentence_id"] = ""
            facts.append(row)
            chunk_id = row.get("chunk_id", "")
            for entity_id in (row.get("head_id", ""), row.get("tail_id", "")):
                if entity_id in graph and chunk_id in graph:
                    self._merge_runtime_edge(graph, entity_id, chunk_id, 1.0, "entity_chunk")
                    self._merge_runtime_edge(graph, chunk_id, entity_id, 1.0, "entity_chunk")
        embeddings = dict(state.get("embeddings", {}))
        embeddings["sentence"] = {}
        new_state["facts"] = facts
        new_state["embeddings"] = embeddings
        new_state["chunk_sentences"] = {}
        new_state["_fact_by_id"] = {fact["fact_id"]: fact for fact in facts}
        new_state["_wo_sentence_layer_stripped"] = True
        self.state = new_state
        return new_state

    def _merge_runtime_edge(self, graph: nx.DiGraph, source: str, target: str, weight: float, edge_type: str) -> None:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] = float(graph[source][target].get("weight", 0.0)) + weight
            kinds = set(graph[source][target].get("edge_kinds", []))
            kinds.add(edge_type)
            graph[source][target]["edge_kinds"] = sorted(kinds)
            return
        graph.add_edge(source, target, weight=weight, edge_type=edge_type, edge_kinds=[edge_type])


_ORIGINAL_BUILD_CONFIG = eval_base.build_config
_ORIGINAL_PARSE_ARGS = eval_base.parse_args


def parse_args_with_ablation_name():
    args = _ORIGINAL_PARSE_ARGS()
    if not args.ablation_name:
        args.ablation_name = "wo_sentence_layer"
    return args


def build_wo_sentence_config(args, save_dir: str):
    config = _ORIGINAL_BUILD_CONFIG(args, save_dir)
    config.enable_sentence_layer = False
    config.enable_granularity_awareness = True
    config.enable_granularity_pagerank_bias = True
    return config


def apply_wo_sentence_defaults(args, ablation_name: str, argv: Sequence[str]) -> None:
    if not eval_base._arg_provided(argv, "--disable_sentence_layer"):
        args.disable_sentence_layer = True
    if not eval_base._arg_provided(argv, "--disable_granularity_awareness"):
        args.disable_granularity_awareness = False
    if not eval_base._arg_provided(argv, "--disable_granularity_pagerank_bias"):
        args.disable_granularity_pagerank_bias = False


def main() -> None:
    if _IMPORT_ERROR is not None:
        if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
            eval_base.parse_args()
            return
        raise _IMPORT_ERROR
    eval_base.parse_args = parse_args_with_ablation_name
    eval_base.build_config = build_wo_sentence_config
    eval_base.apply_ablation_defaults = apply_wo_sentence_defaults
    holorag.HoloRAG = WoSentenceLayerHoloRAG
    eval_base.main()


if __name__ == "__main__":
    main()
