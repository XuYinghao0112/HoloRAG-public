#!/usr/bin/env python3
"""Evaluate the strict wo_sentence_layer ablation.

This variant removes sentence nodes/evidence without reallocating the removed
sentence quota to facts or chunks. It is intended as a no-compensation ablation:
the sentence layer is unavailable, and its evidence units remain unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval as eval_base  # noqa: E402
import eval_wo_sentence_layer as wo_sentence_base  # noqa: E402

try:
    import holorag  # noqa: E402
    _IMPORT_ERROR = wo_sentence_base._IMPORT_ERROR
except ModuleNotFoundError as exc:
    holorag = None
    _IMPORT_ERROR = exc


class StrictWoSentenceProfileRetriever(wo_sentence_base.WoSentenceProfileRetriever):
    """No-compensation sentence-layer ablation retriever."""

    def _renormalized_non_sentence_allocation(self, profile: str, alpha: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if self._use_llm_alpha_evidence(alpha):
            priors = dict(alpha or {})
        else:
            priors = dict(self.config.profile_alpha_priors.get(profile) or self.config.profile_alpha_priors.get("multi_hop", {}))
        return {
            "fact": max(float(priors.get("fact", 0.0)), 0.0),
            "chunk": max(float(priors.get("chunk", 0.0)), 0.0),
        }

    def _non_sentence_count_limits(self, allocation: Dict[str, float]) -> Dict[str, int]:
        total = max(1, int(getattr(self.config, "evidence_alpha_total_units", 20)))
        return {
            "fact": min(
                max(0, int(getattr(self.config, "fact_top_k", 0))),
                int(total * max(float(allocation.get("fact", 0.0)), 0.0)),
            ),
            "chunk": min(
                max(0, int(getattr(self.config, "passage_output_top_k", 0))),
                int(total * max(float(allocation.get("chunk", 0.0)), 0.0)),
            ),
        }

    def _fill_non_sentence_underflow_with_chunks(self, *args, **kwargs) -> None:
        return None


class StrictWoSentenceLayerHoloRAG(wo_sentence_base.WoSentenceLayerHoloRAG):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.retriever = StrictWoSentenceProfileRetriever(self.config, self.embedder, self.llm_client)

    def _strip_sentence_state(self, state: Dict) -> Dict:
        if state.get("_wo_sentence_layer_strict_stripped"):
            return state
        graph = state.get("graph")
        if not isinstance(graph, wo_sentence_base.nx.DiGraph):
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
        embeddings = dict(state.get("embeddings", {}))
        embeddings["sentence"] = {}
        new_state["facts"] = facts
        new_state["embeddings"] = embeddings
        new_state["chunk_sentences"] = {}
        new_state["_fact_by_id"] = {fact["fact_id"]: fact for fact in facts}
        new_state["_wo_sentence_layer_strict_stripped"] = True
        self.state = new_state
        return new_state


_ORIGINAL_PARSE_ARGS = eval_base.parse_args
_ORIGINAL_BUILD_CONFIG = eval_base.build_config


def parse_args_with_ablation_name():
    args = _ORIGINAL_PARSE_ARGS()
    if not args.ablation_name:
        args.ablation_name = "wo_sentence_layer_strict"
    return args


def build_strict_wo_sentence_config(args, save_dir: str):
    return wo_sentence_base.build_wo_sentence_config(args, save_dir)


def apply_strict_wo_sentence_defaults(args, ablation_name: str, argv: Sequence[str]) -> None:
    wo_sentence_base.apply_wo_sentence_defaults(args, ablation_name, argv)


def main() -> None:
    if _IMPORT_ERROR is not None:
        if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
            eval_base.parse_args()
            return
        raise _IMPORT_ERROR
    eval_base.parse_args = parse_args_with_ablation_name
    eval_base.build_config = build_strict_wo_sentence_config
    eval_base.apply_ablation_defaults = apply_strict_wo_sentence_defaults
    holorag.HoloRAG = StrictWoSentenceLayerHoloRAG
    eval_base.main()


if __name__ == "__main__":
    main()
