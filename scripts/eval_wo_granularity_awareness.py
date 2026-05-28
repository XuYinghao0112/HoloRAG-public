#!/usr/bin/env python3
"""Evaluate the wo_granularity_awareness ablation with the finalized v7 defaults."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval as eval_base  # noqa: E402


_ORIGINAL_PARSE_ARGS = eval_base.parse_args


def parse_args_with_ablation_name():
    args = _ORIGINAL_PARSE_ARGS()
    if not args.ablation_name:
        args.ablation_name = "wo_granularity_awareness"
    return args


def apply_wo_granularity_awareness_defaults(args, ablation_name: str, argv: Sequence[str]) -> None:
    defaults = {
        "--topk_passages": ("topk_passages", 4),
        "--passage_output_top_k": ("passage_output_top_k", 10),
        "--qa_evidence_token_budget": ("qa_evidence_token_budget", 1700),
        "--fact_rerank_llm_candidate_k": ("fact_rerank_llm_candidate_k", 20),
        "--fact_rerank_llm_keep_k": ("fact_rerank_llm_keep_k", 6),
        "--evidence_extra_ranked_sentence_k": ("evidence_extra_ranked_sentence_k", 0),
        "--evidence_max_sentences": ("evidence_max_sentences", 7),
        "--evidence_title_limit": ("evidence_title_limit", 2),
        "--evidence_passage_context_k": ("evidence_passage_context_k", 4),
        "--evidence_passage_excerpt_tokens": ("evidence_passage_excerpt_tokens", 300),
    }
    for flag, (attr, value) in defaults.items():
        if not eval_base._arg_provided(argv, flag):
            setattr(args, attr, value)

    if not eval_base._arg_provided(argv, "--disable_granularity_awareness"):
        args.disable_granularity_awareness = True
    if not eval_base._arg_provided(argv, "--disable_sentence_layer"):
        args.disable_sentence_layer = False
    if not eval_base._arg_provided(argv, "--disable_granularity_pagerank_bias"):
        args.disable_granularity_pagerank_bias = True

    if not eval_base._arg_provided(argv, "--enable_fact_source_first_evidence"):
        args.enable_fact_source_first_evidence = False
    if not eval_base._arg_provided(argv, "--enable_fact_chunk_boost"):
        args.enable_fact_chunk_boost = False
    if not eval_base._arg_provided(argv, "--enable_fair_sentence_context"):
        args.enable_fair_sentence_context = False


def main() -> None:
    eval_base.parse_args = parse_args_with_ablation_name
    eval_base.apply_ablation_defaults = apply_wo_granularity_awareness_defaults
    eval_base.main()


if __name__ == "__main__":
    main()
