#!/usr/bin/env python3
"""Evaluate the clean wo_granularity_awareness ablation."""

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
    if not eval_base._arg_provided(argv, "--disable_granularity_awareness"):
        args.disable_granularity_awareness = True
    if not eval_base._arg_provided(argv, "--disable_sentence_layer"):
        args.disable_sentence_layer = False
    if not eval_base._arg_provided(argv, "--disable_granularity_pagerank_bias"):
        args.disable_granularity_pagerank_bias = True
    if not eval_base._arg_provided(argv, "--disable_evidence_alpha_weights"):
        args.disable_evidence_alpha_weights = True


def main() -> None:
    eval_base.parse_args = parse_args_with_ablation_name
    eval_base.apply_ablation_defaults = apply_wo_granularity_awareness_defaults
    eval_base.main()


if __name__ == "__main__":
    main()
