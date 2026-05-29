#!/usr/bin/env bash
set -euo pipefail

cd /data/xyh/code/HoloRAG

mkdir -p results/nohup

PURE_LOG="results/nohup/multiGranularityQA_wo_granularity_awareness_pure_gpu7_overnight.log"
DRIVER_LOG_NOTE="Use tail -f on this script's nohup output or on the two logs above."
SHARED_INDEX_ROOT="results/multiGranularityQA_eval/shared_indexes/full"
FULL_METRICS="results/multiGranularityQA_eval/full/metrics_summary.json"
FULL_WAIT_SECONDS="${FULL_WAIT_SECONDS:-21600}"
FULL_WAIT_INTERVAL_SECONDS="${FULL_WAIT_INTERVAL_SECONDS:-300}"

echo "[$(date '+%F %T')] MultiGranularityQA pure ablation overnight run"
echo "[$(date '+%F %T')] ${DRIVER_LOG_NOTE}"

if [[ -f "$FULL_METRICS" ]]; then
  echo "[$(date '+%F %T')] Full eval metrics already exist, skip full eval: $FULL_METRICS"
else
  echo "[$(date '+%F %T')] Full metrics not found; waiting for GPU6 full eval: $FULL_METRICS"
  waited=0
  while [[ ! -f "$FULL_METRICS" && "$waited" -lt "$FULL_WAIT_SECONDS" ]]; do
    sleep "$FULL_WAIT_INTERVAL_SECONDS"
    waited=$((waited + FULL_WAIT_INTERVAL_SECONDS))
    echo "[$(date '+%F %T')] Still waiting for full metrics (${waited}s/${FULL_WAIT_SECONDS}s): $FULL_METRICS"
  done
  if [[ ! -f "$FULL_METRICS" ]]; then
    echo "[$(date '+%F %T')] ERROR: Full metrics did not appear within ${FULL_WAIT_SECONDS}s. Stop pure ablation."
    exit 1
  fi
fi

echo "[$(date '+%F %T')] Start pure wo_granularity_awareness eval on GPU7" | tee "$PURE_LOG"
CUDA_VISIBLE_DEVICES=7 python scripts/eval.py \
  --dataset_file dataset/MultiGranularityQA.json \
  --dataset_format canonical_json \
  --dataset_name multiGranularityQA \
  --split dev \
  --seed 42 \
  --num_eval_queries 1000 \
  --output_dir results/multiGranularityQA_eval/ablation_runs \
  --shared_index_root "$SHARED_INDEX_ROOT" \
  --run_name wo_granularity_awareness_pure \
  --embedding_device cuda:0 \
  --task_profile auto \
  --chunk_size_words 256 \
  --chunk_overlap_words 64 \
  --disable_paragraph_as_chunk \
  --topk_passages 4 \
  --passage_output_top_k 10 \
  --qa_max_input_tokens 7000 \
  --qa_evidence_token_budget 820 \
  --fact_rerank_llm_candidate_k 20 \
  --fact_rerank_llm_keep_k 7 \
  --evidence_extra_ranked_sentence_k 3 \
  --evidence_max_sentences 15 \
  --evidence_title_limit 3 \
  --evidence_passage_context_k 1 \
  --evidence_passage_excerpt_tokens 100 \
  --disable_granularity_awareness \
  --recompute_only \
  2>&1 | tee -a "$PURE_LOG"

echo "[$(date '+%F %T')] All done."
echo "Full metrics: results/multiGranularityQA_eval/full/metrics_summary.json"
echo "Full granularity summary: results/multiGranularityQA_eval/full/granularity_summary.json"
echo "Pure ablation metrics: results/multiGranularityQA_eval/ablation_runs/wo_granularity_awareness_pure/metrics_summary.json"
