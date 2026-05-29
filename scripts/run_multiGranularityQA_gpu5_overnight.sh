#!/usr/bin/env bash
set -euo pipefail

cd /data/xyh/code/HoloRAG

mkdir -p results/nohup

FULL_LOG="results/nohup/multiGranularityQA_full_gpu6_overnight.log"
ABLATION_LOG="results/nohup/multiGranularityQA_wo_granularity_awareness_gpu6_overnight.log"
SHARED_INDEX_ROOT="results/multiGranularityQA_eval/shared_indexes/full"

echo "[$(date '+%F %T')] Start full MultiGranularityQA eval on GPU6" | tee "$FULL_LOG"
CUDA_VISIBLE_DEVICES=6 python scripts/multiGranularityQA_eval.py \
  --dataset_file dataset/MultiGranularityQA.json \
  --dataset_format canonical_json \
  --dataset_name multiGranularityQA \
  --split dev \
  --seed 42 \
  --num_eval_queries 1000 \
  --output_dir results/multiGranularityQA_eval \
  --shared_index_root "$SHARED_INDEX_ROOT" \
  --run_name full \
  --embedding_device cuda:0 \
  --task_profile auto \
  --chunk_size_words 256 \
  --chunk_overlap_words 64 \
  2>&1 | tee -a "$FULL_LOG"

echo "[$(date '+%F %T')] Full eval finished; start wo_granularity_awareness on GPU6" | tee "$ABLATION_LOG"
CUDA_VISIBLE_DEVICES=6 python scripts/eval_wo_granularity_awareness.py \
  --dataset_file dataset/MultiGranularityQA.json \
  --dataset_format canonical_json \
  --dataset_name multiGranularityQA \
  --split dev \
  --seed 42 \
  --num_eval_queries 1000 \
  --output_dir results/multiGranularityQA_eval/ablation_runs \
  --shared_index_root "$SHARED_INDEX_ROOT" \
  --run_name wo_granularity_awareness \
  --embedding_device cuda:0 \
  --task_profile auto \
  --chunk_size_words 256 \
  --chunk_overlap_words 64 \
  --recompute_only \
  2>&1 | tee -a "$ABLATION_LOG"

echo "[$(date '+%F %T')] All done."
echo "Full metrics: results/multiGranularityQA_eval/full/metrics_summary.json"
echo "Granularity summary: results/multiGranularityQA_eval/full/granularity_summary.json"
echo "Ablation metrics: results/multiGranularityQA_eval/ablation_runs/wo_granularity_awareness/metrics_summary.json"
