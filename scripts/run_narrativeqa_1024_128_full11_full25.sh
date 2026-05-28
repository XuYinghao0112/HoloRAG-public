#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-5}"
PYTHON_BIN="${PYTHON_BIN:-/home/xyh/miniconda3/envs/holorag-dev/bin/python}"

DATASET_FILE="dataset/narrativeqa_canonical.json"
OUTPUT_DIR="results/narrativeqa_eval"
SHARED_INDEX_ROOT="results/narrativeqa_eval/shared_indexes/full"
LOG_DIR="results/nohup"

mkdir -p "${LOG_DIR}"

run_eval() {
  local run_name="$1"
  local topk_passages="$2"
  local evidence_budget="$3"
  local rerank_candidate_k="$4"
  local rerank_keep_k="$5"
  local title_limit="$6"
  local log_file="${LOG_DIR}/narrativeqa_${run_name}_gpu${GPU_ID}.nohup.log"

  if [[ -f "${OUTPUT_DIR}/${run_name}/metrics_summary.json" ]]; then
    echo "[skip] ${run_name} already has metrics_summary.json"
    return 0
  fi

  echo "[start] ${run_name}: topk=${topk_passages}, budget=${evidence_budget}, fact_candidates=${rerank_candidate_k}, fact_keep=${rerank_keep_k}, title_limit=${title_limit}, gpu=${GPU_ID}"
  "${PYTHON_BIN}" scripts/eval.py \
    --dataset_file "${DATASET_FILE}" \
    --dataset_format canonical_json \
    --dataset_name narrativeqa \
    --output_dir "${OUTPUT_DIR}" \
    --shared_index_root "${SHARED_INDEX_ROOT}" \
    --run_name "${run_name}" \
    --num_eval_queries 200 \
    --task_profile long_context \
    --disable_paragraph_as_chunk \
    --chunk_size_words 1024 \
    --chunk_overlap_words 128 \
    --topk_passages "${topk_passages}" \
    --qa_evidence_token_budget "${evidence_budget}" \
    --fact_rerank_llm_candidate_k "${rerank_candidate_k}" \
    --fact_rerank_llm_keep_k "${rerank_keep_k}" \
    --evidence_title_limit "${title_limit}" \
    --embedding_device "cuda:${GPU_ID}" \
    > "${log_file}" 2>&1
  echo "[done] ${run_name}"
}

# The sweep keeps the standard full setup intact:
# - 1024/128 chunking
# - long_context profile
# - all full modules enabled
# - shared 1024/128 indexes reused
# It only varies retrieval depth, evidence budget, fact rerank pool, and
# title diversity cap. Budgets cover a wider 1600-3500 range.
run_eval full11 3 1600 20 7 3
run_eval full12 3 2200 20 7 3
run_eval full13 3 3000 20 7 3
run_eval full14 4 1600 20 7 3
run_eval full15 4 1800 20 7 3
run_eval full16 4 2200 20 7 3
run_eval full17 4 2600 20 7 3
run_eval full18 4 3000 20 7 3
run_eval full19 4 3500 20 7 3
run_eval full20 4 2600 24 5 3
run_eval full21 5 1800 20 7 3
run_eval full22 5 2200 20 7 3
run_eval full23 5 2600 24 5 3
run_eval full24 6 2200 20 7 3
run_eval full25 6 3000 20 7 3

echo "[all done] full11-full25"
