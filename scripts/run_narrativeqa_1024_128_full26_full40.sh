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

# Local refinement around the best run so far:
# full23 = topk 5, budget 2600, fact_candidates 24, fact_keep 5, title_limit 3.
# These are still standard full hyperparameter sweeps: no module is disabled and
# the shared 1024/128 index is reused.
run_eval full26 5 2400 24 5 3
run_eval full27 5 2500 24 5 3
run_eval full28 5 2700 24 5 3
run_eval full29 5 2800 24 5 3
run_eval full30 5 2600 20 5 3
run_eval full31 5 2600 28 5 3
run_eval full32 5 2600 32 5 3
run_eval full33 5 2600 24 4 3
run_eval full34 5 2600 24 6 3
run_eval full35 5 2600 24 5 2
run_eval full36 5 2600 24 5 4
run_eval full37 4 2400 24 5 3
run_eval full38 4 2800 24 5 3
run_eval full39 6 2400 24 5 3
run_eval full40 6 2600 24 5 3

echo "[all done] full26-full40"
