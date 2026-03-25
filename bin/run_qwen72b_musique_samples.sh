#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/xyh/code/HoloRAG"
SAMPLES_GROUP="${SAMPLES_GROUP:-musique_01_10}"
SAMPLES_DIR="${REPO_ROOT}/reproduce/test/groups/${SAMPLES_GROUP}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/qwen_72b_result"
OUTPUT_GROUP_DIR="${OUTPUT_ROOT}/groups/${SAMPLES_GROUP}"
LOG_DIR="${OUTPUT_ROOT}/logs"
EVAL_DIR="${OUTPUT_ROOT}/eval"

LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_NAME="${LLM_NAME:-/data/xyh/models/Qwen2.5-72B-Instruct}"
EMBEDDING_NAME="${EMBEDDING_NAME:-/data/xyh/models/NV-Embed-v2}"

EMBEDDING_VISIBLE_DEVICES="${EMBEDDING_VISIBLE_DEVICES:-2}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-1800}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_GROUP_DIR}" "${LOG_DIR}" "${EVAL_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

wait_for_vllm() {
  local waited=0
  log "Waiting for vLLM at ${LLM_BASE_URL}"
  until curl -fsS "${LLM_BASE_URL}/models" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
      log "Timed out after ${WAIT_TIMEOUT_SECONDS}s waiting for ${LLM_BASE_URL}/models"
      return 1
    fi
  done
  log "vLLM is ready."
}

run_stage() {
  local stage="$1"
  local sample_path="$2"
  local output_dir="$3"
  local sample_name
  sample_name="$(basename "${sample_path}" .json)"

  log "Running ${stage} for ${sample_name}"
  CUDA_VISIBLE_DEVICES="${EMBEDDING_VISIBLE_DEVICES}" \
    python "${REPO_ROOT}/main_holorag.py" "${stage}" \
      --corpus_file "${sample_path}" \
      --output_dir "${output_dir}" \
      --llm_base_url "${LLM_BASE_URL}" \
      --llm_name "${LLM_NAME}" \
      --embedding_name "${EMBEDDING_NAME}" \
      --embedding_device "${EMBEDDING_DEVICE}" \
      2>&1 | tee "${LOG_DIR}/${sample_name}_${stage}.log"
}

wait_for_vllm

shopt -s nullglob
sample_paths=("${SAMPLES_DIR}"/sample_musique*.json)
if (( ${#sample_paths[@]} == 0 )); then
  log "No samples found under ${SAMPLES_DIR}"
  exit 1
fi

for sample_path in "${sample_paths[@]}"; do
  sample_name="$(basename "${sample_path}" .json)"
  output_dir="${OUTPUT_GROUP_DIR}/${sample_name}"
  mkdir -p "${output_dir}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && [[ -f "${output_dir}/holorag_index.pkl" && -f "${output_dir}/last_query_result.json" ]]; then
    log "Skipping ${sample_name} because outputs already exist."
    continue
  fi

  run_stage "index" "${sample_path}" "${output_dir}"
  run_stage "query" "${sample_path}" "${output_dir}"
done

log "Running evaluation summary into ${EVAL_DIR}/holorag_eval_musique_samples_${SAMPLES_GROUP#musique_}.json"
python "${REPO_ROOT}/eval_holorag_musique.py" \
  --samples_glob "${SAMPLES_DIR}/sample_musique*.json" \
  --outputs_dir "${OUTPUT_ROOT}" \
  --result_filename "last_query_result.json" \
  --retrieval_k 5 \
  --llm_base_url "${LLM_BASE_URL}" \
  --llm_name "${LLM_NAME}" \
  --output_json "${EVAL_DIR}/holorag_eval_musique_samples_${SAMPLES_GROUP#musique_}.json" \
  2>&1 | tee "${LOG_DIR}/${SAMPLES_GROUP}_eval.log"

log "All sample runs completed for ${SAMPLES_GROUP}."
