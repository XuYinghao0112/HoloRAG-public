#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/xyh/miniconda3/envs/holorag-dev/bin/python}"
DATASET_FILE="${DATASET_FILE:-dataset/hotpotqa_full/distractor/validation-00000-of-00001.parquet}"
FIXED_QUERIES_FILE="${FIXED_QUERIES_FILE:-results/global_scaling/hotpotqa_G500_G1K_G2K_G5K_G10K/sampled_queries.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/global_scaling/paper_clean_hotpotqa_gold_covered}"
LOG_DIR="${LOG_DIR:-results/nohup/paper_clean_hotpotqa_gold_covered}"
SCALES="${SCALES:-500 1000 2000 5000 10000}"
CORPUS_SEEDS="${CORPUS_SEEDS:-42 43 44}"
QUERY_SEED="${QUERY_SEED:-42}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-100}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:5}"
DEVICES="${DEVICES:-cuda:0 cuda:5}"
ASSIGNMENT_MODE="${ASSIGNMENT_MODE:-dynamic}"
PRIMARY_DEVICE="${PRIMARY_DEVICE:-cuda:0}"
G10K_DEVICE="${G10K_DEVICE:-cuda:3}"
WAIT_FOR_PATTERN="${WAIT_FOR_PATTERN-hotpotqa_G500_G1K_G2K_G5K_G10K}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_G10K="${RUN_G10K:-1}"
TASK_FILE="${TASK_FILE:-${OUTPUT_DIR}/task_queue.tsv}"
STATE_DIR="${STATE_DIR:-${OUTPUT_DIR}/queue_state}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_DIR}/task_queue.lock}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$STATE_DIR"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

wait_for_previous_run() {
  if [[ -z "$WAIT_FOR_PATTERN" ]]; then
    return
  fi
  while true; do
    mapfile -t pids < <(
      pgrep -af "python .*scripts/eval_global_scaling.py" \
        | grep "$WAIT_FOR_PATTERN" \
        | awk '{print $1}' \
        | grep -v "^$$$" || true
    )
    if [[ "${#pids[@]}" -eq 0 ]]; then
      log "No previous eval_global_scaling.py process matching '$WAIT_FOR_PATTERN'; starting paper-clean runs."
      return
    fi
    log "Waiting for previous run matching '$WAIT_FOR_PATTERN' to finish: ${pids[*]}"
    sleep "$POLL_SECONDS"
  done
}

run_name_for() {
  local scale="$1"
  local corpus_seed="$2"
  local scale_label="G${scale}"
  printf 'hotpotqa_gold_covered_clean_q%s_c%s_%s' "$QUERY_SEED" "$corpus_seed" "$scale_label"
}

init_task_queue() {
  local tmp="${TASK_FILE}.tmp"
  : > "$tmp"
  for corpus_seed in $CORPUS_SEEDS; do
    for scale in $SCALES; do
      if [[ "$scale" == "10000" && "$RUN_G10K" != "1" ]]; then
        log "Skipping G10000 because RUN_G10K=$RUN_G10K"
        continue
      fi
      local run_name
      run_name="$(run_name_for "$scale" "$corpus_seed")"
      local run_dir="${OUTPUT_DIR}/${run_name}"
      if [[ -f "${run_dir}/metrics_summary.json" && -f "${run_dir}/config.json" ]]; then
        printf 'completed\t%s\t%s\t-\t-\t-\t%s\n' "$corpus_seed" "$scale" "$run_name" >> "$tmp"
      else
        printf 'pending\t%s\t%s\t-\t-\t-\t%s\n' "$corpus_seed" "$scale" "$run_name" >> "$tmp"
      fi
    done
  done
  mv "$tmp" "$TASK_FILE"
  log "Prepared task queue: $TASK_FILE"
}

claim_task() {
  local device="$1"
  local safe_device="${device//[:\/]/_}"
  local claim_file="${STATE_DIR}/claim_${safe_device}.txt"
  (
    flock -x 9
    local claim
    claim="$(awk -F '\t' '$1 == "pending" {print NR "\t" $2 "\t" $3 "\t" $7; exit}' "$TASK_FILE")"
    if [[ -z "$claim" ]]; then
      : > "$claim_file"
      exit 0
    fi
    local line corpus_seed scale run_name now
    IFS=$'\t' read -r line corpus_seed scale run_name <<< "$claim"
    now="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    awk -F '\t' -v OFS='\t' -v line="$line" -v device="$device" -v now="$now" '
      NR == line {$1 = "running"; $4 = device; $5 = now}
      {print}
    ' "$TASK_FILE" > "${TASK_FILE}.tmp"
    mv "${TASK_FILE}.tmp" "$TASK_FILE"
    printf '%s\t%s\t%s\n' "$corpus_seed" "$scale" "$run_name" > "$claim_file"
  ) 9>"$LOCK_FILE"
  [[ -s "$claim_file" ]] || return 1
  IFS=$'\t' read -r CLAIM_CORPUS_SEED CLAIM_SCALE CLAIM_RUN_NAME < "$claim_file"
}

mark_task() {
  local status="$1"
  local corpus_seed="$2"
  local scale="$3"
  local run_name="$4"
  local now
  now="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  (
    flock -x 9
    awk -F '\t' -v OFS='\t' -v status="$status" -v seed="$corpus_seed" -v scale="$scale" -v run="$run_name" -v now="$now" '
      $2 == seed && $3 == scale && $7 == run {$1 = status; $6 = now}
      {print}
    ' "$TASK_FILE" > "${TASK_FILE}.tmp"
    mv "${TASK_FILE}.tmp" "$TASK_FILE"
  ) 9>"$LOCK_FILE"
}

run_one() {
  local scale="$1"
  local corpus_seed="$2"
  local embedding_device="$3"
  local run_name
  run_name="$(run_name_for "$scale" "$corpus_seed")"
  local run_dir="${OUTPUT_DIR}/${run_name}"
  local log_file="${LOG_DIR}/${run_name}.log"

  if [[ -f "${run_dir}/metrics_summary.json" && -f "${run_dir}/config.json" ]]; then
    log "Skipping completed run: ${run_name}"
    return
  fi

  log "Starting ${run_name} on ${embedding_device}; log=${log_file}"
  "$PYTHON_BIN" scripts/eval_global_scaling.py \
    --dataset_file "$DATASET_FILE" \
    --dataset_format hotpot_parquet \
    --dataset_name hotpotqa \
    --split validation \
    --sampled_queries_file "$FIXED_QUERIES_FILE" \
    --num_eval_queries "$NUM_EVAL_QUERIES" \
    --seed "$QUERY_SEED" \
    --corpus_seed "$corpus_seed" \
    --scales "$scale" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name" \
    --corpus_dataset hotpotqa_train:dataset/hotpotqa_full/distractor/train-00000-of-00002.parquet:hotpot_parquet:train \
    --corpus_dataset hotpotqa_train:dataset/hotpotqa_full/distractor/train-00001-of-00002.parquet:hotpot_parquet:train \
    --embedding_device "$embedding_device" \
    > "$log_file" 2>&1
  log "Finished ${run_name} on ${embedding_device}"
}

worker() {
  local embedding_device="$1"
  log "Worker started on ${embedding_device}"
  while claim_task "$embedding_device"; do
    local corpus_seed="$CLAIM_CORPUS_SEED"
    local scale="$CLAIM_SCALE"
    local run_name="$CLAIM_RUN_NAME"
    if run_one "$scale" "$corpus_seed" "$embedding_device"; then
      mark_task "completed" "$corpus_seed" "$scale" "$run_name"
    else
      mark_task "failed" "$corpus_seed" "$scale" "$run_name"
      log "FAILED ${run_name} on ${embedding_device}; continuing with remaining tasks."
    fi
  done
  log "Worker finished on ${embedding_device}"
}

run_split_g10k_worker() {
  local embedding_device="$1"
  local assigned_scales="$2"
  log "Split worker started on ${embedding_device} for scales: ${assigned_scales}"
  for corpus_seed in $CORPUS_SEEDS; do
    for scale in $assigned_scales; do
      if [[ "$scale" == "10000" && "$RUN_G10K" != "1" ]]; then
        log "Skipping G10000 because RUN_G10K=$RUN_G10K"
        continue
      fi
      run_one "$scale" "$corpus_seed" "$embedding_device"
    done
  done
  log "Split worker finished on ${embedding_device}"
}

wait_for_previous_run

if [[ ! -f "$FIXED_QUERIES_FILE" ]]; then
  log "Missing fixed query file: $FIXED_QUERIES_FILE"
  exit 1
fi

status=0
if [[ "$ASSIGNMENT_MODE" == "split_g10k" ]]; then
  primary_scales=""
  for scale in $SCALES; do
    if [[ "$scale" == "10000" ]]; then
      continue
    fi
    primary_scales="${primary_scales} ${scale}"
  done
  run_split_g10k_worker "$PRIMARY_DEVICE" "$primary_scales" &
  primary_pid="$!"
  run_split_g10k_worker "$G10K_DEVICE" "10000" &
  g10k_pid="$!"
  wait "$primary_pid" || status=1
  wait "$g10k_pid" || status=1
else
  init_task_queue

  worker_pids=()
  for device in $DEVICES; do
    worker "$device" &
    worker_pids+=("$!")
  done

  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done

  if grep -q $'^failed\t' "$TASK_FILE"; then
    status=1
    log "Some paper-clean tasks failed; see $TASK_FILE and $LOG_DIR"
  fi
fi

"$PYTHON_BIN" scripts/summarize_global_scaling_repeats.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/summary_mean_std.csv"
log "All paper-clean runs complete. Summary: ${OUTPUT_DIR}/summary_mean_std.csv"
exit "$status"
