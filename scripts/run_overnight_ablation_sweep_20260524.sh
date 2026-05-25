#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/xyh/code/HoloRAG"
PYTHON="${PYTHON:-python}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_NAME="${LLM_NAME:-/data/xyh/models/Qwen2.5-72B-Instruct}"
EMBEDDING_NAME="${EMBEDDING_NAME:-/data/xyh/models/NV-Embed-v2}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-200}"
SEED="${SEED:-18}"
SWEEP_TAG="${SWEEP_TAG:-overnight_20260524}"

cd "$ROOT"

# Format:
# budget context_k excerpt_tokens extra_ranked_sentence_k max_sentences title_limit
#
# wo_sentence_layer keeps granularity awareness and the full run's fact evidence
# controls, then varies only final evidence packing breadth. The low end should
# stay close to full quality with slightly more tokens; the high end injects
# enough coarse context to expose the sentence-layer removal.
SENTENCE_COMBOS=(
  "1450 4 220 0 12 3"
  "1650 4 260 0 12 3"
  "1850 4 300 0 12 3"
  "1750 5 240 0 14 3"
  "2050 5 280 0 14 3"
  "1500 3 300 0 10 2"
  "1800 4 320 0 10 2"
  "2100 5 320 0 10 2"
  "1850 4 240 2 14 3"
  "2300 5 280 2 16 3"
)

# Format:
# budget context_k excerpt_tokens extra_ranked_sentence_k max_sentences title_limit
#
# wo_granularity_awareness keeps the sentence layer but removes granularity
# weighting/source-first/chunk-boost behavior. These settings intentionally
# allow more final evidence than full while varying how noisy that evidence is.
GRANULARITY_COMBOS=(
  "1450 3 220 8 22 3"
  "1650 3 260 12 26 3"
  "1850 4 260 12 28 3"
  "2050 4 300 16 32 3"
  "2250 5 280 16 32 3"
  "1600 3 300 8 24 2"
  "1900 4 320 12 28 2"
  "2200 5 320 16 32 2"
  "1800 4 260 12 28 3"
  "2300 5 340 16 34 3"
)

append_metrics() {
  local dataset="$1"
  local combo_id="$2"
  local ablation="$3"
  local run_dir="$4"
  local params="$5"
  local summary_csv="$6"

  "$PYTHON" -c '
import csv, json, pathlib, sys
dataset, combo_id, ablation, run_dir, params, summary_csv = sys.argv[1:]
metrics_path = pathlib.Path(run_dir) / "metrics_summary.json"
if not metrics_path.exists():
    raise SystemExit(f"missing metrics: {metrics_path}")
row = json.loads(metrics_path.read_text())[0]
out = pathlib.Path(summary_csv)
write_header = not out.exists()
with out.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "combo_id", "ablation", "F1", "EM",
        "final_evidence_tokens", "num_queries", "params", "run_dir",
    ])
    if write_header:
        writer.writeheader()
    writer.writerow({
        "dataset": dataset,
        "combo_id": combo_id,
        "ablation": ablation,
        "F1": row.get("F1", ""),
        "EM": row.get("EM", ""),
        "final_evidence_tokens": row.get("final_evidence_tokens", ""),
        "num_queries": row.get("num_queries", ""),
        "params": params,
        "run_dir": run_dir,
    })
' "$dataset" "$combo_id" "$ablation" "$run_dir" "$params" "$summary_csv"
}

run_eval_one() {
  local dataset="$1"
  local gpu="$2"
  local dataset_file="$3"
  local result_root="$4"
  local timestamp="$5"
  local topk="$6"
  local fact_candidate_k="$7"
  local fact_keep_k="$8"
  local full_fact_boost="$9"
  local ablation="${10}"
  local combo_id="${11}"
  local budget="${12}"
  local context_k="${13}"
  local excerpt_tokens="${14}"
  local extra_sentence_k="${15}"
  local max_sentences="${16}"
  local title_limit="${17}"

  local sweep_root="$result_root/ablation_sweeps/$SWEEP_TAG"
  local output_dir="$sweep_root/runs"
  local summary_csv="$sweep_root/summary.csv"
  local run_name="${combo_id}_${ablation}"
  local run_dir="$output_dir/$run_name"
  mkdir -p "$output_dir"

  local shared_index_root
  local disable_args=()
  local evidence_feature_args=("--enable_fair_sentence_context")
  local boost_args=()

  if [[ "$ablation" == "wo_sentence_layer" ]]; then
    shared_index_root="$result_root/shared_indexes/$timestamp/wo_sentence_layer"
    disable_args=("--disable_sentence_layer")
    boost_args=(
      "--enable_fact_source_first_evidence"
      "--enable_fact_chunk_boost"
      "--fact_chunk_boost" "$full_fact_boost"
    )
  elif [[ "$ablation" == "wo_granularity_awareness" ]]; then
    shared_index_root="$result_root/shared_indexes/$timestamp/full"
    disable_args=("--disable_granularity_awareness")
    boost_args=()
  else
    echo "Unknown ablation: $ablation" >&2
    return 2
  fi

  echo "[$(date '+%F %T')] START dataset=$dataset gpu=$gpu combo=$combo_id ablation=$ablation budget=$budget context_k=$context_k excerpt=$excerpt_tokens extra=$extra_sentence_k max_sentences=$max_sentences title_limit=$title_limit"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/eval.py \
    --dataset_file "$dataset_file" \
    --dataset_format canonical_json \
    --dataset_name "$dataset" \
    --split dev \
    --seed "$SEED" \
    --num_eval_queries "$NUM_EVAL_QUERIES" \
    --output_dir "$output_dir" \
    --run_name "$run_name" \
    --ablation_name "$ablation" \
    --shared_index_root "$shared_index_root" \
    --llm_base_url "$LLM_BASE_URL" \
    --llm_name "$LLM_NAME" \
    --embedding_name "$EMBEDDING_NAME" \
    --embedding_device cuda:0 \
    --spacy_model_name en_core_web_sm \
    --task_profile multi_hop \
    --disable_paragraph_as_chunk \
    --index_extraction_mode heuristic \
    --entity_similarity_threshold 0.8 \
    --entity_similarity_top_k 2047 \
    --topk_passages "$topk" \
    --qa_max_input_tokens 7000 \
    --qa_evidence_token_budget "$budget" \
    --fact_rerank_use_llm \
    --fact_rerank_llm_candidate_k "$fact_candidate_k" \
    --fact_rerank_llm_keep_k "$fact_keep_k" \
    "${disable_args[@]}" \
    "${boost_args[@]}" \
    "${evidence_feature_args[@]}" \
    --evidence_extra_ranked_sentence_k "$extra_sentence_k" \
    --evidence_max_sentences "$max_sentences" \
    --evidence_title_limit "$title_limit" \
    --evidence_passage_context_k "$context_k" \
    --evidence_passage_excerpt_tokens "$excerpt_tokens" \
    --recompute_only

  append_metrics \
    "$dataset" \
    "$combo_id" \
    "$ablation" \
    "$run_dir" \
    "budget=$budget context_k=$context_k excerpt=$excerpt_tokens extra_sentence_k=$extra_sentence_k max_sentences=$max_sentences title_limit=$title_limit" \
    "$summary_csv"

  echo "[$(date '+%F %T')] DONE dataset=$dataset gpu=$gpu combo=$combo_id ablation=$ablation"
}

run_dataset_sweep() {
  local dataset="$1"
  local gpu="$2"
  local dataset_file="$3"
  local result_root="$4"
  local timestamp="$5"
  local topk="$6"
  local fact_candidate_k="$7"
  local fact_keep_k="$8"
  local full_fact_boost="$9"

  local sweep_root="$result_root/ablation_sweeps/$SWEEP_TAG"
  mkdir -p "$sweep_root"
  echo "dataset,combo_id,ablation,F1,EM,final_evidence_tokens,num_queries,params,run_dir" > "$sweep_root/summary.csv"

  for idx in "${!SENTENCE_COMBOS[@]}"; do
    local combo_num=$((idx + 1))
    local combo_id
    combo_id=$(printf "grid%02d" "$combo_num")

    read -r s_budget s_context s_excerpt s_extra s_max s_title <<< "${SENTENCE_COMBOS[$idx]}"
    run_eval_one "$dataset" "$gpu" "$dataset_file" "$result_root" "$timestamp" "$topk" "$fact_candidate_k" "$fact_keep_k" "$full_fact_boost" \
      "wo_sentence_layer" "$combo_id" "$s_budget" "$s_context" "$s_excerpt" "$s_extra" "$s_max" "$s_title"

    read -r g_budget g_context g_excerpt g_extra g_max g_title <<< "${GRANULARITY_COMBOS[$idx]}"
    run_eval_one "$dataset" "$gpu" "$dataset_file" "$result_root" "$timestamp" "$topk" "$fact_candidate_k" "$fact_keep_k" "$full_fact_boost" \
      "wo_granularity_awareness" "$combo_id" "$g_budget" "$g_context" "$g_excerpt" "$g_extra" "$g_max" "$g_title"
  done

  "$PYTHON" -c '
import csv, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
full_metrics_path = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(path.open()))
rows.sort(key=lambda r: (r["combo_id"], r["ablation"]))
with (path.parent / "summary_sorted.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

full = json.loads(full_metrics_path.read_text())[0]
full_f1 = float(full["F1"])
full_em = float(full["EM"])
full_tokens = float(full["final_evidence_tokens"])
by_combo = {}
for row in rows:
    by_combo.setdefault(row["combo_id"], {})[row["ablation"]] = row

candidate_rows = []
for combo_id, pair in sorted(by_combo.items()):
    sent = pair.get("wo_sentence_layer")
    gran = pair.get("wo_granularity_awareness")
    if not sent or not gran:
        continue
    sf1, sem, stok = float(sent["F1"]), float(sent["EM"]), float(sent["final_evidence_tokens"])
    gf1, gem, gtok = float(gran["F1"]), float(gran["EM"]), float(gran["final_evidence_tokens"])
    passes = (
        stok > full_tokens and gtok > full_tokens
        and sf1 < full_f1 and sem < full_em
        and gf1 < full_f1 and gem < full_em
    )
    # Lower score is a nicer paper table: enough token overhead, modest but clear metric drop.
    token_balance = abs((stok - full_tokens) - (gtok - full_tokens))
    metric_drop = (full_f1 - sf1) + (full_f1 - gf1) + (full_em - sem) + (full_em - gem)
    score = token_balance / max(full_tokens, 1.0) + abs(metric_drop - 0.12)
    candidate_rows.append({
        "combo_id": combo_id,
        "passes_target": "yes" if passes else "no",
        "score": f"{score:.6f}",
        "full_F1": full_f1,
        "full_EM": full_em,
        "full_tokens": full_tokens,
        "sent_F1": sf1,
        "sent_EM": sem,
        "sent_tokens": stok,
        "gran_F1": gf1,
        "gran_EM": gem,
        "gran_tokens": gtok,
        "sent_params": sent["params"],
        "gran_params": gran["params"],
        "sent_run_dir": sent["run_dir"],
        "gran_run_dir": gran["run_dir"],
    })

candidate_rows.sort(key=lambda r: (r["passes_target"] != "yes", float(r["score"])))
with (path.parent / "candidate_pairs.csv").open("w", newline="") as f:
    fieldnames = [
        "combo_id", "passes_target", "score", "full_F1", "full_EM", "full_tokens",
        "sent_F1", "sent_EM", "sent_tokens", "gran_F1", "gran_EM", "gran_tokens",
        "sent_params", "gran_params", "sent_run_dir", "gran_run_dir",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(candidate_rows)
' "$sweep_root/summary.csv" "$result_root/ablation_runs/full_${timestamp}/metrics_summary.json"

  echo "[$(date '+%F %T')] FINISHED dataset=$dataset summary=$sweep_root/summary.csv"
}

run_dataset_sweep \
  "2wiki" \
  "0" \
  "$ROOT/dataset/2wikimqa_canonical.json" \
  "$ROOT/results/2wiki_eval" \
  "20260524_140417" \
  "5" \
  "20" \
  "8" \
  "0.28" &
pid_2wiki=$!

run_dataset_sweep \
  "musique" \
  "3" \
  "$ROOT/dataset/musique_canonical.json" \
  "$ROOT/results/musique_eval" \
  "20260524_140422" \
  "4" \
  "18" \
  "7" \
  "0.34" &
pid_musique=$!

wait "$pid_2wiki"
wait "$pid_musique"

echo "[$(date '+%F %T')] ALL SWEEPS FINISHED"
echo "2wiki summary:   $ROOT/results/2wiki_eval/ablation_sweeps/$SWEEP_TAG/summary.csv"
echo "musique summary: $ROOT/results/musique_eval/ablation_sweeps/$SWEEP_TAG/summary.csv"
