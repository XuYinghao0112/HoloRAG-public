#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/data/xyh/code/HoloRAG
PY=/home/xyh/miniconda3/envs/holorag-dev/bin/python
SCRIPT="$REPO_ROOT/scripts/run_ablation_eval_naive.py"
LLM=http://127.0.0.1:8000/v1
MODEL=/data/xyh/models/Qwen2.5-72B-Instruct
EMB=/data/xyh/models/NV-Embed-v2

COMMON_ARGS=(
  --dataset_format canonical_json
  --split dev
  --seed 18
  --num_eval_queries 200
  --llm_base_url "$LLM"
  --llm_name "$MODEL"
  --embedding_name "$EMB"
  --spacy_model_name en_core_web_sm
  --task_profile multi_hop
  --disable_paragraph_as_chunk
  --index_extraction_mode heuristic
  --entity_similarity_threshold 0.8
  --entity_similarity_top_k 2047
  --topk_passages 5
  --qa_max_input_tokens 7000
  --qa_evidence_token_budget 760
  --fact_rerank_use_llm
  --fact_rerank_llm_candidate_k 16
  --enable_fact_source_first_evidence
  --enable_fact_chunk_boost
  --fact_chunk_boost 0.3
  --enable_fair_sentence_context
  --evidence_extra_ranked_sentence_k 4
  --evidence_max_sentences 15
  --evidence_title_limit 3
  --evidence_passage_context_k 1
  --evidence_passage_excerpt_tokens 110
  --recompute_only
)

run_one() {
  local dataset_file=$1
  local output_dir=$2
  local shared_index_root=$3
  local gpu=$4
  local run_name=$5
  shift 5

  "$PY" "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --dataset_file "$dataset_file" \
    --output_dir "$output_dir" \
    --shared_index_root "$shared_index_root" \
    --embedding_device "cuda:$gpu" \
    --run_name "$run_name" \
    "$@"
}

run_2wiki() {
  cd "$REPO_ROOT"
  local dataset="$REPO_ROOT/dataset/2wikimqa_canonical.json"
  local out="$REPO_ROOT/outputs/2wiki_eval/ablation_runs"
  local shared="$REPO_ROOT/outputs/2wiki_eval/shared_indexes_v1"

  echo "[$(date '+%F %T')] start 2wiki v8a on gpu0"
  run_one "$dataset" "$out" "$shared" 0 v8a_hippo_ppr_2wiki_gpu0 \
    --fact_rerank_llm_keep_k 7 \
    --ppr_seed_mode hippo_entity_passage \
    --enable_entity_occurrence_penalty

  echo "[$(date '+%F %T')] start 2wiki v8b on gpu0"
  run_one "$dataset" "$out" "$shared" 0 v8b_chain_factfilter_2wiki_gpu0 \
    --fact_rerank_llm_keep_k 5 \
    --fact_rerank_prompt_mode chain

  echo "[$(date '+%F %T')] start 2wiki v8c on gpu0"
  run_one "$dataset" "$out" "$shared" 0 v8c_chain_evidence_2wiki_gpu0 \
    --fact_rerank_llm_keep_k 5 \
    --fact_rerank_prompt_mode chain \
    --evidence_selection_mode chain \
    --evidence_max_sentences 14 \
    --evidence_extra_ranked_sentence_k 0 \
    --chain_evidence_per_subquestion 2 \
    --chain_evidence_extra_k 3

  echo "[$(date '+%F %T')] done 2wiki"
}

run_musique() {
  cd "$REPO_ROOT"
  local dataset="$REPO_ROOT/dataset/musique_canonical.json"
  local out="$REPO_ROOT/outputs/musique_eval/ablation_runs"
  local shared="$REPO_ROOT/outputs/musique_eval/shared_indexes_v1"

  echo "[$(date '+%F %T')] start musique v8a on gpu4"
  run_one "$dataset" "$out" "$shared" 4 v8a_hippo_ppr_musique_gpu4 \
    --fact_rerank_llm_keep_k 7 \
    --ppr_seed_mode hippo_entity_passage \
    --enable_entity_occurrence_penalty

  echo "[$(date '+%F %T')] start musique v8b on gpu4"
  run_one "$dataset" "$out" "$shared" 4 v8b_chain_factfilter_musique_gpu4 \
    --fact_rerank_llm_keep_k 5 \
    --fact_rerank_prompt_mode chain

  echo "[$(date '+%F %T')] start musique v8c on gpu4"
  run_one "$dataset" "$out" "$shared" 4 v8c_chain_evidence_musique_gpu4 \
    --fact_rerank_llm_keep_k 5 \
    --fact_rerank_prompt_mode chain \
    --evidence_selection_mode chain \
    --evidence_max_sentences 14 \
    --evidence_extra_ranked_sentence_k 0 \
    --chain_evidence_per_subquestion 2 \
    --chain_evidence_extra_k 3

  echo "[$(date '+%F %T')] done musique"
}

case "${1:-all}" in
  2wiki)
    run_2wiki
    ;;
  musique)
    run_musique
    ;;
  all)
    run_2wiki &
    pid_2wiki=$!
    run_musique &
    pid_musique=$!
    wait "$pid_2wiki"
    wait "$pid_musique"
    ;;
  *)
    echo "Usage: $0 [all|2wiki|musique]" >&2
    exit 2
    ;;
esac
