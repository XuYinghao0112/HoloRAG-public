import argparse
import csv
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval as eval_base  # noqa: E402


def _candidate_ids(result: Dict[str, Any]) -> Dict[str, List[str]]:
    channels = result.get("channel_scores", {}) or {}
    return {
        channel: [str(item.get("id", "")) for item in rows if item.get("id")]
        for channel, rows in channels.items()
        if isinstance(rows, list)
    }


def _final_evidence_ids(result: Dict[str, Any]) -> List[str]:
    evidence = result.get("ranked_evidence", {}) or {}
    ids: List[str] = []
    for key in ("facts", "sentences", "chunks"):
        for item in evidence.get(key, []) or []:
            value = item.get("id") or item.get("node_id") or item.get("fact_id") or item.get("sentence_id") or item.get("chunk_id")
            if value:
                ids.append(str(value))
    return ids


def build_config_args(args: argparse.Namespace, execution_mode: str, save_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_dtype=args.embedding_dtype,
        task_profile=args.task_profile,
        disable_paragraph_as_chunk=args.disable_paragraph_as_chunk,
        index_extraction_mode=args.index_extraction_mode,
        qa_max_input_tokens=args.qa_max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        intent_use_llm=args.intent_use_llm,
        disable_entity_similarity_edges=args.disable_entity_similarity_edges,
        entity_similarity_threshold=args.entity_similarity_threshold,
        entity_similarity_top_k=args.entity_similarity_top_k,
        disable_granularity_awareness=args.disable_granularity_awareness,
        disable_sentence_layer=args.disable_sentence_layer,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        spacy_model_name=args.spacy_model_name,
        topk_passages=args.topk_passages,
        passage_output_top_k=args.passage_output_top_k,
        disable_granularity_pagerank_bias=args.disable_granularity_pagerank_bias,
        fact_rerank_use_llm=args.fact_rerank_use_llm,
        fact_rerank_llm_candidate_k=args.fact_rerank_llm_candidate_k,
        fact_rerank_llm_keep_k=args.fact_rerank_llm_keep_k,
        enable_fact_source_first_evidence=args.enable_fact_source_first_evidence,
        enable_fact_chunk_boost=args.enable_fact_chunk_boost,
        fact_chunk_boost=args.fact_chunk_boost,
        enable_fair_sentence_context=args.enable_fair_sentence_context,
        evidence_extra_ranked_sentence_k=args.evidence_extra_ranked_sentence_k,
        evidence_max_sentences=args.evidence_max_sentences,
        evidence_title_limit=args.evidence_title_limit,
        evidence_passage_context_k=args.evidence_passage_context_k,
        evidence_passage_excerpt_tokens=args.evidence_passage_excerpt_tokens,
        evidence_chunk_max_tokens=args.evidence_chunk_max_tokens,
        evidence_packing_mode=args.evidence_packing_mode,
        evidence_alpha_total_units=args.evidence_alpha_total_units,
        evidence_alpha_uniform_mix=args.evidence_alpha_uniform_mix,
        evidence_soft_token_budget=args.evidence_soft_token_budget,
        evidence_allow_underfill=args.evidence_allow_underfill,
        evidence_min_score=args.evidence_min_score,
        evidence_redundancy_threshold=args.evidence_redundancy_threshold,
        disable_evidence_alpha_weights=args.disable_evidence_alpha_weights,
        execution_mode=execution_mode,
        num_workers=args.num_workers,
        multi_worker_embedding_devices=args.multi_worker_embedding_devices,
        save_dir=str(save_dir),
    )


def run_mode(
    rag,
    mode: str,
    args: argparse.Namespace,
    samples: Sequence[Dict[str, Any]],
    index_records: Sequence[Dict[str, Any]],
    run_dir: Path,
    token_counter: eval_base.TokenCounter,
    logger,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rag.config.execution_mode = mode
    rag.config.num_workers = args.num_workers
    rag.config.multi_worker_embedding_devices = args.multi_worker_embedding_devices
    rag.retriever.config.execution_mode = mode
    rag.retriever.config.num_workers = args.num_workers
    rag.retriever.config.multi_worker_embedding_devices = args.multi_worker_embedding_devices
    record_by_id = {str(item.get("sample_id", "")): item for item in index_records}
    rows: List[Dict[str, Any]] = []
    per_example_path = run_dir / f"per_example_{mode}.jsonl"
    per_example_path.write_text("", encoding="utf-8")

    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", ""))
        record = record_by_id.get(sample_id)
        if not record or not record.get("valid"):
            raise FileNotFoundError(f"Missing valid cached index for sample {sample_id!r} under --shared_index_root.")
        index_path = Path(str(record.get("index_path", "")))
        with index_path.open("rb") as handle:
            rag.state = pickle.load(handle)
        rag.index_path = str(index_path)

        t0 = time.perf_counter()
        result = rag.query(str(sample.get("question", "")).strip())
        total_latency = time.perf_counter() - t0
        timing = result.get("query_timing", {}) or {}

        evidence_text = ""
        messages = result.get("qa_messages", []) or []
        if len(messages) >= 2 and isinstance(messages[1], dict):
            evidence_text = str(messages[1].get("content", ""))
        if not evidence_text:
            evidence_text = eval_base.format_qa_evidence_from_ranked_passages(result.get("ranked_passages", []) or [], rag.config.qa_passage_top_k)
        evidence_values = eval_base._evidence_granularity_row(result, token_counter.count(evidence_text))
        predicted = str(result.get("predicted_answer", "")).strip()
        em, f1 = eval_base.best_em_f1(eval_base.build_gold_answers(sample), predicted)

        row = {
            "execution_mode": mode,
            "query_id": sample_id,
            "question": sample.get("question", ""),
            "F1": f1,
            "EM": em,
            "retrieval_latency": float(timing.get("retrieval_latency", 0.0) or 0.0),
            "qa_latency": float(timing.get("qa_latency", 0.0) or 0.0),
            "total_latency": total_latency,
            "final_evidence_tokens": int(evidence_values["final_evidence_tokens"]),
            "candidate_ids": _candidate_ids(result),
            "final_evidence_ids": _final_evidence_ids(result),
            "index_path": str(index_path),
        }
        rows.append(row)
        with per_example_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[%s][%d/%d] %s | retrieval=%.3fs qa=%.3fs total=%.3fs f1=%.4f em=%.4f",
            mode,
            i,
            len(samples),
            sample_id,
            row["retrieval_latency"],
            row["qa_latency"],
            row["total_latency"],
            f1,
            em,
        )

    metrics = {
        "execution_mode": mode,
        "num_queries": len(rows),
        "F1": eval_base._avg([float(row["F1"]) for row in rows]),
        "EM": eval_base._avg([float(row["EM"]) for row in rows]),
        "retrieval_latency": eval_base._avg([float(row["retrieval_latency"]) for row in rows]),
        "qa_latency": eval_base._avg([float(row["qa_latency"]) for row in rows]),
        "total_latency": eval_base._avg([float(row["total_latency"]) for row in rows]),
        "avg_retrieval_latency": eval_base._avg([float(row["retrieval_latency"]) for row in rows]),
        "avg_total_latency": eval_base._avg([float(row["total_latency"]) for row in rows]),
        "final_evidence_tokens": eval_base._avg([float(row["final_evidence_tokens"]) for row in rows]),
        "final_evidence_tokenizer": token_counter.method,
    }
    return metrics, rows


def compare_rows(sequential: Sequence[Dict[str, Any]], multi_worker: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    multi_by_id = {str(row.get("query_id", "")): row for row in multi_worker}
    comparisons = []
    for seq_row in sequential:
        query_id = str(seq_row.get("query_id", ""))
        mw_row = multi_by_id.get(query_id, {})
        comparisons.append(
            {
                "query_id": query_id,
                "same_F1": float(seq_row.get("F1", 0.0)) == float(mw_row.get("F1", -1.0)),
                "same_EM": float(seq_row.get("EM", 0.0)) == float(mw_row.get("EM", -1.0)),
                "same_final_evidence_tokens": int(seq_row.get("final_evidence_tokens", -1)) == int(mw_row.get("final_evidence_tokens", -2)),
                "same_final_evidence_ids": seq_row.get("final_evidence_ids") == mw_row.get("final_evidence_ids"),
                "same_candidate_ids": seq_row.get("candidate_ids") == mw_row.get("candidate_ids"),
            }
        )
    return {
        "num_compared": len(comparisons),
        "same_F1_count": sum(1 for item in comparisons if item["same_F1"]),
        "same_EM_count": sum(1 for item in comparisons if item["same_EM"]),
        "same_final_evidence_tokens_count": sum(1 for item in comparisons if item["same_final_evidence_tokens"]),
        "same_final_evidence_ids_count": sum(1 for item in comparisons if item["same_final_evidence_ids"]),
        "same_candidate_ids_count": sum(1 for item in comparisons if item["same_candidate_ids"]),
        "rows": comparisons,
    }


def write_summary(run_dir: Path, metrics_rows: Sequence[Dict[str, Any]], comparison: Dict[str, Any]) -> None:
    headers: List[str] = []
    for row in metrics_rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    (run_dir / "metrics_summary.json").write_text(json.dumps(list(metrics_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
    (run_dir / "mode_consistency.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HoloRAG sequential vs multi_worker online retrieval latency using cached indexes.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "musique_jsonl", "canonical_jsonl", "2wiki_json", "canonical_json"])
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shared_index_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--chunk_size_words", type=int, default=256)
    parser.add_argument("--chunk_overlap_words", type=int, default=64)
    parser.add_argument("--spacy_model_name", type=str, default="en_core_web_sm")
    parser.add_argument("--disable_paragraph_as_chunk", action="store_true")
    parser.add_argument("--index_extraction_mode", type=str, default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--intent_use_llm", action="store_true")
    parser.add_argument("--disable_entity_similarity_edges", action="store_true")
    parser.add_argument("--entity_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--entity_similarity_top_k", type=int, default=2047)
    parser.add_argument("--disable_sentence_layer", action="store_true")
    parser.add_argument("--disable_granularity_awareness", action="store_true")
    parser.add_argument("--disable_granularity_pagerank_bias", action="store_true")
    parser.add_argument("--topk_passages", type=int, default=4)
    parser.add_argument("--passage_output_top_k", type=int, default=10)
    parser.add_argument("--qa_max_input_tokens", type=int, default=7000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--fact_rerank_use_llm", action="store_true", default=True)
    parser.add_argument("--fact_rerank_llm_candidate_k", type=int, default=20)
    parser.add_argument("--fact_rerank_llm_keep_k", type=int, default=7)
    parser.add_argument("--enable_fact_source_first_evidence", action="store_true", default=True)
    parser.add_argument("--enable_fact_chunk_boost", action="store_true", default=True)
    parser.add_argument("--fact_chunk_boost", type=float, default=0.4)
    parser.add_argument("--enable_fair_sentence_context", action="store_true", default=True)
    parser.add_argument("--evidence_extra_ranked_sentence_k", type=int, default=3)
    parser.add_argument("--evidence_max_sentences", type=int, default=15)
    parser.add_argument("--evidence_title_limit", type=int, default=3)
    parser.add_argument("--evidence_passage_context_k", type=int, default=1)
    parser.add_argument("--evidence_passage_excerpt_tokens", type=int, default=100)
    parser.add_argument("--evidence_chunk_max_tokens", type=int, default=256)
    parser.add_argument("--evidence_packing_mode", type=str, default="alpha_count")
    parser.add_argument("--evidence_alpha_total_units", type=int, default=20)
    parser.add_argument("--evidence_alpha_uniform_mix", type=float, default=0.0)
    parser.add_argument("--evidence_soft_token_budget", type=int, default=0)
    parser.add_argument("--evidence_allow_underfill", action="store_true", default=True)
    parser.add_argument("--evidence_min_score", type=float, default=0.0)
    parser.add_argument("--evidence_redundancy_threshold", type=float, default=0.85)
    parser.add_argument("--disable_evidence_alpha_weights", action="store_true")
    parser.add_argument("--task_profile", type=str, default="multi_hop", choices=["auto", "single_hop", "multi_hop", "long_context"])
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--multi_worker_embedding_devices", type=str, default="", help="Comma-separated devices for optional multi-worker retrieval encoders, e.g. cuda:0,cuda:3.")
    parser.add_argument("--skip_llm_health_check", action="store_true")
    parser.add_argument("--llm_health_timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_base.set_global_seed(args.seed)
    run_name = args.run_name.strip() or f"multi_worker_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = eval_base.setup_logger(run_dir / "run.log")
    if not args.skip_llm_health_check:
        eval_base.check_llm_server(args.llm_base_url, args.llm_name, args.llm_health_timeout, logger)

    dataset_format = eval_base.detect_dataset_format(args.dataset_file, args.dataset_format)
    all_samples = eval_base.load_samples(args.dataset_file, dataset_format)
    filtered = eval_base.maybe_filter_split(all_samples, args.split)
    candidate_samples = filtered or all_samples
    sample_limit = len(candidate_samples) if args.limit <= 0 else args.limit
    samples = eval_base.sample_queries(candidate_samples, run_dir / "sampled_queries.json", args.seed, sample_limit)
    records = eval_base.records_from_shared_indexes(samples, Path(args.shared_index_root).expanduser().resolve())["records"]
    token_counter = eval_base.TokenCounter(args.llm_name, logger)
    from holorag import HoloRAG

    rag = HoloRAG(eval_base.build_config(build_config_args(args, "sequential", run_dir / "workdir"), save_dir=str(run_dir / "workdir")))

    metrics_rows: List[Dict[str, Any]] = []
    rows_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for mode in ("sequential", "multi_worker"):
        logger.info("Running %s mode on %d cached queries", mode, len(samples))
        metrics, rows = run_mode(rag, mode, args, samples, records, run_dir, token_counter, logger)
        metrics_rows.append(metrics)
        rows_by_mode[mode] = rows

    comparison = compare_rows(rows_by_mode["sequential"], rows_by_mode["multi_worker"])
    write_summary(run_dir, metrics_rows, comparison)
    logger.info("Saved latency comparison to %s", run_dir / "metrics_summary.json")
    logger.info("Consistency summary: %s", json.dumps({k: v for k, v in comparison.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
