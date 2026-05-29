import argparse
import csv
import importlib.util
import json
import pickle
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EVAL_SCRIPT = Path(__file__).with_name("eval.py")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

spec = importlib.util.spec_from_file_location("holorag_base_eval", EVAL_SCRIPT)
base_eval = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_eval)


GRANULARITY_KEYS = ("entity", "fact", "sentence", "chunk")
DEFAULT_SOURCE_INDEX_ROOTS = {
    "2wikimqa": REPO_ROOT / "results/2wiki_eval/shared_indexes/20260524_140417/full",
    "hotpotqa": REPO_ROOT / "results/hotpotqa_eval/shared_indexes/20260524_140426/full",
    "musique": REPO_ROOT / "results/musique_eval/shared_indexes/20260524_140422/full",
    "naturalquestions": REPO_ROOT / "results/naturalquestions_eval/shared_indexes/full",
}


def _source_name(sample: Dict[str, Any]) -> str:
    return str(sample.get("source_dataset") or sample.get("dataset") or "").strip()


def _candidate_source_ids(sample: Dict[str, Any]) -> List[str]:
    sample_id = str(sample.get("id", "")).strip()
    original_id = str(sample.get("original_id", "")).strip()
    source = _source_name(sample)
    candidates = [original_id]
    if sample_id:
        candidates.append(sample_id)
        prefix = f"{source}_"
        if source and sample_id.startswith(prefix):
            candidates.append(sample_id[len(prefix):])
    seen = set()
    ordered = []
    for value in candidates:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _parse_source_index_roots(values: Sequence[str]) -> Dict[str, Path]:
    roots = dict(DEFAULT_SOURCE_INDEX_ROOTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --source_index_root in source=/path form, got: {value}")
        source, path = value.split("=", 1)
        roots[source.strip()] = Path(path).expanduser().resolve()
    return roots


def _copy_index_dir(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return True


def prepopulate_shared_indexes(
    samples: Sequence[Dict[str, Any]],
    target_root: Path,
    source_roots: Dict[str, Path],
    logger,
) -> Dict[str, Any]:
    copied = 0
    reused_existing = 0
    missing_by_source: Dict[str, int] = defaultdict(int)
    copied_by_source: Dict[str, int] = defaultdict(int)

    for sample in samples:
        source = _source_name(sample)
        if source == "narrativeqa":
            missing_by_source[source] += 1
            continue
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            continue
        dst = target_root / sample_id
        if base_eval.resolve_index_path(dst).exists():
            reused_existing += 1
            continue
        source_root = source_roots.get(source)
        if not source_root:
            missing_by_source[source] += 1
            continue
        src = None
        for candidate_id in _candidate_source_ids(sample):
            candidate_dir = source_root / candidate_id
            if base_eval.resolve_index_path(candidate_dir).exists():
                src = candidate_dir
                break
        if src is None:
            missing_by_source[source] += 1
            continue
        if _copy_index_dir(src, dst):
            copied += 1
            copied_by_source[source] += 1

    summary = {
        "target_root": str(target_root),
        "copied": copied,
        "reused_existing": reused_existing,
        "copied_by_source": dict(sorted(copied_by_source.items())),
        "missing_by_source": dict(sorted(missing_by_source.items())),
        "source_roots": {key: str(path) for key, path in sorted(source_roots.items())},
    }
    logger.info("Shared index prepopulate summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def _empty_granularity_stats() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum": {key: 0.0 for key in GRANULARITY_KEYS},
        "avg": {key: 0.0 for key in GRANULARITY_KEYS},
    }


def _add_granularity(stats: Dict[str, Any], alpha: Dict[str, Any]) -> None:
    stats["count"] += 1
    for key in GRANULARITY_KEYS:
        stats["sum"][key] += float(alpha.get(key, 0.0) or 0.0)


def _finalize_granularity_stats(stats_by_source: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    finalized = {}
    for source, stats in sorted(stats_by_source.items()):
        count = max(1, int(stats["count"]))
        stats["avg"] = {key: float(stats["sum"][key] / count) for key in GRANULARITY_KEYS}
        finalized[source] = stats
    return finalized


def run_eval_with_granularity(
    rag,
    samples: Sequence[Dict[str, Any]],
    index_records: Sequence[Dict[str, Any]],
    run_dir: Path,
    dataset_tag: str,
    logger,
    token_counter,
) -> Dict[str, Any]:
    record_by_id = {str(item.get("sample_id", "")): item for item in index_records}
    per_example_path = run_dir / "per_example.jsonl"
    granularity_path = run_dir / "granularity_vectors.jsonl"
    per_example_path.write_text("", encoding="utf-8")
    granularity_path.write_text("", encoding="utf-8")

    em_scores: List[float] = []
    f1_scores: List[float] = []
    retrieval_latencies: List[float] = []
    retrieval_pipeline_latencies: List[float] = []
    qa_latencies: List[float] = []
    query_total_latencies: List[float] = []
    evidence_tokens: List[float] = []
    entity_counts: List[float] = []
    fact_counts: List[float] = []
    sentence_counts: List[float] = []
    chunk_counts: List[float] = []
    node_counts: List[float] = []
    edge_counts: List[float] = []
    edge_type_totals: Dict[str, List[float]] = {}
    granularity_by_source: Dict[str, Dict[str, Any]] = defaultdict(_empty_granularity_stats)
    granularity_overall = _empty_granularity_stats()

    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        source = _source_name(sample)
        if not question:
            continue
        index_record = record_by_id.get(sample_id)
        if not index_record or not index_record.get("valid"):
            logger.warning("[%s][%d/%d] skip missing index: %s", dataset_tag, i, len(samples), sample_id)
            continue
        index_path = Path(str(index_record.get("index_path", "")))
        if not index_path.exists():
            logger.warning("[%s][%d/%d] skip invalid index path: %s", dataset_tag, i, len(samples), index_path)
            continue

        with index_path.open("rb") as handle:
            rag.state = pickle.load(handle)
        rag.index_path = str(index_path)

        t_query = time.perf_counter()
        result = rag.query(question)
        query_elapsed = time.perf_counter() - t_query
        query_timing = result.get("query_timing", {}) or {}
        query_total_latencies.append(query_elapsed)
        retrieval_latencies.append(float(query_timing.get("retrieval_latency", query_elapsed)))
        if "retrieval_pipeline_latency" in query_timing:
            retrieval_pipeline_latencies.append(float(query_timing.get("retrieval_pipeline_latency", 0.0)))
        if "qa_latency" in query_timing:
            qa_latencies.append(float(query_timing.get("qa_latency", 0.0)))

        alpha = {
            key: float((result.get("alpha", {}) or {}).get(key, 0.0) or 0.0)
            for key in GRANULARITY_KEYS
        }
        _add_granularity(granularity_by_source[source or "unknown"], alpha)
        _add_granularity(granularity_overall, alpha)

        ranked_passages = result.get("ranked_passages", []) or []
        qa_messages = result.get("qa_messages", []) or []
        evidence_text = ""
        if len(qa_messages) >= 2 and isinstance(qa_messages[1], dict):
            evidence_text = str(qa_messages[1].get("content", ""))
        if not evidence_text:
            evidence_text = base_eval.format_qa_evidence_from_ranked_passages(ranked_passages, rag.config.qa_passage_top_k)
        final_evidence_tokens = token_counter.count(evidence_text)
        evidence_tokens.append(float(final_evidence_tokens))
        predicted_answer = str(result.get("predicted_answer", "")).strip()
        em, f1 = base_eval.best_em_f1(base_eval.build_gold_answers(sample), predicted_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        stats = index_record.get("stats", {}) or {}
        layers = stats.get("layer_counts", {}) or {}
        entity_counts.append(float(layers.get("entity", 0)))
        fact_counts.append(float(layers.get("fact", 0)))
        sentence_counts.append(float(layers.get("sentence", 0)))
        chunk_counts.append(float(layers.get("chunk", 0)))
        node_counts.append(float(stats.get("nodes", 0)))
        edge_counts.append(float(stats.get("edges", 0)))
        edge_type_counts = stats.get("edge_type_counts", {}) or {}
        for edge_type, count in edge_type_counts.items():
            edge_type_totals.setdefault(str(edge_type), []).append(float(count))

        granularity_row = {
            "query_id": sample_id,
            "source_dataset": source,
            "question": question,
            "task_profile": result.get("task_profile", ""),
            "granularity_vector": alpha,
            "intent_confidence": result.get("intent_confidence"),
        }
        row = {
            "query_id": sample_id,
            "source_dataset": source,
            "original_id": sample.get("original_id", ""),
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "predicted_answer": predicted_answer,
            "F1": f1,
            "EM": em,
            "task_profile": result.get("task_profile", ""),
            "granularity_vector": alpha,
            "intent_confidence": result.get("intent_confidence"),
            "index_path": str(index_path),
            "index_latency": float(index_record.get("index_latency", 0.0) or 0.0),
            "index_reused": bool(index_record.get("reused", False)),
            "query_total_latency": query_elapsed,
            "retrieval_latency": query_timing.get("retrieval_latency"),
            "retrieval_pipeline_latency": query_timing.get("retrieval_pipeline_latency"),
            "qa_latency": query_timing.get("qa_latency"),
            "query_timing": query_timing,
            "entity_nodes": int(layers.get("entity", 0)),
            "fact_nodes": int(layers.get("fact", 0)),
            "sentence_nodes": int(layers.get("sentence", 0)),
            "chunk_nodes": int(layers.get("chunk", 0)),
            "nodes": int(stats.get("nodes", 0)),
            "edges": int(stats.get("edges", 0)),
            "edge_type_counts": {str(key): int(value) for key, value in edge_type_counts.items()},
            "final_evidence_tokens": int(final_evidence_tokens),
            "final_evidence_tokenizer": token_counter.method,
            "qa_answer_mode": result.get("qa_answer_mode", ""),
            "applied_source_base_config": False,
        }
        with granularity_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(granularity_row, ensure_ascii=False) + "\n")
        with per_example_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(
            "[%s][%d/%d] %s | source=%s | alpha=%s | f1=%.4f em=%.4f | running_f1=%.4f | q=%.3fs",
            dataset_tag,
            i,
            len(samples),
            sample_id,
            source,
            ",".join(f"{key}:{alpha[key]:.3f}" for key in GRANULARITY_KEYS),
            f1,
            em,
            base_eval._avg(f1_scores),
            query_elapsed,
        )

    granularity_summary = {
        "overall": _finalize_granularity_stats({"overall": granularity_overall})["overall"],
        "by_source_dataset": _finalize_granularity_stats(granularity_by_source),
    }
    (run_dir / "granularity_summary.json").write_text(
        json.dumps(granularity_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "variant": "multiGranularityQA_holorag",
        "num_queries": len(f1_scores),
        "F1": base_eval._avg(f1_scores),
        "EM": base_eval._avg(em_scores),
        "qa_failures": 0,
        "retrieval_latency": base_eval._avg(retrieval_latencies),
        "retrieval_pipeline_latency": base_eval._avg(retrieval_pipeline_latencies),
        "qa_latency": base_eval._avg(qa_latencies),
        "retrieval_qa_latency": base_eval._avg(query_total_latencies),
        "query_runtime": time.perf_counter() - t_start,
        "nodes": base_eval._avg(node_counts),
        "entity_nodes": base_eval._avg(entity_counts),
        "fact_nodes": base_eval._avg(fact_counts),
        "sentence_nodes": base_eval._avg(sentence_counts),
        "chunk_nodes": base_eval._avg(chunk_counts),
        "edges": base_eval._avg(edge_counts),
        "final_evidence_tokens": base_eval._avg(evidence_tokens),
        "final_evidence_tokenizer": token_counter.method,
        "granularity_summary": granularity_summary,
    }
    for edge_type, values in sorted(edge_type_totals.items()):
        metrics[f"edge_{edge_type}"] = base_eval._avg(values)
    for source, stats in granularity_summary["by_source_dataset"].items():
        for key, value in stats["avg"].items():
            metrics[f"granularity_{source}_{key}"] = value
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified MultiGranularityQA HoloRAG eval and record per-query granularity vectors.")
    parser.add_argument("--dataset_file", type=str, default=str(REPO_ROOT / "dataset/MultiGranularityQA.json"))
    parser.add_argument("--dataset_format", type=str, default="canonical_json", choices=["auto", "musique_jsonl", "canonical_jsonl", "2wiki_json", "canonical_json"])
    parser.add_argument("--dataset_name", type=str, default="multiGranularityQA")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_queries", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "results/multiGranularityQA_eval"))
    parser.add_argument("--shared_index_root", type=str, default=str(REPO_ROOT / "results/multiGranularityQA_eval/shared_indexes/full"))
    parser.add_argument("--run_name", type=str, default="full")
    parser.add_argument("--source_index_root", action="append", default=[], help="Optional source=/path override for prepopulating non-narrativeqa indexes.")
    parser.add_argument("--skip_prepopulate", action="store_true")
    parser.add_argument("--recompute_only", action="store_true", help="Use existing shared indexes only; do not build missing indexes.")

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:5")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--chunk_size_words", type=int, default=256)
    parser.add_argument("--chunk_overlap_words", type=int, default=64)
    parser.add_argument("--spacy_model_name", type=str, default="en_core_web_sm")
    parser.add_argument("--disable_paragraph_as_chunk", action="store_true", default=True, help="Default true here so newly built indexes use 256/64 word chunks.")
    parser.add_argument("--use_paragraph_as_chunk", action="store_false", dest="disable_paragraph_as_chunk", help="Use each paragraph as a chunk for newly built indexes.")
    parser.add_argument("--index_extraction_mode", type=str, default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--disable_intent_llm", action="store_true", help="Disable LLM-predicted granularity vectors and use heuristic fallback.")
    parser.add_argument("--disable_entity_similarity_edges", action="store_true")
    parser.add_argument("--entity_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--entity_similarity_top_k", type=int, default=2047)
    parser.add_argument("--disable_sentence_layer", action="store_true")
    parser.add_argument("--disable_granularity_awareness", action="store_true")
    parser.add_argument("--disable_granularity_pagerank_bias", action="store_true")
    parser.add_argument("--topk_passages", type=int, default=4)
    parser.add_argument("--passage_output_top_k", type=int, default=10)
    parser.add_argument("--qa_max_input_tokens", type=int, default=7000)
    parser.add_argument("--qa_evidence_token_budget", type=int, default=820)
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
    parser.add_argument("--task_profile", type=str, default="auto", choices=["auto", "single_hop", "multi_hop", "long_context"])
    parser.add_argument("--skip_llm_health_check", action="store_true")
    parser.add_argument("--llm_health_timeout", type=float, default=10.0)
    args = parser.parse_args()
    args.intent_use_llm = not args.disable_intent_llm
    args.ablation_name = ""
    args.source_base_config = False
    return args


def write_metrics(run_dir: Path, metrics: Dict[str, Any], index_summary: Dict[str, Any]) -> None:
    metrics["index_latency"] = float(index_summary.get("avg_index_latency", 0.0))
    metrics["shared_index_runtime"] = float(index_summary.get("total_index_runtime", 0.0))
    metrics["total_runtime"] = metrics["query_runtime"] + metrics["shared_index_runtime"]
    metrics["index_pool"] = "shared"
    metric_headers = [
        "variant",
        "num_queries",
        "F1",
        "EM",
        "qa_failures",
        "index_latency",
        "retrieval_latency",
        "retrieval_pipeline_latency",
        "qa_latency",
        "retrieval_qa_latency",
        "query_runtime",
        "shared_index_runtime",
        "total_runtime",
        "final_evidence_tokens",
        "final_evidence_tokenizer",
        "index_pool",
        "nodes",
        "entity_nodes",
        "fact_nodes",
        "sentence_nodes",
        "chunk_nodes",
        "edges",
    ]
    metric_headers.extend(sorted(key for key in metrics if key.startswith("edge_") or key.startswith("granularity_")))
    csv_metrics = {key: metrics.get(key, "") for key in metric_headers}
    (run_dir / "metrics_summary.json").write_text(
        json.dumps([metrics], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_headers)
        writer.writeheader()
        writer.writerow(csv_metrics)


def write_config(run_dir: Path, args: argparse.Namespace, dataset_format: str, shared_index_root: Path, prepopulate_summary: Dict[str, Any]) -> None:
    payload = {
        "dataset_file": args.dataset_file,
        "dataset_format": dataset_format,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "seed": args.seed,
        "num_eval_queries": args.num_eval_queries,
        "llm_base_url": args.llm_base_url,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "embedding_device": args.embedding_device,
        "spacy_model_name": args.spacy_model_name,
        "task_profile": args.task_profile,
        "source_base_config": False,
        "use_paragraph_as_chunk": not args.disable_paragraph_as_chunk,
        "index_extraction_mode": args.index_extraction_mode,
        "intent_use_llm": args.intent_use_llm,
        "enable_entity_similarity_edges": not args.disable_entity_similarity_edges,
        "entity_similarity_threshold": args.entity_similarity_threshold,
        "entity_similarity_top_k": args.entity_similarity_top_k,
        "enable_granularity_awareness": not args.disable_granularity_awareness,
        "enable_sentence_layer": not args.disable_sentence_layer,
        "enable_granularity_pagerank_bias": not args.disable_granularity_pagerank_bias,
        "chunk_size_words": args.chunk_size_words,
        "chunk_overlap_words": args.chunk_overlap_words,
        "topk_passages": args.topk_passages,
        "qa_max_input_tokens": args.qa_max_input_tokens,
        "qa_evidence_token_budget": args.qa_evidence_token_budget,
        "fact_rerank_use_llm": args.fact_rerank_use_llm,
        "fact_rerank_llm_candidate_k": args.fact_rerank_llm_candidate_k,
        "fact_rerank_llm_keep_k": args.fact_rerank_llm_keep_k,
        "enable_fact_source_first_evidence": args.enable_fact_source_first_evidence,
        "enable_fact_chunk_boost": args.enable_fact_chunk_boost,
        "fact_chunk_boost": args.fact_chunk_boost,
        "enable_fair_sentence_context": args.enable_fair_sentence_context,
        "evidence_extra_ranked_sentence_k": args.evidence_extra_ranked_sentence_k,
        "evidence_max_sentences": args.evidence_max_sentences,
        "evidence_title_limit": args.evidence_title_limit,
        "evidence_passage_context_k": args.evidence_passage_context_k,
        "evidence_passage_excerpt_tokens": args.evidence_passage_excerpt_tokens,
        "shared_index_root": str(shared_index_root),
        "prepopulate_summary": prepopulate_summary,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_eval.set_global_seed(args.seed)

    from holorag import HoloRAG

    dataset_format = base_eval.detect_dataset_format(args.dataset_file, args.dataset_format)
    all_samples = base_eval.load_samples(args.dataset_file, dataset_format)
    filtered = base_eval.maybe_filter_split(all_samples, args.split)
    if not filtered:
        raise ValueError("No samples available after split filtering.")

    run_dir = Path(args.output_dir).expanduser().resolve() / (args.run_name.strip() or "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = base_eval.setup_logger(run_dir / "run.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Unified config: task_profile=%s intent_use_llm=%s chunk=%d/%d source_base_config=False",
                args.task_profile, args.intent_use_llm, args.chunk_size_words, args.chunk_overlap_words)
    if not args.skip_llm_health_check:
        base_eval.check_llm_server(args.llm_base_url, args.llm_name, args.llm_health_timeout, logger)

    samples = base_eval.sample_queries(filtered, run_dir / "sampled_queries.json", args.seed, args.num_eval_queries)
    shared_index_root = Path(args.shared_index_root).expanduser().resolve()
    shared_index_root.mkdir(parents=True, exist_ok=True)

    prepopulate_summary: Dict[str, Any] = {}
    if not args.skip_prepopulate:
        prepopulate_summary = prepopulate_shared_indexes(
            samples,
            shared_index_root,
            _parse_source_index_roots(args.source_index_root),
            logger,
        )

    rag = HoloRAG(base_eval.build_config(args, save_dir=str(run_dir / "workdir")))
    token_counter = base_eval.TokenCounter(args.llm_name, logger)
    if args.recompute_only:
        index_data = base_eval.records_from_shared_indexes(samples, shared_index_root)
    else:
        index_data = base_eval.prebuild_or_reuse_indexes(rag, samples, shared_index_root, logger)

    (run_dir / "shared_index_records.json").write_text(
        json.dumps(index_data["records"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "shared_index_summary.json").write_text(
        json.dumps(index_data["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = run_eval_with_granularity(
        rag,
        samples,
        index_data["records"],
        run_dir,
        args.dataset_name,
        logger,
        token_counter,
    )
    write_metrics(run_dir, metrics, index_data["summary"])
    write_config(run_dir, args, dataset_format, shared_index_root, prepopulate_summary)
    logger.info("Saved granularity vectors: %s", run_dir / "granularity_vectors.jsonl")
    logger.info("Saved granularity summary: %s", run_dir / "granularity_summary.json")
    logger.info("Run complete | F1=%.4f EM=%.4f N=%d", metrics["F1"], metrics["EM"], metrics["num_queries"])


if __name__ == "__main__":
    main()
