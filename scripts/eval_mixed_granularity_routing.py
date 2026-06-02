"""Evaluate query-level granularity routing on MultiGranularityQA.

Compares a no-routing Uniform Alpha baseline against LLM Query Routing while
reusing the same chunking and graph indexes for both modes.

Example:
  python code/HoloRAG/scripts/eval_mixed_granularity_routing.py --modes uniform,llm --chunk_size 256 --chunk_overlap 64
"""

import argparse
import csv
import gc
import json
import logging
import pickle
import random
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


UNIFORM_ALPHA = {"fact": 1.0 / 3.0, "sentence": 1.0 / 3.0, "chunk": 1.0 / 3.0}
SUMMARY_KEYS = [
    "mode",
    "variant",
    "num_queries",
    "num_skipped",
    "F1",
    "EM",
    "qa_failures",
    "index_latency",
    "retrieval_latency",
    "qa_latency",
    "total_latency",
    "retrieval_qa_latency",
    "query_runtime",
    "shared_index_runtime",
    "total_runtime",
    "final_evidence_tokens",
    "final_evidence_tokenizer",
    "avg_alpha_fact",
    "avg_alpha_sentence",
    "avg_alpha_chunk",
]


def resolve_path(path_value: str, *, prefer_repo_results: bool = False) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if prefer_repo_results and path.parts and path.parts[0] == "results":
        return (REPO_ROOT / path).resolve()
    if path.parts[:2] == ("code", "HoloRAG"):
        return (REPO_ROOT.parents[1] / path).resolve()
    return path.resolve()


def build_eval_args(args: argparse.Namespace, save_dir: Path, intent_use_llm: bool) -> argparse.Namespace:
    return argparse.Namespace(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_dtype=args.embedding_dtype,
        task_profile="auto",
        disable_paragraph_as_chunk=True,
        index_extraction_mode=args.index_extraction_mode,
        qa_max_input_tokens=args.qa_max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        intent_use_llm=intent_use_llm,
        disable_entity_similarity_edges=False,
        entity_similarity_threshold=args.entity_similarity_threshold,
        entity_similarity_top_k=args.entity_similarity_top_k,
        disable_granularity_awareness=False,
        disable_sentence_layer=False,
        chunk_size_words=args.chunk_size,
        chunk_overlap_words=args.chunk_overlap,
        spacy_model_name=args.spacy_model_name,
        topk_passages=args.topk_passages,
        passage_output_top_k=args.passage_output_top_k,
        disable_granularity_pagerank_bias=False,
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
        disable_evidence_alpha_weights=False,
        save_dir=str(save_dir),
    )


def load_multigranularity_samples(dataset_path: Path, limit: int) -> List[Dict[str, Any]]:
    dataset_format = eval_base.detect_dataset_format(str(dataset_path), "auto")
    samples = eval_base.load_samples(str(dataset_path), dataset_format)
    filtered = eval_base.maybe_filter_split(samples, "dev")
    samples = filtered or samples
    if limit and limit > 0:
        samples = samples[:limit]
    valid = []
    for sample in samples:
        docs = eval_base.build_documents(sample)
        if docs:
            valid.append(sample)
    return valid


def index_config(args: argparse.Namespace, dataset_path: Path, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dataset_path": str(dataset_path),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "use_paragraph_as_chunk": False,
        "seed": args.seed,
        "sample_ids": [str(sample.get("id", "")) for sample in samples],
    }


def manifest_matches(path: Path, expected: Dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(current.get(key) == value for key, value in expected.items())


def build_or_reuse_indexes(
    rag,
    samples: Sequence[Dict[str, Any]],
    shared_index_root: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    shared_index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shared_index_root / "index_config.json"
    expected = index_config(args, dataset_path, samples)
    can_reuse = (not args.rebuild_index) and manifest_matches(manifest_path, expected)
    if can_reuse:
        records = eval_base.records_from_shared_indexes(samples, shared_index_root)
        if records["summary"]["num_valid_samples"] == len(samples):
            logger.info("Reusing shared indexes from %s", shared_index_root)
            return records
        logger.info("Index cache manifest matched, but some indexes are missing; rebuilding missing items.")

    records: List[Dict[str, Any]] = []
    index_latencies: List[float] = []
    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        sample_dir = shared_index_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        index_path = sample_dir / eval_base.INDEX_FILENAME
        metadata_path = sample_dir / "metadata.json"
        if can_reuse and index_path.exists():
            records.append({"sample_id": sample_id, "index_path": str(index_path), "index_latency": 0.0, "valid": True, "stats": {}, "reused": True})
            continue
        t0 = time.perf_counter()
        try:
            documents = eval_base.build_documents(sample)
            if not documents:
                raise ValueError("sample has no usable paragraphs/documents")
            logger.info("[index][%d/%d] start %s", i, len(samples), sample_id)
            rag.index(documents)
            with index_path.open("wb") as handle:
                pickle.dump(rag.state, handle)
            stats = rag.describe_index()
            latency = time.perf_counter() - t0
            index_latencies.append(latency)
            metadata_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "question": sample.get("question", ""),
                        "source_dataset": source_dataset(sample),
                        "index_path": str(index_path),
                        "stats": stats,
                        "index_latency": latency,
                        "chunk_size": args.chunk_size,
                        "chunk_overlap": args.chunk_overlap,
                        "use_paragraph_as_chunk": False,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            records.append({"sample_id": sample_id, "index_path": str(index_path), "index_latency": latency, "valid": True, "stats": stats, "reused": False})
            logger.info("[index][%d/%d] %s | %.3fs", i, len(samples), sample_id, latency)
        except Exception as exc:
            records.append({"sample_id": sample_id, "index_path": str(index_path), "index_latency": 0.0, "valid": False, "error": str(exc), "reused": False})
            logger.exception("Index build failed for sample %s", sample_id)

    summary = {
        "num_samples": len(samples),
        "num_valid_samples": sum(1 for item in records if item.get("valid")),
        "num_reused_samples": sum(1 for item in records if item.get("reused")),
        "avg_index_latency": eval_base._avg(index_latencies),
        "total_index_runtime": time.perf_counter() - t_start,
        "shared_index_root": str(shared_index_root),
    }
    manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": records, "summary": summary}


def source_dataset(sample: Dict[str, Any]) -> str:
    return str(sample.get("source_dataset") or sample.get("dataset") or "").strip()


def patch_uniform_router(rag) -> None:
    def route(query: str, forced_profile: str = "auto") -> Dict[str, Any]:
        return {"profile": "multi_hop", "alpha": dict(UNIFORM_ALPHA), "confidence": 1.0}

    rag.config.intent_use_llm = False
    rag.intent_router.route = route


def evaluate_mode(
    rag,
    samples: Sequence[Dict[str, Any]],
    index_records: Sequence[Dict[str, Any]],
    mode_dir: Path,
    mode: str,
    logger: logging.Logger,
    token_counter: eval_base.TokenCounter,
    index_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    record_by_id = {str(item.get("sample_id", "")): item for item in index_records}
    per_example_path = mode_dir / "per_example.jsonl"
    per_example_path.write_text("", encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    qa_failures = 0
    skipped = 0
    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        index_record = record_by_id.get(sample_id)
        if not question or not index_record or not index_record.get("valid"):
            skipped += 1
            continue
        index_path = Path(str(index_record.get("index_path", "")))
        if not index_path.exists():
            skipped += 1
            continue
        with index_path.open("rb") as handle:
            rag.state = pickle.load(handle)
        rag.index_path = str(index_path)

        t0 = time.perf_counter()
        try:
            result = rag.query(question)
        except Exception as exc:
            qa_failures += 1
            logger.exception("[%s][%d/%d] query failed for %s", mode, i, len(samples), sample_id)
            result = {"predicted_answer": "", "alpha": {}, "query_timing": {}, "task_profile": "", "error": str(exc)}
        total_latency = time.perf_counter() - t0
        timing = result.get("query_timing", {}) or {}

        evidence_text = ""
        qa_messages = result.get("qa_messages", []) or []
        if len(qa_messages) >= 2 and isinstance(qa_messages[1], dict):
            evidence_text = str(qa_messages[1].get("content", ""))
        if not evidence_text:
            evidence_text = eval_base.format_qa_evidence_from_ranked_passages(result.get("ranked_passages", []) or [], rag.config.qa_passage_top_k)
        final_evidence_tokens = token_counter.count(evidence_text)
        alpha_row = eval_base._alpha_row(result.get("alpha", {}) or {})
        evidence_row = eval_base._evidence_granularity_row(result, final_evidence_tokens)
        predicted = str(result.get("predicted_answer", "")).strip()
        em, f1 = eval_base.best_em_f1(eval_base.build_gold_answers(sample), predicted)

        row = {
            "id": sample_id,
            "query_id": sample_id,
            "source_dataset": source_dataset(sample),
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "pred_answer": predicted,
            "predicted_answer": predicted,
            "EM": em,
            "F1": f1,
            "intent": "uniform_alpha" if mode == "uniform" else result.get("task_profile", ""),
            "type": "uniform_alpha" if mode == "uniform" else result.get("task_profile", ""),
            "alpha_fact": alpha_row["alpha_F"],
            "alpha_sentence": alpha_row["alpha_S"],
            "alpha_chunk": alpha_row["alpha_C"],
            "alpha_F": alpha_row["alpha_F"],
            "alpha_S": alpha_row["alpha_S"],
            "alpha_C": alpha_row["alpha_C"],
            "retrieval_latency": float(timing.get("retrieval_latency", 0.0) or 0.0),
            "qa_latency": float(timing.get("qa_latency", 0.0) or 0.0),
            "total_latency": total_latency,
            "query_total_latency": total_latency,
            "retrieval_pipeline_latency": float(timing.get("retrieval_pipeline_latency", 0.0) or 0.0),
            "final_evidence_tokens": int(evidence_row["final_evidence_tokens"]),
            "query_timing": timing,
            "index_path": str(index_path),
            "index_reused": bool(index_record.get("reused", False)),
            "qa_answer_mode": result.get("qa_answer_mode", ""),
        }
        row.update(evidence_row)
        if result.get("error"):
            row["error"] = result["error"]
        rows.append(row)
        with per_example_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[%s][%d/%d] %s | f1=%.4f em=%.4f | running_f1=%.4f | q=%.3fs",
            mode,
            i,
            len(samples),
            sample_id,
            f1,
            em,
            eval_base._avg([float(item["F1"]) for item in rows]),
            total_latency,
        )

    query_runtime = time.perf_counter() - t_start
    metrics = summarize_rows(mode, rows)
    metrics.update(
        {
            "variant": f"holorag_{mode}_routing",
            "num_skipped": skipped,
            "qa_failures": qa_failures,
            "index_latency": float(index_summary.get("avg_index_latency", 0.0) or 0.0),
            "final_evidence_tokenizer": token_counter.method,
            "shared_index_runtime": float(index_summary.get("total_index_runtime", 0.0) or 0.0),
            "query_runtime": query_runtime,
            "total_runtime": query_runtime + float(index_summary.get("total_index_runtime", 0.0) or 0.0),
            "index_pool": "shared",
        }
    )
    write_metrics(mode_dir, [metrics])
    return metrics, rows


def summarize_rows(mode: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "mode": mode,
        "num_queries": len(rows),
        "F1": eval_base._avg([float(row.get("F1", 0.0) or 0.0) for row in rows]),
        "EM": eval_base._avg([float(row.get("EM", 0.0) or 0.0) for row in rows]),
        "retrieval_latency": eval_base._avg([float(row.get("retrieval_latency", 0.0) or 0.0) for row in rows]),
        "qa_latency": eval_base._avg([float(row.get("qa_latency", 0.0) or 0.0) for row in rows]),
        "total_latency": eval_base._avg([float(row.get("total_latency", 0.0) or 0.0) for row in rows]),
        "retrieval_qa_latency": eval_base._avg([float(row.get("total_latency", 0.0) or 0.0) for row in rows]),
        "final_evidence_tokens": eval_base._avg([float(row.get("final_evidence_tokens", 0.0) or 0.0) for row in rows]),
        "avg_alpha_fact": eval_base._avg([float(row.get("alpha_fact", 0.0) or 0.0) for row in rows]),
        "avg_alpha_sentence": eval_base._avg([float(row.get("alpha_sentence", 0.0) or 0.0) for row in rows]),
        "avg_alpha_chunk": eval_base._avg([float(row.get("alpha_chunk", 0.0) or 0.0) for row in rows]),
        "avg_alpha_F": eval_base._avg([float(row.get("alpha_fact", 0.0) or 0.0) for row in rows]),
        "avg_alpha_S": eval_base._avg([float(row.get("alpha_sentence", 0.0) or 0.0) for row in rows]),
        "avg_alpha_C": eval_base._avg([float(row.get("alpha_chunk", 0.0) or 0.0) for row in rows]),
        "final_evidence_tokenizer": "",
    }


def grouped_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        groups.setdefault(source, []).append(row)
    return {
        source: {
            "num_queries": len(items),
            "F1": eval_base._avg([float(item.get("F1", 0.0) or 0.0) for item in items]),
            "EM": eval_base._avg([float(item.get("EM", 0.0) or 0.0) for item in items]),
        }
        for source, items in sorted(groups.items())
    }


def intent_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        intent = str(row.get("intent") or row.get("type") or "").strip()
        if intent:
            counts[intent] = counts.get(intent, 0) + 1
    return dict(sorted(counts.items()))


def write_metrics(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    headers: List[str] = []
    for key in SUMMARY_KEYS:
        if any(key in row for row in rows):
            headers.append(key)
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    (run_dir / "metrics_summary.json").write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def write_comparison(output_dir: Path, metrics_by_mode: Dict[str, Dict[str, Any]], rows_by_mode: Dict[str, List[Dict[str, Any]]]) -> None:
    comparison: Dict[str, Any] = {"modes": {}, "by_source_dataset": {}, "intent_distribution": {}}
    for mode, metrics in metrics_by_mode.items():
        comparison["modes"][mode] = {
            key: metrics.get(key)
            for key in [
                "F1",
                "EM",
                "retrieval_latency",
                "qa_latency",
                "total_latency",
                "final_evidence_tokens",
                "avg_alpha_fact",
                "avg_alpha_sentence",
                "avg_alpha_chunk",
            ]
        }
        comparison["by_source_dataset"][mode] = grouped_metrics(rows_by_mode.get(mode, []))
        dist = intent_distribution(rows_by_mode.get(mode, [])) if mode == "llm" else {}
        if dist:
            comparison["intent_distribution"][mode] = dist
    (output_dir / "routing_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    for mode, values in comparison["modes"].items():
        flat = {"section": "overall", "mode": mode, **values}
        rows.append(flat)
    for mode, groups in comparison["by_source_dataset"].items():
        for source, values in groups.items():
            rows.append({"section": "source_dataset", "mode": mode, "source_dataset": source, **values})
    for mode, dist in comparison.get("intent_distribution", {}).items():
        for intent, count in dist.items():
            rows.append({"section": "intent_distribution", "mode": mode, "intent": intent, "count": count})
    headers: List[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with (output_dir / "routing_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def release_rag(rag: Any) -> None:
    try:
        del rag
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def load_existing_mode_outputs(output_dir: Path, modes_to_skip: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    metrics_by_mode: Dict[str, Dict[str, Any]] = {}
    rows_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for mode in modes_to_skip:
        mode_dir = output_dir / mode
        metrics_path = mode_dir / "metrics_summary.json"
        per_example_path = mode_dir / "per_example.jsonl"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    metrics_by_mode[mode] = dict(payload[0])
                elif isinstance(payload, dict):
                    metrics_by_mode[mode] = dict(payload)
            except Exception:
                pass
        if per_example_path.exists():
            try:
                rows_by_mode[mode] = eval_base.load_jsonl(str(per_example_path))
            except Exception:
                rows_by_mode[mode] = []
    return metrics_by_mode, rows_by_mode


def parse_modes(value: str) -> List[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in {"uniform", "llm"}]
    if invalid:
        raise ValueError(f"Invalid --modes value(s): {invalid}. Use uniform,llm, uniform, or llm.")
    return modes or ["uniform", "llm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Uniform Alpha vs LLM query-level granularity routing on MultiGranularityQA.")
    parser.add_argument("--dataset_path", type=str, default="code/HoloRAG/dataset/MultiGranularityQA.json")
    parser.add_argument("--output_dir", type=str, default="results/MultiGranularityQA_routing")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--modes", type=str, default="uniform,llm")

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--spacy_model_name", type=str, default="en_core_web_sm")
    parser.add_argument("--index_extraction_mode", type=str, default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--entity_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--entity_similarity_top_k", type=int, default=2047)
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
    parser.add_argument("--evidence_alpha_uniform_mix", type=float, default=0.0, help="Mix this fraction of uniform alpha into the alpha used only for evidence packing.")
    parser.add_argument("--evidence_soft_token_budget", type=int, default=0)
    parser.add_argument("--evidence_allow_underfill", action="store_true", default=True)
    parser.add_argument("--evidence_min_score", type=float, default=0.0)
    parser.add_argument("--evidence_redundancy_threshold", type=float, default=0.85)
    parser.add_argument("--skip_llm_health_check", action="store_true")
    parser.add_argument("--llm_health_timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size != 256 or args.chunk_overlap != 64:
        raise ValueError("This experiment is fixed to chunk_size=256 and chunk_overlap=64.")
    random.seed(args.seed)
    eval_base.set_global_seed(args.seed)

    from holorag import HoloRAG

    dataset_path = resolve_path(args.dataset_path)
    output_dir = resolve_path(args.output_dir, prefer_repo_results=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = eval_base.setup_logger(output_dir / "run.log")
    logger.info("Output directory: %s", output_dir)
    logger.info("Dataset path: %s", dataset_path)
    if not args.skip_llm_health_check:
        eval_base.check_llm_server(args.llm_base_url, args.llm_name, args.llm_health_timeout, logger)

    samples = load_multigranularity_samples(dataset_path, args.limit)
    if not samples:
        raise ValueError("No usable samples found. Samples need question, answer, and non-empty paragraphs/documents.")
    (output_dir / "sampled_queries.json").write_text(
        json.dumps({"seed": args.seed, "limit": args.limit, "num_eval_queries": len(samples), "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_args = build_eval_args(args, output_dir / "workdir_index", intent_use_llm=False)
    index_rag = HoloRAG(eval_base.build_config(index_args, save_dir=str(output_dir / "workdir_index")))
    shared_index_root = output_dir / "shared_indexes" / f"chunk{args.chunk_size}_overlap{args.chunk_overlap}"
    index_data = build_or_reuse_indexes(index_rag, samples, shared_index_root, args, dataset_path, logger)
    (output_dir / "shared_index_records.json").write_text(json.dumps(index_data["records"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "shared_index_summary.json").write_text(json.dumps(index_data["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    release_rag(index_rag)

    modes = parse_modes(args.modes)
    token_counter = eval_base.TokenCounter(args.llm_name, logger)
    skipped_modes = [mode for mode in ("uniform", "llm") if mode not in modes]
    metrics_by_mode, rows_by_mode = load_existing_mode_outputs(output_dir, skipped_modes)
    for mode in modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_args = build_eval_args(args, mode_dir / "workdir", intent_use_llm=(mode == "llm"))
        rag = HoloRAG(eval_base.build_config(mode_args, save_dir=str(mode_dir / "workdir")))
        if mode == "uniform":
            patch_uniform_router(rag)
        metrics, rows = evaluate_mode(rag, samples, index_data["records"], mode_dir, mode, logger, token_counter, index_data["summary"])
        metrics_by_mode[mode] = metrics
        rows_by_mode[mode] = rows
        (mode_dir / "config.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "dataset_path": str(dataset_path),
                    "output_dir": str(output_dir),
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "use_paragraph_as_chunk": False,
                    "seed": args.seed,
                    "limit": args.limit,
                    "intent_use_llm": mode == "llm",
                    "uses_dataset_for_routing": False,
                    "uniform_alpha": UNIFORM_ALPHA if mode == "uniform" else None,
                    "evidence_alpha_uniform_mix": args.evidence_alpha_uniform_mix,
                    "shared_index_root": str(shared_index_root),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        release_rag(rag)

    write_comparison(output_dir, metrics_by_mode, rows_by_mode)
    logger.info("Saved comparison: %s", output_dir / "routing_comparison.json")


if __name__ == "__main__":
    main()
