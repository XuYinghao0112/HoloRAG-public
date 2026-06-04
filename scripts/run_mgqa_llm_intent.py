"""Run HoloRAG-LLM-Intent on MultiGranularityQA.

This entrypoint intentionally runs only one method:
HoloRAG-LLM-Intent with question-only continuous alpha routing.
"""

import argparse
import csv
import json
import logging
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
INDEX_FILENAME = "holorag_index.pkl"
FALLBACK_ALPHA = {"fact": 0.33, "sentence": 0.33, "chunk": 0.33}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval as eval_base  # noqa: E402
from holorag.utils import normalize_alpha, safe_parse_json  # noqa: E402


def resolve_path(path_value: str, *, prefer_repo_results: bool = False) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if prefer_repo_results and path.parts and path.parts[0] == "results":
        return (REPO_ROOT / path).resolve()
    if path.parts[:2] == ("code", "HoloRAG"):
        return (REPO_ROOT.parents[1] / path).resolve()
    return path.resolve()


def source_dataset(sample: Dict[str, Any]) -> str:
    return str(sample.get("source_dataset") or sample.get("dataset") or "unknown").strip() or "unknown"


def load_mgqa_samples(dataset_path: Path, limit: int) -> List[Dict[str, Any]]:
    dataset_format = eval_base.detect_dataset_format(str(dataset_path), "auto")
    samples = eval_base.load_samples(str(dataset_path), dataset_format)
    filtered = eval_base.maybe_filter_split(samples, "dev")
    samples = filtered or samples
    if limit and limit > 0:
        samples = samples[:limit]
    return [sample for sample in samples if str(sample.get("question", "")).strip() and eval_base.build_documents(sample)]


def build_eval_args(args: argparse.Namespace, save_dir: Path, *, intent_use_llm: bool) -> argparse.Namespace:
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
        evidence_alpha_uniform_mix=0.0,
        evidence_soft_token_budget=args.evidence_soft_token_budget,
        evidence_allow_underfill=args.evidence_allow_underfill,
        evidence_min_score=args.evidence_min_score,
        evidence_redundancy_threshold=args.evidence_redundancy_threshold,
        disable_evidence_alpha_weights=False,
        execution_mode=args.execution_mode,
        num_workers=args.num_workers,
        multi_worker_embedding_devices=args.multi_worker_embedding_devices,
        save_dir=str(save_dir),
    )


def index_manifest(args: argparse.Namespace, dataset_path: Path, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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


def reuse_existing_indexes(
    samples: Sequence[Dict[str, Any]],
    shared_index_root: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    if args.rebuild_index:
        return None
    manifest_path = shared_index_root / "index_config.json"
    if not manifest_matches(manifest_path, index_manifest(args, dataset_path, samples)):
        return None
    records = eval_base.records_from_shared_indexes(samples, shared_index_root)
    if records["summary"]["num_valid_samples"] == len(samples):
        logger.info("Reusing shared indexes from %s", shared_index_root)
        return records
    logger.info("Shared index manifest matched, but some indexes are missing; rebuilding missing items.")
    return None


def build_or_reuse_indexes(
    rag: Any,
    samples: Sequence[Dict[str, Any]],
    shared_index_root: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    shared_index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shared_index_root / "index_config.json"
    expected = index_manifest(args, dataset_path, samples)
    can_reuse = (not args.rebuild_index) and manifest_matches(manifest_path, expected)
    if can_reuse:
        records = eval_base.records_from_shared_indexes(samples, shared_index_root)
        if records["summary"]["num_valid_samples"] == len(samples):
            logger.info("Reusing shared indexes from %s", shared_index_root)
            return records
        logger.info("Shared index manifest matched, but some indexes are missing; rebuilding missing items.")

    records: List[Dict[str, Any]] = []
    index_latencies: List[float] = []
    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        sample_dir = shared_index_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        index_path = sample_dir / INDEX_FILENAME
        metadata_path = sample_dir / "metadata.json"
        if can_reuse and index_path.exists():
            records.append({"sample_id": sample_id, "index_path": str(index_path), "index_latency": 0.0, "valid": True, "stats": {}, "reused": True})
            continue
        t0 = time.perf_counter()
        try:
            documents = eval_base.build_documents(sample)
            if not documents:
                raise ValueError("sample has no usable documents")
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


def alpha_prompt() -> str:
    # Previous prompt (question_only_chunk_calibrated_v2), kept for
    # reproducibility:
    # return (
    #     "Predict retrieval granularity preference weights from the question text only.\n"
    #     "Return only a JSON object with numeric keys alpha_F, alpha_S, alpha_C.\n"
    #     "The three values are continuous non-negative weights for fact, sentence, and chunk evidence; they should sum to 1.\n"
    #     "Use alpha_F for compact direct facts: entity attributes, relations, names, dates, locations, counts, and direct single-hop lookup. "
    #     "Use alpha_S for localized sentence evidence: one or a few sentences that connect clues, compare entities, explain a relation, or support a short reasoning step. "
    #     "Use alpha_C for broader chunk-level context: narrative/background context, character actions or motivations, ambiguous references, dispersed mentions, "
    #     "event sequences, long-context questions, or cases where surrounding passage context is likely needed.\n"
    #     "Important calibration: a question that starts with who, what, where, or when is not necessarily fact-only. "
    #     "If the answer depends on story context, character behavior, motivation, causality, or surrounding events, assign meaningful chunk weight. "
    #     "For ambiguous non-direct lookup questions, avoid alpha_C below 0.20. For narrative/background/context-dependent questions, prefer alpha_C around 0.30-0.50. "
    #     "Use extreme fact weights only when the question is clearly answerable by a compact standalone fact."
    # )
    return (
        "Infer retrieval granularity from the question text only, then predict calibrated evidence weights.\n"
        "Do not infer or use any dataset name, source, benchmark, source_dataset, domain label, or metadata.\n"
        "First classify the question as one granularity_type:\n"
        "- single_hop_fact: direct lookup of a name, date, location, count, entity attribute, or simple relation.\n"
        "- multi_hop_bridge: requires connecting, comparing, composing, or verifying facts across entities, events, or constraints.\n"
        "- long_context_narrative: requires story/background context, character behavior or motivation, causality, event sequence, ambiguous references, "
        "dispersed mentions, or surrounding passage context.\n"
        "- mixed_uncertain: ambiguous or mixed signals.\n"
        "Then return only a JSON object with keys granularity_type, alpha_F, alpha_S, alpha_C.\n"
        "The alpha values are continuous non-negative weights for fact, sentence, and chunk evidence, and should sum to 1.\n"
        "Use these calibrated ranges:\n"
        "single_hop_fact: alpha_F 0.70-0.90, alpha_S 0.10-0.25, alpha_C 0.00-0.15.\n"
        "multi_hop_bridge: alpha_F 0.30-0.50, alpha_S 0.35-0.55, alpha_C 0.10-0.25.\n"
        "long_context_narrative: alpha_F 0.10-0.30, alpha_S 0.20-0.35, alpha_C 0.45-0.70.\n"
        "mixed_uncertain: alpha_F 0.35-0.45, alpha_S 0.30-0.40, alpha_C 0.20-0.30.\n"
        "A question that starts with who, what, where, or when is not automatically single_hop_fact; use the reasoning need implied by the wording."
    )


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if number < 0:
        return 0.0
    return number


def _values_from_payload(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    candidates = {
        "fact": payload.get("alpha_F", payload.get("alpha_fact", payload.get("fact"))),
        "sentence": payload.get("alpha_S", payload.get("alpha_sentence", payload.get("sentence"))),
        "chunk": payload.get("alpha_C", payload.get("alpha_chunk", payload.get("chunk"))),
    }
    values = {key: _float_or_none(value) for key, value in candidates.items()}
    if any(value is None for value in values.values()):
        return None
    if sum(value or 0.0 for value in values.values()) <= 0:
        return None
    return {key: float(value or 0.0) for key, value in values.items()}


def parse_alpha(raw_text: str, payload: Any) -> Tuple[Dict[str, float], bool]:
    parsed_raw = safe_parse_json(raw_text, None)
    values = _values_from_payload(parsed_raw)
    if values is None:
        values = _values_from_payload(payload)
    if values is None:
        numbers = [_float_or_none(match) for match in re.findall(r"-?\d+(?:\.\d+)?", str(raw_text or ""))]
        numbers = [number for number in numbers if number is not None]
        if len(numbers) >= 3 and sum(numbers[:3]) > 0:
            values = {"fact": numbers[0], "sentence": numbers[1], "chunk": numbers[2]}
    if values is None:
        return normalize_alpha(FALLBACK_ALPHA), True
    return normalize_alpha(values), False


def profile_from_alpha(alpha: Dict[str, float]) -> str:
    if alpha.get("chunk", 0.0) >= 0.42:
        return "long_context"
    if alpha.get("sentence", 0.0) + alpha.get("fact", 0.0) >= 0.62:
        return "multi_hop"
    return "single_hop"


def patch_question_only_llm_alpha_router(rag: Any) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}

    def route(query: str, forced_profile: str = "auto") -> Dict[str, Any]:
        normalized = " ".join(str(query or "").split())
        if normalized in cache:
            cached = cache[normalized]
            return {
                "profile": cached["profile"],
                "alpha": dict(cached["alpha"]),
                "confidence": cached["confidence"],
                "granularity_intent": "llm_continuous_alpha",
            }
        payload, raw_text = rag.intent_router.llm_client.infer_json(
            system_prompt=alpha_prompt(),
            user_prompt=f"Question:\n{normalized}",
            fallback=FALLBACK_ALPHA,
            max_tokens=96,
        )
        alpha, parse_failed = parse_alpha(raw_text, payload)
        result = {
            "profile": profile_from_alpha(alpha),
            "alpha": alpha,
            "confidence": 1.0,
            "raw_alpha_output": raw_text,
            "alpha_parse_failed": parse_failed,
        }
        cache[normalized] = result
        return {
            "profile": result["profile"],
            "alpha": dict(alpha),
            "confidence": result["confidence"],
            "granularity_intent": "llm_continuous_alpha",
        }

    rag.config.intent_use_llm = True
    rag.config.task_profile = "auto"
    rag.intent_router.route = route
    return cache


def evidence_text_for_count(result: Dict[str, Any], rag: Any) -> str:
    qa_messages = result.get("qa_messages", []) or []
    if len(qa_messages) >= 2 and isinstance(qa_messages[1], dict):
        content = str(qa_messages[1].get("content", ""))
        if content:
            return content
    return eval_base.format_qa_evidence_from_ranked_passages(result.get("ranked_passages", []) or [], rag.config.qa_passage_top_k)


def run_queries(
    rag: Any,
    samples: Sequence[Dict[str, Any]],
    index_records: Sequence[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger,
    token_counter: eval_base.TokenCounter,
) -> List[Dict[str, Any]]:
    record_by_id = {str(item.get("sample_id", "")): item for item in index_records}
    router_cache = patch_question_only_llm_alpha_router(rag)
    per_query_path = output_dir / "per_query_results.jsonl"
    per_query_path.write_text("", encoding="utf-8")
    rows: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        record = record_by_id.get(sample_id)
        if not question or not record or not record.get("valid"):
            logger.warning("[query][%d/%d] skipping %s; missing question or valid index", i, len(samples), sample_id)
            continue
        index_path = Path(str(record.get("index_path", "")))
        if not index_path.exists():
            logger.warning("[query][%d/%d] skipping %s; index missing at %s", i, len(samples), sample_id, index_path)
            continue

        with index_path.open("rb") as handle:
            rag.state = pickle.load(handle)
        rag.index_path = str(index_path)

        t0 = time.perf_counter()
        try:
            result = rag.query(question)
        except Exception as exc:
            logger.exception("[query][%d/%d] failed %s", i, len(samples), sample_id)
            result = {"predicted_answer": "", "alpha": {}, "query_timing": {}, "ranked_evidence": {}, "error": str(exc)}
        elapsed = time.perf_counter() - t0

        timing = result.get("query_timing", {}) or {}
        evidence_values = eval_base._evidence_granularity_row(result, token_counter.count(evidence_text_for_count(result, rag)))
        alpha_values = eval_base._alpha_row(result.get("alpha", {}) or {})
        predicted = str(result.get("predicted_answer", "")).strip()
        em, f1 = eval_base.best_em_f1(eval_base.build_gold_answers(sample), predicted)
        alpha_meta = router_cache.get(" ".join(question.split()), {})

        row = {
            "query_id": sample_id,
            "source_dataset": source_dataset(sample),
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "prediction": predicted,
            "f1": f1,
            "em": em,
            "final_evidence_tokens": int(evidence_values["final_evidence_tokens"]),
            "fact_tokens": int(evidence_values["used_tokens_F"]),
            "sentence_tokens": int(evidence_values["used_tokens_S"]),
            "chunk_tokens": int(evidence_values["used_tokens_C"]),
            "alpha_fact": alpha_values["alpha_F"],
            "alpha_sentence": alpha_values["alpha_S"],
            "alpha_chunk": alpha_values["alpha_C"],
            "raw_alpha_output": alpha_meta.get("raw_alpha_output", ""),
            "alpha_parse_failed": bool(alpha_meta.get("alpha_parse_failed", False)),
            "retrieval_latency": float(timing.get("retrieval_latency", 0.0) or 0.0),
            "ppr_latency": float(timing.get("pagerank_latency", 0.0) or 0.0),
            "qa_latency": float(timing.get("qa_latency", 0.0) or 0.0),
            "generation_latency": float(timing.get("qa_latency", 0.0) or 0.0),
            "total_latency": float(timing.get("query_total_latency", elapsed) or elapsed),
        }
        if result.get("error"):
            row["error"] = result["error"]
        rows.append(row)
        with per_query_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[query][%d/%d] %s | f1=%.4f em=%.4f | alpha=(%.3f, %.3f, %.3f) | q=%.3fs",
            i,
            len(samples),
            sample_id,
            f1,
            em,
            row["alpha_fact"],
            row["alpha_sentence"],
            row["alpha_chunk"],
            row["total_latency"],
        )
    return rows


def avg(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return eval_base._avg([float(row.get(key, 0.0) or 0.0) for row in rows])


def fail_rate(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("alpha_parse_failed")) / len(rows)


def summary_row(rows: Sequence[Dict[str, Any]], *, method: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if method is not None:
        row["method"] = method
    if source is not None:
        row["source_dataset"] = source
    row.update(
        {
            "count": len(rows),
            "f1": avg(rows, "f1"),
            "em": avg(rows, "em"),
            "avg_evidence_tokens": avg(rows, "final_evidence_tokens"),
            "avg_fact_tokens": avg(rows, "fact_tokens"),
            "avg_sentence_tokens": avg(rows, "sentence_tokens"),
            "avg_chunk_tokens": avg(rows, "chunk_tokens"),
            "avg_alpha_fact": avg(rows, "alpha_fact"),
            "avg_alpha_sentence": avg(rows, "alpha_sentence"),
            "avg_alpha_chunk": avg(rows, "alpha_chunk"),
            "alpha_parse_fail_rate": fail_rate(rows),
            "retrieval_latency": avg(rows, "retrieval_latency"),
            "ppr_latency": avg(rows, "ppr_latency"),
            "qa_latency": avg(rows, "qa_latency"),
            "total_latency": avg(rows, "total_latency"),
        }
    )
    return row


def group_by_source(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("source_dataset") or "unknown"), []).append(row)
    return dict(sorted(groups.items()))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def token_percent(part: float, total: float) -> float:
    return (part / total) if total > 0 else 0.0


def write_summaries(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    overall_fields = [
        "method",
        "count",
        "f1",
        "em",
        "avg_evidence_tokens",
        "avg_fact_tokens",
        "avg_sentence_tokens",
        "avg_chunk_tokens",
        "avg_alpha_fact",
        "avg_alpha_sentence",
        "avg_alpha_chunk",
        "alpha_parse_fail_rate",
        "retrieval_latency",
        "ppr_latency",
        "qa_latency",
        "total_latency",
    ]
    source_fields = ["source_dataset"] + overall_fields[1:]
    alpha_fields = [
        "source_dataset",
        "mean_alpha_fact",
        "mean_alpha_sentence",
        "mean_alpha_chunk",
        "std_alpha_fact",
        "std_alpha_sentence",
        "std_alpha_chunk",
        "count",
    ]
    evidence_fields = [
        "source_dataset",
        "fact_token_percent",
        "sentence_token_percent",
        "chunk_token_percent",
        "avg_evidence_tokens",
        "count",
    ]

    write_csv(output_dir / "overall_summary.csv", [summary_row(rows, method="HoloRAG-LLM-Intent")], overall_fields)

    by_source = group_by_source(rows)
    write_csv(
        output_dir / "by_source_summary.csv",
        [summary_row(items, source=source) for source, items in by_source.items()],
        source_fields,
    )

    alpha_rows = []
    evidence_rows = []
    for source, items in by_source.items():
        fact_alphas = [float(item.get("alpha_fact", 0.0) or 0.0) for item in items]
        sentence_alphas = [float(item.get("alpha_sentence", 0.0) or 0.0) for item in items]
        chunk_alphas = [float(item.get("alpha_chunk", 0.0) or 0.0) for item in items]
        alpha_rows.append(
            {
                "source_dataset": source,
                "mean_alpha_fact": eval_base._avg(fact_alphas),
                "mean_alpha_sentence": eval_base._avg(sentence_alphas),
                "mean_alpha_chunk": eval_base._avg(chunk_alphas),
                "std_alpha_fact": pstdev(fact_alphas) if len(fact_alphas) > 1 else 0.0,
                "std_alpha_sentence": pstdev(sentence_alphas) if len(sentence_alphas) > 1 else 0.0,
                "std_alpha_chunk": pstdev(chunk_alphas) if len(chunk_alphas) > 1 else 0.0,
                "count": len(items),
            }
        )
        fact_tokens = avg(items, "fact_tokens")
        sentence_tokens = avg(items, "sentence_tokens")
        chunk_tokens = avg(items, "chunk_tokens")
        total = fact_tokens + sentence_tokens + chunk_tokens
        evidence_rows.append(
            {
                "source_dataset": source,
                "fact_token_percent": token_percent(fact_tokens, total),
                "sentence_token_percent": token_percent(sentence_tokens, total),
                "chunk_token_percent": token_percent(chunk_tokens, total),
                "avg_evidence_tokens": avg(items, "final_evidence_tokens"),
                "count": len(items),
            }
        )
    write_csv(output_dir / "alpha_by_source.csv", alpha_rows, alpha_fields)
    write_csv(output_dir / "evidence_token_allocation.csv", evidence_rows, evidence_fields)


def release_rag(rag: Any) -> None:
    try:
        del rag
    except Exception:
        pass
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean HoloRAG-LLM-Intent on MultiGranularityQA.")
    parser.add_argument("--dataset_path", type=str, default="code/HoloRAG/dataset/MultiGranularityQA.json")
    parser.add_argument("--output_dir", type=str, default="results/mgqa_llm_intent")
    parser.add_argument("--shared_index_root", type=str, default="results/MultiGranularityQA_routing/shared_indexes/chunk256_overlap64")
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--execution_mode", type=str, default="sequential", choices=["sequential", "multi_worker"])
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--multi_worker_embedding_devices", type=str, default="")

    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=64)
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
        raise ValueError("MGQA LLM-intent is fixed to chunk_size=256 and chunk_overlap=64.")
    eval_base.set_global_seed(args.seed)

    from holorag import HoloRAG

    dataset_path = resolve_path(args.dataset_path)
    output_dir = resolve_path(args.output_dir, prefer_repo_results=True)
    shared_index_root = resolve_path(args.shared_index_root, prefer_repo_results=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = eval_base.setup_logger(output_dir / "run.log")
    logger.info("Running method: HoloRAG-LLM-Intent")
    logger.info("Output directory: %s", output_dir)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Shared index root: %s", shared_index_root)
    if not args.skip_llm_health_check:
        eval_base.check_llm_server(args.llm_base_url, args.llm_name, args.llm_health_timeout, logger)

    samples = load_mgqa_samples(dataset_path, args.limit)
    if not samples:
        raise ValueError("No usable MultiGranularityQA samples found.")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "method": "HoloRAG-LLM-Intent",
                "dataset_path": str(dataset_path),
                "answer_generator": args.llm_name,
                "embedding_model": args.embedding_name,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "use_paragraph_as_chunk": False,
                "source_dataset_used_for_routing": False,
                "alpha_prompt_version": "question_only_granularity_type_ranges_v3",
                "output_dir": str(output_dir),
                "shared_index_root": str(shared_index_root),
                "seed": args.seed,
                "limit": args.limit,
                "num_samples": len(samples),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    index_data = reuse_existing_indexes(samples, shared_index_root, args, dataset_path, logger)
    if index_data is None:
        index_args = build_eval_args(args, output_dir / "workdir_index", intent_use_llm=False)
        index_rag = HoloRAG(eval_base.build_config(index_args, save_dir=str(output_dir / "workdir_index")))
        index_data = build_or_reuse_indexes(index_rag, samples, shared_index_root, args, dataset_path, logger)
        release_rag(index_rag)
        index_rag = None
    (output_dir / "shared_index_records.json").write_text(json.dumps(index_data["records"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "shared_index_summary.json").write_text(json.dumps(index_data["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    run_args = build_eval_args(args, output_dir / "workdir", intent_use_llm=True)
    rag = HoloRAG(eval_base.build_config(run_args, save_dir=str(output_dir / "workdir")))
    token_counter = eval_base.TokenCounter(args.llm_name, logger)
    rows = run_queries(rag, samples, index_data["records"], output_dir, logger, token_counter)
    write_summaries(output_dir, rows)
    release_rag(rag)
    rag = None
    logger.info("Finished HoloRAG-LLM-Intent. Wrote %d per-query rows.", len(rows))


if __name__ == "__main__":
    main()
