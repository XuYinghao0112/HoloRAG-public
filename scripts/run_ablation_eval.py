import argparse
import json
import logging
import pickle
import random
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def maybe_filter_split(samples: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if not split:
        return list(samples)
    split_key = None
    for key in ("split", "subset"):
        if any(key in sample for sample in samples):
            split_key = key
            break
    if split_key is None:
        return list(samples)
    return [sample for sample in samples if str(sample.get(split_key, "")) == split]


def _normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = _normalize_answer(gold).split()
    pred_tokens = _normalize_answer(pred).split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_counter: Dict[str, int] = {}
    pred_counter: Dict[str, int] = {}
    for token in gold_tokens:
        gold_counter[token] = gold_counter.get(token, 0) + 1
    for token in pred_tokens:
        pred_counter[token] = pred_counter.get(token, 0) + 1
    common = 0
    for token, count in pred_counter.items():
        if token in gold_counter:
            common += min(count, gold_counter[token])
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em_f1(gold_answers: Sequence[str], pred: str) -> Tuple[float, float]:
    if not gold_answers:
        return 0.0, 0.0
    pred_norm = _normalize_answer(pred)
    best_em = 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_norm = _normalize_answer(gold)
        em = 1.0 if gold_norm == pred_norm else 0.0
        f1 = _token_f1(gold, pred)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    return best_em, best_f1


def build_gold_answers(sample: Dict[str, Any]) -> List[str]:
    answers: List[str] = []
    if sample.get("answer"):
        answers.append(str(sample["answer"]))
    answers.extend(str(alias) for alias in sample.get("answer_aliases", []) if alias)
    dedup: List[str] = []
    seen = set()
    for ans in answers:
        key = _normalize_answer(ans)
        if key and key not in seen:
            seen.add(key)
            dedup.append(ans)
    return dedup


def supporting_paragraphs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [p for p in sample.get("paragraphs", []) if p.get("is_supporting")]


def _paragraph_uid(paragraph: Dict[str, Any]) -> Optional[str]:
    for key in ("idx", "id", "paragraph_id", "pid"):
        if key in paragraph and paragraph[key] is not None:
            return str(paragraph[key])
    return None


def _retrieved_uid(item: Dict[str, Any]) -> Optional[str]:
    for key in ("passage_index", "idx", "id", "paragraph_id"):
        if key in item and item[key] is not None:
            return str(item[key])
    node_id = item.get("node_id")
    if node_id is not None:
        return str(node_id)
    return None


def _normalized_title(text: str) -> str:
    return _normalize_answer(text)


def recall_at_k(
    support_paragraph_list: Sequence[Dict[str, Any]],
    retrieved_passages: Sequence[Dict[str, Any]],
    k: int,
) -> float:
    topk = list(retrieved_passages[:k])
    support_ids = {uid for p in support_paragraph_list if (uid := _paragraph_uid(p))}
    retrieved_ids = {uid for r in topk if (uid := _retrieved_uid(r))}
    if support_ids:
        return len(support_ids & retrieved_ids) / len(support_ids)

    support_titles = {
        _normalized_title(str(p.get("title", "")))
        for p in support_paragraph_list
        if str(p.get("title", "")).strip()
    }
    retrieved_titles = {
        _normalized_title(str(r.get("title", "")))
        for r in topk
        if str(r.get("title", "")).strip()
    }
    if not support_titles:
        return 0.0
    return len(support_titles & retrieved_titles) / len(support_titles)


def word_token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def sample_queries(samples: Sequence[Dict[str, Any]], seed: int, num_eval_queries: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    available = list(samples)
    if num_eval_queries >= len(available):
        return available
    return rng.sample(available, num_eval_queries)


def _avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_config(args: argparse.Namespace, save_dir: str, variant_flags: Dict[str, bool]) -> "HoloRAGConfig":
    from src.holorag import HoloRAGConfig

    # Keep legacy eval semantics: topk_triples/topk_passages are the primary knobs
    # for reproducibility with the strong baseline runs.
    retrieval_top_k = max(args.retrieval_top_k, args.topk_passages)
    fact_top_k = args.fact_top_k if args.fact_top_k is not None else args.topk_triples
    fact_rerank_top_k = args.fact_rerank_top_k if args.fact_rerank_top_k is not None else args.topk_triples
    fact_output_top_k = args.fact_output_top_k if args.fact_output_top_k is not None else args.topk_triples
    passage_output_top_k = args.passage_output_top_k if args.passage_output_top_k is not None else max(args.topk_passages, 10)
    qa_passage_top_k = args.qa_passage_top_k if args.qa_passage_top_k is not None else args.topk_passages

    return HoloRAGConfig(
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        save_dir=save_dir,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_dtype=args.embedding_dtype,
        entity_max_length=args.entity_max_length,
        sentence_max_length=args.sentence_max_length,
        chunk_max_length=args.chunk_max_length,
        query_max_length=args.query_max_length,
        linking_top_k=args.linking_top_k,
        fact_candidate_top_k=args.fact_candidate_top_k,
        retrieval_top_k=retrieval_top_k,
        fact_top_k=fact_top_k,
        fact_rerank_top_k=fact_rerank_top_k,
        fact_output_top_k=fact_output_top_k,
        passage_output_top_k=passage_output_top_k,
        qa_passage_top_k=qa_passage_top_k,
        entity_alias_threshold=args.synonym_threshold,
        pagerank_alpha=args.ppr_damping,
        temperature=args.temperature,
        dense_passage_weight=args.dense_passage_weight,
        graph_passage_weight=args.graph_passage_weight,
        fact_passage_weight=args.fact_passage_weight,
        fact_entity_spread_weight=args.fact_entity_spread_weight,
        bridge_entity_top_k=args.bridge_entity_top_k,
        passage_node_weight=args.passage_node_weight,
        enable_sentence_layer=not variant_flags.get("disable_sentence_layer", False),
        enable_intent_routing=not variant_flags.get("disable_intent_routing", False),
        enable_granularity_biased_transition=not variant_flags.get("disable_biased_transition", False),
        enable_recognition_filter=not args.disable_recognition_filter,
        enable_chunk_bridges=not args.disable_chunk_bridges,
        enable_alias_linking=not args.disable_alias_linking,
        enable_llm_judge=args.enable_llm_judge,
    )


def prebuild_shared_indexes(
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    index_pool_name: str,
    builder_flags: Dict[str, bool],
    logger: logging.Logger,
) -> Dict[str, Any]:
    from main_holorag import convert_payload_to_documents
    from src.holorag import HoloRAG

    index_root = run_dir / f"shared_indexes_{index_pool_name}"
    index_root.mkdir(parents=True, exist_ok=True)

    cfg = build_config(
        args,
        save_dir=str(run_dir / f"shared_index_workdir_{index_pool_name}"),
        variant_flags=builder_flags,
    )
    holorag = HoloRAG(cfg)

    index_records: List[Dict[str, Any]] = []
    index_latencies: List[float] = []
    entity_counts: List[float] = []
    sentence_counts: List[float] = []
    chunk_counts: List[float] = []
    edge_counts: List[float] = []

    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        sample_dir = index_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        index_path = sample_dir / "holorag_index.pkl"
        stats_path = sample_dir / "index_stats.json"

        if not question:
            record = {
                "sample_id": sample_id,
                "valid": False,
                "index_path": str(index_path),
                "index_latency": 0.0,
                "stats": {"edges": 0, "layer_counts": {"entity": 0, "sentence": 0, "chunk": 0}},
            }
            index_records.append(record)
            continue

        documents = convert_payload_to_documents(sample)
        t_idx = time.perf_counter()
        index_result = holorag.index(documents)
        index_latency = time.perf_counter() - t_idx

        with index_path.open("wb") as handle:
            pickle.dump(holorag.state, handle)

        stats = index_result.get("stats", {})
        stats_payload = {
            "sample_id": sample_id,
            "index_latency": index_latency,
            "stats": stats,
        }
        stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        layer_counts = stats.get("layer_counts", {})
        index_latencies.append(index_latency)
        entity_counts.append(_safe_float(layer_counts.get("entity", 0)))
        sentence_counts.append(_safe_float(layer_counts.get("sentence", 0)))
        chunk_counts.append(_safe_float(layer_counts.get("chunk", 0)))
        edge_counts.append(_safe_float(stats.get("edges", 0)))

        record = {
            "sample_id": sample_id,
            "valid": True,
            "index_path": str(index_path),
            "index_latency": index_latency,
            "stats": stats,
        }
        index_records.append(record)
        logger.info(
            "[shared-index][%d/%d] %s | idx=%.3fs | entity=%d sentence=%d chunk=%d edges=%d",
            i,
            len(samples),
            sample_id,
            index_latency,
            int(layer_counts.get("entity", 0)),
            int(layer_counts.get("sentence", 0)),
            int(layer_counts.get("chunk", 0)),
            int(stats.get("edges", 0)),
        )

    total_runtime = time.perf_counter() - t_start
    shared_summary = {
        "index_pool_name": index_pool_name,
        "builder_flags": dict(builder_flags),
        "num_samples": len(index_records),
        "num_valid_samples": sum(1 for item in index_records if item.get("valid")),
        "avg_index_latency": _avg(index_latencies),
        "total_index_runtime": total_runtime,
        "entity_nodes": _avg(entity_counts),
        "sentence_nodes": _avg(sentence_counts),
        "chunk_nodes": _avg(chunk_counts),
        "edges": _avg(edge_counts),
    }
    (run_dir / f"shared_index_summary_{index_pool_name}.json").write_text(
        json.dumps(shared_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / f"shared_index_records_{index_pool_name}.json").write_text(
        json.dumps(index_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "records": index_records,
        "summary": shared_summary,
    }


def run_variant(
    variant_name: str,
    variant_flags: Dict[str, bool],
    samples: Sequence[Dict[str, Any]],
    index_records: Sequence[Dict[str, Any]],
    shared_index_summary: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    from src.holorag import HoloRAG

    variant_dir = run_dir / variant_name
    trace_dir = variant_dir / "query_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = variant_dir / "per_example.jsonl"
    per_example_path.write_text("", encoding="utf-8")

    cfg = build_config(args, save_dir=str(variant_dir / "workdir"), variant_flags=variant_flags)
    holorag = HoloRAG(cfg)

    recalls_1: List[float] = []
    recalls_2: List[float] = []
    recalls_5: List[float] = []
    recalls_10: List[float] = []
    em_scores: List[float] = []
    f1_scores: List[float] = []
    index_latencies: List[float] = [_safe_float(shared_index_summary.get("avg_index_latency", 0.0))]
    retrieval_latencies: List[float] = []
    qa_latencies: List[float] = []
    retrieval_qa_latencies: List[float] = []
    entity_counts: List[float] = []
    sentence_counts: List[float] = []
    chunk_counts: List[float] = []
    edge_counts: List[float] = []
    evidence_tokens: List[float] = []

    t_variant_start = time.perf_counter()
    index_record_by_sample = {
        str(item.get("sample_id", "")): item
        for item in index_records
        if str(item.get("sample_id", "")).strip()
    }
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        if not question:
            continue

        index_record = index_record_by_sample.get(sample_id)
        if not index_record:
            logger.warning("[%s] missing shared index record for sample %s; skip", variant_name, sample_id)
            continue
        if not index_record.get("valid"):
            continue
        index_path = Path(str(index_record.get("index_path", "")))
        if not index_path.exists():
            logger.warning("[%s] shared index path missing for %s: %s", variant_name, sample_id, index_path)
            continue
        with index_path.open("rb") as handle:
            holorag.state = pickle.load(handle)
        holorag.index_path = str(index_path)

        index_latency = _safe_float(index_record.get("index_latency", 0.0))
        index_stats = index_record.get("stats", {})
        layer_counts = index_stats.get("layer_counts", {})
        entity_counts.append(_safe_float(layer_counts.get("entity", 0)))
        sentence_counts.append(_safe_float(layer_counts.get("sentence", 0)))
        chunk_counts.append(_safe_float(layer_counts.get("chunk", 0)))
        edge_counts.append(_safe_float(index_stats.get("edges", 0)))

        t_query_start = time.perf_counter()
        query_result = holorag.query(question)
        query_elapsed = time.perf_counter() - t_query_start

        timing = query_result.get("query_timing", {}) or {}
        retrieval_latency = _safe_float(timing.get("retrieval_latency", query_elapsed))
        qa_latency = _safe_float(timing.get("qa_latency", 0.0))
        query_total_latency = _safe_float(timing.get("query_total_latency", query_elapsed))
        retrieval_latencies.append(retrieval_latency)
        qa_latencies.append(qa_latency)
        retrieval_qa_latencies.append(query_total_latency)

        ranked_passages = query_result.get("ranked_passages", []) or []
        support_list = supporting_paragraphs(sample)
        r1 = recall_at_k(support_list, ranked_passages, 1)
        r2 = recall_at_k(support_list, ranked_passages, 2)
        r5 = recall_at_k(support_list, ranked_passages, 5)
        r10 = recall_at_k(support_list, ranked_passages, 10)
        recalls_1.append(r1)
        recalls_2.append(r2)
        recalls_5.append(r5)
        recalls_10.append(r10)

        predicted_answer = str(query_result.get("predicted_answer", "")).strip()
        gold_answers = build_gold_answers(sample)
        em, f1 = best_em_f1(gold_answers, predicted_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        qa_context = str(query_result.get("evidence", {}).get("qa_context", ""))
        evidence_token_cnt = word_token_count(qa_context)
        evidence_tokens.append(float(evidence_token_cnt))

        trace_payload = dict(query_result)
        trace_payload["_meta"] = {
            "variant": variant_name,
            "sample_id": sample_id,
            "index_latency": index_latency,
            "index_path": str(index_path),
            "retrieval_latency": retrieval_latency,
            "qa_latency": qa_latency,
            "query_total_latency": query_total_latency,
            "evidence_tokens": evidence_token_cnt,
        }
        (trace_dir / f"{sample_id}.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        per_item = {
            "variant": variant_name,
            "query_id": sample_id,
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "predicted_answer": predicted_answer,
            "recall@1": r1,
            "recall@2": r2,
            "recall@5": r5,
            "recall@10": r10,
            "F1": f1,
            "EM": em,
            "index_latency": index_latency,
            "index_path": str(index_path),
            "retrieval_latency": retrieval_latency,
            "qa_latency": qa_latency,
            "retrieval_qa_latency": query_total_latency,
            "entity_nodes": int(layer_counts.get("entity", 0)),
            "sentence_nodes": int(layer_counts.get("sentence", 0)),
            "chunk_nodes": int(layer_counts.get("chunk", 0)),
            "edges": int(index_stats.get("edges", 0)),
            "final_evidence_tokens": evidence_token_cnt,
        }
        with per_example_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(per_item, ensure_ascii=False) + "\n")

        logger.info(
            "[%s][%d/%d] %s | R@5=%.4f | F1=%.4f | EM=%.4f | idx=%.3fs | retr=%.3fs | qa=%.3fs",
            variant_name,
            i,
            len(samples),
            sample_id,
            r5,
            f1,
            em,
            index_latency,
            retrieval_latency,
            qa_latency,
        )

    query_runtime = time.perf_counter() - t_variant_start
    shared_index_runtime = _safe_float(shared_index_summary.get("total_index_runtime", 0.0))
    total_runtime = query_runtime + shared_index_runtime
    metrics = {
        "variant": variant_name,
        "num_queries": len(recalls_5),
        "recall@1": _avg(recalls_1),
        "recall@2": _avg(recalls_2),
        "recall@5": _avg(recalls_5),
        "recall@10": _avg(recalls_10),
        "F1": _avg(f1_scores),
        "EM": _avg(em_scores),
        "index_latency": _avg(index_latencies),
        "retrieval_latency": _avg(retrieval_latencies),
        "qa_latency": _avg(qa_latencies),
        "retrieval_qa_latency": _avg(retrieval_qa_latencies),
        "query_runtime": query_runtime,
        "shared_index_runtime": shared_index_runtime,
        "total_runtime": total_runtime,
        "entity_nodes": _avg(entity_counts),
        "sentence_nodes": _avg(sentence_counts),
        "chunk_nodes": _avg(chunk_counts),
        "edges": _avg(edge_counts),
        "final_evidence_tokens": _avg(evidence_tokens),
        "variant_flags": variant_flags,
    }
    (variant_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("seed42_ablation_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline + ablations on MuSiQue samples.")
    parser.add_argument("--dataset_file", type=str, default=str(REPO_ROOT / "reproduce" / "dataset" / "musique_ans_v1.0_dev.jsonl"))
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_queries", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "musique_eval" / "ablation_runs"))
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")

    parser.add_argument("--embedding_batch_size", type=int, default=4)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--entity_max_length", type=int, default=64)
    parser.add_argument("--sentence_max_length", type=int, default=256)
    parser.add_argument("--chunk_max_length", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=128)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--fact_candidate_top_k", type=int, default=24)
    parser.add_argument("--topk_triples", type=int, default=5)
    parser.add_argument("--topk_passages", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--fact_top_k", type=int, default=None)
    parser.add_argument("--fact_rerank_top_k", type=int, default=None)
    parser.add_argument("--fact_output_top_k", type=int, default=None)
    parser.add_argument("--passage_output_top_k", type=int, default=None)
    parser.add_argument("--qa_passage_top_k", type=int, default=None)
    parser.add_argument("--synonym_threshold", type=float, default=0.8)
    parser.add_argument("--ppr_damping", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dense_passage_weight", type=float, default=0.9)
    parser.add_argument("--graph_passage_weight", type=float, default=0.1)
    parser.add_argument("--fact_passage_weight", type=float, default=0.0)
    parser.add_argument("--fact_entity_spread_weight", type=float, default=0.30)
    parser.add_argument("--bridge_entity_top_k", type=int, default=6)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--disable_recognition_filter", action="store_true")
    parser.add_argument("--disable_chunk_bridges", action="store_true")
    parser.add_argument("--disable_alias_linking", action="store_true")
    parser.add_argument("--enable_llm_judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_name = args.run_name.strip() or f"seed{args.seed}_{args.num_eval_queries}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")

    all_samples = load_jsonl(args.dataset_file)
    filtered_samples = maybe_filter_split(all_samples, args.split)
    if not filtered_samples:
        raise ValueError("No samples available after split filtering.")
    samples = sample_queries(filtered_samples, seed=args.seed, num_eval_queries=args.num_eval_queries)

    sampled_payload = {
        "seed": args.seed,
        "num_eval_queries": args.num_eval_queries,
        "split": args.split,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_ids": [str(sample.get("id", f"sample_{idx+1}")) for idx, sample in enumerate(samples)],
    }
    (run_dir / "sampled_queries.json").write_text(json.dumps(sampled_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    variants: List[Tuple[str, Dict[str, bool]]] = [
        ("baseline", {}),
        ("wo_sentence_layer", {"disable_sentence_layer": True}),
        ("wo_intent", {"disable_intent_routing": True}),
        ("wo_biased_transition", {"disable_biased_transition": True}),
    ]

    logger.info("Prebuilding shared FULL indexes once (for baseline / wo_intent / wo_biased_transition)...")
    shared_index_data_full = prebuild_shared_indexes(
        samples=samples,
        args=args,
        run_dir=run_dir,
        index_pool_name="full",
        builder_flags={},
        logger=logger,
    )
    logger.info("Prebuilding shared NO-SENTENCE indexes once (for wo_sentence_layer)...")
    shared_index_data_no_sentence = prebuild_shared_indexes(
        samples=samples,
        args=args,
        run_dir=run_dir,
        index_pool_name="wo_sentence_layer",
        builder_flags={"disable_sentence_layer": True},
        logger=logger,
    )
    shared_index_records_full = shared_index_data_full["records"]
    shared_index_summary_full = shared_index_data_full["summary"]
    shared_index_records_no_sentence = shared_index_data_no_sentence["records"]
    shared_index_summary_no_sentence = shared_index_data_no_sentence["summary"]
    logger.info(
        "Shared FULL index done | valid=%d/%d | avg_index_latency=%.3fs | total_index_runtime=%.2fs",
        int(shared_index_summary_full.get("num_valid_samples", 0)),
        int(shared_index_summary_full.get("num_samples", 0)),
        float(shared_index_summary_full.get("avg_index_latency", 0.0)),
        float(shared_index_summary_full.get("total_index_runtime", 0.0)),
    )
    logger.info(
        "Shared NO-SENTENCE index done | valid=%d/%d | avg_index_latency=%.3fs | total_index_runtime=%.2fs",
        int(shared_index_summary_no_sentence.get("num_valid_samples", 0)),
        int(shared_index_summary_no_sentence.get("num_samples", 0)),
        float(shared_index_summary_no_sentence.get("avg_index_latency", 0.0)),
        float(shared_index_summary_no_sentence.get("total_index_runtime", 0.0)),
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_file": args.dataset_file,
                "split": args.split,
                "seed": args.seed,
                "num_eval_queries": args.num_eval_queries,
                "llm_base_url": args.llm_base_url,
                "llm_name": args.llm_name,
                "embedding_name": args.embedding_name,
                "embedding_device": args.embedding_device,
                "qa_passage_top_k": args.qa_passage_top_k,
                "topk_triples": args.topk_triples,
                "topk_passages": args.topk_passages,
                "synonym_threshold": args.synonym_threshold,
                "ppr_damping": args.ppr_damping,
                "temperature": args.temperature,
                "passage_node_weight": args.passage_node_weight,
                "dense_passage_weight": args.dense_passage_weight,
                "graph_passage_weight": args.graph_passage_weight,
                "fact_passage_weight": args.fact_passage_weight,
                "effective_fact_top_k": args.fact_top_k if args.fact_top_k is not None else args.topk_triples,
                "effective_fact_rerank_top_k": args.fact_rerank_top_k if args.fact_rerank_top_k is not None else args.topk_triples,
                "effective_fact_output_top_k": args.fact_output_top_k if args.fact_output_top_k is not None else args.topk_triples,
                "effective_qa_passage_top_k": args.qa_passage_top_k if args.qa_passage_top_k is not None else args.topk_passages,
                "index_mode": "two_shared_pools",
                "shared_index_summary_full": shared_index_summary_full,
                "shared_index_summary_wo_sentence_layer": shared_index_summary_no_sentence,
                "variants": [{"name": name, "flags": flags} for name, flags in variants],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows: List[Dict[str, Any]] = []
    for variant_name, flags in variants:
        logger.info("Starting variant: %s", variant_name)
        if variant_name == "wo_sentence_layer":
            selected_index_records = shared_index_records_no_sentence
            selected_index_summary = shared_index_summary_no_sentence
            selected_index_pool = "wo_sentence_layer"
        else:
            selected_index_records = shared_index_records_full
            selected_index_summary = shared_index_summary_full
            selected_index_pool = "full"
        metrics = run_variant(
            variant_name=variant_name,
            variant_flags=flags,
            samples=samples,
            index_records=selected_index_records,
            shared_index_summary=selected_index_summary,
            args=args,
            run_dir=run_dir,
            logger=logger,
        )
        metrics["index_pool"] = selected_index_pool
        summary_rows.append(metrics)
        logger.info(
            "Finished %s | R@5=%.4f | F1=%.4f | EM=%.4f | total_runtime=%.2fs",
            variant_name,
            metrics["recall@5"],
            metrics["F1"],
            metrics["EM"],
            metrics["total_runtime"],
        )

    summary_json_path = run_dir / "metrics_summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_headers = [
        "variant",
        "index_pool",
        "num_queries",
        "recall@1",
        "recall@2",
        "recall@5",
        "recall@10",
        "F1",
        "EM",
        "index_latency",
        "retrieval_latency",
        "qa_latency",
        "retrieval_qa_latency",
        "query_runtime",
        "shared_index_runtime",
        "total_runtime",
        "entity_nodes",
        "sentence_nodes",
        "chunk_nodes",
        "edges",
        "final_evidence_tokens",
    ]
    summary_csv_path = run_dir / "metrics_summary.csv"
    csv_lines = [",".join(csv_headers)]
    for row in summary_rows:
        csv_lines.append(",".join(str(row.get(col, "")) for col in csv_headers))
    summary_csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    logger.info("Saved summary json: %s", summary_json_path)
    logger.info("Saved summary csv: %s", summary_csv_path)


if __name__ == "__main__":
    main()
