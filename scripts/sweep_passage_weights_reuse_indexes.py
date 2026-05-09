import argparse
import json
import pickle
import random
import re
import string
import sys
import time
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


def _avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_weight_grid(grid_text: str) -> List[Tuple[float, float, float]]:
    triplets: List[Tuple[float, float, float]] = []
    for item in str(grid_text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid weight triplet: {item}")
        dense, graph, fact = (float(parts[0]), float(parts[1]), float(parts[2]))
        if abs((dense + graph + fact) - 1.0) > 1e-6:
            raise ValueError(f"Triplet must sum to 1.0, got: {item}")
        triplets.append((dense, graph, fact))
    if not triplets:
        raise ValueError("No valid weight triplets provided.")
    return triplets


def build_config(args: argparse.Namespace, save_dir: str, dense: float, graph: float, fact: float) -> Any:
    from src.holorag import HoloRAGConfig

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
        retrieval_top_k=args.retrieval_top_k,
        fact_top_k=args.fact_top_k,
        fact_rerank_top_k=args.fact_rerank_top_k,
        fact_output_top_k=args.fact_output_top_k,
        passage_output_top_k=args.passage_output_top_k,
        qa_passage_top_k=args.qa_passage_top_k,
        dense_passage_weight=dense,
        graph_passage_weight=graph,
        fact_passage_weight=fact,
        fact_entity_spread_weight=args.fact_entity_spread_weight,
        bridge_entity_top_k=args.bridge_entity_top_k,
        passage_node_weight=args.passage_node_weight,
        enable_sentence_layer=True,
        enable_intent_routing=True,
        enable_granularity_biased_transition=True,
        enable_recognition_filter=not args.disable_recognition_filter,
        enable_chunk_bridges=not args.disable_chunk_bridges,
        enable_alias_linking=not args.disable_alias_linking,
        enable_llm_judge=args.enable_llm_judge,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep passage fusion weights with reused shared indexes.")
    parser.add_argument("--dataset_file", type=str, default=str(REPO_ROOT / "reproduce" / "dataset" / "musique_ans_v1.0_dev.jsonl"))
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampled_queries_json", type=str, required=True)
    parser.add_argument("--shared_indexes_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Semicolon-separated triplets, e.g. '0.55,0.30,0.15;0.60,0.25,0.15'")

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
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--fact_top_k", type=int, default=12)
    parser.add_argument("--fact_rerank_top_k", type=int, default=8)
    parser.add_argument("--fact_output_top_k", type=int, default=8)
    parser.add_argument("--passage_output_top_k", type=int, default=10)
    parser.add_argument("--qa_passage_top_k", type=int, default=5)
    parser.add_argument("--fact_entity_spread_weight", type=float, default=0.30)
    parser.add_argument("--bridge_entity_top_k", type=int, default=6)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--disable_recognition_filter", action="store_true")
    parser.add_argument("--disable_chunk_bridges", action="store_true")
    parser.add_argument("--disable_alias_linking", action="store_true")
    parser.add_argument("--enable_llm_judge", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_indexes_dir = Path(args.shared_indexes_dir).expanduser().resolve()
    sampled_queries_json = Path(args.sampled_queries_json).expanduser().resolve()

    payload = json.loads(sampled_queries_json.read_text(encoding="utf-8"))
    sample_ids = payload.get("sample_ids", [])
    if not sample_ids:
        raise ValueError(f"No sample_ids in {sampled_queries_json}")
    sample_id_set = {str(item) for item in sample_ids}

    all_samples = load_jsonl(args.dataset_file)
    filtered = maybe_filter_split(all_samples, args.split)
    samples = [sample for sample in filtered if str(sample.get("id", "")) in sample_id_set]
    samples.sort(key=lambda x: sample_ids.index(str(x.get("id", ""))))

    from src.holorag import HoloRAG

    weights = parse_weight_grid(args.weights)
    summary_rows: List[Dict[str, Any]] = []

    for dense, graph, fact in weights:
        tag = f"d{dense:.2f}_g{graph:.2f}_f{fact:.2f}"
        run_dir = output_dir / tag
        trace_dir = run_dir / "query_traces"
        workdir = run_dir / "workdir"
        trace_dir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)

        cfg = build_config(args=args, save_dir=str(workdir), dense=dense, graph=graph, fact=fact)
        holorag = HoloRAG(cfg)

        recalls_1: List[float] = []
        recalls_2: List[float] = []
        recalls_5: List[float] = []
        recalls_10: List[float] = []
        em_scores: List[float] = []
        f1_scores: List[float] = []
        retrieval_latencies: List[float] = []
        qa_latencies: List[float] = []
        total_latencies: List[float] = []

        t0 = time.perf_counter()
        for idx, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("id", f"sample_{idx}"))
            question = str(sample.get("question", "")).strip()
            if not question:
                continue
            index_path = shared_indexes_dir / sample_id / "holorag_index.pkl"
            if not index_path.exists():
                continue

            with index_path.open("rb") as handle:
                holorag.state = pickle.load(handle)
            holorag.index_path = str(index_path)

            result = holorag.query(question)
            timing = result.get("query_timing", {}) or {}
            retrieval_latencies.append(float(timing.get("retrieval_latency", 0.0)))
            qa_latencies.append(float(timing.get("qa_latency", 0.0)))
            total_latencies.append(float(timing.get("query_total_latency", 0.0)))

            ranked_passages = result.get("ranked_passages", []) or []
            support_list = supporting_paragraphs(sample)
            recalls_1.append(recall_at_k(support_list, ranked_passages, 1))
            recalls_2.append(recall_at_k(support_list, ranked_passages, 2))
            recalls_5.append(recall_at_k(support_list, ranked_passages, 5))
            recalls_10.append(recall_at_k(support_list, ranked_passages, 10))

            predicted_answer = str(result.get("predicted_answer", "")).strip()
            em, f1 = best_em_f1(build_gold_answers(sample), predicted_answer)
            em_scores.append(em)
            f1_scores.append(f1)

            (trace_dir / f"{sample_id}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        elapsed = time.perf_counter() - t0
        metrics = {
            "tag": tag,
            "dense_passage_weight": dense,
            "graph_passage_weight": graph,
            "fact_passage_weight": fact,
            "num_queries": len(recalls_5),
            "recall@1": _avg(recalls_1),
            "recall@2": _avg(recalls_2),
            "recall@5": _avg(recalls_5),
            "recall@10": _avg(recalls_10),
            "F1": _avg(f1_scores),
            "EM": _avg(em_scores),
            "retrieval_latency": _avg(retrieval_latencies),
            "qa_latency": _avg(qa_latencies),
            "retrieval_qa_latency": _avg(total_latencies),
            "query_runtime": elapsed,
            "shared_indexes_dir": str(shared_indexes_dir),
            "sampled_queries_json": str(sampled_queries_json),
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(metrics)
        print(f"[{tag}] R@5={metrics['recall@5']:.4f} F1={metrics['F1']:.4f} EM={metrics['EM']:.4f} N={metrics['num_queries']}")

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

