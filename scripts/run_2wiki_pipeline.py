#!/usr/bin/env python3
import argparse
import json
import random
import re
import string
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DEFAULT = REPO_ROOT / "reproduce" / "dataset" / "2wikimultihopqa.json"


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
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
    pred_norm = normalize_answer(pred)
    best_em = 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        em = 1.0 if gold_norm == pred_norm else 0.0
        f1 = token_f1(gold, pred)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    return best_em, best_f1


def build_sample(item: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    supporting_titles = {str(x[0]) for x in item.get("supporting_facts", []) if isinstance(x, list) and x}
    paragraphs: List[Dict[str, Any]] = []
    idx = 0
    for entry in item.get("context", []):
        if not (isinstance(entry, list) and len(entry) >= 2):
            continue
        title = str(entry[0])
        sentences = entry[1] if isinstance(entry[1], list) else []
        paragraph_text = " ".join(str(s) for s in sentences)
        paragraphs.append(
            {
                "idx": idx,
                "title": title,
                "paragraph_text": paragraph_text,
                "is_supporting": title in supporting_titles,
            }
        )
        idx += 1
    sample_id = str(item.get("_id") or item.get("id") or f"2wiki_{fallback_idx:04d}")
    return {
        "id": sample_id,
        "question": str(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "answer_aliases": [],
        "paragraphs": paragraphs,
    }


def recall_at_k(sample: Dict[str, Any], ranked_passages: Sequence[Dict[str, Any]], k: int) -> float:
    supports = [p for p in sample.get("paragraphs", []) if p.get("is_supporting")]
    support_ids = {str(p.get("idx")) for p in supports if p.get("idx") is not None}
    topk = list(ranked_passages[:k])

    retrieved_ids = set()
    for r in topk:
        for key in ("passage_index", "idx", "id", "paragraph_id"):
            if key in r and r[key] is not None:
                retrieved_ids.add(str(r[key]))
                break

    if support_ids and retrieved_ids:
        return len(support_ids & retrieved_ids) / max(1, len(support_ids))

    support_titles = {normalize_answer(str(p.get("title", ""))) for p in supports if str(p.get("title", "")).strip()}
    retrieved_titles = {normalize_answer(str(r.get("title", ""))) for r in topk if str(r.get("title", "")).strip()}
    if not support_titles:
        return 0.0
    return len(support_titles & retrieved_titles) / len(support_titles)


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2Wiki end-to-end pipeline: sample prep + index/query + evaluation.")
    parser.add_argument("--dataset_file", type=str, default=str(DATASET_DEFAULT))
    parser.add_argument("--num_eval_queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for random sampling; if omitted, sampling is non-deterministic.")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(REPO_ROOT / "outputs" / "2wiki_eval"))
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--embedding_visible_devices", type=str, default="2")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--retrieval_k", type=int, default=5)
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
    parser.add_argument("--qa_passage_top_k", type=int, default=3)
    parser.add_argument("--dense_passage_weight", type=float, default=0.9)
    parser.add_argument("--graph_passage_weight", type=float, default=0.1)
    parser.add_argument("--fact_passage_weight", type=float, default=0.0)
    parser.add_argument("--fact_entity_spread_weight", type=float, default=0.30)
    parser.add_argument("--bridge_entity_top_k", type=int, default=6)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--disable_sentence_layer", action="store_true")
    parser.add_argument("--disable_recognition_filter", action="store_true")
    parser.add_argument("--disable_intent_routing", action="store_true")
    parser.add_argument("--disable_chunk_bridges", action="store_true")
    parser.add_argument("--disable_alias_linking", action="store_true")
    parser.add_argument("--disable_biased_transition", action="store_true")
    parser.add_argument("--enable_llm_judge", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_file)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid dataset format: {dataset_path}")

    run_name = args.run_name.strip() or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    samples_dir = run_dir / "samples"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    n = min(args.num_eval_queries, len(data))
    chosen = rng.sample(data, n) if n < len(data) else list(data)

    sampled_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": str(dataset_path),
        "dataset_size": len(data),
        "num_eval_queries": n,
        "seed": args.seed,
    }
    (run_dir / "sampled_queries.json").write_text(
        json.dumps({"meta": sampled_meta, "samples": [item.get("_id") for item in chosen]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_example: List[Dict[str, Any]] = []
    env = dict(**__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = args.embedding_visible_devices
    shared_holorag_args = [
        "--llm_base_url",
        args.llm_base_url,
        "--llm_name",
        args.llm_name,
        "--embedding_name",
        args.embedding_name,
        "--embedding_device",
        args.embedding_device,
        "--embedding_batch_size",
        str(args.embedding_batch_size),
        "--embedding_max_seq_len",
        str(args.embedding_max_seq_len),
        "--embedding_dtype",
        args.embedding_dtype,
        "--entity_max_length",
        str(args.entity_max_length),
        "--sentence_max_length",
        str(args.sentence_max_length),
        "--chunk_max_length",
        str(args.chunk_max_length),
        "--query_max_length",
        str(args.query_max_length),
        "--linking_top_k",
        str(args.linking_top_k),
        "--fact_candidate_top_k",
        str(args.fact_candidate_top_k),
        "--retrieval_top_k",
        str(args.retrieval_top_k),
        "--fact_top_k",
        str(args.fact_top_k),
        "--fact_rerank_top_k",
        str(args.fact_rerank_top_k),
        "--fact_output_top_k",
        str(args.fact_output_top_k),
        "--passage_output_top_k",
        str(args.passage_output_top_k),
        "--qa_passage_top_k",
        str(args.qa_passage_top_k),
        "--dense_passage_weight",
        str(args.dense_passage_weight),
        "--graph_passage_weight",
        str(args.graph_passage_weight),
        "--fact_passage_weight",
        str(args.fact_passage_weight),
        "--fact_entity_spread_weight",
        str(args.fact_entity_spread_weight),
        "--bridge_entity_top_k",
        str(args.bridge_entity_top_k),
        "--passage_node_weight",
        str(args.passage_node_weight),
    ]
    if args.disable_sentence_layer:
        shared_holorag_args.append("--disable_sentence_layer")
    if args.disable_recognition_filter:
        shared_holorag_args.append("--disable_recognition_filter")
    if args.disable_intent_routing:
        shared_holorag_args.append("--disable_intent_routing")
    if args.disable_chunk_bridges:
        shared_holorag_args.append("--disable_chunk_bridges")
    if args.disable_alias_linking:
        shared_holorag_args.append("--disable_alias_linking")
    if args.disable_biased_transition:
        shared_holorag_args.append("--disable_biased_transition")
    if args.enable_llm_judge:
        shared_holorag_args.append("--enable_llm_judge")

    running_r5: List[float] = []
    running_f1: List[float] = []
    for i, raw in enumerate(chosen, start=1):
        print(f"[{i}/{n}] start", flush=True)
        sample = build_sample(raw, i)
        sample_name = f"sample_2wiki{i:04d}"
        sample_path = samples_dir / f"{sample_name}.json"
        sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

        sample_out = results_dir / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        index_file = sample_out / "holorag_index.pkl"
        query_file = sample_out / "last_query_result.json"

        index_s = 0.0
        query_s = 0.0
        if args.skip_existing and index_file.exists() and query_file.exists():
            timing_file = logs_dir / f"{sample_name}_timing.json"
            if timing_file.exists():
                timing_payload = json.loads(timing_file.read_text(encoding="utf-8"))
                index_s = float(timing_payload.get("index_s", 0.0) or 0.0)
                query_s = float(timing_payload.get("query_s", 0.0) or 0.0)
        else:
            t0 = time.perf_counter()
            run_cmd(
                [
                    sys.executable,
                    str(REPO_ROOT / "main_holorag.py"),
                    "index",
                    "--corpus_file",
                    str(sample_path),
                    "--output_dir",
                    str(sample_out),
                ]
                + shared_holorag_args,
                env=env,
            )
            t1 = time.perf_counter()
            run_cmd(
                [
                    sys.executable,
                    str(REPO_ROOT / "main_holorag.py"),
                    "query",
                    "--corpus_file",
                    str(sample_path),
                    "--output_dir",
                    str(sample_out),
                ]
                + shared_holorag_args,
                env=env,
            )
            t2 = time.perf_counter()
            index_s = t1 - t0
            query_s = t2 - t1
            (logs_dir / f"{sample_name}_timing.json").write_text(
                json.dumps({"index_s": index_s, "query_s": query_s}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        result = json.loads(query_file.read_text(encoding="utf-8"))
        ranked = result.get("ranked_passages", [])
        predicted = str(result.get("predicted_answer", "")).strip()
        gold_answers = [sample.get("answer", "")]
        em, f1 = best_em_f1(gold_answers, predicted)
        r1 = recall_at_k(sample, ranked, 1)
        r2 = recall_at_k(sample, ranked, 2)
        r5 = recall_at_k(sample, ranked, 5)
        rk = recall_at_k(sample, ranked, args.retrieval_k)
        running_r5.append(r5)
        running_f1.append(f1)
        ans_ok = 1 if predicted else 0

        per_example.append(
            {
                "sample_name": sample_name,
                "sample_id": sample.get("id"),
                "question": sample.get("question", ""),
                "gold_answer": sample.get("answer", ""),
                "predicted_answer": predicted,
                "em": em,
                "f1": f1,
                "recall@1": r1,
                "recall@2": r2,
                "recall@5": r5,
                f"recall@{args.retrieval_k}": rk,
                "index_path": str(index_file),
                "result_path": str(query_file),
                "answer_non_empty": bool(ans_ok),
                "index_s": index_s,
                "query_s": query_s,
                "total_s": index_s + query_s,
            }
        )

        print(
            f"[{i}/{n}] {sample_name} "
            f"| ans_ok={ans_ok} | r5={r5:.4f} | f1={f1:.4f} "
            f"| run_r5={mean(running_r5):.4f} | run_f1={mean(running_f1):.4f} "
            f"| index={index_s:.2f}s | query={query_s:.2f}s | total={index_s + query_s:.2f}s"
            ,
            flush=True,
        )

    summary = {
        "num_examples": len(per_example),
        "em": round(mean(item["em"] for item in per_example), 4) if per_example else 0.0,
        "f1": round(mean(item["f1"] for item in per_example), 4) if per_example else 0.0,
        "recall@1": round(mean(item["recall@1"] for item in per_example), 4) if per_example else 0.0,
        "recall@2": round(mean(item["recall@2"] for item in per_example), 4) if per_example else 0.0,
        "recall@5": round(mean(item["recall@5"] for item in per_example), 4) if per_example else 0.0,
        f"recall@{args.retrieval_k}": round(mean(item[f"recall@{args.retrieval_k}"] for item in per_example), 4) if per_example else 0.0,
    }

    payload = {
        "meta": {
            **sampled_meta,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "llm_base_url": args.llm_base_url,
            "llm_name": args.llm_name,
            "embedding_name": args.embedding_name,
            "embedding_device": args.embedding_device,
            "embedding_visible_devices": args.embedding_visible_devices,
        },
        "summary": summary,
        "per_example": per_example,
    }

    (run_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved to: {run_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
