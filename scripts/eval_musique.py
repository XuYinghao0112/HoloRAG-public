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
        if not support_ids:
            return 0.0
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


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("musique_eval")
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


def maybe_filter_split(samples: Sequence[Dict[str, Any]], split: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    if not split:
        return list(samples)
    split_key = None
    for key in ("split", "subset"):
        if any(key in sample for sample in samples):
            split_key = key
            break
    if split_key is None:
        logger.info("No explicit split field found in dataset; using all samples and recording split='%s'.", split)
        return list(samples)
    filtered = [sample for sample in samples if str(sample.get(split_key, "")) == split]
    logger.info("Filtered dataset by %s='%s': %d -> %d", split_key, split, len(samples), len(filtered))
    return filtered


def sample_queries(
    samples: Sequence[Dict[str, Any]],
    sampled_path: Path,
    seed: int,
    num_eval_queries: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    if sampled_path.exists():
        payload = json.loads(sampled_path.read_text(encoding="utf-8"))
        sampled = payload.get("samples", payload)
        logger.info("Loaded existing sampled queries: %s (%d samples)", sampled_path, len(sampled))
        return sampled

    rng = random.Random(seed)
    available = list(samples)
    if num_eval_queries >= len(available):
        sampled = available
        logger.info("Requested %d queries but dataset has %d; using all.", num_eval_queries, len(available))
    else:
        sampled = rng.sample(available, num_eval_queries)
        logger.info("Sampled %d queries from %d with seed=%d.", num_eval_queries, len(available), seed)

    sampled_payload = {
        "seed": seed,
        "num_eval_queries": num_eval_queries,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples": sampled,
    }
    sampled_path.write_text(json.dumps(sampled_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sampled


def build_config(args: argparse.Namespace, save_dir: str) -> Any:
    from src.holorag import HoloRAGConfig

    return HoloRAGConfig(
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        save_dir=save_dir,
        embedding_device=args.embedding_device,
        linking_top_k=args.topk_triples,
        passage_output_top_k=max(args.topk_passages, 10),
        qa_passage_top_k=args.topk_passages,
        retrieval_top_k=max(args.retrieval_top_k, args.topk_passages),
        entity_alias_threshold=args.synonym_threshold,
        pagerank_alpha=args.ppr_damping,
        temperature=args.temperature,
        passage_node_weight=args.passage_node_weight,
        dense_passage_weight=args.dense_passage_weight,
        graph_passage_weight=args.graph_passage_weight,
        fact_passage_weight=args.fact_passage_weight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HoloRAG on MuSiQue with retrieval-only or retrieval+QA modes.")
    parser.add_argument("--dataset_file", type=str, default=str(REPO_ROOT / "reproduce" / "dataset" / "musique_ans_v1.0_dev.jsonl"))
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_queries", type=int, default=1000)
    parser.add_argument("--eval_qa", action="store_true", help="Enable QA generation/evaluation on top of retrieved passages.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "musique_eval" / "runs"),
        help=(
            "Root directory for evaluation run artifacts. "
            "Runs are written to <output_dir>/<run_name>. "
            "For legacy compatibility, if output_dir does not end with 'runs', artifacts are redirected to <output_dir>/runs/<run_name>."
        ),
    )
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")

    parser.add_argument("--topk_triples", type=int, default=5)
    parser.add_argument("--topk_passages", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--synonym_threshold", type=float, default=0.8)
    parser.add_argument("--ppr_damping", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--dense_passage_weight", type=float, default=0.55)
    parser.add_argument("--graph_passage_weight", type=float, default=0.30)
    parser.add_argument("--fact_passage_weight", type=float, default=0.15)
    parser.add_argument(
        "--reuse_indexes_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "musique_eval" / "runs" / "indexes"),
        help="Directory containing reusable per-sample indexes under <run_name>/<sample_id>/holorag_index.pkl.",
    )
    parser.add_argument(
        "--reuse_indexes_run_name",
        type=str,
        default="",
        help="Optional run name under --reuse_indexes_dir to restrict index lookup.",
    )
    parser.add_argument(
        "--allow_reindex_fallback",
        action="store_true",
        help="If reusable index is missing for a sample, rebuild index on the fly. Disabled by default.",
    )
    parser.add_argument(
        "--writeback_rebuilt_index",
        action="store_true",
        help="When --allow_reindex_fallback is enabled, persist rebuilt indexes into --reuse_indexes_dir.",
    )
    parser.add_argument(
        "--writeback_run_name",
        type=str,
        default="",
        help="Run folder name under --reuse_indexes_dir for rebuilt indexes. Defaults to --reuse_indexes_run_name.",
    )
    return parser.parse_args()


def _load_pickle_file(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _parse_saved_at_key(value: str) -> Tuple[int, str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return (0, "")
    return (1, cleaned)


def resolve_reusable_index_path(
    reuse_root: Path,
    sample_id: str,
    question: str,
    preferred_run_name: str = "",
) -> Optional[Path]:
    if not reuse_root.exists():
        return None

    candidate_metadata_paths: List[Path] = []
    if preferred_run_name:
        candidate = reuse_root / preferred_run_name / sample_id / "metadata.json"
        if candidate.exists():
            candidate_metadata_paths.append(candidate)
    else:
        candidate_metadata_paths.extend(reuse_root.glob(f"*/{sample_id}/metadata.json"))
    if not candidate_metadata_paths:
        return None

    question_norm = " ".join(str(question or "").split())
    candidate_records: List[Tuple[Tuple[int, str], Path]] = []
    for metadata_path in candidate_metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata_question = " ".join(str(metadata.get("question", "")).split())
        if question_norm and metadata_question and metadata_question != question_norm:
            continue
        reusable_path = Path(str(metadata.get("reusable_index_path", "")).strip())
        if not reusable_path.exists():
            fallback_path = metadata_path.with_name("holorag_index.pkl")
            reusable_path = fallback_path if fallback_path.exists() else reusable_path
        if not reusable_path.exists():
            continue
        candidate_records.append((_parse_saved_at_key(metadata.get("saved_at", "")), reusable_path))

    if not candidate_records:
        return None
    candidate_records.sort(key=lambda item: item[0], reverse=True)
    return candidate_records[0][1]


def main() -> None:
    args = parse_args()

    from extract_musique_sample import load_jsonl
    from main_holorag import convert_payload_to_documents
    from src.holorag import HoloRAG
    set_global_seed(args.seed)

    run_name = args.run_name.strip() or f"musique_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_dir)
    if output_root.name != "runs":
        output_root = output_root / "runs"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path)

    logger.info("Starting MuSiQue evaluation run: %s", run_name)
    logger.info("Output dir: %s", run_dir)

    sampled_path = run_dir / "sampled_queries.json"
    per_example_path = run_dir / "per_example_results.jsonl"
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.json"
    samples_dump_dir = run_dir / "samples"
    traces_dump_dir = run_dir / "query_traces"
    samples_dump_dir.mkdir(parents=True, exist_ok=True)
    traces_dump_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_jsonl(args.dataset_file)
    samples = maybe_filter_split(all_samples, args.split, logger)
    if not samples:
        raise ValueError("No samples available after split filtering.")

    sampled_samples = sample_queries(
        samples=samples,
        sampled_path=sampled_path,
        seed=args.seed,
        num_eval_queries=args.num_eval_queries,
        logger=logger,
    )

    config_payload = {
        "dataset": "MuSiQue",
        "dataset_file": args.dataset_file,
        "split": args.split,
        "seed": args.seed,
        "num_eval_queries": args.num_eval_queries,
        "eval_qa": args.eval_qa,
        "topk_triples": args.topk_triples,
        "topk_passages": args.topk_passages,
        "synonym_threshold": args.synonym_threshold,
        "ppr_damping": args.ppr_damping,
        "temperature": args.temperature,
        "passage_node_weight": args.passage_node_weight,
        "dense_passage_weight": args.dense_passage_weight,
        "graph_passage_weight": args.graph_passage_weight,
        "fact_passage_weight": args.fact_passage_weight,
        "retrieval_top_k": args.retrieval_top_k,
        "reuse_indexes_dir": args.reuse_indexes_dir,
        "reuse_indexes_run_name": args.reuse_indexes_run_name,
        "allow_reindex_fallback": bool(args.allow_reindex_fallback),
        "writeback_rebuilt_index": bool(args.writeback_rebuilt_index),
        "writeback_run_name": args.writeback_run_name,
        "llm_base_url": args.llm_base_url,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "embedding_device": args.embedding_device,
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    recalls_1: List[float] = []
    recalls_2: List[float] = []
    recalls_5: List[float] = []
    recalls_10: List[float] = []
    em_scores: List[float] = []
    f1_scores: List[float] = []
    index_latencies: List[float] = []
    retrieval_latencies: List[float] = []
    per_example_path.write_text("", encoding="utf-8")

    t_run_start = time.perf_counter()

    shared_workdir = run_dir / "workdir" / "shared_runtime"
    shared_workdir.mkdir(parents=True, exist_ok=True)
    holorag = HoloRAG(build_config(args, save_dir=str(shared_workdir)))
    logger.info("Initialized HoloRAG once and will reuse model weights for all samples.")
    reuse_root = Path(args.reuse_indexes_dir).expanduser().resolve()
    writeback_run_name = (
        args.writeback_run_name.strip()
        or args.reuse_indexes_run_name.strip()
        or f"{run_name}_autofill_indexes"
    )
    logger.info("Reusable index root: %s", reuse_root)
    logger.info("Writeback rebuilt indexes: %s (run=%s)", bool(args.writeback_rebuilt_index), writeback_run_name)

    for i, sample in enumerate(sampled_samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        if not question:
            logger.warning("Skip sample %s: empty question.", sample_id)
            continue
        (samples_dump_dir / f"{sample_id}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("[%d/%d] start sample %s", i, len(sampled_samples), sample_id)

        documents = convert_payload_to_documents(sample)
        llm_before_index = holorag.llm_client.get_stats()
        t_index_start = time.perf_counter()
        reused_index_path = resolve_reusable_index_path(
            reuse_root=reuse_root,
            sample_id=sample_id,
            question=question,
            preferred_run_name=args.reuse_indexes_run_name.strip(),
        )
        index_mode = "reused"
        if reused_index_path is not None:
            holorag.state = _load_pickle_file(reused_index_path)
            holorag.index_path = str(reused_index_path)
        elif args.allow_reindex_fallback:
            index_mode = "rebuilt"
            holorag.index(documents)
            if args.writeback_rebuilt_index and holorag.state is not None:
                writeback_dir = reuse_root / writeback_run_name / sample_id
                writeback_index_path = writeback_dir / "holorag_index.pkl"
                _dump_pickle_file(writeback_index_path, holorag.state)
                metadata_payload = {
                    "run_name": writeback_run_name,
                    "sample_id": sample_id,
                    "question": question,
                    "answer": sample.get("answer"),
                    "answer_aliases": sample.get("answer_aliases", []),
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "source_index_path": str(holorag.index_path),
                    "reusable_index_path": str(writeback_index_path),
                }
                (writeback_dir / "metadata.json").write_text(
                    json.dumps(metadata_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                reused_index_path = writeback_index_path
        else:
            raise FileNotFoundError(
                f"Reusable index not found for sample_id={sample_id}. "
                f"Looked under {reuse_root}. Set --allow_reindex_fallback to permit rebuilding."
            )
        index_latency = time.perf_counter() - t_index_start
        index_latencies.append(index_latency)
        llm_after_index = holorag.llm_client.get_stats()

        llm_before_query = holorag.llm_client.get_stats()
        t_retr_start = time.perf_counter()
        query_result = holorag.query(question)
        retrieval_latency = time.perf_counter() - t_retr_start
        llm_after_query = holorag.llm_client.get_stats()
        retrieval_latencies.append(retrieval_latency)
        sample_trace_dir = traces_dump_dir / sample_id
        sample_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_payload = dict(query_result)
        trace_payload["_meta"] = {
            "sample_id": sample_id,
            "index_mode": index_mode,
            "reused_index_path": str(reused_index_path) if reused_index_path else None,
        }
        (sample_trace_dir / "query_stdout.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (sample_trace_dir / "last_query_result.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        ranked_passages = query_result.get("ranked_passages", [])
        topk_passages = ranked_passages[: args.topk_passages]

        support_list = supporting_paragraphs(sample)
        support_ids = [str(uid) for p in support_list if (uid := _paragraph_uid(p)) is not None]
        support_titles = [str(p.get("title", "")) for p in support_list if str(p.get("title", "")).strip()]

        retrieved_ids = [str(uid) for p in topk_passages if (uid := _retrieved_uid(p)) is not None]
        retrieved_titles = [str(p.get("title", "")) for p in topk_passages if str(p.get("title", "")).strip()]

        r1 = recall_at_k(support_list, ranked_passages, 1)
        r2 = recall_at_k(support_list, ranked_passages, 2)
        r5 = recall_at_k(support_list, ranked_passages, 5)
        r10 = recall_at_k(support_list, ranked_passages, 10)
        recalls_1.append(r1)
        recalls_2.append(r2)
        recalls_5.append(r5)
        recalls_10.append(r10)

        predicted_answer = ""
        em = None
        f1 = None
        qa_latency = None
        if args.eval_qa:
            predicted_answer = str(query_result.get("predicted_answer", "")).strip()
            gold_answers = build_gold_answers(sample)
            em, f1 = best_em_f1(gold_answers, predicted_answer)
            em_scores.append(em)
            f1_scores.append(f1)

        per_item = {
            "query_id": sample_id,
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "predicted_answer": predicted_answer,
            "retrieved_passage_ids": retrieved_ids,
            "retrieved_passage_titles": retrieved_titles,
            "supporting_passage_ids": support_ids,
            "supporting_passage_titles": support_titles,
            "recall@5": r5,
            "F1": f1,
            "EM": em,
            "latency_index": index_latency,
            "latency_retrieval": retrieval_latency,
            "latency_qa": qa_latency if args.eval_qa else None,
            "index_mode": index_mode,
            "reused_index_path": str(reused_index_path) if reused_index_path else None,
            "llm_calls_index": int(llm_after_index.get("completion_calls", 0) - llm_before_index.get("completion_calls", 0)),
            "llm_calls_query": int(llm_after_query.get("completion_calls", 0) - llm_before_query.get("completion_calls", 0)),
        }
        with per_example_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(per_item, ensure_ascii=False) + "\n")

        running_recall5 = float(sum(recalls_5) / len(recalls_5)) if recalls_5 else 0.0
        if args.eval_qa:
            running_f1 = float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
            logger.info(
                "[%d/%d] %s | recall@5=%.4f | running_recall@5=%.4f | f1=%.4f | running_f1=%.4f | index=%.3fs | query=%.3fs | qa=included_in_query_pipeline",
                i,
                len(sampled_samples),
                sample_id,
                r5,
                running_recall5,
                f1 if f1 is not None else 0.0,
                running_f1,
                index_latency,
                retrieval_latency,
            )
        else:
            logger.info(
                "[%d/%d] %s | recall@5=%.4f | running_recall@5=%.4f | index=%.3fs | query=%.3fs",
                i,
                len(sampled_samples),
                sample_id,
                r5,
                running_recall5,
                index_latency,
                retrieval_latency,
            )

    total_runtime = time.perf_counter() - t_run_start

    def _avg(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    results: Dict[str, Any] = {
        "dataset": "MuSiQue",
        "split": args.split,
        "seed": args.seed,
        "num_queries": len(recalls_5),
        "num_passages": args.topk_passages,
        "recall@1": _avg(recalls_1),
        "recall@2": _avg(recalls_2),
        "recall@5": _avg(recalls_5),
        "recall@10": _avg(recalls_10),
        "F1": _avg(f1_scores) if args.eval_qa else None,
        "EM": _avg(em_scores) if args.eval_qa else None,
        "avg_index_latency": _avg(index_latencies),
        "avg_retrieval_latency": _avg(retrieval_latencies),
        "avg_qa_latency": None,
        "total_runtime": total_runtime,
    }

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_ids = [str(sample.get("id", f"sample_{index+1}")) for index, sample in enumerate(sampled_samples)]
    manifest = {"num_samples": len(sample_ids), "sample_ids": sample_ids}
    (samples_dump_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (samples_dump_dir / "sample_ids.tsv").write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")
    logger.info("Saved config: %s", config_path)
    logger.info("Saved sampled queries: %s", sampled_path)
    logger.info("Saved per-example results: %s", per_example_path)
    logger.info("Saved aggregated results: %s", results_path)
    logger.info("Done. total_runtime=%.3fs", total_runtime)


if __name__ == "__main__":
    main()
