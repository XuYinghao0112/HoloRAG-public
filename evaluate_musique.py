import argparse
import json
import logging
import os
import re
import string
import time
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np

from main_holorag import convert_payload_to_documents
from src.holorag import HoloRAG, HoloRAGConfig
from src.holorag.utils import cosine_similarity_matrix, lexical_overlap_score


logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(text)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Iterable[str]) -> float:
    scores = [metric_fn(prediction, answer) for answer in ground_truths if str(answer).strip()]
    return max(scores) if scores else 0.0


def load_jsonl(path: str, max_samples: int | None = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def build_config(args: argparse.Namespace) -> HoloRAGConfig:
    return HoloRAGConfig(
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        save_dir=args.output_dir,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_dtype=args.embedding_dtype,
        entity_max_length=args.entity_max_length,
        sentence_max_length=args.sentence_max_length,
        chunk_max_length=args.chunk_max_length,
        query_max_length=args.query_max_length,
        enable_sentence_layer=not args.disable_sentence_layer,
        enable_recognition_filter=not args.disable_recognition_filter,
        enable_intent_routing=not args.disable_intent_routing,
        enable_chunk_bridges=not args.disable_chunk_bridges,
        enable_alias_linking=not args.disable_alias_linking,
        enable_granularity_biased_transition=not args.disable_biased_transition,
        enable_llm_judge=args.enable_llm_judge,
    )


def get_gold_answers(sample: Dict[str, Any]) -> List[str]:
    answers = [sample.get("answer", "")]
    answers.extend(sample.get("answer_aliases", []) or [])
    return [str(answer) for answer in answers if str(answer).strip()]


def get_paragraph_text(paragraph: Dict[str, Any]) -> str:
    return str(paragraph.get("paragraph_text", paragraph.get("text", ""))).strip()


def get_paragraph_title(paragraph: Dict[str, Any], fallback_index: int) -> str:
    return str(paragraph.get("title", f"passage_{fallback_index}")).strip() or f"passage_{fallback_index}"


def get_supporting_indices(sample: Dict[str, Any]) -> List[int]:
    support = []
    for fallback_index, paragraph in enumerate(sample.get("paragraphs", [])):
        if paragraph.get("is_supporting"):
            support.append(int(paragraph.get("idx", fallback_index)))
    return sorted(set(support))


def build_paragraph_lookup(sample: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for fallback_index, paragraph in enumerate(sample.get("paragraphs", [])):
        passage_index = int(paragraph.get("idx", fallback_index))
        lookup[passage_index] = {
            "passage_index": passage_index,
            "title": get_paragraph_title(paragraph, fallback_index),
            "text": get_paragraph_text(paragraph),
            "is_supporting": bool(paragraph.get("is_supporting", False)),
        }
    return lookup


def _max_similarity(query_embedding: np.ndarray, node_ids: List[str], embedding_table: Dict[str, np.ndarray]) -> float:
    if not node_ids:
        return 0.0
    matrix = np.asarray([embedding_table[node_id] for node_id in node_ids if node_id in embedding_table], dtype=np.float32)
    if matrix.size == 0:
        return 0.0
    scores = cosine_similarity_matrix(query_embedding, matrix)
    return float(np.max(scores)) if len(scores) else 0.0


def canonicalize_ranked_passages(holorag: HoloRAG, query_result: Dict[str, Any], paragraph_lookup: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    state = holorag.load()
    graph = state["graph"]
    embeddings = state["embeddings"]
    alpha = query_result.get("alpha", {})
    query = str(query_result.get("query", ""))
    sub_questions = query_result.get("sub_questions", []) or [query]

    query_embedding = holorag.embedder.encode(
        [query],
        instruction=holorag.config.query_instruction,
        text_type="query",
    )[0]
    sub_question_embeddings = holorag.embedder.encode(
        [str(item) for item in sub_questions],
        instruction="Retrieve the most relevant sentence evidence.",
        text_type="sub_question",
    )

    graph_rank_scores: Dict[str, float] = {
        item["node_id"]: float(item.get("score", 0.0))
        for item in query_result.get("ranked_nodes", [])
    }
    passage_nodes: Dict[int, Dict[str, List[str]]] = {}

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type not in {"chunk", "sentence"}:
            continue
        metadata = attrs.get("metadata", {})
        passage_index = metadata.get("document_index")
        if passage_index is None:
            continue
        passage_index = int(passage_index)
        slot = passage_nodes.setdefault(passage_index, {"chunk": [], "sentence": []})
        slot[node_type].append(node_id)

    ranked_passages: List[Dict[str, Any]] = []
    for passage_index, paragraph in paragraph_lookup.items():
        node_map = passage_nodes.get(passage_index, {"chunk": [], "sentence": []})
        chunk_nodes = node_map["chunk"]
        sentence_nodes = node_map["sentence"]

        chunk_score = _max_similarity(query_embedding, chunk_nodes, embeddings["chunk"])
        sentence_score = 0.0
        for sub_query_embedding in sub_question_embeddings:
            sentence_score = max(
                sentence_score,
                _max_similarity(sub_query_embedding, sentence_nodes, embeddings["sentence"]),
            )
        graph_score = max([graph_rank_scores.get(node_id, 0.0) for node_id in chunk_nodes + sentence_nodes] or [0.0])
        lexical_score = lexical_overlap_score(query, paragraph.get("text", ""))

        alpha_chunk = float(alpha.get("chunk", 0.33))
        alpha_sentence = float(alpha.get("sentence", 0.33))
        combined_score = (
            alpha_chunk * chunk_score
            + alpha_sentence * sentence_score
            + 0.20 * graph_score
            + 0.10 * lexical_score
        )

        source_layer = "chunk" if chunk_score >= sentence_score else "sentence"
        record = {
            "passage_index": passage_index,
            "title": paragraph.get("title", f"passage_{passage_index}"),
            "score": float(combined_score),
            "source_layer": source_layer,
            "text": paragraph.get("text", ""),
            "score_breakdown": {
                "chunk_retrieval": float(chunk_score),
                "sentence_retrieval": float(sentence_score),
                "graph_rank": float(graph_score),
                "lexical": float(lexical_score),
            },
        }
        ranked_passages.append(record)

    ranked_passages.sort(key=lambda row: row["score"], reverse=True)
    return ranked_passages


def build_qa_context(ranked_passages: List[Dict[str, Any]], qa_top_k: int) -> str:
    selected = ranked_passages[:qa_top_k]
    parts = []
    for passage in selected:
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        if title and text:
            parts.append(f"{title}\n{text}")
        elif text:
            parts.append(text)
    return "\n\n".join(part for part in parts if part).strip()


def compute_recall_at_k(gold_indices: List[int], ranked_passages: List[Dict[str, Any]], k: int) -> float:
    if not gold_indices:
        return 0.0
    retrieved = {int(item["passage_index"]) for item in ranked_passages[:k]}
    hits = len(set(gold_indices) & retrieved)
    return hits / len(set(gold_indices))


def compute_retrieval_metrics(gold_indices: List[int], ranked_passages: List[Dict[str, Any]], recall_ks: List[int]) -> Dict[str, Any]:
    metrics = {
        f"recall@{k}": compute_recall_at_k(gold_indices, ranked_passages, k)
        for k in recall_ks
    }
    metrics["supporting_passage_indices"] = gold_indices
    metrics["retrieved_top_k_passage_indices"] = [int(item["passage_index"]) for item in ranked_passages[: max(recall_ks)]]
    metrics["retrieved_top_k_passage_titles"] = [str(item.get("title", "")) for item in ranked_passages[: max(recall_ks)]]
    return metrics


def generate_answer(holorag: HoloRAG, query: str, qa_context: str) -> str:
    fallback = qa_context.split("\n", 1)[0].strip() if qa_context else ""
    answer = holorag.llm_client.infer_text(
        system_prompt=(
            "Answer the question using only the provided passages. "
            "Return only the short final answer string. "
            "If the answer is not supported by the passages, return UNKNOWN."
        ),
        user_prompt=f"Question:\n{query}\n\nPassages:\n{qa_context}",
        fallback=fallback,
        max_tokens=64,
    )
    return answer.strip()


def count_evidence_items(evidence: Dict[str, Any]) -> Dict[str, int]:
    return {
        "entity": len(evidence.get("entity", [])),
        "sentence": len(evidence.get("sentence", [])),
        "chunk": len(evidence.get("chunk", [])),
    }


def classify_error(record: Dict[str, Any], primary_recall_k: int) -> str:
    if record["em"] >= 1.0:
        return "correct"
    recall_key = f"recall@{primary_recall_k}"
    primary_recall = float(record.get("retrieval_metrics", {}).get(recall_key, 0.0))
    if primary_recall == 0.0:
        return "retrieval_miss"
    if primary_recall < 1.0:
        return "partial_retrieval"

    qa_context = normalize_answer(record.get("qa_context", ""))
    gold_answers = [record.get("gold_answer", "")] + list(record.get("gold_aliases", []) or [])
    if any(normalize_answer(answer) and normalize_answer(answer) in qa_context for answer in gold_answers):
        return "reader_error"
    return "reasoning_error"


def evaluate_sample(holorag: HoloRAG, sample: Dict[str, Any], qa_top_k: int, recall_ks: List[int]) -> Dict[str, Any]:
    holorag.llm_client.reset_stats()
    start_time = time.perf_counter()

    documents = convert_payload_to_documents(sample)
    paragraph_lookup = build_paragraph_lookup(sample)
    gold_support_indices = get_supporting_indices(sample)

    holorag.index(documents)
    query_result = holorag.query(str(sample.get("question", "")))

    ranked_passages = canonicalize_ranked_passages(holorag, query_result, paragraph_lookup)
    qa_context = build_qa_context(ranked_passages, qa_top_k=qa_top_k)
    prediction = generate_answer(holorag, query_result["query"], qa_context)

    elapsed_seconds = time.perf_counter() - start_time
    llm_stats = holorag.llm_client.get_stats()
    gold_answers = get_gold_answers(sample)
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)
    f1 = metric_max_over_ground_truths(f1_score, prediction, gold_answers)
    retrieval_metrics = compute_retrieval_metrics(gold_support_indices, ranked_passages, recall_ks)

    record = {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "prediction": prediction,
        "gold_answer": sample.get("answer"),
        "gold_aliases": sample.get("answer_aliases", []),
        "em": em,
        "f1": f1,
        "elapsed_seconds": elapsed_seconds,
        "llm_calls": llm_stats,
        "alpha": query_result.get("alpha", {}),
        "query_entities": query_result.get("query_entities", []),
        "sub_questions": query_result.get("sub_questions", []),
        "seeds": query_result.get("seeds", []),
        "seed_count": len(query_result.get("seeds", [])),
        "seed_count_by_layer": {
            "entity": sum(1 for item in query_result.get("seeds", []) if item.get("layer") == "entity"),
            "sentence": sum(1 for item in query_result.get("seeds", []) if item.get("layer") == "sentence"),
            "chunk": sum(1 for item in query_result.get("seeds", []) if item.get("layer") == "chunk"),
        },
        "ranked_nodes": query_result.get("ranked_nodes", []),
        "ranked_passages": ranked_passages,
        "retrieval_metrics": retrieval_metrics,
        "qa_context": qa_context,
        "qa_top_k": qa_top_k,
        "evidence": query_result.get("evidence", {}),
        "evidence_count": count_evidence_items(query_result.get("evidence", {})),
    }
    for key, value in retrieval_metrics.items():
        if key.startswith("recall@"):
            record[key] = value
    record["error_type"] = classify_error(record, primary_recall_k=5 if 5 in recall_ks else recall_ks[-1])
    return record


def summarize(records: List[Dict[str, Any]], recall_ks: List[int]) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        summary = {"count": 0, "em": 0.0, "f1": 0.0}
        for k in recall_ks:
            summary[f"recall@{k}"] = 0.0
        return summary

    summary: Dict[str, Any] = {
        "count": total,
        "em": sum(float(item["em"]) for item in records) / total,
        "f1": sum(float(item["f1"]) for item in records) / total,
        "avg_elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in records) / total,
        "avg_llm_calls": {
            "completion_calls": sum(int(item.get("llm_calls", {}).get("completion_calls", 0)) for item in records) / total,
            "json_calls": sum(int(item.get("llm_calls", {}).get("json_calls", 0)) for item in records) / total,
            "text_calls": sum(int(item.get("llm_calls", {}).get("text_calls", 0)) for item in records) / total,
        },
        "avg_seed_count": sum(int(item.get("seed_count", 0)) for item in records) / total,
        "avg_evidence_count": {
            "entity": sum(int(item.get("evidence_count", {}).get("entity", 0)) for item in records) / total,
            "sentence": sum(int(item.get("evidence_count", {}).get("sentence", 0)) for item in records) / total,
            "chunk": sum(int(item.get("evidence_count", {}).get("chunk", 0)) for item in records) / total,
        },
    }
    for k in recall_ks:
        summary[f"recall@{k}"] = sum(float(item.get(f"recall@{k}", 0.0)) for item in records) / total

    error_breakdown: Dict[str, int] = {}
    for item in records:
        error_type = item.get("error_type", "unknown")
        error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
    summary["error_breakdown"] = error_breakdown
    return summary


def build_error_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    wrong_records = [item for item in records if item.get("error_type") != "correct"]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in wrong_records:
        grouped.setdefault(item["error_type"], []).append(item)

    details = {}
    for error_type, items in grouped.items():
        ranked = sorted(
            items,
            key=lambda item: (
                item.get("f1", 0.0),
                item.get("recall@5", 0.0),
                -item.get("elapsed_seconds", 0.0),
            ),
        )
        details[error_type] = [
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "prediction": item.get("prediction"),
                "gold_answer": item.get("gold_answer"),
                "em": item.get("em"),
                "f1": item.get("f1"),
                "recall@2": item.get("recall@2"),
                "recall@5": item.get("recall@5"),
                "recall@10": item.get("recall@10"),
                "retrieved_top_k_passage_indices": item.get("retrieval_metrics", {}).get("retrieved_top_k_passage_indices", []),
                "supporting_passage_indices": item.get("retrieval_metrics", {}).get("supporting_passage_indices", []),
                "llm_calls": item.get("llm_calls"),
                "elapsed_seconds": item.get("elapsed_seconds"),
            }
            for item in ranked[:50]
        ]
    return {"wrong_count": len(wrong_records), "groups": details}


def save_outputs(output_dir: str, records: List[Dict[str, Any]], summary: Dict[str, Any], error_analysis: Dict[str, Any]) -> None:
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    error_analysis_path = os.path.join(output_dir, "error_analysis.json")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(predictions_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(error_analysis_path, "w", encoding="utf-8") as handle:
        json.dump(error_analysis, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="HippoRAG-style MuSiQue evaluator for HoloRAG.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/musique_eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--retrieval_top_k", type=int, default=10)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--report_recall_at", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="gpu1")
    parser.add_argument("--embedding_batch_size", type=int, default=1)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--entity_max_length", type=int, default=64)
    parser.add_argument("--sentence_max_length", type=int, default=256)
    parser.add_argument("--chunk_max_length", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=128)
    parser.add_argument("--disable_sentence_layer", action="store_true")
    parser.add_argument("--disable_recognition_filter", action="store_true")
    parser.add_argument("--disable_intent_routing", action="store_true")
    parser.add_argument("--disable_chunk_bridges", action="store_true")
    parser.add_argument("--disable_alias_linking", action="store_true")
    parser.add_argument("--disable_biased_transition", action="store_true")
    parser.add_argument("--enable_llm_judge", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    samples = load_jsonl(args.dataset_file, max_samples=args.max_samples)
    recall_ks = sorted(set(args.report_recall_at))
    logger.info("Loaded %d samples from %s", len(samples), args.dataset_file)

    holorag = HoloRAG(build_config(args))
    records: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        logger.info("Evaluating sample %d/%d: %s", index, len(samples), sample.get("id"))
        record = evaluate_sample(holorag, sample, qa_top_k=args.qa_top_k, recall_ks=recall_ks)
        record["ranked_passages"] = record["ranked_passages"][: args.retrieval_top_k]
        retrieval_metrics = compute_retrieval_metrics(
            record["retrieval_metrics"]["supporting_passage_indices"],
            record["ranked_passages"],
            recall_ks,
        )
        record["retrieval_metrics"] = retrieval_metrics
        for key, value in retrieval_metrics.items():
            if key.startswith("recall@"):
                record[key] = value
        record["qa_context"] = build_qa_context(record["ranked_passages"], qa_top_k=args.qa_top_k)
        record["error_type"] = classify_error(record, primary_recall_k=5 if 5 in recall_ks else recall_ks[-1])
        records.append(record)

        if index % args.save_every == 0 or index == len(samples):
            summary = summarize(records, recall_ks)
            error_analysis = build_error_analysis(records)
            save_outputs(args.output_dir, records, summary, error_analysis)
            logger.info(
                "Progress %d/%d | EM=%.4f | F1=%.4f | Recall@5=%.4f",
                index,
                len(samples),
                summary["em"],
                summary["f1"],
                summary.get("recall@5", 0.0),
            )

    summary = summarize(records, recall_ks)
    error_analysis = build_error_analysis(records)
    save_outputs(args.output_dir, records, summary, error_analysis)

    print(json.dumps(
        {
            "summary": summary,
            "summary_path": os.path.join(args.output_dir, "metrics_summary.json"),
            "predictions_path": os.path.join(args.output_dir, "predictions.jsonl"),
            "error_analysis_path": os.path.join(args.output_dir, "error_analysis.json"),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
