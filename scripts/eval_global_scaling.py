import argparse
import csv
import json
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval import (  # noqa: E402
    INDEX_FILENAME,
    TokenCounter,
    _alpha_row,
    _avg,
    _evidence_granularity_row,
    apply_task_profile_defaults,
    best_em_f1,
    build_config,
    build_documents,
    build_gold_answers,
    check_llm_server,
    detect_dataset_format,
    format_qa_evidence_from_ranked_passages,
    infer_dataset_name,
    load_samples,
    maybe_filter_split,
    sample_queries,
    set_global_seed,
    setup_logger,
)


SCALE_PRESETS = {
    "canonical": [500, 1000, 2000, 5000, 10000],
    "full": [500, 1000, 2000, 5000, 10000],
}


def _slug(text: str) -> str:
    import re

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return value.strip("_.-") or "dataset"


def doc_key(document: Dict[str, Any]) -> str:
    metadata = document.get("metadata", {}) or {}
    title = metadata.get("original_title", document.get("title", ""))
    return " ".join(
        [
            str(title).strip().lower(),
            str(document.get("text", "")).strip(),
        ]
    )


def global_doc_id(dataset_name: str, sample_id: str, doc: Dict[str, Any], local_index: int) -> str:
    raw_idx = doc.get("idx", local_index)
    return f"{_slug(dataset_name)}::{_slug(sample_id)}::{raw_idx}::{local_index}"


def make_global_document(
    dataset_name: str,
    sample: Dict[str, Any],
    doc: Dict[str, Any],
    local_index: int,
    required_for: Sequence[str] = (),
) -> Dict[str, Any]:
    sample_id = str(sample.get("id", "sample"))
    uid = global_doc_id(dataset_name, sample_id, doc, local_index)
    original_title = str(doc.get("title", f"doc_{local_index}")).strip() or f"doc_{local_index}"
    title = f"{dataset_name}::{uid}::{original_title}"
    return {
        "uid": uid,
        "title": title,
        "text": str(doc.get("text", "")),
        "idx": uid,
        "is_supporting": bool(doc.get("is_supporting", False)),
        "metadata": {
            "dataset": dataset_name,
            "sample_id": sample_id,
            "original_title": original_title,
            "original_idx": doc.get("idx", local_index),
            "required_for": list(required_for),
        },
    }


def iter_dataset_docs(dataset_name: str, samples: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        for local_index, doc in enumerate(build_documents(sample)):
            if not str(doc.get("text", "")).strip():
                continue
            yield make_global_document(dataset_name, sample, doc, local_index, required_for=[sample_id])


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return list(value) if isinstance(value, tuple) else [value]


def _hotpot_context_documents(context: Any, supporting_facts: Any = None) -> List[Dict[str, Any]]:
    support_titles = set()
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        support_titles = {str(title) for title in _as_list(titles)}
    elif isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and item:
                support_titles.add(str(item[0]))
            elif isinstance(item, dict) and item.get("title") is not None:
                support_titles.add(str(item.get("title")))

    docs: List[Dict[str, Any]] = []
    if isinstance(context, dict):
        titles = _as_list(context.get("title", []))
        sentences_list = _as_list(context.get("sentences", []))
        for idx, title in enumerate(titles):
            sentences = sentences_list[idx] if idx < len(sentences_list) else []
            if not isinstance(sentences, list):
                sentences = _as_list(sentences)
            text = " ".join(str(sentence) for sentence in sentences)
            docs.append({"idx": idx, "title": str(title), "text": text, "is_supporting": str(title) in support_titles})
        return docs

    if isinstance(context, list):
        for idx, item in enumerate(context):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sentences = item[1] if isinstance(item[1], list) else _as_list(item[1])
                text = " ".join(str(sentence) for sentence in sentences)
                docs.append({"idx": idx, "title": title, "text": text, "is_supporting": title in support_titles})
            elif isinstance(item, dict):
                title = str(item.get("title", f"doc_{idx}"))
                sentences = item.get("sentences", item.get("text", ""))
                text = " ".join(str(sentence) for sentence in sentences) if isinstance(sentences, list) else str(sentences)
                docs.append({"idx": idx, "title": title, "text": text, "is_supporting": title in support_titles})
    return docs


def normalize_hotpot_row(row: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    sample_id = str(row.get("id") or row.get("_id") or f"hotpot_{fallback_idx:06d}")
    return {
        "id": sample_id,
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "answer_aliases": [],
        "paragraphs": _hotpot_context_documents(row.get("context"), row.get("supporting_facts")),
        "type": row.get("type", ""),
        "level": row.get("level", ""),
        "dataset": "hotpotqa",
        "split": row.get("split", ""),
    }


def load_hotpot_parquet(path: str, split: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        columns = table.to_pydict()
        keys = list(columns.keys())
        for idx in range(table.num_rows):
            row = {key: columns[key][idx] for key in keys}
            if split:
                row["split"] = split
            rows.append(normalize_hotpot_row(row, idx + 1))
        return rows
    except ImportError:
        pass
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
        for idx, row in enumerate(frame.to_dict(orient="records"), start=1):
            if split:
                row["split"] = split
            rows.append(normalize_hotpot_row(row, idx))
        return rows
    except ImportError:
        pass
    try:
        from datasets import Dataset

        dataset = Dataset.from_parquet(path)
        for idx, row in enumerate(dataset, start=1):
            if split:
                row["split"] = split
            rows.append(normalize_hotpot_row(row, idx))
        return rows
    except ImportError as exc:
        raise RuntimeError(
            "Reading HotpotQA parquet requires one of: pyarrow, pandas, or datasets. "
            "Install pyarrow in the active environment, e.g. `pip install pyarrow`."
        ) from exc


def load_hotpot_parquet_many(path: str) -> List[Dict[str, Any]]:
    input_path = Path(path).expanduser()
    files: List[Path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
    else:
        files = sorted(input_path.parent.glob(input_path.name)) if any(ch in input_path.name for ch in "*?[]") else [input_path]
    samples: List[Dict[str, Any]] = []
    for file_path in files:
        name = file_path.name.lower()
        split = "validation" if "validation" in name or "dev" in name else "train" if "train" in name else ""
        loaded = load_hotpot_parquet(str(file_path), split=split)
        samples.extend(loaded)
    return samples


def load_scaling_samples(path: str, dataset_format: str) -> List[Dict[str, Any]]:
    if dataset_format == "hotpot_parquet":
        return load_hotpot_parquet_many(path)
    return load_samples(path, dataset_format)


def filter_scaling_split(samples: Sequence[Dict[str, Any]], split: str, dataset_format: str) -> List[Dict[str, Any]]:
    filtered = maybe_filter_split(samples, split)
    if filtered or dataset_format != "hotpot_parquet":
        return filtered
    split_aliases = {
        "dev": "validation",
        "valid": "validation",
        "validation": "dev",
    }
    alias = split_aliases.get(str(split or "").lower())
    if alias:
        return maybe_filter_split(samples, alias)
    return filtered


def detect_scaling_dataset_format(path: str, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format
    input_path = Path(path)
    if input_path.suffix == ".parquet" or any(ch in input_path.name for ch in "*?[]"):
        return "hotpot_parquet"
    if input_path.is_dir() and list(input_path.glob("*.parquet")):
        return "hotpot_parquet"
    return detect_dataset_format(path, explicit_format)


def collect_required_docs(dataset_name: str, eval_samples: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    required: List[Dict[str, Any]] = []
    required_by_query: Dict[str, List[str]] = {}
    seen = set()
    for sample in eval_samples:
        sample_id = str(sample.get("id", ""))
        docs = build_documents(sample)
        supporting_docs = [doc for doc in docs if doc.get("is_supporting")]
        selected = supporting_docs or docs
        query_doc_ids: List[str] = []
        for local_index, doc in enumerate(selected):
            if not str(doc.get("text", "")).strip():
                continue
            global_doc = make_global_document(dataset_name, sample, doc, local_index, required_for=[sample_id])
            key = doc_key(global_doc)
            if key not in seen:
                required.append(global_doc)
                seen.add(key)
            query_doc_ids.append(global_doc["uid"])
        required_by_query[sample_id] = query_doc_ids
    return required, required_by_query


def dedupe_documents(documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for doc in documents:
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def parse_dataset_specs(values: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    specs = []
    for index, value in enumerate(values, start=1):
        parts = value.split(":")
        if len(parts) == 1:
            path = parts[0]
            name = Path(path).stem
            fmt = "auto"
            split = ""
        elif len(parts) == 2:
            name, path = parts
            fmt = "auto"
            split = ""
        elif len(parts) == 3:
            name, path, fmt = parts
            split = ""
        else:
            name, path, fmt, split = parts[0], parts[1], parts[2], ":".join(parts[3:])
        name = _slug(name or f"dataset_{index}")
        specs.append((name, path, fmt or "auto", split))
    return specs


def load_corpus_sources(args: argparse.Namespace, eval_dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    corpus_docs: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    specs = parse_dataset_specs(args.corpus_dataset) if args.corpus_dataset else []
    if not specs:
        specs = [(eval_dataset_name, args.dataset_file, args.dataset_format, args.split)]
    for explicit_name, path, fmt, split in specs:
        dataset_format = detect_scaling_dataset_format(path, fmt)
        samples = load_scaling_samples(path, dataset_format)
        selected = filter_scaling_split(samples, split, dataset_format) if split else samples
        dataset_name = explicit_name or infer_dataset_name(path, samples, "")
        docs = list(iter_dataset_docs(dataset_name, selected))
        corpus_docs.extend(docs)
        source_counts[dataset_name] = source_counts.get(dataset_name, 0) + len(docs)
    return dedupe_documents(corpus_docs), source_counts


def load_fixed_queries(path: str) -> List[Dict[str, Any]]:
    query_path = Path(path).expanduser()
    payload = json.loads(query_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Fixed query file must be a list or contain a samples list: {query_path}")
    return [sample for sample in samples if isinstance(sample, dict)]


def parse_scales(
    scales_arg: str,
    required_count: int,
    available_count: int,
    include_max_scale: bool = False,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    value = str(scales_arg or "auto").strip().lower()
    skipped: List[Dict[str, Any]] = []
    if value == "auto":
        candidates = SCALE_PRESETS["canonical"]
        if available_count >= 50000:
            candidates = SCALE_PRESETS["full"]
        elif available_count >= 20000:
            candidates = [1000, 2000, 5000, 10000, 20000]
        elif available_count >= 10000:
            candidates = [1000, 2000, 5000, 10000]
    else:
        preset = SCALE_PRESETS.get(value)
        candidates = preset if preset is not None else [int(item) for item in value.replace(" ", "").split(",") if item]

    selected: List[int] = []
    for scale in candidates:
        if scale < required_count:
            skipped.append({"scale": scale, "reason": "below_required_docs", "required_docs": required_count})
            continue
        if scale > available_count:
            skipped.append({"scale": scale, "reason": "above_available_docs", "available_docs": available_count})
            continue
        if scale not in selected:
            selected.append(scale)
    if include_max_scale and available_count >= required_count and available_count not in selected:
        selected.append(available_count)
    return sorted(selected), skipped


def build_scaled_corpus(
    required_docs: Sequence[Dict[str, Any]],
    corpus_docs: Sequence[Dict[str, Any]],
    scale: int,
    seed: int,
) -> List[Dict[str, Any]]:
    required = dedupe_documents(required_docs)
    required_keys = {doc_key(doc) for doc in required}
    distractors = [doc for doc in corpus_docs if doc_key(doc) not in required_keys]
    needed = max(0, scale - len(required))
    if needed > len(distractors):
        raise ValueError(f"Scale {scale} requires {needed} distractors, but only {len(distractors)} are available.")
    rng = random.Random(seed + scale)
    sampled = rng.sample(distractors, needed) if needed < len(distractors) else list(distractors)
    return required + sampled


def index_global_corpus(rag, documents: Sequence[Dict[str, Any]], scale_dir: Path, logger) -> Dict[str, Any]:
    scale_dir.mkdir(parents=True, exist_ok=True)
    index_path = scale_dir / INDEX_FILENAME
    metadata_path = scale_dir / "metadata.json"
    if index_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "index_path": str(index_path),
            "stats": metadata.get("stats", {}),
            "index_latency": float(metadata.get("index_latency", 0.0) or 0.0),
            "reused": True,
        }

    logger.info("[index] building global graph with %d passages", len(documents))
    t0 = time.perf_counter()
    rag.config.save_dir = str(scale_dir)
    rag.artifact_dir = str(scale_dir)
    rag.index_path = str(index_path)
    rag.index(list(documents))
    with index_path.open("wb") as handle:
        pickle.dump(rag.state, handle)
    latency = time.perf_counter() - t0
    stats = rag.describe_index()
    metadata = {
        "index_path": str(index_path),
        "num_passages": len(documents),
        "stats": stats,
        "index_latency": latency,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index_path": str(index_path), "stats": stats, "index_latency": latency, "reused": False}


def evaluate_on_global_index(
    rag,
    eval_samples: Sequence[Dict[str, Any]],
    index_record: Dict[str, Any],
    scale_dir: Path,
    scale: int,
    logger,
    token_counter: TokenCounter,
) -> Dict[str, Any]:
    per_example_path = scale_dir / "per_example.jsonl"
    per_example_path.write_text("", encoding="utf-8")
    with Path(index_record["index_path"]).open("rb") as handle:
        rag.state = pickle.load(handle)
    rag.index_path = str(index_record["index_path"])

    em_scores: List[float] = []
    f1_scores: List[float] = []
    retrieval_latencies: List[float] = []
    retrieval_pipeline_latencies: List[float] = []
    qa_latencies: List[float] = []
    query_total_latencies: List[float] = []
    evidence_tokens: List[float] = []
    alpha_f_values: List[float] = []
    alpha_s_values: List[float] = []
    alpha_c_values: List[float] = []
    used_tokens_f: List[float] = []
    used_tokens_s: List[float] = []
    used_tokens_c: List[float] = []

    t_start = time.perf_counter()
    for i, sample in enumerate(eval_samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        if not question:
            continue
        t_query = time.perf_counter()
        result = rag.query(question)
        query_elapsed = time.perf_counter() - t_query
        query_timing = result.get("query_timing", {}) or {}
        query_total_latencies.append(query_elapsed)
        retrieval_latencies.append(float(query_timing.get("retrieval_latency", query_elapsed)))
        retrieval_pipeline_latencies.append(float(query_timing.get("retrieval_pipeline_latency", 0.0)))
        qa_latencies.append(float(query_timing.get("qa_latency", 0.0)))

        ranked_passages = result.get("ranked_passages", []) or []
        qa_messages = result.get("qa_messages", []) or []
        evidence_text = ""
        if len(qa_messages) >= 2 and isinstance(qa_messages[1], dict):
            evidence_text = str(qa_messages[1].get("content", ""))
        if not evidence_text:
            evidence_text = format_qa_evidence_from_ranked_passages(ranked_passages, rag.config.qa_passage_top_k)
        final_evidence_tokens = token_counter.count(evidence_text)
        alpha_values = _alpha_row(result.get("alpha", {}) or {})
        evidence_values = _evidence_granularity_row(result, final_evidence_tokens)
        final_evidence_tokens = int(evidence_values["final_evidence_tokens"])
        evidence_tokens.append(float(final_evidence_tokens))
        alpha_f_values.append(alpha_values["alpha_F"])
        alpha_s_values.append(alpha_values["alpha_S"])
        alpha_c_values.append(alpha_values["alpha_C"])
        used_tokens_f.append(float(evidence_values["used_tokens_F"]))
        used_tokens_s.append(float(evidence_values["used_tokens_S"]))
        used_tokens_c.append(float(evidence_values["used_tokens_C"]))

        predicted_answer = str(result.get("predicted_answer", "")).strip()
        em, f1 = best_em_f1(build_gold_answers(sample), predicted_answer)
        em_scores.append(em)
        f1_scores.append(f1)
        row = {
            "scale_passages": scale,
            "query_id": sample_id,
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "predicted_answer": predicted_answer,
            "F1": f1,
            "EM": em,
            "query_total_latency": query_elapsed,
            "retrieval_latency": query_timing.get("retrieval_latency"),
            "retrieval_pipeline_latency": query_timing.get("retrieval_pipeline_latency"),
            "qa_latency": query_timing.get("qa_latency"),
            "final_evidence_tokens": final_evidence_tokens,
            "final_evidence_tokenizer": token_counter.method,
            "qa_answer_mode": result.get("qa_answer_mode", ""),
        }
        row.update(alpha_values)
        row.update(evidence_values)
        with per_example_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[scale=%d][%d/%d] %s | f1=%.4f em=%.4f | running_f1=%.4f | q=%.3fs",
            scale,
            i,
            len(eval_samples),
            sample_id,
            f1,
            em,
            _avg(f1_scores),
            query_elapsed,
        )

    stats = index_record.get("stats", {}) or {}
    layers = stats.get("layer_counts", {}) or {}
    metrics = {
        "variant": "holorag_global_scaling",
        "scale_passages": scale,
        "num_queries": len(f1_scores),
        "F1": _avg(f1_scores),
        "EM": _avg(em_scores),
        "index_latency": float(index_record.get("index_latency", 0.0) or 0.0),
        "index_reused": bool(index_record.get("reused", False)),
        "retrieval_latency": _avg(retrieval_latencies),
        "retrieval_pipeline_latency": _avg(retrieval_pipeline_latencies),
        "qa_latency": _avg(qa_latencies),
        "retrieval_qa_latency": _avg(query_total_latencies),
        "query_runtime": time.perf_counter() - t_start,
        "total_runtime": float(index_record.get("index_latency", 0.0) or 0.0) + (time.perf_counter() - t_start),
        "nodes": int(stats.get("nodes", 0)),
        "edges": int(stats.get("edges", 0)),
        "entity_nodes": int(layers.get("entity", 0)),
        "fact_nodes": int(layers.get("fact", 0)),
        "sentence_nodes": int(layers.get("sentence", 0)),
        "chunk_nodes": int(layers.get("chunk", 0)),
        "final_evidence_tokens": _avg(evidence_tokens),
        "final_evidence_tokenizer": token_counter.method,
        "avg_alpha_F": _avg(alpha_f_values),
        "avg_alpha_S": _avg(alpha_s_values),
        "avg_alpha_C": _avg(alpha_c_values),
        "avg_used_tokens_F": _avg(used_tokens_f),
        "avg_used_tokens_S": _avg(used_tokens_s),
        "avg_used_tokens_C": _avg(used_tokens_c),
    }
    for edge_type, count in sorted((stats.get("edge_type_counts", {}) or {}).items()):
        metrics[f"edge_{edge_type}"] = int(count)
    return metrics


def write_metrics(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path_json = run_dir / "metrics_summary.json"
    path_csv = run_dir / "metrics_summary.csv"
    path_json.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    headers: List[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HoloRAG on a shared global corpus at increasing passage scales.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Dataset used for evaluation queries.")
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "musique_jsonl", "canonical_jsonl", "2wiki_json", "canonical_json", "hotpot_parquet"])
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--corpus_dataset", action="append", default=[], help="Optional corpus source as name:path[:format[:split]]. Repeat to merge sources.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus_seed", type=int, default=None, help="Seed for sampling distractor corpus documents. Defaults to --seed.")
    parser.add_argument("--sampled_queries_file", type=str, default="", help="Reuse a fixed sampled_queries.json file instead of sampling queries from --dataset_file.")
    parser.add_argument("--num_eval_queries", type=int, default=100)
    parser.add_argument("--scales", type=str, default="auto", help="'auto', 'canonical', 'full', or comma-separated passage counts.")
    parser.add_argument("--include_max_scale", action="store_true", help="Also run one scale using every available deduplicated corpus passage.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--recompute_only", action="store_true", help="Reuse existing global indexes and recompute metrics.")
    parser.add_argument("--prepare_only", action="store_true", help="Only resolve samples, corpus, and scales; do not load models, index, or query.")

    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
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
    parser.add_argument("--entity_similarity_top_k", type=int, default=32)
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
    parser.add_argument("--evidence_soft_token_budget", type=int, default=0)
    parser.add_argument("--evidence_allow_underfill", action="store_true", default=True)
    parser.add_argument("--evidence_min_score", type=float, default=0.0)
    parser.add_argument("--evidence_redundancy_threshold", type=float, default=0.85)
    parser.add_argument("--disable_evidence_alpha_weights", action="store_true")
    parser.add_argument("--task_profile", type=str, default="multi_hop", choices=["auto", "single_hop", "multi_hop", "long_context"])
    parser.add_argument("--skip_llm_health_check", action="store_true")
    parser.add_argument("--llm_health_timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    apply_task_profile_defaults(args, sys.argv[1:])

    from holorag import HoloRAG

    eval_dataset_format = detect_scaling_dataset_format(args.dataset_file, args.dataset_format)
    all_eval_samples = load_scaling_samples(args.dataset_file, eval_dataset_format)
    eval_dataset_name = infer_dataset_name(args.dataset_file, all_eval_samples, args.dataset_name)
    filtered = filter_scaling_split(all_eval_samples, args.split, eval_dataset_format)
    if not filtered:
        raise ValueError("No evaluation samples available after split filtering.")

    run_name = args.run_name.strip() or f"{eval_dataset_name}_global_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")
    logger.info("Run directory: %s", run_dir)
    if not args.skip_llm_health_check:
        check_llm_server(args.llm_base_url, args.llm_name, args.llm_health_timeout, logger)

    corpus_seed = args.seed if args.corpus_seed is None else args.corpus_seed
    if args.sampled_queries_file:
        eval_samples = load_fixed_queries(args.sampled_queries_file)
        if args.num_eval_queries > 0:
            eval_samples = eval_samples[: args.num_eval_queries]
        sampled_payload = {
            "seed": args.seed,
            "corpus_seed": corpus_seed,
            "num_eval_queries": len(eval_samples),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_sampled_queries_file": str(Path(args.sampled_queries_file).expanduser().resolve()),
            "samples": eval_samples,
        }
        (run_dir / "sampled_queries.json").write_text(json.dumps(sampled_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        eval_samples = sample_queries(filtered, run_dir / "sampled_queries.json", args.seed, args.num_eval_queries)
    required_docs, required_by_query = collect_required_docs(eval_dataset_name, eval_samples)
    corpus_docs, source_counts = load_corpus_sources(args, eval_dataset_name)
    required_keys = {doc_key(doc) for doc in required_docs}
    merged_corpus = dedupe_documents(list(required_docs) + list(corpus_docs))
    scales, skipped_scales = parse_scales(args.scales, len(required_docs), len(merged_corpus), args.include_max_scale)
    if not scales:
        raise ValueError(
            f"No valid scales. required_docs={len(required_docs)} available_docs={len(merged_corpus)} "
            f"requested={args.scales}"
        )

    logger.info("Required docs: %d | available unique corpus docs: %d", len(required_docs), len(merged_corpus))
    logger.info("Corpus source counts: %s", json.dumps(source_counts, ensure_ascii=False, sort_keys=True))
    logger.info("Running scales: %s", scales)
    if skipped_scales:
        logger.info("Skipped scales: %s", json.dumps(skipped_scales, ensure_ascii=False))

    prepare_summary = {
        "experiment": "global_corpus_scaling",
        "dataset_file": args.dataset_file,
        "dataset_format": eval_dataset_format,
        "dataset_name": eval_dataset_name,
        "split": args.split,
        "num_eval_queries": args.num_eval_queries,
        "seed": args.seed,
        "corpus_seed": corpus_seed,
        "sampled_queries_file": args.sampled_queries_file,
        "requested_scales": args.scales,
        "include_max_scale": args.include_max_scale,
        "resolved_scales": scales,
        "skipped_scales": skipped_scales,
        "num_required_docs": len(required_docs),
        "num_available_docs": len(merged_corpus),
        "corpus_source_counts": source_counts,
        "corpus_dataset": args.corpus_dataset,
        "required_by_query_preview": dict(list(required_by_query.items())[:10]),
    }
    (run_dir / "prepare_summary.json").write_text(json.dumps(prepare_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        logger.info("prepare_only complete: %s", run_dir / "prepare_summary.json")
        return

    rag = HoloRAG(build_config(args, save_dir=str(run_dir / "workdir")))
    token_counter = TokenCounter(args.llm_name, logger)
    metrics_rows: List[Dict[str, Any]] = []
    scale_records: List[Dict[str, Any]] = []

    for scale in scales:
        scale_dir = run_dir / f"G_{scale}"
        documents = build_scaled_corpus(required_docs, merged_corpus, scale, corpus_seed)
        manifest = {
            "scale_passages": scale,
            "num_required_docs": len(required_docs),
            "num_distractor_docs": max(0, scale - len(required_docs)),
            "num_available_docs": len(merged_corpus),
            "required_doc_fraction": len(required_docs) / max(1, scale),
            "documents_preview": [
                {
                    "uid": doc.get("uid"),
                    "title": doc.get("title"),
                    "metadata": doc.get("metadata", {}),
                }
                for doc in documents[:20]
            ],
        }
        scale_dir.mkdir(parents=True, exist_ok=True)
        (scale_dir / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.recompute_only:
            metadata_path = scale_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning("[scale=%d] missing metadata under recompute_only; skipping", scale)
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            index_record = {
                "index_path": str(scale_dir / INDEX_FILENAME),
                "stats": metadata.get("stats", {}),
                "index_latency": float(metadata.get("index_latency", 0.0) or 0.0),
                "reused": True,
            }
        else:
            index_record = index_global_corpus(rag, documents, scale_dir, logger)
        metrics = evaluate_on_global_index(rag, eval_samples, index_record, scale_dir, scale, logger, token_counter)
        metrics_rows.append(metrics)
        scale_records.append({**manifest, "index": index_record, "metrics": metrics})
        write_metrics(run_dir, metrics_rows)

    config = {
        "experiment": "global_corpus_scaling",
        "dataset_file": args.dataset_file,
        "dataset_format": eval_dataset_format,
        "dataset_name": eval_dataset_name,
        "split": args.split,
        "num_eval_queries": args.num_eval_queries,
        "seed": args.seed,
        "corpus_seed": corpus_seed,
        "sampled_queries_file": args.sampled_queries_file,
        "requested_scales": args.scales,
        "include_max_scale": args.include_max_scale,
        "resolved_scales": scales,
        "skipped_scales": skipped_scales,
        "num_required_docs": len(required_docs),
        "num_available_docs": len(merged_corpus),
        "num_required_docs_already_in_corpus": sum(1 for doc in corpus_docs if doc_key(doc) in required_keys),
        "corpus_source_counts": source_counts,
        "corpus_dataset": args.corpus_dataset,
        "required_by_query": required_by_query,
        "llm_base_url": args.llm_base_url,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "embedding_device": args.embedding_device,
        "task_profile": args.task_profile,
        "use_paragraph_as_chunk": not args.disable_paragraph_as_chunk,
        "index_extraction_mode": args.index_extraction_mode,
        "enable_entity_similarity_edges": not args.disable_entity_similarity_edges,
        "entity_similarity_threshold": args.entity_similarity_threshold,
        "entity_similarity_top_k": args.entity_similarity_top_k,
        "enable_sentence_layer": not args.disable_sentence_layer,
        "enable_granularity_awareness": not args.disable_granularity_awareness,
        "enable_granularity_pagerank_bias": not args.disable_granularity_pagerank_bias,
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "scale_records.json").write_text(json.dumps(scale_records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics(run_dir, metrics_rows)
    logger.info("Saved metrics: %s", run_dir / "metrics_summary.json")
    logger.info("Global scaling complete.")


if __name__ == "__main__":
    main()
