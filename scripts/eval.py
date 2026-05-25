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
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
INDEX_FILENAME = "holorag_index.pkl"
LEGACY_INDEX_FILENAME = "holorag_naive_index.pkl"
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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_sample_list(payload: Any, dataset_file: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("samples", "data", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Expected a sample list in dataset: {dataset_file}")


DOCUMENT_LIST_KEYS = ("documents", "docs", "paragraphs", "contexts", "context", "passages", "items", "data")
TEXT_KEYS = ("text", "content", "contents", "body", "page_content", "paragraph_text")
TITLE_KEYS = ("title", "name", "doc_id", "id", "uid")


def _first_text_value(item: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def _document_from_item(item: Any, idx: int, supporting_titles: set = None) -> Dict[str, Any]:
    supporting_titles = supporting_titles or set()
    if isinstance(item, str):
        return {"idx": idx, "title": f"doc_{idx}", "text": item}
    if isinstance(item, list) and len(item) >= 2:
        title = str(item[0])
        body = item[1]
        text = " ".join(str(s) for s in body) if isinstance(body, list) else str(body)
        return {"idx": idx, "title": title, "text": text, "is_supporting": title in supporting_titles}
    if not isinstance(item, dict):
        return {}
    title = _first_text_value(item, TITLE_KEYS) or f"doc_{idx}"
    text = _first_text_value(item, TEXT_KEYS)
    if not text and isinstance(item.get("sentences"), list):
        text = " ".join(str(sentence) for sentence in item["sentences"])
    if not text:
        return {}
    document = dict(item)
    document["idx"] = document.get("idx", idx)
    document["title"] = title
    document["text"] = text
    if "is_supporting" not in document and title in supporting_titles:
        document["is_supporting"] = True
    return document


def _documents_from_items(items: Any, supporting_titles: set = None) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return documents
    for idx, entry in enumerate(items):
        document = _document_from_item(entry, idx, supporting_titles=supporting_titles)
        if document and str(document.get("text", "")).strip():
            documents.append(document)
    return documents


def _documents_from_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in DOCUMENT_LIST_KEYS:
        documents = _documents_from_items(sample.get(key))
        if documents:
            return documents
    return []


def _documents_from_contexts(contexts: Any) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    if not isinstance(contexts, list):
        return documents
    for idx, entry in enumerate(contexts):
        title = f"doc_{idx}"
        text = ""
        is_supporting = False
        if isinstance(entry, dict):
            title = str(entry.get("title") or entry.get("name") or title)
            text = str(entry.get("paragraph_text") or entry.get("text") or entry.get("contents") or entry.get("content") or "")
            is_supporting = bool(entry.get("is_supporting", False))
        elif isinstance(entry, list) and len(entry) >= 2:
            title = str(entry[0])
            body = entry[1]
            text = " ".join(str(s) for s in body) if isinstance(body, list) else str(body)
        elif isinstance(entry, str):
            text = entry
        if text:
            doc_idx = entry.get("idx", idx) if isinstance(entry, dict) else idx
            documents.append({"idx": doc_idx, "title": title, "text": text, "is_supporting": is_supporting})
    return documents


def normalize_sample(item: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    sample = dict(item)
    sample.setdefault(
        "id",
        str(
            sample.get("_id")
            or sample.get("qid")
            or sample.get("question_id")
            or sample.get("uid")
            or f"sample_{fallback_idx:04d}"
        ),
    )
    if "question" not in sample and sample.get("query") is not None:
        sample["question"] = str(sample.get("query", ""))
    if "answer" not in sample:
        answers = sample.get("answers")
        if isinstance(answers, list) and answers:
            sample["answer"] = str(answers[0])
            aliases = [str(answer) for answer in answers[1:] if answer]
            if aliases and "answer_aliases" not in sample:
                sample["answer_aliases"] = aliases
        elif sample.get("gold_answer") is not None:
            sample["answer"] = str(sample.get("gold_answer", ""))
    if "documents" not in sample:
        for key in DOCUMENT_LIST_KEYS:
            documents = _documents_from_contexts(sample.get(key))
            if documents:
                sample["documents"] = documents
                break
    return sample


def normalize_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_sample(sample, i) for i, sample in enumerate(samples, start=1)]


def convert_2wiki_item(item: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    supporting_titles = {
        str(x[0])
        for x in item.get("supporting_facts", [])
        if isinstance(x, list) and x
    }
    documents: List[Dict[str, Any]] = []
    for idx, entry in enumerate(item.get("context", [])):
        if not (isinstance(entry, list) and len(entry) >= 2):
            continue
        title = str(entry[0])
        sentences = entry[1] if isinstance(entry[1], list) else []
        documents.append(
            {
                "idx": idx,
                "title": title,
                "text": " ".join(str(s) for s in sentences),
                "is_supporting": title in supporting_titles,
            }
        )
    sample_id = str(item.get("_id") or item.get("id") or f"2wiki_{fallback_idx:04d}")
    return {
        "id": sample_id,
        "question": str(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "answer_aliases": [],
        "documents": documents,
    }


def detect_dataset_format(dataset_file: str, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format
    if dataset_file.endswith(".jsonl"):
        return "canonical_jsonl"
    payload = load_json(dataset_file)
    samples = _extract_sample_list(payload, dataset_file)
    if samples:
        first = samples[0]
        if isinstance(first, dict) and "context" in first and "supporting_facts" in first:
            return "2wiki_json"
        if isinstance(first, dict) and any(key in first for key in DOCUMENT_LIST_KEYS) and "question" in first:
            return "canonical_json"
        if isinstance(first, dict):
            return "canonical_json"
    raise ValueError("Cannot auto-detect dataset format; set --dataset_format explicitly.")


def load_samples(dataset_file: str, dataset_format: str) -> List[Dict[str, Any]]:
    if dataset_format in ("musique_jsonl", "canonical_jsonl"):
        return normalize_samples(load_jsonl(dataset_file))
    payload = load_json(dataset_file)
    rows = _extract_sample_list(payload, dataset_file)
    if dataset_format == "2wiki_json":
        return [convert_2wiki_item(item, i) for i, item in enumerate(rows, start=1)]
    if dataset_format == "canonical_json":
        return normalize_samples(rows)
    raise ValueError("Unsupported dataset_format")


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return value.strip("_.-") or "dataset"


def infer_dataset_name(dataset_file: str, samples: Sequence[Dict[str, Any]], explicit_name: str) -> str:
    if explicit_name.strip():
        return _slug(explicit_name)
    counts: Dict[str, int] = {}
    for sample in samples:
        for key in ("dataset", "dataset_name", "source", "task"):
            value = str(sample.get(key, "")).strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
                break
    if counts:
        return _slug(max(counts.items(), key=lambda item: item[1])[0])
    stem = Path(dataset_file).stem
    stem = re.sub(r"(_canonical|_dev|_test|_train|_validation|_eval)$", "", stem, flags=re.IGNORECASE)
    if stem.lower() in ("2wikimqa", "2wiki_mqa"):
        stem = "2wiki"
    return _slug(stem)


def build_log_tag(dataset_name: str, ablation_name: str) -> str:
    if ablation_name.strip():
        return f"{dataset_name}:{_slug(ablation_name)}"
    return dataset_name


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


def sample_queries(samples: Sequence[Dict[str, Any]], sampled_path: Path, seed: int, num_eval_queries: int) -> List[Dict[str, Any]]:
    if sampled_path.exists():
        payload = json.loads(sampled_path.read_text(encoding="utf-8"))
        return payload.get("samples", payload)
    rng = random.Random(seed)
    all_samples = list(samples)
    sampled = all_samples if num_eval_queries >= len(all_samples) else rng.sample(all_samples, num_eval_queries)
    sampled_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "num_eval_queries": num_eval_queries,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "samples": sampled,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return sampled


def _normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


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
        best_em = max(best_em, 1.0 if gold_norm == pred_norm else 0.0)
        best_f1 = max(best_f1, _token_f1(gold, pred))
    return best_em, best_f1


def build_gold_answers(sample: Dict[str, Any]) -> List[str]:
    answers: List[str] = []
    if sample.get("answer"):
        answers.append(str(sample["answer"]))
    answers.extend(str(answer) for answer in sample.get("answers", []) if answer)
    answers.extend(str(alias) for alias in sample.get("answer_aliases", []) if alias)
    dedup: List[str] = []
    seen = set()
    for ans in answers:
        key = _normalize_answer(ans)
        if key and key not in seen:
            seen.add(key)
            dedup.append(ans)
    return dedup


def word_token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


class TokenCounter:
    def __init__(self, tokenizer_name: str, logger: logging.Logger) -> None:
        self.tokenizer = None
        self.method = "regex_nonspace"
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.method = "hf_tokenizer"
        except Exception as exc:
            logger.warning(
                "Could not load tokenizer %s for evidence token counting; falling back to regex count. Error: %s",
                tokenizer_name,
                exc,
            )

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return word_token_count(text)
        return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))


def format_qa_evidence_from_ranked_passages(ranked_passages: Sequence[Dict[str, Any]], qa_top_k: int) -> str:
    rows: List[str] = []
    for passage in list(ranked_passages[:qa_top_k]):
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        if not text:
            continue
        rows.append(f"Wikipedia Title: {title}\n{text}" if title else text)
    return "\n\n".join(rows)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("holorag_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    package_logger = logging.getLogger("holorag")
    package_logger.setLevel(logging.INFO)
    package_logger.handlers.clear()
    package_logger.addHandler(fh)
    package_logger.addHandler(sh)
    package_logger.propagate = False
    return logger


def check_llm_server(llm_base_url: str, llm_name: str, timeout_seconds: float, logger: logging.Logger) -> None:
    import httpx

    models_url = llm_base_url.rstrip("/") + "/models"
    try:
        with httpx.Client(trust_env=False, timeout=timeout_seconds) as client:
            response = client.get(models_url, headers={"Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"LLM server is not reachable at {models_url}. Start vLLM/OpenAI-compatible service first, "
            f"or pass --skip_llm_health_check if you intentionally want fallback behavior. Error: {exc}"
        ) from exc

    model_ids = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
    if model_ids and llm_name not in model_ids:
        logger.warning("LLM server reachable, but configured model %s is not in /models: %s", llm_name, model_ids)
    else:
        logger.info("LLM server reachable: %s", models_url)


def _avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def resolve_index_path(sample_dir: Path) -> Path:
    index_path = sample_dir / INDEX_FILENAME
    if index_path.exists():
        return index_path
    legacy_path = sample_dir / LEGACY_INDEX_FILENAME
    if legacy_path.exists():
        return legacy_path
    return index_path


def build_documents(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    source_documents = _documents_from_sample(sample)
    for idx, document in enumerate(source_documents):
        documents.append(
            {
                "title": str(document.get("title", f"doc_{idx}")),
                "text": str(document.get("text", "")),
                "idx": document.get("idx", idx),
                "is_supporting": document.get("is_supporting", False),
            }
        )
    return documents


def build_config(args: argparse.Namespace, save_dir: str):
    from holorag import HoloRAGConfig

    return HoloRAGConfig(
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        save_dir=save_dir,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_dtype=args.embedding_dtype,
        task_profile=args.task_profile,
        use_paragraph_as_chunk=not args.disable_paragraph_as_chunk,
        index_extraction_mode=args.index_extraction_mode,
        qa_max_input_tokens=args.qa_max_input_tokens,
        qa_evidence_token_budget=args.qa_evidence_token_budget,
        intent_use_llm=args.intent_use_llm,
        enable_entity_similarity_edges=not args.disable_entity_similarity_edges,
        entity_similarity_threshold=args.entity_similarity_threshold,
        entity_similarity_top_k=args.entity_similarity_top_k,
        enable_granularity_awareness=not args.disable_granularity_awareness,
        enable_sentence_layer=not args.disable_sentence_layer,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        spacy_model_name=args.spacy_model_name,
        passage_output_top_k=max(args.topk_passages, args.passage_output_top_k),
        qa_passage_top_k=args.topk_passages,
        enable_granularity_pagerank_bias=not args.disable_granularity_pagerank_bias,
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
    )


def _arg_provided(argv: Sequence[str], name: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in argv)


def apply_ablation_defaults(args: argparse.Namespace, ablation_name: str, argv: Sequence[str]) -> None:
    if ablation_name not in {"wo_sentence_layer", "wo_granularity_awareness"}:
        return

    if ablation_name == "wo_sentence_layer":
        defaults = {
            "--topk_passages": ("topk_passages", 6),
            "--passage_output_top_k": ("passage_output_top_k", 12),
            "--qa_evidence_token_budget": ("qa_evidence_token_budget", 2400),
            "--fact_rerank_llm_candidate_k": ("fact_rerank_llm_candidate_k", 24),
            "--fact_rerank_llm_keep_k": ("fact_rerank_llm_keep_k", 12),
            "--evidence_extra_ranked_sentence_k": ("evidence_extra_ranked_sentence_k", 0),
            "--evidence_max_sentences": ("evidence_max_sentences", 0),
            "--evidence_title_limit": ("evidence_title_limit", 5),
            "--evidence_passage_context_k": ("evidence_passage_context_k", 6),
            "--evidence_passage_excerpt_tokens": ("evidence_passage_excerpt_tokens", 560),
        }
    else:
        defaults = {
            "--topk_passages": ("topk_passages", 6),
            "--passage_output_top_k": ("passage_output_top_k", 12),
            "--qa_evidence_token_budget": ("qa_evidence_token_budget", 2600),
            "--fact_rerank_llm_candidate_k": ("fact_rerank_llm_candidate_k", 28),
            "--fact_rerank_llm_keep_k": ("fact_rerank_llm_keep_k", 14),
            "--evidence_extra_ranked_sentence_k": ("evidence_extra_ranked_sentence_k", 24),
            "--evidence_max_sentences": ("evidence_max_sentences", 40),
            "--evidence_title_limit": ("evidence_title_limit", 4),
            "--evidence_passage_context_k": ("evidence_passage_context_k", 4),
            "--evidence_passage_excerpt_tokens": ("evidence_passage_excerpt_tokens", 320),
        }
    for flag, (attr, value) in defaults.items():
        if not _arg_provided(argv, flag):
            setattr(args, attr, value)

    if ablation_name == "wo_sentence_layer":
        if not _arg_provided(argv, "--disable_sentence_layer"):
            args.disable_sentence_layer = True
        if not _arg_provided(argv, "--disable_granularity_awareness"):
            args.disable_granularity_awareness = False
        if not _arg_provided(argv, "--disable_granularity_pagerank_bias"):
            args.disable_granularity_pagerank_bias = False
    else:
        if not _arg_provided(argv, "--disable_granularity_awareness"):
            args.disable_granularity_awareness = True
        if not _arg_provided(argv, "--disable_sentence_layer"):
            args.disable_sentence_layer = False
        if not _arg_provided(argv, "--disable_granularity_pagerank_bias"):
            args.disable_granularity_pagerank_bias = True

    if not _arg_provided(argv, "--enable_fact_source_first_evidence"):
        args.enable_fact_source_first_evidence = False
    if not _arg_provided(argv, "--enable_fact_chunk_boost"):
        args.enable_fact_chunk_boost = False
    if not _arg_provided(argv, "--enable_fair_sentence_context"):
        args.enable_fair_sentence_context = False


def prebuild_or_reuse_indexes(
    rag,
    samples: Sequence[Dict[str, Any]],
    shared_index_root: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    index_latencies: List[float] = []
    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        sample_dir = shared_index_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        index_path = resolve_index_path(sample_dir)
        metadata_path = sample_dir / "metadata.json"

        if index_path.exists():
            latency = 0.0
            valid = True
            stats = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    stats = metadata.get("stats", {})
                except Exception:
                    stats = {}
            records.append(
                {
                    "sample_id": sample_id,
                    "index_path": str(index_path),
                    "index_latency": latency,
                    "valid": valid,
                    "stats": stats,
                    "reused": True,
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            logger.info("[index][%d/%d] start %s", i, len(samples), sample_id)
            documents = build_documents(sample)
            rag.index(documents)
            index_path = sample_dir / INDEX_FILENAME
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
                        "index_path": str(index_path),
                        "stats": stats,
                        "index_latency": latency,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "index_path": str(index_path),
                    "index_latency": latency,
                    "valid": True,
                    "stats": stats,
                    "reused": False,
                }
            )
            logger.info("[index][%d/%d] %s | %.3fs", i, len(samples), sample_id, latency)
        except Exception as exc:
            records.append(
                {
                    "sample_id": sample_id,
                    "index_path": str(index_path),
                    "index_latency": 0.0,
                    "valid": False,
                    "error": str(exc),
                    "reused": False,
                }
            )
            logger.exception("Index build failed for sample %s", sample_id)

    summary = {
        "num_samples": len(samples),
        "num_valid_samples": sum(1 for item in records if item.get("valid")),
        "num_reused_samples": sum(1 for item in records if item.get("reused")),
        "avg_index_latency": _avg(index_latencies),
        "total_index_runtime": time.perf_counter() - t_start,
        "shared_index_root": str(shared_index_root),
    }
    return {"records": records, "summary": summary}


def records_from_shared_indexes(samples: Sequence[Dict[str, Any]], shared_index_root: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    index_latencies: List[float] = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        sample_dir = shared_index_root / sample_id
        index_path = resolve_index_path(sample_dir)
        metadata_path = sample_dir / "metadata.json"
        stats = {}
        index_latency = 0.0
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                stats = metadata.get("stats", {}) or {}
                index_latency = float(metadata.get("index_latency", 0.0) or 0.0)
            except Exception:
                stats = {}
                index_latency = 0.0
        if index_latency > 0:
            index_latencies.append(index_latency)
        records.append({
            "sample_id": sample_id,
            "index_path": str(index_path),
            "index_latency": index_latency,
            "valid": index_path.exists(),
            "stats": stats,
            "reused": True,
        })
    summary = {
        "num_samples": len(samples),
        "num_valid_samples": sum(1 for item in records if item.get("valid")),
        "num_reused_samples": sum(1 for item in records if item.get("reused")),
        "avg_index_latency": _avg(index_latencies),
        "total_index_runtime": float(sum(index_latencies)),
        "shared_index_root": str(shared_index_root),
    }
    return {"records": records, "summary": summary}


def refresh_index_records(
    samples: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    shared_index_root: Path,
) -> List[Dict[str, Any]]:
    record_by_id = {str(item.get("sample_id", "")): dict(item) for item in records}
    refreshed: List[Dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        record = record_by_id.get(sample_id, {"sample_id": sample_id, "reused": True})
        index_path = resolve_index_path(shared_index_root / sample_id)
        record["index_path"] = str(index_path)
        record["valid"] = index_path.exists()
        record["reused"] = True
        refreshed.append(record)
    return refreshed


def run_eval(
    rag,
    samples: Sequence[Dict[str, Any]],
    index_records: Sequence[Dict[str, Any]],
    run_dir: Path,
    dataset_tag: str,
    logger: logging.Logger,
    token_counter: TokenCounter,
    per_example_filename: str = "per_example.jsonl",
    metrics_variant: str = "baseline",
) -> Dict[str, Any]:
    record_by_id = {str(item.get("sample_id", "")): item for item in index_records}
    per_example_path = run_dir / per_example_filename
    per_example_path.write_text("", encoding="utf-8")

    em_scores: List[float] = []
    f1_scores: List[float] = []
    retrieval_latencies: List[float] = []
    retrieval_pipeline_latencies: List[float] = []
    qa_latencies: List[float] = []
    query_total_latencies: List[float] = []
    entity_counts: List[float] = []
    fact_counts: List[float] = []
    sentence_counts: List[float] = []
    chunk_counts: List[float] = []
    node_counts: List[float] = []
    edge_counts: List[float] = []
    edge_type_totals: Dict[str, List[float]] = {}
    evidence_tokens: List[float] = []
    qa_failures = 0

    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", f"sample_{i}"))
        question = str(sample.get("question", "")).strip()
        if not question:
            continue
        index_record = record_by_id.get(sample_id)
        if not index_record or not index_record.get("valid"):
            continue
        index_path = Path(str(index_record.get("index_path", "")))
        if not index_path.exists():
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

        ranked_passages = result.get("ranked_passages", []) or []
        qa_messages = result.get("qa_messages", []) or []
        evidence_text = ""
        if len(qa_messages) >= 2 and isinstance(qa_messages[1], dict):
            evidence_text = str(qa_messages[1].get("content", ""))
        if not evidence_text:
            evidence_text = format_qa_evidence_from_ranked_passages(ranked_passages, rag.config.qa_passage_top_k)
        final_evidence_tokens = token_counter.count(evidence_text)
        evidence_tokens.append(float(final_evidence_tokens))
        predicted_answer = str(result.get("predicted_answer", "")).strip()
        em, f1 = best_em_f1(build_gold_answers(sample), predicted_answer)
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

        row = {
            "query_id": sample_id,
            "question": question,
            "gold_answer": sample.get("answer", ""),
            "predicted_answer": predicted_answer,
            "F1": f1,
            "EM": em,
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
        }
        with per_example_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(
            "[%s][%d/%d] %s | f1=%.4f em=%.4f | running_f1=%.4f running_em=%.4f | q=%.3fs",
            dataset_tag,
            i,
            len(samples),
            sample_id,
            f1,
            em,
            _avg(f1_scores),
            _avg(em_scores),
            query_elapsed,
        )

    metrics = {
        "variant": metrics_variant,
        "num_queries": len(f1_scores),
        "F1": _avg(f1_scores),
        "EM": _avg(em_scores),
        "qa_failures": qa_failures,
        "retrieval_latency": _avg(retrieval_latencies),
        "retrieval_pipeline_latency": _avg(retrieval_pipeline_latencies),
        "qa_latency": _avg(qa_latencies),
        "retrieval_qa_latency": _avg(query_total_latencies),
        "query_runtime": time.perf_counter() - t_start,
        "nodes": _avg(node_counts),
        "entity_nodes": _avg(entity_counts),
        "fact_nodes": _avg(fact_counts),
        "sentence_nodes": _avg(sentence_counts),
        "chunk_nodes": _avg(chunk_counts),
        "edges": _avg(edge_counts),
        "final_evidence_tokens": _avg(evidence_tokens),
        "final_evidence_tokenizer": token_counter.method,
    }
    for edge_type, values in sorted(edge_type_totals.items()):
        metrics[f"edge_{edge_type}"] = _avg(values)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HoloRAG baseline eval on sampled dataset.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "musique_jsonl", "canonical_jsonl", "2wiki_json", "canonical_json"])
    parser.add_argument("--dataset_name", type=str, default="", help="Name used in logs and output names; inferred from samples or file name when omitted.")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_queries", type=int, default=200)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shared_index_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--ablation_name", type=str, default="", help="Ablation label appended to per-query log tags; defaults to --run_name when set.")

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
    parser.add_argument("--task_profile", type=str, default="multi_hop", choices=["auto", "single_hop", "multi_hop", "long_context"])
    parser.add_argument("--recompute_only", action="store_true", help="Skip indexing and recompute metrics from existing shared indexes only.")
    parser.add_argument("--skip_llm_health_check", action="store_true")
    parser.add_argument("--llm_health_timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    from holorag import HoloRAG

    dataset_format = detect_dataset_format(args.dataset_file, args.dataset_format)
    all_samples = load_samples(args.dataset_file, dataset_format)
    dataset_name = infer_dataset_name(args.dataset_file, all_samples, args.dataset_name)
    output_dir_is_ablation = "ablation" in Path(args.output_dir).as_posix().lower()
    ablation_name = args.ablation_name.strip() or (args.run_name.strip() if output_dir_is_ablation else "")
    apply_ablation_defaults(args, ablation_name, sys.argv[1:])
    log_tag = build_log_tag(dataset_name, ablation_name)
    run_name = args.run_name.strip() or f"{dataset_name}_{args.num_eval_queries}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Python executable: %s", sys.executable)
    if not args.skip_llm_health_check:
        check_llm_server(args.llm_base_url, args.llm_name, args.llm_health_timeout, logger)

    filtered = maybe_filter_split(all_samples, args.split)
    if not filtered:
        raise ValueError("No samples available after split filtering.")
    samples = sample_queries(filtered, run_dir / "sampled_queries.json", args.seed, args.num_eval_queries)

    rag = HoloRAG(build_config(args, save_dir=str(run_dir / "workdir")))
    token_counter = TokenCounter(args.llm_name, logger)
    shared_index_root = Path(args.shared_index_root).expanduser().resolve()
    shared_index_root.mkdir(parents=True, exist_ok=True)
    if args.recompute_only:
        previous_records_path = run_dir / "shared_index_records.json"
        if previous_records_path.exists():
            records = json.loads(previous_records_path.read_text(encoding="utf-8"))
            records = refresh_index_records(samples, records, shared_index_root)
            index_data = {
                "records": records,
                "summary": json.loads((run_dir / "shared_index_summary.json").read_text(encoding="utf-8"))
                if (run_dir / "shared_index_summary.json").exists()
                else {"avg_index_latency": 0.0, "total_index_runtime": 0.0},
            }
            previous_records_path.write_text(json.dumps(index_data["records"], ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            index_data = records_from_shared_indexes(samples, shared_index_root)
            (run_dir / "shared_index_records.json").write_text(json.dumps(index_data["records"], ensure_ascii=False, indent=2), encoding="utf-8")
            (run_dir / "shared_index_summary.json").write_text(json.dumps(index_data["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
        per_example_filename = "per_example.jsonl"
        metrics_variant = "holorag"
    else:
        index_data = prebuild_or_reuse_indexes(rag, samples, shared_index_root, logger)
        (run_dir / "shared_index_records.json").write_text(json.dumps(index_data["records"], ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "shared_index_summary.json").write_text(json.dumps(index_data["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
        per_example_filename = "per_example.jsonl"
        metrics_variant = "holorag"

    metrics = run_eval(
        rag,
        samples,
        index_data["records"],
        run_dir,
        log_tag,
        logger,
        token_counter=token_counter,
        per_example_filename=per_example_filename,
        metrics_variant=metrics_variant,
    )
    metrics["index_latency"] = float(index_data["summary"].get("avg_index_latency", 0.0))
    metrics["shared_index_runtime"] = float(index_data["summary"].get("total_index_runtime", 0.0))
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
    metric_headers.extend(sorted(key for key in metrics if key.startswith("edge_")))
    ordered_metrics = {key: metrics.get(key, "") for key in metric_headers}

    summary_rows = [ordered_metrics]
    metrics_json_name = "metrics_summary.json"
    metrics_csv_name = "metrics_summary.csv"
    (run_dir / metrics_json_name).write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [",".join(metric_headers), ",".join(str(ordered_metrics.get(k, "")) for k in metric_headers)]
    (run_dir / metrics_csv_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_file": args.dataset_file,
                "dataset_format": dataset_format,
                "dataset_name": dataset_name,
                "ablation_name": ablation_name,
                "split": args.split,
                "seed": args.seed,
                "num_eval_queries": args.num_eval_queries,
                "llm_base_url": args.llm_base_url,
                "llm_name": args.llm_name,
                "embedding_name": args.embedding_name,
                "embedding_device": args.embedding_device,
                "spacy_model_name": args.spacy_model_name,
                "task_profile": args.task_profile,
                "use_paragraph_as_chunk": not args.disable_paragraph_as_chunk,
                "index_extraction_mode": args.index_extraction_mode,
                "intent_use_llm": args.intent_use_llm,
                "enable_entity_similarity_edges": not args.disable_entity_similarity_edges,
                "entity_similarity_threshold": args.entity_similarity_threshold,
                "entity_similarity_top_k": args.entity_similarity_top_k,
                "enable_granularity_awareness": rag.config.enable_granularity_awareness,
                "enable_sentence_layer": rag.config.enable_sentence_layer,
                "enable_granularity_pagerank_bias": rag.config.enable_granularity_pagerank_bias,
                "topk_passages": args.topk_passages,
                "qa_max_input_tokens": args.qa_max_input_tokens,
                "qa_evidence_token_budget": args.qa_evidence_token_budget,
                "fact_rerank_use_llm": rag.config.fact_rerank_use_llm,
                "fact_rerank_llm_candidate_k": rag.config.fact_rerank_llm_candidate_k,
                "fact_rerank_llm_keep_k": rag.config.fact_rerank_llm_keep_k,
                "enable_fact_source_first_evidence": rag.config.enable_fact_source_first_evidence,
                "enable_fact_chunk_boost": rag.config.enable_fact_chunk_boost,
                "fact_chunk_boost": rag.config.fact_chunk_boost,
                "enable_fair_sentence_context": rag.config.enable_fair_sentence_context,
                "evidence_extra_ranked_sentence_k": rag.config.evidence_extra_ranked_sentence_k,
                "evidence_max_sentences": rag.config.evidence_max_sentences,
                "evidence_title_limit": rag.config.evidence_title_limit,
                "evidence_passage_context_k": rag.config.evidence_passage_context_k,
                "evidence_passage_excerpt_tokens": rag.config.evidence_passage_excerpt_tokens,
                "shared_index_root": str(shared_index_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Saved metrics: %s", run_dir / metrics_json_name)
    logger.info("Run complete | F1=%.4f EM=%.4f N=%d", metrics["F1"], metrics["EM"], metrics["num_queries"])


if __name__ == "__main__":
    main()
