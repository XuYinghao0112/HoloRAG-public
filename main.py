import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

DOCUMENT_LIST_KEYS = ("documents", "docs", "corpus", "paragraphs", "contexts", "context", "passages", "items", "data")
TEXT_KEYS = ("text", "content", "contents", "body", "page_content", "paragraph_text")
TITLE_KEYS = ("title", "name", "doc_id", "id", "uid")


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_text_value(item: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def _coerce_document(item: Any, index: int) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        text = item
        title = f"doc_{index}"
        return {"title": title, "text": text} if text.strip() else None
    if isinstance(item, list) and len(item) >= 2:
        title = str(item[0]) if item[0] is not None else f"doc_{index}"
        body = item[1]
        text = " ".join(str(part) for part in body) if isinstance(body, list) else str(body)
        return {"title": title, "text": text} if text.strip() else None
    if not isinstance(item, dict):
        return None

    text = _first_text_value(item, list(TEXT_KEYS))
    if not text and isinstance(item.get("sentences"), list):
        text = " ".join(str(sentence) for sentence in item["sentences"])
    if not text.strip():
        return None
    title = _first_text_value(item, list(TITLE_KEYS)) or f"doc_{index}"
    document = {"title": title, "text": text}
    if "idx" in item:
        document["idx"] = item["idx"]
    if "is_supporting" in item:
        document["is_supporting"] = item["is_supporting"]
    return document


def _document_items_from_payload(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in DOCUMENT_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Expected a document list, or a JSON object containing a document-list field.")


def convert_payload_to_documents(payload: Any) -> List[Dict[str, Any]]:
    documents = []
    for index, item in enumerate(_document_items_from_payload(payload)):
        document = _coerce_document(item, index)
        if document:
            documents.append(document)
    if not documents:
        raise ValueError("No valid documents found. Each document needs a non-empty text/content/body field.")
    return documents


def load_documents(corpus_file: str) -> List[Dict[str, Any]]:
    return convert_payload_to_documents(load_json_file(corpus_file))


def load_query_text(corpus_file: str, explicit_query_text: Optional[str]) -> str:
    if explicit_query_text:
        return explicit_query_text
    payload = load_json_file(corpus_file)
    if isinstance(payload, dict):
        for key in ("query", "question"):
            value = payload.get(key)
            if value:
                return str(value)
    raise ValueError("--query_text is required unless corpus_file has a top-level query/question field.")


def load_query_metadata(corpus_file: str) -> Dict[str, Any]:
    payload = load_json_file(corpus_file)
    if not isinstance(payload, dict):
        return {}
    if not any(key in payload for key in ("query", "question", "answer", "answer_aliases", "question_decomposition")):
        return {}
    return {
        "sample_id": payload.get("id"),
        "question": payload.get("question", payload.get("query")),
        "answer": payload.get("answer"),
        "answer_aliases": payload.get("answer_aliases", []),
        "question_decomposition": [
            item.get("question", "")
            for item in payload.get("question_decomposition", [])
            if item.get("question")
        ],
    }


def normalize_answer_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", normalized).strip()


def build_answer_match(result: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [metadata.get("answer"), *(metadata.get("answer_aliases", []) or [])]
    candidates = [candidate for candidate in candidates if candidate]
    blob_parts = [result.get("predicted_answer", "")]
    blob_parts.extend(item.get("text", "") for item in result.get("ranked_passages", []))
    blob_parts.extend(item.get("text", "") for item in result.get("ranked_facts", []))
    blob = normalize_answer_text("\n".join(blob_parts))
    matched = [candidate for candidate in candidates if normalize_answer_text(candidate) in blob]
    return {"gold_answer": metadata.get("answer"), "gold_aliases": metadata.get("answer_aliases", []), "matched": bool(matched), "matched_candidates": matched}


def build_config(args: argparse.Namespace):
    from holorag import HoloRAGConfig

    return HoloRAGConfig(
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        save_dir=args.output_dir,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_dtype=args.embedding_dtype,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        spacy_model_name=args.spacy_model_name,
        use_paragraph_as_chunk=not args.disable_paragraph_as_chunk,
        task_profile=args.task_profile,
        enable_intent_routing=not args.disable_intent_routing,
        intent_use_llm=args.intent_use_llm,
        enable_query_decomposition=not args.disable_query_decomposition,
        enable_entity_similarity_edges=not args.disable_entity_similarity_edges,
        entity_similarity_threshold=args.entity_similarity_threshold,
        entity_similarity_top_k=args.entity_similarity_top_k,
        entity_top_k=args.entity_top_k,
        fact_top_k=args.fact_top_k,
        sentence_top_k=args.sentence_top_k,
        chunk_top_k=args.chunk_top_k,
        passage_output_top_k=args.passage_output_top_k,
        qa_passage_top_k=args.qa_passage_top_k,
        pagerank_alpha=args.pagerank_alpha,
        transition_lambda=args.transition_lambda,
        hub_penalty=args.hub_penalty,
        execution_mode=args.execution_mode,
        num_workers=args.num_workers,
        multi_worker_embedding_devices=args.multi_worker_embedding_devices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HoloRAG baseline.")
    parser.add_argument("command", choices=["index", "query"])
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--query_text", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/holorag_demo")
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="gpu1")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_seq_len", type=int, default=2048)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--chunk_size_words", type=int, default=256)
    parser.add_argument("--chunk_overlap_words", type=int, default=64)
    parser.add_argument("--spacy_model_name", type=str, default="en_core_web_sm")
    parser.add_argument("--disable_paragraph_as_chunk", action="store_true")
    parser.add_argument("--task_profile", type=str, default="auto", choices=["auto", "single_hop", "multi_hop", "long_context"])
    parser.add_argument("--disable_intent_routing", action="store_true")
    parser.add_argument("--intent_use_llm", action="store_true")
    parser.add_argument("--disable_query_decomposition", action="store_true")
    parser.add_argument("--disable_entity_similarity_edges", action="store_true")
    parser.add_argument("--entity_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--entity_similarity_top_k", type=int, default=2047)
    parser.add_argument("--entity_top_k", type=int, default=12)
    parser.add_argument("--fact_top_k", type=int, default=12)
    parser.add_argument("--sentence_top_k", type=int, default=20)
    parser.add_argument("--chunk_top_k", type=int, default=12)
    parser.add_argument("--passage_output_top_k", type=int, default=10)
    parser.add_argument("--qa_passage_top_k", type=int, default=4)
    parser.add_argument("--pagerank_alpha", type=float, default=0.5)
    parser.add_argument("--transition_lambda", type=float, default=1.2)
    parser.add_argument("--hub_penalty", type=float, default=0.08)
    parser.add_argument("--execution_mode", type=str, default="sequential", choices=["sequential", "multi_worker"])
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--multi_worker_embedding_devices", type=str, default="", help="Comma-separated devices for optional multi-worker retrieval encoders, e.g. cuda:0,cuda:3.")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from holorag import HoloRAG

    rag = HoloRAG(build_config(args))
    if args.command == "index":
        result = rag.index(load_documents(args.corpus_file))
        metadata = load_query_metadata(args.corpus_file)
        if metadata:
            result["sample_metadata"] = metadata
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    result = rag.query(load_query_text(args.corpus_file, args.query_text))
    metadata = load_query_metadata(args.corpus_file)
    if metadata:
        result["sample_metadata"] = metadata
        result["answer_match"] = build_answer_match(result, metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
