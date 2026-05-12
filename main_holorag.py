import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def is_musique_single_sample(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("paragraphs"), list)


def convert_payload_to_documents(payload: Any) -> List[Dict[str, str]]:
    if is_musique_single_sample(payload):
        documents = []
        for index, paragraph in enumerate(payload["paragraphs"]):
            title = str(paragraph.get("title", f"doc_{index}"))
            text = str(paragraph.get("paragraph_text", paragraph.get("text", "")))
            documents.append({
                "title": title,
                "text": text,
                "idx": paragraph.get("idx", index),
                "is_supporting": paragraph.get("is_supporting", False),
            })
        return documents

    if not isinstance(payload, list):
        raise ValueError(
            "Unsupported corpus format. Expected a document list or a MuSiQue single-sample JSON object."
        )

    documents = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            documents.append({"title": f"doc_{index}", "text": item})
        elif isinstance(item, dict):
            title = str(item.get("title", f"doc_{index}"))
            text = str(item.get("text", item.get("content", item.get("paragraph_text", ""))))
            documents.append({"title": title, "text": text})
    return documents


def load_documents(corpus_file: str) -> List[Dict[str, str]]:
    payload = load_json_file(corpus_file)
    return convert_payload_to_documents(payload)


def load_query_text(corpus_file: str, explicit_query_text: Optional[str]) -> str:
    if explicit_query_text is not None:
        return explicit_query_text

    payload = load_json_file(corpus_file)
    if is_musique_single_sample(payload):
        question = str(payload.get("question", "")).strip()
        if question:
            return question

    raise ValueError(
        "--query_text is required unless --corpus_file points to a MuSiQue single-sample JSON with a question field."
    )


def load_sample_metadata(corpus_file: str) -> Dict[str, Any]:
    payload = load_json_file(corpus_file)
    if not is_musique_single_sample(payload):
        return {}
    return {
        "sample_id": payload.get("id"),
        "question": payload.get("question"),
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
    gold_answer = metadata.get("answer")
    aliases = metadata.get("answer_aliases", []) or []
    candidates = [candidate for candidate in [gold_answer, *aliases] if candidate]
    normalized_candidates = [normalize_answer_text(candidate) for candidate in candidates]

    searched_texts = []
    searched_texts.extend(item.get("text", "") for item in result.get("evidence", {}).get("entity", []))
    searched_texts.extend(item.get("text", "") for item in result.get("evidence", {}).get("sentence", []))
    searched_texts.extend(item.get("text", "") for item in result.get("evidence", {}).get("chunk", []))
    searched_texts.append(result.get("evidence", {}).get("qa_context", ""))
    searched_blob = normalize_answer_text("\n".join(searched_texts))

    matched_candidates = [candidate for candidate, normalized in zip(candidates, normalized_candidates) if normalized and normalized in searched_blob]
    return {
        "gold_answer": gold_answer,
        "gold_aliases": aliases,
        "matched": bool(matched_candidates),
        "matched_candidates": matched_candidates,
    }


def build_config(args: argparse.Namespace) -> "HoloRAGConfig":
    from src.holorag import HoloRAGConfig

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
        linking_top_k=args.linking_top_k,
        fact_candidate_top_k=args.fact_candidate_top_k,
        retrieval_top_k=args.retrieval_top_k,
        fact_top_k=args.fact_top_k,
        fact_rerank_top_k=args.fact_rerank_top_k,
        fact_output_top_k=args.fact_output_top_k,
        passage_output_top_k=args.passage_output_top_k,
        qa_passage_top_k=args.qa_passage_top_k,
        qa_evidence_max_tokens=args.qa_evidence_max_tokens,
        dense_passage_weight=args.dense_passage_weight,
        graph_passage_weight=args.graph_passage_weight,
        fact_passage_weight=args.fact_passage_weight,
        fact_entity_spread_weight=args.fact_entity_spread_weight,
        bridge_entity_top_k=args.bridge_entity_top_k,
        passage_node_weight=args.passage_node_weight,
        task_profile=args.task_profile,
        intent_query_weight=args.intent_query_weight,
        min_intent_confidence=args.min_intent_confidence,
        enable_terminal_hop_override=not args.disable_terminal_hop_override,
        enable_recognition_filter=not args.disable_recognition_filter,
        enable_intent_routing=not args.disable_intent_routing,
        enable_sentence_layer=not args.disable_sentence_layer,
        enable_chunk_bridges=not args.disable_chunk_bridges,
        enable_alias_linking=not args.disable_alias_linking,
        enable_granularity_biased_transition=not args.disable_biased_transition,
        enable_llm_judge=args.enable_llm_judge,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HoloRAG indexing and query pipeline.")
    parser.add_argument("command", choices=["index", "query"], help="Which stage to run.")
    parser.add_argument("--corpus_file", type=str, default="reproduce/dataset/sample_corpus.json")
    parser.add_argument("--query_text", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/holorag_demo")
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="gpu1")
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
    parser.add_argument("--qa_evidence_max_tokens", type=int, default=400)
    parser.add_argument("--dense_passage_weight", type=float, default=0.55)
    parser.add_argument("--graph_passage_weight", type=float, default=0.30)
    parser.add_argument("--fact_passage_weight", type=float, default=0.15)
    parser.add_argument("--fact_entity_spread_weight", type=float, default=0.30)
    parser.add_argument("--bridge_entity_top_k", type=int, default=6)
    parser.add_argument("--passage_node_weight", type=float, default=0.10)
    parser.add_argument("--task_profile", type=str, default="auto", choices=["auto", "single_hop", "multi_hop", "long_context"])
    parser.add_argument("--intent_query_weight", type=float, default=0.7)
    parser.add_argument("--min_intent_confidence", type=float, default=0.35)
    parser.add_argument("--disable_sentence_layer", action="store_true")
    parser.add_argument("--disable_recognition_filter", action="store_true")
    parser.add_argument("--disable_intent_routing", action="store_true")
    parser.add_argument("--disable_chunk_bridges", action="store_true")
    parser.add_argument("--disable_alias_linking", action="store_true")
    parser.add_argument("--disable_biased_transition", action="store_true")
    parser.add_argument("--disable_terminal_hop_override", action="store_true")
    parser.add_argument("--enable_llm_judge", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from src.holorag import HoloRAG

    holorag = HoloRAG(build_config(args))
    if args.command == "index":
        documents = load_documents(args.corpus_file)
        result = holorag.index(documents)
        metadata = load_sample_metadata(args.corpus_file)
        if metadata:
            result["sample_metadata"] = metadata
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    query_text = load_query_text(args.corpus_file, args.query_text)
    result = holorag.query(query_text)
    metadata = load_sample_metadata(args.corpus_file)
    if metadata:
        result["sample_metadata"] = metadata
        result["answer_match"] = build_answer_match(result, metadata)
        result_path = os.path.join(args.output_dir, "last_query_result.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
