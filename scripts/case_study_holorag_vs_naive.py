import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PROFILE_ALPHA_PRIORS = {
    "single_hop": {"fact": 0.60, "sentence": 0.30, "chunk": 0.10},
    "multi_hop": {"fact": 0.40, "sentence": 0.40, "chunk": 0.20},
    "long_context": {"fact": 0.10, "sentence": 0.20, "chunk": 0.70},
}


DEFAULT_RESULT_FILES = {
    "hotpotqa": REPO_ROOT / "results/hotpotqa_eval/ablation_runs/full/per_example.jsonl",
    "2wiki": REPO_ROOT / "results/2wiki_eval/ablation_runs/full/per_example.jsonl",
    "musique": REPO_ROOT / "results/musique_eval/ablation_runs/full/per_example.jsonl",
    "naturalquestions": REPO_ROOT / "results/naturalquestions_eval/full/per_example.jsonl",
    "narrativeqa": REPO_ROOT / "results/narrativeqa_eval/full/per_example.jsonl",
}


class OfflineLLMClient:
    """Minimal LLM client that always returns the caller-provided fallback."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.stats = {"completion_calls": 0, "json_calls": 0, "text_calls": 0}

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def infer_json(self, system_prompt: str, user_prompt: str, fallback: Dict[str, Any], max_tokens: int = None) -> Tuple[Dict[str, Any], str]:
        self.stats["json_calls"] += 1
        return fallback, ""

    def infer_messages_text(self, messages: List[Dict[str, str]], fallback: str = "", max_tokens: int = None) -> str:
        self.stats["text_calls"] += 1
        return fallback


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def profile_from_row(row: Dict[str, Any], dataset: str) -> str:
    if dataset == "naturalquestions":
        return "single_hop"
    if dataset == "narrativeqa":
        return "long_context"
    alpha_c = float(row.get("alpha_C", 0.0) or 0.0)
    alpha_f = float(row.get("alpha_F", 0.0) or 0.0)
    alpha_s = float(row.get("alpha_S", 0.0) or 0.0)
    if alpha_c >= 0.42:
        return "long_context"
    if alpha_f + alpha_s >= 0.62:
        return "multi_hop"
    return "single_hop"


def alpha_from_row(row: Dict[str, Any], profile: str) -> Dict[str, float]:
    alpha = {
        "fact": float(row.get("alpha_F", 0.0) or 0.0),
        "sentence": float(row.get("alpha_S", 0.0) or 0.0),
        "chunk": float(row.get("alpha_C", 0.0) or 0.0),
    }
    if sum(alpha.values()) > 0:
        return alpha
    return dict(PROFILE_ALPHA_PRIORS.get(profile, PROFILE_ALPHA_PRIORS["multi_hop"]))


def metric_selection_score(row: Dict[str, Any], dataset: str) -> float:
    correct = max(float(row.get("EM", 0.0) or 0.0), float(row.get("F1", 0.0) or 0.0))
    counts = [
        int(row.get("num_fact_evidence", 0) or 0),
        int(row.get("num_sentence_evidence", 0) or 0),
        int(row.get("num_chunk_evidence", 0) or 0),
    ]
    used = [
        int(row.get("used_tokens_F", 0) or 0),
        int(row.get("used_tokens_S", 0) or 0),
        int(row.get("used_tokens_C", 0) or 0),
    ]
    nonzero_counts = sum(1 for value in counts if value > 0)
    nonzero_tokens = sum(1 for value in used if value > 0)
    final_tokens = int(row.get("final_evidence_tokens", 0) or 0)
    graph_size = int(row.get("nodes", 0) or 0)
    profile = profile_from_row(row, dataset)
    profile_bonus = 0.0
    if profile == "long_context":
        profile_bonus = min(120.0, used[2] / 20.0)
    elif profile == "single_hop":
        profile_bonus = min(120.0, used[0] + used[1])
    else:
        profile_bonus = 40.0 * nonzero_tokens + min(80.0, used[1] / 6.0)
    return (
        1000.0 * correct
        + 90.0 * nonzero_counts
        + 70.0 * nonzero_tokens
        + profile_bonus
        + min(80.0, final_tokens / 18.0)
        + min(40.0, graph_size / 80.0)
    )


def select_metric_candidates(rows: Sequence[Dict[str, Any]], dataset: str, limit: int) -> List[Dict[str, Any]]:
    existing = [row for row in rows if Path(str(row.get("index_path", ""))).exists()]
    ranked = sorted(existing, key=lambda row: metric_selection_score(row, dataset), reverse=True)
    return ranked[: max(1, limit)]


def build_config(args: argparse.Namespace, profile: str) -> Any:
    from holorag.config import HoloRAGConfig

    return HoloRAGConfig(
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_dtype=args.embedding_dtype,
        task_profile=profile,
        passage_output_top_k=max(args.naive_top_k, 10),
        qa_passage_top_k=args.naive_top_k,
        fact_rerank_use_llm=False,
        enable_fact_source_first_evidence=True,
        enable_fact_chunk_boost=True,
        fact_chunk_boost=0.4,
        enable_fair_sentence_context=True,
        evidence_extra_ranked_sentence_k=3,
        evidence_max_sentences=15,
        evidence_title_limit=3,
        evidence_passage_context_k=1,
        evidence_passage_excerpt_tokens=100,
        evidence_chunk_max_tokens=args.holorag_chunk_max_tokens,
        evidence_packing_mode="alpha_count",
        evidence_alpha_total_units=20,
        evidence_allow_underfill=True,
        evidence_min_score=0.0,
        evidence_redundancy_threshold=0.85,
        evidence_use_alpha_weights=True,
        execution_mode="sequential",
    )


class EvidenceBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.components_by_profile: Dict[str, Dict[str, Any]] = {}

    def components(self, profile: str) -> Dict[str, Any]:
        if profile in self.components_by_profile:
            return self.components_by_profile[profile]
        from holorag.embedding_model import NVEmbedV2Encoder
        from holorag.extractors import QueryDecomposer, TripleExtractor
        from holorag.intent import IntentRouter
        from holorag.pagerank import GranularityPageRank
        from holorag.retriever import Retriever

        config = build_config(self.args, profile)
        if self.args.enable_llm_calls:
            from holorag.llm_client import LocalLLMClient

            llm_client = LocalLLMClient(config)
        else:
            llm_client = OfflineLLMClient(config)
        embedder = NVEmbedV2Encoder(config)
        components = {
            "config": config,
            "llm_client": llm_client,
            "embedder": embedder,
            "retriever": Retriever(config, embedder, llm_client),
            "pagerank": GranularityPageRank(config),
            "extractor": TripleExtractor(llm_client, index_extraction_mode=config.index_extraction_mode),
            "decomposer": QueryDecomposer(llm_client),
            "router": IntentRouter(config, llm_client),
        }
        self.components_by_profile[profile] = components
        return components

    def build(self, row: Dict[str, Any], dataset: str) -> Dict[str, Any]:
        question = " ".join(str(row.get("question", "")).split())
        profile = profile_from_row(row, dataset)
        alpha = alpha_from_row(row, profile)
        comp = self.components(profile)
        config = comp["config"]
        graph_state = load_pickle(Path(str(row["index_path"])))
        graph = graph_state["graph"]

        query_parse = comp["extractor"].extract_query(question)
        if config.enable_query_decomposition and profile == "multi_hop":
            sub_questions = comp["decomposer"].decompose(question)
        else:
            sub_questions = [question]

        retrieval = comp["retriever"].retrieve(
            query=question,
            query_entities=query_parse.get("entities", []),
            query_facts=query_parse.get("triples", []),
            sub_questions=sub_questions,
            graph=graph,
            state=graph_state,
            alpha=alpha,
        )
        pagerank_scores = comp["pagerank"].run(graph, alpha=alpha, seed_scores=retrieval["seed_scores"])
        ranked_passages = comp["retriever"].rank_passages(
            graph=graph,
            pagerank_scores=pagerank_scores,
            channel_scores=retrieval["channel_scores"],
            ranked_facts=retrieval["ranked_facts"],
            alpha=alpha,
        )
        ranked_evidence = comp["retriever"].rank_evidence(
            graph=graph,
            pagerank_scores=pagerank_scores,
            channel_scores=retrieval["channel_scores"],
            ranked_facts=retrieval["ranked_facts"],
            ranked_passages=ranked_passages,
            profile=profile,
            query=question,
            sub_questions=sub_questions,
            token_budget=0,
            alpha=alpha,
        )
        naive_evidence = self._naive_chunk_evidence(
            graph=graph,
            chunk_scores=retrieval["channel_scores"].get("chunk", {}),
            top_k=self.args.naive_top_k,
        )
        return {
            "dataset": dataset,
            "query_id": row.get("query_id", ""),
            "question": question,
            "gold_answer": row.get("gold_answer", ""),
            "predicted_answer": row.get("predicted_answer", ""),
            "EM": row.get("EM", 0.0),
            "F1": row.get("F1", 0.0),
            "profile": profile,
            "alpha": alpha,
            "sub_questions": sub_questions,
            "index_path": row.get("index_path", ""),
            "metric_row_final_evidence_tokens": row.get("final_evidence_tokens", 0),
            "holorag": self._summarize_holorag(ranked_evidence),
            "naive": naive_evidence,
            "contrast_score": self._contrast_score(ranked_evidence, naive_evidence),
        }

    def _naive_chunk_evidence(self, graph: Any, chunk_scores: Dict[str, float], top_k: int) -> Dict[str, Any]:
        chunks: List[Dict[str, Any]] = []
        lines: List[str] = []
        for rank, (chunk_id, score) in enumerate(sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:top_k], start=1):
            if chunk_id not in graph:
                continue
            attrs = graph.nodes[chunk_id]
            metadata = attrs.get("metadata", {}) or {}
            title = str(metadata.get("title", "")).strip()
            text = " ".join(str(attrs.get("text", "")).split())
            line = f"Chunk {rank}: [{title}] {text}" if title else f"Chunk {rank}: {text}"
            lines.append(line)
            chunks.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "score": float(score),
                "title": title,
                "text": text,
                "tokens": word_count(line),
            })
        packed_text = "\n\n".join(lines)
        return {
            "description": f"NaiveRAG top-{top_k} dense chunks",
            "chunks": chunks,
            "packed_text": packed_text,
            "packed_token_count": word_count(packed_text),
        }

    def _summarize_holorag(self, ranked_evidence: Dict[str, Any]) -> Dict[str, Any]:
        records = list(ranked_evidence.get("packed_records", []) or [])
        return {
            "description": "HoloRAG alpha-guided final evidence",
            "packed_text": str(ranked_evidence.get("packed_text", "")),
            "packed_token_count": int(ranked_evidence.get("packed_token_count", 0) or 0),
            "used_tokens_by_granularity": ranked_evidence.get("used_tokens_by_granularity", {}),
            "evidence_counts_by_granularity": ranked_evidence.get("evidence_counts_by_granularity", {}),
            "evidence_count_limits_by_granularity": ranked_evidence.get("evidence_count_limits_by_granularity", {}),
            "records": [
                {
                    "kind": item.get("kind", ""),
                    "label": item.get("label", ""),
                    "title": item.get("title", ""),
                    "score": item.get("score", 0.0),
                    "tokens": item.get("tokens", 0),
                    "line": item.get("line", ""),
                }
                for item in records
            ],
        }

    def _contrast_score(self, ranked_evidence: Dict[str, Any], naive: Dict[str, Any]) -> float:
        counts = ranked_evidence.get("evidence_counts_by_granularity", {}) or {}
        nonzero = sum(1 for key in ("fact", "sentence", "chunk") if int(counts.get(key, 0) or 0) > 0)
        holo_tokens = max(1, int(ranked_evidence.get("packed_token_count", 0) or 0))
        naive_tokens = max(1, int(naive.get("packed_token_count", 0) or 0))
        return 100.0 * nonzero + min(200.0, naive_tokens / holo_tokens * 40.0)


def choose_case(builder: EvidenceBuilder, rows: Sequence[Dict[str, Any]], dataset: str, candidates_per_dataset: int) -> Dict[str, Any]:
    candidates = select_metric_candidates(rows, dataset, candidates_per_dataset)
    built = [builder.build(row, dataset) for row in candidates]
    return max(built, key=lambda item: item["contrast_score"])


def readable_evidence(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\s+(?=(?:Fact|Fact source evidence|Sentence evidence|Passage|Chunk \d+):)",
        "\n",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def simple_case(case: Dict[str, Any]) -> Dict[str, Any]:
    counts = case["holorag"].get("evidence_counts_by_granularity", {})
    used = case["holorag"].get("used_tokens_by_granularity", {})
    return {
        "dataset": case["dataset"],
        "query_id": case["query_id"],
        "question": case["question"],
        "gold_answer": case["gold_answer"],
        "predicted_answer": case["predicted_answer"],
        "profile": case["profile"],
        "token_stats": {
            "holorag_total_tokens": case["holorag"]["packed_token_count"],
            "naive_total_tokens": case["naive"]["packed_token_count"],
            "holorag_counts_fact_sentence_chunk": [
                counts.get("fact", 0),
                counts.get("sentence", 0),
                counts.get("chunk", 0),
            ],
            "holorag_tokens_fact_sentence_chunk": [
                used.get("fact", 0),
                used.get("sentence", 0),
                used.get("chunk", 0),
            ],
        },
        "holorag_final_evidence": readable_evidence(case["holorag"]["packed_text"]),
        "naive_returned_chunks": readable_evidence(case["naive"]["packed_text"]),
    }


def trim_text(text: str, max_words: int) -> str:
    text = str(text or "")
    if max_words <= 0:
        return text
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= max_words:
        return text
    return text[:matches[max_words - 1].end()].rstrip() + " ..."


def render_markdown(cases: Sequence[Dict[str, Any]], max_words_per_block: int) -> str:
    simple_cases = [case if "token_stats" in case else simple_case(case) for case in cases]
    lines = [
        "# HoloRAG vs NaiveRAG Evidence Case Study",
        "",
        "Only the case-study essentials are kept here: question, answer, token usage, and the final evidence text sent to the answer model.",
        "",
        "## Token Summary",
        "",
        "| Dataset | Profile | HoloRAG tokens | NaiveRAG tokens | HoloRAG counts F/S/C | HoloRAG tokens F/S/C |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in simple_cases:
        stats = case["token_stats"]
        lines.append(
            f"| {case['dataset']} | {case['profile']} | "
            f"{stats['holorag_total_tokens']} | {stats['naive_total_tokens']} | "
            f"{'/'.join(str(item) for item in stats['holorag_counts_fact_sentence_chunk'])} | "
            f"{'/'.join(str(item) for item in stats['holorag_tokens_fact_sentence_chunk'])} |"
        )
    lines.extend([
        "",
        "## Evidence Examples",
        "",
    ])
    for case in simple_cases:
        stats = case["token_stats"]
        lines.extend([
            f"## {case['dataset']}",
            "",
            f"- query_id: `{case['query_id']}`",
            f"- Question: {case['question']}",
            f"- Gold / prediction: {case['gold_answer']} / {case['predicted_answer']}",
            f"- Tokens: HoloRAG {stats['holorag_total_tokens']} vs NaiveRAG {stats['naive_total_tokens']}",
            "",
            "### HoloRAG Final Evidence",
            "",
            "```text",
            trim_text(case["holorag_final_evidence"], max_words_per_block),
            "```",
            "",
            "### NaiveRAG Returned Chunks",
            "",
            "```text",
            trim_text(case["naive_returned_chunks"], max_words_per_block),
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a five-dataset case study comparing HoloRAG final evidence with NaiveRAG top chunks."
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results/case_study_holorag_vs_naive")
    parser.add_argument("--candidates_per_dataset", type=int, default=5)
    parser.add_argument("--naive_top_k", type=int, default=4)
    parser.add_argument("--max_words_per_block", type=int, default=0, help="Maximum words per evidence block in Markdown; 0 keeps full evidence.")
    parser.add_argument("--embedding_name", type=str, default="/data/xyh/models/NV-Embed-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_dtype", type=str, default="bfloat16")
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--enable_llm_calls", action="store_true", help="Use the configured LLM server for query parsing/decomposition; default is offline heuristic fallback.")
    parser.add_argument("--holorag_chunk_max_tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = EvidenceBuilder(args)
    cases: List[Dict[str, Any]] = []
    for dataset, result_path in DEFAULT_RESULT_FILES.items():
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file for {dataset}: {result_path}")
        rows = load_jsonl(result_path)
        print(f"[{dataset}] loaded {len(rows)} rows from {result_path}")
        case = choose_case(builder, rows, dataset, args.candidates_per_dataset)
        print(
            f"[{dataset}] selected {case['query_id']} "
            f"(HoloRAG tokens={case['holorag']['packed_token_count']}, "
            f"Naive tokens={case['naive']['packed_token_count']})"
        )
        cases.append(case)

    json_path = args.output_dir / "case_study.json"
    md_path = args.output_dir / "case_study.md"
    json_path.write_text(json.dumps({"cases": [simple_case(case) for case in cases]}, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(cases, args.max_words_per_block), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
