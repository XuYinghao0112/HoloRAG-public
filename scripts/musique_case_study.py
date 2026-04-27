#!/usr/bin/env python3
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_answer(text: str) -> str:
    import re

    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pick_best_example(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    优先挑选“回答好 + 检索有挑战(Recall@5 < 1)”的样例，模仿论文里展示 hard-but-solved case 的风格。
    """
    candidates = [row for row in rows if float(row.get("recall@5", 0.0) or 0.0) < 1.0]
    pool = candidates or list(rows)

    def score(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        em = float(row.get("EM") or 0.0)
        f1 = float(row.get("F1") or 0.0)
        recall = float(row.get("recall@5") or 0.0)
        predicted = str(row.get("predicted_answer", "")).strip()
        # 目标：正确(EM/F1高) + 有挑战(recall<1 且不要太低)
        challenge = 1.0 - abs(recall - 0.75)
        non_empty = 1.0 if predicted else 0.0
        return (2.0 * em + f1 + 0.6 * challenge + 0.2 * non_empty, em, f1, recall)

    return max(pool, key=score)


def build_runtime_config(
    run_config: Dict[str, Any],
    save_dir: str,
    embedding_device_override: str = "",
) -> "HoloRAGConfig":
    from src.holorag import HoloRAGConfig

    return HoloRAGConfig(
        llm_base_url=str(run_config.get("llm_base_url", "http://127.0.0.1:8000/v1")),
        llm_model_name=str(run_config.get("llm_name", "/data/xyh/models/Qwen2.5-72B-Instruct")),
        embedding_model_name=str(run_config.get("embedding_name", "/data/xyh/models/NV-Embed-v2")),
        save_dir=save_dir,
        embedding_device=(
            embedding_device_override.strip()
            if embedding_device_override.strip()
            else str(run_config.get("embedding_device", "cuda:0"))
        ),
        linking_top_k=int(run_config.get("topk_triples", 5)),
        passage_output_top_k=max(int(run_config.get("topk_passages", 5)), 10),
        qa_passage_top_k=int(run_config.get("topk_passages", 5)),
        retrieval_top_k=max(
            int(run_config.get("retrieval_top_k", 20)),
            int(run_config.get("topk_passages", 5)),
        ),
        entity_alias_threshold=float(run_config.get("synonym_threshold", 0.8)),
        pagerank_alpha=float(run_config.get("ppr_damping", 0.5)),
        temperature=float(run_config.get("temperature", 0.0)),
        passage_node_weight=float(run_config.get("passage_node_weight", 0.05)),
        dense_passage_weight=float(run_config.get("dense_passage_weight", 0.55)),
        graph_passage_weight=float(run_config.get("graph_passage_weight", 0.30)),
        fact_passage_weight=float(run_config.get("fact_passage_weight", 0.15)),
    )


def get_support_titles(sample: Dict[str, Any]) -> List[str]:
    titles = []
    for paragraph in sample.get("paragraphs", []):
        if paragraph.get("is_supporting"):
            title = str(paragraph.get("title", "")).strip()
            if title:
                titles.append(title)
    # 保序去重
    deduped = []
    seen = set()
    for title in titles:
        key = normalize_answer(title)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(title)
    return deduped


def title_recall_at_k(support_titles: Sequence[str], retrieved_titles: Sequence[str]) -> float:
    support_keys = {normalize_answer(item) for item in support_titles if normalize_answer(item)}
    if not support_keys:
        return 0.0
    retrieved_keys = {normalize_answer(item) for item in retrieved_titles if normalize_answer(item)}
    return len(support_keys & retrieved_keys) / len(support_keys)


def resolve_query_context_for_fact_ranking(
    holorag: "HoloRAG",
    query: str,
    graph: Any,
    state: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[str]]:
    raw_query_entities = holorag.triple_extractor.extract_query_entities(query)
    initial_sub_questions = holorag.query_decomposer.decompose(query) or [query.strip()]
    query_entity_resolutions = holorag._resolve_query_entities(query, initial_sub_questions, graph, raw_query_entities)
    confident = [item for item in query_entity_resolutions if item.get("confident")]
    if confident:
        sub_questions = holorag.query_decomposer.decompose(query, resolved_entities=confident) or initial_sub_questions
    else:
        sub_questions = initial_sub_questions
    query_entities = [item["resolved_text"] for item in confident] or raw_query_entities
    query_entity_node_ids = [item["node_id"] for item in confident]
    return sub_questions, query_entity_resolutions, query_entities, query_entity_node_ids


def derive_triples(
    holorag: "HoloRAG",
    query: str,
    state: Dict[str, Any],
    graph: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    holorag._ensure_fact_index(state, graph)
    holorag._prepare_retrieval_objects(state, graph)
    sub_questions, query_entity_resolutions, query_entities, query_entity_node_ids = resolve_query_context_for_fact_ranking(
        holorag=holorag,
        query=query,
        graph=graph,
        state=state,
    )
    fact_scores = holorag._get_multi_query_fact_scores(query, sub_questions, state)
    fact_lookup = {record["fact_id"]: record for record in state.get("facts", [])}
    query_to_triple = []
    for fact_id, score in sorted(fact_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        record = fact_lookup.get(fact_id)
        if not record:
            continue
        query_to_triple.append(
            {
                "fact_id": fact_id,
                "score": float(score),
                "text": str(record.get("text", "")),
                "head_id": record.get("head_id"),
                "tail_id": record.get("tail_id"),
            }
        )
    return {
        "sub_questions": sub_questions,
        "query_entity_resolutions": query_entity_resolutions,
        "query_entities": query_entities,
        "query_entity_node_ids": query_entity_node_ids,
        "query_to_triple_topk": query_to_triple,
    }


def fact_entities_covered(
    facts: Sequence[Dict[str, Any]],
    graph: Any,
    support_titles: Sequence[str],
) -> List[Dict[str, Any]]:
    support_keys = {normalize_answer(t) for t in support_titles if normalize_answer(t)}
    coverage = []
    for item in facts:
        head = str(graph.nodes.get(item.get("head_id"), {}).get("text", "")).strip() if item.get("head_id") else ""
        tail = str(graph.nodes.get(item.get("tail_id"), {}).get("text", "")).strip() if item.get("tail_id") else ""
        text = str(item.get("text", "")).strip()
        hit = False
        # 简单规则：如果 fact 的头/尾出现在支持段标题里，认为与 gold supporting evidence 有对齐
        for candidate in [head, tail]:
            if normalize_answer(candidate) in support_keys:
                hit = True
                break
        coverage.append({"text": text, "head": head, "tail": tail, "aligned_to_support_title": hit})
    return coverage


def format_title_list(titles: Sequence[str]) -> str:
    if not titles:
        return "(empty)"
    return " ".join(f"{idx}. {title}" for idx, title in enumerate(titles, start=1))


def build_markdown_report(payload: Dict[str, Any]) -> str:
    sample = payload["sample"]
    run_row = payload["selected_row"]
    result = payload["rerun_result"]
    derived = payload["derived"]
    support_titles = payload["support_titles"]
    retrieved_titles = payload["retrieved_titles_top5"]
    recall = payload["recall_at_5"]
    gold_answer = str(sample.get("answer", "")).strip()
    pred_answer = str(result.get("predicted_answer", "")).strip()
    question = str(sample.get("question", "")).strip()

    lines: List[str] = []
    lines.append("# MuSiQue Case Study (Auto-selected)")
    lines.append("")
    lines.append("## 1) Example Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Query | {question} |")
    lines.append(f"| Gold Answer | {gold_answer} |")
    lines.append(f"| Predicted Answer | {pred_answer} |")
    lines.append(f"| Recall@5 (title-level) | {recall:.4f} |")
    lines.append(f"| EM/F1 from previous 1000-run | {run_row.get('EM')} / {run_row.get('F1')} |")
    lines.append("")
    lines.append("中文说明：这个样例由脚本自动挑选，优先选择“回答质量高，但检索并非满分”的案例，便于做可解释分析。")
    lines.append("")
    lines.append("## 2) Supporting vs Retrieved Passages (Title)")
    lines.append("")
    lines.append(f"- Supporting Passages (Title): {format_title_list(support_titles)}")
    lines.append(f"- Retrieved Passages Top-5 (Title): {format_title_list(retrieved_titles)}")
    lines.append("")
    lines.append("## 3) Query to Triple (Top-5)")
    lines.append("")
    for idx, item in enumerate(derived["query_to_triple_topk"], start=1):
        lines.append(f"{idx}. ({item['text']}) [score={item['score']:.4f}]")
    if not derived["query_to_triple_topk"]:
        lines.append("(empty)")
    lines.append("")
    lines.append("## 4) Filtered Triple (Top-5 after HoloRAG rerank)")
    lines.append("")
    filtered = list(result.get("ranked_facts", []))[:5]
    for idx, item in enumerate(filtered, start=1):
        lines.append(f"{idx}. ({item.get('text', '')}) [score={float(item.get('score', 0.0)):.4f}]")
    if not filtered:
        lines.append("(empty)")
    lines.append("")
    lines.append("## 5) Intermediate Process (Key Steps)")
    lines.append("")
    lines.append(f"- Alpha (granularity preference): `{json.dumps(result.get('alpha', {}), ensure_ascii=False)}`")
    lines.append(f"- Query Entities: `{json.dumps(result.get('query_entities', []), ensure_ascii=False)}`")
    lines.append(f"- Sub-questions: `{json.dumps(result.get('sub_questions', []), ensure_ascii=False)}`")
    lines.append(f"- Bridge Entities: `{json.dumps(result.get('bridge_entities', []), ensure_ascii=False)}`")
    lines.append("")
    lines.append("### Reasoning Chain")
    for idx, hop in enumerate(result.get("reasoning_chain", []), start=1):
        lines.append(
            f"- Hop {idx}: Q=`{hop.get('sub_question','')}` | "
            f"A=`{hop.get('hop_answer','')}` | "
            f"Source=`{hop.get('source_title','')}`"
        )
    if not result.get("reasoning_chain"):
        lines.append("- (empty)")
    lines.append("")
    lines.append("### Top-5 Passages with Score Breakdown")
    for idx, p in enumerate(result.get("ranked_passages", [])[:5], start=1):
        bd = p.get("score_breakdown", {})
        lines.append(
            f"- {idx}. {p.get('title','')} | final={float(p.get('score',0.0)):.4f} | "
            f"dense={float(bd.get('dense',0.0)):.4f}, graph={float(bd.get('graph',0.0)):.4f}, fact={float(bd.get('fact',0.0)):.4f}"
        )
    if not result.get("ranked_passages"):
        lines.append("- (empty)")
    lines.append("")
    lines.append("### 中文解读")
    lines.append("- `Query to Triple` 反映的是“问题直接拉到的语义事实候选”；")
    lines.append("- `Filtered Triple` 是经过结构/关系/LLM筛选后的高价值事实，更接近最终推理链；")
    lines.append("- 如果 `Recall@5 < 1` 但答案仍正确，通常说明系统通过图扩散与多跳子问题修复了检索缺口。")
    lines.append("")
    lines.append("## 6) Multi-granularity Evidence (Entity / Sentence / Chunk)")
    lines.append("")
    evidence = result.get("evidence", {})
    for layer_name in ["entity", "sentence", "chunk"]:
        records = list(evidence.get(layer_name, []))
        lines.append(f"### {layer_name.title()} Evidence")
        if not records:
            lines.append("- (empty)")
            lines.append("")
            continue
        for idx, record in enumerate(records, start=1):
            text = " ".join(str(record.get("text", "")).strip().split())
            score = float(record.get("score", 0.0) or 0.0)
            metadata = record.get("metadata", {}) or {}
            title = str(metadata.get("title", "")).strip()
            chunk_id = str(metadata.get("chunk_id", "")).strip()
            suffix_parts = []
            if title:
                suffix_parts.append(f"title={title}")
            if chunk_id:
                suffix_parts.append(f"chunk_id={chunk_id}")
            suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
            lines.append(f"- {idx}. score={score:.4f}{suffix}: {text}")
        lines.append("")
    lines.append("中文说明：这三层是 HoloRAG 的核心。Entity 偏实体锚点，Sentence 偏关系表达，Chunk 偏上下文补全。")
    lines.append("")
    lines.append("## 7) Final Evidence Sent to Answer LLM")
    lines.append("")
    qa_context = str(evidence.get("qa_context", "") or "").strip()
    qa_messages = list(evidence.get("qa_messages", []))
    final_user_prompt = ""
    if qa_messages:
        final_user_prompt = str(qa_messages[-1].get("content", "") or "").strip()
    lines.append("- Final reader uses `qa_messages` (multi-message prompt).")
    lines.append(f"- Number of `qa_messages`: {len(qa_messages)}")
    lines.append(f"- `qa_context` length: {len(qa_context)} chars")
    lines.append("")
    lines.append("### qa_context (verbatim)")
    lines.append("```text")
    lines.append(qa_context if qa_context else "(empty)")
    lines.append("```")
    lines.append("")
    lines.append("### Final User Prompt To Reader (verbatim)")
    lines.append("```text")
    lines.append(final_user_prompt if final_user_prompt else "(empty)")
    lines.append("```")
    lines.append("")
    lines.append("中文说明：如果你问“最终给大模型回答的证据是什么样的”，最直接看这两个块：")
    lines.append("- `qa_context`：拼接后的证据文本视图（更像上下文包）；")
    lines.append("- `qa_messages[-1].content`：最终用户提示，包含推理提示 + 事实提示 + 证据段落。")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one strong MuSiQue case from a 1000-run, rerun it, and export a detailed case-study report."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=str(
            REPO_ROOT
            / "outputs"
            / "musique_eval"
            / "runs"
            / "musique_dev_seed42_1000_reuseidx_gpu6_rerank_d090_g010_f000_20260425_212614"
        ),
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default="",
        help="Optional fixed sample_id. If empty, script auto-selects one good example.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for case-study files. Default: <run_dir>/case_study/<sample_id>_<timestamp>",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="",
        help="Optional embedding device override, e.g. cuda:2 / cuda:6 / cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    per_example_path = run_dir / "per_example_results.jsonl"
    samples_dir = run_dir / "samples"
    run_config_path = run_dir / "config.json"
    if not per_example_path.exists():
        raise FileNotFoundError(f"Missing file: {per_example_path}")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing directory: {samples_dir}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing file: {run_config_path}")

    rows = load_jsonl(per_example_path)
    if not rows:
        raise ValueError(f"No rows in {per_example_path}")

    if args.sample_id:
        selected = None
        for row in rows:
            if str(row.get("query_id", "")).strip() == args.sample_id.strip():
                selected = row
                break
        if selected is None:
            raise ValueError(f"sample_id {args.sample_id} not found in {per_example_path}")
    else:
        selected = pick_best_example(rows)

    sample_id = str(selected.get("query_id", "")).strip()
    if not sample_id:
        raise ValueError("Selected row missing query_id")
    sample_path = samples_dir / f"{sample_id}.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    sample = load_json(sample_path)
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError(f"Sample {sample_id} has empty question")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        case_dir = Path(args.output_dir).resolve()
    else:
        case_dir = run_dir / "case_study" / f"{sample_id}_{timestamp}"
    case_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = load_json(run_config_path)
    case_runtime_dir = case_dir / "runtime"
    case_runtime_dir.mkdir(parents=True, exist_ok=True)

    from main_holorag import convert_payload_to_documents
    from src.holorag import HoloRAG

    config = build_runtime_config(
        run_cfg,
        save_dir=str(case_runtime_dir),
        embedding_device_override=args.embedding_device,
    )
    holorag = HoloRAG(config)
    documents = convert_payload_to_documents(sample)
    index_result = holorag.index(documents)
    rerun_result = holorag.query(question)

    state = holorag.load()
    graph = state["graph"]
    derived = derive_triples(holorag=holorag, query=question, state=state, graph=graph, top_k=args.top_k)

    support_titles = get_support_titles(sample)
    retrieved_titles_top5 = [str(item.get("title", "")).strip() for item in rerun_result.get("ranked_passages", [])[: args.top_k]]
    recall = title_recall_at_k(support_titles, retrieved_titles_top5)

    query_triple_cov = fact_entities_covered(derived["query_to_triple_topk"], graph, support_titles)
    filtered_cov = fact_entities_covered(list(rerun_result.get("ranked_facts", []))[: args.top_k], graph, support_titles)

    payload = {
        "selected_row": selected,
        "sample_path": str(sample_path),
        "sample": sample,
        "index_result": index_result,
        "rerun_result": rerun_result,
        "support_titles": support_titles,
        "retrieved_titles_top5": retrieved_titles_top5,
        "recall_at_5": recall,
        "derived": {
            **derived,
            "query_to_triple_coverage": query_triple_cov,
            "filtered_triple_coverage": filtered_cov,
        },
    }

    raw_json_path = case_dir / "case_payload.json"
    rerun_json_path = case_dir / "rerun_result.json"
    derived_json_path = case_dir / "derived_intermediate.json"
    report_path = case_dir / "report.md"

    raw_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rerun_json_path.write_text(json.dumps(rerun_result, ensure_ascii=False, indent=2), encoding="utf-8")
    derived_json_path.write_text(
        json.dumps(
            {
                "selected_row": selected,
                "support_titles": support_titles,
                "retrieved_titles_top5": retrieved_titles_top5,
                "recall_at_5": recall,
                "derived": payload["derived"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(build_markdown_report(payload), encoding="utf-8")

    print(json.dumps(
        {
            "selected_sample_id": sample_id,
            "question": question,
            "gold_answer": sample.get("answer"),
            "predicted_answer": rerun_result.get("predicted_answer", ""),
            "recall_at_5": recall,
            "output_dir": str(case_dir),
            "report": str(report_path),
            "rerun_result": str(rerun_json_path),
            "derived_intermediate": str(derived_json_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
