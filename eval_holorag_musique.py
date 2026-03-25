import argparse
import glob
import httpx
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
SAMPLE_GROUP_ROOT = REPO_ROOT / "reproduce" / "test" / "groups"
OUTPUT_GROUP_ROOTNAME = "groups"
DEFAULT_SAMPLE_GROUP = "musique_01_10"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _normalize_token(token: str) -> str:
    token = str(token or "").lower().strip()
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("ics") and len(token) > 4:
        token = token[:-1]
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    return token


def normalize_answer(answer: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(r'''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'''))

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(answer or "")))))


def normalized_answer_tokens(answer: str) -> List[str]:
    normalized = normalize_answer(answer)
    return [_normalize_token(token) for token in normalized.split() if _normalize_token(token)]


def soft_answer_match(gold_answer: str, predicted_answer: str) -> bool:
    gold_norm = normalize_answer(gold_answer)
    predicted_norm = normalize_answer(predicted_answer)
    if not gold_norm or not predicted_norm:
        return False
    if gold_norm == predicted_norm:
        return True
    gold_tokens = set(normalized_answer_tokens(gold_answer))
    predicted_tokens = set(normalized_answer_tokens(predicted_answer))
    if not gold_tokens or not predicted_tokens:
        return False
    if gold_tokens <= predicted_tokens and len(gold_tokens) <= 2:
        return True
    if predicted_tokens <= gold_tokens and len(predicted_tokens) <= 2:
        return True
    return False


def compute_exact_match(gold_answers: Sequence[str], predicted_answer: str) -> float:
    return max(
        (1.0 if soft_answer_match(gold_answer, predicted_answer) else 0.0)
        for gold_answer in gold_answers
    ) if gold_answers else 0.0


def compute_f1(gold_answers: Sequence[str], predicted_answer: str) -> float:
    predicted_tokens = normalized_answer_tokens(predicted_answer)
    if not gold_answers:
        return 0.0

    best_f1 = 0.0
    for gold_answer in gold_answers:
        gold_tokens = normalized_answer_tokens(gold_answer)
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / max(1, len(predicted_tokens))
        recall = num_same / max(1, len(gold_tokens))
        best_f1 = max(best_f1, 2 * precision * recall / max(1e-8, precision + recall))
    return best_f1


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def supporting_paragraphs(sample: Dict) -> List[Dict]:
    return [paragraph for paragraph in sample.get("paragraphs", []) if paragraph.get("is_supporting")]


def build_gold_answers(sample: Dict) -> List[str]:
    answers = []
    if sample.get("answer"):
        answers.append(str(sample["answer"]))
    answers.extend(str(alias) for alias in sample.get("answer_aliases", []) if alias)
    deduped = []
    seen = set()
    for answer in answers:
        key = normalize_answer(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(answer)
    return deduped


def compute_answer_match(sample: Dict, result: Dict) -> Dict[str, Any]:
    gold_answer = sample.get("answer")
    aliases = sample.get("answer_aliases", []) or []
    candidates = [candidate for candidate in [gold_answer, *aliases] if candidate]
    normalized_candidates = [normalize_answer(candidate) for candidate in candidates]

    searched_texts = []
    searched_texts.extend(item.get("text", "") for item in result.get("evidence", {}).get("entity", []))
    searched_texts.extend(item.get("text", "") for item in result.get("evidence", {}).get("sentence", []))
    searched_texts.extend(item.get("text", "") for item in result.get("evidence", {}).get("chunk", []))
    searched_texts.append(result.get("evidence", {}).get("qa_context", ""))
    searched_blob = normalize_answer("\n".join(str(text) for text in searched_texts))

    matched_candidates = [
        candidate
        for candidate, normalized in zip(candidates, normalized_candidates)
        if normalized and normalized in searched_blob
    ]
    return {
        "gold_answer": gold_answer,
        "gold_aliases": aliases,
        "matched": bool(matched_candidates),
        "matched_candidates": matched_candidates,
    }


class SimpleLLMClient:
    def __init__(self, base_url: str, model_name: str) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY", "sk-"),
            max_retries=3,
            http_client=httpx.Client(trust_env=False),
        )

    def infer_text(self, system_prompt: str, user_prompt: str, fallback: str = "", max_tokens: int = 64) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "".join(
                    item.get("text", "") if isinstance(item, dict) else getattr(item, "text", str(item))
                    for item in content
                ).strip()
            return str(content).strip()
        except Exception:
            return fallback


def generate_answer(llm_client: Any, question: str, qa_context: str, fallback: str = "") -> str:
    if not qa_context.strip():
        return fallback
    response = llm_client.infer_text(
        system_prompt=(
            "Answer the question using only the provided retrieved context. "
            "Respond with a short answer only. If the answer cannot be found, reply with Unknown."
        ),
        user_prompt=f"Context:\n{qa_context}\n\nQuestion:\n{question}\n\nShort answer:",
        fallback=fallback,
        max_tokens=64,
    )
    cleaned = response.strip()
    if not cleaned:
        return fallback
    cleaned = re.sub(r"(?i)^answer\s*:\s*", "", cleaned).strip()
    return cleaned


def sort_sample_paths(paths: Sequence[Path]) -> List[Path]:
    def key(path: Path) -> Tuple[int, str]:
        match = re.search(r"sample_musique(\d+)\.json$", path.name)
        return (int(match.group(1)) if match else 10**9, path.name)

    return sorted(paths, key=key)


def sample_index_from_name(sample_name: str) -> int:
    match = re.search(r"sample_musique(\d+)$", sample_name)
    if not match:
        raise ValueError(f"Unsupported sample name: {sample_name}")
    return int(match.group(1))


def sample_group_name(sample_name: str) -> str:
    index = sample_index_from_name(sample_name)
    group_start = ((index - 1) // 10) * 10 + 1
    group_end = group_start + 9
    return f"musique_{group_start:02d}_{group_end:02d}"


def grouped_result_path(outputs_dir: Path, sample_name: str, result_filename: str) -> Path:
    return outputs_dir / OUTPUT_GROUP_ROOTNAME / sample_group_name(sample_name) / sample_name / result_filename


def evaluate_sample(
    sample_path: Path,
    result_path: Path,
    llm_client: Any,
    generate_qa_answer: bool,
    retrieval_k: int,
) -> Dict:
    sample = load_json(sample_path)
    result = load_json(result_path)

    support_paragraph_list = supporting_paragraphs(sample)
    support_titles = [str(item.get("title", "")).strip() for item in support_paragraph_list if item.get("title")]
    top_passages = result.get("ranked_passages", [])[:retrieval_k]
    top_titles = [str(item.get("title", "")).strip() for item in top_passages if item.get("title")]

    matched_support_titles = sorted(set(support_titles) & set(top_titles))
    passage_recall_at_k = (
        len(matched_support_titles) / len(set(support_titles))
        if support_titles else 0.0
    )
    passage_hit_at_k = 1.0 if matched_support_titles else 0.0

    qa_context = str(result.get("evidence", {}).get("qa_context", "")).strip()
    gold_answers = build_gold_answers(sample)
    stored_predicted_answer = str(result.get("predicted_answer", "")).strip()
    if stored_predicted_answer:
        predicted_answer = stored_predicted_answer
    elif generate_qa_answer and llm_client is not None:
        predicted_answer = generate_answer(
            llm_client=llm_client,
            question=str(sample.get("question", "")),
            qa_context=qa_context,
            fallback="",
        )
    else:
        predicted_answer = ""

    qa_exact_match = compute_exact_match(gold_answers, predicted_answer)
    qa_f1 = compute_f1(gold_answers, predicted_answer)
    qa_context_normalized = normalize_answer(qa_context)
    context_contains_answer = 1.0 if any(normalize_answer(answer) in qa_context_normalized for answer in gold_answers if normalize_answer(answer)) else 0.0

    answer_match = compute_answer_match(sample, result)
    reasoning_chain = result.get("reasoning_chain", [])
    query_resolutions = result.get("query_entity_resolutions", [])

    return {
        "sample_file": sample_path.name,
        "sample_id": sample.get("id"),
        "question": sample.get("question", ""),
        "gold_answers": gold_answers,
        "predicted_answer": predicted_answer,
        "qa_exact_match": qa_exact_match,
        "qa_f1": qa_f1,
        "passage_recall_at_5": passage_recall_at_k,
        "passage_hit_at_5": passage_hit_at_k,
        "support_titles": support_titles,
        "retrieved_titles_top5": top_titles,
        "matched_support_titles": matched_support_titles,
        "context_contains_answer": context_contains_answer,
        "answer_match_flag": bool(answer_match.get("matched")),
        "answer_match_candidates": answer_match.get("matched_candidates", []),
        "qa_context_words": len(qa_context.split()),
        "qa_context_chars": len(qa_context),
        "ranked_passages_count": len(result.get("ranked_passages", [])),
        "ranked_facts_count": len(result.get("ranked_facts", [])),
        "bridge_entities": result.get("bridge_entities", []),
        "query_entities": result.get("query_entities", []),
        "query_entity_resolutions": query_resolutions,
        "sub_questions": result.get("sub_questions", []),
        "reasoning_chain": reasoning_chain,
        "reasoning_chain_count": len(reasoning_chain),
        "retrieved_passages_top5": [
            {
                "title": item.get("title", ""),
                "score": item.get("score", 0.0),
                "words": len(str(item.get("text", "")).split()),
                "score_breakdown": item.get("score_breakdown", {}),
            }
            for item in top_passages
        ],
    }


def summarize_results(per_example_results: Sequence[Dict], qa_enabled: bool) -> Dict:
    if not per_example_results:
        return {}
    summary = {
        "num_examples": len(per_example_results),
        "passage_recall_at_5": round(mean(item["passage_recall_at_5"] for item in per_example_results), 4),
        "passage_hit_at_5": round(mean(item["passage_hit_at_5"] for item in per_example_results), 4),
        "context_contains_answer_rate": round(mean(item["context_contains_answer"] for item in per_example_results), 4),
        "answer_match_rate": round(mean(1.0 if item["answer_match_flag"] else 0.0 for item in per_example_results), 4),
        "avg_qa_context_words": round(mean(item["qa_context_words"] for item in per_example_results), 1),
        "avg_ranked_facts": round(mean(item["ranked_facts_count"] for item in per_example_results), 1),
        "avg_reasoning_chain_count": round(mean(item["reasoning_chain_count"] for item in per_example_results), 1),
    }
    if qa_enabled:
        summary["qa_exact_match"] = round(mean(item["qa_exact_match"] for item in per_example_results), 4)
        summary["qa_f1"] = round(mean(item["qa_f1"] for item in per_example_results), 4)
        summary["qa_non_empty_prediction_rate"] = round(
            mean(1.0 if str(item["predicted_answer"]).strip() else 0.0 for item in per_example_results),
            4,
        )
    return summary


def print_report(summary: Dict, per_example_results: Sequence[Dict], qa_enabled: bool) -> None:
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    print("\nPer-example details:")
    for item in per_example_results:
        payload = {
            "sample_file": item["sample_file"],
            "sample_id": item["sample_id"],
            "qa_exact_match": round(item["qa_exact_match"], 4),
            "qa_f1": round(item["qa_f1"], 4),
            "passage_recall_at_5": round(item["passage_recall_at_5"], 4),
            "context_contains_answer": item["context_contains_answer"],
            "gold_answers": item["gold_answers"],
            "predicted_answer": item["predicted_answer"] if qa_enabled else "",
            "support_titles": item["support_titles"],
            "retrieved_titles_top5": item["retrieved_titles_top5"],
            "matched_support_titles": item["matched_support_titles"],
            "qa_context_words": item["qa_context_words"],
            "bridge_entities": item["bridge_entities"],
            "sub_questions": item["sub_questions"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HoloRAG on MuSiQue grouped sample outputs.")
    parser.add_argument(
        "--samples_glob",
        type=str,
        default=str(SAMPLE_GROUP_ROOT / DEFAULT_SAMPLE_GROUP / "sample_musique*.json"),
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "qwen_72b_result"),
    )
    parser.add_argument(
        "--result_filename",
        type=str,
        default="last_query_result.json",
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=5,
        help="Evaluate passage recall using the top-k ranked passages.",
    )
    parser.add_argument(
        "--skip_qa_generation",
        action="store_true",
        help="Skip final QA answer generation and only compute retrieval/context metrics.",
    )
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_name", type=str, default="/data/xyh/models/Qwen2.5-72B-Instruct")
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(REPO_ROOT / "outputs" / "qwen_72b_result" / "eval" / "holorag_eval_musique_samples_01_10.json"),
    )
    args = parser.parse_args()

    sample_paths = sort_sample_paths([Path(path) for path in glob.glob(args.samples_glob)])
    if not sample_paths:
        raise FileNotFoundError(f"No sample files matched: {args.samples_glob}")

    llm_client = None
    qa_enabled = not args.skip_qa_generation
    if qa_enabled:
        llm_client = SimpleLLMClient(base_url=args.llm_base_url, model_name=args.llm_name)

    per_example_results = []
    outputs_dir = Path(args.outputs_dir)
    for sample_path in sample_paths:
        sample_name = sample_path.stem
        result_path = grouped_result_path(outputs_dir, sample_name, args.result_filename)
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file for {sample_name}: {result_path}")
        per_example_results.append(
            evaluate_sample(
                sample_path=sample_path,
                result_path=result_path,
                llm_client=llm_client,
                generate_qa_answer=qa_enabled,
                retrieval_k=args.retrieval_k,
            )
        )

    summary = summarize_results(per_example_results, qa_enabled=qa_enabled)
    payload = {
        "summary": summary,
        "settings": {
            "samples_glob": args.samples_glob,
            "outputs_dir": args.outputs_dir,
            "result_filename": args.result_filename,
            "retrieval_k": args.retrieval_k,
            "qa_generation_enabled": qa_enabled,
            "llm_base_url": args.llm_base_url if qa_enabled else "",
            "llm_name": args.llm_name if qa_enabled else "",
        },
        "per_example": per_example_results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print_report(summary, per_example_results, qa_enabled=qa_enabled)
    print(f"\nSaved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
