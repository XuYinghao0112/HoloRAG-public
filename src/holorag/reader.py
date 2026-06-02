import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import HoloRAGConfig
from .llm_client import LocalLLMClient


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\S+", str(text or ""))


def _token_count(text: str) -> int:
    return len(_word_tokens(text))


def _truncate_words(text: str, max_tokens: int) -> str:
    tokens = _word_tokens(text)
    if len(tokens) <= max_tokens:
        return str(text or "").strip()
    return " ".join(tokens[:max(0, max_tokens)]).strip()


def _content_terms(text: str) -> set:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "who", "what", "when", "where", "which", "how", "was", "were", "is", "are", "did", "does",
        "do", "this", "that", "its", "his", "her", "their", "person", "place", "film", "song",
    }
    return {term for term in re.findall(r"[A-Za-z0-9']+", str(text or "").lower()) if len(term) > 2 and term not in stopwords}


def _passage_excerpt(text: str, question: str, max_tokens: int) -> str:
    text = " ".join(str(text or "").split())
    if max_tokens <= 0:
        return ""
    if _token_count(text) <= max_tokens:
        return text
    terms = _content_terms(question)
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()] or [text]
    scored = []
    for index, sentence in enumerate(sentences):
        words = _content_terms(sentence)
        lead_bonus = 1.0 if index == 0 else 0.5 if index == 1 else 0.0
        scored.append((len(terms & words) + lead_bonus, -index, sentence))
    selected = []
    used = 0
    for _, _, sentence in sorted(scored, reverse=True):
        cost = _token_count(sentence)
        if cost <= 0:
            continue
        if selected and used + cost > max_tokens:
            continue
        if cost > max_tokens:
            sentence = _truncate_words(sentence, max_tokens - used)
            cost = _token_count(sentence)
        selected.append(sentence)
        used += cost
        if used >= max_tokens:
            break
    return " ".join(selected) if selected else _truncate_words(text, max_tokens)


def format_passages(ranked_passages: Sequence[Dict], top_k: int, question: str = "", max_passage_tokens: int = 900) -> str:
    parts: List[str] = []
    for index, passage in enumerate(list(ranked_passages)[:top_k], start=1):
        title = str(passage.get("title", "")).strip()
        text = _passage_excerpt(str(passage.get("text", "")).strip(), question, max_passage_tokens)
        if not text:
            continue
        header = f"Passage {index}"
        if title:
            header += f" [{title}]"
        parts.append(f"{header}:\n{text}")
    return "\n\n".join(parts).strip()


def parse_answer(raw_text: str) -> Tuple[str, str]:
    text = str(raw_text or "").strip()
    if not text:
        return "", ""
    if "Answer:" in text:
        thought, answer = text.rsplit("Answer:", 1)
        answer = answer.strip().splitlines()[0].strip() if answer.strip() else ""
        return thought.replace("Thought:", "", 1).strip(), answer.strip().strip('"').strip("'").rstrip(".")
    lines = text.splitlines()
    answer = lines[-1].strip() if lines else ""
    return "\n".join(lines[:-1]).replace("Thought:", "", 1).strip(), answer.rstrip(".")


def _extract_or_candidates(question: str) -> List[str]:
    text = " ".join(str(question or "").replace("?", " ?").split())
    lowered = text.lower()
    matches = []
    if "," in text and " or " in lowered:
        matches = re.findall(r"(.+?)\s+or\s+(.+?)(?:\?|$)", text.rsplit(",", 1)[-1], flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b(?:between|of|from|than)\s+(.+?)\s+or\s+(.+?)(?:\?|$)", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"([^,?]+?)\s+or\s+([^,?]+?)(?:\?|$)", text, flags=re.IGNORECASE)
    candidates = []
    for left, right in matches[-1:]:
        for item in (left, right):
            cleaned = item.strip(" ,.?\"'")
            cleaned = re.sub(r"^(?:do|does|did|is|are|was|were|can|could|has|have|had)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^(?:the|film|song|person)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+(?:first|earlier|later|older|younger)$", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if 1 <= len(cleaned.split()) <= 8:
                candidates.append(cleaned)
    deduped = list(dict.fromkeys(candidates))
    if len(deduped) >= 2:
        return deduped[:2]
    if re.match(r"^(do|does|did|is|are|was|were|can|could|has|have|had)\b", lowered):
        return ["yes", "no"]
    return deduped


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _candidate_from_answer(answer: str, candidates: Sequence[str]) -> Optional[str]:
    answer_norm = _normalize_for_match(answer)
    if not answer_norm:
        return None
    for candidate in candidates:
        cand_norm = _normalize_for_match(candidate)
        if cand_norm and (cand_norm in answer_norm or answer_norm in cand_norm):
            return candidate
    return None


def _normalize_answer(question: str, answer: str, candidates: Sequence[str]) -> Tuple[str, str]:
    answer = str(answer or "").strip().strip('"').strip("'").rstrip(".").strip()
    constrained = _candidate_from_answer(answer, candidates)
    if constrained:
        return constrained, "candidate_normalized"
    if candidates == ["yes", "no"]:
        lower = answer.lower()
        if re.search(r"\byes\b", lower) and not re.search(r"\bno\b", lower):
            return "yes", "candidate_normalized"
        if re.search(r"\bno\b", lower) and not re.search(r"\byes\b", lower):
            return "no", "candidate_normalized"
    for separator in ["\n", ";", " because ", " since "]:
        if separator in answer:
            answer = answer.split(separator, 1)[0].strip().rstrip(".").strip()
    return answer, "holorag_single_pass"


class QAReader:
    def __init__(self, config: HoloRAGConfig, llm_client: LocalLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def answer(
        self,
        question: str,
        ranked_passages: Sequence[Dict],
        ranked_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        evidence: Dict = None,
    ) -> Dict:
        candidates = _extract_or_candidates(question)
        messages = self._build_messages(
            question=question,
            ranked_passages=ranked_passages,
            evidence=evidence or {},
            answer_candidates=candidates,
        )
        raw = self.llm_client.infer_messages_text(messages, fallback="Answer: ")
        thought, answer = parse_answer(raw)
        normalized_answer, answer_mode = _normalize_answer(question, answer, candidates)
        return {
            "thought": thought,
            "answer": normalized_answer,
            "raw_response": raw,
            "messages": messages,
            "answer_mode": answer_mode,
        }

    def _build_messages(
        self,
        question: str,
        ranked_passages: Sequence[Dict],
        evidence: Dict,
        answer_candidates: Sequence[str] = (),
    ) -> List[Dict]:
        evidence_text = str(evidence.get("packed_text", "")).strip()
        if not evidence_text:
            per_passage_budget = max(80, min(512, int(self.config.qa_max_input_tokens) // max(1, min(self.config.qa_passage_top_k, 2))))
            evidence_text = format_passages(
                ranked_passages,
                min(self.config.qa_passage_top_k, 2),
                question=question,
                max_passage_tokens=per_passage_budget,
            )
        candidate_block = ""
        if answer_candidates:
            candidate_block = "\nAnswer candidates: " + " | ".join(str(item) for item in answer_candidates)
        user_content = (
            "Use only the retrieved evidence to answer the question. "
            "Return exactly one final line in the form 'Answer: <short answer>'. "
            "The answer should be a concise entity, date, number, title, or phrase; do not include explanation in the final answer.\n\n"
            f"{evidence_text}\n\nQuestion: {question}{candidate_block}\nThought: "
        )
        user_content = self._fit_user_content(user_content)
        system_prompt = (
            "You are a precise reading-comprehension assistant. "
            "Answer from the retrieved evidence only and keep the final answer short."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _fit_user_content(self, content: str) -> str:
        budget = max(256, int(self.config.qa_max_input_tokens))
        if _token_count(content) <= budget:
            return content
        return _truncate_words(content, budget)
