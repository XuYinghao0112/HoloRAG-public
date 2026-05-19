import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import NaiveHoloRAGConfig
from .llm_client import LocalLLMClient


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\S+", str(text or ""))


def _token_count(text: str) -> int:
    return len(_word_tokens(text))


def _truncate_words(text: str, max_tokens: int) -> str:
    tokens = _word_tokens(text)
    if len(tokens) <= max_tokens:
        return str(text or "").strip()
    return " ".join(tokens[:max_tokens]).strip()


def _query_terms(question: str) -> set:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "who", "what", "when", "where", "which", "how", "was", "were", "is", "are", "did", "does",
        "do", "this", "that", "its", "his", "her", "their", "person", "place", "film", "song",
    }
    return {term for term in re.findall(r"[A-Za-z0-9']+", question.lower()) if len(term) > 2 and term not in stopwords}


def _split_sentences(text: str) -> List[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", str(text or "")) if item.strip()]


def _passage_excerpt(text: str, question: str, max_tokens: int) -> str:
    text = str(text or "").strip()
    if max_tokens <= 0:
        return ""
    if _token_count(text) <= max_tokens:
        return text
    terms = _query_terms(question)
    sentences = _split_sentences(text)
    if not sentences:
        return _truncate_words(text, max_tokens)

    scored = []
    for index, sentence in enumerate(sentences):
        words = {word.lower() for word in re.findall(r"[A-Za-z0-9']+", sentence)}
        overlap = len(terms & words)
        # Keep lead sentences competitive because many Wikipedia paragraphs state
        # the bridge entity or answer in the first sentence with little lexical overlap.
        lead_bonus = 1.0 if index == 0 else 0.5 if index == 1 else 0.0
        scored.append((overlap + lead_bonus, -index, sentence))
    selected = []
    used = 0
    for _, _, sentence in sorted(scored, reverse=True):
        cost = _token_count(sentence)
        if cost == 0:
            continue
        if selected and used + cost > max_tokens:
            continue
        if cost > max_tokens:
            sentence = _truncate_words(sentence, max_tokens - used)
            cost = _token_count(sentence)
        if cost <= 0:
            continue
        selected.append(sentence)
        used += cost
        if used >= max_tokens:
            break
    if not selected:
        return _truncate_words(text, max_tokens)
    return " ".join(selected)


def format_passages(ranked_passages: Sequence[Dict], top_k: int, question: str = "", max_passage_tokens: int = 900) -> str:
    parts: List[str] = []
    for passage in list(ranked_passages)[:top_k]:
        title = str(passage.get("title", "")).strip()
        text = _passage_excerpt(str(passage.get("text", "")).strip(), question, max_passage_tokens)
        if not text:
            continue
        parts.append(f"Wikipedia Title: {title}\n{text}" if title else text)
    return "\n\n".join(parts).strip()


def _format_scored_text_block(label: str, rows: Sequence[Dict], max_tokens: int) -> str:
    lines = []
    used = 0
    if max_tokens <= 0:
        return ""
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        title = str(row.get("title", "")).strip()
        prefix = f"{index}. "
        if title:
            prefix += f"[{title}] "
        cost = _token_count(prefix) + _token_count(text)
        if lines and used + cost > max_tokens:
            break
        if cost > max_tokens:
            text = _truncate_words(text, max(1, max_tokens - _token_count(prefix)))
            cost = _token_count(prefix) + _token_count(text)
        lines.append(prefix + text)
        used += cost
    if not lines:
        return ""
    return label + ":\n" + "\n".join(lines)


def _format_evidence_groups(groups: Sequence[Dict], max_tokens: int) -> str:
    blocks = []
    used = 0
    for group in groups:
        rows = list(group.get("items", []) or [])
        if not rows:
            continue
        label = str(group.get("label", "Evidence")).strip() or "Evidence"
        question = str(group.get("question", "")).strip()
        header = label + (f" ({question})" if question else "")
        remaining = max_tokens - used
        if remaining <= 0:
            break
        block = _format_scored_text_block(header, rows, remaining)
        if not block:
            continue
        blocks.append(block)
        used += _token_count(block)
    return "\n\n".join(blocks).strip()


def _fit_blocks(blocks: Sequence[str], max_tokens: int) -> str:
    fitted: List[str] = []
    used = 0
    for block in blocks:
        block = str(block or "").strip()
        if not block:
            continue
        cost = _token_count(block)
        remaining = max_tokens - used
        if remaining <= 0:
            break
        if cost <= remaining:
            fitted.append(block)
            used += cost
        else:
            fitted.append(_truncate_words(block, remaining))
            break
    return "\n\n".join(item for item in fitted if item).strip()


def _block_records(label: str, rows: Sequence[Dict]) -> List[Dict]:
    records = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        title = str(row.get("title", "")).strip()
        prefix = f"{index}. "
        if title:
            prefix += f"[{title}] "
        records.append({"prefix": prefix, "text": text, "label": label})
    return records


def _pack_records(label: str, rows: Sequence[Dict], max_tokens: int) -> str:
    packed = []
    used = 0
    if max_tokens <= 0:
        return ""
    for record in _block_records(label, rows):
        prefix = record["prefix"]
        text = record["text"]
        cost = _token_count(prefix) + _token_count(text)
        remaining = max_tokens - used
        if remaining <= 0:
            break
        if cost > remaining:
            text = _truncate_words(text, max(1, remaining - _token_count(prefix)))
            cost = _token_count(prefix) + _token_count(text)
        if cost <= 0:
            continue
        packed.append(prefix + text)
        used += cost
    if not packed:
        return ""
    return label + ":\n" + "\n".join(packed)


def format_profile_evidence(
    evidence: Dict,
    question: str,
    max_passage_tokens: int,
    max_fact_tokens: int,
    max_total_tokens: int = 680,
) -> str:
    if not evidence:
        return ""

    profile = str(evidence.get("profile", "") or "auto")
    facts = list(evidence.get("facts", []) or [])
    sentences = list(evidence.get("sentences", []) or [])
    chunks = list(evidence.get("chunks", []) or [])
    groups = list(evidence.get("evidence_groups", []) or [])
    budget = max(256, int(max_total_tokens or 680))

    if profile == "long_context":
        fact_budget = min(max_fact_tokens, max(120, budget // 5))
        sentence_budget = max(160, budget // 4)
        chunk_budget = max(0, budget - fact_budget - sentence_budget)
        order = ("facts", "sentences", "chunks")
    elif profile == "single_hop":
        fact_budget = min(max_fact_tokens, max(140, budget // 4))
        sentence_budget = max(220, budget // 2)
        chunk_budget = max(0, budget - fact_budget - sentence_budget)
        order = ("facts", "sentences", "chunks")
    else:
        fact_budget = min(max_fact_tokens, max(160, budget // 5))
        sentence_budget = max(360, int(budget * 0.55))
        chunk_budget = max(0, budget - fact_budget - sentence_budget)
        order = ("sentences", "facts", "chunks")

    fact_block = _pack_records("Facts", facts, fact_budget)
    sentence_block = _format_evidence_groups(groups, sentence_budget)
    if not sentence_block:
        sentence_block = _pack_records("Evidence", sentences, sentence_budget)

    per_passage_budget = 0
    if chunks and chunk_budget > 0:
        per_passage_budget = max(80, min(max_passage_tokens, chunk_budget // max(1, len(chunks))))
    chunk_block = format_passages(
        chunks,
        len(chunks),
        question=question,
        max_passage_tokens=per_passage_budget,
    )

    block_by_name = {"facts": fact_block, "sentences": sentence_block, "chunks": chunk_block}
    blocks = [block_by_name[name] for name in order]

    if not any(blocks):
        fallback = list(evidence.get("fallback_passages", []) or [])
        if not fallback:
            return ""
        per_passage_budget = max(80, budget // max(1, min(len(fallback), 2)))
        return _fit_blocks(
            [format_passages(fallback, min(len(fallback), 2), question=question, max_passage_tokens=per_passage_budget)],
            budget,
        )
    return _fit_blocks(blocks, budget)


def parse_answer(raw_text: str) -> Tuple[str, str]:
    text = str(raw_text or "").strip()
    if not text:
        return "", "Unknown"
    if "Answer:" in text:
        thought, answer = text.split("Answer:", 1)
        return thought.replace("Thought:", "", 1).strip(), answer.strip().strip('"').strip("'").rstrip(".")
    lines = text.splitlines()
    return "\n".join(lines[:-1]).replace("Thought:", "", 1).strip(), lines[-1].strip().rstrip(".")


def _is_unknown_answer(answer: str) -> bool:
    normalized = " ".join(str(answer or "").strip().lower().split())
    return normalized in {"", "unknown", "not enough information", "insufficient evidence", "i don't know"}


def _question_answer_type(question: str) -> str:
    q = " ".join(str(question or "").strip().split())
    ql = q.lower()
    if _extract_or_candidates(q):
        if _extract_or_candidates(q) == ["yes", "no"]:
            return "yes_no"
        return "choice"
    if ql.startswith("where") or " place of " in ql or " birthplace " in ql:
        return "place"
    if ql.startswith("when") or "date of" in ql or "what year" in ql:
        return "date"
    if ql.startswith("who"):
        return "person"
    if "cause of death" in ql:
        return "short_fact"
    return "span"


def _extract_or_candidates(question: str) -> List[str]:
    text = " ".join(str(question or "").replace("?", " ?").split())
    lowered = text.lower()
    choice_matches = []
    if "," in text and " or " in lowered:
        tail = text.rsplit(",", 1)[-1]
        choice_matches = re.findall(r"(.+?)\s+or\s+(.+?)(?:\?|$)", tail, flags=re.IGNORECASE)
    pattern = re.compile(
        r"\b(?:between|of|from|than)\s+(.+?)\s+or\s+(.+?)(?:\?|$)",
        flags=re.IGNORECASE,
    )
    matches = choice_matches or pattern.findall(text)
    if not matches:
        matches = re.findall(r"([^,?]+?)\s+or\s+([^,?]+?)(?:\?|$)", text, flags=re.IGNORECASE)
    candidates = []
    for left, right in matches[-1:]:
        for item in (left, right):
            cleaned = item.strip(" ,.?\"'")
            cleaned = re.sub(r"^(?:do|does|did|is|are|was|were|can|could|has|have|had)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^(?:the|film|song|person)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(
                r"\s+(?:born|died|released|married|founded|established)\s+(?:first|earlier|later|before|after)$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
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


def _looks_like_date(text: str) -> bool:
    value = str(text or "").strip()
    months = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    )
    lower = value.lower()
    return bool(re.search(r"\b\d{1,2}\s+(?:" + "|".join(months) + r")\s+\d{3,4}\b", lower) or re.search(r"\b\d{3,4}\b", value))


def _answer_type_issue(question: str, answer: str, candidates: Sequence[str]) -> Optional[str]:
    answer = str(answer or "").strip()
    if not answer or _is_unknown_answer(answer):
        return None
    answer_type = _question_answer_type(question)
    lower = answer.lower()
    if candidates and candidates != ["yes", "no"] and not _candidate_from_answer(answer, candidates):
        return "choice"
    if candidates == ["yes", "no"] and lower not in {"yes", "no"}:
        return "yes_no"
    if answer_type == "date":
        if not _looks_like_date(answer):
            return "date"
        if re.fullmatch(r"\d{3,4}", answer) and re.search(r"\bdate of\b|\bwhen\b", question.lower()):
            return "full_date"
    if answer_type == "place":
        if _looks_like_date(answer):
            return "place"
        if re.search(r"\b(film|director|performer|father|mother|spouse|song)\b", lower):
            return "place"
    if answer_type == "person":
        if _looks_like_date(answer):
            return "person"
        if re.search(r"\b(father|mother|spouse|husband|wife|director|performer|producer|child)\b", lower):
            return "person"
    if answer_type == "short_fact" and len(answer.split()) > 8:
        return "short_fact"
    return None


def _type_instruction(issue: str, question: str, candidates: Sequence[str]) -> str:
    if issue == "choice":
        return "Choose exactly one of the listed answer candidates. Return the complete candidate text."
    if issue == "yes_no":
        return "Answer exactly yes or no."
    if issue in {"date", "full_date"}:
        return "Return the most specific date mentioned in the evidence, not just a year if a full date is available."
    if issue == "place":
        return "Return the place or organization name asked for, not a person, date, role, or explanation."
    if issue == "person":
        return "Return the person's name, not a relationship description, role, date, or explanation."
    if issue == "short_fact":
        return "Return only the concise cause or fact span, not a full sentence."
    return "Return the concise answer span required by the question."


class NaiveQAReader:
    def __init__(self, config: NaiveHoloRAGConfig, llm_client: LocalLLMClient) -> None:
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
            ranked_facts=ranked_facts,
            sub_questions=sub_questions,
            evidence=evidence,
            retry_mode=False,
            answer_candidates=candidates,
        )
        raw = self.llm_client.infer_messages_text(messages, fallback="Thought: Evidence is insufficient.\nAnswer: Unknown")
        thought, answer = parse_answer(raw)
        constrained_answer = _candidate_from_answer(answer, candidates)
        if constrained_answer:
            return {
                "thought": thought,
                "answer": constrained_answer,
                "raw_response": raw,
                "messages": messages,
                "retry_used": False,
                "retry_mode": "candidate_normalized",
            }
        type_issue = _answer_type_issue(question, answer, candidates)
        if type_issue:
            repaired = self._repair_typed_answer(
                issue=type_issue,
                question=question,
                ranked_passages=ranked_passages,
                ranked_facts=ranked_facts,
                sub_questions=sub_questions,
                evidence=evidence or {},
                candidates=candidates,
            )
            if repaired:
                repair_thought, repair_answer, repair_raw, repair_messages = repaired
                return {
                    "thought": repair_thought,
                    "answer": repair_answer,
                    "raw_response": repair_raw,
                    "messages": repair_messages,
                    "retry_used": True,
                    "retry_mode": f"type_repair_{type_issue}",
                }

        if self.config.qa_retry_on_unknown and _is_unknown_answer(answer):
            retry_evidence = self._retry_evidence(evidence or {}, ranked_passages, ranked_facts)
            retry_messages = self._build_messages(
                question=question,
                ranked_passages=ranked_passages,
                ranked_facts=ranked_facts,
                sub_questions=sub_questions,
                evidence=retry_evidence,
                retry_mode=True,
                answer_candidates=candidates,
            )
            retry_raw = self.llm_client.infer_messages_text(
                retry_messages,
                fallback="Thought: Evidence is insufficient.\nAnswer: Unknown",
            )
            retry_thought, retry_answer = parse_answer(retry_raw)
            constrained_retry = _candidate_from_answer(retry_answer, candidates)
            if constrained_retry:
                retry_answer = constrained_retry
            retry_issue = _answer_type_issue(question, retry_answer, candidates)
            if retry_issue:
                repaired = self._repair_typed_answer(
                    issue=retry_issue,
                    question=question,
                    ranked_passages=ranked_passages,
                    ranked_facts=ranked_facts,
                    sub_questions=sub_questions,
                    evidence=retry_evidence,
                    candidates=candidates,
                )
                if repaired:
                    repair_thought, repair_answer, repair_raw, repair_messages = repaired
                    return {
                        "thought": repair_thought,
                        "answer": repair_answer,
                        "raw_response": repair_raw,
                        "messages": repair_messages,
                        "retry_used": True,
                        "retry_mode": f"type_repair_{retry_issue}",
                    }
            if not _is_unknown_answer(retry_answer):
                return {
                    "thought": retry_thought,
                    "answer": retry_answer,
                    "raw_response": retry_raw,
                    "messages": retry_messages,
                    "retry_used": True,
                    "retry_mode": "best_effort_reader",
                }

            extract_messages = self._build_messages(
                question=question,
                ranked_passages=ranked_passages,
                ranked_facts=ranked_facts,
                sub_questions=sub_questions,
                evidence=retry_evidence,
                retry_mode=True,
                answer_candidates=candidates,
                extract_only=True,
            )
            extract_raw = self.llm_client.infer_messages_text(
                extract_messages,
                fallback="Thought: Select the best supported answer span.\nAnswer: Unknown",
            )
            extract_thought, extract_answer = parse_answer(extract_raw)
            constrained_extract = _candidate_from_answer(extract_answer, candidates)
            if constrained_extract:
                extract_answer = constrained_extract
            extract_issue = _answer_type_issue(question, extract_answer, candidates)
            if extract_issue:
                repaired = self._repair_typed_answer(
                    issue=extract_issue,
                    question=question,
                    ranked_passages=ranked_passages,
                    ranked_facts=ranked_facts,
                    sub_questions=sub_questions,
                    evidence=retry_evidence,
                    candidates=candidates,
                )
                if repaired:
                    repair_thought, repair_answer, repair_raw, repair_messages = repaired
                    return {
                        "thought": repair_thought,
                        "answer": repair_answer,
                        "raw_response": repair_raw,
                        "messages": repair_messages,
                        "retry_used": True,
                        "retry_mode": f"type_repair_{extract_issue}",
                    }
            if not _is_unknown_answer(extract_answer):
                return {
                    "thought": extract_thought,
                    "answer": extract_answer,
                    "raw_response": extract_raw,
                    "messages": extract_messages,
                    "retry_used": True,
                    "retry_mode": "extractive_forced_answer",
                }

        return {
            "thought": thought,
            "answer": answer or "Unknown",
            "raw_response": raw,
            "messages": messages,
            "retry_used": False,
            "retry_mode": "none",
        }

    def _build_messages(
        self,
        question: str,
        ranked_passages: Sequence[Dict],
        ranked_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        evidence: Dict = None,
        retry_mode: bool = False,
        answer_candidates: Sequence[str] = (),
        extract_only: bool = False,
        type_instruction: str = "",
    ) -> List[Dict]:
        passage_block = format_profile_evidence(
            evidence or {},
            question=question,
            max_passage_tokens=self.config.qa_max_passage_tokens,
            max_fact_tokens=self.config.qa_max_fact_tokens,
            max_total_tokens=self._evidence_budget(),
        )
        if not passage_block:
            per_passage_budget = max(80, self._evidence_budget() // max(1, min(self.config.qa_passage_top_k, 2)))
            passage_block = format_passages(
                ranked_passages,
                min(self.config.qa_passage_top_k, 2),
                question=question,
                max_passage_tokens=per_passage_budget,
            )
        fact_lines = []
        fact_tokens = 0
        if not evidence:
            for fact in ranked_facts[: self.config.fact_top_k]:
                text = str(fact.get("text", "")).strip()
                if not text:
                    continue
                cost = _token_count(text) + 1
                if fact_lines and fact_tokens + cost > self.config.qa_max_fact_tokens:
                    break
                fact_lines.append(f"- {text}")
                fact_tokens += cost
        fact_block = "\n".join(fact_lines)
        user_parts = []
        if fact_block:
            user_parts.append("Retrieved facts:\n" + fact_block)
        if passage_block:
            user_parts.append(passage_block)
        candidate_block = ""
        if answer_candidates:
            candidate_block = "Answer candidates: " + " | ".join(str(item) for item in answer_candidates)
        question_block = f"Question: {question}"
        if candidate_block:
            question_block += "\n" + candidate_block
        question_block += "\nThought: "
        user_parts.append(question_block)
        user_content = self._fit_user_content(user_parts)
        system_prompt = (
            "You are a reading comprehension assistant. Use only the provided evidence, reason briefly "
            "after 'Thought:', and finish with 'Answer:' followed by one concise span. Give the best "
            "supported answer; use Unknown only if the evidence contains no usable candidate."
        )
        if retry_mode:
            system_prompt = (
                "Use only the evidence. Give the best supported short answer span. Do not answer Unknown. "
                "If evidence is partial, choose the most plausible span that is directly mentioned. "
                "Finish with 'Answer:' followed by the span."
            )
        if extract_only:
            system_prompt = (
                "Extract the final answer from the evidence. Return a concise answer span after 'Answer:'. "
                "Do not answer Unknown, do not explain uncertainty, and do not add extra text after the answer. "
                "If evidence is incomplete, choose the best directly mentioned candidate span."
            )
        if type_instruction:
            system_prompt += " " + type_instruction
        if answer_candidates:
            system_prompt += " If answer candidates are listed, the final answer must be exactly one listed candidate unless none is supported."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _repair_typed_answer(
        self,
        issue: str,
        question: str,
        ranked_passages: Sequence[Dict],
        ranked_facts: Sequence[Dict],
        sub_questions: Sequence[str],
        evidence: Dict,
        candidates: Sequence[str],
    ) -> Optional[Tuple[str, str, str, List[Dict]]]:
        repair_evidence = evidence or self._retry_evidence({}, ranked_passages, ranked_facts)
        messages = self._build_messages(
            question=question,
            ranked_passages=ranked_passages,
            ranked_facts=ranked_facts,
            sub_questions=sub_questions,
            evidence=repair_evidence,
            retry_mode=True,
            answer_candidates=candidates,
            extract_only=True,
            type_instruction=_type_instruction(issue, question, candidates),
        )
        raw = self.llm_client.infer_messages_text(
            messages,
            fallback="Thought: Extract the answer span required by the question.\nAnswer: Unknown",
        )
        thought, answer = parse_answer(raw)
        constrained = _candidate_from_answer(answer, candidates)
        if constrained:
            answer = constrained
        if _is_unknown_answer(answer):
            return None
        if _answer_type_issue(question, answer, candidates) == issue:
            return None
        return thought, answer, raw, messages

    def _retry_evidence(self, evidence: Dict, ranked_passages: Sequence[Dict], ranked_facts: Sequence[Dict]) -> Dict:
        sentences = list(evidence.get("sentences", []) or [])[:4]
        chunks = list(ranked_passages[: min(3, self.config.qa_passage_top_k)])
        facts = list(ranked_facts[: min(self.config.fact_top_k, 6)])
        groups = []
        if sentences:
            groups.append({"label": "High-signal sentences", "items": sentences})
        return {
            "profile": "long_context",
            "facts": facts,
            "sentences": sentences,
            "chunks": chunks,
            "evidence_groups": groups,
            "fallback_passages": chunks,
        }

    def _evidence_budget(self) -> int:
        return max(256, int(getattr(self.config, "qa_evidence_token_budget", 620)))

    def _fit_user_content(self, user_parts: Sequence[str]) -> str:
        content = "\n\n".join(part for part in user_parts if part)
        budget = max(256, min(int(self.config.qa_max_input_tokens), self._evidence_budget()))
        if _token_count(content) <= budget:
            return content
        prefix_parts = list(user_parts[:-1])
        question_part = user_parts[-1] if user_parts else ""
        remaining = budget - _token_count(question_part)
        fitted = []
        for part in prefix_parts:
            if remaining <= 0:
                break
            tokens = _token_count(part)
            if tokens <= remaining:
                fitted.append(part)
                remaining -= tokens
            else:
                fitted.append(_truncate_words(part, remaining))
                remaining = 0
        fitted.append(question_part)
        return "\n\n".join(part for part in fitted if part)
