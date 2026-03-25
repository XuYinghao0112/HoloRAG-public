from typing import Dict, List, Sequence, Tuple


HIPPORAG_QA_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. "
    "Your response must start after 'Thought: ' with concise reasoning grounded in the passages. "
    "Use the intermediate reasoning hints only as clues, and keep the final answer anchored to the entities established by those hints. "
    "When the evidence is sufficient, synthesize the needed relation or attribute instead of repeating partial context. "
    "Do not switch to a competing, contrastive, or merely nearby entity unless the passages explicitly connect it to the question target. "
    "Conclude with 'Answer: ' followed by one short, definitive answer span. "
    "Do not list multiple alternatives. Use 'Unknown' only when the passages truly do not support a single answer."
)

HIPPORAG_HOP_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, answer the current sub-question using the provided evidence. "
    "Start after 'Thought: ' with brief grounded reasoning. Then write 'Answer: ' followed by one short atomic answer span. "
    "Return a single entity, date, place, organization, number, or attribute phrase rather than a sentence or a list. "
    "If the evidence contains multiple candidates and the current sub-question does not identify one uniquely, answer 'Unknown'. "
    "Use 'Unknown' only when the evidence does not support a single answer."
)

HIPPORAG_NORMALIZE_SYSTEM_PROMPT = (
    "You are given a question, evidence passages, and a candidate answer. "
    "Rewrite the candidate as one short atomic answer span that is directly supported by the evidence and fits the question. "
    "Do not output a sentence, explanation, or multiple alternatives. "
    "If the candidate cannot be reduced to one supported answer span, output 'Unknown'."
)

HIPPORAG_FOCUS_SYSTEM_PROMPT = (
    "You are given a question, a draft answer, intermediate reasoning hints, and evidence passages. "
    "Concentrate only on the answer-bearing evidence that most directly resolves the question. "
    "If the draft answer is unsupported or incomplete, correct it using the passages. "
    "Keep the answer anchored to the entities established by the intermediate reasoning hints, and ignore nearby alternatives or competitors unless the passages explicitly connect them to the target. "
    "Start after 'Thought: ' with concise grounded reasoning focused on the minimal decisive clues. "
    "Conclude with 'Answer: ' followed by one short definitive answer span. "
    "Use 'Unknown' only when the passages do not support a single answer."
)

HIPPORAG_QA_EXAMPLE_USER = (
    "Wikipedia Title: The Last Horse\n"
    "The Last Horse (Spanish:El ultimo caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernan Gomez.\n\n"
    "Wikipedia Title: Southampton\n"
    "The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, "
    "has over 22,000 students. The university is ranked in the top 100 research universities in the world.\n\n"
    "Wikipedia Title: Neville A. Stanton\n"
    "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.\n\n"
    "Question: When was Neville A. Stanton's employer founded?\n"
    "Thought: "
)

HIPPORAG_QA_EXAMPLE_ASSISTANT = (
    "Neville A. Stanton's employer is the University of Southampton. The University of Southampton was founded in 1862.\n"
    "Answer: 1862"
)

HIPPORAG_NORMALIZE_EXAMPLE_USER = (
    "Wikipedia Title: Charles Smith Olden\n"
    "Charles Smith Olden (February 19, 1799April 7, 1876) was an American Republican Party politician, who served as the 19th Governor of New Jersey from 1860 to 1863 during the first part of the American Civil War.\n\n"
    "Question: Which state was this governor from?\n"
    "Candidate Answer: Charles Smith Olden was the governor of New Jersey.\n"
    "Thought: "
)

HIPPORAG_NORMALIZE_EXAMPLE_ASSISTANT = (
    "The candidate answer refers to the state associated with the governor. The supported short answer is New Jersey.\n"
    "Answer: New Jersey"
)


def format_passages_as_wikipedia_docs(ranked_passages: Sequence[Dict], top_k: int) -> str:
    parts: List[str] = []
    for passage in list(ranked_passages)[:top_k]:
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        if not text:
            continue
        if title:
            parts.append(f"Wikipedia Title: {title}\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def format_reasoning_chain(reasoning_chain: Sequence[Dict]) -> str:
    lines: List[str] = []
    for index, hop in enumerate(reasoning_chain, start=1):
        sub_question = str(hop.get("sub_question", "")).strip()
        hop_answer = str(hop.get("hop_answer", "")).strip()
        source_title = str(hop.get("source_title", "")).strip()
        if not sub_question:
            continue
        line = f"{index}. {sub_question}"
        if hop_answer and hop_answer.lower() != "unknown":
            line += f" => {hop_answer}"
        if source_title:
            line += f" (source: {source_title})"
        lines.append(line)
    return "\n".join(lines)


def build_hipporag_qa_messages(
    question: str,
    ranked_passages: Sequence[Dict],
    top_k: int,
    reasoning_chain: Sequence[Dict] | None = None,
    fact_hints: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    passage_block = format_passages_as_wikipedia_docs(ranked_passages, top_k)
    reasoning_block = format_reasoning_chain(reasoning_chain or [])
    user_parts: List[str] = []
    if reasoning_block:
        user_parts.append("Intermediate reasoning hints:\n" + reasoning_block)
    if fact_hints:
        fact_block = "\n".join(f"- {hint}" for hint in fact_hints if str(hint).strip())
        if fact_block:
            user_parts.append("Fact hints:\n" + fact_block)
    if passage_block:
        user_parts.append(passage_block)
    user_parts.append(f"Question: {question}\nThought: ")
    user_prompt = "\n\n".join(part.strip() for part in user_parts if part.strip())
    return [
        {"role": "system", "content": HIPPORAG_QA_SYSTEM_PROMPT},
        {"role": "user", "content": HIPPORAG_QA_EXAMPLE_USER},
        {"role": "assistant", "content": HIPPORAG_QA_EXAMPLE_ASSISTANT},
        {"role": "user", "content": user_prompt},
    ]


def build_hipporag_hop_messages(
    sub_question: str,
    evidence_passages: Sequence[Dict],
    previous_hops: Sequence[Dict] | None = None,
    top_k: int = 3,
) -> List[Dict[str, str]]:
    evidence_block = format_passages_as_wikipedia_docs(evidence_passages, top_k)
    reasoning_block = format_reasoning_chain(previous_hops or [])
    user_parts: List[str] = []
    if reasoning_block:
        user_parts.append("Previous reasoning steps:\n" + reasoning_block)
    if evidence_block:
        user_parts.append(evidence_block)
    user_parts.append(f"Question: {sub_question}\nThought: ")
    return [
        {"role": "system", "content": HIPPORAG_HOP_SYSTEM_PROMPT},
        {"role": "user", "content": HIPPORAG_QA_EXAMPLE_USER},
        {"role": "assistant", "content": HIPPORAG_QA_EXAMPLE_ASSISTANT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_answer_focus_messages(
    question: str,
    candidate_answer: str,
    ranked_passages: Sequence[Dict],
    top_k: int,
    reasoning_chain: Sequence[Dict] | None = None,
    fact_hints: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    passage_block = format_passages_as_wikipedia_docs(ranked_passages, top_k)
    reasoning_block = format_reasoning_chain(reasoning_chain or [])
    user_parts: List[str] = []
    if reasoning_block:
        user_parts.append("Intermediate reasoning hints:\n" + reasoning_block)
    if fact_hints:
        fact_block = "\n".join(f"- {hint}" for hint in fact_hints if str(hint).strip())
        if fact_block:
            user_parts.append("Fact hints:\n" + fact_block)
    if passage_block:
        user_parts.append(passage_block)
    user_parts.append(f"Question: {question}\nDraft Answer: {candidate_answer}\nThought: ")
    return [
        {"role": "system", "content": HIPPORAG_FOCUS_SYSTEM_PROMPT},
        {"role": "user", "content": HIPPORAG_QA_EXAMPLE_USER},
        {"role": "assistant", "content": HIPPORAG_QA_EXAMPLE_ASSISTANT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_short_answer_normalization_messages(
    question: str,
    candidate_answer: str,
    evidence_passages: Sequence[Dict],
    top_k: int,
) -> List[Dict[str, str]]:
    evidence_block = format_passages_as_wikipedia_docs(evidence_passages, top_k)
    user_prompt = "\n\n".join(
        part
        for part in [
            evidence_block,
            f"Question: {question}\nCandidate Answer: {candidate_answer}\nThought: ",
        ]
        if part
    )
    return [
        {"role": "system", "content": HIPPORAG_NORMALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": HIPPORAG_NORMALIZE_EXAMPLE_USER},
        {"role": "assistant", "content": HIPPORAG_NORMALIZE_EXAMPLE_ASSISTANT},
        {"role": "user", "content": user_prompt},
    ]


def parse_hipporag_qa_response(raw_text: str) -> Tuple[str, str]:
    text = str(raw_text or "").strip()
    if not text:
        return "", "Unknown"

    answer = text
    thought = ""
    if "Answer:" in text:
        prefix, suffix = text.split("Answer:", 1)
        answer = suffix.strip()
        thought = prefix.replace("Thought:", "", 1).strip()
    elif "\n" in text:
        answer = text.splitlines()[-1].strip()
        thought = "\n".join(text.splitlines()[:-1]).replace("Thought:", "", 1).strip()

    answer = answer.strip().strip().strip('"').strip("'").strip()
    answer = answer.rstrip(".。")
    if not answer:
        answer = "Unknown"
    return thought, answer
