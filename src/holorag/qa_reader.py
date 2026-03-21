from typing import Dict, List, Sequence, Tuple


HIPPORAG_QA_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze the provided passages and question "
    "carefully. Start your response after 'Thought: ' with a concise reasoning process grounded in the passages. "
    "Then conclude with 'Answer: ' followed by a short, definitive answer. If the answer cannot be found in the "
    "passages, output 'Answer: Unknown'."
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


def build_hipporag_qa_messages(question: str, ranked_passages: Sequence[Dict], top_k: int) -> List[Dict[str, str]]:
    passage_block = format_passages_as_wikipedia_docs(ranked_passages, top_k)
    user_prompt = f"{passage_block}\n\nQuestion: {question}\nThought: ".strip()
    return [
        {"role": "system", "content": HIPPORAG_QA_SYSTEM_PROMPT},
        {"role": "user", "content": HIPPORAG_QA_EXAMPLE_USER},
        {"role": "assistant", "content": HIPPORAG_QA_EXAMPLE_ASSISTANT},
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

    answer = answer.strip().strip(".")
    if not answer:
        answer = "Unknown"
    return thought, answer
