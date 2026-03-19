from typing import Dict

from .config import HoloRAGConfig
from .llm_client import LocalLLMClient
from .utils import lexical_overlap_score


class RecognitionFilter:
    def __init__(self, config: HoloRAGConfig, llm_client: LocalLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def judge(self, query: str, node_text: str, base_score: float) -> float:
        heuristic = lexical_overlap_score(query, node_text)
        if not self.config.enable_recognition_filter:
            return 1.0
        if not self.config.enable_llm_judge:
            return heuristic
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Judge whether the evidence is relevant to the query. "
                "Return JSON with numeric key judge_score in [0, 1]."
            ),
            user_prompt=f"Query:\n{query}\n\nEvidence:\n{node_text}\n\nBase score: {base_score:.4f}",
            fallback={"judge_score": heuristic},
            max_tokens=96,
        )
        return max(0.0, min(1.0, float(payload.get("judge_score", heuristic))))

    def rerank(self, query: str, candidates: Dict[str, float], node_texts: Dict[str, str]) -> Dict[str, float]:
        return {
            node_id: score * self.judge(query, node_texts[node_id], score)
            for node_id, score in candidates.items()
        }
