from typing import Dict

from .llm_client import LocalLLMClient


class IntentParser:
    def __init__(self, llm_client: LocalLLMClient) -> None:
        self.llm_client = llm_client
        self._cache: Dict[str, Dict[str, float]] = {}

    def predict(self, query: str) -> Dict[str, float]:
        cache_key = (query or "").strip()
        if cache_key in self._cache:
            return dict(self._cache[cache_key])

        fallback = self._heuristic_alpha(query)
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Predict a granularity preference vector for the query. "
                "Return JSON with numeric keys alpha_E, alpha_S, alpha_C that sum to 1."
            ),
            user_prompt=f"Query:\n{query}",
            fallback=fallback,
            max_tokens=96,
        )
        alpha = {
            "entity": float(payload.get("alpha_E", fallback["alpha_E"])),
            "sentence": float(payload.get("alpha_S", fallback["alpha_S"])),
            "chunk": float(payload.get("alpha_C", fallback["alpha_C"])),
        }
        total = sum(max(0.0, value) for value in alpha.values()) or 1.0
        normalized = {key: max(0.0, value) / total for key, value in alpha.items()}
        self._cache[cache_key] = dict(normalized)
        return normalized

    def _heuristic_alpha(self, query: str) -> Dict[str, float]:
        lowered = query.lower()
        if any(token in lowered for token in ["summary", "overview", "background", "context", "describe"]):
            return {"alpha_E": 0.15, "alpha_S": 0.25, "alpha_C": 0.60}
        if any(token in lowered for token in ["why", "how", "compare", "reason", "evidence"]):
            return {"alpha_E": 0.20, "alpha_S": 0.55, "alpha_C": 0.25}
        if any(token in lowered for token in ["who", "when", "where", "which", "name", "entity"]):
            return {"alpha_E": 0.55, "alpha_S": 0.30, "alpha_C": 0.15}
        return {"alpha_E": 0.30, "alpha_S": 0.45, "alpha_C": 0.25}
