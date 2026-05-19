from typing import Dict

from .config import NaiveHoloRAGConfig
from .llm_client import LocalLLMClient
from .utils import entropy_confidence, normalize_alpha


class IntentRouter:
    def __init__(self, config: NaiveHoloRAGConfig, llm_client: LocalLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self._cache: Dict[str, Dict[str, float]] = {}

    def route(self, query: str, forced_profile: str = "auto") -> Dict:
        profile = self._resolve_profile(query, forced_profile)
        if profile != "auto":
            alpha = normalize_alpha(self.config.profile_alpha_priors[profile])
            return {"profile": profile, "alpha": alpha, "confidence": 1.0}
        predicted = self.predict_alpha(query)
        return {
            "profile": self._profile_from_alpha(predicted),
            "alpha": predicted,
            "confidence": entropy_confidence(predicted),
        }

    def predict_alpha(self, query: str) -> Dict[str, float]:
        cache_key = " ".join(str(query or "").split())
        if cache_key in self._cache:
            return dict(self._cache[cache_key])
        fallback = self._heuristic_alpha(query)
        if (not self.config.enable_intent_routing) or (not self.config.intent_use_llm):
            alpha = normalize_alpha(fallback)
            self._cache[cache_key] = alpha
            return alpha
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Predict retrieval granularity weights for the question. "
                "Return JSON with numeric keys entity, fact, sentence, chunk that sum to 1."
            ),
            user_prompt=f"Question:\n{query}",
            fallback=fallback,
            max_tokens=96,
        )
        alpha = normalize_alpha({
            "entity": payload.get("entity", fallback["entity"]),
            "fact": payload.get("fact", fallback["fact"]),
            "sentence": payload.get("sentence", fallback["sentence"]),
            "chunk": payload.get("chunk", fallback["chunk"]),
        })
        self._cache[cache_key] = alpha
        return alpha

    def _resolve_profile(self, query: str, forced_profile: str) -> str:
        if forced_profile in {"single_hop", "multi_hop", "long_context"}:
            return forced_profile
        return "auto"

    def _profile_from_alpha(self, alpha: Dict[str, float]) -> str:
        if alpha.get("chunk", 0.0) >= 0.42:
            return "long_context"
        if alpha.get("sentence", 0.0) + alpha.get("fact", 0.0) >= 0.62:
            return "multi_hop"
        return "single_hop"

    def _heuristic_alpha(self, query: str) -> Dict[str, float]:
        lowered = str(query or "").lower()
        if any(token in lowered for token in ["summary", "overview", "background", "context", "describe"]):
            return self.config.profile_alpha_priors["long_context"]
        if any(token in lowered for token in ["employer", "spouse", "father", "mother", "after", "before", "whose", "which"]):
            return self.config.profile_alpha_priors["multi_hop"]
        if any(token in lowered for token in ["who", "when", "where", "name"]):
            return self.config.profile_alpha_priors["single_hop"]
        return {"entity": 0.25, "fact": 0.25, "sentence": 0.30, "chunk": 0.20}
