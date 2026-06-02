from typing import Dict

from .config import HoloRAGConfig
from .llm_client import LocalLLMClient
from .utils import entropy_confidence, normalize_alpha


class IntentRouter:
    def __init__(self, config: HoloRAGConfig, llm_client: LocalLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self._cache: Dict[str, Dict[str, float]] = {}

    def route(self, query: str, forced_profile: str = "auto") -> Dict:
        profile = self._resolve_profile(query, forced_profile)
        if not self.config.enable_granularity_awareness:
            alpha = normalize_alpha({"fact": 1.0, "sentence": 1.0, "chunk": 1.0})
            if profile == "auto":
                profile = self._profile_from_alpha(self._heuristic_alpha(query))
            return {"profile": profile, "alpha": alpha, "confidence": 1.0}
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
            system_prompt=self._granularity_prompt(),
            user_prompt=f"Question:\n{query}",
            fallback=fallback,
            max_tokens=96,
        )
        alpha = normalize_alpha({
            "fact": payload.get("alpha_F", payload.get("fact", fallback["fact"])),
            "sentence": payload.get("alpha_S", payload.get("sentence", fallback["sentence"])),
            "chunk": payload.get("alpha_C", payload.get("chunk", fallback["chunk"])),
        })
        self._cache[cache_key] = alpha
        return alpha

    def _granularity_prompt(self) -> str:
        return (
            "Predict retrieval granularity weights from the question text only.\n"
            "Return only JSON with numeric keys alpha_F, alpha_S, alpha_C. Values must be non-negative and sum to 1.\n"
            "Do not infer or use any dataset name, source, benchmark, domain label, or metadata.\n"
            "alpha_F favors compact factual evidence: entities, relations, attributes, constraints, dates, and counts. "
            "alpha_S favors localized textual evidence: one or a few sentences that connect the question terms and justify the answer. "
            "alpha_C favors broader passage context: questions whose answer may require surrounding context, multiple mentions, narrative/background information, or avoiding ambiguity among related entities.\n"
            "Assign higher alpha_F when the question is likely answerable by a precise fact. "
            "Assign higher alpha_S when the question requires connecting or comparing pieces of evidence in nearby text. "
            "Assign higher alpha_C when the question needs broader context than isolated facts or sentences. "
            "Keep the distribution smooth and calibrated; avoid extreme weights unless the question strongly indicates one granularity. "
            "For mixed or uncertain cases, prefer a balanced distribution rather than over-committing to one source of evidence."
        )

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
        return {"fact": 0.40, "sentence": 0.40, "chunk": 0.20}
