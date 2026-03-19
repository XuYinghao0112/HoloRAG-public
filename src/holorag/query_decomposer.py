import re
from typing import List, Optional, Sequence

from .llm_client import LocalLLMClient


class QueryDecomposer:
    def __init__(self, llm_client: LocalLLMClient) -> None:
        self.llm_client = llm_client

    def decompose(self, query: str, resolved_entities: Optional[Sequence[dict]] = None) -> List[str]:
        fallback = {"sub_questions": self._heuristic_decompose(query)}
        entity_context = ""
        if resolved_entities:
            lines = []
            for item in resolved_entities:
                mention = str(item.get("mention", "")).strip()
                resolved_text = str(item.get("resolved_text", "")).strip()
                if mention and resolved_text:
                    lines.append(f"- {mention} -> {resolved_text}")
            if lines:
                entity_context = "\n\nResolved entity context:\n" + "\n".join(lines)
        payload, _ = self.llm_client.infer_json(
            system_prompt=(
                "Break the query into 1 to 4 atomic retrieval sub-questions. "
                "Preserve multi-hop dependency order. "
                "For nested or relative-clause questions, first resolve the inner referent, "
                "then identify the linked entity, then ask for the final target attribute. "
                "Use the resolved entity context to avoid mixing different entities that share surface forms. "
                "Return JSON with key sub_questions."
            ),
            user_prompt=f"Query:\n{query}{entity_context}",
            fallback=fallback,
            max_tokens=160,
        )
        sub_questions = payload.get("sub_questions", [])
        cleaned = [str(question).strip() for question in sub_questions if str(question).strip()]
        return cleaned or fallback["sub_questions"]

    def _heuristic_decompose(self, query: str) -> List[str]:
        normalized = " ".join(query.strip().split())
        if not normalized:
            return []

        if " which " in normalized.lower() and "," in normalized:
            prefix, suffix = normalized.rsplit(",", 1)
            prefix = prefix.strip(" ?")
            suffix = suffix.strip(" ?")
            if prefix and suffix:
                return [prefix + "?", suffix + "?"]

        connectors = re.split(r"\s+(?:and then|then|after|before|while|versus|vs\.?)\s+", normalized, flags=re.IGNORECASE)
        cleaned = [part.strip(" ?") + "?" for part in connectors if len(part.strip()) > 5]
        if len(cleaned) > 1:
            return cleaned[:4]

        relative_clause = re.search(r"\b(the|a|an)\s+(.+?)\s+(who|which|that)\s+(.+)", normalized, flags=re.IGNORECASE)
        if relative_clause:
            head = relative_clause.group(2).strip()
            clause = relative_clause.group(4).strip(" ?")
            return [f"{clause}?", f"What is the {head}?"]

        return [normalized.rstrip("?") + "?"]
