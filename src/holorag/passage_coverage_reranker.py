from typing import Dict, List, Sequence

from .utils import lexical_overlap_score


class PassageCoverageReranker:
    """Coverage-aware passage reranker for keeping multi-hop evidence in top results."""

    def __init__(
        self,
        top_window: int = 12,
        anchor_keep: int = 2,
        overlap_threshold: float = 0.12,
    ) -> None:
        self.top_window = max(6, int(top_window))
        self.anchor_keep = max(1, int(anchor_keep))
        self.overlap_threshold = max(0.0, float(overlap_threshold))

    def _overlap(self, query_text: str, passage: Dict) -> float:
        query_text = str(query_text or "").strip()
        if not query_text:
            return 0.0
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        if not title and not text:
            return 0.0
        if text and len(text) > 1000:
            text = text[:1000]
        title_overlap = lexical_overlap_score(query_text, title) if title else 0.0
        text_overlap = lexical_overlap_score(query_text, text) if text else 0.0
        return max(title_overlap, text_overlap)

    def rerank(
        self,
        query: str,
        sub_questions: Sequence[str],
        ranked_passages: Sequence[Dict],
        output_top_k: int,
    ) -> List[Dict]:
        ranked_list = list(ranked_passages)
        if len(ranked_list) <= 2:
            return ranked_list

        ordered_queries: List[str] = []
        for item in [query] + list(sub_questions):
            text = str(item or "").strip()
            if text and text not in ordered_queries:
                ordered_queries.append(text)
        if not ordered_queries:
            return ranked_list

        top_window = min(len(ranked_list), max(output_top_k, self.top_window))
        head = list(ranked_list[:top_window])
        tail = list(ranked_list[top_window:])

        primary_k = min(5, len(head))
        topk = list(head[:primary_k])
        rest = list(head[primary_k:])

        def _key(passage: Dict) -> tuple:
            return (
                passage.get("passage_index"),
                str(passage.get("title", "")).strip(),
                str(passage.get("text", "")).strip(),
            )

        # Step 1: conservative de-dup in top-k only.
        used_keys = set()
        deduped_topk: List[Dict] = []
        for passage in topk:
            key = _key(passage)
            if key in used_keys:
                continue
            used_keys.add(key)
            deduped_topk.append(passage)
        if len(deduped_topk) < primary_k:
            for passage in rest:
                key = _key(passage)
                if key in used_keys:
                    continue
                deduped_topk.append(passage)
                used_keys.add(key)
                if len(deduped_topk) >= primary_k:
                    break
        topk = deduped_topk

        # Step 2: only replace when sub-question coverage is missing and candidate is close in score.
        margin = 0.015
        min_gain = 0.06
        for subq in ordered_queries[1:]:
            if len(topk) < primary_k:
                break
            current_best = max((self._overlap(subq, passage) for passage in topk), default=0.0)
            if current_best >= self.overlap_threshold:
                continue

            candidate_idx = None
            candidate_overlap = 0.0
            for idx, passage in enumerate(rest):
                overlap = self._overlap(subq, passage)
                if overlap > candidate_overlap:
                    candidate_overlap = overlap
                    candidate_idx = idx
            if candidate_idx is None or candidate_overlap < self.overlap_threshold:
                continue

            candidate = rest[candidate_idx]
            candidate_score = float(candidate.get("score", 0.0) or 0.0)
            replace_idx = None
            replace_value = float("inf")
            anchor_threshold = sorted((float(p.get("score", 0.0) or 0.0) for p in topk), reverse=True)[: self.anchor_keep]
            anchor_min = min(anchor_threshold) if anchor_threshold else float("inf")
            for idx, passage in enumerate(topk):
                base = float(passage.get("score", 0.0) or 0.0)
                if base >= anchor_min:
                    continue
                value = self._overlap(subq, passage)
                if value < replace_value:
                    replace_value = value
                    replace_idx = idx
            if replace_idx is None:
                continue
            replace_score = float(topk[replace_idx].get("score", 0.0) or 0.0)
            if replace_score - candidate_score > margin:
                continue
            if candidate_overlap - replace_value < min_gain:
                continue

            replaced = topk[replace_idx]
            topk[replace_idx] = candidate
            rest[candidate_idx] = replaced

        # Rebuild head preserving new top-k then original order for remaining unique items.
        selected_keys = {_key(passage) for passage in topk}
        new_head = list(topk)
        for passage in head:
            key = _key(passage)
            if key in selected_keys:
                continue
            new_head.append(passage)
            selected_keys.add(key)
        return new_head + tail
