from typing import Dict, List, Optional

import networkx as nx


class EvidenceExtractor:
    def extract(
        self,
        graph: nx.DiGraph,
        ranked_nodes: List[Dict],
        alpha: Optional[Dict[str, float]] = None,
        ranked_passages: Optional[List[Dict]] = None,
        total_budget: int = 12,
    ) -> Dict:
        alpha = alpha or {"entity": 1 / 3, "sentence": 1 / 3, "chunk": 1 / 3}
        entity_budget = max(1, int(round(total_budget * alpha.get("entity", 0.0))))
        sentence_budget = max(1, int(round(total_budget * alpha.get("sentence", 0.0))))
        chunk_budget = max(1, int(round(total_budget * alpha.get("chunk", 0.0))))
        entity_evidence = []
        sentence_evidence = []
        chunk_evidence = []
        for item in ranked_nodes:
            node_id = item["node_id"]
            if node_id not in graph:
                continue
            attrs = graph.nodes[node_id]
            record = {
                "node_id": node_id,
                "score": item["score"],
                "text": attrs.get("text", ""),
                "metadata": attrs.get("metadata", {}),
            }
            node_type = attrs.get("node_type")
            if node_type == "entity" and len(entity_evidence) < entity_budget:
                entity_evidence.append(record)
            elif node_type == "sentence" and len(sentence_evidence) < sentence_budget:
                sentence_evidence.append(record)
            elif node_type == "chunk" and len(chunk_evidence) < chunk_budget:
                chunk_evidence.append(record)
        context_parts = []
        if ranked_passages:
            context_parts.extend(str(item.get("text", "")) for item in ranked_passages[:chunk_budget])
        else:
            context_parts.extend(record["text"] for record in chunk_evidence)
        context_parts.extend(record["text"] for record in sentence_evidence[: max(1, sentence_budget // 2)])
        return {
            "entity": entity_evidence,
            "sentence": sentence_evidence,
            "chunk": chunk_evidence,
            "qa_context": "\n\n".join(part for part in context_parts if part).strip(),
        }
