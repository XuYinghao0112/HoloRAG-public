import math
from typing import Dict

import networkx as nx

from .config import HoloRAGConfig


class GranularityPageRank:
    def __init__(self, config: HoloRAGConfig) -> None:
        self.config = config

    def run(self, graph: nx.DiGraph, alpha: Dict[str, float], seed_scores: Dict[str, float]) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}
        working = nx.DiGraph()
        for node_id, attrs in graph.nodes(data=True):
            working.add_node(node_id, **attrs)
        for source, target, attrs in graph.edges(data=True):
            edge_kinds = attrs.get("edge_kinds", [attrs.get("edge_type", "default")])
            edge_factor = max(self.config.edge_type_weights.get(kind, 1.0) for kind in edge_kinds)
            target_type = graph.nodes[target].get("node_type", "chunk")
            if self.config.enable_granularity_awareness and self.config.enable_granularity_pagerank_bias:
                target_bias = 1.0 + self.config.transition_lambda * alpha.get(target_type, 0.0)
            else:
                target_bias = 1.0
            hub_scale = 1.0 / (1.0 + self.config.hub_penalty * math.log1p(graph.degree(target)))
            weight = float(attrs.get("weight", 1.0)) * edge_factor * target_bias * hub_scale
            self._keep_max_edge(working, source, target, weight)
            self._keep_max_edge(working, target, source, weight)

        personalization = {node_id: self.config.seed_floor for node_id in working.nodes()}
        for node_id, score in seed_scores.items():
            if node_id in personalization:
                personalization[node_id] += max(float(score), 0.0)
        total = sum(personalization.values()) or 1.0
        personalization = {node_id: score / total for node_id, score in personalization.items()}
        return nx.pagerank(
            working,
            alpha=self.config.pagerank_alpha,
            personalization=personalization,
            weight="weight",
            max_iter=100,
        )

    def _keep_max_edge(self, graph: nx.DiGraph, source: str, target: str, weight: float) -> None:
        old = graph.get_edge_data(source, target, {}).get("weight", 0.0)
        if weight > old:
            graph.add_edge(source, target, weight=weight)
