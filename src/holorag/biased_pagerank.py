import math
from typing import Dict, Optional

import networkx as nx

from .config import HoloRAGConfig


class GranularityBiasedPageRank:
    def __init__(self, config: HoloRAGConfig) -> None:
        self.config = config

    def run(
        self,
        graph: nx.DiGraph,
        alpha: Dict[str, float],
        seed_scores: Dict[str, float],
        prior_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}
        working_graph = nx.DiGraph()
        for node_id, attrs in graph.nodes(data=True):
            working_graph.add_node(node_id, **attrs)
        for source, target, attrs in graph.edges(data=True):
            edge_kinds = attrs.get("edge_kinds", [attrs.get("edge_type", "default")])
            edge_factor = max(self.config.edge_type_weights.get(kind, 1.0) for kind in edge_kinds)
            target_type = graph.nodes[target].get("node_type", "chunk")
            target_bias = (
                1.0 + self.config.transition_lambda * alpha.get(target_type, 0.0)
            ) if self.config.enable_granularity_biased_transition else 1.0
            hub_scale = 1.0 / (1.0 + self.config.hub_penalty * math.log1p(graph.degree(target)))
            weighted_score = float(attrs.get("weight", 1.0) * edge_factor * target_bias * hub_scale)
            working_graph.add_edge(
                source,
                target,
                weight=weighted_score,
            )
            # HoloRAG runs PPR on an undirected graph. We preserve the granularity-biased
            # edge weighting but symmetrize the propagation graph so relevance can travel
            # across alias/relation chains even when the original graph is directional.
            reverse_attrs = graph.get_edge_data(target, source)
            if reverse_attrs is None:
                reverse_kinds = edge_kinds
                reverse_factor = edge_factor
            else:
                reverse_kinds = reverse_attrs.get("edge_kinds", [reverse_attrs.get("edge_type", "default")])
                reverse_factor = max(self.config.edge_type_weights.get(kind, 1.0) for kind in reverse_kinds)
            source_type = graph.nodes[source].get("node_type", "chunk")
            reverse_bias = (
                1.0 + self.config.transition_lambda * alpha.get(source_type, 0.0)
            ) if self.config.enable_granularity_biased_transition else 1.0
            reverse_hub_scale = 1.0 / (1.0 + self.config.hub_penalty * math.log1p(graph.degree(source)))
            reverse_weight = float(attrs.get("weight", 1.0) * reverse_factor * reverse_bias * reverse_hub_scale)
            existing_reverse = working_graph.get_edge_data(target, source, {}).get("weight", 0.0)
            if reverse_weight > existing_reverse:
                working_graph.add_edge(target, source, weight=reverse_weight)
        personalization = {node_id: 0.0 for node_id in working_graph.nodes()}
        total_seed = sum(max(score, 0.0) for score in seed_scores.values()) or 1.0
        for node_id, score in seed_scores.items():
            if node_id in personalization:
                personalization[node_id] = score / total_seed
        if prior_scores:
            total_prior = sum(max(score, 0.0) for score in prior_scores.values()) or 1.0
            for node_id, score in prior_scores.items():
                if node_id in personalization:
                    personalization[node_id] += score / total_prior
        total_personalization = sum(personalization.values())
        if total_personalization > 0:
            personalization = {
                node_id: score / total_personalization
                for node_id, score in personalization.items()
            }
        if sum(personalization.values()) == 0:
            uniform = 1.0 / max(1, working_graph.number_of_nodes())
            personalization = {node_id: uniform for node_id in working_graph.nodes()}
        return nx.pagerank(
            working_graph,
            alpha=self.config.pagerank_alpha,
            personalization=personalization,
            weight="weight",
            max_iter=100,
        )
