from typing import Dict, List


class SeedSelector:
    def select(
        self,
        alpha: Dict[str, float],
        layer_scores: Dict[str, Dict[str, float]],
        seed_budget: int,
        layer_buckets: Dict[str, List[List[Dict]]] | None = None,
    ) -> List[Dict]:
        ordered_layers = ["entity", "sentence", "chunk"]
        layer_buckets = layer_buckets or {}
        budgets = {layer: int(round(alpha.get(layer, 0.0) * seed_budget)) for layer in ordered_layers}
        while sum(budgets.values()) < seed_budget:
            best_layer = max(ordered_layers, key=lambda layer: alpha.get(layer, 0.0))
            budgets[best_layer] += 1
        seeds = []
        for layer in ordered_layers:
            layer_budget = budgets.get(layer, 0)
            chosen_ids = set()
            if layer in layer_buckets and layer_buckets[layer]:
                bucket_positions = [0] * len(layer_buckets[layer])
                while layer_budget > 0:
                    progressed = False
                    for bucket_index, bucket in enumerate(layer_buckets[layer]):
                        while bucket_positions[bucket_index] < len(bucket) and bucket[bucket_positions[bucket_index]]["node_id"] in chosen_ids:
                            bucket_positions[bucket_index] += 1
                        if bucket_positions[bucket_index] >= len(bucket):
                            continue
                        item = bucket[bucket_positions[bucket_index]]
                        seeds.append({"node_id": item["node_id"], "score": float(item["score"]), "layer": layer})
                        chosen_ids.add(item["node_id"])
                        bucket_positions[bucket_index] += 1
                        layer_budget -= 1
                        progressed = True
                        if layer_budget <= 0:
                            break
                    if not progressed:
                        break

            ranked = sorted(layer_scores.get(layer, {}).items(), key=lambda item: item[1], reverse=True)
            for node_id, score in ranked:
                if layer_budget <= 0:
                    break
                if node_id in chosen_ids:
                    continue
                seeds.append({"node_id": node_id, "score": float(score), "layer": layer})
                chosen_ids.add(node_id)
                layer_budget -= 1
        seeds.sort(key=lambda item: item["score"], reverse=True)
        return seeds[:seed_budget]
