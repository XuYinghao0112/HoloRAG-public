import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from .config import HoloRAGConfig
from .embedding_model import NVEmbedV2Encoder
from .sentence_segmenter import SentenceSegmenter
from .triple_extractor import TripleExtractor
from .utils import (
    chunk_text,
    clean_entity_text,
    cosine_similarity_matrix,
    entity_match_score,
    entity_matches_title,
    generate_entity_aliases,
    jaccard_similarity,
    stable_node_id,
)

logger = logging.getLogger(__name__)


class HierarchicalGraphBuilder:
    def __init__(
        self,
        config: HoloRAGConfig,
        sentence_segmenter: SentenceSegmenter,
        triple_extractor: TripleExtractor,
        embedder: NVEmbedV2Encoder,
    ) -> None:
        self.config = config
        self.sentence_segmenter = sentence_segmenter
        self.triple_extractor = triple_extractor
        self.embedder = embedder

    def build(self, documents: List[Dict[str, str]]) -> Dict:
        graph = nx.DiGraph()
        doc_chunks: List[Tuple[str, Dict]] = []
        sentence_payloads: List[Tuple[str, Dict]] = []
        fact_payloads: List[Dict] = []
        sentence_entities = defaultdict(set)
        chunk_entities = defaultdict(set)

        for doc_index, document in enumerate(documents):
            title = str(document.get("title", f"doc_{doc_index}")).strip() or f"doc_{doc_index}"
            text = str(document.get("text", "")).strip()
            if not text:
                continue
            chunk_texts = chunk_text(text, self.config.chunk_size_words, self.config.chunk_overlap_words)
            for chunk_index, chunk_value in enumerate(chunk_texts):
                chunk_id = stable_node_id("chunk", title, str(chunk_index))
                graph.add_node(
                    chunk_id,
                    node_type="chunk",
                    text=f"{title}\n{chunk_value}",
                    metadata={"title": title, "chunk_index": chunk_index, "document_index": doc_index},
                )
                doc_chunks.append((chunk_id, {"title": title, "text": f"{title}\n{chunk_value}"}))

                sentences = self.sentence_segmenter.split(chunk_value)
                sentence_extractions = self._extract_sentences_parallel(sentences)
                previous_sentence_id = None
                for sentence_index, sentence in enumerate(sentences):
                    extraction = sentence_extractions[sentence_index]
                    sentence_id = ""
                    if self.config.enable_sentence_layer:
                        sentence_id = stable_node_id("sentence", title, str(chunk_index), str(sentence_index), sentence[:80])
                        graph.add_node(
                            sentence_id,
                            node_type="sentence",
                            text=sentence,
                            metadata={
                                "title": title,
                                "chunk_id": chunk_id,
                                "sentence_index": sentence_index,
                                "document_index": doc_index,
                            },
                        )
                        self._merge_edge(graph, sentence_id, chunk_id, 1.0, "sentence_chunk")
                        self._merge_edge(graph, chunk_id, sentence_id, 1.0, "sentence_chunk")
                        sentence_payloads.append((sentence_id, {"text": sentence}))
                        if previous_sentence_id is not None:
                            self._merge_edge(graph, previous_sentence_id, sentence_id, 0.95, "sentence_sequence")
                            self._merge_edge(graph, sentence_id, previous_sentence_id, 0.95, "sentence_sequence")
                        previous_sentence_id = sentence_id
                        graph.nodes[sentence_id]["metadata"]["triples"] = extraction["triples"]
                        graph.nodes[sentence_id]["metadata"]["entities"] = extraction["entities"]

                    for triple in extraction["triples"]:
                        head_id = self._add_entity_node(graph, triple["head"], title=title)
                        tail_id = self._add_entity_node(graph, triple["tail"], title=title)
                        fact_payloads.append({
                            "fact_id": stable_node_id(
                                "fact",
                                triple["head"],
                                triple["relation"],
                                triple["tail"],
                                sentence_id or chunk_id,
                                str(sentence_index),
                            ),
                            "text": f"{triple['head']} {triple['relation']} {triple['tail']}",
                            "head_id": head_id,
                            "tail_id": tail_id,
                            "sentence_id": sentence_id,
                            "chunk_id": chunk_id,
                            "document_index": doc_index,
                        })
                        self._merge_edge(graph, head_id, tail_id, 1.0, "entity_relation", relation=triple["relation"])
                        if self.config.enable_sentence_layer and sentence_id:
                            self._merge_edge(graph, head_id, sentence_id, 1.0, "entity_sentence")
                            self._merge_edge(graph, tail_id, sentence_id, 1.0, "entity_sentence")
                            self._merge_edge(graph, sentence_id, head_id, 0.8, "entity_sentence")
                            self._merge_edge(graph, sentence_id, tail_id, 0.8, "entity_sentence")
                            sentence_entities[sentence_id].update([head_id, tail_id])
                        chunk_entities[chunk_id].update([head_id, tail_id])
                    for entity_name in extraction["entities"]:
                        entity_id = self._add_entity_node(graph, entity_name, title=title)
                        if self.config.enable_sentence_layer and sentence_id:
                            self._merge_edge(graph, entity_id, sentence_id, 0.8, "entity_sentence")
                            self._merge_edge(graph, sentence_id, entity_id, 0.6, "entity_sentence")
                            sentence_entities[sentence_id].add(entity_id)
                        chunk_entities[chunk_id].add(entity_id)

        entity_nodes = [(node_id, attrs["text"]) for node_id, attrs in graph.nodes(data=True) if attrs.get("node_type") == "entity"]
        entity_embeddings = self.embedder.encode(
            [text for _, text in entity_nodes],
            instruction="Encode the entity for alignment.",
            text_type="entity",
        )
        sentence_embeddings = self.embedder.encode(
            [payload["text"] for _, payload in sentence_payloads],
            instruction="Encode the sentence for retrieval.",
            text_type="sentence",
        ) if sentence_payloads else np.zeros((0, 1), dtype=np.float32)
        chunk_embeddings = self.embedder.encode(
            [payload["text"] for _, payload in doc_chunks],
            instruction="Encode the chunk for retrieval.",
            text_type="chunk",
        )
        fact_embeddings = self.embedder.encode(
            [payload["text"] for payload in fact_payloads],
            instruction="Encode the triplet fact for retrieval.",
            text_type="sentence",
        ) if fact_payloads else np.zeros((0, 1), dtype=np.float32)

        if self.config.enable_alias_linking and entity_nodes:
            self._link_entity_aliases(graph, entity_nodes, entity_embeddings)
        if self.config.enable_chunk_bridges and doc_chunks:
            self._link_chunk_bridges(graph, doc_chunks, chunk_embeddings, chunk_entities)

        for sentence_id, entities in sentence_entities.items():
            if sentence_id in graph:
                graph.nodes[sentence_id]["metadata"]["entity_ids"] = sorted(entities)
        for chunk_id, entities in chunk_entities.items():
            graph.nodes[chunk_id]["metadata"]["entity_ids"] = sorted(entities)

        return {
            "graph": graph,
            "embeddings": {
                "entity": {node_id: entity_embeddings[idx] for idx, (node_id, _) in enumerate(entity_nodes)},
                "fact": {payload["fact_id"]: fact_embeddings[idx] for idx, payload in enumerate(fact_payloads)},
                "sentence": {node_id: sentence_embeddings[idx] for idx, (node_id, _) in enumerate(sentence_payloads)},
                "chunk": {node_id: chunk_embeddings[idx] for idx, (node_id, _) in enumerate(doc_chunks)},
            },
            "facts": fact_payloads,
            "documents": documents,
        }

    def _extract_sentences_parallel(self, sentences: List[str]) -> List[Dict[str, List]]:
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self.triple_extractor.extract(sentences[0])]
        max_workers = min(8, len(sentences))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.triple_extractor.extract, sentences))

    def _add_entity_node(self, graph: nx.DiGraph, entity_name: str, title: str = "") -> str:
        normalized_entity_name = clean_entity_text(entity_name)
        if not normalized_entity_name:
            normalized_entity_name = str(entity_name).strip()

        use_title_anchor = bool(title and entity_matches_title(normalized_entity_name, title))
        display_text = str(title).strip() if use_title_anchor else normalized_entity_name
        entity_id = stable_node_id("entity", "title_anchor", display_text) if use_title_anchor else stable_node_id("entity", normalized_entity_name)
        aliases = generate_entity_aliases(display_text) + generate_entity_aliases(normalized_entity_name)
        if entity_id not in graph:
            graph.add_node(
                entity_id,
                node_type="entity",
                text=display_text,
                metadata={
                    "canonical_name": display_text,
                    "aliases": sorted(dict.fromkeys(aliases)),
                    "surface_forms": sorted(dict.fromkeys(generate_entity_aliases(normalized_entity_name))),
                    "title_anchor": str(title).strip() if use_title_anchor else "",
                },
            )
        else:
            metadata = graph.nodes[entity_id].setdefault("metadata", {})
            merged_aliases = list(metadata.get("aliases", []))
            for alias in aliases:
                if alias not in merged_aliases:
                    merged_aliases.append(alias)
            metadata["aliases"] = merged_aliases

            surface_forms = list(metadata.get("surface_forms", []))
            for surface_form in generate_entity_aliases(normalized_entity_name):
                if surface_form not in surface_forms:
                    surface_forms.append(surface_form)
            metadata["surface_forms"] = surface_forms
        return entity_id

    def _merge_edge(self, graph: nx.DiGraph, source: str, target: str, weight: float, edge_type: str, relation: str = "") -> None:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
            kinds = set(graph[source][target].get("edge_kinds", []))
            kinds.add(edge_type)
            graph[source][target]["edge_kinds"] = sorted(kinds)
            if relation:
                relations = set(graph[source][target].get("relations", []))
                relations.add(relation)
                graph[source][target]["relations"] = sorted(relations)
        else:
            attrs = {"weight": weight, "edge_type": edge_type, "edge_kinds": [edge_type]}
            if relation:
                attrs["relations"] = [relation]
            graph.add_edge(source, target, **attrs)

    def _link_entity_aliases(self, graph: nx.DiGraph, entity_nodes: List[Tuple[str, str]], entity_embeddings: np.ndarray) -> None:
        if len(entity_nodes) < 2:
            return
        for index, (node_id, _) in enumerate(entity_nodes):
            scores = cosine_similarity_matrix(entity_embeddings[index], entity_embeddings)
            ranked = np.argsort(scores)[::-1]
            added = 0
            for candidate_idx in ranked:
                if candidate_idx == index:
                    continue
                candidate_node_id, _ = entity_nodes[candidate_idx]
                alias_score = entity_match_score(
                    graph.nodes[node_id].get("text", ""),
                    graph.nodes[candidate_node_id].get("text", ""),
                )
                combined_score = max(float(scores[candidate_idx]), 0.85 * alias_score + 0.15 * float(scores[candidate_idx]))
                if alias_score < 0.70 and combined_score < self.config.entity_alias_threshold:
                    break
                self._merge_edge(graph, node_id, candidate_node_id, combined_score, "entity_alias")
                self._merge_edge(graph, candidate_node_id, node_id, combined_score, "entity_alias")
                added += 1
                if added >= self.config.entity_alias_top_k:
                    break

    def _link_chunk_bridges(
        self,
        graph: nx.DiGraph,
        doc_chunks: List[Tuple[str, Dict]],
        chunk_embeddings: np.ndarray,
        chunk_entities: Dict[str, set],
    ) -> None:
        if len(doc_chunks) < 2:
            return
        chunk_ids = [chunk_id for chunk_id, _ in doc_chunks]
        for index, chunk_id in enumerate(chunk_ids):
            scores = cosine_similarity_matrix(chunk_embeddings[index], chunk_embeddings)
            ranked = np.argsort(scores)[::-1]
            added = 0
            for candidate_idx in ranked:
                if candidate_idx == index:
                    continue
                candidate_chunk_id = chunk_ids[candidate_idx]
                overlap = jaccard_similarity(chunk_entities.get(chunk_id, set()), chunk_entities.get(candidate_chunk_id, set()))
                score = self.config.chunk_bridge_eta * overlap + (1.0 - self.config.chunk_bridge_eta) * float(scores[candidate_idx])
                if score < self.config.chunk_bridge_threshold:
                    continue
                self._merge_edge(graph, chunk_id, candidate_chunk_id, score, "chunk_bridge")
                added += 1
                if added >= self.config.chunk_bridge_top_k:
                    break
