import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from .config import HoloRAGConfig
from .embedding_model import NVEmbedV2Encoder
from .extractors import TripleExtractor
from .sentence_segmenter import SentenceSegmenter
from .utils import chunk_text, clean_entity_text, cosine_similarity_matrix, stable_node_id

logger = logging.getLogger(__name__)


class GraphBuilder:
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
        entity_text_by_id: Dict[str, str] = {}
        sentence_payloads: List[Tuple[str, Dict]] = []
        chunk_payloads: List[Tuple[str, Dict]] = []
        fact_payloads: List[Dict] = []
        chunk_sentences: Dict[str, List[str]] = defaultdict(list)

        for doc_index, document in enumerate(documents):
            title = str(document.get("title", f"doc_{doc_index}")).strip() or f"doc_{doc_index}"
            text = str(document.get("text", "")).strip()
            if not text:
                continue
            if self.config.use_paragraph_as_chunk:
                chunk_values = [text]
            else:
                chunk_values = chunk_text(text, self.config.chunk_size_words, self.config.chunk_overlap_words)
            for chunk_index, chunk_value in enumerate(chunk_values):
                chunk_id = stable_node_id("chunk", title, str(chunk_index))
                chunk_text_value = f"{title}\n{chunk_value}"
                graph.add_node(
                    chunk_id,
                    node_type="chunk",
                    text=chunk_text_value,
                    metadata={
                        "title": title,
                        "chunk_index": chunk_index,
                        "document_index": doc_index,
                        "source": dict(document),
                    },
                )
                chunk_payloads.append((chunk_id, {"text": chunk_text_value, "title": title}))
                previous_sentence_id = ""
                for sentence_index, sentence in enumerate(self.sentence_segmenter.split(chunk_value)):
                    sentence_id = stable_node_id("sentence", title, str(chunk_index), str(sentence_index), sentence[:80])
                    extraction = self.triple_extractor.extract_sentence(sentence)
                    if self.config.enable_sentence_layer:
                        graph.add_node(
                            sentence_id,
                            node_type="sentence",
                            text=sentence,
                            metadata={
                                "title": title,
                                "chunk_id": chunk_id,
                                "sentence_index": sentence_index,
                                "document_index": doc_index,
                                "entities": extraction["entities"],
                                "triples": extraction["triples"],
                            },
                        )
                        sentence_payloads.append((sentence_id, {"text": sentence, "title": title, "chunk_id": chunk_id}))
                        chunk_sentences[chunk_id].append(sentence_id)
                        self._merge_edge(graph, sentence_id, chunk_id, 1.0, "sentence_chunk")
                        self._merge_edge(graph, chunk_id, sentence_id, 1.0, "sentence_chunk")
                        if previous_sentence_id:
                            self._merge_edge(graph, previous_sentence_id, sentence_id, 1.0, "sentence_sequence")
                            self._merge_edge(graph, sentence_id, previous_sentence_id, 1.0, "sentence_sequence")
                        previous_sentence_id = sentence_id

                    for entity_name in extraction["entities"]:
                        entity_id = self._add_entity(graph, entity_text_by_id, entity_name)
                        if self.config.enable_sentence_layer:
                            self._merge_edge(graph, entity_id, sentence_id, 1.0, "entity_sentence")
                            self._merge_edge(graph, sentence_id, entity_id, 1.0, "entity_sentence")
                        else:
                            self._merge_edge(graph, entity_id, chunk_id, 1.0, "entity_chunk")
                            self._merge_edge(graph, chunk_id, entity_id, 1.0, "entity_chunk")

                    for triple in extraction["triples"]:
                        head_id = self._add_entity(graph, entity_text_by_id, triple["head"])
                        tail_id = self._add_entity(graph, entity_text_by_id, triple["tail"])
                        relation = str(triple.get("relation", "related_to")).strip() or "related_to"
                        fact_id = stable_node_id("fact", triple["head"], relation, triple["tail"], sentence_id)
                        fact_text = f"{triple['head']} {relation} {triple['tail']}"
                        fact_payloads.append({
                            "fact_id": fact_id,
                            "text": fact_text,
                            "head": triple["head"],
                            "relation": relation,
                            "tail": triple["tail"],
                            "confidence": float(triple.get("confidence", 1.0) or 1.0),
                            "extractor": str(triple.get("extractor", "")),
                            "head_id": head_id,
                            "tail_id": tail_id,
                            "sentence_id": sentence_id if self.config.enable_sentence_layer else "",
                            "chunk_id": chunk_id,
                            "title": title,
                            "document_index": doc_index,
                        })
                        self._merge_edge(graph, head_id, tail_id, 1.0, "entity_relation", relation=relation)
                        if self.config.enable_sentence_layer:
                            self._merge_edge(graph, head_id, sentence_id, 1.0, "entity_sentence")
                            self._merge_edge(graph, tail_id, sentence_id, 1.0, "entity_sentence")
                            self._merge_edge(graph, sentence_id, head_id, 1.0, "entity_sentence")
                            self._merge_edge(graph, sentence_id, tail_id, 1.0, "entity_sentence")
                        else:
                            self._merge_edge(graph, head_id, chunk_id, 1.0, "entity_chunk")
                            self._merge_edge(graph, tail_id, chunk_id, 1.0, "entity_chunk")
                            self._merge_edge(graph, chunk_id, head_id, 1.0, "entity_chunk")
                            self._merge_edge(graph, chunk_id, tail_id, 1.0, "entity_chunk")

        entity_nodes = [(node_id, text) for node_id, text in entity_text_by_id.items() if node_id in graph]
        logger.info(
            "Embedding graph layers: %d entities, %d facts, %d sentences, %d chunks",
            len(entity_nodes),
            len(fact_payloads),
            len(sentence_payloads),
            len(chunk_payloads),
        )
        entity_embeddings = self._encode([text for _, text in entity_nodes], "Encode the entity for retrieval.", "entity")
        fact_embeddings = self._encode([item["text"] for item in fact_payloads], "Encode the factual triple for retrieval.", "fact")
        sentence_embeddings = self._encode([item["text"] for _, item in sentence_payloads], "Encode the sentence for retrieval.", "sentence")
        chunk_embeddings = self._encode([item["text"] for _, item in chunk_payloads], "Encode the chunk for retrieval.", "chunk")
        if self.config.enable_entity_similarity_edges and len(entity_nodes) >= 2:
            self._link_entity_similarity_edges(graph, entity_nodes, entity_embeddings)

        return {
            "graph": graph,
            "facts": fact_payloads,
            "documents": documents,
            "chunk_sentences": dict(chunk_sentences),
            "embeddings": {
                "entity": {node_id: entity_embeddings[index] for index, (node_id, _) in enumerate(entity_nodes)},
                "fact": {item["fact_id"]: fact_embeddings[index] for index, item in enumerate(fact_payloads)},
                "sentence": {node_id: sentence_embeddings[index] for index, (node_id, _) in enumerate(sentence_payloads)},
                "chunk": {node_id: chunk_embeddings[index] for index, (node_id, _) in enumerate(chunk_payloads)},
            },
        }

    def _encode(self, texts: List[str], instruction: str, text_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self.embedder.encode(texts, instruction=instruction, text_type=text_type)

    def _add_entity(self, graph: nx.DiGraph, entity_text_by_id: Dict[str, str], entity_name: str) -> str:
        text = clean_entity_text(entity_name) or str(entity_name).strip()
        entity_id = stable_node_id("entity", text)
        if entity_id not in graph:
            graph.add_node(entity_id, node_type="entity", text=text, metadata={"canonical_name": text})
            entity_text_by_id[entity_id] = text
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
            return
        attrs = {"weight": weight, "edge_type": edge_type, "edge_kinds": [edge_type]}
        if relation:
            attrs["relations"] = [relation]
        graph.add_edge(source, target, **attrs)

    def _link_entity_similarity_edges(
        self,
        graph: nx.DiGraph,
        entity_nodes: List[Tuple[str, str]],
        entity_embeddings: np.ndarray,
    ) -> None:
        entity_ids = [node_id for node_id, _ in entity_nodes]
        for index, source_id in enumerate(entity_ids):
            scores = cosine_similarity_matrix(entity_embeddings[index], entity_embeddings)
            ranked = np.argsort(scores)[::-1]
            added = 0
            for candidate_idx in ranked:
                if candidate_idx == index:
                    continue
                score = float(scores[candidate_idx])
                if score < self.config.entity_similarity_threshold:
                    break
                target_id = entity_ids[candidate_idx]
                self._merge_edge(graph, source_id, target_id, score, "entity_similarity")
                self._merge_edge(graph, target_id, source_id, score, "entity_similarity")
                added += 1
                if added >= self.config.entity_similarity_top_k:
                    break
