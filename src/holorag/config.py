from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HoloRAGConfig:
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model_name: str = "/data/xyh/models/Qwen2.5-72B-Instruct"
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    save_dir: str = "outputs/holorag"
    embedding_device: str = "cuda:1"
    embedding_batch_size: int = 4
    embedding_max_seq_len: int = 2048
    embedding_dtype: str = "bfloat16"
    entity_max_length: int = 64
    sentence_max_length: int = 256
    chunk_max_length: int = 512
    query_max_length: int = 128
    max_new_tokens: int = 512
    temperature: float = 0.0
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    chunk_bridge_top_k: int = 4
    chunk_bridge_threshold: float = 0.42
    chunk_bridge_eta: float = 0.55
    entity_alias_threshold: float = 0.86
    entity_alias_top_k: int = 3
    linking_top_k: int = 5
    fact_candidate_top_k: int = 24
    retrieval_top_k: int = 20
    entity_top_k: int = 12
    fact_top_k: int = 12
    fact_rerank_top_k: int = 8
    fact_output_top_k: int = 8
    sentence_top_k: int = 16
    chunk_top_k: int = 10
    passage_output_top_k: int = 10
    qa_passage_top_k: int = 3
    hop_answer_passage_top_k: int = 1
    entity_resolution_score_threshold: float = 0.68
    entity_resolution_margin_threshold: float = 0.10
    seed_budget: int = 12
    entity_hops: int = 2
    sentence_beam_width: int = 3
    sentence_expand_top_k: int = 3
    sentence_step_top_k: int = 8
    chunk_walk_hops: int = 1
    hub_penalty: float = 0.15
    pagerank_alpha: float = 0.85
    transition_lambda: float = 1.2
    enable_sentence_layer: bool = True
    enable_recognition_filter: bool = True
    enable_intent_routing: bool = True
    enable_chunk_bridges: bool = True
    enable_alias_linking: bool = True
    enable_granularity_biased_transition: bool = True
    enable_llm_judge: bool = False
    lexical_mix_weight: float = 0.30
    dense_passage_weight: float = 0.55
    graph_passage_weight: float = 0.30
    fact_passage_weight: float = 0.15
    fact_entity_spread_weight: float = 0.30
    bridge_entity_top_k: int = 6
    min_fact_passage_signal: float = 0.20
    min_graph_passage_signal: float = 0.10
    passage_node_weight: float = 0.10
    edge_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "entity_relation": 1.0,
        "entity_alias": 0.7,
        "entity_sentence": 1.15,
        "sentence_sequence": 0.95,
        "sentence_peer": 1.0,
        "sentence_dep": 1.1,
        "sentence_chunk": 1.1,
        "chunk_bridge": 0.9,
    })
    query_instruction: str = (
        "Represent the question for retrieval across entity, sentence, and discourse evidence."
    )
