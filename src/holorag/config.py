from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HoloRAGConfig:
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model_name: str = "/data/xyh/models/Qwen2.5-72B-Instruct"
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    save_dir: str = "outputs/holorag"
    embedding_device: str = "cuda:1"
    embedding_batch_size: int = 8
    embedding_max_seq_len: int = 2048
    embedding_dtype: str = "bfloat16"

    entity_max_length: int = 64
    fact_max_length: int = 128
    sentence_max_length: int = 256
    chunk_max_length: int = 512
    query_max_length: int = 128
    llm_context_window: int = 8192
    qa_max_input_tokens: int = 7000
    max_new_tokens: int = 512
    temperature: float = 0.0

    chunk_size_words: int = 256
    chunk_overlap_words: int = 64
    use_paragraph_as_chunk: bool = True
    index_extraction_mode: str = "heuristic"
    spacy_model_name: str = "en_core_web_sm"
    task_profile: str = "auto"
    enable_intent_routing: bool = True
    intent_use_llm: bool = False
    enable_query_decomposition: bool = True
    enable_entity_similarity_edges: bool = True
    entity_similarity_threshold: float = 0.8
    entity_similarity_top_k: int = 2047
    enable_granularity_awareness: bool = True
    enable_sentence_layer: bool = True

    entity_top_k: int = 12
    fact_top_k: int = 12
    sentence_top_k: int = 20
    chunk_top_k: int = 12
    passage_output_top_k: int = 10
    qa_passage_top_k: int = 4

    pagerank_alpha: float = 0.5
    transition_lambda: float = 1.2
    enable_granularity_pagerank_bias: bool = True
    hub_penalty: float = 0.08
    seed_floor: float = 1e-6
    fact_rerank_top_k: int = 24
    fact_rerank_keep_k: int = 12
    fact_rerank_use_llm: bool = False
    fact_rerank_llm_candidate_k: int = 12
    fact_rerank_llm_keep_k: int = 5
    enable_fact_source_first_evidence: bool = False
    enable_fact_chunk_boost: bool = False
    fact_chunk_boost: float = 0.35
    enable_fair_sentence_context: bool = False
    evidence_extra_ranked_sentence_k: int = 6
    evidence_max_sentences: int = 18
    evidence_title_limit: int = 3
    evidence_passage_context_k: int = 2
    evidence_passage_excerpt_tokens: int = 150
    evidence_chunk_max_tokens: int = 256
    evidence_packing_mode: str = "alpha_count"
    evidence_alpha_total_units: int = 20
    evidence_alpha_uniform_mix: float = 0.0
    evidence_soft_token_budget: int = 0
    evidence_allow_underfill: bool = True
    evidence_min_score: float = 0.0
    evidence_redundancy_threshold: float = 0.85
    evidence_use_alpha_weights: bool = True
    enable_no_fact_fallback: bool = True
    entity_hub_suppression: float = 1.0

    edge_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "entity_relation": 1.0,
        "entity_similarity": 0.9,
        "entity_sentence": 1.0,
        "entity_chunk": 0.9,
        "sentence_chunk": 1.0,
        "sentence_sequence": 0.6,
    })
    profile_alpha_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "single_hop": {"fact": 0.60, "sentence": 0.30, "chunk": 0.10},
        "multi_hop": {"fact": 0.40, "sentence": 0.40, "chunk": 0.20},
        "long_context": {"fact": 0.10, "sentence": 0.20, "chunk": 0.70},
    })

    query_instruction: str = "Represent the question for retrieval."
    query_instruction_fact: str = "Represent the question for matching factual triples and constraints."
    query_instruction_text: str = "Represent the question for matching sentence and chunk evidence."
