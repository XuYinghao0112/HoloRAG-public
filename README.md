# HoloRAG

HoloRAG is a hierarchical graph-based Retrieval-Augmented Generation (RAG) pipeline for multi-hop QA and evidence-grounded reasoning.

Current implementation highlights:
- Multi-granularity graph: `entity / sentence / chunk`
- Hybrid retrieval: dense retrieval + graph propagation + fact-level signals
- Query decomposition + hop reasoning chain
- OpenAI-compatible LLM endpoint integration (local vLLM or compatible services)
- Single CLI for indexing and querying (`main_holorag.py`)

---

## Table of Contents

- [1. Features](#1-features)
- [2. Repository Layout](#2-repository-layout)
- [3. Environment Setup](#3-environment-setup)
- [4. Data Format](#4-data-format)
- [5. Quick Start](#5-quick-start)
- [6. Outputs](#6-outputs)
- [7. Key Parameters](#7-key-parameters)
- [8. Ablation Switches](#8-ablation-switches)
- [9. Evaluation & Experiment Scripts](#9-evaluation--experiment-scripts)
- [10. Troubleshooting](#10-troubleshooting)
- [11. License](#11-license)

---

## 1. Features

- Hierarchical graph indexing from raw documents.
- Entity-relation extraction and fact graph construction.
- Query-aware intent routing over granularity preferences.
- Retrieval backbone with:
  - dense passage retrieval
  - fact retrieval and reranking
  - granularity-biased PageRank propagation
- Multi-step QA with intermediate reasoning chain and focused final answer generation.

---

## 2. Repository Layout

```text
HoloRAG/
├── main_holorag.py                      # Main CLI: index/query
├── eval_holorag_musique.py              # Grouped-output evaluator (can be adapted to other datasets)
├── extract_musique_sample.py            # Dataset sample extractor utility (optional)
├── requirements.txt
├── setup.py
├── README.md
├── bin/
│   └── run_qwen72b_musique_samples.sh   # Batch runner example (script name is historical)
├── scripts/
│   ├── eval_musique.py                  # End-to-end eval script example
│   ├── run_ablation_eval.py             # Ablation experiment runner example
│   └── musique_case_study.py            # Case study script example
├── src/holorag/
│   ├── pipeline.py                      # Core HoloRAG pipeline
│   ├── config.py                        # HoloRAGConfig defaults
│   ├── graph_builder.py                 # Multi-granularity graph construction
│   ├── biased_pagerank.py               # Granularity-biased PageRank
│   ├── query_decomposer.py              # Sub-question decomposition
│   ├── intent_parser.py                 # Query intent/granularity routing
│   ├── recognition_filter.py            # Evidence relevance filtering
│   ├── passage_coverage_reranker.py     # Passage diversity/coverage reranking
│   ├── evidence_extractor.py            # Evidence packaging
│   ├── qa_reader.py                     # QA prompting/response parsing
│   ├── triple_extractor.py              # Triple/entity extraction
│   ├── sentence_segmenter.py            # Sentence splitting
│   ├── embedding_model.py               # Embedding encoder wrapper
│   ├── llm_client.py                    # OpenAI-compatible LLM client
│   └── utils.py
├── reproduce/
│   ├── dataset/
│   └── test/
└── outputs/
```

> Note: Some script filenames are historical, but core `main_holorag.py` + `src/holorag` are dataset-agnostic.

---

## 3. Environment Setup

### 3.1 Requirements

- Python `>=3.10`
- CUDA-capable GPU recommended
- OpenAI-compatible Chat Completions endpoint

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

### 3.2 Typical Serving Configuration

- `llm_base_url`: `http://127.0.0.1:8000/v1`
- `llm_name`: your chat model identifier/path
- `embedding_name`: your embedding model identifier/path

Health check:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

---

## 4. Data Format

`main_holorag.py --corpus_file` accepts JSON in these forms:

1. Document list (`list[dict]`):
- each item supports `title`
- text field can be `text` / `content` / `paragraph_text`

2. Plain text list (`list[str]`):
- each string is treated as one document (`title` auto-generated)

3. Single QA sample object (`dict` with `paragraphs`):
- each paragraph supports `title` + `paragraph_text`/`text`
- if `query` mode omits `--query_text`, `question` field is used automatically

Minimal recommended document list example:

```json
[
  {
    "title": "Doc A",
    "text": "Your passage text here."
  },
  {
    "title": "Doc B",
    "text": "Another passage."
  }
]
```

---

## 5. Quick Start

### 5.1 Build Index

```bash
python main_holorag.py index \
  --corpus_file reproduce/test/sample_corpus.json \
  --output_dir outputs/holorag_demo \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /path/to/your/chat-model \
  --embedding_name /path/to/your/embedding-model \
  --embedding_device cuda:0
```

### 5.2 Run Query

```bash
python main_holorag.py query \
  --corpus_file reproduce/test/sample_corpus.json \
  --output_dir outputs/holorag_demo \
  --query_text "Your multi-hop question here" \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /path/to/your/chat-model \
  --embedding_name /path/to/your/embedding-model \
  --embedding_device cuda:0
```

For single-sample QA JSON with a `question` field, `--query_text` can be omitted.

---

## 6. Outputs

Main artifacts written to `--output_dir`:

- `holorag_index.pkl`
  - graph + embeddings + fact structures for retrieval
- `last_query_result.json`
  - retrieval ranking, reasoning chain, evidence bundle, predicted answer, and timing

The query output includes useful fields such as:
- `ranked_passages`
- `ranked_facts`
- `query_entity_resolutions`
- `sub_questions`
- `reasoning_chain`
- `predicted_answer`
- `query_timing`

---

## 7. Key Parameters

CLI entrypoint: `main_holorag.py` (also mapped into `HoloRAGConfig`).

High-impact settings:

- Retrieval budget:
  - `--retrieval_top_k`
  - `--fact_top_k`
  - `--fact_rerank_top_k`
  - `--passage_output_top_k`
  - `--qa_passage_top_k`

- Embedding/runtime:
  - `--embedding_device`
  - `--embedding_batch_size`
  - `--embedding_max_seq_len`
  - `--embedding_dtype`

- Scoring weights:
  - `--dense_passage_weight`
  - `--graph_passage_weight`
  - `--fact_passage_weight`
  - `--fact_entity_spread_weight`
  - `--passage_node_weight`

- Link/bridge controls:
  - `--linking_top_k`
  - `--fact_candidate_top_k`
  - `--bridge_entity_top_k`

Suggestion: keep a log of the exact command + config per run for reproducibility.

---

## 8. Ablation Switches

`main_holorag.py` supports:

- `--disable_sentence_layer`
- `--disable_recognition_filter`
- `--disable_intent_routing`
- `--disable_chunk_bridges`
- `--disable_alias_linking`
- `--disable_biased_transition`
- `--enable_llm_judge`

These are useful for component contribution analysis.

---

## 9. Evaluation & Experiment Scripts

Core pipeline usage for any dataset should be based on `main_holorag.py`.

This repo also contains experiment scripts under `scripts/` and root-level evaluators. Some script names and defaults are dataset-specific (historical naming), but they can be adapted by changing:

- input dataset paths
- split/sampling logic
- metric adapters
- output directory layout assumptions

For custom datasets, treat these scripts as templates rather than fixed evaluation standards.

---

## 10. Troubleshooting

- LLM service unreachable:
  - check `curl -fsS <LLM_BASE_URL>/models`
  - ensure endpoint is OpenAI-compatible chat completions API

- Embedding OOM:
  - reduce `--embedding_batch_size`
  - reduce max lengths (`--chunk_max_length`, etc.)
  - select a smaller embedding model or use a lower-memory device

- Missing index at query time:
  - run `index` first in the same `--output_dir`
  - verify `<output_dir>/holorag_index.pkl` exists

- Unexpectedly weak retrieval:
  - increase retrieval budgets (`--retrieval_top_k`, `--passage_output_top_k`)
  - tune dense/graph/fact weight mix
  - inspect `last_query_result.json` fields (`ranked_passages`, `reasoning_chain`, `query_entity_resolutions`)

---

## 11. License

See `LICENSE`.
