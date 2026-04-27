# HoloRAG

HoloRAG is a hierarchical graph-based Retrieval-Augmented Generation (RAG) pipeline designed for multi-hop QA.
It extends HoloRAG-style graph retrieval with a multi-granularity graph (entity / sentence / chunk), query decomposition, and evidence-aware answer generation.

This repository supports:
- End-to-end indexing and querying via one CLI (`main_holorag.py`)
- MuSiQue single-sample and grouped evaluation workflows
- Retrieval and QA metric evaluation with JSON outputs
- Ablation toggles for fair component-level analysis

---

## Table of Contents

- [1. Features](#1-features)
- [2. Repository Layout](#2-repository-layout)
- [3. Environment Setup](#3-environment-setup)
- [4. Data Format](#4-data-format)
- [5. Quick Start](#5-quick-start)
- [6. MuSiQue Grouped Runs](#6-musique-grouped-runs)
- [7. Evaluation](#7-evaluation)
- [8. Key Parameters](#8-key-parameters)
- [9. Ablation Switches](#9-ablation-switches)
- [10. Reproducibility Checklist](#10-reproducibility-checklist)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Citation](#12-citation)

---

## 1. Features

- Hierarchical graph indexing across different text granularities.
- Hybrid retrieval signals:
  - dense similarity
  - graph propagation (biased transition / PageRank-style)
  - fact-level scoring
- Multi-hop reasoning support through sub-question decomposition.
- OpenAI-compatible LLM endpoint integration (e.g., local vLLM).
- Structured output artifacts for debugging and evaluation.

---

## 2. Repository Layout

```text
HoloRAG/
├── main_holorag.py                      # Main CLI: index/query
├── eval_holorag_musique.py              # MuSiQue-style evaluator
├── extract_musique_sample.py            # Extract dataset samples
├── requirements.txt
├── setup.py
├── README.md
├── bin/
│   └── run_qwen72b_musique_samples.sh   # Grouped batch runner
├── src/holorag/
│   ├── pipeline.py                      # End-to-end orchestration
│   ├── config.py                        # Config dataclass/defaults
│   ├── graph_builder.py                 # Graph construction
│   ├── biased_pagerank.py               # Biased propagation
│   ├── query_decomposer.py              # Sub-question decomposition
│   ├── seed_selector.py                 # Seed selection
│   ├── intent_parser.py                 # Query intent/layer routing
│   ├── recognition_filter.py            # Optional filter/judge
│   ├── evidence_extractor.py            # Evidence packaging
│   ├── qa_reader.py                     # Reader prompting/parsing
│   ├── embedding_model.py               # Embedding wrapper
│   └── llm_client.py                    # OpenAI-compatible client
├── reproduce/
│   ├── dataset/                         # Original datasets
│   └── test/groups/                     # Grouped test samples
└── outputs/
    └── qwen_72b_result/
        ├── groups/                      # Per-sample runtime outputs
        ├── eval/                        # Aggregated eval JSONs
        └── logs/                        # Run logs
```

---

## 3. Environment Setup

### 3.1 Requirements

- Python `>=3.10`
- CUDA-capable GPU (recommended)
- OpenAI-compatible chat completion endpoint

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

### 3.2 Typical Local Serving Configuration

- `llm_base_url`: `http://127.0.0.1:8000/v1`
- `llm_name`: `/data/xyh/models/Qwen2.5-72B-Instruct`
- `embedding_name`: `/data/xyh/models/NV-Embed-v2`

Quick health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 4. Data Format

### 4.1 General Corpus Input

`main_holorag.py index` expects a corpus JSON (or MuSiQue single-sample JSON).

### 4.2 MuSiQue Single-Sample Input

Expected fields include:
- `paragraphs`
- `question`
- `answer` / `answer_aliases`

In query mode, if `--query_text` is omitted, the sample `question` is used automatically.

---

## 5. Quick Start

### 5.1 Index

```bash
python main_holorag.py index \
  --corpus_file reproduce/dataset/sample_corpus.json \
  --output_dir outputs/holorag_demo \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-72B-Instruct \
  --embedding_name /data/xyh/models/NV-Embed-v2 \
  --embedding_device cuda:0
```

### 5.2 Query

```bash
python main_holorag.py query \
  --output_dir outputs/holorag_demo \
  --query_text "Which Stanford neuroscientist is also a CEO and what context connects him to the others?" \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-72B-Instruct \
  --embedding_name /data/xyh/models/NV-Embed-v2
```

### 5.3 Main Output Files

- `holorag_index.pkl`: built index/graph artifacts
- `last_query_result.json`: final retrieval + reasoning + answer output

---

## 6. MuSiQue Grouped Runs

Use grouped folders under:
`reproduce/test/groups/musique_XX_YY/`

Run a full group with the provided script:

```bash
SAMPLES_GROUP=musique_31_40 \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_NAME=/data/xyh/models/Qwen2.5-72B-Instruct \
EMBEDDING_NAME=/data/xyh/models/NV-Embed-v2 \
EMBEDDING_VISIBLE_DEVICES=2 \
EMBEDDING_DEVICE=cuda:0 \
bash bin/run_qwen72b_musique_samples.sh
```

Common script environment variables:
- `SAMPLES_GROUP` (e.g. `musique_01_10`, `musique_11_20`)
- `LLM_BASE_URL`, `LLM_NAME`, `EMBEDDING_NAME`
- `EMBEDDING_VISIBLE_DEVICES`, `EMBEDDING_DEVICE`
- `SKIP_EXISTING` (reuse existing sample outputs)
- `WAIT_TIMEOUT_SECONDS` (service readiness timeout)

Outputs go to:
- per-sample: `outputs/qwen_72b_result/groups/<group>/<sample>/`
- group eval: `outputs/qwen_72b_result/eval/`
- logs: `outputs/qwen_72b_result/logs/`

---

## 7. Evaluation

Run evaluation manually:

```bash
python eval_holorag_musique.py \
  --samples_glob reproduce/test/groups/musique_31_40/sample_musique*.json \
  --outputs_dir outputs/qwen_72b_result/groups/musique_31_40 \
  --result_filename last_query_result.json \
  --retrieval_k 5 \
  --output_json outputs/qwen_72b_result/eval/holorag_eval_musique_samples_31_40.json
```

### 7.1 Metric Definitions

- `passage_recall_at_k`:
  proportion of supporting titles covered in top-k retrieved titles.

- `passage_hit_at_k`:
  binary per-sample metric, 1 if at least one supporting title appears in top-k.

- `qa_exact_match`, `qa_f1`:
  final answer quality against gold answers.

### 7.2 Important Interpretation

`passage_hit_at_5 = 1.0` can still be realistic if every sample hits at least one support title in top-5.
It does not imply all support evidence was retrieved. Always interpret it with:
- `passage_recall_at_5`
- QA metrics (`qa_exact_match`, `qa_f1`)

---

## 8. Key Parameters

Parameter entry points:
- `src/holorag/config.py` (global defaults)
- `main_holorag.py` (CLI override layer)

High-impact runtime settings include:
- Retrieval budget (`retrieval_top_k` / eval `--retrieval_k`)
- Decomposition depth and reasoning chain count
- Graph propagation behavior (biased transition switches)
- Final evidence packaging size for reader prompt
- LLM endpoint/model and decoding settings

Recommendation:
Record exact CLI commands in experiment logs for reproducibility.

---

## 9. Ablation Switches

Available CLI switches (depending on branch/version):

- `--disable_sentence_layer`
- `--disable_recognition_filter`
- `--disable_intent_routing`
- `--disable_chunk_bridges`
- `--disable_alias_linking`
- `--disable_biased_transition`
- `--enable_llm_judge`

Use these to isolate module contributions under controlled settings.

---

## 10. Reproducibility Checklist

For fair comparisons (e.g., vs baseline methods):

1. Use identical sample sets and ordering.
2. Align LLM endpoint/model and embedding model.
3. Align retrieval budget and final evidence budget.
4. Align prompt templates and decoding policy.
5. Report both quality and cost:
   - latency
   - LLM call count
   - token usage
6. Keep all ablation toggles documented.

---

## 11. Troubleshooting

- LLM service unavailable:
  verify `curl <LLM_BASE_URL>/models`.

- GPU process persists after `kill`:
  terminate parent `nohup bash -lc` process first, then child python PIDs.

- Eval cannot find outputs:
  ensure `--outputs_dir` points to the directory containing per-sample result folders.

- `passage_hit_at_k` appears too high:
  inspect `passage_recall_at_k` and per-sample `matched_support_titles` to validate retrieval depth.

---

## 12. Citation

If you use this codebase, please cite the corresponding project/paper and include commit hash + configuration summary in your appendix.

---

## License

See `LICENSE`.
