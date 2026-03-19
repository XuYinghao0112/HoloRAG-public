# HoloRAG

HoloRAG is a clean standalone evolution of HippoRAG focused on one method:

- offline hierarchical graph construction across entity, sentence, and chunk layers
- online granularity-aware retrieval and reasoning
- local OpenAI-compatible LLM calls through a vLLM endpoint
- NV-Embed-v2 for all embedding-based retrieval and graph linking

## Project Layout

```text
main_holorag.py
src/holorag/
reproduce/dataset/sample_corpus.json
```

`main_holorag.py` is the supported entrypoint for both indexing and querying.

## Environment

```bash
conda create -n holorag python=3.10
conda activate holorag
pip install -r requirements.txt
```

If you use a local OpenAI-compatible server, no real OpenAI key is required. The client defaults to a placeholder key for local use.

## Local Models

- LLM endpoint: `http://127.0.0.1:8000/v1`
- LLM model: `/data/xyh/models/Qwen2.5-7B-Instruct`
- Embedding model: `nvidia/NV-Embed-v2`

## Run HoloRAG

Index:

```bash
python main_holorag.py index \
  --corpus_file reproduce/dataset/sample_corpus.json \
  --output_dir outputs/holorag_demo \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-7B-Instruct \
  --embedding_device auto \
  --embedding_batch_size 4 \
  --embedding_name nvidia/NV-Embed-v2
```

Query:

```bash
python main_holorag.py query \
  --output_dir outputs/holorag_demo \
  --query_text "Which Stanford neuroscientist is also a CEO and what context connects him to the others?" \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-7B-Instruct \
  --embedding_name nvidia/NV-Embed-v2
```

Useful ablations:

- `--disable_sentence_layer`
- `--disable_recognition_filter`
- `--disable_intent_routing`
- `--disable_chunk_bridges`
- `--disable_alias_linking`
- `--disable_biased_transition`
- `--enable_llm_judge`

## MuSiQue Single-Sample Format

`main_holorag.py` can also read a MuSiQue-style single-sample JSON object containing `paragraphs`, `question`, and `answer`.

Index directly from the sample:

```bash
python main_holorag.py index \
  --corpus_file reproduce/dataset/sample_musique1.json \
  --output_dir outputs/sample_musique1 \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-7B-Instruct \
  --embedding_device cpu \
  --embedding_batch_size 1 \
  --embedding_max_seq_len 512 \
  --embedding_name nvidia/NV-Embed-v2
```

Query directly from the same sample. If you omit `--query_text`, HoloRAG will automatically use the sample's `question`:

```bash
python main_holorag.py query \
  --corpus_file reproduce/dataset/sample_musique1.json \
  --output_dir outputs/sample_musique1 \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-7B-Instruct \
  --embedding_name nvidia/NV-Embed-v2
```
