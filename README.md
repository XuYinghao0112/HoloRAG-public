# HoloRAG

This is a clean implementation for testing the core HoloRAG idea.

Core mapping:

- query entities -> graph entity nodes
- query triples/facts -> indexed fact triples
- multi-hop sub-questions -> sentence nodes
- original query -> chunk nodes

The final PageRank personalization is controlled by a granularity vector:

```text
alpha = {fact, sentence, chunk}
```

Entity nodes are still used as graph anchors for entity recognition,
normalization, fact endpoints, and PageRank propagation. Legacy four-way alpha
configs are accepted by folding entity mass into fact.

Recommended task profiles:

- `single_hop`: fact focused
- `multi_hop`: fact/sentence focused, enables query decomposition
- `long_context`: chunk focused
- `auto`: ask the intent router to predict alpha

## Install

From this directory:

```bash
pip install -e .
```

It reuses the same local LLM/OpenAI-compatible endpoint and NV-Embed style encoder assumptions as the main HoloRAG project.

## Run

Index:

```bash
python main.py index \
  --corpus_file data/corpus.json \
  --output_dir outputs/demo \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-72B-Instruct \
  --embedding_name /data/xyh/models/NV-Embed-v2 \
  --embedding_device cuda:0
```

Query:

```bash
python main.py query \
  --corpus_file data/corpus.json \
  --query_text "What question should HoloRAG answer?" \
  --output_dir outputs/demo \
  --task_profile multi_hop \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_name /data/xyh/models/Qwen2.5-72B-Instruct \
  --embedding_name /data/xyh/models/NV-Embed-v2 \
  --embedding_device cuda:0
```

For a single-hop dataset, use `--task_profile single_hop`. For long documents, use `--task_profile long_context`.

## Outputs

The query result includes:

- `alpha`
- `query_entities`
- `query_facts`
- `sub_questions`
- `channel_scores`
- `seeds`
- `ranked_facts`
- `ranked_nodes`
- `ranked_passages`
- `predicted_answer`

The latest query result is saved to:

```text
<output_dir>/last_query_result.json
```
