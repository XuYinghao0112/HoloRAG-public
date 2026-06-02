#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


FIELDS = [
    "F1",
    "EM",
    "index_latency",
    "retrieval_latency",
    "retrieval_pipeline_latency",
    "qa_latency",
    "retrieval_qa_latency",
    "query_runtime",
    "total_runtime",
    "nodes",
    "edges",
    "entity_nodes",
    "fact_nodes",
    "sentence_nodes",
    "chunk_nodes",
    "final_evidence_tokens",
]


def parse_seed(run_name: str) -> str:
    match = re.search(r"_c(\d+)_G\d+", run_name)
    return match.group(1) if match else ""


def std(values):
    return stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_global_scaling_repeats.py OUTPUT_DIR", file=sys.stderr)
        return 2

    output_dir = Path(sys.argv[1]).expanduser()
    grouped = defaultdict(list)
    for metrics_path in sorted(output_dir.glob("*/metrics_summary.json")):
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not rows:
            continue
        row = rows[0]
        scale = int(row.get("scale_passages", 0))
        record = {"run_name": metrics_path.parent.name, "corpus_seed": parse_seed(metrics_path.parent.name), **row}
        grouped[scale].append(record)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "scale_passages",
            "num_runs",
            "corpus_seeds",
            *[f"{field}_mean" for field in FIELDS],
            *[f"{field}_std" for field in FIELDS],
        ],
    )
    writer.writeheader()
    for scale in sorted(grouped):
        records = grouped[scale]
        out = {
            "scale_passages": scale,
            "num_runs": len(records),
            "corpus_seeds": ";".join(str(record.get("corpus_seed", "")) for record in records),
        }
        for field in FIELDS:
            values = [float(record[field]) for record in records if isinstance(record.get(field), (int, float))]
            out[f"{field}_mean"] = mean(values) if values else math.nan
            out[f"{field}_std"] = std(values) if values else math.nan
        writer.writerow(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
