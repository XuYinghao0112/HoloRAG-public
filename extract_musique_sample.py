import argparse
import json
import os
import random
import re
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No valid samples found in {path}")
    return samples


def next_sample_index(output_dir: str) -> int:
    max_index = 0
    pattern = re.compile(r"^sample_musique(\d+)\.json$")
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly extract one MuSiQue sample from a jsonl file into reproduce/test/sample_musiqueN.json."
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="/data/xyh/code/HoloRAG/reproduce/dataset/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/xyh/code/HoloRAG/reproduce/test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    samples = load_jsonl(args.dataset_file)
    sample = random.choice(samples)

    sample_index = next_sample_index(args.output_dir)
    output_path = os.path.join(args.output_dir, f"sample_musique{sample_index}.json")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(sample, handle, indent=2, ensure_ascii=False)

    print(json.dumps({
        "created_file": output_path,
        "sample_index": sample_index,
        "sample_id": sample.get("id"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
