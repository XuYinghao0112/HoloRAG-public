import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_FILE = REPO_ROOT / "reproduce" / "dataset" / "musique_ans_v1.0_dev.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reproduce" / "test" / "groups"
DEFAULT_GROUP = "musique_misc"


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


def next_sample_index(output_root: Path) -> int:
    max_index = 0
    pattern = re.compile(r"^sample_musique(\d+)\.json$")
    for path in output_root.rglob("sample_musique*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly extract one MuSiQue sample into reproduce/test/groups/<group>/sample_musiqueN.json."
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=str(DEFAULT_DATASET_FILE),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP,
        help="Target sample group directory name under reproduce/test/groups/.",
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

    output_root = Path(args.output_root)
    output_dir = output_root / args.group
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.dataset_file)
    sample = random.choice(samples)

    sample_index = next_sample_index(output_root)
    output_path = output_dir / f"sample_musique{sample_index}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, indent=2, ensure_ascii=False)

    print(json.dumps({
        "created_file": str(output_path),
        "sample_index": sample_index,
        "sample_id": sample.get("id"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "group": args.group,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
