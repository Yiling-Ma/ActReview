"""
Convert rubrics.jsonl → parquet files for verl GRPO training.

verl expects a parquet dataset with columns:
  - prompt:       List[Dict]  (chat messages, will be formatted by verl)
  - rubric_id:    str
  - task:         str
  - rubric:       str  (JSON-serialized rubric dict, stored in extra_info)
  - data_source:  str  (required by verl, set to "actionreview")
  - reward_model: str  (required by verl for routing, set to "rubric")

"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def load_rubrics(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_rows(records):
    rows = []
    for rec in records:
        rubric_id = rec["rubric_id"]
        task      = rec["task"]
        rubric    = rec["rubric"]
        msgs      = rec["prompt_messages"]

        rows.append({
            # verl reads "prompt" as the chat messages list
            "prompt":      msgs,
            # stored in extra_info → passed to compute_score
            "rubric_id":   rubric_id,
            "task":        task,
            # Keep rubric as dict so reward_fn can use rubric.get(...)
            "rubric":      rubric,
            # verl required fields
            "data_source": "actionreview",
            "reward_model": "rubric",
            # ground_truth is unused (rubric does the scoring) but verl may need it
            "ground_truth": "",
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rubrics",     required=True)
    p.add_argument("--output_dir",  default="rubric_rl/verl_data")
    p.add_argument("--train_ratio", type=float, default=1.0,
                   help="Fraction of data for training (rest goes to validation)")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    records = load_rubrics(args.rubrics)
    print(f"Loaded {len(records)} rubrics")
    print(f"  task1: {sum(1 for r in records if r['task']=='task1')}")
    print(f"  task2: {sum(1 for r in records if r['task']=='task2')}")

    random.seed(args.seed)
    random.shuffle(records)

    rows = build_rows(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_ratio < 1.0:
        split = int(len(rows) * args.train_ratio)
        train_rows = rows[:split]
        val_rows   = rows[split:]
        pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet", index=False)
        pd.DataFrame(val_rows).to_parquet(output_dir / "val.parquet", index=False)
        print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
        print(f"Saved to {output_dir}/train.parquet and val.parquet")
    else:
        pd.DataFrame(rows).to_parquet(output_dir / "train.parquet", index=False)
        print(f"Train: {len(rows)}")
        print(f"Saved to {output_dir}/train.parquet")


if __name__ == "__main__":
    main()