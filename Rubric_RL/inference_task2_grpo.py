# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import (  # noqa: E402
    SYSTEM_PROMPT,
    check_completeness_task2,
    extract_task2_output,
    format_task2_assistant,
    format_task2_user,
    generate_one,
    get_deduped_claims_and_items,
    is_valid_group,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_path", default=INPUT_JSONL)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("Task2-only inference for GRPO checkpoint")
    print("Task2 uses GT claims as input")
    print("=" * 80)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading test data...")
    raw_groups = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_groups.append(json.loads(line))

    print(f"Loaded {len(raw_groups)} groups from {args.input_path}")

    valid_groups = [g for g in raw_groups if is_valid_group(g)]
    print(f"Valid groups: {len(valid_groups)} | Skipped: {len(raw_groups) - len(valid_groups)}")

    if args.n_samples:
        valid_groups = valid_groups[:args.n_samples]
        print(f"Limited to {len(valid_groups)} groups.")

    stats = {"complete": 0, "incomplete": 0}
    total_outputs = 0

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for group in tqdm(valid_groups, desc="Task2 generation by paper"):
            paper_id = group["paper_id"]
            ctx = group.get("paper_context") or {}
            gt_claims, gt_items = get_deduped_claims_and_items(group)

            for claim_idx, (claim_text, item) in enumerate(zip(gt_claims, gt_items)):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_task2_user(group, claim_text, item)},
                ]

                generated = generate_one(model, tokenizer, messages, "task2")
                generated = extract_task2_output(generated)
                is_complete, completeness_msg = check_completeness_task2(generated)

                result = {
                    "paper_id": paper_id,
                    "claim_idx": claim_idx,
                    "l1_id": group.get("l1_id", ""),
                    "l2_id": group.get("l2_id", ""),
                    "l2_name": group.get("l2_name", ""),
                    "task": "task2",
                    "model_path": args.model_path,
                    "input": {
                        "title": ctx.get("title", ""),
                        "abstract": ctx.get("abstract", ""),
                        "claim_text": claim_text,
                        "user_message": format_task2_user(group, claim_text, item),
                    },
                    "ground_truth": format_task2_assistant(group, claim_text, item),
                    "generated": generated,
                    "is_complete": is_complete,
                    "completeness_msg": completeness_msg,
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                total_outputs += 1

                if is_complete:
                    stats["complete"] += 1
                else:
                    stats["incomplete"] += 1
                    print(
                        f"\nIncomplete Task2 | paper_id={paper_id} | "
                        f"claim_idx={claim_idx} | {completeness_msg}"
                    )

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total outputs: {total_outputs}")
    if total_outputs > 0:
        print(f"Complete: {stats['complete']} ({stats['complete'] / total_outputs * 100:.1f}%)")
        print(f"Incomplete: {stats['incomplete']}")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
