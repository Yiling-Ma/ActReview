import argparse
from pathlib import Path
from typing import Dict, List

from rubric_pipeline_common import (
    load_specs,
    ensure_specs_prompt_messages,
    load_sft_cache,
    load_gpt_candidates,
    load_stage3_final_records,
    make_openai_client,
    stage_batch_rubrics,
    stage_batch_verifiers,
    build_final_records,
    write_jsonl,
)


def _merge_final_records(
    specs_in_order: List[Dict],
    existing_by_rid: Dict[str, Dict],
    new_by_rid: Dict[str, Dict],
) -> List[Dict]:
    merged: List[Dict] = []
    for spec in specs_in_order:
        rid = spec["rubric_id"]
        if rid in existing_by_rid:
            merged.append(existing_by_rid[rid])
        elif rid in new_by_rid:
            merged.append(new_by_rid[rid])
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: extract rubrics from cached candidates, then optionally generate verifiers and final records."
    )
    parser.add_argument("--specs", required=True, help="specs.jsonl")
    parser.add_argument("--sft_candidates", required=True, help="sft_candidates.jsonl")
    parser.add_argument("--gpt_candidates", required=True, help="gpt_candidates.jsonl")
    parser.add_argument("--output", required=True, help="final rubrics jsonl")
    parser.add_argument("--artifact_dir", required=True, help="batch artifact dir")
    parser.add_argument("--batch_name_rubrics", default="stage2_rubric_extraction")
    parser.add_argument("--batch_name_verifiers", default="stage3_verifier_generation")
    parser.add_argument("--save_candidates", action="store_true")
    parser.add_argument(
        "--max_specs",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N specs (test / debug). Default: all specs in the file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rubric_ids that already have a completed rubric in --output; merge with new results.",
    )
    args = parser.parse_args()

    specs = load_specs(Path(args.specs))
    ensure_specs_prompt_messages(specs)

    if args.max_specs is not None:
        if args.max_specs < 0:
            raise ValueError("--max_specs must be >= 0")
        if args.max_specs == 0:
            specs = []
            print("  --max_specs 0: no specs selected")
        else:
            specs = specs[: args.max_specs]
            print(
                f"  --max_specs: using first {len(specs)} spec(s) from file (test mode)"
            )

    if not specs:
        print("No specs to process; exiting without writing output.")
        return

    sft_candidates = load_sft_cache(Path(args.sft_candidates))
    gpt_texts = load_gpt_candidates(Path(args.gpt_candidates))

    specs = [
        s
        for s in specs
        if s["rubric_id"] in sft_candidates and s["rubric_id"] in gpt_texts
    ]
    print(f"Specs eligible for rubric extraction: {len(specs)}")
    if not specs:
        print("No specs to process; exiting without writing output.")
        return

    existing_by_rid: Dict[str, Dict] = {}
    if args.resume:
        existing_by_rid = load_stage3_final_records(Path(args.output))
        if existing_by_rid:
            print(
                f"  --resume: loaded {len(existing_by_rid)} completed record(s) from {args.output}"
            )

    specs_to_run = [s for s in specs if s["rubric_id"] not in existing_by_rid]
    skipped = len(specs) - len(specs_to_run)
    if args.resume and skipped:
        print(f"  --resume: skipping {skipped} spec(s) already present in output")

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    client = make_openai_client()

    if not specs_to_run:
        if existing_by_rid:
            final_records = _merge_final_records(
                specs, existing_by_rid, {}
            )
            write_jsonl(Path(args.output), final_records)
            print(
                f"All {len(specs)} spec(s) already completed; rewrote {args.output} in spec order."
            )
        else:
            print("No specs left to run; exiting without writing output.")
        return

    print(f"\n{'─'*60}")
    print(f"Stage 3A: Rubric extraction via Batch API ({len(specs_to_run)} requests)")
    print(f"{'─'*60}")
    rubrics = stage_batch_rubrics(
        client=client,
        specs=specs_to_run,
        sft_candidates=sft_candidates,
        gpt_texts=gpt_texts,
        artifact_dir=artifact_dir,
        batch_name=args.batch_name_rubrics,
    )

    print(f"\n{'─'*60}")
    n_format = sum(
        sum(1 for r in rub["soft_requirements"] if r["type"] == "format")
        for rub in rubrics.values()
    )
    print(f"Stage 3B: Verifier generation ({n_format} format-type requirements)")
    print(f"{'─'*60}")
    verifier_map = stage_batch_verifiers(
        client=client,
        rubrics=rubrics,
        artifact_dir=artifact_dir,
        batch_name=args.batch_name_verifiers,
    )

    new_records = build_final_records(
        specs=specs_to_run,
        sft_candidates=sft_candidates,
        gpt_texts=gpt_texts,
        rubrics=rubrics,
        verifier_map=verifier_map,
        save_candidates=args.save_candidates,
    )
    new_by_rid = {r["rubric_id"]: r for r in new_records}

    final_records = _merge_final_records(specs, existing_by_rid, new_by_rid)

    write_jsonl(Path(args.output), final_records)

    new_ok = len(new_by_rid)
    failed_new = len(specs_to_run) - new_ok
    t1_total = sum(1 for r in final_records if r["task"] == "task1")
    t2_total = sum(1 for r in final_records if r["task"] == "task2")

    print(f"\n{'='*60}")
    print(f"Total records   : {len(final_records)}")
    print(f"  Task1         : {t1_total}")
    print(f"  Task2         : {t2_total}")
    print(f"New this run    : {new_ok} / {len(specs_to_run)} (failed new: {failed_new})")
    print(f"Output          : {args.output}")
    print(f"Batch artifacts : {artifact_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
