import argparse
from pathlib import Path
from typing import Dict, Optional

from rubric_pipeline_common import (
    load_specs,
    ensure_specs_prompt_messages,
    load_sft_cache,
    load_gpt_candidates,
    make_openai_client,
    stage_batch_gpt_candidates,
    stage_sync_openai_gpt_candidates,
    write_gpt_candidates,
)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: generate offline GPT candidates.")
    parser.add_argument("--specs", required=True, help="specs.jsonl")
    parser.add_argument("--sft_candidates", required=True, help="sft_candidates.jsonl")
    parser.add_argument("--output", required=True, help="gpt_candidates.jsonl")
    parser.add_argument(
        "--artifact_dir",
        default=None,
        help="Batch artifact dir (required unless --sync)",
    )
    parser.add_argument("--batch_name", default="stage1_gpt_candidates")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use Chat Completions API instead of Batch API (no --artifact_dir)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model id (default: OPENAI_MODEL env or rubric_pipeline_common default)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Retries per spec when using --sync (default: 3)",
    )
    parser.add_argument(
        "--retry_sleep_seconds",
        type=float,
        default=5.0,
        help="Sleep between retries when using --sync (default: 5)",
    )
    parser.add_argument(
        "--request_sleep_seconds",
        type=float,
        default=0.0,
        help="Optional throttle between successful requests when using --sync (default: 0)",
    )
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
        help="Skip rubric_ids that already have non-empty text in --output; merge with new batch results.",
    )
    args = parser.parse_args()

    if not args.sync and not args.artifact_dir:
        raise RuntimeError("--artifact_dir is required unless --sync is set.")

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
    artifact_dir: Optional[Path] = None
    if not args.sync:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

    specs = [s for s in specs if s["rubric_id"] in sft_candidates]
    print(f"Specs with valid SFT candidates: {len(specs)}")
    if not specs:
        print("No specs to process; exiting without writing output.")
        return

    existing: Dict[str, str] = {}
    if args.resume:
        existing = load_gpt_candidates(Path(args.output))
        if existing:
            print(
                f"  --resume: loaded {len(existing)} existing GPT line(s) from {args.output}"
            )

    specs_to_run = [s for s in specs if s["rubric_id"] not in existing]
    skipped = len(specs) - len(specs_to_run)
    if args.resume and skipped:
        print(f"  --resume: skipping {skipped} spec(s) already present in output")

    if not specs_to_run:
        if existing:
            print(
                f"All {len(specs)} spec(s) already have GPT text in output; nothing to batch."
            )
        else:
            print("No specs left to batch; exiting without writing output.")
        return

    client = make_openai_client()
    if args.sync:
        gpt_texts = stage_sync_openai_gpt_candidates(
            client=client,
            specs=specs_to_run,
            sft_candidates=sft_candidates,
            model=args.model,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            request_sleep_seconds=args.request_sleep_seconds,
        )
    else:
        assert artifact_dir is not None
        gpt_texts = stage_batch_gpt_candidates(
            client=client,
            specs=specs_to_run,
            sft_candidates=sft_candidates,
            artifact_dir=artifact_dir,
            batch_name=args.batch_name,
        )

    merged = {**existing, **gpt_texts}
    write_gpt_candidates(Path(args.output), merged)
    print(f"GPT candidates written to: {args.output}")
    print(f"GPT success this run: {len(gpt_texts)} / {len(specs_to_run)}")
    print(f"Total GPT lines in output: {len(merged)} (target scope: {len(specs)} specs)")


if __name__ == "__main__":
    main()
