import argparse
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from rubric_pipeline_common import (
    load_specs,
    ensure_specs_prompt_messages,
    load_sft_model,
    stage_local_sft_candidates,
    load_sft_cache,
    read_sft_cache_meta,
    write_sft_cache_compact,
    write_sft_cache_meta,
    sft_spec_digest,
    is_valid_sft_candidates,
)


def _chunk_specs(specs: List[Dict], n_chunks: int) -> List[List[Dict]]:
    if not specs or n_chunks <= 0:
        return []
    n_chunks = min(n_chunks, len(specs))
    base, rem = divmod(len(specs), n_chunks)
    out: List[List[Dict]] = []
    i = 0
    for j in range(n_chunks):
        sz = base + (1 if j < rem else 0)
        out.append(specs[i : i + sz])
        i += sz
    return out


def _sft_worker_run(payload: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Run in a child process after CUDA_VISIBLE_DEVICES is set."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_device"])
    from rubric_pipeline_common import load_sft_model, stage_local_sft_candidates

    shard = payload["specs"]
    if not shard:
        return {}
    model, tokenizer = load_sft_model(payload["checkpoint"])
    part_path = Path(payload["part_path"])
    return stage_local_sft_candidates(
        shard,
        model,
        tokenizer,
        cache_path=part_path,
    )


def _parse_gpu_ids(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _worker_part_path(artifact_dir: Path, output_stem: str, worker_idx: int) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", output_stem)
    return artifact_dir / f"{safe}.worker{worker_idx}.jsonl"


def _remove_worker_parts(artifact_dir: Path, output_stem: str) -> None:
    safe = re.sub(r"[^\w.\-]+", "_", output_stem)
    pat = f"{safe}.worker*.jsonl"
    for p in artifact_dir.glob(pat):
        try:
            p.unlink()
        except OSError:
            pass


"""
python rubric_rl/prepare_specs.py \
  --input data/data_splits/checklist.jsonl \
  --output rubric_rl/specs_50groups.jsonl \
  --n_groups 4000 \
  --seed 42
"""
def main():
    parser = argparse.ArgumentParser(description="Stage 1: generate offline SFT candidates.")
    parser.add_argument("--specs", required=True, help="specs.jsonl")
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--output", required=True, help="sft_candidates.jsonl")
    parser.add_argument("--artifact_dir", required=True, help="artifact dir for SFT meta")
    parser.add_argument("--force_sft", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel GPU workers (one model per worker). Requires one GPU per worker.",
    )
    parser.add_argument(
        "--gpu_ids",
        default=None,
        help="Comma-separated CUDA device ids for workers, e.g. 0,1,2,3. "
        "Default: 0,1,...,num_workers-1",
    )
    parser.add_argument(
        "--keep_worker_parts",
        action="store_true",
        help="Keep per-worker jsonl shards under artifact_dir after merge",
    )
    parser.add_argument(
        "--max_specs",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N specs (test / debug). Default: all specs in the file.",
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
        print("No specs to process; exiting without writing cache.")
        return

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sft_cache_path = Path(args.output)

    digest = sft_spec_digest(specs)
    spec_id_set = {s["rubric_id"] for s in specs}

    if args.force_sft:
        if sft_cache_path.exists():
            sft_cache_path.unlink()
        meta_p = artifact_dir / "sft_candidates.meta.json"
        if meta_p.exists():
            meta_p.unlink()
        cached = {}
        print("  --force_sft: cleared SFT cache for this run")
    else:
        cached = load_sft_cache(sft_cache_path)
        meta = read_sft_cache_meta(artifact_dir)
        if meta and meta.get("spec_rubric_ids_sha256") != digest:
            print(
                "  Warning: sft_candidates.meta.json does not match current spec set; "
                "reusing cache lines only for matching rubric_ids (last line wins)."
            )

    cached_in_scope = {
        rid: c for rid, c in cached.items()
        if rid in spec_id_set and is_valid_sft_candidates(c)
    }
    missing_specs = [s for s in specs if s["rubric_id"] not in cached_in_scope]

    if cached_in_scope:
        print(f"Loaded {len(cached_in_scope)} valid SFT entries from {sft_cache_path}")
    if missing_specs:
        print(f"Running SFT for {len(missing_specs)} specs (cache miss or invalid)")

    sft_new: Dict[str, List[Dict]] = {}
    nw = max(1, int(args.num_workers))
    n_parallel = min(nw, len(missing_specs)) if missing_specs else 0

    if missing_specs and n_parallel > 1:
        if args.gpu_ids is not None:
            gpu_ids = _parse_gpu_ids(args.gpu_ids)
        else:
            gpu_ids = list(range(nw))
        if len(gpu_ids) < n_parallel:
            print(
                f"  Warning: --gpu_ids has {len(gpu_ids)} id(s) but need {n_parallel} workers; "
                "cycling device ids (may contend on same GPU)."
            )
        chunks = _chunk_specs(missing_specs, n_parallel)
        _remove_worker_parts(artifact_dir, sft_cache_path.stem)
        print(
            f"  Parallel: {n_parallel} workers on gpu_ids={gpu_ids[:min(len(gpu_ids), n_parallel)]}..."
        )
        ctx = mp.get_context("spawn")
        futures = []
        with ProcessPoolExecutor(max_workers=n_parallel, mp_context=ctx) as ex:
            for wi, chunk in enumerate(chunks):
                if not chunk:
                    continue
                dev = gpu_ids[wi % len(gpu_ids)]
                part_path = _worker_part_path(
                    artifact_dir, sft_cache_path.stem, wi
                )
                payload = {
                    "specs": chunk,
                    "checkpoint": args.sft_checkpoint,
                    "part_path": str(part_path),
                    "cuda_device": dev,
                }
                futures.append(ex.submit(_sft_worker_run, payload))
            for fut in as_completed(futures):
                part = fut.result()
                sft_new.update(part)
        if not args.keep_worker_parts:
            _remove_worker_parts(artifact_dir, sft_cache_path.stem)
    elif missing_specs:
        sft_model, sft_tokenizer = load_sft_model(args.sft_checkpoint)
        sft_new = stage_local_sft_candidates(
            missing_specs,
            sft_model,
            sft_tokenizer,
            cache_path=sft_cache_path,
        )

    sft_candidates = {**cached_in_scope, **sft_new}
    write_sft_cache_compact(sft_cache_path, specs, sft_candidates)
    write_sft_cache_meta(artifact_dir, digest, len(specs))

    print(f"SFT succeeded: {len(sft_candidates)} / {len(specs)}")
    print(f"SFT cache written to: {sft_cache_path}")

if __name__ == "__main__":
    main()