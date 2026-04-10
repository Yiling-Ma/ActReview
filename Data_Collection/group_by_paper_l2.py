"""
Groups enhanced_data.jsonl by (paper_id, l2_id), merging all weakness records
that share the same paper and L2 weakness label into a single grouped record.

Input schema (per line):
  {
    "weakness_id": "PaperID_ReviewerXXX_WN",
    "weakness_category": {"l2_id": "L2.1.2", "l2_name": "...", ...},
    "paper_context": {"title": ..., "abstract": ..., "pdf_url": ..., ...},
    "original_weakness": "...",
    "enhanced_review": {"claim": ..., "evidence": ...,
                        "actionable_suggestions": [...], "severity": ...},
    "aligned_snippets": [...],   # optional, from snippet alignment
    "alignment_status": "...",   # optional
    "alignment_error": "...",    # optional
    ...
  }

Output schema (per line):
  {
    "paper_id": "...",
    "l2_id": "...",
    "l2_name": "...",
    "paper_context": {...},
    "weakness_items": [
      {
        "weakness_id": "...",
        "original_weakness": "...",
        "enhanced_review": {...},
        "aligned_snippets": [...],   # included only when alignment_status=="ok" and non-empty
        ...
      },
      ...
    ]
  }

"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def extract_paper_id(weakness_id: str) -> str:
    return weakness_id.split("_")[0] if "_" in weakness_id else weakness_id


def build_weakness_item(r: dict) -> dict:
    """Build one weakness item for SFT input.

    We only include aligned_snippets when snippet alignment succeeded
    (alignment_status == "ok") and the snippet list is non-empty.
    """
    item = {
        "weakness_id": r.get("weakness_id", ""),
        "original_weakness": (r.get("original_weakness") or "").strip(),
        "enhanced_review": r.get("enhanced_review", {}),
    }

    status = (r.get("alignment_status") or "").strip()
    snips = r.get("aligned_snippets") or []
    if status == "ok" and isinstance(snips, list) and len(snips) > 0:
        item["aligned_snippets"] = snips

    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Path to enhanced_data.jsonl or enhanced_data_with_snippets.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output path for grouped jsonl")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Filter out records below this taxonomy confidence (if present)")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Max weakness items per group (drops excess; keeps highest confidence first)")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    raw = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    print(f"Loaded {len(raw)} records from {args.input}")

    # ── Filter ────────────────────────────────────────────────────────────────
    filtered = []
    skipped = 0
    for r in raw:
        cat = r.get("weakness_category") or {}
        l2_id = cat.get("l2_id", "UNKNOWN")
        if l2_id == "UNKNOWN":
            skipped += 1
            continue
        conf = float(cat.get("confidence", 1.0))
        if conf < args.min_confidence:
            skipped += 1
            continue
        er = r.get("enhanced_review")
        if not er or not (er.get("claim") or "").strip():
            skipped += 1
            continue
        filtered.append(r)

    print(f"After filtering: {len(filtered)} valid records, {skipped} skipped")

    # ── Group by (paper_id, l2_id) ────────────────────────────────────────────
    groups: dict[tuple, list] = defaultdict(list)
    for r in filtered:
        paper_id = extract_paper_id(r.get("weakness_id", ""))
        l2_id    = (r.get("weakness_category") or {}).get("l2_id", "UNKNOWN")
        groups[(paper_id, l2_id)].append(r)

    print(f"Unique (paper_id, l2_id) groups: {len(groups)}")

    # ── Build output records ───────────────────────────────────────────────────
    output_records = []
    for (paper_id, l2_id), members in groups.items():
        # Sort by confidence descending (if available), keep stable order
        members.sort(
            key=lambda r: float((r.get("weakness_category") or {}).get("confidence", 1.0)),
            reverse=True,
        )
        if args.max_items:
            members = members[:args.max_items]

        # Use paper_context from first (highest-confidence) member
        first = members[0]
        cat   = first.get("weakness_category") or {}

        weakness_items = [build_weakness_item(r) for r in members]

        output_records.append({
            "paper_id":      paper_id,
            "l2_id":         l2_id,
            "l2_name":       cat.get("l2_name", ""),
            "l1_id":         cat.get("l1_id", ""),
            "paper_context": first.get("paper_context", {}),
            "weakness_items": weakness_items,
        })

    # ── Stats ─────────────────────────────────────────────────────────────────
    sizes = [len(r["weakness_items"]) for r in output_records]
    from collections import Counter
    size_dist = Counter(sizes)
    single   = size_dist[1]
    multi    = sum(v for k, v in size_dist.items() if k >= 2)

    print(f"\nOutput groups: {len(output_records)}")
    print(f"  Single-weakness groups: {single} ({100*single/len(output_records):.1f}%)")
    print(f"  Multi-weakness groups:  {multi} ({100*multi/len(output_records):.1f}%)")
    print(f"\nGroup size distribution:")
    for k in sorted(size_dist.keys()):
        print(f"  {k} item(s): {size_dist[k]} groups")

    # ── Write ─────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(output_records)} grouped records → {args.output}")


if __name__ == "__main__":
    main()
