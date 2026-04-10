"""
Converts classified enhanced reviews into two-turn SFT training format for
Qwen3-8B-Instruct full fine-tuning.

Input:  classified_enhanced_reviews.jsonl  (step2 output)
Output: sft_train.jsonl
        sft_val.jsonl
        sft_stats.json

Record schema:
{
  "messages": [
    {"role": "system",    "content": "...two-turn system prompt..."},
    {"role": "user",      "content": "...paper context + weakness..."},
    {"role": "assistant", "content": "...one-sentence claim..."},
    {"role": "user",      "content": "...claim + request for suggestions..."},
    {"role": "assistant", "content": "...Evidence + Suggestions + Severity..."}
  ],
  "meta": {
    "weakness_id":    "zrT3HcsWSAt_AnonReviewer1_W2",
    "taxonomy": {"l1_id": "L1.1", "l2_id": "L2.1.3"},
    "source_paper_id": "zrT3HcsWSAt",
    "split": "train"
  }
}

"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# System prompt  (defines the exact output template the model must follow)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert academic peer reviewer specializing in machine learning and AI research.

You work in two turns:

TURN 1 — CLAIM
Given a paper's context and a weakness type, write exactly ONE precise sentence \
that states the core weakness as a clear, testable claim.
Output only that single sentence, with no headings or extra text.

TURN 2 — EVIDENCE AND SUGGESTIONS
Given the claim from Turn 1, produce the structured critique below exactly.

─────────────────────────────────────────────────────────
OUTPUT TEMPLATE  (use these headings verbatim)
─────────────────────────────────────────────────────────

## Evidence
1-2 sentences citing specific paper locations where the issue manifests.
Use the paper's actual section/table names when known; otherwise describe
the location functionally (e.g., "the section introducing the main algorithm").

### Suggestion 1
- **What**: Exact action required (add which baseline / ablation / proof / clarification)
- **Where**: Precise location — table name, section heading, appendix label, or \
functional description if exact name is unknown
- **How**: Step-by-step implementation details (datasets, metrics, seeds, equations, \
code changes)
- **Expected Outcome**: What the result should demonstrate and how it resolves the weakness
- **Priority**: critical | high | medium

### Suggestion 2
(repeat as needed; include only suggestions that are necessary and non-redundant)

## Severity
critical | major | moderate | minor

─────────────────────────────────────────────────────────
REQUIREMENTS
─────────────────────────────────────────────────────────
- Every suggestion must be specific and immediately executable
- WHERE must name an actual location — never write "in the paper" or "somewhere"
- HOW must include concrete steps a researcher can follow without guessing
- NEVER reference the authors' rebuttal or any post-submission response
- Write as if providing initial review guidance, not reactive commentary
- Do NOT add sections not in the template\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Evidence hallucination guard
# ─────────────────────────────────────────────────────────────────────────────

_BARE_SECTION_RE = re.compile(
    r"\b(?:Section|Sec\.?|Appendix|App\.?)\s+([A-Z]?\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
_BARE_TABLEFIG_RE = re.compile(
    r"\b(?:Table|Figure|Fig\.?)\s+(\d+[a-zA-Z]?)\b",
    re.IGNORECASE,
)

# Matches:  Section 'noisy title'  or  Section "noisy title"
# with optional trailing page ref like (p.9) or (page 9)
# Also captures an optional preceding "the " to avoid "the the relevant section"
_QUOTED_SECTION_RE = re.compile(
    r"""(\bthe\s+)?                         # optional preceding "the "
        (?:(?:[Ss]ection|[Ss]ec\.?)\s+)    # "Section " prefix
        (['"])(.*?)\2                       # quoted title
        (\s*\(p(?:age)?\.?\s*\d+\))?       # optional page ref
    """,
    re.VERBOSE,
)

# Tokens that typically end a truncated sentence fragment, not a real heading
_TRAILING_STOPWORDS = {"the", "a", "an", "of", "in", "is", "to", "for", "and", "or",
                       "that", "with", "on", "by", "from", "as", "are", "was", "be"}


def _is_noisy_section_title(title: str) -> bool:
    """
    Heuristic: return True if *title* looks like OCR / extraction noise
    rather than a genuine section heading.

    Positive signals for noise:
      - Very long (> 8 words)
      - Ends with a stop-word / preposition (sentence was cut mid-phrase)
      - Contains sentence-internal periods (not abbreviations)
      - Contains commas followed by lowercase (clause continuation)
      - Opens with a lowercase letter (not an acronym)
      - Ends with an open parenthesis or quote
    """
    words = title.split()
    if not words:
        return True

    # Long titles are almost certainly noise
    if len(words) > 8:
        return True

    # Ends with a stop-word → truncated sentence
    if words[-1].strip(".,;:\"'").lower() in _TRAILING_STOPWORDS:
        return True

    # Contains a mid-title period followed by a space+letter → sentence boundary
    # But skip numbered headings like "4. Convergence Analysis" or "A.3 Sensitivity"
    if re.search(r"\.\s+[A-Za-z]", title):
        # Allow if the period is part of a leading section number (e.g. "4." or "A.3.")
        if not re.match(r"^[A-Z]?\d+(?:\.\d+)*\.\s", title):
            return True

    # Ends with open parenthesis or trailing punctuation artifacts
    stripped = title.rstrip()
    if stripped and stripped[-1] in ("(", '"', "'"):
        return True

    # Ends with a period → sentence text, not a heading
    # (real headings like "A.3" won't match because they won't be the last char)
    if stripped and stripped[-1] == ".":
        return True

    # Starts with lowercase → body text, not a heading
    # (skip if it looks like a numbered sub-section e.g. "e.g." edge case)
    first_alpha = next((c for c in title if c.isalpha()), None)
    if first_alpha and first_alpha.islower():
        return True

    # Contains colon followed by whitespace and a word — truncated compound title
    # like "Methods and Techniques for Proving Inequalities: In Mathematical"
    if re.search(r":\s+[A-Z]", title) and len(words) > 5:
        return True

    return False


def _replace_noisy_quoted_section(m: re.Match) -> str:
    """Replace a noisy quoted section reference, keeping page info if present."""
    # Groups: (1) optional "the ", (2) quote char, (3) title, (4) page ref
    title = m.group(3)
    page_ref = (m.group(4) or "").strip()

    if not _is_noisy_section_title(title):
        return m.group(0)  # keep genuine titles unchanged

    if page_ref:
        return f"the relevant section {page_ref}"
    return "the relevant section"


def sanitize_location_refs(text: str, sections_index: Optional[list]) -> str:
    """
    Prevent the model from learning fake / noisy location references.

    Two independent passes:
    1. **Noisy quoted section titles** (always applied):
       Section 'approaches conventional entropy - no hubs …' (p.9)
         → the relevant section (p.9)
    2. **Bare numeric references** (only when sections_index is absent):
       Section 3.1  → the relevant section
       Table 3      → the results table
    """
    # Pass 1 — always: replace noisy quoted section titles
    text = _QUOTED_SECTION_RE.sub(_replace_noisy_quoted_section, text)

    # Pass 2 — only when no sections_index: replace bare numeric refs
    if not sections_index:
        def replace_section(m: re.Match) -> str:
            prefix = m.group(0).split()[0].rstrip(".").lower()
            if "appendix" in prefix or prefix == "app":
                return "the appendix"
            return "the relevant section"

        def replace_tablefig(m: re.Match) -> str:
            prefix = m.group(0).split()[0].rstrip(".").lower()
            return "the results table" if "table" in prefix else "the figure"

        text = _BARE_SECTION_RE.sub(replace_section, text)
        text = _BARE_TABLEFIG_RE.sub(replace_tablefig, text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_source_paper_id(weakness_id: str) -> str:
    return weakness_id.split("_")[0] if "_" in weakness_id else weakness_id


def format_user_message(record: dict) -> str:
    """Turn-1 user message: paper context + weakness."""
    ctx      = record.get("paper_context", {})
    title    = ctx.get("title", "Unknown Title")
    abstract = ctx.get("abstract", "Not available.")
    keywords = ", ".join(ctx.get("keywords") or []) or "Not specified"
    weakness = (record.get("original_weakness") or "").strip()

    cat    = record.get("weakness_category", {})
    l2name = cat.get("l2_name", "")
    cat_hint = f"\n[Weakness type: {l2name}]" if l2name else ""

    return (
        f"**Paper**: {title}\n\n"
        f"**Keywords**: {keywords}\n\n"
        f"**Abstract**:\n{abstract}\n\n"
        f"**Weakness to elaborate**:{cat_hint}\n{weakness}"
    )


def format_assistant_claim(record: dict) -> str:
    """Turn-1 assistant message: one-sentence claim only."""
    enhanced = record.get("enhanced_review", {})
    return (enhanced.get("claim") or "").strip()


def format_user_turn2(record: dict) -> str:
    """Turn-2 user message: echo claim back and request evidence + suggestions."""
    enhanced = record.get("enhanced_review", {})
    claim    = (enhanced.get("claim") or "").strip()
    return (
        f"**Claim**: {claim}\n\n"
        "Based on the claim above, provide the evidence and actionable suggestions "
        "following the template exactly."
    )


def format_assistant_suggestions(record: dict) -> str:
    """
    Turn-2 assistant message: Evidence + Suggestions + Severity (no Claim header).
    Applies hallucination guard on Evidence + WHERE fields.
    """
    enhanced       = record.get("enhanced_review", {})
    sections_index = record.get("paper_context", {}).get("sections_index")

    lines = []

    # Evidence — sanitize if no sections_index
    evidence = enhanced.get("evidence")
    if isinstance(evidence, str):
        evidence = evidence.strip()
    if not evidence:
        evidence = "Specific paper location not identified from available context."
    evidence = sanitize_location_refs(evidence, sections_index)
    lines.append("## Evidence")
    lines.append(evidence)
    lines.append("")

    # Suggestions
    for i, s in enumerate((enhanced.get("actionable_suggestions") or []), 1):
        what    = (s.get("what")             or "").strip()
        where   = (s.get("where")            or "").strip()
        how     = (s.get("how")              or "").strip()
        outcome = (s.get("expected_outcome") or "").strip()
        prio    = (s.get("priority")         or "medium").strip()

        where = sanitize_location_refs(where, sections_index)

        lines.append(f"### Suggestion {i}")
        lines.append(f"- **What**: {what}")
        lines.append(f"- **Where**: {where}")
        lines.append(f"- **How**: {how}")
        lines.append(f"- **Expected Outcome**: {outcome}")
        lines.append(f"- **Priority**: {prio}")
        lines.append("")

    # Citations (optional)
    citations = enhanced.get("citations") or []
    if citations:
        lines.append("## Relevant Prior Work")
        for c in citations:
            lines.append(f"- {c}")
        lines.append("")

    # Severity
    severity = (enhanced.get("severity") or "major").strip()
    lines.append("## Severity")
    lines.append(severity)

    return "\n".join(lines)


def format_assistant_message(record: dict) -> str:
    """
    Legacy single-turn format (kept for backward compatibility / inspection).
    Returns full review: Claim + Evidence + Suggestions + Severity.
    """
    enhanced       = record.get("enhanced_review", {})
    sections_index = record.get("paper_context", {}).get("sections_index")

    lines = []

    claim = (enhanced.get("claim") or "").strip()
    lines.append("## Claim")
    lines.append(claim)
    lines.append("")

    lines.append(format_assistant_suggestions(record))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_record(record: dict, min_confidence: float) -> tuple[bool, str]:
    er = record.get("enhanced_review")
    if not er:
        return False, "missing enhanced_review"
    if not (er.get("claim") or "").strip():
        return False, "empty claim"
    evidence = er.get("evidence")
    if evidence is not None and not isinstance(evidence, str):
        return False, "invalid evidence type"

    suggestions = er.get("actionable_suggestions") or []
    if not suggestions:
        return False, "no suggestions"
    for i, s in enumerate(suggestions):
        for field in ("what", "where", "how", "expected_outcome", "priority"):
            if not (s.get(field) or "").strip():
                return False, f"suggestion[{i}] missing '{field}'"

    if not (record.get("original_weakness") or "").strip():
        return False, "empty original_weakness"

    cat  = record.get("weakness_category") or {}
    l2id = cat.get("l2_id", "UNKNOWN")
    conf = float(cat.get("confidence", 0.0))
    if l2id == "UNKNOWN":
        return False, "unclassified"
    if conf < min_confidence:
        return False, f"low confidence ({conf:.2f})"

    # Check both assistant turns for rebuttal leakage
    assistant_text = (
        format_assistant_claim(record) + "\n" + format_assistant_suggestions(record)
    ).lower()
    for phrase in [
        "as the authors", "in their response", "the rebuttal",
        "authors mentioned", "authors showed", "authors added",
        "in response to",
    ]:
        if phrase in assistant_text:
            return False, f"rebuttal leakage: '{phrase}'"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampling
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(
    records: list[dict],
    max_per_l2: Optional[int],
    seed: int = 42,
) -> list[dict]:
    if max_per_l2 is None:
        return records

    by_l2: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        l2 = r.get("weakness_category", {}).get("l2_id", "UNKNOWN")
        by_l2[l2].append(r)

    rng = random.Random(seed)
    sampled = []
    for recs in by_l2.values():
        recs_sorted = sorted(
            recs,
            key=lambda r: (
                -float(r.get("weakness_category", {}).get("confidence", 0.0)),
                rng.random(),
            )
        )
        sampled.extend(recs_sorted[:max_per_l2])
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    input_path: str,
    output_dir: str,
    val_ratio: float = 0.05,
    min_confidence: float = 0.6,
    max_per_l2: Optional[int] = None,
    seed: int = 42,
) -> None:
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    print(f"Loaded {len(raw_records)} records")

    valid, skip_reasons = [], Counter()
    for r in raw_records:
        ok, reason = is_valid_record(r, min_confidence)
        if ok:
            valid.append(r)
        else:
            skip_reasons[reason] += 1

    print(f"Valid: {len(valid)} | Skipped: {len(raw_records)-len(valid)}")
    for reason, count in skip_reasons.most_common():
        print(f"  skip — {reason}: {count}")

    sampled = stratified_sample(valid, max_per_l2, seed=seed)
    print(f"After stratified sampling: {len(sampled)}")

    random.shuffle(sampled)
    val_size      = max(1, int(len(sampled) * val_ratio))
    train_records = sampled[val_size:]
    val_records   = sampled[:val_size]
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")

    def to_sft(record: dict, split: str) -> dict:
        wid = record.get("weakness_id", "")
        cat = record.get("weakness_category", {})
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                # Turn 1: paper + weakness → claim
                {"role": "user",      "content": format_user_message(record)},
                {"role": "assistant", "content": format_assistant_claim(record)},
                # Turn 2: claim → evidence + suggestions + severity
                {"role": "user",      "content": format_user_turn2(record)},
                {"role": "assistant", "content": format_assistant_suggestions(record)},
            ],
            "meta": {
                "weakness_id":     wid,
                "taxonomy": {
                    "l1_id": cat.get("l1_id", ""),
                    "l2_id": cat.get("l2_id", ""),
                },
                "source_paper_id": _extract_source_paper_id(wid),
                "split":           split,
            },
        }

    train_sft = [to_sft(r, "train") for r in train_records]
    val_sft   = [to_sft(r, "val")   for r in val_records]

    for path, data in [
        (out / "sft_train.jsonl", train_sft),
        (out / "sft_val.jsonl",   val_sft),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    l1_dist, l2_dist = Counter(), Counter()
    has_sections = sum(
        1 for r in sampled
        if r.get("paper_context", {}).get("sections_index")
    )
    for r in sampled:
        cat = r.get("weakness_category", {})
        l1_dist[cat.get("l1_id", "?")] += 1
        l2_dist[cat.get("l2_id", "?")] += 1

    def avg_tokens_by_turn(records_sft: list[dict], role: str, turn: int) -> float:
        """Average token estimate for a given role's N-th occurrence (1-indexed)."""
        lens = []
        for r in records_sft:
            turns = [m["content"] for m in r["messages"] if m["role"] == role]
            if len(turns) >= turn:
                lens.append(len(turns[turn - 1]) // 4)
        return round(sum(lens) / len(lens)) if lens else 0

    stats = {
        "total_raw":     len(raw_records),
        "total_valid":   len(valid),
        "total_sampled": len(sampled),
        "train_size":    len(train_sft),
        "val_size":      len(val_sft),
        "has_sections_index":  has_sections,
        "no_sections_index":   len(sampled) - has_sections,
        "skip_reasons":        dict(skip_reasons),
        "l1_distribution":     dict(l1_dist),
        "l2_distribution":     dict(l2_dist),
        "avg_user_turn1_tokens_est":       avg_tokens_by_turn(train_sft, "user",      1),
        "avg_assistant_claim_tokens_est":  avg_tokens_by_turn(train_sft, "assistant", 1),
        "avg_user_turn2_tokens_est":       avg_tokens_by_turn(train_sft, "user",      2),
        "avg_assistant_sugg_tokens_est":   avg_tokens_by_turn(train_sft, "assistant", 2),
    }
    with open(out / "sft_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"sft_train.jsonl → {len(train_sft)} records")
    print(f"sft_val.jsonl   → {len(val_sft)} records")
    print(f"Records with sections_index: {has_sections}/{len(sampled)} "
          f"({has_sections/max(len(sampled),1)*100:.1f}%)")
    print(f"\nL1 distribution:")
    for k, v in sorted(l1_dist.items()):
        print(f"  {k}: {v} ({v/len(sampled)*100:.1f}%)")
    print(f"\nEst. avg tokens:")
    print(f"  user turn1:       {stats['avg_user_turn1_tokens_est']}")
    print(f"  assistant claim:  {stats['avg_assistant_claim_tokens_est']}")
    print(f"  user turn2:       {stats['avg_user_turn2_tokens_est']}")
    print(f"  assistant sugg:   {stats['avg_assistant_sugg_tokens_est']}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output_dir",     default="Training_Data_Construction/")
    parser.add_argument("--val_ratio",      type=float, default=0.05)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--max_per_l2",     type=int,   default=None)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        min_confidence=args.min_confidence,
        max_per_l2=args.max_per_l2,
        seed=args.seed,
    )
