#!/usr/bin/env python3
"""
End-to-end script for generating enhanced reviews and enforcing strict post-generation validation.

Input JSONL:
  - aligned_weakness_rebuttal_pairs_finegrained.jsonl

Output JSONL:
  - enhanced_reviews.jsonl

This version includes hard constraints:
1) If sections_index is null/empty, evidence MUST NOT contain explicit references
   like Section X / Table X / Fig X / Appendix X.
2) "where" field must not contain vague banned phrases:
   - in the paper
   - relevant section
   - appropriate section
   - somewhere

Also includes:
- rebuttal leakage detection
- template boundary checks (unexpected keys)
- optional post-filter-only mode
"""

import argparse
import json
import os
import re
import threading
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any


# =============================================================================
# Strict validation regex rules
# =============================================================================

SECTION_REF_PATTERN = re.compile(
    r"\b(?:Section|Sec\.?|Table|Tab\.?|Figure|Fig\.?|Appendix|App\.?)\s*[A-Za-z]?\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)

WHERE_BAN_PATTERNS = [
    re.compile(r"\bin the paper\b", re.IGNORECASE),
    re.compile(r"\brelevant section\b", re.IGNORECASE),
    re.compile(r"\bappropriate section\b", re.IGNORECASE),
    re.compile(r"\bsomewhere\b", re.IGNORECASE),
]

REBUTTAL_LEAKAGE_PATTERNS = [
    re.compile(r"\brebuttal\b", re.IGNORECASE),
    re.compile(r"\bauthors?\s+(?:mentioned|showed|added|stated|clarified|revised)\b", re.IGNORECASE),
    re.compile(r"\bin (?:their|the) response\b", re.IGNORECASE),
    re.compile(r"\bas shown in the response\b", re.IGNORECASE),
    re.compile(r"\bauthors?\s+later\b", re.IGNORECASE),
    re.compile(r"\bafter rebuttal\b", re.IGNORECASE),
]

ALLOWED_TOP_KEYS = {"claim", "evidence", "actionable_suggestions", "citations", "severity"}
ALLOWED_SUGGESTION_KEYS = {"what", "where", "how", "expected_outcome", "priority"}


# =============================================================================
# Prompt template
# =============================================================================

REBUTTAL_TO_ACTIONABLE_PROMPT = """\
You are an expert academic reviewer. Your task is to transform a vague paper review weakness into a highly actionable, specific review comment. You will learn from what the authors actually implemented in their rebuttal, but you must write as if you are the ORIGINAL reviewer providing initial guidance.

CRITICAL CONSTRAINT:
NEVER mention, reference, or hint at the existence of the authors' rebuttal.
Write as if this is your INITIAL review — the first time you are providing feedback.
Your output should read as proactive guidance, not reactive commentary.

INPUT:

PAPER CONTEXT:
Title: {paper_title}
Venue: {venue}
Keywords: {keywords}
Abstract: {abstract}
{sections_block}

ORIGINAL WEAKNESS (vague):
{original_weakness}

AUTHORS' REBUTTAL — for your understanding only, DO NOT reference:
{rebuttal_text}
{multi_turn_block}

TASK:
Read the rebuttal to understand what concrete actions would resolve the weakness.
Then write the enhanced review as if YOU thought of these actions first.

Each suggestion MUST specify:
  WHAT  — exact action (add which baseline / ablation / clarification)
  WHERE — exact paper location (Table 3 / Section 4.2 / Appendix B / new subsection)
  HOW   — implementation steps or experimental setup details
  EXPECTED OUTCOME — what the result should demonstrate

OUTPUT FORMAT — return valid JSON only, no markdown fences

{{
  "claim": "One sentence stating the core weakness precisely.",
  "evidence": "1-2 sentences explaining why this is a problem, citing specific paper locations when available, otherwise functional locations.",
  "actionable_suggestions": [
    {{
      "what": "Precise description of what to add/modify/improve",
      "where": "Exact location: 'Table 3', 'Section 4.2', 'Appendix A.1', 'new subsection 4.3', etc.",
      "how": "Step-by-step guidance: datasets, metrics, seeds, hyperparameters, code release details, or writing changes",
      "expected_outcome": "What the result should demonstrate and how it addresses the weakness",
      "priority": "critical | high | medium"
    }}
  ],
  "citations": [],
  "severity": "critical | major | moderate | minor"
}}

RULES:
- Every suggestion must be specific and executable.
- Never reference rebuttal/response explicitly.
- If section names are unknown, use functional locations.
- Do not invent unsupported details.
- Do not output extra top-level keys beyond the required template.

NOW PROCESS THE INPUT ABOVE.
Return JSON only:
"""


# =============================================================================
# Utility functions
# =============================================================================

def extract_venue(_: str) -> str:
    return "ICLR"


def infer_submission_id(pair: Dict[str, Any]) -> Optional[str]:
    sid = pair.get("submission_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()

    weakness_id = pair.get("weakness_id")
    if isinstance(weakness_id, str) and "_Reviewer_" in weakness_id:
        return weakness_id.split("_Reviewer_", 1)[0]

    web_url = (pair.get("paper_context") or {}).get("web_url")
    if isinstance(web_url, str) and web_url.strip():
        try:
            parsed = urlparse(web_url)
            q = parse_qs(parsed.query)
            vals = q.get("id")
            if vals and isinstance(vals[0], str) and vals[0].strip():
                return vals[0].strip()
        except Exception:
            pass
    return None


def get_original_weakness(pair: dict) -> str:
    cw = pair.get("consolidated_weakness") or {}
    if cw.get("initial"):
        return cw["initial"]
    return pair.get("original_weakness", "")


def get_follow_ups(pair: dict) -> list:
    cw = pair.get("consolidated_weakness") or {}
    if isinstance(cw.get("follow_ups"), list):
        return cw["follow_ups"]
    if isinstance(pair.get("follow_ups"), list):
        return pair["follow_ups"]
    return []


def build_rebuttal_block(pair: dict) -> str:
    parts = []

    for reb in pair.get("rebuttals", []):
        if reb and reb.strip() not in ("No Response", ""):
            parts.append(f"[AUTHORS' RESPONSE]\n{reb.strip()}")

    for fu in get_follow_ups(pair):
        text = (fu.get("text") or "").strip()
        if text:
            parts.append(f"[REVIEWER FOLLOW-UP]\n{text}")

    return "\n\n".join(parts) if parts else "No rebuttal available."


def build_sections_hint(pair: dict) -> str:
    sections = pair.get("paper_context", {}).get("sections_index") or []
    if not sections:
        return ""
    lines = []
    for s in sections[:8]:
        name = s.get("name", "Unknown")
        page = s.get("page", "?")
        lines.append(f"- {name!r} (p.{page})")
    return "\n".join(lines)


def is_multi_turn(pair: dict) -> bool:
    return pair.get("metadata", {}).get("num_turns", 1) > 2 and bool(get_follow_ups(pair))


def build_prompt(pair: dict) -> str:
    ctx = pair.get("paper_context", {}) or {}

    paper_title = ctx.get("title", "Unknown Title")
    venue = extract_venue(ctx.get("web_url", ""))
    abstract = ctx.get("abstract", "Not available.")
    keywords_list = ctx.get("keywords") or []
    keywords = ", ".join(keywords_list) if keywords_list else "Not specified"

    sections_hint = build_sections_hint(pair)
    if sections_hint:
        sections_block = f"KNOWN SECTIONS (may be used in evidence):\n{sections_hint}"
    else:
        sections_block = "KNOWN SECTIONS: None available. Do NOT invent section/table/figure numbers."

    rebuttal_text = build_rebuttal_block(pair)
    original_weakness = get_original_weakness(pair)

    multi_turn_block = ""
    if is_multi_turn(pair):
        n = len(get_follow_ups(pair))
        multi_turn_block = (
            f"\nNOTE — MULTI-TURN THREAD: There are {n} reviewer follow-up(s). "
            f"Consolidate all unresolved concerns proactively in initial-review style.\n"
        )

    return REBUTTAL_TO_ACTIONABLE_PROMPT.format(
        paper_title=paper_title,
        venue=venue,
        keywords=keywords,
        abstract=abstract,
        sections_block=sections_block,
        original_weakness=original_weakness,
        rebuttal_text=rebuttal_text,
        multi_turn_block=multi_turn_block,
    )


# =============================================================================
# Validation
# =============================================================================

def validate_enhanced_review(review: Dict[str, Any], pair: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failed: List[str] = []

    # 1) Template boundary check (top-level keys)
    if not isinstance(review, dict):
        return False, ["enhanced_review: not an object"]

    actual_top_keys = set(review.keys())
    extra_top_keys = sorted(actual_top_keys - ALLOWED_TOP_KEYS)
    missing_top_keys = sorted(ALLOWED_TOP_KEYS - actual_top_keys)

    if extra_top_keys:
        failed.append(f"template_out_of_bounds: unexpected top-level keys {extra_top_keys}")
    if missing_top_keys:
        failed.append(f"template_out_of_bounds: missing top-level keys {missing_top_keys}")

    # 2) Basic field checks
    claim = review.get("claim")
    if not isinstance(claim, str) or len(claim.strip()) < 20:
        failed.append("claim: missing or too short")

    evidence = review.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, str):
            failed.append("evidence: must be string or null")
        elif len(evidence.strip()) < 30:
            failed.append("evidence: too short when provided")

    suggestions = review.get("actionable_suggestions")
    if not isinstance(suggestions, list) or len(suggestions) == 0:
        failed.append("actionable_suggestions: empty or invalid")
        suggestions = []

    # 3) Hard rule: if sections_index is null/empty, evidence cannot include Section/Table/Fig refs
    sections_index = (pair.get("paper_context") or {}).get("sections_index")
    sections_missing = not sections_index  # None or []
    if sections_missing and isinstance(evidence, str) and evidence.strip():
        if SECTION_REF_PATTERN.search(evidence):
            failed.append(
                "evidence_hallucination: sections_index is null/empty but evidence contains explicit section/table/figure/appendix reference"
            )

    # 4) Suggestion-level checks + WHERE ban list + suggestion template boundary
    for i, s in enumerate(suggestions):
        tag = f"suggestion[{i}]"
        if not isinstance(s, dict):
            failed.append(f"{tag}: must be an object")
            continue

        s_keys = set(s.keys())
        s_extra_keys = sorted(s_keys - ALLOWED_SUGGESTION_KEYS)
        if s_extra_keys:
            failed.append(f"{tag}: unexpected keys {s_extra_keys}")

        what = s.get("what")
        where = s.get("where")
        how = s.get("how")
        expected_outcome = s.get("expected_outcome")
        priority = s.get("priority")

        if not isinstance(what, str) or len(what.strip()) < 15:
            failed.append(f"{tag}.what: missing or too short")
        if not isinstance(where, str) or len(where.strip()) < 5:
            failed.append(f"{tag}.where: missing or too vague")
        if not isinstance(how, str) or len(how.strip()) < 20:
            failed.append(f"{tag}.how: missing or too short")
        if not isinstance(expected_outcome, str) or len(expected_outcome.strip()) < 15:
            failed.append(f"{tag}.expected_outcome: missing or too short")
        if priority not in ("critical", "high", "medium"):
            failed.append(f"{tag}.priority: invalid '{priority}'")

        # Hard rule: WHERE vague phrase ban
        if isinstance(where, str):
            for pat in WHERE_BAN_PATTERNS:
                if pat.search(where):
                    failed.append(f"{tag}.where_banned_phrase: matched '{pat.pattern}'")
                    break

    # 5) Rebuttal leakage check over whole object
    dumped = json.dumps(review, ensure_ascii=False)
    for pat in REBUTTAL_LEAKAGE_PATTERNS:
        if pat.search(dumped):
            failed.append(f"rebuttal_leakage: matched '{pat.pattern}'")
            break

    # 6) Severity and citations
    severity = review.get("severity")
    if severity not in ("critical", "major", "moderate", "minor"):
        failed.append(f"severity: invalid '{severity}'")

    citations = review.get("citations")
    if not isinstance(citations, list):
        failed.append("citations: must be a list")

    return len(failed) == 0, failed


# =============================================================================
# LLM call
# =============================================================================

def call_llm(
    prompt: str,
    client,
    model: str,
    max_retries: int = 2,
    max_completion_tokens: int = 1500
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic reviewer. Respond with valid JSON only, no markdown fences."
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=max_completion_tokens,
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        except json.JSONDecodeError:
            if attempt == max_retries:
                return None
            continue
        except Exception:
            if attempt == max_retries:
                return None
            continue


# =============================================================================
# Main generation pipeline
# =============================================================================

def _load_checkpoint(output_path: str, failed_path: str) -> set:
    """Load already-processed weakness_ids from output and failed files."""
    done_ids: set = set()
    for path in (output_path, failed_path):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    wid = obj.get("weakness_id")
                    if wid is not None:
                        done_ids.add(wid)
                except json.JSONDecodeError:
                    continue
    return done_ids


def _process_single_pair(
    pair: Dict[str, Any],
    client,
    model: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Process one pair, returning (weakness_id, success_obj_or_None, fail_obj_or_None)."""
    weakness_id = pair.get("weakness_id", "unknown")
    original_weakness = (get_original_weakness(pair) or "").strip()

    # skip invalid weakness text
    if not original_weakness:
        return weakness_id, None, {
            "weakness_id": weakness_id,
            "reason": "invalid_schema_or_empty_weakness",
            "_status": "skipped",
        }

    # skip if no real rebuttal text
    real_rebuttals = [
        r for r in pair.get("rebuttals", [])
        if isinstance(r, str) and r.strip() and r.strip() != "No Response"
    ]
    if not real_rebuttals:
        return weakness_id, None, {
            "weakness_id": weakness_id,
            "reason": "no_real_rebuttal",
            "_status": "skipped",
        }

    prompt = build_prompt(pair)
    enhanced = call_llm(prompt, client=client, model=model)

    if enhanced is None:
        return weakness_id, None, {
            "weakness_id": weakness_id,
            "reason": "llm_failed",
            "_status": "llm_failed",
        }

    ok, issues = validate_enhanced_review(enhanced, pair)
    if not ok:
        return weakness_id, None, {
            "weakness_id": weakness_id,
            "reason": "validation_failed",
            "issues": issues,
            "enhanced_review": enhanced,
            "_status": "validation_failed",
        }

    out_obj = {
        "venue": pair.get("venue"),
        "venue_id": pair.get("venue_id"),
        "year": pair.get("year"),
        "submission_id": infer_submission_id(pair),
        "weakness_id": weakness_id,
        "paper_context": pair.get("paper_context", {}),
        "original_weakness": original_weakness,
        "follow_ups": get_follow_ups(pair),
        "rebuttals": pair.get("rebuttals", []),
        "weakness_category": pair.get("weakness_category"),
        "enhanced_review": enhanced,
        "metadata": pair.get("metadata", {}),
    }
    return weakness_id, out_obj, None


def process_pairs(
    input_path: str,
    output_path: str,
    failed_path: str,
    client,
    model: str,
    max_records: Optional[int] = None,
    num_workers: int = 1,
    checkpoint: bool = True,
) -> Dict[str, Any]:
    from tqdm import tqdm

    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_records is not None:
        records = records[:max_records]

    # --- checkpoint: skip already-processed records ---
    done_ids: set = set()
    if checkpoint:
        done_ids = _load_checkpoint(output_path, failed_path)
        if done_ids:
            before = len(records)
            records = [r for r in records if r.get("weakness_id") not in done_ids]
            print(f"Checkpoint: {len(done_ids)} already done, {before - len(records)} skipped, {len(records)} remaining.")

    stats = {
        "total": len(records) + len(done_ids),
        "already_done": len(done_ids),
        "skipped": 0,
        "success": 0,
        "llm_failed": 0,
        "validation_failed": 0,
    }

    if not records:
        print("All records already processed. Nothing to do.")
        return stats

    print(f"Processing {len(records)} records with {num_workers} worker(s)...")

    # Use append mode so checkpoint data is preserved
    write_lock = threading.Lock()
    file_mode = "a" if checkpoint and done_ids else "w"

    with open(output_path, file_mode, encoding="utf-8") as out_f, \
         open(failed_path, file_mode, encoding="utf-8") as fail_f:

        def _write_result(success_obj, fail_obj):
            with write_lock:
                if success_obj is not None:
                    out_f.write(json.dumps(success_obj, ensure_ascii=False) + "\n")
                    out_f.flush()
                if fail_obj is not None:
                    # Remove internal _status key before writing
                    status = fail_obj.pop("_status", None)
                    fail_f.write(json.dumps(fail_obj, ensure_ascii=False) + "\n")
                    fail_f.flush()
                    # Restore for stats counting
                    if status:
                        fail_obj["_status"] = status

        if num_workers <= 1:
            # Sequential mode
            for pair in tqdm(records, desc="Generating"):
                _, success_obj, fail_obj = _process_single_pair(pair, client, model)
                _write_result(success_obj, fail_obj)
                if success_obj:
                    stats["success"] += 1
                elif fail_obj:
                    s = fail_obj.get("_status", "")
                    if s == "skipped":
                        stats["skipped"] += 1
                    elif s == "llm_failed":
                        stats["llm_failed"] += 1
                    elif s == "validation_failed":
                        stats["validation_failed"] += 1
        else:
            # Parallel mode
            futures = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for pair in records:
                    fut = executor.submit(_process_single_pair, pair, client, model)
                    futures[fut] = pair.get("weakness_id", "unknown")

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                    _, success_obj, fail_obj = fut.result()
                    _write_result(success_obj, fail_obj)
                    if success_obj:
                        stats["success"] += 1
                    elif fail_obj:
                        s = fail_obj.get("_status", "")
                        if s == "skipped":
                            stats["skipped"] += 1
                        elif s == "llm_failed":
                            stats["llm_failed"] += 1
                        elif s == "validation_failed":
                            stats["validation_failed"] += 1

    attempted = stats["total"] - stats["skipped"] - stats["already_done"]
    success_rate = (stats["success"] / attempted * 100.0) if attempted > 0 else 0.0

    print("\nDone.")
    print(f"  Total:             {stats['total']}")
    print(f"  Already done:      {stats['already_done']}")
    print(f"  Skipped:           {stats['skipped']}")
    print(f"  Success:           {stats['success']} ({success_rate:.2f}%)")
    print(f"  LLM failed:        {stats['llm_failed']}")
    print(f"  Validation failed: {stats['validation_failed']}")

    return stats


# =============================================================================
# Post-filter-only mode (for already generated enhanced JSONL)
# =============================================================================

def post_filter_existing(
    input_path: str,
    output_path: str,
    rejected_path: str
) -> Dict[str, Any]:
    total = 0
    passed = 0
    rejected = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(rejected_path, "w", encoding="utf-8") as frej:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except Exception as e:
                rejected += 1
                frej.write(json.dumps({
                    "weakness_id": None,
                    "reason": [f"invalid_json:{str(e)}"],
                    "raw": line[:1000]
                }, ensure_ascii=False) + "\n")
                continue

            enhanced_review = record.get("enhanced_review")
            ok, issues = validate_enhanced_review(enhanced_review, record) if isinstance(enhanced_review, dict) else (False, ["missing_or_invalid_enhanced_review"])

            if ok:
                passed += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                rejected += 1
                frej.write(json.dumps({
                    "weakness_id": record.get("weakness_id"),
                    "reason": issues,
                    "enhanced_review": enhanced_review
                }, ensure_ascii=False) + "\n")

    print("\nPost-filter done.")
    print(f"  Total:    {total}")
    print(f"  Passed:   {passed}")
    print(f"  Rejected: {rejected}")
    if total > 0:
        print(f"  Pass rate: {passed / total * 100:.2f}%")

    return {"total": total, "passed": passed, "rejected": rejected}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--failed", required=True, help="Failed/rejected JSONL path")
    parser.add_argument("--model", default=None, help="Azure deployment name")
    parser.add_argument("--max", type=int, default=None, help="Max number of records to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint/resume, overwrite output files")
    parser.add_argument("--dry-run", action="store_true", help="Print first prompt only")
    parser.add_argument("--post-filter-only", action="store_true", help="Only post-filter an existing enhanced JSONL")

    args = parser.parse_args()

    if args.post_filter_only:
        post_filter_existing(
            input_path=args.input,
            output_path=args.output,
            rejected_path=args.failed
        )
        return

    # generation mode
    if args.dry_run:
        with open(args.input, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            raise SystemExit("Input file is empty.")
        first = json.loads(first_line)
        prompt = build_prompt(first)
        print(prompt)
        print(f"\n[Prompt length: {len(prompt)} chars | approx tokens: {len(prompt)//4}]")
        return

    model_name = (
        args.model
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL")
    )
    if not model_name:
        raise SystemExit(
            "Model deployment is required. Pass --model or set "
            "AZURE_OPENAI_DEPLOYMENT / OPENAI_MODEL / MODEL."
        )

    from openai import AzureOpenAI

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not api_key:
        raise SystemExit("Missing AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY).")
    if not endpoint:
        raise SystemExit("Missing AZURE_OPENAI_ENDPOINT.")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    process_pairs(
        input_path=args.input,
        output_path=args.output,
        failed_path=args.failed,
        client=client,
        model=model_name,
        max_records=args.max,
        num_workers=args.workers,
        checkpoint=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
