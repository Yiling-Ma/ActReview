# -*- coding: utf-8 -*-
"""
Generate model outputs on test grouped data for both tasks:
  task1: [TASK 1] paper + label + retrieved snippets -> claims list
  task2: [TASK 2] paper + label + GT claim + retrieved snippets -> evidence + suggestions + severity

Task2 always uses ground-truth claims
"""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH = "checkpoints/enhanced_2_tasks/best_model"
TEST_JSONL = "test.jsonl"
OUTPUT_FILE = "test_outputs_task1.jsonl"
MAX_CLAIMS_PER_GROUP = 3

SYSTEM_PROMPT = """You are an expert academic peer reviewer specializing in machine learning and AI research.
You operate in two modes depending on the task header in the user message.

─────────────────────────────────────────────────────────
[TASK 1] — Weakness Claim Discovery
─────────────────────────────────────────────────────────
Given a paper and a weakness category label, identify all distinct concrete weaknesses
that this paper exhibits under that label.
For each weakness found, write exactly ONE precise single sentence as a clear, testable
diagnostic claim. Number the claims: Claim 1: ... Claim 2: ...
If the paper has no weakness under this label, output exactly: None

─────────────────────────────────────────────────────────
[TASK 2] — Actionable Analysis
─────────────────────────────────────────────────────────
Given a paper and a single weakness claim, produce structured evidence and actionable
suggestions following the template below exactly.

OUTPUT TEMPLATE
─────────────────────────────────────────────────────────

## Claim
(exact claim text)

## Evidence
1-2 sentences citing specific paper locations where the issue manifests.

### Suggestion 1
- **What**: Exact action required
- **Where**: Precise location — section heading, table name, appendix label, or
  functional description if exact name is unknown
- **How**: Step-by-step implementation details
- **Expected Outcome**: What the result should demonstrate and how it resolves the weakness
- **Priority**: critical | high | medium

### Suggestion 2
(repeat as needed)

## Severity
critical | major | moderate | minor

─────────────────────────────────────────────────────────
REQUIREMENTS (both tasks)
─────────────────────────────────────────────────────────
- Task 1: each claim must be a single sentence, specific and testable, no duplicates
- Task 2: WHERE must name an actual location — never write "in the paper" or "somewhere"
- Task 2: HOW must include concrete steps a researcher can follow without guessing
- NEVER reference the authors' rebuttal or any post-submission response
- Do NOT add sections not in the template"""


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

_REBUTTAL_PHRASES = [
    "as the authors",
    "in their response",
    "the rebuttal",
    "authors mentioned",
    "authors showed",
    "authors added",
    "in response to",
]

_BARE_SECTION_RE = re.compile(
    r"\b(?:Section|Sec\.?|Appendix|App\.?)\s+([A-Z]?\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
_BARE_TABLEFIG_RE = re.compile(
    r"\b(?:Table|Figure|Fig\.?)\s+(\d+[a-zA-Z]?)\b",
    re.IGNORECASE,
)
_QUOTED_SECTION_RE = re.compile(
    r"""(\bthe\s+)?(?:(?:[Ss]ection|[Ss]ec\.?)\s+)(['"])(.*?)\2(\s*\(p(?:age)?\.?\s*\d+\))?""",
    re.VERBOSE,
)
_TRAILING_STOPWORDS = {
    "the", "a", "an", "of", "in", "is", "to", "for", "and", "or",
    "that", "with", "on", "by", "from", "as", "are", "was", "be"
}

_CLAIM_STOPWORDS = {
    "the", "a", "an", "of", "in", "is", "to", "for", "and", "or",
    "that", "with", "on", "by", "from", "as", "are", "was", "be",
    "this", "paper", "work", "method", "approach", "model", "it", "its",
    "their", "there", "has", "have", "had", "at", "under", "into", "than",
    "then", "very", "more", "less", "not",
}

_SPECIFICITY_ANCHORS = {
    "ablation", "baseline", "table", "figure", "fig", "section", "appendix",
    "dataset", "benchmark", "metric", "error", "failure", "case",
    "counterexample", "latency", "throughput", "runtime", "complexity",
    "memory", "robustness", "ood", "hyperparameter", "seed", "variance",
    "significance", "confidence", "calibration", "leakage", "prompt",
    "instruction", "retrieval", "evidence", "citation", "hallucination",
    "equation", "formula", "algorithm", "procedure", "definition",
}


def _has_rebuttal_leakage(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _REBUTTAL_PHRASES)


def _is_noisy_section_title(title: str) -> bool:
    words = title.split()
    if not words:
        return True
    if len(words) > 8:
        return True
    if words[-1].strip(".,;:\"'").lower() in _TRAILING_STOPWORDS:
        return True
    stripped = title.rstrip()
    if stripped and stripped[-1] in ("(", '"', "'", "."):
        return True
    first_alpha = next((c for c in title if c.isalpha()), None)
    if first_alpha and first_alpha.islower():
        return True
    return False


def _replace_noisy_quoted_section(m: re.Match) -> str:
    title = m.group(3)
    page_ref = (m.group(4) or "").strip()
    if not _is_noisy_section_title(title):
        return m.group(0)
    return f"the relevant section {page_ref}" if page_ref else "the relevant section"


def sanitize_location_refs(text: str, sections_index) -> str:
    if not isinstance(text, str):
        return text
    text = _QUOTED_SECTION_RE.sub(_replace_noisy_quoted_section, text)
    if not sections_index:
        def replace_section(m: re.Match) -> str:
            prefix = m.group(0).split()[0].rstrip(".").lower()
            return "the appendix" if ("appendix" in prefix or prefix == "app") else "the relevant section"

        def replace_tablefig(m: re.Match) -> str:
            prefix = m.group(0).split()[0].rstrip(".").lower()
            return "the results table" if "table" in prefix else "the figure"

        text = _BARE_SECTION_RE.sub(replace_section, text)
        text = _BARE_TABLEFIG_RE.sub(replace_tablefig, text)
    return text


def normalize_claim(text: str) -> str:
    """Normalize claim for comparison"""
    text = text.strip().lower()
    text = re.sub(r"^claim\s+\d+\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in _CLAIM_STOPWORDS]
    return " ".join(tokens)


def _get_core_concepts(claim: str) -> set[str]:
    """Extract core concepts (length >= 4)"""
    norm = normalize_claim(claim)
    tokens = set(norm.split())
    return {t for t in tokens if len(t) >= 4}


def _is_near_duplicate_claim(text_a: str, text_b: str) -> bool:
    """Check if two claims are near-duplicates (60% threshold)"""
    core_a = _get_core_concepts(text_a)
    core_b = _get_core_concepts(text_b)
    
    if not core_a or not core_b:
        return False
    
    overlap = len(core_a & core_b)
    containment = max(
        overlap / len(core_a) if core_a else 0,
        overlap / len(core_b) if core_b else 0
    )
    
    return containment >= 0.60


def _compute_claim_similarity(claim1: str, claim2: str) -> float:
    """
    Compute similarity score between two claims
    Returns value in [0, 1]
    """
    core1 = _get_core_concepts(claim1)
    core2 = _get_core_concepts(claim2)
    
    if not core1 or not core2:
        return 0.0
    
    overlap = len(core1 & core2)
    union = len(core1 | core2)
    
    # Jaccard similarity
    return overlap / union if union > 0 else 0.0


def _specificity_score(text: str) -> int:
    """Calculate specificity score for a claim"""
    lower = text.strip().lower()
    score = 0
    
    if re.search(r"\d", lower):
        score += 2
    
    if re.search(r"\b(section|sec\.|table|figure|fig\.|appendix|app\.)\b", lower):
        score += 2
    
    score += sum(1 for word in _SPECIFICITY_ANCHORS if re.search(rf"\b{re.escape(word)}\b", lower))
    
    if re.search(r"\b(because|therefore|which|leading to|causing|so that)\b", lower):
        score += 1
    
    score += min(len(lower.split()) // 10, 3)
    
    return score


def _clean_single_sentence_claim(text: str) -> str:
    """Clean and extract single sentence from claim"""
    text = re.sub(r"^\s*(?:[-*]\s*)?", "", text.strip())
    text = re.sub(r"^Claim\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip()


def _aggressive_dedup_claims(claims: list[str]) -> list[str]:
    """
    Group claims by semantic similarity and pick most specific from each group
    Uses 50% containment threshold
    """
    if len(claims) <= 1:
        return claims
    
    groups = []
    for claim in claims:
        core = _get_core_concepts(claim)
        
        merged = False
        for group in groups:
            group_core = _get_core_concepts(group[0])
            
            if not core or not group_core:
                continue
            
            overlap = len(core & group_core)
            containment = max(
                overlap / len(core) if core else 0,
                overlap / len(group_core) if group_core else 0
            )
            
            if containment >= 0.50:
                group.append(claim)
                merged = True
                break
        
        if not merged:
            groups.append([claim])
    
    result = []
    for group in groups:
        best_claim = max(group, key=lambda c: _specificity_score(c))
        result.append(best_claim)
    
    return result


def _force_merge_similar_claims(claims: list[str], max_output: int = 2) -> list[str]:
    """Force merge similar claims if we have too many"""
    if len(claims) <= max_output:
        return claims
    
    pairs = []
    for i in range(len(claims)):
        for j in range(i+1, len(claims)):
            core_i = _get_core_concepts(claims[i])
            core_j = _get_core_concepts(claims[j])
            
            if not core_i or not core_j:
                continue
            
            overlap = len(core_i & core_j)
            similarity = overlap / min(len(core_i), len(core_j))
            
            pairs.append((i, j, similarity))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    merged_claims = list(claims)
    merged_indices = set()
    
    for i, j, sim in pairs:
        if i in merged_indices or j in merged_indices:
            continue
        
        if len(merged_claims) - len(merged_indices) <= max_output:
            break
        
        if _specificity_score(claims[i]) >= _specificity_score(claims[j]):
            merged_indices.add(j)
        else:
            merged_indices.add(i)
    
    result = [c for idx, c in enumerate(merged_claims) if idx not in merged_indices]
    return result[:max_output]


def _refine_task1_claim_lines(claim_lines: list[str]) -> list[str]:
    """Refine and deduplicate task1 claim lines"""
    cleaned_claims = []
    for line in claim_lines:
        match = re.match(r"^Claim\s+\d+\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
        if not match:
            continue
        claim = _clean_single_sentence_claim(match.group(1))
        if claim:
            cleaned_claims.append(claim)

    if not cleaned_claims:
        return []

    ranked_claims = sorted(cleaned_claims, key=lambda c: _specificity_score(c), reverse=True)
    deduped_claims = _aggressive_dedup_claims(ranked_claims)
    final_claims = _force_merge_similar_claims(deduped_claims, max_output=2)
    
    return [f"Claim {idx}: {claim}" for idx, claim in enumerate(final_claims, 1)]


def get_deduped_claims_and_items(group: dict):
    """Extract deduplicated claims from a grouped record"""
    items = group.get("weakness_items") or []
    seen = set()
    claims = []
    claim_items = []
    for item in items:
        er = item.get("enhanced_review") or {}
        claim = (er.get("claim") or "").strip()
        if not claim:
            continue
        norm = normalize_claim(claim)
        if norm in seen:
            continue
        seen.add(norm)
        claims.append(claim)
        claim_items.append(item)
    return claims, claim_items


def is_valid_item(item: dict, sections_index) -> bool:
    er = item.get("enhanced_review")
    if not er:
        return False
    if not (er.get("claim") or "").strip():
        return False
    if _has_rebuttal_leakage(json.dumps(er, ensure_ascii=False)):
        return False

    suggestions = er.get("actionable_suggestions") or []
    if not suggestions:
        return False

    for s in suggestions:
        for field in ("what", "where", "how", "expected_outcome", "priority"):
            if not (s.get(field) or "").strip():
                return False
    return True


def is_valid_group(group: dict) -> bool:
    if not group.get("paper_id"):
        return False
    if not group.get("l2_id"):
        return False
    if not (group.get("paper_context") or {}).get("abstract", "").strip():
        return False

    sections_index = (group.get("paper_context") or {}).get("sections_index")
    valid_items = [item for item in (group.get("weakness_items") or []) if is_valid_item(item, sections_index)]
    if not valid_items:
        return False

    group["weakness_items"] = valid_items
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════

def _paper_metadata_block(group: dict) -> str:
    """Shared paper metadata formatting"""
    ctx = group.get("paper_context") or {}
    title = ctx.get("title", "Unknown Title")
    abstract = ctx.get("abstract", "Not available.")
    keywords = ", ".join(ctx.get("keywords") or []) or "Not specified"
    l2_id = group.get("l2_id", "")
    l2_name = group.get("l2_name", "")
    label = f"{l2_id} — {l2_name}" if l2_id else l2_name

    return (
        f"**Paper**: {title}\n\n"
        f"**Keywords**: {keywords}\n\n"
        f"**Abstract**:\n{abstract}\n\n"
        f"**Weakness label**: {label}"
    )


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _format_snippets_block(snippets: list[dict], max_snippets: int) -> str:
    if not snippets or max_snippets <= 0:
        return ""
    lines: list[str] = ["**Retrieved snippets**:"]
    count = 0
    for sn in snippets:
        if count >= max_snippets:
            break
        text = (sn.get("text") or "").strip()
        if not text:
            continue
        text = text[:1200]
        page = sn.get("page")
        score = sn.get("score")
        meta = []
        if page is not None:
            meta.append(f"p.{page}")
        if score is not None:
            try:
                meta.append(f"score={float(score):.4f}")
            except Exception:
                pass
        meta_str = f" ({', '.join(meta)})" if meta else ""
        lines.append(f"[Snippet {count+1}]{meta_str}\n{text}")
        count += 1
    if count == 0:
        return ""
    return "\n\n".join(lines).strip()


def _collect_group_snippets(group: dict, max_snippets: int) -> list[dict]:
    """Collect task1 snippets across all weakness_items in a group, ranked by score."""
    if max_snippets <= 0:
        return []
    items = group.get("weakness_items") or []
    all_snips: list[dict] = []
    for it in items:
        # Prefer task1-specific snippets, fall back to generic aligned_snippets
        snips = it.get("aligned_snippets_task1") or it.get("aligned_snippets") or []
        if isinstance(snips, list):
            all_snips.extend([s for s in snips if isinstance(s, dict)])
    if not all_snips:
        return []
    all_snips.sort(key=lambda s: float(s.get("score", 0.0) or 0.0), reverse=True)
    selected: list[dict] = []
    seen: set[str] = set()
    for s in all_snips:
        key = _normalize_ws((s.get("text") or "")[:300])
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(s)
        if len(selected) >= max_snippets:
            break
    return selected


MAX_SNIPPETS = 3


def format_task1_user(group: dict) -> str:
    base = "[TASK 1] Weakness Claim Discovery\n\n" + _paper_metadata_block(group)
    snippets = _collect_group_snippets(group, MAX_SNIPPETS)
    snippets_block = _format_snippets_block(snippets, MAX_SNIPPETS)
    if snippets_block:
        base += "\n\n" + snippets_block
    return (
        base
        + "\n\nIdentify all concrete weaknesses in this paper under the label above and "
        "state each as a single-sentence diagnostic claim. Prefer the most specific reviewer "
        "concerns tied to a concrete experiment, metric, dataset, component, or paper location, "
        "and avoid repeating the same weakness in paraphrased form."
        + "\n\n**Important**: Generate 1-2 distinct claims if the paper has multiple "
        "independent weaknesses under this label. If multiple potential claims address "
        "similar underlying issues, consolidate them into one precise claim rather than "
        "listing variations."
    )


def format_task1_assistant_from_claims(claims: list[str]) -> str:
    if not claims:
        return "None"
    return "\n".join(f"Claim {i}: {c}" for i, c in enumerate(claims, 1))


def format_task2_user(group: dict, claim_text: str, item: dict | None = None) -> str:
    base = "[TASK 2] Actionable Analysis\n\n" + _paper_metadata_block(group)
    if item is not None:
        snips = item.get("aligned_snippets_task2") or item.get("aligned_snippets") or []
        if isinstance(snips, list) and snips:
            snippets_block = _format_snippets_block(snips, MAX_SNIPPETS)
            if snippets_block:
                base += "\n\n" + snippets_block
    return (
        base
        + f"\n\n**Claim**: {claim_text}\n\n"
        "Provide evidence, actionable suggestions, and severity for the claim above "
        "following the template exactly."
    )


def format_task2_assistant(group: dict, claim_text: str, item: dict) -> str:
    er = item.get("enhanced_review") or {}
    sections_index = (group.get("paper_context") or {}).get("sections_index")

    evidence = (er.get("evidence") or "").strip()
    if not evidence:
        evidence = "Specific paper location not identified from available context."
    evidence = sanitize_location_refs(evidence, sections_index)

    lines = [
        "## Claim",
        claim_text,
        "",
        "## Evidence",
        evidence,
        "",
    ]

    for i, s in enumerate((er.get("actionable_suggestions") or []), 1):
        what = (s.get("what") or "").strip()
        where = sanitize_location_refs((s.get("where") or "").strip(), sections_index)
        how = (s.get("how") or "").strip()
        outcome = (s.get("expected_outcome") or "").strip()
        priority = (s.get("priority") or "medium").strip()

        lines.extend([
            f"### Suggestion {i}",
            f"- **What**: {what}",
            f"- **Where**: {where}",
            f"- **How**: {how}",
            f"- **Expected Outcome**: {outcome}",
            f"- **Priority**: {priority}",
            "",
        ])

    severity = (er.get("severity") or "major").strip()
    lines.extend([
        "## Severity",
        severity,
    ])

    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Stopping criteria
# ═══════════════════════════════════════════════════════════════════════════

_TASK1_STOP_PATTERNS = [
    r"(?:\n|\A)\s*(?:user|assistant|system)\s*(?:\n|$)",
    r"\[\s*TASK\s*[12]\s*\]",
    r"\n\s*Claim\s+\d+\s*:\s*.+\n\s*\n",
]

_TASK2_STOP_PATTERNS = [
    r"##\s*Severity\s*\n\s*(critical|major|moderate|minor)\s*(?:\n|$)",
    r"(?:\n|\A)\s*(?:user|assistant|system)\s*(?:\n|$)",
    r"\[\s*TASK\s*[12]\s*\]",
]


class RegexStopping(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, patterns: list[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.pattern = re.compile("|".join(f"({p})" for p in patterns), re.IGNORECASE)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self.pattern.search(text) is not None


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════════════════

def extract_task1_output(text: str) -> str:
    """Extract and clean task1 output"""
    text = text.strip()

    markers = []
    for pat in _TASK1_STOP_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            markers.append(m.start())
    if markers:
        text = text[:min(markers)].strip()

    if text.lower() == "none":
        return "None"

    claim_lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^Claim\s+\d+\s*:", line, re.IGNORECASE):
            claim_lines.append(line)
        elif claim_lines:
            break

    if claim_lines:
        refined = _refine_task1_claim_lines(claim_lines)
        if refined:
            return "\n".join(refined).strip()
        return "\n".join(claim_lines).strip()

    return text.strip()


def parse_task1_claims(text: str) -> list[str]:
    """
    Parse Task1 output and return list of claims
    Returns empty list if no claims found or output is "None"
    """
    text = text.strip()
    
    if text.lower() == "none":
        return []
    
    claims = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^Claim\s+\d+\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            claim_text = match.group(1).strip()
            if claim_text:
                claims.append(claim_text)
    
    return claims


def extract_task2_output(text: str) -> str:
    """Extract and clean task2 output"""
    match = re.search(
        r"##\s*Severity\s*\n\s*(critical|major|moderate|minor)\s*(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return text[:match.end()].strip()

    markers = []
    for pat in _TASK2_STOP_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            markers.append(m.start())
    if markers:
        text = text[:min(markers)]

    return text.strip()


def check_completeness_task1(text: str):
    """Validate task1 output completeness"""
    text = text.strip()
    if not text:
        return False, "Empty output"

    if text.lower() == "none":
        return True, "Complete (no weaknesses)"

    if re.search(r"(?:^|\n)\s*(?:user|assistant|system)\s*(?:\n|$)", text, re.IGNORECASE):
        return False, "Contains stray role marker"
    if re.search(r"\[\s*TASK\s*[12]\s*\]", text, re.IGNORECASE):
        return False, "Contains stray task header"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, "Empty output"

    claim_lines = [ln for ln in lines if re.match(r"^Claim\s+\d+\s*:", ln, re.IGNORECASE)]
    if not claim_lines:
        return False, "Missing numbered claims"

    if len(claim_lines) != len(lines):
        return False, "Contains non-claim trailing content"

    expected_num = 1
    seen_norm = set()
    accepted_claims = []
    
    for ln in claim_lines:
        m = re.match(r"^Claim\s+(\d+)\s*:\s*(.+)$", ln, re.IGNORECASE)
        if not m:
            return False, "Malformed claim line"
        num = int(m.group(1))
        claim_text = m.group(2).strip()
        if num != expected_num:
            return False, f"Claim numbering not sequential at Claim {num}"
        if not claim_text:
            return False, f"Empty text for Claim {num}"
        
        norm = normalize_claim(claim_text)
        if norm in seen_norm:
            return False, f"Duplicate claim detected at Claim {num}"
        if any(_is_near_duplicate_claim(claim_text, prev) for prev in accepted_claims):
            return False, f"Near-duplicate claim detected at Claim {num}"
        
        seen_norm.add(norm)
        accepted_claims.append(claim_text)
        expected_num += 1

    return True, "Complete"


def check_completeness_task2(review: str):
    """Validate task2 output completeness"""
    required_sections = [
        (r"##\s*Claim", "## Claim"),
        (r"##\s*Evidence", "## Evidence"),
        (r"###\s*Suggestion\s+\d+", "### Suggestion"),
        (r"##\s*Severity", "## Severity"),
    ]
    for pattern, name in required_sections:
        if not re.search(pattern, review, re.IGNORECASE):
            return False, f"Missing section: {name}"

    if re.search(r"(?:^|\n)\s*(?:user|assistant|system)\s*(?:\n|$)", review, re.IGNORECASE):
        return False, "Contains stray role marker"
    if re.search(r"\[\s*TASK\s*[12]\s*\]", review, re.IGNORECASE):
        return False, "Contains stray task header"

    suggestions = re.findall(
        r"###\s*Suggestion\s+\d+.*?(?=###\s*Suggestion|##\s*Severity|$)",
        review,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not suggestions:
        return False, "No suggestions found"

    for idx, suggestion in enumerate(suggestions, 1):
        for field in ("What", "Where", "How", "Expected Outcome", "Priority"):
            pat = rf"\*\*{re.escape(field)}\*\*\s*[:：]|{re.escape(field)}\s*[:：]"
            if not re.search(pat, suggestion, re.IGNORECASE):
                return False, f"Suggestion {idx} missing field: {field}"

    if not re.search(
        r"##\s*Severity\s*\n\s*(critical|major|moderate|minor)\b", review, re.IGNORECASE
    ):
        return False, "Severity section missing a valid value"

    return True, "Complete"


# ═══════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════

def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_one(model, tokenizer, messages: list[dict], task: str) -> str:
    prompt_text = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if task == "task1":
        stopping = StoppingCriteriaList([RegexStopping(tokenizer, input_len, _TASK1_STOP_PATTERNS)])
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=6,
            repetition_penalty=1.15,
            length_penalty=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    else:
        stopping = StoppingCriteriaList([RegexStopping(tokenizer, input_len, _TASK2_STOP_PATTERNS)])
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=1536,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline mode helpers
# ═══════════════════════════════════════════════════════════════════════════

def match_generated_claims_to_gt_items(
    generated_claims: list[str],
    gt_claims: list[str],
    gt_items: list[dict]
) -> list[dict]:
    """
    Match generated claims to GT items using semantic similarity
    
    For each generated claim, find the most similar GT claim and use its item.
    If no good match, use the first GT item as fallback.
    
    Returns:
        List of items corresponding to generated_claims
    """
    if not generated_claims:
        return []
    
    if not gt_claims or not gt_items:
        # No GT available, return None for each generated claim
        return [None] * len(generated_claims)
    
    matched_items = []
    
    for gen_claim in generated_claims:
        best_match_idx = 0
        best_similarity = 0.0
        
        # Find most similar GT claim
        for idx, gt_claim in enumerate(gt_claims):
            similarity = _compute_claim_similarity(gen_claim, gt_claim)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx
        
        # Use matched item (or first item as fallback)
        matched_item = gt_items[best_match_idx] if best_match_idx < len(gt_items) else gt_items[0]
        matched_items.append(matched_item)
    
    return matched_items


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument(
        "--input_path",
        default=None,
        help="Path to input test JSONL (preferred; replaces --test_jsonl)",
    )
    parser.add_argument(
        "--test_jsonl",
        default=None,
        help="Deprecated; use --input_path instead",
    )
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--max_claims", type=int, default=MAX_CLAIMS_PER_GROUP)
    parser.add_argument("--n_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument(
        "--task",
        choices=["task1", "task2", "both"],
        default="both",
        help="Which task to run: task1 only, task2 only (uses GT claims), or both (default)",
    )
    args = parser.parse_args()

    input_path = args.input_path or args.test_jsonl or TEST_JSONL

    print("=" * 80)
    print("Task2 always uses GT claims (oracle mode).")
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
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_groups.append(json.loads(line))
    print(f"Loaded {len(raw_groups)} groups from {input_path}")

    valid_groups = [g for g in raw_groups if is_valid_group(g)]
    print(f"Valid groups: {len(valid_groups)} | Skipped: {len(raw_groups) - len(valid_groups)}")

    if args.n_samples:
        valid_groups = valid_groups[:args.n_samples]
        print(f"Limited to {len(valid_groups)} groups for testing.")

    run_task1 = args.task in ("task1", "both")
    run_task2 = args.task in ("task2", "both")

    print("\n" + "=" * 80)
    print(f"STEP: Generating outputs for --task={args.task} (Task2 always uses GT claims)")
    print("=" * 80)
    
    results = []
    stats = {
        "task1": {"complete": 0, "incomplete": 0},
        "task2": {"complete": 0, "incomplete": 0},
    }
    
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for group in tqdm(valid_groups, desc="Generation by paper"):
            paper_id = group["paper_id"]
            ctx = group.get("paper_context") or {}
            
            # ───────────────────────────────────────────────────────────────
            # Prepare GT claims/items for this group
            # ───────────────────────────────────────────────────────────────
            gt_claims, gt_items = get_deduped_claims_and_items(group)
            if args.max_claims:
                gt_claims = gt_claims[:args.max_claims]
                gt_items = gt_items[:args.max_claims]
            
            # ───────────────────────────────────────────────────────────────
            # Task1: generate claims
            # ───────────────────────────────────────────────────────────────
            if run_task1:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_task1_user(group)},
                ]
                task1_generated = generate_one(model, tokenizer, messages, "task1")
                task1_generated = extract_task1_output(task1_generated)
                task1_complete, task1_msg = check_completeness_task1(task1_generated)

                if not task1_complete:
                    print(f"\nIncomplete Task1 | paper_id={paper_id} | {task1_msg}")

                task1_result = {
                    "paper_id": paper_id,
                    "l1_id": group.get("l1_id", ""),
                    "l2_id": group.get("l2_id", ""),
                    "l2_name": group.get("l2_name", ""),
                    "task": "task1",
                    "input": {
                        "title": ctx.get("title", ""),
                        "abstract": ctx.get("abstract", ""),
                        "user_message": format_task1_user(group),
                    },
                    "ground_truth": format_task1_assistant_from_claims(gt_claims),
                    "generated": task1_generated,
                    "is_complete": task1_complete,
                    "completeness_msg": task1_msg,
                }
                results.append(task1_result)
                out_f.write(json.dumps(task1_result, ensure_ascii=False) + "\n")
                out_f.flush()
                stats["task1"]["complete" if task1_complete else "incomplete"] += 1
            
            # ───────────────────────────────────────────────────────────────
            # Task2: always use GT claims
            # ───────────────────────────────────────────────────────────────
            if run_task2:
                for claim_idx, (claim_text, item) in enumerate(zip(gt_claims, gt_items)):
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": format_task2_user(group, claim_text, item)},
                    ]
                    
                    task2_generated = generate_one(model, tokenizer, messages, "task2")
                    task2_generated = extract_task2_output(task2_generated)
                    task2_complete, task2_msg = check_completeness_task2(task2_generated)
                    stats["task2"]["complete" if task2_complete else "incomplete"] += 1
                    
                    if item is not None:
                        gt_text = format_task2_assistant(group, claim_text, item)
                    else:
                        gt_text = "[No matching GT item available]"
                    
                    task2_result = {
                        "paper_id": paper_id,
                        "l1_id": group.get("l1_id", ""),
                        "l2_id": group.get("l2_id", ""),
                        "l2_name": group.get("l2_name", ""),
                        "task": "task2",
                        "input": {
                            "title": ctx.get("title", ""),
                            "abstract": ctx.get("abstract", ""),
                            "user_message": format_task2_user(group, claim_text),
                        },
                        "claim": claim_text,
                        "ground_truth": gt_text,
                        "generated": task2_generated,
                        "is_complete": task2_complete,
                        "completeness_msg": task2_msg,
                        "input_mode": "gt",
                    }
                    results.append(task2_result)
                    out_f.write(json.dumps(task2_result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    
                    if not task2_complete:
                        print(f"\nIncomplete Task2 | paper_id={paper_id} | claim #{claim_idx+1} | {task2_msg}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Mode: --task={args.task} | Task2 uses ground-truth claims")
    print(f"Total samples evaluated: {len(results)}")
    
    for task in ("task1", "task2"):
        c = stats[task]["complete"]
        inc = stats[task]["incomplete"]
        total = c + inc
        if total:
            print(f"  {task}: {total} samples | complete: {c} ({c/total*100:.1f}%) | incomplete: {inc}")
    
    print(f"\nResults saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()