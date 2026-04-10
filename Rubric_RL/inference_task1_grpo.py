# -*- coding: utf-8 -*-
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

# ============================================================
# Default paths
# ============================================================

MAX_SNIPPETS = 3

SYSTEM_PROMPT = """You are an expert academic peer reviewer specializing in machine learning and AI research.

[TASK 1] — Weakness Claim Discovery

Given a paper and a weakness category label, identify all distinct concrete weaknesses
that this paper exhibits under that label.

For each weakness found, write exactly ONE precise single sentence as a clear, testable
diagnostic claim.

Output format:
Claim 1: ...
Claim 2: ...

If the paper has no weakness under this label, output exactly:
None

Requirements:
- each claim must be a single sentence
- claims must be specific and testable
- no duplicate or near-duplicate claims
- never reference the authors' rebuttal or any post-submission response
"""

# ============================================================
# Validation helpers
# ============================================================

_REBUTTAL_PHRASES = [
    "as the authors",
    "in their response",
    "the rebuttal",
    "authors mentioned",
    "authors showed",
    "authors added",
    "in response to",
]

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


def normalize_claim(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^claim\s+\d+\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in _CLAIM_STOPWORDS]
    return " ".join(tokens)


def _get_core_concepts(claim: str) -> set[str]:
    norm = normalize_claim(claim)
    tokens = set(norm.split())
    return {t for t in tokens if len(t) >= 4}


def _compute_claim_similarity(claim1: str, claim2: str) -> float:
    core1 = _get_core_concepts(claim1)
    core2 = _get_core_concepts(claim2)
    if not core1 or not core2:
        return 0.0
    overlap = len(core1 & core2)
    union = len(core1 | core2)
    return overlap / union if union > 0 else 0.0


def _is_near_duplicate_claim(text_a: str, text_b: str) -> bool:
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


def _specificity_score(text: str) -> int:
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
    text = re.sub(r"^\s*(?:[-*]\s*)?", "", text.strip())
    text = re.sub(r"^Claim\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip()


def _aggressive_dedup_claims(claims: list[str]) -> list[str]:
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
    if len(claims) <= max_output:
        return claims

    pairs = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            core_i = _get_core_concepts(claims[i])
            core_j = _get_core_concepts(claims[j])
            if not core_i or not core_j:
                continue
            overlap = len(core_i & core_j)
            similarity = overlap / min(len(core_i), len(core_j))
            pairs.append((i, j, similarity))

    pairs.sort(key=lambda x: x[2], reverse=True)

    merged_indices = set()
    for i, j, sim in pairs:
        if i in merged_indices or j in merged_indices:
            continue
        if len(claims) - len(merged_indices) <= max_output:
            break
        if _specificity_score(claims[i]) >= _specificity_score(claims[j]):
            merged_indices.add(j)
        else:
            merged_indices.add(i)

    return [c for idx, c in enumerate(claims) if idx not in merged_indices][:max_output]


def _refine_task1_claim_lines(claim_lines: list[str]) -> list[str]:
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


def parse_task1_claims(text: str) -> list[str]:
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


def format_task1_assistant_from_claims(claims: list[str]) -> str:
    if not claims:
        return "None"
    return "\n".join(f"Claim {i}: {c}" for i, c in enumerate(claims, 1))


# ============================================================
# Dataset helpers
# ============================================================

def is_valid_item(item: dict) -> bool:
    er = item.get("enhanced_review")
    if not er:
        return False
    if not (er.get("claim") or "").strip():
        return False
    if _has_rebuttal_leakage(json.dumps(er, ensure_ascii=False)):
        return False
    return True


def is_valid_group(group: dict) -> bool:
    if not group.get("paper_id"):
        return False
    if not group.get("l2_id"):
        return False
    if not (group.get("paper_context") or {}).get("abstract", "").strip():
        return False

    valid_items = [item for item in (group.get("weakness_items") or []) if is_valid_item(item)]
    if not valid_items:
        return False

    group["weakness_items"] = valid_items
    return True


def get_deduped_claims(group: dict):
    items = group.get("weakness_items") or []
    seen = set()
    claims = []
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
    return claims


def _paper_metadata_block(group: dict) -> str:
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
    lines = ["**Retrieved snippets**:"]
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
    if max_snippets <= 0:
        return []
    items = group.get("weakness_items") or []
    all_snips = []
    for it in items:
        snips = it.get("aligned_snippets_task1") or it.get("aligned_snippets") or []
        if isinstance(snips, list):
            all_snips.extend([s for s in snips if isinstance(s, dict)])
    if not all_snips:
        return []
    all_snips.sort(key=lambda s: float(s.get("score", 0.0) or 0.0), reverse=True)

    selected = []
    seen = set()
    for s in all_snips:
        key = _normalize_ws((s.get("text") or "")[:300])
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(s)
        if len(selected) >= max_snippets:
            break
    return selected


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


# ============================================================
# Generation
# ============================================================

_TASK1_STOP_PATTERNS = [
    r"(?:\n|\A)\s*(?:user|assistant|system)\s*(?:\n|$)",
    r"\[\s*TASK\s*[12]\s*\]",
    r"\n\s*Claim\s+\d+\s*:\s*.+\n\s*\n",
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


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_task1_output(text: str) -> str:
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


def check_completeness_task1(text: str):
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


def generate_one(model, tokenizer, messages: list[dict]) -> str:
    prompt_text = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

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

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_path", default=INPUT_JSONL)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("Task1-only inference for GRPO checkpoint")
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

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for group in tqdm(valid_groups, desc="Task1 generation"):
            paper_id = group["paper_id"]
            ctx = group.get("paper_context") or {}
            gt_claims = get_deduped_claims(group)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_task1_user(group)},
            ]

            generated = generate_one(model, tokenizer, messages)
            generated = extract_task1_output(generated)
            is_complete, completeness_msg = check_completeness_task1(generated)

            result = {
                "paper_id": paper_id,
                "l1_id": group.get("l1_id", ""),
                "l2_id": group.get("l2_id", ""),
                "l2_name": group.get("l2_name", ""),
                "task": "task1",
                "model_path": args.model_path,
                "input": {
                    "title": ctx.get("title", ""),
                    "abstract": ctx.get("abstract", ""),
                    "user_message": format_task1_user(group),
                },
                "ground_truth": format_task1_assistant_from_claims(gt_claims),
                "generated": generated,
                "generated_claims": parse_task1_claims(generated),
                "gt_claims": gt_claims,
                "is_complete": is_complete,
                "completeness_msg": completeness_msg,
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if is_complete:
                stats["complete"] += 1
            else:
                stats["incomplete"] += 1
                print(f"\nIncomplete | paper_id={paper_id} | {completeness_msg}")

    total = stats["complete"] + stats["incomplete"]
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total: {total}")
    if total > 0:
        print(f"Complete: {stats['complete']} ({stats['complete'] / total * 100:.1f}%)")
        print(f"Incomplete: {stats['incomplete']}")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()