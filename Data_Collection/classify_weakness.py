"""
Classifies each weakness point into the 7 L1 / 17 L2
taxonomy categories using an LLM.

Input:  aligned_weakness_rebuttal_pairs_finegrained.jsonl 
Output: classified_weakness_pairs.jsonl
        — same schema as input, each record gains a new field:
          "weakness_category": {
              "l1_id":     "L1.1",
              "l1_name":   "Experimental Design and Empirical Validation Weaknesses",
              "l2_id":     "L2.1.2",
              "l2_name":   "Missing or Inadequate Comparative and Component Analysis",
              "confidence": 0.91,
              "reasoning": "..."   (brief chain-of-thought from the LLM)
          }
"""

import argparse
import json
import os
import re
import signal
import sys
import time
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI, AzureOpenAI
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ──────────────────────────────────────────────────────────────────────────────
# Taxonomy definition (hard-coded so no external file is needed)
# ──────────────────────────────────────────────────────────────────────────────

TAXONOMY = {
    "L1.1": {
        "name": "Experimental Design and Empirical Validation Weaknesses",
        "definition": (
            "Weaknesses related to the design, scope, rigor, fairness, and completeness "
            "of experiments and evaluations. These issues limit confidence in empirical "
            "claims, comparative performance, robustness, or generalization."
        ),
        "children": {
            "L2.1.1": {
                "name": "Insufficient or Narrow Experimental Evaluation",
                "definition": (
                    "The experimental evaluation is too limited, simplistic, or narrow in "
                    "scope to support the paper's claims. Includes reliance on too few "
                    "datasets/tasks, overly simplistic benchmarks, weak generalization "
                    "evidence, or failure to demonstrate robustness or scalability."
                ),
                "keywords": "limited scope, few datasets, generalization, toy datasets, narrow evaluation, robustness, scalability",
            },
            "L2.1.2": {
                "name": "Missing or Inadequate Comparative and Component Analysis",
                "definition": (
                    "The paper lacks adequate baselines, ablation studies, or component-wise "
                    "analysis to justify performance claims. It is unclear what drives "
                    "improvements or how the method compares to existing work."
                ),
                "keywords": "baseline, comparison, ablation, component, hyperparameter, design choice",
            },
            "L2.1.3": {
                "name": "Weak, Unreliable, or Flawed Empirical Evidence",
                "definition": (
                    "Reported results are unconvincing, unreliable, or methodologically flawed. "
                    "Includes marginal/inconsistent gains, lack of statistical rigor, "
                    "inappropriate metrics, unfair comparisons, or contradictory results."
                ),
                "keywords": "statistical significance, metrics, marginal gains, unfair comparison, inconsistent results",
            },
        },
    },
    "L1.2": {
        "name": "Methodological Clarity and Reproducibility Issues",
        "definition": (
            "Weaknesses related to unclear, incomplete, or confusing descriptions of "
            "methods, assumptions, or implementation details that hinder understanding "
            "or reproducibility."
        ),
        "children": {
            "L2.2.1": {
                "name": "Unclear or Incomplete Method Description",
                "definition": (
                    "The proposed method, algorithm, or pipeline is poorly explained, "
                    "underspecified, or confusing. Missing explanations of components, "
                    "training/inference procedures, equations, or interactions."
                ),
                "keywords": "unclear method, algorithm description, pipeline, architecture, training, inference, implementation",
            },
            "L2.2.2": {
                "name": "Missing or Insufficient Experimental and Reproducibility Details",
                "definition": (
                    "Critical experimental or implementation details are missing, making "
                    "results hard to reproduce. Includes missing hyperparameters, training "
                    "setups, hardware, code availability, or unexplained sensitivity to tuning."
                ),
                "keywords": "reproducibility, experimental setup, hyperparameters, training details, code, replication, sensitivity",
            },
            "L2.2.3": {
                "name": "Unclear Problem Definition, Assumptions, or Scope",
                "definition": (
                    "The paper does not clearly define the problem, objectives, assumptions, "
                    "or scope of applicability. Core concepts or conditions are ambiguous."
                ),
                "keywords": "problem formulation, definitions, assumptions, scope, applicability, conditions",
            },
        },
    },
    "L1.3": {
        "name": "Theoretical Soundness and Justification Gaps",
        "definition": (
            "Weaknesses related to missing, weak, or flawed theoretical analysis, "
            "assumptions, or guarantees that undermine confidence in correctness or rigor."
        ),
        "children": {
            "L2.3.1": {
                "name": "Missing or Insufficient Theoretical Justification",
                "definition": (
                    "Lacks adequate theoretical analysis or justification. Includes missing "
                    "proofs, absent guarantees, unclear theoretical arguments, and failure "
                    "to formally support correctness, convergence, or complexity claims."
                ),
                "keywords": "missing theory, no proof, theoretical justification, formal analysis, guarantees, correctness, convergence",
            },
            "L2.3.2": {
                "name": "Flawed or Unjustified Theoretical Assumptions",
                "definition": (
                    "Theoretical results rely on assumptions that are unrealistic, overly "
                    "strong, poorly justified, or violated in practice."
                ),
                "keywords": "assumptions, unrealistic assumptions, theoretical soundness, weak rigor, invalid assumptions",
            },
            "L2.3.3": {
                "name": "Theory–Practice Misalignment",
                "definition": (
                    "Clear disconnect between theoretical analysis and empirical evaluation. "
                    "Theoretical claims not reflected in experiments, or assumptions do not "
                    "hold in practice."
                ),
                "keywords": "theory-practice gap, empirical validation, applicability, disconnect, theory vs experiment",
            },
        },
    },
    "L1.4": {
        "name": "Novelty, Contribution, and Positioning Limitations",
        "definition": (
            "Weaknesses related to limited, incremental, unclear, or overstated novelty, "
            "and poor positioning relative to prior work."
        ),
        "children": {
            "L2.4.1": {
                "name": "Insufficient Positioning and Related Work Coverage",
                "definition": (
                    "Fails to adequately situate contributions within existing literature. "
                    "Missing or unclear discussion of prior work, weak comparisons, or "
                    "poor explanation of how the approach differs from existing methods."
                ),
                "keywords": "related work, prior literature, missing citations, positioning, comparison, literature review",
            },
            "L2.4.2": {
                "name": "Weak, Incremental, or Overstated Novelty",
                "definition": (
                    "The contribution offers limited or unclear novelty; often incremental, "
                    "derivative, or a repackaging of known ideas. Reviewers question whether "
                    "the paper meaningfully advances the state of the art."
                ),
                "keywords": "incremental novelty, limited contribution, overstated claims, derivative, repackaging, already known",
            },
        },
    },
    "L1.5": {
        "name": "Motivation, Claims, and Practical Relevance Issues",
        "definition": (
            "Weaknesses related to unclear or weak motivation, overstated or unsupported "
            "claims, and questionable practical relevance or impact."
        ),
        "children": {
            "L2.5.1": {
                "name": "Weak or Unclear Motivation and Framing",
                "definition": (
                    "Fails to clearly motivate the problem, task, or method, or provides "
                    "insufficient intuition for why the work matters or why design choices "
                    "are made."
                ),
                "keywords": "motivation, framing, intuition, rationale, why, significance, scope, insight",
            },
            "L2.5.2": {
                "name": "Unsupported, Overstated, or Incorrect Claims",
                "definition": (
                    "Makes claims not adequately supported by theory or experiments, "
                    "overstated relative to evidence, misleadingly phrased, or technically "
                    "incorrect."
                ),
                "keywords": "claims, evidence, unsupported, overstated, misleading, incorrect, verification",
            },
            "L2.5.3": {
                "name": "Limited Practical Relevance or Real-World Applicability",
                "definition": (
                    "The approach is questioned for being impractical or unrealistic due to "
                    "assumptions, scalability, cost, or deployment constraints. May overclaim "
                    "practical impact."
                ),
                "keywords": "practical, real-world, applicability, deployment, scalability, limitations, failure cases, impact",
            },
        },
    },
    "L1.6": {
        "name": "Writing, Presentation, and Communication Problems",
        "definition": (
            "Weaknesses related to poor writing quality, organization, notation, "
            "formatting, or visual presentation that hinder comprehension."
        ),
        "children": {
            "L2.6.1": {
                "name": "Unclear Writing, Organization, or Notation",
                "definition": (
                    "The manuscript is difficult to understand due to poor writing quality, "
                    "weak organization, confusing exposition, or unclear/overloaded notation."
                ),
                "keywords": "writing, clarity, organization, confusing, hard to follow, notation, inconsistent, readability",
            },
            "L2.6.2": {
                "name": "Formatting, Figures, or Submission Issues",
                "definition": (
                    "Problems with formatting, figures, diagrams, or visual presentation "
                    "that reduce readability or violate submission guidelines."
                ),
                "keywords": "formatting, template, layout, submission, figures, diagrams, visualization",
            },
        },
    },
    "L1.7": {
        "name": "Scalability, Efficiency, and Resource Considerations",
        "definition": (
            "Weaknesses related to missing or insufficient analysis of computational cost, "
            "scalability, runtime, or resource requirements."
        ),
        "children": {
            "L2.7.1": {
                "name": "Missing Computational Cost, Runtime, and Scalability Analysis",
                "definition": (
                    "Fails to adequately analyze or report computational cost, runtime, "
                    "memory usage, or scalability, especially when making efficiency claims."
                ),
                "keywords": "computational cost, runtime, memory, efficiency, scalability, complexity, compute",
            },
        },
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Build a flat lookup: l2_id → (l1_id, l1_name, l2_name)
# ──────────────────────────────────────────────────────────────────────────────

L2_LOOKUP: dict[str, dict] = {}
for l1_id, l1_data in TAXONOMY.items():
    for l2_id, l2_data in l1_data["children"].items():
        L2_LOOKUP[l2_id] = {
            "l1_id":   l1_id,
            "l1_name": l1_data["name"],
            "l2_id":   l2_id,
            "l2_name": l2_data["name"],
        }

VALID_L2_IDS = set(L2_LOOKUP.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Deterministic tie-break for L1.3 sub-categories (L2.3.1 vs L2.3.2 vs L2.3.3)
# ──────────────────────────────────────────────────────────────────────────────

# Keyword phrases that signal L2.3.1 (Missing/Insufficient Theoretical Justification)
_L2_3_1_PHRASES = [
    "incorrect proof", "theorem wrong", "theorem is wrong", "proof is wrong",
    "proof is incorrect", "derivation invalid", "invalid derivation",
    "not shown", "missing proof", "does not follow", "not proven",
    "proof missing", "proof is missing", "no proof", "without proof",
    "lack of proof", "unproven", "proof error", "proof flawed",
]

# Keyword phrases that signal L2.3.2 (Flawed/Unjustified Theoretical Assumptions)
_L2_3_2_PHRASES = [
    "assumption unrealistic", "unrealistic assumption",
    "too strong assumption", "assumption too strong", "overly strong assumption",
    "assumption violated", "violated in practice",
    "non-differentiability ignored", "differentiability assumption",
    "assumption does not hold", "unjustified assumption",
    "assumption is not realistic", "strong assumption",
    "questionable assumption", "invalid assumption",
]

# Keyword phrases that signal L2.3.3 (Theory–Practice Misalignment)
_L2_3_3_PHRASES = [
    "theory and experiment", "theoretical and empirical",
    "theory does not match", "theory-practice gap", "theory practice gap",
    "disconnect between theory", "gap between theory",
    "theoretical results do not match", "theory vs experiment",
    "inconsistent with theory", "not reflected in experiment",
]


def _count_phrase_hits(text: str, phrases: list[str]) -> int:
    """Count how many keyword phrases appear in the (lowered) text."""
    text_lower = text.lower()
    return sum(1 for p in phrases if p in text_lower)


def _l13_tiebreak(weakness_text: str, current_l2: str) -> str | None:
    """
    Deterministic tie-break for L1.3 sub-categories.

    Only fires when the current classification is L2.3.1, L2.3.2, or L2.3.3.
    Returns a corrected l2_id if the keyword evidence clearly favours a
    different sub-category, otherwise returns None (keep original).
    """
    if current_l2 not in ("L2.3.1", "L2.3.2", "L2.3.3"):
        return None

    hits_1 = _count_phrase_hits(weakness_text, _L2_3_1_PHRASES)
    hits_2 = _count_phrase_hits(weakness_text, _L2_3_2_PHRASES)
    hits_3 = _count_phrase_hits(weakness_text, _L2_3_3_PHRASES)

    # Rule 3: if L2.3.3 phrases present AND it's the dominant signal, pick L2.3.3
    if hits_3 > 0 and hits_3 >= hits_1 and hits_3 >= hits_2:
        return "L2.3.3" if current_l2 != "L2.3.3" else None

    # Rule 1: L2.3.1 phrases dominate → pick L2.3.1
    if hits_1 > 0 and hits_1 > hits_2:
        return "L2.3.1" if current_l2 != "L2.3.1" else None

    # Rule 2: L2.3.2 phrases dominate → pick L2.3.2
    if hits_2 > 0 and hits_2 > hits_1:
        return "L2.3.2" if current_l2 != "L2.3.2" else None

    # No clear signal or equal hits — keep original
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in academic peer review analysis.
Your task is to classify a peer-review weakness point into exactly one of the
17 Level-2 categories defined in the taxonomy below.

TAXONOMY
========
L1.1 Experimental Design and Empirical Validation Weaknesses
  L2.1.1 Insufficient or Narrow Experimental Evaluation
          Keywords: limited scope, few datasets, generalization, toy datasets, narrow evaluation, robustness, scalability
  L2.1.2 Missing or Inadequate Comparative and Component Analysis
          Keywords: baseline, comparison, ablation, component, hyperparameter, design choice
  L2.1.3 Weak, Unreliable, or Flawed Empirical Evidence
          Keywords: statistical significance, metrics, marginal gains, unfair comparison, inconsistent results

L1.2 Methodological Clarity and Reproducibility Issues
  L2.2.1 Unclear or Incomplete Method Description
          Keywords: unclear method, algorithm description, pipeline, architecture, training, inference
  L2.2.2 Missing or Insufficient Experimental and Reproducibility Details
          Keywords: reproducibility, hyperparameters, training details, code, replication, sensitivity
  L2.2.3 Unclear Problem Definition, Assumptions, or Scope
          Keywords: problem formulation, definitions, assumptions, scope, applicability

L1.3 Theoretical Soundness and Justification Gaps
  L2.3.1 Missing or Insufficient Theoretical Justification
          Keywords: missing theory, no proof, formal analysis, guarantees, correctness, convergence
  L2.3.2 Flawed or Unjustified Theoretical Assumptions
          Keywords: unrealistic assumptions, theoretical soundness, weak rigor, invalid assumptions
  L2.3.3 Theory–Practice Misalignment
          Keywords: theory-practice gap, disconnect, theory vs experiment

L1.4 Novelty, Contribution, and Positioning Limitations
  L2.4.1 Insufficient Positioning and Related Work Coverage
          Keywords: related work, prior literature, missing citations, positioning, literature review
  L2.4.2 Weak, Incremental, or Overstated Novelty
          Keywords: incremental novelty, limited contribution, derivative, repackaging, already known

L1.5 Motivation, Claims, and Practical Relevance Issues
  L2.5.1 Weak or Unclear Motivation and Framing
          Keywords: motivation, framing, intuition, rationale, why, significance
  L2.5.2 Unsupported, Overstated, or Incorrect Claims
          Keywords: claims, unsupported, overstated, misleading, incorrect, verification
  L2.5.3 Limited Practical Relevance or Real-World Applicability
          Keywords: practical, real-world, applicability, deployment, scalability, limitations

L1.6 Writing, Presentation, and Communication Problems
  L2.6.1 Unclear Writing, Organization, or Notation
          Keywords: writing, clarity, organization, confusing, notation, readability
  L2.6.2 Formatting, Figures, or Submission Issues
          Keywords: formatting, template, figures, diagrams, visualization

L1.7 Scalability, Efficiency, and Resource Considerations
  L2.7.1 Missing Computational Cost, Runtime, and Scalability Analysis
          Keywords: computational cost, runtime, memory, efficiency, scalability, complexity

INSTRUCTIONS
============
1. Read the paper context (title + abstract) and the weakness point carefully.
2. Select the SINGLE most appropriate L2 category.
3. Assign a confidence score between 0.0 and 1.0.
4. Respond in this EXACT JSON format (no markdown, no extra text):

{
  "l2_id": "<e.g. L2.1.2>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining why>"
}
"""


def build_user_prompt(record: dict, retry_hint: str = "") -> str:
    ctx = record.get("paper_context", {})
    title    = ctx.get("title", "Unknown title")
    abstract = ctx.get("abstract", "No abstract available.")

    # ── Resolve weakness text ──────────────────────────────────────────────
    # Priority 1: original_weakness (direct field, used when input is
    #             already-enhanced records)
    weakness = (record.get("original_weakness") or "").strip()

    # Priority 2: consolidated_weakness.initial (step1 raw output format)
    if not weakness:
        cw = record.get("consolidated_weakness") or {}
        weakness = (cw.get("initial") or "").strip()

    # Priority 3: fallback to enhanced_review.claim (last resort)
    if not weakness:
        er = record.get("enhanced_review") or {}
        claim = (er.get("claim") or "").strip()
        evidence = (er.get("evidence") or "").strip()
        if claim:
            weakness = claim
            if evidence:
                weakness += f"\n\nAdditional context: {evidence}"

    if not weakness:
        weakness = "(No weakness text available — classify based on paper context only)"

    # ── Optional retry hint (shown on low-confidence re-attempts) ──────────
    hint_block = ""
    if retry_hint:
        hint_block = f"\n\nCLASSIFIER HINT (from previous attempt): {retry_hint}\n"

    return (
        f"PAPER TITLE: {title}\n\n"
        f"ABSTRACT: {abstract}\n\n"
        f"WEAKNESS POINT TO CLASSIFY:\n{weakness}"
        f"{hint_block}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM call with retry
# ──────────────────────────────────────────────────────────────────────────────

def classify_one(
    client: OpenAI,
    record: dict,
    model: str,
    max_retries: int = 3,
    low_conf_threshold: float = 0.6,
) -> dict:
    """
    Returns the record with a new 'weakness_category' field added.
    On repeated failure, the field is set to an error placeholder.

    Low-confidence results (< low_conf_threshold) are automatically retried
    once with a hint derived from the first attempt's reasoning.
    """

    def _single_call(retry_hint: str = "") -> tuple[dict | None, str]:
        """Returns (parsed_result_or_None, raw_text)."""
        user_prompt = build_user_prompt(record, retry_hint=retry_hint)
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_completion_tokens=256,
                    timeout=30,
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                parsed = json.loads(raw)
                l2_id = parsed.get("l2_id", "").strip()
                if l2_id not in VALID_L2_IDS:
                    raise ValueError(f"Unknown l2_id returned: {l2_id!r}")
                return parsed, raw
            except json.JSONDecodeError as e:
                logging.warning(
                    f"[{record.get('weakness_id')}] JSON parse error (attempt {attempt+1}): {e}"
                )
            except ValueError as e:
                logging.warning(
                    f"[{record.get('weakness_id')}] Validation error (attempt {attempt+1}): {e}"
                )
            except Exception as e:
                logging.warning(
                    f"[{record.get('weakness_id')}] API error (attempt {attempt+1}): {e}"
                )
                time.sleep(2 ** attempt)
        return None, ""

    # ── First call ─────────────────────────────────────────────────────────
    parsed, _ = _single_call()

    if parsed is None:
        record["weakness_category"] = {
            "l1_id":      "UNKNOWN",
            "l1_name":    "Classification Failed",
            "l2_id":      "UNKNOWN",
            "l2_name":    "Classification Failed",
            "confidence": 0.0,
            "reasoning":  "LLM classification failed after all retries.",
        }
        return record

    confidence = float(parsed.get("confidence", 0.0))

    # ── Low-confidence retry ───────────────────────────────────────────────
    # If the first attempt returned low confidence, retry with a hint that
    # explains what the model was uncertain about, nudging it to reconsider
    # more specific L1.3 / L1.1 categories before defaulting to L1.6.
    if confidence < low_conf_threshold:
        first_l2   = parsed.get("l2_id", "")
        first_why  = parsed.get("reasoning", "")
        hint = (
            f"Your previous attempt returned '{first_l2}' with confidence "
            f"{confidence:.2f}: \"{first_why}\". "
            f"This is below the confidence threshold. "
            f"Please reconsider carefully. "
            f"In particular: if the weakness mentions proofs, theorems, "
            f"equations, logical steps, or formal guarantees, prefer L1.3 "
            f"(Theoretical Soundness). If it mentions incorrect or inconsistent "
            f"notation / unclear equation presentation, it could still be L2.3.1 "
            f"(Missing Theoretical Justification) rather than L2.6.1 (Writing). "
            f"Re-read the weakness and pick the MOST SPECIFIC matching category."
        )
        logging.info(
            f"[{record.get('weakness_id')}] Low confidence ({confidence:.2f}) "
            f"on {first_l2} — retrying with hint."
        )
        parsed2, _ = _single_call(retry_hint=hint)
        if parsed2 is not None:
            conf2 = float(parsed2.get("confidence", 0.0))
            # Accept the retry result if it is more confident OR equal
            if conf2 >= confidence:
                parsed = parsed2
                logging.info(
                    f"[{record.get('weakness_id')}] Retry improved confidence: "
                    f"{confidence:.2f} → {conf2:.2f} "
                    f"({first_l2} → {parsed2.get('l2_id')})"
                )
            else:
                logging.info(
                    f"[{record.get('weakness_id')}] Retry did not improve "
                    f"({conf2:.2f} < {confidence:.2f}), keeping original."
                )

    # ── Build final result ─────────────────────────────────────────────────
    l2_id = parsed["l2_id"]

    # ── L1.3 deterministic tie-break (L2.3.1 vs L2.3.2 vs L2.3.3) ────────
    if l2_id in ("L2.3.1", "L2.3.2", "L2.3.3"):
        # Resolve weakness text (same priority chain as build_user_prompt)
        _wtext = (record.get("original_weakness") or "").strip()
        if not _wtext:
            _cw = record.get("consolidated_weakness") or {}
            _wtext = (_cw.get("initial") or "").strip()
        if not _wtext:
            _er = record.get("enhanced_review") or {}
            _wtext = (_er.get("claim") or "").strip()

        corrected = _l13_tiebreak(_wtext, l2_id)
        if corrected is not None:
            logging.info(
                f"[{record.get('weakness_id')}] L1.3 tie-break: "
                f"{l2_id} → {corrected}"
            )
            l2_id = corrected

    info  = L2_LOOKUP[l2_id]
    record["weakness_category"] = {
        "l1_id":      info["l1_id"],
        "l1_name":    info["l1_name"],
        "l2_id":      l2_id,
        "l2_name":    info["l2_name"],
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning":  str(parsed.get("reasoning", "")),
    }
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Load / save helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_done_ids(path: str) -> set[str]:
    """Read already-classified IDs from an existing output file (for --resume)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if "weakness_category" in r:
                        done.add(r["weakness_id"])
                except Exception:
                    pass
    return done


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify weakness points via LLM.")
    parser.add_argument(
        "--input", required=True,
        help="Path to aligned_weakness_rebuttal_pairs_finegrained.jsonl (step1 output)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write classified_weakness_pairs.jsonl"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model/deployment name. If omitted, read AZURE_OPENAI_DEPLOYMENT / OPENAI_MODEL / MODEL."
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel threads (default: 8)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=20,
        help="Records to flush to disk per batch (default: 20)"
    )
    parser.add_argument(
        "--progress_every", type=int, default=None,
        help="Log progress every N completed records (independent of --batch_size)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip records already present in --output"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to write progress checkpoint JSON (default: <output>.checkpoint.json)"
    )
    parser.add_argument(
        "--reclassify_low_conf", action="store_true",
        help=(
            "Re-run classification for records in --input whose existing "
            "weakness_category has confidence below --conf_threshold. "
            "Useful to fix bad classifications without reprocessing everything. "
            "Reads existing results from --output (if present) and overwrites "
            "only the low-confidence records."
        )
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.6,
        help="Confidence threshold for --reclassify_low_conf (default: 0.6)"
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Resolve model/deployment name
    model_name = (
        args.model
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL")
    )
    if not model_name:
        raise SystemExit(
            "Model/deployment is required. Pass --model or set "
            "AZURE_OPENAI_DEPLOYMENT (preferred), OPENAI_MODEL, or MODEL."
        )

    # Initialize client:
    # - If Azure endpoint exists, use AzureOpenAI with AZURE_* env vars.
    # - Otherwise use OpenAI with OPENAI_API_KEY.
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=azure_endpoint,
        )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load input
    logging.info(f"Loading input from: {args.input}")
    records = load_jsonl(args.input)
    logging.info(f"Total records loaded: {len(records)}")

    # ── Mode: reclassify low-confidence records ────────────────────────────
    if args.reclassify_low_conf:
        # Build a map of existing classifications from --output (if present)
        existing: dict[str, dict] = {}
        if os.path.exists(args.output):
            for r in load_jsonl(args.output):
                wid = r.get("weakness_id")
                if wid:
                    existing[wid] = r

        # Identify low-confidence records from the INPUT file
        # (input may already have weakness_category if it's a previously
        #  classified file, or may lack it if it's the raw step1 output)
        low_conf_records = []
        high_conf_records = []
        for r in records:
            wid = r.get("weakness_id")
            # Check existing output first, then fall back to inline field
            src = existing.get(wid, r)
            cat = src.get("weakness_category", {})
            conf = cat.get("confidence", 0.0)
            l2   = cat.get("l2_id", "UNKNOWN")
            if l2 == "UNKNOWN" or conf < args.conf_threshold:
                # Use the richer version from existing output if available
                low_conf_records.append(existing.get(wid, r))
            else:
                high_conf_records.append(existing.get(wid, r))

        logging.info(
            f"Reclassify mode: {len(low_conf_records)} low-confidence "
            f"(< {args.conf_threshold}) | {len(high_conf_records)} keeping as-is"
        )

        if not low_conf_records:
            logging.info("No low-confidence records found. Nothing to do.")
            return

        # Reclassify the low-conf records
        reclassified: dict[str, dict] = {}
        failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    classify_one, client, r, model_name,
                    low_conf_threshold=args.conf_threshold
                ): r
                for r in low_conf_records
            }
            done_count = 0
            pbar = tqdm(total=len(low_conf_records), desc="Reclassify", unit="rec") if tqdm else None
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    orig = futures[future]
                    logging.error(f"Unexpected error for {orig.get('weakness_id')}: {e}")
                    result = orig
                cat = result.get("weakness_category", {})
                if cat.get("l2_id") == "UNKNOWN":
                    failed += 1
                reclassified[result["weakness_id"]] = result
                done_count += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(failed=failed)
                if done_count % args.batch_size == 0:
                    logging.info(
                        f"Reclassified {done_count}/{len(low_conf_records)} | failed: {failed}"
                    )
            if pbar:
                pbar.close()

        # Merge: high-conf unchanged + reclassified low-conf
        # Preserve original record order
        merged = []
        id_order = [r.get("weakness_id") for r in records]
        high_map = {r["weakness_id"]: r for r in high_conf_records}
        for wid in id_order:
            if wid in reclassified:
                merged.append(reclassified[wid])
            elif wid in high_map:
                merged.append(high_map[wid])
            # else: record not in either (shouldn't happen)

        save_jsonl(merged, args.output)
        logging.info(
            f"\nReclassify complete. "
            f"Updated {len(reclassified)} records | failed: {failed} | "
            f"Output: {args.output}"
        )
        # Fall through to print distribution stats
        all_output = merged

    # ── Mode: normal classification ────────────────────────────────────────
    else:
        # Resume: skip already-done IDs
        done_ids: set[str] = set()
        if args.resume:
            done_ids = load_done_ids(args.output)
            logging.info(f"Resuming: {len(done_ids)} records already classified, skipping.")

        to_process = [r for r in records if r.get("weakness_id") not in done_ids]
        logging.info(f"Records to classify: {len(to_process)}")

        if not to_process:
            logging.info("Nothing to do. All records already classified.")
            return

        # Output file — append mode if resuming, write mode otherwise
        out_mode = "a" if args.resume and os.path.exists(args.output) else "w"
        out_f = open(args.output, out_mode, encoding="utf-8")

        total_done  = len(done_ids)
        total_all   = len(records)
        failed      = 0
        low_conf    = 0
        completed_this_run = 0
        batch_buf: list[dict] = []
        start_ts = time.time()
        checkpoint_path = args.checkpoint or f"{args.output}.checkpoint.json"
        checkpoint_enabled = True
        shutdown_requested = False

        def _signal_handler(signum, frame):
            nonlocal shutdown_requested
            if shutdown_requested:
                logging.warning("Second interrupt received, forcing exit.")
                sys.exit(1)
            shutdown_requested = True
            logging.warning(
                "\nInterrupt received! Finishing current batch and saving progress... "
                "(press Ctrl+C again to force quit)"
            )

        prev_sigint = signal.signal(signal.SIGINT, _signal_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

        def flush_batch(buf: list[dict]) -> None:
            for r in buf:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()

        def write_checkpoint() -> None:
            if not checkpoint_enabled:
                return
            elapsed = time.time() - start_ts
            speed = (completed_this_run / elapsed) if elapsed > 0 else 0.0
            remaining = max(total_all - total_done, 0)
            eta_seconds = (remaining / speed) if speed > 0 else None
            payload = {
                "output_path": args.output,
                "input_path": args.input,
                "model": model_name,
                "resume": args.resume,
                "total_records": total_all,
                "already_done_before_run": len(done_ids),
                "completed_this_run": completed_this_run,
                "total_done": total_done,
                "remaining": remaining,
                "failed_this_run": failed,
                "low_conf_this_run": low_conf,
                "elapsed_seconds": round(elapsed, 2),
                "speed_records_per_sec": round(speed, 4),
                "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, checkpoint_path)

        def log_progress() -> None:
            elapsed = time.time() - start_ts
            speed = (completed_this_run / elapsed) if elapsed > 0 else 0.0
            remaining = max(total_all - total_done, 0)
            eta = _format_duration(remaining / speed) if speed > 0 else "--:--"
            logging.info(
                f"Progress: {total_done}/{total_all} "
                f"({total_done/total_all*100:.1f}%) | failed: {failed} "
                f"| low_conf: {low_conf} | speed: {speed:.2f} rec/s | ETA: {eta}"
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    classify_one, client, r, model_name,
                    low_conf_threshold=args.conf_threshold
                ): r
                for r in to_process
            }
            pbar = tqdm(total=len(to_process), desc="Classify", unit="rec") if tqdm else None
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    orig = futures[future]
                    logging.error(f"Unexpected error for {orig.get('weakness_id')}: {e}")
                    orig["weakness_category"] = {
                        "l1_id": "UNKNOWN", "l1_name": "Classification Failed",
                        "l2_id": "UNKNOWN", "l2_name": "Classification Failed",
                        "confidence": 0.0,
                        "reasoning": f"Unexpected error: {e}",
                    }
                    result = orig

                cat = result.get("weakness_category", {})
                if cat.get("l2_id") == "UNKNOWN":
                    failed += 1
                if float(cat.get("confidence", 0.0)) < args.conf_threshold:
                    low_conf += 1

                batch_buf.append(result)
                total_done += 1
                completed_this_run += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(
                        failed=failed,
                        low_conf=low_conf,
                        done=f"{total_done}/{total_all}",
                    )

                if (
                    args.progress_every
                    and args.progress_every > 0
                    and completed_this_run % args.progress_every == 0
                    and len(batch_buf) < args.batch_size
                ):
                    write_checkpoint()
                    log_progress()

                if len(batch_buf) >= args.batch_size:
                    flush_batch(batch_buf)
                    batch_buf.clear()
                    write_checkpoint()
                    log_progress()

                # ── Graceful shutdown on Ctrl+C / SIGTERM ─────────────
                if shutdown_requested:
                    logging.warning("Shutting down: cancelling pending tasks...")
                    for f in futures:
                        f.cancel()
                    break

            if pbar:
                pbar.close()

        # Flush remaining results in buffer
        if batch_buf:
            flush_batch(batch_buf)
            write_checkpoint()
        # Final checkpoint snapshot
        write_checkpoint()
        out_f.close()

        # Restore original signal handlers
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

        all_output = load_jsonl(args.output)
        logging.info(f"Checkpoint written to: {checkpoint_path}")

        if shutdown_requested:
            logging.info(
                f"Gracefully stopped. {completed_this_run} records saved this run. "
                f"Re-run with --resume to continue."
            )
            sys.exit(0)

    # ── Summary statistics (shared by both modes) ──────────────────────────
    logging.info("\n" + "="*60)
    logging.info("CLASSIFICATION COMPLETE")
    logging.info("="*60)

    from collections import Counter
    l1_counter: Counter = Counter()
    l2_counter: Counter = Counter()
    conf_scores = []
    low_conf_remaining = 0

    for r in all_output:
        cat = r.get("weakness_category", {})
        l1  = cat.get("l1_id", "UNKNOWN")
        l2  = cat.get("l2_id", "UNKNOWN")
        c   = cat.get("confidence", 0.0)
        if l1 != "UNKNOWN":
            l1_counter[l1] += 1
            l2_counter[l2] += 1
            conf_scores.append(c)
            if c < args.conf_threshold:
                low_conf_remaining += 1

    logging.info("\nL1 distribution:")
    for k, v in sorted(l1_counter.items()):
        l1_name = TAXONOMY.get(k, {}).get("name", "")
        logging.info(f"  {k} ({l1_name[:40]}): {v}")

    logging.info("\nL2 distribution (top 10):")
    for k, v in l2_counter.most_common(10):
        l2_name = L2_LOOKUP.get(k, {}).get("l2_name", "")
        logging.info(f"  {k} ({l2_name[:45]}): {v}")

    if conf_scores:
        avg_conf = sum(conf_scores) / len(conf_scores)
        logging.info(f"\nAverage confidence:           {avg_conf:.3f}")
        logging.info(
            f"Still low-confidence "
            f"(< {args.conf_threshold}): {low_conf_remaining} "
            f"({low_conf_remaining/len(conf_scores)*100:.1f}%)"
        )
        if low_conf_remaining > 0 and not args.reclassify_low_conf:
            logging.info(
                f"\nTIP: Re-run with --reclassify_low_conf to fix "
                f"these {low_conf_remaining} records."
            )

    logging.info(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
