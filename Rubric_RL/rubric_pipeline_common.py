import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from urllib.parse import urlparse, urlunparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AzureOpenAI, OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OpenAIClient = Union[OpenAI, AzureOpenAI]


def _azure_openai_api_key() -> str:
    return (os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or "").strip()


def uses_azure_openai() -> bool:
    """True when Azure Chat Completions should be used (endpoint + key set)."""
    ep = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    return bool(ep and _azure_openai_api_key())


def _normalize_azure_endpoint(raw: str) -> str:
    """
    AzureOpenAI expects the resource base URL (scheme + host), not a full path
    like /openai/responses?... Strip path and query.
    """
    raw = raw.strip()
    if not raw:
        return raw
    if "://" not in raw:
        raw = "https://" + raw
    p = urlparse(raw)
    scheme = p.scheme or "https"
    return urlunparse((scheme, p.netloc, "", "", "", ""))


def ensure_openai_credentials() -> None:
    """Raise if neither OpenAI nor Azure credentials are configured."""
    if uses_azure_openai():
        return
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Set OPENAI_API_KEY for the OpenAI API, or set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_KEY (or AZURE_OPENAI_API_KEY) for Azure OpenAI."
        )


def make_openai_client() -> OpenAIClient:
    """
    Build a client for Chat Completions.

    - Azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY or AZURE_OPENAI_API_KEY;
      optional AZURE_OPENAI_API_VERSION (default 2024-12-01-preview);
      model name must be the deployment name (AZURE_OPENAI_DEPLOYMENT or OPENAI_MODEL).
    - OpenAI: OPENAI_API_KEY; optional OPENAI_BASE_URL.
    """
    if uses_azure_openai():
        endpoint = _normalize_azure_endpoint(os.environ["AZURE_OPENAI_ENDPOINT"])
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        return AzureOpenAI(
            api_key=_azure_openai_api_key(),
            api_version=api_version,
            azure_endpoint=endpoint,
        )
    ensure_openai_credentials()
    return OpenAI()


def get_openai_model_name() -> str:
    """Model id (OpenAI) or deployment name (Azure)."""
    if uses_azure_openai():
        dep = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if dep:
            return dep
    return os.getenv("OPENAI_MODEL", "gpt-5.1")
SFT_TEMPERATURES = [0.1, 0.7, 1.0]

BATCH_POLL_SECONDS = int(os.getenv("OPENAI_BATCH_POLL_SECONDS", "60"))
BATCH_COMPLETION_WINDOW = os.getenv("OPENAI_BATCH_COMPLETION_WINDOW", "24h")

# GPT candidate generation budgets
GPT_CANDIDATE_MAX_TOKENS_TASK1 = 1024
GPT_CANDIDATE_MAX_TOKENS_TASK2 = 1536

# ─────────────────────────────────────────────────────────────────────────────
# System prompt  (must match SFT training exactly)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert academic peer reviewer specializing in machine learning and AI research.
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
- Do NOT add sections not in the template
- Output must be written in English only
"""

# ─────────────────────────────────────────────────────────────────────────────
# GT text helpers
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_review_to_text(er: Dict) -> str:
    if not er:
        return ""
    claim = (er.get("claim") or "").strip()
    evidence = (er.get("evidence") or "Specific paper location not identified.").strip()
    lines = ["## Claim", claim, "", "## Evidence", evidence, ""]
    for i, s in enumerate(er.get("actionable_suggestions") or [], 1):
        what = (s.get("what") or "").strip()
        where = (s.get("where") or "").strip()
        how = (s.get("how") or "").strip()
        outcome = (s.get("expected_outcome") or "").strip()
        priority = (s.get("priority") or "medium").strip()
        lines += [
            f"### Suggestion {i}",
            f"- **What**: {what}",
            f"- **Where**: {where}",
            f"- **How**: {how}",
            f"- **Expected Outcome**: {outcome}",
            f"- **Priority**: {priority}",
            "",
        ]
    severity = (er.get("severity") or "major").strip()
    lines += ["## Severity", severity]
    return "\n".join(lines).strip()


def items_to_task1_gt(weakness_items: List[Dict]) -> str:
    claims = [
        it.get("original_weakness", "").strip()
        for it in weakness_items
        if it.get("original_weakness", "").strip()
    ]
    if not claims:
        return "None"
    return "\n".join(f"Claim {i+1}: {c}" for i, c in enumerate(claims))

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders  (must match SFT training prompts exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_task1_messages(paper_context: Dict, l2_id: str, l2_name: str) -> List[Dict]:
    title = paper_context.get("title", "Unknown")
    abstract = paper_context.get("abstract", "")
    keywords = ", ".join(paper_context.get("keywords") or [])
    user = (
        f"[TASK 1] Weakness Claim Discovery\n\n"
        f"**Paper**: {title}\n\n"
        f"**Keywords**: {keywords}\n\n"
        f"**Abstract**:\n{abstract}\n\n"
        f"**Weakness label**: {l2_id} — {l2_name}\n\n"
        "Identify all concrete weaknesses in this paper under the label above "
        "and state each as a single-sentence diagnostic claim. "
        "Prefer the most specific reviewer concerns tied to a concrete experiment, "
        "metric, dataset, component, or paper location, and avoid repeating the same "
        "weakness in paraphrased form.\n\n"
        "**Important**: Generate 1-2 distinct claims if the paper has multiple "
        "independent weaknesses under this label. If multiple potential claims address "
        "similar underlying issues, consolidate them into one precise claim rather than "
        "listing variations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_task2_messages(
    paper_context: Dict,
    l2_id: str,
    l2_name: str,
    claim_text: str,
) -> List[Dict]:
    title = paper_context.get("title", "Unknown")
    abstract = paper_context.get("abstract", "")
    keywords = ", ".join(paper_context.get("keywords") or [])
    user = (
        f"[TASK 2] Actionable Analysis\n\n"
        f"**Paper**: {title}\n\n"
        f"**Keywords**: {keywords}\n\n"
        f"**Abstract**:\n{abstract}\n\n"
        f"**Weakness label**: {l2_id} — {l2_name}\n\n"
        f"**Claim**: {claim_text}\n\n"
        "Provide evidence, actionable suggestions, and severity for the claim above "
        "following the template exactly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_task2_compact_gpt_messages(
    paper_context: Dict,
    l2_id: str,
    l2_name: str,
    claim_text: str,
) -> List[Dict]:
    title = paper_context.get("title", "Unknown")
    abstract = paper_context.get("abstract", "")
    keywords = ", ".join(paper_context.get("keywords") or [])

    system = """\
You are an expert academic peer reviewer.

Produce a COMPACT structured actionable analysis.

Rules:
- Use exactly the section headers below.
- Include exactly 1 suggestion only.
- Keep Evidence to 1 short sentence.
- Keep How to 1-2 concise sentences.
- Keep the total answer concise while preserving specificity.
- WHERE must name a concrete paper location if possible.
- NEVER reference the authors' rebuttal or any post-submission response.
- Output must be written in English only.

Required output format:

## Claim
(exact claim text)

## Evidence
(1 short sentence)

### Suggestion 1
- **What**: ...
- **Where**: ...
- **How**: ...
- **Expected Outcome**: ...
- **Priority**: critical | high | medium

## Severity
critical | major | moderate | minor
"""

    user = (
        f"[TASK 2] Compact Actionable Analysis Candidate\n\n"
        f"**Paper**: {title}\n\n"
        f"**Keywords**: {keywords}\n\n"
        f"**Abstract**:\n{abstract}\n\n"
        f"**Weakness label**: {l2_id} — {l2_name}\n\n"
        f"**Claim**: {claim_text}\n\n"
        "Provide one compact but specific structured analysis for the claim above."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ─────────────────────────────────────────────────────────────────────────────
# SFT model
# ─────────────────────────────────────────────────────────────────────────────

def load_sft_model(checkpoint_path: str):
    print(f"Loading SFT model from {checkpoint_path} ...")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model = model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.eval()
    print("SFT model loaded.")
    return model, tokenizer


def generate_with_sft(
    model,
    tokenizer,
    messages: List[Dict],
    temperature: float,
    max_new_tokens: int = 1024,
) -> Optional[str]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"    SFT generation error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Rubric extraction prompts
# ─────────────────────────────────────────────────────────────────────────────

def _format_candidates_block(candidates: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        temp_str = str(c["temperature"]) if c["temperature"] is not None else "GT"
        text = (c["text"] or "")[:800]
        lines.append(
            f"### Candidate {i+1} (source={c['source']}, temp={temp_str}):\n{text}..."
        )
    return "\n\n".join(lines)


def _format_non_gt_candidates_block(candidates: List[Dict]) -> str:
    non_gt = [c for c in candidates if c["source"] != "gt"]
    lines = []
    for i, c in enumerate(non_gt, 1):
        temp_str = str(c["temperature"]) if c["temperature"] is not None else "N/A"
        text = (c["text"] or "")[:800]
        lines.append(
            f"### Candidate {i} (source={c['source']}, temp={temp_str}):\n{text}..."
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ONLY TASK1 MODIFIED BELOW
# ─────────────────────────────────────────────────────────────────────────────

TASK1_RUBRIC_EXTRACTION_PROMPT = """\
You are designing a grading rubric for Task 1 (Weakness Claim Discovery) peer-review outputs.

# CONTEXT
**Paper**: {title}
**Weakness Type**: {l2_name} ({l2_id})
**Abstract**: {abstract_snippet}...

# GROUND-TRUTH REFERENCE ANSWER
This is the best available answer for this specific sample. Use it to infer what this sample most deserves credit for.
{gt_text}

# NON-GT CANDIDATE OUTPUTS (varying quality)
Use these to infer common failure modes that should be penalized for this sample.
{non_gt_candidates_text}

# YOUR TASK
Design 5-6 soft requirements for evaluating Task 1 outputs.

## WHAT TASK 1 OUTPUTS LOOK LIKE
The model outputs a numbered list: "Claim 1: ... Claim 2: ..." or "None".
Each claim is a single sentence identifying a concrete paper deficiency.
Do NOT use crude count-only checks, but you SHOULD use the GT to infer how many DISTINCT weakness families this sample actually contains.
If the GT clearly contains K distinct weakness families, at least one soft requirement should check whether the candidate also preserves that multiplicity rather than collapsing everything into one broad claim.

## REQUIRED RUBRIC DESIGN STRATEGY
First infer:
(1) the 2-3 most important SAMPLE-SPECIFIC weakness families captured by the GT answer;
(2) the 2-3 most important SAMPLE-SPECIFIC failure modes visible in the bad candidates.
Then convert them into 5-6 soft requirements.

## VERY IMPORTANT: REWARD WEAKNESS FAMILIES, NOT GT WORDING
The rubric should reward answers that capture the SAME SAMPLE-SPECIFIC WEAKNESS FAMILY as the GT,
even if they use different wording, a different sentence structure, or a different but equivalent concrete expression.

Examples of acceptable variation:
- a candidate may describe the same missing evaluation problem using different metrics, nouns, or phrasing;
- a candidate may state the same experimental gap more compactly or more explicitly than the GT;
- a candidate may merge two closely related GT details into one strong claim if the underlying weakness family is still captured clearly.

Do NOT require exact noun overlap with the GT unless that noun is essential to the sample's core weakness.
Do NOT turn the rubric into an answer-key matching checklist.

## SAMPLE-SPECIFIC TASK1 EMPHASIS
This sample may contain more than one distinct experimental weakness family.
A strong Task 1 rubric should usually reward outputs that:
- capture the main GT-specific weakness families for this paper;
- distinguish separate weakness families when they are clearly different;
- stay grounded in the paper's actual evaluation setting;
- avoid generic "insufficient experiments" complaints with no sample-specific content.

If the GT suggests two or more distinct weakness families, the rubric should usually expect the answer to preserve that separation.
Do NOT give full credit to a single broad umbrella claim when the GT contains multiple clearly different weaknesses, unless those GT weaknesses are truly near-duplicates.

## CRITICAL RULES
- At least 2 soft requirements must be clearly derived from GT-specific content.
- At least 2 soft requirements must be clearly derived from bad-candidate failure modes.
- At least 2 soft requirements must explicitly mention sample-specific concepts from this instance.
- Do NOT make all requirements generic category-level criteria.
- Do NOT make all requirements exact GT-detail checks.
- Prefer requirements that would be noticeably less appropriate for a different paper under the same label.
- Each soft requirement must test a DIFFERENT grading dimension. Do NOT write overlapping requirements that would reward the same underlying fact twice.
- If two requirements could both be satisfied by the same sentence/span/evidence in the answer, rewrite them until their scopes are clearly distinct.
- Requirements must be written as NARROW, NECESSARY conditions, not broad topic hints.
- Avoid open-ended wording such as "for example", "e.g.", "such as", "including", or "mentions the experimental context" unless the exact allowed alternatives are the thing being tested.
- Prefer formulations like "At least one claim must explicitly mention X and identify Y" over broad formulations like "Claims should reference the experimental context".
- A candidate that only vaguely touches the topic must NOT receive credit; the requirement should demand the specific comparison, control, omission, or weakness family that the GT makes central.

## DO REWARD
- Capturing the specific experimental weakness families highlighted by the GT for this paper.
- Mentioning the paper's actual evaluation setting when appropriate.
- Distinguishing multiple different weaknesses when they are truly different.
- Using alternative but still sample-grounded phrasing to express the same core weakness.
- Staying concrete and testable rather than generic.

## DO PENALIZE
- Repetition or duplicated claims.
- Generic claims that fail to mention the sample's core experimental setting.
- Hallucinated or off-topic complaints not supported by the paper context.
- Drifting to standard benchmark/baseline complaints when the GT is about a different missing evidence family.
- Missing the main GT-specific weakness families entirely.
- Collapsing multiple different weaknesses into one vague umbrella complaint when that removes important sample-specific information.

## DO NOT EVALUATE
- Exact wording match to GT.
- Exact noun overlap with GT.
- Pure count-based checks such as "has >= 2 claims" with no semantic grounding.
- Superficial topic overlap where the answer only mentions one nearby concept but misses the GT's concrete missing control / comparison / omission.

## HARD CONSTRAINTS (exactly 2)
1. "Output must not reference the authors' rebuttal or any post-submission response"
2. "Output must be formatted as numbered claims (Claim 1: ...) or the word None"

## OUTPUT FORMAT — return ONLY valid JSON
{{
  "soft_requirements": [
    {{
      "requirement": "...",
      "weight": 95,
      "type": "semantic",
      "rationale": "..."
    }}
  ],
  "hard_constraints": [
    {{"requirement": "...", "rationale": "..."}}
  ]
}}

Rules:
- 5-6 soft requirements total
- weights will be normalised to 100
- type must be "semantic" or "format"
- Keep hard constraints general and stable; put sample-specificity into soft requirements
- The final rubric should be sample-specific but should still give credit to alternative valid answers that capture the same core weakness families as the GT
- Before finalizing, check that each soft requirement is mutually non-overlapping with the others and that satisfying one does not almost automatically satisfy another.
- Before finalizing, rewrite any requirement that could be satisfied by merely name-dropping one loosely related concept without stating the GT-central missing comparison, control, or omission.
"""

# TASK2 NOT MODIFIED
TASK2_RUBRIC_EXTRACTION_PROMPT = """\
You are designing a grading rubric for Task 2 (Actionable Analysis) peer-review outputs.

# CONTEXT
**Paper**: {title}
**Weakness Type**: {l2_name} ({l2_id})
**Claim being elaborated**: {claim_text}
**Abstract**: {abstract_snippet}...

# GROUND-TRUTH REFERENCE ANSWER
This is the best available answer for this specific sample. Use it to infer what this sample most deserves credit for.
{gt_text}

# NON-GT CANDIDATE OUTPUTS (varying quality)
Use these to infer common failure modes that should be penalized for this sample.
{non_gt_candidates_text}

# YOUR TASK
Design 5-6 soft requirements for evaluating Task 2 outputs.

## WHAT TASK 2 OUTPUTS LOOK LIKE
Structured markdown:
## Claim
## Evidence
### Suggestion N with:
- **What**
- **Where**
- **How**
- **Expected Outcome**
- **Priority**
## Severity

## REQUIRED RUBRIC DESIGN STRATEGY
First infer:
(1) the 2-3 most important SAMPLE-SPECIFIC reward dimensions from the GT answer;
(2) the 2-3 most important SAMPLE-SPECIFIC failure modes visible in the bad candidates.

Then convert them into 5-6 soft requirements.

## CORE TASK2 PRINCIPLE
A Task 2 answer is NOT high quality merely because it is well formatted.
It should earn credit if it captures the same core diagnostic goal as the GT,
while allowing different but equally valid concrete implementations of the remedy.

Many weak candidates may look polished, mention plausible experiments, or have all required fields,
but they should still score lower if they drift away from the GT's central missing evidence
or replace a focused fix with a much broader and less diagnostic one.

## CRITICAL RULES
- At least 2 soft requirements must be clearly derived from GT-specific content.
- At least 2 soft requirements must be clearly derived from bad-candidate failure modes.
- At least 2 soft requirements must explicitly mention concrete nouns, datasets, perturbations, metrics, baselines, sections, or experiment types from this specific instance.
- Do NOT make all requirements generic template-level checks like "Evidence should be specific" or "How should be detailed."
- Prefer requirements that would be noticeably less appropriate for a different sample under the same weakness label.
- At least 2 soft requirements should still give credit to alternative strong answers that differ from the GT in wording or implementation details.
- Soft requirements should be atomic and checkable: each one should test a single concrete grading question rather than a vague umbrella notion like "is the answer detailed and specific".
- Do NOT reward mere topic overlap or broadly related terminology if the answer misses the sample's core missing evidence.
- Each soft requirement must cover a distinct grading dimension. Do NOT let two requirements reward the same intervention, evidence family, or missing comparison twice.
- If two requirements could both be satisfied by the same sentence/span, merge or rewrite them so the scopes are clearly different.
- Requirements must be narrow enough that a superficially related answer cannot pass by only mentioning one nearby concept.
- Avoid open-ended phrasing like "e.g.", "for example", "such as", "related to", or "mentions the context" unless the allowed alternatives are explicitly the target of the requirement.
- Prefer formulations that require the candidate to identify the GT-central comparison / control / intervention / evidence family, not just adjacent terminology.

## WHAT TO REWARD
- Capturing the same missing evidence or diagnostic need as the GT, even if the proposed fix is not phrased identically.
- Evidence that is grounded in the specific part of the paper implicated by the claim.
- Suggestions whose What/Where/How remain concretely tied to the current sample's missing experiment design, without requiring exact overlap with the GT's section names, datasets, metrics, or baselines unless those details are essential to the claim itself.
- Expected Outcome that reflects the purpose of the proposed experiment in this sample.
- If the GT contains multiple distinct remedy components, reward outputs that preserve those distinctions rather than collapsing them.
- Navigable and actionable Where/How fields even when they use different but equally concrete placements or experiment choices from the GT.

## WHAT TO PENALIZE
- Generic but well-formatted answers that could fit many different papers.
- Suggestions that broaden the issue into a larger but less relevant complaint.
- Drift toward unrelated baselines, architectures, tasks, or evaluation settings not central to this sample.
- Vague or default Where/How fields such as "in the paper" or "in experiments", but do NOT penalize alternative concrete placements solely because they differ from the GT's exact subsection naming.
- Outputs that mention superficially relevant concepts but miss the GT's main concrete intervention.
- Outputs with extra repeated content, assistant pollution, prompt leakage, or continuation artifacts.
- Outputs that are structurally correct but sample-wise too broad.

## IMPORTANT TASK2 DISTINCTIONS
When comparing good vs bad candidates, prioritize whether the answer:
- targets the same diagnostic intent or evidence family as the GT,
- uses a controllable variable / perturbation / comparison axis that addresses the same missing evidence, even if not identical to the GT,
- proposes metrics and outcomes aligned with the same evaluative purpose as the GT,
- avoids replacing a focused fix with a broader but less diagnostic experiment.

## SPECIAL INSTRUCTION ABOUT BAD CANDIDATES
If bad candidates contain superficially related answers that would wrongly receive high scores under a generic rubric,
write soft requirements that explicitly block those false positives.

For example:
- do NOT ask only "Does the answer discuss robustness?"
- instead ask whether it proposes the specific family of robustness experiment, degradation, metric, comparison, or diagnostic goal that the GT makes central.

Likewise:
- do NOT ask only "Does the answer suggest quantitative evaluation?"
- instead ask whether it proposes the same kind of quantitative evidence or measurement family highlighted by the GT, while still allowing equivalent alternative formulations.

## MANDATORY CALIBRATION STEP — DO THIS BEFORE WRITING THE FINAL RUBRIC

After drafting your 5-6 soft requirements, mentally score the BEST non-GT candidate...

If YES (best non-GT averages above about 4.2/5) — your rubric is too lenient. You MUST revise it.
  - Identify which requirement(s) the best non-GT candidate satisfies easily...
  - Rewrite to be more sample-specific and discriminating...
  - A well-calibrated rubric should have GT scoring at least about 0.5 points higher than best non-GT on the 1-5 scale.

If NO — proceed to output.

This calibration step is mandatory. Do not skip it.

## HARD-CONSTRAINT EVALUATION NOTE
Hard constraints are structural only.
Do NOT turn semantic disagreement or missing sample-specific content into a hard-constraint failure.
Failure on soft requirements must not be used to mark a hard constraint as failed.

## HARD CONSTRAINTS (exactly 2)
1. "Output must not reference the authors' rebuttal or any post-submission response"
2. "Output must contain all four sections: ## Claim, ## Evidence, at least one ### Suggestion, ## Severity"

## OUTPUT FORMAT — return ONLY valid JSON
{{
  "soft_requirements": [
    {{
      "requirement": "...",
      "weight": 88,
      "type": "semantic",
      "rationale": "..."
    }}
  ],
  "hard_constraints": [
    {{"requirement": "...", "rationale": "..."}}
  ]
}}

Rules:
- 5-6 soft requirements total
- weights will be normalised to 100
- type must be "semantic" or "format"
- Keep hard constraints general and stable; put sample-specificity into soft requirements
- Before finalizing, verify that the best non-GT candidate cannot score highly by satisfying multiple requirements with the same broad, partially related statement.
"""

def build_rubric_extraction_prompt(
    task: str,
    paper_context: Dict,
    l2_id: str,
    l2_name: str,
    candidates: List[Dict],
    gt_text: str,
    claim_text: str = "",
) -> str:
    title = paper_context.get("title", "Unknown")
    abstract_snippet = paper_context.get("abstract", "")[:400]
    non_gt_candidates_text = _format_non_gt_candidates_block(candidates)

    if task == "task1":
        return TASK1_RUBRIC_EXTRACTION_PROMPT.format(
            title=title,
            l2_id=l2_id,
            l2_name=l2_name,
            abstract_snippet=abstract_snippet,
            gt_text=gt_text,
            non_gt_candidates_text=non_gt_candidates_text,
        )
    else:
        return TASK2_RUBRIC_EXTRACTION_PROMPT.format(
            title=title,
            l2_id=l2_id,
            l2_name=l2_name,
            claim_text=claim_text,
            abstract_snippet=abstract_snippet,
            gt_text=gt_text,
            non_gt_candidates_text=non_gt_candidates_text,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Rubric validation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_weights(soft_reqs: List[Dict]) -> List[Dict]:
    total = sum(r.get("weight", 0) for r in soft_reqs)
    if total == 0 and soft_reqs:
        for r in soft_reqs:
            r["weight"] = round(100.0 / len(soft_reqs), 2)
    elif total > 0:
        for r in soft_reqs:
            r["weight"] = round(r["weight"] * 100.0 / total, 2)
    return soft_reqs


def validate_rubric(rubric: Dict) -> bool:
    soft = rubric.get("soft_requirements", [])
    hard = rubric.get("hard_constraints", [])
    if not (5 <= len(soft) <= 6):
        print(f"    Rubric invalid: {len(soft)} soft reqs (need 5-6)")
        return False
    if len(hard) != 2:
        print(f"    Rubric invalid: {len(hard)} hard constraints (need 2)")
        return False
    for r in soft:
        for f in ["requirement", "weight", "type", "rationale"]:
            if f not in r:
                print(f"    Rubric invalid: soft req missing '{f}'")
                return False
        if r["type"] not in ("semantic", "format"):
            print(f"    Rubric invalid: unknown type '{r['type']}'")
            return False
    for r in hard:
        for f in ["requirement", "rationale"]:
            if f not in r:
                print(f"    Rubric invalid: hard constraint missing '{f}'")
                return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────────────

VERIFIER_PROMPT = """\
Generate a Python verification function for this requirement, OR return "NONE".

Requirement: {requirement}

Rules:
- ONLY write code for EXACT format/syntax checks (regex, counts, keyword presence)
- Do NOT write code for semantic meaning
- Function signature: def verify(text: str) -> bool
- Must not crash on any input
- Return "NONE" if not verifiable with simple code

Code:"""


def extract_verifier_code(content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    content = content.strip()
    if content.upper().startswith("NONE"):
        return None
    match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    code = match.group(1) if match else content
    if "def verify" not in code:
        return None
    try:
        compile(code, "<string>", "exec")
        env: Dict[str, Any] = {}
        exec(code, env)
        fn = env.get("verify")
        if not callable(fn):
            return None
        fn("test string")
        return code
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# File / JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_line(path: Path, row: Dict) -> None:
    """Append a single JSON object as one line (for incremental stage outputs)."""
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sft_spec_digest(specs: List[Dict]) -> str:
    ids = sorted(s["rubric_id"] for s in specs)
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def is_valid_sft_candidates(cands: Any) -> bool:
    if not isinstance(cands, list) or len(cands) != len(SFT_TEMPERATURES):
        return False
    for c in cands:
        if not isinstance(c, dict):
            return False
        if not (c.get("text") or "").strip():
            return False
        if c.get("temperature") is None:
            return False
    return True


def load_sft_cache(path: Path) -> Dict[str, List[Dict]]:
    if not path.exists():
        return {}
    out: Dict[str, List[Dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row["rubric_id"]
            out[rid] = row["candidates"]
    return out


def append_sft_cache_line(path: Path, rubric_id: str, candidates: List[Dict]) -> None:
    ensure_parent(path)
    row = {"rubric_id": rubric_id, "candidates": candidates}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sft_cache_compact(
    path: Path, specs: List[Dict], sft_candidates: Dict[str, List[Dict]]
) -> None:
    rows = []
    for spec in specs:
        rid = spec["rubric_id"]
        if rid in sft_candidates and is_valid_sft_candidates(sft_candidates[rid]):
            rows.append({"rubric_id": rid, "candidates": sft_candidates[rid]})
    write_jsonl(path, rows)


def write_sft_cache_meta(artifact_dir: Path, digest: str, n_specs: int) -> None:
    meta_path = artifact_dir / "sft_candidates.meta.json"
    write_json(
        meta_path,
        {"spec_rubric_ids_sha256": digest, "n_specs": n_specs},
    )


def read_sft_cache_meta(artifact_dir: Path) -> Optional[Dict]:
    meta_path = artifact_dir / "sft_candidates.meta.json"
    if not meta_path.exists():
        return None
    try:
        return read_json(meta_path)
    except Exception:
        return None


def load_specs(path: Path) -> List[Dict]:
    return load_jsonl(path)


def raise_if_jsonl_is_sft_cache_not_specs(specs: List[Dict]) -> None:
    """Catch common mistake: passing sft_candidates.jsonl as --specs."""
    if not specs:
        return
    s0 = specs[0]
    if (
        "candidates" in s0
        and isinstance(s0.get("candidates"), list)
        and "paper_context" not in s0
    ):
        raise RuntimeError(
            "The --specs file looks like sft_candidates.jsonl (each line has "
            "'rubric_id' + 'candidates'), not specs.jsonl from prepare_specs.py. "
            "Use the checklist-derived specs file (e.g. rubric_rl/specs_groups.jsonl) "
            "for --specs. The SFT cache file belongs in --output (stage1) or "
            "--sft_candidates (stage2/3)."
        )


def ensure_spec_prompt_messages(spec: Dict) -> None:
    """Fill spec['prompt_messages'] using the same builders as prepare_specs_from_groups."""
    if spec.get("prompt_messages"):
        return
    rid = spec.get("rubric_id", "<unknown>")
    task = spec.get("task")
    ctx = spec.get("paper_context")
    l2_id = spec.get("l2_id")
    l2_name = spec.get("l2_name", "")
    if ctx is None or not l2_id:
        raise RuntimeError(
            f"spec {rid!r} has no prompt_messages and cannot rebuild: "
            "need paper_context and l2_id"
        )
    if task == "task1":
        spec["prompt_messages"] = build_task1_messages(ctx, l2_id, l2_name)
    elif task == "task2":
        claim = (spec.get("claim_text") or "").strip()
        spec["prompt_messages"] = build_task2_messages(ctx, l2_id, l2_name, claim)
    else:
        raise RuntimeError(
            f"spec {rid!r} has no prompt_messages and invalid or missing task={task!r}"
        )


def ensure_specs_prompt_messages(specs: List[Dict]) -> None:
    raise_if_jsonl_is_sft_cache_not_specs(specs)
    for spec in specs:
        ensure_spec_prompt_messages(spec)


def write_specs(path: Path, specs: List[Dict]) -> None:
    write_jsonl(path, specs)


def load_gpt_candidates(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for row in load_jsonl(path):
        rid = row["rubric_id"]
        text = (row.get("text") or "").strip()
        if text:
            out[rid] = text
    return out


def write_gpt_candidates(path: Path, gpt_texts: Dict[str, str]) -> None:
    rows = [{"rubric_id": rid, "text": text} for rid, text in sorted(gpt_texts.items())]
    write_jsonl(path, rows)


def load_rubrics_file(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    for row in load_jsonl(path):
        rid = row["rubric_id"]
        if "rubric" in row:
            out[rid] = row["rubric"]
    return out


def write_rubrics_file(path: Path, rubrics: Dict[str, Dict]) -> None:
    rows = [{"rubric_id": rid, "rubric": rubric} for rid, rubric in sorted(rubrics.items())]
    write_jsonl(path, rows)


def load_stage3_final_records(path: Path) -> Dict[str, Dict]:
    """
    Load completed Stage-3 jsonl rows keyed by rubric_id.
    A row counts as completed if it has a non-empty ``rubric`` dict.
    """
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    for row in load_jsonl(path):
        rid = row.get("rubric_id")
        if not rid:
            continue
        rub = row.get("rubric")
        if isinstance(rub, dict) and rub:
            out[rid] = row
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Batch API helpers
# ─────────────────────────────────────────────────────────────────────────────

def verifier_api_temperature() -> Optional[float]:
    """
    Temperature for verifier code-generation calls.

    Some chat deployments (e.g. certain GPT-5 family models) reject ``temperature=0.0``
    and only allow the server default. In that case omit the parameter entirely.

    Env ``RUBRIC_VERIFIER_TEMPERATURE``:
      - unset, empty, or ``omit`` / ``default`` / ``none``: omit parameter (API default)
      - a number string (e.g. ``0``, ``0.0``, ``1``): send that value
    """
    raw = os.getenv("RUBRIC_VERIFIER_TEMPERATURE", "").strip().lower()
    if raw in ("", "omit", "default", "none"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _verifier_chat_completion_kwargs() -> Dict[str, Any]:
    t = verifier_api_temperature()
    if t is None:
        return {}
    return {"temperature": t}


def make_chat_batch_request(
    custom_id: str,
    messages: List[Dict],
    max_completion_tokens: int,
    temperature: Optional[float] = None,
    response_format: Optional[Dict] = None,
) -> Dict:
    body: Dict = {
        "model": get_openai_model_name(),
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def batch_meta_paths(artifact_dir: Path, batch_name: str) -> Dict[str, Path]:
    return {
        "request": artifact_dir / f"{batch_name}.requests.jsonl",
        "result": artifact_dir / f"{batch_name}.results.jsonl",
        "meta": artifact_dir / f"{batch_name}.meta.json",
    }


def download_batch_output_if_needed(client: OpenAIClient, batch, result_path: Path) -> None:
    if result_path.exists() and result_path.stat().st_size > 0:
        return
    if not getattr(batch, "output_file_id", None):
        raise RuntimeError("Batch completed but no output_file_id found.")
    result_bytes = client.files.content(batch.output_file_id).content
    ensure_parent(result_path)
    with open(result_path, "wb") as f:
        f.write(result_bytes)


def maybe_load_existing_completed_batch(
    client: OpenAIClient, meta_path: Path, result_path: Path
) -> Optional[Dict[str, Dict]]:
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    batch_id = meta.get("batch_id")
    if not batch_id:
        return None
    batch = client.batches.retrieve(batch_id)
    meta.update({
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    })
    write_json(meta_path, meta)
    if batch.status == "completed":
        download_batch_output_if_needed(client, batch, result_path)
        return {row["custom_id"]: row for row in load_jsonl(result_path)}
    return None


def _raise_batch_failure(client: OpenAIClient, batch_name: str, batch) -> None:
    detail = f"Batch {batch_name} ended with status={batch.status}"
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        try:
            detail += f"\n{client.files.content(error_file_id).content.decode('utf-8', errors='replace')}"
        except Exception as e:
            detail += f"\n(Could not read error file: {e})"
    raise RuntimeError(detail)


def run_chat_batch(
    client: OpenAIClient,
    requests: List[Dict],
    artifact_dir: Path,
    batch_name: str,
) -> Dict[str, Dict]:
    if not requests:
        return {}

    paths = batch_meta_paths(artifact_dir, batch_name)
    req_path = paths["request"]
    res_path = paths["result"]
    meta_path = paths["meta"]

    existing = maybe_load_existing_completed_batch(client, meta_path, res_path)
    if existing is not None:
        print(f"[Batch:{batch_name}] reusing completed batch")
        return existing

    write_jsonl(req_path, requests)
    request_sha = sha256_file(req_path)

    if meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("request_sha256") and meta["request_sha256"] != request_sha:
            raise RuntimeError(
                f"[Batch:{batch_name}] request hash mismatch. "
                "Use a new batch_name or delete existing meta/request files."
            )
        batch_id = meta.get("batch_id")
        if batch_id:
            print(f"[Batch:{batch_name}] resuming batch_id={batch_id}")
            while True:
                batch = client.batches.retrieve(batch_id)
                meta.update({
                    "status": batch.status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                })
                write_json(meta_path, meta)
                print(f"[Batch:{batch_name}] status={batch.status}")
                if batch.status == "completed":
                    download_batch_output_if_needed(client, batch, res_path)
                    return {row["custom_id"]: row for row in load_jsonl(res_path)}
                if batch.status in {"failed", "expired", "cancelled"}:
                    _raise_batch_failure(client, batch_name, batch)
                time.sleep(BATCH_POLL_SECONDS)

    print(f"[Batch:{batch_name}] uploading {len(requests)} requests ...")
    with open(req_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"name": batch_name},
    )
    meta = {
        "batch_name": batch_name,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "status": batch.status,
        "request_file": str(req_path),
        "result_file": str(res_path),
        "request_sha256": request_sha,
        "model": get_openai_model_name(),
        "completion_window": BATCH_COMPLETION_WINDOW,
        "created_at": int(time.time()),
    }
    write_json(meta_path, meta)
    print(f"[Batch:{batch_name}] created batch_id={batch.id}, polling ...")

    while True:
        batch = client.batches.retrieve(batch.id)
        meta.update({
            "status": batch.status,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
        })
        write_json(meta_path, meta)
        print(f"[Batch:{batch_name}] status={batch.status}")
        if batch.status == "completed":
            download_batch_output_if_needed(client, batch, res_path)
            return {row["custom_id"]: row for row in load_jsonl(res_path)}
        if batch.status in {"failed", "expired", "cancelled"}:
            _raise_batch_failure(client, batch_name, batch)
        time.sleep(BATCH_POLL_SECONDS)


def parse_batch_chat_content(result_row: Dict) -> Optional[str]:
    try:
        msg = result_row["response"]["body"]["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        refusal = msg.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return f"[REFUSAL] {refusal.strip()}"
        return None
    except Exception:
        return None


def parse_batch_error(result_row: Dict) -> Optional[str]:
    if result_row.get("error") is not None:
        return json.dumps(result_row["error"], ensure_ascii=False)
    try:
        body_error = result_row["response"]["body"].get("error")
        if body_error is not None:
            return json.dumps(body_error, ensure_ascii=False)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_groups(input_file: str) -> List[Dict]:
    groups, skipped = [], 0
    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARN line {i}: invalid JSON")
                skipped += 1
                continue

            if not rec.get("paper_id") or not rec.get("l2_id"):
                skipped += 1
                continue
            ctx = rec.get("paper_context", {})
            if not ctx.get("title") or not ctx.get("abstract"):
                skipped += 1
                continue
            valid_items = [
                it for it in rec.get("weakness_items", [])
                if it.get("original_weakness", "").strip() and it.get("enhanced_review")
            ]
            if not valid_items:
                skipped += 1
                continue
            rec["weakness_items"] = valid_items
            groups.append(rec)

    print(f"Loaded {len(groups)} valid groups, skipped {skipped}")
    return groups


def stratified_sample(groups: List[Dict], n: int) -> List[Dict]:
    import random
    if n <= 0 or not groups:
        return []
    by_l2: Dict[str, List] = defaultdict(list)
    for g in groups:
        by_l2[g["l2_id"]].append(g)
    per_l2 = max(1, n // len(by_l2))
    sampled = []
    for bucket in by_l2.values():
        sampled.extend(random.sample(bucket, min(per_l2, len(bucket))))
    remaining = n - len(sampled)
    if remaining > 0:
        sampled_ids = {id(g) for g in sampled}
        pool = [g for g in groups if id(g) not in sampled_ids]
        sampled.extend(random.sample(pool, min(remaining, len(pool))))
    return sampled[:n]

# ─────────────────────────────────────────────────────────────────────────────
# Spec preparation  — single pass per group
# ─────────────────────────────────────────────────────────────────────────────

def prepare_specs_from_groups(
    groups: List[Dict],
    completed_ids: set,
) -> List[Dict]:
    specs: List[Dict] = []
    for group in groups:
        paper_id = group["paper_id"]
        l2_id = group["l2_id"]
        l2_name = group.get("l2_name", "")
        ctx = group["paper_context"]
        items = group["weakness_items"]

        t1_rid = f"{paper_id}_{l2_id}_task1"
        if t1_rid not in completed_ids:
            specs.append({
                "rubric_id": t1_rid,
                "task": "task1",
                "paper_id": paper_id,
                "l2_id": l2_id,
                "l2_name": l2_name,
                "weakness_id": "group",
                "paper_context": ctx,
                "gt_text": items_to_task1_gt(items),
                "claim_text": "",
                "prompt_messages": build_task1_messages(ctx, l2_id, l2_name),
            })

        for item in items:
            t2_rid = f"{item['weakness_id']}_task2"
            if t2_rid in completed_ids:
                continue
            er = item["enhanced_review"]
            claim_text = (er.get("claim") or item.get("original_weakness", "")).strip()
            specs.append({
                "rubric_id": t2_rid,
                "task": "task2",
                "paper_id": paper_id,
                "l2_id": l2_id,
                "l2_name": l2_name,
                "weakness_id": item["weakness_id"],
                "paper_context": ctx,
                "gt_text": enhanced_review_to_text(er),
                "claim_text": claim_text,
                "prompt_messages": build_task2_messages(ctx, l2_id, l2_name, claim_text),
            })

    return specs

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_local_sft_candidates(
    specs: List[Dict],
    sft_model,
    sft_tokenizer,
    cache_path: Optional[Path] = None,
) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    labels = ["sft_conservative", "sft_balanced", "sft_creative"]
    for i, spec in enumerate(specs, 1):
        rid = spec["rubric_id"]
        print(f"  [SFT {i}/{len(specs)}] {rid}")
        candidates, ok = [], True
        for label, temp in zip(labels, SFT_TEMPERATURES):
            text = generate_with_sft(
                sft_model, sft_tokenizer, spec["prompt_messages"], temp
            )
            if not text:
                print(f"    Failed SFT {label} for {rid}")
                ok = False
                break
            candidates.append({"source": label, "temperature": temp, "text": text})
        if ok:
            out[rid] = candidates
            if cache_path is not None:
                append_sft_cache_line(cache_path, rid, candidates)
    return out


def _gpt_candidate_messages_and_budget(spec: Dict) -> Tuple[List[Dict], int]:
    if spec["task"] == "task1":
        return spec["prompt_messages"], GPT_CANDIDATE_MAX_TOKENS_TASK1

    compact_messages = build_task2_compact_gpt_messages(
        paper_context=spec["paper_context"],
        l2_id=spec["l2_id"],
        l2_name=spec["l2_name"],
        claim_text=spec["claim_text"],
    )
    return compact_messages, GPT_CANDIDATE_MAX_TOKENS_TASK2


def stage_batch_gpt_candidates(
    client: OpenAIClient,
    specs: List[Dict],
    sft_candidates: Dict[str, List[Dict]],
    artifact_dir: Path,
    batch_name: str = "stage1_gpt_candidates",
) -> Dict[str, str]:
    requests = []
    for spec in specs:
        rid = spec["rubric_id"]
        if rid not in sft_candidates:
            continue
        gpt_messages, max_tok = _gpt_candidate_messages_and_budget(spec)
        requests.append(
            make_chat_batch_request(
                custom_id=f"gpt_candidate::{rid}",
                messages=gpt_messages,
                temperature=1.0,
                max_completion_tokens=max_tok,
            )
        )

    results = run_chat_batch(client, requests, artifact_dir, batch_name)

    gpt_texts: Dict[str, str] = {}
    for spec in specs:
        rid = spec["rubric_id"]
        row = results.get(f"gpt_candidate::{rid}")
        if row is None:
            print(f"  Missing GPT candidate for {rid}")
            continue
        err = parse_batch_error(row)
        if err is not None:
            print(f"  GPT candidate error for {rid}: {err}")
            continue

        text = parse_batch_chat_content(row)
        if text:
            gpt_texts[rid] = text
        else:
            try:
                choice = row["response"]["body"]["choices"][0]
                msg = choice.get("message", {})
                print(
                    f"  GPT candidate empty for {rid} | "
                    f"finish_reason={choice.get('finish_reason')} | "
                    f"content={repr(msg.get('content'))} | "
                    f"refusal={repr(msg.get('refusal'))}"
                )
            except Exception:
                print(f"  GPT candidate empty for {rid}")
    return gpt_texts


def _openai_chat_completion_message_text(resp: Any) -> str:
    """Extract assistant text from a Chat Completions API response object."""
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg is None:
            return ""
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts).strip()
    except Exception:
        return ""
    return ""


def stage_sync_openai_gpt_candidates(
    client: OpenAIClient,
    specs: List[Dict],
    sft_candidates: Dict[str, List[Dict]],
    *,
    model: Optional[str] = None,
    max_retries: int = 3,
    retry_sleep_seconds: float = 5.0,
    request_sleep_seconds: float = 0.0,
) -> Dict[str, str]:
    """
    Same prompts/budgets as ``stage_batch_gpt_candidates``, but uses the Chat
    Completions API (one synchronous request per spec).
    """
    resolved_model = model or get_openai_model_name()
    out: Dict[str, str] = {}
    for i, spec in enumerate(specs, 1):
        rid = spec["rubric_id"]
        if rid not in sft_candidates:
            continue
        gpt_messages, max_tok = _gpt_candidate_messages_and_budget(spec)
        print(f"  [OpenAI chat {i}/{len(specs)}] {rid}")
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=resolved_model,
                    messages=gpt_messages,
                    temperature=1.0,
                    max_completion_tokens=max_tok,
                )
                text = _openai_chat_completion_message_text(resp)
                if text:
                    out[rid] = text
                else:
                    print(f"    Empty OpenAI output for {rid} (attempt {attempt})")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"    attempt {attempt} failed: {e!r}")
                if attempt < max_retries:
                    time.sleep(retry_sleep_seconds)
        if last_err is not None:
            print(
                f"  OpenAI candidate failed for {rid} after {max_retries} attempt(s)"
            )

        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)
    return out


def openai_chat_messages_to_anthropic(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Split OpenAI-style chat messages into Anthropic ``system`` + ``messages``."""
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": content})
        else:
            out.append({"role": "user", "content": content})
    system = "\n\n".join(system_parts) if system_parts else ""
    return system, out


def _anthropic_message_text(message: Any) -> str:
    parts: List[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def stage_claude_haiku_candidates(
    client: Any,
    specs: List[Dict],
    sft_candidates: Dict[str, List[Dict]],
    model: str,
    *,
    max_retries: int = 3,
    retry_sleep_seconds: float = 5.0,
    request_sleep_seconds: float = 0.0,
) -> Dict[str, str]:
    """
    Generate GPT-slot candidates via Anthropic Messages API (sync, one request per spec).
    Reuses the same prompts as ``stage_batch_gpt_candidates``.
    """
    out: Dict[str, str] = {}
    for i, spec in enumerate(specs, 1):
        rid = spec["rubric_id"]
        if rid not in sft_candidates:
            continue
        gpt_messages, max_tok = _gpt_candidate_messages_and_budget(spec)
        system, msgs = openai_chat_messages_to_anthropic(gpt_messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tok,
            "temperature": 1.0,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system

        print(f"  [Claude {i}/{len(specs)}] {rid}")
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.messages.create(**kwargs)
                text = _anthropic_message_text(resp)
                if text:
                    out[rid] = text
                else:
                    print(f"    Empty Claude output for {rid} (attempt {attempt})")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"    attempt {attempt} failed: {e!r}")
                if attempt < max_retries:
                    time.sleep(retry_sleep_seconds)
        if last_err is not None:
            print(f"  Claude candidate failed for {rid} after {max_retries} attempt(s)")

        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)
    return out


def stage_batch_rubrics(
    client: OpenAIClient,
    specs: List[Dict],
    sft_candidates: Dict[str, List[Dict]],
    gpt_texts: Dict[str, str],
    artifact_dir: Path,
    batch_name: str = "stage2_rubric_extraction",
) -> Dict[str, Dict]:
    requests = []
    for spec in specs:
        rid = spec["rubric_id"]
        if rid not in sft_candidates or rid not in gpt_texts:
            continue
        candidates = list(sft_candidates[rid]) + [
            {"source": "gpt", "temperature": 1.0, "text": gpt_texts[rid]},
            {"source": "gt", "temperature": None, "text": spec["gt_text"]},
        ]
        prompt = build_rubric_extraction_prompt(
            task=spec["task"],
            paper_context=spec["paper_context"],
            l2_id=spec["l2_id"],
            l2_name=spec["l2_name"],
            candidates=candidates,
            gt_text=spec["gt_text"],
            claim_text=spec["claim_text"],
        )
        requests.append(
            make_chat_batch_request(
                custom_id=f"rubric::{rid}",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
        )

    results = run_chat_batch(client, requests, artifact_dir, batch_name)

    rubrics: Dict[str, Dict] = {}
    for spec in specs:
        rid = spec["rubric_id"]
        row = results.get(f"rubric::{rid}")
        if row is None:
            continue
        err = parse_batch_error(row)
        if err is not None:
            print(f"  Rubric error {rid}: {err}")
            continue
        text = parse_batch_chat_content(row)
        if not text:
            print(f"  Rubric empty {rid}")
            continue
        try:
            rubric = json.loads(text)
            rubric["soft_requirements"] = normalize_weights(
                rubric.get("soft_requirements", [])
            )
            if validate_rubric(rubric):
                for req in rubric["soft_requirements"]:
                    req.setdefault("verifier_code", None)
                rubrics[rid] = rubric
            else:
                print(f"  Invalid rubric for {rid}")
        except Exception as e:
            print(f"  Rubric JSON parse error {rid}: {e}")
    return rubrics


def _parse_verifier_custom_id(custom_id: str) -> Tuple[str, int]:
    parts = custom_id.split("::")
    idx = int(parts[-1])
    rid = "::".join(parts[1:-1])
    return rid, idx


def stage_batch_verifiers(
    client: OpenAIClient,
    rubrics: Dict[str, Dict],
    artifact_dir: Path,
    batch_name: str = "stage3_verifier_generation",
) -> Dict[Tuple[str, int], Optional[str]]:
    verifier_map: Dict[Tuple[str, int], Optional[str]] = {}
    requests = []
    for rid, rubric in rubrics.items():
        for idx, req in enumerate(rubric["soft_requirements"]):
            if req["type"] == "format":
                verifier_map[(rid, idx)] = None
                requests.append(
                    make_chat_batch_request(
                        custom_id=f"verifier::{rid}::{idx}",
                        messages=[{"role": "user", "content": VERIFIER_PROMPT.format(
                            requirement=req["requirement"]
                        )}],
                        max_completion_tokens=512,
                        temperature=verifier_api_temperature(),
                    )
                )
    if not requests:
        return verifier_map

    results = run_chat_batch(client, requests, artifact_dir, batch_name)
    for custom_id, row in results.items():
        rid, idx = _parse_verifier_custom_id(custom_id)
        err = parse_batch_error(row)
        if err is not None:
            print(f"  Verifier error {custom_id}: {err}")
            continue
        verifier_map[(rid, idx)] = extract_verifier_code(parse_batch_chat_content(row))
    return verifier_map


def _sync_chat_text(resp: Any) -> str:
    """Plain text from chat.completions response."""
    try:
        choice = resp.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content.strip()
        if content is not None:
            return str(content).strip()
    except Exception:
        pass
    return ""


def sync_extract_rubric_one_spec(
    client: OpenAIClient,
    spec: Dict,
    sft_candidates: Dict[str, List[Dict]],
    gpt_texts: Dict[str, str],
    *,
    max_retries: int = 5,
    retry_sleep_seconds: float = 10.0,
    request_sleep_seconds: float = 0.0,
) -> Optional[Dict]:
    """
    Extract and validate a single rubric dict for one spec (sync Chat Completions).
    Returns None on failure.
    """
    rid = spec["rubric_id"]
    if rid not in sft_candidates or rid not in gpt_texts:
        return None
    candidates = list(sft_candidates[rid]) + [
        {"source": "gpt", "temperature": 1.0, "text": gpt_texts[rid]},
        {"source": "gt", "temperature": None, "text": spec["gt_text"]},
    ]
    prompt = build_rubric_extraction_prompt(
        task=spec["task"],
        paper_context=spec["paper_context"],
        l2_id=spec["l2_id"],
        l2_name=spec["l2_name"],
        candidates=candidates,
        gt_text=spec["gt_text"],
        claim_text=spec["claim_text"],
    )
    text = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=get_openai_model_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
            text = _sync_chat_text(resp)
            if text:
                break
            print(f"    Empty rubric response (attempt {attempt})")
        except Exception as e:
            print(f"    Rubric API attempt {attempt} failed: {e!r}")
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)
    if not text:
        print(f"  Rubric failed for {rid} after {max_retries} attempt(s)")
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)
        return None
    try:
        rubric = json.loads(text)
        rubric["soft_requirements"] = normalize_weights(
            rubric.get("soft_requirements", [])
        )
        if validate_rubric(rubric):
            for req in rubric["soft_requirements"]:
                req.setdefault("verifier_code", None)
            if request_sleep_seconds > 0:
                time.sleep(request_sleep_seconds)
            return rubric
        print(f"  Invalid rubric for {rid}")
    except Exception as e:
        print(f"  Rubric JSON parse error {rid}: {e}")
    if request_sleep_seconds > 0:
        time.sleep(request_sleep_seconds)
    return None


def stage_sync_rubrics(
    client: OpenAIClient,
    specs: List[Dict],
    sft_candidates: Dict[str, List[Dict]],
    gpt_texts: Dict[str, str],
    *,
    max_retries: int = 5,
    retry_sleep_seconds: float = 10.0,
    request_sleep_seconds: float = 0.0,
) -> Dict[str, Dict]:
    """
    Same as ``stage_batch_rubrics`` but uses synchronous ``chat.completions.create``
    (no Files / Batch API). For gateways that only support chat or Responses.
    """
    rubrics: Dict[str, Dict] = {}
    n = len(specs)
    for i, spec in enumerate(specs, 1):
        rid = spec["rubric_id"]
        print(f"  [Rubric {i}/{n}] {rid}")
        rubric = sync_extract_rubric_one_spec(
            client,
            spec,
            sft_candidates,
            gpt_texts,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            request_sleep_seconds=request_sleep_seconds,
        )
        if rubric is not None:
            rubrics[rid] = rubric
    return rubrics


def stage_sync_verifiers(
    client: OpenAIClient,
    rubrics: Dict[str, Dict],
    *,
    max_retries: int = 5,
    retry_sleep_seconds: float = 10.0,
    request_sleep_seconds: float = 0.0,
) -> Dict[Tuple[str, int], Optional[str]]:
    """Same as ``stage_batch_verifiers`` but synchronous chat completions."""
    verifier_map: Dict[Tuple[str, int], Optional[str]] = {}
    jobs: List[Tuple[str, int, str]] = []
    for rid, rubric in rubrics.items():
        for idx, req in enumerate(rubric["soft_requirements"]):
            if req["type"] == "format":
                verifier_map[(rid, idx)] = None
                user_msg = VERIFIER_PROMPT.format(requirement=req["requirement"])
                jobs.append((rid, idx, user_msg))

    if not jobs:
        return verifier_map

    total = len(jobs)
    for j, (rid, idx, user_msg) in enumerate(jobs, 1):
        print(f"  [Verifier {j}/{total}] {rid} idx={idx}")
        code: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=get_openai_model_name(),
                    messages=[{"role": "user", "content": user_msg}],
                    max_completion_tokens=512,
                    **_verifier_chat_completion_kwargs(),
                )
                raw = _sync_chat_text(resp)
                if raw:
                    code = extract_verifier_code(raw)
                    break
                print(f"    Empty verifier response (attempt {attempt})")
            except Exception as e:
                print(f"    Verifier API attempt {attempt} failed: {e!r}")
                if attempt < max_retries:
                    time.sleep(retry_sleep_seconds)
        verifier_map[(rid, idx)] = code
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)
    return verifier_map


# ─────────────────────────────────────────────────────────────────────────────
# Final assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_final_records(
    specs: List[Dict],
    sft_candidates: Dict[str, List[Dict]],
    gpt_texts: Dict[str, str],
    rubrics: Dict[str, Dict],
    verifier_map: Dict[Tuple[str, int], Optional[str]],
    save_candidates: bool,
) -> List[Dict]:
    final = []
    for spec in specs:
        rid = spec["rubric_id"]
        if rid not in sft_candidates or rid not in gpt_texts or rid not in rubrics:
            continue
        candidates = list(sft_candidates[rid]) + [
            {"source": "gpt", "temperature": 1.0, "text": gpt_texts[rid]},
            {"source": "gt", "temperature": None, "text": spec["gt_text"]},
        ]
        rubric = rubrics[rid]
        for idx, req in enumerate(rubric["soft_requirements"]):
            if req["type"] == "format":
                req["verifier_code"] = verifier_map.get((rid, idx))

        final.append({
            "rubric_id": rid,
            "task": spec["task"],
            "paper_id": spec["paper_id"],
            "l2_id": spec["l2_id"],
            "l2_name": spec["l2_name"],
            "weakness_id": spec["weakness_id"],
            "prompt_messages": spec["prompt_messages"],
            "gt_text": spec["gt_text"],
            "rubric": rubric,
            "candidates": candidates if save_candidates else None,
        })
    return final
