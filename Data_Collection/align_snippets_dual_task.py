"""
Purpose
-------
Align top-k PDF snippets for each enhanced-review record, with separate retrieval
logic for Task 1 and Task 2.

Dependencies
------------
pip install pymupdf requests tqdm
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import requests
from tqdm import tqdm


# ============================================================
# Global lexicons
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by",
    "with", "from", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "their", "them", "as", "if", "then",
    "than", "into", "about", "around", "under", "over", "after", "before",
    "paper", "new", "add", "include", "provide", "clarify", "using", "used",
    "use", "show", "shows", "shown", "based", "section", "table", "figure"
}

SECTION_HINT_WORDS = [
    "abstract", "introduction", "background", "method", "methods", "methodology",
    "approach", "model", "algorithm", "training", "experiment", "experiments",
    "evaluation", "results", "discussion", "conclusion", "appendix",
    "implementation", "setup", "hyperparameter", "ablation", "analysis",
    "related work", "limitation", "limitations", "proof", "theory", "theoretical",
    "table", "protocol", "fine-tuning", "finetuning", "baseline",
    "decoder", "architecture", "pipeline", "novelty", "contribution", "value",
    "robustness", "accuracy", "corruption", "significance", "kl", "divergence",
    "lemma", "proposition", "theorem", "equation", "eq", "variance", "comparison",
    "supplementary", "calibration", "reproducibility", "statistical"
]

NEGATIVE_SECTION_WORDS = [
    "acknowledgements", "references", "possible negative societal impacts",
    "negative societal impacts", "societal impacts", "environment"
]

FAILED_PDF_URLS = set()


# ============================================================
# Text normalization / tokenization
# ============================================================

def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s\-\.\(\)/%λ]", " ", text)
    toks = []
    for tok in text.split():
        tok = tok.strip(".,:;!?()[]{}\"'")
        if len(tok) < 2:
            continue
        if tok in STOPWORDS:
            continue
        toks.append(tok)
    return toks


def safe_join(parts: List[str]) -> str:
    return "\n".join([p for p in parts if p and str(p).strip()])


# ============================================================
# Record field readers
# ============================================================

def get_title_text(record: Dict[str, Any]) -> str:
    return ((record.get("paper_context") or {}).get("title") or "").strip()


def get_abstract_text(record: Dict[str, Any]) -> str:
    return ((record.get("paper_context") or {}).get("abstract") or "").strip()


def get_keywords_text(record: Dict[str, Any]) -> str:
    kws = ((record.get("paper_context") or {}).get("keywords") or [])
    if isinstance(kws, list):
        return " ".join(str(x) for x in kws)
    return str(kws)


def get_original_weakness(record: Dict[str, Any]) -> str:
    return (record.get("original_weakness") or "").strip()


def get_l2_name_text(record: Dict[str, Any]) -> str:
    wc = record.get("weakness_category") or {}
    return (wc.get("l2_name") or record.get("l2_name") or "").strip()


def get_l1_name_text(record: Dict[str, Any]) -> str:
    wc = record.get("weakness_category") or {}
    return (wc.get("l1_name") or record.get("l1_name") or "").strip()


def get_claim_text(record: Dict[str, Any]) -> str:
    enhanced = record.get("enhanced_review") or {}
    return (enhanced.get("claim") or "").strip()


def get_evidence_text(record: Dict[str, Any]) -> str:
    enhanced = record.get("enhanced_review") or {}
    return (enhanced.get("evidence") or "").strip()


def get_suggestions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    enhanced = record.get("enhanced_review") or {}
    sugs = enhanced.get("actionable_suggestions") or []
    return sugs if isinstance(sugs, list) else []


def get_severity_text(record: Dict[str, Any]) -> str:
    enhanced = record.get("enhanced_review") or {}
    return (enhanced.get("severity") or "").strip()


def get_followups_text(record: Dict[str, Any]) -> str:
    follow_ups = record.get("follow_ups") or []
    parts = []
    for fu in follow_ups:
        if isinstance(fu, dict):
            t = fu.get("text")
            if t:
                parts.append(str(t))
        elif isinstance(fu, str):
            parts.append(fu)
    return "\n".join(parts).strip()


def get_rebuttals_text(record: Dict[str, Any]) -> str:
    rebs = record.get("rebuttals") or []
    if isinstance(rebs, list):
        return "\n".join(str(x) for x in rebs if str(x).strip())
    return str(rebs)


# ============================================================
# Suggestion field readers
# ============================================================

def get_suggestion_fields(record: Dict[str, Any]) -> Dict[str, str]:
    sugs = get_suggestions(record)
    what_parts, where_parts, how_parts, outcome_parts, priority_parts = [], [], [], [], []

    for s in sugs:
        if not isinstance(s, dict):
            continue
        if s.get("what"):
            what_parts.append(str(s["what"]))
        if s.get("where"):
            where_parts.append(str(s["where"]))
        if s.get("how"):
            how_parts.append(str(s["how"]))
        if s.get("expected_outcome"):
            outcome_parts.append(str(s["expected_outcome"]))
        if s.get("priority"):
            priority_parts.append(str(s["priority"]))

    return {
        "what": "\n".join(what_parts).strip(),
        "where": "\n".join(where_parts).strip(),
        "how": "\n".join(how_parts).strip(),
        "expected_outcome": "\n".join(outcome_parts).strip(),
        "priority": "\n".join(priority_parts).strip(),
    }


# ============================================================
# PDF download / extraction
# ============================================================

def derive_openreview_pdf_fallbacks(record: Dict[str, Any]) -> List[str]:
    paper_context = record.get("paper_context") or {}
    web_url = (paper_context.get("web_url") or "").strip()
    paper_id = (record.get("weakness_id") or "").split("_")[0].strip()

    fallbacks = []
    if paper_id:
        fallbacks.append(f"https://openreview.net/pdf?id={paper_id}")

    m = re.search(r"[?&]id=([^&]+)", web_url)
    if m:
        fallbacks.append(f"https://openreview.net/pdf?id={m.group(1)}")

    deduped = []
    seen = set()
    for x in fallbacks:
        if x and x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def download_pdf(pdf_urls: List[str], cache_dir: Path, timeout: int = 60) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://openreview.net/",
    })

    errors = []
    for pdf_url in pdf_urls:
        if not pdf_url:
            continue
        if pdf_url in FAILED_PDF_URLS:
            continue

        filename = hashlib.md5(pdf_url.encode("utf-8")).hexdigest() + ".pdf"
        pdf_path = cache_dir / filename

        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return pdf_path

        try:
            resp = session.get(pdf_url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").lower()
            if "pdf" not in content_type:
                raise RuntimeError(f"unexpected content-type: {resp.headers.get('content-type')}")
            pdf_path.write_bytes(resp.content)
            return pdf_path
        except Exception as e:
            FAILED_PDF_URLS.add(pdf_url)
            errors.append(f"{pdf_url} -> {e}")

    raise RuntimeError("all pdf urls failed: " + " | ".join(errors[:3]))


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


def split_into_chunks(page_num: int, text: str, max_chars: int = 1400) -> List[Dict[str, Any]]:
    if not text:
        return []

    raw_parts = re.split(r"\n\s*\n", text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()]

    chunks = []
    buf = []
    cur_len = 0

    for part in raw_parts:
        if len(part) > max_chars:
            sents = re.split(r"(?<=[.!?])\s+", part)
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                if cur_len + len(sent) + 1 > max_chars and buf:
                    chunks.append(" ".join(buf).strip())
                    buf = [sent]
                    cur_len = len(sent)
                else:
                    buf.append(sent)
                    cur_len += len(sent) + 1
            continue

        if cur_len + len(part) + 2 > max_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf = [part]
            cur_len = len(part)
        else:
            buf.append(part)
            cur_len += len(part) + 2

    if buf:
        chunks.append(" ".join(buf).strip())

    out = []
    for ch in chunks:
        if ch and len(ch) >= 80:
            out.append({"page": page_num, "text": ch})
    return out


def build_chunks_from_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for page in pages:
        chunks.extend(split_into_chunks(page["page_num"], page["text"]))
    return chunks


# ============================================================
# Chunk typing
# ============================================================

def infer_chunk_type(text: str) -> str:
    t = normalize_text(text)

    if any(x in t for x in ["table ", "table:", "tab.", "apbox", "apmask", "aa (%)", "sa (%)"]):
        return "table_result"
    if any(x in t for x in ["algorithm ", "proposition", "lemma", "theorem", "proof", "eq.", "equation", "kl("]):
        return "method"
    if any(x in t for x in ["implementation details", "batch size", "optimizer", "epochs", "learning rate"]):
        return "implementation"
    if any(x in t for x in ["evaluation protocols", "protocol", "test accuracy", "autoattack", "robustbench"]):
        return "protocol"
    if any(x in t for x in ["experiments", "results", "ablation", "analysis", "appendix c.", "appendix b."]):
        return "analysis"
    return "generic"


# ============================================================
# Weakness-type detectors
# ============================================================

def weakness_blob(record: Dict[str, Any], include_suggestions: bool = False) -> str:
    sf = get_suggestion_fields(record)
    parts = [
        get_l1_name_text(record),
        get_l2_name_text(record),
        get_original_weakness(record),
        get_claim_text(record),
        get_evidence_text(record),
        get_followups_text(record),
        get_rebuttals_text(record),
    ]
    if include_suggestions:
        parts.extend([sf["what"], sf["where"], sf["how"]])
    return normalize_text(" ".join(parts))


def is_novelty_case(record: Dict[str, Any]) -> bool:
    blob = weakness_blob(record, include_suggestions=False)
    return any(k in blob for k in [
        "novelty", "originality", "incremental", "overstated novelty",
        "weak novelty", "positioning", "prior work", "existing theorem",
        "existing invariant", "relative to prior", "lack of originality",
        "contribution seem", "contribution appears"
    ])


def is_theory_case(record: Dict[str, Any]) -> bool:
    blob = weakness_blob(record, include_suggestions=False)
    return any(k in blob for k in [
        "theoretical", "theory", "theorem", "lemma", "proposition", "proof",
        "decomposition", "assumption", "generalization guarantees",
        "justification", "theoretical soundness"
    ])


def is_method_clarity_case(record: Dict[str, Any]) -> bool:
    blob = weakness_blob(record, include_suggestions=False)
    return any(k in blob for k in [
        "unclear", "incomplete method", "methodological clarity", "reproduce",
        "reproducibility", "notation", "definition", "inputs", "input types",
        "kl divergence", "equation", "algorithm", "implementation detail"
    ])


def is_empirical_case(record: Dict[str, Any]) -> bool:
    blob = weakness_blob(record, include_suggestions=False)
    return any(k in blob for k in [
        "baseline", "comparison", "accuracy", "results", "empirical",
        "significance", "p-value", "variance", "small gains", "robustness",
        "corruption", "evaluation", "protocol", "fine-tuning", "finetuning",
        "source and correctness", "baseline results"
    ])


def is_ablation_case(record: Dict[str, Any]) -> bool:
    blob = weakness_blob(record, include_suggestions=True)
    return any(k in blob for k in [
        "ablation", "sir-only", "air-only", "sir+air", "combining sir and air",
        "why both", "why not solely", "why not only", "necessary to employ both",
        "joint contributions", "individual and joint contributions",
        "isolate the contributions", "variants", "lambda1", "lambda2", "λ1", "λ2"
    ])


def is_mechanism_case(record: Dict[str, Any]) -> bool:
    blob = weakness_blob(record, include_suggestions=False)
    return any(k in blob for k in [
        "why the proposed method", "why models trained", "mechanism", "translate into",
        "generalize well to common corruption", "common corruption benchmarks",
        "why enforcing", "why this should", "clear justification", "mechanistic"
    ])


# ============================================================
# Section hints / page hints
# ============================================================

def extract_section_hints(record: Dict[str, Any], mode: str) -> List[str]:
    pieces = [
        get_claim_text(record),
        get_evidence_text(record),
        get_original_weakness(record),
        get_l2_name_text(record),
        get_l1_name_text(record),
        get_followups_text(record),
        get_rebuttals_text(record),
    ]
    if mode == "task2_support":
        sf = get_suggestion_fields(record)
        pieces.extend([sf["what"], sf["where"], sf["how"]])

    blob = normalize_text(" ".join(pieces))
    hints = set()
    for w in SECTION_HINT_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", blob):
            hints.add(w)
    return sorted(hints)


def extract_explicit_page_hints(record: Dict[str, Any], mode: str) -> List[int]:
    pieces = [
        get_claim_text(record),
        get_evidence_text(record),
        get_original_weakness(record),
        get_followups_text(record),
    ]
    if mode == "task2_support":
        sf = get_suggestion_fields(record)
        pieces.extend([sf["where"]])

    blob = " ".join(pieces)
    pages = []
    for m in re.finditer(r"\bp(?:age)?\.?\s*(\d+)\b", blob, re.IGNORECASE):
        try:
            pages.append(int(m.group(1)))
        except Exception:
            pass
    return sorted(set(pages))


def fuzzy_section_page_hints(record: Dict[str, Any], mode: str) -> List[int]:
    sections_index = ((record.get("paper_context") or {}).get("sections_index") or [])
    if not sections_index:
        return []

    pieces = [
        get_claim_text(record),
        get_evidence_text(record),
        get_original_weakness(record),
        get_l2_name_text(record),
        get_l1_name_text(record),
        get_followups_text(record),
        get_rebuttals_text(record),
    ]
    if mode == "task2_support":
        sf = get_suggestion_fields(record)
        pieces.extend([sf["where"], sf["what"], sf["how"]])

    blob = normalize_text(" ".join(pieces))
    pages = set()

    for sec in sections_index:
        sec_name = normalize_text(sec.get("name", ""))
        sec_page = sec.get("page")
        if not sec_name or not sec_page:
            continue
        sec_tokens = set(tokenize(sec_name))
        if not sec_tokens:
            continue
        overlap = sum(1 for tok in sec_tokens if tok in blob)
        if overlap > 0:
            try:
                pages.add(int(sec_page))
            except Exception:
                pass

    return sorted(pages)


def section_name_bias(record: Dict[str, Any], chunk_page: int, mode: str) -> float:
    sections_index = ((record.get("paper_context") or {}).get("sections_index") or [])
    if not sections_index:
        return 0.0

    preferred = []
    discouraged = [normalize_text(x) for x in NEGATIVE_SECTION_WORDS]

    is_method = is_method_clarity_case(record)
    is_experiment = is_empirical_case(record)
    is_novelty = is_novelty_case(record)
    is_theory = is_theory_case(record)
    is_ablation = is_ablation_case(record)
    is_mechanism = is_mechanism_case(record)

    is_claim_support = mode in {"task1", "task2_evidence"}
    is_suggestion_support = (mode == "task2_support")

    if is_method and is_claim_support:
        preferred += ["methodology", "method", "methods", "analysis", "proof", "appendix", "implementation"]
    if is_experiment and is_claim_support:
        preferred += ["experiments", "results", "evaluation", "analysis", "implementation", "protocol", "appendix"]
    if is_novelty and is_claim_support:
        preferred += ["related work", "methodology", "analysis", "proof", "appendix"]
    if is_theory and is_claim_support:
        preferred += ["methodology", "analysis", "proof", "appendix", "related work"]
    if is_ablation and is_claim_support:
        preferred += ["experiments", "analysis", "appendix", "ablation", "implementation"]
    if is_mechanism and is_claim_support:
        preferred += ["methodology", "analysis", "experiments", "appendix", "related work"]
    if is_suggestion_support:
        preferred += ["implementation", "experiments", "results", "appendix", "methodology", "analysis"]

    preferred = [normalize_text(x) for x in preferred]

    score = 0.0
    for sec in sections_index:
        sec_name = normalize_text(sec.get("name", ""))
        sec_page = sec.get("page")
        if not sec_name or sec_page is None:
            continue
        try:
            sec_page = int(sec_page)
        except Exception:
            continue

        page_dist = abs(chunk_page - sec_page)
        local = 0.0

        for p in preferred:
            if p and p in sec_name:
                if page_dist == 0:
                    local = max(local, 0.18)
                elif page_dist == 1:
                    local = max(local, 0.10)

        for d in discouraged:
            if d and d in sec_name:
                if page_dist == 0:
                    local = min(local, -0.18)
                elif page_dist == 1:
                    local = min(local, -0.08)

        score += local

    return score


def page_hint_score(chunk_page: int, hinted_pages: List[int]) -> float:
    if not hinted_pages:
        return 0.0
    best = min(abs(chunk_page - p) for p in hinted_pages)
    if best == 0:
        return 0.18
    if best == 1:
        return 0.10
    if best == 2:
        return 0.05
    return 0.0


# ============================================================
# Query builders
# ============================================================

def task1_query_fields(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        "claim": get_claim_text(record),
        "evidence": get_evidence_text(record),
        "weakness": get_original_weakness(record),
        "followups": get_followups_text(record),
        "meta": safe_join([
            get_title_text(record),
            get_l1_name_text(record),
            get_l2_name_text(record),
            get_keywords_text(record),
        ])
    }


def task2_evidence_query_fields(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        "claim": get_claim_text(record),
        "evidence": get_evidence_text(record),
        "weakness": get_original_weakness(record),
        "followups": get_followups_text(record),
        "rebuttals": get_rebuttals_text(record),
        "meta": safe_join([
            get_title_text(record),
            get_l1_name_text(record),
            get_l2_name_text(record),
            get_keywords_text(record),
        ])
    }


def task2_support_query_fields(record: Dict[str, Any]) -> Dict[str, str]:
    sf = get_suggestion_fields(record)
    return {
        "claim": get_claim_text(record),
        "evidence": get_evidence_text(record),
        "what": sf["what"],
        "where": sf["where"],
        "how": sf["how"],
        "outcome": sf["expected_outcome"],
        "priority": sf["priority"],
        "meta": safe_join([
            get_title_text(record),
            get_l1_name_text(record),
            get_l2_name_text(record),
            get_keywords_text(record),
        ])
    }


# ============================================================
# Lexical / phrase / anchor scoring
# ============================================================

def lexical_overlap_score(query_tokens: List[str], chunk_text: str) -> float:
    chunk_tokens = set(tokenize(chunk_text))
    query_token_set = set(query_tokens)
    if not query_token_set or not chunk_tokens:
        return 0.0
    overlap = sum(1 for tok in query_token_set if tok in chunk_tokens)
    return overlap / max(1, len(query_token_set))


def phrase_overlap_score(query_text: str, chunk_text: str) -> float:
    q = normalize_text(query_text)
    c = normalize_text(chunk_text)
    if not q or not c:
        return 0.0

    phrases = []
    for part in re.split(r"[.;:\n]", q):
        part = normalize_text(part)
        if 2 <= len(part.split()) <= 18 and len(part) <= 120:
            phrases.append(part)

    phrases = phrases[:18]
    hits = sum(1 for p in phrases if p and p in c)
    return min(0.30, 0.05 * hits)


def section_hint_score(chunk_text: str, section_hints: List[str]) -> float:
    if not section_hints:
        return 0.0
    lower = normalize_text(chunk_text)
    hits = sum(1 for hint in section_hints if hint in lower)
    return min(0.20, 0.03 * hits)


def extract_structural_anchors(text: str) -> List[str]:
    text = text or ""
    lowered = normalize_text(text)

    patterns = [
        r"\bequation\s*\(?\d+(?:\.\d+)?\)?",
        r"\beqn?\.?\s*\(?\d+(?:\.\d+)?\)?",
        r"\beq\.\s*\(?\d+(?:\.\d+)?\)?",
        r"\bfigure\s*\d+",
        r"\bfig\.\s*\d+",
        r"\btable\s*\d+",
        r"\btab\.\s*\d+",
        r"\bsection\s*\d+(?:\.\d+)*",
        r"\bsec\.\s*\d+(?:\.\d+)*",
        r"\balgorithm\s*\d+",
        r"\bappendix\s*[a-z](?:\.\d+)?",
        r"\bproposition\s*\d+",
        r"\blemma\s*\d+",
        r"\btheorem\s*\d+",
        r"\bablation\b",
        r"\brelated work\b",
        r"\bkl\b",
        r"\bautoattack\b",
        r"\brobustbench\b",
        r"\bclip\b",
        r"\bvit-b\b",
        r"\bvit-l\b",
        r"\bsir\b",
        r"\bair\b",
        r"\bλ1\b",
        r"\bλ2\b",
        r"\blambda1\b",
        r"\blambda2\b",
        r"\bp-value\b",
        r"\bt-test\b",
    ]

    matches = []
    for pat in patterns:
        for m in re.finditer(pat, lowered, flags=re.IGNORECASE):
            matches.append(normalize_text(m.group(0)))

    deduped = []
    seen = set()
    for a in matches:
        if a and a not in seen:
            deduped.append(a)
            seen.add(a)
    return deduped


def has_anchor_match(record: Dict[str, Any], chunk_text: str, mode: str) -> bool:
    if mode == "task1":
        source = safe_join([
            get_claim_text(record),
            get_evidence_text(record),
            get_original_weakness(record),
            get_followups_text(record),
        ])
    elif mode == "task2_evidence":
        source = safe_join([
            get_claim_text(record),
            get_evidence_text(record),
            get_original_weakness(record),
            get_followups_text(record),
            get_rebuttals_text(record),
        ])
    else:
        sf = get_suggestion_fields(record)
        source = safe_join([
            get_claim_text(record),
            get_evidence_text(record),
            sf["what"],
            sf["where"],
            sf["how"],
        ])

    anchors = extract_structural_anchors(source)
    c = normalize_text(chunk_text)
    return any(a in c for a in anchors)


def special_pattern_score(record: Dict[str, Any], chunk_text: str, mode: str) -> float:
    if mode == "task1":
        source = safe_join([
            get_claim_text(record),
            get_evidence_text(record),
            get_original_weakness(record),
            get_followups_text(record),
        ])
    elif mode == "task2_evidence":
        source = safe_join([
            get_claim_text(record),
            get_evidence_text(record),
            get_original_weakness(record),
            get_followups_text(record),
            get_rebuttals_text(record),
        ])
    else:
        sf = get_suggestion_fields(record)
        source = safe_join([
            get_claim_text(record),
            sf["what"],
            sf["where"],
            sf["how"],
        ])

    anchors = extract_structural_anchors(source)
    c = normalize_text(chunk_text)
    score = 0.0
    for a in anchors:
        if a in c:
            score += 0.20 if mode != "task2_support" else 0.12
    return min(0.45, score)


# ============================================================
# Mode-specific score components
# ============================================================

def generic_intro_abstract_penalty(record: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> float:
    """
    Reduce abstract/introduction/generic summary dominance for:
    - novelty
    - theory
    - mechanism/justification
    - ablation
    """
    text = normalize_text(chunk["text"])
    ctype = chunk["chunk_type"]
    page = chunk["page"]

    if mode != "task2_evidence":
        return 0.0

    penalty = 0.0
    trigger = is_novelty_case(record) or is_theory_case(record) or is_mechanism_case(record) or is_ablation_case(record)
    if not trigger:
        return 0.0

    generic_markers = [
        "to the best of our knowledge",
        "our experimental results show",
        "our proposed method",
        "we demonstrate that",
        "we propose",
        "state-of-the-art",
        "significantly improves"
    ]
    structural_markers = [
        "lemma", "proposition", "theorem", "proof", "decomposition", "related work",
        "appendix", "equation", "eq.", "algorithm", "sir", "air", "ablation", "λ1", "λ2", "lambda1", "lambda2"
    ]

    if page <= 2 and ctype in {"generic", "analysis"}:
        if any(m in text for m in generic_markers) and not any(m in text for m in structural_markers):
            penalty -= 0.18

    if is_novelty_case(record) and page <= 2 and ctype == "generic":
        if "related work" not in text and not any(x in text for x in ["mitrovic", "prior", "existing", "previous works"]):
            penalty -= 0.10

    return penalty


def theory_novelty_bias(record: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> float:
    if mode != "task2_evidence":
        return 0.0

    text = normalize_text(chunk["text"])
    ctype = chunk["chunk_type"]
    score = 0.0

    if is_novelty_case(record) or is_theory_case(record):
        if ctype == "method":
            score += 0.12
        if any(x in text for x in ["lemma", "proposition", "theorem", "proof", "decomposition"]):
            score += 0.14
        if any(x in text for x in ["related work", "prior work", "previous works", "mitrovic", "sir"]):
            score += 0.12
        if any(x in text for x in ["appendix b", "appendix c", "proof of proposition", "proof of lemma"]):
            score += 0.08

    if is_mechanism_case(record):
        if any(x in text for x in ["style-independent", "style-independence", "common corruption", "corruption", "downstream tasks"]):
            score += 0.08
        if ctype in {"method", "analysis"}:
            score += 0.05

    return score


def ablation_bias(record: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> float:
    if mode not in {"task2_evidence", "task2_support"}:
        return 0.0

    if not is_ablation_case(record):
        return 0.0

    text = normalize_text(chunk["text"])
    ctype = chunk["chunk_type"]
    score = 0.0

    if any(x in text for x in ["ablation", "variants", "acl variants", "appendix c.2", "appendix c.3", "lambda1", "lambda2", "λ1", "λ2"]):
        score += 0.14
    if ctype in {"analysis", "implementation", "table_result"}:
        score += 0.06
    return score


def empirical_bias(record: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> float:
    text = normalize_text(chunk["text"])
    ctype = chunk["chunk_type"]
    score = 0.0

    if is_empirical_case(record):
        if ctype in {"table_result", "protocol", "implementation"}:
            score += 0.06
        if any(x in text for x in ["standard deviation", "median results", "repeated", "3 times", "p-value", "t-test"]):
            score += 0.10
        if any(x in text for x in ["clip", "vit-b", "vit-l", "baseline", "fine-tuning", "finetuning"]):
            score += 0.08

    return score


def method_clarity_bias(record: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> float:
    if not is_method_clarity_case(record):
        return 0.0

    text = normalize_text(chunk["text"])
    ctype = chunk["chunk_type"]
    score = 0.0

    if ctype == "method":
        score += 0.08
    if any(x in text for x in ["equation", "eq.", "algorithm", "kl", "pdo", "input", "augmented", "adversarial variant"]):
        score += 0.12
    return score


def penalty_for_irrelevant_sections(record: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> float:
    text = normalize_text(chunk["text"])
    ctype = chunk["chunk_type"]

    penalty = 0.0

    if any(x in text for x in [
        "possible negative societal impacts",
        "negative societal impacts",
        "acknowledgements",
        "references"
    ]):
        penalty -= 0.35

    if mode == "task2_evidence":
        if is_method_clarity_case(record):
            if ctype in {"table_result", "protocol"} and not any(
                x in text for x in ["kl", "equation", "proposition", "lemma", "theorem", "algorithm", "pdo", "augmented", "adversarial variant"]
            ):
                penalty -= 0.12

        if is_empirical_case(record):
            if ctype == "method" and not any(
                x in text for x in ["clip", "vit-b", "vit-l", "table", "fine-tune", "implementation", "standard deviation", "p-value", "t-test"]
            ):
                penalty -= 0.10

        if is_novelty_case(record) or is_theory_case(record):
            if ctype in {"protocol", "table_result"} and not any(
                x in text for x in ["related work", "sir", "air", "proposition", "lemma", "theory", "decomposition", "proof"]
            ):
                penalty -= 0.12

    if mode == "task2_support":
        if "limitations" in text and not any(x in text for x in ["appendix", "table", "implementation", "protocol"]):
            penalty -= 0.10

    return penalty


# ============================================================
# Task score functions
# ============================================================

def task1_score(record: Dict[str, Any], chunk: Dict[str, Any], section_hints: List[str], hinted_pages: List[int]) -> Dict[str, Any]:
    q = task1_query_fields(record)

    claim_tokens = tokenize(q["claim"])
    evidence_tokens = tokenize(q["evidence"])
    weakness_tokens = tokenize(q["weakness"])
    followup_tokens = tokenize(q["followups"])
    meta_tokens = tokenize(q["meta"])

    claim_score = lexical_overlap_score(claim_tokens, chunk["text"])
    evidence_score = lexical_overlap_score(evidence_tokens, chunk["text"])
    weakness_score = lexical_overlap_score(weakness_tokens, chunk["text"])
    followup_score = lexical_overlap_score(followup_tokens, chunk["text"])
    meta_score = lexical_overlap_score(meta_tokens, chunk["text"])

    score = (
        0.48 * claim_score +
        0.20 * evidence_score +
        0.18 * weakness_score +
        0.08 * followup_score +
        0.06 * meta_score
    )

    score += phrase_overlap_score(q["claim"], chunk["text"])
    score += phrase_overlap_score(q["evidence"], chunk["text"])
    score += phrase_overlap_score(q["weakness"], chunk["text"])
    score += phrase_overlap_score(q["followups"], chunk["text"])
    score += section_hint_score(chunk["text"], section_hints)
    score += page_hint_score(chunk["page"], hinted_pages)
    score += special_pattern_score(record, chunk["text"], mode="task1")
    score += section_name_bias(record, chunk["page"], mode="task1")
    score += method_clarity_bias(record, chunk, mode="task1")
    score += empirical_bias(record, chunk, mode="task1")
    score += theory_novelty_bias(record, chunk, mode="task1")
    score += ablation_bias(record, chunk, mode="task1")
    score += penalty_for_irrelevant_sections(record, chunk, mode="task1")

    return {
        "page": chunk["page"],
        "text": chunk["text"],
        "score": round(score, 6),
        "claim_score": round(claim_score, 6),
        "weakness_score": round(weakness_score, 6),
        "evidence_score": round(evidence_score, 6),
        "followup_score": round(followup_score, 6),
        "meta_score": round(meta_score, 6),
        "anchor_match": has_anchor_match(record, chunk["text"], mode="task1"),
        "chunk_type": chunk["chunk_type"],
    }


def task2_evidence_score(record: Dict[str, Any], chunk: Dict[str, Any], section_hints: List[str], hinted_pages: List[int]) -> Dict[str, Any]:
    q = task2_evidence_query_fields(record)

    claim_tokens = tokenize(q["claim"])
    evidence_tokens = tokenize(q["evidence"])
    weakness_tokens = tokenize(q["weakness"])
    followup_tokens = tokenize(q["followups"])
    rebuttal_tokens = tokenize(q["rebuttals"])
    meta_tokens = tokenize(q["meta"])

    claim_score = lexical_overlap_score(claim_tokens, chunk["text"])
    evidence_score = lexical_overlap_score(evidence_tokens, chunk["text"])
    weakness_score = lexical_overlap_score(weakness_tokens, chunk["text"])
    followup_score = lexical_overlap_score(followup_tokens, chunk["text"])
    rebuttal_score = lexical_overlap_score(rebuttal_tokens, chunk["text"])
    meta_score = lexical_overlap_score(meta_tokens, chunk["text"])

    score = (
        0.38 * claim_score +
        0.28 * evidence_score +
        0.16 * weakness_score +
        0.07 * followup_score +
        0.06 * rebuttal_score +
        0.05 * meta_score
    )

    score += phrase_overlap_score(q["claim"], chunk["text"])
    score += phrase_overlap_score(q["evidence"], chunk["text"])
    score += phrase_overlap_score(q["weakness"], chunk["text"])
    score += phrase_overlap_score(q["followups"], chunk["text"])
    score += phrase_overlap_score(q["rebuttals"], chunk["text"])
    score += section_hint_score(chunk["text"], section_hints)
    score += page_hint_score(chunk["page"], hinted_pages)
    score += special_pattern_score(record, chunk["text"], mode="task2_evidence")
    score += section_name_bias(record, chunk["page"], mode="task2_evidence")

    score += method_clarity_bias(record, chunk, mode="task2_evidence")
    score += empirical_bias(record, chunk, mode="task2_evidence")
    score += theory_novelty_bias(record, chunk, mode="task2_evidence")
    score += ablation_bias(record, chunk, mode="task2_evidence")
    score += generic_intro_abstract_penalty(record, chunk, mode="task2_evidence")
    score += penalty_for_irrelevant_sections(record, chunk, mode="task2_evidence")

    return {
        "page": chunk["page"],
        "text": chunk["text"],
        "score": round(score, 6),
        "claim_score": round(claim_score, 6),
        "evidence_score": round(evidence_score, 6),
        "weakness_score": round(weakness_score, 6),
        "followup_score": round(followup_score, 6),
        "rebuttal_score": round(rebuttal_score, 6),
        "meta_score": round(meta_score, 6),
        "anchor_match": has_anchor_match(record, chunk["text"], mode="task2_evidence"),
        "chunk_type": chunk["chunk_type"],
    }


def task2_support_score(record: Dict[str, Any], chunk: Dict[str, Any], section_hints: List[str], hinted_pages: List[int]) -> Dict[str, Any]:
    q = task2_support_query_fields(record)

    claim_tokens = tokenize(q["claim"])
    evidence_tokens = tokenize(q["evidence"])
    what_tokens = tokenize(q["what"])
    where_tokens = tokenize(q["where"])
    how_tokens = tokenize(q["how"])
    outcome_tokens = tokenize(q["outcome"])
    meta_tokens = tokenize(q["meta"])

    claim_score = lexical_overlap_score(claim_tokens, chunk["text"])
    evidence_score = lexical_overlap_score(evidence_tokens, chunk["text"])
    what_score = lexical_overlap_score(what_tokens, chunk["text"])
    where_score = lexical_overlap_score(where_tokens, chunk["text"])
    how_score = lexical_overlap_score(how_tokens, chunk["text"])
    outcome_score = lexical_overlap_score(outcome_tokens, chunk["text"])
    meta_score = lexical_overlap_score(meta_tokens, chunk["text"])

    score = (
        0.18 * claim_score +
        0.14 * evidence_score +
        0.20 * what_score +
        0.12 * where_score +
        0.16 * how_score +
        0.10 * outcome_score +
        0.10 * meta_score
    )

    score += phrase_overlap_score(q["claim"], chunk["text"])
    score += phrase_overlap_score(q["what"], chunk["text"])
    score += phrase_overlap_score(q["where"], chunk["text"])
    score += phrase_overlap_score(q["how"], chunk["text"])
    score += section_hint_score(chunk["text"], section_hints)
    score += page_hint_score(chunk["page"], hinted_pages)
    score += special_pattern_score(record, chunk["text"], mode="task2_support")
    score += section_name_bias(record, chunk["page"], mode="task2_support")
    score += ablation_bias(record, chunk, mode="task2_support")
    score += penalty_for_irrelevant_sections(record, chunk, mode="task2_support")

    where_text = normalize_text(q["where"])
    how_text = normalize_text(q["how"])
    blob = " ".join([where_text, how_text, normalize_text(q["what"])])

    if any(x in blob for x in ["implementation", "training", "protocol", "hyperparameter", "fine-tuning", "finetuning"]):
        if chunk["chunk_type"] in {"implementation", "protocol"}:
            score += 0.08

    if any(x in blob for x in ["table", "results", "comparison", "benchmark", "appendix", "ablation", "variants"]):
        if chunk["chunk_type"] in {"table_result", "analysis"}:
            score += 0.08

    if any(x in blob for x in ["methodology", "equation", "algorithm", "proposition", "theorem", "lemma"]):
        if chunk["chunk_type"] == "method":
            score += 0.07

    return {
        "page": chunk["page"],
        "text": chunk["text"],
        "score": round(score, 6),
        "claim_score": round(claim_score, 6),
        "evidence_score": round(evidence_score, 6),
        "what_score": round(what_score, 6),
        "where_score": round(where_score, 6),
        "how_score": round(how_score, 6),
        "query_score": round(outcome_score, 6),
        "meta_score": round(meta_score, 6),
        "anchor_match": has_anchor_match(record, chunk["text"], mode="task2_support"),
        "chunk_type": chunk["chunk_type"],
    }


# ============================================================
# Ranking helpers
# ============================================================

def add_chunk_types(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for idx, ch in enumerate(chunks):
        item = dict(ch)
        item["chunk_id"] = idx
        item["chunk_type"] = infer_chunk_type(ch["text"])
        out.append(item)
    return out


def dedup_ranked_items(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    selected = []
    seen = set()

    for item in items:
        norm = normalize_text(item["text"][:350])
        if norm in seen:
            continue
        seen.add(norm)
        selected.append(item)
        if len(selected) >= top_k:
            break

    return selected


def neighborhood_expand_candidates(
    initial_ranked: List[Dict[str, Any]],
    num_chunks: int,
    seed_top_n: int = 10,
    width: int = 2
) -> List[int]:
    candidate_ids = set()
    if not initial_ranked:
        return []

    best_score = initial_ranked[0]["score"]

    for item in initial_ranked[:seed_top_n]:
        idx = item["chunk_id"]
        candidate_ids.add(idx)

        expand = item.get("anchor_match", False) or item["score"] >= best_score - 0.10
        if expand:
            for delta in range(1, width + 1):
                if idx - delta >= 0:
                    candidate_ids.add(idx - delta)
                if idx + delta < num_chunks:
                    candidate_ids.add(idx + delta)

    return sorted(candidate_ids)


def rank_task1(record: Dict[str, Any], chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    section_hints = extract_section_hints(record, mode="task1")
    hinted_pages = sorted(set(
        extract_explicit_page_hints(record, mode="task1") +
        fuzzy_section_page_hints(record, mode="task1")
    ))

    first = []
    for ch in chunks:
        scored = task1_score(record, ch, section_hints, hinted_pages)
        scored["chunk_id"] = ch["chunk_id"]
        first.append(scored)

    first.sort(key=lambda x: x["score"], reverse=True)
    candidate_ids = neighborhood_expand_candidates(first, len(chunks), seed_top_n=10, width=2)

    second = []
    for idx in candidate_ids:
        ch = chunks[idx]
        scored = task1_score(record, ch, section_hints, hinted_pages)
        scored["chunk_id"] = ch["chunk_id"]
        second.append(scored)

    second.sort(key=lambda x: x["score"], reverse=True)
    return dedup_ranked_items(second, top_k=top_k)


def rank_task2_evidence(record: Dict[str, Any], chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    section_hints = extract_section_hints(record, mode="task2_evidence")
    hinted_pages = sorted(set(
        extract_explicit_page_hints(record, mode="task2_evidence") +
        fuzzy_section_page_hints(record, mode="task2_evidence")
    ))

    first = []
    for ch in chunks:
        scored = task2_evidence_score(record, ch, section_hints, hinted_pages)
        scored["chunk_id"] = ch["chunk_id"]
        first.append(scored)

    first.sort(key=lambda x: x["score"], reverse=True)
    candidate_ids = neighborhood_expand_candidates(first, len(chunks), seed_top_n=12, width=2)

    second = []
    for idx in candidate_ids:
        ch = chunks[idx]
        scored = task2_evidence_score(record, ch, section_hints, hinted_pages)
        scored["chunk_id"] = ch["chunk_id"]
        second.append(scored)

    second.sort(key=lambda x: x["score"], reverse=True)
    return dedup_ranked_items(second, top_k=top_k)


def rank_task2_support(record: Dict[str, Any], chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    section_hints = extract_section_hints(record, mode="task2_support")
    hinted_pages = sorted(set(
        extract_explicit_page_hints(record, mode="task2_support") +
        fuzzy_section_page_hints(record, mode="task2_support")
    ))

    first = []
    for ch in chunks:
        scored = task2_support_score(record, ch, section_hints, hinted_pages)
        scored["chunk_id"] = ch["chunk_id"]
        first.append(scored)

    first.sort(key=lambda x: x["score"], reverse=True)
    candidate_ids = neighborhood_expand_candidates(first, len(chunks), seed_top_n=12, width=3)

    second = []
    for idx in candidate_ids:
        ch = chunks[idx]
        scored = task2_support_score(record, ch, section_hints, hinted_pages)
        scored["chunk_id"] = ch["chunk_id"]
        second.append(scored)

    second.sort(key=lambda x: x["score"], reverse=True)
    return dedup_ranked_items(second, top_k=top_k)


def merge_task2_lists(
    evidence_snippets: List[Dict[str, Any]],
    support_snippets: List[Dict[str, Any]],
    final_top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Balanced merge:
    - Prefer 3 evidence + 2 support when possible
    - Deduplicate by normalized prefix
    """
    merged = []
    seen = set()

    def try_add(item: Dict[str, Any]) -> None:
        norm = normalize_text(item["text"][:350])
        if norm in seen:
            return
        seen.add(norm)
        merged.append(item)

    for item in evidence_snippets[:3]:
        try_add(item)
    for item in support_snippets[:2]:
        try_add(item)

    pool = sorted(
        evidence_snippets[3:] + support_snippets[2:],
        key=lambda x: x["score"],
        reverse=True
    )
    for item in pool:
        if len(merged) >= final_top_k:
            break
        try_add(item)

    if len(merged) < final_top_k:
        pool2 = sorted(evidence_snippets + support_snippets, key=lambda x: x["score"], reverse=True)
        for item in pool2:
            if len(merged) >= final_top_k:
                break
            try_add(item)

    return merged[:final_top_k]


# ============================================================
# Main per-record processing
# ============================================================

def process_record(
    record: Dict[str, Any],
    pdf_cache_dir: Path,
    sleep_seconds: float = 0.0,
    top_k_task1: int = 3,
    top_k_task2_final: int = 5,
    top_k_task2_channel: int = 4,
) -> Dict[str, Any]:
    record = dict(record)
    paper_context = record.get("paper_context") or {}
    pdf_url = paper_context.get("pdf_url")

    record["aligned_snippets_task1"] = []
    record["aligned_snippets_task2_evidence"] = []
    record["aligned_snippets_task2_support"] = []
    record["aligned_snippets_task2"] = []
    record["alignment_status_task1"] = "not_run"
    record["alignment_status_task2"] = "not_run"

    if not pdf_url:
        record["alignment_status_task1"] = "no_pdf_url"
        record["alignment_status_task2"] = "no_pdf_url"
        return record

    try:
        pdf_candidates = [pdf_url] + derive_openreview_pdf_fallbacks(record)
        pdf_path = download_pdf(pdf_candidates, pdf_cache_dir)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        pages = extract_pdf_pages(pdf_path)
        chunks = build_chunks_from_pages(pages)
        chunks = add_chunk_types(chunks)

        if not chunks:
            record["alignment_status_task1"] = "no_text_extracted"
            record["alignment_status_task2"] = "no_text_extracted"
            return record

        task1_snips = rank_task1(record, chunks, top_k=top_k_task1)
        record["aligned_snippets_task1"] = task1_snips
        record["alignment_status_task1"] = "ok" if task1_snips else "no_snippet_found"

        task2_evidence = rank_task2_evidence(record, chunks, top_k=top_k_task2_channel)
        task2_support = rank_task2_support(record, chunks, top_k=top_k_task2_channel)
        task2_final = merge_task2_lists(task2_evidence, task2_support, final_top_k=top_k_task2_final)

        record["aligned_snippets_task2_evidence"] = task2_evidence
        record["aligned_snippets_task2_support"] = task2_support
        record["aligned_snippets_task2"] = task2_final
        record["alignment_status_task2"] = "ok" if task2_final else "no_snippet_found"

        record["aligned_snippets"] = task1_snips
        return record

    except Exception as e:
        record["alignment_status_task1"] = "error"
        record["alignment_status_task2"] = "error"
        record["alignment_error"] = str(e)[:500]
        return record


# ============================================================
# CLI / file loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Dual-task PDF snippet alignment for Task1/Task2")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--pdf_cache_dir", default="data/pdf_cache", help="PDF cache dir")
    parser.add_argument("--top_k_task1", type=int, default=3, help="Top-k for Task 1")
    parser.add_argument("--top_k_task2_final", type=int, default=5, help="Final top-k for Task 2 merged output")
    parser.add_argument("--top_k_task2_channel", type=int, default=4, help="Top-k per Task 2 channel")
    parser.add_argument("--sleep_seconds", type=float, default=0.0, help="Sleep between downloads")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")
    parser.add_argument("--resume", action="store_true", help="Resume and append new records")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    pdf_cache_dir = Path(args.pdf_cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = set()
    append_mode = False

    if args.resume and output_path.exists():
        print(f"Resume: loading existing weakness_ids from {output_path} ...")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    wid = rec.get("weakness_id")
                    if wid:
                        processed_ids.add(wid)
                except Exception:
                    pass
        append_mode = True
        print(f"Resume: found {len(processed_ids)} processed records.")

    if args.limit is not None:
        total_lines = args.limit
    else:
        print("Counting input lines...")
        total_lines = 0
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_lines += 1
        print(f"Total input records: {total_lines}")

    total = 0
    with_pdf = 0
    ok_task1 = 0
    ok_task2 = 0
    no_pdf = 0
    errors = 0
    skipped = 0

    pbar = tqdm(total=total_lines, desc="Aligning snippets", unit="rec")
    fout_mode = "a" if append_mode else "w"

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(fout_mode, encoding="utf-8") as fout:
        for line in fin:
            if args.limit is not None and total >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except Exception:
                errors += 1
                pbar.update(1)
                pbar.set_postfix(ok1=ok_task1, ok2=ok_task2, err=errors, skip=skipped, refresh=False)
                continue

            wid = record.get("weakness_id") or ""
            if append_mode and wid in processed_ids:
                skipped += 1
                pbar.update(1)
                pbar.set_postfix(ok1=ok_task1, ok2=ok_task2, err=errors, skip=skipped, refresh=False)
                continue

            if ((record.get("paper_context") or {}).get("pdf_url")):
                with_pdf += 1

            updated = process_record(
                record=record,
                pdf_cache_dir=pdf_cache_dir,
                sleep_seconds=args.sleep_seconds,
                top_k_task1=args.top_k_task1,
                top_k_task2_final=args.top_k_task2_final,
                top_k_task2_channel=args.top_k_task2_channel,
            )

            st1 = updated.get("alignment_status_task1")
            st2 = updated.get("alignment_status_task2")

            if st1 == "ok":
                ok_task1 += 1
            if st2 == "ok":
                ok_task2 += 1
            if st1 == "no_pdf_url" or st2 == "no_pdf_url":
                no_pdf += 1
            if st1 == "error" or st2 == "error":
                errors += 1

            fout.write(json.dumps(updated, ensure_ascii=False) + "\n")
            fout.flush()

            pbar.update(1)
            pbar.set_postfix(ok1=ok_task1, ok2=ok_task2, err=errors, skip=skipped, refresh=False)

    pbar.close()

    print("=" * 80)
    print("DONE")
    print(f"Total input lines seen: {total}")
    if append_mode:
        print(f"Skipped (already in output): {skipped}")
    print(f"With pdf_url: {with_pdf}")
    print(f"Task1 aligned ok: {ok_task1}")
    print(f"Task2 aligned ok: {ok_task2}")
    print(f"No pdf_url: {no_pdf}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}" + (" (appended)" if append_mode else ""))
    print("=" * 80)


if __name__ == "__main__":
    main()