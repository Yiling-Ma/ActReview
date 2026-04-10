#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect ICML 2022-2023 multi-turn review threads from OpenReview (v2 primary, v1 fallback).

Output: one JSONL row per submission, with a flat `thread_events` list that
preserves conversation tree (review -> rebuttal -> follow-up -> ...), aligned
with ICLR/NeurIPS data schema.

Env:
    export OPENREVIEW_USERNAME="your_email"
    export OPENREVIEW_PASSWORD="your_password"

"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import openreview

# Optional: pymupdf for PDF section extraction
try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# Optional: requests for PDF download
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =========================
# Constants
# =========================
BASEURL_V2 = "https://api2.openreview.net"
BASEURL_V1 = "https://api.openreview.net"
OPENREVIEW_WEB_BASE = "https://openreview.net"

RATING_KEYS = [
    "overall_rating", "rating", "recommendation", "reviewer_rating",
    "overall_assessment", "overall", "score", "assessment",
]

CONFIDENCE_KEYS = [
    "confidence", "reviewer_confidence", "confidence_rating",
]

TEXT_FIELDS = ["comment", "response", "text", "message", "rebuttal", "discussion"]

# ICML venue id patterns (most likely first)
ICML_VENUE_CANDIDATES = [
    "ICML.cc/{year}/Conference",
    "icml.cc/{year}/Conference",
    "ICML/{year}/Conference",
    "ICML.cc/{year}",
    "icml.cc/{year}",
]


# =========================
# Retry decorator
# =========================
def with_retry(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        sys.stderr.write(f"Failed after {max_retries} attempts: {e}\n")
                        sys.stderr.flush()
                        raise
                    wait = delay * (2 ** attempt)
                    sys.stderr.write(f"Retry {attempt + 1}/{max_retries} in {wait}s: {e}\n")
                    sys.stderr.flush()
                    time.sleep(wait)
        return wrapper
    return decorator


# =========================
# Field helpers
# =========================
def _val(x):
    """Unwrap v2 {'value': ...} or return as-is."""
    if isinstance(x, dict):
        return x.get("value")
    return x


def safe_str(x) -> Optional[str]:
    x = _val(x)
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None


def safe_list_str(x) -> Optional[List[str]]:
    x = _val(x)
    if x is None:
        return None
    if isinstance(x, list):
        out = [item.strip() for item in x if isinstance(item, str) and item.strip()]
        return out if out else None
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return None


def parse_numeric_rating(v) -> Tuple[Optional[float], Optional[str]]:
    """Parse '8: Accept' or '6' -> (8.0, 'Accept')."""
    v = _val(v)
    if v is None:
        return None, None
    if isinstance(v, (int, float)):
        return float(v), None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None, None
        m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*[:：]?\s*(.*)\s*$", s)
        if m:
            return float(m.group(1)), m.group(2).strip() or None
        return None, s
    return None, None


def find_rating_field(content: dict) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    keys = list(content.keys())
    lowmap = {k: k.lower() for k in keys}
    for pref in RATING_KEYS:
        for k in keys:
            if pref in lowmap[k]:
                return k
    for k in keys:
        lk = lowmap[k]
        if any(t in lk for t in ["rating", "score", "recommend", "assess"]):
            return k
    return None


def extract_rating(content: dict) -> Optional[float]:
    fname = find_rating_field(content)
    if not fname:
        return None
    num, _ = parse_numeric_rating(content.get(fname))
    return num


def extract_confidence(content: dict) -> Optional[float]:
    if not isinstance(content, dict):
        return None
    for field in CONFIDENCE_KEYS:
        if field in content:
            val = _val(content[field])
            if val is not None:
                num, _ = parse_numeric_rating(val)
                return num
    return None


def extract_text(content: dict) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    for field in TEXT_FIELDS:
        if field in content:
            s = safe_str(content[field])
            if s:
                return s
    return None


def join_review_text(content: dict) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    if "review" in content:
        s = safe_str(content["review"])
        if s:
            return s
    texts = []
    for k, v in content.items():
        val = _val(v)
        if isinstance(val, str):
            s = val.strip()
            if s and not re.fullmatch(r"[0-9.:/,_-]+", s) and len(s) >= 3:
                texts.append(f"## {k}\n{s}" if k.lower() not in ["review", "summary"] else s)
    return "\n\n".join(texts) if texts else None


# =========================
# Submission metadata
# =========================
def safe_note_field(n, field: str) -> Optional[str]:
    if hasattr(n, "content") and isinstance(n.content, dict):
        return safe_str(n.content.get(field))
    return None


def safe_note_keywords(n) -> Optional[List[str]]:
    if hasattr(n, "content") and isinstance(n.content, dict):
        kws = n.content.get("keywords") or n.content.get("keyword")
        return safe_list_str(kws)
    return None


def normalize_pdf_url(pdf_val: Optional[str]) -> Optional[str]:
    if not pdf_val:
        return None
    if pdf_val.startswith("http://") or pdf_val.startswith("https://"):
        return pdf_val
    if pdf_val.startswith("/"):
        return OPENREVIEW_WEB_BASE + pdf_val
    return OPENREVIEW_WEB_BASE + "/" + pdf_val.lstrip("/")


# =========================
# PDF section extraction
# =========================
def extract_sections_index(pdf_url: str) -> Optional[List[Dict]]:
    if not HAS_FITZ or not HAS_REQUESTS:
        return None
    try:
        resp = _requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            doc = fitz.open(tmp.name)

            all_spans: List[Tuple[float, str, int]] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                try:
                    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                except Exception as e:
                    msg = str(e)
                    if "appearance stream" in msg or "Screen annotations" in msg:
                        sys.stderr.write(
                            f"MuPDF warning: skip page {page_num + 1} due to annotation appearance error\n"
                        )
                        sys.stderr.flush()
                        continue
                    raise
                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span["text"].strip()
                            if text:
                                all_spans.append((span["size"], text, page_num + 1))

            if not all_spans:
                doc.close()
                return None

            from collections import Counter
            size_counts = Counter(round(s[0], 1) for s in all_spans)
            body_size = size_counts.most_common(1)[0][0]

            section_pattern = re.compile(
                r"^(?:(\d+(?:\.\d+)*)\s+)?"
                r"(Introduction|Related\s+Work|Background|Method|Approach|Model|"
                r"Experiments?|Results?|Evaluation|Discussion|Conclusion|"
                r"Acknowledgements?|References|Appendix|Limitations|"
                r"Abstract|Supplementary|Analysis|Setup|Ablation|"
                r"Implementation|Training|Dataset)",
                re.IGNORECASE,
            )

            sections = []
            sid = 0
            seen = set()
            for size, text, page in all_spans:
                if round(size, 1) <= body_size:
                    continue
                if len(text) > 80 or len(text) < 3:
                    continue
                m = section_pattern.match(text)
                if m:
                    name = text.strip()
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        sid += 1
                        sections.append({
                            "section_id": f"s{sid}",
                            "name": name,
                            "page": page,
                        })

            doc.close()
            return sections if sections else None
    except Exception:
        return None


# =========================
# Reply type classification
# =========================
def classify_reply(reply: dict, venue_id: str) -> str:
    invs = reply.get("invitations", []) or []
    sigs = reply.get("signatures", []) or []

    invs_lower = [x.lower() for x in invs if isinstance(x, str)]

    # Decision
    if any(x.endswith("decision") or "/decision" in x for x in invs_lower):
        return "decision"

    # Official review
    if any(("official_review" in x) or x.endswith("/review") for x in invs_lower):
        return "review"

    # Meta review
    if any(("meta_review" in x) or ("metareview" in x) for x in invs_lower):
        return "meta_review"

    # Author rebuttal/response
    for inv in invs_lower:
        if any(p in inv for p in [
            "/authors/-/official_comment", "/authors/-/rebuttal",
            "author_response", "authors_official_comment", "rebuttal"
        ]):
            return "rebuttal"
        if "/authors" in inv and ("comment" in inv or "response" in inv):
            return "rebuttal"

    for sig in sigs:
        if isinstance(sig, str) and (sig.endswith("/Authors") or sig.lower().endswith("/authors")):
            return "rebuttal"

    return "comment"


def extract_role_and_actor(reply: dict, venue_id: str, event_type: str) -> Tuple[str, Optional[str]]:
    sigs = reply.get("signatures", []) or []
    if not sigs:
        return "unknown", None
    sig = sigs[0]

    if event_type == "rebuttal":
        return "author", sig
    if event_type == "decision":
        return "program_chair", sig
    if event_type == "meta_review":
        return "area_chair", sig

    sig_lower = str(sig).lower()
    if "area_chair" in sig_lower or "ac" in sig_lower:
        return "area_chair", sig
    if "reviewer" in sig_lower:
        return "reviewer", sig
    if sig_lower.endswith("/authors"):
        return "author", sig

    return "unknown", sig


# =========================
# Participant extraction
# =========================
def extract_participants(replies: List[dict], venue_id: str) -> Dict:
    reviewers = set()
    authors = set()
    area_chairs = set()

    for r in replies:
        etype = classify_reply(r, venue_id)
        role, actor = extract_role_and_actor(r, venue_id, etype)
        if actor is None:
            continue
        if role == "reviewer":
            reviewers.add(actor)
        elif role == "author":
            authors.add(actor)
        elif role == "area_chair":
            area_chairs.add(actor)

    return {
        "reviewers": [{"reviewer_id": r} for r in sorted(reviewers)],
        "authors": [{"author_id": a} for a in sorted(authors)],
        "area_chairs": [{"ac_id": ac} for ac in sorted(area_chairs)],
    }


# =========================
# Thread event builder
# =========================
def ts_to_iso(ts) -> Optional[str]:
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        return str(ts)
    except Exception:
        return str(ts)


@with_retry(max_retries=3, delay=2)
def _get_note_edits(client, note_id: str):
    return client.get_note_edits(note_id=note_id, sort="tcdate:asc")


def build_rating_update_events(client, review_reply: dict, review_event_id: str,
                               venue_id: str) -> List[Dict]:
    rid = review_reply["id"]
    try:
        edits = _get_note_edits(client, rid)
    except Exception as e:
        sys.stderr.write(f"  Skipping edits for {rid}: {e}\n")
        sys.stderr.flush()
        return []

    if len(edits) <= 1:
        return []

    events = []
    prev_rating = None
    prev_confidence = None

    for edit in edits:
        note = edit.note
        if not note or not isinstance(note.content, dict):
            continue
        cur_rating = extract_rating(note.content)
        cur_confidence = extract_confidence(note.content)

        if prev_rating is not None and (cur_rating != prev_rating or cur_confidence != prev_confidence):
            role, actor = extract_role_and_actor(review_reply, venue_id, "review")
            events.append({
                "event_id": f"{edit.id}_rating_update",
                "timestamp": ts_to_iso(edit.tcdate or edit.cdate),
                "role": role,
                "actor_id": actor,
                "event_type": "rating_update",
                "text": None,
                "reply_to_event_id": review_event_id,
                "rating": cur_rating,
                "confidence": cur_confidence,
            })

        if cur_rating is not None:
            prev_rating = cur_rating
        if cur_confidence is not None:
            prev_confidence = cur_confidence

    return events


_PHASE_MAP = {
    "review": "initial_review",
    "rebuttal": "rebuttal",
    "comment": "discussion",
    "meta_review": "meta_review",
    "decision": "decision",
    "rating_update": "post_rebuttal",
}


def _assign_turn_and_phase(events: List[Dict]):
    if not events:
        return

    prev_role = None
    turn = 0

    for idx, ev in enumerate(events):
        ev["event_index"] = idx
        etype = ev["event_type"]
        role = ev["role"]

        if etype in ("review",) and prev_role != "reviewer":
            turn += 1
        elif etype == "rebuttal" and prev_role != "author":
            turn += 1
        elif etype == "comment" and role != prev_role:
            turn += 1
        elif etype in ("meta_review", "decision"):
            turn += 1
        elif etype == "rating_update":
            pass
        elif turn == 0:
            turn = 1

        ev["turn"] = turn
        ev["phase"] = _PHASE_MAP.get(etype, "discussion")

        if etype == "comment" and role == "reviewer":
            has_prior_rebuttal = any(events[j]["event_type"] == "rebuttal" for j in range(idx))
            if has_prior_rebuttal:
                ev["phase"] = "post_rebuttal"

        prev_role = role


def build_thread_events(replies: List[dict], venue_id: str, client=None, with_edits: bool = False) -> List[Dict]:
    events = []

    for r in replies:
        eid = r.get("id")
        content = r.get("content", {}) or {}
        etype = classify_reply(r, venue_id)
        role, actor = extract_role_and_actor(r, venue_id, etype)
        timestamp = ts_to_iso(r.get("tcdate") or r.get("cdate"))
        reply_to = r.get("replyto")

        if etype == "review":
            text = join_review_text(content)
        elif etype == "decision":
            text = safe_str(content.get("decision"))
            comment = safe_str(content.get("comment"))
            if comment:
                text = f"{text}\n\n{comment}" if text else comment
        elif etype == "meta_review":
            text = extract_text(content) or safe_str(content.get("metareview"))
        else:
            text = extract_text(content)

        rating = extract_rating(content) if etype == "review" else None
        confidence = extract_confidence(content) if etype == "review" else None

        event = {
            "event_id": eid,
            "timestamp": timestamp,
            "role": role,
            "actor_id": actor,
            "event_type": etype,
            "text": text,
            "reply_to_event_id": reply_to,
            "rating": rating,
            "confidence": confidence,
            "note_id": eid,
            "forum_id": r.get("forum"),
            "invitation": r.get("invitations", [None])[0] if r.get("invitations") else None,
        }
        events.append(event)

        if with_edits and etype == "review" and client is not None:
            events.extend(build_rating_update_events(client, r, eid, venue_id))

    events.sort(key=lambda e: e["timestamp"] or "")
    _assign_turn_and_phase(events)
    return events


# =========================
# Fetch helpers
# =========================
def _v1_note_to_dict(r) -> Optional[dict]:
    if hasattr(r, "__dict__"):
        d = {
            "id": getattr(r, "id", None),
            "forum": getattr(r, "forum", None),
            "replyto": getattr(r, "replyto", None),
            "content": getattr(r, "content", {}),
            "signatures": getattr(r, "signatures", []),
            "tcdate": getattr(r, "tcdate", None),
            "cdate": getattr(r, "cdate", None),
            "invitations": [getattr(r, "invitation", "")] if getattr(r, "invitation", None) else [],
        }
        return d
    elif isinstance(r, dict):
        d = dict(r)
        if "invitation" in d and "invitations" not in d:
            d["invitations"] = [d["invitation"]] if d["invitation"] else []
        return d
    return None


def _normalize_replies(notes) -> List[dict]:
    out: List[dict] = []
    for n in notes or []:
        d = _v1_note_to_dict(n)
        if d:
            out.append(d)
    return out


def _extract_v2_replies_from_submission(sub) -> List[dict]:
    """
    OpenReview v2 submission.details keys vary by venue/year.
    Try common reply containers in order.
    """
    details = getattr(sub, "details", None) or {}
    if not isinstance(details, dict):
        return []

    for key in ("replies", "directReplies", "replyNotes"):
        raw = details.get(key) or []
        replies = _normalize_replies(raw)
        if replies:
            return replies
    return []


@with_retry(max_retries=3, delay=2)
def _v1_fetch_all_forum_notes(client_v1, forum_id: str) -> List[dict]:
    all_notes = client_v1.get_all_notes(forum=forum_id)
    replies = []
    for n in all_notes:
        if n.id == forum_id:
            continue
        d = _v1_note_to_dict(n)
        if d:
            replies.append(d)
    return replies


@with_retry(max_retries=3, delay=2)
def _v2_fetch_all_forum_notes(client_v2, forum_id: str) -> List[dict]:
    all_notes = client_v2.get_all_notes(forum=forum_id)
    replies = []
    for n in all_notes:
        d = _v1_note_to_dict(n)
        if not d:
            continue
        if d.get("id") == forum_id:
            continue
        replies.append(d)
    return replies


def _find_latest_output(output_dir: str, year: int) -> Optional[Path]:
    pattern = f"icml{year}_threads_*.jsonl"
    candidates = sorted(
        Path(output_dir).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_existing(output_path: str) -> Tuple[set, Dict[str, int]]:
    processed_ids = set()
    stats = {
        "submissions_with_events": 0,
        "total_events": 0,
        "event_types": {},
    }
    try:
        with open(output_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pid = obj.get("submission_id")
                if pid:
                    processed_ids.add(pid)
                events = obj.get("thread_events") or []
                if events:
                    stats["submissions_with_events"] += 1
                stats["total_events"] += len(events)
                for ev in events:
                    et = ev.get("event_type")
                    if et:
                        stats["event_types"][et] = stats["event_types"].get(et, 0) + 1
    except FileNotFoundError:
        pass
    return processed_ids, stats


def _resolve_icml_venue(client_v2, year: int) -> Optional[str]:
    for pat in ICML_VENUE_CANDIDATES:
        venue_id = pat.format(year=year)
        try:
            _ = client_v2.get_group(venue_id)
            sys.stderr.write(f"  Resolved venue_id: {venue_id}\n")
            sys.stderr.flush()
            return venue_id
        except Exception:
            continue
    return None


def _fetch_submissions_v2(client_v2, venue_id: str):
    submission_name = None
    try:
        venue_group = client_v2.get_group(venue_id)
        submission_name = venue_group.content["submission_name"]["value"]
    except Exception:
        pass

    candidate_names = []
    if submission_name:
        candidate_names.append(submission_name)
    candidate_names.extend(["Blind_Submission", "Submission"])

    best_subs = []
    best_name = None
    best_reply_signal = -1

    for name in candidate_names:
        invitation = f"{venue_id}/-/{name}"
        sys.stderr.write(f"  [v2] Trying invitation: {invitation}\n")
        sys.stderr.flush()
        try:
            subs = client_v2.get_all_notes(invitation=invitation, details="replies")
            if subs:
                # Heuristic: prefer invitations where submission.details already exposes
                # non-empty reply containers on at least some papers.
                probe_n = min(50, len(subs))
                reply_signal = 0
                for s in subs[:probe_n]:
                    if _extract_v2_replies_from_submission(s):
                        reply_signal += 1

                sys.stderr.write(
                    f"  [v2] -> Found {len(subs)} submissions, "
                    f"reply_signal={reply_signal}/{probe_n}\n"
                )
                sys.stderr.flush()
                if (
                    reply_signal > best_reply_signal
                    or (reply_signal == best_reply_signal and len(subs) > len(best_subs))
                ):
                    best_subs = subs
                    best_name = name
                    best_reply_signal = reply_signal
            sys.stderr.write("  [v2] -> 0 submissions, trying next...\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"  [v2] -> Error: {e}, trying next...\n")
            sys.stderr.flush()

    if best_subs:
        sys.stderr.write(
            f"  [v2] Selected invitation {venue_id}/-/{best_name} "
            f"(submissions={len(best_subs)}, reply_signal={best_reply_signal})\n"
        )
        sys.stderr.flush()
        return best_subs, "v2"

    # Fallback by venueid in content
    sys.stderr.write("  [v2] Trying venueid search...\n")
    sys.stderr.flush()
    try:
        subs = client_v2.get_all_notes(content={"venueid": venue_id}, details="replies")
        if subs:
            sys.stderr.write(f"  [v2] -> Found {len(subs)} submissions via venueid\n")
            sys.stderr.flush()
            return subs, "v2"
    except Exception as e:
        sys.stderr.write(f"  [v2] -> venueid search failed: {e}\n")
        sys.stderr.flush()

    return [], None


def _fetch_submissions_v1(client_v1, venue_id: str):
    candidate_names = ["Blind_Submission", "Submission"]
    best_subs = []
    best_name = None
    for name in candidate_names:
        invitation = f"{venue_id}/-/{name}"
        sys.stderr.write(f"  [v1] Trying invitation: {invitation}\n")
        sys.stderr.flush()
        try:
            subs = client_v1.get_all_notes(invitation=invitation, details="directReplies")
            if subs:
                sys.stderr.write(f"  [v1] -> Found {len(subs)} submissions\n")
                sys.stderr.flush()
                if len(subs) > len(best_subs):
                    best_subs = subs
                    best_name = name
            sys.stderr.write("  [v1] -> 0 submissions, trying next...\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"  [v1] -> Error: {e}, trying next...\n")
            sys.stderr.flush()

    if best_subs:
        sys.stderr.write(
            f"  [v1] Selected invitation {venue_id}/-/{best_name} "
            f"(submissions={len(best_subs)})\n"
        )
        sys.stderr.flush()
        return best_subs, "v1"

    return [], None


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Collect ICML multi-turn review threads")
    p.add_argument("--years", nargs="+", type=int, choices=[2022, 2023], default=[2022, 2023],
                   help="ICML years to collect (default: 2022 2023; allowed: 2022 2023)")
    p.add_argument("--with-edits", action="store_true",
                   help="Query edit history per review to emit rating_update events (slow)")
    p.add_argument("--skip-pdf", action="store_true",
                   help="Skip PDF download and section extraction")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Output directory")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing output file if present")
    p.add_argument("--output-path", type=str, default=None,
                   help="Explicit output JSONL path")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    username = os.environ["OPENREVIEW_USERNAME"]
    password = os.environ["OPENREVIEW_PASSWORD"]

    client_v2 = openreview.api.OpenReviewClient(
        baseurl=BASEURL_V2,
        username=username,
        password=password,
    )
    client_v1 = openreview.Client(
        baseurl=BASEURL_V1,
        username=username,
        password=password,
    )

    for year in args.years:
        sys.stderr.write(f"\n{'='*60}\n")
        sys.stderr.write(f"Collecting ICML {year}\n")
        sys.stderr.write(f"{'='*60}\n")
        sys.stderr.flush()

        venue_id = _resolve_icml_venue(client_v2, year)
        if not venue_id:
            # last fallback guess
            venue_id = f"ICML.cc/{year}/Conference"
            sys.stderr.write(f"  Warning: cannot validate venue via get_group, fallback to {venue_id}\n")
            sys.stderr.flush()

        sys.stderr.write("Fetching submissions...\n")
        sys.stderr.flush()

        subs, api_version = _fetch_submissions_v2(client_v2, venue_id)
        if not subs:
            sys.stderr.write("  v2 API found nothing, trying v1 API...\n")
            sys.stderr.flush()
            subs, api_version = _fetch_submissions_v1(client_v1, venue_id)

        if not subs:
            sys.stderr.write(f"No submissions found for ICML {year}, skipping.\n")
            sys.stderr.flush()
            continue

        sys.stderr.write(f"Using {api_version} API, {len(subs)} submissions\n")
        sys.stderr.flush()

        total_subs = len(subs)

        output_path = None
        if args.output_path:
            output_path = args.output_path
        elif args.resume:
            latest = _find_latest_output(args.output_dir, year)
            if latest:
                output_path = str(latest)

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(args.output_dir, f"icml{year}_threads_{timestamp}.jsonl")

        stats = {
            "total_submissions": total_subs,
            "submissions_with_events": 0,
            "total_events": 0,
            "event_types": {},
            "fallback_v2_forum_fetch": 0,
            "fallback_v1_forum_fetch": 0,
            "empty_replies_after_fallback": 0,
        }

        processed_ids = set()
        if args.resume and os.path.exists(output_path):
            processed_ids, existing_stats = _load_existing(output_path)
            stats["submissions_with_events"] += existing_stats["submissions_with_events"]
            stats["total_events"] += existing_stats["total_events"]
            for et, cnt in existing_stats["event_types"].items():
                stats["event_types"][et] = stats["event_types"].get(et, 0) + cnt
            sys.stderr.write(f"Resuming from {output_path}, already have {len(processed_ids)} papers\n")
            sys.stderr.flush()

        open_mode = "a" if args.resume and os.path.exists(output_path) else "w"
        with open(output_path, open_mode, encoding="utf-8") as fout:
            for sub_idx, sub in enumerate(subs):
                paper_id = sub.forum or sub.id
                if paper_id in processed_ids:
                    sys.stderr.write(
                        f"\r[{year}] {sub_idx + 1}/{total_subs} (paper: {paper_id[:20]}...) skipped (resume)    "
                    )
                    sys.stderr.flush()
                    continue

                if api_version == "v1":
                    try:
                        replies = _v1_fetch_all_forum_notes(client_v1, paper_id)
                    except Exception as e:
                        sys.stderr.write(f"\n  Failed to fetch forum {paper_id}: {e}\n")
                        sys.stderr.flush()
                        replies = []
                else:
                    replies = _extract_v2_replies_from_submission(sub)
                    if not replies:
                        # Fallback 1: query v2 notes by forum directly.
                        try:
                            replies = _v2_fetch_all_forum_notes(client_v2, paper_id)
                            if replies:
                                stats["fallback_v2_forum_fetch"] += 1
                        except Exception as e:
                            sys.stderr.write(f"\n  [v2 forum fallback failed] {paper_id}: {e}\n")
                            sys.stderr.flush()
                            replies = []

                    if not replies:
                        # Fallback 2: query v1 notes by forum directly.
                        try:
                            replies = _v1_fetch_all_forum_notes(client_v1, paper_id)
                            if replies:
                                stats["fallback_v1_forum_fetch"] += 1
                        except Exception as e:
                            sys.stderr.write(f"\n  [v1 forum fallback failed] {paper_id}: {e}\n")
                            sys.stderr.flush()
                            replies = []

                    if not replies:
                        stats["empty_replies_after_fallback"] += 1

                sys.stderr.write(
                    f"\r[{year}] {sub_idx + 1}/{total_subs} (paper: {paper_id[:20]}...) replies={len(replies)}    "
                )
                sys.stderr.flush()

                title = safe_note_field(sub, "title")
                abstract = safe_note_field(sub, "abstract")
                keywords = safe_note_keywords(sub)
                pdf_val = safe_note_field(sub, "pdf")
                pdf_url = normalize_pdf_url(pdf_val)
                web_url = f"{OPENREVIEW_WEB_BASE}/forum?id={paper_id}"

                sections_index = None
                if not args.skip_pdf and pdf_url:
                    sections_index = extract_sections_index(pdf_url)

                paper_context = {
                    "title": title,
                    "abstract": abstract,
                    "keywords": keywords,
                    "pdf_url": pdf_url,
                    "web_url": web_url,
                    "sections_index": sections_index,
                }

                participants = extract_participants(replies, venue_id)

                thread_events = build_thread_events(
                    replies, venue_id,
                    client=client_v2 if args.with_edits else None,
                    with_edits=args.with_edits,
                )
                if not thread_events:
                    raise RuntimeError(
                        f"Empty thread_events for submission {paper_id} (ICML {year}). "
                        "Fail-fast enabled: stop to avoid silently writing empty data."
                    )

                decision = None
                for ev in thread_events:
                    if ev["event_type"] == "decision":
                        decision = ev["text"]
                        if decision and "\n" in decision:
                            first_line = decision.split("\n")[0].strip()
                            if any(kw in first_line.lower() for kw in ["accept", "reject"]):
                                decision = first_line
                        break

                row = {
                    "venue": "ICML",
                    "year": year,
                    "submission_id": paper_id,
                    "paper_context": paper_context,
                    "participants": participants,
                    "decision": decision,
                    "thread_events": thread_events,
                }

                line = json.dumps(row, ensure_ascii=False)
                line = re.sub(
                    r"[\u2028\u2029\u0000-\u0008\u000b\u000c\u000e-\u001f\ufeff\u200b-\u200f\u202a-\u202e\u2066-\u2069]",
                    " ",
                    line,
                )
                fout.write(line + "\n")

                if thread_events:
                    stats["submissions_with_events"] += 1
                stats["total_events"] += len(thread_events)
                for ev in thread_events:
                    et = ev["event_type"]
                    stats["event_types"][et] = stats["event_types"].get(et, 0) + 1

        sys.stderr.write(f"\n\n--- ICML {year} Summary ---\n")
        sys.stderr.write(f"Total submissions: {stats['total_submissions']}\n")
        sys.stderr.write(f"Submissions with events: {stats['submissions_with_events']}\n")
        sys.stderr.write(f"Total events: {stats['total_events']}\n")
        sys.stderr.write(
            "Fallback stats: "
            f"v2_forum={stats['fallback_v2_forum_fetch']}, "
            f"v1_forum={stats['fallback_v1_forum_fetch']}, "
            f"still_empty={stats['empty_replies_after_fallback']}\n"
        )
        for et, count in sorted(stats["event_types"].items()):
            sys.stderr.write(f"  {et}: {count}\n")
        if stats["submissions_with_events"] == 0:
            raise RuntimeError(
                f"0 submissions have events for ICML {year}; stopping because empty output is not allowed."
            )
        sys.stderr.write(f"Output: {output_path}\n")
        sys.stderr.flush()

        print(f"[ICML {year}] Saved {stats['total_submissions']} submissions "
              f"({stats['total_events']} events) to {output_path}")


if __name__ == "__main__":
    main()
