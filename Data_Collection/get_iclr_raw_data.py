# -*- coding: utf-8 -*-
"""
Collect ICLR 2021-2025 multi-turn review threads from OpenReview v2 API.

Output: one JSONL row per submission, with a flat `thread_events` list that
preserves the full conversation tree (review → rebuttal → follow-up → …).

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
        m = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)\s*[:：]?\s*(.*)\s*$', s)
        if m:
            return float(m.group(1)), m.group(2).strip() or None
        return None, s
    return None, None


def find_rating_field(content: dict) -> Optional[str]:
    """Find the most likely rating field name in content keys."""
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
    """Extract numeric rating from content."""
    fname = find_rating_field(content)
    if not fname:
        return None
    num, _ = parse_numeric_rating(content.get(fname))
    return num


def extract_confidence(content: dict) -> Optional[float]:
    """Extract numeric confidence from content."""
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
    """Extract the main text body from a reply's content."""
    if not isinstance(content, dict):
        return None
    for field in TEXT_FIELDS:
        if field in content:
            s = safe_str(content[field])
            if s:
                return s
    return None


def join_review_text(content: dict) -> Optional[str]:
    """Merge all text-like fields in a review into one string."""
    if 'review' in content:
        s = safe_str(content['review'])
        if s:
            return s
    texts = []
    for k, v in content.items():
        val = _val(v)
        if isinstance(val, str):
            s = val.strip()
            if s and not re.fullmatch(r'[0-9.:/,_-]+', s) and len(s) >= 3:
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
    """Download PDF and extract section headings via pymupdf font-size heuristic."""
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

            # Collect all text spans with font sizes
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

            # Find the body font size (most common)
            from collections import Counter
            size_counts = Counter(round(s[0], 1) for s in all_spans)
            body_size = size_counts.most_common(1)[0][0]

            # Headings: larger than body, short text, match section patterns
            section_pattern = re.compile(
                r'^(?:(\d+(?:\.\d+)*)\s+)?'
                r'(Introduction|Related\s+Work|Background|Method|Approach|Model|'
                r'Experiments?|Results?|Evaluation|Discussion|Conclusion|'
                r'Acknowledgements?|References|Appendix|Limitations|'
                r'Abstract|Supplementary|Analysis|Setup|Ablation|'
                r'Implementation|Training|Dataset)',
                re.IGNORECASE
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
    """Classify a reply into: review | rebuttal | comment | meta_review | decision."""
    invs = reply.get("invitations", [])
    sigs = reply.get("signatures", [])

    # Decision
    if any(x.endswith("Decision") for x in invs):
        return "decision"

    # Official review
    if any(x.endswith("Official_Review") for x in invs):
        return "review"

    # Meta review
    for inv in invs:
        inv_lower = inv.lower()
        if "meta_review" in inv_lower or "metareview" in inv_lower:
            return "meta_review"

    # Author rebuttal
    for inv in invs:
        inv_lower = inv.lower()
        if any(p in inv_lower for p in [
            "/authors/-/official_comment", "/authors/-/rebuttal",
            "author_response", "authors_official_comment"
        ]):
            return "rebuttal"
        if "/authors" in inv_lower and ("comment" in inv_lower or "rebuttal" in inv_lower):
            return "rebuttal"
    for sig in sigs:
        if sig.endswith("/Authors") and venue_id in sig:
            return "rebuttal"

    # Everything else is a comment (reviewer/AC discussion)
    return "comment"


def extract_role_and_actor(reply: dict, venue_id: str, event_type: str) -> Tuple[str, Optional[str]]:
    """Extract (role, actor_id) from signatures."""
    sigs = reply.get("signatures", [])
    if not sigs:
        return "unknown", None

    sig = sigs[0]  # primary signature

    if event_type == "rebuttal":
        return "author", sig

    if event_type == "decision":
        return "program_chair", sig

    if event_type == "meta_review":
        return "area_chair", sig

    # For review and comment, infer from signature
    sig_lower = sig.lower()
    if "area_chair" in sig_lower or "area_chairs" in sig_lower:
        return "area_chair", sig
    if "reviewer" in sig_lower:
        return "reviewer", sig
    if sig.endswith("/Authors"):
        return "author", sig

    return "unknown", sig


# =========================
# Participant extraction
# =========================
def extract_participants(replies: List[dict], venue_id: str) -> Dict:
    """Extract unique participants from all reply signatures."""
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
    """Convert millisecond timestamp to ISO 8601 string."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            # OpenReview timestamps are in milliseconds
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        return str(ts)
    except Exception:
        return str(ts)


@with_retry(max_retries=3, delay=2)
def _get_note_edits(client, note_id: str):
    return client.get_note_edits(note_id=note_id, sort="tcdate:asc")


def build_rating_update_events(client, review_reply: dict, review_event_id: str,
                                venue_id: str) -> List[Dict]:
    """Query edit history for a review and emit rating_update events for rating changes."""
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


def build_thread_events(replies: List[dict], venue_id: str,
                         client=None, with_edits: bool = False) -> List[Dict]:
    """Build flat list of thread events from all replies."""
    events = []

    for r in replies:
        eid = r.get("id")
        content = r.get("content", {}) or {}
        etype = classify_reply(r, venue_id)
        role, actor = extract_role_and_actor(r, venue_id, etype)
        timestamp = ts_to_iso(r.get("tcdate") or r.get("cdate"))
        reply_to = r.get("replyto")

        # Extract text
        if etype == "review":
            text = join_review_text(content)
        elif etype == "decision":
            text = safe_str(content.get("decision"))
            # Also try to get the full decision comment
            comment = safe_str(content.get("comment"))
            if comment:
                text = f"{text}\n\n{comment}" if text else comment
        elif etype == "meta_review":
            text = extract_text(content) or safe_str(content.get("metareview"))
        else:
            text = extract_text(content)

        # Rating & confidence (for reviews)
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
            # OpenReview provenance (for back-referencing / re-fetching)
            "note_id": eid,
            "forum_id": r.get("forum"),
            "invitation": r.get("invitations", [None])[0] if r.get("invitations") else None,
        }
        events.append(event)

        # Optional: rating_update events from edit history
        if with_edits and etype == "review" and client is not None:
            update_events = build_rating_update_events(client, r, eid, venue_id)
            events.extend(update_events)

    # Sort by timestamp, then assign event_index / turn / phase
    events.sort(key=lambda e: e["timestamp"] or "")
    _assign_turn_and_phase(events)
    return events


# Phase mapping: event_type -> default phase name
_PHASE_MAP = {
    "review": "initial_review",
    "rebuttal": "rebuttal",
    "comment": "discussion",
    "meta_review": "meta_review",
    "decision": "decision",
    "rating_update": "post_rebuttal",
}


def _assign_turn_and_phase(events: List[Dict]):
    """In-place: assign event_index, turn, and phase to sorted events.

    Turn logic:
    - turn starts at 1 with the first review
    - Each time the role switches between reviewer↔author, turn increments
    - meta_review / decision / rating_update get their own turn bump
    """
    if not events:
        return

    prev_role = None
    turn = 0

    for idx, ev in enumerate(events):
        ev["event_index"] = idx

        etype = ev["event_type"]
        role = ev["role"]

        # Determine if we should bump the turn
        if etype in ("review",) and prev_role != "reviewer":
            turn += 1
        elif etype == "rebuttal" and prev_role != "author":
            turn += 1
        elif etype == "comment" and role != prev_role:
            turn += 1
        elif etype in ("meta_review", "decision"):
            turn += 1
        elif etype == "rating_update":
            # rating_update follows the reviewer's previous turn
            pass
        elif turn == 0:
            turn = 1

        ev["turn"] = turn
        ev["phase"] = _PHASE_MAP.get(etype, "discussion")

        # Refine phase: if a reviewer comments after rebuttal, it's post_rebuttal
        if etype == "comment" and role == "reviewer":
            # Check if any rebuttal came before this event
            has_prior_rebuttal = any(
                events[j]["event_type"] == "rebuttal" for j in range(idx)
            )
            if has_prior_rebuttal:
                ev["phase"] = "post_rebuttal"

        prev_role = role


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Collect ICLR multi-turn review threads")
    p.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024],
                    help="ICLR years to collect (default: 2021-2024)")
    p.add_argument("--with-edits", action="store_true",
                    help="Query edit history per review to emit rating_update events (slow)")
    p.add_argument("--skip-pdf", action="store_true",
                    help="Skip PDF download and section extraction")
    p.add_argument("--output-dir", type=str, default=".",
                    help="Output directory (default: current dir)")
    p.add_argument("--resume", action="store_true",
                    help="Resume from an existing output file if present")
    p.add_argument("--output-path", type=str, default=None,
                    help="Explicit output JSONL path (useful with --resume)")
    return p.parse_args()


def _fetch_submissions_v2(client_v2, venue_id: str):
    """Try fetching submissions via the v2 API with multiple invitation patterns."""
    # Try to get submission_name from venue group
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

    for name in candidate_names:
        invitation = f"{venue_id}/-/{name}"
        sys.stderr.write(f"  [v2] Trying invitation: {invitation}\n")
        sys.stderr.flush()
        try:
            subs = client_v2.get_all_notes(invitation=invitation, details="replies")
            if subs:
                sys.stderr.write(f"  [v2] -> Found {len(subs)} submissions\n")
                sys.stderr.flush()
                return subs, "v2"
            sys.stderr.write(f"  [v2] -> 0 submissions, trying next...\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"  [v2] -> Error: {e}, trying next...\n")
            sys.stderr.flush()

    # Last resort: venueid search
    sys.stderr.write(f"  [v2] Trying venueid search...\n")
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
    """Fetch submissions via the v1 API (for older conferences like ICLR 2021-2022)."""
    candidate_names = ["Blind_Submission", "Submission"]

    for name in candidate_names:
        invitation = f"{venue_id}/-/{name}"
        sys.stderr.write(f"  [v1] Trying invitation: {invitation}\n")
        sys.stderr.flush()
        try:
            subs = client_v1.get_all_notes(invitation=invitation, details="directReplies")
            if subs:
                sys.stderr.write(f"  [v1] -> Found {len(subs)} submissions\n")
                sys.stderr.flush()
                return subs, "v1"
            sys.stderr.write(f"  [v1] -> 0 submissions, trying next...\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"  [v1] -> Error: {e}, trying next...\n")
            sys.stderr.flush()

    return [], None


def _find_latest_output(output_dir: str, year: int) -> Optional[Path]:
    pattern = f"iclr{year}_threads_*.jsonl"
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


def _v1_note_to_dict(r) -> Optional[dict]:
    """Convert a v1 Note object or dict into the normalised dict format."""
    if hasattr(r, "__dict__"):
        d = {
            "id": getattr(r, "id", None),
            "forum": getattr(r, "forum", None),
            "replyto": getattr(r, "replyto", None),
            "content": getattr(r, "content", {}),
            "signatures": getattr(r, "signatures", []),
            "tcdate": getattr(r, "tcdate", None),
            "cdate": getattr(r, "cdate", None),
            # v1 uses "invitation" (singular string), convert to list
            "invitations": [getattr(r, "invitation", "")] if getattr(r, "invitation", None) else [],
        }
        return d
    elif isinstance(r, dict):
        d = dict(r)
        if "invitation" in d and "invitations" not in d:
            d["invitations"] = [d["invitation"]] if d["invitation"] else []
        return d
    return None


@with_retry(max_retries=3, delay=2)
def _v1_fetch_all_forum_notes(client_v1, forum_id: str) -> List[dict]:
    """Fetch ALL notes in a v1 forum thread (all levels of nesting).

    v1's directReplies only returns first-level replies. To get the full thread
    (rebuttals, discussions, nested replies), we fetch all notes with forum=forum_id
    and exclude the submission itself.
    """
    all_notes = client_v1.get_all_notes(forum=forum_id)
    replies = []
    for n in all_notes:
        # Skip the submission itself (replyto is None or id == forum_id)
        if n.id == forum_id:
            continue
        d = _v1_note_to_dict(n)
        if d:
            replies.append(d)
    return replies


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    username = os.environ["OPENREVIEW_USERNAME"]
    password = os.environ["OPENREVIEW_PASSWORD"]

    # Create both v2 and v1 clients
    client_v2 = openreview.api.OpenReviewClient(
        baseurl=BASEURL_V2, username=username, password=password,
    )
    client_v1 = openreview.Client(
        baseurl=BASEURL_V1, username=username, password=password,
    )

    for year in args.years:
        venue_id = f"ICLR.cc/{year}/Conference"
        sys.stderr.write(f"\n{'='*60}\n")
        sys.stderr.write(f"Collecting ICLR {year} (venue: {venue_id})\n")
        sys.stderr.write(f"{'='*60}\n")
        sys.stderr.flush()

        sys.stderr.write(f"Fetching submissions...\n")
        sys.stderr.flush()

        # Try v2 first, then fall back to v1
        subs, api_version = _fetch_submissions_v2(client_v2, venue_id)
        if not subs:
            sys.stderr.write(f"  v2 API found nothing, trying v1 API...\n")
            sys.stderr.flush()
            subs, api_version = _fetch_submissions_v1(client_v1, venue_id)

        if not subs:
            sys.stderr.write(f"No submissions found for ICLR {year}, skipping.\n")
            sys.stderr.flush()
            continue

        sys.stderr.write(f"Using {api_version} API, {len(subs)} submissions\n")
        sys.stderr.flush()

        total_subs = len(subs)
        sys.stderr.write(f"Found {total_subs} submissions\n")
        sys.stderr.flush()

        output_path = None
        if args.output_path:
            output_path = args.output_path
        elif args.resume:
            latest = _find_latest_output(args.output_dir, year)
            if latest:
                output_path = str(latest)

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                args.output_dir, f"iclr{year}_threads_{timestamp}.jsonl"
            )

        stats = {
            "total_submissions": total_subs,
            "submissions_with_events": 0,
            "total_events": 0,
            "event_types": {},
        }

        processed_ids = set()
        if args.resume and os.path.exists(output_path):
            processed_ids, existing_stats = _load_existing(output_path)
            stats["submissions_with_events"] += existing_stats["submissions_with_events"]
            stats["total_events"] += existing_stats["total_events"]
            for et, count in existing_stats["event_types"].items():
                stats["event_types"][et] = stats["event_types"].get(et, 0) + count

            sys.stderr.write(
                f"Resuming from {output_path}, already have {len(processed_ids)} papers\n"
            )
            sys.stderr.flush()

        open_mode = "a" if args.resume and os.path.exists(output_path) else "w"
        with open(output_path, open_mode, encoding="utf-8") as fout:
            for sub_idx, sub in enumerate(subs):
                paper_id = sub.forum or sub.id
                if paper_id in processed_ids:
                    sys.stderr.write(
                        f"\r[{year}] {sub_idx + 1}/{total_subs} "
                        f"(paper: {paper_id[:20]}...) "
                        f"skipped (resume)    "
                    )
                    sys.stderr.flush()
                    continue

                # Get replies: v2 uses details["replies"], v1 needs full forum fetch
                if api_version == "v1":
                    try:
                        replies = _v1_fetch_all_forum_notes(client_v1, paper_id)
                    except Exception as e:
                        sys.stderr.write(f"\n  Failed to fetch forum {paper_id}: {e}\n")
                        sys.stderr.flush()
                        replies = []
                else:
                    replies = sub.details.get("replies", []) or []

                # Progress
                sys.stderr.write(
                    f"\r[{year}] {sub_idx + 1}/{total_subs} "
                    f"(paper: {paper_id[:20]}...) "
                    f"replies={len(replies)}    "
                )
                sys.stderr.flush()

                # Paper context
                title = safe_note_field(sub, "title")
                abstract = safe_note_field(sub, "abstract")
                keywords = safe_note_keywords(sub)
                pdf_val = safe_note_field(sub, "pdf")
                pdf_url = normalize_pdf_url(pdf_val)
                web_url = f"{OPENREVIEW_WEB_BASE}/forum?id={paper_id}"

                # Sections index (optional)
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

                # Participants
                participants = extract_participants(replies, venue_id)

                # Thread events
                thread_events = build_thread_events(
                    replies, venue_id,
                    client=client_v2 if args.with_edits else None,
                    with_edits=args.with_edits,
                )

                # Decision (extract from events for convenience)
                decision = None
                for ev in thread_events:
                    if ev["event_type"] == "decision":
                        decision = ev["text"]
                        # Often "Accept" or "Reject" is the first line
                        if decision and "\n" in decision:
                            decision_first_line = decision.split("\n")[0].strip()
                            if any(kw in decision_first_line.lower() for kw in ["accept", "reject"]):
                                decision = decision_first_line
                        break

                row = {
                    "venue": "ICLR",
                    "year": year,
                    "submission_id": paper_id,
                    "paper_context": paper_context,
                    "participants": participants,
                    "decision": decision,
                    "thread_events": thread_events,
                }

                # Remove problematic Unicode characters that cause editor warnings
                line = json.dumps(row, ensure_ascii=False)
                # U+2028 Line Separator, U+2029 Paragraph Separator
                # U+0000-U+001F control chars (except \n \r \t which json.dumps escapes)
                # U+FEFF BOM, U+FFFD replacement char, U+200B-U+200F zero-width chars
                line = re.sub(r'[\u2028\u2029\u0000-\u0008\u000b\u000c\u000e-\u001f\ufeff\u200b-\u200f\u202a-\u202e\u2066-\u2069]', ' ', line)
                fout.write(line + "\n")

                # Stats
                if thread_events:
                    stats["submissions_with_events"] += 1
                stats["total_events"] += len(thread_events)
                for ev in thread_events:
                    et = ev["event_type"]
                    stats["event_types"][et] = stats["event_types"].get(et, 0) + 1

        # Summary
        sys.stderr.write(f"\n\n--- ICLR {year} Summary ---\n")
        sys.stderr.write(f"Total submissions: {stats['total_submissions']}\n")
        sys.stderr.write(f"Submissions with events: {stats['submissions_with_events']}\n")
        sys.stderr.write(f"Total events: {stats['total_events']}\n")
        for et, count in sorted(stats["event_types"].items()):
            sys.stderr.write(f"  {et}: {count}\n")
        sys.stderr.write(f"Output: {output_path}\n")
        sys.stderr.flush()

        print(f"[ICLR {year}] Saved {stats['total_submissions']} submissions "
              f"({stats['total_events']} events) to {output_path}")


if __name__ == "__main__":
    main()
