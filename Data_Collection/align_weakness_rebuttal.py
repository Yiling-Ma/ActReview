import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_request_ts = 0.0
_shutdown_event = threading.Event()

def _interruptible_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    deadline = time.time() + seconds
    while not _shutdown_event.is_set():
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))
    raise KeyboardInterrupt("Shutdown requested")

def _get_azure_config():
    cfg = getattr(_thread_local, "azure_cfg", None)
    if cfg:
        return cfg
    endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key        = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
    if not (endpoint and key and deployment):
        raise RuntimeError("Missing Azure OpenAI env vars.")
    cfg = {
        "endpoint": endpoint.rstrip("/"),
        "key": key,
        "deployment": deployment,
        "api_version": api_version,
    }
    _thread_local.azure_cfg = cfg
    return cfg

def _azure_chat_completion(
    messages: List[Dict[str, str]],
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    cfg = _get_azure_config()
    path = f"/openai/deployments/{cfg['deployment']}/chat/completions"
    query = urllib.parse.urlencode({"api-version": cfg["api_version"]})
    url = f"{cfg['endpoint']}{path}?{query}"

    payload: Dict[str, Any] = {"messages": messages}
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if response_format is not None:
        payload["response_format"] = response_format

    max_retries = int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", "8"))
    backoff_base = float(os.environ.get("AZURE_OPENAI_BACKOFF_BASE", "2.0"))
    backoff_cap = float(os.environ.get("AZURE_OPENAI_MAX_BACKOFF", "90.0"))
    min_interval = float(os.environ.get("AZURE_OPENAI_MIN_INTERVAL_SEC", "0.6"))
    request_timeout = float(os.environ.get("AZURE_OPENAI_TIMEOUT_SEC", "120"))

    def _should_retry_http(status: int, detail: str) -> bool:
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if status == 403 and "temporarily blocked" in detail.lower():
            return True
        return False

    last_error = None
    for attempt in range(max_retries + 1):
        if _shutdown_event.is_set():
            raise KeyboardInterrupt("Shutdown requested")
        # Global pacing across threads to avoid burst traffic.
        global _last_request_ts
        with _rate_lock:
            now = time.time()
            wait = (_last_request_ts + min_interval) - now
            if wait > 0:
                _interruptible_sleep(wait)
            _last_request_ts = time.time()

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": cfg["key"],
                "x-ms-client-request-id": str(uuid.uuid4()),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"Azure OpenAI HTTP {e.code}: {detail[:500]}")
            if attempt >= max_retries or not _should_retry_http(e.code, detail):
                raise last_error from e

            retry_after = e.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_s = float(retry_after)
            else:
                sleep_s = min(backoff_cap, backoff_base ** attempt) + random.uniform(0, 0.8)
            _interruptible_sleep(sleep_s)
        except Exception as e:
            last_error = RuntimeError(f"Azure OpenAI request failed: {e}")
            if attempt >= max_retries:
                raise last_error from e
            sleep_s = min(backoff_cap, backoff_base ** attempt) + random.uniform(0, 0.8)
            _interruptible_sleep(sleep_s)
    raise last_error if last_error else RuntimeError("Azure OpenAI request failed with unknown error")


FILTER_OUT_LABELS = {"debate", "score_update", "acknowledgment"}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Segment initial review into weakness points
# ═══════════════════════════════════════════════════════════════════

WEAKNESS_SEGMENTATION_PROMPT = """
You are analyzing a peer review. Segment the review into individual, independent weakness points.

REVIEW TEXT:
{review_text}

TASK:
1. Identify all distinct weakness points, concerns, or questions raised by the reviewer
2. Each point should be a complete, self-contained critique
3. Number sequentially (Point 1, Point 2, ...)

OUTPUT FORMAT:
Point 1: [First weakness]

Point 2: [Second weakness]

GUIDELINES:
- Merge closely related sub-points into one
- Each point should be substantial
- Preserve the reviewer's original wording
- Extract from "Weaknesses:" / "Concerns:" sections if present
- Extract ONLY technical concerns, questions, or requests for improvement
- DO NOT extract attitude statements or score announcements as weaknesses
  (e.g., "I will give a low score", "I am not convinced overall", "borderline reject")
  These are the reviewer's verdict, not technical weakness points
- DO NOT extract any text that references a rebuttal or revision
  (e.g., "even after the rebuttal...", "the revised paper still...", "after revisions...")
  Such sentences belong to a follow-up comment, not the initial review
"""


def segment_review_into_weaknesses(review_text: str) -> List[Dict[str, str]]:
    try:
        result_text = _azure_chat_completion(
            messages=[
                {"role": "system", "content": "You segment peer reviews into weakness points."},
                {"role": "user",   "content": WEAKNESS_SEGMENTATION_PROMPT.format(review_text=review_text)}
            ],
            max_completion_tokens=2000
        )
        weakness_points = []
        for num, content in re.findall(
            r'Point\s+(\d+):\s*(.*?)(?=Point\s+\d+:|$)',
            result_text, re.DOTALL | re.IGNORECASE
        ):
            weakness_points.append({"id": f"W{num}", "content": content.strip()})
        return weakness_points
    except Exception as e:
        print(f"Segmentation failed: {e}")
        return [{"id": "W1", "content": review_text}]


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Map each weakness to its specific rebuttal segment
# ═══════════════════════════════════════════════════════════════════

WEAKNESS_REBUTTAL_MAPPING_PROMPT = """
You are analyzing a peer review discussion. Map each weakness point to the SPECIFIC part of the rebuttal that directly addresses it.

WEAKNESS POINTS:
{weakness_points}

REBUTTAL TEXT:
{rebuttal_text}

TASK:
For each weakness (W1, W2, ...), extract ONLY the rebuttal segment that directly responds to that specific weakness.

OUTPUT FORMAT (one line per weakness):
W1 -> [Exact rebuttal text addressing W1] (Confidence: 0.95)
W2 -> No Response (Confidence: 1.0)

CRITICAL RULES:
- Extract EXACT text from the rebuttal — do NOT paraphrase
- Map each weakness to its OWN specific rebuttal segment
- Do NOT assign the same rebuttal segment to multiple weaknesses unless it explicitly addresses both
- If a weakness is not addressed, write "No Response"
- If multiple weaknesses share the same rebuttal segment, DUPLICATE the full text for each — do NOT write "Same segment as W3" or any cross-reference
- Confidence: 0.0-1.0
- If the rebuttal starts with a general acknowledgment paragraph followed by numbered or
  lettered responses (e.g., "A1.", "Q1.", "(1)", "**W1**"), extract the SPECIFIC numbered
  response for each weakness — not the general preamble that appears before all responses
- If you cannot find a specific response for a weakness but only a general preamble, write "No Response"
- IMPORTANT: A rebuttal sometimes QUOTES the reviewer's original weakness before answering it
  (e.g., "> Lack of baselines..." or "\"Missing comparison...\""). The quoted reviewer text is NOT
  the response. Only extract the AUTHORS' actual reply that follows the quote. If there is no
  substantive author reply after the quote, write "No Response"
- If the extracted segment is identical or near-identical to the weakness text itself, that means
  the rebuttal only quoted the concern without addressing it — write "No Response" instead
"""


def _is_quote_only(rebuttal: str, weakness: str) -> bool:
    """Return True if the rebuttal segment appears to just quote/restate the weakness
    without providing a substantive author reply.

    Uses a symmetric overlap check: both the fraction of weakness tokens in the rebuttal
    AND the fraction of rebuttal tokens that are weakness tokens must be high.
    This avoids false positives when a real reply naturally shares topic keywords.
    """
    _STOPWORDS = {
        'the','a','an','in','on','at','to','for','of','and','or','but',
        'is','are','was','were','be','this','that','with','from','by','as',
        'it','i','we','our','my','your','their','not','no','do','does',
        'can','will','may','have','has','had','been','would','could',
        'should','if','so','than','then','also','more','most','very',
        'about','which','who','what','when','where','how','why','all',
        'any','such','these','those','its','they','them','their','there'
    }
    def _tokens(s: str):
        return set(w for w in re.findall(r'\b[a-z]{3,}\b', s.lower()) if w not in _STOPWORDS)

    wt, rt = _tokens(weakness), _tokens(rebuttal)
    if not wt or not rt:
        return False
    intersection = wt & rt
    # w_in_r: how much of the weakness vocabulary is covered by the rebuttal
    # r_in_w: how much of the rebuttal vocabulary comes from the weakness
    # A quote-only segment scores high on BOTH (it IS the weakness).
    # A real rebuttal shares topic words but adds new vocabulary → r_in_w stays low.
    # Thresholds: w_in_r > 0.55 (most weakness words present in the rebuttal)
    #             r_in_w > 0.45 (most rebuttal words come from the weakness)
    # Asymmetric because quoted rebuttals often carry metadata like "(incomplete...)"
    # or "> ..." prefixes that slightly dilute r_in_w, so we use a looser bound there.
    w_in_r = len(intersection) / len(wt)
    r_in_w = len(intersection) / len(rt)
    return w_in_r > 0.55 and r_in_w > 0.45


def map_weaknesses_to_rebuttals(
    weakness_points: List[Dict[str, str]],
    rebuttal_text: str
) -> List[Dict[str, Any]]:
    weakness_text = "\n\n".join(f"{w['id']}: {w['content']}" for w in weakness_points)
    try:
        result_text = _azure_chat_completion(
            messages=[
                {"role": "system", "content": "You map peer review weaknesses to specific rebuttal segments."},
                {"role": "user",   "content": WEAKNESS_REBUTTAL_MAPPING_PROMPT.format(
                    weakness_points=weakness_text,
                    rebuttal_text=rebuttal_text[:3000]
                )}
            ],
            max_completion_tokens=3000
        )
        mappings = []
        seen = set()
        for pattern in [
            r'W(\d+)\s*->\s*(No Response)\s*\(Confidence:\s*([\d.]+)\)',
            r'W(\d+)\s*->\s*(.*?)\s*\(Confidence:\s*([\d.]+)\)',
        ]:
            for wid, rebuttal_content, confidence in re.findall(
                pattern, result_text, re.DOTALL | re.IGNORECASE
            ):
                if wid in seen:
                    continue
                weakness = next((w for w in weakness_points if w["id"] == f"W{wid}"), None)
                if weakness:
                    seen.add(wid)
                    rebuttal_clean = rebuttal_content.strip()

                    # ── Post-processing A: quote-only detection ────────────
                    # If the rebuttal just quotes / restates the weakness without
                    # adding an author reply, discard it as "No Response".
                    # Uses symmetric token overlap so real replies sharing topic
                    # keywords (e.g., "We added ODIN comparison in Table 3") are kept.
                    if rebuttal_clean != "No Response":
                        if _is_quote_only(rebuttal_clean, weakness["content"]):
                            rebuttal_clean = "No Response"

                    # ── Post-processing B: cross-reference resolution ─────────
                    # LLM sometimes outputs "Same segment as W4" instead of the
                    # actual rebuttal text.  Resolve by copying the referenced
                    # weakness's rebuttal so every entry is a real text span.
                    resolved_from_ref = False
                    if rebuttal_clean != "No Response":
                        ref_match = re.match(
                            r'^[Ss]ame\s+(segment|rebuttal|response|text)\s+as\s+W(\d+)',
                            rebuttal_clean
                        )
                        if ref_match:
                            ref_wid = ref_match.group(2)
                            ref_mapping = next(
                                (m for m in mappings
                                 if m["weakness"]["id"] == f"W{ref_wid}"
                                 and m["rebuttal"] != "No Response"),
                                None
                            )
                            if ref_mapping:
                                rebuttal_clean = ref_mapping["rebuttal"]
                                resolved_from_ref = True
                            else:
                                rebuttal_clean = "No Response"

                    # ── Post-processing C: shared-preamble deduplication ───────
                    # If this exact segment was already assigned to a previous weakness,
                    # it is a generic preamble rather than a specific response.
                    # Skip this check for cross-reference resolved entries — those
                    # intentionally share a rebuttal segment across weaknesses.
                    if rebuttal_clean != "No Response" and not resolved_from_ref:
                        already_used = any(
                            m["rebuttal"] == rebuttal_clean
                            for m in mappings
                            if m["rebuttal"] != "No Response"
                        )
                        if already_used:
                            rebuttal_clean = "No Response"

                    mappings.append({
                        "weakness":   weakness,
                        "rebuttal":   rebuttal_clean,
                        "confidence": float(confidence)
                    })
        return mappings
    except Exception as e:
        print(f"Mapping failed: {e}")
        return [{"weakness": w, "rebuttal": rebuttal_text, "confidence": 0.5}
                for w in weakness_points]


# ═══════════════════════════════════════════════════════════════════
# STEP 3: Map each follow-up comment to its target weakness
#
# WHY THIS IS NEEDED:
#   A reviewer's follow-up comment (turn > 1) typically addresses ONE
#   specific weakness from their original review, not all of them.
#   Without this step every follow-up gets attached to every weakness,
#   leading to wrong classification and noisy pairs.
#
# OUTPUT:
#   {
#     "W1": ["follow-up text A", "follow-up text B"],
#     "W2": [],
#     "W3": ["follow-up text C"],
#     "UNRELATED": ["follow-up text D"]   ← not about any specific weakness
#   }
# ═══════════════════════════════════════════════════════════════════

FOLLOWUP_WEAKNESS_MAPPING_PROMPT = """
You are analyzing a peer review discussion thread.

A reviewer wrote an initial review with multiple weakness points (listed below).
The authors wrote a rebuttal.
Now the reviewer has written one or more follow-up comments.

Your task: for each follow-up comment, identify WHICH weakness point(s) from the original review it is responding to.

ORIGINAL WEAKNESS POINTS:
{weakness_points}

FOLLOW-UP COMMENTS:
{followup_comments}

TASK:
For each follow-up comment (F1, F2, ...), output which weakness ID(s) it targets.

OUTPUT FORMAT (one line per follow-up):
F1 -> W3 (Confidence: 0.90)
F2 -> W1, W2 (Confidence: 0.75)
F3 -> UNRELATED (Confidence: 0.85)

RULES:
- Use the weakness ID (W1, W2, ...) from the list above
- If a follow-up addresses multiple weaknesses, list all of them separated by commas
- If the follow-up is not specifically about any weakness (e.g., general score update,
  general satisfaction), write "UNRELATED"
- Confidence: 0.0-1.0
- One line per follow-up comment
"""


def map_followups_to_weaknesses(
    weakness_points: List[Dict[str, str]],
    followup_events: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a dict mapping weakness_id -> list of follow-up events targeting that weakness.
    Also includes key "UNRELATED" for follow-ups not tied to any specific weakness.

    If there are no follow-ups, returns empty lists for each weakness.
    """
    # Initialize result: every weakness gets an empty list
    result: Dict[str, List] = {w["id"]: [] for w in weakness_points}
    result["UNRELATED"] = []

    if not followup_events:
        return result

    # If only one weakness, no need to call LLM — all follow-ups belong to it
    if len(weakness_points) == 1:
        result[weakness_points[0]["id"]] = followup_events
        return result

    weakness_text = "\n\n".join(f"{w['id']}: {_safe_text(w.get('content'))}" for w in weakness_points)
    followup_text = "\n\n".join(
        f"F{i+1}: {_safe_text(e.get('text'))[:600]}"
        for i, e in enumerate(followup_events)
    )

    try:
        result_text = _azure_chat_completion(
            messages=[
                {"role": "system", "content": "You map reviewer follow-up comments to specific weakness points."},
                {"role": "user",   "content": FOLLOWUP_WEAKNESS_MAPPING_PROMPT.format(
                    weakness_points=weakness_text,
                    followup_comments=followup_text
                )}
            ],
            max_completion_tokens=1000
        )

        # Parse: "F1 -> W3 (Confidence: 0.90)" or "F2 -> W1, W2 (...)"
        for match in re.finditer(
            r'F(\d+)\s*->\s*([\w,\s]+?)\s*\(Confidence:\s*([\d.]+)\)',
            result_text, re.IGNORECASE
        ):
            f_idx   = int(match.group(1)) - 1   # 0-based
            targets = [t.strip().upper() for t in match.group(2).split(",")]
            conf    = float(match.group(3))

            if f_idx < 0 or f_idx >= len(followup_events):
                continue

            event = followup_events[f_idx]

            for target in targets:
                if target == "UNRELATED":
                    result["UNRELATED"].append(event)
                elif target in result:
                    result[target].append(event)
                # If target not recognized, fall back to UNRELATED
                else:
                    result["UNRELATED"].append(event)

        return result

    except Exception as e:
        print(f"Follow-up mapping failed: {e}")
        # Fallback: attach all follow-ups to all weaknesses (old behavior)
        # This is safe but noisy — better than losing data
        for w in weakness_points:
            result[w["id"]] = followup_events
        return result


# ═══════════════════════════════════════════════════════════════════
# STEP 4a: Rule-based follow-up classifier
# ═══════════════════════════════════════════════════════════════════

def classify_turn_relationship_rule_based(
    initial_weakness: str,
    follow_up_text: str
) -> Tuple[str, str, float]:
    """Returns (label, confidence_tier, confidence_score)."""
    fl = follow_up_text.lower()
    il = initial_weakness.lower()

    # ── 0a. Score update ──────────────────────────────────────────
    # Only trigger when the reviewer is ANNOUNCING a score change.
    # Guard: if they already raised their score (past tense), it's an acknowledgment —
    # checked in block 0b. Skip score_update for past-tense patterns.
    _past_tense_raise = bool(re.search(
        r'\bhave\s+(raised|increased|updated)\s+(my\s+)?score\b', fl
    ))
    if not _past_tense_raise and re.search(
        r'\b(willing|happy|glad)\s+to\s+(increase|raise|update|change)\s+(my\s+)?(score|rating)\b'
        r'|\bi\s+(will|would|am\s+going\s+to)\s+(raise|increase|update|change)\s+(my\s+)?(score|rating)\b'
        r'|\braise\s+my\s+score\s+(to|from)\b'
        r'|\bscore\s+(of|to)\s+\d\b'
        r'|\b(increase|raise)\s+(the\s+)?score\b',
        fl
    ):
        return "score_update", "high", 0.95

    # ── 0b. Acknowledgment (no concurrent negative signal) ────────
    has_negative_signal = bool(re.search(
        r'\b(still|however|but|unfortunately|remains?'
        r'|not\s+(convinced|clear|addressed|resolved|sufficient)'
        r'|concern\s+remains?|disagree)\b',
        fl
    ))
    if not has_negative_signal:
        if re.search(
            r'\b(authors?\s+have\s+(addressed|answered|resolved|clarified))'
            r'|\b(my\s+)?(concern|question|issue)s?\s+(have\s+been|has\s+been)\s+(addressed|resolved)'
            r'|\bsatisfied\s+with\s+the\s+(response|rebuttal)'
            r'|\bno\s+(further|more|additional)\s+(concern|question|issue|comment)s?\b'
            r'|\bi\s+(will\s+)?(accept|support)\s+(the\s+)?(paper|submission)'
            r'|\b(happy|glad|willing)\s+to\s+(accept|support)\b'
            r'|\b(response|rebuttal|clarification)\s+is\s+(clear|satisfactory|convincing|adequate)\b'
            r'|\bthank\s+you[,\.]?\s+(the\s+)?(response|clarification)\s+is\b'
            r'|\bi\s+have\s+(raised|increased|updated)\s+(my\s+)?score\b'
            r'|\bmy\s+(concern|question|issue)s?\s+(have\s+been|are\s+(now\s+)?)addressed\b',
            fl
        ):
            return "acknowledgment", "high", 0.95

    # ── 1. Debate Tier A: explicit rebuttal rejection ─────────────
    debate_tier_a = [
        r'\b(this|the)\s+(response|rebuttal|answer)\s+(does\s+not|didn\'t|doesn\'t|failed\s+to)\s+(address|resolve|answer|convince)\b',
        r'\bthis\s+does\s+not\s+address\b',
        r'\bdoes\s+not\s+(address|resolve|answer|clarify)\s+(my|the|this)\b',
        r'\bfail(s|ed)\s+to\s+(address|resolve|answer|clarify)\b',
        r'\b(concern|issue|question|problem)\s+remains?\s+(unaddressed|open|unresolved)\b',
        r'\bremains?\s+unaddressed\b',
        r'\b(primary|main|key|major)\s+concern\s+remains?\b',
        r'\bi\s+am\s+not\s+convinced\s+by\b',
        r'\bnot\s+convinced\s+by\s+(the|this|your)\b',
        r'\bi\s+maintain\s+(my|that)\b',
        r'\bi\s+still\s+disagree\b',
        r'\bstill\s+not\s+(convinced|addressed|resolved|clear|adequate|satisfied)\b',
        r'\bi\s+still\s+(do\s+not|don\'t)\s+(agree|accept|believe)\b',
    ]
    for p in debate_tier_a:
        if re.search(p, fl):
            return "debate", "high", 0.92

    # Pre-check: polite opener + contrast word + rejection signal
    if re.search(r'\bthank(s|\s+you)\b', fl) and re.search(
        r'\b(unfortunately|however|yet|nonetheless|that said)\b', fl
    ):
        if re.search(
            r'\b(concern|issue|question)\s+remains?'
            r'|remains?\s+(unaddressed|unconvincing|problematic)'
            r'|not\s+(addressed|resolved|answered|convinced)'
            r'|not\s+all\s+of\s+(my\s+)?(concern|question|issue)'
            r'|some\s+but\s+not\s+all'
            r'|does\s+not\s+(address|resolve)'
            r'|fail(s|ed)?\s+to\s+(address|resolve)',
            fl
        ):
            return "debate", "high", 0.90

    # ── 1b. Debate Tier B: weaker signals ─────────────────────────
    # Only fire when no "I want more work" signals are present.
    # "I still think X is needed" = same_issue (action request), NOT debate.
    same_issue_present = bool(re.search(
        r'\bcould\s+you\s+(also|please)'
        r'|\bplease\s+(also|consider|add|include)'
        r'|\badditionally[,\s]'
        r'|\bit\s+would\s+(be|also)\s+(nice|good|better|helpful)'
        r'|\bone\s+more\s+(suggestion|request|question)'
        r'|\bfurther(more)?\b'
        r'|\bin\s+addition\b',
        fl
    ))
    action_request_present = bool(re.search(
        r'\b(needed|required|necessary|should\s+be|must\s+be|need\s+to|have\s+to'
        r'|are\s+needed|is\s+needed|would\s+be\s+(helpful|useful|necessary|important))\b',
        fl
    ))
    if not same_issue_present and not action_request_present and has_negative_signal:
        for p in [
            r'\bi\s+still\s+think\b',
            r'\bi\s+still\s+believe\b',
            r'\bstill\s+think\s+(this|that|the)\b',
            r'\bnot\s+fully\s+(convinced|addressed|resolved)\b',
            r'\bit\s+is\s+still\s+(unclear|not\s+clear)\b',
            r'\bremains?\s+(unclear|unconvincing)\b',
        ]:
            if re.search(p, fl):
                return "debate", "medium", 0.75

    # ── 2. Same issue: refinement / additional request ────────────
    for p in [
        r'\bthank\s+you\b.*\b(address|fix|clarif|revis)',
        r'\badditionally,?\s+(it\s+would|please|could\s+you)',
        r'\bone\s+more\s+(suggestion|question|request|concern)\b',
        r'\bfinal\s+suggestion\b',
        r'\bminor\s+(point|comment|suggestion)\b',
        r'\bit\s+would\s+(be|also)\s+(nice|good|better|helpful)',
        r'\bcould\s+you\s+(also|please)',
        r'\bplease\s+(also|consider|add|include)',
        r'\b(furthermore|moreover|in\s+addition)\b',
        r'\balso,?\s+(could|would|please)',
        r'\bsome\s+(final|more|additional)\s+(suggestion|comment|point)',
    ]:
        if re.search(p, fl):
            return "same_issue", "high", 0.88

    # ── 3. New issue: EXPLICIT topic shift required ───────────────
    same_topic_ref = bool(re.search(
        r'\bthis\s+(issue|concern|problem|point)\b'
        r'|\bmy\s+(original|initial|previous|earlier)\s+(concern|question|point)\b'
        r'|\bas\s+(mentioned|stated|noted)\s+(earlier|above|before)\b',
        fl
    ))
    for p in [
        r'\banother\s+(concern|issue|problem|question)\b',
        r'\bseparately[,\s]',
        r'\bon\s+a\s+(completely\s+)?different\s+(note|topic|point)\b',
        r'\bnew\s+(concern|issue|question)\b',
        r'\bunrelated\s+to\s+(the\s+above|this|my\s+previous)\b',
        r'\bmoving\s+(on\s+)?to\s+(a\s+)?(different|new|separate|another)\b',
    ]:
        if re.search(p, fl) and not same_topic_ref:
            return "new_issue", "high", 0.85

    # ── 4. Keyword overlap ────────────────────────────────────────
    stopwords = {
        'the','a','an','in','on','at','to','for','of','and','or','but',
        'is','are','was','were','be','this','that','with','from','by','as',
        'it','thank','you','your','please','would','could','have','has',
        'not','do','does','did','can','will','may','also','very','more'
    }
    def kw(text):
        return set(w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in stopwords)

    ik, fk = kw(il), kw(fl)
    if ik and fk:
        ratio = len(ik & fk) / len(ik)
        if ratio > 0.35:
            return "same_issue", "medium", min(0.5 + ratio * 0.3, 0.82)

    # ── 5. Pronoun references ─────────────────────────────────────
    if len(re.findall(r'\b(this|that|it|these|those)\b', fl)) >= 3:
        return "same_issue", "low", 0.55

    # ── 6. Gratitude without negative signal ──────────────────────
    if re.search(r'\bthank|\bappreciate|\bgrateful', fl) and not has_negative_signal:
        return "same_issue", "medium", 0.68

    return "same_issue", "medium", 0.60


def needs_llm_verification(
    label: str, confidence_tier: str, confidence_score: float,
    follow_up_text: str
) -> bool:
    if confidence_tier in ("low", "medium") and len(follow_up_text) > 150:
        return True
    if confidence_score < 0.65:
        return True
    if label == "debate" and confidence_tier != "high":
        return True
    if label == "new_issue" and confidence_score < 0.80:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# STEP 4b: LLM follow-up classifier
# ═══════════════════════════════════════════════════════════════════

LLM_CLASSIFICATION_PROMPT = """
You are classifying a reviewer's follow-up comment in a peer review rebuttal thread.

─── INITIAL WEAKNESS (Turn 1) ───
{initial_weakness}

─── AUTHORS' REBUTTAL ───
{rebuttal_summary}

─── REVIEWER'S FOLLOW-UP ───
{follow_up_text}

═══ LABEL DEFINITIONS ═══

1. same_issue
   The reviewer continues, refines, or adds a minor request about THE SAME concern.
   ✓ "Could you also add an experiment on dataset X?"
   ✓ "Thank you. One more thing: please clarify Y."
   ✓ "I still think you need more ablations." ← pushing for more, not rejecting
   ✓ "Additionally, it would help to show Z."

2. new_issue
   The reviewer raises a COMPLETELY DIFFERENT technical concern not in the initial weakness.
   ✓ "Another concern: computational complexity." (if initial was about accuracy)
   ✗ "I still think the ablations are insufficient." ← same topic = same_issue

3. debate
   The reviewer EXPLICITLY states the rebuttal FAILED to address their concern.
   ✓ "This does not address my concern."
   ✓ "I am not convinced by this explanation."
   ✓ "The concern remains unaddressed."
   ✓ "Thank you, but my primary concern remains unaddressed."
   ✗ "I still think more experiments are needed." ← same_issue, not debate
   ✗ "I still believe X." ← ambiguous → default same_issue

4. score_update
   The reviewer ANNOUNCES they are changing their score (present or future tense).
   ✓ "I am willing to raise my score to 6."
   ✓ "I will increase my score."
   ✗ "I have raised my score." ← past tense = already done → acknowledgment

5. acknowledgment
   The reviewer is fully satisfied with no remaining concerns, OR reports they have
   already changed their score (past tense).
   ✓ "The authors have addressed all my concerns."
   ✓ "I have raised my score to 6." ← past tense raise = acknowledgment
   ✓ "In light of the revisions, I have increased my score."

═══ DECISION RULES ═══
Q1: Does reviewer explicitly say the rebuttal FAILED / did NOT address the concern? → debate
Q2: Does reviewer ANNOUNCE a score change in present/future tense? → score_update
Q3: Did reviewer ALREADY raise their score (past tense), or are they fully satisfied? → acknowledgment
Q4: Is this a COMPLETELY different topic from the initial weakness? → new_issue
Q5: Everything else (including "I still think X needs more work") → same_issue

DEFAULT: same_issue

═══ OUTPUT (JSON only) ═══
{{
  "label": "same_issue" | "new_issue" | "debate" | "score_update" | "acknowledgment",
  "confidence": 0.0-1.0,
  "reasoning": "One sentence."
}}
"""


def classify_with_llm(
    initial_weakness: str,
    follow_up_text: str,
    rebuttal_text: str = ""
) -> Tuple[str, float, str]:
    try:
        result_text = _azure_chat_completion(
            messages=[
                {"role": "system", "content": "You classify peer review follow-up comments. Output JSON only."},
                {"role": "user",   "content": LLM_CLASSIFICATION_PROMPT.format(
                    initial_weakness=initial_weakness[:500],
                    rebuttal_summary=(rebuttal_text[:300] if rebuttal_text else "Not available"),
                    follow_up_text=follow_up_text[:800]
                )}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(result_text)
        return (
            result.get("label", "same_issue"),
            float(result.get("confidence", 0.7)),
            result.get("reasoning", "")
        )
    except Exception as e:
        print(f"LLM classification failed: {e}")
        return "same_issue", 0.5, "LLM failed"


# ═══════════════════════════════════════════════════════════════════
# STEP 5: Align weakness-rebuttal pairs from a submission
# ═══════════════════════════════════════════════════════════════════

def align_weakness_rebuttal_pairs(submission_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    thread_events = submission_data.get("thread_events", [])
    paper_context = submission_data["paper_context"]
    if not thread_events:
        return []

    def normalize_role(role: str) -> str:
        r = role.lower().strip()
        if r in ("reviewer", "review"):   return "reviewer"
        if r in ("author",  "authors"):   return "author"
        return r

    reviewer_threads: Dict[str, List] = defaultdict(list)
    for event in thread_events:
        if normalize_role(event.get("role", "")) == "reviewer":
            reviewer_threads[event.get("actor_id", "")].append(event)

    author_events = [
        e for e in thread_events
        if normalize_role(e.get("role", "")) == "author"
    ]

    all_pairs = []

    for reviewer_id, rev_events in reviewer_threads.items():
        rev_events = sorted(rev_events, key=lambda x: x.get("turn", 0))

        # ── Find initial review (turn 1) ──────────────────────────
        initial_review = next((e for e in rev_events if e.get("turn") == 1), None)
        if initial_review is None:
            initial_review = next(
                (e for e in rev_events
                 if e.get("event_type", "").lower() in ("review", "official_review")),
                None
            )
        if initial_review is None and rev_events:
            initial_review = rev_events[0]
        review_text = _safe_text(initial_review.get("text")) if initial_review else ""
        if not initial_review or not review_text:
            continue

        # ── Step 1: Segment weaknesses ────────────────────────────
        print(f"  Segmenting review from {reviewer_id.split('/')[-1]}...")
        weakness_points = segment_review_into_weaknesses(review_text)
        if not weakness_points:
            continue
        print(f"    Found {len(weakness_points)} weakness points")

        # ── Step 2: Map weaknesses → rebuttal segments ────────────
        all_rebuttal_text = "\n\n".join(
            f"[Rebuttal Turn {e.get('turn','?')}]\n{_safe_text(e.get('text'))}"
            for e in author_events
        )
        if not all_rebuttal_text:
            continue

        print(f"    Mapping weaknesses to rebuttals...")
        w2r_mappings = map_weaknesses_to_rebuttals(weakness_points, all_rebuttal_text)
        print(f"    Mapped {len(w2r_mappings)} weakness-rebuttal pairs")

        # ── Step 3 (NEW): Map follow-ups → weakness points ────────
        followup_events = [e for e in rev_events if e.get("turn", 0) > 1]

        if followup_events:
            print(f"    Mapping {len(followup_events)} follow-up(s) to weaknesses...")
            fu2w = map_followups_to_weaknesses(weakness_points, followup_events)
        else:
            fu2w = {w["id"]: [] for w in weakness_points}
            fu2w["UNRELATED"] = []

        # ── Build one pair per weakness ───────────────────────────
        for mapping in w2r_mappings:
            weakness  = mapping["weakness"]
            rebuttal  = mapping["rebuttal"]

            if rebuttal == "No Response":
                continue

            # Rebuttal turn
            turns = [{
                "turn":      2,
                "type":      "rebuttal",
                "text":      rebuttal,
                "timestamp": ""
            }]

            # Only attach follow-ups that specifically target THIS weakness
            for fu_event in fu2w.get(weakness["id"], []):
                turns.append({
                    "turn":      fu_event.get("turn", 3),
                    "type":      "follow_up",
                    "text":      _safe_text(fu_event.get("text")),
                    "timestamp": fu_event.get("timestamp", "")
                })

            turns = sorted(turns, key=lambda x: x["turn"])

            all_pairs.append({
                "venue": submission_data.get("venue"),
                "venue_id": submission_data.get("venue_id"),
                "year": submission_data.get("year"),
                "submission_id": submission_data.get("submission_id"),
                "weakness_id": (
                    f"{submission_data['submission_id']}"
                    f"_{reviewer_id.split('/')[-1]}"
                    f"_{weakness['id']}"
                ),
                "paper_context":    paper_context,
                "reviewer_id":      reviewer_id,
                "initial_weakness": {
                    "text":      weakness["content"],
                    "turn":      1,
                    "timestamp": initial_review.get("timestamp", "")
                },
                "turns": turns
            })

    return all_pairs


# ═══════════════════════════════════════════════════════════════════
# STEP 6: Consolidate multi-turn weakness (classify follow-ups)
# ═══════════════════════════════════════════════════════════════════

def consolidate_multi_turn_weakness(
    weakness_thread: Dict[str, Any],
    use_llm: bool = True
) -> Dict[str, Any]:
    rebuttals  = [t for t in weakness_thread["turns"] if t["type"] == "rebuttal"]
    follow_ups = [t for t in weakness_thread["turns"] if t["type"] == "follow_up"]
    rebuttal_summary = "\n\n".join(_safe_text(r.get("text"))[:200] for r in rebuttals[:2])

    classified_fus = []
    llm_calls = 0

    for fu in follow_ups:
        fu_text = _safe_text(fu.get("text"))
        label, conf_tier, conf_score = classify_turn_relationship_rule_based(
            weakness_thread["initial_weakness"]["text"],
            fu_text
        )
        if use_llm and needs_llm_verification(label, conf_tier, conf_score, fu_text):
            llm_label, llm_conf, reasoning = classify_with_llm(
                weakness_thread["initial_weakness"]["text"],
                fu_text,
                rebuttal_summary
            )
            label      = llm_label
            conf_score = llm_conf
            conf_tier  = "high" if llm_conf > 0.80 else "medium"
            llm_calls += 1
            fu["text"] = fu_text
            fu["classification"] = {
                "label": label, "confidence": conf_tier,
                "confidence_score": conf_score, "method": "llm",
                "reasoning": reasoning
            }
        else:
            fu["text"] = fu_text
            fu["classification"] = {
                "label": label, "confidence": conf_tier,
                "confidence_score": conf_score, "method": "rule"
            }
        classified_fus.append(fu)

    same_issue_fus = [fu for fu in classified_fus if fu["classification"]["label"] == "same_issue"]
    new_issue_fus  = [fu for fu in classified_fus if fu["classification"]["label"] == "new_issue"]
    filtered_fus   = [fu for fu in classified_fus if fu["classification"]["label"] in FILTER_OUT_LABELS]

    return {
        "venue": weakness_thread.get("venue"),
        "venue_id": weakness_thread.get("venue_id"),
        "year": weakness_thread.get("year"),
        "submission_id": weakness_thread.get("submission_id"),
        "weakness_id":   weakness_thread["weakness_id"],
        "paper_context": weakness_thread["paper_context"],

        "consolidated_weakness": {
            "initial":    weakness_thread["initial_weakness"]["text"],
            "follow_ups": [
                {"text": fu["text"],
                 "confidence": fu["classification"]["confidence"],
                 "method":     fu["classification"]["method"]}
                for fu in same_issue_fus
            ]
        },

        "rebuttals": [_safe_text(r.get("text")) for r in rebuttals],

        "metadata": {
            "num_turns":          len(weakness_thread["turns"]) + 1,
            "num_follow_ups":     len(follow_ups),
            "num_same_issue":     len(same_issue_fus),
            "num_new_issues":     len(new_issue_fus),
            "num_debates":        sum(1 for fu in filtered_fus
                                      if fu["classification"]["label"] == "debate"),
            "num_score_updates":  sum(1 for fu in filtered_fus
                                      if fu["classification"]["label"] == "score_update"),
            "num_acknowledgments":sum(1 for fu in filtered_fus
                                      if fu["classification"]["label"] == "acknowledgment"),
            "llm_calls_needed":   llm_calls
        },

        # stored for inspection, NOT split into new units
        "new_issues_text": [fu["text"] for fu in new_issue_fus]
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def _write_jsonl(records: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _checkpoint_paths(output_file: str) -> Tuple[str, str]:
    checkpoint_file = output_file + ".checkpoint"
    state_file = checkpoint_file + ".state.json"
    return checkpoint_file, state_file


def _save_checkpoint(
    output_file: str,
    records: List[Dict[str, Any]],
    processed_indices: set,
    total_submissions: int,
    total_followups: int,
    total_llm_classify: int,
    total_seg: int,
    total_map_wr: int,
    total_map_fu: int,
    label_dist: Dict[str, int]
) -> None:
    checkpoint_file, state_file = _checkpoint_paths(output_file)
    _write_jsonl(records, checkpoint_file)
    state = {
        "total_submissions": total_submissions,
        "processed_indices": sorted(processed_indices),
        "total_followups": total_followups,
        "total_llm_classify": total_llm_classify,
        "total_seg": total_seg,
        "total_map_wr": total_map_wr,
        "total_map_fu": total_map_fu,
        "label_dist": dict(label_dist),
    }
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_resume_state(output_file: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    checkpoint_file, state_file = _checkpoint_paths(output_file)
    if not (os.path.exists(checkpoint_file) and os.path.exists(state_file)):
        return None
    try:
        records = _read_jsonl(checkpoint_file)
        with open(state_file, encoding='utf-8') as f:
            state = json.load(f)
        return records, state
    except Exception as e:
        print(f"Failed to load checkpoint state, starting fresh: {e}")
        return None


def _process_submission_task(
    idx: int,
    submission: Dict[str, Any],
    use_llm: bool
) -> Dict[str, Any]:
    pairs = align_weakness_rebuttal_pairs(submission)

    num_reviewers = len(set(p["reviewer_id"] for p in pairs))
    total_followups = 0
    total_llm_classify = 0
    label_updates: Dict[str, int] = defaultdict(int)
    consolidated_records: List[Dict[str, Any]] = []

    for pair in pairs:
        consolidated = consolidate_multi_turn_weakness(pair, use_llm=use_llm)
        consolidated_records.append(consolidated)

        m = consolidated["metadata"]
        total_followups += m["num_follow_ups"]
        total_llm_classify += m["llm_calls_needed"]
        label_updates["same_issue"] += m["num_same_issue"]
        label_updates["new_issue"] += m["num_new_issues"]
        label_updates["debate"] += m["num_debates"]
        label_updates["score_update"] += m["num_score_updates"]
        label_updates["acknowledgment"] += m["num_acknowledgments"]

    return {
        "index": idx,
        "submission_id": submission.get("submission_id", "Unknown"),
        "records": consolidated_records,
        "num_reviewers": num_reviewers,
        "total_followups": total_followups,
        "total_llm_classify": total_llm_classify,
        "label_updates": dict(label_updates),
    }


def process_all_submissions(
    input_file:  str  = "actionable_threads_2turn.jsonl",
    output_file: str  = "aligned_weakness_rebuttal_pairs.jsonl",
    use_llm:     bool = True,
    batch_size:  int  = 100,
    max_workers: int  = 2,
    resume:      bool = True
) -> List[Dict[str, Any]]:
    _shutdown_event.clear()

    print("\n" + "="*70)
    print("FINE-GRAINED WEAKNESS-REBUTTAL ALIGNMENT v5")
    print("(with per-weakness follow-up mapping + quote/dedup post-processing + tense-aware classification)")
    print("="*70)

    all_pairs: List[Dict] = []
    processed_indices: set = set()
    total_followups = total_llm_classify = total_seg = total_map_wr = total_map_fu = 0
    label_dist: Dict[str, int] = defaultdict(int)

    with open(input_file, encoding='utf-8') as f:
        submissions = [json.loads(line) for line in f]
    total_submissions = len(submissions)
    print(f"Processing {total_submissions} submissions...")
    print(f"use_llm={use_llm}, max_workers={max_workers}, resume={resume}")

    if resume:
        loaded = _load_resume_state(output_file)
        if loaded is not None:
            all_pairs, state = loaded
            processed_indices = set(state.get("processed_indices", []))
            total_followups = int(state.get("total_followups", 0))
            total_llm_classify = int(state.get("total_llm_classify", 0))
            total_seg = int(state.get("total_seg", 0))
            total_map_wr = int(state.get("total_map_wr", 0))
            total_map_fu = int(state.get("total_map_fu", 0))
            label_dist = defaultdict(int, state.get("label_dist", {}))
            print(
                f"Resumed from checkpoint: {len(processed_indices)}/{total_submissions} "
                f"submissions, {len(all_pairs)} records"
            )

    pending = [
        (idx, submission)
        for idx, submission in enumerate(submissions)
        if idx not in processed_indices
    ]
    print(f"Pending submissions: {len(pending)}")

    completed_count = len(processed_indices)

    def _consume_result(result: Dict[str, Any]) -> None:
        nonlocal completed_count, total_followups, total_llm_classify
        nonlocal total_seg, total_map_wr, total_map_fu
        idx = result["index"]
        if idx in processed_indices:
            return

        processed_indices.add(idx)
        completed_count += 1
        all_pairs.extend(result["records"])

        num_reviewers = result["num_reviewers"]
        total_seg += num_reviewers
        total_map_wr += num_reviewers
        total_map_fu += num_reviewers

        total_followups += result["total_followups"]
        total_llm_classify += result["total_llm_classify"]
        for label, count in result["label_updates"].items():
            label_dist[label] += count

        if batch_size > 0 and completed_count % batch_size == 0:
            _save_checkpoint(
                output_file=output_file,
                records=all_pairs,
                processed_indices=processed_indices,
                total_submissions=total_submissions,
                total_followups=total_followups,
                total_llm_classify=total_llm_classify,
                total_seg=total_seg,
                total_map_wr=total_map_wr,
                total_map_fu=total_map_fu,
                label_dist=label_dist,
            )
            print(f"  Checkpoint: {completed_count}/{total_submissions}")

    interrupted = False
    try:
        if max_workers <= 1:
            for idx, submission in tqdm(pending, desc="Aligning"):
                result = _process_submission_task(idx, submission, use_llm)
                _consume_result(result)
        else:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            pbar = tqdm(total=len(pending), desc="Aligning")
            pending_futures = set()
            try:
                for idx, submission in pending:
                    pending_futures.add(
                        executor.submit(_process_submission_task, idx, submission, use_llm)
                    )

                while pending_futures:
                    done, pending_futures = wait(
                        pending_futures, timeout=15, return_when=FIRST_COMPLETED
                    )
                    if not done:
                        print(
                            f"  Still running... completed {pbar.n}/{pbar.total}, "
                            f"in-flight {len(pending_futures)}"
                        )
                        continue
                    for future in done:
                        result = future.result()
                        _consume_result(result)
                        pbar.update(1)
            finally:
                pbar.close()
                executor.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
        _shutdown_event.set()
        print("\nInterrupted by user. Saving checkpoint before exit...")
        _save_checkpoint(
            output_file=output_file,
            records=all_pairs,
            processed_indices=processed_indices,
            total_submissions=total_submissions,
            total_followups=total_followups,
            total_llm_classify=total_llm_classify,
            total_seg=total_seg,
            total_map_wr=total_map_wr,
            total_map_fu=total_map_fu,
            label_dist=label_dist,
        )
        print(f"Checkpoint saved. Processed {len(processed_indices)}/{total_submissions} submissions.")
        raise SystemExit(130)

    if interrupted:
        return all_pairs

    _write_jsonl(all_pairs, output_file)
    _save_checkpoint(
        output_file=output_file,
        records=all_pairs,
        processed_indices=processed_indices,
        total_submissions=total_submissions,
        total_followups=total_followups,
        total_llm_classify=total_llm_classify,
        total_seg=total_seg,
        total_map_wr=total_map_wr,
        total_map_fu=total_map_fu,
        label_dist=label_dist,
    )

    total_llm = total_llm_classify + total_seg + total_map_wr + total_map_fu
    summary = {
        "total_submissions":            len(submissions),
        "total_output_records":         len(all_pairs),
        "total_follow_ups":             total_followups,
        "llm_calls_segmentation":       total_seg,
        "llm_calls_map_weakness_rebuttal": total_map_wr,
        "llm_calls_map_followup_weakness": total_map_fu,
        "llm_calls_follow_up_classify": total_llm_classify,
        "total_llm_calls":              total_llm,
        "label_distribution":           dict(label_dist),
        "filter_out_count":             (label_dist["debate"]
                                         + label_dist["score_update"]
                                         + label_dist["acknowledgment"])
    }
    summary_file = output_file.replace(".jsonl", ".summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Output:              {output_file}")
    print(f"Total records:       {len(all_pairs)}")
    print(f"Total follow-ups:    {total_followups}")
    print(f"\nLLM calls breakdown:")
    print(f"  Segmentation:          {total_seg}")
    print(f"  Map weakness→rebuttal: {total_map_wr}")
    print(f"  Map follow-up→weakness:{total_map_fu}  ← NEW in v4")
    print(f"  Follow-up classify:    {total_llm_classify}")
    print(f"  TOTAL:                 {total_llm}")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        tag = ("→ KEEP" if label == "same_issue"
               else "→ INSPECT" if label == "new_issue"
               else "→ FILTER OUT")
        print(f"  {label:20s}: {count:4d}  {tag}")
    print(f"\nFiltered out total:  {summary['filter_out_count']}")
    return all_pairs


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════

def run_unit_tests():
    print("\n" + "="*65)
    print("UNIT TESTS: classify_turn_relationship_rule_based")
    print("="*65)

    cases = [
        # DEBATE: clear rebuttal rejection
        ("Weighted Bellman update lacks motivation.",
         "Thank you for the response. Unfortunately, my primary concern remains unaddressed.",
         "debate", "Polite + concern remains → DEBATE"),
        ("Proof is incorrect.",
         "This does not address my concern about the proof.",
         "debate", "Explicit rejection → DEBATE"),
        ("Results not significant.",
         "I am not convinced by this explanation.",
         "debate", "Not convinced by explanation → DEBATE"),
        ("Baseline missing.",
         "I still disagree — the comparison is unfair.",
         "debate", "Explicit disagreement → DEBATE"),
        ("Calibration term misused.",
         "Thank you. I think the rebuttal addressed some but not all of my concerns. I still do not agree with the term calibration.",
         "debate", "Some but not all + still not agree → DEBATE"),
        ("Minor conceptual ambiguities.",
         "Thank you for the in-depth response. Unfortunately, a primary concern remains unaddressed: the lack of significance.",
         "debate", "Polite + primary concern remains → DEBATE"),
        # SAME_ISSUE: pushing for more (not rejecting)
        ("Missing ablation study.",
         "I still think more ablations are needed to confirm this.",
         "same_issue", "'still think X needed' → SAME_ISSUE not debate"),
        ("Missing ablation study.",
         "Thank you. Could you also include a comparison on dataset Y?",
         "same_issue", "Polite + additional request → SAME_ISSUE"),
        ("Experiments lack diversity.",
         "Thank you for adding experiments. Could you also add results on CIFAR-100?",
         "same_issue", "Thank you + more request → SAME_ISSUE"),
        ("Section 5 poorly structured.",
         "Additionally, it would help to add a diagram explaining the flow.",
         "same_issue", "Additionally → SAME_ISSUE"),
        ("Evaluation weak.",
         "I still think the evaluation is insufficient; more datasets are needed.",
         "same_issue", "'still think... needed' = action request → SAME_ISSUE"),
        # ACKNOWLEDGMENT
        ("Proof is unclear.",
         "The authors have addressed my concerns. I'm happy to accept.",
         "acknowledgment", "Full acceptance → ACKNOWLEDGMENT"),
        ("Missing experiments.",
         "Thank you, the clarification is satisfactory.",
         "acknowledgment", "Satisfied → ACKNOWLEDGMENT"),
        # NEW ISSUE
        ("Notation is unclear.",
         "Another concern is the computational complexity of Algorithm 2.",
         "new_issue", "'Another concern' on different topic → NEW_ISSUE"),
        ("Proof is wrong.",
         "On a different note, the writing in Section 3 is quite unclear.",
         "new_issue", "Different note → NEW_ISSUE"),
        # SCORE UPDATE — present/future tense (announces change)
        ("Missing baseline.",
         "I am willing to raise my score to 6 given the new experiments.",
         "score_update", "Willing to raise → SCORE_UPDATE"),
        ("Missing baseline.",
         "I will increase my score to 7.",
         "score_update", "Future tense raise → SCORE_UPDATE"),
        # ACKNOWLEDGMENT — past tense raise (already done = satisfied)
        ("Method unclear.",
         "I have raised my score to 6.",
         "acknowledgment", "Past-tense raise → ACKNOWLEDGMENT not SCORE_UPDATE"),
        ("Method unclear.",
         "In light of the revisions, I have raised my score to 6.",
         "acknowledgment", "'In light of... have raised' → ACKNOWLEDGMENT"),
    ]

    passed = 0
    for initial, follow_up, expected, desc in cases:
        label, tier, score = classify_turn_relationship_rule_based(initial, follow_up)
        ok = (label == expected)
        if ok:
            passed += 1
        print(f"{'correct' if ok else 'error'} {desc}")
        if not ok:
            print(f"   Expected : {expected}")
            print(f"   Got      : {label}  (tier={tier}, score={score:.2f})")

    # ── Test _is_quote_only ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("UNIT TESTS: _is_quote_only (Defect 2a post-processing)")
    print("─"*65)

    quote_cases = [
        # True: rebuttal IS the weakness (should become No Response)
        ("The paper lacks comparison with ODIN and Mahalanobis baselines.",
         "The paper lacks comparison with ODIN and Mahalanobis baselines.",
         True,  "Identical text → quote-only"),
        ("Lack of discussion on related work. Could the authors compare with ODIN?",
         "\"Lack of discussion on related work. Could the authors compare their method to these methods...\" (incomplete; no actual response provided)",
         True,  "Quoted weakness, no author reply → quote-only"),
        # False: real rebuttal (shares topic keywords but has new content)
        ("The paper lacks comparison with ODIN and Mahalanobis baselines.",
         "We have added comparisons with ODIN and Mahalanobis in Table 3. Our method outperforms both.",
         False, "Real rebuttal sharing topic words → not quote-only"),
        ("Missing ablation study.",
         "Thank you for this suggestion. We conducted ablation studies and results in Appendix D show each component contributes.",
         False, "Real response → not quote-only"),
    ]
    q_passed = 0
    for w, r, expected, desc in quote_cases:
        result = _is_quote_only(r, w)
        ok = result == expected
        if ok: q_passed += 1
        print(f"{'correct' if ok else 'error'} {desc}")
        if not ok:
            print(f"   Expected: {expected}, Got: {result}")

    total = len(cases) + len(quote_cases)
    total_passed = passed + q_passed
    print(f"\n{'='*65}")
    print(f"Results: {total_passed}/{total} passed")
    print("="*65)
    return total_passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align weaknesses with rebuttals and write JSONL output."
    )
    parser.add_argument(
        "--input",
        default="train_val_pool_test.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="aligned_weakness_rebuttal_pairs_finegrained.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker threads (default: 2). Use 1 for single-thread mode.",
    )
    args = parser.parse_args()

    all_passed = run_unit_tests()
    if not all_passed:
        print("\n Fix failing tests before running on full data.")
    else:
        print("\n All tests passed.")
        process_all_submissions(
            input_file  = args.input,
            output_file = args.output,
            use_llm     = True,
            batch_size  = 100,
            max_workers = args.workers,
        )
