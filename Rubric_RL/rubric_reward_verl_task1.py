"""
Task1-only reward function for verl GRPO training.

Reward formula:
  r = (alpha * S(x) + (1 - alpha) * G(x)) * Q(n) * P(x) * F(x)

  H(x)  : hard gate on RAW text (pre-truncation)
  S(x)  : weighted API soft score in [0, 1]
  Q(n)  : mild length prior (NOT GT-count-matched — rubric design explicitly
           excludes count matching; see TASK1_RUBRIC_EXTRACTION_PROMPT)
  P(x)  : repetition penalty on raw text
  G(x)  : GPT-judged per-GT-claim coverage, averaged → partial credit in [0,1]
  F(x)  : light format penalty for multi-sentence or duplicate claims
  alpha : weight between rubric score and GT match (default 0.8)

  If ALL API calls fail → alpha = 0, fall back to G(x) only.
  If ground_truth missing → alpha = 1, use rubric score only.

Key design decisions
--------------------
Q(n): uses a static mild prior (n=1→1.0, n=2→0.9, n≥3→decay).
  Rationale: the rubric generation pipeline explicitly instructs GPT *not* to
  evaluate exact claim counts ("Do NOT evaluate exact count matching",
  "Pure count-based checks such as 'has >= 2 claims'"). Using GT count in Q(n)
  would contradict that philosophy and create a confounding signal.

G(x): per-GT-claim coverage scoring.
  For GT with N claims, score each GT claim independently against the full
  generated output (partial-credit prompt), then average. This gives ~0.5 when
  the model covers 1 of 2 GT claims, rather than 0 (old whole-text comparison).
  Cache key uses suffix _v2 to avoid collision with old whole-text cache entries.

H(x): near-duplicate claim gate added.
  Catches the 22-claim repetition failure mode seen in sft_conservative outputs.
  If >50% of claim pairs have Jaccard >= 0.7, reward = 0.

F(x): duplicate threshold tightened 0.80 -> 0.65 Jaccard.
"""

import argparse
import atexit
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------

JUDGE_MODEL             = os.environ.get("OPENAI_MODEL") or os.environ.get("JUDGE_MODEL", "gpt-5.1")
JUDGE_MAX_CONCURRENCY   = max(1, int(os.environ.get("JUDGE_MAX_CONCURRENCY", "20")))
JUDGE_CACHE_DISABLE     = os.environ.get("JUDGE_CACHE_DISABLE", "").lower() in {"1", "true", "yes"}
JUDGE_MEMORY_CACHE_SIZE = max(0, int(os.environ.get("JUDGE_MEMORY_CACHE_SIZE", "20000")))
JUDGE_CACHE_PATH        = Path(
    os.environ.get(
        "JUDGE_CACHE_PATH",
        str(Path(__file__).resolve().parent / ".judge_cache_task1.sqlite3"),
    )
)

ALPHA = float(os.environ.get("REWARD_ALPHA", "0.8"))

# -- Rubric soft-requirement scorer ----------------------------------
JUDGE_SYSTEM = """\
You are a strict but fair grader evaluating the quality of an academic peer review output.
You will be given a Task 1 (Weakness Claim Discovery) output and a single grading requirement.

Score how well the output satisfies the requirement using ONE integer from 1 to 5.

Scale:
- 5 = Requirement is fully and clearly satisfied
- 4 = Requirement is mostly satisfied, with only minor gaps
- 3 = Requirement is partially satisfied
- 2 = Requirement is barely satisfied
- 1 = Requirement is not satisfied

Important rules:
- Use the full 1-5 range when appropriate.
- Do not default to 3 unless the evidence is genuinely mixed.
- Return ONLY a single integer: 1, 2, 3, 4, or 5.
- Do not provide any explanation.
"""

# -- Per-GT-claim coverage scorer ------------------------------------
# Scores how well the generated output covers ONE specific GT claim.
# Uses partial-credit guide so direction-correct answers receive > 0.
GT_CLAIM_COVERAGE_SYSTEM = """\
You are a strict grader for Task 1 (Weakness Claim Discovery).
You will be given one reference weakness claim (the ground truth) and a
model-generated output that may contain one or more claims. Score how well
the generated output covers the core weakness described in the reference
claim, using ONE integer from 1 to 5.

Scale:
- 5 = The generated output contains a claim that matches the reference
      weakness almost exactly (same specific gap, same paper component
      or location if mentioned)
- 4 = The generated output contains a claim in the correct weakness
      category that captures the main point, even if less specific or
      differently worded
- 3 = The generated output partially addresses the reference weakness,
      e.g. correct area but misses a key detail, or touches the issue only
      in passing
- 2 = The generated output only weakly overlaps with the reference
      weakness, e.g. related topic but clearly a different concern
- 1 = The generated output does not address the reference weakness at
      all, or describes something entirely different

Important: judge semantic coverage, not exact wording.
A generated claim counts as covering the reference even if it is more
general, provided the core weakness is addressed. Do NOT penalize for
additional claims beyond the reference.
Return ONLY a single integer: 1, 2, 3, 4, or 5. No explanation."""


# -----------------------------------------------------------------
# numpy-safe conversion
# -----------------------------------------------------------------

def _to_python(obj):
    if hasattr(obj, 'tolist'):
        return _to_python(obj.tolist())
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


# -----------------------------------------------------------------
# OpenAI client + persistent cache
# -----------------------------------------------------------------

_client_local       = threading.local()
_cache_lock         = threading.Lock()
_memory_cache_lock  = threading.Lock()
_memory_cache: "OrderedDict[str, float]" = OrderedDict()
_cache_conn: Optional[sqlite3.Connection] = None
_loop_lock          = threading.Lock()
_loop_thread: Optional[threading.Thread] = None
_loop_start_event: Optional[threading.Event] = None
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_semaphores: Dict[int, asyncio.Semaphore] = {}
_semaphore_lock     = threading.Lock()


def _get_client() -> AsyncOpenAI:
    client = getattr(_client_local, "client", None)
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client = AsyncOpenAI(api_key=api_key)
        _client_local.client = client
    return client


def _get_cache_conn() -> Optional[sqlite3.Connection]:
    global _cache_conn
    if JUDGE_CACHE_DISABLE:
        return None
    with _cache_lock:
        if _cache_conn is None:
            JUDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                JUDGE_CACHE_PATH,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key  TEXT PRIMARY KEY,
                    score      REAL NOT NULL,
                    created_at REAL NOT NULL,
                    model      TEXT NOT NULL,
                    task       TEXT NOT NULL
                )
                """
            )
            _cache_conn = conn
        return _cache_conn


def _close_resources() -> None:
    global _cache_conn, _background_loop, _loop_thread, _loop_start_event
    with _cache_lock:
        if _cache_conn is not None:
            _cache_conn.close()
            _cache_conn = None
    with _loop_lock:
        loop   = _background_loop
        thread = _loop_thread
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=5.0)
    with _loop_lock:
        if _loop_thread is thread and (thread is None or not thread.is_alive()):
            _loop_thread = None
        if _background_loop is loop and (loop is None or not loop.is_running()):
            _background_loop = None
        _loop_start_event = None


atexit.register(_close_resources)


def _cache_key(kind: str, payload: Dict) -> str:
    payload = json.dumps(
        {"kind": kind, "model": JUDGE_MODEL, **payload},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[float]:
    with _memory_cache_lock:
        score = _memory_cache.get(key)
        if score is not None:
            _memory_cache.move_to_end(key)
            return score
    conn = _get_cache_conn()
    if conn is None:
        return None
    with _cache_lock:
        row = conn.execute(
            "SELECT score FROM judge_cache WHERE cache_key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    score = float(row[0])
    if JUDGE_MEMORY_CACHE_SIZE > 0:
        with _memory_cache_lock:
            _memory_cache[key] = score
            _memory_cache.move_to_end(key)
            while len(_memory_cache) > JUDGE_MEMORY_CACHE_SIZE:
                _memory_cache.popitem(last=False)
    return score


def _cache_set(key: str, score: float) -> None:
    if JUDGE_MEMORY_CACHE_SIZE > 0:
        with _memory_cache_lock:
            _memory_cache[key] = score
            _memory_cache.move_to_end(key)
            while len(_memory_cache) > JUDGE_MEMORY_CACHE_SIZE:
                _memory_cache.popitem(last=False)
    conn = _get_cache_conn()
    if conn is None:
        return
    with _cache_lock:
        conn.execute(
            """
            INSERT INTO judge_cache(cache_key, score, created_at, model, task)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                score=excluded.score, created_at=excluded.created_at,
                model=excluded.model, task=excluded.task
            """,
            (key, float(score), time.time(), JUDGE_MODEL, "task1"),
        )


def _get_api_semaphore() -> asyncio.Semaphore:
    loop    = asyncio.get_running_loop()
    loop_id = id(loop)
    with _semaphore_lock:
        sem = _loop_semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(JUDGE_MAX_CONCURRENCY)
            _loop_semaphores[loop_id] = sem
        return sem


# -----------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------

def _build_rubric_messages(requirement: str, text: str) -> List[Dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                "## Task Type\nTask 1 (Weakness Claim Discovery)\n\n"
                f"## Grading Requirement\n{requirement}\n\n"
                f"## Review Output to Grade\n{text}\n\n"
                "Score (1-5):"
            ),
        },
    ]


def _build_gt_claim_coverage_messages(gt_claim: str, generated_text: str) -> List[Dict]:
    """Score how well generated_text covers a single GT claim."""
    return [
        {"role": "system", "content": GT_CLAIM_COVERAGE_SYSTEM},
        {
            "role": "user",
            "content": (
                "## Task Type\nTask 1 (Weakness Claim Discovery)\n\n"
                f"## Reference Weakness Claim\n{gt_claim}\n\n"
                f"## Generated Output\n{generated_text}\n\n"
                "Score (1-5):"
            ),
        },
    ]


# -----------------------------------------------------------------
# Score parsing
# -----------------------------------------------------------------

def _parse_score(raw: str) -> Optional[float]:
    m = re.search(r"\b([1-5])\b", raw.strip())
    return float(int(m.group(1))) if m else None


def _score_1_to_reward_continuous(score_1_to_5: float) -> float:
    score_1_to_5 = max(1.0, min(5.0, float(score_1_to_5)))
    return float((score_1_to_5 - 1.0) / 4.0)


# -----------------------------------------------------------------
# API call with retry
# -----------------------------------------------------------------

async def _call_judge_api(
    client: AsyncOpenAI,
    messages: List[Dict],
    retries: int = 3,
) -> Optional[float]:
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=8,
            )
            score = _parse_score(resp.choices[0].message.content)
            if score is not None:
                return score
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"[REWARD] API judge failed after {retries} retries: {e}", flush=True)
    return None


# -----------------------------------------------------------------
# S(x) -- rubric soft-requirement scoring
# -----------------------------------------------------------------

async def _score_async(text: str, rubric: Dict) -> Tuple[float, bool]:
    """Returns (S, api_ok)."""
    soft = rubric.get("soft_requirements", [])
    if not soft:
        return 0.5, True

    try:
        client    = _get_client()
        semaphore = _get_api_semaphore()
    except Exception as e:
        print(f"[REWARD] API unavailable before rubric scoring: {e}", flush=True)
        return 0.0, False

    async def _score_one(req: Dict) -> Optional[float]:
        try:
            requirement = str(req.get("requirement", ""))
            key = _cache_key(
                "task1_soft_requirement",
                {"system": JUDGE_SYSTEM, "task": "task1",
                 "requirement": requirement, "text": text},
            )
            cached = _cache_get(key)
            if cached is not None:
                return cached
            msgs = _build_rubric_messages(requirement, text)
            async with semaphore:
                score = await _call_judge_api(client, msgs)
            if score is None:
                return None
            _cache_set(key, score)
            return score
        except Exception as e:
            print(f"[REWARD] _score_one exception: {e}", flush=True)
            return None

    raw_scores = await asyncio.gather(*[_score_one(r) for r in soft])
    weights    = [float(r.get("weight", 0.0)) for r in soft]

    valid_pairs = [(s, w) for s, w in zip(raw_scores, weights) if s is not None]
    if not valid_pairs:
        return 0.0, False

    valid_scores, valid_weights = zip(*valid_pairs)
    total_w = sum(valid_weights)
    if total_w == 0:
        return 0.5, True

    mean_score = sum(sc * w for sc, w in zip(valid_scores, valid_weights)) / total_w
    s = float(min(1.0, max(0.0, _score_1_to_reward_continuous(mean_score))))
    api_ok = len(valid_pairs) == len(soft)
    return s, api_ok


# -----------------------------------------------------------------
# GT claim extraction
# -----------------------------------------------------------------

def _extract_gt_claims(ground_truth: str) -> List[str]:
    """
    Extract individual claim texts from a GT string produced by
    items_to_task1_gt(), i.e. "Claim 1: ...\nClaim 2: ..." format.
    Falls back to treating the whole string as one claim if no markers found.
    """
    gt_clean = (ground_truth or "").strip()
    if not gt_clean or gt_clean.lower() == "none":
        return []

    parts = re.findall(
        r'(?:Claim\s+\d+\s*:\s*)(.*?)(?=\s*Claim\s+\d+\s*:|$)',
        gt_clean,
        re.IGNORECASE | re.DOTALL,
    )
    if parts:
        return [p.strip() for p in parts if p.strip()]

    return [gt_clean]


# -----------------------------------------------------------------
# G(x) -- per-GT-claim coverage scoring
# -----------------------------------------------------------------

async def _score_gt_match_async(text: str, ground_truth: str) -> Tuple[float, bool]:
    """
    For GT with N claims, score each GT claim independently against the full
    generated output, then average.

    Partial credit example:
      GT has 2 claims, model covers 1 well  -> G ~0.5  (was 0.0 before)
      GT has 1 claim,  model covers it well -> G ~0.8+

    Cache key suffix _v2 avoids reusing old whole-text comparison cache entries.
    """
    gt_claims = _extract_gt_claims(ground_truth)
    if not gt_claims:
        return 0.0, False

    try:
        client    = _get_client()
        semaphore = _get_api_semaphore()
    except Exception as e:
        print(f"[REWARD] API unavailable before GT scoring: {e}", flush=True)
        return 0.0, False

    async def _score_one_gt_claim(gt_claim: str) -> Optional[float]:
        key = _cache_key(
            "task1_gt_claim_coverage_v2",
            {"system": GT_CLAIM_COVERAGE_SYSTEM, "task": "task1",
             "gt_claim": gt_claim, "text": text},
        )
        cached = _cache_get(key)
        if cached is not None:
            return cached
        msgs = _build_gt_claim_coverage_messages(gt_claim, text)
        async with semaphore:
            score = await _call_judge_api(client, msgs)
        if score is None:
            return None
        _cache_set(key, score)
        return score

    raw_scores = await asyncio.gather(*[_score_one_gt_claim(c) for c in gt_claims])
    valid = [s for s in raw_scores if s is not None]

    if not valid:
        return 0.0, False

    mean_score = sum(valid) / len(valid)
    g     = float(min(1.0, max(0.0, _score_1_to_reward_continuous(mean_score))))
    gt_ok = len(valid) == len(gt_claims)
    return g, gt_ok


# -----------------------------------------------------------------
# Qwen3 thinking strip
# -----------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# -----------------------------------------------------------------
# H(x) -- hard gate on RAW stripped text
# -----------------------------------------------------------------

_REBUTTAL_PATTERNS = [
    r'\brebuttal\b',
    r'\bpost.submission\b',
    r'\bauthors?.respond',
    r'\bin their response\b',
    r'\bauthor response\b',
]

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _token_set(text: str) -> frozenset:
    return frozenset(re.findall(r'[a-z]+', text.lower()))

def _check_hard_constraints(raw_text: str) -> bool:
    """
    Returns False -> reward = 0.

    Gates:
    1. Rebuttal reference
    2. Exact-line repetition (>3 identical lines when len > 5)
    3. Must contain "Claim N:" or be exactly "None"
    4. Near-duplicate claim gate: if >=3 claims and >50% of pairs have
       Jaccard >= 0.7, reject. Catches the 22-claim repetition failure mode.
    """
    text_lower = raw_text.lower()

    for pat in _REBUTTAL_PATTERNS:
        if re.search(pat, text_lower):
            return False

    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    if len(lines) > 5 and Counter(lines).most_common(1)[0][1] > 3:
        return False

    clean = raw_text.strip()
    if clean.lower() == "none":
        return True
    if not re.search(r'^\s*Claim\s+\d+\s*:', clean, re.MULTILINE | re.IGNORECASE):
        return False

    # Near-duplicate claim gate
    claim_bodies = re.findall(
        r'^\s*Claim\s+\d+\s*:\s*(.+?)(?=\s*Claim\s+\d+\s*:|$)',
        clean,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    claim_bodies = [c.strip() for c in claim_bodies if c.strip()]
    if len(claim_bodies) >= 3:
        tokenized = [_token_set(c) for c in claim_bodies]
        n         = len(tokenized)
        max_pairs = n * (n - 1) / 2
        dup_pairs = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if _jaccard(tokenized[i], tokenized[j]) >= 0.7
        )
        if max_pairs > 0 and dup_pairs / max_pairs > 0.5:
            return False

    return True


# -----------------------------------------------------------------
# Truncation
# -----------------------------------------------------------------

def _truncate(text: str) -> str:
    clean = text.strip()
    if clean.lower() == "none":
        return "None"
    claims = re.findall(
        r'(Claim\s+\d+\s*:.*?)(?=\s*Claim\s+\d+\s*:|$)',
        clean,
        re.IGNORECASE | re.DOTALL,
    )
    return '\n'.join(c.strip() for c in claims) if claims else clean


# -----------------------------------------------------------------
# B(x) -- degenerate base score
# -----------------------------------------------------------------

def _base_score(text: str) -> Optional[float]:
    clean = text.strip()
    if clean.lower() == "none":
        return 0.10
    n = len(re.findall(r'^\s*Claim\s+\d+\s*:', clean, re.MULTILINE | re.IGNORECASE))
    return 0.0 if n == 0 else None


# -----------------------------------------------------------------
# Q(n) -- static mild length prior
#
# Rationale: the rubric generation pipeline explicitly instructs GPT
# *not* to check exact claim counts ("Do NOT evaluate exact count
# matching", "Pure count-based checks such as 'has >= 2 claims'").
# Using GT count here would contradict that design philosophy and add
# a confounding signal separate from S(x) and G(x).
# A static mild prior is sufficient to discourage degenerate
# over-generation without competing with content signals.
# -----------------------------------------------------------------

def _count_claims(text: str) -> int:
    return len(re.findall(r'^\s*Claim\s+\d+\s*:', text, re.MULTILINE | re.IGNORECASE))

def _claim_count_bonus(text: str) -> float:
    """
    Static prior calibrated to GT distribution (74% n=1, 19% n=2, 7% n>=3):
      n=1 -> 1.00   modal answer, no penalty
      n=2 -> 0.90   plausible, slight discount
      n>=3 -> max(0.60, 0.90 - 0.15*(n-2))
    """
    n = _count_claims(text)
    if n <= 1:
        return 1.00
    if n == 2:
        return 0.90
    return max(0.60, 0.90 - 0.15 * (n - 2))


# -----------------------------------------------------------------
# F(x) -- light format penalty
# -----------------------------------------------------------------

def _extract_claim_bodies(text: str) -> List[str]:
    claims = re.findall(
        r'^\s*Claim\s+\d+\s*:\s*(.+?)\s*$',
        text,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    return [c.strip() for c in claims if c.strip()]

def _format_penalty(text: str) -> float:
    """
    Softly discourages:
      - multi-sentence claims  (task1 claims should be one sentence)
      - near-duplicate claims  (Jaccard threshold tightened 0.80 -> 0.65)
    """
    claims = _extract_claim_bodies(text)
    if not claims:
        return 1.0

    penalty = 1.0

    multi_sentence_claims = sum(
        1 for c in claims
        if len(re.findall(r'[.!?](?=\s|$)', c)) > 1
    )
    if multi_sentence_claims:
        penalty *= max(0.75, 1.0 - 0.12 * multi_sentence_claims)

    tokenized       = [_token_set(c) for c in claims]
    duplicate_pairs = sum(
        1 for i in range(len(tokenized)) for j in range(i + 1, len(tokenized))
        if _jaccard(tokenized[i], tokenized[j]) >= 0.65
    )
    if duplicate_pairs:
        penalty *= max(0.70, 1.0 - 0.15 * duplicate_pairs)

    return max(0.0, min(1.0, penalty))


# -----------------------------------------------------------------
# P(x) -- repetition penalty on RAW text
# -----------------------------------------------------------------

def _repetition_penalty(raw_text: str) -> float:
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    if len(lines) < 4:
        return 1.0
    counts       = Counter(lines)
    repeat_count = sum(1 for l in lines if counts[l] > 2)
    return max(0.0, 1.0 - 2.0 * (repeat_count / len(lines)))


# -----------------------------------------------------------------
# asyncio background loop
# -----------------------------------------------------------------

def _background_loop_main() -> None:
    global _background_loop, _loop_thread, _loop_start_event
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with _loop_lock:
        _background_loop = loop
        start_event = _loop_start_event
    if start_event is not None:
        loop.call_soon(start_event.set)
    try:
        loop.run_forever()
    finally:
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        with _loop_lock:
            if _background_loop is loop:
                _background_loop = None
            if _loop_thread is threading.current_thread():
                _loop_thread = None
            _loop_start_event = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _loop_thread, _background_loop, _loop_start_event
    with _loop_lock:
        if _background_loop is not None and _background_loop.is_running():
            return _background_loop
        start_event = _loop_start_event
        if start_event is None or _loop_thread is None or not _loop_thread.is_alive():
            start_event       = threading.Event()
            _loop_start_event = start_event
            _loop_thread      = threading.Thread(
                target=_background_loop_main,
                name="rubric-reward-loop-task1",
                daemon=True,
            )
            _loop_thread.start()
    if not start_event.wait(timeout=5.0):
        raise RuntimeError("Timed out while starting background asyncio loop.")
    with _loop_lock:
        if _background_loop is None or not _background_loop.is_running():
            raise RuntimeError("Failed to start background asyncio loop.")
        return _background_loop


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop   = _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)


# -----------------------------------------------------------------
# verl entry point
# -----------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
) -> float:
    """
    r = (alpha * S(x) + (1 - alpha) * G(x)) * Q(n) * P(x) * F(x)

    Fallback rules:
      Rubric API failure -> alpha = 0  (G only)
      GT missing/fail   -> alpha = 1  (S only)
      Both fail         -> r = 0
    """
    try:
        if extra_info is None:
            return 0.0

        extra_info = _to_python(extra_info)
        task       = extra_info.get("task", "task1")

        if task != "task1":
            print(f"[REWARD] unexpected task={task}, expected task1", flush=True)
            return 0.0

        rubric = extra_info.get("rubric")
        if not rubric:
            return 0.0
        if isinstance(rubric, str):
            try:
                rubric = json.loads(rubric)
            except Exception:
                return 0.0
        if not isinstance(rubric, dict):
            return 0.0

        # Step 1: strip thinking
        raw_text = _strip_thinking(solution_str)

        # Step 2: H(x) hard gate
        if not _check_hard_constraints(raw_text):
            return 0.0

        # Step 3: truncate
        text = _truncate(raw_text)

        # Step 4: B(x) degenerate base score
        base = _base_score(text)
        if base is not None:
            return base

        # Step 5: load GT
        gt_str = ground_truth if isinstance(ground_truth, str) else ""
        if not gt_str:
            rm = extra_info.get("reward_model", {})
            if isinstance(rm, dict):
                gt_str = rm.get("ground_truth", "")
        has_gt = bool(gt_str and gt_str.strip().lower() not in {"", "none"})

        # Step 6: Q(n), P(x), F(x) -- cheap, no API
        q = _claim_count_bonus(text)   # static prior, NOT GT-count-matched
        p = _repetition_penalty(raw_text)
        f = _format_penalty(text)

        # Step 7: S(x) and G(x) via GPT judges
        soft_score, api_ok = _run_async(_score_async(text, rubric))
        g, gt_ok           = _run_async(_score_gt_match_async(text, gt_str))

        # Step 8: combine
        if not api_ok and not gt_ok:
            print("[REWARD] task1: both judges unavailable -> 0", flush=True)
            return 0.0

        alpha = ALPHA
        if not api_ok:
            alpha = 0.0
        elif not has_gt or not gt_ok:
            alpha = 1.0

        blended = alpha * soft_score + (1.0 - alpha) * g
        final   = blended * q * p * f

        print(
            f"[REWARD] task1 S={soft_score:.3f} G={g:.3f} alpha={alpha:.1f} "
            f"Q={q:.2f} P={p:.3f} F={f:.3f} api_ok={api_ok} gt_ok={gt_ok} "
            f"-> r={final:.4f}",
            flush=True,
        )
        return float(max(0.0, min(1.0, final)))

    except Exception as e:
        print(f"[REWARD] compute_score exception: {e}", flush=True)
        return 0.0


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate",    action="store_true")
    parser.add_argument("--rubrics",     type=str, default=None)
    parser.add_argument("--max_records", type=int, default=20)
    args = parser.parse_args()

    if args.validate:
        if not args.rubrics:
            parser.error("--validate requires --rubrics")
        print("Validation mode: load rubrics JSONL and call compute_score per record.")
    else:
        parser.print_help()
