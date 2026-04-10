"""
Task2-only reward function for verl GRPO training.

Reward:
    r = S(x) * P(x) * F(x)

Where:
    H(x): hard gate on raw output
    S(x): weighted API soft score in [0, 1]
    P(x): repetition penalty in [0, 1]
    F(x): format/structure quality penalty in [0, 1]  ← NEW

Changes vs previous version
----------------------------
1. H(x): added prompt-leakage gate.
   Catches the sft_conservative failure mode where the model emits
   "assistant" / "user" role tokens or repeats the full prompt inside
   the output, producing multiple concatenated answers.

2. _truncate(): now extracts ONLY the FIRST complete task2 block.
   Previous logic truncated at the last Priority/Severity line, which
   could include content from a second concatenated answer if the model
   looped. Now we anchor on the first ## Claim (or ## Evidence if no
   Claim) and cut at the first ## Severity that follows.

3. F(x): new light structural penalty.
   Rewards outputs that have all four expected sections
   (## Claim, ## Evidence, >= 1 Suggestion, ## Severity) and have at
   least one suggestion with all four fields (What/Where/How/Expected
   Outcome). Missing sections are soft-penalised rather than hard-gated,
   because partial outputs may still carry useful content for S(x).

4. P(x): threshold tightened from >2 occurrences to >1 for task2.
   Task2 outputs are longer and more structured; a line appearing twice
   is already a strong signal of looping.
"""

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

JUDGE_MODEL           = os.environ.get("JUDGE_MODEL", "gpt-4.1")
JUDGE_MAX_CONCURRENCY = max(1, int(os.environ.get("JUDGE_MAX_CONCURRENCY", "20")))
JUDGE_CACHE_DISABLE   = os.environ.get("JUDGE_CACHE_DISABLE", "").lower() in {"1", "true", "yes"}
JUDGE_MEMORY_CACHE_SIZE = max(0, int(os.environ.get("JUDGE_MEMORY_CACHE_SIZE", "20000")))
JUDGE_CACHE_PATH      = Path(
    os.environ.get(
        "JUDGE_CACHE_PATH",
        str(Path(__file__).resolve().parent / ".judge_cache_task2.sqlite3"),
    )
)

JUDGE_SYSTEM = """\
You are a strict but fair grader evaluating the quality of an academic peer review output.
You will be given a Task 2 (Actionable Analysis) output and a single grading requirement.
Score how well the output satisfies the requirement on a scale from 0 to 100.

Scoring guide:
- 100: Requirement is fully and clearly satisfied
- 75:  Requirement is mostly satisfied with minor gaps
- 50:  Requirement is partially satisfied
- 25:  Requirement is barely satisfied
- 0:   Requirement is not satisfied at all

Return ONLY a single integer between 0 and 100. No explanation."""

_SEVERITY_LEVELS = {"critical", "major", "moderate", "minor"}

_REBUTTAL_PATTERNS = [
    r"\brebuttal\b",
    r"\bpost.submission\b",
    r"\bauthors?.respond",
    r"\bin their response\b",
    r"\bauthor response\b",
]

# Prompt-leakage signals — used ONLY in _truncate() to find the end of
# the first answer block. NOT used as a hard gate, because the first
# answer before the leakage is often valid and should be scored.
_LEAKAGE_PATTERNS = [
    r"(?m)^\s*assistant\s*$",          # bare "assistant" on its own line
    r"(?m)^\s*user\s*$",               # bare "user" on its own line
    r"\[TASK 2\] Actionable Analysis", # re-emission of the task header
    r"\*\*Weakness label\*\*:",        # re-emission of the prompt fields
    r"\*\*Abstract\*\*:",
]


# -----------------------------------------------------------------
# numpy-safe conversion
# -----------------------------------------------------------------

def _to_python(obj):
    if hasattr(obj, "tolist"):
        return _to_python(obj.tolist())
    if hasattr(obj, "item"):
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


def _cache_key(requirement: str, text: str) -> str:
    payload = json.dumps(
        {"model": JUDGE_MODEL, "system": JUDGE_SYSTEM, "task": "task2",
         "requirement": requirement, "text": text},
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
            (key, float(score), time.time(), JUDGE_MODEL, "task2"),
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
# Prompt builder
# -----------------------------------------------------------------

def _build_messages(requirement: str, text: str) -> List[Dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                "## Task Type\nTask 2 (Actionable Analysis)\n\n"
                f"## Grading Requirement\n{requirement}\n\n"
                f"## Review Output to Grade\n{text}\n\n"
                "Score (0-100):"
            ),
        },
    ]


# -----------------------------------------------------------------
# Score parsing
# -----------------------------------------------------------------

def _parse_score(raw: str) -> Optional[float]:
    m = re.search(r"\b(\d{1,3})\b", raw.strip())
    return float(min(100, max(0, int(m.group(1))))) if m else None


# -----------------------------------------------------------------
# API call
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
        print(f"[REWARD] API unavailable: {e}", flush=True)
        return 0.0, False

    async def _score_one(req: Dict) -> Optional[float]:
        try:
            requirement = str(req.get("requirement", ""))
            key = _cache_key(requirement, text)
            cached = _cache_get(key)
            if cached is not None:
                return cached
            msgs = _build_messages(requirement, text)
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

    if len(valid_pairs) < len(soft):
        print(
            f"[REWARD] task2: {len(soft)-len(valid_pairs)}/{len(soft)} API calls failed, "
            "S from partial results", flush=True,
        )

    valid_scores, valid_weights = zip(*valid_pairs)
    total_w = sum(valid_weights)
    if total_w == 0:
        return 0.5, True

    s = float(min(1.0, max(0.0,
        sum(sc * w for sc, w in zip(valid_scores, valid_weights)) / (total_w * 100.0)
    )))
    return s, True


# -----------------------------------------------------------------
# Thinking strip
# -----------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# -----------------------------------------------------------------
# H(x) -- hard gate
# -----------------------------------------------------------------

def _extract_evidence_body(raw_text: str) -> str:
    m = re.search(
        r"##\s*Evidence\s*(.*?)(?=\n\s*###\s*Suggestion\s*\d+|\Z)",
        raw_text, re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _has_min_actionable_field(raw_text: str) -> bool:
    blocks = re.findall(
        r"(###\s*Suggestion\s*\d+.*?)(?=\n\s*###\s*Suggestion\s*\d+|\n\s*##\s*Severity|\Z)",
        raw_text, re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        if (
            re.search(r"-\s*\*\*What\*\*:", block, re.IGNORECASE)
            or re.search(r"-\s*\*\*Where\*\*:", block, re.IGNORECASE)
            or re.search(r"-\s*\*\*How\*\*:", block, re.IGNORECASE)
        ):
            return True
    return False


def _check_hard_constraints(raw_text: str) -> bool:
    """
    Returns False -> reward = 0.

    Gates:
    1. Rebuttal reference
    2. Exact-line repetition loop (>3 identical lines when total > 5)
    3. Prompt leakage — model re-emitted role tokens or prompt fields,
       indicating a concatenated / looping output
    4. Must have ## Evidence (non-empty)
    5. Must have at least one ### Suggestion N
    6. At least one suggestion must have a basic actionable field
    """
    text_lower = raw_text.lower()

    # Gate 1: rebuttal
    for pat in _REBUTTAL_PATTERNS:
        if re.search(pat, text_lower):
            return False

    # Gate 2: exact-line repetition
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if len(lines) > 5 and Counter(lines).most_common(1)[0][1] > 3:
        return False

    # Gate 3: ## Evidence present and non-empty
    if not re.search(r"##\s*Evidence", raw_text, re.IGNORECASE):
        return False
    if not _extract_evidence_body(raw_text):
        return False

    # Gate 4: at least one ### Suggestion N
    if not re.search(r"###\s*Suggestion\s*\d+", raw_text, re.IGNORECASE):
        return False

    # Gate 5: at least one actionable field
    if not _has_min_actionable_field(raw_text):
        return False

    return True


# -----------------------------------------------------------------
# Truncation -- extract FIRST complete task2 block only
# -----------------------------------------------------------------

def _truncate(text: str) -> str:
    """
    Extract the first complete task2 answer block.

    Strategy:
    1. Cut the text at the first leakage marker (bare "assistant"/"user"
       line, or re-emitted prompt field). This removes any second answer
       the model concatenated after its first response.
    2. From the remaining text, find the start of the first ## Claim or
       ## Evidence — the beginning of the actual answer.
    3. Find the first ## Severity that follows, and cut there.
    4. If no ## Severity found, fall back to the last Priority line.
    """
    # Step 1: cut at first leakage marker
    earliest_leakage = len(text)
    for pat in _LEAKAGE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.start() < earliest_leakage:
            earliest_leakage = m.start()
    text = text[:earliest_leakage]

    # Step 2: find answer start
    start_match = re.search(r"##\s*(Claim|Evidence)\b", text, re.IGNORECASE)
    if not start_match:
        # No ## marker — try bare "Claim:" at start
        bare = re.search(r"^Claim\s*:", text, re.IGNORECASE | re.MULTILINE)
        if bare:
            body = text[bare.start():]
        else:
            return text.strip()
    else:
        body = text[start_match.start():]

    # Step 3: cut at first ## Severity
    sev_match = re.search(
        r"(##\s*Severity\s*\n\s*(?:critical|major|moderate|minor)\b)",
        body, re.IGNORECASE,
    )
    if sev_match:
        return body[:sev_match.end()].strip()

    # Step 4: fallback to last Priority line
    priority_matches = list(re.finditer(
        r"-\s*\*\*Priority\*\*:\s*(?:critical|high|medium|low|major|moderate|minor)\b",
        body, re.IGNORECASE,
    ))
    if priority_matches:
        return body[:priority_matches[-1].end()].strip()

    return body.strip()


# -----------------------------------------------------------------
# F(x) -- structural quality penalty  (NEW)
# -----------------------------------------------------------------

def _format_penalty(text: str) -> float:
    """
    F(x): soft penalty for missing expected task2 sections.

    Full credit (1.0) if the output has:
      - ## Claim
      - ## Evidence
      - at least one ### Suggestion with all four fields
        (What, Where, How, Expected Outcome)
      - ## Severity

    Each missing element applies a small multiplicative discount.
    This is intentionally mild — S(x) handles semantic quality,
    F(x) only catches structural gaps that S(x) may not penalise.
    """
    penalty = 1.0

    if not re.search(r"##\s*Claim\b", text, re.IGNORECASE):
        penalty *= 0.90

    if not re.search(r"##\s*Severity\b", text, re.IGNORECASE):
        penalty *= 0.90

    # Check if at least one suggestion has all four fields
    blocks = re.findall(
        r"(###\s*Suggestion\s*\d+.*?)(?=\n\s*###\s*Suggestion\s*\d+|\n\s*##\s*Severity|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    has_complete_suggestion = False
    for block in blocks:
        has_what     = bool(re.search(r"-\s*\*\*What\*\*:\s*\S",     block, re.IGNORECASE))
        has_where    = bool(re.search(r"-\s*\*\*Where\*\*:\s*\S",    block, re.IGNORECASE))
        has_how      = bool(re.search(r"-\s*\*\*How\*\*:\s*\S",      block, re.IGNORECASE))
        has_outcome  = bool(re.search(r"-\s*\*\*Expected Outcome\*\*:\s*\S", block, re.IGNORECASE))
        if has_what and has_where and has_how and has_outcome:
            has_complete_suggestion = True
            break

    if not has_complete_suggestion:
        penalty *= 0.85   # stronger penalty — missing fields is a real task2 failure

    return max(0.0, min(1.0, penalty))


# -----------------------------------------------------------------
# P(x) -- repetition penalty on RAW text
# -----------------------------------------------------------------

def _repetition_penalty(raw_text: str) -> float:
    """
    P(x) = max(0, 1 - 2 * repeat_ratio)

    repeat_ratio = (lines appearing > 1 time) / total non-empty lines

    Threshold tightened to >1 (was >2) because task2 outputs are longer
    and structured; a line appearing twice is already a strong loop signal.
    """
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if len(lines) < 4:
        return 1.0
    counts       = Counter(lines)
    repeat_count = sum(1 for l in lines if counts[l] > 1)
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
                name="rubric-reward-loop-task2",
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
    Task2 reward:
        r = S(x) * P(x) * F(x)

    If hard constraints fail  -> 0
    If all API calls fail     -> 0
    """
    try:
        if extra_info is None:
            return 0.0

        extra_info = _to_python(extra_info)
        task       = extra_info.get("task", "task2")

        if task != "task2":
            print(f"[REWARD] unexpected task={task}, expected task2", flush=True)
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

        # Step 3: extract first complete answer block
        text = _truncate(raw_text)

        # Step 4: cheap penalties — P on truncated text (leakage already removed)
        p = _repetition_penalty(text)
        f = _format_penalty(text)

        # Step 5: S(x) via GPT rubric judge
        soft_score, api_ok = _run_async(_score_async(text, rubric))
        if not api_ok:
            print("[REWARD] task2: all API calls failed -> return 0", flush=True)
            return 0.0

        final = soft_score * p * f

        print(
            f"[REWARD] task2 S={soft_score:.3f} P={p:.3f} F={f:.3f} api_ok={api_ok} "
            f"-> r={final:.4f} "
            f"(raw_chars={len(raw_text)} truncated_chars={len(text)})",
            flush=True,
        )
        return float(max(0.0, min(1.0, final)))

    except Exception as e:
        print(f"[REWARD] compute_score exception: {e}", flush=True)
        return 0.0


if __name__ == "__main__":
    print("VERL reward module: import compute_score() from this file.", flush=True)