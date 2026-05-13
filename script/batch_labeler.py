#!/usr/bin/env python3
"""
batch_labeler.py
Labels unlabeled Arabic YouTube comments in dataset_v2.csv via the Gemini Batch API.

Each comment gets:
  label      : 0 (Negative) or 1 (Positive)
  confidence : 0.0 - 1.0  (filter on >= 0.75 for highest-quality training data)

Usage
-----
  python batch_labeler.py --submit              # submit all rows using primary key
  python batch_labeler.py --submit --new-key    # submit all rows using secondary key
  python batch_labeler.py --status              # status for primary key batches
  python batch_labeler.py --status  --new-key   # status for secondary key batches
  python batch_labeler.py --collect             # collect primary results
  python batch_labeler.py --collect --new-key   # collect secondary results
  python batch_labeler.py --sample              # send 15-comment smoke-test

Dual-key workflow
-----------------
  Primary key  -> batch_state.json      (GEMINI_API_KEY)
  Secondary key-> batch_state_new.json  (GEMINI_API_KEY_NEW)
  Both write to labeled_results.json (deduplication by original_index)

Environment (.env)
------------------
  GEMINI_API_KEY      required -- primary Google AI Studio API key
  GEMINI_API_KEY_NEW  optional -- secondary key for parallel submission
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

DATASET_PATH = DATASET_DIR / "dataset_v2.csv"
OUTPUT_FILE  = DATASET_DIR / "labeled_results.json"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL               = "gemini-2.5-flash"   # best cost/quality for Arabic SA (batch-compatible)
ENCODING            = "utf-8-sig"
TEXT_COL            = "comment_text"
LABEL_COL           = "label"

# 5 comments per request: with thinking disabled, each request needs ~150 output tokens
# 2000 requests per batch: 10,000 comments per batch job
COMMENTS_PER_REQ    = 5       # REDUCED from 15 -- avoids token budget issues
MAX_REQS_PER_BATCH  = 2_000   # Gemini Batch API supports up to 200,000; 2000 is safe
SAMPLE_COMMENTS     = 15      # rows sent in --sample (3 requests of 5)
POLL_INTERVAL       = 60      # seconds between status checks
BATCH_SUBMIT_DELAY  = 70      # seconds between batch submissions (~5 jobs/min limit)

# Filter threshold: records with confidence below this are kept but flagged
CONFIDENCE_THRESHOLD = 0.75

# ── Runtime state (set by CLI args in main()) ─────────────────────────────────
_STATE_FILE    = SCRIPT_DIR / "batch_state.json"   # overridden by --new-key
_API_KEY_ENV   = "GEMINI_API_KEY"                  # overridden by --new-key
_BATCH_PREFIX  = "arabic-sa"                       # overridden by --name-prefix

# ── Few-shot examples from the known labeled sample ───────────────────────────
FEW_SHOT = [
    ("هذا الفيديو رائع ومفيد جداً شكراً لك", 1),
    ("محتوى سيء ومضيعة للوقت لن أشاهد مجدداً", 0),
    ("الشرح واضح وبسيط أفضل قناة", 1),
    ("لا يوجد أي فائدة من هذا الكلام", 0),
]

# ── Logging: UTF-8 everywhere, no Unicode crashes on Windows ──────────────────
def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler: wrap stdout in a UTF-8 stream to avoid cp1256 errors
    stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = logging.StreamHandler(stdout_utf8)
    console.setFormatter(fmt)

    file_h = logging.FileHandler(SCRIPT_DIR / "batch_labeler.log", encoding="utf-8")
    file_h.setFormatter(fmt)

    logger = logging.getLogger("batch_labeler")
    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(file_h)
    return logger


log = _setup_logging()


# =============================================================================
# Data helpers
# =============================================================================

def load_unlabeled() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, encoding=ENCODING)
    unlabeled = df[df[LABEL_COL].isna()].copy()
    log.info(
        "Dataset: %d total | %d labeled | %d unlabeled",
        len(df), (~df[LABEL_COL].isna()).sum(), len(unlabeled),
    )
    return unlabeled.reset_index()   # preserves original 'index' column (row in CSV)


def chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# =============================================================================
# Prompt builders
# =============================================================================

_EXAMPLES = "\n".join(
    f"  Comment: {t}\n  Label: {l}" for t, l in FEW_SHOT
)

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Arabic YouTube comments.\n"
    "Classify each comment as:\n"
    "  0 = Negative (criticism, complaints, insults, negative opinions)\n"
    "  1 = Positive (praise, appreciation, encouragement, positive opinions)\n\n"
    "Examples of correctly labeled comments:\n"
    + _EXAMPLES
)


def build_user_prompt(comments: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {c.strip()}" for i, c in enumerate(comments))
    return (
        f"Label these {len(comments)} comments.\n"
        "For each comment return an object with:\n"
        '  "label"      : 0 or 1\n'
        '  "confidence" : float 0.0-1.0 (how certain you are)\n'
        "Return ONLY a JSON array of these objects in the same order.\n\n"
        + numbered
    )


# =============================================================================
# InlinedRequest builder
# =============================================================================

_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "label":      {"type": "INTEGER"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["label", "confidence"],
    },
}

_GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.0,
    # thinking_budget=0: CRITICAL FIX for Gemini 2.5 Flash.
    # Without this, the model uses ~1963 "thinking tokens" internally which
    # consume almost the entire max_output_tokens budget, causing MAX_TOKENS
    # truncation. Disabling thinking gives full token budget to JSON output.
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    max_output_tokens=256,       # 5 comments x ~25 tokens each = ~125 tokens needed
    response_mime_type="application/json",
    response_schema=_RESPONSE_SCHEMA,
)


def make_request(
    row_indices: list[int],
    comments: list[str],
    batch_tag: str,
    req_idx: int,
) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=build_user_prompt(comments),
        config=_GEN_CONFIG,
        metadata={
            # All values must be strings (Gemini API requirement)
            "batch_tag":     batch_tag,
            "req_idx":       str(req_idx),
            "row_indices":   ",".join(str(i) for i in row_indices),
            "comment_count": str(len(comments)),
        },
    )


# =============================================================================
# State management
# =============================================================================

def load_state() -> dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    return {
        "created_at":      _now(),
        "submitted_count": 0,      # total unlabeled comments already submitted
        "batches":         [],
        "sample":          None,
    }


def save_state(state: dict) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Gemini API wrappers
# =============================================================================

def get_client() -> genai.Client:
    api_key = os.environ.get(_API_KEY_ENV, "")
    if not api_key:
        sys.exit(f"ERROR: {_API_KEY_ENV} environment variable is not set.")
    return genai.Client(api_key=api_key)


def submit_batch(
    client: genai.Client,
    requests: list[types.InlinedRequest],
    display_name: str,
    max_retries: int = 8,
    backoff_base: int = 90,
) -> types.BatchJob:
    from google.genai import errors as _err
    delay = backoff_base
    for attempt in range(1, max_retries + 1):
        try:
            job = client.batches.create(
                model=MODEL,
                src=requests,
                config=types.CreateBatchJobConfig(display_name=display_name),
            )
            log.info(
                "Submitted batch '%s' -> job_name=%s  requests=%d",
                display_name, job.name, len(requests),
            )
            return job
        except (_err.ClientError, _err.ServerError) as exc:
            code = getattr(exc, "status_code", None) or getattr(exc, "code", 0)
            if code == 429 or code >= 500:
                if attempt == max_retries:
                    raise
                log.warning(
                    "HTTP %d on submit '%s' (attempt %d/%d) — retrying in %ds ...",
                    code, display_name, attempt, max_retries, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 600)  # cap at 10 min
            else:
                raise


def get_status(client: genai.Client, job_name: str) -> types.BatchJob:
    return client.batches.get(name=job_name)


def job_state_str(job: types.BatchJob) -> str:
    return str(job.state).split(".")[-1] if job.state else "UNKNOWN"


def is_done(job: types.BatchJob) -> bool:
    return bool(job.done)


# =============================================================================
# Result parsing — multi-strategy, recovers partial results
# =============================================================================

import re as _re

_FALLBACK_ITEM = {"label": -1, "confidence": 0.0}


def _coerce_item(item: object) -> dict:
    """Convert one raw parsed item into {label, confidence}, tolerating format variation."""
    # Strategy A: correct schema — {"label": 0, "confidence": 0.9}
    if isinstance(item, dict):
        # Try canonical keys first, then common aliases
        lbl = item.get("label", item.get("sentiment", item.get("class", -1)))
        conf = item.get("confidence", item.get("score", item.get("prob", 0.5)))
        try:
            lbl = int(lbl)
            lbl = lbl if lbl in (0, 1) else -1
        except (ValueError, TypeError):
            lbl = -1
        try:
            conf = round(float(conf), 4)
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 0.5
        return {"label": lbl, "confidence": conf}

    # Strategy B: plain integer [0, 1, 0, ...] — label only, confidence unknown
    if isinstance(item, (int, float)):
        lbl = int(item) if int(item) in (0, 1) else -1
        return {"label": lbl, "confidence": 0.5}  # 0.5 = "confidence unknown"

    return dict(_FALLBACK_ITEM)


def _extract_json_array(text: str) -> list | None:
    """
    Try multiple strategies to pull a JSON array out of a potentially messy response.
    Returns the parsed list or None if all strategies fail.
    """
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = _re.sub(r"```(?:json)?\s*", "", text).strip()

    # Strategy 1: direct parse on the stripped text
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
        # Gemini sometimes wraps: {"results": [...]}
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the first [...] block in the text (handles leading/trailing prose)
    match = _re.search(r"\[.*?\]", cleaned, _re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 3: truncated JSON — try adding closing brackets and re-parse
    # This recovers partial arrays cut off at max_output_tokens
    for suffix in ("]", "]}]", "}]"):
        try:
            obj = json.loads(cleaned + suffix)
            if isinstance(obj, list):
                log.debug("Recovered truncated JSON with suffix %r", suffix)
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 4: extract integers via regex (last resort for [0,1,0,...] style)
    ints = _re.findall(r"\b[01]\b", cleaned)
    if ints:
        return [int(i) for i in ints]

    return None


def parse_response(text: str | None, expected: int) -> list[dict]:
    """
    Parse Gemini's response into a list of {label, confidence} dicts.

    Attempts multiple recovery strategies before giving up:
      1. Correct schema  [{label, confidence}, ...]
      2. Alias keys      [{sentiment, score}, ...]
      3. Plain integers  [0, 1, 0, ...]
      4. Markdown fences stripped
      5. JSON inside prose text
      6. Truncated JSON repaired
      7. Regex integer extraction

    Partial results (fewer items than expected) are padded with -1 rather than
    discarding the valid items already found.
    """
    if not text:
        log.warning("Empty response for request expecting %d items.", expected)
        return [dict(_FALLBACK_ITEM) for _ in range(expected)]

    arr = _extract_json_array(text)

    if arr is None:
        log.warning(
            "All parse strategies failed (expected %d items). Raw: %r",
            expected, text[:200],
        )
        return [dict(_FALLBACK_ITEM) for _ in range(expected)]

    items = [_coerce_item(x) for x in arr]

    if len(items) == expected:
        return items                          # perfect match

    if len(items) > expected:
        log.debug("Response had %d items, expected %d — truncating.", len(items), expected)
        return items[:expected]

    # Partial recovery: use what we have, pad the rest with -1
    missing = expected - len(items)
    log.warning(
        "Partial response: got %d items, expected %d — padding %d with label=-1.",
        len(items), expected, missing,
    )
    return items + [dict(_FALLBACK_ITEM) for _ in range(missing)]


def extract_results(job: types.BatchJob) -> list[dict]:
    """
    Extract labeled records from a completed BatchJob.
    Returns list of {original_index, label, confidence, low_confidence}.
    """
    if not job.dest or not job.dest.inlined_responses:
        log.warning("Job %s has no inlined_responses.", job.name)
        return []

    records = []
    for resp in job.dest.inlined_responses:
        meta        = resp.metadata or {}
        row_indices = [int(i) for i in meta.get("row_indices", "").split(",") if i]
        expected    = int(meta.get("comment_count", len(row_indices)))

        if resp.error:
            log.warning("Request error in %s: %s", job.name, resp.error)
            items = [{"label": -1, "confidence": 0.0}] * expected
        else:
            text  = resp.response.text if resp.response else None
            items = parse_response(text, expected)

        for idx, item in zip(row_indices, items):
            records.append({
                "original_index":  idx,
                "label":           item["label"],
                "confidence":      item["confidence"],
                "low_confidence":  item["confidence"] < CONFIDENCE_THRESHOLD,
            })

    log.info("Extracted %d label records from job %s", len(records), job.name)
    return records


# =============================================================================
# Pipeline stages
# =============================================================================

def run_sample(client: genai.Client) -> None:
    """Submit one small batch job (15 comments), wait, and print results."""
    log.info("=== SAMPLE MODE: %d comments ===", SAMPLE_COMMENTS)
    unlabeled   = load_unlabeled()
    sample      = unlabeled.head(SAMPLE_COMMENTS)
    comments    = sample[TEXT_COL].tolist()
    row_indices = sample["index"].tolist()

    reqs = [
        make_request(
            row_indices[i : i + COMMENTS_PER_REQ],
            comments[i : i + COMMENTS_PER_REQ],
            "sample",
            ri,
        )
        for ri, i in enumerate(range(0, len(comments), COMMENTS_PER_REQ))
    ]

    sample_name = f"{_BATCH_PREFIX}-sample"
    job   = submit_batch(client, reqs, display_name=sample_name)
    state = load_state()
    state["sample"] = {
        "job_name":      job.name,
        "display_name":  sample_name,
        "submitted_at":  _now(),
        "request_count": len(reqs),
        "comment_count": len(comments),
        "state":         job_state_str(job),
    }
    save_state(state)
    log.info("Sample job_name saved: %s", job.name)

    log.info("Polling for sample completion (interval: %ds) ...", POLL_INTERVAL)
    while not is_done(job):
        time.sleep(POLL_INTERVAL)
        job = get_status(client, job.name)
        log.info("  sample state: %s", job_state_str(job))

    final_state = job_state_str(job)
    # Reload from disk before saving so we never clobber batches added by --submit
    state = load_state()
    state["sample"]["state"]        = final_state
    state["sample"]["completed_at"] = _now()
    save_state(state)

    if final_state != "JOB_STATE_SUCCEEDED":
        log.error("Sample batch ended with state: %s", final_state)
        return

    records = extract_results(job)
    log.info("--- Sample results (%d records) ---", len(records))
    for rec in records:
        if rec["label"] == 1:
            sentiment = "Positive"
        elif rec["label"] == 0:
            sentiment = "Negative"
        else:
            sentiment = "ERROR"
        orig = unlabeled[unlabeled["index"] == rec["original_index"]][TEXT_COL].values
        preview = orig[0][:70] if len(orig) else "?"
        conf_flag = " [low-conf]" if rec["low_confidence"] else ""
        log.info("  [%s %.0f%%%s]  %s", sentiment, rec["confidence"] * 100, conf_flag, preview)


def run_submit(client: genai.Client) -> None:
    """Submit all not-yet-submitted unlabeled rows as batch jobs."""
    unlabeled = load_unlabeled()
    state     = load_state()

    already_submitted = state.get("submitted_count", 0)
    pending = unlabeled.iloc[already_submitted:]

    if pending.empty:
        log.info("All %d unlabeled rows already submitted.", already_submitted)
        return

    log.info(
        "Submitting %d remaining unlabeled rows (skipping first %d already in flight) ...",
        len(pending), already_submitted,
    )

    comments    = pending[TEXT_COL].tolist()
    row_indices = pending["index"].tolist()

    all_requests = [
        make_request(
            row_indices[i : i + COMMENTS_PER_REQ],
            comments[i : i + COMMENTS_PER_REQ],
            "full",
            global_req_idx,
        )
        for global_req_idx, i in enumerate(range(0, len(comments), COMMENTS_PER_REQ))
    ]
    log.info("Total InlinedRequests to send: %d", len(all_requests))

    today         = datetime.now(timezone.utc).strftime("%Y%m%d")
    batch_num_offset = len(state["batches"]) + 1
    total_batches    = -(-len(all_requests) // MAX_REQS_PER_BATCH)  # ceil division
    log.info("Will submit %d batch jobs (%d requests each, %ds apart).",
             total_batches, MAX_REQS_PER_BATCH, BATCH_SUBMIT_DELAY)

    for batch_offset, req_chunk in enumerate(chunks(all_requests, MAX_REQS_PER_BATCH)):
        batch_num    = batch_num_offset + batch_offset
        comment_cnt  = sum(int(r.metadata["comment_count"]) for r in req_chunk)
        # Visible display name in Google AI Studio:
        # format: <prefix>-<YYYYMMDD>-b<NNN>   e.g. arabic-sa-20260513-b001
        display_name = f"{_BATCH_PREFIX}-{today}-b{batch_num:03d}"

        job = submit_batch(client, req_chunk, display_name=display_name)

        state["batches"].append({
            "job_name":      job.name,
            "display_name":  display_name,
            "batch_num":     batch_num,
            "submitted_at":  _now(),
            "request_count": len(req_chunk),
            "comment_count": comment_cnt,
            "state":         job_state_str(job),
            "completed_at":  None,
            "collected":     False,
        })
        state["submitted_count"] = state.get("submitted_count", 0) + comment_cnt
        save_state(state)  # persist after every batch so job_names are never lost
        log.info(
            "  [%d/%d] Submitted %s -> %s  (%d reqs, %d comments)",
            batch_offset + 1, total_batches,
            display_name, job.name, len(req_chunk), comment_cnt,
        )
        if batch_offset < total_batches - 1:   # no delay after last batch
            log.info("  Waiting %ds before next submission ...", BATCH_SUBMIT_DELAY)
            time.sleep(BATCH_SUBMIT_DELAY)

    log.info("All batches submitted. Run --status to monitor progress.")


def run_status(client: genai.Client) -> None:
    """Print live status of every tracked batch job."""
    state = load_state()
    smpl  = state.get("sample")

    header = (
        f"{'JOB NAME':<48} {'DISPLAY NAME':<38} "
        f"{'STATE':<25} {'REQS':>6} {'COMMENTS':>10}"
    )
    print(header)
    print("-" * len(header))

    entries = []
    if smpl:
        entries.append({"_sample": True, **smpl})
    for b in state["batches"]:
        entries.append({"_sample": False, **b})

    for entry in entries:
        job = get_status(client, entry["job_name"])
        s   = job_state_str(job)

        tag = "[SAMPLE] " if entry["_sample"] else ""
        print(
            f"{entry['job_name']:<48} "
            f"{tag}{entry['display_name']:<38} "
            f"{s:<25} "
            f"{entry.get('request_count', '?'):>6} "
            f"{entry.get('comment_count', '?'):>10}"
        )

        if s != entry.get("state"):
            entry["state"] = s
            if is_done(job) and not entry.get("completed_at"):
                entry["completed_at"] = _now()

    # Persist state changes (strip internal _sample key)
    state["batches"] = [
        {k: v for k, v in e.items() if k != "_sample"}
        for e in entries if not e["_sample"]
    ]
    if smpl and entries:
        sample_entry = next((e for e in entries if e["_sample"]), None)
        if sample_entry:
            state["sample"] = {k: v for k, v in sample_entry.items() if k != "_sample"}
    save_state(state)


def run_collect(client: genai.Client) -> None:
    """Download results from all completed batches, save to labeled_results.json."""
    state   = load_state()
    records = []

    if OUTPUT_FILE.exists():
        records = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        collected_indices = {r["original_index"] for r in records}
        log.info("Loaded %d existing records from %s", len(records), OUTPUT_FILE.name)
    else:
        collected_indices = set()

    changed = False
    for entry in state["batches"]:
        if entry.get("collected"):
            continue

        job = get_status(client, entry["job_name"])
        s   = job_state_str(job)
        entry["state"] = s

        if not is_done(job):
            log.info("Batch %s is still %s -- skipping.", entry["display_name"], s)
            continue

        if s != "JOB_STATE_SUCCEEDED":
            log.warning("Batch %s ended with %s -- marking done.", entry["display_name"], s)
            entry["collected"]    = True
            entry["completed_at"] = entry.get("completed_at") or _now()
            changed = True
            continue

        new_records = [
            r for r in extract_results(job)
            if r["original_index"] not in collected_indices
        ]
        records.extend(new_records)
        collected_indices.update(r["original_index"] for r in new_records)
        entry["collected"]    = True
        entry["completed_at"] = entry.get("completed_at") or _now()
        changed = True
        log.info("Collected %d new records from %s", len(new_records), entry["display_name"])

    if changed:
        save_state(state)
        OUTPUT_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        labels   = [r["label"] for r in records]
        low_conf = sum(1 for r in records if r.get("low_confidence"))
        log.info(
            "Saved %d records -> %s  [pos=%d neg=%d err=%d low_conf=%d]",
            len(records), OUTPUT_FILE,
            labels.count(1), labels.count(0), labels.count(-1), low_conf,
        )
    else:
        log.info("No new completed batches to collect.")


def run_pipeline(client: genai.Client) -> None:
    """Full pipeline: submit -> poll -> collect."""
    run_submit(client)

    log.info("Polling until all batches complete (interval: %ds) ...", POLL_INTERVAL)
    while True:
        state   = load_state()
        pending = [b for b in state["batches"] if not b.get("collected")]
        if not pending:
            log.info("All batches collected.")
            break

        all_done = True
        for entry in pending:
            job = get_status(client, entry["job_name"])
            s   = job_state_str(job)
            entry["state"] = s
            if not is_done(job):
                all_done = False
                log.info("  %s -> %s", entry["display_name"], s)
        save_state(state)

        if all_done:
            log.info("All jobs finished. Collecting results ...")
            run_collect(client)
            break

        time.sleep(POLL_INTERVAL)


# =============================================================================
# CLI
# =============================================================================

def _load_dot_env() -> None:
    """Load .env from project root (same logic as poller.py)."""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    global _STATE_FILE, _API_KEY_ENV, _BATCH_PREFIX

    parser = argparse.ArgumentParser(
        description="Label Arabic YouTube comments via Gemini Batch API"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--sample",  action="store_true", help="send 15-comment smoke test")
    grp.add_argument("--submit",  action="store_true", help="submit all unlabeled rows")
    grp.add_argument("--status",  action="store_true", help="print live batch status")
    grp.add_argument("--collect", action="store_true", help="collect completed results")
    parser.add_argument(
        "--new-key",
        action="store_true",
        help="use GEMINI_API_KEY_NEW (secondary account) with batch_state_new.json",
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="prefix for batch display names in AI Studio (default: arabic-sa or arabic-sa-new)",
    )
    args = parser.parse_args()

    _load_dot_env()

    # ── Apply runtime overrides based on flags ────────────────────────────────
    if args.new_key:
        _STATE_FILE   = SCRIPT_DIR / "batch_state_new.json"
        _API_KEY_ENV  = "GEMINI_API_KEY_NEW"
        _BATCH_PREFIX = args.name_prefix or "arabic-sa-new"
        log.info("Mode: SECONDARY KEY  |  state: batch_state_new.json  |  prefix: %s", _BATCH_PREFIX)
    else:
        _STATE_FILE   = SCRIPT_DIR / "batch_state.json"
        _API_KEY_ENV  = "GEMINI_API_KEY"
        _BATCH_PREFIX = args.name_prefix or "arabic-sa"
        log.info("Mode: PRIMARY KEY    |  state: batch_state.json      |  prefix: %s", _BATCH_PREFIX)

    client = get_client()

    if args.sample:
        run_sample(client)
    elif args.submit:
        run_submit(client)
    elif args.status:
        run_status(client)
    elif args.collect:
        run_collect(client)
    else:
        run_pipeline(client)


if __name__ == "__main__":
    main()
