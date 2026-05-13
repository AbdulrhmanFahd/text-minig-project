#!/usr/bin/env python3
"""
poller.py
Polls Gemini Batch API every 5 minutes.
Collects results as jobs complete, exits when everything is done.

Reads GEMINI_API_KEY from project-root/.env automatically.

Usage:
    python poller.py
"""

import json
import logging
import os
import sys
import time
import io
from pathlib import Path

from google import genai
from google.genai import types

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Default paths -- overridden by CLI flags in main()
STATE_FILE  = SCRIPT_DIR / "batch_state.json"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "labeled_results.json"
ENV_FILE    = PROJECT_DIR / ".env"

POLL_INTERVAL = 5 * 60   # 5 minutes

# ── Logging ───────────────────────────────────────────────────────────────────
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_console = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_console.setFormatter(_fmt)

_file = logging.FileHandler(SCRIPT_DIR / "poller.log", encoding="utf-8")
_file.setFormatter(_fmt)

log = logging.getLogger("poller")
log.setLevel(logging.INFO)
log.addHandler(_console)
log.addHandler(_file)


# ── .env loader (no external deps) ───────────────────────────────────────────
def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:   # don't overwrite existing env vars
            os.environ[key] = value


# ── State helpers ─────────────────────────────────────────────────────────────
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── Resilient API call with exponential backoff ───────────────────────────────
# Retries on transient server/network errors; raises immediately on auth/quota.
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES     = 5
_BACKOFF_BASE    = 10   # seconds — doubles each retry: 10, 20, 40, 80, 160


def get_job(client: genai.Client, job_name: str) -> types.BatchJob | None:
    """
    Fetch a batch job, retrying up to _MAX_RETRIES times on transient errors.
    Returns None (instead of crashing) if all retries are exhausted.
    """
    from google.genai import errors as _err
    delay = _BACKOFF_BASE
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.batches.get(name=job_name)
        except _err.ServerError as exc:
            code = getattr(exc, "status_code", None) or getattr(exc, "code", 0)
            if code not in _TRANSIENT_CODES:
                log.error("Non-retryable API error for %s: %s", job_name, exc)
                raise
            if attempt == _MAX_RETRIES:
                log.error("All %d retries exhausted for %s. Last error: %s", _MAX_RETRIES, job_name, exc)
                return None
            log.warning(
                "Transient %s for %s (attempt %d/%d) — retrying in %ds ...",
                code, job_name, attempt, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay *= 2
        except Exception as exc:
            log.error("Unexpected error fetching %s: %s", job_name, exc)
            return None


# ── Job helpers ───────────────────────────────────────────────────────────────
def job_state_str(job: types.BatchJob) -> str:
    return str(job.state).split(".")[-1] if job.state else "UNKNOWN"


def is_done(job: types.BatchJob) -> bool:
    return bool(job.done)


# ── Result parsing (same logic as batch_labeler.py) ──────────────────────────
import re as _re

_FALLBACK_ITEM = {"label": -1, "confidence": 0.0}


def _coerce_item(item: object) -> dict:
    if isinstance(item, dict):
        lbl  = item.get("label", item.get("sentiment", item.get("class", -1)))
        conf = item.get("confidence", item.get("score", item.get("prob", 0.5)))
        try:
            lbl = int(lbl); lbl = lbl if lbl in (0, 1) else -1
        except (ValueError, TypeError):
            lbl = -1
        try:
            conf = round(max(0.0, min(1.0, float(conf))), 4)
        except (ValueError, TypeError):
            conf = 0.5
        return {"label": lbl, "confidence": conf}
    if isinstance(item, (int, float)):
        lbl = int(item) if int(item) in (0, 1) else -1
        return {"label": lbl, "confidence": 0.5}
    return dict(_FALLBACK_ITEM)


def _extract_json_array(text: str) -> list | None:
    cleaned = _re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list): return v
    except json.JSONDecodeError:
        pass
    match = _re.search(r"\[.*?\]", cleaned, _re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list): return obj
        except json.JSONDecodeError:
            pass
    for suffix in ("]", "]}]", "}]"):
        try:
            obj = json.loads(cleaned + suffix)
            if isinstance(obj, list): return obj
        except json.JSONDecodeError:
            pass
    ints = _re.findall(r"\b[01]\b", cleaned)
    if ints: return [int(i) for i in ints]
    return None


def parse_response(text: str | None, expected: int) -> list[dict]:
    if not text:
        return [dict(_FALLBACK_ITEM) for _ in range(expected)]
    arr = _extract_json_array(text)
    if arr is None:
        log.warning("All parse strategies failed (expected %d). Raw: %r", expected, text[:160])
        return [dict(_FALLBACK_ITEM) for _ in range(expected)]
    items = [_coerce_item(x) for x in arr]
    if len(items) >= expected:
        return items[:expected]
    missing = expected - len(items)
    log.warning("Partial response: got %d / %d — padding %d with label=-1.", len(items), expected, missing)
    return items + [dict(_FALLBACK_ITEM) for _ in range(missing)]


def extract_results(job: types.BatchJob) -> list[dict]:
    if not job.dest or not job.dest.inlined_responses:
        log.warning("Job %s: no inlined_responses in dest.", job.name)
        return []
    records = []
    for resp in job.dest.inlined_responses:
        meta        = resp.metadata or {}
        row_indices = [int(i) for i in meta.get("row_indices", "").split(",") if i]
        expected    = int(meta.get("comment_count", len(row_indices)))
        if resp.error:
            log.warning("Request error: %s", resp.error)
            items = [dict(_FALLBACK_ITEM)] * expected
        else:
            text  = resp.response.text if resp.response else None
            items = parse_response(text, expected)
        for idx, item in zip(row_indices, items):
            records.append({
                "original_index": idx,
                "label":          item["label"],
                "confidence":     item["confidence"],
                "low_confidence": item["confidence"] < 0.75,
            })
    return records


# ── Collect one completed job ─────────────────────────────────────────────────
def collect_job(client: genai.Client, entry: dict, collected_indices: set, records: list) -> bool:
    """
    Fetch results for one SUCCEEDED job, append to records, mark collected.
    Returns True if records were added.
    """
    job = get_job(client, entry["job_name"])
    if job is None:
        log.error("Could not fetch %s for collection — will retry next tick.", entry["job_name"])
        return False
    state = job_state_str(job)

    if state != "JOB_STATE_SUCCEEDED":
        log.warning("Job %s ended with %s — skipping collection.", entry["display_name"], state)
        entry["collected"]    = True
        entry["completed_at"] = entry.get("completed_at") or _now()
        return False

    new_records = [r for r in extract_results(job) if r["original_index"] not in collected_indices]
    records.extend(new_records)
    collected_indices.update(r["original_index"] for r in new_records)

    entry["collected"]    = True
    entry["completed_at"] = entry.get("completed_at") or _now()
    log.info("  Collected %d records from %s", len(new_records), entry["display_name"])
    return True


def flush_output(records: list) -> None:
    OUTPUT_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    labels   = [r["label"] for r in records]
    low_conf = sum(1 for r in records if r.get("low_confidence"))
    log.info(
        "Saved %d records -> %s  [pos=%d neg=%d err=%d low_conf=%d]",
        len(records), OUTPUT_FILE.name,
        labels.count(1), labels.count(0), labels.count(-1), low_conf,
    )


# ── Status table printer ──────────────────────────────────────────────────────
_STATE_ICON = {
    "JOB_STATE_PENDING":    "[ PENDING  ]",
    "JOB_STATE_RUNNING":    "[ RUNNING  ]",
    "JOB_STATE_SUCCEEDED":  "[SUCCEEDED ]",
    "JOB_STATE_FAILED":     "[ FAILED   ]",
    "JOB_STATE_CANCELLED":  "[CANCELLED ]",
    "UNKNOWN":              "[ UNKNOWN  ]",
}


def print_status_table(batches: list, collected_count: int, tick: int, started_at: float) -> None:
    from datetime import datetime, timezone

    now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed   = int(time.time() - started_at)
    h, rem    = divmod(elapsed, 3600)
    m, s      = divmod(rem, 60)
    elapsed_s = f"{h:02d}h {m:02d}m {s:02d}s"

    total     = len(batches)
    done      = sum(1 for b in batches if b.get("collected"))
    pending   = total - done

    # Header
    print()
    print("=" * 72)
    print(f"  Batch Status Report  |  {now_str}  |  Elapsed: {elapsed_s}  |  Tick #{tick}")
    print("=" * 72)
    print(f"  {'BATCH':<30} {'STATE':<15} {'COMMENTS':>10}  {'SUBMITTED':>22}  {'COMPLETED':>22}")
    print("  " + "-" * 68)

    for b in batches:
        icon      = _STATE_ICON.get(b.get("state", "UNKNOWN"), "[ UNKNOWN  ]")
        sub_at    = (b.get("submitted_at") or "")[:19].replace("T", " ")
        comp_at   = (b.get("completed_at") or "-")[:19].replace("T", " ")
        collected = " [collected]" if b.get("collected") else ""
        print(
            f"  {b['display_name']:<30} {icon} {b.get('comment_count', '?'):>10,}"
            f"  {sub_at:>22}  {comp_at:>22}{collected}"
        )

    print("  " + "-" * 68)
    print(f"  Total: {total} batches | Done: {done} | Pending: {pending} | Labels saved: {collected_count:,}")
    print("=" * 72)
    print()


# ── Main poll loop ────────────────────────────────────────────────────────────
def poll(client: genai.Client) -> None:
    if not STATE_FILE.exists():
        log.error("batch_state.json not found. Run batch_labeler.py --submit first.")
        sys.exit(1)

    # Load existing output for incremental collection
    if OUTPUT_FILE.exists():
        records = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        collected_indices: set[int] = {r["original_index"] for r in records}
        log.info("Resuming: %d records already collected.", len(records))
    else:
        records, collected_indices = [], set()

    started_at = time.time()
    tick = 0

    while True:
        state   = load_state()
        batches = state.get("batches", [])
        pending = [b for b in batches if not b.get("collected")]

        if not pending:
            print_status_table(batches, len(records), tick, started_at)
            log.info("All %d batch jobs collected. Exiting.", len(batches))
            break

        changed = False

        for entry in pending:
            job = get_job(client, entry["job_name"])
            if job is None:
                log.warning("Skipping %s this tick (API unavailable).", entry["display_name"])
                continue

            new_state = job_state_str(job)
            old_state = entry.get("state", "")

            if new_state != old_state:
                log.info(
                    "STATE CHANGE  %s:  %s  ->  %s",
                    entry["display_name"], old_state or "?", new_state,
                )
                entry["state"] = new_state

            if is_done(job) and not entry.get("collected"):
                log.info("Collecting %s ...", entry["display_name"])
                collect_job(client, entry, collected_indices, records)
                changed = True

        # Always print the full table every tick so you can see current time & progress
        print_status_table(batches, len(records), tick, started_at)

        if changed:
            flush_output(records)
            save_state(state)

        if not any(not b.get("collected") for b in batches):
            log.info("All jobs finished and collected. Exiting.")
            break

        log.info("Next check in %d minutes ...", POLL_INTERVAL // 60)
        time.sleep(POLL_INTERVAL)
        tick += 1


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    global STATE_FILE

    parser = argparse.ArgumentParser(
        description="Poll Gemini Batch API until all jobs complete and collect results."
    )
    parser.add_argument(
        "--new-key",
        action="store_true",
        help="Watch batch_state_new.json (secondary key / --new-key submissions)",
    )
    parser.add_argument(
        "--state-file",
        default="",
        help="Explicit path to a batch_state.json file (overrides --new-key)",
    )
    args = parser.parse_args()

    load_env(ENV_FILE)

    # Resolve which state file to watch
    if args.state_file:
        STATE_FILE = Path(args.state_file)
    elif args.new_key:
        STATE_FILE = SCRIPT_DIR / "batch_state_new.json"
        log.info("Watching SECONDARY key state: batch_state_new.json")
    else:
        STATE_FILE = SCRIPT_DIR / "batch_state.json"
        log.info("Watching PRIMARY key state: batch_state.json")

    api_key_env = "GEMINI_API_KEY_NEW" if args.new_key else "GEMINI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        sys.exit(f"ERROR: {api_key_env} not set. Add it to .env or export it.")

    client = genai.Client(api_key=api_key)
    log.info("Poller started. Watching %d-minute intervals.", POLL_INTERVAL // 60)
    log.info("State file : %s", STATE_FILE)
    log.info("Output file: %s", OUTPUT_FILE)

    try:
        poll(client)
    except KeyboardInterrupt:
        log.info("Interrupted by user. Progress saved to %s.", STATE_FILE.name)
        sys.exit(0)


if __name__ == "__main__":
    main()
