#!/usr/bin/env python3
"""
build_dataset.py
================
Merges the original dataset CSV with labeled_results.json to produce a
complete, human-readable dataset with both the comment text and its label.

Safe to re-run at any time — it always rebuilds from the latest sources.
New batches collected by the poller are automatically included on the next run.

Outputs
-------
  dataset/full_dataset.json   — JSON array, one object per labeled comment
  dataset/full_dataset.csv    — Same data as CSV for easy inspection / training

Usage
-----
  python script/build_dataset.py              # merge + save
  python script/build_dataset.py --stats      # print label stats only
  python script/build_dataset.py --format csv # save as CSV only (skip JSON)
  python script/build_dataset.py --format all # save both (default)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
DATASET_DIR  = PROJECT_DIR / "dataset"

DATASET_CSV     = DATASET_DIR / "dataset_v2.csv"
RESULTS_JSON    = DATASET_DIR / "labeled_results.json"
OUT_JSON        = DATASET_DIR / "full_dataset.json"
OUT_CSV         = DATASET_DIR / "full_dataset.csv"

ENCODING  = "utf-8-sig"
TEXT_COL  = "comment_text"
LABEL_COL = "label"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("build_dataset")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv() -> pd.DataFrame:
    """Load the original dataset, preserving row index as 'csv_index'."""
    df = pd.read_csv(DATASET_CSV, encoding=ENCODING)
    df.index.name = "csv_index"
    df = df.reset_index()   # csv_index becomes a regular column (0-based row number)
    log.info("CSV loaded: %d rows, columns: %s", len(df), df.columns.tolist())
    return df


def load_results() -> pd.DataFrame:
    """Load labeled_results.json into a DataFrame keyed on original_index."""
    if not RESULTS_JSON.exists():
        log.error("labeled_results.json not found: %s", RESULTS_JSON)
        sys.exit(1)

    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    df = pd.DataFrame(results)
    log.info("labeled_results.json loaded: %d labeled records", len(df))

    # Deduplicate — keep the last occurrence if an index appears more than once
    before = len(df)
    df = df.drop_duplicates(subset="original_index", keep="last")
    if len(df) < before:
        log.warning("Removed %d duplicate original_index entries.", before - len(df))

    return df.set_index("original_index")


def merge(csv_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CSV rows with their labels.

    Sources for labels (in priority order):
      1. labeled_results.json  — batch API predictions
      2. Original CSV label    — pre-existing manual labels (16 000 rows)

    Rows with no label from either source are excluded.
    """
    # ── AI-predicted labels (from batch jobs) ─────────────────────────────────
    # We rename csv_df label to avoid conflict during merge
    csv_core = csv_df.rename(columns={LABEL_COL: "original_label"})

    predicted = (
        csv_core
        .merge(
            results_df[["label", "confidence", "low_confidence"]],
            left_on="csv_index",
            right_index=True,
            how="inner",
        )
        .rename(columns={"label": "ai_label", "confidence": "ai_confidence"})
    )
    predicted["label_source"] = "ai_batch"
    predicted["label"]        = predicted["ai_label"].astype(int)
    predicted["confidence"]   = predicted["ai_confidence"]

    ai_indices = set(predicted["csv_index"].tolist())

    # ── Pre-existing manual labels (rows NOT covered by batch predictions) ─────
    manual = csv_df[
        csv_df[LABEL_COL].notna() &
        ~csv_df["csv_index"].isin(ai_indices)
    ].copy()
    manual["label"]        = manual[LABEL_COL].astype(int)
    manual["confidence"]   = 1.0   # manual labels are certain
    manual["low_confidence"] = False
    manual["label_source"] = "manual"

    log.info(
        "Labels: %d from AI batches + %d manual pre-existing = %d total",
        len(predicted), len(manual), len(predicted) + len(manual),
    )

    # ── Combine ────────────────────────────────────────────────────────────────
    keep_cols = [TEXT_COL, "label", "confidence", "low_confidence", "label_source",
                 "nationality", "likes", "csv_index"]
    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in predicted.columns or c in manual.columns]

    merged = pd.concat(
        [
            predicted[[c for c in keep_cols if c in predicted.columns]],
            manual[[c for c in keep_cols if c in manual.columns]],
        ],
        ignore_index=True,
    )
    merged = merged.sort_values("csv_index").reset_index(drop=True)
    return merged


def print_stats(df: pd.DataFrame) -> None:
    """Print a summary of the merged dataset."""
    total   = len(df)
    pos     = (df["label"] == 1).sum()
    neg     = (df["label"] == 0).sum()
    err     = (df["label"] == -1).sum()
    low_c   = df["low_confidence"].sum() if "low_confidence" in df.columns else 0
    high_q  = (df["confidence"] >= 0.75).sum() if "confidence" in df.columns else total

    ai_cnt  = (df["label_source"] == "ai_batch").sum() if "label_source" in df.columns else 0
    man_cnt = (df["label_source"] == "manual").sum() if "label_source" in df.columns else 0

    print()
    print("=" * 52)
    print("  DATASET SUMMARY")
    print("=" * 52)
    print(f"  Total labeled rows  : {total:>8,}")
    print(f"  Positive (label=1)  : {pos:>8,}  ({100*pos/total:.1f}%)")
    print(f"  Negative (label=0)  : {neg:>8,}  ({100*neg/total:.1f}%)")
    if err:
        print(f"  Errors   (label=-1) : {err:>8,}  ({100*err/total:.1f}%)")
    print()
    print(f"  AI batch labels     : {ai_cnt:>8,}")
    print(f"  Manual labels       : {man_cnt:>8,}")
    print()
    print(f"  High-confidence >=75%: {high_q:>8,}  ({100*high_q/total:.1f}%)")
    print(f"  Low-confidence  <75%: {low_c:>8,}  ({100*low_c/total:.1f}%)")
    print("=" * 52)
    print()


def save_json(df: pd.DataFrame) -> None:
    records = df.to_dict(orient="records")
    OUT_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved %d records -> %s  (%.1f MB)", len(records), OUT_JSON.name,
             OUT_JSON.stat().st_size / 1_048_576)


def save_csv_file(df: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d rows -> %s  (%.1f MB)", len(df), OUT_CSV.name,
             OUT_CSV.stat().st_size / 1_048_576)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge dataset CSV + labeled_results.json into a full labeled dataset"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print label distribution stats and exit (no file output)",
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "all"], default="all",
        help="Output format: json, csv, or all (default: all)",
    )
    args = parser.parse_args()

    csv_df     = load_csv()
    results_df = load_results()
    merged     = merge(csv_df, results_df)

    print_stats(merged)

    if args.stats:
        log.info("--stats mode: no files written.")
        return

    if args.format in ("json", "all"):
        save_json(merged)
    if args.format in ("csv", "all"):
        save_csv_file(merged)

    log.info("Done. Re-run this script anytime after new batches are collected.")


if __name__ == "__main__":
    main()
