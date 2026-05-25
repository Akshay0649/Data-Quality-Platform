"""
cleaner.py
----------
All data-cleaning operations for the Finance Cleaning Pipeline.
Each function takes a DataFrame + config dict and returns:
    (cleaned_df, stats_dict)

Operations (8 total):
  1. clean_nulls             — Fill / flag missing values
  2. remove_duplicates        — Drop exact duplicate rows on key columns
  3. normalize_dates          — Parse & standardize date columns to YYYY-MM-DD
  4. validate_amounts         — Flag negative amounts and out-of-range values
  5. standardize_currency     — Upper-case, validate against accepted list
  6. normalize_vendor_names   — Strip whitespace, title-case, replace placeholders
  7. normalize_statuses       — Strip, upper-case, validate status values
  8. flag_outliers            — Z-score based outlier detection on amount columns
"""

import logging
import re
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_list(config: dict, key: str) -> list:
    """Split a semicolon-separated config value into a list."""
    raw = config.get(key, "")
    return [v.strip() for v in raw.split(";") if v.strip()]


# ---------------------------------------------------------------------------
# 1. Null Handling
# ---------------------------------------------------------------------------

def clean_nulls(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Fill missing values:
      - Object/string columns → null_fill_string  (default 'UNKNOWN')
      - Numeric columns       → null_fill_numeric (default 0.0)
    Adds 'had_nulls' flag column = 1 if ANY field was null for that row.
    """
    fill_str = config.get("null_fill_string", "UNKNOWN")
    fill_num = float(config.get("null_fill_numeric", 0.0))

    # Track rows that had at least one null before filling
    df = df.copy()
    df["had_nulls"] = df.isnull().any(axis=1).astype(int)

    null_counts_before = df.isnull().sum().sum()

    for col in df.columns:
        if col in ("had_nulls",):
            continue
        if df[col].dtype == object:
            df[col] = df[col].fillna(fill_str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(fill_num)

    null_counts_after = df.isnull().sum().sum()
    rows_with_nulls = int(df["had_nulls"].sum())

    stats = {
        "nulls_filled": int(null_counts_before - null_counts_after),
        "rows_with_nulls": rows_with_nulls,
    }
    logger.info(f"[1/8] Nulls filled: {stats['nulls_filled']} cells across {rows_with_nulls} rows")
    return df, stats


# ---------------------------------------------------------------------------
# 2. Duplicate Removal
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Drop duplicate rows based on the columns listed in 'duplicate_subset'.
    Keeps the first occurrence.
    """
    df = df.copy()
    subset_cols = _parse_list(config, "duplicate_subset")

    # Only use columns that actually exist in the DataFrame
    subset_cols = [c for c in subset_cols if c in df.columns]

    rows_before = len(df)

    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
    else:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

    dupes_removed = rows_before - len(df)
    stats = {"duplicates_removed": dupes_removed}
    logger.info(f"[2/8] Duplicates removed: {dupes_removed}")
    return df, stats


# ---------------------------------------------------------------------------
# 3. Date Normalization
# ---------------------------------------------------------------------------

def normalize_dates(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Parse date columns to datetime and reformat to YYYY-MM-DD strings.
    Unparseable values are replaced with NaT (then filled to 'UNKNOWN').
    Adds a '<col>_parse_error' boolean flag column for each date column.
    """
    df = df.copy()
    date_cols = _parse_list(config, "date_columns")
    date_cols = [c for c in date_cols if c in df.columns]

    total_errors = 0
    for col in date_cols:
        original = df[col].copy()
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
        error_mask = parsed.isna() & original.notna() & (original != "") & (original != "UNKNOWN")
        df[f"{col}_parse_error"] = error_mask.astype(int)
        total_errors += int(error_mask.sum())
        # Format valid dates as ISO string; keep UNKNOWN for nulls
        df[col] = parsed.dt.strftime("%Y-%m-%d").where(~parsed.isna(), other="UNKNOWN")

    stats = {"date_parse_errors": total_errors, "date_columns_processed": len(date_cols)}
    logger.info(f"[3/8] Date columns processed: {len(date_cols)} | Parse errors: {total_errors}")
    return df, stats


# ---------------------------------------------------------------------------
# 4. Amount Validation
# ---------------------------------------------------------------------------

def validate_amounts(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    For each amount column:
      - Flag negative values  → 'amount_negative' column
      - Flag values below min → 'amount_below_min' column
      - Flag values above max → 'amount_above_max' column
    Does NOT modify the original values — only flags.
    """
    df = df.copy()
    amount_cols = _parse_list(config, "amount_columns")
    amount_cols = [c for c in amount_cols if c in df.columns]

    min_amt = float(config.get("min_amount", 0.0))
    max_amt = float(config.get("max_amount", 10_000_000.0))

    neg_count = 0
    below_min = 0
    above_max = 0

    for col in amount_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_negative"]  = (df[col] < 0).astype(int)
        df[f"{col}_below_min"] = (df[col] < min_amt).astype(int)
        df[f"{col}_above_max"] = (df[col] > max_amt).astype(int)
        neg_count  += int((df[col] < 0).sum())
        below_min  += int((df[col] < min_amt).sum())
        above_max  += int((df[col] > max_amt).sum())

    stats = {
        "amount_negative_flags": neg_count,
        "amount_below_min_flags": below_min,
        "amount_above_max_flags": above_max,
    }
    logger.info(f"[4/8] Amount flags — negative: {neg_count} | below_min: {below_min} | above_max: {above_max}")
    return df, stats


# ---------------------------------------------------------------------------
# 5. Currency Standardization
# ---------------------------------------------------------------------------

def standardize_currency(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Upper-cases the 'currency' column and validates against the accepted list.
    Invalid currencies are replaced with 'UNKNOWN' and flagged.
    """
    if "currency" not in df.columns:
        return df, {"currency_invalid": 0}

    df = df.copy()
    valid = set(_parse_list(config, "valid_currencies"))

    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    invalid_mask = ~df["currency"].isin(valid)
    df["currency_invalid"] = invalid_mask.astype(int)
    df.loc[invalid_mask, "currency"] = "UNKNOWN"

    invalid_count = int(invalid_mask.sum())
    stats = {"currency_invalid": invalid_count}
    logger.info(f"[5/8] Invalid currency codes corrected: {invalid_count}")
    return df, stats


# ---------------------------------------------------------------------------
# 6. Vendor Name Normalization
# ---------------------------------------------------------------------------

_VENDOR_PLACEHOLDERS = {"n/a", "na", "unknown", "unknown vendor", "none", ""}

def normalize_vendor_names(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    - Strip leading/trailing whitespace
    - Collapse internal multiple spaces
    - Title-case
    - Replace placeholder values with null_fill_string
    """
    if "vendor_name" not in df.columns:
        return df, {"vendor_names_normalized": 0}

    df = df.copy()
    fill_str = config.get("null_fill_string", "UNKNOWN")

    original = df["vendor_name"].astype(str)
    cleaned = (
        original
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    is_placeholder = cleaned.str.lower().isin(_VENDOR_PLACEHOLDERS)
    cleaned[is_placeholder] = fill_str
    changed = int((cleaned != original).sum())
    df["vendor_name"] = cleaned

    stats = {"vendor_names_normalized": changed}
    logger.info(f"[6/8] Vendor names normalized/fixed: {changed}")
    return df, stats


# ---------------------------------------------------------------------------
# 7. Status Normalization
# ---------------------------------------------------------------------------

def normalize_statuses(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Strip whitespace, upper-case, then validate against the accepted status list.
    Invalid statuses are replaced with 'UNKNOWN' and flagged.
    """
    if "status" not in df.columns:
        return df, {"status_invalid": 0}

    df = df.copy()
    valid = set(_parse_list(config, "valid_statuses"))
    fill_str = config.get("null_fill_string", "UNKNOWN")

    df["status"] = (
        df["status"].astype(str)
        .str.strip()
        .str.upper()
        .replace({"": fill_str, "NAN": fill_str})
    )

    invalid_mask = ~df["status"].isin(valid)
    df["status_invalid"] = invalid_mask.astype(int)
    df.loc[invalid_mask, "status"] = fill_str

    invalid_count = int(invalid_mask.sum())
    stats = {"status_invalid": invalid_count}
    logger.info(f"[7/8] Invalid statuses corrected: {invalid_count}")
    return df, stats


# ---------------------------------------------------------------------------
# 8. Outlier Flagging
# ---------------------------------------------------------------------------

def flag_outliers(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    For each amount column, compute a Z-score and flag rows where
    |z| > outlier_zscore_threshold. Adds '<col>_outlier' boolean column.
    """
    df = df.copy()
    amount_cols = _parse_list(config, "amount_columns")
    amount_cols = [c for c in amount_cols if c in df.columns]
    threshold = float(config.get("outlier_zscore_threshold", 3.0))

    total_outliers = 0
    for col in amount_cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        mean = numeric_col.mean()
        std  = numeric_col.std()
        if std and std > 0:
            z_scores = (numeric_col - mean) / std
        else:
            z_scores = pd.Series(0.0, index=df.index)
        outlier_mask = z_scores.abs() > threshold
        df[f"{col}_outlier"] = outlier_mask.astype(int)
        total_outliers += int(outlier_mask.sum())

    stats = {"outliers_flagged": total_outliers}
    logger.info(f"[8/8] Outliers flagged (z > {threshold}): {total_outliers}")
    return df, stats


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_cleaners(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Run all 8 cleaning operations in sequence and return the cleaned DataFrame
    along with a combined stats dictionary.
    """
    all_stats = {}
    steps = [
        clean_nulls,
        remove_duplicates,
        normalize_dates,
        validate_amounts,
        standardize_currency,
        normalize_vendor_names,
        normalize_statuses,
        flag_outliers,
    ]
    for step in steps:
        df, stats = step(df, config)
        all_stats.update(stats)

    return df, all_stats
