"""
remediation.py — B3: Detection -> Remediation for DataQual AI.
DataQual AI · v3.0 Phase 3 · RAK Pvt. Ltd

WHY THIS MODULE EXISTS
----------------------
Scoring tells you WHAT is wrong. Remediation closes the loop: it records what the
cleaning engine already FIXED, lists what still needs a human, and produces a
clean, business-ready export plus an auditable remediation log. This is the step
where the tool starts SAVING labour ("spend time deciding, not cleaning"), not
just surfacing problems.

It reuses the existing 8-step `run_cleaning()` output (`clean_stats`) — no new
cleaning logic — and turns it into:
  * a remediation LOG (what was auto-fixed vs flagged, + the settings used),
  * SUGGESTED ACTIONS (prioritised manual next-steps from the scored frame),
  * a CLEANED business export (internal engineering flag columns stripped).

Pure functions, no Streamlit. Standard library + pandas only.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd

# Internal engineering columns the cleaning/anomaly engines add (not business data).
_INTERNAL_SUFFIXES = ("_parse_error", "_negative", "_below_min", "_above_max", "_outlier")
_INTERNAL_EXACT = {"had_nulls", "isolation_score", "iqr_outlier_count"}


def build_remediation_log(clean_stats: dict, config: dict, created_at: Optional[str] = None) -> dict:
    """Translate run_cleaning()'s clean_stats + the config used into an audit log."""
    if created_at is None:
        created_at = datetime.now().isoformat(timespec="seconds")
    cs = clean_stats or {}
    cfg = config or {}
    fs = cfg.get("null_fill_string", "UNKNOWN")
    fn = cfg.get("null_fill_numeric", 0)

    actions: list[dict] = []

    def add(action: str, count, detail: str, kind: str) -> None:
        if count and int(count) > 0:
            actions.append({"action": action, "count": int(count), "detail": detail, "type": kind})

    # Auto-fixed (the engine corrected these)
    add("Filled blank cells", cs.get("nulls_filled"),
        f"Empty text -> '{fs}', empty numbers -> {fn}", "fixed")
    add("Removed duplicate rows", cs.get("duplicates_removed"),
        "Exact / key duplicates dropped (first occurrence kept)", "fixed")
    add("Standardised category labels", cs.get("categorical_fixed"),
        "Trimmed spaces, unified case, mapped blanks/placeholders", "fixed")
    # Flagged (marked for review, value preserved)
    add("Flagged unparseable dates", cs.get("date_parse_errors"),
        "Values that are not valid dates, marked for review", "flagged")
    add("Flagged negative numbers", cs.get("negative_amount_flags"),
        "Numeric values below zero, marked for review", "flagged")
    add("Flagged statistical outliers", cs.get("outliers_flagged"),
        "Values far from the column norm (z-score), marked for review", "flagged")

    return {
        "created_at": created_at,
        "rows_input": int(cs.get("rows_input", 0)),
        "rows_output": int(cs.get("rows_output", 0)),
        "config_used": {
            "null_fill_string": fs,
            "null_fill_numeric": fn,
            "min_amount": cfg.get("min_amount"),
            "max_amount": cfg.get("max_amount"),
            "outlier_zscore_threshold": cfg.get("outlier_zscore_threshold"),
            "duplicate_subset": cfg.get("duplicate_subset", ""),
        },
        "actions": actions,
        "total_fixed": sum(a["count"] for a in actions if a["type"] == "fixed"),
        "total_flagged": sum(a["count"] for a in actions if a["type"] == "flagged"),
    }


def remediation_log_df(log: dict) -> pd.DataFrame:
    """Tabular view of the log for display / CSV."""
    rows = [{
        "Action": a["action"],
        "Type": "Auto-fixed" if a["type"] == "fixed" else "Flagged for review",
        "Count": a["count"],
        "Detail": a["detail"],
    } for a in log.get("actions", [])]
    return pd.DataFrame(rows, columns=["Action", "Type", "Count", "Detail"])


def remediation_log_json(log: dict) -> str:
    return json.dumps(log, indent=2, ensure_ascii=False)


def remediation_log_markdown(log: dict) -> str:
    """Printable Markdown remediation report."""
    lines = [
        "# Remediation Log",
        f"Generated: {log.get('created_at', '')}  ",
        f"Rows: {log.get('rows_input', 0):,} in -> {log.get('rows_output', 0):,} out  ",
        f"Auto-fixed: {log.get('total_fixed', 0):,} cells/rows · "
        f"Flagged for review: {log.get('total_flagged', 0):,}",
        "",
        "| Action | Type | Count | Detail |",
        "|---|---|---:|---|",
    ]
    for a in log.get("actions", []):
        t = "Auto-fixed" if a["type"] == "fixed" else "Flagged"
        lines.append(f"| {a['action']} | {t} | {a['count']:,} | {a['detail']} |")
    lines += ["", "## Settings used"]
    for k, v in (log.get("config_used", {}) or {}).items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def suggested_actions(scored_df: pd.DataFrame, col_mapping: dict) -> list[dict]:
    """Prioritised manual next-steps derived from the scored frame (self-contained)."""
    out: list[dict] = []

    if "dq_severity" in scored_df.columns:
        crit = int((scored_df["dq_severity"] == "CRITICAL").sum())
        if crit:
            out.append({"priority": "High", "issue": "Critical-severity rows", "count": crit,
                        "recommendation": "Review before using this data for decisions."})
        high = int((scored_df["dq_severity"] == "HIGH").sum())
        if high:
            out.append({"priority": "Medium", "issue": "High-severity rows", "count": high,
                        "recommendation": "Schedule a review; usually fixable at the source."})

    if "is_anomaly" in scored_df.columns:
        anom = int(scored_df["is_anomaly"].sum())
        if anom:
            out.append({"priority": "High", "issue": "Statistical anomalies", "count": anom,
                        "recommendation": "Verify these unusual rows — errors or genuine edge cases."})

    if "dq_grade" in scored_df.columns:
        low = int(scored_df["dq_grade"].isin(["D", "F"]).sum())
        if low:
            out.append({"priority": "Medium", "issue": "Low-grade rows (D/F)", "count": low,
                        "recommendation": "Multiple issues per row; start with the column hotspots below."})

    # Per-column hotspots from the engine's flag columns.
    hotspots: list[tuple[int, str, str]] = []
    for col in scored_df.columns:
        for suffix, label in (("_outlier", "outliers"),
                              ("_above_max", "above max"),
                              ("_below_min", "below min")):
            if col.endswith(suffix):
                cnt = int(pd.to_numeric(scored_df[col], errors="coerce").fillna(0).sum())
                if cnt:
                    hotspots.append((cnt, col[: -len(suffix)], label))
    hotspots.sort(reverse=True)
    for cnt, base, label in hotspots[:5]:
        out.append({"priority": "Low", "issue": f"'{base}' — {label}", "count": cnt,
                    "recommendation": f"Inspect '{base}'; adjust the valid range/threshold or fix source values."})
    return out


def suggested_actions_df(scored_df: pd.DataFrame, col_mapping: dict) -> pd.DataFrame:
    rows = suggested_actions(scored_df, col_mapping)
    return pd.DataFrame(
        rows, columns=["priority", "issue", "count", "recommendation"]
    ).rename(columns={"priority": "Priority", "issue": "Issue",
                      "count": "Count", "recommendation": "Recommended action"})


def cleaned_business_df(scored_df: pd.DataFrame, keep_scores: bool = True) -> pd.DataFrame:
    """Business-ready cleaned export: drop internal engineering flag columns.

    keep_scores=True keeps the dq_* score/grade/severity columns (useful context);
    set False for a pristine copy of just the user's cleaned columns.
    """
    drop = [c for c in scored_df.columns
            if c.endswith(_INTERNAL_SUFFIXES) or c in _INTERNAL_EXACT]
    out = scored_df.drop(columns=drop, errors="ignore").copy()
    if not keep_scores:
        out = out.drop(columns=[c for c in out.columns if c.startswith("dq_")], errors="ignore")
        out = out.drop(columns=[c for c in ("is_anomaly",) if c in out.columns], errors="ignore")
    return out
