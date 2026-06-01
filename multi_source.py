"""
multi_source.py — B4: multi-file comparison + schema-drift for DataQual AI.
DataQual AI · v3.0 Phase 4 · RAK Pvt. Ltd

Businesses don't have one CSV. This module compares the quality of several scored
sources side by side and detects schema drift (added / removed / type-changed
columns) between them — the first step toward the connectors vision. The per-file
scoring reuses the existing pipeline (detect → clean → score) in app.py; this
module only does the pure comparison/drift maths over the resulting manifests.

Pure functions, no Streamlit. Standard library + pandas only.
"""

from __future__ import annotations

import pandas as pd


def _colmap(columns) -> dict:
    """Normalise a manifest `columns` list ([{name,dtype}]) into {name: dtype}."""
    out: dict[str, str] = {}
    for c in (columns or []):
        if isinstance(c, dict) and "name" in c:
            out[str(c["name"])] = str(c.get("dtype", ""))
    return out


def schema_drift(cols_a, cols_b) -> dict:
    """Drift going from baseline A to B.

    Returns {added, removed, type_changed[{column,from,to}], n_stable, has_drift}.
    """
    a, b = _colmap(cols_a), _colmap(cols_b)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    shared = set(a) & set(b)
    type_changed = [{"column": k, "from": a[k], "to": b[k]} for k in sorted(shared) if a[k] != b[k]]
    n_stable = sum(1 for k in shared if a[k] == b[k])
    return {
        "added": added,
        "removed": removed,
        "type_changed": type_changed,
        "n_stable": n_stable,
        "has_drift": bool(added or removed or type_changed),
    }


def comparison_df(manifests) -> pd.DataFrame:
    """One row per source: quality + size + signature, for side-by-side comparison."""
    rows = []
    for m in (manifests or []):
        dims = m.get("dimensions", {}) or {}
        rows.append({
            "Source": m.get("dataset_name", "?"),
            "Rows": m.get("row_count"),
            "Overall DQ": m.get("overall_dq_score"),
            "Critical": m.get("critical_count"),
            "Completeness": dims.get("completeness"),
            "Validity": dims.get("validity"),
            "Accuracy": dims.get("accuracy"),
            "Consistency": dims.get("consistency"),
            "Uniqueness": dims.get("uniqueness"),
            "Columns": m.get("column_count"),
            "Signature": m.get("dataset_signature"),
        })
    return pd.DataFrame(rows, columns=[
        "Source", "Rows", "Overall DQ", "Critical", "Completeness", "Validity",
        "Accuracy", "Consistency", "Uniqueness", "Columns", "Signature",
    ])


def drift_report(manifests) -> list[dict]:
    """Schema drift of every source vs the first one (the baseline)."""
    out: list[dict] = []
    if not manifests:
        return out
    base = manifests[0]
    base_cols = base.get("columns")
    for m in manifests[1:]:
        cols = m.get("columns")
        entry = {"source": m.get("dataset_name"), "baseline": base.get("dataset_name")}
        if base_cols is None or cols is None:
            entry["available"] = False
        else:
            entry.update(schema_drift(base_cols, cols))
            entry["available"] = True
        out.append(entry)
    return out


def drift_summary_text(entry: dict) -> str:
    """One-line human summary of a single drift_report entry."""
    if not entry.get("available", False):
        return "schema not available (older manifest)"
    if not entry.get("has_drift"):
        return "no schema drift — columns match the baseline"
    parts = []
    if entry.get("added"):
        parts.append(f"+{len(entry['added'])} added")
    if entry.get("removed"):
        parts.append(f"-{len(entry['removed'])} removed")
    if entry.get("type_changed"):
        parts.append(f"{len(entry['type_changed'])} type-changed")
    return "drift: " + ", ".join(parts)
