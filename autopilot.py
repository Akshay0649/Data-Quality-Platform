"""
autopilot.py — B5+: the DataQual Autopilot decision engine.
DataQual AI · v3.0 Phase 5 · RAK Pvt. Ltd

WHY THIS MODULE EXISTS
----------------------
Scoring tells you WHAT is wrong (scorer). Remediation tells you what got FIXED
(remediation). Persistence remembers it (run_manifest). The missing piece is the
one a busy stakeholder actually wants: a DECISION. "Given this data and our
quality bar, can I use it — yes, with cleanup, or not at all?"

Autopilot closes that loop. Feed it a scored frame + the run manifest + a
"quality policy" (the SLA bar) and it returns a single, explainable verdict:

  • APPROVE     — meets the bar; publish the cleaned file, no human needed.
  • REMEDIATE   — recoverable; auto-fixes applied, here's the prioritised work.
  • QUARANTINE  — do not use; escalate the critical rows and fix at source.

…together with a confidence (how clear-cut the call is), the exact policy gates
that drove it, a prioritised action plan (reused from `remediation`), and the
recommended next steps. It is the autonomous orchestrator on top of the existing
engines — it adds NO new scoring/cleaning logic, only a transparent decision
rule over numbers the pipeline already produced.

Design DNA (matches the rest of the platform): pure functions, no Streamlit,
standard library + pandas only. Deterministic and explainable — every verdict
can be traced to the gate(s) that caused it. The decision rule lives here so it
can be unit-tested offline and, later, reused by the B6 scheduler/alerting layer
without dragging in the UI.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd

# Optional siblings — both are pure (stdlib + pandas). Imported defensively so a
# missing sibling degrades gracefully rather than breaking the whole app import.
try:  # pragma: no cover - exercised indirectly
    import run_manifest as _run_manifest
except Exception:  # pragma: no cover
    _run_manifest = None
try:  # pragma: no cover
    import remediation as _remediation
except Exception:  # pragma: no cover
    _remediation = None


# ──────────────────────────────────────────────────────────────────────────────
# Policy — the quality bar Autopilot decides against.
# ──────────────────────────────────────────────────────────────────────────────
# Per-dimension minimums mirror the Scorecard defaults (accuracy/uniqueness are
# stricter because, in finance, a wrong number costs more than a missing one and a
# double-paid invoice is expensive). The two "max %" gates cap how much of the
# dataset may be unusable before Autopilot refuses to wave it through.
DEFAULT_POLICY = {
    "min_overall":       80,
    "min_completeness":  80,
    "min_validity":      80,
    "min_accuracy":      85,
    "min_consistency":   80,
    "min_uniqueness":    90,
    "max_critical_pct":  1.0,   # % of rows allowed at CRITICAL severity
    "max_anomaly_pct":   5.0,   # % of rows allowed to be statistical anomalies
}

# How far BELOW a minimum a hard gate must fall before it blocks (QUARANTINE)
# rather than merely flags (REMEDIATE). A small miss is recoverable; a large one
# means the data is not trustworthy enough to publish even after cleanup.
HARD_MARGIN = 15

_DIM_KEYS = ["overall", "completeness", "validity", "accuracy", "consistency", "uniqueness"]

# Gates that, when failed badly (beyond HARD_MARGIN), force QUARANTINE.
_HARD_DIMENSIONS = {"overall", "accuracy", "consistency"}

_LABELS = {
    "overall":       "Overall quality",
    "completeness":  "Completeness (missing data)",
    "validity":      "Validity (format)",
    "accuracy":      "Accuracy (values)",
    "consistency":   "Consistency",
    "uniqueness":    "Uniqueness (duplicates)",
    "critical_pct":  "Critical rows %",
    "anomaly_pct":   "Anomaly rows %",
}

_WEIGHTS = {  # dimension weights (mirror run_manifest / scorer) — for display + tie-breaks
    "completeness": 20, "validity": 25, "accuracy": 35,
    "consistency": 15, "uniqueness": 5,
}

_VERDICT_META = {
    "APPROVE":    {"label": "APPROVE",    "icon": "✅", "color": "#1D9E75", "bg": "#f0fdf4"},
    "REMEDIATE":  {"label": "REMEDIATE",  "icon": "🔧", "color": "#BA7517", "bg": "#fffbeb"},
    "QUARANTINE": {"label": "QUARANTINE", "icon": "🚫", "color": "#E24B4A", "bg": "#fff5f5"},
}


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def merge_policy(policy: Optional[dict]) -> dict:
    """A full policy dict = defaults overlaid with any user overrides."""
    return {**DEFAULT_POLICY, **(policy or {})}


# ──────────────────────────────────────────────────────────────────────────────
# Policy gates
# ──────────────────────────────────────────────────────────────────────────────
def policy_checks(manifest: dict, scored_df: Optional[pd.DataFrame], policy: Optional[dict]) -> list[dict]:
    """Evaluate every policy gate against a run manifest.

    Returns a list of gate dicts:
      {key, label, kind('min'|'max'), target, actual, pass, weight, margin}
    `margin` is how far the actual is on the passing side of the target
    (positive = pass with room to spare; negative = failing). It drives both the
    verdict (large negative on a hard gate ⇒ block) and the confidence.
    """
    p = merge_policy(policy)
    dims = manifest.get("dimensions", {}) or {}
    rows = manifest.get("row_count")
    if not rows and scored_df is not None:
        rows = len(scored_df)
    rows = int(rows or 0)

    actuals = {"overall": manifest.get("overall_dq_score"), **dims}
    checks: list[dict] = []

    # Per-dimension minimums (higher is better).
    for k in _DIM_KEYS:
        tkey = "min_" + k
        if tkey not in p or actuals.get(k) is None:
            continue
        a, t = float(actuals[k]), float(p[tkey])
        checks.append({
            "key": k, "label": _LABELS[k], "kind": "min",
            "target": round(t, 1), "actual": round(a, 1),
            "pass": a >= t, "margin": round(a - t, 1),
            "weight": _WEIGHTS.get(k),
        })

    # Critical-rows ceiling (lower is better).
    crit = int(manifest.get("critical_count") or 0)
    crit_pct = (100.0 * crit / rows) if rows else 0.0
    t = float(p["max_critical_pct"])
    checks.append({
        "key": "critical_pct", "label": _LABELS["critical_pct"], "kind": "max",
        "target": round(t, 2), "actual": round(crit_pct, 2),
        "pass": crit_pct <= t, "margin": round(t - crit_pct, 2), "weight": None,
        "count": crit,
    })

    # Anomaly-rows ceiling (lower is better) — only if the manifest carries it.
    anom = manifest.get("anomaly_count")
    if anom is not None and rows:
        anom_pct = 100.0 * int(anom) / rows
        t = float(p["max_anomaly_pct"])
        checks.append({
            "key": "anomaly_pct", "label": _LABELS["anomaly_pct"], "kind": "max",
            "target": round(t, 2), "actual": round(anom_pct, 2),
            "pass": anom_pct <= t, "margin": round(t - anom_pct, 2), "weight": None,
            "count": int(anom),
        })

    return checks


def _is_blocking(check: dict) -> bool:
    """A failed gate that is severe enough to force QUARANTINE."""
    if check["pass"]:
        return False
    if check["key"] == "critical_pct":
        return True  # any breach of the critical ceiling is disqualifying
    if check["key"] in _HARD_DIMENSIONS and check["margin"] <= -HARD_MARGIN:
        return True
    return False


def decide(checks: list[dict]) -> tuple[str, list[dict]]:
    """Map gate results to a verdict.

    QUARANTINE if any blocking gate trips; APPROVE if every gate passes;
    REMEDIATE otherwise (recoverable misses). Returns (verdict, driver_checks).
    """
    failed = [c for c in checks if not c["pass"]]
    blockers = [c for c in checks if _is_blocking(c)]
    if blockers:
        return "QUARANTINE", blockers
    if not failed:
        return "APPROVE", []
    return "REMEDIATE", failed


def confidence(checks: list[dict], verdict: str) -> int:
    """How clear-cut the call is (0–100) — NOT the data-quality score.

    Decisiveness = average distance of each gate from its threshold. Gates that
    pass (or fail) by a wide margin make the verdict obvious → high confidence;
    gates sitting right on the line make it a judgement call → lower confidence.
    """
    if not checks:
        return 60
    # Dimension margins are on a 0–100 scale; the % gates are on a smaller scale,
    # so weight them up a touch to keep them comparable.
    mags = []
    for c in checks:
        m = abs(float(c["margin"]))
        mags.append(m * (3.0 if c["key"].endswith("_pct") else 1.0))
    avg = sum(mags) / len(mags)
    return int(round(_clamp(55 + avg * 3.0, 55, 99)))


def _reasons(verdict: str, checks: list[dict], drivers: list[dict]) -> list[str]:
    out: list[str] = []
    passed = [c for c in checks if c["pass"]]
    failed = [c for c in checks if not c["pass"]]
    if verdict == "APPROVE":
        out.append(f"All {len(checks)} quality gates passed.")
        # Call out the strongest area for a positive note.
        dim_passed = [c for c in passed if c["kind"] == "min"]
        if dim_passed:
            best = max(dim_passed, key=lambda c: c["margin"])
            out.append(f"Strongest area: {best['label']} ({best['actual']:.0f} vs target {best['target']:.0f}).")
    else:
        for c in drivers:
            if c["kind"] == "min":
                out.append(f"{c['label']} is {c['actual']:.0f}, below the target of {c['target']:.0f}.")
            else:
                cnt = c.get("count")
                tail = f" ({cnt:,} rows)" if cnt is not None else ""
                out.append(f"{c['label']} is {c['actual']:.2f}%, above the {c['target']:.2f}% limit{tail}.")
        clean_passed = len(passed)
        if clean_passed:
            out.append(f"{clean_passed} of {len(checks)} gates still passed.")
    return out


def _next_steps(verdict: str, manifest: dict, n_actions: int) -> list[str]:
    crit = int(manifest.get("critical_count") or 0)
    if verdict == "APPROVE":
        return [
            "Publish the cleaned dataset to your gold / BI layer — no human review required.",
            "Archive this run's manifest so the History tab can trend it over time.",
            "Re-run Autopilot on the next refresh to keep the bar enforced.",
        ]
    if verdict == "REMEDIATE":
        steps = [
            "Download the cleaned dataset below — the safe auto-fixes are already applied.",
        ]
        if n_actions:
            steps.append(f"Work the {n_actions} prioritised action(s) below, highest priority first.")
        steps.append("Re-run Autopilot afterwards to confirm the verdict clears to APPROVE.")
        return steps
    # QUARANTINE
    steps = ["Do NOT use this data for decisions yet — it failed a hard quality gate."]
    if crit:
        steps.append(f"Escalate the {crit:,} critical row(s) to the data owner for source correction.")
    steps += [
        "Fix the root cause at source (or adjust the policy if the bar is wrong).",
        "Re-run the pipeline and Autopilot once the source data is corrected.",
    ]
    return steps


def _headline(verdict: str, manifest: dict) -> str:
    overall = manifest.get("overall_dq_score")
    overall = float(overall) if overall is not None else 0.0
    return {
        "APPROVE":    f"Data approved for use — overall quality {overall:.0f}/100 meets every target.",
        "REMEDIATE":  f"Usable after cleanup — overall quality {overall:.0f}/100, a few gates need attention.",
        "QUARANTINE": f"Hold this data — overall quality {overall:.0f}/100 breached a hard quality gate.",
    }[verdict]


def evaluate(
    manifest: dict,
    scored_df: Optional[pd.DataFrame] = None,
    policy: Optional[dict] = None,
    col_mapping: Optional[dict] = None,
) -> dict:
    """The Autopilot decision. Pure: same inputs → same verdict.

    Returns a self-contained decision dict (JSON-serialisable except for any
    pandas objects, which are not included):
      {verdict, verdict_icon, verdict_color, confidence, headline, reasons[],
       checks[], drivers[], action_plan[], next_steps[], policy, created_at,
       counts{...}}
    """
    checks = policy_checks(manifest, scored_df, policy)
    verdict, drivers = decide(checks)

    action_plan: list[dict] = []
    if scored_df is not None and _remediation is not None:
        try:
            action_plan = _remediation.suggested_actions(scored_df, col_mapping or {})
        except Exception:  # pragma: no cover - never let advice crash the verdict
            action_plan = []

    meta = _VERDICT_META[verdict]
    rows = int(manifest.get("row_count") or (len(scored_df) if scored_df is not None else 0))
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": manifest.get("dataset_name", "dataset"),
        "verdict": verdict,
        "verdict_icon": meta["icon"],
        "verdict_color": meta["color"],
        "verdict_bg": meta["bg"],
        "confidence": confidence(checks, verdict),
        "headline": _headline(verdict, manifest),
        "reasons": _reasons(verdict, checks, drivers),
        "checks": checks,
        "drivers": drivers,
        "action_plan": action_plan,
        "next_steps": _next_steps(verdict, manifest, len(action_plan)),
        "policy": merge_policy(policy),
        "counts": {
            "rows": rows,
            "gates_total": len(checks),
            "gates_passed": sum(1 for c in checks if c["pass"]),
            "gates_failed": sum(1 for c in checks if not c["pass"]),
            "critical": int(manifest.get("critical_count") or 0),
            "anomalies": int(manifest.get("anomaly_count") or 0) if manifest.get("anomaly_count") is not None else None,
            "overall": manifest.get("overall_dq_score"),
        },
    }


def run_autopilot(
    scored_df: pd.DataFrame,
    score_stats: dict,
    col_mapping: dict,
    policy: Optional[dict] = None,
    dataset_name: str = "dataset",
) -> dict:
    """One-call convenience: build the manifest, then decide.

    Requires `run_manifest`. Returns {"manifest": ..., "decision": ...}.
    The UI can call this directly; tests usually call `evaluate` with a hand-built
    manifest so they don't depend on the scorer.
    """
    if _run_manifest is None:  # pragma: no cover
        raise RuntimeError("run_manifest module is required for run_autopilot()")
    manifest = _run_manifest.build_manifest(scored_df, score_stats, col_mapping, dataset_name)
    decision = evaluate(manifest, scored_df, policy, col_mapping)
    return {"manifest": manifest, "decision": decision}


# ──────────────────────────────────────────────────────────────────────────────
# Reporting (downloadable artefacts) — pure string/dict builders.
# ──────────────────────────────────────────────────────────────────────────────
def checks_df(decision: dict) -> pd.DataFrame:
    """Tabular view of the policy gates for display / CSV."""
    rows = []
    for c in decision.get("checks", []):
        suffix = "%" if c["kind"] == "max" else ""
        rows.append({
            "Gate": c["label"],
            "Target": (f"≤ {c['target']:g}%" if c["kind"] == "max" else f"≥ {c['target']:g}"),
            "Actual": f"{c['actual']:g}{suffix}",
            "Status": "✅ Pass" if c["pass"] else "❌ Fail",
        })
    return pd.DataFrame(rows, columns=["Gate", "Target", "Actual", "Status"])


def action_plan_df(decision: dict) -> pd.DataFrame:
    rows = decision.get("action_plan", [])
    return pd.DataFrame(
        rows, columns=["priority", "issue", "count", "recommendation"]
    ).rename(columns={"priority": "Priority", "issue": "Issue",
                      "count": "Count", "recommendation": "Recommended action"})


def report_json(decision: dict) -> str:
    return json.dumps(decision, indent=2, ensure_ascii=False, default=str)


def report_markdown(decision: dict) -> str:
    """Printable Markdown decision report — shareable without the app."""
    c = decision
    lines = [
        f"# DataQual Autopilot — {c['verdict_icon']} {c['verdict']}",
        f"_{c['headline']}_",
        "",
        f"- **Dataset:** {c.get('dataset_name', 'dataset')}",
        f"- **Generated:** {c.get('created_at', '')}",
        f"- **Confidence:** {c.get('confidence', 0)}/100",
        f"- **Gates passed:** {c['counts']['gates_passed']}/{c['counts']['gates_total']}",
        f"- **Rows:** {c['counts']['rows']:,} · "
        f"Critical: {c['counts']['critical']:,} · "
        f"Anomalies: {('%d' % c['counts']['anomalies']) if c['counts']['anomalies'] is not None else 'n/a'}",
        "",
        "## Why",
    ]
    lines += [f"- {r}" for r in c.get("reasons", [])] or ["- (no reasons recorded)"]

    lines += ["", "## Policy gates", "", "| Gate | Target | Actual | Status |", "|---|---|---|---|"]
    for ch in c.get("checks", []):
        tgt = f"≤ {ch['target']:g}%" if ch["kind"] == "max" else f"≥ {ch['target']:g}"
        suffix = "%" if ch["kind"] == "max" else ""
        status = "PASS" if ch["pass"] else "FAIL"
        lines.append(f"| {ch['label']} | {tgt} | {ch['actual']:g}{suffix} | {status} |")

    plan = c.get("action_plan", [])
    if plan:
        lines += ["", "## Prioritised action plan", "",
                  "| Priority | Issue | Count | Recommended action |", "|---|---|---:|---|"]
        for a in plan:
            lines.append(f"| {a.get('priority','')} | {a.get('issue','')} | "
                         f"{a.get('count',0):,} | {a.get('recommendation','')} |")

    lines += ["", "## Recommended next steps"]
    lines += [f"{i}. {s}" for i, s in enumerate(c.get("next_steps", []), 1)]

    lines += ["", "## Policy used"]
    for k, v in (c.get("policy", {}) or {}).items():
        lines.append(f"- **{k}**: {v}")

    lines += ["", "_Generated by DataQual AI · Autopilot · scores are 0–100, higher is better._"]
    return "\n".join(lines)
