"""
run_manifest.py — Persistence keystone (roadmap B1) for DataQual AI.
DataQual AI · v3.0 Phase 2 · RAK Pvt. Ltd

WHY THIS MODULE EXISTS
----------------------
Today every run is throwaway: upload → score → close tab → gone. The single
highest-leverage upgrade is letting a user compare this month's run to last
month's. That turns a one-shot DQ *score* into a DQ *program*.

This module produces a small, portable JSON "run manifest" for each scoring run
(per-dimension scores, row/issue counts, grade & severity distribution, a stable
dataset signature, and a timestamp). Users download manifests and re-upload past
ones to render trend lines — automatic history WITHOUT a backend, honouring the
zero-backend promise for as long as possible.

THE MIGRATION CONTRACT (Systems-Architecture decision)
------------------------------------------------------
Storage sits behind the `StorageBackend` interface. Today the only implementation
is `JSONFileStore` (local files / user downloads — no server). The SAME manifest
schema drops into DuckDB/SQLite (local, still no server) and later Postgres
(hosted via Docker) with a one-class change. The JSON document IS the schema
contract — design it once, carry it all the way to the platform phase.

Dependencies: standard library + pandas only.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import pandas as pd

MANIFEST_SCHEMA_VERSION = 1

# The five scoring dimensions and the stats keys run_scoring() emits for each.
_DIMENSION_STATS = {
    "completeness": "avg_completeness",
    "validity": "avg_validity",
    "accuracy": "avg_accuracy",
    "consistency": "avg_consistency",
    "uniqueness": "avg_uniqueness",
}
_DIMENSION_WEIGHTS = {
    "completeness": 20, "validity": 25, "accuracy": 35,
    "consistency": 15, "uniqueness": 5,
}


def dataset_signature(df: pd.DataFrame) -> str:
    """Stable id for a dataset's SHAPE (column names + dtypes), not its contents.

    Re-uploading the same dataset next month yields the same signature, so history
    groups automatically. Renaming/adding columns yields a new signature — which is
    exactly the schema-drift boundary we want for B4 later.
    """
    parts = [f"{c}:{df[c].dtype}" for c in df.columns]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def build_manifest(
    scored_df: pd.DataFrame,
    score_stats: dict,
    col_mapping: dict,
    dataset_name: str = "dataset",
    created_at: Optional[str] = None,
) -> dict:
    """Build a run manifest dict from a scored dataframe + run_scoring() stats."""
    if created_at is None:
        created_at = datetime.now().isoformat(timespec="seconds")

    dims = {
        name: round(float(score_stats.get(key, float("nan"))), 2)
        for name, key in _DIMENSION_STATS.items()
    }

    overall = score_stats.get("mean_dq_score")
    if overall is None and "dq_score" in scored_df.columns:
        overall = float(scored_df["dq_score"].mean())

    grades: dict[str, int] = {}
    if "dq_grade" in scored_df.columns:
        grades = {str(k): int(v) for k, v in scored_df["dq_grade"].value_counts().items()}

    severities: dict[str, int] = {}
    if "dq_severity" in scored_df.columns:
        severities = {str(k): int(v) for k, v in scored_df["dq_severity"].value_counts().items()}

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": str(dataset_name),
        "dataset_signature": dataset_signature(scored_df),
        "created_at": created_at,
        "row_count": int(len(scored_df)),
        "column_count": int(scored_df.shape[1]),
        "overall_dq_score": round(float(overall), 2) if overall is not None else None,
        "dimensions": dims,
        "dimension_weights": dict(_DIMENSION_WEIGHTS),
        "grade_counts": grades,
        "severity_counts": severities,
        "critical_count": int(severities.get("CRITICAL", 0)),
        "anomaly_count": int(scored_df["is_anomaly"].sum()) if "is_anomaly" in scored_df.columns else None,
    }


def manifest_to_json(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False)


def manifest_to_bytes(manifest: dict) -> bytes:
    return manifest_to_json(manifest).encode("utf-8")


def suggested_filename(manifest: dict) -> str:
    stamp = (manifest.get("created_at") or "run").replace(":", "").replace("-", "")
    name = (manifest.get("dataset_name") or "dataset").replace(" ", "_")
    return f"dqmanifest_{name}_{stamp}.json"


_REQUIRED_KEYS = {"schema_version", "created_at", "overall_dq_score", "dimensions"}


def parse_manifest(data) -> dict:
    """Parse + validate an uploaded manifest (str/bytes/dict). Raise ValueError if invalid."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}")
    elif isinstance(data, dict):
        obj = data
    else:
        raise ValueError(f"unsupported manifest type: {type(data)}")

    missing = _REQUIRED_KEYS - set(obj)
    if missing:
        raise ValueError(f"manifest missing required keys: {sorted(missing)}")
    if int(obj.get("schema_version", 0)) > MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version {obj['schema_version']} is newer than supported "
                         f"({MANIFEST_SCHEMA_VERSION}); upgrade the app.")
    return obj


def manifests_to_trend_df(manifests: list[dict]) -> pd.DataFrame:
    """Flatten manifests into a tidy, time-sorted DataFrame for trend charts."""
    rows = []
    for m in manifests:
        dims = m.get("dimensions", {}) or {}
        rows.append({
            "created_at": m.get("created_at"),
            "dataset_name": m.get("dataset_name", "dataset"),
            "dataset_signature": m.get("dataset_signature", ""),
            "overall": m.get("overall_dq_score"),
            "completeness": dims.get("completeness"),
            "validity": dims.get("validity"),
            "accuracy": dims.get("accuracy"),
            "consistency": dims.get("consistency"),
            "uniqueness": dims.get("uniqueness"),
            "rows": m.get("row_count"),
            "critical": m.get("critical_count"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df.sort_values("created_at").reset_index(drop=True)


def compute_deltas(trend_df: pd.DataFrame) -> dict:
    """Latest-vs-previous deltas per dimension (seeds the B2 SLA/scorecard story).

    Returns {} when there is fewer than two runs to compare.
    """
    if trend_df is None or len(trend_df) < 2:
        return {}
    last, prev = trend_df.iloc[-1], trend_df.iloc[-2]
    out: dict[str, Any] = {"from": str(prev["created_at"]), "to": str(last["created_at"]), "dimensions": {}}
    for dim in ["overall", "completeness", "validity", "accuracy", "consistency", "uniqueness"]:
        a, b = prev.get(dim), last.get(dim)
        if pd.notna(a) and pd.notna(b):
            out["dimensions"][dim] = round(float(b) - float(a), 2)
    return out


def evaluate_slas(manifest: dict, slas: dict[str, float]) -> list[dict]:
    """Check a manifest against per-dimension SLA targets (B2 seed).

    `slas` maps dimension name (or 'overall') -> minimum acceptable score.
    Returns a list of {metric, target, actual, pass} rows.
    """
    results = []
    dims = manifest.get("dimensions", {}) or {}
    for metric, target in (slas or {}).items():
        actual = manifest.get("overall_dq_score") if metric == "overall" else dims.get(metric)
        if actual is None:
            continue
        results.append({
            "metric": metric,
            "target": float(target),
            "actual": round(float(actual), 2),
            "pass": bool(float(actual) >= float(target)),
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Storage interface — the migration seam (JSON now → DuckDB → Postgres later).
# ──────────────────────────────────────────────────────────────────────────────
class StorageBackend(ABC):
    @abstractmethod
    def save(self, manifest: dict) -> str: ...

    @abstractmethod
    def load_all(self, signature: Optional[str] = None) -> list[dict]: ...


class JSONFileStore(StorageBackend):
    """No-backend store: one JSON file per run under a local directory.

    On Streamlit Cloud the filesystem is ephemeral, so the primary UX is
    download/re-upload; this class is the local-dev / self-hosted convenience and
    the reference implementation the DB stores will mirror.
    """

    def __init__(self, directory: str = "output/manifests") -> None:
        import os
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def save(self, manifest: dict) -> str:
        import os
        path = os.path.join(self.directory, suggested_filename(manifest))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(manifest_to_json(manifest))
        return path

    def load_all(self, signature: Optional[str] = None) -> list[dict]:
        import glob
        import os
        out = []
        for p in glob.glob(os.path.join(self.directory, "*.json")):
            try:
                with open(p, encoding="utf-8") as fh:
                    m = parse_manifest(fh.read())
                if signature is None or m.get("dataset_signature") == signature:
                    out.append(m)
            except (ValueError, OSError):
                continue
        return out

# Future implementations (documented seam — implement at the platform phase):
#   class DuckDBStore(StorageBackend):   # local file db, still no server
#   class PostgresStore(StorageBackend): # hosted via Docker, multi-user
