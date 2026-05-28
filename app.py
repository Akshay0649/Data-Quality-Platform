"""
app.py — DataQual AI  v2.4
=============================================
Universal AI-powered DQ platform.
Works with ANY dataset — no domain selection needed.
Auto-detects column types and applies the right checks automatically.

New in v2.1:
  - Auto column-type detection (numeric · date · categorical · id · text · boolean)
  - Universal statistical profiler (per-column health stats)
  - Smart column-role mapper (dates, amounts, IDs detected automatically)
  - Profile tab: null heatmap, type breakdown, sample values, distribution stats
  - Cleaning & scoring now fully dynamic — no hardcoded column names

Run locally:  streamlit run app.py
Deploy:       push to GitHub → share.streamlit.io
"""

import json
import re
import random
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataQual AI — v2.4",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Targeted UI polish (safe selectors only) ──────────────────────────────────
st.markdown("""
<style>
/* Tabs — rosewood palette */
.stTabs [data-baseweb="tab-list"] {
    background: #F5EEF0;
    padding: 4px;
    border-radius: 12px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 8px 16px;
    color: #5E6472;
    font-weight: 600;
    font-size: 13px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #915466;
    color: #FDFFFF;
    box-shadow: 0 2px 8px rgba(145,84,102,0.3);
}
/* Metric labels */
[data-testid="stMetricLabel"] { color: #5E6472 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #23022E !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)



# ── Design System — Matte Pastel ─────────────────────────────────────────────
# Palette: soft lavender bg · white cards · pastel accents
CHART_BG   = "#ffffff"
CHART_PLOT = "#fafafa"
CHART_FONT = "#5E6472"

# Pastel accent map for dimensions / severity
PASTEL = {
    "indigo":  "#915466",   # rosewood primary
    "violet":  "#C79192",   # rosy taupe
    "sky":     "#5E6472",   # blue slate
    "teal":    "#1D9E75",
    "emerald": "#3B6D11",
    "amber":   "#EF9F27",
    "rose":    "#D85A30",
    "slate":   "#5E6472",
}

def _chart_layout(**kw):
    """Base Plotly layout — clean light theme. Pass any plotly layout keys as kwargs."""
    base = dict(
        template="plotly_white",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=CHART_FONT, family="Inter, system-ui, sans-serif", size=12),
        margin=dict(t=32, b=32, l=16, r=32),
        showlegend=False,
    )
    base.update(kw)  # caller kwargs win — no conflicts
    return base

# Shared axis defaults (use directly in update_layout)
_GRID_X = dict(showgrid=True,  gridcolor="#f1f5f9", zeroline=False, tickfont=dict(color="#5E6472"))
_GRID_Y = dict(showgrid=True,  gridcolor="#f1f5f9", zeroline=False, tickfont=dict(color="#5E6472"))
_NO_GRID_Y = dict(showgrid=False, zeroline=False, tickfont=dict(color="#5E6472"))

def kpi_card(label, value, subtitle="", color="#915466", icon="", bg=""):
    """Clean KPI card — white background, coloured top border."""
    ico_html = f"{icon} " if icon else ""
    sub_html = f'<p style="color:#5E6472;font-size:11px;margin:5px 0 0;">{subtitle}</p>' if subtitle else ""
    return (
        f'<div style="background:#fff;border-radius:12px;padding:18px 20px;'
        f'border:1px solid #E2E8F0;border-top:4px solid {color};'
        f'box-shadow:0 1px 6px rgba(0,0,0,0.05);height:100%;">'
        f'<p style="color:#5E6472;font-size:10px;font-weight:700;'
        f'letter-spacing:1px;text-transform:uppercase;margin:0 0 8px;">'
        f'{ico_html}{label}</p>'
        f'<p style="color:#23022E;font-size:26px;font-weight:800;margin:0;line-height:1;">'
        f'{value}</p>{sub_html}</div>'
    )

def section_head(title, subtitle=""):
    sub = f'<p style="color:#5E6472;font-size:12px;margin:3px 0 0;">{subtitle}</p>' if subtitle else ""
    return (f'<div style="margin:20px 0 12px;">' +
            f'<h3 style="color:#23022E;font-size:16px;font-weight:700;margin:0;">{title}</h3>' +
            sub + "</div>")


# ═══════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE QUERY ENGINE  (v2.4)
# ═══════════════════════════════════════════════════════════════════════════════
import re as _re

_STOPWORDS = {"show","me","find","get","list","give","display","all","the",
              "records","rows","entries","data","where","which","that","with",
              "have","has","are","is","a","an","of","in","for","and","or"}

def _tokens(q: str) -> list[str]:
    return [t for t in _re.sub(r"[^\w\s<>=!\.%]", " ", q.lower()).split()
            if t not in _STOPWORDS]

# ── Intent patterns ────────────────────────────────────────────────────────────
_SEV_MAP = {
    "critical":"CRITICAL","crit":"CRITICAL",
    "high":"HIGH",
    "medium":"MEDIUM","med":"MEDIUM","moderate":"MEDIUM",
    "low":"LOW",
    "clean":"CLEAN","good":"CLEAN","healthy":"CLEAN","ok":"CLEAN",
}
_GRADE_MAP = {"a":"A","b":"B","c":"C","d":"D","f":"F"}
_DIM_MAP = {
    "completeness":"dq_score_completeness","missing":"dq_score_completeness","null":"dq_score_completeness","empty":"dq_score_completeness",
    "validity":"dq_score_validity","format":"dq_score_validity","invalid":"dq_score_validity",
    "accuracy":"dq_score_accuracy","value":"dq_score_accuracy","range":"dq_score_accuracy",
    "consistency":"dq_score_consistency","consistent":"dq_score_consistency",
    "uniqueness":"dq_score_uniqueness","duplicate":"dq_score_uniqueness","duplication":"dq_score_uniqueness","duplicates":"dq_score_uniqueness",
}
_DIM_HUMAN = {
    "dq_score_completeness":"Missing Data",
    "dq_score_validity":"Format Accuracy",
    "dq_score_accuracy":"Value Accuracy",
    "dq_score_consistency":"Data Consistency",
    "dq_score_uniqueness":"Duplicate Check",
}

def _parse_number(tokens: list[str]) -> float | None:
    for t in tokens:
        try: return float(t.replace("%","").replace(",",""))
        except ValueError: pass
    return None

def parse_nlq(query: str, df: "pd.DataFrame", col_mapping: dict) -> dict:
    """
    Parse a natural-language query and return:
      {"mask": pd.Series(bool), "summary": str, "intent": str,
       "sort_col": str|None, "sort_asc": bool, "agg_df": pd.DataFrame|None}
    Returns None mask means 'return all'.
    """
    import pandas as pd, numpy as np
    q    = query.strip().lower()
    toks = _tokens(q)

    mask   = pd.Series([True]*len(df), index=df.index)
    intent = "filter"
    summary_parts = []
    sort_col  = None
    sort_asc  = False
    agg_df    = None
    limit     = None

    num_cols = col_mapping.get("numeric_columns", [])
    cat_cols = col_mapping.get("categorical_columns", [])
    id_cols  = col_mapping.get("id_columns", [])

    # ── 1. Severity filter ────────────────────────────────────────────────────
    sev_hits = [_SEV_MAP[t] for t in toks if t in _SEV_MAP]
    if sev_hits and "dq_severity" in df.columns:
        mask &= df["dq_severity"].isin(sev_hits)
        summary_parts.append(f"severity = {' or '.join(sev_hits)}")

    # ── 2. Grade filter ───────────────────────────────────────────────────────
    grade_m = _re.search(r'\bgrade\s+([a-fA-F])\b', q)
    if grade_m and "dq_grade" in df.columns:
        g = grade_m.group(1).upper()
        mask &= df["dq_grade"] == g
        summary_parts.append(f"grade = {g}")

    # ── 3. DQ score threshold ─────────────────────────────────────────────────
    score_m = _re.search(r'score\s*(below|under|less than|<|above|over|greater than|>|between)\s*([\d\.]+)(?:\s*and\s*([\d\.]+))?', q)
    if score_m and "dq_score" in df.columns:
        op, v1, v2 = score_m.group(1), float(score_m.group(2)), score_m.group(3)
        if op in ("below","under","less than","<"):
            mask &= df["dq_score"] < v1
            summary_parts.append(f"DQ score < {v1:.0f}")
        elif op in ("above","over","greater than",">"):
            mask &= df["dq_score"] > v1
            summary_parts.append(f"DQ score > {v1:.0f}")
        elif op == "between" and v2:
            mask &= df["dq_score"].between(v1, float(v2))
            summary_parts.append(f"DQ score between {v1:.0f}–{float(v2):.0f}")

    # ── 4. Anomaly / unusual filter ───────────────────────────────────────────
    anom_words = {"unusual","anomal","anomaly","anomalous","outlier","outliers","flagged","suspicious","weird","strange","risky","risk"}
    if any(w in q for w in anom_words):
        if "is_anomaly" in df.columns:
            mask &= df["is_anomaly"] == True
            summary_parts.append("flagged as anomalous by AI")
        elif "isolation_score" in df.columns:
            mask &= df["isolation_score"] >= 70
            summary_parts.append("AI risk score ≥ 70")

    # ── 5. Duplicate records ──────────────────────────────────────────────────
    dup_words = {"duplicate","duplicates","duplicated","dup","dups","repeated"}
    if any(w in q for w in dup_words):
        dup_col = next((c for c in id_cols if c in df.columns), None)
        if dup_col:
            dupes = df[dup_col].duplicated(keep=False)
            mask &= dupes
            summary_parts.append(f"duplicate {dup_col}")
        elif "dq_score_uniqueness" in df.columns:
            mask &= df["dq_score_uniqueness"] < 50
            summary_parts.append("uniqueness score < 50")

    # ── 6. Missing / null values ──────────────────────────────────────────────
    miss_words = {"missing","null","nulls","empty","blank","nan","incomplete"}
    if any(w in q for w in miss_words) and not any(w in q for w in ("completeness","score")):
        null_mask = df.isnull().any(axis=1)
        mask &= null_mask
        summary_parts.append("has missing values")

    # ── 7. Dimension weakness filter ─────────────────────────────────────────
    dim_hits = [(k, _DIM_MAP[k]) for k in _DIM_MAP if k in q]
    if dim_hits:
        thresh_m = _re.search(r'(below|under|<|less than)\s*([\d\.]+)', q)
        thresh = float(thresh_m.group(2)) if thresh_m else 70.0
        for kw, col in dim_hits:
            if col in df.columns:
                mask &= df[col] < thresh
                summary_parts.append(f"{_DIM_HUMAN[col]} score < {thresh:.0f}")

    # ── 8. Numeric column value filter  (amount > 50000, price < 100) ─────────
    num_m = _re.search(
        r'([\w_]+)\s*(>|>=|<|<=|=|==|above|below|over|under|greater than|less than)\s*([\d,\.]+)', q)
    if num_m:
        col_hint = num_m.group(1).replace(" ","_")
        op_str   = num_m.group(2)
        val      = float(num_m.group(3).replace(",",""))
        # fuzzy-match col_hint to actual df columns
        matched_col = next(
            (c for c in df.columns if col_hint in c.lower() or c.lower() in col_hint), None)
        if matched_col and pd.api.types.is_numeric_dtype(df[matched_col]):
            op_map = {">":">",">=":">=","<":"<","<=":"<=","=":"==","==":"==",
                      "above":">","over":">","greater than":">",
                      "below":"<","under":"<","less than":"<"}
            op = op_map.get(op_str, ">")
            cond = df[matched_col].dropna()
            if op == ">":   mask &= df[matched_col].fillna(-np.inf) > val
            elif op == ">=": mask &= df[matched_col].fillna(-np.inf) >= val
            elif op == "<":  mask &= df[matched_col].fillna(np.inf) < val
            elif op == "<=": mask &= df[matched_col].fillna(np.inf) <= val
            elif op == "==": mask &= df[matched_col] == val
            summary_parts.append(f"{matched_col} {op} {val:,.0f}")

    # ── 9. Categorical value filter  (vendor = 'Accenture', status = paid) ───
    cat_m = _re.search(r'([\w_]+)\s*(?:=|is|equals|called|named)\s*["\']?([\w\s]+)["\']?', q)
    if cat_m:
        col_hint = cat_m.group(1).replace(" ","_")
        val_hint = cat_m.group(2).strip()
        matched_col = next(
            (c for c in df.columns if col_hint in c.lower() or c.lower() in col_hint), None)
        if matched_col and matched_col not in ("dq_severity","dq_grade","is_anomaly"):
            col_vals = df[matched_col].astype(str).str.lower()
            hit = col_vals.str.contains(val_hint, na=False)
            if hit.sum() > 0:
                mask &= hit
                summary_parts.append(f"{matched_col} contains '{val_hint}'")

    # ── 10. Aggregation: which column has most issues ──────────────────────────
    agg_words = {"which","who","top","worst","best","most","fewest","rank","breakdown","by","group"}
    if any(w in toks for w in agg_words):
        grp_col = next((c for c in cat_cols if c in df.columns and
                        any(hint in q for hint in [c.lower(), c.split("_")[0].lower()])), None)
        if not grp_col and cat_cols:
            # default: first categorical column
            grp_col = next((c for c in cat_cols if c in df.columns), None)
        if grp_col:
            intent = "aggregate"
            agg_df = (
                df[mask].groupby(grp_col, dropna=False)
                .agg(
                    records=("dq_score","count"),
                    avg_score=("dq_score","mean"),
                    critical=(
                        "dq_severity",
                        lambda s: (s == "CRITICAL").sum()
                    ),
                )
                .reset_index()
                .rename(columns={grp_col: "Value", "records":"Records",
                                  "avg_score":"Avg DQ Score","critical":"Critical Issues"})
                .sort_values("Avg DQ Score")
            )
            agg_df["Avg DQ Score"] = agg_df["Avg DQ Score"].round(1)
            summary_parts.append(f"grouped by {grp_col}")

    # ── 11. Top N ─────────────────────────────────────────────────────────────
    top_m = _re.search(r'top\s+(\d+)', q)
    if top_m:
        limit = int(top_m.group(1))
        sort_col = "dq_score"
        sort_asc = True

    # ── 12. Sort direction hints ──────────────────────────────────────────────
    if any(w in q for w in ("worst","lowest","lowest score","bad")):
        sort_col = sort_col or "dq_score"; sort_asc = True
    if any(w in q for w in ("best","highest","highest score")):
        sort_col = sort_col or "dq_score"; sort_asc = False
    if "risk" in q or "risky" in q:
        if "isolation_score" in df.columns:
            sort_col = "isolation_score"; sort_asc = False

    # ── Build final summary ───────────────────────────────────────────────────
    n = int(mask.sum())
    if not summary_parts:
        if n == len(df):
            summary_txt = f"Showing all {len(df):,} records — try a more specific query."
        else:
            summary_txt = f"Found {n:,} records."
    else:
        criteria = " · ".join(summary_parts)
        summary_txt = f"Found **{n:,} records** matching: {criteria}."

    if limit:
        summary_txt += f" Showing top {limit}."

    return {
        "mask":     mask,
        "summary":  summary_txt,
        "intent":   intent,
        "sort_col": sort_col,
        "sort_asc": sort_asc,
        "agg_df":   agg_df,
        "limit":    limit,
        "n":        n,
    }

# Suggested example queries for the UI
NLQ_EXAMPLES = [
    "Show critical records",
    "Find anomalous records",
    "Records with score below 60",
    "Top 20 worst records",
    "Show records with missing values",
    "Find duplicate records",
    "High severity with low completeness",
    "Which vendors have the most issues",
    "Show high risk anomalies",
    "Records with format accuracy below 50",
    "Grade F records",
    "Show flagged records sorted by risk",
]


# ── Column-type metadata ──────────────────────────────────────────────────────
COL_TYPE_META = {
    "numeric":     {"icon": "🔢", "label": "Numeric",     "color": "#378ADD"},
    "date":        {"icon": "📅", "label": "Date",        "color": "#1D9E75"},
    "categorical": {"icon": "🏷️",  "label": "Categorical", "color": "#E24B4A"},
    "id":          {"icon": "🆔", "label": "ID / Key",    "color": "#534AB7"},
    "text":        {"icon": "📝", "label": "Free text",   "color": "#BA7517"},
    "boolean":     {"icon": "☑️",  "label": "Boolean",    "color": "#639922"},
}

# ── Severity / grade constants ────────────────────────────────────────────────
SEV_ORDER  = ["CLEAN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEV_COLORS = {"CLEAN": "#639922", "LOW": "#EF9F27",
              "MEDIUM": "#BA7517", "HIGH": "#D85A30", "CRITICAL": "#E24B4A"}
SEV_LABELS = {
    "CLEAN":    "✅ Clean — Ready to use",
    "LOW":      "🟡 Low — Worth monitoring",
    "MEDIUM":   "🟠 Medium — Schedule a fix",
    "HIGH":     "🔶 High — Review today",
    "CRITICAL": "🔴 Critical — Act now",
}
DIM_COLORS = {
    "Missing Data":     "#378ADD",
    "Format Accuracy":  "#E24B4A",
    "Value Accuracy":   "#3B6D11",
    "Data Consistency": "#1D9E75",
    "Duplicate Check":  "#534AB7",
}
PLACEHOLDER_VALUES = {"UNKNOWN", "N/A", "NA", "NONE", "", "NULL", "NAN", "-"}

# kept for demo generator only
VENDORS    = ["Accenture Ltd","  KPMG Advisory  ","Deloitte & Touche","ernst & young",
              "PwC Services","GARTNER INC","Infosys BPO","Wipro Technologies",
              "IBM Global  ","Capgemini SE","  Oracle Corp","SAP AG",
              "Microsoft Azure","AWS Finance","Cognizant Tech",None,"N/A","UNKNOWN VENDOR"]
CURRENCIES = ["USD","EUR","GBP","AUD","INR","XYZ","ZZZ","usd","Eur",None]
STATUSES   = ["PAID","PENDING","OVERDUE","CANCELLED","DRAFT",
              "paid","  Pending  ","overdue","settled","VOID",None,""]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="background:#F5EEF0;border-radius:12px;padding:14px 16px;margin-bottom:12px;
                border:1px solid #C79192;">
      <p style="color:#915466;font-size:15px;font-weight:800;margin:0 0 4px 0;">⚙️ Configuration</p>
      <p style="color:#5E6472;font-size:12px;margin:0;">Adjust thresholds before running the pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    null_fill_str = st.text_input("Fill empty text with", "UNKNOWN")
    null_fill_num = st.number_input("Fill empty numbers with", value=0.0)
    min_amount    = st.number_input("Minimum valid number", value=0.0)
    max_amount    = st.number_input("Maximum valid number", value=10_000_000.0, step=100_000.0)
    outlier_z     = st.number_input("Outlier sensitivity (higher = less strict)", value=3.0, step=0.5)
    dup_cols      = st.text_input("Duplicate key columns (;-separated, blank = all)", "")

    st.divider()
    st.caption("Built by Akshay — Data Engineer · v2.4")

CONFIG = {
    "null_fill_string":         null_fill_str,
    "null_fill_numeric":        null_fill_num,
    "min_amount":               min_amount,
    "max_amount":               max_amount,
    "outlier_zscore_threshold": outlier_z,
    "duplicate_subset":         dup_cols,
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def _rnd_date(start, end):
    d   = start + timedelta(days=random.randint(0, (end - start).days))
    fmt = random.choice(["%Y-%m-%d"]*5 + ["%d/%m/%Y", "%m-%d-%Y", "bad-date", ""])
    return None if fmt == "" else d.strftime(fmt)

@st.cache_data(show_spinner=False)
def generate_demo(n: int = 300, seed: int = 42) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    s, e = date(2023, 1, 1), date(2025, 12, 31)
    rows = []
    for i in range(1, n + 1):
        vendor = random.choice(VENDORS)
        inv    = f"INV-{random.randint(1000,9999)}"
        amt    = round(random.uniform(-500, 250_000), 2)
        if random.random() < 0.05: amt = round(random.uniform(1_000_000, 15_000_000), 2)
        if random.random() < 0.08: amt = -abs(amt)
        tax   = round(amt * random.uniform(0.05, 0.18), 2) if amt > 0 else None
        total = round(amt + (tax or 0), 2)
        rows.append({
            "invoice_id":     i,
            "invoice_number": inv,
            "vendor_name":    vendor,
            "invoice_date":   _rnd_date(s, e),
            "due_date":       _rnd_date(s, e),
            "payment_date":   _rnd_date(s, e) if random.random() > 0.35 else None,
            "amount":         amt if random.random() > 0.05 else None,
            "tax_amount":     tax,
            "total_amount":   total if random.random() > 0.05 else None,
            "currency":       random.choice(CURRENCIES),
            "status":         random.choice(STATUSES),
            "department":     random.choice(["Finance", "IT", "HR", "Ops", None]),
            "cost_centre":    random.choice(["CC100", "CC200", "CC300", "CC999", None]),
        })
    df    = pd.DataFrame(rows)
    dupes = df.sample(20, random_state=7).copy()
    df    = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df["invoice_id"] = range(1, len(df) + 1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL COLUMN-TYPE DETECTOR  (v2.1 NEW)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def detect_col_types(df: pd.DataFrame) -> dict:
    """
    Auto-detect semantic type for every column.
    Returns {col_name: 'numeric'|'date'|'categorical'|'id'|'text'|'boolean'}
    No domain knowledge required — works purely from data shape.
    """
    types = {}
    n = max(len(df), 1)

    for col in df.columns:
        s = df[col]

        # Already a datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "date"; continue

        # Numeric dtype
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique() <= 2 and set(s.dropna().unique()).issubset({0, 1}):
                types[col] = "boolean"
            elif s.nunique() / n > 0.90 and pd.api.types.is_integer_dtype(s):
                types[col] = "id"
            else:
                types[col] = "numeric"
            continue

        # Object / string columns
        sample = s.dropna().head(200).astype(str)
        if len(sample) == 0:
            types[col] = "text"; continue

        # Boolean-like strings
        uniq_lower = {v.lower().strip() for v in sample.unique()}
        if uniq_lower.issubset({"true","false","yes","no","1","0","y","n","t","f"}):
            types[col] = "boolean"; continue

        # Date strings — try parsing a sample
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.70:
                types[col] = "date"; continue
        except Exception:
            pass

        # Cardinality-based split
        nunique      = s.nunique()
        unique_ratio = nunique / n
        avg_len      = sample.str.len().mean()

        if unique_ratio > 0.85 and avg_len < 40:
            types[col] = "id"
        elif nunique <= max(30, n * 0.05):
            types[col] = "categorical"
        elif avg_len > 60:
            types[col] = "text"
        else:
            types[col] = "categorical"

    return types


# ═══════════════════════════════════════════════════════════════════════════════
# SMART COLUMN-ROLE MAPPER  (v2.1 NEW)
# ═══════════════════════════════════════════════════════════════════════════════
def map_columns_to_roles(df: pd.DataFrame, detected_types: dict) -> dict:
    """
    Maps columns to pipeline roles based purely on detected type.
    No domain knowledge needed.
    Returns:
      {
        'date_columns':        [...],
        'numeric_columns':     [...],
        'id_columns':          [...],
        'categorical_columns': [...],
        'text_columns':        [...],
        'critical_fields':     [...],   # highest-importance cols for completeness
      }
    """
    mapping = {
        "date_columns":        [],
        "numeric_columns":     [],
        "id_columns":          [],
        "categorical_columns": [],
        "text_columns":        [],
        "boolean_columns":     [],
    }
    for col, dtype in detected_types.items():
        if   dtype == "date":        mapping["date_columns"].append(col)
        elif dtype == "numeric":     mapping["numeric_columns"].append(col)
        elif dtype == "id":          mapping["id_columns"].append(col)
        elif dtype == "categorical": mapping["categorical_columns"].append(col)
        elif dtype == "text":        mapping["text_columns"].append(col)
        elif dtype == "boolean":     mapping["boolean_columns"].append(col)

    # Critical fields = IDs + first 3 numerics + first 3 categoricals
    mapping["critical_fields"] = (
        mapping["id_columns"][:3]
        + mapping["numeric_columns"][:3]
        + mapping["categorical_columns"][:3]
    )
    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL PROFILER  (v2.1 NEW)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame, types_json: str) -> list:
    """
    Per-column statistical profile.
    types_json: JSON-serialised detected_types dict (for cache key).
    """
    detected_types = json.loads(types_json)
    n = max(len(df), 1)
    profiles = []

    for col in df.columns:
        s     = df[col]
        dtype = detected_types.get(col, "text")
        null_count = int(s.isna().sum())
        nunique    = int(s.nunique())

        p = {
            "column":       col,
            "type":         dtype,
            "null_pct":     round(null_count / n * 100, 1),
            "null_count":   null_count,
            "unique_count": nunique,
            "unique_pct":   round(nunique / n * 100, 1),
            "sample":       " · ".join(str(v) for v in s.dropna().unique()[:4]),
            # numeric extras (filled below if applicable)
            "min": None, "max": None, "mean": None,
            "zero_pct": None, "neg_pct": None,
        }

        if dtype in ("numeric", "id") or pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                p["min"]      = round(float(num.min()), 2)
                p["max"]      = round(float(num.max()), 2)
                p["mean"]     = round(float(num.mean()), 2)
                p["zero_pct"] = round(float((num == 0).mean() * 100), 1)
                p["neg_pct"]  = round(float((num < 0).mean() * 100), 1)

        profiles.append(p)

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL CLEANING ENGINE  (v2.1 — fully dynamic, no hardcoded column names)
# ═══════════════════════════════════════════════════════════════════════════════
def run_cleaning(df: pd.DataFrame, col_mapping: dict) -> tuple:
    fs  = CONFIG["null_fill_string"]
    fn  = float(CONFIG["null_fill_numeric"])
    mn  = float(CONFIG["min_amount"])
    mx  = float(CONFIG["max_amount"])
    thr = float(CONFIG["outlier_zscore_threshold"])
    dup_raw = CONFIG["duplicate_subset"].strip()

    df = df.copy()

    # ── 1. Track rows that had nulls ──────────────────────────────────────────
    df["had_nulls"] = df.isnull().any(axis=1).astype(int)
    nulls_before = int(df.isnull().sum().sum())

    # ── 2. Fill nulls by type ─────────────────────────────────────────────────
    for col in df.columns:
        if col == "had_nulls": continue
        if df[col].dtype == object:
            df[col] = df[col].fillna(fs)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(fn)
    nulls_filled = nulls_before - int(df.isnull().sum().sum())

    # ── 3. Deduplication ─────────────────────────────────────────────────────
    if dup_raw:
        subset = [c.strip() for c in dup_raw.split(";") if c.strip() in df.columns]
    else:
        subset = col_mapping.get("id_columns", []) or None
    rows_before   = len(df)
    df            = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    dupes_removed = rows_before - len(df)

    # ── 4. Date normalization — all detected date columns ─────────────────────
    date_errors = 0
    for col in col_mapping.get("date_columns", []):
        if col not in df.columns: continue
        orig   = df[col].copy()
        parsed = pd.to_datetime(df[col], errors="coerce")
        err    = parsed.isna() & orig.notna() & (~orig.astype(str).isin(["", fs, "nan", "None"]))
        df[f"{col}_parse_error"] = err.astype(int)
        date_errors += int(err.sum())
        df[col] = parsed.dt.strftime("%Y-%m-%d").where(~parsed.isna(), other=fs)

    # ── 5. Numeric validation — all detected numeric columns ──────────────────
    neg_flags = 0
    for col in col_mapping.get("numeric_columns", []):
        if col not in df.columns: continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fn)
        df[f"{col}_negative"]  = (df[col] < 0).astype(int)
        df[f"{col}_below_min"] = (df[col] < mn).astype(int)
        df[f"{col}_above_max"] = (df[col] > mx).astype(int)
        neg_flags += int((df[col] < 0).sum())

    # ── 6. Categorical standardisation ────────────────────────────────────────
    cat_fixed = 0
    for col in col_mapping.get("categorical_columns", []):
        if col not in df.columns: continue
        before = df[col].astype(str).copy()
        df[col] = (df[col].astype(str).str.strip()
                          .str.upper()
                          .replace({"NAN": fs, "NONE": fs, "": fs, "NULL": fs}))
        cat_fixed += int((df[col] != before).sum())

    # ── 7. ID / text columns — strip whitespace, fill placeholders ────────────
    for col in col_mapping.get("id_columns", []) + col_mapping.get("text_columns", []):
        if col not in df.columns: continue
        df[col] = df[col].astype(str).str.strip()
        ph = df[col].str.upper().isin(PLACEHOLDER_VALUES)
        df.loc[ph, col] = fs

    # ── 8. Outlier flagging — all numeric columns ─────────────────────────────
    outliers = 0
    for col in col_mapping.get("numeric_columns", []):
        if col not in df.columns: continue
        num = pd.to_numeric(df[col], errors="coerce")
        std = num.std()
        z   = (num - num.mean()) / std if std and std > 0 else pd.Series(0.0, index=df.index)
        df[f"{col}_outlier"] = (z.abs() > thr).astype(int)
        outliers += int((z.abs() > thr).sum())

    stats = {
        "rows_input":            rows_before,
        "rows_output":           len(df),
        "duplicates_removed":    dupes_removed,
        "nulls_filled":          nulls_filled,
        "rows_with_nulls":       int(df["had_nulls"].sum()),
        "date_parse_errors":     date_errors,
        "negative_amount_flags": neg_flags,
        "categorical_fixed":     cat_fixed,
        "outliers_flagged":      outliers,
        "date_cols_processed":   len(col_mapping.get("date_columns", [])),
        "numeric_cols_processed":len(col_mapping.get("numeric_columns", [])),
        "cat_cols_processed":    len(col_mapping.get("categorical_columns", [])),
    }
    return df, stats


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL SCORING ENGINE  (v2.1 — fully dynamic)
# ═══════════════════════════════════════════════════════════════════════════════
def _clamp(s): return s.clip(0, 100)

def run_scoring(df: pd.DataFrame, col_mapping: dict) -> tuple:
    df  = df.copy()
    n   = len(df)
    fs  = CONFIG["null_fill_string"]

    critical = col_mapping.get("critical_fields", [f for f in df.columns[:6]])
    num_cols = col_mapping.get("numeric_columns", [])
    id_cols  = col_mapping.get("id_columns", [])
    cat_cols = col_mapping.get("categorical_columns", [])
    date_cols= col_mapping.get("date_columns", [])

    # ── COMPLETENESS (20%) ────────────────────────────────────────────────────
    sc = pd.Series(100.0, index=df.index)
    if critical:
        ppf = 100.0 / len(critical)
        for f in critical:
            if f not in df.columns: continue
            sc -= (df[f].isna() | df[f].astype(str).str.strip().str.upper()
                   .isin(PLACEHOLDER_VALUES)).astype(float) * ppf
    df["dq_score_completeness"] = _clamp(sc).round(1)

    # ── VALIDITY (25%) ────────────────────────────────────────────────────────
    sv = pd.Series(100.0, index=df.index)
    # Date parse errors — penalise per column
    date_pen = max(10, int(60 / max(len(date_cols), 1)))
    for col in date_cols:
        ecol = f"{col}_parse_error"
        if ecol in df.columns:
            sv -= df[ecol].fillna(0).astype(int) * date_pen
    # Categorical columns with very high placeholder rate get a light penalty
    for col in cat_cols[:5]:
        if col in df.columns:
            ph_rate = df[col].astype(str).str.upper().isin(PLACEHOLDER_VALUES).mean()
            if ph_rate > 0.5:
                sv -= pd.Series(20.0, index=df.index)
    df["dq_score_validity"] = _clamp(sv).round(1)

    # ── ACCURACY (35%) ────────────────────────────────────────────────────────
    sa = pd.Series(100.0, index=df.index)
    neg_pen = max(20, int(70 / max(len(num_cols), 1)))
    out_pen = max(10, int(30 / max(len(num_cols), 1)))
    abv_pen = max(10, int(30 / max(len(num_cols), 1)))
    for col in num_cols:
        if f"{col}_negative"  in df.columns: sa -= df[f"{col}_negative"].fillna(0)  * neg_pen
        if f"{col}_above_max" in df.columns: sa -= df[f"{col}_above_max"].fillna(0)  * abv_pen
        if f"{col}_outlier"   in df.columns: sa -= df[f"{col}_outlier"].fillna(0)    * out_pen
    df["dq_score_accuracy"] = _clamp(sa).round(1)

    # ── CONSISTENCY (15%) ─────────────────────────────────────────────────────
    sco = pd.Series(100.0, index=df.index)
    # Universal: check pairs of date columns for logical ordering
    # e.g. start < end, created < updated, invoice < due
    START_HINTS = {"start", "created", "invoice", "hire", "open",  "from", "begin", "order"}
    END_HINTS   = {"end",   "due",     "close",   "term", "closed","to",   "expire","delivery","ship"}
    date_pairs  = []
    for c1 in date_cols:
        for c2 in date_cols:
            if c1 == c2: continue
            c1l = c1.lower(); c2l = c2.lower()
            c1_start = any(h in c1l for h in START_HINTS)
            c2_end   = any(h in c2l for h in END_HINTS)
            if c1_start and c2_end:
                date_pairs.append((c1, c2))
    for c1, c2 in date_pairs[:3]:  # check up to 3 pairs
        d1 = pd.to_datetime(df[c1], errors="coerce")
        d2 = pd.to_datetime(df[c2], errors="coerce")
        bad = d1.notna() & d2.notna() & (d2 < d1)
        sco -= bad.astype(float) * 40
    df["dq_score_consistency"] = _clamp(sco).round(1)

    # ── UNIQUENESS (5%) ───────────────────────────────────────────────────────
    su = pd.Series(100.0, index=df.index)
    if id_cols:
        # Penalise if ANY id column has duplicates
        for col in id_cols[:2]:
            if col in df.columns:
                su -= df[col].duplicated(keep=False).astype(float) * (50 / min(len(id_cols), 2))
    df["dq_score_uniqueness"] = _clamp(su).round(1)

    # ── COMPOSITE ────────────────────────────────────────────────────────────
    overall = (
        df["dq_score_completeness"] * 0.20 +
        df["dq_score_validity"]     * 0.25 +
        df["dq_score_accuracy"]     * 0.35 +
        df["dq_score_consistency"]  * 0.15 +
        df["dq_score_uniqueness"]   * 0.05
    ).clip(0, 100).round(1)

    df["dq_score"] = overall.astype(int)
    df["dq_grade"] = overall.apply(
        lambda s: "A" if s>=90 else "B" if s>=75 else "C" if s>=60 else "D" if s>=40 else "F"
    )

    def sev(r):
        if r.dq_score_accuracy < 30 or r.dq_score_consistency < 40: return "CRITICAL"
        if r.dq_score_accuracy < 60 or r.dq_score_validity < 40 or r.dq_score < 50: return "HIGH"
        if r.dq_score < 70 or min(r.dq_score_accuracy, r.dq_score_consistency,
                                  r.dq_score_validity, r.dq_score_completeness) < 60: return "MEDIUM"
        if r.dq_score < 85: return "LOW"
        return "CLEAN"

    df["dq_severity"] = df[["dq_score","dq_score_accuracy","dq_score_consistency",
                             "dq_score_validity","dq_score_completeness"]].apply(sev, axis=1)

    # ── ISSUE LOG ─────────────────────────────────────────────────────────────
    issue_rules = (
        [(f"{c}_parse_error", f"[Validity] {c}: date unparseable")   for c in date_cols] +
        [(f"{c}_negative",    f"[Accuracy] {c}: negative value")      for c in num_cols] +
        [(f"{c}_above_max",   f"[Accuracy] {c}: exceeds max threshold") for c in num_cols] +
        [(f"{c}_outlier",     f"[Accuracy] {c}: statistical outlier") for c in num_cols] +
        [("had_nulls",        "[Completeness] Row had missing values")]
    )
    issues = [""] * n
    for flag_col, label in issue_rules:
        if flag_col not in df.columns: continue
        for i, v in enumerate(df[flag_col].fillna(0).astype(int).tolist()):
            if v:
                issues[i] = (issues[i] + " | " + label) if issues[i] else label
    # Consistency date pair issues
    for c1, c2 in date_pairs[:3]:
        d1 = pd.to_datetime(df[c1], errors="coerce")
        d2 = pd.to_datetime(df[c2], errors="coerce")
        bad = (d1.notna() & d2.notna() & (d2 < d1)).tolist()
        lbl = f"[Consistency] {c2} is before {c1}"
        for i, v in enumerate(bad):
            if v:
                issues[i] = (issues[i] + " | " + lbl) if issues[i] else lbl
    df["dq_issues"] = issues

    grade_dist = df["dq_grade"].value_counts().to_dict()
    sev_dist   = df["dq_severity"].value_counts().to_dict()
    stats = {
        "mean_dq_score":    round(float(overall.mean()), 1),
        "median_dq_score":  round(float(overall.median()), 1),
        "perfect_rows":     int((overall >= 99).sum()),
        "avg_completeness": round(float(df["dq_score_completeness"].mean()), 1),
        "avg_validity":     round(float(df["dq_score_validity"].mean()), 1),
        "avg_accuracy":     round(float(df["dq_score_accuracy"].mean()), 1),
        "avg_consistency":  round(float(df["dq_score_consistency"].mean()), 1),
        "avg_uniqueness":   round(float(df["dq_score_uniqueness"].mean()), 1),
        **{f"grade_{k}": v for k, v in grade_dist.items()},
        **{f"sev_{k}":   v for k, v in sev_dist.items()},
    }
    return df, stats


# ═══════════════════════════════════════════════════════════════════════════════
# APP HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#23022E 0%,#3D1040 100%);
            border-radius:16px;padding:24px 32px;margin-bottom:20px;
            border:1.5px solid #915466;
            box-shadow:0 4px 24px rgba(145,84,102,0.25);">
  <div style="display:flex;align-items:center;gap:20px;">
    <div style="background:linear-gradient(135deg,#915466,#C79192);
                border-radius:16px;width:58px;height:58px;display:flex;
                align-items:center;justify-content:center;font-size:28px;
                box-shadow:0 4px 16px rgba(145,84,102,0.45);flex-shrink:0;">🔬</div>
    <div style="flex:1;">
      <h1 style="color:#FDFFFF;margin:0;font-size:27px;font-weight:800;
                 letter-spacing:-0.5px;">DataQual AI</h1>
      <p style="color:#C79192;margin:6px 0 0;font-size:13.5px;line-height:1.5;">
        Upload any dataset &nbsp;·&nbsp; AI detects issues instantly &nbsp;·&nbsp; Make confident decisions
      </p>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <span style="background:#915466;color:#FDFFFF;font-size:11px;font-weight:700;
                   padding:6px 14px;border-radius:20px;letter-spacing:0.8px;
                   border:1.5px solid #C79192;">v2.4</span>
      <p style="color:#C79192;font-size:10px;margin:6px 0 0;letter-spacing:0.5px;text-transform:uppercase;">Universal DQ Platform</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📂 Start here — upload your data")

src_col1, src_col2 = st.columns([3, 1])
with src_col1:
    uploaded_file = st.file_uploader(
        "Upload any CSV — finance, HR, sales, supply chain, or anything else",
        type=["csv"],
        help="The pipeline auto-detects column types and applies the right checks.",
    )
with src_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    n_demo    = st.number_input("Demo rows", 100, 1000, 300, 50)
    demo_seed = st.number_input("Seed", value=42, step=1)
    gen_btn   = st.button("🎲 Generate demo data", use_container_width=True)

if gen_btn:
    st.session_state["input_df"] = generate_demo(int(n_demo), int(demo_seed))
    st.session_state.pop("scored_df", None)
    st.session_state.pop("detected_types", None)
    st.session_state["_file_key"] = "demo"
    st.success(f"Demo dataset ready — {len(st.session_state['input_df'])} rows")

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_file_key") != file_key:
        try:
            st.session_state["input_df"] = pd.read_csv(uploaded_file)
            st.session_state.pop("scored_df", None)
            st.session_state.pop("detected_types", None)
            st.session_state["_file_key"] = file_key
            df_tmp = st.session_state["input_df"]
            st.success(f"Uploaded — {len(df_tmp):,} rows × {len(df_tmp.columns)} columns")
        except Exception as ex:
            st.error(f"Could not read CSV: {ex}")


# ── Anomaly Detection ─────────────────────────────────────────────────────────
def run_anomaly_detection(df: pd.DataFrame, col_mapping: dict) -> pd.DataFrame:
    """
    Adds two columns:
      isolation_score    – IsolationForest score mapped to 0–100 (higher = more anomalous)
      iqr_outlier_count  – number of numeric cols where the row is an IQR outlier
    """
    num_cols = [c for c in col_mapping.get("numeric_columns", []) if c in df.columns]
    result = df.copy()

    # ── IQR outlier count (per-column, independent) ───────────────────────────
    iqr_flags = pd.DataFrame(False, index=df.index, columns=num_cols)
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_flags[col] = df[col].lt(lo) | df[col].gt(hi)
    result["iqr_outlier_count"] = iqr_flags.sum(axis=1).astype(int)

    # ── IsolationForest (multivariate across all numeric cols) ────────────────
    if len(num_cols) >= 1 and len(df) >= 10:
        X = df[num_cols].copy()
        X = X.fillna(X.median())          # impute for model
        try:
            iso = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
                n_jobs=-1,
            )
            raw_scores = iso.score_samples(X)   # more negative = more anomalous
            # Normalise to 0–100 (100 = most anomalous)
            lo_s, hi_s = raw_scores.min(), raw_scores.max()
            if hi_s > lo_s:
                norm = 100 * (1 - (raw_scores - lo_s) / (hi_s - lo_s))
            else:
                norm = np.zeros(len(raw_scores))
            result["isolation_score"] = np.round(norm, 1)
        except Exception:
            result["isolation_score"] = 0.0
    else:
        result["isolation_score"] = 0.0

    # ── Combined anomaly flag ─────────────────────────────────────────────────
    result["is_anomaly"] = (
        (result["isolation_score"] >= 70) |
        (result["iqr_outlier_count"] >= 2)
    )
    return result

# ─── Raw preview ─────────────────────────────────────────────────────────────
if "input_df" in st.session_state:
    input_df = st.session_state["input_df"]

    with st.expander("👁️ Raw data preview", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.metric("Rows",           f"{len(input_df):,}")
        p2.metric("Columns",        len(input_df.columns))
        p3.metric("Missing values", f"{int(input_df.isnull().sum().sum()):,}")
        st.dataframe(input_df.head(20), use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # v2.1 — DATA PROFILE SECTION (before running the pipeline)
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("🔍 Auto-Detected Data Profile")
    st.caption("Column types detected automatically — no configuration needed")

    # Run detection (cached per dataframe)
    if "detected_types" not in st.session_state:
        with st.spinner("Detecting column types…"):
            detected_types = detect_col_types(input_df)
        st.session_state["detected_types"] = detected_types
    else:
        detected_types = st.session_state["detected_types"]

    col_mapping = map_columns_to_roles(input_df, detected_types)
    st.session_state["col_mapping"] = col_mapping

    # Type-count KPIs
    type_counts = Counter(detected_types.values())
    tk = st.columns(6)
    for i, (dtype, meta) in enumerate(COL_TYPE_META.items()):
        cnt = type_counts.get(dtype, 0)
        tk[i].metric(f"{meta['icon']} {meta['label']}", cnt)

    st.divider()

    # Column mapping summary
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.markdown("**📅 Date columns**")
    cm1.write("\n".join(f"• `{c}`" for c in col_mapping["date_columns"]) or "_none detected_")
    cm2.markdown("**🔢 Numeric columns**")
    cm2.write("\n".join(f"• `{c}`" for c in col_mapping["numeric_columns"]) or "_none detected_")
    cm3.markdown("**🆔 ID columns**")
    cm3.write("\n".join(f"• `{c}`" for c in col_mapping["id_columns"]) or "_none detected_")
    cm4.markdown("**🏷️ Categorical columns**")
    cm4.write("\n".join(f"• `{c}`" for c in col_mapping["categorical_columns"]) or "_none detected_")

    st.divider()

    # Column health table
    with st.expander("📊 Full column profile", expanded=True):
        profiles = profile_dataframe(input_df, json.dumps(detected_types))
        prof_df  = pd.DataFrame(profiles)

        display_cols = ["column","type","null_pct","unique_count","unique_pct","sample"]
        if "min" in prof_df.columns:
            display_cols += ["min","max","mean","zero_pct","neg_pct"]

        st.dataframe(
            prof_df[[c for c in display_cols if c in prof_df.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "null_pct":    st.column_config.ProgressColumn("Null %",    min_value=0, max_value=100, format="%.1f%%"),
                "unique_pct":  st.column_config.ProgressColumn("Unique %",  min_value=0, max_value=100, format="%.1f%%"),
                "zero_pct":    st.column_config.ProgressColumn("Zero %",    min_value=0, max_value=100, format="%.1f%%"),
                "neg_pct":     st.column_config.ProgressColumn("Negative %",min_value=0, max_value=100, format="%.1f%%"),
                "type":        st.column_config.TextColumn("Detected type"),
                "sample":      st.column_config.TextColumn("Sample values"),
            }
        )

        # Null heatmap bar chart
        null_df = prof_df[prof_df["null_pct"] > 0].sort_values("null_pct", ascending=False)
        if not null_df.empty:
            st.subheader("🕳️ Missing data by column")
            fig_null = px.bar(
                null_df, x="column", y="null_pct",
                color="null_pct",
                color_continuous_scale=["#86efac","#fde68a","#fca5a5","#f87171"],
                range_color=[0,100],
                labels={"null_pct":"Missing %","column":"Column"},
                text="null_pct",
            )
            fig_null.update_traces(
                texttemplate="%{y:.1f}%", textposition="outside",
                textfont=dict(color="#23022E", size=11),
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% missing<extra></extra>",
            )
            fig_null.update_layout(
                **_chart_layout(height=300, coloraxis_showscale=False),
                xaxis=dict(**_GRID_X, title=None),
                yaxis=dict(**_GRID_Y, title="% Missing"),
                bargap=0.35,
            )
            st.plotly_chart(fig_null, use_container_width=True)

    st.divider()

    # ─── Run Pipeline ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#23022E,#3D1040);border-radius:14px;
                padding:20px 28px;margin-bottom:16px;border:1px solid #915466;
                text-align:center;">
      <p style="color:#C79192;font-size:12px;font-weight:700;letter-spacing:1px;
                text-transform:uppercase;margin:0 0 6px;">Ready to analyse?</p>
      <p style="color:#FDFFFF;font-size:15px;font-weight:600;margin:0 0 4px;">
        Run the full AI quality pipeline on your data
      </p>
      <p style="color:#C79192;font-size:12px;margin:0;">
        Cleaning · Scoring · Anomaly Detection — all in one click
      </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Analyse My Data — Run Full Pipeline", type="primary", use_container_width=True):
        with st.spinner("Cleaning data…"):
            cleaned_df, clean_stats = run_cleaning(input_df, col_mapping)
        with st.spinner("Scoring across 5 dimensions…"):
            scored_df, score_stats  = run_scoring(cleaned_df, col_mapping)
        with st.spinner("Running anomaly detection (IsolationForest + IQR)…"):
            scored_df = run_anomaly_detection(scored_df, col_mapping)
        st.session_state["scored_df"]   = scored_df
        st.session_state["clean_stats"] = clean_stats
        st.session_state["score_stats"] = score_stats
        n_anom = int(scored_df.get("is_anomaly", pd.Series(dtype=bool)).sum())
        st.success(f"Pipeline complete — {len(scored_df):,} records scored · {n_anom} anomalies detected.")
        st.info("💬 **Next step:** Click the **Ask Your Data** tab (second tab) to query your results in plain English.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS — 7 TABS

# ── Story Banner (plain-English narrative) ────────────────────────────────────
def build_story_banner(df_f: pd.DataFrame, col_mapping: dict) -> str:
    total     = len(df_f)
    if total == 0:
        return ""
    avg_score = df_f["dq_score"].mean()
    critical  = int((df_f["dq_severity"] == "CRITICAL").sum())
    high      = int((df_f["dq_severity"] == "HIGH").sum())
    anomalies = int(df_f["is_anomaly"].sum()) if "is_anomaly" in df_f.columns else 0
    clean     = int((df_f["dq_severity"] == "CLEAN").sum())

    if avg_score >= 85:
        headline  = "✅ Your data looks healthy overall."
        bdr, bg   = "#1D9E75", "#f0fdf4"
    elif avg_score >= 65:
        headline  = "⚠️ Your data is mostly reliable — a few things need attention before major decisions."
        bdr, bg   = "#BA7517", "#fffbeb"
    else:
        headline  = "🔴 Your data has significant quality issues that could affect business decisions."
        bdr, bg   = "#E24B4A", "#fff5f5"

    bullets = []
    if critical > 0:
        bullets.append(f"🔴 <strong>{critical:,} records</strong> need immediate action — resolve these before relying on this data.")
    if high > 0:
        bullets.append(f"🟠 <strong>{high:,} records</strong> have significant issues — review today.")
    if anomalies > 0:
        bullets.append(f"🚨 <strong>{anomalies:,} records</strong> look statistically unusual compared to the rest — verify before decisions.")
    if clean > 0:
        pct = round(100 * clean / total)
        bullets.append(f"✅ <strong>{clean:,} records ({pct}%)</strong> are fully clean and ready to use.")

    dims = {
        "Missing Data":     df_f["dq_score_completeness"].mean(),
        "Format Accuracy":  df_f["dq_score_validity"].mean(),
        "Value Accuracy":   df_f["dq_score_accuracy"].mean(),
        "Data Consistency": df_f["dq_score_consistency"].mean(),
        "Duplicate Check":  df_f["dq_score_uniqueness"].mean(),
    }
    lowest_name, lowest_val = min(dims.items(), key=lambda x: x[1])
    if lowest_val < 80:
        bullets.append(f"📐 <strong>{lowest_name}</strong> is your weakest area ({lowest_val:.0f}/100) — start here.")

    bullets_html = "".join(f'<li style="margin:7px 0;line-height:1.5">{b}</li>' for b in bullets)
    return f"""
    <div style="background:#ffffff;border:1.5px solid #C79192;border-left:5px solid {bdr};
                border-radius:14px;padding:20px 24px;margin-bottom:24px;
                box-shadow:0 2px 12px rgba(0,0,0,0.04);">
      <p style="margin:0 0 14px;font-size:15px;font-weight:700;color:#23022E;">{headline}</p>
      <ul style="margin:0;padding-left:20px;color:#5E6472;font-size:13.5px;line-height:1.9;">{bullets_html}</ul>
    </div>"""

# ═══════════════════════════════════════════════════════════════════════════════
if "scored_df" in st.session_state:
    df          = st.session_state["scored_df"]
    clean_stats = st.session_state["clean_stats"]
    score_stats = st.session_state["score_stats"]
    col_mapping = st.session_state.get("col_mapping", {})
    det_types   = st.session_state.get("detected_types", {})

    st.divider()
    st.markdown("""
    <div style="margin:8px 0 20px;">
      <h2 style="margin:0 0 4px;font-size:22px;font-weight:800;color:#23022E;">📊 Your Data Report</h2>
      <p style="color:#5E6472;margin:0;font-size:13px;">Pipeline complete — here's what we found</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar result filters
    with st.sidebar:
        st.divider()
        st.subheader("🎯 Filter Results")
        sev_filter   = st.multiselect("Severity", SEV_ORDER, default=SEV_ORDER)
        grade_filter = st.multiselect("Grade", ["A","B","C","D","F"], default=["A","B","C","D","F"])
        score_range  = st.slider("DQ Score", 0, 100, (0, 100))

    df_f = df[
        df["dq_severity"].isin(sev_filter) &
        df["dq_grade"].isin(grade_filter) &
        df["dq_score"].between(score_range[0], score_range[1])
    ].copy()


    # ── Tab persistence — keep active tab across reruns ─────────────────────
    st.markdown("""
<script>
(function() {
  function restoreTab() {
    var saved = sessionStorage.getItem("dq_active_tab");
    if (!saved) return;
    var tabs = document.querySelectorAll('[data-baseweb="tab"]');
    var idx = parseInt(saved, 10);
    if (tabs && tabs[idx]) { tabs[idx].click(); }
  }
  function attachListeners() {
    var tabs = document.querySelectorAll('[data-baseweb="tab"]');
    tabs.forEach(function(tab, idx) {
      tab.addEventListener("click", function() {
        sessionStorage.setItem("dq_active_tab", idx);
      });
    });
  }
  var tries = 0;
  var timer = setInterval(function() {
    var tabs = document.querySelectorAll('[data-baseweb="tab"]');
    if (tabs.length > 0 || tries > 20) {
      clearInterval(timer);
      attachListeners();
      restoreTab();
    }
    tries++;
  }, 200);
})();
</script>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🔍 Profile",
        "💬 Ask Your Data",
        "📊 Dashboard",
        "📐 Dimension Analysis",
        "📋 Record Explorer",
        "🔬 Deep Dive",
        "📈 Statistics",
        "⬇️ Export",
        "🚨 Risk Signals",
    ])

    # =========================================================================
    # TAB 1 — PROFILE  (v2.1 NEW)
    # =========================================================================
    with tab1:
        st.subheader("Column type breakdown")
        type_counts = Counter(det_types.values())
        # ── Interactive type filter cards ─────────────────────────────────────
        if "profile_type_filter" not in st.session_state:
            st.session_state["profile_type_filter"] = "all"

        tk2 = st.columns(6)
        for i, (dtype, meta) in enumerate(COL_TYPE_META.items()):
            cnt = type_counts.get(dtype, 0)
            is_active = st.session_state["profile_type_filter"] == dtype
            border_style = f"4px solid {meta['color']}" if is_active else f"4px solid {meta['color']}44"
            bg_style     = f"{meta['color']}12" if is_active else "#ffffff"
            with tk2[i]:
                st.markdown(
                    f'<div style="background:{bg_style};border-radius:12px;padding:16px 14px;' +
                    f'border:1px solid #E2E8F0;border-top:{border_style};' +
                    f'box-shadow:0 1px 6px rgba(0,0,0,0.05);cursor:pointer;margin-bottom:4px;">' +
                    f'<p style="color:#5E6472;font-size:10px;font-weight:700;' +
                    f'letter-spacing:1px;text-transform:uppercase;margin:0 0 6px;">' +
                    f'{meta["icon"]} {meta["label"]}</p>' +
                    f'<p style="color:#23022E;font-size:24px;font-weight:800;margin:0;">{cnt}</p>' +
                    f'<p style="font-size:10px;color:{meta["color"]};margin:4px 0 0;font-weight:600;">' +
                    ('● Active' if is_active else '○ Click to filter') + "</p></div>",
                    unsafe_allow_html=True
                )
                if st.button(
                    "All" if is_active else "Filter",
                    key=f"type_btn_{dtype}",
                    help=f"Filter profile table to {meta['label']} columns",
                    use_container_width=True,
                ):
                    if is_active:
                        st.session_state["profile_type_filter"] = "all"
                    else:
                        st.session_state["profile_type_filter"] = dtype
                    st.rerun()

        active_filter = st.session_state["profile_type_filter"]
        if active_filter != "all":
            active_meta = COL_TYPE_META.get(active_filter, {})
            st.info(f"{active_meta.get('icon','')} Showing only **{active_meta.get('label',active_filter)}** columns — click the active card again to clear.")

        st.divider()
        st.subheader("Role mapping (used by pipeline)")
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown("**📅 Date columns**")
        for c in col_mapping.get("date_columns",[]): r1.markdown(f"• `{c}`")
        r2.markdown("**🔢 Numeric columns**")
        for c in col_mapping.get("numeric_columns",[]): r2.markdown(f"• `{c}`")
        r3.markdown("**🆔 ID columns**")
        for c in col_mapping.get("id_columns",[]): r3.markdown(f"• `{c}`")
        r4.markdown("**🏷️ Categorical columns**")
        for c in col_mapping.get("categorical_columns",[]): r4.markdown(f"• `{c}`")

        st.divider()
        st.divider()
        filter_label = COL_TYPE_META.get(active_filter, {}).get("label", "All") if active_filter != "all" else "All"
        st.subheader(f"Full column profile — {filter_label}" if active_filter != "all" else "Full column profile")
        profiles  = profile_dataframe(input_df, json.dumps(det_types))
        prof_df2  = pd.DataFrame(profiles)
        if active_filter != "all":
            prof_df2 = prof_df2[prof_df2["type"] == active_filter]
        disp_cols = ["column","type","null_pct","unique_count","unique_pct","sample","min","max","mean","zero_pct","neg_pct"]
        st.dataframe(
            prof_df2[[c for c in disp_cols if c in prof_df2.columns]],
            use_container_width=True, hide_index=True, height=450,
            column_config={
                "null_pct":  st.column_config.ProgressColumn("Null %",    min_value=0, max_value=100, format="%.1f%%"),
                "unique_pct":st.column_config.ProgressColumn("Unique %",  min_value=0, max_value=100, format="%.1f%%"),
                "zero_pct":  st.column_config.ProgressColumn("Zero %",    min_value=0, max_value=100, format="%.1f%%"),
                "neg_pct":   st.column_config.ProgressColumn("Negative %",min_value=0, max_value=100, format="%.1f%%"),
            }
        )

    # =========================================================================
    # TAB 2 — DASHBOARD
    # =========================================================================
    with tab3:
        # ── Story Banner ─────────────────────────────────────────────────────
        banner_html = build_story_banner(df_f, col_mapping)
        if banner_html:
            st.markdown(banner_html, unsafe_allow_html=True)
        # ── Quick NLQ search bar (Dashboard) ─────────────────────────────────
        with st.expander("💬 Ask your data — type a question to filter records instantly", expanded=False):
            qs_col1, qs_col2 = st.columns([5, 1])
            with qs_col1:
                quick_q = st.text_input(
                    "Quick query", key="dash_nlq",
                    placeholder='e.g. "Show critical records" · "amount above 50000" · "Find anomalies"',
                    label_visibility="collapsed",
                )
            with qs_col2:
                qs_clear = st.button("✕ Clear", key="dash_nlq_clear", use_container_width=True)
            if qs_clear:
                st.session_state["dash_nlq"] = ""
                quick_q = ""
            if quick_q and quick_q.strip():
                qs_result = parse_nlq(quick_q, df_f, col_mapping)
                st.caption(qs_result["summary"])
                if qs_result["intent"] == "aggregate" and qs_result["agg_df"] is not None:
                    st.dataframe(qs_result["agg_df"], use_container_width=True, hide_index=True)
                else:
                    qs_df = df_f[qs_result["mask"]]
                    if qs_result["sort_col"] and qs_result["sort_col"] in qs_df.columns:
                        qs_df = qs_df.sort_values(qs_result["sort_col"], ascending=qs_result["sort_asc"])
                    if qs_result["limit"]:
                        qs_df = qs_df.head(qs_result["limit"])
                    st.dataframe(qs_df.reset_index(drop=True), use_container_width=True, height=300)



        lowest_dim = min(
            [("Missing Data",     df_f["dq_score_completeness"].mean()),
             ("Format Accuracy",  df_f["dq_score_validity"].mean()),
             ("Value Accuracy",   df_f["dq_score_accuracy"].mean()),
             ("Data Consistency", df_f["dq_score_consistency"].mean()),
             ("Duplicate Check",  df_f["dq_score_uniqueness"].mean())],
            key=lambda x: x[1]
        )
        avg_score = df_f["dq_score"].mean()
        score_color = "#22c55e" if avg_score>=85 else "#f59e0b" if avg_score>=65 else "#ef4444"
        k1,k2,k3,k4,k5 = st.columns(5)
        cards = [
            ("Total Records",   f"{len(df_f):,}",                                 "in your dataset",    "#915466", "📁"),
            ("Quality Score",   f"{avg_score:.1f} / 100",                          "overall average",    score_color,"⭐"),
            ("Ready to Use",    f"{int((df_f['dq_score']>=99).sum()):,}",         "clean records",      "#22C55E", "✅"),
            ("Act Now",         f"{int((df_f['dq_severity']=='CRITICAL').sum()):,}","critical records", "#EF4444", "🔴"),
            ("Weakest Area",    f"{lowest_dim[1]:.0f} / 100",                      lowest_dim[0],        "#F59E0B", "⚠️"),
        ]
        for col_obj, (lbl, val, sub, clr, ico) in zip([k1,k2,k3,k4,k5], cards):
            with col_obj:
                st.markdown(kpi_card(lbl, val, sub, clr, ico), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Severity + Grade side by side ──────────────────────────────────────
        PASTEL_SEV = {"CLEAN":"#86efac","LOW":"#fde68a","MEDIUM":"#fed7aa",
                      "HIGH":"#fca5a5","CRITICAL":"#f87171"}
        c1, c2 = st.columns([3,2])
        with c1:
            st.markdown(section_head("Record Severity Breakdown",
                "👆 Click a bar to see those records below"), unsafe_allow_html=True)
            sev_df = df_f["dq_severity"].value_counts().reindex(SEV_ORDER, fill_value=0).reset_index()
            sev_df.columns = ["Severity","Count"]
            fig_sev = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                             color_discrete_map=PASTEL_SEV, text="Count",
                             custom_data=["Severity"])
            fig_sev.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>%{y:,} records<extra></extra>",
                textfont_size=13, textfont_color="#23022E",
            )
            fig_sev.update_layout(
                **_chart_layout(height=320),
                xaxis=dict(**_GRID_X, title=None),
                yaxis=dict(**_GRID_Y, title="Records"),
                bargap=0.35,
            )
            sev_event = st.plotly_chart(fig_sev, use_container_width=True,
                                        on_select="rerun", key="sev_bar")
        with c2:
            st.markdown(section_head("Quality Grade Mix"), unsafe_allow_html=True)
            grd_df = df_f["dq_grade"].value_counts().reindex(["A","B","C","D","F"], fill_value=0).reset_index()
            grd_df.columns = ["Grade","Count"]
            PASTEL_GRADE = {"A":"#86efac","B":"#6ee7b7","C":"#fde68a","D":"#fca5a5","F":"#f87171"}
            fig_grd = px.pie(grd_df, values="Count", names="Grade", hole=0.58,
                             color="Grade", color_discrete_map=PASTEL_GRADE)
            fig_grd.update_traces(
                textposition="outside", textinfo="label+percent",
                marker=dict(line=dict(color="#ffffff", width=3)),
                hovertemplate="Grade <b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>",
                pull=[0.04,0,0,0,0],
            )
            fig_grd.update_layout(
                **_chart_layout(height=320, showlegend=True,
                                legend=dict(orientation="h", y=-0.15,
                                            font=dict(color="#5E6472"))),
            )
            st.plotly_chart(fig_grd, use_container_width=True)

        # ── INTERACTIVE: click severity bar → filter records inline ────────────
        if sev_event and sev_event.selection and sev_event.selection.points:
            clicked_sev = sev_event.selection.points[0]["x"]
            filtered_click = df_f[df_f["dq_severity"] == clicked_sev]
            sev_label = {"CLEAN":"✅ Clean","LOW":"🟡 Low","MEDIUM":"🟠 Medium",
                         "HIGH":"🔶 High","CRITICAL":"🔴 Critical"}.get(clicked_sev, clicked_sev)
            st.markdown(f"""
            <div style="background:#F5EEF0;border:1.5px solid #C79192;border-radius:14px;
                        padding:16px 20px;margin:12px 0;">
              <p style="color:#915466;font-weight:700;font-size:15px;margin:0 0 4px;">
                {sev_label} — {len(filtered_click):,} records</p>
              <p style="color:#5E6472;font-size:12px;margin:0;">
                Click another bar to switch · Click same bar to deselect</p>
            </div>""", unsafe_allow_html=True)
            id_cols  = col_mapping.get("id_columns", [])
            num_cols = col_mapping.get("numeric_columns", [])[:3]
            cat_cols = col_mapping.get("categorical_columns", [])[:2]
            show_c   = [c for c in id_cols[:2]+num_cols+cat_cols+
                        ["dq_score","dq_grade","dq_severity","dq_issues"]
                        if c in filtered_click.columns][:10]
            st.dataframe(filtered_click[show_c].reset_index(drop=True),
                         use_container_width=True, height=360,
                         column_config={"dq_score": st.column_config.ProgressColumn(
                             "Quality Score", min_value=0, max_value=100, format="%d")})

        st.divider()

        # ── Score histogram ────────────────────────────────────────────────────
        st.markdown(section_head("Overall Quality Score Distribution",
            "See how your records cluster across the 0–100 quality range"), unsafe_allow_html=True)
        fig_hist = px.histogram(df_f, x="dq_score", nbins=25,
                                color_discrete_sequence=["#915466"],
                                labels={"dq_score":"Quality Score"})
        fig_hist.update_traces(
            marker_line_width=0,
            hovertemplate="Score %{x}<br>%{y:,} records<extra></extra>",
        )
        fig_hist.add_vline(x=df_f["dq_score"].mean(), line_dash="dash",
                           line_color="#915466", line_width=2,
                           annotation_text=f"Avg {df_f['dq_score'].mean():.0f}",
                           annotation_font_color="#915466", annotation_position="top right")
        fig_hist.update_layout(
            **_chart_layout(height=280),
            xaxis=dict(**_GRID_X, title="Quality Score (0–100)"),
            yaxis=dict(**_GRID_Y, title="Records"),
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # =========================================================================
    # TAB 3 — DIMENSION ANALYSIS
    # =========================================================================
    with tab4:
        dim_avgs = {
            "Missing Data":     df_f["dq_score_completeness"].mean(),
            "Format Accuracy":  df_f["dq_score_validity"].mean(),
            "Value Accuracy":   df_f["dq_score_accuracy"].mean(),
            "Data Consistency": df_f["dq_score_consistency"].mean(),
            "Duplicate Check":  df_f["dq_score_uniqueness"].mean(),
        }
        weights = {"Missing Data":"20%","Format Accuracy":"25%","Value Accuracy":"35%",
                   "Data Consistency":"15%","Duplicate Check":"5%"}
        DIM_COLORS_MAP = {
            "Missing Data":"#915466","Format Accuracy":"#0EA5E9",
            "Value Accuracy":"#10B981","Data Consistency":"#F59E0B","Duplicate Check":"#EF4444"
        }
        DIM_ICONS = {
            "Missing Data":"🕳️","Format Accuracy":"✅","Value Accuracy":"🎯",
            "Data Consistency":"🔗","Duplicate Check":"🪪"
        }
        d1,d2,d3,d4,d5 = st.columns(5)
        for col_obj, (dim, avg) in zip([d1,d2,d3,d4,d5], dim_avgs.items()):
            clr = DIM_COLORS_MAP.get(dim, "#915466")
            ico = DIM_ICONS.get(dim, "📊")
            status = "✅ Good" if avg>=80 else ("⚠️ Needs work" if avg>=60 else "🔴 Critical")
            sub = f"{weights[dim]} weight · {status}"
            with col_obj:
                st.markdown(kpi_card(dim, f"{avg:.1f}", sub, clr, ico), unsafe_allow_html=True)

        st.divider()
        dim_df = pd.DataFrame({
            "Dimension": list(dim_avgs.keys()),
            "Score":     [round(v,1) for v in dim_avgs.values()],
            "Weight":    list(weights.values()),
        }).sort_values("Score")
        PASTEL_DIMS = {
            "Missing Data":"#915466","Format Accuracy":"#2dd4bf",
            "Value Accuracy":"#34d399","Data Consistency":"#fbbf24","Duplicate Check":"#fb7185",
        }
        fig_dim = px.bar(dim_df, x="Score", y="Dimension", orientation="h",
                         color="Dimension", color_discrete_map=PASTEL_DIMS,
                         text="Score", range_x=[0,100], custom_data=["Weight"])
        fig_dim.update_traces(
            texttemplate="  %{x:.0f}",
            textposition="outside",
            textfont=dict(color="#23022E", size=13, family="Inter"),
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<br>Weight: %{customdata[0]}<extra></extra>",
        )
        fig_dim.update_layout(
            **_chart_layout(height=340),
            xaxis=dict(**_GRID_X, title="Score (0–100)"),
            yaxis=dict(**_NO_GRID_Y),
            bargap=0.35,
        )
        # Add a vertical "target" line at 80
        fig_dim.add_vline(x=80, line_dash="dot", line_color="#5E6472", line_width=1.5,
                          annotation_text="Target (80)", annotation_font_color="#5E6472",
                          annotation_position="top right")
        st.plotly_chart(fig_dim, use_container_width=True)

        st.subheader("Top failing checks")
        all_issues = []
        for iss in df_f["dq_issues"].dropna():
            if iss:
                for i in iss.split(" | "):
                    i = i.strip()
                    if i: all_issues.append(i)
        if all_issues:
            top = Counter(all_issues).most_common(10)
            iss_df = pd.DataFrame(top, columns=["Issue","Count"])
            iss_df["Dimension"] = iss_df["Issue"].str.extract(r"\[(\w+)\]")[0]
            iss_df["Label"]     = iss_df["Issue"].str.replace(r"\[\w+\] ","",regex=True)
            PASTEL_DIMS2 = {
                "Missing Data":"#915466","Format Accuracy":"#2dd4bf",
                "Value Accuracy":"#34d399","Data Consistency":"#fbbf24","Duplicate Check":"#fb7185",
            }
            fig_iss = px.bar(iss_df.sort_values("Count"), x="Count", y="Label",
                             orientation="h", color="Dimension",
                             color_discrete_map={d: PASTEL_DIMS2.get(d,"#5E6472")
                                                 for d in iss_df["Dimension"].unique()},
                             text="Count")
            fig_iss.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>%{x:,} records affected<extra></extra>",
            )
            fig_iss.update_layout(
                **_chart_layout(height=420, showlegend=True,
                                legend=dict(orientation="h", y=1.08,
                                            font=dict(color="#5E6472", size=11))),
                xaxis=dict(**_GRID_X, title="Records affected"),
                yaxis=dict(**_NO_GRID_Y),
                bargap=0.3,
            )
            st.plotly_chart(fig_iss, use_container_width=True)
        else:
            st.success("No issues found across all records.")

    # =========================================================================
    # TAB 4 — RECORD EXPLORER
    # =========================================================================
    with tab5:
        score_cols_all = ["dq_score","dq_score_completeness","dq_score_validity",
                          "dq_score_accuracy","dq_score_consistency","dq_score_uniqueness"]
        id_cols_avail  = col_mapping.get("id_columns",[])[:2]
        num_cols_avail = col_mapping.get("numeric_columns",[])[:2]
        cat_cols_avail = col_mapping.get("categorical_columns",[])[:2]
        show = (id_cols_avail + num_cols_avail + cat_cols_avail +
                ["dq_score","dq_grade","dq_severity"] + score_cols_all)
        show = [c for c in show if c in df_f.columns]
        # deduplicate while preserving order
        seen = set(); show = [c for c in show if not (c in seen or seen.add(c))]

        r1, r2 = st.columns(2)
        sort_col = r1.selectbox("Sort by", score_cols_all + num_cols_avail)
        sort_asc = r2.checkbox("Ascending (worst first)", value=True)

        table_df = df_f[show].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
        st.dataframe(
            table_df, use_container_width=True, height=500,
            column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%d")
                           for c in score_cols_all if c in table_df.columns},
        )

    # =========================================================================
    # TAB 5 — DEEP DIVE
    # =========================================================================
    with tab6:
        id_col = (col_mapping.get("id_columns") or list(df_f.columns))[0]
        if id_col in df_f.columns:
            options  = df_f[id_col].tolist()
            selected = st.selectbox(f"Select a record by `{id_col}`", options)
            row      = df_f[df_f[id_col] == selected].iloc[0]
        else:
            idx      = st.selectbox("Select row index", list(range(len(df_f))))
            row      = df_f.iloc[idx]

        dd1,dd2,dd3 = st.columns(3)
        dd1.metric("Overall DQ Score", f"{row['dq_score']} / 100")
        dd2.metric("Grade",    row["dq_grade"])
        dd3.metric("Severity", row["dq_severity"])

        dims = ["Completeness","Validity","Accuracy","Consistency","Uniqueness"]
        st.markdown("**Dimension breakdown**")
        dim_cols_ui = st.columns(5)
        for i, dim in enumerate(dims):
            val = float(row.get(f"dq_score_{dim.lower()}", 0))
            dim_cols_ui[i].metric(dim, f"{val:.0f}")
            dim_cols_ui[i].progress(int(val)/100)

        st.markdown("**Radar chart**")
        fig_radar = go.Figure(go.Scatterpolar(
            r=[float(row.get(f"dq_score_{d.lower()}", 0)) for d in dims] +
              [float(row.get("dq_score_completeness", 0))],
            theta=dims + [dims[0]],
            fill="toself", line_color="#378ADD",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            height=350, margin=dict(t=10,b=10), paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        if row.get("dq_issues",""):
            st.markdown("**Issues found:**")
            for iss in str(row["dq_issues"]).split(" | "):
                dim_tag = iss.split("]")[0].replace("[","") if "]" in iss else "Info"
                label   = iss.split("] ")[-1] if "] " in iss else iss
                color   = DIM_COLORS.get(dim_tag, "#888")
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:6px 12px;margin:4px 0;'
                    f'background:rgba(128,128,128,0.07);border-radius:4px;">'
                    f'<span style="font-size:11px;color:{color};font-weight:600;">{dim_tag.upper()}</span>'
                    f'<br><span style="font-size:14px;">{label}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No issues — this record passed all checks.")

    # =========================================================================
    # TAB 6 — STATISTICS
    # =========================================================================
    with tab7:
        st.subheader("Cleaning statistics")
        st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in clean_stats.items()]),
                     use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("Scoring statistics")
        st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in score_stats.items()]),
                     use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("Column role mapping used")
        role_rows = []
        for role, cols in col_mapping.items():
            if isinstance(cols, list):
                for c in cols:
                    role_rows.append({"Role": role, "Column": c,
                                      "Detected type": det_types.get(c,"—")})
        if role_rows:
            st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 7 — EXPORT
    # =========================================================================
    with tab8:
        st.subheader("⬇️ Download full scored dataset")
        st.caption(f"{len(df_f):,} records · all severity levels · includes all DQ columns")
        st.download_button(
            label="⬇️ Download full scored dataset (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name=f"dq_scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.subheader("🔎 Filter, preview & download by severity")
        fc1, fc2 = st.columns([2,1])
        with fc1:
            sev_dl = st.selectbox("Severity to preview", ["ALL"] + SEV_ORDER)
        with fc2:
            preview_cols_opt = st.multiselect(
                "Columns to show",
                options=list(df_f.columns),
                default=[c for c in (
                    col_mapping.get("id_columns",[])[:1] +
                    col_mapping.get("numeric_columns",[])[:2] +
                    col_mapping.get("categorical_columns",[])[:2] +
                    ["dq_score","dq_grade","dq_severity",
                     "isolation_score","iqr_outlier_count","is_anomaly","dq_issues"]
                ) if c in df_f.columns][:12],
            )

        anom_only  = st.checkbox("🚨 Flagged records only", value=False,
                                   help="Filter to rows where is_anomaly = True")
        export_df = df_f if sev_dl == "ALL" else df_f[df_f["dq_severity"] == sev_dl]
        if anom_only and "is_anomaly" in export_df.columns:
            export_df = export_df[export_df["is_anomaly"] == True]

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Records",      f"{len(export_df):,}")
        m2.metric("Avg DQ Score", f"{export_df['dq_score'].mean():.1f}" if len(export_df) else "—")
        m3.metric("Grades A/B",   int(export_df["dq_grade"].isin(["A","B"]).sum()) if len(export_df) else 0)
        m4.metric("Grades D/F",   int(export_df["dq_grade"].isin(["D","F"]).sum()) if len(export_df) else 0)

        if len(export_df) == 0:
            st.info("No records match the selected severity.")
        else:
            show_cols = [c for c in preview_cols_opt if c in export_df.columns]
            if not show_cols:
                show_cols = [c for c in ["dq_score","dq_severity"] if c in export_df.columns]
            score_cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%d")
                         for c in ["dq_score","dq_score_completeness","dq_score_validity",
                                   "dq_score_accuracy","dq_score_consistency","dq_score_uniqueness"]
                         if c in show_cols}
            st.dataframe(export_df[show_cols].reset_index(drop=True),
                         use_container_width=True, height=420, column_config=score_cfg)
            st.caption(f"Showing {len(export_df):,} {sev_dl} records")
            st.download_button(
                label=f"⬇️ Download {sev_dl} records ({len(export_df):,} rows)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"dq_{sev_dl.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # =========================================================================
    # TAB 8 — ANOMALY DETECTION  (v2.2 NEW)
    # =========================================================================
    with tab9:
        has_iso = "isolation_score"   in df.columns
        has_iqr = "iqr_outlier_count" in df.columns
        has_flag= "is_anomaly"        in df.columns

        if not has_iso:
            st.info("Run the pipeline to see anomaly detection results.")
        else:
            # ── KPIs ─────────────────────────────────────────────────────────
            total      = len(df_f)
            n_anom     = int(df_f["is_anomaly"].sum())            if has_flag else 0
            pct_anom   = round(100 * n_anom / total, 1)           if total else 0
            avg_iso    = round(df_f["isolation_score"].mean(), 1) if has_iso else 0
            max_iqr    = int(df_f["iqr_outlier_count"].max())     if has_iqr else 0

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("🚨 Flagged Records",          f"{n_anom:,}")
            k2.metric("% Anomalous",           f"{pct_anom}%")
            k3.metric("Avg Risk Level",   f"{avg_iso}")
            k4.metric("Most Unusual Columns",    f"{max_iqr}")

            st.divider()

            col_a, col_b = st.columns(2)

            # ── IsolationForest score distribution ───────────────────────────
            with col_a:
                st.subheader("🔍 How unusual are your records?")
                st.caption("Each bar shows how many records fall at that risk level. Bars on the right need your attention.")
                fig_iso = px.histogram(
                    df_f, x="isolation_score", nbins=40,
                    color_discrete_sequence=["#915466"],
                    labels={"isolation_score": "AI Risk Score"},
                )
                fig_iso.update_traces(
                    marker_line_width=0,
                    hovertemplate="Risk score ~%{x}<br>%{y:,} records<extra></extra>",
                )
                fig_iso.add_vline(x=70, line_dash="dash", line_color="#f87171", line_width=2,
                                  annotation_text="⚠️ Flag threshold (70)",
                                  annotation_font_color="#ef4444",
                                  annotation_position="top right")
                fig_iso.update_layout(
                    **_chart_layout(height=350),
                    xaxis=dict(**_GRID_X, title="AI Risk Score (0–100)"),
                    yaxis=dict(**_GRID_Y, title="Records"),
                    bargap=0.05,
                )
                st.plotly_chart(fig_iso, use_container_width=True)

            # ── IQR violations per numeric column ────────────────────────────
            with col_b:
                st.subheader("📊 Which columns have unusual values?")
                st.caption("Taller bars mean more records with out-of-range values in that column.")
                num_cols = col_mapping.get("numeric_columns", [])
                iqr_data = {}
                for col in num_cols:
                    if col not in df_f.columns:
                        continue
                    s = df_f[col].dropna()
                    if len(s) < 4:
                        continue
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr_val = q3 - q1
                    if iqr_val == 0:
                        continue
                    lo, hi = q1 - 1.5*iqr_val, q3 + 1.5*iqr_val
                    iqr_data[col] = int(((df_f[col] < lo) | (df_f[col] > hi)).sum())
                if iqr_data:
                    iqr_df = pd.DataFrame(
                        {"Column": list(iqr_data.keys()),
                         "Outlier Rows": list(iqr_data.values())}
                    ).sort_values("Outlier Rows", ascending=False)
                    fig_iqr = px.bar(
                        iqr_df, x="Column", y="Outlier Rows",
                        color="Outlier Rows",
                        color_continuous_scale=["#C79192","#915466","#915466","#fb7185"],
                        labels={"Outlier Rows": "Records with unusual values"},
                    )
                    fig_iqr.update_traces(
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>%{y:,} records out of range<extra></extra>",
                    )
                    fig_iqr.update_layout(
                        **_chart_layout(height=350, coloraxis_showscale=False),
                        xaxis=dict(**_GRID_X, title=None),
                        yaxis=dict(**_GRID_Y, title="Records out of normal range"),
                        bargap=0.35,
                    )
                    st.plotly_chart(fig_iqr, use_container_width=True)
                else:
                    st.info("No IQR outlier data — no numeric columns detected.")

            st.divider()

            # ── Top anomalous records ─────────────────────────────────────────
            st.subheader("🔴 Records flagged for your review")
            st.caption("Sorted by AI risk level. Review the top rows before using this data in reports or decisions.")
            top_n = st.slider("Records to review", 10, 200, 50, step=10)
            anom_df = df_f.sort_values("isolation_score", ascending=False).head(top_n)
            id_cols = col_mapping.get("id_columns", [])
            show_anom_cols = [c for c in (
                id_cols[:2] +
                col_mapping.get("numeric_columns", [])[:3] +
                ["isolation_score", "iqr_outlier_count", "is_anomaly",
                 "dq_score", "dq_severity"]
            ) if c in anom_df.columns]
            anom_cfg = {}
            if "isolation_score" in show_anom_cols:
                anom_cfg["isolation_score"] = st.column_config.ProgressColumn(
                    "AI Risk Score", min_value=0, max_value=100, format="%d")
            if "dq_score" in show_anom_cols:
                anom_cfg["dq_score"] = st.column_config.ProgressColumn(
                    "Data Quality Score", min_value=0, max_value=100, format="%d")
            st.dataframe(
                anom_df[show_anom_cols].reset_index(drop=True),
                use_container_width=True, height=450,
                column_config=anom_cfg,
            )
            st.download_button(
                label=f"⬇️ Download flagged records ({n_anom:,} rows)",
                data=df_f[df_f["is_anomaly"] == True].to_csv(index=False).encode("utf-8"),
                file_name=f"flagged_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # =========================================================================
    # TAB 9 — ASK YOUR DATA  (v2.4 NLQ)
    # =========================================================================
    with tab2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#23022E 0%,#3D1040 100%);
                    border-radius:14px;padding:20px 28px;margin-bottom:20px;
                    border:1px solid #915466;">
          <h2 style="color:#FDFFFF;margin:0 0 6px;font-size:20px;font-weight:800;">💬 Ask Your Data</h2>
          <p style="color:#C79192;margin:0;font-size:13px;line-height:1.6;">
            Type a plain-English question — no SQL, no filters, no code.
            The AI reads your intent and returns matching records instantly.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── State init ───────────────────────────────────────────────────────
        if "nlq_chip" not in st.session_state:
            st.session_state["nlq_chip"] = ""

        # ── Search bar FIRST — results appear immediately below ───────────────
        prefill = st.session_state.get("nlq_chip", "")
        nlq_col1, nlq_col2 = st.columns([6, 1])
        with nlq_col1:
            nlq_input = st.text_input(
                "Your question",
                value=prefill,
                placeholder='e.g. "Show critical records" · "Which vendors have issues?" · "Top 20 worst records"',
                label_visibility="collapsed",
                key="nlq_typed",
            )
            if nlq_input != prefill:
                st.session_state["nlq_chip"] = nlq_input
        with nlq_col2:
            nlq_run = st.button("🔍 Search", key="nlq_run", use_container_width=True, type="primary")

        # ── Execute query — RIGHT under search bar ────────────────────────────
        active_query = nlq_input.strip() if nlq_input else ""

        if active_query:
            result = parse_nlq(active_query, df_f, col_mapping)
            n_found = result["n"]

            # ── Result summary banner ────────────────────────────────────────
            if n_found == 0:
                st.warning(f'🔍 No records matched **"{active_query}"** — try different keywords or check the example queries above.')
            else:
                color = "#1D9E75" if n_found > 10 else "#BA7517"
                st.markdown(
                    f'<div style="background:#f0fdf4;border-left:4px solid {color};'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:16px;">'
                    f'<p style="margin:0;color:#23022E;font-size:14px;">{result["summary"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Aggregate view ────────────────────────────────────────────
                if result["intent"] == "aggregate" and result["agg_df"] is not None:
                    agg = result["agg_df"]
                    st.subheader("📊 Breakdown")

                    # Bar chart
                    fig_agg = px.bar(
                        agg.head(20), x="Value", y="Avg DQ Score",
                        color="Avg DQ Score",
                        color_continuous_scale=["#E24B4A","#EF9F27","#1D9E75"],
                        range_color=[0, 100],
                        text="Avg DQ Score",
                        labels={"Value": "", "Avg DQ Score": "Avg Quality Score"},
                    )
                    fig_agg.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                    fig_agg.update_layout(
                        **_chart_layout(height=380),
                        xaxis=dict(**_GRID_X, title=""),
                        yaxis=dict(**_GRID_Y, title="Avg Quality Score", range=[0, 110]),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_agg, use_container_width=True)

                    st.subheader("📋 Full Breakdown Table")
                    agg_disp = agg.copy()
                    agg_disp["Avg DQ Score"] = agg_disp["Avg DQ Score"].round(1)
                    st.dataframe(
                        agg_disp.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Avg DQ Score": st.column_config.ProgressColumn(
                                "Avg Quality Score", min_value=0, max_value=100, format="%.1f"
                            ),
                            "Critical Issues": st.column_config.NumberColumn(
                                "🔴 Critical", format="%d"
                            ),
                        },
                    )

                # ── Record Explorer View ─────────────────────────────────────
                else:
                    res_df = df_f[result["mask"]].copy()
                    if result["sort_col"] and result["sort_col"] in res_df.columns:
                        res_df = res_df.sort_values(result["sort_col"], ascending=result["sort_asc"])
                    if result["limit"]:
                        res_df = res_df.head(result["limit"])
                    res_df = res_df.reset_index(drop=True)

                    # ── KPI strip ─────────────────────────────────────────────
                    avg_q  = res_df["dq_score"].mean() if "dq_score" in res_df.columns and len(res_df) else 0
                    crit_n = int((res_df["dq_severity"] == "CRITICAL").sum()) if "dq_severity" in res_df.columns else 0
                    anom_n = int(res_df["is_anomaly"].sum()) if "is_anomaly" in res_df.columns else 0
                    pct    = round(100 * len(res_df) / len(df_f), 1) if len(df_f) else 0
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Records Found",    f"{len(res_df):,}",  f"{pct}% of total")
                    k2.metric("Avg Quality Score", f"{avg_q:.1f}",     delta_color="normal")
                    k3.metric("Critical Issues",  f"{crit_n:,}")
                    k4.metric("Anomalies Flagged",f"{anom_n:,}")

                    st.divider()

                    # ── Mini insight charts ───────────────────────────────────
                    ch1, ch2 = st.columns(2)

                    with ch1:
                        st.markdown("**Severity breakdown in results**")
                        if "dq_severity" in res_df.columns:
                            sev_counts = res_df["dq_severity"].value_counts().reindex(SEV_ORDER, fill_value=0)
                            fig_sev_r = px.bar(
                                x=sev_counts.index,
                                y=sev_counts.values,
                                color=sev_counts.index,
                                color_discrete_map=SEV_COLORS,
                                labels={"x": "", "y": "Records"},
                                text=sev_counts.values,
                            )
                            fig_sev_r.update_traces(textposition="outside", showlegend=False)
                            fig_sev_r.update_layout(
                                **_chart_layout(height=240),
                                xaxis=dict(**_GRID_X, title=""),
                                yaxis=dict(**_GRID_Y, title="Records"),
                                bargap=0.3, showlegend=False,
                            )
                            st.plotly_chart(fig_sev_r, use_container_width=True)
                        else:
                            st.info("No severity data.")

                    with ch2:
                        st.markdown("**Quality score distribution in results**")
                        if "dq_score" in res_df.columns:
                            fig_sc_r = px.histogram(
                                res_df, x="dq_score", nbins=20,
                                color_discrete_sequence=["#915466"],
                                labels={"dq_score": "Quality Score"},
                            )
                            fig_sc_r.update_traces(marker_line_width=0)
                            fig_sc_r.update_layout(
                                **_chart_layout(height=240),
                                xaxis=dict(**_GRID_X, title="Quality Score (0–100)"),
                                yaxis=dict(**_GRID_Y, title="Records"),
                                bargap=0.05,
                            )
                            st.plotly_chart(fig_sc_r, use_container_width=True)
                        else:
                            st.info("No score data.")

                    st.divider()

                    # ── Column picker ─────────────────────────────────────────
                    priority_cols = (
                        col_mapping.get("id_columns", [])[:2] +
                        col_mapping.get("date_columns", [])[:1] +
                        col_mapping.get("numeric_columns", [])[:3] +
                        col_mapping.get("categorical_columns", [])[:2]
                    )
                    score_cols = ["dq_score", "dq_severity", "dq_grade"]
                    anom_cols  = [c for c in ["isolation_score", "is_anomaly"] if c in res_df.columns]
                    default_cols = [c for c in (priority_cols + score_cols + anom_cols) if c in res_df.columns]
                    if not default_cols:
                        default_cols = list(res_df.columns[:10])

                    with st.expander("🎛️ Choose columns to display", expanded=False):
                        chosen_cols = st.multiselect(
                            "Columns",
                            options=list(res_df.columns),
                            default=default_cols,
                            label_visibility="collapsed",
                            key="nlq_col_picker",
                        )
                    show_cols = chosen_cols if chosen_cols else default_cols

                    # ── Sort control ──────────────────────────────────────────
                    sort_c1, sort_c2, sort_c3 = st.columns([3, 2, 1])
                    with sort_c1:
                        sort_by = st.selectbox(
                            "Sort by", options=show_cols,
                            index=show_cols.index("dq_score") if "dq_score" in show_cols else 0,
                            key="nlq_sort_col", label_visibility="visible"
                        )
                    with sort_c2:
                        sort_dir = st.radio("Order", ["⬆ Ascending","⬇ Descending"],
                                            index=1, horizontal=True,
                                            key="nlq_sort_dir", label_visibility="visible")
                    with sort_c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.caption(f"{len(res_df):,} rows")

                    sort_asc_ui = sort_dir == "⬆ Ascending"
                    if sort_by in res_df.columns:
                        res_df = res_df.sort_values(sort_by, ascending=sort_asc_ui)

                    # ── Main data table ───────────────────────────────────────
                    col_cfg = {}
                    if "dq_score" in show_cols:
                        col_cfg["dq_score"] = st.column_config.ProgressColumn(
                            "Quality Score", min_value=0, max_value=100, format="%d")
                    if "isolation_score" in show_cols:
                        col_cfg["isolation_score"] = st.column_config.ProgressColumn(
                            "AI Risk Score", min_value=0, max_value=100, format="%d")
                    if "dq_score_completeness" in show_cols:
                        col_cfg["dq_score_completeness"] = st.column_config.ProgressColumn(
                            "Missing Data", min_value=0, max_value=100, format="%d")
                    if "dq_score_validity" in show_cols:
                        col_cfg["dq_score_validity"] = st.column_config.ProgressColumn(
                            "Format Accuracy", min_value=0, max_value=100, format="%d")

                    st.dataframe(
                        res_df[show_cols].reset_index(drop=True),
                        use_container_width=True,
                        height=400,
                        column_config=col_cfg,
                    )

                    # ── Row drill-down ────────────────────────────────────────
                    st.divider()
                    with st.expander("🔍 Drill into a specific record", expanded=False):
                        if len(res_df) > 0:
                            row_idx = st.number_input(
                                "Row number (0-based)",
                                min_value=0,
                                max_value=max(0, len(res_df) - 1),
                                value=0,
                                step=1,
                                key="nlq_row_drill",
                            )
                            row = res_df.iloc[int(row_idx)]
                            # Split into data cols and DQ cols
                            dq_keys   = [c for c in row.index if c.startswith("dq_") or c in ("is_anomaly","isolation_score","iqr_outlier_count")]
                            data_keys = [c for c in row.index if c not in dq_keys]

                            drill_c1, drill_c2 = st.columns(2)
                            with drill_c1:
                                st.markdown("**📄 Record data**")
                                for col_name in data_keys:
                                    val = row[col_name]
                                    display = "_(empty)_" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
                                    st.markdown(f"**{col_name}:** {display}")
                            with drill_c2:
                                st.markdown("**🎯 Quality scores**")
                                for col_name in dq_keys:
                                    val = row[col_name]
                                    if isinstance(val, float) and not np.isnan(val) and "score" in col_name:
                                        color = "#1D9E75" if val >= 80 else "#EF9F27" if val >= 60 else "#E24B4A"
                                        label = col_name.replace("dq_score_","").replace("dq_","").replace("_"," ").title()
                                        st.markdown(
                                            f'<div style="display:flex;justify-content:space-between;'
                                            f'padding:5px 10px;border-radius:8px;background:#F5EEF0;margin:3px 0;">'
                                            f'<span style="color:#23022E;font-size:13px;">{label}</span>'
                                            f'<span style="color:{color};font-weight:800;font-size:13px;">{val:.1f}</span>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        display = str(val) if val is not None else "—"
                                        st.markdown(f"**{col_name}:** {display}")
                        else:
                            st.info("No records to drill into.")

                    # ── Download ──────────────────────────────────────────────
                    dl1, dl2 = st.columns([3, 1])
                    with dl1:
                        st.download_button(
                            label=f"⬇️ Download these {len(res_df):,} records as CSV",
                            data=res_df[show_cols].to_csv(index=False).encode("utf-8"),
                            file_name=f"nlq_results_{len(res_df)}rows.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with dl2:
                        st.caption(f"{len(res_df):,} of {len(df_f):,} total")

        else:
            # ── Empty state ───────────────────────────────────────────────────
            st.markdown("""
            <div style="text-align:center;padding:32px 24px 16px;color:#5E6472;">
              <p style="font-size:40px;margin:0 0 10px;">💬</p>
              <p style="font-size:15px;font-weight:700;color:#23022E;margin:0 0 6px;">Ask anything about your data</p>
              <p style="font-size:12px;margin:0;line-height:1.7;">
                Type a question above or pick an example below.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Example chips — collapsible, always below results ────────────────
        with st.expander("💡 Example queries — click any to run instantly", expanded=not bool(active_query)):
            st.markdown('<p style="color:#5E6472;font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;margin:0 0 8px;">Try these examples</p>', unsafe_allow_html=True)
            chip_rows = [NLQ_EXAMPLES[:4], NLQ_EXAMPLES[4:8], NLQ_EXAMPLES[8:]]
            for row in chip_rows:
                cols = st.columns(len(row))
                for i, example in enumerate(row):
                    with cols[i]:
                        if st.button(example, key=f"chip_{example}", use_container_width=True):
                            st.session_state["nlq_chip"] = example


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("DataQual AI · Universal Data Quality Platform · v2.4 · Built by Akshay")
