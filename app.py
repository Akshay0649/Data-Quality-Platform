"""
app.py — Finance Data Quality Streamlit Application
====================================================
Self-contained: generates synthetic finance data, runs the full
cleaning + 5-dimension DQ scoring pipeline in-memory, and presents
an interactive analytics dashboard.

Run locally:
    streamlit run app.py

Deploy: push to GitHub → connect at share.streamlit.io
"""

import random
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance DQ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared config (mirrors config.csv) ──────────────────────────────────────
CONFIG = {
    "duplicate_subset":           "vendor_name;invoice_number",
    "date_columns":               "invoice_date;due_date;payment_date",
    "amount_columns":             "amount;tax_amount;total_amount",
    "valid_currencies":           "USD;EUR;GBP;AUD;CAD;INR;SGD",
    "valid_statuses":             "PAID;PENDING;OVERDUE;CANCELLED;DRAFT",
    "outlier_zscore_threshold":   "3.0",
    "min_amount":                 "0.0",
    "max_amount":                 "10000000.0",
    "null_fill_string":           "UNKNOWN",
    "null_fill_numeric":          "0.0",
}

PLACEHOLDER_VALUES = {"UNKNOWN", "N/A", "NA", "NONE", ""}
CRITICAL_FIELDS    = ["vendor_name", "invoice_number", "amount",
                      "total_amount", "currency", "status"]

SEV_ORDER  = ["CLEAN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEV_COLORS = {"CLEAN": "#639922", "LOW": "#EF9F27",
              "MEDIUM": "#BA7517", "HIGH": "#D85A30", "CRITICAL": "#E24B4A"}
DIM_COLORS = {
    "Completeness": "#378ADD", "Validity": "#E24B4A",
    "Accuracy": "#3B6D11", "Consistency": "#1D9E75", "Uniqueness": "#534AB7",
}
DIM_WEIGHTS = {"completeness": 0.20, "validity": 0.25,
               "accuracy": 0.35, "consistency": 0.15, "uniqueness": 0.05}

# ────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ────────────────────────────────────────────────────────────────────────────
VENDORS = [
    "Accenture Ltd", "  KPMG Advisory  ", "Deloitte & Touche",
    "ernst & young", "PwC Services", "GARTNER INC", "Infosys BPO",
    "Wipro Technologies", "IBM Global  ", "Capgemini SE",
    "  Oracle Corp", "SAP AG", "Microsoft Azure", "AWS Finance",
    "Cognizant Tech", None, "N/A", "UNKNOWN VENDOR",
]
CURRENCIES = ["USD","EUR","GBP","AUD","INR","XYZ","ZZZ","usd","Eur",None]
STATUSES   = ["PAID","PENDING","OVERDUE","CANCELLED","DRAFT",
              "paid","  Pending  ","overdue","settled","VOID",None,""]

def _rnd_date(start, end):
    delta = (end - start).days
    d = start + timedelta(days=random.randint(0, delta))
    fmt = random.choice(["%Y-%m-%d"]*5 + ["%d/%m/%Y", "%m-%d-%Y", "bad-date", ""])
    return None if fmt == "" else d.strftime(fmt)

@st.cache_data(show_spinner=False)
def generate_and_score(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate dirty data, clean it, score it — returns final scored DataFrame."""
    random.seed(seed)
    np.random.seed(seed)

    start, end = date(2023, 1, 1), date(2025, 12, 31)
    rows = []
    for i in range(1, n_rows + 1):
        vendor  = random.choice(VENDORS)
        inv_num = f"INV-{random.randint(1000,9999)}"
        amount  = round(random.uniform(-500, 250_000), 2)
        if random.random() < 0.05:  amount = round(random.uniform(1_000_000, 15_000_000), 2)
        if random.random() < 0.08:  amount = -abs(amount)
        tax   = round(amount * random.uniform(0.05, 0.18), 2) if amount > 0 else None
        total = round(amount + (tax or 0), 2)
        rows.append({
            "invoice_id":     i,
            "invoice_number": inv_num,
            "vendor_name":    vendor,
            "invoice_date":   _rnd_date(start, end),
            "due_date":       _rnd_date(start, end),
            "payment_date":   _rnd_date(start, end) if random.random() > 0.35 else None,
            "amount":         amount if random.random() > 0.05 else None,
            "tax_amount":     tax,
            "total_amount":   total if random.random() > 0.05 else None,
            "currency":       random.choice(CURRENCIES),
            "status":         random.choice(STATUSES),
            "department":     random.choice(["Finance","IT","HR","Ops",None]),
            "cost_centre":    random.choice(["CC100","CC200","CC300","CC999",None]),
        })

    df = pd.DataFrame(rows)
    dupes = df.sample(20, random_state=7).copy()
    df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df["invoice_id"] = range(1, len(df)+1)

    df = _clean(df)
    df = _score(df)
    return df


# ────────────────────────────────────────────────────────────────────────────
# CLEANING  (inline — no file imports needed for cloud)
# ────────────────────────────────────────────────────────────────────────────
def _parse_list(key):
    return [v.strip() for v in CONFIG[key].split(";") if v.strip()]

def _clean(df):
    fill_str = CONFIG["null_fill_string"]
    fill_num = float(CONFIG["null_fill_numeric"])
    df = df.copy()
    df["had_nulls"] = df.isnull().any(axis=1).astype(int)
    for col in df.columns:
        if col == "had_nulls": continue
        if df[col].dtype == object:         df[col] = df[col].fillna(fill_str)
        elif pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(fill_num)

    subset = [c for c in _parse_list("duplicate_subset") if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    for col in [c for c in _parse_list("date_columns") if c in df.columns]:
        orig   = df[col].copy()
        parsed = pd.to_datetime(df[col], errors="coerce")
        err    = parsed.isna() & orig.notna() & (orig != "") & (orig != fill_str)
        df[f"{col}_parse_error"] = err.astype(int)
        df[col] = parsed.dt.strftime("%Y-%m-%d").where(~parsed.isna(), other=fill_str)

    mn, mx = float(CONFIG["min_amount"]), float(CONFIG["max_amount"])
    for col in [c for c in _parse_list("amount_columns") if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_negative"]  = (df[col] < 0).astype(int)
        df[f"{col}_below_min"] = (df[col] < mn).astype(int)
        df[f"{col}_above_max"] = (df[col] > mx).astype(int)

    valid_curr = set(_parse_list("valid_currencies"))
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.strip().str.upper()
        inv = ~df["currency"].isin(valid_curr)
        df["currency_invalid"] = inv.astype(int)
        df.loc[inv, "currency"] = fill_str

    if "vendor_name" in df.columns:
        df["vendor_name"] = (df["vendor_name"].astype(str).str.strip()
                              .str.replace(r"\s+", " ", regex=True).str.title())
        ph = df["vendor_name"].str.lower().isin({v.lower() for v in PLACEHOLDER_VALUES})
        df.loc[ph, "vendor_name"] = fill_str

    valid_st = set(_parse_list("valid_statuses"))
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.upper().replace({"": fill_str, "NAN": fill_str})
        inv_st = ~df["status"].isin(valid_st)
        df["status_invalid"] = inv_st.astype(int)
        df.loc[inv_st, "status"] = fill_str

    thr = float(CONFIG["outlier_zscore_threshold"])
    for col in [c for c in _parse_list("amount_columns") if c in df.columns]:
        num = pd.to_numeric(df[col], errors="coerce")
        std = num.std()
        z   = (num - num.mean()) / std if std and std > 0 else pd.Series(0.0, index=df.index)
        df[f"{col}_outlier"] = (z.abs() > thr).astype(int)

    return df


# ────────────────────────────────────────────────────────────────────────────
# SCORING
# ────────────────────────────────────────────────────────────────────────────
def _clamp(s): return s.clip(0, 100)

def _score(df):
    df = df.copy()

    # Completeness
    pf = [f for f in CRITICAL_FIELDS if f in df.columns]
    ppf = 100.0 / len(pf) if pf else 0
    sc = pd.Series(100.0, index=df.index)
    for f in pf:
        sc -= (df[f].isna() | df[f].astype(str).str.strip().str.upper().isin(PLACEHOLDER_VALUES)).astype(float) * ppf
    df["dq_score_completeness"] = _clamp(sc).round(1)

    # Validity
    sv = pd.Series(100.0, index=df.index)
    for col, pen in [("currency_invalid",60),("status_invalid",50),
                     ("invoice_date_parse_error",30),("due_date_parse_error",25),
                     ("payment_date_parse_error",20)]:
        if col in df.columns: sv -= df[col].fillna(0).astype(int) * pen
    df["dq_score_validity"] = _clamp(sv).round(1)

    # Accuracy
    sa = pd.Series(100.0, index=df.index)
    for col, pen in [("amount_negative",70),("total_amount_negative",70),
                     ("tax_amount_negative",40),("amount_above_max",40),
                     ("total_amount_above_max",40),("amount_outlier",20),
                     ("total_amount_outlier",20),("tax_amount_outlier",10)]:
        if col in df.columns: sa -= df[col].fillna(0).astype(int) * pen
    df["dq_score_accuracy"] = _clamp(sa).round(1)

    # Consistency
    sco = pd.Series(100.0, index=df.index)
    if "status" in df.columns and "payment_date" in df.columns:
        paid     = df["status"].astype(str).str.upper() == "PAID"
        no_date  = df["payment_date"].astype(str).str.strip().str.upper().isin(PLACEHOLDER_VALUES)
        sco -= (paid & no_date).astype(float) * 60
    if all(c in df.columns for c in ["amount","tax_amount","total_amount"]):
        amt = pd.to_numeric(df["amount"], errors="coerce")
        tax = pd.to_numeric(df["tax_amount"], errors="coerce").fillna(0)
        tot = pd.to_numeric(df["total_amount"], errors="coerce")
        mm  = amt.notna() & tot.notna() & ((tot - (amt+tax)).abs() > 0.02)
        sco -= mm.astype(float) * 60
    if "invoice_date" in df.columns and "due_date" in df.columns:
        inv = pd.to_datetime(df["invoice_date"], errors="coerce")
        due = pd.to_datetime(df["due_date"], errors="coerce")
        sco -= (inv.notna() & due.notna() & (due < inv)).astype(float) * 40
    df["dq_score_consistency"] = _clamp(sco).round(1)

    # Uniqueness
    su = pd.Series(100.0, index=df.index)
    if "invoice_number" in df.columns:
        su -= df["invoice_number"].duplicated(keep=False).astype(float) * 100
    df["dq_score_uniqueness"] = _clamp(su).round(1)

    # Overall weighted score
    overall = (
        df["dq_score_completeness"] * 0.20 +
        df["dq_score_validity"]     * 0.25 +
        df["dq_score_accuracy"]     * 0.35 +
        df["dq_score_consistency"]  * 0.15 +
        df["dq_score_uniqueness"]   * 0.05
    ).clip(0, 100).round(1)
    df["dq_score"] = overall.astype(int)

    grade_map = lambda s: "A" if s>=90 else "B" if s>=75 else "C" if s>=60 else "D" if s>=40 else "F"
    df["dq_grade"] = overall.apply(grade_map)

    def severity(row):
        if row.dq_score_accuracy < 30 or row.dq_score_consistency < 40: return "CRITICAL"
        if row.dq_score_accuracy < 60 or row.dq_score_validity < 40 or row.dq_score < 50: return "HIGH"
        if row.dq_score < 70 or min(row.dq_score_accuracy, row.dq_score_consistency,
                                    row.dq_score_validity, row.dq_score_completeness) < 60: return "MEDIUM"
        if row.dq_score < 85: return "LOW"
        return "CLEAN"
    df["dq_severity"] = df[["dq_score","dq_score_accuracy","dq_score_consistency",
                             "dq_score_validity","dq_score_completeness"]].apply(severity, axis=1)

    # Issues
    ISSUE_RULES = [
        ("amount_negative",          "[Accuracy] Negative invoice amount"),
        ("total_amount_negative",    "[Accuracy] Negative total amount"),
        ("tax_amount_negative",      "[Accuracy] Negative tax amount"),
        ("amount_above_max",         "[Accuracy] Amount exceeds max threshold"),
        ("total_amount_above_max",   "[Accuracy] Total exceeds max threshold"),
        ("amount_outlier",           "[Accuracy] Amount is a statistical outlier"),
        ("total_amount_outlier",     "[Accuracy] Total is a statistical outlier"),
        ("currency_invalid",         "[Validity] Invalid currency code"),
        ("status_invalid",           "[Validity] Invalid invoice status"),
        ("invoice_date_parse_error", "[Validity] Invoice date unparseable"),
        ("due_date_parse_error",     "[Validity] Due date unparseable"),
        ("payment_date_parse_error", "[Validity] Payment date unparseable"),
        ("had_nulls",                "[Completeness] Row had missing values"),
    ]
    issues = [""] * len(df)
    for col, label in ISSUE_RULES:
        if col not in df.columns: continue
        for i, v in enumerate(df[col].fillna(0).astype(int).tolist()):
            if v: issues[i] = (issues[i] + " | " + label) if issues[i] else label
    df["dq_issues"] = issues
    return df


# ────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ────────────────────────────────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Finance Data Quality Dashboard")
st.caption("Automated pipeline: data generation → 8 cleaning ops → 5-dimension DQ scoring")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    n_rows = st.slider("Sample size (rows)", 100, 1000, 300, 50)
    seed   = st.number_input("Random seed", value=42, step=1)

    if st.button("🔄 Regenerate data", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    st.subheader("🔍 Filters")

    sev_filter   = st.multiselect("Severity", SEV_ORDER, default=SEV_ORDER)
    grade_filter = st.multiselect("Grade", ["A","B","C","D","F"], default=["A","B","C","D","F"])
    score_range  = st.slider("DQ Score range", 0, 100, (0, 100))
    dim_filter   = st.selectbox("Highlight dimension", ["Overall"] + list(DIM_COLORS.keys()))

    st.divider()
    st.caption("Built by Akshay — Data Engineer")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Running pipeline: generating → cleaning → scoring…"):
    df_full = generate_and_score(n_rows=n_rows, seed=int(seed))

df = df_full[
    df_full["dq_severity"].isin(sev_filter) &
    df_full["dq_grade"].isin(grade_filter) &
    df_full["dq_score"].between(score_range[0], score_range[1])
].copy()

score_col = f"dq_score_{dim_filter.lower()}" if dim_filter != "Overall" else "dq_score"

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total records",   len(df), f"{len(df_full)-len(df)} filtered out")
k2.metric("Mean DQ score",   f"{df['dq_score'].mean():.1f}", "/100")
k3.metric("Perfect rows",    int((df["dq_score"] >= 99).sum()))
k4.metric("Critical records",int((df["dq_severity"]=="CRITICAL").sum()),
          delta_color="inverse")
k5.metric("Validity score",  f"{df['dq_score_validity'].mean():.1f}",
          "⚠️ lowest dimension" if df["dq_score_validity"].mean() < 60 else "✅ ok",
          delta_color="inverse")

st.divider()

# ── Row 1: Dimension scores + Severity donut ──────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Dimension scores")
    dim_avgs = {
        "Completeness": df["dq_score_completeness"].mean(),
        "Validity":     df["dq_score_validity"].mean(),
        "Accuracy":     df["dq_score_accuracy"].mean(),
        "Consistency":  df["dq_score_consistency"].mean(),
        "Uniqueness":   df["dq_score_uniqueness"].mean(),
    }
    dim_df = pd.DataFrame({
        "Dimension": list(dim_avgs.keys()),
        "Score":     [round(v, 1) for v in dim_avgs.values()],
        "Weight":    ["20%","25%","35%","15%","5%"],
        "Color":     [DIM_COLORS[d] for d in dim_avgs],
    }).sort_values("Score")

    fig_dim = px.bar(
        dim_df, x="Score", y="Dimension", orientation="h",
        color="Dimension", color_discrete_map=DIM_COLORS,
        text="Score", range_x=[0, 100],
        custom_data=["Weight"],
    )
    fig_dim.update_traces(
        texttemplate="%{x:.1f}", textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<br>Weight: %{customdata[0]}<extra></extra>"
    )
    fig_dim.update_layout(
        showlegend=False, height=300, margin=dict(l=0, r=40, t=10, b=10),
        xaxis_title="", yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_dim, use_container_width=True)

with col2:
    st.subheader("Severity breakdown")
    sev_counts = df["dq_severity"].value_counts().reindex(SEV_ORDER, fill_value=0).reset_index()
    sev_counts.columns = ["Severity", "Count"]

    fig_sev = px.pie(
        sev_counts, values="Count", names="Severity",
        color="Severity", color_discrete_map=SEV_COLORS,
        hole=0.55,
    )
    fig_sev.update_traces(
        textposition="outside", textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} records (%{percent})<extra></extra>"
    )
    fig_sev.update_layout(
        showlegend=False, height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sev, use_container_width=True)

# ── Row 2: Histogram + Top issues ─────────────────────────────────────────────
col3, col4 = st.columns([2, 3])

with col3:
    st.subheader("Score distribution")
    hist_df = df.copy()
    hist_df["band"] = (df[score_col] // 10 * 10).clip(0, 90).astype(int)
    hist_counts = hist_df.groupby("band").size().reset_index(name="count")
    hist_counts["label"] = hist_counts["band"].astype(str) + "–" + (hist_counts["band"]+9).astype(str)
    hist_counts["color"] = hist_counts["band"].apply(
        lambda b: "#E24B4A" if b<40 else "#D85A30" if b<60 else "#BA7517" if b<70 else "#378ADD" if b<80 else "#639922"
    )
    fig_hist = px.bar(hist_counts, x="label", y="count", color="color",
                      color_discrete_map="identity", text="count")
    fig_hist.update_traces(textposition="outside",
                           hovertemplate="<b>%{x}</b><br>%{y} records<extra></extra>")
    fig_hist.update_layout(
        showlegend=False, height=300, margin=dict(l=0, r=10, t=10, b=10),
        xaxis_title="Score band", yaxis_title="Records",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col4:
    st.subheader("Top failing checks")
    all_issues = []
    for issues in df["dq_issues"].dropna():
        if issues:
            for iss in issues.split(" | "):
                iss = iss.strip()
                if iss: all_issues.append(iss)
    top = Counter(all_issues).most_common(10)
    if top:
        issue_df = pd.DataFrame(top, columns=["Issue", "Count"])
        issue_df["Dimension"] = issue_df["Issue"].str.extract(r'\[(\w+)\]')[0]
        issue_df["ShortIssue"] = issue_df["Issue"].str.replace(r'\[\w+\] ', '', regex=True)
        fig_issues = px.bar(
            issue_df.sort_values("Count"), x="Count", y="ShortIssue",
            orientation="h", color="Dimension",
            color_discrete_map={d: DIM_COLORS.get(d, "#888") for d in issue_df["Dimension"].unique()},
            text="Count",
        )
        fig_issues.update_traces(textposition="outside",
                                 hovertemplate="<b>%{y}</b><br>%{x} records<extra></extra>")
        fig_issues.update_layout(
            height=300, margin=dict(l=0, r=40, t=10, b=10),
            xaxis_title="", yaxis_title="",
            legend_title="Dimension",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_issues, use_container_width=True)

st.divider()

# ── Interactive data table ─────────────────────────────────────────────────────
st.subheader("📋 Record explorer")

display_cols = ["invoice_number", "vendor_name", "amount", "currency",
                "status", "dq_score", "dq_grade", "dq_severity",
                "dq_score_completeness", "dq_score_validity",
                "dq_score_accuracy", "dq_score_consistency", "dq_score_uniqueness"]
display_cols = [c for c in display_cols if c in df.columns]

sort_col = st.selectbox("Sort by", ["dq_score", "dq_score_accuracy",
                                    "dq_score_validity", "dq_score_completeness",
                                    "dq_score_consistency", "amount"], index=0)
sort_asc = st.checkbox("Ascending (worst first)", value=True)

table_df = df[display_cols].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

st.dataframe(
    table_df,
    use_container_width=True,
    height=400,
    column_config={
        "dq_score": st.column_config.ProgressColumn(
            "dq_score", min_value=0, max_value=100, format="%d"),
        "dq_score_accuracy": st.column_config.ProgressColumn(
            "dq_score_accuracy", min_value=0, max_value=100, format="%d"),
        "dq_score_validity": st.column_config.ProgressColumn(
            "dq_score_validity", min_value=0, max_value=100, format="%d"),
        "dq_score_completeness": st.column_config.ProgressColumn(
            "dq_score_completeness", min_value=0, max_value=100, format="%d"),
        "dq_score_consistency": st.column_config.ProgressColumn(
            "dq_score_consistency", min_value=0, max_value=100, format="%d"),
    },
)

st.divider()

# ── Record deep-dive ──────────────────────────────────────────────────────────
st.subheader("🔬 Record deep-dive")

inv_options = df["invoice_number"].tolist()
selected    = st.selectbox("Select an invoice to inspect", inv_options)
row         = df[df["invoice_number"] == selected].iloc[0]

d1, d2, d3 = st.columns(3)
d1.metric("Overall DQ Score", f"{row['dq_score']} / 100")
d2.metric("Grade",    row["dq_grade"])
d3.metric("Severity", row["dq_severity"])

st.markdown("**Dimension breakdown**")
dims = ["Completeness","Validity","Accuracy","Consistency","Uniqueness"]
dim_cols = st.columns(5)
for i, dim in enumerate(dims):
    key = f"dq_score_{dim.lower()}"
    val = row.get(key, 0)
    col_obj = dim_cols[i]
    col_obj.metric(dim, f"{val:.0f}")
    col_obj.progress(int(val) / 100)

if row["dq_issues"]:
    st.markdown("**Issues found:**")
    for iss in row["dq_issues"].split(" | "):
        dim_tag = iss.split("]")[0].replace("[","") if "]" in iss else "Info"
        color   = DIM_COLORS.get(dim_tag, "#888")
        label   = iss.split("] ")[-1] if "] " in iss else iss
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:6px 12px;margin:4px 0;'
            f'background:rgba(128,128,128,0.05);border-radius:4px;">'
            f'<span style="font-size:11px;color:{color};font-weight:600;">{dim_tag.upper()}</span>'
            f'<br><span style="font-size:14px;">{label}</span></div>',
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No issues found — this record passed all checks.")

st.divider()

# ── Download ──────────────────────────────────────────────────────────────────
st.subheader("⬇️ Export")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered results as CSV",
    data=csv_data,
    file_name="invoices_dq_scored.csv",
    mime="text/csv",
    use_container_width=True,
)
