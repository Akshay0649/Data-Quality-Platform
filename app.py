"""
app.py — Finance Data Quality Streamlit Application
====================================================
Merged version:
  - Upload your own CSV  OR  generate synthetic demo data
  - Editable pipeline config in sidebar
  - Explicit Run Pipeline button
  - Tabbed results: Dashboard · Dimension Analysis · Record Explorer · Deep Dive · Stats · Export
  - Self-contained: no external file imports needed (works on Streamlit Cloud)

Run locally:  streamlit run app.py
Deploy:       push to GitHub → share.streamlit.io
"""

import random
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance DQ Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
SEV_ORDER  = ["CLEAN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEV_COLORS = {"CLEAN": "#639922", "LOW": "#EF9F27",
              "MEDIUM": "#BA7517", "HIGH": "#D85A30", "CRITICAL": "#E24B4A"}
DIM_COLORS = {
    "Completeness": "#378ADD", "Validity": "#E24B4A",
    "Accuracy":     "#3B6D11", "Consistency": "#1D9E75", "Uniqueness": "#534AB7",
}
PLACEHOLDER_VALUES = {"UNKNOWN", "N/A", "NA", "NONE", ""}
CRITICAL_FIELDS    = ["vendor_name", "invoice_number", "amount",
                      "total_amount", "currency", "status"]

VENDORS    = ["Accenture Ltd","  KPMG Advisory  ","Deloitte & Touche","ernst & young",
              "PwC Services","GARTNER INC","Infosys BPO","Wipro Technologies",
              "IBM Global  ","Capgemini SE","  Oracle Corp","SAP AG",
              "Microsoft Azure","AWS Finance","Cognizant Tech",None,"N/A","UNKNOWN VENDOR"]
CURRENCIES = ["USD","EUR","GBP","AUD","INR","XYZ","ZZZ","usd","Eur",None]
STATUSES   = ["PAID","PENDING","OVERDUE","CANCELLED","DRAFT",
              "paid","  Pending  ","overdue","settled","VOID",None,""]

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Config
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    st.caption("Adjust thresholds — applied when you run the pipeline")

    null_fill_str  = st.text_input("Null fill (text fields)", "UNKNOWN")
    null_fill_num  = st.number_input("Null fill (numeric fields)", value=0.0)
    min_amount     = st.number_input("Min valid amount", value=0.0)
    max_amount     = st.number_input("Max valid amount", value=10_000_000.0, step=100_000.0)
    outlier_z      = st.number_input("Outlier Z-score threshold", value=3.0, step=0.5)
    valid_curr     = st.text_input("Valid currencies (;-separated)", "USD;EUR;GBP;AUD;CAD;INR;SGD")
    valid_stat     = st.text_input("Valid statuses (;-separated)", "PAID;PENDING;OVERDUE;CANCELLED;DRAFT")
    dup_cols       = st.text_input("Duplicate key columns (;-separated)", "vendor_name;invoice_number")

    st.divider()
    st.caption("Built by Akshay — Data Engineer")

CONFIG = {
    "null_fill_string":           null_fill_str,
    "null_fill_numeric":          str(null_fill_num),
    "duplicate_subset":           dup_cols,
    "date_columns":               "invoice_date;due_date;payment_date",
    "amount_columns":             "amount;tax_amount;total_amount",
    "valid_currencies":           valid_curr,
    "valid_statuses":             valid_stat,
    "outlier_zscore_threshold":   str(outlier_z),
    "min_amount":                 str(min_amount),
    "max_amount":                 str(max_amount),
}

# ────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR
# ────────────────────────────────────────────────────────────────────────────
def _rnd_date(start, end):
    d   = start + timedelta(days=random.randint(0, (end - start).days))
    fmt = random.choice(["%Y-%m-%d"]*5 + ["%d/%m/%Y","%m-%d-%Y","bad-date",""])
    return None if fmt == "" else d.strftime(fmt)

@st.cache_data(show_spinner=False)
def generate_demo(n: int = 300, seed: int = 42) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    s, e = date(2023,1,1), date(2025,12,31)
    rows = []
    for i in range(1, n+1):
        vendor = random.choice(VENDORS)
        inv    = f"INV-{random.randint(1000,9999)}"
        amt    = round(random.uniform(-500, 250_000), 2)
        if random.random() < 0.05: amt = round(random.uniform(1_000_000,15_000_000), 2)
        if random.random() < 0.08: amt = -abs(amt)
        tax   = round(amt * random.uniform(0.05,0.18), 2) if amt > 0 else None
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
            "department":     random.choice(["Finance","IT","HR","Ops",None]),
            "cost_centre":    random.choice(["CC100","CC200","CC300","CC999",None]),
        })
    df = pd.DataFrame(rows)
    dupes = df.sample(20, random_state=7).copy()
    df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df["invoice_id"] = range(1, len(df)+1)
    return df

# ────────────────────────────────────────────────────────────────────────────
# CLEANING ENGINE
# ────────────────────────────────────────────────────────────────────────────
def _plist(key): return [v.strip() for v in CONFIG[key].split(";") if v.strip()]

def run_cleaning(df: pd.DataFrame) -> tuple:
    fs, fn = CONFIG["null_fill_string"], float(CONFIG["null_fill_numeric"])
    df = df.copy()
    df["had_nulls"] = df.isnull().any(axis=1).astype(int)
    nulls_before = int(df.isnull().sum().sum())
    for col in df.columns:
        if col == "had_nulls": continue
        if df[col].dtype == object:              df[col] = df[col].fillna(fs)
        elif pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(fn)
    nulls_filled = nulls_before - int(df.isnull().sum().sum())

    subset = [c for c in _plist("duplicate_subset") if c in df.columns]
    rows_before = len(df)
    df = df.drop_duplicates(subset=subset or None, keep="first").reset_index(drop=True)
    dupes_removed = rows_before - len(df)

    date_errors = 0
    for col in [c for c in _plist("date_columns") if c in df.columns]:
        orig   = df[col].copy()
        parsed = pd.to_datetime(df[col], errors="coerce")
        err    = parsed.isna() & orig.notna() & (~orig.astype(str).isin(["", fs]))
        df[f"{col}_parse_error"] = err.astype(int)
        date_errors += int(err.sum())
        df[col] = parsed.dt.strftime("%Y-%m-%d").where(~parsed.isna(), other=fs)

    mn, mx = float(CONFIG["min_amount"]), float(CONFIG["max_amount"])
    neg_flags = 0
    for col in [c for c in _plist("amount_columns") if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_negative"]  = (df[col] < 0).astype(int)
        df[f"{col}_below_min"] = (df[col] < mn).astype(int)
        df[f"{col}_above_max"] = (df[col] > mx).astype(int)
        neg_flags += int((df[col] < 0).sum())

    curr_invalid = 0
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.strip().str.upper()
        inv = ~df["currency"].isin(set(_plist("valid_currencies")))
        df["currency_invalid"] = inv.astype(int)
        df.loc[inv, "currency"] = fs
        curr_invalid = int(inv.sum())

    vendor_fixed = 0
    if "vendor_name" in df.columns:
        orig_v = df["vendor_name"].astype(str).copy()
        df["vendor_name"] = (df["vendor_name"].astype(str).str.strip()
                              .str.replace(r"\s+", " ", regex=True).str.title())
        ph = df["vendor_name"].str.lower().isin({v.lower() for v in PLACEHOLDER_VALUES})
        df.loc[ph, "vendor_name"] = fs
        vendor_fixed = int((df["vendor_name"] != orig_v).sum())

    stat_invalid = 0
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.upper().replace({"": fs,"NAN": fs})
        inv_s = ~df["status"].isin(set(_plist("valid_statuses")))
        df["status_invalid"] = inv_s.astype(int)
        df.loc[inv_s, "status"] = fs
        stat_invalid = int(inv_s.sum())

    thr = float(CONFIG["outlier_zscore_threshold"])
    outliers = 0
    for col in [c for c in _plist("amount_columns") if c in df.columns]:
        num = pd.to_numeric(df[col], errors="coerce")
        std = num.std()
        z   = (num - num.mean())/std if std and std > 0 else pd.Series(0.0, index=df.index)
        df[f"{col}_outlier"] = (z.abs() > thr).astype(int)
        outliers += int((z.abs() > thr).sum())

    stats = {
        "rows_input":           rows_before,
        "rows_output":          len(df),
        "duplicates_removed":   dupes_removed,
        "nulls_filled":         nulls_filled,
        "rows_with_nulls":      int(df["had_nulls"].sum()),
        "date_parse_errors":    date_errors,
        "negative_amount_flags":neg_flags,
        "currency_invalid":     curr_invalid,
        "vendor_names_fixed":   vendor_fixed,
        "statuses_fixed":       stat_invalid,
        "outliers_flagged":     outliers,
    }
    return df, stats

# ────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ────────────────────────────────────────────────────────────────────────────
def _clamp(s): return s.clip(0, 100)

def run_scoring(df: pd.DataFrame) -> tuple:
    df = df.copy()

    pf  = [f for f in CRITICAL_FIELDS if f in df.columns]
    ppf = 100.0 / len(pf) if pf else 0
    sc  = pd.Series(100.0, index=df.index)
    for f in pf:
        sc -= (df[f].isna() | df[f].astype(str).str.strip().str.upper()
               .isin(PLACEHOLDER_VALUES)).astype(float) * ppf
    df["dq_score_completeness"] = _clamp(sc).round(1)

    sv = pd.Series(100.0, index=df.index)
    for col, pen in [("currency_invalid",60),("status_invalid",50),
                     ("invoice_date_parse_error",30),("due_date_parse_error",25),
                     ("payment_date_parse_error",20)]:
        if col in df.columns: sv -= df[col].fillna(0).astype(int) * pen
    df["dq_score_validity"] = _clamp(sv).round(1)

    sa = pd.Series(100.0, index=df.index)
    for col, pen in [("amount_negative",70),("total_amount_negative",70),
                     ("tax_amount_negative",40),("amount_above_max",40),
                     ("total_amount_above_max",40),("amount_outlier",20),
                     ("total_amount_outlier",20),("tax_amount_outlier",10)]:
        if col in df.columns: sa -= df[col].fillna(0).astype(int) * pen
    df["dq_score_accuracy"] = _clamp(sa).round(1)

    sco = pd.Series(100.0, index=df.index)
    if "status" in df.columns and "payment_date" in df.columns:
        paid    = df["status"].astype(str).str.upper() == "PAID"
        no_date = df["payment_date"].astype(str).str.strip().str.upper().isin(PLACEHOLDER_VALUES)
        sco -= (paid & no_date).astype(float) * 60
    if all(c in df.columns for c in ["amount","tax_amount","total_amount"]):
        amt = pd.to_numeric(df["amount"], errors="coerce")
        tax = pd.to_numeric(df["tax_amount"], errors="coerce").fillna(0)
        tot = pd.to_numeric(df["total_amount"], errors="coerce")
        sco -= (amt.notna() & tot.notna() & ((tot-(amt+tax)).abs() > 0.02)).astype(float) * 60
    if "invoice_date" in df.columns and "due_date" in df.columns:
        inv = pd.to_datetime(df["invoice_date"], errors="coerce")
        due = pd.to_datetime(df["due_date"], errors="coerce")
        sco -= (inv.notna() & due.notna() & (due < inv)).astype(float) * 40
    df["dq_score_consistency"] = _clamp(sco).round(1)

    su = pd.Series(100.0, index=df.index)
    if "invoice_number" in df.columns:
        su -= df["invoice_number"].duplicated(keep=False).astype(float) * 100
    df["dq_score_uniqueness"] = _clamp(su).round(1)

    overall = (df["dq_score_completeness"]*0.20 + df["dq_score_validity"]*0.25 +
               df["dq_score_accuracy"]*0.35   + df["dq_score_consistency"]*0.15 +
               df["dq_score_uniqueness"]*0.05).clip(0,100).round(1)
    df["dq_score"] = overall.astype(int)
    df["dq_grade"] = overall.apply(lambda s: "A" if s>=90 else "B" if s>=75 else "C" if s>=60 else "D" if s>=40 else "F")

    def sev(r):
        if r.dq_score_accuracy<30 or r.dq_score_consistency<40: return "CRITICAL"
        if r.dq_score_accuracy<60 or r.dq_score_validity<40 or r.dq_score<50: return "HIGH"
        if r.dq_score<70 or min(r.dq_score_accuracy,r.dq_score_consistency,
                                r.dq_score_validity,r.dq_score_completeness)<60: return "MEDIUM"
        if r.dq_score<85: return "LOW"
        return "CLEAN"
    df["dq_severity"] = df[["dq_score","dq_score_accuracy","dq_score_consistency",
                             "dq_score_validity","dq_score_completeness"]].apply(sev, axis=1)

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

    grade_dist = df["dq_grade"].value_counts().to_dict()
    sev_dist   = df["dq_severity"].value_counts().to_dict()
    stats = {
        "mean_dq_score":        round(float(overall.mean()), 1),
        "median_dq_score":      round(float(overall.median()), 1),
        "perfect_rows":         int((overall >= 99).sum()),
        "avg_completeness":     round(float(df["dq_score_completeness"].mean()), 1),
        "avg_validity":         round(float(df["dq_score_validity"].mean()), 1),
        "avg_accuracy":         round(float(df["dq_score_accuracy"].mean()), 1),
        "avg_consistency":      round(float(df["dq_score_consistency"].mean()), 1),
        "avg_uniqueness":       round(float(df["dq_score_uniqueness"].mean()), 1),
        **{f"grade_{k}": v for k, v in grade_dist.items()},
        **{f"sev_{k}":   v for k, v in sev_dist.items()},
    }
    return df, stats

# ────────────────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────────────────
st.title("📊 Finance Data Quality Pipeline")
st.caption("Upload your CSV or generate demo data · configure thresholds · run pipeline · explore results")
st.divider()

# ────────────────────────────────────────────────────────────────────────────
# DATA SOURCE SECTION
# ────────────────────────────────────────────────────────────────────────────
st.header("📁 Data Source")

src_col1, src_col2 = st.columns([3, 1])

with src_col1:
    uploaded_file = st.file_uploader(
        "Upload your finance CSV",
        type=["csv"],
        help="Any CSV with finance data — the pipeline will auto-detect columns.",
    )

with src_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    n_demo  = st.number_input("Demo rows", 100, 1000, 300, 50)
    demo_seed = st.number_input("Seed", value=42, step=1)
    gen_btn = st.button("🎲 Generate demo data", use_container_width=True)

if gen_btn:
    st.session_state["input_df"] = generate_demo(int(n_demo), int(demo_seed))
    st.session_state.pop("scored_df", None)
    st.success(f"Demo dataset ready — {len(st.session_state['input_df'])} rows (with injected dirty data)")

if uploaded_file is not None:
    # Fingerprint the file so we only reload when the user picks a NEW file.
    # Without this, every sidebar-filter change triggers a rerun, re-reads the
    # same file, pops scored_df, and resets the view to an empty Dashboard.
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_uploaded_file_key") != file_key:
        try:
            st.session_state["input_df"] = pd.read_csv(uploaded_file)
            st.session_state.pop("scored_df", None)
            st.session_state["_uploaded_file_key"] = file_key
            st.success(f"Uploaded successfully — {len(st.session_state['input_df'])} rows × {len(st.session_state['input_df'].columns)} columns")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# ────────────────────────────────────────────────────────────────────────────
# RAW DATA PREVIEW
# ────────────────────────────────────────────────────────────────────────────
if "input_df" in st.session_state:
    input_df = st.session_state["input_df"]

    with st.expander("👁️ Raw data preview", expanded=True):
        p1, p2, p3 = st.columns(3)
        p1.metric("Rows", f"{len(input_df):,}")
        p2.metric("Columns", len(input_df.columns))
        p3.metric("Missing values", f"{int(input_df.isnull().sum().sum()):,}")
        st.dataframe(input_df.head(20), use_container_width=True)

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Cleaning data…"):
            cleaned_df, clean_stats = run_cleaning(input_df)
        with st.spinner("Scoring across 5 dimensions…"):
            scored_df, score_stats = run_scoring(cleaned_df)
        st.session_state["scored_df"]   = scored_df
        st.session_state["clean_stats"] = clean_stats
        st.session_state["score_stats"] = score_stats
        st.success(f"Pipeline complete — {len(scored_df):,} records scored.")

# ────────────────────────────────────────────────────────────────────────────
# RESULTS
# ────────────────────────────────────────────────────────────────────────────
if "scored_df" in st.session_state:
    df          = st.session_state["scored_df"]
    clean_stats = st.session_state["clean_stats"]
    score_stats = st.session_state["score_stats"]

    st.divider()
    st.header("✅ Pipeline Results")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📐 Dimension Analysis",
        "📋 Record Explorer",
        "🔬 Deep Dive",
        "📈 Statistics",
        "⬇️ Export",
    ])

    # ── Sidebar filters (applied across all tabs) ────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("🔍 Result Filters")
        sev_filter   = st.multiselect("Severity", SEV_ORDER, default=SEV_ORDER)
        grade_filter = st.multiselect("Grade", ["A","B","C","D","F"], default=["A","B","C","D","F"])
        score_range  = st.slider("DQ Score", 0, 100, (0, 100))

    df_f = df[
        df["dq_severity"].isin(sev_filter) &
        df["dq_grade"].isin(grade_filter) &
        df["dq_score"].between(score_range[0], score_range[1])
    ].copy()

    # =========================================================================
    # TAB 1 — DASHBOARD
    # =========================================================================
    with tab1:
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total records",    f"{len(df_f):,}")
        k2.metric("Mean DQ score",    f"{df_f['dq_score'].mean():.1f} / 100")
        k3.metric("Perfect rows",     int((df_f["dq_score"]>=99).sum()))
        k4.metric("Critical records", int((df_f["dq_severity"]=="CRITICAL").sum()))
        k5.metric("Validity score",   f"{df_f['dq_score_validity'].mean():.1f}",
                  "⚠️ lowest" if df_f["dq_score_validity"].mean() < 60 else "✅ ok",
                  delta_color="inverse")

        c1, c2 = st.columns([3,2])
        with c1:
            st.subheader("Severity breakdown")
            sev_df = df_f["dq_severity"].value_counts().reindex(SEV_ORDER, fill_value=0).reset_index()
            sev_df.columns = ["Severity","Count"]
            fig = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                         color_discrete_map=SEV_COLORS, text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=320,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Grade distribution")
            grd_df = df_f["dq_grade"].value_counts().reindex(["A","B","C","D","F"], fill_value=0).reset_index()
            grd_df.columns = ["Grade","Count"]
            fig2 = px.pie(grd_df, values="Count", names="Grade", hole=0.55,
                          color="Grade",
                          color_discrete_map={"A":"#639922","B":"#3B6D11","C":"#EF9F27","D":"#D85A30","F":"#E24B4A"})
            fig2.update_traces(textposition="outside", textinfo="label+percent")
            fig2.update_layout(showlegend=False, height=320,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Score distribution")
        fig3 = px.histogram(df_f, x="dq_score", nbins=20,
                            color_discrete_sequence=["#378ADD"],
                            labels={"dq_score":"DQ Score"})
        fig3.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # =========================================================================
    # TAB 2 — DIMENSION ANALYSIS
    # =========================================================================
    with tab2:
        dim_avgs = {
            "Completeness": df_f["dq_score_completeness"].mean(),
            "Validity":     df_f["dq_score_validity"].mean(),
            "Accuracy":     df_f["dq_score_accuracy"].mean(),
            "Consistency":  df_f["dq_score_consistency"].mean(),
            "Uniqueness":   df_f["dq_score_uniqueness"].mean(),
        }
        weights = {"Completeness":"20%","Validity":"25%","Accuracy":"35%",
                   "Consistency":"15%","Uniqueness":"5%"}

        d1,d2,d3,d4,d5 = st.columns(5)
        for col_obj, (dim, avg) in zip([d1,d2,d3,d4,d5], dim_avgs.items()):
            col_obj.metric(f"{dim} ({weights[dim]})", f"{avg:.1f}")
            col_obj.progress(int(avg)/100)

        st.divider()

        dim_df = pd.DataFrame({
            "Dimension": list(dim_avgs.keys()),
            "Score":     [round(v,1) for v in dim_avgs.values()],
            "Weight":    list(weights.values()),
        }).sort_values("Score")

        fig_dim = px.bar(dim_df, x="Score", y="Dimension", orientation="h",
                         color="Dimension", color_discrete_map=DIM_COLORS,
                         text="Score", range_x=[0,100], custom_data=["Weight"])
        fig_dim.update_traces(texttemplate="%{x:.1f}", textposition="outside",
                              hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<br>Weight: %{customdata[0]}<extra></extra>")
        fig_dim.update_layout(showlegend=False, height=320,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0,r=40,t=10,b=10))
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
            iss_df["Dimension"] = iss_df["Issue"].str.extract(r'\[(\w+)\]')[0]
            iss_df["Label"]     = iss_df["Issue"].str.replace(r'\[\w+\] ','',regex=True)
            fig_iss = px.bar(iss_df.sort_values("Count"), x="Count", y="Label",
                             orientation="h", color="Dimension",
                             color_discrete_map={d: DIM_COLORS.get(d,"#888") for d in iss_df["Dimension"].unique()},
                             text="Count")
            fig_iss.update_traces(textposition="outside")
            fig_iss.update_layout(height=380, showlegend=True,
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(l=0,r=40,t=10,b=10), legend_title="Dimension",
                                  xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_iss, use_container_width=True)

    # =========================================================================
    # TAB 3 — RECORD EXPLORER
    # =========================================================================
    with tab3:
        disp_cols = ["invoice_number","vendor_name","amount","currency","status",
                     "dq_score","dq_grade","dq_severity",
                     "dq_score_completeness","dq_score_validity",
                     "dq_score_accuracy","dq_score_consistency","dq_score_uniqueness"]
        disp_cols = [c for c in disp_cols if c in df_f.columns]

        r1, r2 = st.columns(2)
        sort_col = r1.selectbox("Sort by", ["dq_score","dq_score_accuracy","dq_score_validity",
                                            "dq_score_completeness","dq_score_consistency","amount"])
        sort_asc = r2.checkbox("Ascending (worst first)", value=True)

        table_df = df_f[disp_cols].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

        st.dataframe(
            table_df,
            use_container_width=True,
            height=500,
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

    # =========================================================================
    # TAB 4 — DEEP DIVE
    # =========================================================================
    with tab4:
        if "invoice_number" in df_f.columns:
            inv_options = df_f["invoice_number"].tolist()
            selected    = st.selectbox("Select an invoice to inspect", inv_options)
            row         = df_f[df_f["invoice_number"] == selected].iloc[0]
        else:
            idx_options = list(range(len(df_f)))
            selected    = st.selectbox("Select a row index", idx_options)
            row         = df_f.iloc[selected]

        dd1,dd2,dd3 = st.columns(3)
        dd1.metric("Overall DQ Score", f"{row['dq_score']} / 100")
        dd2.metric("Grade",    row["dq_grade"])
        dd3.metric("Severity", row["dq_severity"])

        st.markdown("**Dimension breakdown**")
        dims = ["Completeness","Validity","Accuracy","Consistency","Uniqueness"]
        dim_cols_ui = st.columns(5)
        for i, dim in enumerate(dims):
            key = f"dq_score_{dim.lower()}"
            val = float(row.get(key, 0))
            dim_cols_ui[i].metric(dim, f"{val:.0f}")
            dim_cols_ui[i].progress(int(val)/100)

        st.markdown("**Radar chart**")
        fig_radar = go.Figure(go.Scatterpolar(
            r=[float(row.get(f"dq_score_{d.lower()}", 0)) for d in dims] +
              [float(row.get("dq_score_completeness", 0))],
            theta=dims + [dims[0]],
            fill="toself",
            line_color="#378ADD",
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                                height=350, margin=dict(t=10,b=10),
                                paper_bgcolor="rgba(0,0,0,0)")
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
    # TAB 5 — STATISTICS
    # =========================================================================
    with tab5:
        st.subheader("Cleaning statistics")
        clean_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in clean_stats.items()])
        st.dataframe(clean_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Scoring statistics")
        score_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in score_stats.items()])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 6 — EXPORT
    # =========================================================================
    with tab6:
        st.subheader("Download results")
        fname = f"finance_dq_scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        st.download_button(
            label="⬇️ Download full scored dataset (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Filter & download by severity")
        sev_dl = st.selectbox("Severity to export", ["ALL"] + SEV_ORDER)
        export_df = df_f if sev_dl == "ALL" else df_f[df_f["dq_severity"] == sev_dl]
        st.caption(f"{len(export_df)} rows will be exported")
        st.download_button(
            label=f"⬇️ Download {sev_dl} records",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"finance_dq_{sev_dl.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Finance Data Quality Pipeline · 8 cleaning ops · 5-dimension DQ scoring · Built by Akshay")
