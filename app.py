"""
app.py — DataQual AI  (v3.0 — makeover)
=======================================
A universal, AI-powered data-quality platform.

    "Ask your data anything — and know whether you can trust the answer."

Point it at ANY CSV (or generate sample data). It auto-detects column types,
cleans, scores across 5 quality dimensions, flags anomalies, and hands you a
plain-English trust verdict. No domain selection, no config, no SQL.

Architecture
------------
All heavy logic lives in `engine.py` (pure, Streamlit-free, unit-testable).
This file is only the presentation layer, organised as ONE story in 4 beats:

    Ask  →  Quality  →  Verdict & Fix  →  Trends

Optional add-ons (all degrade gracefully if missing): `llm_nlq` (opt-in LLM
query planner), `autopilot` (trust verdict), `remediation` (fix log + cleaned
export), `run_manifest` / `multi_source` (history + cross-source comparison).

Run locally:  streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

import engine

# ── Optional add-on modules — never let a missing add-on break the core app ────
try:
    import llm_nlq
    import run_manifest
    import remediation
    import multi_source
    import autopilot
    _ADDONS_OK = True
    _ADDONS_ERR = ""
except Exception as _e:                      # pragma: no cover
    llm_nlq = run_manifest = remediation = multi_source = autopilot = None
    _ADDONS_OK = False
    _ADDONS_ERR = str(_e)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE + THEME
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataQual AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"]{background:#F5EEF0;padding:4px;border-radius:12px;gap:2px;}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:8px 18px;color:#5E6472;font-weight:600;font-size:13.5px;border:none;}
.stTabs [aria-selected="true"]{background:#915466;color:#FDFFFF;box-shadow:0 2px 8px rgba(145,84,102,0.3);}
[data-testid="stMetricLabel"]{color:#5E6472 !important;font-size:11px !important;}
[data-testid="stMetricValue"]{color:#23022E !important;font-weight:800 !important;}
</style>
""", unsafe_allow_html=True)

CHART_BG, CHART_PLOT, CHART_FONT = "#ffffff", "#fafafa", "#5E6472"
_GRID = dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, tickfont=dict(color="#5E6472"))


def _layout(**kw):
    base = dict(template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                font=dict(color=CHART_FONT, family="Inter, system-ui, sans-serif", size=12),
                margin=dict(t=32, b=32, l=16, r=32), showlegend=False)
    base.update(kw)
    return base


def kpi_card(label, value, subtitle="", color="#915466"):
    sub = f'<p style="color:#5E6472;font-size:11px;margin:5px 0 0;">{subtitle}</p>' if subtitle else ""
    return (f'<div style="background:#fff;border-radius:12px;padding:18px 20px;border:1px solid #E2E8F0;'
            f'border-top:4px solid {color};box-shadow:0 1px 6px rgba(0,0,0,0.05);height:100%;">'
            f'<p style="color:#5E6472;font-size:10px;font-weight:700;letter-spacing:1px;'
            f'text-transform:uppercase;margin:0 0 8px;">{label}</p>'
            f'<p style="color:#23022E;font-size:26px;font-weight:800;margin:0;line-height:1;">{value}</p>{sub}</div>')


# Presentation constants pulled from the engine (single source of truth)
SEV_ORDER   = engine.SEV_ORDER
SEV_COLORS  = engine.SEV_COLORS
COL_TYPE_META = engine.COL_TYPE_META
DEMO_DOMAINS  = engine.DEMO_DOMAINS
DIMENSIONS = [   # (display, score_stats key, scored-df column, colour, weight)
    ("Completeness", "avg_completeness", "dq_score_completeness", "#378ADD", 20),
    ("Validity",     "avg_validity",     "dq_score_validity",     "#E24B4A", 25),
    ("Accuracy",     "avg_accuracy",     "dq_score_accuracy",     "#3B6D11", 35),
    ("Consistency",  "avg_consistency",  "dq_score_consistency",  "#1D9E75", 15),
    ("Uniqueness",   "avg_uniqueness",   "dq_score_uniqueness",   "#534AB7", 5),
]


def _grade_color(grade: str) -> str:
    return {"A": "#1D9E75", "B": "#639922", "C": "#EF9F27", "D": "#D85A30", "F": "#E24B4A"}.get(grade, "#5E6472")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — brand · cleaning settings · optional LLM · (result filters appear later)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="background:#F5EEF0;border-radius:12px;padding:14px 16px;margin-bottom:12px;border:1px solid #C79192;">'
        '<p style="margin:0;font-size:16px;font-weight:800;color:#23022E;">🔬 DataQual AI</p>'
        '<p style="margin:4px 0 0;font-size:11.5px;color:#5E6472;line-height:1.5;">'
        'Ask your data anything — and know whether you can trust the answer.</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Cleaning & scoring settings", expanded=False):
        _fs  = st.text_input("Fill empty text with", "UNKNOWN")
        _fn  = st.number_input("Fill empty numbers with", value=0.0)
        _mn  = st.number_input("Minimum valid number", value=0.0)
        _mx  = st.number_input("Maximum valid number", value=10_000_000.0, step=100_000.0)
        _oz  = st.number_input("Outlier sensitivity (higher = less strict)", value=3.0, step=0.5)
        _dup = st.text_input("Duplicate key columns (;-separated, blank = all)", "")
    engine.configure(null_fill_string=_fs, null_fill_numeric=_fn, min_amount=_mn,
                     max_amount=_mx, outlier_zscore_threshold=_oz, duplicate_subset=_dup)

    # Optional LLM query planner (default OFF — keeps the $0 / no-API promise).
    LLM_CFG = {"enabled": False}
    _llm_cap = 20
    with st.expander("🤖 Smart Query (optional LLM) — off by default", expanded=False):
        if not _ADDONS_OK or llm_nlq is None:
            st.warning("LLM add-on not loaded: " + _ADDONS_ERR)
        else:
            st.caption("Only column names/types are ever sent — never your data rows. Falls back to the rule engine on any error.")
            _on = st.toggle("Enable LLM query planner", value=False)
            _provs = list(llm_nlq.PROVIDER_PRESETS.keys())
            _prov = st.selectbox("Provider (free tiers)", _provs, index=0)
            _preset = llm_nlq.PROVIDER_PRESETS[_prov]
            _secret = ""
            try:
                _secret = st.secrets.get("LLM_API_KEY", "")
            except Exception:
                _secret = ""
            _base = st.text_input("API base URL", value=_preset["base_url"])
            _model = st.text_input("Model", value=_preset["model"])
            _key = st.text_input("API key", value=_secret, type="password")
            if _preset.get("signup"):
                st.caption("Get a free key → " + _preset["signup"])
            _samples = st.checkbox("Also send a few sample category values (less private)", value=False)
            _llm_cap = st.number_input("Max LLM calls this session", 1, 200, 20)
            LLM_CFG = {"enabled": bool(_on), "base_url": _base, "model": _model,
                       "api_key": _key, "send_samples": bool(_samples), "timeout": 20}

    st.divider()
    st.caption("Universal · $0 · no backend by default · Built by Akshay")

if _ADDONS_OK and llm_nlq is not None:
    if "llm_guard" not in st.session_state or st.session_state.get("_llm_cap") != int(_llm_cap):
        st.session_state["llm_guard"] = llm_nlq.CostGuard(max_calls=int(_llm_cap))
        st.session_state["_llm_cap"] = int(_llm_cap)


def smart_nlq(query, frame, mapping):
    """Route a question through the LLM planner when enabled, else the rule engine.
    Returns (result_dict, engine_label, note). Never raises."""
    if LLM_CFG.get("enabled") and _ADDONS_OK and llm_nlq is not None:
        return llm_nlq.smart_query(query, frame, mapping, engine.parse_nlq,
                                   LLM_CFG, st.session_state.get("llm_guard"))
    return engine.parse_nlq(query, frame, mapping), "rules", ""


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER + DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="background:linear-gradient(135deg,#23022E 0%,#3D1040 100%);border-radius:16px;'
    'padding:22px 30px;margin-bottom:18px;border:1.5px solid #915466;box-shadow:0 4px 24px rgba(145,84,102,0.25);">'
    '<h1 style="color:#FDFFFF;margin:0;font-size:24px;font-weight:800;">🔬 DataQual AI</h1>'
    '<p style="color:#C79192;margin:6px 0 0;font-size:14px;">'
    'Ask your data anything — and know whether you can trust the answer.</p></div>',
    unsafe_allow_html=True,
)

st.subheader("📂 Start here — bring your data")
src1, src2 = st.columns([3, 1])
with src1:
    uploaded = st.file_uploader("Upload a CSV", type=["csv"],
                                help="Any CSV. Column types are detected automatically — no setup.")
with src2:
    st.markdown("<br>", unsafe_allow_html=True)
    demo_label = st.selectbox("…or try sample data", list(DEMO_DOMAINS.keys()), index=0)
    dcol = st.columns(2)
    n_demo = dcol[0].number_input("Rows", 100, 2000, 300, 50)
    seed = dcol[1].number_input("Seed", value=42, step=1)
    gen = st.button("🎲 Generate sample", width='stretch', type="primary")

if gen:
    st.session_state["input_df"] = engine.generate_demo(int(n_demo), int(seed), DEMO_DOMAINS[demo_label])
    for k in ("scored_df", "detected_types", "decision"):
        st.session_state.pop(k, None)
    st.session_state["_file_key"] = "demo"
    st.session_state["dataset_name"] = demo_label
    st.success(f"{demo_label} sample ready — {len(st.session_state['input_df']):,} rows with intentional issues to stress-test.")

if uploaded is not None:
    fkey = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_file_key") != fkey:
        try:
            st.session_state["input_df"] = pd.read_csv(uploaded)
            for k in ("scored_df", "detected_types", "decision"):
                st.session_state.pop(k, None)
            st.session_state["_file_key"] = fkey
            st.session_state["dataset_name"] = uploaded.name
            st.success(f"Uploaded {uploaded.name} — {len(st.session_state['input_df']):,} rows × {len(st.session_state['input_df'].columns)} cols")
        except Exception as ex:
            st.error(f"Could not read CSV: {ex}")


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE + RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
if "input_df" in st.session_state:
    input_df = st.session_state["input_df"]

    with st.expander("👁️ Raw preview & auto-detected profile", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.metric("Rows", f"{len(input_df):,}")
        p2.metric("Columns", f"{len(input_df.columns):,}")
        p3.metric("Cells", f"{input_df.size:,}")
        st.dataframe(input_df.head(15), width='stretch')

        if "detected_types" not in st.session_state:
            st.session_state["detected_types"] = engine.detect_col_types(input_df)
        det_types = st.session_state["detected_types"]
        col_mapping = engine.map_columns_to_roles(input_df, det_types)
        st.session_state["col_mapping"] = col_mapping

        prof = pd.DataFrame(engine.profile_dataframe(input_df, json.dumps(det_types)))
        show = [c for c in ["column", "type", "null_pct", "unique_count", "sample", "min", "max", "mean"] if c in prof.columns]
        st.dataframe(prof[show], width='stretch', hide_index=True,
                     column_config={"null_pct": st.column_config.ProgressColumn("Null %", min_value=0, max_value=100, format="%.1f%%")})
    col_mapping = st.session_state.get("col_mapping", engine.map_columns_to_roles(input_df, st.session_state["detected_types"]))

    if st.button("🔍 Analyse data", type="primary", width='stretch'):
        with st.spinner("Cleaning · scoring · detecting anomalies…"):
            cleaned, clean_stats = engine.run_cleaning(input_df, col_mapping)
            scored, score_stats = engine.run_scoring(cleaned, col_mapping)
            scored = engine.run_anomaly_detection(scored, col_mapping)
        st.session_state.update(scored_df=scored, clean_stats=clean_stats, score_stats=score_stats)
        # Precompute the trust verdict once so the Ask tab can show it cheaply.
        if _ADDONS_OK and autopilot is not None:
            try:
                st.session_state["decision"] = autopilot.run_autopilot(
                    scored, score_stats, col_mapping,
                    dataset_name=st.session_state.get("dataset_name", "dataset"))
            except Exception:
                st.session_state["decision"] = None
        st.success(f"Done — {len(scored):,} records scored across 5 dimensions.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS — 4 SURFACES:  Ask · Quality · Verdict & Fix · Trends
# ═══════════════════════════════════════════════════════════════════════════════
if "scored_df" in st.session_state:
    df = st.session_state["scored_df"]
    clean_stats = st.session_state.get("clean_stats", {})
    score_stats = st.session_state.get("score_stats", {})
    col_mapping = st.session_state.get("col_mapping", {})
    det_types = st.session_state.get("detected_types", {})
    decision = st.session_state.get("decision")

    overall = float(df["dq_score"].mean())
    grade = df["dq_grade"].mode().iloc[0] if "dq_grade" in df and len(df) else "—"

    # ── Sidebar result filters ────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("🎯 Filter results")
        sev_filter = st.multiselect("Severity", SEV_ORDER, default=SEV_ORDER)
        grade_filter = st.multiselect("Grade", ["A", "B", "C", "D", "F"], default=["A", "B", "C", "D", "F"])
        score_rng = st.slider("DQ score", 0, 100, (0, 100))
    df_f = df[df["dq_severity"].isin(sev_filter) & df["dq_grade"].isin(grade_filter)
              & df["dq_score"].between(*score_rng)].copy()

    tab_ask, tab_quality, tab_verdict, tab_trends = st.tabs(
        ["💬 Ask", "📊 Quality", "🛸 Verdict & Fix", "🕒 Trends"])

    # ─────────────────────────────────────────────────────────────────────────
    # SURFACE 1 — ASK  (the hero: NLQ + trust-aware answers)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_ask:
        # Trust chip — every answer is framed by how trustworthy the data is.
        v_txt = decision["decision"]["verdict"] if decision else "—"
        v_col = decision["decision"]["verdict_color"] if decision else "#5E6472"
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap;">'
            f'<span style="background:{_grade_color(grade)}22;color:{_grade_color(grade)};border:1px solid {_grade_color(grade)};'
            f'border-radius:20px;padding:5px 14px;font-size:12.5px;font-weight:700;">Data grade {grade} · {overall:.0f}/100</span>'
            f'<span style="background:{v_col}18;color:{v_col};border:1px solid {v_col};'
            f'border-radius:20px;padding:5px 14px;font-size:12.5px;font-weight:700;">Trust verdict: {v_txt}</span>'
            f'<span style="color:#5E6472;font-size:12px;">— answers below reflect this data. Treat with matching care.</span>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="background:linear-gradient(135deg,#23022E 0%,#3D1040 100%);border-radius:14px;'
            'padding:18px 24px;margin-bottom:16px;border:1px solid #915466;">'
            '<h2 style="color:#FDFFFF;margin:0 0 4px;font-size:19px;font-weight:800;">💬 Ask your data</h2>'
            '<p style="color:#C79192;margin:0;font-size:13px;">Plain English — no SQL, no filters. e.g. '
            '“show critical records”, “average score by status”, “top 20 worst rows”.</p></div>',
            unsafe_allow_html=True)

        if "nlq_q" not in st.session_state:
            st.session_state["nlq_q"] = ""
        qcol1, qcol2 = st.columns([6, 1])
        q = qcol1.text_input("Your question", value=st.session_state["nlq_q"],
                             placeholder='e.g. "Which categories have the most issues?"',
                             label_visibility="collapsed", key="nlq_typed")
        qcol2.button("🔍 Ask", width='stretch', type="primary")

        # Clickable example chips (data-driven)
        try:
            examples = engine._dynamic_examples(df_f, col_mapping)[:6]
        except Exception:
            examples = []
        if examples:
            ecols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                if ecols[i].button(ex, key=f"ex_{i}", width='stretch'):
                    st.session_state["nlq_q"] = ex
                    st.rerun()

        active = (q or "").strip()
        if active:
            result, eng_label, note = smart_nlq(active, df_f, col_mapping)
            badge = "🤖 LLM plan" if eng_label == "llm" else "⚙️ Rule engine"
            if note:
                badge += f" — {note}"
            st.caption(badge)

            if result["n"] == 0:
                st.warning(f'No records matched **"{active}"** — try different words or an example above.')
            else:
                st.markdown(
                    f'<div style="background:#f0fdf4;border-left:4px solid #1D9E75;border-radius:10px;'
                    f'padding:13px 18px;margin-bottom:14px;"><p style="margin:0;color:#23022E;font-size:14px;">'
                    f'{result["summary"]}</p></div>', unsafe_allow_html=True)

                if result["intent"] == "aggregate" and result.get("agg_df") is not None:
                    agg = result["agg_df"]
                    fig = px.bar(agg.head(20), x="Value", y="Avg DQ Score", color="Avg DQ Score",
                                 color_continuous_scale=["#E24B4A", "#EF9F27", "#1D9E75"], range_color=[0, 100],
                                 text="Avg DQ Score")
                    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                    fig.update_layout(**_layout(height=380), xaxis=dict(**_GRID, title=""),
                                      yaxis=dict(**_GRID, title="Avg quality", range=[0, 110]), coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')
                    st.dataframe(agg.reset_index(drop=True), width='stretch', hide_index=True)
                else:
                    res = df_f[result["mask"]].copy()
                    if result.get("sort_col") and result["sort_col"] in res.columns:
                        res = res.sort_values(result["sort_col"], ascending=result["sort_asc"])
                    if result.get("limit"):
                        res = res.head(result["limit"])
                    res = res.reset_index(drop=True)
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Records found", f"{len(res):,}", f"{round(100*len(res)/max(len(df_f),1),1)}% of view")
                    k2.metric("Avg quality", f"{res['dq_score'].mean():.1f}" if len(res) else "—")
                    k3.metric("Critical", f"{int((res.get('dq_severity')=='CRITICAL').sum()):,}")
                    k4.metric("Anomalies", f"{int(res.get('is_anomaly', pd.Series(dtype=bool)).sum()):,}")
                    st.dataframe(res, width='stretch', hide_index=True)
        else:
            st.info("Type a question above, or click an example, to explore your data.")

    # ─────────────────────────────────────────────────────────────────────────
    # SURFACE 2 — QUALITY  (the evidence: score · dimensions · issues · profile)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_quality:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("Overall quality", f"{overall:.0f}", "out of 100", _grade_color(grade)), unsafe_allow_html=True)
        c2.markdown(kpi_card("Grade", grade, "most common", _grade_color(grade)), unsafe_allow_html=True)
        clean_n = int((df["dq_severity"] == "CLEAN").sum())
        c3.markdown(kpi_card("Clean records", f"{clean_n:,}", f"{round(100*clean_n/max(len(df),1))}% ready to use", "#1D9E75"), unsafe_allow_html=True)
        crit_n = int((df["dq_severity"] == "CRITICAL").sum())
        c4.markdown(kpi_card("Critical records", f"{crit_n:,}", "need attention now", "#E24B4A"), unsafe_allow_html=True)

        st.divider()
        st.subheader("Quality by dimension")
        dim_rows = [{"Dimension": f"{name} ({w}%)", "Score": float(score_stats.get(key, df[col].mean())), "c": color}
                    for name, key, col, color, w in DIMENSIONS]
        dim_df = pd.DataFrame(dim_rows)
        fig = px.bar(dim_df, x="Score", y="Dimension", orientation="h", text="Score",
                     color="Dimension", color_discrete_sequence=dim_df["c"].tolist())
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(**_layout(height=300), xaxis=dict(**_GRID, range=[0, 110], title="Score /100"),
                          yaxis=dict(showgrid=False, title=""))
        st.plotly_chart(fig, width='stretch')

        st.divider()
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Severity mix")
            sev_counts = df["dq_severity"].value_counts().reindex(SEV_ORDER).fillna(0)
            figs = px.bar(x=sev_counts.index, y=sev_counts.values, color=sev_counts.index,
                          color_discrete_map=SEV_COLORS, text=sev_counts.values)
            figs.update_layout(**_layout(height=300), xaxis=dict(title=""), yaxis=dict(**_GRID, title="Records"))
            st.plotly_chart(figs, width='stretch')
        with cc2:
            st.subheader("Missing data by column")
            prof = pd.DataFrame(engine.profile_dataframe(input_df, json.dumps(det_types)))
            nulls = prof[prof["null_pct"] > 0].sort_values("null_pct", ascending=False) if "null_pct" in prof else pd.DataFrame()
            if not nulls.empty:
                fign = px.bar(nulls, x="column", y="null_pct", color="null_pct",
                              color_continuous_scale=["#86efac", "#fde68a", "#f87171"], range_color=[0, 100])
                fign.update_layout(**_layout(height=300, coloraxis_showscale=False), xaxis=dict(title=""),
                                   yaxis=dict(**_GRID, title="% missing"))
                st.plotly_chart(fign, width='stretch')
            else:
                st.success("No missing values detected. 🎉")

        st.divider()
        st.subheader("Records with issues")
        issue_view = df_f[df_f["dq_severity"] != "CLEAN"][
            [c for c in ["dq_score", "dq_grade", "dq_severity", "dq_issues"] if c in df_f.columns]]
        st.dataframe(issue_view.reset_index(drop=True), width='stretch', hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SURFACE 3 — VERDICT & FIX  (the decision + the cleaned file out)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_verdict:
        if not (_ADDONS_OK and autopilot is not None):
            st.warning("The Autopilot add-on isn't loaded — verdict unavailable. " + _ADDONS_ERR)
        else:
            if decision is None:
                decision = autopilot.run_autopilot(df, score_stats, col_mapping,
                                                   dataset_name=st.session_state.get("dataset_name", "dataset"))
                st.session_state["decision"] = decision
            d = decision["decision"]
            st.markdown(
                f'<div style="background:{d["verdict_color"]}12;border:2px solid {d["verdict_color"]};'
                f'border-radius:16px;padding:22px 26px;margin-bottom:16px;">'
                f'<p style="margin:0;font-size:30px;font-weight:800;color:{d["verdict_color"]};">'
                f'{d["verdict_icon"]} {d["verdict"]}</p>'
                f'<p style="margin:8px 0 0;font-size:14px;color:#23022E;">{d["headline"]}</p>'
                f'<p style="margin:6px 0 0;font-size:12.5px;color:#5E6472;">Confidence: {d["confidence"]}%</p></div>',
                unsafe_allow_html=True)

            if d.get("reasons"):
                st.markdown("**Why:** " + "  ·  ".join(d["reasons"]))

            st.subheader("Policy gates")
            st.dataframe(autopilot.checks_df(d), width='stretch', hide_index=True)

            st.subheader("Prioritised action plan")
            ap = autopilot.action_plan_df(d)
            if len(ap):
                st.dataframe(ap, width='stretch', hide_index=True)
            else:
                st.success("No actions required — data passed every gate.")

            if d.get("next_steps"):
                st.markdown("**Next steps:**")
                for s in d["next_steps"]:
                    st.markdown(f"- {s}")

            st.divider()
            st.subheader("Fix & export")
            quarantined = d["verdict"] == "QUARANTINE"
            fcols = st.columns(3)
            fcols[0].download_button("⬇️ Decision report (MD)", autopilot.report_markdown(d),
                                     "trust_verdict.md", width='stretch')
            fcols[1].download_button("⬇️ Decision report (JSON)", autopilot.report_json(d),
                                     "trust_verdict.json", width='stretch')
            if remediation is not None:
                clean_csv = remediation.cleaned_business_df(df, keep_scores=True).to_csv(index=False)
                fcols[2].download_button("⬇️ Cleaned data (CSV)", clean_csv, "cleaned_data.csv",
                                         width='stretch', disabled=quarantined,
                                         help="Disabled while QUARANTINE — resolve the verdict first." if quarantined else "")

            if remediation is not None:
                with st.expander("🛠️ Remediation log & suggested fixes", expanded=False):
                    log = remediation.build_remediation_log(clean_stats, engine.CONFIG)
                    st.dataframe(remediation.remediation_log_df(log), width='stretch', hide_index=True)
                    st.markdown("**Suggested fixes (prioritised):**")
                    st.dataframe(remediation.suggested_actions_df(df, col_mapping),
                                 width='stretch', hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SURFACE 4 — TRENDS  (compare over time & across sources)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_trends:
        if not (_ADDONS_OK and run_manifest is not None):
            st.warning("The history add-on isn't loaded. " + _ADDONS_ERR)
        else:
            manifest = (decision or {}).get("manifest") if decision else None
            if manifest is None:
                manifest = run_manifest.build_manifest(df, score_stats, col_mapping,
                                                       dataset_name=st.session_state.get("dataset_name", "dataset"))
            st.subheader("This run")
            st.caption("Download this run's manifest, then re-upload past runs to see trends & drift.")
            st.download_button("⬇️ Save this run (manifest JSON)", run_manifest.manifest_to_bytes(manifest),
                               run_manifest.suggested_filename(manifest), width='stretch')

            ups = st.file_uploader("Upload past run manifests to compare", type=["json"], accept_multiple_files=True)
            manifests = [manifest]
            for f in ups or []:
                try:
                    manifests.append(run_manifest.parse_manifest(json.load(f)))
                except Exception as ex:
                    st.error(f"Skipped {f.name}: {ex}")

            if len(manifests) > 1:
                trend = run_manifest.manifests_to_trend_df(manifests)
                st.subheader("Quality over time")
                figt = px.line(trend, x="timestamp", y="overall", markers=True)
                figt.update_layout(**_layout(height=320), xaxis=dict(**_GRID, title=""),
                                   yaxis=dict(**_GRID, title="Overall score", range=[0, 105]))
                st.plotly_chart(figt, width='stretch')

                if multi_source is not None:
                    st.subheader("Cross-source comparison")
                    st.dataframe(multi_source.comparison_df(manifests), width='stretch', hide_index=True)
                    st.subheader("Schema drift")
                    for entry in multi_source.drift_report(manifests):
                        st.markdown("- " + multi_source.drift_summary_text(entry))
            else:
                st.info("Only this run so far. Save it, run another dataset, then upload the saved file(s) here to compare.")

else:
    st.info("👆 Upload a CSV or generate sample data, then click **Analyse data** to get your trust verdict.")
