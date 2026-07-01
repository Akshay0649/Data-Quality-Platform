# CLAUDE.md — DataQual AI

Project context for Claude Code. Read automatically on launch.

## What this project is
**DataQual AI** — a universal, AI-powered data-quality platform.

> **One line:** *"Ask your data anything — and know whether you can trust the answer."*

- Point it at ANY CSV (or generate sample data). It auto-detects column types,
  cleans, scores across 5 quality dimensions, flags anomalies, and returns a
  plain-English **trust verdict**. No domain selection, no config, no SQL.
- Live deploy: https://finance-data-analyzer.streamlit.app/ (URL is legacy; product is universal).
- Repo: https://github.com/Akshay0649/Data-Quality-Platform (default branch `main`).
- Owner: Akshay (Data Engineer). Inspired by Anomalo (*trust in data*).
- **Identity (locked):** UNIVERSAL, not finance-specific. $0 / no external API / no
  backend is the DEFAULT out-of-the-box behaviour; the LLM query path is opt-in
  (default OFF). A backend arrives later (B6) as a clean upgrade, not a rewrite.

## The makeover (v3.0) — what changed and why
The app had accreted into a **3,485-line `app.py` with 14 tabs** (features bolted
on one PR at a time: `tab1, tab_sc, tab_ap, tab2 … tab_ms, tab9, tab10`). No
information architecture; five tabs all showed "quality numbers," two showed
"what to do," etc. The makeover **subtracts** that sprawl:

1. **Engine extracted** → `engine.py` (pure, Streamlit-free, unit-testable): the
   rule-based NLQ parser, column-type detection, role mapping, profiling, the
   8-step cleaning engine, the 5-dimension scoring engine, IsolationForest
   anomaly detection, the multi-domain demo generator, and shared severity/
   grade/colour constants. **No `st.` calls.** UI settings reach it via
   `engine.configure(**overrides)` (mutates `engine.CONFIG`, defaults in
   `engine.DEFAULT_CONFIG`).
2. **UI rebuilt** → new thin `app.py` (~430 lines): **14 tabs → 4 surfaces**, one
   story in four beats — **Ask → Quality → Verdict & Fix → Trends.**
3. **Branding unified** to universal (was fractured: README said "finance/SQL
   Server", CLAUDE.md said "universal", URL says "finance-data-analyzer").

`app_backup_prerefactor.py` is the pre-makeover monolith (local only — do NOT commit).

## Stack
Python / Streamlit. Plotly (charts), scikit-learn `IsolationForest`, pandas, NumPy.
**stdlib + pandas + streamlit + plotly + sklearn only — no new pip deps** (keeps $0).

## Core architecture
- **`engine.py`** — all logic (see above). Key entry points:
  `generate_demo(n, seed, domain)` · `detect_col_types(df)` ·
  `map_columns_to_roles(df, types)` · `profile_dataframe(df, types_json)` ·
  `run_cleaning(df, mapping) -> (df, stats)` ·
  `run_scoring(df, mapping) -> (df, stats)` · `run_anomaly_detection(df, mapping)` ·
  `parse_nlq(query, df, mapping) -> {"mask","summary","intent","sort_col","sort_asc","agg_df","limit","n"}`.
- **5-dimension DQ score** (`run_scoring`): completeness 20% · validity 25% ·
  accuracy 35% · consistency 15% · uniqueness 5%. Adds columns `dq_score`,
  `dq_grade` (A–F), `dq_severity` (CLEAN/LOW/MEDIUM/HIGH/CRITICAL), per-dimension
  `dq_score_*`, `dq_issues` (pipe-joined log), `is_anomaly`.
- Date columns are stored as `"%Y-%m-%d"` **strings** after cleaning — re-parse
  with `pd.to_datetime(..., errors="coerce")` inside NLQ.

## The 4 surfaces (`app.py`)
1. **💬 Ask** *(hero)* — NLQ box + clickable data-driven example chips. Every answer
   is **trust-aware**: a chip shows the data's grade + Autopilot verdict above the
   results. Router `smart_nlq()` uses the LLM planner when enabled, else `parse_nlq`.
2. **📊 Quality** — overall score + grade + clean/critical KPIs, 5-dimension bars,
   severity mix, missing-data chart, issue table. (The evidence behind the verdict.)
3. **🛸 Verdict & Fix** — `autopilot.run_autopilot()` verdict hero
   (APPROVE/REMEDIATE/QUARANTINE + confidence), policy-gate table, prioritised
   action plan, next steps, decision-report (MD/JSON) + cleaned-CSV downloads
   (CSV disabled while QUARANTINE), and the remediation log / suggested fixes.
4. **🕒 Trends** — save this run's manifest, upload past runs → quality-over-time
   line, cross-source comparison + schema drift.

## Add-on modules (all optional; app degrades gracefully if any fail to import)
- **`autopilot.py`** — decision engine. `run_autopilot(scored_df, score_stats,
  col_mapping, policy=None, dataset_name=...)` → `{"manifest", "decision"}`. The
  **inner `decision`** dict (verdict/verdict_icon/verdict_color/confidence/headline/
  reasons/checks/action_plan/next_steps/counts) is what `checks_df`,
  `action_plan_df`, `report_markdown`, `report_json` consume — pass the inner dict,
  not the wrapper. `evaluate()` is the pure rule (QUARANTINE on hard-gate breach,
  APPROVE if all gates pass, else REMEDIATE).
- **`remediation.py`** — `build_remediation_log(clean_stats, config)` →
  `remediation_log_df/_markdown/_json`; `suggested_actions_df(scored_df, mapping)`;
  `cleaned_business_df(scored_df, keep_scores=True)` (drops internal flag columns).
- **`run_manifest.py`** — portable JSON manifest (schema v2, carries `columns`).
  `build_manifest(scored_df, score_stats, col_mapping, dataset_name)` ·
  `manifest_to_bytes` · `suggested_filename` · `parse_manifest` ·
  `manifests_to_trend_df` · `compute_deltas` · `build_scorecard` · `evaluate_slas`.
  `StorageBackend` interface = the migration seam (JSON now → DuckDB → Postgres).
- **`multi_source.py`** — `schema_drift(cols_a, cols_b)` · `comparison_df(manifests)`
  · `drift_report` / `drift_summary_text`. Pure; works off manifests.
- **`llm_nlq.py`** — optional privacy-first LLM NLQ. NL → strict JSON query plan via
  any OpenAI-compatible free tier (`PROVIDER_PRESETS`; Groq default). Only column
  names/types are sent (sample values opt-in, default OFF). `CostGuard` caps calls.
  Transparent fallback to `parse_nlq` on disable/over-budget/error. Entry:
  `smart_query(query, df, mapping, parse_nlq, cfg, guard)`.

## Verification
- `python -m py_compile app.py engine.py` — clean.
- Engine smoke test: `import engine` → generate_demo → detect → clean → score →
  anomaly → parse_nlq all run headless (no Streamlit).
- Headless Streamlit **AppTest**: generate → analyse → 4 tabs → NLQ record &
  aggregate queries with **no exceptions** and no on-screen errors.
- `.streamlit/secrets.toml` is git-ignored — put `LLM_API_KEY` there.

## Known cleanup / follow-ups
- **`use_container_width`** is deprecated (removed after 2025-12-31) — sweep to
  `width='stretch'` across `app.py`.
- **Legacy finance/SQL CLI** (`pipeline.py`, `db_connector.py`, `config.csv`,
  `dq_dashboard.html`, `generate_sample_data.py`) are unused by the app and
  reinforce the old finance identity — candidates for removal.
- `README.md` still describes the old finance/SQL pipeline — needs a universal
  rewrite to match this file.
- The old per-feature selftests (`_selftest_phase2.py`, `_selftest_autopilot.py`)
  should be promoted into a real `tests/` suite that imports `engine` directly.

## Conventions
- Branches: kebab-case, intent-prefixed (`feat/`, `fix/`, `chore/`, `data/`).
- Commits: concise, imperative subject. PRs against `main`; reviewer is the owner.
- `git add <file>` explicitly — never `git add .` (keeps local backups out).

## Roadmap (post-makeover)
- **B5** analytics NLQ (time-series / comparisons via the LLM plan) ← suggested next.
- **B6** backend (Docker Postgres + scheduler + alerting + root-cause) — the moat.
  Autopilot's `evaluate()` is the decision rule it will call; the manifest schema
  is the migration contract, so this is an upgrade, not a rewrite.
