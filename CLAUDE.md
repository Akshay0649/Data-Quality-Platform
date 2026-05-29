# CLAUDE.md — DataQual AI

Project context for Claude Code. This file is read automatically on launch.

## What this project is
**DataQual AI** — a universal, AI-powered data quality platform.
- Main Streamlit app: `app.py` (**v2.7**, ~3,050 lines) + Phase-2 modules `llm_nlq.py`, `run_manifest.py` (no longer strictly single-file).
- Live deploy: https://finance-data-analyzer.streamlit.app/
- Repo: https://github.com/Akshay0649/Data-Quality-Platform (default branch `main`).
- Owner: Akshay (Data Engineer). Inspired by the Anomalo data-quality platform.
- Design identity (DEFAULT, opt-out): **zero external AI/API, no backend, single-session, stateless.** Owner has DECIDED to outgrow this into a full platform — keep it as the default for Phases 0–3; a backend arrives at Phase 5 (B6). The LLM NLQ path is opt-in (default OFF), so "zero external API" stays the out-of-the-box behaviour.

## Stack
Python / Streamlit. Plotly (charts), scikit-learn `IsolationForest` (anomaly detection), pandas, NumPy.
Supporting files: `cleaner.py`, `scorer.py`, `pipeline.py`, `db_connector.py`, `generate_sample_data.py`, `requirements.txt`, `.streamlit/config.toml` (Rosewood theme).

## Core architecture
- **5-dimension DQ score** (`run_scoring()`): completeness 20% · validity 25% · accuracy 35% · consistency 15% · uniqueness 5%.
- **Auto column-type detection** → `col_mapping` with `numeric_columns` / `categorical_columns` / `id_columns` / `date_columns`.
- Date columns are stored as `"%Y-%m-%d"` **strings** after cleaning — must re-parse with `pd.to_datetime(..., errors="coerce")` inside NLQ.
- **Rule-based NLQ engine** (`parse_nlq(query, df, col_mapping)`): regex + keyword, no LLM. Returns `{"mask","summary","intent","sort_col","sort_asc","agg_df","limit","n"}`.
- 8-step cleaning engine: `run_cleaning()`.

## v2.7 — SHIPPED (PR #1 open)
Multi-condition AND/OR NLQ · date-range NLQ · config explainer · version stamp. **PR #1** open against `main` (branch `feature/nlq-v2.7-multicondition`, `app.py` only). Live app lags until merged + Streamlit Cloud redeploys. Local backups `app_backup_v26_live.py` / `app_backup_v251.py` must NOT be committed (`git add app.py`, never `git add .`).

## v3.0 Phase 2 + B2 — LLM NLQ · persistence · executive scorecard (PR #2)
Two NEW modules, **stdlib + pandas only — ZERO new pip deps** (keeps the $0 promise):
- **`llm_nlq.py`** — optional, privacy-first LLM NLQ. NL → strict JSON "query plan" (`QUERY_PLAN_SCHEMA`) via any OpenAI-compatible **free tier** (Groq default; Gemini / OpenRouter presets, BYO key). Our own deterministic `validate_plan()` + `execute_plan()` run the plan and return the SAME dict contract as `parse_nlq`. **Privacy:** only column names/types are sent — never data rows (categorical sample values are opt-in, default OFF). `CostGuard` caps calls/chars per session. Transparent fallback to `parse_nlq` on disable / over-budget / error / invalid plan. Entry point: `smart_query()`.
- **`run_manifest.py`** — B1 keystone. `build_manifest()` → portable JSON (per-dimension scores, row/issue counts, grade+severity dist, `dataset_signature`, timestamp). `manifests_to_trend_df` / `compute_deltas` / `evaluate_slas` / **`build_scorecard` + `exec_summary_html` (B2)**. `StorageBackend` interface (JSONFileStore now; DuckDB/Postgres are a documented seam) = **the migration contract** (same schema all the way to the platform phase).

**app.py wiring:** sidebar "🤖 Smart Query" expander (opt-in, default OFF; provider/model/key/cost-cap/privacy toggle) → builds `LLM_CFG` + session `CostGuard`; `_smart_nlq()` router replaces BOTH `parse_nlq` UI call sites (Ask tab + Dashboard quick bar) and shows an engine badge; new **"🎯 Scorecard" tab (2nd tab, `tab_sc`)** = SLA target inputs → PASS/AT-RISK/FAIL status + per-area pass/fail grid + printable HTML export + optional prev-manifest deltas; new **"🕒 History" tab (`tab10`)** = download/upload manifests + trend lines + deltas. NOTE: Scorecard was inserted at tab-list position 2 as `tab_sc`, so `tab2`–`tab10` still map to Ask…History unchanged. `.streamlit/secrets.toml` is git-ignored — put `LLM_API_KEY` there.

Verified: `py_compile` clean; `_selftest_phase2.py` = **30/30** offline checks; headless Streamlit **AppTest** (streamlit 1.58 now installed locally) drives demo→score→**11 tabs**→NLQ→scorecard with **no exceptions**. `_selftest_phase2.py` is a local-only regression test (not committed; promote to `tests/` later).

## North-star architecture (Anomalo-grade target)
Layers: (1) connectors/ingest · (2) check engine [HAVE: `run_scoring` + IsolationForest] · (3) results store [BUILDING: `run_manifest` → DuckDB → Postgres] · (4) scheduler · (5) alerting · (6) UI [HAVE] · plus NLQ/analytics differentiator [`llm_nlq`]. **Current gap = layers 3/4/5.** B1 builds layer 3 in its no-backend form; B6 (Phase 5, Dockerised Postgres + FastAPI) builds 4/5.

## Conventions
- Branches: kebab-case, intent-prefixed (`feat/`, `fix/`, `chore/`, `data/`).
- Commits: concise, imperative subject.
- PRs against `main`; reviewer is the account owner.

## Roadmap (v3.0 vision — see DataQual_AI_Roadmap_v2.6_to_v3.md if present)
- **B1 Persistence/history** (keystone): downloadable JSON "run manifest" → trend lines, no backend. Fuller version needs SQLite/DuckDB.
- **B2** Business scorecard + per-column/source SLAs + trend deltas.
- **B3** Detection → remediation (suggested fixes, cleaned-file export, remediation log).
- **B4** Multi-file / multi-source + schema-drift detection.
- **B5** NLQ → analytics layer (time-series, comparisons); optional opt-in LLM-to-query translation behind a feature flag (keeps "no external API" as default).
- **B6** Scheduling/monitoring — the point where a backend becomes mandatory.

## Product decisions — RESOLVED (owner annotations on the v2.6→v3.0 roadmap)
1. **Outgrow single-file / no-backend?** YES — become a full platform. Keep $0 / no-backend / no-external-API as the DEFAULT for Phases 0–3; introduce a backend at Phase 5 (B6).
2. **Target user:** ANY business that wants to read, understand, and act on its data to drive decisions (outcome-led, not just data-engineer tooling).
3. **NLQ:** add an optional LLM layer behind a flag (built → `llm_nlq.py`); rule engine stays the default AND the fallback. Step-by-step, no rush.
4. **Scale / "proper hosted Docker DB"?** YES, eventually — Postgres via Docker at the platform phase. NOT yet: JSON manifest now → local DuckDB next → hosted Postgres later. The manifest schema is the migration contract, so this is a clean upgrade, not a rewrite.

## Suggested next steps (Phase 3+)
- ✅ Phase 2 committed → **PR #2** (`feat/llm-nlq-and-history`, base `main`). B2 scorecard bundled into the same PR.
- ✅ **B2 DONE** — per-DIMENSION SLAs + scorecard + printable HTML export + deltas. *Follow-up:* **per-COLUMN SLAs** (needs per-column dimension scores surfaced into the manifest first).
- **B3** remediation export (cleaned file + remediation log from `run_cleaning()`) ← suggested next.
- Then **B4** multi-file + schema drift (reuse `dataset_signature`), **B5** analytics NLQ (time-series/comparisons via the LLM plan), **B6** backend (Docker Postgres + scheduler + alerting).
- Housekeeping: PR #1 (v2.7) is redundant since `main` already has v2.7 — owner may close it or merge for the record.
