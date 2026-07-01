# 🔬 DataQual AI

> **Ask your data anything — and know whether you can trust the answer.**

A universal, AI-powered data-quality platform. Point it at **any CSV**, and it
auto-detects column types, cleans, scores every record across **5 weighted
quality dimensions**, flags anomalies, and hands you a plain-English **trust
verdict** — APPROVE, REMEDIATE, or QUARANTINE. No domain setup, no config, no SQL.

**Live app → [finance-data-analyzer.streamlit.app](https://finance-data-analyzer.streamlit.app)**
*(the URL is legacy — the product is domain-agnostic)*

Inspired by [Anomalo](https://www.anomalo.com/): the goal is **trust in data**,
not just a dashboard of numbers.

---

## Why it exists

Real-world data — from ERPs, CRMs, billing systems, spreadsheets, vendor exports —
arrives messy: nulls and placeholders (`N/A`, `UNKNOWN`), duplicates, malformed
dates, out-of-range and outlier values, inconsistent casing/currencies, and
cross-field contradictions. DataQual AI detects all of it automatically and, more
importantly, tells you **whether the dataset is safe to use** and **what to fix first**.

---

## The product — one story, four surfaces

The whole app is a single narrative: **Load → Ask → understand Quality → get a
Verdict & Fix → track Trends.**

| Surface | What it does |
|---|---|
| **💬 Ask** *(the hero)* | Ask a plain-English question — *"which categories have the most issues?"*, *"top 20 worst rows"*. Every answer is **trust-aware**: a chip shows the data's grade and verdict, so you always know how much to trust what you're looking at. Rule-based by default; an optional LLM planner can be switched on. |
| **📊 Quality** | The evidence: overall score + grade, clean vs. critical counts, the 5-dimension breakdown, severity mix, missing-data-by-column, and the full issue table. |
| **🛸 Verdict & Fix** | The decision: an **APPROVE / REMEDIATE / QUARANTINE** verdict with a confidence score, the policy gates that drove it, a prioritised action plan, and one-click exports — cleaned CSV, plus a shareable Markdown/JSON decision report. |
| **🕒 Trends** | Save each run as a portable JSON manifest, re-upload past runs, and see quality-over-time, cross-source comparison, and schema drift. |

---

## Architecture

```
              ┌──────────────────────────────────────────────┐
  app.py  →   │  UI only (Streamlit) — 4 surfaces, ~430 lines │
              └──────────────────────────────────────────────┘
                                  │  imports
                                  ▼
              ┌──────────────────────────────────────────────┐
 engine.py →  │  Pure logic, NO Streamlit — unit-testable     │
              │  • column-type detection & role mapping       │
              │  • profiling                                  │
              │  • 8-step cleaning engine                     │
              │  • 5-dimension scoring engine                 │
              │  • IsolationForest + IQR anomaly detection    │
              │  • rule-based NLQ parser                       │
              │  • multi-domain sample-data generator         │
              └──────────────────────────────────────────────┘
                                  │  reused by
                                  ▼
   optional add-ons (all degrade gracefully if absent):
   autopilot.py · remediation.py · run_manifest.py · multi_source.py · llm_nlq.py
```

**Design principles:** `$0` to run · **no external API** and **no backend** by
default · single-session, stateless · **stdlib + pandas + streamlit + plotly +
sklearn only** (no exotic deps). The optional LLM query planner is **off by
default**, so "no external API" is the out-of-the-box behaviour.

---

## 5-dimension quality score

Every record gets a `dq_score` (0–100), a `dq_grade` (A–F), a `dq_severity`
(CLEAN / LOW / MEDIUM / HIGH / CRITICAL), and a pipe-separated `dq_issues` log.

| Dimension | Weight | Checks |
|---|---:|---|
| Completeness | 20% | nulls, blanks, placeholder values |
| Validity | 25% | format/type correctness (dates, numbers, categories) |
| Accuracy | 35% | range violations, outliers, negatives |
| Consistency | 15% | casing/currency normalisation, cross-field logic |
| Uniqueness | 5% | duplicate detection |

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then either **upload a CSV** or click **Generate sample** to try it on realistic
messy data (finance, sales/CRM, HR, e-commerce, supply chain, or product
analytics), and hit **Analyse data**.

### Optional: enable the LLM query planner
Off by default. To turn it on, open the sidebar → **🤖 Smart Query**, pick a free-tier
provider (Groq / Gemini / OpenRouter presets, or bring your own OpenAI-compatible
endpoint), and paste a key. Only **column names and types** are ever sent — never
your data rows (sending a few sample category values is a separate opt-in). It falls
back to the rule engine transparently on any error or budget cap. Put your key in
`.streamlit/secrets.toml` as `LLM_API_KEY` (git-ignored).

---

## Project structure

```
app.py            # Streamlit UI — the 4 surfaces
engine.py         # all core logic (pure, no Streamlit) — the testable heart
autopilot.py      # APPROVE/REMEDIATE/QUARANTINE decision engine
remediation.py    # remediation log, suggested fixes, cleaned-data export
run_manifest.py   # portable run manifests → history & trends (migration seam)
multi_source.py   # cross-source comparison + schema-drift detection
llm_nlq.py        # optional privacy-first LLM query planner
requirements.txt
.streamlit/        # theme + (git-ignored) secrets
```

---

## Roadmap

- **Analytics NLQ** — time-series and comparison questions via the query planner.
- **Backend (the moat)** — Dockerised Postgres + scheduler + alerting + root-cause
  analysis. The run-manifest schema is the migration contract, so moving from
  JSON → DuckDB → Postgres is an upgrade, not a rewrite. Autopilot's decision rule
  becomes the engine the scheduler calls to auto-quarantine and notify.

---

Built by **Akshay** — Data Engineer.
