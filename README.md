# 📊 Finance Data Quality Pipeline

> A production-ready Python + Streamlit platform that ingests finance data from **SQL Server / Azure SQL** or any CSV, applies **8 automated cleaning operations**, scores every record across **5 weighted quality dimensions**, and surfaces results in a rich interactive web application — all configurable without touching the code.

**Live App → [finance-data-analyzer.streamlit.app](https://finance-data-analyzer.streamlit.app)**

---

## Table of Contents

- [What This Does](#what-this-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Streamlit App — Full Feature Tour](#streamlit-app--full-feature-tour)
- [8 Cleaning Operations](#8-cleaning-operations)
- [5-Dimension DQ Scoring Engine](#5-dimension-dq-scoring-engine)
- [Severity Classification](#severity-classification)
- [Output Columns](#output-columns)
- [Configuration Reference](#configuration-reference)
- [SQL Server / Azure SQL Setup](#sql-server--azure-sql-setup)
- [Pipeline CLI Output](#pipeline-cli-output)
- [Deployment](#deployment)
- [Future Scope](#future-scope)

---

## What This Does

Finance data arriving from ERPs, billing systems, and vendor portals is rarely clean. This pipeline automates the detection and remediation of:

- **Nulls and placeholders** (`N/A`, `UNKNOWN`, blank vendor names)
- **Duplicate invoice records** (double-payment risk)
- **Malformed or inconsistent dates** (`bad-date`, `02-01-2025`, `16/08/2023`)
- **Invalid financial amounts** (negative, out-of-range, statistical outliers)
- **Non-standard currencies and statuses** (`usd`, `Eur`, `settled`, `VOID`)
- **Cross-field logic failures** (PAID with no payment date, total ≠ amount + tax)

Every record exits the pipeline with a **data quality score (0–100)**, a **grade (A–F)**, a **severity label**, and a **pipe-separated issue log** — ready for audit, remediation, or downstream analytics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources                                │
│   SQL Server / Azure SQL   ──OR──   Upload CSV / Demo Data     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  8 Cleaning Operations                          │
│  Nulls → Dupes → Dates → Amounts → Currency → Vendor →        │
│  Status → Outliers                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           5-Dimension DQ Scoring Engine                        │
│   Completeness (20%) · Validity (25%) · Accuracy (35%)        │
│   Consistency (15%) · Uniqueness (5%)                          │
│   → dq_score · dq_grade · dq_severity · dq_issues             │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Cleaned CSV    DB Table Write   Streamlit App
    (timestamped)  (scored layer)   (Interactive)
```

---

## Project Structure

```
finance_pipeline/
├── app.py                   # Streamlit web application (self-contained)
├── pipeline.py              # CLI orchestrator — for scheduled/batch runs
├── cleaner.py               # 8 cleaning operations (used by CLI)
├── scorer.py                # 5-dimension DQ scoring engine (used by CLI)
├── db_connector.py          # SQL Server + SQLite connector
├── generate_sample_data.py  # Synthetic dirty-data generator
├── dq_dashboard.html        # Standalone HTML analytics dashboard
├── config.csv               # All pipeline parameters (no code edits needed)
├── .env.example             # Credential template — copy to .env
├── requirements.txt         # Python dependencies
└── output/                  # Generated at runtime (gitignored)
    ├── invoices_cleaned.csv
    └── pipeline_run_log.csv
```

---

## Quick Start

### Option A — Streamlit App (recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open your browser to `http://localhost:8501`. No database or config file needed.

### Option B — CLI Pipeline (batch/scheduled runs)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 300 rows of synthetic dirty finance data into SQLite
python generate_sample_data.py

# 3. Run the full pipeline
python pipeline.py
```

Cleaned CSV → `output/invoices_cleaned.csv`  
Run audit log → `output/pipeline_run_log.csv`

---

## Streamlit App — Full Feature Tour

The Streamlit app (`app.py`) is entirely self-contained — all cleaning and scoring logic is embedded, so it deploys to Streamlit Cloud with zero additional files.

### Data Source

Two modes, switchable without reloading:

| Mode | How |
|------|-----|
| **Upload CSV** | Drag-and-drop any finance CSV. Auto-detected columns. Supports up to 200 MB. |
| **Demo generator** | Configurable row count (100–1000) + random seed. Injects deliberate dirty data: nulls, dupes, bad dates, invalid currencies, negative amounts, outliers. |

A **raw data preview** expander shows row count, column count, missing value count, and the first 20 rows — so you know exactly what's going into the pipeline before you run it.

> **Bug fixed:** Sidebar filter changes no longer clear pipeline results. The app fingerprints each uploaded file (`filename + size`) and only re-reads the CSV when a genuinely new file is detected — so changing a filter stays on your current tab.

---

### Sidebar — Pipeline Configuration

Every threshold is editable live. Changes take effect on the next pipeline run.

| Setting | Default | Purpose |
|---------|---------|---------|
| Null fill (text) | `UNKNOWN` | Replace missing strings |
| Null fill (numeric) | `0.00` | Replace missing numbers |
| Min valid amount | `0.00` | Flag amounts below this |
| Max valid amount | `10,000,000` | Flag amounts above this |
| Outlier Z-score threshold | `3.0` | Statistical outlier sensitivity |
| Valid currencies | `USD;EUR;GBP;AUD;CAD;INR;SGD` | Accepted ISO currency codes |
| Valid statuses | `PAID;PENDING;OVERDUE;CANCELLED;DRAFT` | Accepted status values |
| Duplicate key columns | `vendor_name;invoice_number` | Columns for dedup logic |

A **Result Filters** section (Severity, Grade, DQ Score range) applies across all six result tabs without triggering a re-run.

---

### Tab 1 — Dashboard

![Dashboard](docs/screenshots/Dashbaord.png)

Five headline KPIs at a glance:

- **Total records** after cleaning
- **Mean DQ Score** (weighted composite)
- **Perfect rows** (score ≥ 99)
- **Critical records** — require manual remediation
- **Validity score** — flagged as lowest when < 60 (often the weakest dimension due to date parsing and status/currency mismatches)

Below the KPIs: a **severity breakdown bar chart** (CLEAN → CRITICAL), a **grade distribution donut** (A–F), and a **score distribution histogram** — immediate visual triage of the dataset's health.

**Example output on 1,000-row demo dataset:**

| Metric | Value |
|--------|-------|
| Records processed | 998 (after 22 dupes removed) |
| Mean DQ Score | 76.3 / 100 |
| Perfect rows | 57 |
| Critical records | 130 |
| Validity score | 44.9 ⚠️ (lowest) |

Severity breakdown: CLEAN 233 · LOW 2 · MEDIUM 264 · HIGH 369 · CRITICAL 130  
Grade distribution: A 21.9% · B 35.5% · C 31.4% · D 7.7% · F 3.5%

---

### Tab 2 — Dimension Analysis

![Dimension Analysis](docs/screenshots/Dimension%20Analysis.png)

Five metric tiles with progress bars — one per dimension — give an instant read on where quality breaks down:

| Dimension | Score (demo) | Weight |
|-----------|-------------|--------|
| Completeness | 86.8 | 20% |
| Validity | **44.9** ← weakest | 25% |
| Accuracy | 89.3 | 35% |
| Consistency | 82.5 | 15% |
| Uniqueness | 89.2 | 5% |

A **horizontal bar chart** ranks dimensions visually. Below it, a **Top Failing Checks** chart shows the 10 most frequent issues, coloured by dimension, so you can target the highest-ROI fixes first.

**Top issues on the demo dataset:**

| Check | Count | Dimension |
|-------|-------|-----------|
| Row had missing values | 801 | Completeness |
| Invoice date unparseable | 346 | Validity |
| Due date unparseable | 345 | Validity |
| Invalid invoice status | 340 | Validity |
| Invalid currency code | 327 | Validity |
| Payment date unparseable | 231 | Validity |
| Negative invoice amount | 84 | Accuracy |
| Negative total amount | 82 | Accuracy |
| Amount is statistical outlier | 31 | Accuracy |
| Total is statistical outlier | 30 | Accuracy |

---

### Tab 3 — Record Explorer

![Record Explorer](docs/screenshots/Record%20Explorer.png)

A sortable, scrollable table of every record with inline **progress bars** for all six score columns — no matplotlib required, using Streamlit's native `ProgressColumn`.

Columns shown: `invoice_number`, `vendor_name`, `amount`, `currency`, `status`, `dq_score`, `dq_grade`, `dq_severity`, and all five dimension sub-scores.

Sort by any score column, ascending (worst first) or descending — useful for targeted remediation queues.

---

### Tab 4 — Deep Dive

![Deep Dive](docs/screenshots/Deep%20Dive.png)

Select any invoice number to inspect it in full detail:

- **Overall score, grade, severity** at a glance
- **Per-dimension breakdown** — five progress bars showing where exactly the record fails
- **Radar chart** — visualises the five-dimensional quality profile in one shape
- **Colour-coded issue list** — every failed check labelled with its dimension tag, styled with the dimension's brand colour (red for Validity, green for Accuracy, blue for Completeness, etc.)

**Example — INV-1339 (CRITICAL):**  
Completeness 100 · Validity 100 · Accuracy **0** · Consistency 100 · Uniqueness 0  
Issues: Amount exceeds max threshold · Total exceeds max threshold · Amount is a statistical outlier · Total is a statistical outlier

---

### Tab 5 — Statistics

![Statistics](docs/screenshots/Statistics.png)

Two plain tables — no charts, just numbers:

**Cleaning Statistics** — what the pipeline actually did:

| Metric | Value (demo) |
|--------|-------------|
| rows_input | 1,020 |
| rows_output | 998 |
| duplicates_removed | 22 |
| nulls_filled | 186 |
| rows_with_nulls | 801 |
| date_parse_errors | 922 |
| negative_amount_flags | 166 |
| currency_invalid | 327 |
| vendor_names_fixed | 741 |
| statuses_fixed | 340 |

**Scoring Statistics** — distribution summary across the scored dataset.

---

### Tab 6 — Export

![Export](docs/screenshots/Export.png)

Two download options:

**Full dataset** — all records, all DQ columns, timestamped filename.

**Filter → Preview → Download** — a purpose-built severity-aware export workflow:

1. Pick a severity level (ALL / CLEAN / LOW / MEDIUM / HIGH / CRITICAL)
2. Choose which columns to include via a multiselect picker
3. See **4 KPIs** instantly: record count, average DQ score, A/B grade count, D/F grade count
4. Preview the exact records in a scrollable table with progress-bar score columns
5. Download only when you're satisfied — button label shows row count (e.g. *"Download CRITICAL records (130 rows)"*)

**Example — CRITICAL filter:**  
130 records · Avg DQ Score 52.8 · 16 A/B grades · 86 D/F grades

---

## 8 Cleaning Operations

| # | Operation | Details |
|---|-----------|---------|
| 1 | **Null handling** | Fills missing strings → configurable fill value (default `UNKNOWN`); numerics → `0.0`. Flags every row that originally contained any null with `had_nulls = 1`. |
| 2 | **Duplicate removal** | Drops rows with identical values on the configured key columns (default: `vendor_name + invoice_number`). Keeps first occurrence. Reports exact count removed. |
| 3 | **Date normalization** | Parses all date columns to ISO `YYYY-MM-DD` using `pd.to_datetime(errors="coerce")`. Unparseable values get a `{col}_parse_error = 1` flag and the null fill value. Handles `DD/MM/YYYY`, `MM-DD-YYYY`, and free text. |
| 4 | **Amount validation** | Coerces all amount columns to numeric. Adds `{col}_negative`, `{col}_below_min`, `{col}_above_max` flags per column. |
| 5 | **Currency standardisation** | Strips whitespace, upper-cases, validates against the accepted list. Invalid codes (`XYZ`, `usd`, `Eur`) are replaced with the null fill value and flagged `currency_invalid = 1`. |
| 6 | **Vendor name cleanup** | Strips leading/trailing whitespace, collapses internal spaces, applies title-case (`KPMG ADVISORY` → `Kpmg Advisory`). Placeholder values (`N/A`, `UNKNOWN VENDOR`) are replaced. |
| 7 | **Status normalisation** | Strips, upper-cases, validates against the accepted list. Non-standard values (`settled`, `VOID`, `paid`) are replaced and flagged `status_invalid = 1`. |
| 8 | **Outlier flagging** | Computes Z-score per amount column. Records with `|z| > threshold` (default `3.0`) get `{col}_outlier = 1`. Does not alter the value — flags only. |

---

## 5-Dimension DQ Scoring Engine

Every record receives six scores: one per dimension and a weighted composite.

### Weights & Business Rationale

| Dimension | Weight | Core question | Why this weight |
|-----------|--------|---------------|-----------------|
| **Completeness** | 20% | Are all required fields populated? | Missing vendor or amount blocks reconciliation and audit |
| **Validity** | 25% | Do values conform to format and domain rules? | Invalid currency or status breaks ERP payment runs and workflow routing |
| **Accuracy** | **35%** | Are values mathematically and financially correct? | A wrong number processed downstream costs real money — highest business risk |
| **Consistency** | 15% | Are fields logically coherent with each other? | PAID with no payment date is unreconcilable; total ≠ amount + tax is an accounting error |
| **Uniqueness** | 5% | Is each record distinct? | Duplicate invoice numbers create double-payment risk |

### Scoring Detail

**Completeness** — penalty per missing or placeholder-valued critical field:
`vendor_name`, `invoice_number`, `amount`, `total_amount`, `currency`, `status`

**Validity** — penalties applied for:
- Invalid currency code: −60 pts
- Invalid invoice status: −50 pts
- Invoice date unparseable: −30 pts
- Due date unparseable: −25 pts
- Payment date unparseable: −20 pts

**Accuracy** — penalties applied for:
- Negative `amount` or `total_amount`: −70 pts each
- Negative `tax_amount`: −40 pts
- Amount above max threshold: −40 pts
- Statistical outlier (Z-score): −20 pts per column

**Consistency** — three cross-field checks:
- `status = PAID` and `payment_date` is missing/placeholder → −60 pts
- `|total_amount − (amount + tax_amount)| > 0.02` → −60 pts
- `due_date` < `invoice_date` (chronologically impossible) → −40 pts

**Uniqueness** — duplicate `invoice_number` (any copy, keep=False) → 0 pts

### Composite Score

```
dq_score = (completeness × 0.20) + (validity × 0.25) +
           (accuracy × 0.35) + (consistency × 0.15) +
           (uniqueness × 0.05)
```

All sub-scores are clamped to [0, 100].

---

## Severity Classification

| Severity | Trigger condition | Recommended action |
|----------|------------------|--------------------|
| `CLEAN` | score ≥ 85 AND all dimensions ≥ 70 | Publish to gold layer |
| `LOW` | score < 85 | Safe for most analytics use |
| `MEDIUM` | score < 70 OR any dimension < 60 | Review recommended before publishing |
| `HIGH` | Accuracy < 60 OR Validity < 40 OR score < 50 | Use with caution; flag for remediation |
| `CRITICAL` | Accuracy < 30 OR Consistency < 40 | Do not use; manual remediation required |

---

## Output Columns

Every record in the cleaned/scored output gains these columns:

| Column | Type | Description |
|--------|------|-------------|
| `had_nulls` | int (0/1) | Row had at least one null before cleaning |
| `{col}_parse_error` | int (0/1) | Date column was unparseable |
| `{col}_negative` | int (0/1) | Amount was negative |
| `{col}_below_min` | int (0/1) | Amount below configured minimum |
| `{col}_above_max` | int (0/1) | Amount above configured maximum |
| `{col}_outlier` | int (0/1) | Amount exceeded Z-score threshold |
| `currency_invalid` | int (0/1) | Currency not in accepted list |
| `status_invalid` | int (0/1) | Status not in accepted list |
| `dq_score` | int 0–100 | Weighted overall quality score |
| `dq_grade` | A/B/C/D/F | Letter grade |
| `dq_severity` | string | CLEAN / LOW / MEDIUM / HIGH / CRITICAL |
| `dq_score_completeness` | float | Completeness dimension sub-score |
| `dq_score_validity` | float | Validity dimension sub-score |
| `dq_score_accuracy` | float | Accuracy dimension sub-score |
| `dq_score_consistency` | float | Consistency dimension sub-score |
| `dq_score_uniqueness` | float | Uniqueness dimension sub-score |
| `dq_issues` | string | Pipe-separated list of all failed checks with dimension tag |

### Example `dq_issues` value

```
[Accuracy] Negative invoice amount | [Validity] Invalid currency code | [Consistency] PAID status with no payment date
```

---

## Configuration Reference (`config.csv`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_mode` | `demo` | `demo` (SQLite) or `sqlserver` |
| `source_table` | `invoices` | Table to load |
| `cleaned_table` | `invoices_cleaned` | Table to write scored data to |
| `duplicate_subset` | `vendor_name;invoice_number` | Key columns for dedup |
| `date_columns` | `invoice_date;due_date;payment_date` | Date columns to normalize |
| `amount_columns` | `amount;tax_amount;total_amount` | Amount columns to validate |
| `valid_currencies` | `USD;EUR;GBP;AUD;CAD;INR;SGD` | Accepted ISO codes |
| `valid_statuses` | `PAID;PENDING;OVERDUE;CANCELLED;DRAFT` | Accepted statuses |
| `outlier_zscore_threshold` | `3.0` | Z-score cutoff |
| `null_fill_string` | `UNKNOWN` | Fill for missing text |
| `null_fill_numeric` | `0.0` | Fill for missing numerics |
| `min_amount` | `0.0` | Minimum valid amount |
| `max_amount` | `10000000.0` | Maximum valid amount |

Credentials are stored as `${ENV_VAR}` placeholders in `config.csv` and resolved from `.env` at runtime — never hardcoded.

---

## SQL Server / Azure SQL Setup

```bash
# 1. Copy credential template
cp .env.example .env
```

Edit `.env`:
```
DB_SERVER=your-server.database.windows.net
DB_NAME=FinanceDB
DB_USER=your_username
DB_PASSWORD=your_password
DB_DRIVER=ODBC Driver 18 for SQL Server
```

Set `db_mode=sqlserver` in `config.csv`, then:

```bash
python pipeline.py
```

Requires the [Microsoft ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

---

## Pipeline CLI Output

```
════════════════════════════════════════════════════════════════════
  PIPELINE SUMMARY
════════════════════════════════════════════════════════════════════
  Rows loaded (raw):                        1,020
  Rows output (cleaned):                      998
  Duplicates removed:                          22
────────────────────────────────────────────────────────────────────
  Null cells filled:                          186
  Date parse errors:                          922
  Negative amount flags:                      166
  Invalid currency codes:                     327
  Invalid statuses corrected:                 340
  Outliers flagged:                           166

════════════════════════════════════════════════════════════════════
  DATA QUALITY SCORES  (weighted composite)
────────────────────────────────────────────────────────────────────
  Overall DQ Score   ██████████████░░░░   76.3 / 100

  Dimension       Wt    Score  Bar
  ─────────────  ────  ──────  ───────────────────────────────────
  Completeness    20%    86.8  ████████████████░░
  Validity        25%    44.9  ████████░░░░░░░░░░  ← weakest
  Accuracy        35%    89.3  █████████████████░
  Consistency     15%    82.5  ███████████████░░░
  Uniqueness       5%    89.2  █████████████████░

  Severity:  ✅ CLEAN:233  🟡 LOW:2  🟠 MEDIUM:264  🔴 HIGH:369  🚨 CRITICAL:130
════════════════════════════════════════════════════════════════════
```

---

## Deployment

### Streamlit Cloud

1. Push the repo to GitHub (must be public, or connected to your Streamlit account)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `main`, main file `app.py`
4. Deploy — no secrets needed for demo mode

The app is self-contained. No `cleaner.py`, `scorer.py`, or `config.csv` is needed at runtime; all logic is embedded in `app.py`.

### GitHub Pages (HTML Dashboard)

1. Go to your repo Settings → Pages
2. Set Source branch to `main`, folder `/` (root)
3. Save — the dashboard is live at `https://<username>.github.io/<repo>/dq_dashboard.html`

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
pyodbc>=4.0.39
streamlit>=1.35.0
plotly>=5.20.0
```

Python 3.8+. `pyodbc` is only required for SQL Server mode.

---

## Future Scope

- **Airflow / Azure Data Factory scheduling** — run the pipeline on a daily cadence with retries and SLA alerting
- **dbt integration** — use the scored output as a silver layer source model with data contracts
- **Reconciliation layer** — cross-table validation (every PAID invoice has a matching payment record in the payments table)
- **Expand to more tables** — `payments`, `vendors`, `purchase_orders`, `general_ledger` via config change only
- **Power BI / Grafana connector** — push DQ metrics to a live operations dashboard
- **Rule engine** — define custom validation rules in `config.csv` without writing Python
- **Email / Teams alerting** — notify the data team when CRITICAL record count exceeds a threshold

---

*Built by Akshay — Data Engineer*  
*Stack: Python · pandas · Streamlit · Plotly · SQLAlchemy · SQL Server / SQLite*
