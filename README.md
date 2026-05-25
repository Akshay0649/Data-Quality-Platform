# Finance Data Cleaning Pipeline

A production-ready Python pipeline that connects to **SQL Server / Azure SQL**, loads financial data (invoices, payments, vendors), applies **8 automated cleaning operations**, and outputs both a cleaned CSV file and a clean table written back to the database.

All pipeline parameters — table names, cleaning thresholds, column lists — are controlled via a single `config.csv` file with no code changes required.

---

## Project Structure

```
finance_pipeline/
├── pipeline.py              # Main orchestrator — run this
├── cleaner.py               # All 8 cleaning operations
├── db_connector.py          # SQL Server + SQLite demo mode connector
├── generate_sample_data.py  # Synthetic dirty-data generator (demo/testing)
├── config.csv               # All pipeline parameters
├── .env.example             # Credential template — copy to .env
├── requirements.txt         # Python dependencies
└── output/                  # Generated at runtime (gitignored)
    ├── invoices_cleaned.csv
    └── pipeline_run_log.csv
```

---

## Quick Start (Demo Mode — no DB needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dirty finance data into a local SQLite DB
python generate_sample_data.py

# 3. Run the pipeline
python pipeline.py
```

The cleaned CSV lands in `output/invoices_cleaned.csv` and a run log in `output/pipeline_run_log.csv`.

---

## Connecting to SQL Server / Azure SQL

1. Copy the credentials template:
   ```bash
   cp .env.example .env
   ```

2. Fill in your real values in `.env`:
   ```
   DB_SERVER=your-server.database.windows.net
   DB_NAME=FinanceDB
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_DRIVER=ODBC Driver 18 for SQL Server
   ```

3. Set `db_mode` to `sqlserver` in `config.csv`.

4. Run:
   ```bash
   python pipeline.py
   ```

> **Note:** You need the [Microsoft ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) installed on your machine.

---

## Cleaning Operations

| # | Operation | What it does |
|---|-----------|--------------|
| 1 | **Null Handling** | Fills missing strings with `UNKNOWN`, numerics with `0.0`; flags rows that had nulls |
| 2 | **Duplicate Removal** | Drops duplicate rows based on configurable key columns |
| 3 | **Date Normalization** | Parses all date columns to `YYYY-MM-DD`; flags unparseable values |
| 4 | **Amount Validation** | Flags negative, below-minimum, and above-maximum amounts |
| 5 | **Currency Standardization** | Upper-cases codes and validates against accepted ISO list |
| 6 | **Vendor Name Cleanup** | Strips whitespace, title-cases, replaces placeholder values |
| 7 | **Status Normalization** | Strips/upper-cases status field, validates against accepted list |
| 8 | **Outlier Flagging** | Z-score based detection on amount columns; configurable threshold |

---

## Configuration Reference (`config.csv`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_mode` | `demo` | `demo` (SQLite) or `sqlserver` |
| `source_table` | `invoices` | Table to load from DB |
| `cleaned_table` | `invoices_cleaned` | Table to write cleaned data to |
| `duplicate_subset` | `vendor_name;invoice_number` | Columns for duplicate detection |
| `date_columns` | `invoice_date;due_date;payment_date` | Columns to parse as dates |
| `amount_columns` | `amount;tax_amount;total_amount` | Columns to validate |
| `valid_currencies` | `USD;EUR;GBP;...` | Accepted ISO currency codes |
| `valid_statuses` | `PAID;PENDING;OVERDUE;...` | Accepted status values |
| `outlier_zscore_threshold` | `3.0` | Z-score cutoff for outlier flagging |
| `null_fill_string` | `UNKNOWN` | Fill value for missing text fields |
| `null_fill_numeric` | `0.0` | Fill value for missing numeric fields |

---

## Sample Run Output

```
════════════════════════════════════════════════════════════
  PIPELINE SUMMARY REPORT
════════════════════════════════════════════════════════════
  Rows loaded (raw):                       320
  Rows output (cleaned):                   299
  Rows removed (duplicates):                21
────────────────────────────────────────────────────────────
  Null cells filled:                       484
  Date parse errors:                       198
  Negative amount flags:                    38
  Invalid currency codes fixed:             99
  Vendor names normalized:                 216
  Invalid statuses corrected:              117
  Outliers flagged:                         22
════════════════════════════════════════════════════════════
```

---

## Requirements

- Python 3.8+
- pandas, numpy, sqlalchemy
- pyodbc (SQL Server mode only)

---

*Built by Akshay — Data Engineer*
