"""
pipeline.py
-----------
Finance Data Cleaning Pipeline — Main Orchestrator
===================================================

What it does
------------
1. Loads parameters from config.csv
2. Connects to SQL Server (or SQLite in demo mode)
3. Extracts the source table into a Pandas DataFrame
4. Runs 8 cleaning operations via cleaner.py
5. Exports the cleaned data to a CSV file
6. Writes the cleaned data back to the database as a new table
7. Appends a run summary to the pipeline audit log (CSV)
8. Prints a human-readable summary report

Quick start (demo mode — no DB credentials needed):
    pip install -r requirements.txt
    python generate_sample_data.py   # creates finance_demo.db with dirty data
    python pipeline.py               # runs the full pipeline

Switch to SQL Server:
    Edit config.csv → set db_mode=sqlserver and fill in db_server, db_name, etc.
    Then just: python pipeline.py
"""

import csv
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from db_connector import get_engine, table_exists
from cleaner import run_all_cleaners

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     Finance Data Cleaning Pipeline  |  by Akshay         ║
╚══════════════════════════════════════════════════════════╝
"""

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.csv") -> dict:
    """Read config.csv and return a {parameter: value} dict."""
    if not os.path.exists(path):
        logger.error(f"config.csv not found at '{path}'. Aborting.")
        sys.exit(1)
    config = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config[row["parameter"].strip()] = row["value"].strip()
    logger.info(f"[Config] Loaded {len(config)} parameters from '{path}'")
    return config


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_data(engine, config: dict) -> pd.DataFrame:
    """Load the source table from the database into a DataFrame."""
    mode   = config.get("db_mode", "demo").lower()
    table  = config["source_table"]

    if mode == "demo":
        # SQLite — no schema prefix
        query = f"SELECT * FROM {table}"
    else:
        schema = config.get("source_schema", "dbo")
        query  = f"SELECT * FROM [{schema}].[{table}]"

    logger.info(f"[Extract] Running: {query}")
    df = pd.read_sql(query, engine)
    logger.info(f"[Extract] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, config: dict) -> str:
    """Write cleaned DataFrame to a CSV file. Returns the output path."""
    path = config.get("export_csv_path", "output/invoices_cleaned.csv")
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"[Export] CSV written → {path}  ({len(df):,} rows)")
    return path


# ---------------------------------------------------------------------------
# DB write-back
# ---------------------------------------------------------------------------

def write_to_db(df: pd.DataFrame, engine, config: dict) -> None:
    """Write the cleaned DataFrame to the cleaned_table in the database."""
    mode          = config.get("db_mode", "demo").lower()
    cleaned_table = config.get("cleaned_table", "invoices_cleaned")
    cleaned_schema = config.get("cleaned_schema", "dbo")

    if mode == "demo":
        # SQLite — schema not supported
        df.to_sql(cleaned_table, engine, if_exists="replace", index=False)
        logger.info(f"[Write-back] Table '{cleaned_table}' written to SQLite demo DB")
    else:
        df.to_sql(
            cleaned_table,
            engine,
            schema=cleaned_schema,
            if_exists="replace",
            index=False,
            chunksize=500,
        )
        logger.info(f"[Write-back] Table '[{cleaned_schema}].[{cleaned_table}]' written to SQL Server")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def append_run_log(config: dict, stats: dict, rows_in: int, rows_out: int) -> None:
    """Append one row to the pipeline run log CSV."""
    log_path = config.get("log_csv_path", "output/pipeline_run_log.csv")
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    run_record = {
        "run_timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_table":         config.get("source_table"),
        "cleaned_table":        config.get("cleaned_table"),
        "rows_input":           rows_in,
        "rows_output":          rows_out,
        **stats,
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_record)
    logger.info(f"[Audit] Run logged → {log_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(stats: dict, rows_in: int, rows_out: int, csv_path: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print("  PIPELINE SUMMARY REPORT")
    print("═" * width)
    print(f"  {'Rows loaded (raw):':<35} {rows_in:>8,}")
    print(f"  {'Rows output (cleaned):':<35} {rows_out:>8,}")
    print(f"  {'Rows removed (duplicates):':<35} {stats.get('duplicates_removed', 0):>8,}")
    print("─" * width)
    print(f"  {'Null cells filled:':<35} {stats.get('nulls_filled', 0):>8,}")
    print(f"  {'Rows that had nulls:':<35} {stats.get('rows_with_nulls', 0):>8,}")
    print(f"  {'Date parse errors:':<35} {stats.get('date_parse_errors', 0):>8,}")
    print(f"  {'Negative amount flags:':<35} {stats.get('amount_negative_flags', 0):>8,}")
    print(f"  {'Above-max amount flags:':<35} {stats.get('amount_above_max_flags', 0):>8,}")
    print(f"  {'Invalid currency codes fixed:':<35} {stats.get('currency_invalid', 0):>8,}")
    print(f"  {'Vendor names normalized:':<35} {stats.get('vendor_names_normalized', 0):>8,}")
    print(f"  {'Invalid statuses corrected:':<35} {stats.get('status_invalid', 0):>8,}")
    print(f"  {'Outliers flagged:':<35} {stats.get('outliers_flagged', 0):>8,}")
    print("─" * width)
    print(f"  CSV output  →  {csv_path}")
    print("═" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(BANNER)
    start = datetime.now()

    # 1. Load config
    config = load_config("config.csv")

    # 2. Connect to database
    engine = get_engine(config)

    # 3. Extract source data
    df_raw = extract_data(engine, config)
    rows_in = len(df_raw)

    # 4. Run all cleaning operations
    logger.info("[Clean] Starting cleaning operations...")
    df_clean, stats = run_all_cleaners(df_raw, config)
    rows_out = len(df_clean)

    # 5. Export to CSV
    csv_path = export_csv(df_clean, config)

    # 6. Write cleaned table back to DB
    write_to_db(df_clean, engine, config)

    # 7. Append to audit log
    append_run_log(config, stats, rows_in, rows_out)

    # 8. Print summary
    elapsed = (datetime.now() - start).total_seconds()
    print_summary(stats, rows_in, rows_out, csv_path)
    logger.info(f"[Done] Pipeline completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
