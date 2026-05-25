"""
generate_sample_data.py
-----------------------
Creates a synthetic Finance / Accounting dataset with intentional data quality
issues so you can test the cleaning pipeline without a real SQL Server.

Dirty data injected:
  - NULL vendor names, amounts, statuses
  - Duplicate invoice rows
  - Malformed dates (wrong format, nonsense strings)
  - Invalid currency codes
  - Negative amounts (invalid for invoices)
  - Outlier amounts (suspiciously large values)
  - Mixed-case / extra whitespace in vendor names and statuses
  - Missing payment dates for PAID invoices

Run:
    python generate_sample_data.py
Writes the 'invoices' table into finance_demo.db (SQLite).
"""

import random
import sqlite3
from datetime import date, timedelta

import pandas as pd

random.seed(42)

VENDORS = [
    "Accenture Ltd", "  KPMG Advisory  ", "Deloitte & Touche",
    "ernst & young", "PwC Services", "GARTNER INC", "infosys BPO",
    "wipro technologies", "IBM Global  ", "Capgemini SE",
    "  Oracle Corp", "SAP AG", "Microsoft Azure", "AWS Finance",
    "Cognizant Tech", None, "N/A", "UNKNOWN VENDOR"
]

CURRENCIES = ["USD", "EUR", "GBP", "AUD", "INR", "XYZ", "ZZZ", "usd", "Eur", None]

STATUSES = [
    "PAID", "PENDING", "OVERDUE", "CANCELLED", "DRAFT",
    "paid", "  Pending  ", "overdue", "settled", "VOID", None, ""
]

def random_date(start: date, end: date) -> str:
    """Return a random date string, occasionally malformed."""
    delta = (end - start).days
    d = start + timedelta(days=random.randint(0, delta))
    # 10% chance of bad format
    fmt = random.choice([
        "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d",
        "%d/%m/%Y", "%m-%d-%Y", "not-a-date", ""
    ])
    if fmt == "":
        return None
    return d.strftime(fmt)


def generate_invoices(n: int = 300) -> pd.DataFrame:
    rows = []
    base_date = date(2023, 1, 1)
    end_date  = date(2025, 12, 31)

    for i in range(1, n + 1):
        vendor  = random.choice(VENDORS)
        inv_num = f"INV-{random.randint(1000, 9999)}"
        amount  = round(random.uniform(-500, 250_000), 2)

        # 5% chance of outlier
        if random.random() < 0.05:
            amount = round(random.uniform(1_000_000, 15_000_000), 2)

        # 8% chance of negative amount
        if random.random() < 0.08:
            amount = -abs(amount)

        tax     = round(amount * random.uniform(0.05, 0.18), 2) if amount > 0 else None
        total   = round(amount + (tax or 0), 2)
        currency = random.choice(CURRENCIES)
        status   = random.choice(STATUSES)

        inv_date  = random_date(base_date, end_date)
        due_date  = random_date(base_date, end_date)
        pay_date  = random_date(base_date, end_date) if random.random() > 0.35 else None

        rows.append({
            "invoice_id":    i,
            "invoice_number": inv_num,
            "vendor_name":    vendor,
            "invoice_date":   inv_date,
            "due_date":       due_date,
            "payment_date":   pay_date,
            "amount":         amount if random.random() > 0.05 else None,
            "tax_amount":     tax,
            "total_amount":   total if random.random() > 0.05 else None,
            "currency":       currency,
            "status":         status,
            "department":     random.choice(["Finance", "IT", "HR", "Ops", None]),
            "cost_centre":    random.choice(["CC100", "CC200", "CC300", "CC999", None]),
        })

    df = pd.DataFrame(rows)

    # Inject 20 exact duplicates
    dupes = df.sample(20, random_state=7).copy()
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df["invoice_id"] = range(1, len(df) + 1)  # re-assign IDs

    return df


if __name__ == "__main__":
    print("Generating synthetic finance data...")
    df = generate_invoices(300)
    print(f"  → {len(df)} rows generated (including duplicates)")

    conn = sqlite3.connect("finance_demo.db")
    df.to_sql("invoices", conn, if_exists="replace", index=False)
    conn.close()

    print("  → Written to 'invoices' table in finance_demo.db")
    print("\nDirty data summary:")
    print(f"  Null vendor_name   : {df['vendor_name'].isna().sum()}")
    print(f"  Null amount        : {df['amount'].isna().sum()}")
    print(f"  Null status        : {df['status'].isna().sum()}")
    print(f"  Null currency      : {df['currency'].isna().sum()}")
    print(f"  Negative amounts   : {(df['amount'] < 0).sum()}")
    print(f"  Approx duplicates  : ~20 injected")
