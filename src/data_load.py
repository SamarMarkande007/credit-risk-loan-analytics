"""
data_load.py  -  Clean the raw CSVs and load them into MySQL.

One command does everything (safe to re-run, it rebuilds the tables):

    python src/data_load.py

Steps
  1. Read the 4 raw CSVs from datasets/
  2. Basic cleaning (whitespace, casing, dates, integrity checks) -> logged
  3. Create the database (if missing) and the 4 tables from sql/schema.sql
  4. Bulk-load in FK order: customers -> credit_history -> loan_applications -> transactions
  5. Validate: row counts, orphan keys, PRD portfolio metrics

Credentials come from a .env file in the project root (copy .env.example).
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

# --------------------------------------------------------------------------
# Paths (no hard-coded absolute paths: everything is relative to the project)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
SCHEMA_FILE = ROOT / "sql" / "schema.sql"
LOG_FILE = ROOT / "documentation" / "cleaning_log.txt"

# Rows we expect (from the PRD) - used in validation
EXPECTED_ROWS = {
    "customers": 5000,
    "credit_history": 5000,
    "loan_applications": 7500,
    "transactions": 40079,
}

LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    """Print and remember a line for the cleaning log file."""
    print(msg)
    LOG_LINES.append(msg)


# --------------------------------------------------------------------------
# Database connection
# --------------------------------------------------------------------------
def get_engine(with_database: bool = True) -> Engine:
    """Build a SQLAlchemy engine from .env settings."""
    load_dotenv(ROOT / ".env")
    required = ["DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        sys.exit(
            f"Missing {missing} in .env  ->  copy .env.example to .env and fill it in."
        )
    url = URL.create(
        "mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),  # URL.create escapes special characters safely
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME") if with_database else None,
    )
    return create_engine(url, pool_pre_ping=True)


# --------------------------------------------------------------------------
# 1. Read raw files
# --------------------------------------------------------------------------
def read_raw() -> dict[str, pd.DataFrame]:
    files = {
        "customers": "customers.csv",
        "credit_history": "credit_history.csv",
        "loan_applications": "loan_applications.csv",
        "transactions": "transactions.csv",
    }
    raw = {}
    for table, fname in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            sys.exit(f"Missing file: {path}  (did you rename 'customers (1).csv' to customers.csv?)")
        raw[table] = pd.read_csv(path)
        log(f"  read {fname:<24} {raw[table].shape[0]:>7,} rows x {raw[table].shape[1]} cols")
    return raw


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def _strip_text(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Trim and collapse whitespace in every text column."""
    changed = 0
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        cleaned = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        changed += int((cleaned.fillna("") != df[col].astype("string").fillna("")).sum())
        df[col] = cleaned
    log(f"  [{name}] whitespace fixed in {changed} cells")
    return df


def _parse_dates(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    for col in cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad = int(parsed.isna().sum() - df[col].isna().sum())
        if bad:
            sys.exit(f"[{name}] {bad} unparseable dates in {col}")
        df[col] = parsed.dt.date  # python date -> MySQL DATE, NaT -> NULL
        nulls = int(parsed.isna().sum())
        if nulls:
            log(f"  [{name}] {col}: {nulls} empty dates kept as NULL")
    return df


def clean(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cust = raw["customers"].copy()
    cred = raw["credit_history"].copy()
    loan = raw["loan_applications"].copy()
    txn = raw["transactions"].copy()

    # --- whitespace on text columns --------------------------------------
    cust = _strip_text(cust, "customers")
    cred = _strip_text(cred, "credit_history")
    loan = _strip_text(loan, "loan_applications")
    txn = _strip_text(txn, "transactions")

    # --- IDs upper-case, emails lower-case -------------------------------
    for df in (cust, cred, loan, txn):
        df["customer_id"] = df["customer_id"].str.upper()
    cust["email"] = cust["email"].str.lower()

    # --- primary-key duplicates (exact repeats are dropped) ---------------
    for name, df, pk in [
        ("customers", cust, "customer_id"),
        ("credit_history", cred, "credit_id"),
        ("loan_applications", loan, "application_id"),
        ("transactions", txn, "transaction_id"),
    ]:
        dups = int(df[pk].duplicated().sum())
        if dups:
            df.drop_duplicates(subset=pk, keep="first", inplace=True)
        log(f"  [{name}] duplicate {pk}: {dups} dropped")

    # --- duplicate emails: reported, NOT removed (two different customers) -
    dup_email = cust[cust["email"].duplicated(keep=False)].sort_values("email")
    if len(dup_email):
        log(f"  [customers] {dup_email['email'].nunique()} email(s) shared by "
            f"{len(dup_email)} customers -> kept (email is not used in modelling):")
        for _, r in dup_email.iterrows():
            log(f"      {r['customer_id']}  {r['first_name']} {r['last_name']}  {r['email']}")

    # --- dates ------------------------------------------------------------
    cust = _parse_dates(cust, ["date_of_birth", "customer_since"], "customers")
    cred = _parse_dates(cred, ["as_of_date"], "credit_history")
    loan = _parse_dates(loan, ["application_date", "decision_date"], "loan_applications")
    txn = _parse_dates(txn, ["transaction_date"], "transactions")

    # --- referential integrity: drop orphans -------------------------------
    valid = set(cust["customer_id"])
    for name, df in [("credit_history", cred), ("loan_applications", loan), ("transactions", txn)]:
        orphan = ~df["customer_id"].isin(valid)
        if orphan.any():
            log(f"  [{name}] {int(orphan.sum())} orphan rows (no matching customer) dropped")
            df.drop(df[orphan].index, inplace=True)
        else:
            log(f"  [{name}] orphan customer_ids: 0")

    # --- target logic: loan_default must exist only for disbursed loans ----
    loan["loan_default"] = loan["loan_default"].astype("Int64")  # nullable int, keeps NULLs
    bad_target = loan["loan_default"].notna() & (loan["disbursed_flag"] != 1)
    if bad_target.any():
        log(f"  [loan_applications] {int(bad_target.sum())} rows had a default label but were not "
            f"disbursed -> label set to NULL")
        loan.loc[bad_target, "loan_default"] = pd.NA
    missing_target = (loan["disbursed_flag"] == 1) & loan["loan_default"].isna()
    log(f"  [loan_applications] disbursed loans missing a target: {int(missing_target.sum())}")
    log(f"  [loan_applications] target NULL (not disbursed): {int(loan['loan_default'].isna().sum()):,} "
        f"(valid - these are rejected / under-review applications)")

    # --- range checks that mirror the DB constraints -----------------------
    checks = {
        "customers": (cust, "age >= 18 and age <= 100 and annual_income >= 0"),
        "credit_history": (cred, "credit_score >= 300 and credit_score <= 900 "
                                 "and credit_utilization_ratio >= 0 and credit_utilization_ratio <= 1"),
        "loan_applications": (loan, "requested_amount > 0 and loan_term_months > 0 "
                                    "and debt_to_income_ratio >= 0 and interest_rate >= 0"),
        "transactions": (txn, "amount >= 0"),
    }
    for name, (df, expr) in checks.items():
        bad = len(df) - len(df.query(expr))
        log(f"  [{name}] rows violating range rules: {bad}")
        if bad:
            sys.exit(f"[{name}] has {bad} rows that would be rejected by MySQL constraints; fix the data first.")

    # --- numpy NaN -> None so MySQL stores real NULLs ----------------------
    out = {"customers": cust, "credit_history": cred, "loan_applications": loan, "transactions": txn}
    for name, df in out.items():
        out[name] = df.astype(object).where(df.notna(), None)
    return out


# --------------------------------------------------------------------------
# 3. Schema
# --------------------------------------------------------------------------
def create_database(engine_no_db: Engine) -> None:
    db = os.getenv("DB_NAME")
    if not re.fullmatch(r"[A-Za-z0-9_]+", db or ""):
        sys.exit("DB_NAME may only contain letters, digits and underscores.")
    with engine_no_db.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db}` DEFAULT CHARACTER SET utf8mb4"))
    log(f"  database `{db}` ready")


def run_schema(engine: Engine) -> None:
    """Execute sql/schema.sql statement by statement."""
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    sql = re.sub(r"--[^\n]*", "", sql)  # strip full-line AND trailing comments before splitting on ';'
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    log(f"  schema.sql executed ({len(statements)} statements)")


def check_mysql_version(engine: Engine) -> None:
    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).scalar()
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    parts = tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
    log(f"  MySQL version: {version}")
    if parts < (8, 0, 16):
        log("  WARNING: MySQL < 8.0.16 silently ignores CHECK constraints. Please upgrade to 8.0.16+.")


# --------------------------------------------------------------------------
# 4. Load
# --------------------------------------------------------------------------
def load_tables(engine: Engine, data: dict[str, pd.DataFrame]) -> None:
    order = ["customers", "credit_history", "loan_applications", "transactions"]  # parents first
    for table in order:
        data[table].to_sql(table, engine, if_exists="append", index=False,
                           chunksize=2000, method="multi")
        log(f"  loaded {table:<20} {len(data[table]):>7,} rows")


# --------------------------------------------------------------------------
# 5. Validate
# --------------------------------------------------------------------------
def validate(engine: Engine) -> bool:
    ok = True
    with engine.connect() as conn:
        log("  Row counts:")
        for table, expected in EXPECTED_ROWS.items():
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            flag = "OK " if n == expected else "!! "
            ok &= n == expected
            log(f"    {flag}{table:<20} {n:>7,}  (expected {expected:,})")

        log("  Orphan foreign keys (must all be 0):")
        orphan_sql = {
            "credit_history": "SELECT COUNT(*) FROM credit_history t LEFT JOIN customers c USING (customer_id) WHERE c.customer_id IS NULL",
            "loan_applications": "SELECT COUNT(*) FROM loan_applications t LEFT JOIN customers c USING (customer_id) WHERE c.customer_id IS NULL",
            "transactions": "SELECT COUNT(*) FROM transactions t LEFT JOIN customers c USING (customer_id) WHERE c.customer_id IS NULL",
        }
        for table, q in orphan_sql.items():
            n = conn.execute(text(q)).scalar()
            ok &= n == 0
            log(f"    {'OK ' if n == 0 else '!! '}{table:<20} {n}")

        log("  Portfolio metrics vs PRD:")
        decided, approved = conn.execute(text(
            "SELECT SUM(approval_status IN ('Approved','Rejected')), SUM(approval_status='Approved') "
            "FROM loan_applications")).one()
        disbursed, default_rate, disb_value = conn.execute(text(
            "SELECT COUNT(*), AVG(loan_default), SUM(approved_amount) "
            "FROM loan_applications WHERE disbursed_flag = 1")).one()
        log(f"    approval rate (decided apps) : {100 * approved / decided:.1f}%   (PRD: 46.4%)")
        log(f"    disbursed loans              : {int(disbursed):,}      (PRD: 3,403)")
        log(f"    observed default rate        : {100 * float(default_rate):.1f}%   (PRD: 23.4%)")
        log(f"    total disbursed value        : INR {float(disb_value) / 1e7:,.0f} crore  (PRD: ~441)")
    return ok


# --------------------------------------------------------------------------
def main() -> None:
    log(f"Credora data load  -  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("\n[1/5] Reading raw CSVs")
    raw = read_raw()

    log("\n[2/5] Cleaning")
    data = clean(raw)

    log("\n[3/5] Preparing MySQL")
    create_database(get_engine(with_database=False))
    engine = get_engine()
    check_mysql_version(engine)
    run_schema(engine)

    log("\n[4/5] Loading tables")
    load_tables(engine, data)

    log("\n[5/5] Validating")
    ok = validate(engine)

    log("\nRESULT: " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - read the lines marked !!"))
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text("\n".join(LOG_LINES), encoding="utf-8")
    print(f"\n(log saved to {LOG_FILE.relative_to(ROOT)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
