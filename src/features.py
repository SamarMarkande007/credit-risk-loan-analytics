"""
features.py  -  Turn the four MySQL tables into one clean modelling table.

What this module gives you
    load_raw()                -> the 4 tables from MySQL
    aggregate_transactions()  -> ONE row per customer from the transaction table
    build_features()          -> one row per loan: target + engineered features
    get_xy()                  -> X, y, groups ready for train/test splitting
    make_preprocessor()       -> imputation + scaling + one-hot encoding (sklearn)

Run it directly to build the table and print a validation report:

    python src/features.py

Design decisions (all backed by the EDA notebook)
    * Train only on disbursed loans - the only rows where the outcome is known.
    * Transactions are aggregated to customer level BEFORE joining (no row duplication).
    * Post-decision columns are excluded (leakage); PII and names are excluded.
    * Redundant columns are dropped: monthly income (= annual / 12), and
      oldest_account_age_months (~0.99 correlated with age).
    * debt_to_limit was tried and dropped: it equals credit_utilization_ratio
      (outstanding debt / credit limit) to within rounding, so it added nothing.
    * Gender / marital status / city / state are NOT used as model inputs.
    * Tenure is measured at the bureau snapshot date, because ~12% of customers
      have customer_since AFTER their application date (negative tenure otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets notebooks import this file too
from data_load import get_engine  # noqa: E402  (re-use the same .env based connection)

# --------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------
TARGET = "loan_default"
ID_COLS = ["application_id", "customer_id", "application_date"]

# Only known AFTER the underwriting decision -> would leak the answer into the model
LEAKAGE_COLS = ["approval_status", "approved_amount", "decision_date", "disbursed_flag"]
# Personal data: not predictive, and must not appear in models or dashboards
PII_COLS = ["first_name", "last_name", "email", "phone", "date_of_birth"]

DEMOGRAPHIC = ["age", "dependents", "years_at_residence", "tenure_years"]
BUREAU = ["credit_score", "credit_utilization_ratio", "num_late_payments_30d",
          "num_late_payments_90d", "num_defaults_prior", "bankruptcies",
          "num_credit_inquiries_6m", "num_open_accounts", "total_credit_limit",
          "total_outstanding_debt"]
APPLICATION = ["annual_income", "requested_amount", "interest_rate", "loan_term_months",
               "debt_to_income_ratio", "existing_emi", "loan_to_income", "emi_to_income"]
TRANSACTION = ["txn_count", "total_inflow", "total_outflow", "net_cash_flow",
               "emi_outflow", "avg_balance", "inflow_outflow_ratio"]

NUMERIC_FEATURES = DEMOGRAPHIC + BUREAU + APPLICATION + TRANSACTION
BINARY_FEATURES = ["collateral_flag"]
CATEGORICAL_FEATURES = ["loan_type", "employment_type", "residence_type", "education"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

FEATURE_GROUPS = {
    "Demographic": DEMOGRAPHIC + ["employment_type", "residence_type", "education"],
    "Credit bureau": BUREAU,
    "Application": APPLICATION + ["loan_type", "collateral_flag"],
    "Transaction behaviour": TRANSACTION,
}


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_raw(engine=None) -> dict[str, pd.DataFrame]:
    """Read the four tables from MySQL and convert DATE columns to datetimes."""
    engine = engine or get_engine()
    raw = {t: pd.read_sql(f"SELECT * FROM {t}", engine)
           for t in ["customers", "credit_history", "loan_applications", "transactions"]}
    for table, cols in {"customers": ["date_of_birth", "customer_since"],
                        "credit_history": ["as_of_date"],
                        "loan_applications": ["application_date", "decision_date"],
                        "transactions": ["transaction_date"]}.items():
        for c in cols:
            raw[table][c] = pd.to_datetime(raw[table][c])
    return raw


# --------------------------------------------------------------------------
# 2. Transactions -> one row per customer
# --------------------------------------------------------------------------
def aggregate_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse ~40k transactions into one row per customer (the last 6 months).
    Joining raw transactions to loans would multiply every loan row, which is the
    most common leakage/duplication mistake in this project.
    """
    t = txn.assign(
        inflow=txn["amount"].where(txn["transaction_type"] == "Credit", 0.0),
        outflow=txn["amount"].where(txn["transaction_type"] == "Debit", 0.0),
        emi_outflow=txn["amount"].where(txn["category"] == "EMI", 0.0),
    )
    agg = t.groupby("customer_id").agg(
        txn_count=("amount", "size"),
        total_inflow=("inflow", "sum"),
        total_outflow=("outflow", "sum"),
        emi_outflow=("emi_outflow", "sum"),
        avg_balance=("balance_after", "mean"),
    ).reset_index()
    agg["net_cash_flow"] = agg["total_inflow"] - agg["total_outflow"]
    # outflow == 0 -> ratio undefined -> NaN (filled later by the median imputer)
    agg["inflow_outflow_ratio"] = agg["total_inflow"] / agg["total_outflow"].replace(0, np.nan)
    return agg


# --------------------------------------------------------------------------
# 3. Build the modelling table
# --------------------------------------------------------------------------
def build_features(raw: dict[str, pd.DataFrame] | None = None,
                   only_disbursed: bool = True) -> pd.DataFrame:
    """
    Return one row per loan application with: ID columns, the target and every feature.

    only_disbursed=True  -> training population (target known, 3,403 rows)
    only_disbursed=False -> all 7,500 applications (target is NaN where unknown);
                            used later to score applicants.
    """
    raw = raw or load_raw()
    loans = raw["loan_applications"]
    if only_disbursed:
        loans = loans[(loans["disbursed_flag"] == 1) & loans[TARGET].notna()]

    cust_cols = ["customer_id", "age", "dependents", "education", "employment_type",
                 "annual_income", "residence_type", "years_at_residence", "customer_since"]
    credit_cols = ["customer_id", "as_of_date", "credit_score", "num_open_accounts",
                   "num_credit_inquiries_6m", "credit_utilization_ratio",
                   "num_late_payments_30d", "num_late_payments_90d", "num_defaults_prior",
                   "bankruptcies", "total_credit_limit", "total_outstanding_debt"]

    # validate="m:1" makes pandas RAISE an error if any join would duplicate loan rows
    df = (loans
          .merge(raw["customers"][cust_cols], on="customer_id", how="left", validate="m:1")
          .merge(raw["credit_history"][credit_cols], on="customer_id", how="left", validate="m:1")
          .merge(aggregate_transactions(raw["transactions"]), on="customer_id",
                 how="left", validate="m:1"))

    # ---- engineered features ------------------------------------------------
    df["tenure_years"] = (df["as_of_date"] - df["customer_since"]).dt.days / 365.25
    df["loan_to_income"] = df["requested_amount"] / df["annual_income"]
    df["emi_to_income"] = df["existing_emi"] / df["applicant_monthly_income"]

    df = df.replace([np.inf, -np.inf], np.nan)  # division by zero -> NaN (imputed later)

    out = df[ID_COLS + [TARGET] + ALL_FEATURES].reset_index(drop=True)
    _check(out, only_disbursed)
    return out


def _check(df: pd.DataFrame, only_disbursed: bool) -> None:
    """Hard guarantees - fail loudly instead of silently training on bad data."""
    banned = set(LEAKAGE_COLS + PII_COLS) & set(ALL_FEATURES)
    assert not banned, f"Leakage/PII columns in feature list: {banned}"
    assert df["application_id"].is_unique, "duplicate loan rows after joins"
    if only_disbursed:
        assert df[TARGET].notna().all(), "missing target in training population"


def get_xy(df: pd.DataFrame):
    """Split the table into X (features), y (0/1 target) and groups (customer ids)."""
    return df[ALL_FEATURES].copy(), df[TARGET].astype(int), df["customer_id"]


# --------------------------------------------------------------------------
# 4. Preprocessing pipeline (fit on TRAIN data only, inside the model pipeline)
# --------------------------------------------------------------------------
def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    """
    Missing values + encoding, reproducibly.
        numeric     : median imputation  (+ standardisation when scale=True)
        categorical : missing -> 'Unknown', then one-hot encoding
        binary flag : passed through unchanged
    scale=True  for Logistic Regression (sensitive to feature scale).
    scale=False for trees / Random Forest / XGBoost (scale-free).
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    numeric = Pipeline(steps)

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES),
         ("cat", categorical, CATEGORICAL_FEATURES),
         ("bin", "passthrough", BINARY_FEATURES)],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")  # keep column names (needed for feature importance / SHAP)


# --------------------------------------------------------------------------
# 5. Validation report
# --------------------------------------------------------------------------
def main() -> None:
    df = build_features()
    X, y, groups = get_xy(df)

    print(f"Modelling table : {df.shape[0]:,} loans x {len(ALL_FEATURES)} features")
    print(f"Target          : {y.mean() * 100:.1f}% default  ({y.sum():,} defaults / {(1 - y).sum():,.0f} repaid)")
    print(f"Customers       : {groups.nunique():,} unique  ({groups.duplicated().sum():,} extra loan rows -> split by customer)")
    print("\nFeature groups:")
    for name, cols in FEATURE_GROUPS.items():
        print(f"  {name:<22}{len(cols):>3}  {', '.join(cols)}")

    miss = X.isna().sum()
    miss = miss[miss > 0]
    print("\nMissing values before imputation:",
          "none" if miss.empty else miss.to_dict())

    for scale in (True, False):
        Xt = make_preprocessor(scale=scale).fit_transform(X)
        assert not Xt.isna().any().any(), "NaN left after preprocessing"
        print(f"Preprocessor (scale={scale!s:<5}) -> {Xt.shape[1]} columns, 0 NaN")

    print("\nChecks passed: no duplicate loans, no leakage/PII columns, no NaN after preprocessing.")


if __name__ == "__main__":
    main()
