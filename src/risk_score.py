"""
risk_score.py  -  Turn default probabilities into a 0-100 risk score and Low/Medium/High
                  labels, and write them to MySQL for the dashboard.

Run after training:

    python src/risk_score.py

Output tables (rebuilt on every run)
    scored_applications : one row per loan application (7,500)
    scored_customers    : one row per customer (5,000)   <- feeds the dashboard

Key design points
  * risk_score = round(100 x calibrated P(default))                       (PRD 13.7)
  * Risk bands are TUNED (Low <= 18, Medium 19-30, High >= 31).  The PRD's fixed
    0-33 / 34-66 / 67-100 bands put ~79% of loans in "Low" for this model, so they
    do not separate risk.  The PRD allows tuning "against the confusion matrix and
    business risk appetite".  Change LOW_MAX / MEDIUM_MAX below to change appetite.
  * Honest scores: loans the model was trained on are scored with CROSS-FITTED
    (out-of-fold) predictions, so a loan is never scored by a model that saw it.
  * Customer score = mean probability across the customer's applications.
  * 1,096 customers never applied for a loan.  They are scored from their bureau,
    demographic and transaction profile only (loan-specific fields are filled with
    portfolio medians) and labelled scoring_basis = 'profile only'.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import get_engine  # noqa: E402
from features import (ALL_FEATURES, TARGET, aggregate_transactions, build_features,  # noqa: E402
                      get_xy, load_raw)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "default_model.pkl"

# ---- risk appetite (tune here) ------------------------------------------------
LOW_MAX = 18        # score <= 18          -> Low
MEDIUM_MAX = 30     # 19 <= score <= 30    -> Medium ;  score >= 31 -> High
PRD_BANDS = (33, 66)  # the PRD's original fixed bands, kept for comparison
CATEGORIES = ["Low", "Medium", "High"]
RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------------------------------
# Score and band transforms
# --------------------------------------------------------------------------
def probability_to_score(p) -> np.ndarray:
    """round(100 x P) as an integer 0-100."""
    return np.clip(np.round(np.asarray(p, dtype=float) * 100), 0, 100).astype(int)


def score_to_category(score, low_max: int = LOW_MAX, medium_max: int = MEDIUM_MAX) -> pd.Categorical:
    """Map 0-100 scores to Low / Medium / High using the given upper limits."""
    return pd.cut(np.asarray(score), bins=[-1, low_max, medium_max, 100], labels=CATEGORIES)


# --------------------------------------------------------------------------
# Model loading and prediction
# --------------------------------------------------------------------------
def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        sys.exit(f"{path} not found - run  python src/train.py  first.")
    return joblib.load(path)


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibrate(bundle: dict, raw_proba) -> np.ndarray:
    """Platt scaling learned in train.py: makes probabilities match real default frequencies."""
    return bundle["calibrator"].predict_proba(_logit(raw_proba).reshape(-1, 1))[:, 1]


def score_applications(bundle: dict, apps: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrated P(default) for every application.
    Disbursed loans (the model's training population) get cross-fitted out-of-fold
    predictions; all other applications are scored by the final model.
    """
    pipe = bundle["pipeline"]
    proba = np.full(len(apps), np.nan)
    oof_flag = np.zeros(len(apps), dtype=int)

    labelled = apps[TARGET].notna().to_numpy()
    if labelled.any():
        Xl, yl, gl = get_xy(apps[labelled])
        idx = np.flatnonzero(labelled)
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for tr, te in sgkf.split(Xl, yl, gl):
            model = clone(pipe).fit(Xl.iloc[tr], yl.iloc[tr])
            proba[idx[te]] = model.predict_proba(Xl.iloc[te])[:, 1]
        oof_flag[labelled] = 1
    if (~labelled).any():
        proba[~labelled] = pipe.predict_proba(apps.loc[~labelled, ALL_FEATURES])[:, 1]

    out = apps[["application_id", "customer_id"]].copy()
    out["default_probability"] = calibrate(bundle, proba)
    out["risk_score"] = probability_to_score(out["default_probability"])
    out["risk_category"] = score_to_category(out["risk_score"]).astype(str)
    out["actual_default"] = apps[TARGET].astype("Int64")
    out["is_out_of_fold"] = oof_flag
    return out


def _profile_only_rows(raw: dict, customer_ids: list[str]) -> pd.DataFrame:
    """Feature rows for customers with no loan application (loan-specific fields left empty)."""
    cust = raw["customers"][raw["customers"]["customer_id"].isin(customer_ids)]
    df = (cust[["customer_id", "age", "dependents", "education", "employment_type", "annual_income",
                "residence_type", "years_at_residence", "customer_since"]]
          .merge(raw["credit_history"].drop(columns=["credit_id"]), on="customer_id", how="left", validate="1:1")
          .merge(aggregate_transactions(raw["transactions"]), on="customer_id", how="left", validate="1:1"))
    df["tenure_years"] = (df["as_of_date"] - df["customer_since"]).dt.days / 365.25
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in ["requested_amount", "interest_rate", "loan_term_months", "debt_to_income_ratio",
                "existing_emi", "loan_to_income", "emi_to_income", "loan_type"]:
        df[col] = np.nan                      # imputed with portfolio medians / 'Unknown' by the pipeline
    df["collateral_flag"] = 0                 # assume no collateral (the pipeline does not impute flags)
    return df


def score_customers(bundle: dict, raw: dict, app_scores: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: mean probability over their applications, or profile-only if none."""
    with_apps = (app_scores.groupby("customer_id")
                 .agg(n_applications=("application_id", "size"), default_probability=("default_probability", "mean"))
                 .reset_index().assign(scoring_basis="applications"))

    all_ids = raw["customers"]["customer_id"].tolist()
    missing = sorted(set(all_ids) - set(with_apps["customer_id"]))
    if missing:
        prof = _profile_only_rows(raw, missing)
        p = calibrate(bundle, bundle["pipeline"].predict_proba(prof[ALL_FEATURES])[:, 1])
        no_apps = pd.DataFrame({"customer_id": prof["customer_id"], "n_applications": 0,
                                "default_probability": p, "scoring_basis": "profile only"})
        with_apps = pd.concat([with_apps, no_apps], ignore_index=True)

    out = with_apps.copy()
    out["risk_score"] = probability_to_score(out["default_probability"])
    out["risk_category"] = score_to_category(out["risk_score"]).astype(str)
    out = out.sort_values("customer_id").reset_index(drop=True)
    return out[["customer_id", "n_applications", "default_probability", "risk_score",
                "risk_category", "scoring_basis"]]


# --------------------------------------------------------------------------
# MySQL output
# --------------------------------------------------------------------------
DDL = [
    "DROP TABLE IF EXISTS scored_applications",
    "DROP TABLE IF EXISTS scored_customers",
    """CREATE TABLE scored_customers (
        customer_id          VARCHAR(12)  NOT NULL,
        n_applications       SMALLINT     NOT NULL,
        default_probability  DECIMAL(6,4) NOT NULL,
        risk_score           SMALLINT     NOT NULL,
        risk_category        VARCHAR(10)  NOT NULL,
        scoring_basis        VARCHAR(20)  NOT NULL,
        PRIMARY KEY (customer_id),
        CONSTRAINT fk_sc_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
        CONSTRAINT chk_sc_score CHECK (risk_score BETWEEN 0 AND 100),
        CONSTRAINT chk_sc_cat   CHECK (risk_category IN ('Low','Medium','High'))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE scored_applications (
        application_id       VARCHAR(12)  NOT NULL,
        customer_id          VARCHAR(12)  NOT NULL,
        default_probability  DECIMAL(6,4) NOT NULL,
        risk_score           SMALLINT     NOT NULL,
        risk_category        VARCHAR(10)  NOT NULL,
        actual_default       SMALLINT     NULL,
        is_out_of_fold       SMALLINT     NOT NULL,
        PRIMARY KEY (application_id),
        CONSTRAINT fk_sa_app FOREIGN KEY (application_id) REFERENCES loan_applications (application_id) ON DELETE CASCADE,
        CONSTRAINT fk_sa_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
        CONSTRAINT chk_sa_score CHECK (risk_score BETWEEN 0 AND 100),
        CONSTRAINT chk_sa_cat   CHECK (risk_category IN ('Low','Medium','High'))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def write_to_mysql(engine, customers: pd.DataFrame, apps: pd.DataFrame) -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    apps_out = apps.astype(object).where(apps.notna(), None)
    apps_out.to_sql("scored_applications", engine, if_exists="append", index=False, chunksize=2000, method="multi")
    customers.to_sql("scored_customers", engine, if_exists="append", index=False, chunksize=2000, method="multi")


# --------------------------------------------------------------------------
def run_scoring(write: bool = True, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    bundle = load_model()
    raw = load_raw()
    apps_feat = build_features(raw, only_disbursed=False)

    app_scores = score_applications(bundle, apps_feat)
    cust_scores = score_customers(bundle, raw, app_scores)

    if write:
        write_to_mysql(get_engine(), cust_scores, app_scores)
        log("Wrote tables: scored_applications, scored_customers")

    if verbose:
        log(f"\nModel: {bundle['champion']}   bands: Low <= {LOW_MAX}, Medium <= {MEDIUM_MAX}, High > {MEDIUM_MAX}")
        log(f"Applications scored: {len(app_scores):,}   Customers scored: {len(cust_scores):,}")
        mix = cust_scores["risk_category"].value_counts().reindex(CATEGORIES)
        log("\nCustomer risk mix:  " + "   ".join(f"{k} {v:,} ({v / len(cust_scores) * 100:.1f}%)" for k, v in mix.items()))
        lab = app_scores[app_scores["actual_default"].notna()]
        t = lab.groupby("risk_category")["actual_default"].agg(loans="size", default_rate="mean").reindex(CATEGORIES)
        log("\nHonest check on disbursed loans (out-of-fold scores vs actual outcomes):")
        for k, r in t.iterrows():
            log(f"  {k:<7} {int(r['loans']):>5,} loans   actual default rate {r['default_rate'] * 100:5.1f}%")
    return {"bundle": bundle, "raw": raw, "apps_feat": apps_feat,
            "app_scores": app_scores, "cust_scores": cust_scores}


if __name__ == "__main__":
    run_scoring()
