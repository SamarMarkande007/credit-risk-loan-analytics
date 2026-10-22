"""
train.py  -  Train, tune, compare and save the loan-default model.

Run the whole pipeline with one command (reproducible: fixed seeds):

    python src/train.py

What it does
  1. Builds the modelling table (src/features.py)  - 3,403 disbursed loans
  2. Splits 80/20, stratified on the target AND grouped by customer
     (a customer's loans never appear in both train and test)
  3. Tunes 4 models with 5-fold grouped CV on the TRAIN set only:
        Logistic Regression, Decision Tree, Random Forest, XGBoost
  4. Picks the champion by cross-validated ROC-AUC
  5. Calibrates probabilities (Platt scaling on out-of-fold predictions), because
     class weighting makes raw probabilities too high for a 0-100 risk score
  6. Chooses the decision threshold on TRAIN out-of-fold data (never on test):
     the highest threshold that still reaches the PRD recall target (0.70)
  7. Evaluates ONCE on the held-out test set
  8. Saves models/default_model.pkl, model_comparison.csv, feature_importance.csv,
     metrics.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ALL_FEATURES, build_features, get_xy, make_preprocessor  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
N_SPLITS = 5            # 5-fold CV; the first fold also gives the 80/20 train/test split
RECALL_TARGET = 0.70    # PRD success metric
AUC_TARGET = 0.80       # PRD success metric (reported honestly even if missed)
N_ITER = 15             # random-search candidates per model
N_ITER_OVERRIDE = {"Random Forest": 8}   # slowest model: fewer candidates keeps training well under 5 min


# --------------------------------------------------------------------------
# Models and their search spaces
# --------------------------------------------------------------------------
def model_zoo(pos_weight: float) -> dict:
    """name -> (pipeline, hyper-parameter search space).  Class imbalance handled per model."""
    return {
        "Logistic Regression": (
            Pipeline([("prep", make_preprocessor(scale=True)),
                      ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            {"model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        ),
        "Decision Tree": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
            {"model__max_depth": [3, 4, 5, 6, 8],
             "model__min_samples_leaf": [10, 20, 40, 80]},
        ),
        "Random Forest": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("model", RandomForestClassifier(class_weight="balanced_subsample",
                                                       random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [150, 300],
             "model__max_depth": [4, 6, 8, None],
             "model__min_samples_leaf": [5, 10, 20],
             "model__max_features": ["sqrt", 0.5]},
        ),
        "XGBoost": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("model", XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                              random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [150, 300, 500],
             "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.02, 0.05, 0.1],
             "model__subsample": [0.7, 0.9],
             "model__colsample_bytree": [0.6, 0.9],
             "model__min_child_weight": [1, 5, 10]},
        ),
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def split_train_test(X, y, groups):
    """80/20 split: stratified on y and grouped by customer (no customer in both sets)."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]), "customer leakage across split"
    return train_idx, test_idx


def metrics_at(y_true, proba, threshold: float) -> dict:
    """All PRD metrics for a given decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pick_threshold(y_true, proba, recall_target: float = RECALL_TARGET) -> float:
    """Highest threshold whose recall still meets the target (= best precision at that recall)."""
    best = 0.0
    for t in np.unique(np.round(proba, 4)):
        if recall_score(y_true, (proba >= t).astype(int)) >= recall_target:
            best = max(best, t)
    return float(best)


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Built-in importance of the final estimator, mapped to the encoded feature names."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    est = pipeline.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values, kind = est.feature_importances_, "feature_importances_"
    else:  # logistic regression: size of the standardised coefficient
        values, kind = np.abs(est.coef_.ravel()), "abs(coefficient)"
    out = pd.DataFrame({"feature": names, "importance": values})
    out["importance_pct"] = 100 * out["importance"] / out["importance"].sum()
    out["method"] = kind
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_training(save: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    # 1-2. data + split ------------------------------------------------------
    df = build_features()
    X, y, groups = get_xy(df)
    tr, te = split_train_test(X, y, groups)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    g_tr = groups.iloc[tr]
    log(f"Train: {len(tr):,} loans ({y_tr.mean() * 100:.1f}% default)   "
        f"Test: {len(te):,} loans ({y_te.mean() * 100:.1f}% default)   [no shared customers]")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X_tr, y_tr, g_tr))  # fixed folds -> every model sees identical splits
    pos_weight = float((1 - y_tr.mean()) / y_tr.mean())

    # 3. tune every model ---------------------------------------------------
    searches, rows = {}, []
    for name, (pipe, space) in model_zoo(pos_weight).items():
        t1 = time.time()
        n_iter = min(N_ITER_OVERRIDE.get(name, N_ITER), int(np.prod([len(v) for v in space.values()])))
        rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv_splits,
                                random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        rs.fit(X_tr, y_tr)
        searches[name] = rs
        # out-of-fold probabilities of the best config (for recall + threshold, still TRAIN only)
        oof = cross_val_predict(clone(rs.best_estimator_), X_tr, y_tr, cv=cv_splits, method="predict_proba")[:, 1]
        rows.append({"model": name, "cv_roc_auc": rs.best_score_,
                     "cv_auc_std": rs.cv_results_["std_test_score"][rs.best_index_],
                     "cv_recall@0.5": recall_score(y_tr, (oof >= 0.5).astype(int)),
                     "best_params": {k.replace("model__", ""): v for k, v in rs.best_params_.items()}})
        log(f"  {name:<20} CV ROC-AUC {rs.best_score_:.3f} +/- {rows[-1]['cv_auc_std']:.3f}   ({time.time() - t1:.0f}s)")

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    # 4. champion --------------------------------------------------------------
    champ_name = comparison.loc[0, "model"]
    champion = searches[champ_name].best_estimator_
    log(f"\nChampion: {champ_name}")

    # 5. calibration (Platt scaling fitted on TRAIN out-of-fold probabilities) --
    oof_raw = cross_val_predict(clone(champion), X_tr, y_tr, cv=cv_splits, method="predict_proba")[:, 1]
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof_raw).reshape(-1, 1), y_tr)
    calibrate = lambda p: calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    oof_cal = calibrate(oof_raw)

    # 6. threshold on TRAIN OOF (never on the test set) ------------------------
    threshold = pick_threshold(y_tr, oof_cal)
    log(f"Decision threshold (calibrated P) = {threshold:.3f}  -> recall >= {RECALL_TARGET} on train OOF")

    # 7. ONE evaluation on the test set -----------------------------------------
    p_te = calibrate(champion.predict_proba(X_te)[:, 1])
    test_default = metrics_at(y_te, p_te, 0.5)
    test_tuned = metrics_at(y_te, p_te, threshold)
    # baseline: predict "no default" for everyone
    baseline_acc = 1 - y_te.mean()

    log("\nTEST SET (touched once)")
    log(f"  {'':<22}{'ROC-AUC':>8}{'Accuracy':>10}{'Precision':>10}{'Recall':>8}{'F1':>7}")
    for label, m in [("threshold 0.50", test_default), (f"threshold {threshold:.2f} (tuned)", test_tuned)]:
        log(f"  {label:<22}{m['roc_auc']:>8.3f}{m['accuracy']:>10.3f}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}")
    log(f"  'predict nobody defaults' baseline accuracy: {baseline_acc:.3f}  (recall 0)")
    log(f"  PRD targets: ROC-AUC >= {AUC_TARGET}: {'MET' if test_tuned['roc_auc'] >= AUC_TARGET else 'NOT met'}  |  "
        f"Recall >= {RECALL_TARGET}: {'MET' if test_tuned['recall'] >= RECALL_TARGET else 'NOT met'}")

    importance = feature_importance(champion)

    metrics = {
        "champion": champ_name,
        "random_state": RANDOM_STATE,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_default_rate": float(y_tr.mean()), "test_default_rate": float(y_te.mean()),
        "threshold": threshold,
        "test_at_0.5": test_default, "test_at_tuned_threshold": test_tuned,
        "baseline_accuracy": float(baseline_acc),
        "targets": {"roc_auc": AUC_TARGET, "recall": RECALL_TARGET},
        "targets_met": {"roc_auc": bool(test_tuned["roc_auc"] >= AUC_TARGET),
                        "recall": bool(test_tuned["recall"] >= RECALL_TARGET)},
        "cv_comparison": comparison.assign(best_params=comparison["best_params"].astype(str)).to_dict("records"),
    }

    # 8. save ---------------------------------------------------------------------
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        bundle = {"pipeline": champion, "calibrator": calibrator, "threshold": threshold,
                  "features": ALL_FEATURES, "champion": champ_name, "metrics": metrics}
        joblib.dump(bundle, MODELS_DIR / "default_model.pkl")
        comparison.drop(columns="best_params").assign(
            best_params=comparison["best_params"].astype(str)).to_csv(MODELS_DIR / "model_comparison.csv", index=False)
        importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
        (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        log(f"\nSaved to models/: default_model.pkl, model_comparison.csv, feature_importance.csv, metrics.json")
    log(f"Total time: {time.time() - t0:.0f}s")

    return {"df": df, "X": X, "y": y, "groups": groups, "train_idx": tr, "test_idx": te,
            "searches": searches, "comparison": comparison, "champion_name": champ_name,
            "champion": champion, "calibrator": calibrator, "threshold": threshold,
            "oof_train_cal": oof_cal, "p_test": p_te, "y_train": y_tr, "y_test": y_te,
            "X_train": X_tr, "X_test": X_te, "importance": importance, "metrics": metrics}


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    run_training()
