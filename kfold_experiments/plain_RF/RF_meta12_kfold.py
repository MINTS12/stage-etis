"""
RF_meta12_kfold.py

Native multi-output RandomForest baseline (12 metacategory odor labels),
run through the same k=5 IterativeStratification fold splits used by
baseline_meta12_kfold.py (BR+RF) and the HMCN-F kfold pipeline, so this
number is directly comparable to both.

Unlike BR+RF (skmultilearn.problem_transform.BinaryRelevance, which trains
12 fully independent RandomForestClassifiers), this fits ONE
RandomForestClassifier per fold on the full (n_samples, 12) label matrix
directly -- sklearn's tree ensembles support multi-output natively, so
splits are chosen jointly across all 12 labels rather than in isolation.

CONFIRMED: SPLIT_PATH pickle is a list of N_FOLDS tuples
  (train_idx, val_idx, test_idx) -- numpy arrays of row indices into the
  dataframe loaded from DATASET_PATH, sizes 3379/614/983 (~68/12/20).

REMAINING ASSUMPTION TO VERIFY:
  1. The metric functions below (macro_pr_auc, macro_roc_auc,
     find_optimal_thresholds, macro_f1_at_thresholds) reimplement the
     logic described for hmcn_eval.py (macro averaging, skip constant
     columns, per-label tau* via F1 sweep on validation only). If you'd
     rather call the real hmcn_eval functions, replace this block with
     `from hmcn_eval import ...` and adjust the calls in run_fold().

Usage (on espadon1):
    nohup python3 -u RF_meta12_kfold.py > rf_meta12_stdout.log 2>&1 &

Run a quick smoke test first:
    Set DEBUG = True below, run directly (not under nohup), confirm it
    finishes in seconds with sane-looking numbers, then set DEBUG = False
    for the real run.
"""

import logging
import pickle
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# ============================================================
# CONFIG
# ============================================================
DEBUG = False              # True -> quick random-split smoke test, no pickle needed
DEBUG_N_ROWS = 300

DATASET_PATH = Path("hmcn_dataset.csv")                 # comma-delimited
SPLIT_PATH   = Path("hmcn_kfold_split_indices.pkl")
OUTPUT_DIR   = Path("kfold_results")
LOG_PATH     = Path("RF_meta12_kfold.log")

LABEL_PREFIX = "meta_"     # 12 metacategories
N_FOLDS = 5

ID_COLUMNS = ("SMILES",)
NON_FEATURE_PREFIXES = ("fine_", "meta_")

# Hyperparameter grid -- matches the BR+RF grid for apples-to-apples comparison
N_ESTIMATORS_GRID = [100, 300]
MAX_FEATURES_GRID = ["sqrt", "log2"]

THRESH_CANDIDATES = np.linspace(0.05, 0.95, 19)   # same sweep as HMCN-F calibration

RANDOM_STATE = 42
N_JOBS = -1                 # shared server -- lower this if you want to be polite to other users

# ============================================================
# LOGGING  (FileHandler, not print -- survives nohup buffering)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w")],
)
log = logging.getLogger(__name__)


# ============================================================
# METRICS
# ============================================================

def macro_pr_auc(y_true, y_prob):
    """Macro-averaged PR-AUC, skipping labels with zero positives in y_true."""
    aps = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        aps.append(average_precision_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aps)), len(aps)


def macro_roc_auc(y_true, y_prob):
    """Macro-averaged ROC-AUC, skipping constant columns (undefined AUC)."""
    aucs = []
    for i in range(y_true.shape[1]):
        pos = y_true[:, i].sum()
        if pos == 0 or pos == y_true.shape[0]:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aucs)), len(aucs)


def find_optimal_thresholds(y_true, y_prob):
    """Per-label tau* maximizing F1, searched on validation set only."""
    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5)
    for i in range(n_labels):
        if y_true[:, i].sum() == 0:
            continue
        best_f1, best_tau = 0.0, 0.5
        for tau in THRESH_CANDIDATES:
            f1 = f1_score(y_true[:, i], (y_prob[:, i] >= tau).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        thresholds[i] = best_tau
    return thresholds


def macro_f1_at_thresholds(y_true, y_prob, thresholds):
    f1s = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        pred = (y_prob[:, i] >= thresholds[i]).astype(int)
        f1s.append(f1_score(y_true[:, i], pred, zero_division=0))
    return float(np.mean(f1s))


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    log.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, sep=",")

    label_cols = [c for c in df.columns if c.startswith(LABEL_PREFIX)]
    feature_cols = [
        c for c in df.columns
        if c not in ID_COLUMNS and not c.startswith(NON_FEATURE_PREFIXES)
    ]
    log.info(f"  {len(feature_cols)} feature columns, {len(label_cols)} label columns ({LABEL_PREFIX}*)")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y = df[label_cols].to_numpy(dtype=np.int32)
    return X, Y, label_cols


def load_splits():
    log.info(f"Loading cached fold indices from {SPLIT_PATH}")
    with open(SPLIT_PATH, "rb") as f:
        splits = pickle.load(f)
    return splits


def make_debug_split(n_total, n_debug):
    """Quick random 70/15/15 split on a subsample -- for smoke testing only,
    NOT for real results (ignores the cached fold indices on purpose)."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n_total, size=min(n_debug, n_total), replace=False)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train_idx": idx[:n_train],
        "val_idx": idx[n_train:n_train + n_val],
        "test_idx": idx[n_train + n_val:],
    }


# ============================================================
# TRAIN / EVAL ONE FOLD, ONE CONFIG
# ============================================================

def run_fold(X, Y, train_idx, val_idx, test_idx, n_estimators, max_features):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    clf.fit(X[train_idx], Y[train_idx])

    def predict_proba_matrix(idx):
        # multi-output predict_proba returns a list of (n_samples, 2) arrays,
        # one per label -- stack the positive-class column from each
        probs_per_label = clf.predict_proba(X[idx])
        return np.column_stack([p[:, 1] for p in probs_per_label])

    val_prob = predict_proba_matrix(val_idx)
    test_prob = predict_proba_matrix(test_idx)

    val_pr_auc, _ = macro_pr_auc(Y[val_idx], val_prob)
    test_pr_auc, n_valid_pr = macro_pr_auc(Y[test_idx], test_prob)
    test_roc_auc, n_valid_roc = macro_roc_auc(Y[test_idx], test_prob)

    thresholds = find_optimal_thresholds(Y[val_idx], val_prob)
    test_f1 = macro_f1_at_thresholds(Y[test_idx], test_prob, thresholds)

    return {
        "val_PR_AUC": val_pr_auc,
        "test_PR_AUC": test_pr_auc,
        "test_ROC_AUC": test_roc_auc,
        "test_F1": test_f1,
        "n_valid_labels_PR": n_valid_pr,
        "n_valid_labels_ROC": n_valid_roc,
    }


# ============================================================
# MAIN
# ============================================================

def run_debug():
    log.info("=== DEBUG SMOKE TEST (random split, small subsample) ===")
    X, Y, label_cols = load_data()
    split = make_debug_split(len(X), DEBUG_N_ROWS)
    row = run_fold(X, Y, split["train_idx"], split["val_idx"], split["test_idx"],
                    n_estimators=100, max_features="sqrt")
    log.info(f"DEBUG result: {row}")
    print("DEBUG result:", row)
    log.info("=== DEBUG OK -- set DEBUG = False for the real run ===")


def main():
    if DEBUG:
        run_debug()
        return

    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, Y, label_cols = load_data()
    splits = load_splits()
    assert len(splits) == N_FOLDS, f"Expected {N_FOLDS} folds, got {len(splits)}"

    grid = list(product(N_ESTIMATORS_GRID, MAX_FEATURES_GRID))
    log.info(f"Running {len(grid)} configs x {N_FOLDS} folds = {len(grid) * N_FOLDS} runs")

    all_rows = []
    for n_estimators, max_features in grid:
        config_name = f"n{n_estimators}_{max_features}"
        log.info(f"--- Config {config_name} ---")

        for fold_i, split in enumerate(splits):
            fold_start = time.time()
            train_idx, val_idx, test_idx = split   # pickle stores (train_idx, val_idx, test_idx) tuples
            row = run_fold(
                X, Y,
                train_idx, val_idx, test_idx,
                n_estimators, max_features,
            )
            row.update({
                "config": config_name,
                "n_estimators": n_estimators,
                "max_features": max_features,
                "fold": fold_i,
            })
            all_rows.append(row)

            log.info(
                f"  fold {fold_i}: val PR-AUC={row['val_PR_AUC']:.4f}  "
                f"test PR-AUC={row['test_PR_AUC']:.4f}  "
                f"test ROC-AUC={row['test_ROC_AUC']:.4f}  "
                f"test F1={row['test_F1']:.4f}  "
                f"({time.time() - fold_start:.1f}s)"
            )

    results_df = pd.DataFrame(all_rows)
    per_run_path = OUTPUT_DIR / "RF_meta12_kfold_per_run.csv"
    results_df.to_csv(per_run_path, index=False)
    log.info(f"Saved per-run results to {per_run_path}")

    summary = (
        results_df.groupby("config")
        .agg(
            mean_val_PR_AUC=("val_PR_AUC", "mean"),
            std_val_PR_AUC=("val_PR_AUC", "std"),
            mean_test_PR_AUC=("test_PR_AUC", "mean"),
            std_test_PR_AUC=("test_PR_AUC", "std"),
            mean_test_ROC_AUC=("test_ROC_AUC", "mean"),
            std_test_ROC_AUC=("test_ROC_AUC", "std"),
            mean_test_F1=("test_F1", "mean"),
            std_test_F1=("test_F1", "std"),
        )
        .reset_index()
        .sort_values("mean_val_PR_AUC", ascending=False)
    )
    summary_path = OUTPUT_DIR / "RF_meta12_kfold_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved config summary to {summary_path}")

    winner = summary.iloc[0]
    log.info("=" * 60)
    log.info(f"WINNING CONFIG: {winner['config']}")
    log.info(f"  test PR-AUC  = {winner['mean_test_PR_AUC']:.4f} +/- {winner['std_test_PR_AUC']:.4f}")
    log.info(f"  test ROC-AUC = {winner['mean_test_ROC_AUC']:.4f} +/- {winner['std_test_ROC_AUC']:.4f}")
    log.info(f"  test F1      = {winner['mean_test_F1']:.4f} +/- {winner['std_test_F1']:.4f}")
    log.info(f"Total runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
