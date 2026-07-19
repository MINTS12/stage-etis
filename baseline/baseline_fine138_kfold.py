"""
Binary Relevance baseline on the 138 fine-grained odor descriptors -- k-fold version.

Changes from baseline_fine138.py:
    1. Single train/test split (iterative_train_test_split) replaced with the
       SAME 5-fold split used by the HMCN k-fold sweep: reads/writes
       `hmcn_kfold_split_indices.pkl` (built from `hmcn_dataset.csv`, stratified
       on the 138 fine labels via IterativeStratification(order=2), seeded).
       If HMCN already generated this file, it's loaded as-is -- both scripts
       then evaluate on the exact same molecules per fold, since both load the
       exact same CSV in the exact same row order. If missing, it's computed
       here fresh; the algorithm is deterministic given the seed, so either
       script can produce it first without producing a different split.
    2. Per-label thresholds are now calibrated on a held-out VAL fold, not on
       the train set the model was fit on. Train-set calibration is optimistic
       -- fitted models (especially RF/KNN) predict train rows more confidently
       than genuinely unseen ones, so a threshold picked to maximize F1 on train
       predictions isn't necessarily the threshold that maximizes F1 on new data.
    3. Zero-variance filtering, the Mordred correlation filter, and the
       StandardScaler are now all fit per fold, on that fold's train split only,
       and applied unchanged to that fold's val/test -- not fit once globally.
    4. Added Precision to the per-label metric set.
    5. LogisticRegression now gets random_state (the "saga" solver is
       stochastic -- randomized coordinate order -- and was previously unseeded).
    6. Restart-skip: on relaunch, already-completed (fold, Model) pairs found
       in the existing results CSV are skipped rather than rerun.
    7. All progress messages go through `logger.info(...)`, never `print()` --
       logging.FileHandler flushes after every line, so the log file updates
       live even under `nohup`, unlike buffered stdout redirection.

Usage:
    Smoke test:  DEBUG=True in CONFIG, then run (uses 1 fold, 1 model, tiny grid).
    Full run (server, background):
        nohup python3 -u baseline_fine138_kfold.py > /dev/null 2>&1 &
        (log file is the source of truth either way; /dev/null here is fine)
"""

import logging
import sys
import time
import random
from datetime import datetime
from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score, confusion_matrix
)
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance

# ======================================================================
# CONFIG
# ======================================================================
DEBUG = False  # True -> 1 fold, 1 model, tiny grid, for a smoke test

DATA_PATH    = "hmcn_dataset.csv"
OUTPUT_DIR   = Path("outputs_fine138_kfold")
LOG_DIR      = Path("logs")
RANDOM_STATE = 42
CV_FOLDS     = 3          # inner CV used by GridSearchCV, within each fold's train split
K            = 5
VAL_RATIO    = 0.15       # matches the HMCN k-fold split exactly
SPLIT_CACHE  = "hmcn_kfold_split_indices.pkl"

MODELS_TO_RUN = ["LR", "RF", "XGB", "SVM", "KNN"]

PARAM_GRIDS = {
    "LR":  {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    "RF":  {"classifier__n_estimators": [100, 300], "classifier__max_features": ["sqrt", "log2"]},
    "XGB": {"classifier__n_estimators": [100, 300], "classifier__max_depth": [3, 6]},
    "SVM": {"classifier__C": [0.1, 1, 10]},
    "KNN": {"classifier__n_neighbors": [5, 11, 21], "classifier__metric": ["euclidean", "cosine"]},
}

if DEBUG:
    OUTPUT_DIR = Path("outputs_fine138_kfold_debug")
    MODELS_TO_RUN = ["LR"]
    PARAM_GRIDS["LR"] = {"classifier__C": [0.1]}
    CV_FOLDS = 2
    DEBUG_N_FOLDS = 1     # only run fold 0, not a row subsample -- see module docstring
    DEBUG_N_LABELS = 6

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# ======================================================================
# REPRODUCIBILITY
# ======================================================================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)   # also fixes skmultilearn's IterativeStratification, whose
                            # internal tie-breaking falls back to the global numpy RNG
                            # regardless of what random_state is passed

set_seed(RANDOM_STATE)

# ======================================================================
# LOGGING (file + stdout, auto-timestamped, flushes every line)
# ======================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"baseline_fine138_kfold_{'debug' if DEBUG else 'full'}_{timestamp}.log"

logger = logging.getLogger("baseline_fine138_kfold")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

fh = logging.FileHandler(log_path)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

logger.info(f"DEBUG={DEBUG} | log file: {log_path}")
SCRIPT_START_TIME = datetime.now()
logger.info(f"Script started at: {SCRIPT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")

# ======================================================================
# LOAD (full dataset, no split yet -- splitting happens via cached fold indices)
# ======================================================================
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {DATA_PATH}: {df.shape}")

    label_cols = [c for c in df.columns if c.startswith("fine_")]
    meta_cols  = [c for c in df.columns if c.startswith("meta_")]
    fp_cols    = [c for c in df.columns if c.startswith("MACCS_") or c.startswith("morgan_")]
    mordred_cols = [c for c in df.columns if c not in label_cols + meta_cols + ["SMILES"] + fp_cols]

    logger.info(f"Labels (fine_*)   : {len(label_cols)}")
    logger.info(f"Meta cols excluded: {len(meta_cols)}")
    logger.info(f"FP cols           : {len(fp_cols)}")
    logger.info(f"Mordred cols      : {len(mordred_cols)}")

    n_nan = df[fp_cols + mordred_cols].isna().sum().sum()
    logger.info(f"NaNs in features  : {n_nan}")
    df_clean = df.dropna(subset=fp_cols + mordred_cols).reset_index(drop=True)

    # SAFETY CHECK: the cached fold indices are positions into the ORIGINAL row
    # order of hmcn_dataset.csv. If dropna ever actually removes rows, those
    # positions silently stop lining up with the right molecules. The dataset
    # is documented as already NaN-free (4976 rows), so this should always be
    # a no-op -- fail loudly instead of silently misaligning if that changes.
    if len(df_clean) != len(df):
        raise RuntimeError(
            f"dropna removed {len(df) - len(df_clean)} rows -- this invalidates the "
            f"cached fold indices, which assume df's original row order is preserved. "
            f"Do not proceed without re-deriving the split (or fixing the NaNs upstream)."
        )

    pos_counts = df_clean[label_cols].sum().sort_values()
    n_rare = (pos_counts < 10).sum()
    if n_rare:
        logger.warning(f"{n_rare} labels have <10 positive examples in the full set "
                        f"(rarest: {pos_counts.index[0]}={int(pos_counts.iloc[0])}). "
                        f"Expect noisy per-label metrics for these, especially at fold-sized test sets.")

    if DEBUG:
        label_cols = label_cols[:DEBUG_N_LABELS]
        logger.info(f"[DEBUG] restricted to {len(label_cols)} labels (full row set kept, "
                     f"so cached fold indices stay valid)")

    return df_clean, label_cols, fp_cols, mordred_cols


def build_or_load_kfold_splits(X, Y, k=K, val_ratio=VAL_RATIO, seed=RANDOM_STATE,
                                cache_path=SPLIT_CACHE):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            fold_splits = pickle.load(f)
        logger.info(f"Loaded cached {len(fold_splits)}-fold split from '{cache_path}'.")
        return fold_splits

    placeholder = np.arange(len(X)).reshape(-1, 1)
    np.random.seed(seed)
    outer = IterativeStratification(n_splits=k, order=2)
    outer_folds = list(outer.split(placeholder, Y))

    fold_splits = []
    for fold_i, (trainval_idx, test_idx) in enumerate(outer_folds):
        np.random.seed(seed + fold_i + 1)
        X_tv = np.arange(len(trainval_idx)).reshape(-1, 1)
        y_tv = Y[trainval_idx]
        inner = IterativeStratification(
            n_splits=2, order=2, sample_distribution_per_fold=[val_ratio, 1 - val_ratio],
        )
        train_rel, val_rel = next(inner.split(X_tv, y_tv))
        fold_splits.append((trainval_idx[train_rel], trainval_idx[val_rel], test_idx))

    with open(cache_path, "wb") as f:
        pickle.dump(fold_splits, f)
    logger.info(f"Computed a fresh {k}-fold split and cached it to '{cache_path}'.")
    return fold_splits


# ======================================================================
# PER-FOLD FEATURE PIPELINE (fit on that fold's train split only)
# ======================================================================
def build_fold_features(X_fp_raw, X_mordred_raw, y, train_idx, val_idx, test_idx):
    Xfp_tr, Xfp_va, Xfp_te = X_fp_raw[train_idx], X_fp_raw[val_idx], X_fp_raw[test_idx]
    Xmo_tr, Xmo_va, Xmo_te = X_mordred_raw[train_idx], X_mordred_raw[val_idx], X_mordred_raw[test_idx]
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

    vt_fp = VarianceThreshold(threshold=0)
    Xfp_tr = vt_fp.fit_transform(Xfp_tr)
    Xfp_va = vt_fp.transform(Xfp_va)
    Xfp_te = vt_fp.transform(Xfp_te)

    vt_mo = VarianceThreshold(threshold=0)
    Xmo_tr = vt_mo.fit_transform(Xmo_tr)
    Xmo_va = vt_mo.transform(Xmo_va)
    Xmo_te = vt_mo.transform(Xmo_te)

    scaler = StandardScaler()
    Xmo_tr = scaler.fit_transform(Xmo_tr)
    Xmo_va = scaler.transform(Xmo_va)
    Xmo_te = scaler.transform(Xmo_te)

    corr = np.abs(np.corrcoef(Xmo_tr.T))
    upper = np.triu(corr, k=1)
    drop = set()
    rows, cols = np.where(upper > 0.95)
    for r, c in zip(rows, cols):
        if c not in drop:
            drop.add(c)
    keep = [i for i in range(Xmo_tr.shape[1]) if i not in drop]
    Xmo_tr, Xmo_va, Xmo_te = Xmo_tr[:, keep], Xmo_va[:, keep], Xmo_te[:, keep]

    X_train = np.hstack([Xfp_tr, Xmo_tr])
    X_val   = np.hstack([Xfp_va, Xmo_va])
    X_test  = np.hstack([Xfp_te, Xmo_te])

    logger.info(f"  Fold features: fp={Xfp_tr.shape[1]}, mordred={Xmo_tr.shape[1]} "
                f"(dropped {len(drop)} correlated) | "
                f"train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    return X_train, y_tr, X_val, y_va, X_test, y_te


# ======================================================================
# THRESHOLDS (on VAL, not train) + EVALUATION (on TEST)
# ======================================================================
def find_thresholds(y_true, y_prob, label_cols, thresholds=np.arange(0.1, 0.91, 0.05)):
    best_thresholds = {}
    for i, label in enumerate(label_cols):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            y_pred_t = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], y_pred_t, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[label] = best_t
    return best_thresholds


def evaluate(name, fold_i, clf, X_val, y_val, X_test, y_test, label_cols):
    """clf is already fit. Thresholds are calibrated on VAL, metrics reported on TEST."""
    y_prob_val = np.array(clf.predict_proba(X_val).todense())
    y_prob_test = np.array(clf.predict_proba(X_test).todense())

    thresholds = find_thresholds(y_val, y_prob_val, label_cols)

    rows = []
    for i, label in enumerate(label_cols):
        yt, ypr = y_test[:, i], y_prob_test[:, i]
        yp = (ypr >= thresholds[label]).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        try:    roc = roc_auc_score(yt, ypr)
        except: roc = float("nan")
        try:    pr = average_precision_score(yt, ypr)
        except: pr = float("nan")
        try:    mcc = matthews_corrcoef(yt, yp)
        except: mcc = float("nan")

        rows.append({
            "Fold": fold_i, "Model": name, "Label": label, "Threshold": round(thresholds[label], 2),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Bal.Acc": round(balanced_accuracy_score(yt, yp), 3),
            "MCC": round(mcc, 3) if mcc == mcc else float("nan"),
            "F1": round(f1_score(yt, yp, zero_division=0), 3),
            "ROC_AUC": round(roc, 3),
            "PR_AUC": round(pr, 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Sensitivity": round(recall_score(yt, yp, zero_division=0), 3),
            "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else float("nan"), 3),
        })

    result_df = pd.DataFrame(rows)
    macro = result_df.drop(columns=["Fold", "Model", "Label"]).mean(numeric_only=True)
    logger.info(f"  [{name}] fold {fold_i} MACRO  F1={macro['F1']:.3f}  PR_AUC={macro['PR_AUC']:.3f}  "
                f"ROC_AUC={macro['ROC_AUC']:.3f}  Bal.Acc={macro['Bal.Acc']:.3f}  "
                f"Precision={macro['Precision']:.3f}")
    return result_df


# ======================================================================
# MODEL BUILDERS
# ======================================================================
def _make_auto_weight_xgb(**kwargs):
    from xgboost import XGBClassifier

    class _AutoWeightXGB(XGBClassifier):
        def fit(self, X, y, **fit_kwargs):
            y_arr = np.asarray(y).ravel()
            n_pos = int(y_arr.sum())
            n_neg = len(y_arr) - n_pos
            self.scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            return super().fit(X, y_arr, **fit_kwargs)

    return _AutoWeightXGB(**kwargs)


def get_model(name):
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        return BinaryRelevance(
            classifier=LogisticRegression(class_weight="balanced", solver="saga", max_iter=300,
                                           random_state=RANDOM_STATE),  # saga is stochastic -- was unseeded before
            require_dense=[True, True],
        )
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return BinaryRelevance(
            classifier=RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            require_dense=[True, True],
        )
    if name == "XGB":
        return BinaryRelevance(
            classifier=_make_auto_weight_xgb(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            require_dense=[True, True],
        )
    if name == "SVM":
        from sklearn.svm import SVC
        return BinaryRelevance(
            classifier=SVC(class_weight="balanced", kernel="rbf", probability=True, random_state=RANDOM_STATE),
            require_dense=[True, True],
        )
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return BinaryRelevance(classifier=KNeighborsClassifier(n_jobs=-1), require_dense=[True, True])
    raise ValueError(name)


def run_model(name, fold_i, X_train, y_train, X_val, y_val, X_test, y_test, label_cols):
    logger.info(f"  [{name}] fold {fold_i}: starting GridSearchCV (grid={PARAM_GRIDS[name]})")
    scorer = make_scorer(f1_score, average="macro")
    clf = get_model(name)
    gs = GridSearchCV(clf, PARAM_GRIDS[name], scoring=scorer, cv=CV_FOLDS, n_jobs=-1, verbose=0)

    t0 = time.time()
    gs.fit(X_train, y_train)
    logger.info(f"  [{name}] fold {fold_i}: GridSearchCV done in {time.time() - t0:.1f}s | "
                f"best_params={gs.best_params_} | best CV macro-F1={gs.best_score_:.3f}")

    return evaluate(name, fold_i, gs.best_estimator_, X_val, y_val, X_test, y_test, label_cols)


# ======================================================================
# MAIN
# ======================================================================
def main():
    df_clean, label_cols, fp_cols, mordred_cols = load_and_prepare()
    X_fp_raw = df_clean[fp_cols].values.astype(float)
    X_mordred_raw = df_clean[mordred_cols].values.astype(float)
    y_full = df_clean[label_cols].values

    # NOTE: the fold split is stratified on the FULL 138-label set, even in
    # DEBUG mode with a restricted label_cols -- this keeps the split file
    # identical to the one HMCN uses/produces, rather than a debug-only variant.
    y_for_split = df_clean[[c for c in df_clean.columns if c.startswith("fine_")]].values
    fold_splits = build_or_load_kfold_splits(X_fp_raw, y_for_split)

    if DEBUG:
        fold_splits = fold_splits[:DEBUG_N_FOLDS]
        logger.info(f"[DEBUG] restricted to {len(fold_splits)} fold(s)")

    results_path = OUTPUT_DIR / "baseline_fine138_kfold_all_results.csv"

    # --- Restart-skip: which (fold, Model) pairs are already done? ---
    completed = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        completed = set(zip(prev["Fold"], prev["Model"]))
        logger.info(f"Found {len(completed)} completed (fold, model) pairs in "
                    f"'{results_path}' -- these will be skipped.")

    all_results = []
    if results_path.exists():
        all_results.append(pd.read_csv(results_path))

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold_i+1}/{len(fold_splits)}  "
                    f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test)")
        logger.info(f"{'='*60}")

        models_remaining = [m for m in MODELS_TO_RUN if (fold_i, m) not in completed]
        if not models_remaining:
            logger.info(f"  All models already completed for fold {fold_i} -- skipping feature build.")
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = build_fold_features(
            X_fp_raw, X_mordred_raw, y_full, train_idx, val_idx, test_idx
        )

        for name in MODELS_TO_RUN:
            if (fold_i, name) in completed:
                logger.info(f"  [{name}] fold {fold_i}: already completed, skipping")
                continue
            try:
                res = run_model(name, fold_i, X_train, y_train, X_val, y_val, X_test, y_test, label_cols)
                all_results.append(res)
                pd.concat(all_results, ignore_index=True).to_csv(results_path, index=False)
            except Exception:
                logger.exception(f"  [{name}] fold {fold_i}: failed, continuing with remaining models")

    if not all_results:
        logger.error("No model completed successfully.")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(results_path, index=False)

    metric_cols = ["Bal.Acc", "MCC", "F1", "ROC_AUC", "PR_AUC", "Precision", "Sensitivity", "Specificity"]

    # Per-fold macro average (mean across the 138 labels), one row per (fold, Model)
    per_fold_macro = all_results_df.groupby(["Fold", "Model"])[metric_cols].mean().round(3)
    per_fold_macro.to_csv(OUTPUT_DIR / "baseline_fine138_kfold_per_fold_macro.csv")

    # Mean +/- std across folds -- the headline number to report
    summary = all_results_df.groupby(["Fold", "Model"])[metric_cols].mean().groupby("Model").agg(["mean", "std"]).round(3)
    summary.to_csv(OUTPUT_DIR / "baseline_fine138_kfold_summary_mean_std.csv")
    logger.info("Mean +/- std across folds, per model:\n" + summary.to_string())

    logger.info(f"Done. Results in {OUTPUT_DIR}/")

    script_end_time = datetime.now()
    elapsed = script_end_time - SCRIPT_START_TIME
    logger.info(f"Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
