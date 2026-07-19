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
import itertools
from datetime import datetime
from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score, confusion_matrix,
    accuracy_score, hamming_loss, jaccard_score
)
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance
import hmcn_eval  # shared eval utility -- place hmcn_eval.py in the same directory

# Same schema extension as the HMCN k-fold script -- see that file for rationale.
# 'model' is added separately from param_value: param_value encodes the model's
# winning hyperparams too (e.g. "LR_{'classifier__C': 0.1}"), which can differ
# across folds since GridSearchCV is refit per fold -- grouping by param_value
# for the cross-fold summary would then put each fold in its own group of size 1.
if 'fold' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(3, 'fold')
if 'model' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(4, 'model')

# ======================================================================
# CONFIG
# ======================================================================
DEBUG = False  # True -> 1 fold, 1 model, tiny grid, for a smoke test

DATA_PATH    = "hmcn_dataset.csv"
OUTPUT_DIR   = Path("outputs_fine138_kfold")
LOG_DIR      = Path("logs")
RANDOM_STATE = 42
K            = 5
VAL_RATIO    = 0.15       # matches the HMCN k-fold split exactly
SPLIT_CACHE  = "kfold_results/hmcn_kfold_split_indices.pkl"  # MUST match the HMCN k-fold script's path exactly

MODELS_TO_RUN = ["LR", "RF", "XGB", "SVM", "KNN"]

# Each grid gives EVERY config to be run on EVERY fold, unchanged -- no per-fold
# auto-tuning (that was the bug: GridSearchCV was picking a possibly different
# "best" config on each fold's own internal CV, making the winning hyperparams
# inconsistent across folds and impossible to report as a single config in the
# paper). This mirrors exactly how the HMCN k-fold sweep evaluates a fixed grid
# identically across all folds.
MODEL_GRIDS = {
    "LR":  {"C": [0.01, 0.1, 1, 10, 100]},
    "RF":  {"n_estimators": [100, 300], "max_features": ["sqrt", "log2"]},
    "XGB": {"n_estimators": [100, 300], "max_depth": [3, 6]},
    "SVM": {"C": [0.1, 1, 10]},
    "KNN": {"n_neighbors": [5, 11, 21], "metric": ["euclidean", "cosine"]},
}

if DEBUG:
    OUTPUT_DIR = Path("outputs_fine138_kfold_debug")
    MODELS_TO_RUN = ["LR"]
    MODEL_GRIDS["LR"] = {"C": [0.1]}
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

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
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
# ======================================================================
# THRESHOLDS (on VAL, not train) + EVALUATION (on TEST)
# ======================================================================
def compute_fine_only_metrics(fine_probs, fine_true, fine_thresholds):
    """Same formulas as the fine-138 block of hmcn_eval.compute_all_metrics,
    copied rather than called directly because that function requires meta
    predictions the BR baseline doesn't produce. hier_violation_rate_138 is
    deliberately omitted (it needs a parent/meta prediction to check against)
    -- hmcn_eval.save_experiment will fill it, and every meta_*/12 column,
    with NaN automatically.
    """
    fine_pred = (fine_probs >= fine_thresholds[np.newaxis, :]).astype(int)
    valid_fine = [i for i in range(fine_true.shape[1]) if fine_true[:, i].sum() > 0]

    roc_auc_138 = float(np.mean([
        roc_auc_score(fine_true[:, i], fine_probs[:, i]) for i in valid_fine
    ]))
    pr_auc_138 = float(np.mean([
        average_precision_score(fine_true[:, i], fine_probs[:, i]) for i in valid_fine
    ]))

    return {
        'roc_auc_138'             : round(roc_auc_138, 4),
        'pr_auc_138'              : round(pr_auc_138, 4),
        'f1_macro_138'            : round(float(f1_score(fine_true, fine_pred, average='macro', zero_division=0)), 4),
        'f1_micro_138'            : round(float(f1_score(fine_true, fine_pred, average='micro', zero_division=0)), 4),
        'precision_macro_138'     : round(float(precision_score(fine_true, fine_pred, average='macro', zero_division=0)), 4),
        'recall_macro_138'        : round(float(recall_score(fine_true, fine_pred, average='macro', zero_division=0)), 4),
        'precision_micro_138'     : round(float(precision_score(fine_true, fine_pred, average='micro', zero_division=0)), 4),
        'recall_micro_138'        : round(float(recall_score(fine_true, fine_pred, average='micro', zero_division=0)), 4),
        'matched_accuracy_138'    : round(float(accuracy_score(fine_true, fine_pred)), 4),
        'hamming_loss_138'        : round(float(hamming_loss(fine_true, fine_pred)), 4),
        'jaccard_138'             : round(float(jaccard_score(fine_true, fine_pred, average='macro', zero_division=0)), 4),
        # hier_violation_rate_138 intentionally omitted -- see docstring
    }


def evaluate(name, param_value, fold_i, clf, X_val, y_val, X_test, y_test, label_cols):
    """clf is already fit. Thresholds are calibrated on VAL, metrics reported on TEST.
    Returns (per_label_df, ablation_metrics_dict) -- the former keeps full TP/TN/FP/FN
    traceability per label (as before), the latter matches the hmcn_ablation_results.csv
    schema for direct cross-track comparison with the HMCN k-fold sweep.
    """
    y_prob_val = np.array(clf.predict_proba(X_val).todense())
    y_prob_test = np.array(clf.predict_proba(X_test).todense())

    # Same 19-candidate sweep in [0.05, 0.95] as HMCN's calibration -- previously
    # this used a slightly different 17-candidate [0.1, 0.9] sweep; switched for
    # exact fidelity with the shared eval utility.
    thr_array = hmcn_eval.find_optimal_thresholds(y_prob_val, y_val)
    thresholds = dict(zip(label_cols, thr_array))

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
            "Fold": fold_i, "Model": name, "Config": param_value, "Label": label,
            "Threshold": round(float(thresholds[label]), 2),
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
    macro = result_df.drop(columns=["Fold", "Model", "Config", "Label"]).mean(numeric_only=True)
    logger.info(f"  [{name}] fold {fold_i} [{param_value}] MACRO  F1={macro['F1']:.3f}  PR_AUC={macro['PR_AUC']:.3f}  "
                f"ROC_AUC={macro['ROC_AUC']:.3f}  Bal.Acc={macro['Bal.Acc']:.3f}  "
                f"Precision={macro['Precision']:.3f}")

    ablation_metrics = compute_fine_only_metrics(y_prob_test, y_test, thr_array)
    return result_df, ablation_metrics


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


def get_model(name, cfg):
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        return BinaryRelevance(
            classifier=LogisticRegression(class_weight="balanced", solver="saga", max_iter=300,
                                           random_state=RANDOM_STATE, **cfg),  # saga is stochastic -- was unseeded before
            require_dense=[True, True],
        )
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return BinaryRelevance(
            classifier=RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **cfg),
            require_dense=[True, True],
        )
    if name == "XGB":
        return BinaryRelevance(
            classifier=_make_auto_weight_xgb(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, **cfg),
            require_dense=[True, True],
        )
    if name == "SVM":
        from sklearn.svm import SVC
        return BinaryRelevance(
            classifier=SVC(class_weight="balanced", kernel="rbf", probability=True, random_state=RANDOM_STATE, **cfg),
            require_dense=[True, True],
        )
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return BinaryRelevance(classifier=KNeighborsClassifier(n_jobs=-1, **cfg), require_dense=[True, True])
    raise ValueError(name)


def run_model(name, cfg, param_value, fold_i, X_train, y_train, X_val, y_val, X_test, y_test,
              label_cols, ablation_csv_path):
    logger.info(f"  [{name}] fold {fold_i}: fitting {param_value}")
    clf = get_model(name, cfg)

    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info(f"  [{name}] fold {fold_i}: fit done in {time.time() - t0:.1f}s")

    per_label_df, ablation_metrics = evaluate(name, param_value, fold_i, clf, X_val, y_val, X_test, y_test, label_cols)

    config = dict(
        experiment='baseline_fine138_kfold', param_name='model', param_value=param_value,
        fold=fold_i, model=name, seed=RANDOM_STATE,
        # global_dim/local_dim/dropout/lr/weight_decay/lambda_viol/beta/batch_size,
        # best_epoch/train_loss_at_best/val_meta_roc_auc, test_loss: not applicable
        # to sklearn models -- left unset, hmcn_eval.save_experiment NaN-fills them.
    )
    hmcn_eval.save_experiment(config, ablation_metrics, csv_path=ablation_csv_path)

    return per_label_df


def expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid_dict.values())]


def format_param_value(name, cfg):
    return name + "_" + "_".join(f"{k}={v}" for k, v in cfg.items())


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
    ablation_csv_path = OUTPUT_DIR / "baseline_fine138_kfold_ablation_results.csv"

    # Every (model, config) pair to run, flattened once up front.
    runs = [(name, cfg, format_param_value(name, cfg))
            for name in MODELS_TO_RUN for cfg in expand_grid(MODEL_GRIDS[name])]
    logger.info(f"Configs total: {len(runs)} | Folds: {len(fold_splits)} | "
                f"Total runs: {len(runs) * len(fold_splits)}")

    # --- Restart-skip: which (fold, Model, Config) triples are already done? ---
    completed = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        completed = set(zip(prev["Fold"], prev["Model"], prev["Config"]))
        logger.info(f"Found {len(completed)} completed (fold, model, config) triples in "
                    f"'{results_path}' -- these will be skipped.")

    all_results = []
    if results_path.exists():
        all_results.append(pd.read_csv(results_path))

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold_i+1}/{len(fold_splits)}  "
                    f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test)")
        logger.info(f"{'='*60}")

        runs_remaining = [r for r in runs if (fold_i, r[0], r[2]) not in completed]
        if not runs_remaining:
            logger.info(f"  All {len(runs)} (model, config) runs already completed for fold {fold_i} -- skipping.")
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = build_fold_features(
            X_fp_raw, X_mordred_raw, y_full, train_idx, val_idx, test_idx
        )

        for i, (name, cfg, param_value) in enumerate(runs, 1):
            if (fold_i, name, param_value) in completed:
                logger.info(f"  [{i}/{len(runs)}] {param_value}: already completed, skipping")
                continue
            try:
                res = run_model(name, cfg, param_value, fold_i, X_train, y_train, X_val, y_val,
                                 X_test, y_test, label_cols, ablation_csv_path)
                all_results.append(res)
                pd.concat(all_results, ignore_index=True).to_csv(results_path, index=False)
            except Exception:
                logger.exception(f"  [{i}/{len(runs)}] {param_value}: failed, continuing")

    if not all_results:
        logger.error("No run completed successfully.")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(results_path, index=False)

    metric_cols = ["Bal.Acc", "MCC", "F1", "ROC_AUC", "PR_AUC", "Precision", "Sensitivity", "Specificity"]

    # Per-fold macro average (mean across the 138 labels), one row per (fold, Model, Config)
    per_fold_macro = all_results_df.groupby(["Fold", "Model", "Config"])[metric_cols].mean().round(3)
    per_fold_macro.to_csv(OUTPUT_DIR / "baseline_fine138_kfold_per_fold_macro.csv")

    # Mean +/- std across folds, per (Model, Config) -- this is now directly analogous
    # to HMCN's per-config summary, since every config ran identically on every fold.
    summary = all_results_df.groupby(["Fold", "Model", "Config"])[metric_cols].mean() \
                             .groupby(["Model", "Config"]).agg(["mean", "std"]).round(3)
    summary.to_csv(OUTPUT_DIR / "baseline_fine138_kfold_summary_mean_std.csv")
    logger.info("Mean +/- std across folds, per (Model, Config):\n" + summary.to_string())

    # Same aggregation for the ablation-format CSV (matches HMCN's schema) -- grouped
    # by (model, param_value), which is now a stable key across folds by construction.
    if ablation_csv_path.exists():
        ablation_df = pd.read_csv(ablation_csv_path)
        exclude_cols = ['experiment', 'param_name', 'param_value', 'fold', 'model',
                         'global_dim', 'local_dim', 'dropout', 'lr', 'weight_decay',
                         'lambda_viol', 'beta', 'batch_size', 'seed',
                         'best_epoch', 'train_loss_at_best', 'val_meta_roc_auc']
        metric_cols_ablation = [c for c in hmcn_eval._CSV_COLUMNS if c not in exclude_cols]
        ablation_summary = ablation_df.groupby(['model', 'param_value'])[metric_cols_ablation].agg(['mean', 'std'])
        ablation_summary_path = OUTPUT_DIR / "baseline_fine138_kfold_ablation_summary_mean_std.csv"
        ablation_summary.to_csv(ablation_summary_path)
        logger.info(f"Ablation-format mean +/- std summary (compare directly against "
                    f"HMCN's hmcn_kfold_summary_mean_std.csv): {ablation_summary_path}")

        # One row per model: the single best config (by mean pr_auc_138 across folds) --
        # this is the number/config to actually put in the paper's methods table.
        mean_only = ablation_df.groupby(['model', 'param_value'])[metric_cols_ablation].mean()
        winners = mean_only.loc[mean_only.groupby('model')['pr_auc_138'].idxmax()]
        winners_path = OUTPUT_DIR / "baseline_fine138_kfold_winning_configs.csv"
        winners.to_csv(winners_path)
        logger.info("Winning config per model (best mean pr_auc_138 across folds):\n" +
                    winners[['pr_auc_138', 'f1_macro_138', 'roc_auc_138']].to_string())
        logger.info(f"Winning configs saved to: {winners_path}")

    logger.info(f"Done. Per-label results in {results_path}")
    logger.info(f"Done. Ablation-format results (matches hmcn_ablation_results.csv schema) in {ablation_csv_path}")

    script_end_time = datetime.now()
    elapsed = script_end_time - SCRIPT_START_TIME
    logger.info(f"Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
