"""
Binary Relevance baseline on the 138 fine-grained odor descriptors.

Dataset: hmcn_dataset.csv
    - Already merges MACCS(166) + Morgan(512) + Mordred(327) features with
      138 `fine_*` labels and 12 `meta_*` labels, cleaned to 4976 rows (no NaNs).
    - This is the SAME feature pipeline used for the 12-metacategory baseline,
      so no separate feature engineering is needed -- only the label set changes.

Differences from the 12-label baseline notebook:
    1. LABEL_COLS is built programmatically from the `fine_*` columns (138 labels)
       instead of the hardcoded 12 metacategories.
    2. `meta_*` columns are explicitly excluded from the feature matrix (they were
       not present in the old CSV, so the old exclusion list didn't need to know
       about them).
    3. evaluate() now stores TP/TN/FP/FN per label per model in the results
       CSV, so any confusion-matrix-derived metric (precision, recall,
       specificity, F1, MCC, balanced accuracy, ...) can be recomputed later
       without refitting anything.
    4. Designed to run as a script on the server: config block, dual logging
       (file + stdout, timestamped), DEBUG smoke-test mode.

Usage:
    Smoke test (Colab or local, ~1 min):   DEBUG=True  in CONFIG, then run.
    Full run (server, background):
        nohup python3 baseline_fine138.py > /dev/null 2>&1 &
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
)
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.problem_transform import BinaryRelevance

# ======================================================================
# CONFIG
# ======================================================================
DEBUG = False  # True -> tiny subsample, 1 model, single param, for a smoke test

DATA_PATH   = "hmcn_dataset.csv"
OUTPUT_DIR  = Path("outputs_fine138")
LOG_DIR     = Path("logs")
RANDOM_STATE = 42
TEST_SIZE   = 0.2
CV_FOLDS    = 3

# which models to run, in order
MODELS_TO_RUN = ["LR", "RF", "XGB", "SVM", "KNN"]

# Full-run hyperparameter grids. Values carried over from the 12-metacategory
# tuning (grid ranges are about model capacity vs. training-set size, which is
# unchanged at 138 labels -- only the number of labels changed, which affects
# runtime, not what values are appropriate).
# NOTE: with 138 labels x CV_FOLDS, every grid point costs 138 x CV_FOLDS
# classifier fits. Total across all 5 models here is ~9,100 fits -- expect a
# genuinely long server job, not a quick check.
PARAM_GRIDS = {
    "LR":  {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    "RF":  {"classifier__n_estimators": [100, 300], "classifier__max_features": ["sqrt", "log2"]},
    "XGB": {"classifier__n_estimators": [100, 300], "classifier__max_depth": [3, 6]},
    # SVM with probability=True is the expensive one (Platt scaling = internal
    # 5-fold CV per fit, x CV_FOLDS x 138 labels x grid points).
    "SVM": {"classifier__C": [0.1, 1, 10]},
    "KNN": {"classifier__n_neighbors": [5, 11, 21], "classifier__metric": ["euclidean", "cosine"]},
}

if DEBUG:
    OUTPUT_DIR = Path("outputs_fine138_debug")
    MODELS_TO_RUN = ["LR"]
    PARAM_GRIDS["LR"] = {"classifier__C": [0.1]}
    CV_FOLDS = 2
    DEBUG_N_ROWS = 400
    DEBUG_N_LABELS = 6

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# ======================================================================
# LOGGING (file + stdout, auto-timestamped)
# ======================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"baseline_fine138_{'debug' if DEBUG else 'full'}_{timestamp}.log"

logger = logging.getLogger("baseline_fine138")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

fh = logging.FileHandler(log_path)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

logger.info(f"DEBUG={DEBUG} | log file: {log_path}")

# ======================================================================
# LOAD + CLEAN
# ======================================================================
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)  # comma-delimited, unlike the semicolon-delimited CSVs elsewhere in the project
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
    logger.info(f"Rows after dropna : {len(df_clean)} (dropped {len(df) - len(df_clean)})")

    if DEBUG:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(df_clean), size=min(DEBUG_N_ROWS, len(df_clean)), replace=False)
        df_clean = df_clean.iloc[idx].reset_index(drop=True)
        label_cols = label_cols[:DEBUG_N_LABELS]
        logger.info(f"[DEBUG] subsampled to {len(df_clean)} rows, {len(label_cols)} labels")

    # sanity check: warn about labels with very few positives, since GridSearchCV's
    # internal folds and the per-label threshold search can behave erratically for them
    pos_counts = df_clean[label_cols].sum().sort_values()
    n_rare = (pos_counts < 10).sum()
    if n_rare:
        logger.warning(f"{n_rare} labels have <10 positive examples in the full set "
                        f"(rarest: {pos_counts.index[0]}={int(pos_counts.iloc[0])}). "
                        f"Expect noisy per-label metrics and possible zero-positive CV folds for these.")

    return df_clean, label_cols, fp_cols, mordred_cols


def build_features(df_clean, fp_cols, mordred_cols, label_cols):
    X_fp = df_clean[fp_cols].values.astype(float)
    X_mordred = df_clean[mordred_cols].values.astype(float)
    y = df_clean[label_cols].values

    vt_fp = VarianceThreshold(threshold=0)
    X_fp = vt_fp.fit_transform(X_fp)
    vt_mordred = VarianceThreshold(threshold=0)
    X_mordred = vt_mordred.fit_transform(X_mordred)
    logger.info(f"After zero-variance filter: fp={X_fp.shape[1]}, mordred={X_mordred.shape[1]}")

    X_combined = np.hstack([X_fp, X_mordred])
    n_fp = X_fp.shape[1]

    X_train_comb, y_train, X_test_comb, y_test = iterative_train_test_split(
        X_combined, y, test_size=TEST_SIZE
    )
    X_fp_train, X_mordred_train = X_train_comb[:, :n_fp], X_train_comb[:, n_fp:]
    X_fp_test,  X_mordred_test  = X_test_comb[:, :n_fp],  X_test_comb[:, n_fp:]
    logger.info(f"Train: {X_fp_train.shape[0]} | Test: {X_fp_test.shape[0]}")

    scaler = StandardScaler()
    X_mordred_train = scaler.fit_transform(X_mordred_train)
    X_mordred_test = scaler.transform(X_mordred_test)

    corr = np.abs(np.corrcoef(X_mordred_train.T))
    upper = np.triu(corr, k=1)
    drop = set()
    rows, cols = np.where(upper > 0.95)
    for r, c in zip(rows, cols):
        if c not in drop:
            drop.add(c)
    keep = [i for i in range(X_mordred_train.shape[1]) if i not in drop]
    X_mordred_train = X_mordred_train[:, keep]
    X_mordred_test = X_mordred_test[:, keep]
    logger.info(f"Mordred after correlation filter: {X_mordred_train.shape[1]} (dropped {len(drop)})")

    X_train = np.hstack([X_fp_train, X_mordred_train])
    X_test = np.hstack([X_fp_test, X_mordred_test])
    logger.info(f"Final X_train {X_train.shape} | X_test {X_test.shape}")

    return X_train, y_train, X_test, y_test


# ======================================================================
# THRESHOLDS + EVALUATION (persists TP/TN/FP/FN per label per model)
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


def evaluate(name, clf, X_tr, y_tr, X_te, y_te, label_cols):
    """Fit clf, find per-label thresholds on train, evaluate on test.
    Stores TP/TN/FP/FN per label so any confusion-matrix-derived metric
    (precision, recall, specificity, F1, MCC, balanced accuracy, ...) can
    be recomputed later without refitting.
    """
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    logger.info(f"[{name}] fit done in {time.time() - t0:.1f}s")

    y_prob_train = np.array(clf.predict_proba(X_tr).todense())
    y_prob_test = np.array(clf.predict_proba(X_te).todense())

    thresholds = find_thresholds(y_tr, y_prob_train, label_cols)

    rows = []
    for i, label in enumerate(label_cols):
        yt, ypr = y_te[:, i], y_prob_test[:, i]
        yp = (ypr >= thresholds[label]).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        try:    roc = roc_auc_score(yt, ypr)
        except: roc = float("nan")
        try:    pr = average_precision_score(yt, ypr)
        except: pr = float("nan")

        rows.append({
            "Model": name, "Label": label, "Threshold": round(thresholds[label], 2),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Bal.Acc": round(balanced_accuracy_score(yt, yp), 3),
            "MCC": round(matthews_corrcoef(yt, yp), 3),
            "F1": round(f1_score(yt, yp, zero_division=0), 3),
            "ROC_AUC": round(roc, 3),
            "PR_AUC": round(pr, 3),
            "Sensitivity": round(recall_score(yt, yp, zero_division=0), 3),
            "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else float("nan"), 3),
        })

    result_df = pd.DataFrame(rows)

    macro = result_df.drop(columns=["Model", "Label"]).mean(numeric_only=True)
    logger.info(f"[{name}] MACRO  F1={macro['F1']:.3f}  PR_AUC={macro['PR_AUC']:.3f}  "
                f"ROC_AUC={macro['ROC_AUC']:.3f}  Bal.Acc={macro['Bal.Acc']:.3f}")

    return result_df


# ======================================================================
# MODEL BUILDERS
# ======================================================================
def _make_auto_weight_xgb(**kwargs):
    """Build an XGBClassifier whose scale_pos_weight is computed from the
    label it's being fit on, instead of left at the default of 1 (= no
    imbalance correction). LR/RF/SVM get this for free via
    class_weight='balanced'; XGBoost has no such option, only a single
    scalar scale_pos_weight, so it has to be computed and set per label.

    Because BinaryRelevance clones this classifier fresh for each of the
    138 labels (and GridSearchCV clones BinaryRelevance fresh for each CV
    fold/param combination), the weight computed for one label never leaks
    into another label's fit -- each fit() call recomputes it from scratch
    from just that call's y.
    """
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
            classifier=LogisticRegression(class_weight="balanced", solver="saga", max_iter=300),
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


def run_model(name, X_train, y_train, X_test, y_test, label_cols):
    logger.info(f"=== {name}: starting GridSearchCV (grid={PARAM_GRIDS[name]}) ===")
    scorer = make_scorer(f1_score, average="macro")
    clf = get_model(name)
    gs = GridSearchCV(clf, PARAM_GRIDS[name], scoring=scorer, cv=CV_FOLDS, n_jobs=-1, verbose=1)

    t0 = time.time()
    gs.fit(X_train, y_train)
    logger.info(f"[{name}] GridSearchCV done in {time.time() - t0:.1f}s | "
                f"best_params={gs.best_params_} | best CV macro-F1={gs.best_score_:.3f}")

    return evaluate(name, gs.best_estimator_, X_train, y_train, X_test, y_test, label_cols)


# ======================================================================
# MAIN
# ======================================================================
def main():
    df_clean, label_cols, fp_cols, mordred_cols = load_and_prepare()
    X_train, y_train, X_test, y_test = build_features(df_clean, fp_cols, mordred_cols, label_cols)

    all_results = []
    for name in MODELS_TO_RUN:
        try:
            res = run_model(name, X_train, y_train, X_test, y_test, label_cols)
            all_results.append(res)
            # save incrementally so a crash on model N doesn't lose models 1..N-1
            pd.concat(all_results, ignore_index=True).to_csv(
                OUTPUT_DIR / "baseline_fine138_results.csv", index=False
            )
        except Exception:
            logger.exception(f"[{name}] failed, continuing with remaining models")

    if not all_results:
        logger.error("No model completed successfully.")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(OUTPUT_DIR / "baseline_fine138_results.csv", index=False)

    metric_cols = ["Bal.Acc", "MCC", "F1", "ROC_AUC", "PR_AUC", "Sensitivity", "Specificity"]
    summary = all_results_df.groupby("Model")[metric_cols].mean().round(3)
    summary.to_csv(OUTPUT_DIR / "baseline_fine138_summary.csv")
    logger.info("Macro-averaged test metrics per model:\n" + summary.to_string())

    logger.info(f"Done. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
