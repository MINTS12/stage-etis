"""
Flat MLP baseline -- HMCN-F's exact building blocks (shared trunk, local fine
head, local meta head, global head, beta blend), minus the hierarchy
violation loss. Jointly predicts fine138 + meta12 from the same shared
5-fold split cache used by every other k-fold script in this project.

Purpose (see paper's central question -- does complexity help at this scale):
this isolates "does the hierarchy machinery itself help" from "does going
neural help at all", by holding architecture capacity fixed and only
removing lambda_viol * vloss(...) from the training loss. Concretely this is
the lambda_viol=0 point on the same axis HMCN-F's own sweep already varies --
config['lambda_viol'] is set to 0.0 explicitly below (not left NaN) so the
ablation results table reads unambiguously.

Design notes / things to verify against hmcn_eval.py before treating these
numbers as final:
    - meta12 metric keys are named by direct analogy with the fine138 keys
      already used elsewhere in this project (roc_auc_12, pr_auc_12,
      f1_macro_12, etc.). If hmcn_eval.py's HMCN-F script uses different
      names for these, rename compute_metrics()'s output keys to match
      before the two result tables are merged.
    - hier_violation_rate_138 is computed explicitly here (see
      compute_hier_violation_rate docstring) rather than via hmcn_eval,
      because hmcn_eval's own violation-rate formula wasn't available to
      copy from. Cross-check the definition against whatever hmcn_eval /
      the HMCN k-fold script uses internally -- if they differ, this
      column is not directly comparable to HMCN-F's until reconciled.
    - Early stopping / model selection uses val macro PR-AUC (mean of
      fine138 PR-AUC and meta12 PR-AUC), not val meta ROC-AUC like the
      older HMCN scripts -- a deliberate change per project's stated
      preference for PR-AUC under class imbalance. 'val_pr_auc_combined'
      is inserted into hmcn_eval._CSV_COLUMNS the same guarded way 'fold'
      and 'model' were inserted in baseline_fine138_kfold.py.

Usage:
    Smoke test:  DEBUG=True in CONFIG, then run (1 fold, 2 configs, few epochs).
    Full run (server, background):
        nohup python3 -u mlp_kfold.py > /dev/null 2>&1 &
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score,
    confusion_matrix, accuracy_score, hamming_loss, jaccard_score
)
from skmultilearn.model_selection import IterativeStratification
import hmcn_eval  # shared eval utility -- place hmcn_eval.py in the same directory

# Same guarded-insert pattern as baseline_fine138_kfold.py.
if 'fold' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(3, 'fold')
if 'model' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(4, 'model')
if 'val_pr_auc_combined' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.append('val_pr_auc_combined')

# ======================================================================
# CONFIG
# ======================================================================
DEBUG = False  # True -> 1 fold, 2 configs, 5 epochs, for a smoke test

DATA_PATH    = "hmcn_dataset.csv"
OUTPUT_DIR   = Path("outputs_mlp_kfold")
LOG_DIR      = Path("logs")
RANDOM_STATE = 42
K            = 5
VAL_RATIO    = 0.15       # matches the HMCN k-fold split exactly
SPLIT_CACHE  = "kfold_results/hmcn_kfold_split_indices.pkl"  # MUST match the HMCN k-fold script's path exactly

BATCH_SIZE  = 32
EPOCHS      = 150
PATIENCE    = 30
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BETA        = 0.5   # local/global blend weight, matches HMCN-F's tuned value

# Small fixed grid, run identically on every fold -- no per-fold auto-tuning,
# same discipline as baseline_fine138_kfold.py's MODEL_GRIDS.
GRID = {
    "global_dim": [64, 128],
    "local_dim":  [32, 64],
    "dropout":    [0.5, 0.6],
}

if DEBUG:
    OUTPUT_DIR = Path("outputs_mlp_kfold_debug")
    GRID = {"global_dim": [64], "local_dim": [32], "dropout": [0.5]}
    EPOCHS = 5
    PATIENCE = 3
    DEBUG_N_FOLDS = 1

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# ======================================================================
# REPRODUCIBILITY
# ======================================================================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)

# ======================================================================
# LOGGING (file + stdout, auto-timestamped, flushes every line)
# ======================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"mlp_kfold_{'debug' if DEBUG else 'full'}_{timestamp}.log"

logger = logging.getLogger("mlp_kfold")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

fh = logging.FileHandler(log_path)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"DEBUG={DEBUG} | device={device} | log file: {log_path}")
SCRIPT_START_TIME = datetime.now()
logger.info(f"Script started at: {SCRIPT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")

# ======================================================================
# LABEL HIERARCHY (only used for the post-hoc violation-rate diagnostic --
# the model is never trained against it) -- copied from hmcn_tune.py.
# ======================================================================
META_CATEGORIES = {
    'floral':['floral','rose','jasmin','lily','muguet','violet','hyacinth','geranium','lavender','orangeflower','chamomile','hawthorn'],
    'fruity':['fruity','apple','apricot','banana','berry','cherry','grape','grapefruit','lemon','melon','orange','peach','pear','pineapple','plum','raspberry','strawberry','tropical','black currant','fruit skin'],
    'sweet':['sweet','vanilla','caramellic','honey','chocolate','cocoa','coconut','creamy','buttery','milky','dairy'],
    'woody':['woody','cedar','sandalwood','pine','vetiver','terpenic','balsamic','cortex'],
    'green':['green','grassy','herbal','leafy','hay','tea','fresh','cucumber','vegetable','weedy'],
    'spicy':['spicy','cinnamon','clove','warm','pungent','sharp','cooling','mint','camphoreous'],
    'animal_musk':['animal','musk','leathery','fishy','sweaty','meaty','beefy','musty'],
    'earthy':['earthy','mushroom','nutty','hazelnut','roasted','coffee','tobacco','smoky','popcorn'],
    'citrus':['citrus','bergamot','ozone','clean','soapy'],
    'chemical':['solvent','ethereal','metallic','medicinal','phenolic','sulfurous','gassy','burnt','oily'],
    'gourmand':['almond','malty','rummy','brandy','cognac','winey','cooked','potato','savory','celery','tomato','radish','onion','garlic','cabbage','cheesy'],
    'powdery_amber':['amber','powdery','anisic','coumarinic','orris','waxy','aldehydic','ketonic','lactonic'],
}

def build_pairs(fine_names, meta_names):
    fi = {n: i for i, n in enumerate(fine_names)}
    mi = {n: i for i, n in enumerate(meta_names)}
    return [(fi[m], mi[meta]) for meta, mems in META_CATEGORIES.items()
            for m in mems if m in fi and meta in mi]


# ======================================================================
# LOAD (full dataset, no split yet -- splitting happens via cached fold indices)
# ======================================================================
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {DATA_PATH}: {df.shape}")

    fine_cols = [c for c in df.columns if c.startswith("fine_")]
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    fp_cols   = [c for c in df.columns if c.startswith("MACCS_") or c.startswith("morgan_")]
    mordred_cols = [c for c in df.columns if c not in fine_cols + meta_cols + ["SMILES"] + fp_cols]

    logger.info(f"Fine labels (fine_*) : {len(fine_cols)}")
    logger.info(f"Meta labels (meta_*) : {len(meta_cols)}")
    logger.info(f"FP cols              : {len(fp_cols)}")
    logger.info(f"Mordred cols         : {len(mordred_cols)}")

    n_nan = df[fp_cols + mordred_cols].isna().sum().sum()
    logger.info(f"NaNs in features     : {n_nan}")
    df_clean = df.dropna(subset=fp_cols + mordred_cols).reset_index(drop=True)

    # Same safety check as baseline_fine138_kfold.py -- cached fold indices are
    # positions into the ORIGINAL row order; fail loudly if that ever shifts.
    if len(df_clean) != len(df):
        raise RuntimeError(
            f"dropna removed {len(df) - len(df_clean)} rows -- this invalidates the "
            f"cached fold indices, which assume df's original row order is preserved. "
            f"Do not proceed without re-deriving the split (or fixing the NaNs upstream)."
        )

    fine_names = [c.replace('fine_', '') for c in fine_cols]
    meta_names = [c.replace('meta_', '') for c in meta_cols]

    return df_clean, fine_cols, meta_cols, fine_names, meta_names, fp_cols, mordred_cols


def build_or_load_kfold_splits(X, Y, k=K, val_ratio=VAL_RATIO, seed=RANDOM_STATE,
                                cache_path=SPLIT_CACHE):
    """Identical to baseline_fine138_kfold.py's version -- loads the existing
    cache as-is (this project already has it), only computes fresh if missing.
    """
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
# PER-FOLD FEATURE PIPELINE (fit on that fold's train split only) --
# identical to baseline_fine138_kfold.py's build_fold_features, generalized
# to carry both label blocks (fine + meta) through the same row indexing.
# ======================================================================
def build_fold_features(X_fp_raw, X_mordred_raw, y_fine, y_meta, train_idx, val_idx, test_idx):
    Xfp_tr, Xfp_va, Xfp_te = X_fp_raw[train_idx], X_fp_raw[val_idx], X_fp_raw[test_idx]
    Xmo_tr, Xmo_va, Xmo_te = X_mordred_raw[train_idx], X_mordred_raw[val_idx], X_mordred_raw[test_idx]

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

    X_train = np.hstack([Xfp_tr, Xmo_tr]).astype(np.float32)
    X_val   = np.hstack([Xfp_va, Xmo_va]).astype(np.float32)
    X_test  = np.hstack([Xfp_te, Xmo_te]).astype(np.float32)

    logger.info(f"  Fold features: fp={Xfp_tr.shape[1]}, mordred={Xmo_tr.shape[1]} "
                f"(dropped {len(drop)} correlated) | "
                f"train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    return (X_train, y_fine[train_idx].astype(np.float32), y_meta[train_idx].astype(np.float32),
            X_val,   y_fine[val_idx].astype(np.float32),   y_meta[val_idx].astype(np.float32),
            X_test,  y_fine[test_idx].astype(np.float32),  y_meta[test_idx].astype(np.float32))


# ======================================================================
# MODEL -- HMCN-F's exact Block/HMCNF structure (proj -> local fine head ->
# local meta head -> global head -> beta blend). Copied unchanged from
# hmcn_tune.py. The only difference from HMCN-F lives in the training loop
# below, which never adds a violation-loss term.
# ======================================================================
class Block(nn.Module):
    def __init__(self, idim, gdim, ldim, nl, dr):
        super().__init__()
        self.gfc = nn.Sequential(nn.Linear(gdim + idim, gdim), nn.BatchNorm1d(gdim), nn.ReLU(), nn.Dropout(dr))
        self.tr  = nn.Sequential(nn.Linear(gdim, ldim), nn.ReLU(), nn.Dropout(dr))
        self.out = nn.Linear(ldim, nl)

    def forward(self, x, A):
        A = self.gfc(torch.cat([A, x], 1))
        return A, torch.sigmoid(self.out(self.tr(A)))


class JointMLP(nn.Module):
    """Same architecture class as HMCN-F (HMCNF in hmcn_tune.py), renamed to
    make clear in logs/output that this run never sees a violation loss."""
    def __init__(self, idim, nf, nm, gd, ld, dr, beta):
        super().__init__()
        self.beta = beta
        self.proj = nn.Sequential(nn.Linear(idim, gd), nn.BatchNorm1d(gd), nn.ReLU(), nn.Dropout(dr))
        self.l1 = Block(idim, gd, ld, nf, dr)
        self.l2 = Block(idim, gd, ld, nm, dr)
        self.go = nn.Linear(gd, nf + nm)

    def forward(self, x):
        A = self.proj(x)
        A, P1 = self.l1(x, A)
        A, P2 = self.l2(x, A)
        PG = torch.sigmoid(self.go(A))
        PF = self.beta * torch.cat([P1, P2], 1) + (1 - self.beta) * PG
        return PF, P1, P2, PG


def bce(P, Y, e=1e-7):
    P = torch.clamp(P, e, 1 - e)
    return -torch.mean(Y * torch.log(P) + (1 - Y) * torch.log(1 - P))


def loss_fn_no_hierarchy(P1, P2, PG, Y1, Y2):
    """Same three BCE terms as HMCN-F's loss_fn, with lambda_viol * vloss(...)
    removed entirely -- this is the one substantive difference from HMCN-F."""
    return bce(P1, Y1) + bce(P2, Y2) + bce(PG, torch.cat([Y1, Y2], 1))


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    fp, ft, mp, mt = [], [], [], []
    for Xb, Y1b, Y2b in loader:
        _, P1, P2, _ = model(Xb.to(device))
        fp.append(P1.cpu().numpy()); ft.append(Y1b.numpy())
        mp.append(P2.cpu().numpy()); mt.append(Y2b.numpy())
    return np.vstack(fp), np.vstack(ft), np.vstack(mp), np.vstack(mt)


def mean_pr_auc(probs, true):
    valid = [i for i in range(true.shape[1]) if true[:, i].sum() > 0]
    if not valid:
        return float("nan")
    return float(np.mean([average_precision_score(true[:, i], probs[:, i]) for i in valid]))


# ======================================================================
# HIERARCHY-VIOLATION DIAGNOSTIC (post-hoc only -- see module docstring
# for the caveat about matching hmcn_eval's own definition)
# ======================================================================
def compute_hier_violation_rate(fine_pred, meta_pred, pairs):
    """Fraction of (sample, child-parent pair) instances where the child
    (fine label) is predicted positive but its parent (meta label) is
    predicted negative, averaged over pairs -- i.e. the binarized-prediction
    analogue of the training-time vloss hinge term. Computed here purely as
    a diagnostic on a model that was never penalized for violating it.
    """
    if not pairs:
        return float("nan")
    rates = []
    for fi, mi in pairs:
        child_pos = fine_pred[:, fi] == 1
        if child_pos.sum() == 0:
            continue
        viol = (meta_pred[child_pos, mi] == 0).mean()
        rates.append(viol)
    return float(np.mean(rates)) if rates else float("nan")


# ======================================================================
# METRICS (mirrors baseline_fine138_kfold.py's compute_fine_only_metrics,
# generalized to a suffix so it produces both the _138 and _12 blocks)
# ======================================================================
def compute_metrics(probs, true, thresholds, suffix):
    pred = (probs >= thresholds[np.newaxis, :]).astype(int)
    valid = [i for i in range(true.shape[1]) if true[:, i].sum() > 0]

    roc_auc = float(np.mean([roc_auc_score(true[:, i], probs[:, i]) for i in valid]))
    pr_auc  = float(np.mean([average_precision_score(true[:, i], probs[:, i]) for i in valid]))

    return {
        f'roc_auc_{suffix}'         : round(roc_auc, 4),
        f'pr_auc_{suffix}'          : round(pr_auc, 4),
        f'f1_macro_{suffix}'        : round(float(f1_score(true, pred, average='macro', zero_division=0)), 4),
        f'f1_micro_{suffix}'        : round(float(f1_score(true, pred, average='micro', zero_division=0)), 4),
        f'precision_macro_{suffix}' : round(float(precision_score(true, pred, average='macro', zero_division=0)), 4),
        f'recall_macro_{suffix}'    : round(float(recall_score(true, pred, average='macro', zero_division=0)), 4),
        f'precision_micro_{suffix}' : round(float(precision_score(true, pred, average='micro', zero_division=0)), 4),
        f'recall_micro_{suffix}'    : round(float(recall_score(true, pred, average='micro', zero_division=0)), 4),
        f'matched_accuracy_{suffix}': round(float(accuracy_score(true, pred)), 4),
        f'hamming_loss_{suffix}'    : round(float(hamming_loss(true, pred)), 4),
        f'jaccard_{suffix}'         : round(float(jaccard_score(true, pred, average='macro', zero_division=0)), 4),
    }, pred


def per_label_rows(fold_i, param_value, label_set, label_names, probs, true, pred, thresholds):
    rows = []
    for i, label in enumerate(label_names):
        yt, ypr, yp = true[:, i], probs[:, i], pred[:, i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        try:    roc = roc_auc_score(yt, ypr)
        except: roc = float("nan")
        try:    pr = average_precision_score(yt, ypr)
        except: pr = float("nan")
        try:    mcc = matthews_corrcoef(yt, yp)
        except: mcc = float("nan")
        rows.append({
            "Fold": fold_i, "Model": "MLP", "Config": param_value,
            "LabelSet": label_set, "Label": label,
            "Threshold": round(float(thresholds[i]), 2),
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
    return pd.DataFrame(rows)


# ======================================================================
# TRAIN + EVALUATE ONE (fold, config)
# ======================================================================
def run_config(cfg, param_value, fold_i, fold_data, fine_names, meta_names, pairs,
               ablation_csv_path, all_results_accum, results_path):
    (X_train, y1_tr, y2_tr, X_val, y1_va, y2_va, X_test, y1_te, y2_te) = fold_data

    def T(*a): return [torch.tensor(x, dtype=torch.float32) for x in a]
    train_loader = DataLoader(TensorDataset(*T(X_train, y1_tr, y2_tr)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(*T(X_val, y1_va, y2_va)), batch_size=128)
    test_loader  = DataLoader(TensorDataset(*T(X_test, y1_te, y2_te)), batch_size=128)

    model = JointMLP(
        idim=X_train.shape[1], nf=y1_tr.shape[1], nm=y2_tr.shape[1],
        gd=cfg['global_dim'], ld=cfg['local_dim'], dr=cfg['dropout'], beta=BETA,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    logger.info(f"  [MLP] fold {fold_i}: fitting {param_value}")
    t0 = time.time()
    best_val_score, best_state, best_epoch, patience_counter = 0.0, None, 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, Y1b, Y2b in train_loader:
            Xb, Y1b, Y2b = Xb.to(device), Y1b.to(device), Y2b.to(device)
            optimizer.zero_grad()
            _, P1, P2, PG = model(Xb)
            loss = loss_fn_no_hierarchy(P1, P2, PG, Y1b, Y2b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        fp_v, ft_v, mp_v, mt_v = collect(model, val_loader, device)
        # Val macro PR-AUC, averaged across fine138 and meta12 -- the early
        # stopping / model-selection criterion (see module docstring).
        val_score = 0.5 * (mean_pr_auc(fp_v, ft_v) + mean_pr_auc(mp_v, mt_v))
        scheduler.step(-val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.to(device)
    logger.info(f"  [MLP] fold {fold_i} [{param_value}]: fit done in {time.time() - t0:.1f}s, "
                f"best_epoch={best_epoch}, best_val_pr_auc_combined={best_val_score:.4f}")

    # Thresholds calibrated on VAL, metrics reported on TEST.
    fp_v, ft_v, mp_v, mt_v = collect(model, val_loader, device)
    fine_thr = hmcn_eval.find_optimal_thresholds(fp_v, ft_v)
    meta_thr = hmcn_eval.find_optimal_thresholds(mp_v, mt_v)

    fp_t, ft_t, mp_t, mt_t = collect(model, test_loader, device)
    fine_metrics, fine_pred = compute_metrics(fp_t, ft_t, fine_thr, suffix='138')
    meta_metrics, meta_pred = compute_metrics(mp_t, mt_t, meta_thr, suffix='12')
    hier_violation_rate_138 = compute_hier_violation_rate(fine_pred, meta_pred, pairs)

    macro_f1 = fine_metrics['f1_macro_138']
    logger.info(f"  [MLP] fold {fold_i} [{param_value}] TEST  "
                f"pr_auc_138={fine_metrics['pr_auc_138']:.3f}  f1_macro_138={macro_f1:.3f}  "
                f"pr_auc_12={meta_metrics['pr_auc_12']:.3f}  hier_violation_rate_138={hier_violation_rate_138:.4f}")

    ablation_metrics = {**fine_metrics, **meta_metrics, 'hier_violation_rate_138': round(hier_violation_rate_138, 4)}
    config = dict(
        experiment='mlp_kfold', param_name='model', param_value=param_value,
        fold=fold_i, model='MLP', seed=RANDOM_STATE,
        global_dim=cfg['global_dim'], local_dim=cfg['local_dim'], dropout=cfg['dropout'],
        lr=LR, weight_decay=WEIGHT_DECAY, lambda_viol=0.0, beta=BETA, batch_size=BATCH_SIZE,
        best_epoch=best_epoch, val_pr_auc_combined=round(best_val_score, 4),
    )
    hmcn_eval.save_experiment(config, ablation_metrics, csv_path=ablation_csv_path)

    fine_rows = per_label_rows(fold_i, param_value, 'fine138', fine_names, fp_t, ft_t, fine_pred, fine_thr)
    meta_rows = per_label_rows(fold_i, param_value, 'meta12', meta_names, mp_t, mt_t, meta_pred, meta_thr)
    all_results_accum.append(pd.concat([fine_rows, meta_rows], ignore_index=True))
    pd.concat(all_results_accum, ignore_index=True).to_csv(results_path, index=False)


def expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid_dict.values())]


def format_param_value(cfg):
    return "MLP_" + "_".join(f"{k}={v}" for k, v in cfg.items())


# ======================================================================
# MAIN
# ======================================================================
def main():
    df_clean, fine_cols, meta_cols, fine_names, meta_names, fp_cols, mordred_cols = load_and_prepare()
    X_fp_raw = df_clean[fp_cols].values.astype(float)
    X_mordred_raw = df_clean[mordred_cols].values.astype(float)
    y_fine_full = df_clean[fine_cols].values
    y_meta_full = df_clean[meta_cols].values
    pairs = build_pairs(fine_names, meta_names)
    logger.info(f"Hierarchy pairs (diagnostic only): {len(pairs)}")

    # Fold split is stratified on the full 138-label set -- matches the
    # cache exactly, same as baseline_fine138_kfold.py.
    fold_splits = build_or_load_kfold_splits(X_fp_raw, y_fine_full)

    if DEBUG:
        fold_splits = fold_splits[:DEBUG_N_FOLDS]
        logger.info(f"[DEBUG] restricted to {len(fold_splits)} fold(s)")

    results_path = OUTPUT_DIR / "mlp_kfold_all_results.csv"
    ablation_csv_path = OUTPUT_DIR / "mlp_kfold_ablation_results.csv"

    configs = [(cfg, format_param_value(cfg)) for cfg in expand_grid(GRID)]
    logger.info(f"Configs total: {len(configs)} | Folds: {len(fold_splits)} | "
                f"Total runs: {len(configs) * len(fold_splits)}")

    # --- Restart-skip: which (fold, Config) pairs are already done? ---
    completed = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        completed = set(zip(prev["Fold"], prev["Config"]))
        logger.info(f"Found {len(completed)} completed (fold, config) entries in "
                    f"'{results_path}' -- these will be skipped.")

    all_results = []
    if results_path.exists():
        all_results.append(pd.read_csv(results_path))

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold_i+1}/{len(fold_splits)}  "
                    f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test)")
        logger.info(f"{'='*60}")

        runs_remaining = [(cfg, pv) for cfg, pv in configs if (fold_i, pv) not in completed]
        if not runs_remaining:
            logger.info(f"  All {len(configs)} configs already completed for fold {fold_i} -- skipping.")
            continue

        fold_data = build_fold_features(X_fp_raw, X_mordred_raw, y_fine_full, y_meta_full,
                                         train_idx, val_idx, test_idx)

        for i, (cfg, param_value) in enumerate(configs, 1):
            if (fold_i, param_value) in completed:
                logger.info(f"  [{i}/{len(configs)}] {param_value}: already completed, skipping")
                continue
            try:
                run_config(cfg, param_value, fold_i, fold_data, fine_names, meta_names, pairs,
                           ablation_csv_path, all_results, results_path)
            except Exception:
                logger.exception(f"  [{i}/{len(configs)}] {param_value}: failed, continuing")

    if not all_results:
        logger.error("No run completed successfully.")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(results_path, index=False)

    metric_cols = ["Bal.Acc", "MCC", "F1", "ROC_AUC", "PR_AUC", "Precision", "Sensitivity", "Specificity"]

    # Per-fold macro average (mean across labels within each LabelSet block).
    per_fold_macro = all_results_df.groupby(["Fold", "Model", "Config", "LabelSet"])[metric_cols].mean().round(3)
    per_fold_macro.to_csv(OUTPUT_DIR / "mlp_kfold_per_fold_macro.csv")

    summary = all_results_df.groupby(["Fold", "Model", "Config", "LabelSet"])[metric_cols].mean() \
                             .groupby(["Model", "Config", "LabelSet"]).agg(["mean", "std"]).round(3)
    summary.to_csv(OUTPUT_DIR / "mlp_kfold_summary_mean_std.csv")
    logger.info("Mean +/- std across folds, per (Config, LabelSet):\n" + summary.to_string())

    if ablation_csv_path.exists():
        ablation_df = pd.read_csv(ablation_csv_path)
        exclude_cols = ['experiment', 'param_name', 'param_value', 'fold', 'model', 'seed']
        metric_cols_ablation = [c for c in hmcn_eval._CSV_COLUMNS if c not in exclude_cols]
        metric_cols_ablation = [c for c in metric_cols_ablation if c in ablation_df.columns]
        ablation_summary = ablation_df.groupby(['model', 'param_value'])[metric_cols_ablation].agg(['mean', 'std'])
        ablation_summary_path = OUTPUT_DIR / "mlp_kfold_ablation_summary_mean_std.csv"
        ablation_summary.to_csv(ablation_summary_path)
        logger.info(f"Ablation-format mean +/- std summary (compare directly against "
                    f"HMCN-F's own hmcn_kfold_summary_mean_std.csv): {ablation_summary_path}")

        # Single winning config -- by mean pr_auc_138 across folds, same
        # selection rule as baseline_fine138_kfold.py, for the methods table.
        mean_only = ablation_df.groupby(['model', 'param_value'])[metric_cols_ablation].mean()
        winners = mean_only.loc[mean_only.groupby('model')['pr_auc_138'].idxmax()]
        winners_path = OUTPUT_DIR / "mlp_kfold_winning_config.csv"
        winners.to_csv(winners_path)
        report_cols = [c for c in ['pr_auc_138', 'f1_macro_138', 'roc_auc_138',
                                    'pr_auc_12', 'f1_macro_12', 'hier_violation_rate_138']
                       if c in winners.columns]
        logger.info("Winning config (best mean pr_auc_138 across folds):\n" + winners[report_cols].to_string())
        logger.info(f"Winning config saved to: {winners_path}")

    logger.info(f"Done. Per-label results in {results_path}")
    logger.info(f"Done. Ablation-format results (matches hmcn_ablation_results.csv schema) in {ablation_csv_path}")

    script_end_time = datetime.now()
    elapsed = script_end_time - SCRIPT_START_TIME
    logger.info(f"Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
